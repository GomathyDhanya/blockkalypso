"""Opt-in telemetry (vllm/kalypso/telemetry.py): off by default, does not
change query results, and its records are consistent with what ran.

Uses a deterministic fake LLM: a row passes the filter question ``Q_KEEP``
when its text contains ``kk`` and a pair passes the join question
``Q_MATCH`` when the pair's text contains ``mm``.
"""

import asyncio
import copy
import re

import pytest

from vllm.kalypso import telemetry
from vllm.kalypso.controller.plan import SemanticPlan
from vllm.kalypso.execution.vllm_executor import trace_llm_call
from vllm.kalypso.query import Query

from .test_block_filter import FakeOut, row

_ITEM = re.compile(r"^(\d+)\. ", re.M)


class RuleExecutor:
    """Answers block and single prompts from the rule in the module doc."""

    def __init__(self):
        self.prompts = []

    @staticmethod
    def _passes(question_text: str, item: str) -> bool:
        return ("mm" in item) if "Q_MATCH" in question_text else ("kk" in item)

    async def execute(self, raw_request, prompt, max_tokens, pin=False, priority=0):
        body = prompt[-1]["content"]
        self.prompts.append(body)
        ids = [(m.group(1), m.end()) for m in _ITEM.finditer(body)]
        if ids and "Items:" in body:
            question = body.split("Items:")[0]
            lines = []
            for n, (i, start) in enumerate(ids):
                end = ids[n + 1][1] - len(ids[n + 1][0]) - 2 if n + 1 < len(ids) else len(body)
                lines.append(f"{i}: {'YES' if self._passes(question, body[start:end]) else 'NO'}")
            return FakeOut("\n".join(lines))
        whole = "\n".join(str(m.get("content", "")) for m in prompt)
        verdict = self._passes(whole, whole.split("Q_KEEP")[0].split("Q_MATCH")[0] or whole)
        if body.startswith("Item: "):
            return FakeOut("YES" if verdict else "NO")
        return FakeOut("true" if verdict else "false")


def _source(executor, n=12):
    return [row(f"row{i} {'kk' if i % 3 else 'xx'}", i, executor) for i in range(n)]


def _right(n=4):
    return [row(f"right{j} {'mm' if j % 2 else 'zz'}", j) for j in range(n)]


PIPELINES = {
    "block_block": [
        {"op": "sem_filter", "args": {"prompt": "Q_KEEP", "block_size": 3}},
        {"op": "sem_filter", "args": {"prompt": "Q_KEEP again", "block_size": 2}},
    ],
    "tuple_block": [
        {"op": "sem_filter", "args": {"prompt": "Q_KEEP"}},
        {"op": "sem_filter", "args": {"prompt": "Q_KEEP again", "block_size": 4}},
    ],
    "filter_blockjoin": [
        {"op": "sem_filter", "args": {"prompt": "Q_KEEP", "block_size": 4}},
        {"op": "join", "args": {"instruction": "Q_MATCH", "right_table": "r",
                                "pair_block_size": 3}},
    ],
    "filter_join_tuple": [
        {"op": "sem_filter", "args": {"prompt": "Q_KEEP"}},
        {"op": "join", "args": {"instruction": "Q_MATCH", "right_table": "r"}},
    ],
}


def _run(ops, monkeypatch, sink=None):
    monkeypatch.setattr("vllm.kalypso.controller.plan.data._data_source",
                        lambda raw, path, ex=None: iter(_right()) if path == "r"
                        else iter(_source(ex)))
    ex = RuleExecutor()
    telemetry.configure(sink, run_id="test")
    try:
        out = asyncio.run(SemanticPlan(ex).execute(None, Query(copy.deepcopy(ops), "left")))
    finally:
        telemetry.configure(None)
    result = sorted((r.state.idx, getattr(r.state, "right_idx", None),
                     tuple(str(o) for o in r.output)) for r in out)
    return result, sorted(ex.prompts)


@pytest.fixture(autouse=True)
def _off():
    telemetry.configure(None)
    yield
    telemetry.configure(None)


def test_off_by_default():
    assert not telemetry.enabled()
    telemetry.emit("nothing")  # no sink: a no-op, not an error


@pytest.mark.parametrize("name", sorted(PIPELINES))
def test_tracing_does_not_change_results(name, fake_kv_manager, monkeypatch):
    plain, plain_prompts = _run(PIPELINES[name], monkeypatch)
    events = []
    traced, traced_prompts = _run(PIPELINES[name], monkeypatch, events)
    assert traced == plain
    # Which rows share a downstream block depends on the order in which
    # concurrent upstream tasks finish, which varies even with tracing off
    # (asyncio.wait returns a set; see BASELINE_AUDIT.md). So compare the
    # number of LLM calls and every item each stage asked about, not the
    # grouping into blocks.
    assert len(traced_prompts) == len(plain_prompts)
    assert _items(traced_prompts) == _items(plain_prompts)
    assert events, "tracing on must record events"


def _items(prompts):
    out = []
    for body in prompts:
        question = body.split("Items:")[0] if "Items:" in body else ""
        if "Items:" in body:
            out += [question + line.split(". ", 1)[1]
                    for line in body.split("Items:")[1].split("\n\nAnswer")[0].strip().split("\n")
                    if ". " in line]
        else:
            out.append(body)
    return sorted(out)


@pytest.mark.parametrize("name", sorted(PIPELINES))
def test_trace_is_consistent(name, fake_kv_manager, monkeypatch):
    events = []
    result, prompts = _run(PIPELINES[name], monkeypatch, events)
    kinds = [e["event"] for e in events]
    assert kinds[0] == "query_start" and kinds[-1] == "query_end"
    assert "plan_built" in kinds and "kv_stage_registered" in kinds
    query_ids = {e["query_id"] for e in events}
    assert len(query_ids) == 1 and None not in query_ids
    end = events[-1]
    assert end["rows_out"] == len(result)
    assert end["rows_in"] == 12

    starts = [e for e in events if e["event"] == "task_start"]
    ends = [e for e in events if e["event"] == "task_end"]
    assert {e["task_id"] for e in starts} == {e["task_id"] for e in ends}
    assert all(e["wait_s"] >= 0 for e in starts) and all(e["exec_s"] >= 0 for e in ends)
    # Rows conserved between consecutive stages: what stage i emits is what
    # stage i+1 receives (every enqueued row runs exactly once here: no retries).
    stages = sorted({e["stage_id"] for e in ends})
    for a, b in zip(stages, stages[1:]):
        produced = sum(e["rows_out"] for e in ends if e["stage_id"] == a)
        consumed = sum(e["rows_in"] for e in ends if e["stage_id"] == b)
        assert produced == consumed
    last = stages[-1]
    assert sum(e["rows_out"] for e in ends if e["stage_id"] == last) == len(result)
    # One block_eval per block-operator call group, rows add up to block input.
    for stage in stages:
        evals = [e for e in events if e["event"] == "block_eval" and e.get("stage_id") == stage]
        if evals:
            assert sum(e["rows"] for e in evals) == sum(
                e["rows_in"] for e in ends if e["stage_id"] == stage)


def test_ids_attributed_to_stage_and_task(fake_kv_manager, monkeypatch):
    events = []
    _run(PIPELINES["block_block"], monkeypatch, events)
    evals = [e for e in events if e["event"] == "block_eval"]
    assert evals and all("stage_id" in e and "task_id" in e and e["op"] == "BlockSemFilter"
                         for e in evals)


def test_counters_reported_and_reset(fake_kv_manager, monkeypatch):
    fake_kv_manager._capacity = 1  # every reservation is too big: forced admission
    events = []
    _run(PIPELINES["block_block"], monkeypatch, events)
    counters = events[-1]["counters"]
    assert counters.get("kv_force_admitted", 0) > 0
    assert telemetry.take_counters() == {}


def test_llm_call_record_from_response():
    events = []
    telemetry.configure(events, run_id="t")
    token = telemetry.stage_id.set(7)
    try:
        trace_llm_call({"usage": {"prompt_tokens": 120, "completion_tokens": 3,
                                  "prompt_tokens_details": {"cached_tokens": 64}},
                        "choices": [{"message": {"content": "true"},
                                     "finish_reason": "stop"}]},
                       "chat", started=0.0, max_tokens=8, priority=-1, pin=False)
    finally:
        telemetry.stage_id.reset(token)
    (e,) = events
    assert e["event"] == "llm_call" and e["stage_id"] == 7
    assert (e["prompt_tokens"], e["completion_tokens"], e["cached_tokens"]) == (120, 3, 64)
    assert e["output_head"] == "true" and e["finish_reason"] == "stop"


def test_file_sink_writes_jsonl(tmp_path):
    import json
    path = tmp_path / "t" / "trace.jsonl"
    telemetry.configure(path, run_id="f")
    telemetry.emit("x", a=1, obj=object())
    telemetry.configure(None)
    (line,) = path.read_text().splitlines()
    record = json.loads(line)
    assert record["event"] == "x" and record["a"] == 1 and record["run_id"] == "f"
    assert isinstance(record["obj"], str)
