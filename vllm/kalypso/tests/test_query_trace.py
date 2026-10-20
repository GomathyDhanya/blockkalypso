"""QueryTrace: per-stage rows, timing, occupancy and operator stats."""

import asyncio
from types import SimpleNamespace

import pytest

from vllm.kalypso.controller.plan import SemanticPlan
from vllm.kalypso.execution import trace as tr
from vllm.kalypso.execution.pipeline_execution import AsyncPipelineExecutor

from .test_block_filter import FakeOut, row
from .test_block_join_pipeline import QueryExecutor


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


def _stage(stage_id, block_size=1, ops=()):
    return SimpleNamespace(stage_id=stage_id, block_size=block_size,
                           operators=ops, fanout_op=None)


def _task(task_id, rows):
    return SimpleNamespace(task_id=task_id, num_rows=rows)


def test_trace_records_timing_occupancy_and_counts_retries_once():
    clock = Clock()
    trace = tr.QueryTrace(clock=clock)
    stats = SimpleNamespace(as_dict=lambda: {"llm_calls": 3})
    s0 = _stage(0, block_size=4, ops=(SimpleNamespace(stats=stats),))
    s1 = _stage(1)
    trace.register_stages([s0, s1])

    clock.t = 101.0
    trace.task_started(s0, _task(1, 4))
    trace.task_started(s0, _task(2, 2))
    clock.t = 102.0
    trace.task_finished(_task(1, 4), 3, tr.DONE)
    trace.task_finished(_task(2, 2), 0, tr.RETRY)
    trace.task_started(s0, _task(3, 2))        # the retry of task 2
    clock.t = 103.0
    trace.task_finished(_task(3, 2), 1, tr.DONE)
    trace.task_started(s1, _task(4, 1))
    clock.t = 104.5
    trace.task_finished(_task(4, 1), 1, tr.DONE)
    trace.outputs_emitted(1)
    trace.outputs_emitted(0)

    d = trace.to_dict()
    first = d["stages"]["0"]
    assert first["tasks"] == 2 and first["retries"] == 1
    assert first["rows_in"] == 6 and first["rows_out"] == 4
    assert first["first_start"] == 1.0 and first["last_end"] == 3.0
    assert first["mean_occupancy"] == 0.75          # (4 + 2) / (4 * 2)
    assert first["operator_stats"] == {"SimpleNamespace": {"llm_calls": 3}}
    assert d["stages"]["1"]["mean_occupancy"] is None   # not a block stage
    assert d["first_output_s"] == 4.5 and d["outputs"] == 1
    assert [t["outcome"] for t in d["tasks"]] == [tr.DONE, tr.RETRY, tr.DONE, tr.DONE]


def test_trace_of_filter_join_filter_pipeline(fake_kv_manager, monkeypatch):
    monkeypatch.setattr(AsyncPipelineExecutor, "LOG_SCHEDULER", False)
    executor = QueryExecutor()
    left = [row(text, idx, executor) for idx, text in
            enumerate(("apple", "banana", "carrot", "date"))]
    right = [row(text, 100 + idx) for idx, text in
             enumerate(("carrot", "apple", "pear"))]
    monkeypatch.setattr("vllm.kalypso.controller.plan.data._data_source",
                        lambda *args: iter(right))
    plan = SemanticPlan(executor=executor)
    physical = plan.parse_ops([
        {"op": "sem_filter", "args": {"prompt": "keep all", "block_size": 2}},
        {"op": "join", "args": {"instruction": "same item",
                                "right_table": "right.csv",
                                "left_block_size": 2, "right_block_size": 2}},
        {"op": "sem_filter", "args": {"prompt": "only apple", "block_size": 2}},
    ])
    built = plan.build(left, physical)
    trace = tr.QueryTrace()
    output, _ = asyncio.run(plan.plan_executor.execute(left, built, trace=trace))
    assert [(c.state.idx, c.state.right_idx) for c in output] == [(0, 101)]

    d = trace.to_dict()
    f1, join, f2 = (d["stages"][str(s.stage_id)] for s in built[0])
    assert f1["operators"] == ["BlockSemFilter"] and join["operators"] == ["BlockSemJoin"]
    assert (f1["rows_in"], f1["rows_out"], f1["mean_occupancy"]) == (4, 4, 1.0)
    assert (join["tasks"], join["rows_in"], join["rows_out"]) == (2, 4, 2)
    # Each join task yields one match and F2 blocks are cut per join task, so
    # every F2 block holds one row of a possible two.
    assert (f2["tasks"], f2["rows_in"], f2["rows_out"]) == (2, 2, 1)
    assert f2["mean_occupancy"] == 0.5
    assert join["operator_stats"]["BlockSemJoin"]["block_calls"] == 4
    assert f1["operator_stats"]["BlockSemFilter"]["llm_calls"] == 2
    assert d["outputs"] == 1
    assert 0 <= d["first_output_s"] <= d["elapsed_s"]
    assert all(t["end"] is not None for t in d["tasks"])
    # The F2 block holding the carrot match passes nothing, so it is dropped.
    assert sorted(t["outcome"] for t in d["tasks"]) == [tr.DONE] * 5 + [tr.DROPPED]
    assert f1["first_start"] <= join["first_start"] <= f2["first_start"]


class PairAwareExecutor(QueryExecutor):
    """QueryExecutor that also answers Kalypso's per-pair join prompt."""

    async def execute(self, raw_request, prompt, max_tokens, pin=False, priority=0):
        body = prompt[-1]["content"]
        if "Condition:same item" in body:
            self.join_calls += 1
            left, right = (m["content"].split('"text":', 1)[1].split("}")[0].strip()
                           for m in prompt[-3:-1])
            return FakeOut("TRUE" if left == right else "FALSE")
        if "Condition:only apple" in body:
            self.filter_calls += 1
            context = " ".join(m["content"] for m in prompt[:-1] if m["role"] != "system")
            return FakeOut("TRUE" if "apple" in context else "FALSE")
        return await super().execute(raw_request, prompt, max_tokens, pin=pin,
                                     priority=priority)


def _filter_join_filter(monkeypatch, blocked=True):
    executor = PairAwareExecutor()
    left = [row(text, idx, executor) for idx, text in
            enumerate(("apple", "banana", "carrot", "date"))]
    right = [row(text, 100 + idx) for idx, text in
             enumerate(("carrot", "apple", "pear"))]
    monkeypatch.setattr("vllm.kalypso.controller.plan.data._data_source",
                        lambda *args: iter(right))
    plan = SemanticPlan(executor=executor)
    sizes = ({"block_size": 2}, {"left_block_size": 2, "right_block_size": 2}) if blocked \
        else ({}, {})
    physical = plan.parse_ops([
        {"op": "sem_filter", "args": {"prompt": "keep all", **sizes[0]}},
        {"op": "join", "args": {"instruction": "same item", "right_table": "right.csv",
                                **sizes[1]}},
        {"op": "sem_filter", "args": {"prompt": "only apple", **sizes[0]}},
    ])
    return plan, plan.build(left, physical), left


@pytest.mark.parametrize("blocked", [True, False])
def test_materialized_runs_stages_one_after_another(fake_kv_manager, monkeypatch, blocked):
    monkeypatch.setattr(AsyncPipelineExecutor, "LOG_SCHEDULER", False)
    results = {}
    for materialize in (False, True):
        plan, built, left = _filter_join_filter(monkeypatch, blocked)
        trace = tr.QueryTrace()
        out, _ = asyncio.run(plan.plan_executor.execute(
            left, built, trace=trace, materialize=materialize))
        results[materialize] = (sorted((c.state.idx, c.state.right_idx) for c in out),
                                trace.to_dict())
    assert results[True][0] == results[False][0] == [(0, 101)]   # same answers
    d = results[True][1]
    stages = [d["stages"][k] for k in sorted(d["stages"], key=int)]
    for prev, cur in zip(stages, stages[1:]):
        assert cur["first_start"] >= prev["last_end"]               # no overlap
    assert d["outputs"] == 1                                         # final stage only
    assert d["first_output_s"] >= stages[-1]["first_start"]


def test_materialized_flag_reaches_the_executor(fake_kv_manager, monkeypatch):
    seen = {}

    async def fake_execute(self, ctxs, plan, trace=None, materialize=False, coalesce=False):
        seen.update(materialize=materialize, coalesce=coalesce)
        return [], []

    from vllm.kalypso.execution.pipeline_execution import PlanExecutor
    monkeypatch.setattr(PlanExecutor, "execute", fake_execute)
    monkeypatch.setattr("vllm.kalypso.controller.plan.data._data_source",
                        lambda *args: iter([]))
    from vllm.kalypso.query import Query
    plan = SemanticPlan(executor=None)
    monkeypatch.setattr(plan, "build", lambda ctxs, ops: [])
    asyncio.run(plan.execute(None, Query([], "x"), materialize=True))
    assert seen == {"materialize": True, "coalesce": False}
    asyncio.run(plan.execute(None, Query([], "x"), coalesce=True))
    assert seen == {"materialize": False, "coalesce": True}


@pytest.mark.parametrize("blocked", [True, False])
def test_trace_counts_each_operator_even_in_fused_stages(fake_kv_manager, monkeypatch,
                                                         blocked):
    monkeypatch.setattr(AsyncPipelineExecutor, "LOG_SCHEDULER", False)
    plan, built, left = _filter_join_filter(monkeypatch, blocked)
    trace = tr.QueryTrace()
    asyncio.run(plan.plan_executor.execute(left, built, trace=trace))
    ops = {}
    for stage in trace.to_dict()["stages"].values():
        for op in stage["operator_rows"]:
            ops.setdefault(op["position"], []).append(op)
    count = lambda pos, key: sum(o[key] for o in ops[pos])
    assert (count(0, "rows_in"), count(0, "rows_out")) == (4, 4)          # keep all
    assert count(1, "rows_out") == 2                                       # 2 matches
    assert (count(2, "rows_in"), count(2, "rows_out")) == (2, 1)          # only apple
    if not blocked:   # per-tuple: join check and F2 share a stage, still separate here
        assert count(1, "rows_in") == 12                                   # 4 x 3 pairs
        heads = {}
        for o in ops[1]:
            for k, v in o["answers"].items():
                heads[k] = heads.get(k, 0) + v
        assert heads == {"true": 2, "false": 10}


def test_coalesce_fills_filter2_blocks_across_join_tasks(fake_kv_manager, monkeypatch):
    """Each 2x2 join task yields one match; F2 blocks of 2 are half full per
    join task, and full once the join's output is coalesced."""
    monkeypatch.setattr(AsyncPipelineExecutor, "LOG_SCHEDULER", False)
    seen = {}
    for coalesce in (False, True):
        plan, built, left = _filter_join_filter(monkeypatch, blocked=True)
        trace = tr.QueryTrace()
        out, _ = asyncio.run(plan.plan_executor.execute(left, built, trace=trace,
                                                        coalesce=coalesce))
        stages = [trace.to_dict()["stages"][str(st.stage_id)] for st in built[0]]
        seen[coalesce] = (sorted((c.state.idx, c.state.right_idx) for c in out), stages[2])
    assert seen[False][0] == seen[True][0] == [(0, 101)]         # same answers
    assert (seen[False][1]["tasks"], seen[False][1]["mean_occupancy"]) == (2, 0.5)
    assert (seen[True][1]["tasks"], seen[True][1]["mean_occupancy"]) == (1, 1.0)


def test_coalesce_with_materialize_still_matches(fake_kv_manager, monkeypatch):
    monkeypatch.setattr(AsyncPipelineExecutor, "LOG_SCHEDULER", False)
    plan, built, left = _filter_join_filter(monkeypatch, blocked=True)
    out, _ = asyncio.run(plan.plan_executor.execute(left, built, materialize=True,
                                                    coalesce=True))
    assert sorted((c.state.idx, c.state.right_idx) for c in out) == [(0, 101)]
