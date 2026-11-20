"""Local checks of the Phase 2 functional run's pieces (no GPU, no Modal).

The run itself (research/phase2/gpu/modal_phase2.py) is launched by hand on
Modal. Here: its module imports and its plan names real workloads; the
container preflight passes with the controller on; the optimizer step works
on a synthetic trace that carries Session 1's real KV capacity; and the summary
counts a synthetic trace correctly.
"""

import json
import sys
import types
from unittest import mock

import pytest

from vllm.kalypso.benchmark.eval import costmodel as cm

SESSION1_CAPACITY = 25_631_373_374
BYTES_PER_TOKEN = 57_344


@pytest.fixture
def phase2_module(monkeypatch):
    monkeypatch.setitem(sys.modules, "modal", mock.MagicMock())
    for name in ("research.phase2.gpu.modal_phase2", "research.baseline.gpu.modal_baseline"):
        sys.modules.pop(name, None)
    import research.phase2.gpu.modal_phase2 as m
    yield m
    for name in ("research.phase2.gpu.modal_phase2", "research.baseline.gpu.modal_baseline"):
        sys.modules.pop(name, None)


def test_plan_uses_real_workloads_and_turns_the_controller_on(phase2_module):
    from research.baseline import workloads  # noqa: F401
    from vllm.kalypso.benchmark.eval.workloads import WORKLOADS
    exp = phase2_module.EXPERIMENT
    assert exp["env"] == {"KALYPSO_ADAPTIVE_STAGE": "1"}
    for part in exp["parts"]:
        w = WORKLOADS[part["workload"]](part["n"] or None)
        assert len(w.stages) == 2  # a managed stage after the first
    assert "adaptive_selectivity" not in json.dumps(exp)  # step C stays off


def test_container_preflight_passes(phase2_module, monkeypatch):
    monkeypatch.setenv("KALYPSO_ADAPTIVE_STAGE", "1")
    exec(phase2_module.PREFLIGHT, {})


def _trace(tmp_path, capacity=SESSION1_CAPACITY):
    lines = [{"event": "query_start", "query_id": "q1"},
             {"event": "kv_stage_registered", "query_id": "q1", "stage_id": 1,
              "total_capacity_bytes": capacity},
             {"event": "adaptive_switch", "query_id": "q1", "stage_id": 2, "action": "switch",
              "mode": "tuple", "size": 1, "from_mode": "static", "pass_rate": 0.0},
             {"event": "adaptive_switch", "query_id": "q1", "stage_id": 2, "action": "veto",
              "mode": "block", "size": 4, "reason": "quality guard", "vetoed": [4, 2]},
             {"event": "query_end", "query_id": "q1"},
             {"event": "query_start", "query_id": "q2"}]
    (tmp_path / "trace-on.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines))


def _tokens_json(tmp_path, parts):
    """What modal_baseline._tokens writes, with a stand-in 40 tokens per row
    (the real workloads' rows; real counts come from the container's tokenizer)."""
    from research.baseline import workloads  # noqa: F401
    from vllm.kalypso.benchmark.eval.workloads import WORKLOADS
    out = {}
    for p in parts:
        w = WORKLOADS[p["workload"]](p["n"] or None)
        out[f"{p['workload']}:{p['n'] or ''}"] = {
            "left": [40] * len(w.rows), "right": [40] * len(w.right_rows or [])}
    (tmp_path / "tokens.json").write_text(json.dumps(out))


def test_optimizer_step_on_real_workloads(tmp_path, monkeypatch, phase2_module):
    from research.phase2 import optimizer_step

    def no_fetch(*a, **k):
        raise AssertionError("item tokens missing from the shipped cache")
    monkeypatch.setattr(cm, "_count", no_fetch)
    _trace(tmp_path)
    _tokens_json(tmp_path, phase2_module.PARTS)
    out = optimizer_step.run(tmp_path, phase2_module.PARTS, BYTES_PER_TOKEN)
    assert out["kv_budget_tokens"] == pytest.approx(SESSION1_CAPACITY / BYTES_PER_TOKEN)
    for part in phase2_module.PARTS:
        p = out["parts"][part["name"]]
        assert p["kv_rejected"] == []                  # nothing near the real budget
        assert p["arm"].startswith("kalypso_pipe:") and len(p["blocks"]) == 2
    # Every workload has 2 stages: the automatic cache must not engage.
    assert out["decision_cache"] == {"hits": 0, "misses": 0, "evictions": 0, "entries": 0}
    assert json.loads((tmp_path / "optimizer.json").read_text()) == out


def test_optimizer_step_tiny_budget_rejects_and_records(tmp_path, phase2_module):
    from research.phase2 import optimizer_step
    _trace(tmp_path, capacity=BYTES_PER_TOKEN * 300)  # 300 tokens of KV
    _tokens_json(tmp_path, phase2_module.PARTS[:1])
    out = optimizer_step.run(tmp_path, phase2_module.PARTS[:1], BYTES_PER_TOKEN)
    assert out["parts"]["rpipe_s05"]["kv_rejected"]


def test_functional_summary_counts(tmp_path):
    from research.phase2.functional_summary import summarize
    _trace(tmp_path)
    part = tmp_path / "rpipe_s05"
    part.mkdir()
    (part / "run-on.jsonl").write_text(
        json.dumps({"type": "meta"}) + "\n"
        + json.dumps({"arm": "kalypso_pipe:4,4", "outputs": {"0": "yes"}}) + "\n"
        + json.dumps({"arm": "kalypso_pipe:4,4", "outputs": {}, "error": "boom"}) + "\n")
    s = summarize(tmp_path)
    assert s["queries_started"] == 2 and s["queries_finished"] == 1
    assert s["queries_unfinished"] == ["q2"]
    assert {d["action"] for d in s["controller_decisions"]} == {"switch", "veto"}
    assert s["arms"]["rpipe_s05 kalypso_pipe:4,4"] == {"trials": 2, "with_outputs": 1, "errors": 1}
