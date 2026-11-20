"""Local checks for the static-vs-adaptive comparison (no GPU, no Modal)."""

import json
import sys
from unittest import mock

import pytest

from research.phase2 import workloads as P  # noqa: F401
from vllm.kalypso.benchmark.eval.workloads import WORKLOADS


def test_drift_workload_is_rpipe_s50_reordered():
    base, d = WORKLOADS["rpipe_s50"](None), WORKLOADS["rpipe_s50_drift"](None)
    assert sorted(r.text for r in base.rows) == sorted(r.text for r in d.rows)
    assert d.right_rows == base.right_rows and d.stages == base.stages
    text = {r.idx: r.text for r in d.rows}
    base_text = {r.idx: r.text for r in base.rows}
    assert {(text[i], j) for i, j in d.positives} == {(base_text[i], j) for i, j in base.positives}
    left = {i for i, _ in d.positives}
    labels = [r.idx in left for r in d.rows]
    assert not any(labels[:P.DRIFT_NEGATIVES]) and sum(labels[P.DRIFT_NEGATIVES:]) == 48


@pytest.fixture
def phase2_module(monkeypatch):
    monkeypatch.setitem(sys.modules, "modal", mock.MagicMock())
    for name in ("research.phase2.gpu.modal_phase2", "research.baseline.gpu.modal_baseline"):
        sys.modules.pop(name, None)
    import research.phase2.gpu.modal_phase2 as m
    yield m
    for name in ("research.phase2.gpu.modal_phase2", "research.baseline.gpu.modal_baseline"):
        sys.modules.pop(name, None)


def test_compare_plan(phase2_module):
    c = phase2_module.COMPARE
    assert [on for _, on in c["phases"]] == [False, True, False, True]
    assert c["warmup"] == 1 and c["repeats"] == 2
    for part in c["parts"]:
        assert part["workload"] in WORKLOADS
        assert part["arm"].startswith("kalypso_pipe:")


def test_compare_summary(tmp_path):
    from research.phase2.compare_summary import summarize
    plan = {"parts": [{"name": "p", "workload": "rpipe_s05", "n": 0, "arm": "kalypso_pipe:1,2"}],
            "phases": [["off-1", False], ["on-1", True]]}
    (tmp_path / "plan.json").write_text(json.dumps(plan))
    (tmp_path / "p").mkdir()
    meta = {"type": "meta", "kind": "join", "rows": 2, "right_rows": 1, "positives": [[0, 0]],
            "data_path": "/d.csv"}
    for tag, lat, sel in (("off-1", 2.0, [[0, 0]]), ("on-1", 1.0, [[0, 0], [1, 0]])):
        (tmp_path / "p" / f"run-{tag}.jsonl").write_text(
            json.dumps(meta) + "\n" + json.dumps({"type": "trial", "arm": "a", "latency_s": lat,
                                                  "selected": sel, "llm_requests": 3}) + "\n")
    (tmp_path / "trace-on-1.jsonl").write_text(
        json.dumps({"event": "query_start", "query_id": "q", "data_path": "/d.csv"}) + "\n"
        + json.dumps({"event": "adaptive_switch", "query_id": "q", "action": "veto",
                      "mode": "block", "size": 2, "vetoed": [2]}) + "\n")
    (tmp_path / "trace-off-1.jsonl").write_text(
        json.dumps({"event": "query_start", "query_id": "r", "data_path": "/d.csv"}) + "\n")
    s = summarize(tmp_path)
    assert s["parts"]["p"]["static"]["latency_s"]["mean"] == 2.0
    assert s["parts"]["p"]["static"]["f1"]["mean"] == 1.0
    assert s["parts"]["p"]["adaptive"]["f1"]["mean"] == pytest.approx(2 / 3, abs=1e-4)
    assert s["decision_counts"]["static"] == []
    assert s["decision_counts"]["adaptive"][0]["vetoed"] == [2]
    assert s["queries_per_config"] == {"static": {"p": 1}, "adaptive": {"p": 1}}
