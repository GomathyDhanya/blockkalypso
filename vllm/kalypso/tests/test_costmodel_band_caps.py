"""Join quality caps by selectivity band (Phase 2 step A).

Ground truth: the per-workload ΔF1 in research/docs/QUALITY_CAP_DECISION.md,
rebuilt read-only by research/baseline/analysis/quality_cap_audit.py.
"""

import json
from dataclasses import fields
from types import SimpleNamespace

import pytest

from vllm.kalypso.benchmark.eval import costmodel as cm
from vllm.kalypso.controller.adaptive_stage import (
    AdaptiveStageConfig, AdaptiveStageController, costmodel_quality_veto)
from vllm.kalypso.sem_ops.block_filter import BlockSemFilter

from .test_block_filter import row

FITTED = cm.RESULTS / "analysis/costmodel.json"

# Join workloads: (true selectivity, regime) from QUALITY_CAP_DECISION.md.
JOINS = {
    "movie_fanout10": (0.09, "short"), "movie_fanout25": (0.25, "short"),
    "movie_fanout50": (0.50, "short"), "movie_fanout75": (0.75, "short"),
    "movie_fanout90": (0.91, "short"), "movie_opposite": (0.50, "short"),
    "pipe_filter_join_s10_join": (0.47, "short"), "pipe_filter_join_s50_join": (0.47, "short"),
    "pipe_filter_join_s90_join": (0.47, "short"),
    "cnli_join_low": (0.21, "long"), "cnli_join_high": (0.65, "long"),
}
# fanout75/90 kept ΔF1 >= -0.02 at k = 2, 4 and 6 and fail at 11. Every other
# join stays at 1: fanout50 fails at k = 2 (-0.048) before passing at 4, and
# the cap stops at the first failing k; cnli_join_low's lone +0.007 at k = 2 is
# one workload, below MIN_BAND_WORKLOADS.
EXPECTED_CAP = {w: (6 if w in ("movie_fanout75", "movie_fanout90") else 1) for w in JOINS}
TOKENS = {"short": 100.0, "long": 1600.0}


@pytest.fixture(scope="module")
def model():
    d = json.loads(FITTED.read_text())
    names = {f.name for f in fields(cm.KindParams)}
    kinds = {k: cm.KindParams(**{n: v for n, v in p.items() if n in names})
             for k, p in d["kinds"].items()}
    return cm.Model(g=d["g"], alpha=d["alpha"], beta=d["beta"], gamma=d["gamma"],
                    delta=d["delta"], tail=d["tail"], kinds=kinds,
                    f1_slack=d["f1_slack"], lam=d["lambda"], retry=d["retry"])


def _items(regime):
    return cm.Items(400, TOKENS[regime], TOKENS[regime])


def test_fit_reproduces_committed_band_caps(monkeypatch):
    from research.baseline.analysis.quality_cap_audit import SLACK, fit_rows
    monkeypatch.setattr(cm, "_count", cm._count)  # fit_rows replaces it; restore after
    fitted = cm.fit(fit_rows(), SLACK)
    committed = json.loads(FITTED.read_text())["kinds"]
    for kind, p in fitted.kinds.items():
        assert p.caps == committed[kind]["caps"]            # pooled caps unchanged
        assert p.band_caps == committed[kind]["band_caps"]
    assert committed["join"]["band_caps"] == {"short": {"0": 1, "1": 6},
                                              "long": {"0": 1, "1": 1}}


@pytest.mark.parametrize("workload", sorted(JOINS))
def test_join_cap_per_workload(model, workload):
    sel, regime = JOINS[workload]
    assert model.k_max("join", _items(regime), quality=True, selectivity=sel) \
        == EXPECTED_CAP[workload]


def test_previously_miscapped_joins_now_get_larger_k(model):
    for w in ("movie_fanout75", "movie_fanout90"):
        sel, regime = JOINS[w]
        assert model.k_max("join", _items(regime), quality=True) == 1  # pooled, as before
        assert model.k_max("join", _items(regime), quality=True, selectivity=sel) == 6
        pick = model.choose_k("join", _items(regime), quality=True, selectivity=sel)
        assert pick["k_max"] == 6


def test_low_selectivity_joins_still_cap_at_one(model):
    for sel in (0.0, 0.09, 0.25, 0.47, 0.5, 0.59):
        assert model.k_max("join", _items("short"), quality=True, selectivity=sel) == 1
        assert model.k_max("join", _items("long"), quality=True, selectivity=sel) == 1


def test_no_selectivity_is_unchanged(model):
    for regime in ("short", "long"):
        assert model.k_max("join", _items(regime), quality=True) == 1
        assert model.choose_k("join", _items(regime), quality=True) == \
            model.choose_k("join", _items(regime), quality=True, selectivity=None)


@pytest.mark.parametrize("sel", [None, 0.05, 0.5, 0.9])
def test_filters_and_maps_ignore_selectivity(model, sel):
    assert model.k_max("filter", _items("short"), quality=True, selectivity=sel) == 20
    assert model.k_max("filter", _items("long"), quality=True, selectivity=sel) == 1
    assert model.k_max("map", _items("short"), quality=True, selectivity=sel) == 1


def test_default_veto_uses_band_cap():
    veto = costmodel_quality_veto()
    assert veto("join", 4, 100.0) is True          # pooled cap 1
    assert veto("join", 4, 100.0, 0.3) is True     # low band
    assert veto("join", 4, 100.0, 0.8) is False    # high band, cap 6
    assert veto("join", 8, 100.0, 0.8) is True     # above the band cap
    assert veto("join", 4, 1600.0, 0.8) is True    # long rows: cap 1 in every band
    assert veto("map", 2, 40.0, 0.9) is True


def test_controller_passes_the_stage_own_pass_rate():
    seen = []
    upstream = SimpleNamespace(stage_id=1, block_tasks=False, operators=[], block_size=1)
    down = SimpleNamespace(stage_id=2, block_tasks=True, block_size=4,
                           operators=[BlockSemFilter("q", block_size=4)])
    config = AdaptiveStageConfig(
        enabled=True, quality_veto=lambda kind, k, t, sel=None: seen.append(sel) or False)
    c = AdaptiveStageController([upstream, down], config, token_length=len)
    c.formation_size(1, row("x"))

    def end(stage_id, passed, rows_in=1):
        return {"event": "task_end", "stage_id": stage_id, "rows_in": rows_in,
                "rows_out": passed, "rows_passed": passed, "outcome": "row"}

    # The managed stage's own tasks: 3 of 4 rows passed.
    c.observe(end(2, 3, rows_in=4))
    c.observe({**end(2, 0, rows_in=4), "outcome": "retry"})  # retries ignored
    for passed in (0, 0, 0, 1, 1, 1):  # upstream: to per-tuple, then back to block
        c.observe(end(1, passed))
    assert seen == [0.75]
    assert c.states[1].selectivity() == 0.75
