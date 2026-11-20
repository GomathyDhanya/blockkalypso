"""Adaptive selectivity plumbing (Phase 2 step C): gated off by default.

Hard requirement: with the flag off, every existing call is bit-for-bit
identical, even when the model holds estimates. With it on (tests only), the
stored estimate moves toward repeated observations.
"""

import json
import re
from dataclasses import fields
from pathlib import Path

import pytest

from vllm.kalypso.benchmark.eval import costmodel as cm

from .test_costmodel_candidate_set import RECORDED, _pipe

FITTED = cm.RESULTS / "analysis/costmodel.json"


def _model():
    d = json.loads(FITTED.read_text())
    names = {f.name for f in fields(cm.KindParams)}
    kinds = {k: cm.KindParams(**{n: v for n, v in p.items() if n in names})
             for k, p in d["kinds"].items()}
    return cm.Model(g=d["g"], alpha=d["alpha"], beta=d["beta"], gamma=d["gamma"],
                    delta=d["delta"], tail=d["tail"], kinds=kinds,
                    f1_slack=d["f1_slack"], lam=d["lambda"], retry=d["retry"])


def _primed(pipe):
    """A model that has seen live observations for every stage of ``pipe``."""
    m = _model()
    for i in range(len(pipe.ops)):
        for _ in range(5):
            m.observe_selectivity(pipe.ops, i, 0.05, prior=pipe.true_share(i))
    return m


# --- flag off: bit-for-bit identical ----------------------------------------------

@pytest.mark.parametrize("family,depth", sorted(RECORDED))
def test_flag_off_choose_pipeline_bit_identical(family, depth):
    pipe = _pipe(family, depth)
    fresh, primed = _model(), _primed(pipe)
    assert primed.selectivity_estimates  # it does hold estimates
    for candidate_set in ("exhaustive", "pow2"):
        want = fresh.choose_pipeline(pipe, candidate_set=candidate_set)
        assert primed.choose_pipeline(pipe, candidate_set=candidate_set) == want
        assert primed.choose_pipeline(pipe, candidate_set=candidate_set,
                                      adaptive_selectivity=False) == want
    ex_blocks, ex_lat, p2_blocks, p2_lat = RECORDED[(family, depth)]
    assert primed.choose_pipeline(pipe)["blocks"] == ex_blocks
    assert primed.choose_pipeline(pipe)["latency_s"] == pytest.approx(ex_lat, rel=1e-9)


@pytest.mark.parametrize("family,depth", sorted(RECORDED))
def test_flag_off_predict_pipeline_bit_identical(family, depth):
    pipe = _pipe(family, depth)
    blocks = RECORDED[(family, depth)][0]
    assert _primed(pipe).predict_pipeline(pipe, blocks) == _model().predict_pipeline(pipe, blocks)


def test_flag_on_without_estimates_is_identical():
    pipe = _pipe("filter_join", 2)
    m = _model()
    assert m.adapted_pipeline(pipe) is pipe
    assert m.choose_pipeline(pipe, adaptive_selectivity=True) == m.choose_pipeline(pipe)


def test_choose_k_has_no_selectivity_input_to_adapt():
    items = cm.Items(400, 31.0, 120.0)
    assert _primed(_pipe("filters", 2)).choose_k("filter", items) == \
        _model().choose_k("filter", items)


# --- flag on: the estimate moves ---------------------------------------------------

def test_estimate_moves_toward_repeated_observations():
    m = _model()
    ops = ("filter", "join")
    values = [m.observe_selectivity(ops, 0, 0.1, prior=0.9) for _ in range(30)]
    assert values[0] == pytest.approx(0.9 + 0.2 * (0.1 - 0.9))  # one EWMA step
    assert all(b < a for a, b in zip(values, values[1:]))       # monotone toward 0.1
    assert values[-1] == pytest.approx(0.1, abs=0.001)
    assert m.selectivity_estimates == {"filter+join#0": values[-1]}


def test_first_observation_without_prior_starts_at_it():
    m = _model()
    assert m.observe_selectivity(("filter",), 0, 0.4) == 0.4
    with pytest.raises(ValueError):
        m.observe_selectivity(("filter",), 0, 0.4, weight=0.0)


def test_flag_on_uses_estimates_in_planning():
    pipe = _pipe("filter_join", 2)
    m = _model()
    blocks = (1, 1)
    before = m.predict_pipeline(pipe, blocks, adaptive_selectivity=True)
    for _ in range(30):  # the leading filter matches far fewer rows than planned
        m.observe_selectivity(pipe.ops, 0, 0.02, prior=pipe.true_share(0))
    adapted = m.adapted_pipeline(pipe)
    assert adapted.true_share(0) == pytest.approx(0.02, abs=0.01)
    assert adapted.true_share(1) == pipe.true_share(1)   # no estimate: unchanged
    after = m.predict_pipeline(pipe, blocks, adaptive_selectivity=True)
    assert after["latency_s"] < before["latency_s"]       # fewer rows reach the join
    assert m.predict_pipeline(pipe, blocks) == before     # flag off: unchanged


# --- not enabled anywhere -----------------------------------------------------------

def test_no_source_turns_the_flag_on():
    root = Path(cm.__file__).resolve().parents[3]  # vllm/
    hits = [p for p in list(root.rglob("*.py")) + list(Path(root.parent, "research").rglob("*.py"))
            if "tests" not in p.parts
            and re.search(r"adaptive_selectivity\s*=\s*True", p.read_text(errors="ignore"))]
    assert hits == []
