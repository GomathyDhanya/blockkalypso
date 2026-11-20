"""Bounded decision cache for choose_k / choose_pipeline (costmodel.DECISION_CACHE).

Correctness only. The cache is opt-in and present for completeness: the
search is offline and mostly sub-10 ms (see the note above DecisionCache in
costmodel.py), so no speed claim is tested or made here.
"""

import json
from dataclasses import fields

import pytest

from vllm.kalypso.benchmark.eval import costmodel as cm

FITTED = cm.RESULTS / "analysis/costmodel.json"


def _model():
    d = json.loads(FITTED.read_text())
    names = {f.name for f in fields(cm.KindParams)}
    kinds = {k: cm.KindParams(**{n: v for n, v in p.items() if n in names})
             for k, p in d["kinds"].items()}
    return cm.Model(g=d["g"], alpha=d["alpha"], beta=d["beta"], gamma=d["gamma"],
                    delta=d["delta"], tail=d["tail"], kinds=kinds,
                    f1_slack=d["f1_slack"], lam=d["lambda"], retry=d["retry"])


@pytest.fixture(autouse=True)
def _fresh_cache():
    cm.DECISION_CACHE.clear()
    yield
    cm.DECISION_CACHE.clear()


def _stats():
    return cm.DECISION_CACHE.stats.as_dict()


SHORT = cm.Items(400, 31.0, 120.0)
LONG = cm.Items(400, 1600.0, 2400.0)
PIPE = cm.Pipeline(("filter", "join"), 400, 31.0, 120.0, 0.5, 32, 31.0, 120.0)


# --- same inputs twice: identical result, one hit ---------------------------

def test_choose_k_same_inputs_hit():
    m = _model()
    first = m.choose_k("filter", SHORT, cache=True)
    second = m.choose_k("filter", SHORT, cache=True)
    assert second == first
    assert _stats() == {"hits": 1, "misses": 1, "evictions": 0}


def test_choose_pipeline_same_inputs_hit():
    m = _model()
    first = m.choose_pipeline(PIPE, cache=True)
    second = m.choose_pipeline(PIPE, cache=True)
    assert second == first
    assert _stats()["hits"] == 1 and _stats()["misses"] == 1


def test_near_identical_inputs_share_a_bucket():
    m = _model()
    a = m.choose_k("filter", cm.Items(400, 30.9, 119.8), cache=True)
    b = m.choose_k("filter", cm.Items(400.2, 31.1, 120.3), cache=True)
    assert a == b
    assert _stats()["hits"] == 1


def test_cache_off_by_default_and_unchanged():
    m = _model()
    assert m.choose_k("filter", SHORT) == m.choose_k("filter", SHORT, cache=True)
    assert m.choose_pipeline(PIPE) == m.choose_pipeline(PIPE, cache=True)
    assert _stats()["misses"] == 2 and _stats()["hits"] == 0  # uncached calls not counted


def test_hit_returns_a_copy():
    m = _model()
    first = m.choose_pipeline(PIPE, cache=True)
    first["stages"].clear()
    assert m.choose_pipeline(PIPE, cache=True)["stages"]


# --- different inputs miss ---------------------------------------------------

def test_different_regime_misses():
    m = _model()
    m.choose_k("filter", SHORT, cache=True)
    m.choose_k("filter", LONG, cache=True)
    assert _stats() == {"hits": 0, "misses": 2, "evictions": 0}


def test_different_item_count_misses():
    m = _model()
    m.choose_k("filter", SHORT, cache=True)
    m.choose_k("filter", cm.Items(401, 31.0, 120.0), cache=True)
    assert _stats()["hits"] == 0 and _stats()["misses"] == 2


def test_different_candidate_mode_misses():
    m = _model()
    pipe3 = cm.Pipeline(("filter", "join", "filter"), 400, 31.0, 120.0, 0.5, 32, 31.0, 120.0)
    exhaustive = m.choose_pipeline(pipe3, cache=True)
    pow2 = m.choose_pipeline(pipe3, candidate_set="pow2", cache=True)
    assert _stats()["hits"] == 0 and _stats()["misses"] == 2
    # Recorded in optimizer_scaling.md: exhaustive [1, 30, 2], pow2 [1, 32, 2].
    assert exhaustive["blocks"] == (1, 30, 2) and pow2["blocks"] == (1, 32, 2)


def test_different_kind_and_explicit_candidates_miss():
    m = _model()
    m.choose_k("filter", SHORT, cache=True)
    m.choose_k("join", SHORT, cache=True)
    m.choose_k("filter", SHORT, candidates=[1, 2, 4], cache=True)
    assert _stats()["hits"] == 0 and _stats()["misses"] == 3


# --- refit invalidates -------------------------------------------------------

def test_refit_invalidates_old_entries():
    m = _model()
    old = m.choose_k("filter", SHORT, cache=True)
    old_fp = m.fingerprint()
    # A refit, as main() does after fit(): parameters change in place.
    m.alpha *= 10
    m.kinds["filter"].measured_k["short"] = [1, 2]
    assert m.fingerprint() != old_fp
    new = m.choose_k("filter", SHORT, cache=True)
    assert _stats()["hits"] == 0 and _stats()["misses"] == 2
    assert new == m.choose_k("filter", SHORT)  # matches a fresh uncached computation
    assert new["k"] != old["k"]  # the stale pick (k=3, k_max 20) was not reused
    assert new["k_max"] == 2


def test_separately_fitted_identical_models_share_entries():
    a, b = _model(), _model()
    assert a.fingerprint() == b.fingerprint()
    a.choose_k("filter", SHORT, cache=True)
    b.choose_k("filter", SHORT, cache=True)
    assert _stats()["hits"] == 1


# --- bounded -------------------------------------------------------------------

def test_lru_is_bounded(monkeypatch):
    monkeypatch.setattr(cm, "DECISION_CACHE", cm.DecisionCache(max_size=2))
    m = _model()
    for n in (100, 200, 300):
        m.choose_k("filter", cm.Items(n, 31.0, 120.0), cache=True)
    assert len(cm.DECISION_CACHE) == 2
    assert cm.DECISION_CACHE.stats.evictions == 1
    m.choose_k("filter", cm.Items(100, 31.0, 120.0), cache=True)  # evicted: misses
    assert cm.DECISION_CACHE.stats.hits == 0


# --- automatic activation (Phase 2 step D) -------------------------------------------

PIPE4 = cm.Pipeline(("filter", "filter", "join", "filter"), 400, 31.4, 120.0, 0.5,
                    32, 31.0, 120.0)


def _no_key(monkeypatch):
    """Fail if a call builds a cache key or touches the cache."""
    def boom(*a, **k):
        raise AssertionError("cache touched")
    monkeypatch.setattr(cm.Model, "fingerprint", boom)
    monkeypatch.setattr(cm.DECISION_CACHE, "get_or_compute", boom)


@pytest.mark.parametrize("ops", [("filter",), ("filter", "join"), ("filter", "join", "filter")])
def test_up_to_three_stages_never_touch_the_cache(monkeypatch, ops):
    pipe = cm.Pipeline(ops, 400, 31.4, 120.0, 0.5, 32, 31.0, 120.0)
    want = _model().choose_pipeline(pipe, cache=False)
    _no_key(monkeypatch)  # no fingerprint, no lookup: the uncached path exactly
    assert _model().choose_pipeline(pipe, candidate_set="exhaustive") == want
    assert len(cm.DECISION_CACHE) == 0


def test_four_stages_use_the_cache_automatically():
    m = _model()
    first = m.choose_pipeline(PIPE4, candidate_set="pow2")
    second = m.choose_pipeline(PIPE4, candidate_set="pow2")
    assert _stats() == {"hits": 1, "misses": 1, "evictions": 0}
    assert first == second


def test_automatic_cache_is_bit_identical_to_uncached():
    # Exact-input key: the fractional 31.4 tokens are not rounded, unlike cache=True.
    m = _model()
    auto = m.choose_pipeline(PIPE4, candidate_set="pow2")
    off = m.choose_pipeline(PIPE4, candidate_set="pow2", cache=False)
    assert auto == off
    assert _stats()["misses"] == 1  # cache=False did not count


def test_explicit_flags_still_win(monkeypatch):
    m = _model()
    m.choose_pipeline(PIPE, cache=True)                          # 2 stages, forced on
    assert _stats()["misses"] == 1
    m.choose_pipeline(PIPE4, candidate_set="pow2", cache=False)  # 4 stages, forced off
    assert _stats()["misses"] == 1
    assert cm.AUTO_CACHE_MIN_STAGES == 4


def test_choose_k_never_caches_by_default(monkeypatch):
    _no_key(monkeypatch)
    _model().choose_k("filter", SHORT)
