"""KV feasibility gate on choose_k / choose_pipeline (costmodel ``kv_budget_tokens``).

A safety check for pathological inputs, not a resource model (see the note
above costmodel.Model). Checks: clear overflows are rejected and listed; sizes
well within budget are unaffected; with the real Session 1 budget no existing
choice changes.
"""

import json
from dataclasses import fields

import pytest

from vllm.kalypso.benchmark.eval import costmodel as cm

from .test_costmodel_candidate_set import RECORDED, _pipe

FITTED = cm.RESULTS / "analysis/costmodel.json"

# Session 1 (L40S, Qwen2.5-7B-Instruct): Kalypso's traced KV capacity
# 25,631,373,374 bytes (23.87 GiB; kv_stage_registered.total_capacity_bytes in
# research/baseline/results/gpu/research-session1/E1-*/trace-on.jsonl.gz) over
# 57,344 bytes/token (2 x 28 layers x 4 KV heads x 128 dims x 2 bytes).
SESSION1_BUDGET_TOKENS = 25_631_373_374 / 57_344


@pytest.fixture(scope="module")
def model():
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


SHORT = cm.Items(400, 31.0, 120.0)
LONG = cm.Items(400, 1600.0, 2400.0)


# --- clear overflows are rejected ----------------------------------------------

def test_budget_rejects_sizes_that_do_not_fit(model):
    k_max = model.k_max("filter", SHORT)
    budget = model.call_kv_tokens("filter", SHORT, 4)  # exactly fits k = 4
    res = model.choose_k("filter", SHORT, kv_budget_tokens=budget)
    assert res["kv_rejected"] == list(range(5, k_max + 1))
    assert res["k"] <= 4


def test_tiny_budget_falls_back_to_per_tuple(model):
    res = model.choose_k("filter", SHORT, kv_budget_tokens=10)
    assert res["k"] == 1
    assert 1 not in res["kv_rejected"] and res["kv_rejected"]  # k = 1 never rejected


def test_pipeline_gate_is_per_stage_and_recorded(model):
    pipe = _pipe("filter_join", 2)
    items = model.stage_items(pipe, pipe.rows)
    # Stage 0 (filter) may only use k <= 2; stage 1 (join) gets plenty.
    tight = model.call_kv_tokens("filter", items[0], 2)
    res = model.choose_pipeline(pipe, kv_budget_tokens=[tight, SESSION1_BUDGET_TOKENS])
    assert res["blocks"][0] <= 2
    assert res["kv_rejected"] and all(i == 0 and k > 2 for i, k in res["kv_rejected"])


# --- within budget: unaffected --------------------------------------------------

def test_large_budget_changes_nothing(model):
    plain = model.choose_k("filter", SHORT)
    gated = model.choose_k("filter", SHORT, kv_budget_tokens=1e12)
    assert gated.pop("kv_rejected") == []
    assert gated == plain


def test_no_budget_adds_no_field(model):
    assert "kv_rejected" not in model.choose_k("filter", SHORT)
    assert "kv_rejected" not in model.choose_pipeline(_pipe("filters", 2))


# --- no existing choice changes at the real budget ---------------------------------

@pytest.mark.parametrize("family,depth", sorted(RECORDED))
def test_session1_budget_changes_no_recorded_pipeline_choice(model, family, depth):
    pipe = _pipe(family, depth)
    ex_blocks, ex_lat, p2_blocks, p2_lat = RECORDED[(family, depth)]
    for candidate_set, blocks, lat in (("exhaustive", ex_blocks, ex_lat),
                                       ("pow2", p2_blocks, p2_lat)):
        res = model.choose_pipeline(pipe, candidate_set=candidate_set,
                                    kv_budget_tokens=SESSION1_BUDGET_TOKENS)
        assert res["kv_rejected"] == []
        assert res["blocks"] == blocks
        assert res["latency_s"] == pytest.approx(lat, rel=1e-9)


@pytest.mark.parametrize("kind,items", [("filter", SHORT), ("join", SHORT), ("map", SHORT),
                                        ("filter", LONG), ("join", LONG)])
@pytest.mark.parametrize("quality", [False, True])
def test_session1_budget_changes_no_choose_k(model, kind, items, quality):
    plain = model.choose_k(kind, items, quality=quality)
    gated = model.choose_k(kind, items, quality=quality, kv_budget_tokens=SESSION1_BUDGET_TOKENS)
    assert gated.pop("kv_rejected") == []
    assert gated == plain


# --- budget from KVMemoryManager; cache key ------------------------------------------

def test_budget_from_kv_memory_manager(fake_kv_manager):
    # conftest: capacity 10,000,000 bytes, 1 byte per token.
    assert cm.kv_budget_tokens() == 10_000_000
    fake_kv_manager.register_stage(7, 0.25)
    assert cm.kv_budget_tokens(7) == 2_500_000


def test_budget_is_part_of_cache_key(model):
    model.choose_k("filter", SHORT, cache=True, kv_budget_tokens=1e12)
    small = model.choose_k("filter", SHORT, cache=True, kv_budget_tokens=10)
    assert cm.DECISION_CACHE.stats.hits == 0
    assert small["k"] == 1
