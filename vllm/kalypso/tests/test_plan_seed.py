"""The planner's KV-capacity sample is seeded (BASELINE_AUDIT.md §H / §4.1).

``SemanticPlan.build`` sizes each non-final stage's minimum KV capacity from
the largest task budget in a sample of at most ESTIMATED_CTX_SAMPLE_SIZE
input contexts. With more inputs than that, the sample used to come from the
unseeded global ``random`` module, so the registered capacities could differ
between runs of the same query.
"""

import random

from vllm.kalypso.budget import KVMemoryManager
from vllm.kalypso.controller.plan import SemanticPlan
from vllm.kalypso.sem_ops import ops

from .test_block_filter import row

N_ROWS = 3 * SemanticPlan.ESTIMATED_CTX_SAMPLE_SIZE


def _source():
    # Row lengths vary, so the sample's largest budget depends on the draw.
    return [row("x" * (7 * i), i) for i in range(N_ROWS)]


def _min_caps(plan_seed, global_seed):
    random.seed(global_seed)  # stands in for whatever else used `random` first
    plan = SemanticPlan(executor=None, plan_seed=plan_seed)
    physical = [ops.SemFilter("first"), ops.BlockSemFilter("second", block_size=4)]
    plan.build(_source(), physical)
    return dict(KVMemoryManager.get_instance()._stage_min_capacity)


def test_default_seed_gives_identical_capacities(fake_kv_manager):
    caps = [_min_caps(SemanticPlan.DEFAULT_PLAN_SEED, g) for g in range(5)]
    assert all(c == caps[0] for c in caps)


def test_different_seeds_can_differ(fake_kv_manager):
    caps = {tuple(sorted(_min_caps(s, 0).items())) for s in range(5)}
    assert len(caps) > 1


def test_unseeded_still_varies(fake_kv_manager):
    # plan_seed=None keeps the old behaviour; this is the hazard reproduced.
    caps = {tuple(sorted(_min_caps(None, g).items())) for g in range(5)}
    assert len(caps) > 1
