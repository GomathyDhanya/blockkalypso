"""Adaptive per-stage blocking (controller/adaptive_stage.py).

Correctness only; no performance or quality evaluation. Covers:

1. disabled -> output identical to the unmodified executor (golden
   fingerprints recorded from the executor before this change);
2. a mid-run switch never resizes a task already formed, queued or running;
3. the quality veto blocks a switch to the small block size;
4. hysteresis: one noisy block does not flip the mode.

Completion order is pinned (creation order) wherever exact block contents are
compared, because asyncio.wait's ``done`` set order is not stable
(test_block_membership_order.py).
"""

import asyncio
import copy
import hashlib
import json
from types import SimpleNamespace

import pytest

from vllm.kalypso import telemetry
from vllm.kalypso.controller.adaptive_stage import (
    BLOCK, STATIC, TUPLE, AdaptiveStageConfig, AdaptiveStageController,
    candidate_sizes, costmodel_quality_veto)
from vllm.kalypso.controller.plan import SemanticPlan
from vllm.kalypso.execution import pipeline_execution
from vllm.kalypso.query import Query
from vllm.kalypso.sem_ops.block_filter import BlockSemFilter

from .test_block_filter import row
from .test_block_membership_order import _downstream_blocks
from .test_telemetry import PIPELINES, RuleExecutor, _run


def _pin_completion_order(monkeypatch):
    real_wait = asyncio.wait

    async def ordered(fs, *args, **kwargs):
        done, pending = await real_wait(fs, *args, **kwargs)
        return sorted(done, key=lambda f: int(f.get_name().rsplit("-", 1)[1])), pending

    monkeypatch.setattr(pipeline_execution.asyncio, "wait", ordered)


def _h(x):
    return hashlib.sha256(json.dumps(x).encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# 1. Disabled == unmodified executor
# ---------------------------------------------------------------------------

# Recorded from the executor *before* the adaptive change (phase2/02 tree),
# same harness: test_telemetry._run with completion order pinned.
GOLDEN_UNMODIFIED = {
    "tuple_block": {"result": "86790b5ede3722dd", "prompts": "07844565db2b825f",
                    "blocks": [[1, 2, 4, 5], [7, 8, 10, 11]], "n_prompts": 14},
    "filter_blockjoin": {"result": "9a1f014c9995612c", "prompts": "a3b38cc63c131712",
                         "blocks": [[1], [1, 2], [2, 4], [4], [5], [5, 7], [7, 8], [8],
                                    [10], [10, 11], [11]], "n_prompts": 14},
    "block_block": {"result": "d6ca79aa0d6db142", "prompts": "65f5813d8d206b85",
                    "blocks": [[1, 2], [4, 5], [7, 8], [10, 11]], "n_prompts": 8},
    "filter_join_tuple": {"result": "5ea0d1eeee1406f2", "prompts": "b427de7961e16b32",
                          "blocks": [], "n_prompts": 44},
}


@pytest.mark.parametrize("explicit", [False, True], ids=["default", "enabled=False"])
@pytest.mark.parametrize("name", sorted(GOLDEN_UNMODIFIED))
def test_disabled_reproduces_unmodified_executor(name, explicit, fake_kv_manager, monkeypatch):
    monkeypatch.delenv("KALYPSO_ADAPTIVE_STAGE", raising=False)
    if explicit:
        real_init = SemanticPlan.__init__

        def init(self, executor, *a, **kw):
            real_init(self, executor, *a, adaptive=AdaptiveStageConfig(enabled=False), **kw)
        monkeypatch.setattr(SemanticPlan, "__init__", init)
    _pin_completion_order(monkeypatch)
    result, prompts = _run(PIPELINES[name], monkeypatch)
    got = {"result": _h([list(map(str, r)) for r in result]), "prompts": _h(prompts),
           "blocks": sorted(sorted(b) for b in _downstream_blocks(prompts)),
           "n_prompts": len(prompts)}
    assert got == GOLDEN_UNMODIFIED[name]


def test_disabled_creates_no_controller_or_subscription(fake_kv_manager, monkeypatch):
    monkeypatch.delenv("KALYPSO_ADAPTIVE_STAGE", raising=False)
    plan = SemanticPlan(RuleExecutor())
    assert plan.adaptive.enabled is False
    seen = []
    monkeypatch.setattr(telemetry, "subscribe", lambda fn: seen.append(fn))
    _run(PIPELINES["tuple_block"], monkeypatch)
    assert seen == []


def test_env_flag_enables(monkeypatch):
    monkeypatch.setenv("KALYPSO_ADAPTIVE_STAGE", "1")
    assert AdaptiveStageConfig.from_env().enabled
    monkeypatch.setenv("KALYPSO_ADAPTIVE_STAGE", "0")
    assert not AdaptiveStageConfig.from_env().enabled


# ---------------------------------------------------------------------------
# Unit-level controller fixtures
# ---------------------------------------------------------------------------

def _stages(block_size=4):
    upstream = SimpleNamespace(stage_id=1, block_tasks=False, operators=[], block_size=1)
    down = SimpleNamespace(stage_id=2, block_tasks=True, block_size=block_size,
                           operators=[BlockSemFilter("q", block_size=block_size)])
    return [upstream, down]


def _end(passed, rows_in=1, stage_id=1, outcome="row"):
    return {"event": "task_end", "stage_id": stage_id, "rows_in": rows_in,
            "rows_out": passed, "rows_passed": passed, "outcome": outcome}


def _controller(veto=lambda kind, k, tokens, selectivity=None: False, **cfg):
    config = AdaptiveStageConfig(enabled=True, quality_veto=veto, **cfg)
    c = AdaptiveStageController(_stages(), config, token_length=len)
    c.formation_size(1, row("x"))  # a row has reached the stage: regime known
    return c


# ---------------------------------------------------------------------------
# 4. Hysteresis
# ---------------------------------------------------------------------------

def test_needs_min_consecutive_observations():
    c = _controller(min_consecutive=3)
    c.observe(_end(0))
    c.observe(_end(0))
    assert c.states[1].mode == STATIC          # two low observations: no switch
    c.observe(_end(0))
    assert c.states[1].mode == TUPLE           # third: switch


def test_single_noisy_block_does_not_flip():
    c = _controller(min_consecutive=3)
    for _ in range(3):
        c.observe(_end(0))
    assert c.states[1].mode == TUPLE
    switches = len(c.decisions)
    # One block where every row passed, then the low rate resumes. The block
    # keeps the rolling rate >= 20% for several observations, but it is the
    # only observation that is itself above the threshold.
    c.observe(_end(4, rows_in=4, outcome="block"))
    for _ in range(10):
        c.observe(_end(0))
    assert c.states[1].mode == TUPLE
    assert len(c.decisions) == switches
    assert c.formation_size(1, row("y")) == 1


def test_sustained_high_rate_switches_back():
    c = _controller(min_consecutive=3)
    for _ in range(3):
        c.observe(_end(0))
    for _ in range(3):
        c.observe(_end(1))
    assert c.states[1].mode == BLOCK
    assert c.formation_size(1, row("z")) == 4


def test_dead_band_resets_streak():
    c = _controller(min_consecutive=3, low_pass_rate=0.05, high_pass_rate=0.2, window_tasks=1)
    c.observe(_end(0))
    c.observe(_end(0))
    c.observe(_end(1, rows_in=10))             # 10%: inside the band, resets
    c.observe(_end(0))
    assert c.states[1].mode == STATIC


def test_retries_and_other_stages_ignored():
    c = _controller(min_consecutive=1)
    c.observe(_end(0, outcome="retry"))
    c.observe(_end(0, stage_id=2))
    c.observe({"event": "task_start", "stage_id": 1})
    assert c.states[1].mode == STATIC and c.decisions == []


# ---------------------------------------------------------------------------
# 3. Quality veto
# ---------------------------------------------------------------------------

def _to_tuple_then_high(c):
    for _ in range(3):
        c.observe(_end(0))
    assert c.states[1].mode == TUPLE
    for _ in range(3):
        c.observe(_end(1))


def test_quality_veto_blocks_switch():
    c = _controller(veto=lambda kind, k, tokens, selectivity=None: True)
    _to_tuple_then_high(c)
    assert c.states[1].mode == TUPLE
    assert c.decisions[-1]["action"] == "veto"
    assert c.decisions[-1]["reason"] == "quality guard"
    assert c.formation_size(1, row("w")) == 1


def test_same_signal_without_veto_switches():
    c = _controller(veto=lambda kind, k, tokens, selectivity=None: False)
    _to_tuple_then_high(c)
    assert c.states[1].mode == BLOCK


def test_unknown_regime_vetoes():
    config = AdaptiveStageConfig(enabled=True, quality_veto=lambda *a: False)
    c = AdaptiveStageController(_stages(), config, token_length=len)
    for _ in range(3):
        c.observe(_end(0))
    for _ in range(3):
        c.observe(_end(1))
    assert c.states[1].mode == TUPLE
    assert "regime unknown" in c.decisions[-1]["reason"]


def test_default_veto_is_costmodel_choose_k_guard():
    # The committed v3 fit caps join and map at k=1 in every regime and
    # filters at 20 (short rows) / 1 (long rows).
    veto = costmodel_quality_veto()
    assert veto("join", 4, 40.0) is True
    assert veto("map", 4, 40.0) is True
    assert veto("filter", 4, 40.0) is False
    assert veto("filter", 4, 1600.0) is True


# ---------------------------------------------------------------------------
# 2. Mid-run switch never resizes a formed / queued / running task
# ---------------------------------------------------------------------------

def test_mid_run_switch_does_not_resize_formed_tasks(fake_kv_manager, monkeypatch):
    # Upstream per-tuple filter: rows 0-9 pass, 10-23 fail, 24-35 pass.
    # Downstream block filter (k=4). With low=0.5 the controller switches the
    # downstream stage to per-tuple during the failing run; rows 8 and 9 are
    # then still in a forming block, which must complete at its formation
    # size (4) together with rows 24 and 25.
    ex = RuleExecutor()
    source = [row(f"row{i} {'kk' if i < 10 or i >= 24 else 'xx'}", i, ex) for i in range(36)]
    monkeypatch.setattr("vllm.kalypso.controller.plan.data._data_source",
                        lambda *a: iter(source))
    _pin_completion_order(monkeypatch)

    enqueued, ran = {}, {}
    from vllm.kalypso.controller.stage import Stage
    orig_enqueue, orig_run = Stage.enqueue, Stage.run_task

    def enqueue(self, task):
        if isinstance(task.ctx, list) and self.stage_id == 2:
            enqueued[id(task)] = (len(task.ctx), [c.state.idx for c in task.ctx])
        return orig_enqueue(self, task)

    async def run_task(self, task):
        if isinstance(task.ctx, list) and self.stage_id == 2:
            ran[id(task)] = (len(task.ctx), [c.state.idx for c in task.ctx])
        return await orig_run(self, task)

    monkeypatch.setattr(Stage, "enqueue", enqueue)
    monkeypatch.setattr(Stage, "run_task", run_task)

    config = AdaptiveStageConfig(enabled=True, low_pass_rate=0.5, high_pass_rate=0.95,
                                 min_consecutive=3, window_tasks=16,
                                 quality_veto=lambda *a: False)
    plan = SemanticPlan(ex, adaptive=config)
    ops = [{"op": "sem_filter", "args": {"prompt": "Q_KEEP"}},
           {"op": "sem_filter", "args": {"prompt": "Q_KEEP again", "block_size": 4}}]
    out = asyncio.run(plan.execute(None, Query(copy.deepcopy(ops), "left")))
    controller = plan.plan_executor.pipeline_executor.last_controller

    # Exactly one switch, to per-tuple, after the first 3 blocks started forming.
    switches = [d for d in controller.decisions if d["action"] == "switch"]
    assert [(d["mode"], d["after_formations"]) for d in switches] == [(TUPLE, 3)]
    assert [s for _, s in controller.formations] == [4, 4, 4] + [1] * 10

    # Every task ran with exactly the rows it was formed with.
    assert ran == enqueued
    sizes = sorted(enqueued.values(), key=lambda v: v[1])
    # The block forming at switch time kept its size 4 (rows 8, 9 + 24, 25).
    assert (4, [8, 9, 24, 25]) in sizes
    assert [s for s, _ in sizes].count(4) == 3
    assert [s for s, _ in sizes].count(1) == 10
    # Output unaffected in content: every passing row comes out once.
    assert sorted(r.state.idx for r in out) == list(range(10)) + list(range(24, 36))


# ---------------------------------------------------------------------------
# 6. Candidate sizes: the search's power-of-two set
# ---------------------------------------------------------------------------

def _switch_targets(block_size, small_block_size):
    config = AdaptiveStageConfig(enabled=True, small_block_size=small_block_size,
                                 quality_veto=lambda *a: False)
    state = AdaptiveStageController(_stages(block_size), config, token_length=len).states[1]
    return state, {state.size(TUPLE), state.size(BLOCK)}


def test_candidates_are_the_search_pow2_set():
    from vllm.kalypso.benchmark.eval.costmodel import POW2_CANDIDATES
    assert candidate_sizes(64) == POW2_CANDIDATES
    assert candidate_sizes(10) == (1, 2, 4, 8)
    assert candidate_sizes(3) == (1, 2)
    assert candidate_sizes(1) == (1,)


@pytest.mark.parametrize("block_size", range(1, 41))
@pytest.mark.parametrize("small_block_size", [2, 3, 4, 5, 6, 8, 12, 16, 40])
def test_every_switch_target_is_a_power_of_two(block_size, small_block_size):
    from vllm.kalypso.benchmark.eval.costmodel import POW2_CANDIDATES
    state, targets = _switch_targets(block_size, small_block_size)
    assert targets <= set(POW2_CANDIDATES)
    assert targets <= set(state.candidate_sizes)
    assert max(targets) <= min(block_size, small_block_size)


def test_non_pow2_query_size_now_gives_pow2_block():
    # Before: min(4, 3) = 3. Now the largest power of two <= 3.
    assert _switch_targets(3, 4)[0].small_size == 2
    assert _switch_targets(10, 4)[0].small_size == 4  # default unchanged when size >= 4
    assert _switch_targets(10, 6)[0].small_size == 4


def test_real_quality_veto_rejects_size_from_new_list():
    # Long rows: the committed fit caps filters at k = 1 there, so the
    # candidate 4 must be vetoed by the unchanged costmodel guard.
    config = AdaptiveStageConfig(enabled=True, quality_veto=costmodel_quality_veto())
    c = AdaptiveStageController(_stages(8), config, token_length=lambda text: 1600)
    c.formation_size(1, row("x"))
    assert c.states[1].small_size in c.states[1].candidate_sizes
    _to_tuple_then_high(c)
    assert c.states[1].mode == TUPLE
    assert c.decisions[-1]["action"] == "veto"
    assert c.decisions[-1]["reason"] == "quality guard"
    assert c.decisions[-1]["size"] == 4
    # Same signal on short rows (cap 20): the same candidate is allowed.
    c = AdaptiveStageController(_stages(8), config, token_length=lambda text: 40)
    c.formation_size(1, row("x"))
    _to_tuple_then_high(c)
    assert c.states[1].mode == BLOCK and c.formation_size(1, row("y")) == 4


# ---------------------------------------------------------------------------
# 7. Over-cap fallback: step down through the power-of-two candidates
# ---------------------------------------------------------------------------

def _block_controller(veto, block_size=16, small_block_size=8, row_tokens=len):
    config = AdaptiveStageConfig(enabled=True, small_block_size=small_block_size,
                                 quality_veto=veto)
    c = AdaptiveStageController(_stages(block_size), config, token_length=row_tokens)
    c.formation_size(1, row("x"))
    return c


def test_vetoed_top_candidate_steps_down_not_to_one():
    asked = []

    def veto(kind, k, tokens, selectivity=None):
        asked.append(k)
        return k > 2
    c = _block_controller(veto)
    _to_tuple_then_high(c)
    assert c.states[1].mode == BLOCK
    assert asked == [8, 4, 2]                      # largest first, stops at the first pass
    assert c.decisions[-1]["action"] == "switch"
    assert c.decisions[-1]["size"] == 2
    assert c.decisions[-1]["vetoed"] == [8, 4]
    assert c.formation_size(1, row("y")) == 2


def test_real_veto_steps_down_to_the_join_band_cap():
    # Short-row join at selectivity 0.8: band cap 6, so 8 is vetoed and 4 passes.
    c = _block_controller(costmodel_quality_veto(), row_tokens=lambda text: 100)
    c.states[1].kind = "join"
    for _ in range(16):  # a real reading: a full own window (window_tasks) at 0.8
        c.observe({"event": "task_end", "stage_id": 2, "rows_in": 10, "rows_out": 8,
                   "rows_passed": 8, "outcome": "block"})
    _to_tuple_then_high(c)
    assert c.states[1].mode == BLOCK and c.formation_size(1, row("y")) == 4
    assert c.decisions[-1]["vetoed"] == [8]


def test_all_candidates_vetoed_stays_per_tuple():
    c = _block_controller(lambda kind, k, tokens, selectivity=None: True)
    _to_tuple_then_high(c)
    assert c.states[1].mode == TUPLE
    assert c.decisions[-1]["action"] == "veto"
    assert c.decisions[-1]["reason"] == "quality guard"
    assert c.decisions[-1]["vetoed"] == [8, 4, 2]
    assert c.decisions[-1]["size"] == 8            # reports the preferred size
    assert c.formation_size(1, row("y")) == 1


def test_step_down_cost_is_a_few_cpu_veto_calls():
    # Worst case: one veto call per candidate in (1, small_size], i.e. at most
    # five for the largest set (32, 16, 8, 4, 2); no GPU call, no new signal.
    c = _block_controller(lambda *a: True, block_size=64, small_block_size=32)
    _to_tuple_then_high(c)
    assert c.veto_checks == 5
    assert c.decisions[-1]["vetoed"] == [32, 16, 8, 4, 2]


def test_switch_back_recomputes_size():
    allowed = {"max": 2}
    c = _block_controller(lambda kind, k, tokens, selectivity=None: k > allowed["max"])
    _to_tuple_then_high(c)
    assert c.formation_size(1, row("a")) == 2
    allowed["max"] = 8
    for _ in range(18):  # 16 to flush the rolling window, then 3 votes below 5%
        c.observe(_end(0))
    assert c.states[1].mode == TUPLE
    for _ in range(6):  # the rolling rate reaches 20% at the 4th, then 3 votes
        c.observe(_end(1))
    assert c.states[1].mode == BLOCK
    assert c.formation_size(1, row("b")) == 8


# ---------------------------------------------------------------------------
# 8. Cold start: a join's switch up waits for a real selectivity reading
# ---------------------------------------------------------------------------

def _join_controller(veto=None, block_size=8, small_block_size=8):
    calls = []
    real = veto or costmodel_quality_veto()

    def recording(kind, k, tokens, selectivity=None):
        calls.append((k, selectivity))
        return real(kind, k, tokens, selectivity)
    c = _block_controller(recording, block_size=block_size,
                          small_block_size=small_block_size, row_tokens=lambda text: 100)
    c.states[1].kind = "join"
    return c, calls


def _own(c, passed, rows_in=1):
    c.observe({"event": "task_end", "stage_id": 2, "rows_in": rows_in,
               "rows_out": passed, "rows_passed": passed, "outcome": "row"})


def _drift_recovery(c, n=40):
    """The real drift shape (P2-compare): the upstream filter's first rows
    fail, then it recovers, all before any join task has finished."""
    for _ in range(3):
        c.observe(_end(0))
    for _ in range(n):
        c.observe(_end(1))


def test_switch_to_tuple_fires_immediately_without_a_reading():
    c, calls = _join_controller()
    for _ in range(3):
        c.observe(_end(0))
    assert c.states[1].mode == TUPLE
    assert c.states[1].selectivity() is None and not c.has_selectivity_reading(c.states[1])
    assert [d["action"] for d in c.decisions] == ["switch"]
    assert calls == [] and c.deferrals == 0


def test_recovery_with_no_join_completions_defers_not_caps():
    c, calls = _join_controller()
    _drift_recovery(c)
    assert c.states[1].mode == TUPLE                      # stays at its current size
    assert calls == []                                   # the pooled cap was never asked
    assert [d["action"] for d in c.decisions] == ["switch"]  # no veto recorded
    assert c.deferrals == 40 - 2                         # every vote after the 3rd


def test_partial_window_still_defers():
    c, calls = _join_controller()
    for _ in range(15):          # one short of window_tasks
        _own(c, 1)
    _drift_recovery(c, n=10)
    assert calls == [] and c.states[1].mode == TUPLE and c.deferrals > 0


def test_switches_once_real_data_allows_it():
    # Short-row join, own rate 0.75 (band cap 6): preferred 8 is vetoed, 4 passes.
    c, calls = _join_controller()
    _drift_recovery(c, n=5)
    assert calls == [] and c.states[1].mode == TUPLE
    pattern = [1, 1, 1, 0] * 4   # 16 own tasks at 0.75
    for passed in pattern:
        _own(c, passed)
    c.observe(_end(1))           # the next block vote: streak was kept, evaluates at once
    assert c.states[1].mode == BLOCK
    assert calls == [(8, 0.75), (4, 0.75)]
    assert c.decisions[-1]["action"] == "switch" and c.decisions[-1]["vetoed"] == [8]
    assert c.formation_size(1, row("y")) == 4


def test_real_low_selectivity_reading_still_vetoes():
    c, calls = _join_controller()
    _drift_recovery(c, n=5)
    for passed in [1, 0, 0, 0] * 4:   # 0.25: band cap 1
        _own(c, passed)
    c.observe(_end(1))
    assert c.states[1].mode == TUPLE
    assert c.decisions[-1]["action"] == "veto" and c.decisions[-1]["vetoed"] == [8, 4, 2]
    assert all(sel == 0.25 for _, sel in calls)


def test_filters_are_not_deferred():
    asked = []
    c = _block_controller(lambda kind, k, t, sel=None: asked.append(sel) or False)
    _to_tuple_then_high(c)       # filter stage, no own data: decides at once, as before
    assert c.states[1].mode == BLOCK and asked == [None] and c.deferrals == 0
