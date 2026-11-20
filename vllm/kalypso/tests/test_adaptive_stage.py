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
    costmodel_quality_veto)
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


def _controller(veto=lambda kind, k, tokens: False, **cfg):
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
    c = _controller(veto=lambda kind, k, tokens: True)
    _to_tuple_then_high(c)
    assert c.states[1].mode == TUPLE
    assert c.decisions[-1]["action"] == "veto"
    assert c.decisions[-1]["reason"] == "quality guard"
    assert c.formation_size(1, row("w")) == 1


def test_same_signal_without_veto_switches():
    c = _controller(veto=lambda kind, k, tokens: False)
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
