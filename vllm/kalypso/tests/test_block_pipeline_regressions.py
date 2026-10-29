"""Behavioral regressions for block/tuple pipeline equivalence."""

import asyncio
import re

from vllm.kalypso.controller.plan import SemanticPlan
from vllm.kalypso.sem_ops import ops
from vllm.kalypso.sem_ops.block_filter import build_single_messages
from .test_block_filter import FakeOut, row


class YesExecutor:
    def __init__(self):
        self.prompts = []

    async def execute(self, raw_request, prompt, max_tokens, pin=False, priority=0):
        body = prompt[-1]["content"]
        self.prompts.append(body)
        ids = re.findall(r"^(\d+)\. ", body, re.M)
        return FakeOut("\n".join(f"{i}: YES" for i in ids) if ids else "true")


def test_planner_preserves_each_block_size(fake_kv_manager):
    ex = YesExecutor()
    source = [row(str(i), i, ex) for i in range(6)]
    plan = SemanticPlan(ex)
    physical = plan.parse_ops([
        {"op": "sem_filter", "args": {"prompt": "first", "block_size": 3}},
        {"op": "sem_filter", "args": {"prompt": "second", "block_size": 2}},
    ])
    built = plan.build(source, physical)
    assert len(built) == 1
    assert [s.block_size for s in built[0]] == [3, 2]
    result, stats = asyncio.run(plan.plan_executor.execute(source, built))
    assert len(result) == 6
    assert stats[0][built[0][1].stage_id]["input"] == 3


def test_tuple_producer_accumulates_block_filter(fake_kv_manager):
    ex = YesExecutor()
    source = [row(str(i), i, ex) for i in range(5)]
    plan = SemanticPlan(ex)
    physical = [ops.SemFilter("first"), ops.BlockSemFilter("second", block_size=2)]
    built = plan.build(source, physical)
    result, stats = asyncio.run(plan.plan_executor.execute(source, built))
    assert len(result) == 5
    assert stats[0][built[0][1].stage_id]["input"] == 3
    assert physical[1].stats.block_calls == 2


def test_streamed_join_obeys_token_limit(fake_kv_manager, monkeypatch):
    ex = YesExecutor()
    source = [row("left", i, ex) for i in range(2)]
    right = [row("right " * 100, i) for i in range(3)]
    monkeypatch.setattr("vllm.kalypso.controller.plan.data._data_source",
                        lambda *args: iter(right))
    plan = SemanticPlan(ex)
    physical = plan.parse_ops([
        {"op": "sem_filter", "args": {"prompt": "first"}},
        {"op": "join", "args": {"instruction": "join", "right_table": "r",
                                    "pair_block_size": 6}},
    ])
    join = physical[-1]
    assert isinstance(join, ops.BlockSemJoin)
    pairs = ops.CartesianProduct(right)(source[0])
    join.max_block_tokens = join.estimate_tokens(pairs[:2])
    built = plan.build(source, physical)
    result, _ = asyncio.run(plan.plan_executor.execute(source, built))
    assert len(result) == 6
    assert join.stats.block_calls == 3


def test_join_result_keeps_both_sides_for_next_join(fake_kv_manager):
    ex = YesExecutor()
    first = ops.CartesianProduct([row("middle", 10)])(row("left", 0, ex))
    result = asyncio.run(ops.BlockSemJoin("yes").run_block(first))
    assert result[0].input.right_data == []
    second = ops.CartesianProduct([row("last", 20)])(result[0])
    text = ops.BlockSemJoin("yes").row_text_for_prompt(second[0])
    assert "left" in text and "middle" in text and "last" in text


def test_empty_right_value_is_valid_join_candidate(fake_kv_manager):
    pair = ops.CartesianProduct([row("", 1)])(row("left", 0, YesExecutor()))[0]
    assert "Right item: (empty)" in ops.BlockSemJoin("yes").row_text_for_prompt(pair)


def test_repairs_share_one_request_reservation(fake_kv_manager):
    class RepairExecutor(YesExecutor):
        active = 0
        peak = 0

        async def execute(self, raw_request, prompt, max_tokens, pin=False, priority=0):
            if prompt[-1]["content"].startswith("Question for each item:"):
                return FakeOut("malformed")
            self.active += 1
            self.peak = max(self.peak, self.active)
            await asyncio.sleep(0)
            self.active -= 1
            return FakeOut("YES")

    ex = RepairExecutor()
    result = asyncio.run(ops.BlockSemFilter("q").evaluate_block(ex, None, ["a", "b", "c"]))
    assert result.verdicts == [True, True, True]
    assert ex.peak == 1


def test_single_budget_matches_actual_single_prompt(fake_kv_manager):
    op = ops.BlockSemFilter("q")
    messages = build_single_messages("q", "a")
    expected = len("\n".join(m["content"] for m in messages)) + op.single_max_tokens
    assert op.estimate_tokens([row("a")], token_length=len) == expected


def test_block_filter_overlaps_tuple_producer(fake_kv_manager):
    class Streaming(YesExecutor):
        def __init__(self):
            super().__init__()
            self.started = asyncio.Event()

        async def execute(self, raw_request, prompt, max_tokens, pin=False, priority=0):
            body = prompt[-1]["content"]
            if body.startswith("Question for each item: second"):
                self.started.set()
            elif "Condition:first" in body and "slow" in str(prompt):
                await self.started.wait()
            return await super().execute(raw_request, prompt, max_tokens, pin, priority)

    ex = Streaming()
    source = [row(t, i, ex) for i, t in enumerate(["a", "b", "slow"])]
    plan = SemanticPlan(ex)
    built = plan.build(source, [ops.SemFilter("first"), ops.BlockSemFilter("second", 2)])
    result, _ = asyncio.run(asyncio.wait_for(plan.plan_executor.execute(source, built), 2))
    assert len(result) == 3 and ex.started.is_set()


def test_map_after_block_retries_with_new_reservation(fake_kv_manager):
    from vllm.kalypso.context import RETRY_TASK
    from vllm.kalypso.sem_ops.base import BaseOp, OpBehavior

    class RetryingRowOp(BaseOp):
        def __init__(self):
            super().__init__(OpBehavior.TUPLE_INDEPENDENT, 1)
            self.calls = {}

        def estimate_tokens(self, ctx):
            return 500 if self.calls.get(ctx.state.idx) else 10

        async def __call__(self, ctx, priority=0):
            n = self.calls.get(ctx.state.idx, 0)
            self.calls[ctx.state.idx] = n + 1
            if not n:
                return RETRY_TASK
            assert fake_kv_manager.stage_usage(ctx.state.stage_id)[0] >= 500
            return True

    ex = YesExecutor()
    source = [row("a", i, ex) for i in range(2)]
    retry = RetryingRowOp()
    plan = SemanticPlan(ex)
    built = plan.build(source, [ops.BlockSemFilter("first", 2), retry])
    result, _ = asyncio.run(plan.plan_executor.execute(source, built))
    assert len(result) == 2 and retry.calls == {0: 2, 1: 2}
    assert all(fake_kv_manager.stage_usage(s.stage_id)[0] == 0 for s in built[0])


def test_block_stage_breaks_physical_pin_chain(fake_kv_manager, monkeypatch):
    monkeypatch.setattr("vllm.kalypso.controller.plan.data._data_source",
                        lambda *args: iter([row("right")]))
    physical = SemanticPlan(None, virtual_pinning=False).parse_ops([
        {"op": "sem_filter", "args": {"prompt": "first"}},
        {"op": "sem_filter", "args": {"prompt": "block", "block_size": 2}},
        {"op": "join", "args": {"instruction": "join", "right_table": "r",
                                    "pair_block_size": 2}},
    ])
    assert not any(op.pin or op.unpin for op in physical)


def test_benchmark_uses_global_candidate_blocks(fake_kv_manager, monkeypatch):
    from pathlib import Path
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "benchmark"))
    from block_join_standalone import evaluate_candidates
    ex = YesExecutor()
    op = ops.BlockSemJoin("q", left_block_size=2, right_block_size=2)
    matches, _ = asyncio.run(evaluate_candidates(op, ex, ["a", "b", "c"], ["x", "y", "z"]))
    assert matches == [(i, j) for i in range(3) for j in range(3)]
    assert op.stats.block_calls == 2 and op.stats.fallback_calls == 1
