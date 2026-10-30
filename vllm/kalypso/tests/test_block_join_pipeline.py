"""Cartesian-product block join through the normal planner and pipeline."""

import asyncio
import re

import pytest

from vllm.kalypso.controller.plan import SemanticPlan
from vllm.kalypso.execution.pipeline_execution import AsyncPipelineExecutor
from vllm.kalypso.sem_ops import ops

from .test_block_filter import FakeOut, row


@pytest.fixture(autouse=True)
def quiet_scheduler(monkeypatch):
    monkeypatch.setattr(AsyncPipelineExecutor, "LOG_SCHEDULER", False)


class QueryExecutor:
    def __init__(self):
        self.join_calls = 0
        self.filter_calls = 0

    async def execute(self, raw_request, prompt, max_tokens, pin=False, priority=0):
        body = prompt[-1]["content"]
        if body.startswith("Question for each item:"):
            self.filter_calls += 1
            items = re.findall(r"^(\d+)\. (.*)$", body, re.MULTILINE)
            if "same item" in body.splitlines()[0]:
                self.join_calls += 1
                answers = []
                for idx, value in items:
                    pair = re.search(r"Left item: (.*?) Right item: (.*)$", value)
                    assert pair, value
                    answers.append(f"{idx}: {'YES' if pair.group(1) == pair.group(2) else 'NO'}")
                return FakeOut("\n".join(answers))
            only_apple = "only apple" in body.splitlines()[0]
            return FakeOut("\n".join(
                f"{idx}: {'YES' if not only_apple or 'apple' in value else 'NO'}"
                for idx, value in items
            ))
        if body.startswith("Item:"):
            self.filter_calls += 1
            item = body.split("\n\n", 1)[0]
            return FakeOut("YES" if "only apple" not in body or "apple" in item else "NO")
        if "Condition:keep all" in body:
            self.filter_calls += 1
            return FakeOut("TRUE")
        raise AssertionError(f"unexpected model prompt: {body}")


def test_planner_lowers_block_join_to_cartesian_product_then_filter(fake_kv_manager, monkeypatch):
    right = [row("apple", 10)]
    monkeypatch.setattr("vllm.kalypso.controller.plan.data._data_source",
                        lambda *args: iter(right))
    plan = SemanticPlan(executor=None)
    physical = plan.parse_ops([{"op": "join", "args": {
        "instruction": "same item", "right_table": "right.csv",
        "pair_block_size": 6}}])
    assert isinstance(physical[0], ops.CartesianProduct)
    assert isinstance(physical[1], ops.BlockSemJoin)
    assert physical[1].block_size == 6


def test_planner_rejects_legacy_join_grid_sizes(fake_kv_manager, monkeypatch):
    monkeypatch.setattr("vllm.kalypso.controller.plan.data._data_source",
                        lambda *args: iter(()))
    with pytest.raises(ValueError, match="pair_block_size"):
        SemanticPlan(executor=None).parse_ops([{"op": "join", "args": {
            "instruction": "same item", "right_table": "right.csv",
            "left_block_size": 2, "right_block_size": 3}}])


@pytest.mark.parametrize("first_block_size", [1, 2])
def test_filter_cartesian_product_block_join_filter_query(
    fake_kv_manager, monkeypatch, first_block_size
):
    executor = QueryExecutor()
    left = [row(text, idx, executor) for idx, text in
            enumerate(("apple", "banana", "carrot", "date"))]
    right = [row(text, 100 + idx) for idx, text in
             enumerate(("carrot", "apple", "pear"))]
    monkeypatch.setattr("vllm.kalypso.controller.plan.data._data_source",
                        lambda *args: iter(right))

    plan = SemanticPlan(executor=executor)
    physical = plan.parse_ops([
        {"op": "sem_filter", "args": {"prompt": "keep all", "block_size": first_block_size}},
        {"op": "join", "args": {"instruction": "same item", "right_table": "right.csv",
                                "pair_block_size": 4}},
        {"op": "sem_filter", "args": {"prompt": "only apple", "block_size": 2}},
    ])
    built = plan.build(left, physical)
    assert len(built) == 1
    assert isinstance(built[0][0].fanout_op, ops.CartesianProduct)
    assert isinstance(built[0][1].operators[0], ops.BlockSemJoin)

    output, stats = asyncio.run(plan.plan_executor.execute(left, built))
    assert [(ctx.state.idx, ctx.state.right_idx) for ctx in output] == [(0, 101)]
    assert len(output[0].output) == 2
    assert executor.join_calls == 3  # 12 Cartesian-product pairs in blocks of four
    join_stage = built[0][1]
    assert stats[0][join_stage.stage_id]["input"] == 3
    assert all(fake_kv_manager.stage_usage(stage.stage_id)[0] == 0
               for stage in built[0])


def test_block_join_starts_before_upstream_filter_finishes(fake_kv_manager, monkeypatch):
    class StreamingExecutor(QueryExecutor):
        def __init__(self):
            super().__init__()
            self.join_started = asyncio.Event()
            self.upstream_finished = False
            self.join_overlapped_upstream = False

        async def execute(self, raw_request, prompt, max_tokens, pin=False, priority=0):
            body = prompt[-1]["content"]
            if body.startswith("Question for each item: keep all") and "carrot" in body:
                await self.join_started.wait()
                self.upstream_finished = True
            if body.startswith("Question for each item: same item"):
                self.join_overlapped_upstream |= not self.upstream_finished
                self.join_started.set()
            return await super().execute(raw_request, prompt, max_tokens,
                                         pin=pin, priority=priority)

    executor = StreamingExecutor()
    left = [row(text, idx, executor) for idx, text in
            enumerate(("apple", "banana", "carrot", "date"))]
    right = [row("apple", 100), row("carrot", 101)]
    monkeypatch.setattr("vllm.kalypso.controller.plan.data._data_source",
                        lambda *args: iter(right))
    plan = SemanticPlan(executor=executor)
    physical = plan.parse_ops([
        {"op": "sem_filter", "args": {"prompt": "keep all", "block_size": 2}},
        {"op": "join", "args": {"instruction": "same item",
                                "right_table": "right.csv",
                                "pair_block_size": 4}},
    ])
    output, _ = asyncio.run(asyncio.wait_for(
        plan.plan_executor.execute(left, plan.build(left, physical)), timeout=2,
    ))
    assert executor.join_overlapped_upstream
    assert sorted((ctx.state.idx, ctx.state.right_idx) for ctx in output) == [
        (0, 100), (2, 101),
    ]


def test_block_join_fallback_keeps_ordinary_join_ids(fake_kv_manager, monkeypatch):
    class MissingAnswerExecutor(QueryExecutor):
        async def execute(self, raw_request, prompt, max_tokens, pin=False, priority=0):
            body = prompt[-1]["content"]
            if body.startswith("Question for each item:") and "same item" in body:
                self.join_calls += 1
                return FakeOut("1: NO")  # remaining Cartesian candidates retry singly
            if body.startswith("Item:") and "same item" in body:
                pair = re.search(r"Left item: (.*?) Right item: (.*?)(?:\n\n|$)", body)
                return FakeOut("YES" if pair.group(1) == pair.group(2) else "NO")
            return await super().execute(raw_request, prompt, max_tokens, pin=pin, priority=priority)

    executor = MissingAnswerExecutor()
    left = [row("apple", 0, executor), row("carrot", 1, executor)]
    right = [row("carrot", 100), row("apple", 101)]
    monkeypatch.setattr("vllm.kalypso.controller.plan.data._data_source",
                        lambda *args: iter(right))
    plan = SemanticPlan(executor=executor)
    physical = plan.parse_ops([{"op": "join", "args": {
        "instruction": "same item", "right_table": "right.csv",
        "pair_block_size": 4}}])
    output, _ = asyncio.run(plan.plan_executor.execute(left, plan.build(left, physical)))
    assert sorted((ctx.state.idx, ctx.state.right_idx) for ctx in output) == [(0, 101), (1, 100)]
