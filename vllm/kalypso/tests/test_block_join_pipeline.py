"""Block join through the normal planner and asynchronous pipeline."""

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
        if body.startswith("Join condition:"):
            self.join_calls += 1
            left = re.findall(r"^L(\d+)\. (.*)$", body, re.M)
            right = re.findall(r"^R(\d+)\. (.*)$", body, re.M)
            pairs = [f"L{li}, R{rj}" for li, lt in left for rj, rt in right
                     if lt == rt]
            return FakeOut("\n".join(pairs) or "NONE")
        if body.startswith("Left item:"):
            self.join_calls += 1
            left = re.search(r"Left item: (.*)$", body, re.M).group(1)
            right = re.search(r"Right item: (.*)$", body, re.M).group(1)
            return FakeOut("YES" if left == right else "NO")
        if body.startswith("Question for each item:"):
            self.filter_calls += 1
            items = re.findall(r"^(\d+)\. (.*)$", body, re.M)
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


def test_planner_keeps_legacy_join_without_block_sizes(fake_kv_manager, monkeypatch):
    right = [row("apple", 10)]
    monkeypatch.setattr("vllm.kalypso.controller.plan.data._data_source",
                        lambda *args: iter(right))
    plan = SemanticPlan(executor=None)
    legacy = plan.parse_ops([{"op": "join", "args": {
        "instruction": "same item", "right_table": "right.csv"}}])
    assert isinstance(legacy[0], ops.CartesianProduct)
    block = plan.parse_ops([{"op": "join", "args": {
        "instruction": "same item", "right_table": "right.csv",
        "left_block_size": 2, "right_block_size": 2}}])
    assert isinstance(block[0], ops.BlockSemJoin)
    assert block[0].right_table == right


@pytest.mark.parametrize("first_block_size", [1, 2])
def test_filter_block_join_filter_query(fake_kv_manager, monkeypatch,
                                        first_block_size):
    executor = QueryExecutor()
    left = [row(text, idx, executor) for idx, text in
            enumerate(("apple", "banana", "carrot", "date"))]
    right = [row(text, 100 + idx) for idx, text in
             enumerate(("carrot", "apple", "pear"))]
    monkeypatch.setattr("vllm.kalypso.controller.plan.data._data_source",
                        lambda *args: iter(right))

    plan = SemanticPlan(executor=executor)
    physical = plan.parse_ops([
        {"op": "sem_filter", "args": {"prompt": "keep all",
                                        "block_size": first_block_size}},
        {"op": "join", "args": {"instruction": "same item",
                                "right_table": "right.csv",
                                "left_block_size": 2, "right_block_size": 2}},
        {"op": "sem_filter", "args": {"prompt": "only apple", "block_size": 2}},
    ])
    built = plan.build(left, physical)
    assert len(built) == 1 and [s.block_size for s in built[0]] == [first_block_size, 2, 2]
    assert isinstance(built[0][1].operators[0], ops.BlockSemJoin)

    output, stats = asyncio.run(plan.plan_executor.execute(left, built))
    assert [(ctx.state.idx, ctx.state.right_idx) for ctx in output] == [(0, 101)]
    assert len(output[0].output) == 2  # both filter decisions survive the join
    assert executor.join_calls == 4  # 2 left blocks x 2 right blocks
    assert physical[1].stats.candidate_pairs == 12
    assert stats[0][built[0][1].stage_id]["input"] == 2
    assert all(fake_kv_manager.stage_usage(stage.stage_id)[0] == 0
               for stage in built[0])


def test_join_starts_before_upstream_filter_finishes(fake_kv_manager, monkeypatch):
    class StreamingExecutor(QueryExecutor):
        def __init__(self):
            super().__init__()
            self.join_started = asyncio.Event()
            self.upstream_finished = False
            self.join_overlapped_upstream = False

        async def execute(self, raw_request, prompt, max_tokens, pin=False,
                          priority=0):
            body = prompt[-1]["content"]
            if body.startswith("Question for each item: keep all") and "carrot" in body:
                await self.join_started.wait()
                self.upstream_finished = True
            if body.startswith("Join condition:"):
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
                                "left_block_size": 2, "right_block_size": 2}},
    ])
    built = plan.build(left, physical)
    output, _ = asyncio.run(asyncio.wait_for(
        plan.plan_executor.execute(left, built), timeout=2,
    ))
    assert executor.join_overlapped_upstream
    assert sorted((ctx.state.idx, ctx.state.right_idx) for ctx in output) == [
        (0, 100), (2, 101),
    ]


def test_join_flushes_final_partial_left_block(fake_kv_manager, monkeypatch):
    class DroppingExecutor(QueryExecutor):
        async def execute(self, raw_request, prompt, max_tokens, pin=False,
                          priority=0):
            body = prompt[-1]["content"]
            if body.startswith("Question for each item: keep all"):
                self.filter_calls += 1
                items = re.findall(r"^(\d+)\. (.*)$", body, re.M)
                return FakeOut("\n".join(
                    f"{idx}: {'NO' if value == 'date' else 'YES'}"
                    for idx, value in items
                ))
            return await super().execute(raw_request, prompt, max_tokens,
                                         pin=pin, priority=priority)

    executor = DroppingExecutor()
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
                                "left_block_size": 2, "right_block_size": 2}},
    ])
    built = plan.build(left, physical)
    output, stats = asyncio.run(asyncio.wait_for(
        plan.plan_executor.execute(left, built), timeout=2,
    ))
    assert sorted((ctx.state.idx, ctx.state.right_idx) for ctx in output) == [
        (0, 100), (2, 101),
    ]
    assert stats[0][built[0][1].stage_id]["input"] == 2
    assert physical[1].stats.candidate_pairs == 6
