"""Unit tests for block semantic map (no model or server needed)."""

import asyncio
from types import SimpleNamespace

import pytest

from vllm.kalypso.sem_ops.block_map import (
    BlockSemMap,
    build_block_map_messages,
    parse_block_map_outputs,
)
from vllm.kalypso.sem_ops.map import SemMap
from vllm.kalypso.sem_ops.prompt_utils import add_assistant_prompt

from .test_block_filter import FakeOut, row


class MapExecutor:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    async def execute(self, raw_request, prompt, max_tokens, pin=False, priority=0):
        self.calls.append((prompt, max_tokens))
        reply = self.replies.pop(0)
        return reply if isinstance(reply, FakeOut) else FakeOut(reply)


def run(coro):
    return asyncio.run(coro)


def mapped_rows(texts, executor):
    rows = [row(text, index, executor) for index, text in enumerate(texts)]
    for item in rows:
        item.state.retry_op_position = -1
        item.state.retry_max_tokens = 0
    return rows


def test_block_prompt_preserves_multiline_text_as_json():
    body = build_block_map_messages("summarize", ["first\n2. not an item"])[-1]["content"]
    assert '"id": 1' in body
    assert "first\\n2. not an item" in body
    assert body.count("summarize") == 1


def test_parse_block_map_requires_unambiguous_string_outputs():
    assert parse_block_map_outputs(
        '{"items": [{"id": 1, "output": "a"}, {"id": 2, "output": "b"}]}', 2
    ) == {1: "a", 2: "b"}
    assert parse_block_map_outputs(
        '{"items": [{"id": 1, "output": "a"}, {"id": 1, "output": "b"}]}', 2
    ) == {}
    assert parse_block_map_outputs("not json", 2) == {}


def test_block_map_materializes_logical_per_row_state_not_block_transcript(fake_kv_manager):
    executor = MapExecutor([
        '{"items": [{"id": 1, "output": "alpha"}, {"id": 2, "output": "beta"}]}'
    ])
    rows = mapped_rows(["one", "two"], executor)
    expected = []
    for item, answer in zip(rows, ["alpha", "beta"]):
        prompt = SemMap("summarize")._build_prompt(item)
        expected.append(add_assistant_prompt(prompt, answer)[0])

    op = BlockSemMap("summarize", block_size=2)
    assert run(op.run_block(rows)) == rows
    assert len(executor.calls) == 1
    assert "Contexts (JSON)" in executor.calls[0][0][-1]["content"]
    assert [item.input.data for item in rows] == expected
    assert all("Contexts (JSON)" not in str(item.input.data) for item in rows)
    assert [item.output[-1][str(BlockSemMap)] for item in rows] == ["alpha", "beta"]


def test_missing_block_entries_repair_with_ordinary_semmap(fake_kv_manager):
    executor = MapExecutor([
        '{"items": [{"id": 1, "output": "from block"}]}',
        "repaired",
    ])
    rows = mapped_rows(["one", "two"], executor)
    op = BlockSemMap("summarize", block_size=2)
    run(op.run_block(rows))
    assert len(executor.calls) == 2
    assert [item.output[-1][str(BlockSemMap)] for item in rows] == ["from block", "repaired"]
    # The repair request is the normal SemMap prompt, not a one-item block prompt.
    assert "Map Instruction:summarize" in executor.calls[1][0][-1]["content"]
    assert op.stats.rows_from_block == 1
    assert op.stats.rows_from_fallback == 1


def test_block_size_one_uses_the_ordinary_semmap_request(fake_kv_manager):
    executor = MapExecutor(["singleton"])
    item = mapped_rows(["one"], executor)[0]
    op = BlockSemMap("summarize", block_size=8)
    run(op.run_block([item]))
    assert len(executor.calls) == 1
    assert "Map Instruction:summarize" in executor.calls[0][0][-1]["content"]
    assert item.output[-1][str(BlockSemMap)] == "singleton"


def test_planner_selects_block_map_only_for_block_size_above_one(fake_kv_manager):
    from vllm.kalypso.controller.plan import SemanticPlan

    physical = SemanticPlan(executor=None).parse_ops([
        {"op": "sem_map", "args": {"prompt": "a", "block_size": 3}},
        {"op": "sem_map", "args": {"prompt": "b", "block_size": 1}},
    ])
    assert isinstance(physical[0], BlockSemMap)
    assert physical[0].block_size == 3
    assert type(physical[1]) is SemMap


def test_block_map_rejects_individual_kv_pinning():
    with pytest.raises(ValueError, match="pin=True"):
        BlockSemMap("summarize", pin=True)


def test_biodex_dataset_runs_through_a_block_map_pipeline(fake_kv_manager):
    """The real map workload is scheduled as one two-row block task."""
    from vllm.kalypso.benchmark.eval.workloads import load_biodex
    from vllm.kalypso.controller.stage import stage_builder
    from vllm.kalypso.data import data
    from vllm.kalypso.execution.pipeline_execution import AsyncPipelineExecutor

    workload = load_biodex(2)
    executor = MapExecutor([
        '{"items": [{"id": 1, "output": "reaction one"}, '
        '{"id": 2, "output": "reaction two"}]}'
    ])
    ctxs = list(data._data_source(None, str(workload.data_path), None))
    for ctx in ctxs:
        ctx.state.executor = executor
        ctx.state.retry_op_position = -1
        ctx.state.retry_max_tokens = 0

    stage = stage_builder([BlockSemMap(workload.instruction, block_size=2)], 1)
    fake_kv_manager.register_stage(1, 1.0)
    previous = AsyncPipelineExecutor.LOG_SCHEDULER
    AsyncPipelineExecutor.LOG_SCHEDULER = False
    try:
        result, stats = run(AsyncPipelineExecutor().execute_tasks(ctxs, [stage]))
    finally:
        AsyncPipelineExecutor.LOG_SCHEDULER = previous

    assert [ctx.state.idx for ctx in result] == [1, 2]
    assert stats[1] == {"input": 1, "output": 2}
    assert len(executor.calls) == 1
    assert [ctx.output[-1][str(BlockSemMap)] for ctx in result] == [
        "reaction one", "reaction two"
    ]
