"""Block tasks: several rows share one Task and one KV memory reservation."""

import asyncio
from types import SimpleNamespace

import pytest

from vllm.kalypso.controller.stage import Task, stage_builder
from vllm.kalypso.execution.pipeline_execution import AsyncPipelineExecutor
from vllm.kalypso.sem_ops.block_filter import BlockSemFilter

from .test_block_filter import FakeExecutor, row


@pytest.fixture(autouse=True)
def quiet_scheduler(monkeypatch):
    monkeypatch.setattr(AsyncPipelineExecutor, "LOG_SCHEDULER", False)


def rows(texts, ex):
    return [row(t, i, ex) for i, t in enumerate(texts)]


def test_task_block_properties():
    single = Task(ctx=SimpleNamespace(state=SimpleNamespace(idx=7)), stage_index=0)
    block = Task(ctx=[SimpleNamespace(state=SimpleNamespace(idx=i)) for i in (1, 2)], stage_index=0)
    assert (single.is_block, single.num_rows, single.row_ids) == (False, 1, [7])
    assert (block.is_block, block.num_rows, block.row_ids) == (True, 2, [1, 2])


def test_stage_takes_block_size_from_block_operator(fake_kv_manager):
    assert stage_builder([BlockSemFilter("q", block_size=4)], 1).block_size == 4
    assert stage_builder([], 2).block_size == 1


def test_token_budget_limits_group_size(fake_kv_manager):
    op = BlockSemFilter("q", block_size=4)
    ex = FakeExecutor([])
    sample = rows(["short", "long " * 100, "short"], ex)
    two = op.estimate_tokens(sample[:2])
    three = op.estimate_tokens(sample)
    op.max_block_tokens = (two + three) // 2
    stage = stage_builder([op], 1)
    assert [len(group) for group in stage.make_blocks(sample)] == [2, 1]


def test_queue_signals_account_for_underfilled_blocks(fake_kv_manager):
    stage = stage_builder([BlockSemFilter("q", block_size=4)], 1)
    ex = FakeExecutor([])
    stage.low_threshold = 1
    stage.high_threshold = 1
    for item in rows("ab", ex):
        stage.enqueue(Task(ctx=[item], stage_index=0))
    assert stage.ready_count() == 2
    assert stage.ready_rows() == 2
    assert stage.ready_work() == 0.5
    assert stage.is_starving() and not stage.is_saturated()
    for group in stage.make_blocks(rows("cdefghij", ex)):
        stage.enqueue(Task(ctx=group, stage_index=0))
    assert stage.ready_work() == 2.5
    assert stage.is_saturated()


def test_block_task_reserves_once_for_the_whole_block(fake_kv_manager):
    stage = stage_builder([BlockSemFilter("q", block_size=4)], 1)
    ex = FakeExecutor([])
    block = rows(["a" * 40] * 4, ex)
    one_block = stage.estimate_budget(Task(ctx=block, stage_index=0))
    per_row = sum(stage.estimate_budget(Task(ctx=r, stage_index=0)) for r in block)
    # one shared prompt + output cap is cheaper than four separate requests
    assert 0 < one_block < per_row


def test_block_reservation_is_taken_and_released(fake_kv_manager):
    fake_kv_manager.register_stage(1, 1.0)
    stage = stage_builder([BlockSemFilter("q", block_size=3)], 1)
    ex = FakeExecutor(["1: YES\n2: NO\n3: YES"])
    task = Task(ctx=rows("abc", ex), stage_index=0)

    async def go():
        budget = await stage.accept(task, fake_kv_manager)
        assert fake_kv_manager.stage_usage(1)[0] == budget > 0
        assert fake_kv_manager.stage_inflight(1) == 1  # one reservation, three rows
        out = await stage.run_task(task)
        await stage.release_budget(stage.detach_budget(task), fake_kv_manager)
        return out

    out = asyncio.run(go())
    assert [r.state.idx for r in out] == [0, 2]
    assert fake_kv_manager.stage_usage(1)[0] == 0
    assert fake_kv_manager.stage_inflight(1) == 0


def test_block_task_with_no_survivors_returns_none(fake_kv_manager):
    stage = stage_builder([BlockSemFilter("q", block_size=2)], 1)
    ex = FakeExecutor(["1: NO\n2: NO"])
    assert asyncio.run(stage.run_task(Task(ctx=rows("ab", ex), stage_index=0))) is None


def run_pipeline(stages, ctxs, mgr):
    for i, st in enumerate(stages):
        mgr.register_stage(st.stage_id, 1.0 / len(stages), min_fraction=0.0, max_fraction=1.0)
    return asyncio.run(AsyncPipelineExecutor().execute_tasks(ctxs, stages))


def test_pipeline_runs_one_task_per_block(fake_kv_manager):
    ex = FakeExecutor(["1: YES\n2: NO\n3: YES", "1: YES\n2: YES\n3: NO", "1: NO"])
    stage = stage_builder([BlockSemFilter("q", block_size=3)], 1)
    out, stats = run_pipeline([stage], rows("abcdefg", ex), fake_kv_manager)
    assert sorted(r.state.idx for r in out) == [0, 2, 3, 4]
    assert stats[1]["input"] == 3          # 7 rows -> 3 block tasks
    assert stats[1]["output"] == 4
    assert len(ex.calls) == 2 + 1          # two block calls + block of one


def test_two_block_stages_keep_block_boundaries(fake_kv_manager):
    ex = FakeExecutor(["1: YES\n2: YES\n3: NO\n4: YES", "1: YES\n2: NO\n3: YES"])
    s1 = stage_builder([BlockSemFilter("first", block_size=4)], 1)
    s2 = stage_builder([BlockSemFilter("second", block_size=4)], 2)
    out, stats = run_pipeline([s1, s2], rows("abcd", ex), fake_kv_manager)
    assert sorted(r.state.idx for r in out) == [0, 3]
    assert stats[2]["input"] == 1          # survivors stay one block task
    assert len(ex.calls) == 2
    assert fake_kv_manager.stage_usage(1)[0] == fake_kv_manager.stage_usage(2)[0] == 0


def test_downstream_stage_respects_smaller_block_limit(fake_kv_manager):
    ex = FakeExecutor(["1: YES\n2: YES\n3: YES\n4: YES",
                       "1: YES\n2: YES", "1: YES\n2: YES"])
    s1 = stage_builder([BlockSemFilter("first", block_size=4)], 1)
    s2 = stage_builder([BlockSemFilter("second", block_size=2)], 2)
    out, stats = run_pipeline([s1, s2], rows("abcd", ex), fake_kv_manager)
    assert len(out) == 4
    assert stats[2]["input"] == 2


def test_block_stage_feeds_per_row_stage(fake_kv_manager):
    ex = FakeExecutor(["1: YES\n2: NO\n3: YES"], truth={"a": True, "c": False})
    s1 = stage_builder([BlockSemFilter("first", block_size=3)], 1)
    s2 = stage_builder([BlockSemFilter("second", block_size=1)], 2)
    out, stats = run_pipeline([s1, s2], rows("abc", ex), fake_kv_manager)
    assert [r.state.idx for r in out] == [0]
    assert stats[2]["input"] == 2          # each survivor is its own task


def test_empty_input(fake_kv_manager):
    stage = stage_builder([BlockSemFilter("q", block_size=3)], 1)
    out, _ = run_pipeline([stage], [], fake_kv_manager)
    assert out == []


def test_per_row_semfilter_pipeline_is_unchanged(fake_kv_manager):
    """Kalypso's original path: one task per row, one request per row."""
    from vllm.kalypso.sem_ops.filters import SemFilter

    class PerRowExecutor:
        def __init__(self):
            self.calls = 0

        async def execute(self, raw_request, prompt, max_tokens, pin=False, priority=0):
            self.calls += 1
            text = prompt[-2]["content"] if len(prompt) > 1 else ""
            return SimpleNamespace(text="true" if "keep" in text else "false",
                                   request_id=str(self.calls), finish_reason="stop")

    ex = PerRowExecutor()
    stage = stage_builder([SemFilter("q")], 1)
    assert stage.block_size == 1
    out, stats = run_pipeline([stage], rows(["keep me", "drop me", "keep too"], ex), fake_kv_manager)
    assert sorted(r.state.idx for r in out) == [0, 2]
    assert stats[1]["input"] == 3 and ex.calls == 3


def test_planner_builds_block_filter_from_block_size(fake_kv_manager):
    from vllm.kalypso.controller.plan import SemanticPlan
    from vllm.kalypso.sem_ops import ops

    plan = SemanticPlan(executor=None)
    physical = plan.parse_ops([
        {"op": "sem_filter", "args": {"prompt": "a", "block_size": 8}},
        {"op": "sem_filter", "args": {"prompt": "b"}},
        {"op": "sem_filter", "args": {"prompt": "c", "block_size": 1}},
    ])
    assert isinstance(physical[0], ops.BlockSemFilter) and physical[0].block_size == 8
    assert type(physical[1]) is ops.SemFilter
    assert type(physical[2]) is ops.SemFilter
    built = plan.build([], physical)
    assert built[0][0].block_size == 8


def test_planner_passes_token_budget(fake_kv_manager):
    from vllm.kalypso.controller.plan import SemanticPlan

    op = SemanticPlan(executor=None).parse_ops([
        {"op": "sem_filter", "args": {"prompt": "q", "block_size": 8,
                                        "max_block_tokens": 256}},
    ])[0]
    assert op.max_block_tokens == 256
