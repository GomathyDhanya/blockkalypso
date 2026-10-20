"""QueryTrace: per-stage rows, timing, occupancy and operator stats."""

import asyncio
from types import SimpleNamespace

from vllm.kalypso.controller.plan import SemanticPlan
from vllm.kalypso.execution import trace as tr
from vllm.kalypso.execution.pipeline_execution import AsyncPipelineExecutor

from .test_block_filter import row
from .test_block_join_pipeline import QueryExecutor


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


def _stage(stage_id, block_size=1, ops=()):
    return SimpleNamespace(stage_id=stage_id, block_size=block_size,
                           operators=ops, fanout_op=None)


def _task(task_id, rows):
    return SimpleNamespace(task_id=task_id, num_rows=rows)


def test_trace_records_timing_occupancy_and_counts_retries_once():
    clock = Clock()
    trace = tr.QueryTrace(clock=clock)
    stats = SimpleNamespace(as_dict=lambda: {"llm_calls": 3})
    s0 = _stage(0, block_size=4, ops=(SimpleNamespace(stats=stats),))
    s1 = _stage(1)
    trace.register_stages([s0, s1])

    clock.t = 101.0
    trace.task_started(s0, _task(1, 4))
    trace.task_started(s0, _task(2, 2))
    clock.t = 102.0
    trace.task_finished(_task(1, 4), 3, tr.DONE)
    trace.task_finished(_task(2, 2), 0, tr.RETRY)
    trace.task_started(s0, _task(3, 2))        # the retry of task 2
    clock.t = 103.0
    trace.task_finished(_task(3, 2), 1, tr.DONE)
    trace.task_started(s1, _task(4, 1))
    clock.t = 104.5
    trace.task_finished(_task(4, 1), 1, tr.DONE)
    trace.outputs_emitted(1)
    trace.outputs_emitted(0)

    d = trace.to_dict()
    first = d["stages"]["0"]
    assert first["tasks"] == 2 and first["retries"] == 1
    assert first["rows_in"] == 6 and first["rows_out"] == 4
    assert first["first_start"] == 1.0 and first["last_end"] == 3.0
    assert first["mean_occupancy"] == 0.75          # (4 + 2) / (4 * 2)
    assert first["operator_stats"] == {"SimpleNamespace": {"llm_calls": 3}}
    assert d["stages"]["1"]["mean_occupancy"] is None   # not a block stage
    assert d["first_output_s"] == 4.5 and d["outputs"] == 1
    assert [t["outcome"] for t in d["tasks"]] == [tr.DONE, tr.RETRY, tr.DONE, tr.DONE]


def test_trace_of_filter_join_filter_pipeline(fake_kv_manager, monkeypatch):
    monkeypatch.setattr(AsyncPipelineExecutor, "LOG_SCHEDULER", False)
    executor = QueryExecutor()
    left = [row(text, idx, executor) for idx, text in
            enumerate(("apple", "banana", "carrot", "date"))]
    right = [row(text, 100 + idx) for idx, text in
             enumerate(("carrot", "apple", "pear"))]
    monkeypatch.setattr("vllm.kalypso.controller.plan.data._data_source",
                        lambda *args: iter(right))
    plan = SemanticPlan(executor=executor)
    physical = plan.parse_ops([
        {"op": "sem_filter", "args": {"prompt": "keep all", "block_size": 2}},
        {"op": "join", "args": {"instruction": "same item",
                                "right_table": "right.csv",
                                "left_block_size": 2, "right_block_size": 2}},
        {"op": "sem_filter", "args": {"prompt": "only apple", "block_size": 2}},
    ])
    built = plan.build(left, physical)
    trace = tr.QueryTrace()
    output, _ = asyncio.run(plan.plan_executor.execute(left, built, trace=trace))
    assert [(c.state.idx, c.state.right_idx) for c in output] == [(0, 101)]

    d = trace.to_dict()
    f1, join, f2 = (d["stages"][str(s.stage_id)] for s in built[0])
    assert f1["operators"] == ["BlockSemFilter"] and join["operators"] == ["BlockSemJoin"]
    assert (f1["rows_in"], f1["rows_out"], f1["mean_occupancy"]) == (4, 4, 1.0)
    assert (join["tasks"], join["rows_in"], join["rows_out"]) == (2, 4, 2)
    # Each join task yields one match and F2 blocks are cut per join task, so
    # every F2 block holds one row of a possible two.
    assert (f2["tasks"], f2["rows_in"], f2["rows_out"]) == (2, 2, 1)
    assert f2["mean_occupancy"] == 0.5
    assert join["operator_stats"]["BlockSemJoin"]["block_calls"] == 4
    assert f1["operator_stats"]["BlockSemFilter"]["llm_calls"] == 2
    assert d["outputs"] == 1
    assert 0 <= d["first_output_s"] <= d["elapsed_s"]
    assert all(t["end"] is not None for t in d["tasks"])
    # The F2 block holding the carrot match passes nothing, so it is dropped.
    assert sorted(t["outcome"] for t in d["tasks"]) == [tr.DONE] * 5 + [tr.DROPPED]
    assert f1["first_start"] <= join["first_start"] <= f2["first_start"]
