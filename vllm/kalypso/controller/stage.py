import asyncio
from dataclasses import dataclass, field
import math
from itertools import count
from typing import Any

from vllm.kalypso.budget import KVMemoryManager
from vllm.kalypso.context import RETRY_TASK, RetryTaskResult
from vllm.kalypso.sem_ops import OpBehavior, ops


_TASK_IDS = count()


@dataclass
class Task:
    """Unit of scheduling: one row, or a block of rows sharing one reservation.

    ``ctx`` is a single ``SemContext`` (Kalypso's original per-tuple task) or
    a list of them (a block task). A block task is admitted, budgeted and
    released as a whole, so its rows share one KV memory reservation.
    """

    ctx: Any
    stage_index: int
    trackers: tuple = ()
    op_index: int = 0
    retry_priority: int | None = None
    reserved_budget: int = 0
    task_id: int = field(default_factory=lambda: next(_TASK_IDS))

    @property
    def is_block(self) -> bool:
        return isinstance(self.ctx, list)

    @property
    def rows(self) -> list:
        return list(self.ctx) if self.is_block else [self.ctx]

    @property
    def num_rows(self) -> int:
        return len(self.ctx) if self.is_block else 1

    @property
    def row_ids(self) -> list:
        return [getattr(getattr(r, "state", None), "idx", None) for r in self.rows]


def supports_blocks(op) -> bool:
    """True for operators that evaluate a whole block in one call."""
    return bool(getattr(op, "supports_blocks", False))


class Stage:
    def __init__(
        self,
        stage_id: int,
        operators,
        behavior: OpBehavior | None = None,
        fanout_op=None,
        priority_offset: int = 0,
    ):
        self.stage_id = stage_id
        self.operators = tuple(operators)
        self.behavior = behavior or self._infer_behavior()
        self.fanout_op = fanout_op
        self.priority_offset = priority_offset
        self.waiting_tasks = []
        self.running_tasks = {}
        self.bytes_per_token = KVMemoryManager.get_instance().bytes_per_token
        self.low_threshold = 1
        self.high_threshold = 5
        # Rows per task. >1 when the stage starts with a block operator; the
        # scheduler then enqueues blocks of this size instead of single rows.
        self.block_size = self._infer_block_size()
        self.block_tasks = self.block_size > 1 or any(
            getattr(op, "requires_block_task", False) for op in self.operators
        )

    def _infer_block_size(self) -> int:
        for op in self.operators:
            if supports_blocks(op):
                return max(1, int(getattr(op, "block_size", 1)))
        return 1

    def _infer_behavior(self) -> OpBehavior:
        return OpBehavior.TUPLE_INDEPENDENT

    def task_priority(self, task: Task) -> int:
        if task.retry_priority is not None:
            return task.retry_priority
        return -(self.priority_offset + task.op_index)

    def enqueue(self, task: Task) -> None:
        task_priority = self.task_priority(task)
        insert_at = len(self.waiting_tasks)
        for idx, queued_task in enumerate(self.waiting_tasks):
            queued_priority = self.task_priority(queued_task)
            if (
                task_priority < queued_priority
                or (
                    task_priority == queued_priority
                    and task.task_id < queued_task.task_id
                )
            ):
                insert_at = idx
                break
        self.waiting_tasks.insert(insert_at, task)

    def clear_tasks(self) -> None:
        self.waiting_tasks.clear()
        self.running_tasks.clear()

    def has_waiting_tasks(self) -> bool:
        return bool(self.waiting_tasks)

    def ready_count(self) -> int:
        return len(self.waiting_tasks)

    def ready_rows(self) -> int:
        """Number of queued rows, including rows inside block tasks."""
        return sum(task.num_rows for task in self.waiting_tasks)

    def ready_work(self) -> float:
        """Queued work measured in full-block equivalents."""
        return self.ready_rows() / self.block_size

    def running_count(self) -> int:
        return len(self.running_tasks)

    def is_starving(self) -> bool:
        return self.ready_work() < self.low_threshold

    def is_saturated(self) -> bool:
        return self.ready_work() > self.high_threshold

    def make_blocks(self, rows) -> list[list]:
        """Level 1 fixed-count blocks, optionally bounded by Level 2 tokens.

        The token estimate includes the prompt and output allowance. An
        oversized row is emitted by itself so grouping always progresses.
        """
        block_op = next((op for op in self.operators if supports_blocks(op)), None)
        if block_op is None:
            return [[row] for row in rows]
        limit = getattr(block_op, "max_block_tokens", None)
        blocks, current = [], []
        for row in rows:
            candidate = current + [row]
            if current and (len(candidate) > self.block_size or
                            (limit is not None and block_op.estimate_tokens(candidate) > limit)):
                blocks.append(current)
                current = [row]
            else:
                current = candidate
        if current:
            blocks.append(current)
        return blocks

    def tune_thresholds(self):
        manager = KVMemoryManager.get_instance()
        min_budget = int(manager._stage_min_capacity.get(self.stage_id, 0))
        if min_budget <= 0:
            min_budget = max(1, int(self.bytes_per_token))

        _, cur_cap = manager.stage_usage(self.stage_id)
        self.high_threshold = max(1, int(1 * cur_cap // min_budget))
        self.low_threshold = max(1, int(0.5 * cur_cap // min_budget))
        # print(f'{self.stage_id}-{self.low_threshold}, self.is_starving(): {self.is_starving()}, self.is_saturated(): {self.is_saturated()}')

    def peek_task(self) -> Task | None:
        if not self.waiting_tasks:
            return None
        return self.waiting_tasks[0]

    def pop_task(self) -> Task | None:
        if not self.waiting_tasks:
            return None
        return self.waiting_tasks.pop(0)

    def estimate_tokens_for(self, op, task: Task) -> int:
        """Tokens ``op`` can occupy while running ``task``.

        A block operator sizes the whole block (one call). A per-row operator
        inside a block task runs once per row concurrently, so its rows add up.
        """
        if not hasattr(op, "estimate_tokens"):
            raise AttributeError(f"{op} must define `estimate_tokens`")
        if not task.is_block:
            return op.estimate_tokens(task.ctx)
        if supports_blocks(op):
            return op.estimate_tokens(task.rows)
        return sum(op.estimate_tokens(row) for row in task.rows)

    def estimate_budget(self, task: Task) -> int:
        """Bytes to reserve for ``task``: one reservation for all its rows."""
        max_boundary = -1
        for op in self.operators[task.op_index:]:
            if isinstance(op, ops.CartesianProduct):
                break
            estimated_tokens = self.estimate_tokens_for(op, task)
            if estimated_tokens > max_boundary:
                max_boundary = estimated_tokens

        return max_boundary * self.bytes_per_token


    async def accept(self, task: Task, manager=None) -> int | None:
        manager = manager or KVMemoryManager.get_instance()
        if task.reserved_budget > 0:
            budget = task.reserved_budget
            task.reserved_budget = 0
            self.running_tasks[task.task_id] = budget
            return budget

        budget = self.estimate_budget(task)
        if not await manager.try_allocate_stage(self.stage_id, budget):
            return None
        self.running_tasks[task.task_id] = budget
        return budget


    async def force_accept(self, task: Task, manager=None) -> int:
        manager = manager or KVMemoryManager.get_instance()
        if task.reserved_budget > 0:
            budget = task.reserved_budget
            task.reserved_budget = 0
            self.running_tasks[task.task_id] = budget
            return budget

        budget = self.estimate_budget(task)
        await manager.force_allocate_stage(self.stage_id, budget)
        self.running_tasks[task.task_id] = budget
        return budget

    def detach_budget(self, task: Task) -> int:
        return self.running_tasks.pop(task.task_id, 0)

    async def release_budget(self, budget: int, manager=None) -> None:
        manager = manager or KVMemoryManager.get_instance()
        await manager.release_stage(self.stage_id, budget)

    # async def release(self, task: Task, manager=None) -> None:
    #     budget = self.detach_budget(task)
    #     await self.release_budget(budget, manager)

    async def run_task(self, task: Task):
        if task.is_block:
            return await self._run_block_task(task)

        task.ctx.state.stage_id = self.stage_id

        keep_going = True
        for idx in range(task.op_index, len(self.operators)):
            op = self.operators[idx]
            if keep_going:
                priority = (
                    task.retry_priority
                    if idx == task.op_index and task.retry_priority is not None
                    else -(self.priority_offset + idx)
                )
                keep_going = await op(
                    task.ctx,
                    priority=priority,
                )
                if keep_going is RETRY_TASK:
                    return RetryTaskResult(
                        ctx=task.ctx,
                        op_index=idx,
                        retain_budget=False,
                    )
                if keep_going is False:
                    return None

        return task.ctx


    async def _run_block_task(self, task: Task):
        """Run every operator of the stage over a block of rows.

        Block operators get the whole block in one call. Per-row operators run
        once per surviving row, concurrently; a per-row RETRY_TASK is retried
        in place (as Kalypso's blocking executor does) instead of re-queueing
        the whole block. Returns the surviving rows, or None if none survive.
        """
        rows = task.rows
        for row in rows:
            row.state.stage_id = self.stage_id

        for idx in range(task.op_index, len(self.operators)):
            if not rows:
                break
            op = self.operators[idx]
            priority = -(self.priority_offset + idx)
            if supports_blocks(op):
                rows = await op.run_block(rows, priority=priority)
            else:
                keep = await asyncio.gather(
                    *(self._run_row_op(op, row, priority) for row in rows)
                )
                rows = [row for row, k in zip(rows, keep) if k is not False]

        return rows or None

    @staticmethod
    async def _run_row_op(op, ctx, priority):
        while True:
            out = await op(ctx, priority=priority)
            if out is not RETRY_TASK:
                return out


def stage_builder(operators, stage_id, priority_offset: int = 0):
    return Stage(
        stage_id=stage_id,
        operators=operators,
        priority_offset=priority_offset,
    )
