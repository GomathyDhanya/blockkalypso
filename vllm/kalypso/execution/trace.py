"""Per-query execution trace for the pipeline executor.

Records, for one query, when every task ran, how many rows it took in and
passed on, and when the first final output appeared. ``to_dict`` summarises
this per stage (rows in and out, first start and last finish, block
occupancy, block-operator fallback counters) and keeps the per-task records
so stage overlap can be computed afterwards.

Times are seconds since the trace was created, which the API does when the
query arrives.
"""

import time
from dataclasses import asdict, dataclass

# Outcomes a task attempt can end with.
DONE, DROPPED, RETRY, ERROR = "done", "dropped", "retry", "error"


@dataclass
class TaskRecord:
    task_id: int
    stage_id: int
    rows: int
    start: float
    end: float | None = None
    rows_out: int | None = None
    outcome: str | None = None


class QueryTrace:
    def __init__(self, clock=time.perf_counter):
        self._clock = clock
        self._t0 = clock()
        self._stages = {}   # stage_id -> Stage
        self._open = {}     # task_id -> TaskRecord still running
        self.records: list[TaskRecord] = []
        self.first_output: float | None = None
        self.outputs = 0

    def now(self) -> float:
        return self._clock() - self._t0

    def register_stages(self, stages) -> None:
        for stage in stages:
            self._stages[stage.stage_id] = stage

    def task_started(self, stage, task) -> None:
        record = TaskRecord(task.task_id, stage.stage_id, task.num_rows, self.now())
        self.records.append(record)
        self._open[task.task_id] = record

    def task_finished(self, task, rows_out: int, outcome: str) -> None:
        record = self._open.pop(task.task_id, None)
        if record is not None:
            record.end = self.now()
            record.rows_out = rows_out
            record.outcome = outcome

    def outputs_emitted(self, count: int) -> None:
        if count <= 0:
            return
        if self.first_output is None:
            self.first_output = self.now()
        self.outputs += count

    def _stage_summary(self, stage_id: int) -> dict:
        stage = self._stages.get(stage_id)
        records = [r for r in self.records if r.stage_id == stage_id]
        finished = [r for r in records if r.end is not None]
        # Retried attempts run the same rows again; count each row once.
        first_attempts = [r for r in records if r.outcome != RETRY]
        block_size = getattr(stage, "block_size", 1)
        summary = {
            "operators": [type(op).__name__ for op in getattr(stage, "operators", ())],
            "fanout": (type(stage.fanout_op).__name__
                       if getattr(stage, "fanout_op", None) is not None else None),
            "block_size": block_size,
            "tasks": len(first_attempts),
            "retries": sum(r.outcome == RETRY for r in records),
            "rows_in": sum(r.rows for r in first_attempts),
            "rows_out": sum(r.rows_out or 0 for r in finished),
            "first_start": min((r.start for r in records), default=None),
            "last_end": max((r.end for r in finished), default=None),
            "mean_occupancy": None,
            "operator_stats": {},
        }
        if block_size > 1 and first_attempts:
            summary["mean_occupancy"] = round(
                sum(r.rows for r in first_attempts) / (block_size * len(first_attempts)), 4)
        for op in getattr(stage, "operators", ()):
            stats = getattr(op, "stats", None)
            if stats is not None and hasattr(stats, "as_dict"):
                summary["operator_stats"][type(op).__name__] = stats.as_dict()
        return summary

    def to_dict(self) -> dict:
        stage_ids = sorted(set(self._stages) | {r.stage_id for r in self.records})
        return {
            "elapsed_s": round(self.now(), 4),
            "first_output_s": (None if self.first_output is None
                               else round(self.first_output, 4)),
            "outputs": self.outputs,
            "stages": {str(sid): self._stage_summary(sid) for sid in stage_ids},
            "tasks": [asdict(r) for r in self.records],
        }
