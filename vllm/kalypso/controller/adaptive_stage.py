"""Adaptive per-stage blocking: per-tuple vs. a small block size, switched on
the observed pass rate of the stage that feeds it.

Implements SESSION1_REPORT.md §H, recommendation 7, and nothing more. The
evidence (BASELINE_RESULTS.md §1-2, E1 ``rpipe``): in filter -> join,
blocking the downstream stage loses slightly when the upstream filter passes
~5% of rows (0.97x) and wins from ~20% upward (1.25-1.52x). So the
controlled stage is a downstream block stage, and the signal is the
**observed pass rate of the stage that feeds it**.

Off by default. The single on/off switch is ``AdaptiveStageConfig.enabled``
(env ``KALYPSO_ADAPTIVE_STAGE=1``). Disabled, the executor never creates a
controller and runs the original static code path.

Which stages are managed: every stage after the first that the planner built
as a block stage (``Stage.block_tasks``; the query asked for a block operator
there, e.g. ``block_size``/``pair_block_size``). Its operator
(BlockSemFilter / BlockSemJoin / BlockSemMap) already runs a one-row block as
its ordinary single-row prompt, so "per-tuple" is simply a block size of 1.
No operator is swapped.

Modes of a managed stage:

- ``static`` (initial): the size the query asked for, until the controller
  has seen enough evidence to decide;
- ``tuple``: blocks of 1;
- ``block``: blocks of the largest size in ``candidate_sizes`` that is
  <= ``small_block_size``.

Candidate sizes: the search's power-of-two set, ``costmodel.POW2_CANDIDATES``
(1, 2, 4, 8, 16, 32), up to the stage's requested size. ``tuple`` (1) and
``block`` are the only sizes the controller switches to, so both are always
powers of two (a requested size of 3 gives blocks of 2, not 3). ``static``
is the query's own size and is only the starting mode: the controller never
switches back to it. The k_max cap applies through the quality veto below
(``Model.k_max(..., quality=True)`` includes the context-window bound): a
block size above it is vetoed, and the stage stays per-tuple.

Signal: ``task_end`` telemetry events of the upstream stage, received through
an in-process subscription (``telemetry.subscribe``; works with no trace sink
configured). Pass rate = sum(rows_passed) / sum(rows_in) over the last
``window_tasks`` upstream tasks. ``rows_passed`` counts surviving rows before
any fanout; ``rows_out / rows_in`` cannot be used because on a fanout stage
``rows_out`` counts Cartesian children (E1: ~16 at a 50% pass rate with 32
right rows). Retried tasks are ignored.

Decision with hysteresis: an observation votes for block when both the
rolling rate and that task's own pass rate are ``>= high_pass_rate``, and for
per-tuple when both are ``< low_pass_rate``. Anything else keeps the current
mode and resets the count. The mode changes only after ``min_consecutive``
consecutive votes for the other mode. Requiring the task's own rate as well
as the rolling one matters: a single outlier block keeps the rolling rate
past a threshold for up to ``window_tasks`` later observations, but it can
cast only one vote, so one noisy block cannot flip the mode.

Quality veto: before switching *to* the small block size, the controller asks
``quality_veto(kind, k, row_tokens)``. The default reuses the cost model's
existing guard from ``choose_k(quality=True)``: ``k`` must not exceed
``Model.k_max(kind, items, quality=True)`` (the learned per-(kind, regime) cap
from ``f1_slack`` / ``analyze.MIN_SIMILARITY``, plus the context-window bound).
If vetoed, or if no row has reached the stage yet so its regime is unknown,
the stage stays per-tuple. Switching to per-tuple is never vetoed: per-tuple
is the quality reference those caps are measured against.

Where it applies: ``formation_size`` is called by the executor only when a
row lands in an *empty* pending buffer of a downstream block stage, i.e. when
a new block starts forming (``enqueue_block_rows`` in
``execution/pipeline_execution.py``). The size is fixed for that block at
that moment. Blocks already forming, queued or running are never resized.

``small_block_size`` defaults to 4: E2 measured join blocks best at k=2-4,
with quality collapsing above 4 (F1 0.83 at 4, 0.70 at 8). Filters at k=4
were 1.88-1.90x per-tuple with higher F1. All E1 block arms used 4.
SESSION1_REPORT.md §H.5 caps join blocks at 4.
"""

from __future__ import annotations

import json
import os
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from vllm.kalypso import telemetry

STATIC, TUPLE, BLOCK = "static", "tuple", "block"


@dataclass(frozen=True)
class AdaptiveStageConfig:
    enabled: bool = False
    small_block_size: int = 4
    low_pass_rate: float = 0.05
    high_pass_rate: float = 0.20
    min_consecutive: int = 3
    window_tasks: int = 16
    # (kind, k, row_tokens) -> True to forbid blocks of k. None: cost model guard.
    quality_veto: Callable[[str, int, float], bool] | None = None

    def __post_init__(self):
        if not 0.0 <= self.low_pass_rate <= self.high_pass_rate <= 1.0:
            raise ValueError("need 0 <= low_pass_rate <= high_pass_rate <= 1")
        if self.small_block_size < 2 or self.min_consecutive < 1 or self.window_tasks < 1:
            raise ValueError("small_block_size >= 2, min_consecutive >= 1, window_tasks >= 1")

    @classmethod
    def from_env(cls) -> "AdaptiveStageConfig":
        return cls(enabled=os.environ.get("KALYPSO_ADAPTIVE_STAGE", "") not in ("", "0", "false"))


def operator_kind(op) -> str | None:
    """Cost-model kind of a block operator (imported lazily: no cycle)."""
    from vllm.kalypso.sem_ops.block_join import BlockSemJoin
    from vllm.kalypso.sem_ops.block_filter import BlockSemFilter
    from vllm.kalypso.sem_ops.block_map import BlockSemMap
    if isinstance(op, BlockSemJoin):
        return "join"
    if isinstance(op, BlockSemFilter):
        return "filter"
    if isinstance(op, BlockSemMap):
        return "map"
    return None


_FITTED = (Path(__file__).resolve().parents[1]
           / "benchmark/eval/results/analysis/costmodel.json")


def costmodel_quality_veto(path: Path = _FITTED) -> Callable[[str, int, float], bool]:
    """The existing choose_k quality guard as a veto: forbid k when it exceeds
    ``Model.k_max(kind, items, quality=True)`` for the row's regime."""
    from dataclasses import fields as dc_fields

    from vllm.kalypso.benchmark.eval import costmodel as cm

    d = json.loads(Path(path).read_text())
    names = {f.name for f in dc_fields(cm.KindParams)}
    kinds = {k: cm.KindParams(**{n: v for n, v in p.items() if n in names})
             for k, p in d["kinds"].items()}
    model = cm.Model(g=d["g"], alpha=d["alpha"], beta=d["beta"], gamma=d["gamma"],
                     delta=d["delta"], tail=d["tail"], kinds=kinds,
                     f1_slack=d["f1_slack"], lam=d["lambda"], retry=d["retry"])

    def veto(kind: str, k: int, row_tokens: float) -> bool:
        if kind not in model.kinds:
            return True  # no guard for this kind: do not block
        items = cm.Items(n=k, tokens=row_tokens, max_tokens=row_tokens)
        return k > model.k_max(kind, items, quality=True)

    return veto


def candidate_sizes(requested_size: int) -> tuple[int, ...]:
    """Block sizes the controller may use at a stage whose query asked for
    ``requested_size``: the search's power-of-two set up to that size."""
    from vllm.kalypso.benchmark.eval.costmodel import POW2_CANDIDATES

    return tuple(k for k in POW2_CANDIDATES if k <= requested_size) or (1,)


@dataclass
class _StageState:
    index: int
    stage_id: int
    upstream_stage_id: int
    kind: str | None
    op: object
    static_size: int
    small_size: int
    candidate_sizes: tuple[int, ...] = (1,)
    mode: str = STATIC
    row_tokens: float | None = None
    window: deque = field(default_factory=deque)
    vote: str | None = None
    streak: int = 0

    def size(self, mode: str | None = None) -> int:
        mode = mode or self.mode
        return {STATIC: self.static_size, TUPLE: 1, BLOCK: self.small_size}[mode]


class AdaptiveStageController:
    def __init__(self, stages, config: AdaptiveStageConfig, token_length=None):
        self.config = config
        self._veto = config.quality_veto
        self._token_length = token_length
        self.states: dict[int, _StageState] = {}
        self._by_upstream: dict[int, _StageState] = {}
        self.decisions: list[dict] = []   # switches and vetoes, in order
        self.formations: list[tuple[int, int]] = []  # (stage index, size) per new block
        for index, stage in enumerate(stages):
            if index == 0 or not stage.block_tasks:
                continue
            op = next((o for o in stage.operators if getattr(o, "supports_blocks", False)), None)
            if op is None:
                continue
            cands = candidate_sizes(stage.block_size)
            state = _StageState(
                index=index, stage_id=stage.stage_id,
                upstream_stage_id=stages[index - 1].stage_id, kind=operator_kind(op), op=op,
                static_size=stage.block_size,
                small_size=max(k for k in cands if k <= config.small_block_size),
                candidate_sizes=cands,
                window=deque(maxlen=config.window_tasks))
            self.states[index] = state
            self._by_upstream[state.upstream_stage_id] = state

    # -- telemetry subscriber -------------------------------------------------

    def observe(self, record: dict) -> None:
        if record.get("event") != "task_end":
            return
        state = self._by_upstream.get(record.get("stage_id"))
        if state is None or record.get("outcome") == "retry":
            return
        rows_in, passed = record.get("rows_in") or 0, record.get("rows_passed")
        if passed is None or rows_in <= 0:
            return
        state.window.append((passed, rows_in))
        rate = sum(p for p, _ in state.window) / sum(n for _, n in state.window)
        own = passed / rows_in
        if rate >= self.config.high_pass_rate and own >= self.config.high_pass_rate:
            vote = BLOCK
        elif rate < self.config.low_pass_rate and own < self.config.low_pass_rate:
            vote = TUPLE
        else:
            vote = None
        current_is_block = state.size() > 1
        if vote is None or (vote == BLOCK) == current_is_block:
            state.vote, state.streak = None, 0
            return
        state.streak = state.streak + 1 if vote == state.vote else 1
        state.vote = vote
        if state.streak < self.config.min_consecutive:
            return
        state.vote, state.streak = None, 0
        if vote == BLOCK and self._vetoed(state):
            return
        self._switch(state, vote, rate)

    def _vetoed(self, state: _StageState) -> bool:
        k = state.small_size
        reason = None
        if k <= 1:
            reason = "requested block size is 1"
        elif state.row_tokens is None:
            reason = "regime unknown: no row has reached the stage yet"
        elif state.kind is None:
            reason = "no cost-model kind for this operator"
        else:
            if self._veto is None:
                self._veto = costmodel_quality_veto()
            if self._veto(state.kind, k, state.row_tokens):
                reason = "quality guard"
        if reason is None:
            return False
        self._record(state, "veto", BLOCK, reason=reason)
        return True

    def _switch(self, state: _StageState, mode: str, rate: float) -> None:
        old = state.mode
        state.mode = mode
        self._record(state, "switch", mode, from_mode=old, pass_rate=rate)

    def _record(self, state, action, mode, **extra):
        entry = {"action": action, "stage_index": state.index, "stage_id": state.stage_id,
                 "mode": mode, "size": state.size(mode),
                 "after_formations": len(self.formations), **extra}
        self.decisions.append(entry)
        telemetry.emit("adaptive_switch", **entry)

    # -- executor hook ----------------------------------------------------------

    def formation_size(self, index: int, first_row) -> int | None:
        """Size of the block that starts forming now at stage ``index`` with
        ``first_row``. None: stage not managed (use the static size)."""
        state = self.states.get(index)
        if state is None:
            return None
        if state.row_tokens is None:
            state.row_tokens = self._row_tokens(state.op, first_row)
        size = state.size()
        self.formations.append((index, size))
        return size

    def _row_tokens(self, op, row) -> float | None:
        try:
            text = op.row_text_for_prompt(row)
            if self._token_length is not None:
                return float(self._token_length(text))
            from vllm.kalypso.budget import KVMemoryManager
            return float(KVMemoryManager.get_instance().token_length(text))
        except Exception:  # noqa: BLE001 -- unknown regime: vetoes blocking
            return None
