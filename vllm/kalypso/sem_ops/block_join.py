"""Block semantic join implemented as a Cartesian product plus block filter.

The ordinary Kalypso join expands every left/right candidate with
``CartesianProduct`` and lets ``SemFilter`` decide the predicate for each
pair. ``BlockSemJoin`` keeps that shape, but replaces the per-pair filter
with a join-aware ``BlockSemFilter``: one request decides YES/NO for a block
of already-materialized candidate pairs.

The logical planner owns the Cartesian product. This module owns only the
predicate stage, so every surviving context retains the ordinary join's
``idx`` and ``right_idx`` values.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from .block_filter import BlockFilterResult, BlockSemFilter
from .block_utils import first_executor, item_line, make_blocks, strip_context_wrapper


def _messages_text(data: Any) -> str:
    """Render one side of a joined ``SemContext`` without the system prompt."""
    messages = [data] if isinstance(data, str) else list(data or [])
    parts: list[str] = []
    for message in messages:
        if isinstance(message, str):
            parts.append(strip_context_wrapper(message))
        elif isinstance(message, dict) and message.get("role") != "system":
            parts.append(strip_context_wrapper(str(message.get("content", ""))))
    return "\n".join(part for part in parts if part)


def format_join_candidate(left: str, right: str) -> str:
    """Make pair boundaries explicit inside a numbered block-filter item."""
    return f"Left item: {item_line(left)}\nRight item: {item_line(right)}"


@dataclass
class BlockJoinResult:
    """Matches for a direct table-to-table call, using zero-based positions."""

    matches: list[tuple[int, int]]
    raw_outputs: list[str] = field(default_factory=list)


class BlockSemJoin(BlockSemFilter):
    """A join-aware block filter run after ``CartesianProduct``.

    ``pair_block_size`` is the number of already-materialized Cartesian
    candidates decided by one block-filter request.  It intentionally has no
    left/right dimensions: candidates arrive asynchronously from the
    Cartesian-product stage and are packed in their arrival order.
    """

    def __init__(
        self,
        condition: str,
        pair_block_size: int = 25,
        max_tokens_per_pair: int = 6,
        max_tokens_overhead: int = 16,
        pair_max_tokens: int = 4,
        system_prompt: str | None = None,
        right_table: Sequence[Any] | None = None,
        position: int = -1,
        max_block_tokens: int | None = None,
    ):
        if pair_block_size < 1:
            raise ValueError("pair_block_size must be >= 1")
        super().__init__(
            instruction=condition,
            block_size=pair_block_size,
            max_tokens_per_row=max_tokens_per_pair,
            max_tokens_overhead=max_tokens_overhead,
            single_max_tokens=pair_max_tokens,
            system_prompt=system_prompt,
            position=position,
            max_block_tokens=max_block_tokens,
        )
        self.condition = condition
        # Retained for compatibility with direct callers. The planner now
        # materializes this through CartesianProduct before this operator.
        self.right_table = list(right_table or [])

    def row_text_for_prompt(self, ctx) -> str:
        """Render a Cartesian-product context with explicit pair boundaries."""
        left = _messages_text(ctx.input.data)
        right = _messages_text(getattr(ctx.input, "right_data", None))
        if not getattr(ctx.input, "right_data", None):
            raise ValueError("BlockSemJoin requires Cartesian-product pair contexts")
        return format_join_candidate(left, right)

    async def evaluate_block_pair(
        self,
        executor,
        raw_request,
        left_texts: Sequence[str],
        right_texts: Sequence[str],
        priority: int = 0,
    ) -> BlockJoinResult:
        """Compatibility helper that applies the pair-filter flow to raw rows."""
        candidates = [
            (left_index, right_index, format_join_candidate(left, right))
            for left_index, left in enumerate(left_texts)
            for right_index, right in enumerate(right_texts)
        ]
        matches: list[tuple[int, int]] = []
        raw_outputs: list[str] = []
        for block in make_blocks(candidates, self.block_size):
            result: BlockFilterResult = await self.evaluate_block(
                executor,
                raw_request,
                [candidate[2] for candidate in block],
                priority,
            )
            raw_outputs.extend(result.raw_outputs)
            matches.extend(
                (left_index, right_index)
                for (left_index, right_index, _), verdict in zip(block, result.verdicts)
                if verdict
            )
        return BlockJoinResult(matches=matches, raw_outputs=raw_outputs)

    async def join(
        self, left: Sequence[Any], right: Sequence[Any], priority: int = 0
    ) -> list[tuple[int, int]]:
        """Direct table helper with the same Cartesian-product semantics."""
        if not left or not right:
            return []
        executor, raw_request = first_executor(list(left) + list(right))
        left_texts = [
            _messages_text(row.input.data) if hasattr(row, "input") else str(row)
            for row in left
        ]
        right_texts = [
            _messages_text(row.input.data) if hasattr(row, "input") else str(row)
            for row in right
        ]
        result = await self.evaluate_block_pair(
            executor, raw_request, left_texts, right_texts, priority
        )
        return result.matches
