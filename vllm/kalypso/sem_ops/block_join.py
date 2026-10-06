"""Block semantic join: prompt construction and response parsing.

One LLM call compares a block of left rows with a block of right rows and
returns every matching pair.

Milestone 1, brute force:

* left rows are labeled ``L1..Lk`` and right rows ``R1..Rj`` (block-local);
* one prompt carries the join condition once, the left block and the right
  block, and asks for the matching pairs as lines ``L<i>, R<j>``, or the
  single word ``NONE`` when nothing matches;
* the response is parsed into pairs and every pair is validated against the
  ids that were actually in the two blocks (proposal step 3);
* pairs naming ids that are not in the blocks are dropped and counted. If
  the response cannot be parsed at all (no pair lines and no ``NONE``), the
  block is answered pair by pair with one call per (left, right) pair, the
  same as Kalypso's per-pair join. Candidate sub-blocking repair is
  Milestone 2.

Accuracy is not a Milestone 1 goal; ``BlockJoinStats`` records invalid and
duplicate pairs so it can be studied later.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

from .base import BaseOp, OpBehavior
from .block_utils import (
    call_llm,
    first_executor,
    item_line,
    make_blocks,
    make_join_blocks,
    parse_single_answer,
    row_text,
    with_system,
)

# Ids in an answer line: "L1", "l 3", "R12". Pairs are formed per line from
# consecutive left/right ids, so "L1, R3", "(L1, R3)", "L1 - R3", "R3, L1",
# "**L1, R3**" and "L1, R1; L2, R2" all parse, and ids never pair across lines.
_ID_RE = re.compile(r"(?<![A-Za-z])([LR])\s*(\d+)(?![0-9])", re.IGNORECASE)


def _pairs_in_line(line: str) -> list[tuple[int, int]]:
    ids = [(side.upper(), int(num)) for side, num in _ID_RE.findall(line)]
    pairs, i = [], 0
    while i + 1 < len(ids):
        (s1, n1), (s2, n2) = ids[i], ids[i + 1]
        if s1 != s2:
            pairs.append((n1, n2) if s1 == "L" else (n2, n1))
            i += 2
        else:
            i += 1
    return pairs


_NONE_RE = re.compile(r"^\W*(NONE|NO MATCHES?|NO PAIRS?)\W*$", re.IGNORECASE | re.MULTILINE)


def build_block_join_messages(
    condition: str,
    left_texts: Sequence[str],
    right_texts: Sequence[str],
    system_prompt: str | None = None,
) -> list[dict]:
    """Prompt for one k-by-j block pair. Ids are 1-based and block-local."""
    left = "\n".join(f"L{i}. {item_line(t)}" for i, t in enumerate(left_texts, 1))
    right = "\n".join(f"R{j}. {item_line(t)}" for j, t in enumerate(right_texts, 1))
    user = (
        f"Join condition: {condition}\n\n"
        f"Left items:\n{left}\n\n"
        f"Right items:\n{right}\n\n"
        "Check every left item against every right item. List every pair that "
        "satisfies the join condition, one per line, as `L<number>, R<number>`. "
        "If no pair satisfies it, write NONE. Write nothing else."
    )
    return with_system(system_prompt, user)


def build_pair_messages(
    condition: str, left_text: str, right_text: str, system_prompt: str | None = None
) -> list[dict]:
    """Prompt for one (left, right) pair: the fallback and the per-pair baseline."""
    return with_system(
        system_prompt,
        f"Left item: {item_line(left_text)}\n"
        f"Right item: {item_line(right_text)}\n\n"
        f"Join condition: {condition}\n"
        "Do the two items satisfy the join condition? Answer YES or NO only.",
    )


@dataclass
class ParsedPairs:
    """Result of parsing one block-join response."""

    pairs: list[tuple[int, int]]      # valid, de-duplicated, in output order
    invalid: list[tuple[int, int]]    # pairs naming an id outside the blocks
    duplicates: int                   # repeated valid pairs that were dropped
    said_none: bool                   # the model answered NONE
    parseable: bool                   # pairs or NONE were found


def parse_block_join(text: str | None, k: int, j: int) -> ParsedPairs:
    """Parse ``L<i>, R<j>`` lines and validate ids against a k-by-j block."""
    text = text or ""
    found = [pair for line in text.splitlines() for pair in _pairs_in_line(line)]
    said_none = bool(_NONE_RE.search(text))

    pairs: list[tuple[int, int]] = []
    invalid: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    duplicates = 0
    for li, rj in found:
        if not (1 <= li <= k and 1 <= rj <= j):
            invalid.append((li, rj))
        elif (li, rj) in seen:
            duplicates += 1
        else:
            seen.add((li, rj))
            pairs.append((li, rj))
    return ParsedPairs(
        pairs=pairs,
        invalid=invalid,
        duplicates=duplicates,
        said_none=said_none and not found,
        parseable=bool(found) or said_none,
    )


@dataclass
class BlockJoinStats:
    """Counters for one operator instance, summed over every block pair."""

    block_pairs: int = 0
    candidate_pairs: int = 0
    block_calls: int = 0
    fallback_calls: int = 0
    matches: int = 0
    invalid_pairs: int = 0
    duplicate_pairs: int = 0
    unparseable_blocks: int = 0
    truncated_block_calls: int = 0

    @property
    def llm_calls(self) -> int:
        return self.block_calls + self.fallback_calls

    def as_dict(self) -> dict[str, int]:
        return {**asdict(self), "llm_calls": self.llm_calls}


@dataclass
class BlockJoinResult:
    """Matches for one block pair, as positions within the two blocks (0-based)."""

    matches: list[tuple[int, int]]
    invalid: list[tuple[int, int]] = field(default_factory=list)
    raw_outputs: list[str] = field(default_factory=list)


class BlockSemJoin(BaseOp):
    """Semantic join that evaluates a k-by-j block of candidates per call."""

    supports_blocks = True
    requires_block_task = True

    def __init__(
        self,
        condition: str,
        left_block_size: int = 5,
        right_block_size: int = 5,
        max_tokens_per_pair: int = 6,
        max_tokens_overhead: int = 16,
        pair_max_tokens: int = 4,
        system_prompt: str | None = None,
        right_table: Sequence[Any] | None = None,
        position: int = -1,
    ):
        super().__init__(behavior=OpBehavior.TUPLE_INDEPENDENT, position=position)
        if left_block_size < 1 or right_block_size < 1:
            raise ValueError("block sizes must be >= 1")
        self.condition = condition
        self.left_block_size = left_block_size
        self.right_block_size = right_block_size
        self.block_size = left_block_size
        self.right_table = list(right_table or [])
        self.max_tokens_per_pair = max_tokens_per_pair
        self.max_tokens_overhead = max_tokens_overhead
        self.pair_max_tokens = pair_max_tokens
        self.system_prompt = system_prompt
        self.pin = False
        self.unpin = False
        self.stats = BlockJoinStats()

    def block_max_tokens(self, k: int, j: int) -> int:
        # Worst case every pair matches; NONE fits in the overhead.
        return self.max_tokens_overhead + self.max_tokens_per_pair * k * j

    def estimate_tokens(self, left_ctxs, right_ctxs=None, token_length=None) -> int:
        """Prompt + output tokens for one block pair (``token_length``: str -> int)."""
        if right_ctxs is None:
            from vllm.kalypso.budget import KVMemoryManager

            token_length = KVMemoryManager.get_instance().token_length
            return max((
                self.estimate_tokens(left_ctxs, block, token_length)
                for block in make_blocks(self.right_table, self.right_block_size)
            ), default=0)
        if not left_ctxs or not right_ctxs:
            return 0
        msgs = build_block_join_messages(
            self.condition,
            [row_text(c) for c in left_ctxs],
            [row_text(c) for c in right_ctxs],
            self.system_prompt,
        )
        text = "\n".join(m["content"] for m in msgs)
        return token_length(text) + self.block_max_tokens(len(left_ctxs), len(right_ctxs))

    async def run_block(self, left_ctxs, priority: int = 0):
        """Evaluate one incoming left block and emit Kalypso join contexts."""
        if not left_ctxs or not self.right_table:
            return []
        from .joins import CartesianProduct

        executor, raw_request = first_executor(left_ctxs)
        left_texts = [row_text(ctx) for ctx in left_ctxs]
        right_blocks = make_join_blocks(
            left_texts, self.right_table,
            len(left_ctxs), self.right_block_size,
        )
        # A pipeline task reserves room for its largest pair prompt. Run the
        # right blocks one at a time so concurrent calls cannot exceed it.
        results = [
            await self.evaluate_block_pair(
                executor, raw_request, left_texts,
                [row_text(ctx) for ctx in right_block], priority,
            )
            for _, _, _, right_block in right_blocks
        ]
        joined = []
        for (_, _, right_offset, _), result in zip(right_blocks, results):
            for left_pos, right_pos in result.matches:
                right_ctx = self.right_table[right_offset + right_pos]
                left_ctx = left_ctxs[left_pos]
                for pair in CartesianProduct([right_ctx])(left_ctx):
                    pair.output = list(left_ctx.output)
                    joined.append(pair)
        return joined

    async def evaluate_block_pair(
        self,
        executor,
        raw_request,
        left_texts: Sequence[str],
        right_texts: Sequence[str],
        priority: int = 0,
    ) -> BlockJoinResult:
        """One call for a k-by-j block pair; per-pair calls if unparseable."""
        k, j = len(left_texts), len(right_texts)
        self.stats.block_pairs += 1
        self.stats.candidate_pairs += k * j
        if k == 0 or j == 0:
            return BlockJoinResult(matches=[])
        if k * j == 1:
            return await self._pairwise(executor, raw_request, left_texts, right_texts, priority)

        self.stats.block_calls += 1
        text, finish = await call_llm(
            executor, raw_request,
            build_block_join_messages(self.condition, left_texts, right_texts, self.system_prompt),
            self.block_max_tokens(k, j), priority,
        )
        if finish == "length":
            self.stats.truncated_block_calls += 1
        parsed = parse_block_join(text, k, j)
        self.stats.invalid_pairs += len(parsed.invalid)
        self.stats.duplicate_pairs += parsed.duplicates

        if not parsed.parseable:
            self.stats.unparseable_blocks += 1
            result = await self._pairwise(executor, raw_request, left_texts, right_texts, priority)
            result.raw_outputs.insert(0, text)
            return result

        matches = [(li - 1, rj - 1) for li, rj in parsed.pairs]
        self.stats.matches += len(matches)
        return BlockJoinResult(matches=matches, invalid=parsed.invalid, raw_outputs=[text])

    async def _pairwise(self, executor, raw_request, left_texts, right_texts, priority):
        """Exhaustive one-call-per-pair fallback (the per-pair baseline)."""
        cells = [(a, b) for a in range(len(left_texts)) for b in range(len(right_texts))]
        replies = [
            await call_llm(
                executor, raw_request,
                build_pair_messages(self.condition, left_texts[a], right_texts[b], self.system_prompt),
                self.pair_max_tokens, priority,
            )
            for a, b in cells
        ]
        self.stats.fallback_calls += len(cells)
        matches = [cell for cell, (text, _) in zip(cells, replies) if parse_single_answer(text)]
        self.stats.matches += len(matches)
        return BlockJoinResult(matches=matches, raw_outputs=[t for t, _ in replies])

    async def join(self, left: Sequence[Any], right: Sequence[Any], priority: int = 0):
        """Join two tables of Kalypso rows (or strings).

        Returns ``(left_position, right_position)`` pairs into the two input
        lists, sorted. All k-by-j block pairs run concurrently.
        """
        if not left or not right:
            return []
        executor, raw_request = first_executor(list(left) + list(right))
        left_texts = [row_text(r) for r in left]
        right_texts = [row_text(r) for r in right]
        blocks = make_join_blocks(left_texts, right_texts,
                                  self.left_block_size, self.right_block_size)
        results = await asyncio.gather(*(
            self.evaluate_block_pair(executor, raw_request, lb, rb, priority)
            for _, lb, _, rb in blocks
        ))
        return sorted(
            (lo + a, ro + b)
            for (lo, _, ro, _), res in zip(blocks, results)
            for a, b in res.matches
        )
