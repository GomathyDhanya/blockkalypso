"""Block semantic filter: one LLM call decides a predicate for a whole block.

Milestone 1, brute force:

* rows are cut into fixed-size blocks (proposal Level 1, ``rows[i:i+N]``);
* one prompt carries the instruction once plus every row of the block, each
  numbered with a block-local id ``1..n``;
* the model answers one ``<id>: YES|NO`` line per row (TRUE/FALSE accepted);
* a row whose answer is missing, conflicting or out of range is answered by
  its own one-row call (the proposal's ``SINGLE_TUPLE_CALL``). Recursive
  sub-blocking on partial failure is Milestone 2.

Accuracy is not a Milestone 1 goal; ``BlockFilterStats`` records how many
rows the block call actually answered so it can be studied later.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import asdict, dataclass, field
from typing import Sequence

from .base import BaseOp, OpBehavior
from .block_utils import (
    call_llm,
    first_executor,
    item_line,
    make_blocks,
    parse_single_answer,
    row_text,
    with_system,
)

# Optional system prompt. Off by default: on Qwen2.5-0.5B/1.5B a system turn
# cut one-row accuracy on the 20-review sample from 16/20 to 10/20.
BLOCK_SYSTEM_PROMPT = (
    "You are a helpful assistant for executing semantic operators.\n"
    "Judge every numbered item independently of the others.\n"
)

# One answer line: "3: YES", "[3] no", "3 - TRUE", "**3**: False", "3. Yes" ...
# Anchored to the start of a line so "1: NO 2: YES" on one line only counts 1.
_ANSWER_RE = re.compile(
    r"^[\s*>#`-]*\[?\s*(\d+)\s*\]?[*\s]*[:.)=\]-]?[\s*]*(YES|NO|TRUE|FALSE)\b",
    re.IGNORECASE | re.MULTILINE,
)
_TRUE_WORDS = {"yes", "true"}


def build_block_messages(
    instruction: str, texts: Sequence[str], system_prompt: str | None = None
) -> list[dict]:
    """Prompt for one block; item ids are 1-based and local to the block."""
    n = len(texts)
    items = "\n".join(f"{i}. {item_line(t)}" for i, t in enumerate(texts, start=1))
    user = (
        f"Question for each item: {instruction}\n\n"
        f"Items:\n{items}\n\n"
        f"Answer the question separately for each of the {n} items.\n"
        f"Write exactly {n} lines, in order, as `<number>: YES` or "
        "`<number>: NO`, and nothing else."
    )
    return with_system(system_prompt, user)


def build_single_messages(
    instruction: str, text: str, system_prompt: str | None = None
) -> list[dict]:
    """Prompt for one row: the fallback path and the per-tuple baseline."""
    return with_system(
        system_prompt,
        f"Item: {item_line(text)}\n\nQuestion: {instruction}\nAnswer YES or NO only.",
    )


def parse_block_answers(text: str | None, n: int) -> dict[int, bool]:
    """Map block-local id (1..n) to a verdict.

    Ids outside 1..n are ignored. An id repeated with the same verdict is
    kept; an id repeated with conflicting verdicts is dropped so that row
    goes to the fallback. If the model numbered the items 0..n-1 (all n
    answers, ids exactly 0..n-1) the ids are shifted by one.
    """
    seen: dict[int, set[bool]] = {}
    for m in _ANSWER_RE.finditer(text or ""):
        seen.setdefault(int(m.group(1)), set()).add(m.group(2).lower() in _TRUE_WORDS)
    if n > 0 and set(seen) == set(range(n)):
        seen = {i + 1: v for i, v in seen.items()}
    return {i: next(iter(v)) for i, v in seen.items() if 1 <= i <= n and len(v) == 1}


@dataclass
class BlockFilterStats:
    """Counters for one operator instance, summed over every block it ran."""

    blocks: int = 0
    rows: int = 0
    block_calls: int = 0
    fallback_calls: int = 0
    rows_from_block: int = 0
    rows_from_fallback: int = 0
    rows_undecided: int = 0
    truncated_block_calls: int = 0

    @property
    def llm_calls(self) -> int:
        return self.block_calls + self.fallback_calls

    def as_dict(self) -> dict[str, int]:
        return {**asdict(self), "llm_calls": self.llm_calls}


@dataclass
class BlockFilterResult:
    verdicts: list[bool | None]
    raw_outputs: list[str] = field(default_factory=list)


class BlockSemFilter(BaseOp):
    """Semantic filter that evaluates a whole block of rows per LLM call."""

    # Tells Kalypso's Stage to schedule this operator one block per task.
    supports_blocks = True

    def __init__(
        self,
        instruction: str,
        block_size: int = 8,
        negate: bool = False,
        max_tokens_per_row: int = 6,
        max_tokens_overhead: int = 16,
        single_max_tokens: int = 4,
        system_prompt: str | None = None,
        position: int = -1,
        predicate: bool = True,
    ):
        super().__init__(
            behavior=OpBehavior.TUPLE_INDEPENDENT,
            position=position,
            predicate=predicate,
        )
        if block_size < 1:
            raise ValueError(f"block_size must be >= 1, got {block_size}")
        self.instruction = instruction
        self.block_size = block_size
        self.negate = negate
        self.max_tokens_per_row = max_tokens_per_row
        self.max_tokens_overhead = max_tokens_overhead
        self.single_max_tokens = single_max_tokens
        self.system_prompt = system_prompt
        # Kalypso's plan builder resets these; block operators never pin yet.
        self.pin = False
        self.unpin = False
        self.stats = BlockFilterStats()

    # ---- budgeting (Kalypso convention: prompt tokens + max output tokens) --

    def block_max_tokens(self, n: int) -> int:
        return self.max_tokens_overhead + self.max_tokens_per_row * n

    def estimate_tokens(self, ctxs, token_length=None) -> int:
        """Tokens one block call can occupy: its prompt plus its output cap.

        ``token_length`` maps text to a token count; by default Kalypso's
        KVMemoryManager tokenizer is used (it must be initialised).
        """
        ctxs = ctxs if isinstance(ctxs, list) else [ctxs]
        msgs = build_block_messages(
            self.instruction, [row_text(c) for c in ctxs], self.system_prompt
        )
        if token_length is None:
            from vllm.kalypso.budget import KVMemoryManager

            mgr = KVMemoryManager.get_instance()
            prompt_tokens = mgr.token_length(mgr.apply_chat_template(msgs))
        else:
            prompt_tokens = token_length("\n".join(m["content"] for m in msgs))
        out = self.block_max_tokens(len(ctxs)) if len(ctxs) > 1 else self.single_max_tokens
        return prompt_tokens + out

    # ---- execution ---------------------------------------------------------

    async def evaluate_block(
        self, executor, raw_request, texts: Sequence[str], priority: int = 0
    ) -> BlockFilterResult:
        """One block call, then a one-row call for every unusable answer."""
        n = len(texts)
        self.stats.blocks += 1
        self.stats.rows += n
        raw: list[str] = []
        answers: dict[int, bool] = {}
        if n > 1:
            self.stats.block_calls += 1
            text, finish = await call_llm(
                executor, raw_request,
                build_block_messages(self.instruction, texts, self.system_prompt),
                self.block_max_tokens(n), priority,
            )
            raw.append(text)
            if finish == "length":
                self.stats.truncated_block_calls += 1
            answers = parse_block_answers(text, n)
            self.stats.rows_from_block += len(answers)

        verdicts: list[bool | None] = [answers.get(i) for i in range(1, n + 1)]
        missing = [i for i, v in enumerate(verdicts) if v is None]
        replies = await asyncio.gather(*(
            call_llm(
                executor, raw_request,
                build_single_messages(self.instruction, texts[i], self.system_prompt),
                self.single_max_tokens, priority,
            )
            for i in missing
        ))
        self.stats.fallback_calls += len(missing)
        for i, (text, _) in zip(missing, replies):
            raw.append(text)
            verdicts[i] = parse_single_answer(text)
            if verdicts[i] is None:
                self.stats.rows_undecided += 1
            else:
                self.stats.rows_from_fallback += 1

        if self.negate:
            verdicts = [None if v is None else not v for v in verdicts]
        return BlockFilterResult(verdicts=verdicts, raw_outputs=raw)

    async def run_block(self, ctxs: list, priority: int = 0) -> list:
        """Filter one block of Kalypso rows; return the rows that pass.

        Every row gets a ``"true"``, ``"false"`` or ``"undecided"`` output
        entry. An undecided row is dropped. (Kalypso's SemFilter instead
        passes any answer that does not contain "false".)
        """
        if not ctxs:
            return []
        executor, raw_request = first_executor(ctxs)
        result = await self.evaluate_block(
            executor, raw_request, [row_text(c) for c in ctxs], priority
        )
        label = {True: "true", False: "false", None: "undecided"}
        passed = []
        for ctx, v in zip(ctxs, result.verdicts):
            ctx.output.append({str(self.__class__): label[v]})
            if v:
                passed.append(ctx)
        return passed

    async def __call__(self, ctx, priority: int = 0):
        """Kalypso operator entry point.

        * a list of rows is one table: it is cut into ``block_size`` blocks
          that run concurrently, and the passing rows come back in order;
        * a single row is a block of one, returning a pass/fail bool like
          Kalypso's per-tuple ``SemFilter``.
        """
        if isinstance(ctx, list):
            results = await asyncio.gather(
                *(self.run_block(b, priority) for b in make_blocks(ctx, self.block_size))
            )
            return [c for block in results for c in block]
        passed = await self.run_block([ctx], priority)
        return bool(passed) if self.predicate else ctx
