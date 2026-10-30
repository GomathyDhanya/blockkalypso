"""Block semantic map: one LLM request maps a block of Kalypso rows.

Each successful block answer is materialized as ordinary per-row ``SemMap``
state: the ordinary map prompt plus an assistant turn becomes the row's new
input.  This is logical row state, not a transcript of the shared JSON block
request.  Keeping the shared transcript would expose other rows to later
operators; consequently a later operator cannot reuse the block request's KV
prefix. Missing or malformed entries are repaired with the ordinary
single-row map path.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict, dataclass, field
from typing import Sequence

from .base import OpBehavior
from .block_utils import BLOCK_SAMPLING, call_llm, first_executor, make_blocks, row_text
from .map import SemMap
from .prompt_utils import add_assistant_prompt
from vllm.kalypso.budget import KVMemoryManager
from vllm.kalypso.controller.map_estimator import MapRatioEstimator
from vllm.kalypso.context import RETRY_TASK


def build_block_map_messages(instruction: str, texts: Sequence[str]) -> list[dict]:
    """Build a JSON-safe, one-request prompt for several map inputs.

    JSON strings preserve row whitespace and prevent a row containing a line
    such as ``2. ...`` from becoming a synthetic block item.
    """
    items = [
        {"id": index, "context": text}
        for index, text in enumerate(texts, start=1)
    ]
    user = (
        "You are given several independent contexts and one mapping instruction.\n"
        f"Mapping instruction: {instruction}\n\n"
        "Contexts (JSON):\n"
        f"{json.dumps(items, ensure_ascii=False)}\n\n"
        "Apply the instruction independently to every context. Return JSON only, "
        'with exactly this shape: {"items": [{"id": 1, "output": "..."}]}. '
        "Include one object for every input id. Each output must be a JSON string."
    )
    return [{"role": "user", "content": user}]


def parse_block_map_outputs(text: str | None, n: int) -> dict[int, str]:
    """Return unambiguous, in-range ``id -> output`` entries from JSON only."""
    if not text or n < 1:
        return {}
    candidate = text.strip()
    if candidate.startswith("```") and candidate.endswith("```"):
        candidate = candidate.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        value = json.loads(candidate)
    except (TypeError, ValueError):
        return {}
    entries = value.get("items") if isinstance(value, dict) else None
    if not isinstance(entries, list):
        return {}

    seen: dict[int, set[str]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        item_id, output = entry.get("id"), entry.get("output")
        # bool is an int subclass, but must not be accepted as an item id.
        if isinstance(item_id, bool) or not isinstance(item_id, int):
            continue
        if not isinstance(output, str):
            continue
        seen.setdefault(item_id, set()).add(output)
    return {
        item_id: next(iter(outputs))
        for item_id, outputs in seen.items()
        if 1 <= item_id <= n and len(outputs) == 1
    }


@dataclass
class BlockMapStats:
    blocks: int = 0
    rows: int = 0
    block_calls: int = 0
    fallback_calls: int = 0
    rows_from_block: int = 0
    rows_from_fallback: int = 0
    truncated_block_calls: int = 0
    malformed_block_calls: int = 0

    @property
    def llm_calls(self) -> int:
        return self.block_calls + self.fallback_calls

    def as_dict(self) -> dict[str, int]:
        return {**asdict(self), "llm_calls": self.llm_calls}


@dataclass
class BlockMapResult:
    outputs: list[str | None]
    raw_outputs: list[str] = field(default_factory=list)


class BlockSemMap(SemMap):
    """A block-aware ``SemMap`` with ordinary-map repair for bad entries."""

    supports_blocks = True

    def __init__(
        self,
        instruction: str,
        block_size: int = 8,
        max_tokens_overhead: int = 32,
        max_block_tokens: int | None = None,
        position: int = -1,
        pin: bool = False,
        unpin: bool = False,
    ):
        if block_size < 1:
            raise ValueError(f"block_size must be >= 1, got {block_size}")
        if pin:
            raise ValueError(
                "BlockSemMap does not support pin=True: one block request "
                "cannot own independent per-row KV pins."
            )
        if max_block_tokens is not None and max_block_tokens < 1:
            raise ValueError("max_block_tokens must be positive")
        super().__init__(instruction=instruction, position=position, pin=pin, unpin=unpin)
        self.behavior = OpBehavior.TUPLE_INDEPENDENT
        self.block_size = block_size
        self.max_tokens_overhead = max_tokens_overhead
        self.max_block_tokens = max_block_tokens
        self.stats = BlockMapStats()

    def row_max_tokens(self, ctx) -> int:
        """Output tokens one row may need, by ``SemMap``'s own rule.

        ``SemMap._planned_max_tokens`` scales the row's prompt length by the
        ratio ``MapRatioEstimator`` has learned for this operator, or uses the
        prompt length before any row has finished. A row's share is capped at
        ``SemMap.MAX_TOKEN_LIMIT``, the length ``SemMap`` retries a cut-off
        row with, so one long prompt cannot inflate the whole block request.
        """
        return max(1, min(SemMap.MAX_TOKEN_LIMIT, self._planned_max_tokens(ctx)))

    def block_max_tokens(self, ctxs) -> int:
        """Output tokens for one block: every row's ``SemMap`` share plus overhead."""
        return self.max_tokens_overhead + sum(self.row_max_tokens(ctx) for ctx in ctxs)

    def estimate_tokens(self, ctxs, token_length=None) -> int:
        """Peak tokens for one block request or its sequential row repairs."""
        ctxs = ctxs if isinstance(ctxs, list) else [ctxs]
        if len(ctxs) <= 1:
            return super().estimate_tokens(ctxs[0])
        texts = [row_text(ctx) for ctx in ctxs]
        if token_length is None:
            manager = KVMemoryManager.get_instance()
            count = lambda messages: manager.token_length(manager.apply_chat_template(messages))
        else:
            count = lambda messages: token_length("\n".join(m["content"] for m in messages))
        block_peak = count(build_block_map_messages(self.instruction, texts)) + self.block_max_tokens(ctxs)
        single_peak = max(
            (SemMap.estimate_tokens(self, ctx) for ctx in ctxs), default=0
        )
        return max(block_peak, single_peak)

    async def _apply_block_output(self, ctx, output_text: str) -> None:
        """Materialize logical per-row state, as ``SemMap._run_single`` does.

        This deliberately differs from the request that produced
        ``output_text``: the request was one shared block prompt, whereas the
        stored state must be independent and must not include sibling rows.
        It therefore cannot reuse that block request as a vLLM KV prefix.
        """
        previous_pin_req_id = ctx.state.pin_req_id
        prompt = self._build_prompt(ctx)
        prompt_str = KVMemoryManager.get_instance().apply_chat_template(prompt)
        input_token_len = KVMemoryManager.get_instance().token_length(prompt_str)
        appended_prompt, appended_prompt_str = add_assistant_prompt(prompt, output_text)
        ctx.input.data = appended_prompt
        MapRatioEstimator.instance().update(
            self.position,
            input_token_len,
            KVMemoryManager.get_instance().token_length(appended_prompt_str) - input_token_len,
        )
        ctx.output.append({str(self.__class__): output_text})
        if self.unpin and previous_pin_req_id is not None:
            await ctx.state.executor.unpin(ctx.state.raw_request, previous_pin_req_id)
            ctx.state.pin_req_id = None

    async def _run_single_map(self, ctx, priority: int):
        """Use the inherited retry and pin behaviour for a repair call."""
        while True:
            result = await self._run_single(ctx, priority=priority)
            if result is not RETRY_TASK:
                return result

    async def evaluate_block(
        self, executor, raw_request, texts: Sequence[str], priority: int = 0,
        max_tokens: int | None = None,
    ) -> BlockMapResult:
        """One block request. ``run_block`` passes ``max_tokens`` from its rows."""
        n = len(texts)
        self.stats.blocks += 1
        self.stats.rows += n
        if n <= 1:
            return BlockMapResult(outputs=[None] * n)

        self.stats.block_calls += 1
        raw, finish = await call_llm(
            executor,
            raw_request,
            build_block_map_messages(self.instruction, texts),
            max_tokens if max_tokens is not None
            else self.max_tokens_overhead + SemMap.MAX_TOKEN_LIMIT * n,
            priority,
            sampling=BLOCK_SAMPLING,
        )
        answers = parse_block_map_outputs(raw, n)
        if finish == "length":
            self.stats.truncated_block_calls += 1
        if not answers:
            self.stats.malformed_block_calls += 1
        self.stats.rows_from_block += len(answers)
        return BlockMapResult(
            outputs=[answers.get(index) for index in range(1, n + 1)],
            raw_outputs=[raw],
        )

    async def run_block(self, ctxs: list, priority: int = 0) -> list:
        """Map every row, repairing absent block outputs with ordinary SemMap."""
        if not ctxs:
            return []
        if len(ctxs) == 1:
            await self._run_single_map(ctxs[0], priority)
            return ctxs

        executor, raw_request = first_executor(ctxs)
        result = await self.evaluate_block(
            executor, raw_request, [row_text(ctx) for ctx in ctxs], priority,
            max_tokens=self.block_max_tokens(ctxs),
        )
        for ctx, output in zip(ctxs, result.outputs):
            if output is not None:
                if getattr(ctx.state, "retry_op_position", -1) == self.position:
                    ctx.state.retry_op_position = -1
                    ctx.state.retry_max_tokens = 0
                await self._apply_block_output(ctx, output)

        for ctx, output in zip(ctxs, result.outputs):
            if output is None:
                self.stats.fallback_calls += 1
                await self._run_single_map(ctx, priority)
                self.stats.rows_from_fallback += 1
        return ctxs

    async def __call__(self, ctx, priority: int = 0):
        if isinstance(ctx, list):
            blocks = await asyncio.gather(
                *(self.run_block(block, priority) for block in make_blocks(ctx, self.block_size))
            )
            return [row for block in blocks for row in block]
        return await super().__call__(ctx, priority=priority)
