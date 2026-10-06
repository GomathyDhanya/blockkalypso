"""Helpers shared by the block operators (block filter, block join).

Everything here is a pure function with no vLLM, torch or tokenizer import,
so it can be unit-tested in milliseconds.
"""

from __future__ import annotations

import re
from typing import Any, Sequence

_CONTEXT_WRAPPER = re.compile(
    r'\s*CONTEXT:\s*\{\s*"text":\s*(.*?)\s*\}\s*', flags=re.DOTALL
)
_VERDICT_RE = re.compile(r"\b(yes|no|true|false)\b", re.IGNORECASE)
_TRUE_WORDS = {"yes", "true"}
EMPTY_ITEM = "(empty)"


def strip_context_wrapper(text: str) -> str:
    """Undo Kalypso's ``CONTEXT: { "text": ... }`` wrapper (see prompt_utils)."""
    m = _CONTEXT_WRAPPER.fullmatch(text)
    return m.group(1).strip() if m else text.strip()


def row_text(ctx) -> str:
    """Plain text of one Kalypso row (``SemContext``) or of a bare string.

    System messages are skipped. For a row produced by a join, the right-hand
    data is appended after the left-hand data.
    """
    if isinstance(ctx, str):
        return strip_context_wrapper(ctx)
    data = ctx.input.data
    messages = [data] if isinstance(data, str) else list(data or [])
    messages += list(getattr(ctx.input, "right_data", None) or [])
    parts = []
    for m in messages:
        if isinstance(m, str):
            parts.append(strip_context_wrapper(m))
        elif isinstance(m, dict) and m.get("role") != "system":
            parts.append(strip_context_wrapper(str(m.get("content", ""))))
    return "\n".join(p for p in parts if p)


def item_line(text: str) -> str:
    """Render one item as a single prompt line.

    All whitespace, newlines included, is collapsed so that a row containing
    something like ``"\\n2. ..."`` cannot pose as an extra numbered item.
    An empty row is rendered as a placeholder so its number still appears.
    """
    flat = " ".join(strip_context_wrapper(text).split())
    return flat or EMPTY_ITEM


def make_blocks(rows: Sequence[Any], block_size: int) -> list[list[Any]]:
    """Level-1 blocking from the proposal: fixed tuple count, ``rows[i:i+N]``."""
    if block_size < 1:
        raise ValueError(f"block_size must be >= 1, got {block_size}")
    return [list(rows[i : i + block_size]) for i in range(0, len(rows), block_size)]


def make_join_blocks(
    left: Sequence[Any], right: Sequence[Any], k: int, j: int
) -> list[tuple[int, list[Any], int, list[Any]]]:
    """Every (left block, right block) pair for a k-by-j block join.

    Returns ``(left_offset, left_block, right_offset, right_block)`` so block-
    local positions can be mapped back to positions in the full tables.
    """
    lefts = make_blocks(left, k)
    rights = make_blocks(right, j)
    return [
        (li * k, lb, ri * j, rb)
        for li, lb in enumerate(lefts)
        for ri, rb in enumerate(rights)
    ]


def parse_single_answer(text: str | None) -> bool | None:
    """First YES/NO/TRUE/FALSE word in a one-item answer, or None."""
    m = _VERDICT_RE.search(text or "")
    return None if m is None else m.group(1).lower() in _TRUE_WORDS


# Sampling for block operators. Kalypso's VLLMExecutor defaults to
# frequency_penalty=0.5 and repetition_penalty=1.3; on a block answer, one
# short line per row, those penalties push the model into prose and every row
# then falls back to its own call. Deterministic and penalty-free instead.
BLOCK_SAMPLING = {
    "temperature": 0.0,
    "top_p": 1.0,
    "frequency_penalty": 0.0,
    "repetition_penalty": 1.0,
}


def with_system(system_prompt: str | None, user: str) -> list[dict]:
    """Chat messages: optional system turn, then one user turn."""
    msgs = [{"role": "user", "content": user}]
    if system_prompt:
        msgs.insert(0, {"role": "system", "content": system_prompt})
    return msgs


async def call_llm(executor, raw_request, messages, max_tokens, priority=0,
                   sampling=BLOCK_SAMPLING):
    """Call Kalypso's ``LLMExecutor.execute``; return ``(text, finish_reason)``.

    ``sampling`` is passed only to executors that accept it (Kalypso's
    VLLMExecutor does); others are called with the original signature.
    """
    kwargs = dict(raw_request=raw_request, prompt=messages, max_tokens=max_tokens,
                  pin=False, priority=priority)
    if sampling is not None and _accepts_sampling(executor):
        kwargs["sampling"] = sampling
    out = await executor.execute(**kwargs)
    return (out.text or ""), getattr(out, "finish_reason", None)


def _accepts_sampling(executor) -> bool:
    import inspect

    try:
        params = inspect.signature(executor.execute).parameters
    except (TypeError, ValueError):
        return False
    return "sampling" in params or any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())


def first_executor(ctxs):
    """Executor and raw request of the first row that has one.

    Rows loaded as a join's right table have no executor, so callers pass
    the left rows first.
    """
    for c in ctxs:
        state = getattr(c, "state", None)
        if state is not None and getattr(state, "executor", None) is not None:
            return state.executor, state.raw_request
    raise ValueError("no row in the block carries an executor")
