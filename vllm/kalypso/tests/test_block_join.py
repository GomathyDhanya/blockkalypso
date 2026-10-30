"""Unit tests for the Cartesian-product block join (no model needed)."""

import asyncio
import re
from dataclasses import dataclass, field

import pytest

from vllm.kalypso.sem_ops.block_join import BlockSemJoin, format_join_candidate

from .test_block_filter import FakeOut, row


def run(coro):
    return asyncio.run(coro)


@dataclass
class JoinExecutor:
    """Answers each candidate independently, like ordinary SemFilter."""

    match: callable
    block_replies: list = field(default_factory=list)
    calls: list = field(default_factory=list)

    async def execute(self, raw_request, prompt, max_tokens, pin=False, priority=0):
        self.calls.append((prompt, max_tokens))
        body = prompt[-1]["content"]
        if body.startswith("Item: "):
            left, right = _pair_from_text(body)
            return FakeOut("YES" if self.match(left, right) else "NO")
        if self.block_replies:
            reply = self.block_replies.pop(0)
            return reply if isinstance(reply, FakeOut) else FakeOut(reply)
        pairs = [_pair_from_text(item) for _, item in re.findall(
            r"^(\d+)\. (.*)$", body, re.MULTILINE
        )]
        return FakeOut("\n".join(
            f"{index}: {'YES' if self.match(left, right) else 'NO'}"
            for index, (left, right) in enumerate(pairs, start=1)
        ))


def _pair_from_text(text):
    match = re.search(r"Left item: (.*?)\s+Right item: (.*?)(?:\n\n|\nQuestion:|$)", text, re.DOTALL)
    assert match, text
    return match.group(1).strip(), match.group(2).strip()


def same_letter(left, right):
    return left[0] == right[0]


def test_candidate_keeps_left_and_right_labels():
    assert format_join_candidate("left\nrow", "right\nrow") == (
        "Left item: left row\nRight item: right row"
    )


def test_one_block_filter_call_decides_every_pair():
    ex = JoinExecutor(same_letter)
    op = BlockSemJoin("same category", pair_block_size=6)
    result = run(op.evaluate_block_pair(ex, None, ["a1", "b1"], ["a2", "c2", "b2"]))
    assert sorted(result.matches) == [(0, 0), (1, 2)]
    assert op.block_size == 6
    assert op.stats.llm_calls == 1


def test_missing_block_answer_falls_back_for_that_candidate():
    ex = JoinExecutor(same_letter, block_replies=["1: YES\n3: NO\n4: YES"])
    op = BlockSemJoin("same category", pair_block_size=4)
    result = run(op.evaluate_block_pair(ex, None, ["a", "b"], ["a", "b"]))
    assert sorted(result.matches) == [(0, 0), (1, 1)]
    assert (op.stats.block_calls, op.stats.fallback_calls) == (1, 1)


def test_pair_block_size_controls_how_candidates_are_packed():
    ex = JoinExecutor(same_letter)
    op = BlockSemJoin("same category", pair_block_size=2)
    result = run(op.evaluate_block_pair(ex, None, ["a", "b"], ["a", "b", "c"]))
    assert sorted(result.matches) == [(0, 0), (1, 1)]
    assert op.stats.block_calls == 3


def test_direct_join_uses_the_same_pair_filter_logic():
    ex = JoinExecutor(same_letter)
    left = [row("a1", 0, ex), row("b1", 1, ex)]
    right = [row("a2", 2), row("c2", 3)]
    assert run(BlockSemJoin("same category", pair_block_size=4).join(
        left, right
    )) == [(0, 0)]


def test_invalid_pair_block_size():
    with pytest.raises(ValueError):
        BlockSemJoin("c", pair_block_size=0)
