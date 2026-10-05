"""Unit tests for the block semantic join (no model or server needed)."""

import asyncio
import re
from dataclasses import dataclass, field

import pytest

from vllm.kalypso.sem_ops.block_join import (
    BlockSemJoin,
    build_block_join_messages,
    build_pair_messages,
    parse_block_join,
)

from .test_block_filter import FakeOut, row


def run(coro):
    return asyncio.run(coro)


@dataclass
class JoinExecutor:
    """Block prompts get scripted replies, or answers computed from ``match``.

    ``match(left_text, right_text) -> bool`` is the ground truth.
    """

    match: callable
    block_replies: list = field(default_factory=list)
    calls: list = field(default_factory=list)

    async def execute(self, raw_request, prompt, max_tokens, pin=False, priority=0):
        self.calls.append(max_tokens)
        body = prompt[-1]["content"]
        if body.startswith("Left item: "):
            left = re.search(r"Left item: (.*)\n", body).group(1)
            right = re.search(r"Right item: (.*)\n", body).group(1)
            return FakeOut("YES" if self.match(left, right) else "NO")
        if self.block_replies:
            reply = self.block_replies.pop(0)
            return reply if isinstance(reply, FakeOut) else FakeOut(reply)
        lefts = re.findall(r"^L(\d+)\. (.*)$", body, re.M)
        rights = re.findall(r"^R(\d+)\. (.*)$", body, re.M)
        lines = [f"L{a}, R{b}" for a, lt in lefts for b, rt in rights if self.match(lt, rt)]
        return FakeOut("\n".join(lines) or "NONE")


same_letter = lambda left, right: left[0] == right[0]


# ---------------- parsing ----------------

def test_parse_clean_pairs():
    p = parse_block_join("L1, R2\nL3, R1", 3, 2)
    assert p.pairs == [(1, 2), (3, 1)] and p.parseable and not p.said_none


@pytest.mark.parametrize("text", [
    "L1, R2", "(L1, R2)", "L1 - R2", "L1 & R2", "L1,R2", "l1, r2",
    "**L1, R2**", "- L1, R2", "L1 , R2", "R2, L1", "L1 and R2", "L1R2",
])
def test_parse_tolerates_pair_formats(text):
    assert parse_block_join(text, 2, 2).pairs == [(1, 2)]


def test_parse_several_pairs_on_one_line():
    assert parse_block_join("L1, R1; L2, R2", 2, 2).pairs == [(1, 1), (2, 2)]


def test_parse_none():
    p = parse_block_join("NONE", 2, 2)
    assert p.pairs == [] and p.said_none and p.parseable
    assert parse_block_join("No matches.", 2, 2).said_none


def test_parse_drops_invalid_ids_and_duplicates():
    p = parse_block_join("L1, R1\nL1, R1\nL9, R1\nL0, R2\nL2, R5", 2, 2)
    assert p.pairs == [(1, 1)]
    assert p.invalid == [(9, 1), (0, 2), (2, 5)]
    assert p.duplicates == 1


def test_parse_garbage_is_unparseable():
    p = parse_block_join("The first left item matches the second right item.", 2, 2)
    assert p.pairs == [] and not p.parseable
    assert not parse_block_join("", 2, 2).parseable
    assert not parse_block_join(None, 2, 2).parseable


def test_parse_never_pairs_ids_across_lines():
    assert parse_block_join("L1, R2\nL3, R1", 3, 2).pairs == [(1, 2), (3, 1)]
    assert parse_block_join("L1\nR2", 2, 2).pairs == []


def test_parse_ignores_words_containing_l_or_r():
    assert parse_block_join("RL1, R2 URL3", 2, 2).pairs == []


def test_parse_pairs_win_over_none_word():
    p = parse_block_join("L1, R1\nNONE", 1, 1)
    assert p.pairs == [(1, 1)] and not p.said_none


# ---------------- prompts ----------------

def test_block_join_prompt_lists_both_blocks_and_condition_once():
    body = build_block_join_messages("same topic", ["a", "b"], ["x", "y", "z"])[-1]["content"]
    assert body.count("same topic") == 1
    assert "L1. a\nL2. b" in body and "R1. x\nR2. y\nR3. z" in body


def test_block_join_prompt_flattens_multiline_rows():
    body = build_block_join_messages("c", ["a\nR1. fake"], ["x"])[-1]["content"]
    assert "L1. a R1. fake\n" in body and body.count("\nR1. ") == 1


def test_pair_prompt():
    body = build_pair_messages("c", "a", "x")[-1]["content"]
    assert "Left item: a\nRight item: x" in body


# ---------------- operator ----------------

def test_one_call_for_a_block_pair():
    ex = JoinExecutor(same_letter)
    op = BlockSemJoin("c")
    res = run(op.evaluate_block_pair(ex, None, ["a1", "b1"], ["a2", "c2", "b2"]))
    assert sorted(res.matches) == [(0, 0), (1, 2)]
    assert op.stats.llm_calls == 1


def test_none_answer_means_no_matches_without_fallback():
    ex = JoinExecutor(same_letter, block_replies=["NONE"])
    op = BlockSemJoin("c")
    assert run(op.evaluate_block_pair(ex, None, ["a", "b"], ["c", "d"])).matches == []
    assert op.stats.llm_calls == 1


def test_invalid_ids_are_dropped_valid_pairs_kept():
    ex = JoinExecutor(same_letter, block_replies=["L1, R1\nL7, R1\nL2, R9"])
    op = BlockSemJoin("c")
    res = run(op.evaluate_block_pair(ex, None, ["a", "b"], ["a", "c"]))
    assert res.matches == [(0, 0)] and res.invalid == [(7, 1), (2, 9)]
    assert op.stats.invalid_pairs == 2 and op.stats.fallback_calls == 0


def test_unparseable_block_falls_back_to_every_pair():
    ex = JoinExecutor(same_letter, block_replies=["I think the first ones match."])
    op = BlockSemJoin("c")
    res = run(op.evaluate_block_pair(ex, None, ["a", "b"], ["a", "c", "b"]))
    assert sorted(res.matches) == [(0, 0), (1, 2)]
    assert (op.stats.block_calls, op.stats.fallback_calls) == (1, 6)
    assert op.stats.unparseable_blocks == 1


def test_one_by_one_block_uses_pair_prompt_only():
    ex = JoinExecutor(same_letter)
    op = BlockSemJoin("c")
    assert run(op.evaluate_block_pair(ex, None, ["a"], ["a"])).matches == [(0, 0)]
    assert (op.stats.block_calls, op.stats.fallback_calls) == (0, 1)


def test_empty_sides():
    op = BlockSemJoin("c")
    ex = JoinExecutor(same_letter)
    assert run(op.evaluate_block_pair(ex, None, [], ["a"])).matches == []
    assert run(op.join([], [row("a", 0, ex)])) == []
    assert op.stats.llm_calls == 0


def test_truncated_block_is_counted():
    ex = JoinExecutor(same_letter, block_replies=[FakeOut("L1, R1\nL2,", finish_reason="length")])
    op = BlockSemJoin("c")
    run(op.evaluate_block_pair(ex, None, ["a", "b"], ["a", "b"]))
    assert op.stats.truncated_block_calls == 1


def test_output_budget_scales_with_candidates():
    ex = JoinExecutor(same_letter)
    op = BlockSemJoin("c", max_tokens_per_pair=5, max_tokens_overhead=10)
    run(op.evaluate_block_pair(ex, None, ["a", "b"], ["a", "b", "c"]))
    assert ex.calls == [10 + 5 * 6]


def test_join_maps_block_positions_to_table_positions():
    lefts = ["a1", "b1", "c1", "d1", "e1"]
    rights = ["e2", "d2", "c2", "b2", "a2", "x2", "a3"]
    ex = JoinExecutor(same_letter)
    left_rows = [row(t, i, ex) for i, t in enumerate(lefts)]
    right_rows = [row(t, 100 + i) for i, t in enumerate(rights)]  # no executor
    op = BlockSemJoin("c", left_block_size=2, right_block_size=3)
    got = run(op.join(left_rows, right_rows))
    want = sorted((i, j) for i, l in enumerate(lefts) for j, r in enumerate(rights) if l[0] == r[0])
    assert got == want
    assert op.stats.block_pairs == 3 * 3 and op.stats.candidate_pairs == 35


def test_five_by_five_matches_pairwise_baseline():
    """Milestone 1 target shape: one 5x5 block call agrees with 25 pair calls."""
    lefts = ["apple pie", "bread", "carrot", "date", "egg"]
    rights = ["egg salad", "apple juice", "fig", "carrot cake", "bread roll"]
    ex = JoinExecutor(lambda l, r: l.split()[0] == r.split()[0])
    block_op = BlockSemJoin("c")
    pair_op = BlockSemJoin("c", left_block_size=1, right_block_size=1)
    got = run(block_op.join([row(t, i, ex) for i, t in enumerate(lefts)], rights))
    baseline = run(pair_op.join([row(t, i, ex) for i, t in enumerate(lefts)], rights))
    assert got == baseline == [(0, 1), (1, 4), (2, 3), (4, 0)]
    assert block_op.stats.llm_calls == 1 and pair_op.stats.llm_calls == 25


def test_invalid_block_sizes():
    with pytest.raises(ValueError):
        BlockSemJoin("c", left_block_size=0)


def test_estimate_tokens_grows_with_block():
    op = BlockSemJoin("c")
    count = lambda s: len(s) // 4
    small = op.estimate_tokens([row("a" * 40)], [row("b" * 40)], count)
    big = op.estimate_tokens([row("a" * 40)] * 3, [row("b" * 40)] * 3, count)
    assert big > small
