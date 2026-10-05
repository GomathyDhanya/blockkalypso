"""Unit tests for the block semantic join (no model or server needed)."""

import pytest

from vllm.kalypso.sem_ops.block_join import (
    build_block_join_messages,
    build_pair_messages,
    parse_block_join,
)



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
