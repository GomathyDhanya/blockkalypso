"""Unit tests for the block semantic filter (no model or server needed)."""

import asyncio
from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest

from vllm.kalypso.sem_ops.block_filter import (
    BLOCK_SYSTEM_PROMPT,
    BlockSemFilter,
    build_block_messages,
    build_single_messages,
    parse_block_answers,
)


@dataclass
class FakeOut:
    text: str
    finish_reason: str = "stop"
    request_id: str = "fake"


@dataclass
class FakeExecutor:
    """Scripted block replies; one-row prompts are answered from ``truth``."""

    block_replies: list
    truth: dict = field(default_factory=dict)
    calls: list = field(default_factory=list)

    async def execute(self, raw_request, prompt, max_tokens, pin=False, priority=0):
        self.calls.append((prompt, max_tokens))
        body = prompt[-1]["content"]
        if body.startswith("Item: "):
            item = body[len("Item: "):].split("\n\n")[0]
            if item not in self.truth:
                return FakeOut("I am not sure.")
            return FakeOut("YES" if self.truth[item] else "NO")
        reply = self.block_replies.pop(0)
        return reply if isinstance(reply, FakeOut) else FakeOut(reply)


def row(text, idx=0, executor=None):
    content = f'CONTEXT:\n  {{\n    "text": {text}\n  }}\n'
    return SimpleNamespace(
        input=SimpleNamespace(data=[{"role": "user", "type": "text", "content": content}],
                              right_data=[]),
        state=SimpleNamespace(executor=executor, raw_request=None, idx=idx),
        output=[],
    )


def run(coro):
    return asyncio.run(coro)


# ---------------- parsing ----------------

def test_parse_clean_lines():
    assert parse_block_answers("1: YES\n2: NO\n3: yes", 3) == {1: True, 2: False, 3: True}


@pytest.mark.parametrize("text", [
    "1: TRUE\n2: FALSE",
    "[1] YES\n[2] NO",
    "1 - YES\n2 - NO",
    "1. Yes\n2. No",
    "1) yes\n2) no",
    "**1**: YES\n**2**: NO",
    "**1: YES**\n**2: NO**",
    "- 1: YES\n- 2: NO",
    "```\n1: YES\n2: NO\n```",
    "Here are the answers:\n1: YES\n2: NO\nHope this helps.",
])
def test_parse_tolerates_common_formats(text):
    assert parse_block_answers(text, 2) == {1: True, 2: False}


def test_parse_counts_only_line_starts():
    assert parse_block_answers("1: NO 2: YES 3: YES", 3) == {1: False}


def test_parse_ignores_out_of_range_ids():
    assert parse_block_answers("1: YES\n4: NO\n99: YES", 3) == {1: True}


def test_parse_conflicting_duplicate_is_dropped_agreeing_is_kept():
    assert parse_block_answers("1: YES\n1: NO\n2: NO\n2: NO", 2) == {2: False}


def test_parse_zero_based_numbering_is_shifted():
    assert parse_block_answers("0: YES\n1: NO\n2: YES", 3) == {1: True, 2: False, 3: True}


def test_parse_partial_zero_based_is_not_shifted():
    assert parse_block_answers("0: YES\n1: NO", 3) == {1: False}


@pytest.mark.parametrize("text", ["", None, "I cannot answer that.", "YES\nNO"])
def test_parse_garbage_gives_nothing(text):
    assert parse_block_answers(text, 2) == {}


# ---------------- prompts ----------------

def test_block_prompt_has_instruction_once_and_every_item():
    msgs = build_block_messages("is negative", ["a", "b", "c"])
    body = msgs[-1]["content"]
    assert body.count("is negative") == 1
    assert all(f"{i}. {t}" in body for i, t in enumerate("abc", 1))
    assert "exactly 3 lines" in body


def test_block_prompt_flattens_multiline_rows():
    body = build_block_messages("q", ["first\n2. injected", "second"])[-1]["content"]
    assert "1. first 2. injected\n2. second" in body


def test_system_prompt_is_optional():
    assert [m["role"] for m in build_block_messages("q", ["a"])] == ["user"]
    roles = [m["role"] for m in build_block_messages("q", ["a"], BLOCK_SYSTEM_PROMPT)]
    assert roles == ["system", "user"]
    assert build_single_messages("q", "a", "s")[0]["role"] == "system"


# ---------------- operator ----------------

def test_one_call_when_block_answers_are_complete():
    ex = FakeExecutor(["1: YES\n2: NO\n3: YES\n4: NO"])
    op = BlockSemFilter("q", block_size=4)
    assert run(op.evaluate_block(ex, None, list("abcd"))).verdicts == [True, False, True, False]
    assert op.stats.llm_calls == 1 and op.stats.rows_from_block == 4


def test_missing_and_malformed_rows_fall_back():
    ex = FakeExecutor(["1: YES\n3: maybe\n4: NO"], truth={"b": True, "c": False})
    op = BlockSemFilter("q", block_size=4)
    assert run(op.evaluate_block(ex, None, list("abcd"))).verdicts == [True, True, False, False]
    assert (op.stats.block_calls, op.stats.fallback_calls) == (1, 2)


def test_unparseable_block_falls_back_for_every_row():
    ex = FakeExecutor(["Sorry, I cannot help."], truth={"a": True, "b": False})
    op = BlockSemFilter("q")
    assert run(op.evaluate_block(ex, None, ["a", "b"])).verdicts == [True, False]
    assert op.stats.llm_calls == 3


def test_truncated_block_is_counted_and_repaired():
    ex = FakeExecutor([FakeOut("1: YES\n2: N", finish_reason="length")], truth={"b": False})
    op = BlockSemFilter("q")
    assert run(op.evaluate_block(ex, None, ["a", "b"])).verdicts == [True, False]
    assert op.stats.truncated_block_calls == 1


def test_row_undecided_after_fallback_is_none_and_dropped():
    ex = FakeExecutor(["1: YES"])  # "b" is not in truth -> unusable single answer
    rows = [row("a", 0, ex), row("b", 1, ex)]
    op = BlockSemFilter("q")
    assert [r.state.idx for r in run(op(rows))] == [0]
    assert op.stats.rows_undecided == 1
    assert rows[1].output[-1][str(BlockSemFilter)] == "undecided"


def test_block_of_one_uses_single_prompt_only():
    ex = FakeExecutor([], truth={"a": True})
    op = BlockSemFilter("q")
    assert run(op.evaluate_block(ex, None, ["a"])).verdicts == [True]
    assert (op.stats.block_calls, op.stats.fallback_calls) == (0, 1)


def test_empty_block_and_empty_table():
    op = BlockSemFilter("q")
    assert run(op.evaluate_block(FakeExecutor([]), None, [])).verdicts == []
    assert run(op([])) == []
    assert op.stats.llm_calls == 0


def test_negate_flips_decided_rows_only():
    ex = FakeExecutor(["1: YES\n2: NO"], truth={})
    op = BlockSemFilter("q", negate=True)
    assert run(op.evaluate_block(ex, None, ["a", "b"])).verdicts == [False, True]


def test_output_budget_scales_with_block_size():
    ex = FakeExecutor(["1: YES\n2: YES\n3: YES"])
    op = BlockSemFilter("q", max_tokens_per_row=5, max_tokens_overhead=10)
    run(op.evaluate_block(ex, None, list("abc")))
    assert ex.calls[0][1] == 25


def test_table_is_blocked_and_passing_rows_keep_order():
    ex = FakeExecutor(["1: YES\n2: NO\n3: YES", "1: NO\n2: YES"])
    rows = [row(t, i, ex) for i, t in enumerate("abcde")]
    op = BlockSemFilter("q", block_size=3)
    assert [r.state.idx for r in run(op(rows))] == [0, 2, 4]
    assert op.stats.block_calls == 2 and op.stats.fallback_calls == 0
    assert [r.output[-1][str(BlockSemFilter)] for r in rows] == ["true", "false", "true", "false", "true"]


def test_single_row_call_returns_bool_like_semfilter():
    ex = FakeExecutor([], truth={"a": False})
    assert run(BlockSemFilter("q")(row("a", 0, ex))) is False


def test_invalid_block_size():
    with pytest.raises(ValueError):
        BlockSemFilter("q", block_size=0)


def test_estimate_tokens_grows_with_block_and_uses_single_cap_for_one():
    op = BlockSemFilter("q", single_max_tokens=4)
    count = lambda s: len(s) // 4
    one = op.estimate_tokens([row("x" * 40)], count)
    four = op.estimate_tokens([row("x" * 40)] * 4, count)
    assert four > one
    assert op.estimate_tokens(row("x" * 40), count) == one
