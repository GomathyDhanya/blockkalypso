"""Unit tests for the shared block-operator helpers.

Run from any directory outside the repo's ``tests/`` folder, e.g.:
    python -m pytest vllm/kalypso/tests -q -p no:cacheprovider
"""

import asyncio
from types import SimpleNamespace

import pytest

from vllm.kalypso.sem_ops.block_utils import (
    EMPTY_ITEM,
    call_llm,
    first_executor,
    item_line,
    make_blocks,
    make_join_blocks,
    parse_single_answer,
    row_text,
    strip_context_wrapper,
    with_system,
)


def kalypso_row(text, right=None, idx=0, executor=None):
    wrap = lambda t: {"role": "user", "type": "text",
                      "content": f'CONTEXT:\n  {{\n    "text": {t}\n  }}\n'}
    return SimpleNamespace(
        input=SimpleNamespace(
            data=[{"role": "system", "content": "sys"}, wrap(text)],
            right_data=[wrap(right)] if right else [],
        ),
        state=SimpleNamespace(executor=executor, raw_request=None, idx=idx),
        output=[],
    )


def test_strip_context_wrapper():
    assert strip_context_wrapper('CONTEXT:\n  {\n    "text": hi there\n  }\n') == "hi there"
    assert strip_context_wrapper("  plain  ") == "plain"


def test_row_text_skips_system_and_appends_right_side():
    assert row_text(kalypso_row("left")) == "left"
    assert row_text(kalypso_row("left", right="right")) == "left\nright"
    assert row_text("bare string") == "bare string"


def test_item_line_flattens_newlines_and_marks_empty():
    assert item_line("a\n2. fake item\n  b") == "a 2. fake item b"
    assert item_line("   ") == EMPTY_ITEM


def test_make_blocks():
    assert make_blocks(list(range(7)), 3) == [[0, 1, 2], [3, 4, 5], [6]]
    assert make_blocks([], 3) == []
    with pytest.raises(ValueError):
        make_blocks([1], 0)


def test_make_join_blocks_offsets_cover_every_pair():
    blocks = make_join_blocks(list("abc"), list("xyzw"), 2, 3)
    assert [(lo, ro) for lo, _, ro, _ in blocks] == [(0, 0), (0, 3), (2, 0), (2, 3)]
    pairs = {(lo + i, ro + j) for lo, lb, ro, rb in blocks
             for i in range(len(lb)) for j in range(len(rb))}
    assert pairs == {(i, j) for i in range(3) for j in range(4)}


@pytest.mark.parametrize("text,want", [
    ("YES", True), ("no.", False), ("TRUE\n", True), (" False", False),
    ("The answer is yes", True), ("maybe", None), ("", None), (None, None),
])
def test_parse_single_answer(text, want):
    assert parse_single_answer(text) is want


def test_with_system():
    assert [m["role"] for m in with_system(None, "u")] == ["user"]
    assert [m["role"] for m in with_system("s", "u")] == ["system", "user"]


def test_call_llm_and_first_executor():
    class Ex:
        async def execute(self, raw_request, prompt, max_tokens, pin, priority):
            return SimpleNamespace(text=None, finish_reason="stop")

    ex = Ex()
    assert asyncio.run(call_llm(ex, None, [], 4)) == ("", "stop")
    assert first_executor([kalypso_row("r"), kalypso_row("l", executor=ex)])[0] is ex
    with pytest.raises(ValueError):
        first_executor([kalypso_row("r")])
