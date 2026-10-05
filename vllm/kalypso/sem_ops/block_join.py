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

import re
from dataclasses import dataclass
from typing import Sequence

from .block_utils import item_line, with_system

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
