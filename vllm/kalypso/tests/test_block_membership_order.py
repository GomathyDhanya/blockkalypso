"""Does task completion order change downstream block MEMBERSHIP or only ORDER?

BASELINE_AUDIT.md §4.6: the scheduler iterates ``asyncio.wait(...)``'s
``done`` *set* (``execution/pipeline_execution.py``, the ``for future in done``
loop), and a downstream block stage packs rows in the order they arrive
(``enqueue_block_rows``). Set iteration order is not stable, so the order in
which finished upstream tasks are handled is not stable either.

These tests pin that order explicitly (creation order vs. an interleaved
order: even creation index first, then odd) instead of relying on the set's
hash order, so they are deterministic. (Plain ascending vs. descending is not
a good probe: with 8 survivors in blocks of 4 it happens to give the same two
groups mirrored.)
Findings they lock in:

1. Membership, not just order: the same query packs *different sets of rows*
   into the downstream blocks depending only on completion handling order.
2. With an LLM whose per-row verdict does not depend on its block-mates, the
   final result set is still identical.
3. With an LLM whose verdict does depend on block-mates (as a real LLM's can,
   even at temperature 0), the final result set changes.
4. A block -> block pipeline with no per-tuple producer keeps fixed blocks,
   because upstream block boundaries are preserved.
"""

import asyncio
import re

import pytest

from vllm.kalypso.execution import pipeline_execution

from .test_telemetry import PIPELINES, RuleExecutor, _run

_ROW = re.compile(r"row(\d+)")


def _creation_index(future) -> int:
    # Scheduler futures come from asyncio.create_task with default names
    # "Task-<n>", n increasing in creation order.
    return int(future.get_name().rsplit("-", 1)[1])


def _order_key(interleaved: bool):
    if not interleaved:
        return _creation_index
    return lambda f: (_creation_index(f) % 2, _creation_index(f))


def _force_completion_order(monkeypatch, interleaved: bool):
    real_wait = asyncio.wait

    async def ordered_wait(fs, *args, **kwargs):
        done, pending = await real_wait(fs, *args, **kwargs)
        return sorted(done, key=_order_key(interleaved)), pending

    monkeypatch.setattr(pipeline_execution.asyncio, "wait", ordered_wait)


def _downstream_blocks(prompts) -> frozenset:
    """Row ids grouped per downstream block call (stage 2 question)."""
    blocks = []
    for body in prompts:
        if "Items:" not in body:
            continue
        question = body.split("Items:")[0]
        if "again" in question or "Q_MATCH" in question:
            blocks.append(frozenset(int(i) for i in _ROW.findall(body.split("Items:")[1])))
    return frozenset(blocks)


def _run_ordered(name, monkeypatch, interleaved, executor_cls=RuleExecutor):
    _force_completion_order(monkeypatch, interleaved)
    monkeypatch.setattr(
        "vllm.kalypso.tests.test_telemetry.RuleExecutor", executor_cls)
    try:
        result, prompts = _run(PIPELINES[name], monkeypatch)
    finally:
        monkeypatch.undo()
    return result, _downstream_blocks(prompts)


@pytest.mark.parametrize("name", ["tuple_block", "filter_blockjoin"])
def test_completion_order_changes_block_membership_not_results(name, fake_kv_manager, monkeypatch):
    fwd_result, fwd_blocks = _run_ordered(name, monkeypatch, interleaved=False)
    rev_result, rev_blocks = _run_ordered(name, monkeypatch, interleaved=True)

    # Same rows reach the downstream stage either way ...
    assert set().union(*fwd_blocks) == set().union(*rev_blocks)
    # ... but they are grouped into different blocks: membership changed,
    # not merely the order blocks were issued in (frozensets ignore order).
    assert fwd_blocks != rev_blocks
    # A block-mate-independent LLM gives the same final answer set.
    assert fwd_result == rev_result


def test_block_to_block_membership_is_stable(fake_kv_manager, monkeypatch):
    _, fwd_blocks = _run_ordered("block_block", monkeypatch, interleaved=False)
    _, rev_blocks = _run_ordered("block_block", monkeypatch, interleaved=True)
    assert fwd_blocks == rev_blocks


class BlockMateSensitiveExecutor(RuleExecutor):
    """Stage-2 block verdicts depend on block-mates: an item passes only if it
    passes the rule AND its block contains a row with id < 4. A stand-in for
    a real LLM whose per-row answer shifts with the other rows in the prompt."""

    async def execute(self, raw_request, prompt, max_tokens, pin=False, priority=0):
        body = prompt[-1]["content"]
        if "Items:" in body and "again" in body.split("Items:")[0]:
            self.prompts.append(body)
            items = re.findall(r"^(\d+)\. (.*)$", body.split("Items:")[1], re.M)
            ids = [int(r) for _, text in items for r in _ROW.findall(text)]
            ok = bool(ids) and min(ids) < 4
            from .test_block_filter import FakeOut
            return FakeOut("\n".join(
                f"{i}: {'YES' if ok and 'kk' in text else 'NO'}" for i, text in items))
        return await super().execute(raw_request, prompt, max_tokens, pin, priority)


def test_block_mate_sensitive_llm_changes_results(fake_kv_manager, monkeypatch):
    fwd_result, fwd_blocks = _run_ordered(
        "tuple_block", monkeypatch, interleaved=False, executor_cls=BlockMateSensitiveExecutor)
    rev_result, rev_blocks = _run_ordered(
        "tuple_block", monkeypatch, interleaved=True, executor_cls=BlockMateSensitiveExecutor)
    assert fwd_blocks != rev_blocks
    assert fwd_result != rev_result
