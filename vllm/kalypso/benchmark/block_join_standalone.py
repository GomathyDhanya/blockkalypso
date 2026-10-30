"""Milestone 1 standalone check for the block join.

Joins a left table with a right table (default: 5 complaints x 5 complaint
categories) twice against an OpenAI-compatible server, e.g. Kalypso's own
vLLM server:

* pairwise baseline: one call per (left, right) pair, like Kalypso's join;
* block join: Cartesian candidates in blocks of ``--pair-block-size`` pairs.

Prints LLM calls, tokens, latency, the matches each produced, agreement
between the two, and agreement with the hand-written expected matches.

    python vllm/kalypso/benchmark/block_join_standalone.py \
        --base-url http://localhost:8003/v1 --pair-block-size 25 --show-raw
"""

import argparse
import asyncio
import csv
import json
from pathlib import Path

import httpx

from block_ops_common import SAMPLE_DATA, HTTPExecutor, Timer, resolve_model

from vllm.kalypso.sem_ops.block_join import BlockSemJoin, format_join_candidate
from vllm.kalypso.sem_ops.block_utils import make_blocks

DATA = SAMPLE_DATA / "block_join"
DEFAULT_CONDITION = "The complaint in the left item belongs to the category in the right item."


def load_table(path):
    with open(path, newline="", encoding="utf-8") as f:
        return [r["data"] for r in csv.DictReader(f)]


def load_matches(path):
    if not path or not Path(path).exists():
        return None
    with open(path, newline="", encoding="utf-8") as f:
        return sorted((int(r["left_id"]), int(r["right_id"])) for r in csv.DictReader(f))


def fmt(pairs):
    """Pairs as a compact string, e.g. "L0-R2 L1-R4" (0-based table rows)."""
    return " ".join(f"L{a}-R{b}" for a, b in sorted(pairs)) or "-"


def compare(got, want):
    got, want = set(got), set(want)
    return {
        "agree": f"{len(got & want)}/{len(want)}",
        "missed": fmt(want - got),
        "extra": fmt(got - want),
    }


async def evaluate_candidates(op, executor, left, right):
    """Benchmark the same left-major candidate blocks as query execution."""
    candidates = [(i, j, format_join_candidate(l, r))
                  for i, l in enumerate(left) for j, r in enumerate(right)]
    blocks = make_blocks(candidates, op.block_size)
    results = await asyncio.gather(*(
        op.evaluate_block(executor, None, [text for _, _, text in block])
        for block in blocks
    ))
    matches = sorted((i, j) for block, result in zip(blocks, results)
                     for (i, j, _), verdict in zip(block, result.verdicts)
                     if verdict)
    return matches, results


async def run(args):
    left = load_table(args.left)
    right = load_table(args.right)
    expected = load_matches(args.expected)
    async with httpx.AsyncClient(timeout=args.timeout) as client:
        model = await resolve_model(client, args.base_url, args.model)

        base_ex = HTTPExecutor(client, args.base_url, model)
        pairwise = BlockSemJoin(args.condition, pair_block_size=1)
        with Timer() as t_base:
            results = await asyncio.gather(*(
                pairwise.evaluate_block_pair(base_ex, None, [l], [r])
                for l in left for r in right
            ))
        base_matches = sorted(
            (i, j) for (i, j), res in zip(((i, j) for i in range(len(left)) for j in range(len(right))), results)
            if res.matches
        )

        blk_ex = HTTPExecutor(client, args.base_url, model)
        op = BlockSemJoin(args.condition, pair_block_size=args.pair_block_size)
        with Timer() as t_blk:
            blk_matches, blk_results = await evaluate_candidates(op, blk_ex, left, right)

    report = {
        "model": model,
        "left_rows": len(left),
        "right_rows": len(right),
        "pair_block_size": args.pair_block_size,
        "pairwise": {**base_ex.summary(t_base.seconds), "matches": fmt(base_matches)},
        "block": {**blk_ex.summary(t_blk.seconds), "matches": fmt(blk_matches),
                  "operator_stats": op.stats.as_dict()},
        "block_vs_pairwise": compare(blk_matches, base_matches),
        "full_agreement_with_pairwise": set(blk_matches) == set(base_matches),
    }
    if expected is not None:
        report["pairwise_vs_expected"] = compare(base_matches, expected)
        report["block_vs_expected"] = compare(blk_matches, expected)
    print(json.dumps(report, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))
    if args.show_raw:
        for i, res in enumerate(blk_results, 1):
            print(f"---- raw output, block pair {i} ----")
            print("\n---\n".join(res.raw_outputs))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base-url", default="http://localhost:8003/v1")
    ap.add_argument("--model", default=None, help="default: first model the server lists")
    ap.add_argument("--left", default=str(DATA / "left_5.csv"))
    ap.add_argument("--right", default=str(DATA / "right_5.csv"))
    ap.add_argument("--expected", default=str(DATA / "matches_5x5.csv"),
                    help="CSV of left_id,right_id expected matches (optional)")
    ap.add_argument("--condition", default=DEFAULT_CONDITION)
    ap.add_argument("--pair-block-size", type=int, default=25,
                    help="Cartesian candidate pairs per block-filter request")
    ap.add_argument("--timeout", type=float, default=600)
    ap.add_argument("--out", default=None, help="also write the JSON report here")
    ap.add_argument("--show-raw", action="store_true")
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
