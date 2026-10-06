"""Milestone 1 standalone check for the block filter.

Runs one predicate over a labeled CSV (default: 20 synthetic reviews) twice
against an OpenAI-compatible server, e.g. Kalypso's own vLLM server:

* per-tuple baseline: one call per row, like Kalypso's SemFilter;
* block filter: the rows in blocks of ``--block-size``.

Prints LLM calls, tokens, latency and agreement with the labels (all rows and
the hand-checked subset), and writes the same JSON with ``--out``.

    python vllm/kalypso/benchmark/block_filter_standalone.py \
        --base-url http://localhost:8003/v1 --block-size 20 --show-raw
"""

import argparse
import asyncio
import csv
import json
from pathlib import Path

import httpx

from block_ops_common import SAMPLE_DATA, HTTPExecutor, Timer, resolve_model

from vllm.kalypso.sem_ops.block_filter import BlockSemFilter, build_single_messages
from vllm.kalypso.sem_ops.block_utils import make_blocks, parse_single_answer

DEFAULT_DATA = SAMPLE_DATA / "block_filter" / "reviews_20.csv"
DEFAULT_INSTRUCTION = "Does the review complain about the restaurant?"


def load_rows(path):
    """CSV with columns data, label (true/false) and optional hand_checked."""
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [
        {
            "text": r["data"],
            "label": r["label"].strip().lower() == "true",
            "hand": r.get("hand_checked", "false").strip().lower() == "true",
        }
        for r in rows
    ]


def agreement(preds, rows, only_hand=False):
    pairs = [(p, r["label"]) for p, r in zip(preds, rows) if r["hand"] or not only_hand]
    return f"{sum(p == want for p, want in pairs)}/{len(pairs)}"


async def run(args):
    rows = load_rows(args.data)
    texts = [r["text"] for r in rows]
    async with httpx.AsyncClient(timeout=args.timeout) as client:
        model = await resolve_model(client, args.base_url, args.model)

        base = HTTPExecutor(client, args.base_url, model)
        with Timer() as t_base:
            outs = await asyncio.gather(*(
                base.execute(None, build_single_messages(args.instruction, t), 4)
                for t in texts
            ))
        base_preds = [parse_single_answer(o.text) for o in outs]

        blk = HTTPExecutor(client, args.base_url, model)
        op = BlockSemFilter(args.instruction, block_size=args.block_size)
        with Timer() as t_blk:
            results = await asyncio.gather(*(
                op.evaluate_block(blk, None, b) for b in make_blocks(texts, args.block_size)
            ))
        blk_preds = [v for r in results for v in r.verdicts]

    report = {
        "model": model,
        "rows": len(rows),
        "block_size": args.block_size,
        "per_tuple": {
            **base.summary(t_base.seconds),
            "agree_all": agreement(base_preds, rows),
            "agree_hand_checked": agreement(base_preds, rows, only_hand=True),
        },
        "block": {
            **blk.summary(t_blk.seconds),
            "agree_all": agreement(blk_preds, rows),
            "agree_hand_checked": agreement(blk_preds, rows, only_hand=True),
            "agree_with_per_tuple": f"{sum(a == b for a, b in zip(blk_preds, base_preds))}/{len(rows)}",
            "operator_stats": op.stats.as_dict(),
        },
    }
    print(json.dumps(report, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))
    if args.show_raw:
        for i, r in enumerate(results, 1):
            print(f"---- raw output, block {i} ----")
            print("\n---\n".join(r.raw_outputs))
        print("---- per row: hand / label / per-tuple / block ----")
        for r, bp, kp in zip(rows, base_preds, blk_preds):
            print(f"{'H' if r['hand'] else ' '} {r['label']!s:5} {bp!s:5} {kp!s:5}  {r['text']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base-url", default="http://localhost:8003/v1")
    ap.add_argument("--model", default=None, help="default: first model the server lists")
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    ap.add_argument("--block-size", type=int, default=20)
    ap.add_argument("--timeout", type=float, default=600)
    ap.add_argument("--out", default=None, help="also write the JSON report here")
    ap.add_argument("--show-raw", action="store_true")
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
