"""Repeatable standalone filter and join block-size sweeps.

Uses the same prompts for the singleton and block arms. Reports each trial,
including calls, tokens, latency, labels, and sampled KV-cache occupancy.
Optional server-process RSS is a separate process-memory proxy.
"""

import argparse
import asyncio
import json
import re
import statistics
import subprocess
from pathlib import Path

import httpx

from block_filter_standalone import (DEFAULT_DATA as FILTER_DATA,
                                     DEFAULT_INSTRUCTION as FILTER_PROMPT,
                                     load_rows)
from block_join_standalone import (DATA as JOIN_DATA,
                                   DEFAULT_CONDITION as JOIN_PROMPT,
                                   load_matches, load_table, evaluate_candidates)
from block_ops_common import HTTPExecutor, Timer, resolve_model
from vllm.kalypso.sem_ops.block_filter import BlockSemFilter
from vllm.kalypso.sem_ops.block_join import BlockSemJoin
from vllm.kalypso.sem_ops.block_utils import make_blocks


async def peak_rss(pid, stop, samples):
    while not stop.is_set():
        try:
            out = await asyncio.to_thread(
                subprocess.check_output, ["ps", "-o", "rss=", "-p", str(pid)],
                text=True,
            )
            samples.append(int(out.strip()))
        except (OSError, ValueError, subprocess.CalledProcessError):
            pass
        try:
            await asyncio.wait_for(stop.wait(), timeout=0.1)
        except asyncio.TimeoutError:
            pass


async def peak_kv_usage(client, base_url, stop, samples):
    """Sample vLLM's KV occupancy gauge while a trial is active."""
    metrics_url = base_url.removesuffix("/v1") + "/metrics"
    while not stop.is_set():
        try:
            response = await client.get(metrics_url, timeout=2.0)
            response.raise_for_status()
            for line in response.text.splitlines():
                if line.startswith("vllm:kv_cache_usage_perc{"):
                    value = re.search(r"\}\s+([0-9.eE+-]+)$", line)
                    if value:
                        samples.append(float(value.group(1)) * 100)
        except (httpx.HTTPError, ValueError):
            pass
        try:
            await asyncio.wait_for(stop.wait(), timeout=0.25)
        except asyncio.TimeoutError:
            pass


async def timed_run(work, server_pid, client, base_url):
    stop = asyncio.Event()
    rss_samples, kv_samples = [], []
    rss_monitor = asyncio.create_task(peak_rss(server_pid, stop, rss_samples)) if server_pid else None
    kv_monitor = asyncio.create_task(peak_kv_usage(client, base_url, stop, kv_samples))
    try:
        with Timer() as timer:
            result = await work()
    finally:
        stop.set()
        if rss_monitor:
            await rss_monitor
        await kv_monitor
    return result, timer.seconds, max(rss_samples, default=None), max(kv_samples, default=None)


async def filter_trial(client, args, model, size, rows):
    executor = HTTPExecutor(client, args.base_url, model)
    op = BlockSemFilter(args.filter_prompt, block_size=size)
    texts = [r["text"] for r in rows]

    async def work():
        results = await asyncio.gather(*(
            op.evaluate_block(executor, None, group)
            for group in make_blocks(texts, size)
        ))
        return [v for result in results for v in result.verdicts]

    predictions, seconds, rss, kv_usage = await timed_run(
        work, args.server_pid, client, args.base_url)
    checked = [(p, r["label"]) for p, r in zip(predictions, rows)]
    hand = [(p, r["label"]) for p, r in zip(predictions, rows) if r["hand"]]
    return {
        "operator": "filter", "block_size": size,
        **executor.summary(seconds), "peak_server_rss_kb": rss,
        "peak_kv_cache_usage_pct": kv_usage,
        "correct": sum(p == label for p, label in checked), "total": len(checked),
        "hand_correct": sum(p == label for p, label in hand), "hand_total": len(hand),
        "undecided": sum(p is None for p in predictions),
        "operator_stats": op.stats.as_dict(),
    }


async def join_trial(client, args, model, k, j, left, right, expected):
    executor = HTTPExecutor(client, args.base_url, model)
    op = BlockSemJoin(args.join_prompt, left_block_size=k, right_block_size=j)

    async def work():
        matches, _ = await evaluate_candidates(op, executor, left, right)
        return set(matches)

    matches, seconds, rss, kv_usage = await timed_run(
        work, args.server_pid, client, args.base_url)
    truth = set(expected)
    return {
        "operator": "join", "left_block_size": k, "right_block_size": j,
        **executor.summary(seconds), "peak_server_rss_kb": rss,
        "peak_kv_cache_usage_pct": kv_usage,
        "true_positives": len(matches & truth),
        "false_positives": len(matches - truth),
        "false_negatives": len(truth - matches),
        "operator_stats": op.stats.as_dict(),
    }


def parse_sizes(value):
    sizes = [int(part) for part in value.split(",")]
    if not sizes or any(size < 1 for size in sizes):
        raise argparse.ArgumentTypeError("sizes must be positive integers")
    return sizes


def parse_pairs(value):
    try:
        pairs = [tuple(map(int, part.lower().split("x"))) for part in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("use kxj,kxj,...") from exc
    if not pairs or any(len(pair) != 2 for pair in pairs):
        raise argparse.ArgumentTypeError("use kxj,kxj,...")
    return pairs


async def run(args):
    rows = load_rows(args.filter_data)
    left, right = load_table(args.join_left), load_table(args.join_right)
    expected = load_matches(args.join_expected)
    if expected is None:
        raise ValueError("the join ablation requires an expected-match CSV")
    trials = []
    async with httpx.AsyncClient(timeout=args.timeout) as client:
        model = await resolve_model(client, args.base_url, args.model)
        for repeat in range(args.repeats):
            for size in args.filter_sizes:
                trial = await filter_trial(client, args, model, size, rows)
                trial["repeat"] = repeat + 1
                trials.append(trial)
                print(json.dumps(trial), flush=True)
                if args.out:
                    Path(args.out).write_text(json.dumps(
                        {"model": model, "trials": trials, "complete": False}, indent=2))
            for k, j in args.join_sizes:
                trial = await join_trial(client, args, model, k, j, left, right, expected)
                trial["repeat"] = repeat + 1
                trials.append(trial)
                print(json.dumps(trial), flush=True)
                if args.out:
                    Path(args.out).write_text(json.dumps(
                        {"model": model, "trials": trials, "complete": False}, indent=2))
    groups = {}
    for trial in trials:
        key = (trial["operator"], trial.get("block_size"),
               trial.get("left_block_size"), trial.get("right_block_size"))
        groups.setdefault(str(key), []).append(trial)
    medians = {key: {
        "latency_s": statistics.median(t["latency_s"] for t in group),
        "llm_calls": statistics.median(t["llm_calls"] for t in group),
        "prompt_tokens": statistics.median(t["prompt_tokens"] for t in group),
    } for key, group in groups.items()}
    report = {"model": model, "trials": trials, "medians": medians,
              "complete": True,
              "memory_note": "peak_kv_cache_usage_pct is a sampled vLLM gauge; peak_server_rss_kb is optional process RSS"}
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-url", default="http://localhost:8003/v1")
    parser.add_argument("--model")
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--filter-sizes", type=parse_sizes, default=[1, 2, 5, 10, 20])
    parser.add_argument("--join-sizes", type=parse_pairs, default=[(1, 1), (1, 5), (5, 1), (5, 5)])
    parser.add_argument("--filter-data", default=str(FILTER_DATA))
    parser.add_argument("--filter-prompt", default=FILTER_PROMPT)
    parser.add_argument("--join-left", default=str(JOIN_DATA / "left_5.csv"))
    parser.add_argument("--join-right", default=str(JOIN_DATA / "right_5.csv"))
    parser.add_argument("--join-expected", default=str(JOIN_DATA / "matches_5x5.csv"))
    parser.add_argument("--join-prompt", default=JOIN_PROMPT)
    parser.add_argument("--server-pid", type=int, help="optional local server PID for peak RSS sampling")
    parser.add_argument("--out", help="write all trials and medians to JSON")
    args = parser.parse_args()
    if args.repeats < 1 or any(k < 1 or j < 1 for k, j in args.join_sizes):
        parser.error("repeat count and join sizes must be positive")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
