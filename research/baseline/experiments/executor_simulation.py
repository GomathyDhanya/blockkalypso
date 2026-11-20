"""CPU executor simulation: the real planner, scheduler and KV admission with a
simulated LLM, for filter -> join at several selectivities and block sizes.

    python -m research.baseline.experiments.executor_simulation \
        --out research/baseline/results/executor_simulation

What is real: ``SemanticPlan`` (plan building, stage segmentation, KV stage
registration), ``AsyncPipelineExecutor`` (block buffering, partial-block
flush, rebalancing, admission), the block filter/join operators and their
prompt construction and parsing, and the telemetry hooks.

What is simulated: the LLM. A call sleeps ``base + per_item * items`` seconds
(defaults 20 ms + 2 ms per item) under a semaphore of ``--slots`` concurrent
calls, standing in for the server's batch capacity, and answers by a rule:
row ``i`` passes the filter iff it is in the chosen pass set (exactly
``round(sel * n)`` rows, fixed seed) and a pair passes the join iff the right
row's id is odd (fanout 0.5 of the 32 right rows). The KV manager is the
real class with a fake tokenizer (``len(text)//4`` tokens) and capacity
``--kv-tokens``.

So latencies here measure the scheduler's structure (call counts, block fill,
waiting, critical path) under a fixed per-call cost model; they are NOT GPU
latencies. GPU numbers come from research/baseline/gpu/.

Outputs: ``runs.jsonl`` (one record per run, including the per-stage trace
summary), ``summary.md``, and the raw traces under ``traces/``.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import random
import re
import statistics
import time
from pathlib import Path
from types import SimpleNamespace

import vllm.kalypso.query_processor  # noqa: F401  (import order, see tests/conftest.py)
from vllm.kalypso import telemetry
from vllm.kalypso.budget import KVMemoryManager
from vllm.kalypso.controller.plan import SemanticPlan
from vllm.kalypso.query import Query

from research.baseline.analysis.trace_summary import summarise

_ITEM = re.compile(r"^(\d+)\. ", re.M)
_ROW = re.compile(r"\bL(\d+)\b")
_RIGHT = re.compile(r"\bR(\d+)\b")


class FakeOut:
    def __init__(self, text):
        self.text, self.finish_reason, self.request_id = text, "stop", "sim"


class SimExecutor:
    """Rule-answering LLM with a fixed per-call latency and bounded slots."""

    def __init__(self, pass_rows: set, base_s: float, per_item_s: float, slots: int):
        self.pass_rows, self.base_s, self.per_item_s = pass_rows, base_s, per_item_s
        self.sem = asyncio.Semaphore(slots)
        self.calls = 0

    def _passes(self, question: str, item: str) -> bool:
        if "Q_MATCH" in question:
            m = _RIGHT.search(item)
            return bool(m) and int(m.group(1)) % 2 == 1
        m = _ROW.search(item)
        return bool(m) and int(m.group(1)) in self.pass_rows

    async def execute(self, raw_request, prompt, max_tokens, pin=False, priority=0):
        body = prompt[-1]["content"]
        whole = "\n".join(str(m.get("content", "")) for m in prompt)
        ids = [(m.group(1), m.start(), m.end()) for m in _ITEM.finditer(body)]
        block = bool(ids) and "Items:" in body
        async with self.sem:
            self.calls += 1
            await asyncio.sleep(self.base_s + self.per_item_s * (len(ids) if block else 1))
        if block:
            question = body.split("Items:")[0]
            lines = []
            for n, (i, _, start) in enumerate(ids):
                end = ids[n + 1][1] if n + 1 < len(ids) else len(body)
                lines.append(f"{i}: {'YES' if self._passes(question, body[start:end]) else 'NO'}")
            return FakeOut("\n".join(lines))
        verdict = self._passes(whole, whole)
        if body.startswith("Item: "):
            return FakeOut("YES" if verdict else "NO")
        return FakeOut("true" if verdict else "false")


def _row(text, idx, executor=None):
    content = f'CONTEXT:\n  {{\n    "text": {text}\n  }}\n'
    return SimpleNamespace(
        input=SimpleNamespace(data=[{"role": "user", "type": "text", "content": content}],
                              right_data=[]),
        state=SimpleNamespace(executor=executor, raw_request=None, idx=idx,
                              pin_req_id=None, stage_id=-1),
        output=[],
    )


class _FakeTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        return "\n".join(str(m.get("content", "")) for m in messages)

    def encode(self, text, add_special_tokens=False):
        return [0] * max(1, len(text) // 4)


def _install_kv(capacity_tokens: int):
    mgr = KVMemoryManager.__new__(KVMemoryManager)
    mgr.tokenizer = _FakeTokenizer()
    mgr.bytes_per_token = 1
    mgr._capacity = capacity_tokens
    mgr._global_used = 0
    mgr._stage_capacity, mgr._stage_min_capacity = {}, {}
    mgr._stage_max_capacity, mgr._stage_used, mgr._stage_inflight = {}, {}, {}
    mgr._cond = asyncio.Condition()
    KVMemoryManager._instance = mgr


def pipeline(k_filter: int, k_join: int):
    f = {"op": "sem_filter", "args": {"prompt": "Q_KEEP: is this review clearly positive?"}}
    j = {"op": "join", "args": {"instruction": "Q_MATCH: do these reviews disagree?",
                                "right_table": "right"}}
    if k_filter > 1:
        f["args"]["block_size"] = k_filter
    if k_join > 1:
        j["args"]["pair_block_size"] = k_join
    return [f, j]


FILLER = ("The pacing drags in the middle act but the leads carry it, and the "
          "score does a lot of quiet work. ")


async def _run_once(ops, n_left, n_right, pass_rows, args):
    _install_kv(args.kv_tokens)
    ex = SimExecutor(pass_rows, args.base_ms / 1000, args.per_item_ms / 1000, args.slots)
    left = [_row(f"L{i} {FILLER}", i, ex) for i in range(n_left)]
    right = [_row(f"R{j} {FILLER}", j) for j in range(n_right)]
    from vllm.kalypso.controller import plan as plan_mod
    plan_mod.data._data_source = (lambda raw, path, e=None:
                                  iter(copy.deepcopy(right)) if path == "right" else iter(left))
    t0 = time.perf_counter()
    out = await SemanticPlan(ex).execute(None, Query(copy.deepcopy(ops), "left"))
    return time.perf_counter() - t0, ex.calls, out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--selectivities", default="0.05,0.2,0.5,0.8,0.95")
    ap.add_argument("--configs", default="1x1,4x1,1x4,4x4,8x8")
    ap.add_argument("--n-left", type=int, default=100)
    ap.add_argument("--n-right", type=int, default=32)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--base-ms", type=float, default=20.0)
    ap.add_argument("--per-item-ms", type=float, default=2.0)
    ap.add_argument("--slots", type=int, default=64)
    ap.add_argument("--kv-tokens", type=int, default=2_000_000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "traces").mkdir(exist_ok=True)
    records = []
    for sel in [float(s) for s in args.selectivities.split(",")]:
        rng = random.Random(args.seed)
        pass_rows = set(rng.sample(range(args.n_left), round(sel * args.n_left)))
        for cfg in args.configs.split(","):
            kf, kj = (int(x) for x in cfg.split("x"))
            ops = pipeline(kf, kj)
            for rep in range(-args.warmup, args.repeats):
                trace_path = args.out / "traces" / f"s{int(sel*100):02d}_{cfg}_r{rep}.jsonl"
                events = []
                telemetry.configure(events, run_id=f"sim-s{sel}-{cfg}-r{rep}")
                try:
                    wall, calls, out = asyncio.run(
                        _run_once(ops, args.n_left, args.n_right, pass_rows, args))
                finally:
                    telemetry.configure(None)
                if rep < 0:
                    continue  # warm-up
                trace_path.write_text("".join(json.dumps(e) + "\n" for e in events))
                q = summarise(events)[0]
                expected_out = len(pass_rows) * sum(1 for j in range(args.n_right) if j % 2)
                rec = {
                    "selectivity": sel, "config": cfg, "k_filter": kf, "k_join": kj,
                    "repeat": rep, "latency_s": wall, "llm_calls": calls,
                    "rows_out": len(out), "rows_out_expected": expected_out,
                    "correct": len(out) == expected_out,
                    "stages": q["stages"], "peak_queued_rows": q["peak_queued_rows"],
                    "counters": q["counters"],
                    "join_block_signature": _join_groups(events),
                }
                records.append(rec)
                print(f"sel {sel:.2f} {cfg:>4} r{rep}: {wall:.3f}s, {calls} calls, "
                      f"{len(out)}/{expected_out} rows")
    with (args.out / "runs.jsonl").open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    (args.out / "config.json").write_text(json.dumps(vars(args), default=str, indent=1))
    (args.out / "summary.md").write_text(render(records, args))
    print(f"wrote {args.out}")


def _join_groups(events):
    """Sorted list of (rows per task) for the join stage: used to measure how
    much downstream block composition changes between repeats."""
    ends = [e for e in events if e["event"] == "task_end" and e["outcome"] != "retry"]
    if not ends:
        return None
    last = max(e["stage_id"] for e in ends)
    return sorted(e["rows_in"] for e in ends if e["stage_id"] == last)


def render(records, args):
    lines = [
        "# Executor simulation (CPU, simulated LLM)",
        "",
        f"filter -> join, {args.n_left} left x {args.n_right} right rows; simulated call "
        f"cost {args.base_ms:g} ms + {args.per_item_ms:g} ms/item, {args.slots} slots; "
        f"{args.repeats} repeats after {args.warmup} warm-up. Latency is the scheduler's "
        "behaviour under this cost model, **not** GPU latency.",
        "",
        "| selectivity | config (k_filter x k_join) | latency s median [min,max] | LLM calls median | "
        "join tasks | join partial blocks | join rows/task | join wait p95 s | results correct |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    groups = {}
    for r in records:
        groups.setdefault((r["selectivity"], r["config"]), []).append(r)
    for (sel, cfg), rs in groups.items():
        lat = [r["latency_s"] for r in rs]
        join = [r["stages"][-1] for r in rs]
        lines.append(
            f"| {sel:.2f} | {cfg} | {statistics.median(lat):.3f} [{min(lat):.3f}, {max(lat):.3f}] | "
            f"{statistics.median(r['llm_calls'] for r in rs):g} | "
            f"{statistics.median(j['tasks'] for j in join):g} | "
            f"{_fmt(statistics.median(j['partial_blocks'] or 0 for j in join))} | "
            f"{statistics.median(j['mean_rows_per_task'] or 0 for j in join):.2f} | "
            f"{statistics.median(j['wait_s_p95'] or 0 for j in join):.3f} | "
            f"{sum(r['correct'] for r in rs)}/{len(rs)} |")
    lines += ["", "## Downstream block composition across repeats", "",
              "Distinct join-stage task-size multisets over the repeats of one configuration "
              "(1 = identical grouping every time).", "",
              "| selectivity | config | distinct groupings / repeats |", "|---|---|---|"]
    for (sel, cfg), rs in groups.items():
        sigs = {json.dumps(r["join_block_signature"]) for r in rs}
        lines.append(f"| {sel:.2f} | {cfg} | {len(sigs)}/{len(rs)} |")
    return "\n".join(lines) + "\n"


def _fmt(x):
    return f"{x:g}"


if __name__ == "__main__":
    main()
