"""Summarise a Kalypso telemetry trace (KALYPSO_TRACE JSONL) per query.

    python -m research.baseline.analysis.trace_summary TRACE.jsonl [--json OUT]

Per query: latency, planning time, and per stage the observed cardinality
(rows in/out, selectivity or fanout), task/block counts and fill, waiting and
execution times, retries; per query the LLM calls (count, tokens, latency by
stage), block-operator parse outcomes, KV admission counters and the peak
scheduler queue. Used for both the CPU simulation and the GPU traces.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def _pct(values, q):
    v = sorted(values)
    if not v:
        return None
    i = min(len(v) - 1, max(0, round(q * (len(v) - 1))))
    return v[i]


def load(path: Path) -> list[dict]:
    import gzip
    path = Path(path)
    text = (gzip.open(path, "rt").read() if path.suffix == ".gz" else path.read_text())
    out = []
    for line in text.splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # a line cut off when the server was stopped
    return out


def summarise(events: list[dict]) -> list[dict]:
    by_query = defaultdict(list)
    for e in events:
        if e.get("query_id"):
            by_query[e["query_id"]].append(e)
    out = []
    for qid, evs in by_query.items():
        start = next((e for e in evs if e["event"] == "query_start"), None)
        end = next((e for e in evs if e["event"] == "query_end"), None)
        plan = next((e for e in evs if e["event"] == "plan_built"), None)
        stage_meta = {}
        if plan:
            for item in plan["plan"]:
                for st in item.get("segment", []):
                    stage_meta[st["stage_id"]] = st
        ends = [e for e in evs if e["event"] == "task_end"]
        starts = {e["task_id"]: e for e in evs if e["event"] == "task_start"}
        stages = []
        for sid in sorted({e["stage_id"] for e in ends}):
            se = [e for e in ends if e["stage_id"] == sid]
            ss = [starts[e["task_id"]] for e in se if e["task_id"] in starts]
            meta = stage_meta.get(sid, {})
            k = meta.get("block_size", 1)
            rows_in = sum(e["rows_in"] for e in se if e["outcome"] != "retry")
            rows_out = sum(e["rows_out"] for e in se)
            sizes = [e["rows_in"] for e in se if e["outcome"] != "retry"]
            calls = [e for e in evs if e["event"] == "llm_call" and e.get("stage_id") == sid]
            evals = [e for e in evs if e["event"] == "block_eval" and e.get("stage_id") == sid]
            stages.append({
                "stage_id": sid, "ops": meta.get("ops"), "block_size": k,
                "fanout": meta.get("fanout"),
                "tasks": len(sizes), "retries": sum(e["outcome"] == "retry" for e in se),
                "rows_in": rows_in, "rows_out": rows_out,
                "out_per_in": rows_out / rows_in if rows_in else None,
                "full_blocks": sum(s == k for s in sizes) if k > 1 else None,
                "partial_blocks": sum(s < k for s in sizes) if k > 1 else None,
                "mean_rows_per_task": statistics.mean(sizes) if sizes else None,
                "wait_s_p50": _pct([s["wait_s"] for s in ss], 0.5),
                "wait_s_p95": _pct([s["wait_s"] for s in ss], 0.95),
                "exec_s_p50": _pct([e["exec_s"] for e in se], 0.5),
                "exec_s_p95": _pct([e["exec_s"] for e in se], 0.95),
                "forced_launches": sum(bool(s.get("forced")) for s in ss),
                "budget_tokens_max": max((s["budget_bytes"] or 0 for s in ss), default=None),
                "llm_calls": len(calls),
                "prompt_tokens": _sum(c.get("prompt_tokens") for c in calls),
                "completion_tokens": _sum(c.get("completion_tokens") for c in calls),
                "cached_tokens": _sum(c.get("cached_tokens") for c in calls),
                "call_latency_s_p50": _pct([c["latency_s"] for c in calls], 0.5),
                "call_latency_s_p95": _pct([c["latency_s"] for c in calls], 0.95),
                "truncated_calls": sum(c.get("finish_reason") == "length" for c in calls),
                "block_fallback_calls": sum(e.get("fallback_calls", 0) for e in evals),
                "block_undecided_rows": sum(e.get("undecided", 0) for e in evals),
                "block_malformed": sum(bool(e.get("malformed")) for e in evals),
            })
        samples = [e for e in evs if e["event"] == "scheduler_sample"]
        peak_wait = {}
        for s in samples:
            for st in s["stages"]:
                peak_wait[st["stage_id"]] = max(peak_wait.get(st["stage_id"], 0),
                                                st["waiting_rows"] + st["pending_block_rows"])
        out.append({
            "query_id": qid, "ops": (start or {}).get("ops"),
            "t_wall_start": (start or {}).get("t_wall"), "t_wall_end": (end or {}).get("t_wall"),
            "rows_in": (end or {}).get("rows_in"), "rows_out": (end or {}).get("rows_out"),
            "latency_s": (end or {}).get("latency_s"), "plan_s": (plan or {}).get("plan_s"),
            "stages": stages, "peak_queued_rows": peak_wait,
            "counters": (end or {}).get("counters"), "op_stats": (end or {}).get("op_stats"),
            "llm_calls": sum(s["llm_calls"] for s in stages),
            "complete": end is not None,
        })
    return sorted(out, key=lambda q: q["t_wall_start"] or 0)


def _sum(values):
    v = [x for x in values if x is not None]
    return sum(v) if v else None


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("trace", type=Path)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    queries = summarise(load(args.trace))
    if args.json:
        args.json.write_text(json.dumps(queries, indent=1))
    for q in queries:
        print(f"query {q['query_id']}: {q['rows_in']} -> {q['rows_out']} rows, "
              f"{q['latency_s']}s, plan {q['plan_s']}s, {q['llm_calls']} LLM calls")
        for s in q["stages"]:
            print(f"  stage {s['stage_id']} {s['ops']} k={s['block_size']}: "
                  f"{s['rows_in']} -> {s['rows_out']} ({s['tasks']} tasks, "
                  f"{s['partial_blocks']} partial), wait p50 {s['wait_s_p50']}")


if __name__ == "__main__":
    main()
