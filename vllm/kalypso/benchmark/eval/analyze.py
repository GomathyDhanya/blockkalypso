"""Turn every results JSONL into one table for analysis and the cost model.

    python3 -m vllm.kalypso.benchmark.eval.analyze [--results DIR] [--out DIR]

Reads every ``run.jsonl`` under ``eval/results/`` (the block-size search and
cloud/experiments.py's runs) and writes to ``eval/results/analysis/``:

- ``arms.csv``: one row per (run, arm). Medians over repeats of every
  measured column, plus what the cost model needs: speedup over the
  per-tuple arm of the same run, calls, prompt and generated tokens per
  call, prompt tokens actually computed (after prefix-cache hits), decode
  seconds per generated token, waves (calls / max_num_seqs) and quality
  deltas against per-tuple.
- ``summary.md``: per run, a compact table and the best arm (fastest whose F1
  is within ``--f1-slack`` of per-tuple's, or, for unlabeled maps, whose
  agreement with per-tuple is at least ``--min-agreement``); per pipeline
  experiment, the pipeline's latency against its operators run alone in the
  same container (sum and max of the stages).
"""

import argparse
import csv
import math
import re
import statistics
from pathlib import Path

from . import report

RESULTS = Path(__file__).resolve().parent / "results"
BASELINE = "kalypso_per_tuple"
# vLLM 0.29.0's API-server default on GPUs under 70 GiB, for runs that did not
# record it (the block-size search on the L40S).
DEFAULT_MAX_NUM_SEQS = 256

MEASURED = ("latency_s", "llm_requests", "block_calls", "fallback_calls", "truncated",
            "prompt_tokens", "generation_tokens", "prefix_cache_hit_rate",
            "peak_kv_cache_pct", "peak_running", "peak_waiting", "preemptions",
            "mean_prefill_time_s", "mean_decode_time_s", "rows_per_s", "rows_out")
QUALITY = ("accuracy", "precision", "recall", "f1")
COLUMNS = (
    "source", "experiment", "phase", "part", "workload", "kind", "stages", "items",
    "rows", "right_rows", "positives", "positive_share", "arm", "blocks", "trials",
    "failed_trials", *MEASURED, "speedup", "calls_per_item", "prompt_tokens_per_call",
    "gen_tokens_per_call", "computed_prompt_tokens", "decode_s_per_gen_token",
    "waves", *QUALITY, "f1_delta", "recall_delta", "agreement", "self_agreement",
    "model", "gpu", "max_num_seqs", "git_sha",
)


def _median(values):
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else None


def _blocks(arm: str, stages: list | None) -> str:
    """Block size per stage, e.g. "6,4"; 1 is per-tuple."""
    n = len(stages) if stages else 1
    kind, _, size = arm.partition(":")
    if kind == "kalypso_pipe":
        return size
    if kind == "kalypso_block":
        return ",".join([size] * n)
    if kind in ("kalypso_per_tuple", "vllm_direct"):
        return ",".join(["1"] * n)
    return ""


def _ratio(a, b, digits=4):
    return round(a / b, digits) if a is not None and b else None


def run_rows(path: Path, root: Path) -> list[dict]:
    meta, trials = report.load(path)
    groups = report.by_arm(trials)
    failed = report.failures(trials)
    agree = report.agreement(meta, trials, BASELINE) if trials else {}
    items = len(report.universe(meta))
    positives = len(meta["positives"]) if meta.get("positives") is not None else None
    max_num_seqs = meta.get("max_num_seqs") or DEFAULT_MAX_NUM_SEQS
    rel = path.parent.relative_to(root)
    out = []
    for arm in meta["arms"]:
        group = groups.get(arm, [])
        row = {
            "source": str(rel), "experiment": meta.get("experiment") or "",
            "phase": meta.get("phase") or rel.parts[0], "part": meta.get("part") or "",
            "workload": meta["workload"], "kind": meta.get("kind", "map"),
            "stages": ">".join(s["op"] for s in meta["stages"]) if meta.get("stages") else "",
            "items": items, "rows": meta["rows"], "right_rows": meta.get("right_rows"),
            "positives": positives,
            "positive_share": _ratio(positives, items) if positives is not None else None,
            "arm": arm, "blocks": _blocks(arm, meta.get("stages")),
            "trials": len(group), "failed_trials": len(failed.get(arm, [])),
            "model": meta.get("model"), "gpu": meta.get("gpu") or "",
            "max_num_seqs": max_num_seqs, "git_sha": (meta.get("git_sha") or "")[:9],
        }
        for key in MEASURED:
            row[key] = _median(t.get(key) for t in group)
        quality = [report.quality(meta, t) for t in group]
        for key in QUALITY:
            row[key] = _median(q.get(key) for q in quality)
        if arm in agree:
            row["agreement"] = agree[arm]["vs_reference"]
            row["self_agreement"] = agree[arm]["self"]
        calls, hit = row["llm_requests"], row["prefix_cache_hit_rate"]
        row["calls_per_item"] = _ratio(calls, items)
        row["prompt_tokens_per_call"] = _ratio(row["prompt_tokens"], calls, 1)
        row["gen_tokens_per_call"] = _ratio(row["generation_tokens"], calls, 2)
        if row["prompt_tokens"] is not None and hit is not None:
            row["computed_prompt_tokens"] = round(row["prompt_tokens"] * (1 - hit))
        row["decode_s_per_gen_token"] = _ratio(row["mean_decode_time_s"],
                                               row["gen_tokens_per_call"])
        row["waves"] = math.ceil(calls / max_num_seqs) if calls else None
        out.append(row)
    base = next((r for r in out if r["arm"] == BASELINE), None)
    for row in out:
        if base:
            row["speedup"] = _ratio(base["latency_s"], row["latency_s"], 3)
            for key in ("f1", "recall"):
                if row[key] is not None and base[key] is not None:
                    row[f"{key}_delta"] = round(row[key] - base[key], 4)
    return out


def best_arm(rows: list[dict], f1_slack: float, min_agreement: float) -> dict | None:
    """Fastest arm whose quality holds against per-tuple (None if no baseline)."""
    base = next((r for r in rows if r["arm"] == BASELINE), None)
    if not base or base["latency_s"] is None:
        return None

    def ok(r):
        if r["latency_s"] is None:
            return False
        if base.get("f1") is not None and r.get("f1") is not None:
            return r["f1"] >= base["f1"] - f1_slack
        return (r.get("agreement") or 0) >= min_agreement or r is base

    return min((r for r in rows if ok(r)), key=lambda r: r["latency_s"])


def _fmt(value, digits=2):
    if value is None or value == "":
        return "–"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


RUN_TABLE = (
    ("arm", "Arm", 0), ("latency_s", "Latency (s)", 2), ("speedup", "Speedup", 2),
    ("llm_requests", "Calls", 0), ("fallback_calls", "Fallbacks", 0),
    ("waves", "Waves", 0), ("peak_running", "Peak running", 0),
    ("peak_waiting", "Peak waiting", 0), ("computed_prompt_tokens", "Computed prompt tok", 0),
    ("gen_tokens_per_call", "Gen tok/call", 1), ("mean_decode_time_s", "Decode/call (s)", 2),
    ("decode_s_per_gen_token", "s/gen tok", 3), ("prefix_cache_hit_rate", "Hit rate", 2),
    ("peak_kv_cache_pct", "Peak KV %", 1), ("preemptions", "Preempt", 0),
    ("recall", "Recall", 2), ("f1", "F1", 2), ("f1_delta", "ΔF1", 2),
    ("agreement", "Agree", 2), ("self_agreement", "Self", 2),
)


def run_table(rows: list[dict]) -> list[str]:
    lines = ["| " + " | ".join(label for _, label, _ in RUN_TABLE) + " |",
             "|" + "---|" * len(RUN_TABLE)]
    for r in rows:
        cells = []
        for key, _, digits in RUN_TABLE:
            value = r.get(key)
            cells.append(f"`{value}`" if key == "arm" else _fmt(
                float(value) if isinstance(value, int) and digits else value, digits))
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def composition(rows_by_part: dict[str, list[dict]]) -> list[str]:
    """Pipeline latency against its stages run alone in the same container."""
    pipeline = next((rows for rows in rows_by_part.values() if rows and rows[0]["stages"]), None)
    if not pipeline:
        return []
    ops = pipeline[0]["stages"].split(">")
    alone = {op: {r["blocks"]: r for r in rows_by_part.get(f"alone-{op}", [])} for op in ops}
    if not all(alone.values()):
        return []
    lines = ["| Pipeline arm | Pipeline (s) | " + " | ".join(f"{op} alone (s)" for op in ops)
             + " | Sum (s) | Max (s) | Pipeline / sum | Pipeline / max |",
             "|" + "---|" * (len(ops) + 6)]
    for r in pipeline:
        sizes = r["blocks"].split(",")
        parts = [alone[op].get(k) for op, k in zip(ops, sizes)]
        if r["latency_s"] is None or any(p is None or p["latency_s"] is None for p in parts):
            continue
        times = [p["latency_s"] for p in parts]
        lines.append(f"| `{r['arm']}` | {r['latency_s']:.2f} | "
                     + " | ".join(f"{t:.2f}" for t in times)
                     + f" | {sum(times):.2f} | {max(times):.2f} | "
                     f"{r['latency_s'] / sum(times):.2f} | {r['latency_s'] / max(times):.2f} |")
    return lines if len(lines) > 2 else []


def summary(all_rows: list[dict], f1_slack: float, min_agreement: float) -> str:
    by_source = {}
    for r in all_rows:
        by_source.setdefault(r["source"], []).append(r)
    by_experiment = {}
    for source, rows in by_source.items():
        key = rows[0]["experiment"] or source
        by_experiment.setdefault(key, {})[rows[0]["part"] or source] = rows

    lines = ["# Batching study results", "",
             f"Best arm: fastest with F1 at least per-tuple's minus {f1_slack} "
             f"(unlabeled maps: agreement with per-tuple at least {min_agreement:.0%}). "
             "Medians over repeats; speedup is against per-tuple in the same container.", ""]
    phases = {}
    for exp, parts in by_experiment.items():
        phase = next(iter(parts.values()))[0]["phase"]
        phases.setdefault(phase, []).append((exp, parts))
    for phase in sorted(phases):
        lines += [f"## {phase}", ""]
        for exp, parts in sorted(phases[phase]):
            lines += [f"### {exp}", ""]
            for part_name, rows in parts.items():
                first = rows[0]
                size = (f"{first['rows']} x {first['right_rows']} = {first['items']} pairs"
                        if first["right_rows"] else f"{first['items']} rows")
                share = (f", {first['positive_share']:.0%} positive"
                         if first["positive_share"] is not None else "")
                stages = f", stages {first['stages']}" if first["stages"] else ""
                lines.append(f"**{part_name}**: `{first['workload']}` ({first['kind']}), "
                             f"{size}{share}{stages}; GPU {first['gpu'] or 'L40S (not recorded)'}, "
                             f"max_num_seqs {first['max_num_seqs']}, git `{first['git_sha']}`.")
                lines.append("")
                lines += run_table(rows)
                best = best_arm(rows, f1_slack, min_agreement)
                if best:
                    lines += ["", f"Best: `{best['arm']}` ({_fmt(best['speedup'])}x)."]
                failed = [r for r in rows if r["failed_trials"]]
                for r in failed:
                    lines.append(f"- `{r['arm']}` failed {r['failed_trials']} trial(s); "
                                 "see run.jsonl")
                lines.append("")
            comp = composition(parts)
            if comp:
                lines += ["Composition (same container):", ""] + comp + [""]
    return "\n".join(lines) + "\n"


def collect(root: Path) -> list[dict]:
    rows = []
    for path in sorted(root.rglob("run.jsonl")):
        if "analysis" in path.relative_to(root).parts:
            continue
        try:
            rows += run_rows(path, root)
        except (ValueError, KeyError) as exc:
            print(f"skipping {path}: {exc}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--out", type=Path, default=None,
                        help="default: <results>/analysis")
    parser.add_argument("--f1-slack", type=float, default=0.02)
    parser.add_argument("--min-agreement", type=float, default=0.9)
    args = parser.parse_args()
    out = args.out or args.results / "analysis"
    rows = collect(args.results)
    if not rows:
        raise SystemExit(f"no run.jsonl under {args.results}")
    out.mkdir(parents=True, exist_ok=True)
    with (out / "arms.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    (out / "summary.md").write_text(summary(rows, args.f1_slack, args.min_agreement),
                                    encoding="utf-8")
    runs = len({r["source"] for r in rows})
    print(f"{len(rows)} arms from {runs} runs -> {out / 'arms.csv'}, {out / 'summary.md'}")


if __name__ == "__main__":
    main()
