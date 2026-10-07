"""Run every Milestone 1 workload with its default arms, then write one report.

    python -m vllm.kalypso.benchmark.eval.suite \\
        --out-dir vllm/kalypso/benchmark/eval/results/m1

Writes ``<workload>.jsonl`` per workload and ``summary.md`` with every table.
Each workload runs with the same flags ``run.py`` takes.
"""

import argparse
import asyncio
from argparse import Namespace
from pathlib import Path

from . import report, run

WORKLOADS = ("biodex", "reviews", "medec", "join5x5")


def write_summary(out_dir: Path, workloads) -> Path:
    sections = ["# Milestone 1 evaluation", ""]
    for name in workloads:
        path = out_dir / f"{name}.jsonl"
        if not path.exists():
            sections += [f"## {name}", "", "Not run.", ""]
            continue
        meta, trials = report.load(path)
        sections += [f"## {name}", ""]
        if not trials:
            sections += ["No trials recorded.", ""]
            continue
        sections.append(report.to_markdown(
            meta, report.summarize(meta, trials),
            report.agreement(meta, trials, "kalypso_per_tuple"),
            "kalypso_per_tuple", report.failures(trials)))
    summary = out_dir / "summary.md"
    summary.write_text("\n".join(sections), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--workloads", default=",".join(WORKLOADS),
                        type=lambda s: s.split(","))
    parser.add_argument("--base-url", default="http://localhost:8003")
    parser.add_argument("--model", default=None)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--server-pid", type=int, default=None)
    parser.add_argument("--allow-warm-cache", action="store_true")
    parser.add_argument("--timeout", type=float, default=3600)
    args = parser.parse_args()
    unknown = [w for w in args.workloads if w not in run.WORKLOADS]
    if unknown:
        raise SystemExit(f"unknown workloads {unknown}; choose from {sorted(run.WORKLOADS)}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    base_url = args.base_url.rstrip("/").removesuffix("/v1")
    for name in args.workloads:
        print(f"== {name} ==", flush=True)
        asyncio.run(run.run(Namespace(
            base_url=base_url, model=args.model, workload=name, n=None, arms=None,
            warmup=args.warmup, repeats=args.repeats, server_pid=args.server_pid,
            allow_warm_cache=args.allow_warm_cache, timeout=args.timeout,
            out=str(args.out_dir / f"{name}.jsonl"))))
    print(f"wrote {write_summary(args.out_dir, args.workloads)}")


if __name__ == "__main__":
    main()
