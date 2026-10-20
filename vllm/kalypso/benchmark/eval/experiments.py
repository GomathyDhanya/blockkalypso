"""The Milestone 2 experiments, as grids of eval runs.

    python -m vllm.kalypso.benchmark.eval.experiments E1 \\
        --base-url http://localhost:8003 --out-dir results/m2/E1

Each experiment is a list of runs: one workload at one setting (a
"condition") with the arms to compare. Every run writes ``<run>.jsonl``;
the experiment writes ``summary.md`` with every run's report.

Choices the experiment plan leaves open, made here:
- E1 row lengths are three real datasets with the paper's own filter
  prompts: arXiv abstracts (short, ~235 tokens), MEDEC notes (medium,
  ~390) and ContractNLI contracts (long, ~1750).
- E1's Level 2 arms cap blocks at 4096 estimated tokens (N@4096): about 13
  abstracts, 10 notes or 2 contracts.
- E2's "all" right rows is 8, the size of its right table.
"""

import argparse
import asyncio
from argparse import Namespace
from dataclasses import dataclass, field
from pathlib import Path

from . import report, run

FILTER_KS = (2, 4, 8, 16, 32)
LEVEL2_BUDGET = 4096


@dataclass(frozen=True)
class Run:
    name: str                 # output file stem, unique within an experiment
    workload: str
    arms: tuple[str, ...]
    n: int | None = None
    params: dict = field(default_factory=dict)
    condition: str = ""       # what this run varies, as a section title


def _filter_arms(ks=FILTER_KS, budget=None):
    blocks = tuple(f"kalypso_block:{k}" for k in ks)
    if budget:
        blocks += tuple(f"kalypso_block:{k}@{budget}" for k in ks)
    return ("kalypso_per_tuple",) + blocks


def e1() -> list[Run]:
    """Block filter alone: k, row length, Level 1 vs Level 2."""
    arms = _filter_arms(budget=LEVEL2_BUDGET)
    return [
        Run("short_arxiv", "arxiv_filter", arms, n=64, condition="short rows (arXiv)"),
        Run("medium_medec", "medec_full", arms, n=64, condition="medium rows (MEDEC)"),
        Run("long_contracts", "contractnli_filter", arms, n=64,
            condition="long rows (ContractNLI)"),
    ]


def e2() -> list[Run]:
    """Block join alone: every k x j shape on a 16 x 8 join."""
    shapes = [f"kalypso_block:{k}x{j}" for k in (1, 2, 4, 8) for j in (1, 2, 4, 8)
              if (k, j) != (1, 1)]
    return [Run("join_16x8", "arxiv_join", ("kalypso_per_tuple", *shapes), n=16,
                params={"right": 8, "f": 1.0}, condition="16 x 8, f = 1")]


def e3() -> list[Run]:
    """Input size vs block size: does the best filter k depend on N?"""
    arms = ("kalypso_per_tuple", "kalypso_block:4", "kalypso_block:16")
    return [Run(f"n{n}", "arxiv_filter", arms, n=n, condition=f"N = {n}")
            for n in (16, 32, 64, 128)]


def check_pipeline_data() -> list[Run]:
    """Before E4: does the model's per-tuple answer match the arXiv proxy labels?

    Per-tuple only, so the measured fanout (pairs the join keeps) and
    selectivity (papers Filter1 keeps) can be compared with the targets the
    workloads are built for (f = 1, s = 0.5).
    """
    return [
        Run("join_16x8", "arxiv_join", ("kalypso_per_tuple",), n=16,
            params={"right": 8, "f": 1.0}, condition="join, target f = 1"),
        Run("fjf_s05_f1", "arxiv_fjf", ("kalypso_per_tuple",), n=32,
            params={"s": 0.5, "f": 1.0}, condition="pipeline, target s = 0.5, f = 1"),
    ]


def check_join_prompt() -> list[Run]:
    """Does the arXiv join's wording decide the answers? Paper vs statement.

    Per-pair Kalypso and direct vLLM (which records every raw answer) on the
    16 x 8 join with f = 1, once per wording.
    """
    arms = ("kalypso_per_tuple", "vllm_direct")
    return [Run(f"join_{name}", "arxiv_join", arms, n=16,
                params={"right": 8, "f": 1.0, "join_prompt": name},
                condition=f"join wording: {name}")
            for name in ("paper", "statement", "related", "task")]


PIPELINE_N, PIPELINE_RIGHT = 32, 32


def e4(arms=None) -> list[Run]:
    """Pipeline, one block size k for F1, the join (k x k) and F2."""
    arms = arms or ("kalypso_per_tuple",) + tuple(
        f"kalypso_block:{k}/{k}x{k}/{k}" for k in (2, 4, 8, 16))
    return [Run("fjf_s05_f1", "arxiv_fjf", tuple(arms), n=PIPELINE_N,
                params={"s": 0.5, "f": 1.0}, condition="s = 0.5, f = 1")]


def e5(arms=None) -> list[Run]:
    """Pipeline, a block size per stage (the plan's reduced grid)."""
    joins = ("2x2", "4x4", f"4x{PIPELINE_RIGHT}")          # 4 x all right rows
    arms = arms or ("kalypso_per_tuple",) + tuple(
        f"kalypso_block:{f1}/{j}/{f2}" for f1 in (4, 16) for j in joins for f2 in (4, 16))
    return [Run("fjf_s05_f1", "arxiv_fjf", tuple(arms), n=PIPELINE_N,
                params={"s": 0.5, "f": 1.0}, condition="s = 0.5, f = 1")]


def _with_baseline(arms):
    if not arms:
        raise ValueError("E6/E7 need --arms: the best 2-3 settings from E5")
    return ("kalypso_per_tuple",) + tuple(a for a in arms if a != "kalypso_per_tuple")


def e6(arms=None) -> list[Run]:
    """Selectivity sweep with E5's best settings (f = 1)."""
    return [Run(f"s{int(s * 100):02d}", "arxiv_fjf", _with_baseline(arms), n=PIPELINE_N,
                params={"s": s, "f": 1.0}, condition=f"s = {s}, f = 1")
            for s in (0.1, 0.3, 0.6, 0.9)]


def e7(arms=None) -> list[Run]:
    """Fanout sweep with E5's best settings (s = 0.5)."""
    return [Run(f"f{str(f).replace('.', '')}", "arxiv_fjf", _with_baseline(arms),
                n=PIPELINE_N, params={"s": 0.5, "f": f}, condition=f"s = 0.5, f = {f}")
            for f in (0.2, 1.0, 3.0, 5.0)]


def e8(arms=None) -> list[Run]:
    """Pipelined vs materialized: the best setting(s) run both ways."""
    if not arms:
        raise ValueError("E8 needs --arms: the best setting(s) from E4/E5")
    base = _with_baseline([a.removesuffix("+materialized") for a in arms])
    paired = tuple(x for a in base for x in (a, a + "+materialized"))
    return [Run("fjf_s05_f1", "arxiv_fjf", paired, n=PIPELINE_N,
                params={"s": 0.5, "f": 1.0}, condition="s = 0.5, f = 1")]


def e9(arms=None) -> list[Run]:
    """Block reforming off vs on: each setting with and without coalescing
    Filter2's input, at s = 0.5, f = 1 and at the two low-occupancy corners
    of E6/E7 (low selectivity, low fanout)."""
    if not arms:
        raise ValueError("E9 needs --arms: the best block setting(s)")
    base = [a.removesuffix("+coalesce") for a in arms if a != "kalypso_per_tuple"]
    paired = ("kalypso_per_tuple",) + tuple(x for a in base for x in (a, a + "+coalesce"))
    conditions = [("s05_f1", 0.5, 1.0), ("s01_f1", 0.1, 1.0), ("s05_f02", 0.5, 0.2)]
    return [Run(name, "arxiv_fjf", paired, n=PIPELINE_N, params={"s": s, "f": f},
                condition=f"s = {s}, f = {f}")
            for name, s, f in conditions]


def e11() -> list[Run]:
    """Trummer's cost model (Level 3) vs the measured best.

    The model's (k, j) runs next to the fastest shapes measured on
    Qwen2.5-14B: 8x4 for the join alone (E2) and 4/4x4/4 for the pipeline
    (E4/E5). The chosen sizes are in each trace's stage "sizing".
    """
    return [
        Run("join_16x8", "arxiv_join",
            ("kalypso_per_tuple", "kalypso_block:trummer", "kalypso_block:8x4"), n=16,
            params={"right": 8, "f": 1.0}, condition="join alone, 16 x 8, f = 1"),
        Run("fjf_s05_f1", "arxiv_fjf",
            ("kalypso_per_tuple", "kalypso_block:4/trummer/4", "kalypso_block:4/4x4/4"),
            n=PIPELINE_N, params={"s": 0.5, "f": 1.0}, condition="pipeline, s = 0.5, f = 1"),
    ]


EXPERIMENTS = {"E1": e1, "E2": e2, "E3": e3, "E4": e4, "E5": e5, "E6": e6, "E7": e7,
               "E8": e8, "E9": e9, "E11": e11, "CHECK": check_pipeline_data,
               "JOINCHECK": check_join_prompt}
TAKES_ARMS = {"E4", "E5", "E6", "E7", "E8", "E9"}


def build(name: str, arms=None) -> list[Run]:
    return EXPERIMENTS[name](arms) if name in TAKES_ARMS else EXPERIMENTS[name]()


async def run_experiment(name: str, args) -> Path:
    runs = build(name, args.arms)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for r in runs:
        print(f"== {name} {r.name} ==", flush=True)
        await run.run(Namespace(
            base_url=args.base_url, model=args.model, workload=r.workload, n=r.n,
            params=dict(r.params), arms=list(r.arms), warmup=args.warmup,
            repeats=args.repeats, server_pid=args.server_pid,
            allow_warm_cache=args.allow_warm_cache, timeout=args.timeout,
            out=str(out_dir / f"{r.name}.jsonl")))
    return write_reports(name, runs, out_dir)


def write_reports(name: str, runs: list[Run], out_dir: Path) -> Path:
    sections = [f"# {name}", ""]
    for r in runs:
        meta, trials = report.load(out_dir / f"{r.name}.jsonl")
        sections += [f"## {r.name}: {r.condition}", ""]
        if not trials:
            sections += ["No trials recorded.", ""]
            continue
        sections.append(report.full_report(meta, trials))
    summary = out_dir / "summary.md"
    summary.write_text("\n".join(sections), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("experiment", choices=sorted(EXPERIMENTS))
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--base-url", default="http://localhost:8003")
    parser.add_argument("--model", default=None)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--server-pid", type=int, default=None)
    parser.add_argument("--allow-warm-cache", action="store_true")
    parser.add_argument("--timeout", type=float, default=3600)
    parser.add_argument("--arms", default=None, type=lambda s: s.split(","),
                        help="E4-E9: override the arms (E6-E9 need E4/E5's best settings)")
    args = parser.parse_args()
    args.base_url = args.base_url.rstrip("/").removesuffix("/v1")
    print(f"wrote {asyncio.run(run_experiment(args.experiment, args))}")


if __name__ == "__main__":
    main()
