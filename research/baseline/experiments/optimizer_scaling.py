"""Step 4A: how the original exhaustive block-size search scales (CPU only).

    python -m research.baseline.experiments.optimizer_scaling \
        --out research/baseline/results/optimizer_scaling.jsonl

Optimizer-only, synthetic pipelines: no LLM is called and no latency here is
a query latency. The model is the committed v3 fit (costmodel.json) and the
code is costmodel.Model.choose_pipeline, unchanged.

Grid: pipeline families x depth 1..6 x candidate sets:

- ``exhaustive``: the original default, every k in 1..k_max per stage;
- ``pow2``: k in {1, 2, 4, 8, 16, 32};
- ``small``: k in {1, 2, 3, 4}.

Each configuration runs ``--repeats`` times (planning latency: median, min,
max). A configuration whose legal space exceeds ``--max-candidates`` is not
run; it is recorded with ``executed: false`` and the reason, and its
per-candidate cost is measured on a deterministic sample of
``--sample`` candidates (``sample_*`` fields) so the full cost can be
discussed, clearly labelled as an extrapolation.
"""

from __future__ import annotations

import argparse
import datetime as dt
import itertools
import json
import platform
import random
import statistics
import subprocess
import sys
from pathlib import Path

from vllm.kalypso.benchmark.eval import costmodel as cm

from ..instrumentation.optimizer_profile import (
    legal_space, load_model, profile_choose_pipeline)

REPO = Path(__file__).resolve().parents[3]

# Row shapes from the measured workloads: SemBench reviews (~31 tokens, short
# regime), 400 rows, 32 right rows for joins.
ROWS, TOKENS, MAX_TOKENS = 400, 31.0, 120.0
RIGHT_ROWS, RIGHT_TOKENS, RIGHT_MAX = 32, 31.0, 120.0
SELECTIVITY = 0.5

FAMILIES = {
    # Filters only: the cheapest per candidate, k_max 20 each.
    "filters": ["filter"] * 6,
    # Alternating filter / join: joins expand, filters contract.
    "filter_join": ["filter", "join", "filter", "join", "filter", "join"],
    # Mixed with maps (retry shares apply).
    "mixed": ["filter", "join", "map", "filter", "join", "map"],
}
CANDIDATE_SETS = {
    "exhaustive": None,
    "pow2": (1, 2, 4, 8, 16, 32),
    "small": (1, 2, 3, 4),
}


def pipeline(ops) -> cm.Pipeline:
    return cm.Pipeline(tuple(ops), ROWS, TOKENS, MAX_TOKENS, SELECTIVITY,
                       RIGHT_ROWS if "join" in ops else 0,
                       RIGHT_TOKENS if "join" in ops else 0.0,
                       RIGHT_MAX if "join" in ops else 0.0)


def candidates(model, pipe, cset):
    if cset is None:
        return None
    return list(itertools.product(cset, repeat=len(pipe.ops)))


def sample_cost(model, pipe, cset, n, seed=0) -> dict:
    """Seconds per evaluated candidate on n candidates drawn uniformly from
    the legal space (with replacement, fixed seed)."""
    reasons = [model.k_max(op, it) for op, it in zip(pipe.ops, model.stage_items(pipe, pipe.rows))]
    per_stage = [[k for k in (cset or range(1, m + 1)) if k <= m] or [1]
                 for m in reasons]
    rng = random.Random(seed)
    sample = [tuple(rng.choice(ks) for ks in per_stage) for _ in range(n)]
    rec = profile_choose_pipeline(model, pipe, candidates=sample, memory=True)
    return {"sample_size": n, "sample_planning_s": rec["planning_s"],
            "sample_s_per_candidate": rec["planning_s"] / n,
            "sample_peak_heap_bytes": rec["peak_python_heap_bytes"],
            "sample_heap_bytes_per_candidate": rec["peak_python_heap_bytes"] / n}


def git(*args):
    try:
        return subprocess.check_output(["git", "-C", str(REPO), *args], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--max-depth", type=int, default=6)
    ap.add_argument("--max-candidates", type=int, default=2_000_000)
    ap.add_argument("--sample", type=int, default=20_000)
    args = ap.parse_args()
    model = load_model()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    meta = {"type": "meta", "experiment": "optimizer_scaling",
            "timestamp": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
            "git_sha": git("rev-parse", "HEAD"),
            "git_dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
            "python": platform.python_version(), "platform": platform.platform(),
            "cpu": platform.processor() or platform.machine(),
            "model": "costmodel v3 (costmodel.json)", "repeats": args.repeats,
            "max_candidates": args.max_candidates, "sample": args.sample,
            "rows": ROWS, "tokens": TOKENS, "right_rows": RIGHT_ROWS,
            "selectivity": SELECTIVITY, "synthetic": True}
    with args.out.open("w") as f:
        f.write(json.dumps(meta) + "\n")
        for family, ops in FAMILIES.items():
            for depth in range(1, args.max_depth + 1):
                pipe = pipeline(ops[:depth])
                for cname, cset in CANDIDATE_SETS.items():
                    enumerated, legal = legal_space(model, pipe, candidate_set=cset)
                    base = {"type": "config", "family": family, "depth": depth,
                            "candidate_set": cname, "candidates_enumerated": enumerated,
                            "candidates_legal": legal}
                    if legal > args.max_candidates:
                        rec = {**base, "executed": False,
                               "reason": f"legal space {legal:,} > --max-candidates "
                                         f"{args.max_candidates:,}",
                               **sample_cost(model, pipe, cset, args.sample)}
                        f.write(json.dumps(rec) + "\n")
                        f.flush()
                        print(f"{family} d{depth} {cname}: NOT RUN ({legal:,} candidates), "
                              f"{rec['sample_s_per_candidate'] * 1e6:.1f} us/candidate")
                        continue
                    runs = []
                    for r in range(args.repeats):
                        runs.append(profile_choose_pipeline(
                            model, pipe, candidates=candidates(model, pipe, cset),
                            memory=(r == 0)))
                    lat = [x["planning_s"] for x in runs]
                    first = runs[0]
                    rec = {**base, "executed": True, **{k: v for k, v in first.items()
                                                       if k != "planning_s"},
                           "planning_s_runs": lat,
                           "planning_s_median": statistics.median(lat),
                           "planning_s_min": min(lat), "planning_s_max": max(lat)}
                    f.write(json.dumps(rec) + "\n")
                    f.flush()
                    print(f"{family} d{depth} {cname}: {legal:,} legal, "
                          f"{first['candidates_evaluated']:,} evaluated, "
                          f"median {rec['planning_s_median']:.4f}s, "
                          f"heap {first['peak_python_heap_bytes'] / 1e6:.1f} MB, "
                          f"pick {first['chosen']}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    sys.exit(main())
