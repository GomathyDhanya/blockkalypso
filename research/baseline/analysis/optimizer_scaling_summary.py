"""Tables for the optimizer scaling experiment (Experiment A).

    python -m research.baseline.analysis.optimizer_scaling_summary \
        research/baseline/results/optimizer_scaling.jsonl > research/baseline/results/optimizer_scaling.md

Exhaustive configurations above the candidate cap were not run; for those the
table shows the candidate count and an extrapolation (legal candidates x the
measured per-candidate time and heap of a sample), clearly labelled.

"Plan regret" compares a restricted candidate set's chosen plan with the
exhaustive search's chosen plan, both under the cost model's own predicted
latency (not measured latency). A negative regret is possible: ``_pick``
(costmodel.py:310-317) takes the smallest blocks within 2% of the best
prediction, so the exhaustive pick need not be the minimum.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main(path: str):
    recs = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    meta = next(r for r in recs if r.get("type") == "meta")
    cfg = [r for r in recs if r.get("type") == "config"]
    ex = {(r["family"], r["depth"]): r for r in cfg if r["candidate_set"] == "exhaustive"}
    out = [
        "# Optimizer scaling (Experiment A)",
        "",
        f"Cost model v3 (`costmodel.json`), synthetic shape: {meta['rows']} rows, "
        f"{meta['tokens']:g} tokens/row, {meta['right_rows']} right rows, selectivity "
        f"{meta['selectivity']}. {meta['repeats']} timed repeats per configuration; "
        f"cap {meta['max_candidates']:,} candidates. CPU: {meta['platform']}, Python "
        f"{meta['python']}, git {meta['git_sha'][:7]}.",
        "",
        "| family | stages | candidate set | legal plans | run? | planning s median [min,max] | peak heap MB | chosen blocks | predicted s | regret vs exhaustive |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in cfg:
        e = ex[(r["family"], r["depth"])]
        if r["executed"]:
            t = f"{r['planning_s_median']:.4f} [{r['planning_s_min']:.4f}, {r['planning_s_max']:.4f}]"
            heap = f"{r['peak_python_heap_bytes'] / 1e6:.1f}"
            pred = f"{r['predicted_latency_s']:.2f}"
            chosen = str(r["chosen"])
        else:
            n = r["candidates_legal"]
            s = r.get("sample_s_per_candidate")
            h = r.get("sample_heap_bytes_per_candidate")
            t = f"NOT RUN (extrapolated ~{n * s:,.0f})" if s else "NOT RUN"
            heap = f"NOT RUN (extrapolated ~{n * h / 1e6:,.0f})" if h else "NOT RUN"
            pred, chosen = "-", "-"
        if r["executed"] and e["executed"] and r is not e:
            reg = f"{(r['predicted_latency_s'] / e['predicted_latency_s'] - 1) * 100:+.1f}%"
        elif r is e:
            reg = "(reference)" if r["executed"] else "-"
        else:
            reg = "n/a (exhaustive not run)"
        out.append(f"| {r['family']} | {r['depth']} | {r['candidate_set']} | "
                   f"{r['candidates_legal']:,} | {'yes' if r['executed'] else 'no'} | {t} | {heap} | "
                   f"{chosen} | {pred} | {reg} |")
    print("\n".join(out))


if __name__ == "__main__":
    main(sys.argv[1])
