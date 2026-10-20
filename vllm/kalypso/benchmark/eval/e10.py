"""E10: the accuracy cost of larger blocks, collected from all experiments.

    python -m vllm.kalypso.benchmark.eval.e10 RESULTS_DIR [RESULTS_DIR ...] \\
        --out results/m2-E10-qwen14b.md

The plan's E10 is "collected from all of the above": no new runs. For every
results file (a condition) and arm it reports the plan's three measures:

- F1 against the workload's labels (gold or proxy);
- agreement with per-tuple Kalypso, the proposal's own accuracy measure
  ("checked against per-tuple execution as ground truth");
- fallback calls (block answers that had to be redone per row or pair);

plus, for pipelines, whether the arm's work drifted from per-tuple. Medians
over repeats. Accuracy is recorded, not used to choose a block size.
"""

import argparse
from pathlib import Path

from . import report
from .arms import parse_arm

REFERENCE = "kalypso_per_tuple"


def block_label(arm: str) -> str:
    spec = parse_arm(arm)
    flags = ("+materialized" if spec.materialize else "") + ("+coalesce" if spec.coalesce else "")
    if spec.kind != "kalypso_block":
        return ("per-tuple" if spec.kind == REFERENCE else spec.kind) + flags
    parts = ["x".join(map(str, g)) + (f"@{b}" if b else "")
             for g, b in zip(spec.groups, spec.budgets or (None,) * len(spec.groups))]
    return "/".join(parts) + flags


def f1_vs_reference(meta: dict, trials: list[dict], arm: str) -> float | None:
    """Median F1 of the arm's selected items, taking per-tuple's selected
    items in the same repeat as the truth. Unlike agreement over every
    candidate, this is not diluted when almost every candidate is a "no"."""
    groups = report.by_arm(trials)
    ref = {t["repeat"]: {report._key(x) for x in t.get("selected") or []}
           for t in groups.get(REFERENCE, [])}
    scores = []
    for t in groups.get(arm, []):
        if t["repeat"] not in ref or t.get("selected") is None:
            continue
        truth, got = ref[t["repeat"]], {report._key(x) for x in t["selected"]}
        tp = len(got & truth)
        if not truth and not got:
            scores.append(1.0)
        elif tp == 0:
            scores.append(0.0)
        else:
            precision, recall = tp / len(got), tp / len(truth)
            scores.append(2 * precision * recall / (precision + recall))
    return sorted(scores)[len(scores) // 2] if scores else None


def collect(dirs) -> list[dict]:
    rows = []
    for d in map(Path, dirs):
        for f in sorted(d.glob("*.jsonl")):
            meta, trials = report.load(f)
            if not trials:
                continue
            summary = report.summarize(meta, trials)
            agree = report.agreement(meta, trials, REFERENCE)
            drift = (report.drift(meta, trials, REFERENCE)
                     if meta.get("kind") == "pipeline" else {})
            failed = report.failures(trials)
            for arm in meta["arms"]:
                if arm not in summary:
                    rows.append({"experiment": d.name, "condition": f.stem, "arm": arm,
                                 "blocks": block_label(arm), "failed": len(failed.get(arm, []))})
                    continue
                med = lambda key: (summary[arm][key] or {}).get("median")
                rows.append({
                    "experiment": d.name, "condition": f.stem, "arm": arm,
                    "blocks": block_label(arm), "f1": med("f1"),
                    "agreement": agree[arm]["vs_reference"],
                    "f1_vs_reference": f1_vs_reference(meta, trials, arm),
                    "fallbacks": med("fallback_calls"),
                    "drifted": drift.get(arm, {}).get("drifted"),
                    "failed": 0,
                })
    return rows


def _fmt(v, kind="num"):
    if v is None:
        return "n/a"
    return f"{v:.0%}" if kind == "pct" else f"{v:.2f}" if isinstance(v, float) else str(v)


def to_markdown(rows: list[dict]) -> str:
    lines = ["# E10: accuracy cost of larger blocks", "",
             "F1 against labels, agreement with per-tuple Kalypso (the proposal's accuracy "
             "measure) and fallback calls, medians over repeats. Drifted = the pipeline's "
             "work differed from per-tuple by more than 10% at Filter1, the join or the "
             "output.", "",
             "F1 vs per-tuple takes per-tuple's output as the truth; agreement counts "
             "every candidate, so it stays high when outputs are sparse.", "",
             "| Experiment | Condition | Blocks | F1 vs labels | F1 vs per-tuple | "
             "Agrees w/ per-tuple | Fallback calls | Drifted |",
             "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if r.get("failed") and "f1" not in r:
            lines.append(f"| {r['experiment']} | {r['condition']} | {r['blocks']} | "
                         f"failed ({r['failed']} trials) | | | | |")
            continue
        drift = "" if r["drifted"] is None else ("yes" if r["drifted"] else "no")
        lines.append(f"| {r['experiment']} | {r['condition']} | {r['blocks']} | "
                     f"{_fmt(r['f1'])} | {_fmt(r['f1_vs_reference'])} | "
                     f"{_fmt(r['agreement'], 'pct')} | "
                     f"{_fmt(r['fallbacks'])} | {drift} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dirs", nargs="+", help="experiment results folders")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    text = to_markdown(collect(args.dirs))
    print(text, end="")
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
