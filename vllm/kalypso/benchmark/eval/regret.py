"""Static vs dynamic block size, exactly as the Milestone 2 plan defines it.

    python -m vllm.kalypso.benchmark.eval.regret RESULTS [RESULTS ...]

Each RESULTS is a results file (one condition: a row length, N, s or f) or
a results folder (every .jsonl in it is a condition). For every condition c,
k*(c) is the arm with the lowest median completion time. For each arm,

    regret(c) = latency(arm, c) / latency(k*(c), c)

and k_static is the one arm that does best across all conditions: the
lowest worst-case regret (ties broken by mean regret). If its regret stays
at or below ``THRESHOLD`` everywhere, a static k is enough; otherwise the
conditions where it is exceeded show where k must adapt (and which signals
a Level 4/5 controller should watch).

Only arms that every condition ran are compared; an arm that failed in a
condition has infinite regret there. Latency only, as the plan specifies:
an arm whose pipeline work drifted from per-tuple (report.drift) is kept and
marked, since its time reflects different work.
"""

import argparse
import math
import statistics
from pathlib import Path

from . import report

THRESHOLD = 1.1
REFERENCE = "kalypso_per_tuple"


def _condition(path: Path) -> dict:
    meta, trials = report.load(path)
    latencies = {arm: statistics.median(t["latency_s"] for t in group)
                 for arm, group in report.by_arm(trials).items()}
    drifted = set()
    if meta.get("kind") == "pipeline":
        drifted = {arm for arm, row in report.drift(meta, trials, REFERENCE).items()
                   if row["drifted"]}
    best = min(latencies, key=latencies.get) if latencies else None
    params = meta.get("params") or {}
    label = f"{path.parent.name}/{path.stem}" + (
        " (" + ", ".join(f"{k}={v}" for k, v in params.items()) + ")" if params else "")
    return {"label": label, "arms": list(meta["arms"]), "latency": latencies,
            "drifted": drifted, "best": best}


def condition_files(paths) -> list[Path]:
    files = []
    for p in map(Path, paths):
        files += sorted(p.glob("*.jsonl")) if p.is_dir() else [p]
    return files


def analyze(files) -> dict:
    conditions = [_condition(Path(f)) for f in files]
    if not conditions:
        raise ValueError("no results files given")
    common = [a for a in conditions[0]["arms"] if all(a in c["arms"] for c in conditions)]
    regrets = {}
    for arm in common:
        regrets[arm] = {}
        for c in conditions:
            if c["best"] is None or arm not in c["latency"]:
                regrets[arm][c["label"]] = math.inf          # failed here
            else:
                regrets[arm][c["label"]] = c["latency"][arm] / c["latency"][c["best"]]
    worst = {a: max(r.values()) for a, r in regrets.items()}
    mean = {a: statistics.mean(r.values()) if all(map(math.isfinite, r.values())) else math.inf
            for a, r in regrets.items()}
    static = min(regrets, key=lambda a: (worst[a], mean[a])) if regrets else None
    exceeded = ([c["label"] for c in conditions if regrets[static][c["label"]] > THRESHOLD]
                if static else [])
    return {"conditions": conditions, "regrets": regrets, "worst": worst, "mean": mean,
            "static": static, "exceeded": exceeded}


def _num(v):
    return "failed" if math.isinf(v) else f"{v:.2f}"


def to_markdown(result: dict) -> str:
    labels = [c["label"] for c in result["conditions"]]
    lines = ["# Static vs dynamic block size", "",
             "regret(c) = latency(arm, c) / latency(k*(c), c); k_static = the arm with the "
             f"lowest worst-case regret; threshold {THRESHOLD}. `*` = the arm's pipeline work "
             "drifted from per-tuple in that condition.", "",
             "| Condition | k*(c) | Latency (s) |", "|---|---|---|"]
    for c in result["conditions"]:
        mark = " *" if c["best"] in c["drifted"] else ""
        lines.append(f"| {c['label']} | `{c['best']}`{mark} | {c['latency'][c['best']]:.2f} |")
    lines += ["", "| Arm | " + " | ".join(labels) + " | Worst | Mean |",
              "|" + "---|" * (len(labels) + 3)]
    for arm, row in result["regrets"].items():
        cells = []
        for c in result["conditions"]:
            cells.append(_num(row[c["label"]]) + (" *" if arm in c["drifted"] else ""))
        lines.append(f"| `{arm}` | " + " | ".join(cells)
                     + f" | {_num(result['worst'][arm])} | {_num(result['mean'][arm])} |")
    static = result["static"]
    lines += ["", f"k_static: `{static}` (worst-case regret {_num(result['worst'][static])}, "
              f"mean {_num(result['mean'][static])})."]
    if result["exceeded"]:
        lines.append(f"Verdict: dynamic k / reforming is needed; regret exceeds {THRESHOLD} "
                     "for: " + ", ".join(result["exceeded"]) + ".")
    else:
        lines.append(f"Verdict: a static k is enough (regret <= {THRESHOLD} in every condition).")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("results", nargs="+", help="results files or folders")
    parser.add_argument("--out", default=None, help="also write the markdown here")
    args = parser.parse_args()
    text = to_markdown(analyze(condition_files(args.results)))
    print(text, end="")
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
