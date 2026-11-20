"""Did the Phase 2 functional run work end to end? (read-only, no metrics)

Reads one run directory of research/phase2/gpu/modal_phase2.py and reports
counts only. Was every query that started also finished? Did the adaptive
controller observe, veto and switch? Did each arm produce outputs? What did
the optimizer step choose, with the KV check and the automatic cache? No
latency, throughput or quality numbers: this is a functional check, not an
evaluation.

    python -m research.phase2.functional_summary research/phase2/results/P2-functional
"""

from __future__ import annotations

import gzip
import json
import sys
from collections import Counter
from pathlib import Path


def _lines(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def summarize(root: Path) -> dict:
    root = Path(root)
    out: dict = {"run": root.name}
    traces = sorted(root.glob("trace-*.jsonl*"))
    events: Counter = Counter()
    started, ended = set(), set()
    decisions: Counter = Counter()
    switches = []
    for trace in traces:
        for r in _lines(trace):
            events[r["event"]] += 1
            if r["event"] == "query_start":
                started.add(r.get("query_id"))
            elif r["event"] == "query_end":
                ended.add(r.get("query_id"))
            elif r["event"] == "adaptive_switch":
                decisions[(r.get("action"), r.get("mode"), r.get("size"),
                           r.get("reason"), tuple(r.get("vetoed") or ()))] += 1
                if len(switches) < 20:
                    switches.append({k: r.get(k) for k in (
                        "query_id", "stage_id", "action", "mode", "size", "reason",
                        "vetoed", "from_mode", "pass_rate", "after_formations")})
    out["trace_files"] = [t.name for t in traces]
    out["events"] = dict(events)
    out["queries_started"] = len(started)
    out["queries_finished"] = len(ended & started)
    out["queries_unfinished"] = sorted(q for q in started - ended if q is not None)[:20]
    out["controller_decisions"] = [
        {"action": a, "mode": m, "size": s, "reason": why, "vetoed": list(v), "count": n}
        for (a, m, s, why, v), n in sorted(decisions.items(), key=lambda x: -x[1])]
    out["controller_decision_examples"] = switches

    arms: dict = {}
    for run in sorted(root.glob("*/run-*.jsonl")):
        for r in _lines(run):
            if r.get("type") == "meta":
                continue
            key = f"{run.parent.name} {r.get('arm')}"
            a = arms.setdefault(key, {"trials": 0, "with_outputs": 0, "errors": 0})
            a["trials"] += 1
            a["with_outputs"] += bool(r.get("outputs"))
            a["errors"] += bool(r.get("error"))
    out["arms"] = arms
    opt = root / "optimizer.json"
    if opt.exists():
        out["optimizer"] = json.loads(opt.read_text())
    status = root / "status.json"
    if status.exists():
        out["status"] = json.loads(status.read_text())
    return out


def main(argv=None) -> None:
    argv = argv if argv is not None else sys.argv[1:]
    print(json.dumps(summarize(Path(argv[0])), indent=2, default=str))


if __name__ == "__main__":
    main()
