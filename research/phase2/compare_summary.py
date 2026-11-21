"""Static vs adaptive controller (P2-compare): measured numbers per arm.

For each part and configuration (static = controller off, adaptive = on),
over the measured repeats of every phase, it reports:

- end-to-end latency (eval.run's ``latency_s``);
- F1 against the workload's labels (``report.quality``, the metric every
  filter/join report already uses);
- mean, min and max for both.

It also lists every controller decision in the adaptive phases' traces
(switch, veto with the sizes vetoed, step-down) per query, warmups included:
the trace does not mark which query was a warmup. It also counts decisions
in the static phases, which must be 0. Per adaptive query it also reports
the controller's ``adaptive_summary``: deferrals, veto checks, and per
managed stage its own finished tasks, whether its window filled (a real
selectivity reading) and how many upstream votes came after that. Mean and
spread only, no statistics.

    python -m research.phase2.compare_summary research/phase2/results/P2-compare
"""

from __future__ import annotations

import gzip
import json
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

from vllm.kalypso.benchmark.eval import report


def _lines(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def _stats(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    return {"mean": round(st.mean(values), 4), "min": round(min(values), 4),
            "max": round(max(values), 4), "n": len(values)}


def summarize(root: Path) -> dict:
    root = Path(root)
    plan = json.loads((root / "plan.json").read_text())
    adaptive_of = {tag: on for tag, on in plan["phases"]}
    out = {"parts": {}}
    data_path_to_part = {}
    for part in plan["parts"]:
        per = {"static": defaultdict(list), "adaptive": defaultdict(list)}
        for tag, on in plan["phases"]:
            path = root / part["name"] / f"run-{tag}.jsonl"
            if not path.exists():
                continue
            meta, trials = report.load(path)
            data_path_to_part[str(meta["data_path"])] = part["name"]
            cfg = "adaptive" if on else "static"
            for t in trials:
                if "error" in t:
                    per[cfg]["errors"].append(t["error"])
                    continue
                per[cfg]["latency_s"].append(t["latency_s"])
                per[cfg]["f1"].append(report.quality(meta, t).get("f1"))
                per[cfg]["llm_requests"].append(t.get("llm_requests"))
        out["parts"][part["name"]] = {
            "arm": part["arm"],
            **{cfg: {"latency_s": _stats(v["latency_s"]), "f1": _stats(v["f1"]),
                     "llm_requests": _stats(v["llm_requests"]), "errors": len(v["errors"])}
               for cfg, v in per.items()}}

    decisions = defaultdict(list)
    summaries = defaultdict(list)
    counts = {"static": Counter(), "adaptive": Counter()}
    queries = {"static": Counter(), "adaptive": Counter()}
    for trace in sorted(root.glob("trace-*.jsonl*")):
        tag = trace.name.split(".")[0].removeprefix("trace-")
        cfg = "adaptive" if adaptive_of.get(tag) else "static"
        part_of = {}
        ends = defaultdict(list)  # query -> [(stage_id, t_mono)] of own task ends
        for r in _lines(trace):
            if r["event"] == "task_end" and r.get("outcome") != "retry":
                ends[r.get("query_id")].append((r.get("stage_id"), r.get("t_mono")))
            if r["event"] == "adaptive_summary":
                q = r.get("query_id")
                entry = {"phase": tag, "query_id": q, "deferrals": r.get("deferrals"),
                         "veto_checks": r.get("veto_checks"), "stages": r.get("stages")}
                for sid, st_ in (r.get("stages") or {}).items():
                    # Was there an upstream vote left after the window filled?
                    own = [t for s, t in ends[q] if str(s) == sid]
                    up = [t for s, t in ends[q] if str(s) == str(int(sid) - 1)]
                    if len(own) >= 16:
                        filled = sorted(own)[15]
                        st_["upstream_votes_after_reading"] = sum(t > filled for t in up)
                    else:
                        st_["upstream_votes_after_reading"] = None
                summaries[part_of.get(q, "?")].append(entry)
            if r["event"] == "query_start":
                part_of[r["query_id"]] = data_path_to_part.get(str(r.get("data_path")), "?")
                queries[cfg][part_of[r["query_id"]]] += 1
            elif r["event"] == "adaptive_switch":
                part = part_of.get(r.get("query_id"), "?")
                counts[cfg][(part, r["action"], r["mode"], r.get("size"),
                             tuple(r.get("vetoed") or ()))] += 1
                if cfg == "adaptive":
                    decisions[part].append({k: r.get(k) for k in (
                        "query_id", "stage_id", "action", "mode", "size", "from_mode",
                        "reason", "vetoed", "pass_rate", "after_formations")} | {"phase": tag})
    out["queries_per_config"] = {c: dict(q) for c, q in queries.items()}
    out["decision_counts"] = {
        cfg: [{"part": p, "action": a, "mode": m, "size": s, "vetoed": list(v), "count": n}
              for (p, a, m, s, v), n in sorted(c.items())]
        for cfg, c in counts.items()}
    out["adaptive_decisions"] = dict(decisions)
    out["adaptive_summaries"] = dict(summaries)
    return out


def main(argv=None) -> None:
    argv = argv if argv is not None else sys.argv[1:]
    print(json.dumps(summarize(Path(argv[0])), indent=2, default=str))


if __name__ == "__main__":
    main()
