"""Do pipeline stages overlap in time? Measured from KALYPSO_TRACE task events.

For every query with two or more stages, from ``task_start`` / ``task_end``
(``t_mono``, one process, so comparable):

- ``s{i}_span``: first task_start to last task_end of stage i.
- ``gap_s``: stage 2's first task_start minus stage 1's last task_end.
  Negative = stage 2 started while stage 1 was still running.
- ``s2_started_early``: share of stage-2 tasks that started before stage 1's
  last task ended.
- ``both_running_frac``: share of the query's latency during which at least one
  task of each stage was running.
- ``s2_exec_overlapped``: share of stage 2's summed task time (exec_s) that
  fell while stage 1 still had a task running.
- ``additive_ratio``: (s1_span + s2_span) / latency. About 1 if the stages
  run one after the other, above 1 if they overlap.

    python -m research.baseline.analysis.stage_overlap \
        research/baseline/results/gpu/research-session1/E1-s*/trace-on.jsonl.gz
"""

import argparse
import gzip
import json
import statistics
from collections import defaultdict
from pathlib import Path


def _stages(plan):
    out = []
    for item in plan:
        for s in item.get("segment", []) if isinstance(item, dict) else []:
            out.append(s)
    return out


def _arm(stages):
    return " -> ".join(
        f"{'+'.join(s['ops'])}(k={s['block_size']}{', fanout' if s.get('fanout') else ''})"
        for s in stages)


def _union_len(intervals):
    total, cur_s, cur_e = 0.0, None, None
    for s, e in sorted(intervals):
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        total += cur_e - cur_s
    return total


def _intersect(a, b):
    """Total length where some interval of `a` and some of `b` are both active."""
    def merged(iv):
        out = []
        for s, e in sorted(iv):
            if out and s <= out[-1][1]:
                out[-1][1] = max(out[-1][1], e)
            else:
                out.append([s, e])
        return out
    ma, mb, i, j, total = merged(a), merged(b), 0, 0, 0.0
    while i < len(ma) and j < len(mb):
        lo, hi = max(ma[i][0], mb[j][0]), min(ma[i][1], mb[j][1])
        if hi > lo:
            total += hi - lo
        if ma[i][1] < mb[j][1]:
            i += 1
        else:
            j += 1
    return total


def queries(path):
    q = defaultdict(lambda: {"starts": {}, "ends": {}, "stage": {}})
    with gzip.open(path, "rt") as f:
        for line in f:
            e = json.loads(line)
            qid = e.get("query_id")
            if qid is None:
                continue
            ev, rec = e["event"], q[qid]
            if ev == "plan_built":
                rec["stages"] = _stages(e["plan"])
            elif ev == "task_start":
                rec["starts"][e["task_id"]] = e["t_mono"]
                rec["stage"][e["task_id"]] = e["stage_id"]
            elif ev == "task_end":
                rec["ends"][e["task_id"]] = e["t_mono"]
            elif ev == "query_start":
                rec["t0"] = e["t_mono"]
            elif ev == "query_end":
                rec["t1"], rec["latency"] = e["t_mono"], e["latency_s"]
    return q


def measure(rec):
    stages = rec.get("stages") or []
    if len(stages) < 2 or "latency" not in rec:
        return None
    iv = defaultdict(list)
    for tid, s in rec["starts"].items():
        if tid in rec["ends"]:
            iv[rec["stage"][tid]].append((s, rec["ends"][tid]))
    ids = [s["stage_id"] for s in stages]
    if not all(iv.get(i) for i in ids[:2]):
        return None
    a, b = iv[ids[0]], iv[ids[1]]
    s1_first, s1_last = min(s for s, _ in a), max(e for _, e in a)
    s2_first, s2_last = min(s for s, _ in b), max(e for _, e in b)
    lat = rec["latency"]
    s2_exec = sum(e - s for s, e in b)
    s2_exec_ov = sum(max(0.0, min(e, s1_last) - s) for s, e in b)
    return {
        "arm": _arm(stages),
        "latency": lat,
        "s1_tasks": len(a), "s2_tasks": len(b),
        "s1_span": s1_last - s1_first,
        "s2_span": s2_last - s2_first,
        "gap_s": s2_first - s1_last,
        "s2_started_early": sum(s < s1_last for s, _ in b) / len(b),
        "both_running_frac": _intersect(a, b) / lat,
        "s2_exec_overlapped": s2_exec_ov / s2_exec if s2_exec else 0.0,
        "additive_ratio": ((s1_last - s1_first) + (s2_last - s2_first)) / lat,
        "tail_after_s1": (s2_last - s1_last) / lat,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traces", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, help="per-query JSONL")
    args = ap.parse_args()
    rows = []
    for p in args.traces:
        for qid, rec in queries(p).items():
            m = measure(rec)
            if m:
                rows.append({"trace": str(p), "query_id": qid, **m})
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    by_arm = defaultdict(list)
    for r in rows:
        by_arm[r["arm"]].append(r)
    keys = ["latency", "gap_s", "s2_started_early", "both_running_frac",
            "s2_exec_overlapped", "additive_ratio", "tail_after_s1"]
    print("| arm | n | " + " | ".join(keys) + " |")
    print("|---|---|" + "---|" * len(keys))
    for arm, rs in sorted(by_arm.items()):
        cells = [f"{statistics.median(r[k] for r in rs):.2f}" for k in keys]
        print(f"| {arm} | {len(rs)} | " + " | ".join(cells) + " |")
    print(f"\nall multi-stage queries: {len(rows)}; "
          f"stage 2 started before stage 1 finished in "
          f"{sum(r['gap_s'] < 0 for r in rows)} of them")


if __name__ == "__main__":
    main()
