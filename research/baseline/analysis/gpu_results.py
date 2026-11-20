"""Analyse the Session 1 GPU runs (E0-E3) downloaded from the Modal volume.

    modal volume get blockkalypso-results research-session1 ./research/baseline/results/gpu/
    python -m research.baseline.analysis.gpu_results research/baseline/results/gpu/research-session1 \
        --out research/baseline/results/gpu_analysis

Per experiment folder (see research/baseline/gpu/modal_baseline.py for the
layout) it reads ``status.json``, ``env.json``, every ``<part>/run-*.jsonl``
(eval.run output; only measured repeats are written there, warm-ups are not)
and, when present, the telemetry trace.

Per arm: latency median [min, max] and a 95% bootstrap CI of the median,
LLM requests, quality (accuracy/precision/recall/F1 against labels, via
eval/report.py), achieved selectivity (selected / items), peak KV %, peak
waiting, preemptions. Against the per-tuple baseline: per-repeat paired
latency ratio, whether all repeats agree in direction.

E0: trace off vs on, per arm. E3: arm x load level. Traces: per stage
cardinality, block fill, forced launches, KV admission counters, fallbacks,
truncations (research/baseline/analysis/trace_summary.py).

Outputs ``arms.jsonl``, ``summary.json`` and ``summary.md`` in ``--out``.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
from collections import defaultdict
from pathlib import Path

from research.baseline.analysis.trace_summary import load as load_trace
from research.baseline.analysis.trace_summary import summarise
from vllm.kalypso.benchmark.eval import report

BASELINE = "kalypso_per_tuple"


def boot_ci(values, n=10_000, seed=0):
    if len(values) < 2:
        return None
    rng = random.Random(seed)
    meds = sorted(statistics.median(rng.choices(values, k=len(values))) for _ in range(n))
    return [meds[int(0.025 * n)], meds[int(0.975 * n) - 1]]


def _stat(values):
    v = [x for x in values if x is not None]
    if not v:
        return None
    return {"median": statistics.median(v), "min": min(v), "max": max(v), "n": len(v)}


def arm_stats(meta, trials):
    groups = defaultdict(list)
    errors = defaultdict(list)
    for t in trials:
        (errors if "error" in t else groups)[t["arm"]].append(t)
    items = len(report.universe(meta)) if meta.get("rows") else None
    out = {}
    for arm, group in groups.items():
        rows = [{**t, **report.quality(meta, t)} for t in group]
        lat = [r["latency_s"] for r in rows]
        sel = ([len(r["selected"]) / items for r in rows if r.get("selected") is not None]
               if items else [])
        out[arm] = {
            "repeats": len(rows), "errors": len(errors.get(arm, [])),
            "latency_s": _stat(lat), "latency_ci95": boot_ci(lat),
            "latency_by_repeat": {r.get("repeat"): r["latency_s"] for r in rows},
            "llm_requests": _stat([r.get("llm_requests") for r in rows]),
            "fallback_calls": _stat([r.get("fallback_calls") for r in rows]),
            "prompt_tokens": _stat([r.get("prompt_tokens") for r in rows]),
            "generation_tokens": _stat([r.get("generation_tokens") for r in rows]),
            "prefix_cache_hit_rate": _stat([r.get("prefix_cache_hit_rate") for r in rows]),
            "peak_kv_cache_pct": _stat([r.get("peak_kv_cache_pct") for r in rows]),
            "peak_waiting": _stat([r.get("peak_waiting") for r in rows]),
            "peak_running": _stat([r.get("peak_running") for r in rows]),
            "preemptions": _stat([r.get("preemptions") for r in rows]),
            "truncated": _stat([r.get("truncated") for r in rows]),
            "rows_out": _stat([r.get("rows_out") for r in rows]),
            "achieved_selectivity": _stat(sel),
            "f1": _stat([r.get("f1") for r in rows]),
            "precision": _stat([r.get("precision") for r in rows]),
            "recall": _stat([r.get("recall") for r in rows]),
            "accuracy": _stat([r.get("accuracy") for r in rows]),
        }
    base = out.get(BASELINE)
    for arm, s in out.items():
        if base and arm != BASELINE:
            pairs = [(s["latency_by_repeat"][k], base["latency_by_repeat"][k])
                     for k in s["latency_by_repeat"] if k in base["latency_by_repeat"]]
            ratios = [a / b for a, b in pairs if b]
            s["speedup_vs_baseline"] = _stat([1 / r for r in ratios])
            s["all_repeats_faster"] = bool(ratios) and all(r < 1 for r in ratios)
            s["all_repeats_slower"] = bool(ratios) and all(r > 1 for r in ratios)
    return out


def load_run(path: Path):
    meta, trials = report.load(path)
    return meta, trials


def trace_stats(path: Path, arms_order: list[str] | None, warmup: int, repeats: int):
    events = load_trace(path)
    queries = summarise(events)
    agg = {"queries": len(queries), "complete": sum(q["complete"] for q in queries)}
    counters = defaultdict(int)
    for q in queries:
        for k, v in (q.get("counters") or {}).items():
            counters[k] += v
    agg["counters"] = dict(counters)
    per_stage = defaultdict(lambda: defaultdict(list))
    for q in queries:
        for s in q["stages"]:
            key = f"stage{s['stage_id']}:k{s['block_size']}"
            for f in ("out_per_in", "partial_blocks", "tasks", "forced_launches", "retries",
                      "wait_s_p95", "exec_s_p50", "truncated_calls", "block_fallback_calls",
                      "block_undecided_rows", "cached_tokens", "prompt_tokens"):
                if s.get(f) is not None:
                    per_stage[key][f].append(s[f])
    agg["stages"] = {k: {f: _stat(v) for f, v in d.items()} for k, d in per_stage.items()}
    agg["plan_s"] = _stat([q["plan_s"] for q in queries])
    # Queries run one per trial, in eval.run's order (warm-ups, then repeats,
    # arm by arm). Attribute them to arms only when the counts match exactly.
    if arms_order and len(queries) == len(arms_order) * (warmup + repeats):
        by_arm = defaultdict(list)
        for i, q in enumerate(queries):
            if i >= len(arms_order) * warmup:
                by_arm[arms_order[i % len(arms_order)]].append(q)
        agg["by_arm"] = {
            arm: {"latency_s_trace": _stat([q["latency_s"] for q in qs]),
                  "stage_out_per_in": {
                      f"stage{s['stage_id']}": _stat([x["out_per_in"] for qq in qs
                                                      for x in qq["stages"]
                                                      if x["stage_id"] == s["stage_id"]])
                      for s in qs[0]["stages"]} if qs and qs[0]["stages"] else {}}
            for arm, qs in by_arm.items()}
        agg["attribution"] = "by order (counts match)"
    else:
        agg["attribution"] = (f"not attributed: {len(queries)} queries vs "
                              f"{len(arms_order or [])} arms x {warmup + repeats} trials "
                              "(each arm may issue more than one query)")
    return agg


def analyse_experiment(root: Path) -> dict:
    exp = {"id": root.name}
    for name in ("status.json", "env.json", "plan.json"):
        p = root / name
        if p.exists():
            exp[name.split(".")[0]] = json.loads(p.read_text())
    exp["parts"] = {}
    for part_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        runs = sorted(part_dir.glob("run-*.jsonl"))
        if not runs:
            continue
        part = {}
        load_runs = defaultdict(list)
        for run in runs:
            m = re.match(r"run-L(\d+)-r(\d+)\.jsonl", run.name)
            meta, trials = load_run(run)
            if m:
                level, rep = int(m.group(1)), int(m.group(2))
                for t in trials:
                    t["repeat"] = rep
                load_runs[level].append((meta, trials))
            else:
                tag = run.stem.split("-", 1)[1]
                part[tag] = {"meta": {k: meta.get(k) for k in
                                      ("workload", "rows", "right_rows", "kind", "arms",
                                       "warmup", "repeats", "git_sha", "git_dirty")},
                             "arms": arm_stats(meta, trials)}
        for level, lst in sorted(load_runs.items()):
            meta = lst[0][0]
            trials = [t for _, ts in lst for t in ts]
            part[f"L{level}"] = {"meta": {"workload": meta.get("workload"),
                                          "load_streams": level, "repeats": len(lst)},
                                 "arms": arm_stats(meta, trials)}
        exp["parts"][part_dir.name] = part
    exp["traces"] = {}
    plan = exp.get("plan") or {}
    for trace in sorted([*root.glob("trace-*.jsonl"), *root.glob("trace-*.jsonl.gz")]):
        arms_order = None
        if len(plan.get("parts", [])) == 1 and not plan.get("load_levels"):
            arms_order = plan["parts"][0]["arms"]
        exp["traces"][trace.name.split(".")[0]] = trace_stats(trace, arms_order, 1, 5)
    exp["vllm_metrics"] = {}
    for mf in sorted(root.glob("vllm-metrics-*.jsonl")):
        rows = [json.loads(l) for l in mf.read_text().splitlines() if l.strip()]
        exp["vllm_metrics"][mf.stem] = {
            "samples": len(rows),
            "kv_cache_pct": _stat([r["kv_cache_pct"] for r in rows]),
            "waiting": _stat([r["waiting"] for r in rows]),
            "preemptions_total": (rows[-1]["preemptions"] - rows[0]["preemptions"])
            if rows and rows[-1].get("preemptions") is not None
            and rows[0].get("preemptions") is not None else None}
    return exp


def _f(s, key="median", fmt="{:.2f}"):
    if not s or s.get(key) is None:
        return "-"
    return fmt.format(s[key])


def render(exps: list[dict]) -> str:
    lines = ["# Session 1 GPU results (E0-E3)", "",
             "L40S, Qwen2.5-7B-Instruct; 1 warm-up + 5 repeats per arm unless noted. "
             "Latency: median [min, max] over repeats; CI: 95% bootstrap of the median. "
             "Speedup: per-repeat paired baseline/arm, median; ✓ = faster in every repeat.", ""]
    for e in exps:
        st = e.get("status", {})
        lines += [f"## {e['id']}", "",
                  f"status: {st.get('parts')} {('error: ' + st['error']) if st.get('error') else ''}",
                  ""]
        for part, tags in e["parts"].items():
            for tag, d in tags.items():
                lines += [f"### {part} ({tag})", "",
                          "| arm | n | latency s | CI95 | speedup | LLM req | F1 | achieved sel | rows out | peak KV % | peak waiting | preempt |",
                          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
                for arm, s in d["arms"].items():
                    lat = s["latency_s"]
                    ci = s["latency_ci95"]
                    sp = s.get("speedup_vs_baseline")
                    mark = " ✓" if s.get("all_repeats_faster") else (" ✗" if s.get("all_repeats_slower") else "")
                    lines.append(
                        f"| {arm} | {s['repeats']}{' (' + str(s['errors']) + ' err)' if s['errors'] else ''} | "
                        f"{_f(lat)} [{_f(lat, 'min')}, {_f(lat, 'max')}] | "
                        f"{'[%.2f, %.2f]' % tuple(ci) if ci else '-'} | "
                        f"{(_f(sp) + 'x' + mark) if sp else '-'} | {_f(s['llm_requests'], fmt='{:.0f}')} | "
                        f"{_f(s['f1'])} | {_f(s['achieved_selectivity'], fmt='{:.3f}')} | "
                        f"{_f(s['rows_out'], fmt='{:.0f}')} | {_f(s['peak_kv_cache_pct'], fmt='{:.1f}')} | "
                        f"{_f(s['peak_waiting'], fmt='{:.0f}')} | {_f(s['preemptions'], 'max', '{:.0f}')} |")
                lines.append("")
        for name, t in e["traces"].items():
            lines += [f"### Trace {name}", "",
                      f"{t['queries']} queries ({t['complete']} complete); KV admission counters "
                      f"{t['counters'] or 'none'}; plan time median {_f(t['plan_s'], fmt='{:.4f}')} s; "
                      f"{t['attribution']}.", "",
                      "| stage:k | out/in median [min,max] | partial blocks max | forced launches max | retries max | wait p95 s median | truncated max | block fallbacks max |",
                      "|---|---|---|---|---|---|---|---|"]
            for k, s in sorted(t["stages"].items()):
                lines.append(
                    f"| {k} | {_f(s.get('out_per_in'), fmt='{:.3f}')} [{_f(s.get('out_per_in'), 'min', '{:.3f}')}, "
                    f"{_f(s.get('out_per_in'), 'max', '{:.3f}')}] | {_f(s.get('partial_blocks'), 'max', '{:.0f}')} | "
                    f"{_f(s.get('forced_launches'), 'max', '{:.0f}')} | {_f(s.get('retries'), 'max', '{:.0f}')} | "
                    f"{_f(s.get('wait_s_p95'), fmt='{:.3f}')} | {_f(s.get('truncated_calls'), 'max', '{:.0f}')} | "
                    f"{_f(s.get('block_fallback_calls'), 'max', '{:.0f}')} |")
            lines.append("")
        for name, m in e["vllm_metrics"].items():
            lines.append(f"- {name}: {m['samples']} samples, KV % max {_f(m['kv_cache_pct'], 'max', '{:.1f}')}, "
                         f"waiting max {_f(m['waiting'], 'max', '{:.0f}')}, preemptions {m['preemptions_total']}")
        lines.append("")
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("root", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    exps = [analyse_experiment(p) for p in sorted(args.root.iterdir()) if p.is_dir()]
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(exps, indent=1, default=str))
    (args.out / "summary.md").write_text(render(exps))
    print(f"wrote {args.out} ({len(exps)} experiments)")


if __name__ == "__main__":
    main()
