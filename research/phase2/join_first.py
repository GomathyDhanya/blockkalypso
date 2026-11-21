"""Join-first composition error (Phase 2, item 4): where does the held-out
join -> filter over-prediction come from?

    python -m research.phase2.join_first --out research/phase2/results/join_first

Read-only over committed results: no GPU run and no cost-model change. The
model is refitted as costmodel_validation does (research/baseline/analysis/).
It covers the four held-out validate-jf workloads (pipe_join_filter_*,
join -> filter; the catalog has no join -> filter -> join plan) and tests
three candidate causes:

1. **Cardinality.** The model sizes the filter from the join's predicted
   pass share. ``costmodel.passed_rows`` returns None for join-first plans,
   so COST_MODEL_VALIDATION's "given cardinality" prediction never used the
   real cardinality there: it equals the full prediction. Here the filter
   stage is re-predicted from the pairs the join really passed, which is
   (llm_requests - the join stage's calls) x k2, an upper bound that is off
   by less than one block. That is cross-checked against the join run alone
   (validate-jf/*/alone-join, rows_out).
2. **Standalone-fit vs in-pipeline cost of the join.** The model's join-stage
   latency is compared with the same join measured alone (alone-join, k = 1,
   2, 4, 6).
3. **Stage overlap.** step 01's ``stage_overlap.measure`` is applied to the
   only join-first traces that exist: pipe_join_filter_j62_f90 in the Phase 2
   Modal runs, with the controller-off phases only.
"""

from __future__ import annotations

import argparse
import gzip
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

from vllm.kalypso.benchmark.eval import analyze, report
from vllm.kalypso.benchmark.eval import costmodel as cm

from research.baseline.analysis import stage_overlap
from research.baseline.analysis.costmodel_validation import fit_like_original

JF = cm.RESULTS / "experiments/validate-jf"
TRACES = {  # (trace, controller on?) for the join-first traced queries
    "P2-compare/trace-off-1": False, "P2-compare/trace-off-2": False,
    "P2-compare-v2/trace-off-1": False, "P2-compare-v2/trace-off-2": False,
    # Controller on: any query with an adaptive_switch event is dropped.
    "P2-functional/trace-on": True,
}
PHASE2_RESULTS = Path(__file__).resolve().parent / "results"


def _ape(pred, actual):
    return abs(pred - actual) / actual


def _signed(pred, actual):
    return (pred - actual) / actual


def arm_records(model, everything, cache):
    prows = cm.attach_tokens(cm._pipeline_rows(everything), cache)
    out = []
    for source in sorted({r["source"] for r in prows if r["phase"] == "validate-jf"}):
        run = [r for r in prows if r["source"] == source]
        base = run[0]
        pipe = cm.pipeline_of(
            base, cm.workload_selectivity(base["workload"], cm._n_arg(base), cache),
            cm.workload_stage_selectivity(base["workload"], cm._n_arg(base), cache))
        assert pipe.ops[:2] == ("join", "filter"), pipe.ops
        items = model.stage_items(pipe, pipe.rows)
        for r in run:
            k1, k2 = r["blocks_t"]
            full = model.predict_pipeline(pipe, r["blocks_t"], r["max_num_seqs"])
            s1, s2 = full["stages"]
            filter_calls = max(r["llm_requests"] - s1["calls"], 0)
            passed = min(filter_calls * k2, items[0].n)
            s2_true = model.predict("filter", cm.Items(passed, items[1].tokens,
                                                       items[1].max_tokens), k2,
                                    r["max_num_seqs"]) if passed else {"latency_s": 0.0}
            true_card = model.lam * (s1["latency_s"] + s2_true["latency_s"])
            out.append({
                "workload": r["workload"], "arm": r["arm"], "k1": k1, "k2": k2,
                "latency_s": r["latency_s"], "pred_s": full["latency_s"],
                "pred_true_card_s": true_card,
                "join_pred_s": s1["latency_s"], "filter_pred_s": s2["latency_s"],
                "filter_pred_true_card_s": s2_true["latency_s"],
                "join_passed_pred": items[0].n * model.kinds["join"].pass_share(
                    items[0].regime, k1, pipe.true_share(0)),
                "join_passed_actual": passed, "join_calls_pred": s1["calls"],
                "calls": r["llm_requests"], "pred_calls": full["calls"]})
    return out


def alone_join():
    """The join run alone (validate-jf/*/alone-join) vs the model's join stage."""
    out = []
    for path in sorted(JF.glob("*/alone-join/run.jsonl")):
        meta, trials = report.load(path)
        by_arm = defaultdict(list)
        for t in trials:
            if "error" not in t:
                by_arm[t["arm"]].append(t)
        pipeline_name = meta["workload"].removesuffix("_join")
        for arm, ts in by_arm.items():
            k = 1 if arm == "kalypso_per_tuple" else int(arm.split(":")[1])
            out.append({"workload": pipeline_name, "k": k,
                        "latency_s": st.median(t["latency_s"] for t in ts),
                        "rows_out": st.median(t["rows_out"] for t in ts),
                        "true_pairs": len(meta["positives"])})
    return out


def join_stage_predictions(model, everything, cache, alone):
    """Model join-stage latency and kept pairs at each alone-join k."""
    prows = cm.attach_tokens(cm._pipeline_rows(everything), cache)
    pipes = {}
    for r in prows:
        if r["phase"] == "validate-jf" and r["workload"] not in pipes:
            pipes[r["workload"]] = (cm.pipeline_of(
                r, cm.workload_selectivity(r["workload"], cm._n_arg(r), cache),
                cm.workload_stage_selectivity(r["workload"], cm._n_arg(r), cache)),
                r["max_num_seqs"])
    for a in alone:
        pipe, seqs = pipes[a["workload"]]
        items = model.stage_items(pipe, pipe.rows)[0]
        pred = model.predict("join", items, a["k"], seqs)
        a["join_pred_s"] = pred["latency_s"]
        a["kept_pred"] = items.n * model.kinds["join"].pass_share(items.regime, a["k"],
                                                                 pipe.true_share(0))
    return alone


def overlap(root: Path = PHASE2_RESULTS):
    rows = []
    for name in TRACES:
        path = root / f"{name}.jsonl.gz"
        if not path.exists():
            continue
        with gzip.open(path, "rt") as f:
            switched = {e.get("query_id") for e in map(json.loads, f)
                        if e["event"] == "adaptive_switch"}
        for qid, rec in stage_overlap.queries(path).items():
            if qid in switched:
                continue
            stages = rec.get("stages") or []
            if not stages or not any("Join" in op for op in stages[0]["ops"]):
                continue
            m = stage_overlap.measure(rec)
            if m:
                rows.append({"trace": name, "controller_on": TRACES[name],
                             "query_id": qid, **m})
    return rows


def _summ(vals):
    vals = list(vals)
    return {"median": round(st.median(vals), 3), "min": round(min(vals), 3),
            "max": round(max(vals), 3), "n": len(vals)} if vals else None


def summarize(arms, alone, ov):
    out = {"arms": {}, "by_k1": {}, "alone_join": alone, "overlap": {}}
    for label, f in (("all", lambda r: True),
                     ("k1=1", lambda r: r["k1"] == 1),
                     ("k1 in 2..8", lambda r: 2 <= r["k1"] <= 8),
                     ("k1>=9", lambda r: r["k1"] >= 9)):
        a = [r for r in arms if f(r)]
        if not a:
            continue
        out["by_k1"][label] = {
            "n": len(a),
            "mape_full": round(st.mean(_ape(r["pred_s"], r["latency_s"]) for r in a), 4),
            "mape_true_card": round(st.mean(_ape(r["pred_true_card_s"], r["latency_s"])
                                            for r in a), 4),
            "bias_full": round(st.mean(_signed(r["pred_s"], r["latency_s"]) for r in a), 4),
            "bias_true_card": round(st.mean(_signed(r["pred_true_card_s"], r["latency_s"])
                                            for r in a), 4),
            "over_full": sum(r["pred_s"] > r["latency_s"] for r in a),
            "over_true_card": sum(r["pred_true_card_s"] > r["latency_s"] for r in a),
            "join_passed_pred_median": round(st.median(r["join_passed_pred"] for r in a), 1),
            "join_passed_actual_median": st.median(r["join_passed_actual"] for r in a),
        }
    by_arm = defaultdict(list)
    for r in ov:
        by_arm[r["arm"]].append(r)
    keys = ["latency", "s1_span", "s2_span", "gap_s", "s2_started_early",
            "both_running_frac", "additive_ratio", "tail_after_s1"]
    out["overlap"] = {arm: {"n": len(rs), **{k: _summ(r[k] for r in rs) for k in keys}}
                      for arm, rs in by_arm.items()}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    cache = json.loads((cm.RESULTS / "analysis/item_tokens.json").read_text())
    everything = analyze.collect(cm.RESULTS)
    model = fit_like_original(everything, cache)
    arms = arm_records(model, everything, cache)
    alone = join_stage_predictions(model, everything, cache, alone_join())
    ov = overlap()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "arms.jsonl").write_text("".join(json.dumps(r) + "\n" for r in arms))
    (args.out / "overlap.jsonl").write_text("".join(json.dumps(r) + "\n" for r in ov))
    summary = summarize(arms, alone, ov)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
