"""Step 5: empirical validation of the original cost model (v3) on the
committed GPU results (no new runs).

    python -m research.baseline.analysis.costmodel_validation \
        --out research/baseline/results/costmodel_validation

Data: every ``run.jsonl`` under ``vllm/kalypso/benchmark/eval/results/``
(L40S, Qwen2.5-7B-Instruct, 1 warmup + 3 repeats; arms are medians over
repeats, as ``eval/analyze.py`` computes them). The model is refitted exactly
as ``costmodel.main`` does (every phase except the held-out ones) and the
coefficients are checked against the committed ``costmodel.json``.

Outputs (in ``--out``):

- ``arms.jsonl``: one record per measured Kalypso arm with prediction,
  measurement, split (``fit`` or ``held_out``) and cardinality fields;
- ``summary.json``: every metric below;
- ``summary.md``: the tables used in COST_MODEL_VALIDATION.md.

Metrics: latency MAE / RMSE / MAPE / median APE / signed bias, with under-
and over-prediction counted separately; per run with >= 3 measured arms,
Spearman rank correlation between predicted and measured latency, top-1
agreement and the regret of the model's pick among the measured arms;
cardinality q-error = max(pred/actual, actual/pred) for rows a filter passes
(single filters from labels and selected counts; pipelines from per-stage
call counts), with zero cardinalities counted separately (both zero: q = 1;
one zero: reported as "zero mismatch", excluded from q statistics).
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

import numpy as np

from vllm.kalypso.benchmark.eval import analyze
from vllm.kalypso.benchmark.eval import costmodel as cm

REPO = Path(__file__).resolve().parents[3]
COMMITTED = cm.RESULTS / "analysis" / "costmodel.json"


def _spearman(a, b):
    if len(a) < 3:
        return None
    ra = _rank(a)
    rb = _rank(b)
    if np.std(ra) == 0 or np.std(rb) == 0:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


def _rank(x):
    """Average ranks (ties share their mean rank)."""
    order = sorted(range(len(x)), key=lambda i: x[i])
    ranks = [0.0] * len(x)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and x[order[j + 1]] == x[order[i]]:
            j += 1
        for t in range(i, j + 1):
            ranks[order[t]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def q_error(pred, actual):
    if pred is None or actual is None:
        return None, "missing"
    if pred == 0 and actual == 0:
        return 1.0, "both_zero"
    if pred <= 0 or actual <= 0:
        return None, "zero_mismatch"
    return max(pred / actual, actual / pred), "ok"


def error_stats(pairs):
    """pairs: [(pred, meas)] -> MAE, RMSE, MAPE, median APE, bias, under/over."""
    pairs = [(p, m) for p, m in pairs if p is not None and m]
    if not pairs:
        return None
    err = np.array([p - m for p, m in pairs])
    ape = np.array([abs(p - m) / m for p, m in pairs])
    rel = np.array([(p - m) / m for p, m in pairs])
    under, over = rel[rel < 0], rel[rel > 0]
    return {
        "n": len(pairs),
        "mae_s": float(np.mean(np.abs(err))), "rmse_s": float(np.sqrt(np.mean(err ** 2))),
        "mape": float(np.mean(ape)), "median_ape": float(np.median(ape)),
        "p90_ape": float(np.percentile(ape, 90)), "max_ape": float(np.max(ape)),
        "mean_signed_rel": float(np.mean(rel)),
        "under_n": int(len(under)), "under_mean_rel": float(np.mean(under)) if len(under) else None,
        "over_n": int(len(over)), "over_mean_rel": float(np.mean(over)) if len(over) else None,
    }


def q_stats(records):
    qs = [r["q_error"] for r in records if r["q_status"] in ("ok", "both_zero")]
    status = {}
    for r in records:
        status[r["q_status"]] = status.get(r["q_status"], 0) + 1
    if not qs:
        return {"n": 0, "status": status}
    under = [r for r in records if r["q_status"] == "ok" and r["card_pred"] < r["card_actual"]]
    over = [r for r in records if r["q_status"] == "ok" and r["card_pred"] > r["card_actual"]]
    return {"n": len(qs), "median_q": float(np.median(qs)), "p90_q": float(np.percentile(qs, 90)),
            "max_q": float(np.max(qs)), "gmean_q": float(np.exp(np.mean(np.log(qs)))),
            "under_n": len(under), "over_n": len(over), "status": status}


def fit_like_original(everything, cache):
    collected = [r for r in everything if r["phase"] not in cm.HELD_OUT]
    rows = cm.attach_tokens(cm._single_operator_rows(collected), cache)
    pipeline_rows = cm.attach_tokens(cm._pipeline_rows(collected), cache)
    model = cm.fit(rows, 0.02)
    model.retry = cm.fit_retry(pipeline_rows)
    model.lam = cm.fit_lambda(model, cm.pipeline_runs(model, pipeline_rows, cache))
    return model


def check_against_committed(model):
    d = json.loads(COMMITTED.read_text())
    diffs = {k: abs(getattr(model, k) - d[k]) / abs(d[k]) for k in ("alpha", "beta", "gamma", "delta")}
    diffs["lambda"] = abs(model.lam - d["lambda"]) / abs(d["lambda"])
    return {k: float(v) for k, v in diffs.items()}


def single_records(model, everything, cache):
    out = []
    rows = cm.attach_tokens(cm._single_operator_rows(everything), cache)
    for r in rows:
        split = "held_out" if r["phase"] in cm.HELD_OUT else "fit"
        pred = model.predict(r["kind"], cm.items_of(r), r["k"], r["max_num_seqs"])
        rec = {"source": r["source"], "phase": r["phase"], "split": split,
               "workload": r["workload"], "kind": r["kind"], "regime": cm._regime(r),
               "arm": r["arm"], "k": r["k"], "items": r["items"],
               "latency_s": r["latency_s"], "pred_s": pred["latency_s"],
               "pred_from_tokens_s": model.latency(r["kind"], r["computed_prompt_tokens"],
                                                   r["waves"], r["gen_tokens_per_call"],
                                                   r["llm_requests"], r["prompt_tokens_per_call"]),
               "calls": r["llm_requests"], "pred_calls": pred["calls"],
               "computed_prompt_tokens": r["computed_prompt_tokens"],
               "pred_computed_prompt_tokens": pred["computed_prompt_tokens"],
               "peak_kv_cache_pct": r.get("peak_kv_cache_pct"),
               "peak_waiting": r.get("peak_waiting"), "preemptions": r.get("preemptions"),
               "f1": r.get("f1"), "f1_delta": r.get("f1_delta")}
        # Cardinality: rows a single filter passes, from its recall/FPR at k
        # and the workload's true selectivity (labels).
        if r["kind"] == "filter" and r.get("positives") is not None and r.get("selected_count") is not None:
            s = r["positives"] / r["items"]
            card_pred = r["items"] * model.kinds["filter"].pass_share(cm._regime(r), r["k"], s)
            q, status = q_error(card_pred, r["selected_count"])
            rec.update(card_kind="filter_pass", card_pred=card_pred,
                       card_actual=r["selected_count"], card_true=r["positives"],
                       selectivity_true=s, q_error=q, q_status=status)
        elif r["kind"] == "join" and r.get("positives") is not None and r.get("selected_count") is not None:
            s = r["positives"] / r["items"]
            card_pred = r["items"] * model.kinds["join"].pass_share(cm._regime(r), r["k"], s)
            q, status = q_error(card_pred, r["selected_count"])
            rec.update(card_kind="join_pass", card_pred=card_pred,
                       card_actual=r["selected_count"], card_true=r["positives"],
                       selectivity_true=s, q_error=q, q_status=status)
        out.append(rec)
    return out


def pipeline_records(model, everything, cache):
    out, skipped = [], []
    prows = cm.attach_tokens(cm._pipeline_rows(everything), cache)
    for source in sorted({r["source"] for r in prows}):
        run = [r for r in prows if r["source"] == source]
        base = run[0]
        try:
            pipe = cm.pipeline_of(
                base, cm.workload_selectivity(base["workload"], cm._n_arg(base), cache),
                cm.workload_stage_selectivity(base["workload"], cm._n_arg(base), cache))
        except Exception as exc:  # noqa: BLE001 -- e.g. dataset not downloadable here
            skipped.append({"source": source, "reason": f"{type(exc).__name__}: {exc}"[:200]})
            continue
        for r in run:
            passed = cm.passed_rows(r, pipe)
            full = model.predict_pipeline(pipe, r["blocks_t"], r["max_num_seqs"])
            comp = model.predict_pipeline(pipe, r["blocks_t"], r["max_num_seqs"], passed=passed,
                                          retried=cm.retried_rows(r, pipe))
            rec = {"source": source, "phase": r["phase"],
                   "split": "held_out" if r["phase"] in cm.HELD_OUT else "fit",
                   "workload": r["workload"], "stages": r["stages"], "arm": r["arm"],
                   "blocks": list(r["blocks_t"]), "latency_s": r["latency_s"],
                   "pred_s": full["latency_s"], "pred_given_cardinality_s": comp["latency_s"],
                   "calls": r["llm_requests"], "pred_calls": full["calls"],
                   "peak_kv_cache_pct": r.get("peak_kv_cache_pct"),
                   "peak_waiting": r.get("peak_waiting"), "preemptions": r.get("preemptions"),
                   "f1_delta": r.get("f1_delta")}
            if passed is not None and pipe.ops[0] == "filter":
                # Predicted rows out of the leading filter (as predict_pipeline
                # computes them) vs rows that reached stage 2.
                items0 = model.stage_items(pipe, pipe.rows)[0]
                card_pred = pipe.rows * model.kinds["filter"].pass_share(
                    items0.regime, r["blocks_t"][0], pipe.true_share(0))
                q, status = q_error(card_pred, passed)
                rec.update(card_kind="pipeline_stage1_out", card_pred=card_pred,
                           card_actual=passed, selectivity_true=pipe.true_share(0),
                           q_error=q, q_status=status)
            out.append(rec)
    return out, skipped


def ranking(records, key="pred_s"):
    """Per run with >= 3 measured arms: Spearman, top-1, regret of the pick."""
    out = []
    for source in sorted({r["source"] for r in records}):
        run = [r for r in records if r["source"] == source and r["latency_s"]]
        if len(run) < 3:
            continue
        pred = [r[key] for r in run]
        meas = [r["latency_s"] for r in run]
        pick = min(run, key=lambda r: r[key])
        best = min(run, key=lambda r: r["latency_s"])
        out.append({"source": source, "split": run[0]["split"], "arms": len(run),
                    "spearman": _spearman(pred, meas),
                    "top1": pick["arm"] == best["arm"],
                    "regret": pick["latency_s"] / best["latency_s"] - 1,
                    "pick": pick["arm"], "best": best["arm"]})
    return out


def rank_stats(rows):
    if not rows:
        return None
    sp = [r["spearman"] for r in rows if r["spearman"] is not None]
    rg = [r["regret"] for r in rows]
    return {"runs": len(rows), "arms_total": sum(r["arms"] for r in rows),
            "spearman_median": float(np.median(sp)) if sp else None,
            "spearman_mean": float(np.mean(sp)) if sp else None,
            "spearman_min": float(np.min(sp)) if sp else None,
            "top1_rate": float(np.mean([r["top1"] for r in rows])),
            "regret_mean": float(np.mean(rg)), "regret_median": float(np.median(rg)),
            "regret_max": float(np.max(rg))}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", type=Path, default=cm.RESULTS)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    cache_path = args.results / "analysis" / "item_tokens.json"
    cache = json.loads(cache_path.read_text())  # read only; never written back
    everything = analyze.collect(args.results)
    model = fit_like_original(everything, cache)
    singles = single_records(model, everything, cache)
    pipes, skipped = pipeline_records(model, everything, cache)

    summary = {"coefficient_rel_diff_vs_committed": check_against_committed(model),
               "pipelines_skipped": skipped, "latency": {}, "ranking": {}, "cardinality": {}}
    for split in ("fit", "held_out", "all"):
        sel = (lambda r: True) if split == "all" else (lambda r, s=split: r["split"] == s)
        s = [r for r in singles if sel(r)]
        p = [r for r in pipes if sel(r)]
        summary["latency"][split] = {
            "single_full": error_stats([(r["pred_s"], r["latency_s"]) for r in s]),
            "single_from_measured_tokens": error_stats(
                [(r["pred_from_tokens_s"], r["latency_s"]) for r in s]),
            "pipeline_full": error_stats([(r["pred_s"], r["latency_s"]) for r in p]),
            "pipeline_given_cardinality": error_stats(
                [(r["pred_given_cardinality_s"], r["latency_s"]) for r in p]),
            "calls_single": error_stats([(r["pred_calls"], r["calls"]) for r in s]),
            "computed_prompt_tokens_single": error_stats(
                [(r["pred_computed_prompt_tokens"], r["computed_prompt_tokens"]) for r in s]),
        }
        by_kind = {}
        for kind in sorted({r["kind"] for r in s}):
            for regime in ("short", "long"):
                ks = [r for r in s if r["kind"] == kind and r["regime"] == regime]
                if ks:
                    by_kind[f"{kind}/{regime}"] = error_stats([(r["pred_s"], r["latency_s"]) for r in ks])
        summary["latency"][split]["single_by_kind_regime"] = by_kind
        summary["ranking"][split] = {"single": rank_stats(ranking(s)),
                                     "pipeline": rank_stats(ranking(p))}
        card = [r for r in s + p if "q_status" in r]
        summary["cardinality"][split] = {
            kind: q_stats([r for r in card if r["card_kind"] == kind])
            for kind in sorted({r["card_kind"] for r in card})}
    summary["ranking_runs"] = {"single": ranking(singles), "pipeline": ranking(pipes)}
    summary["kv"] = {
        "modeled": False,
        "note": "costmodel.py has no KV-memory, queueing or preemption term; "
                "only measured peaks are available (below).",
        "peak_kv_cache_pct": _dist([r["peak_kv_cache_pct"] for r in singles + pipes]),
        "peak_waiting": _dist([r["peak_waiting"] for r in singles + pipes]),
        "preemptions_nonzero_arms": sum(1 for r in singles + pipes if r.get("preemptions")),
        "arms": len(singles + pipes),
    }

    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "arms.jsonl").open("w") as f:
        for r in singles + pipes:
            f.write(json.dumps({"type": "single" if "k" in r else "pipeline", **r}) + "\n")
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    (args.out / "summary.md").write_text(render(summary))
    print(render(summary))


def _dist(values):
    v = [x for x in values if x is not None]
    if not v:
        return None
    return {"n": len(v), "median": float(np.median(v)), "p90": float(np.percentile(v, 90)),
            "max": float(np.max(v))}


def _pct(x):
    return "–" if x is None else f"{100 * x:.1f}%"


def _f(x, spec=".2f"):
    return "–" if x is None else format(x, spec)


def render(s):
    lines = ["# Cost-model validation (v3, committed GPU results)", "",
             "Coefficient relative difference after refitting vs committed costmodel.json: "
             + ", ".join(f"{k} {v:.1e}" for k, v in s["coefficient_rel_diff_vs_committed"].items()),
             ""]
    lines += ["## Latency", "",
              "| Split | Prediction | n | MAE (s) | RMSE (s) | MAPE | median APE | p90 APE | max APE | "
              "under (n, mean) | over (n, mean) |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for split, d in s["latency"].items():
        for name in ("single_full", "single_from_measured_tokens", "pipeline_full",
                     "pipeline_given_cardinality"):
            e = d[name]
            if not e:
                continue
            lines.append(f"| {split} | {name} | {e['n']} | {e['mae_s']:.2f} | {e['rmse_s']:.2f} | "
                         f"{_pct(e['mape'])} | {_pct(e['median_ape'])} | {_pct(e['p90_ape'])} | "
                         f"{_pct(e['max_ape'])} | {e['under_n']}, {_pct(e['under_mean_rel'])} | "
                         f"{e['over_n']}, {_pct(e['over_mean_rel'])} |")
    lines += ["", "### Single operators by kind and regime (full model)", "",
              "| Split | Kind/regime | n | MAPE | median APE | max APE | under n | over n |",
              "|---|---|---|---|---|---|---|---|"]
    for split, d in s["latency"].items():
        for name, e in d["single_by_kind_regime"].items():
            lines.append(f"| {split} | {name} | {e['n']} | {_pct(e['mape'])} | "
                         f"{_pct(e['median_ape'])} | {_pct(e['max_ape'])} | {e['under_n']} | "
                         f"{e['over_n']} |")
    lines += ["", "## Ranking among measured arms (per run with >= 3 arms)", "",
              "| Split | Level | Runs | Arms | Spearman median | Spearman min | Top-1 | "
              "Regret mean | Regret max |", "|---|---|---|---|---|---|---|---|---|"]
    for split, d in s["ranking"].items():
        for level, e in d.items():
            if not e:
                continue
            lines.append(f"| {split} | {level} | {e['runs']} | {e['arms_total']} | "
                         f"{_f(e['spearman_median'])} | {_f(e['spearman_min'])} | "
                         f"{_pct(e['top1_rate'])} | {_pct(e['regret_mean'])} | "
                         f"{_pct(e['regret_max'])} |")
    lines += ["", "## Cardinality (q-error; zero cardinalities counted separately)", "",
              "| Split | Quantity | n | median q | p90 q | max q | under n | over n | status |",
              "|---|---|---|---|---|---|---|---|---|"]
    for split, d in s["cardinality"].items():
        for kind, e in d.items():
            lines.append(f"| {split} | {kind} | {e['n']} | {_f(e.get('median_q'))} | "
                         f"{_f(e.get('p90_q'))} | {_f(e.get('max_q'))} | {e.get('under_n', '–')} | "
                         f"{e.get('over_n', '–')} | {e['status']} |")
    kv = s["kv"]
    lines += ["", "## KV / resources", "", kv["note"], "",
              f"- peak KV cache %: {kv['peak_kv_cache_pct']}",
              f"- peak waiting requests: {kv['peak_waiting']}",
              f"- arms with any preemption: {kv['preemptions_nonzero_arms']} of {kv['arms']}"]
    if s["pipelines_skipped"]:
        lines += ["", "Pipelines skipped:", *[f"- {x['source']}: {x['reason']}"
                                             for x in s["pipelines_skipped"]]]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
