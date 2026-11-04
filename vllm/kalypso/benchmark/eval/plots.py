"""Plots for the batching study write-up (results/analysis/STUDY.md).

    python3 -m vllm.kalypso.benchmark.eval.plots [--results DIR]

Fits the cost model the way costmodel.py does (everything except the held-out
phase) and writes PNGs to ``results/analysis/plots/``. Needs matplotlib.
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from . import analyze, costmodel as cm  # noqa: E402

# Representative single-operator runs: (experiment, label).
RUNS = (
    ("single/filter-nall", "filter, short rows, 1865"),
    ("single/filter-n100", "filter, short rows, 100"),
    ("longrows/cnli-filter-s50", "filter, long rows, 400"),
    ("single/join-fanout50", "join, short rows, 32x32"),
    ("longrows/cnli-join-high", "join, long rows, 24x8"),
    ("map/map-n500", "map, short rows, 500"),
)
FILTER_JOIN = ("pipeline/filter-join-s10", "validate/filter-join-s25", "pipeline/filter-join-s50",
               "validate3/filter-join-s75", "pipeline/filter-join-s90")
SHOWN = ("Tuple-wise", "Static 4", "Best in isolation", "Algorithm", cm.ORACLE)
LABELS = {"Tuple-wise": "per-tuple", "Static 4": "always block 4",
          "Best in isolation": "best size per stage alone", "Algorithm": "algorithm (v3)",
          cm.ORACLE: "fastest measured"}


def _by_k(rows, experiment):
    run = [r for r in rows if r["experiment"] == experiment]
    return sorted(run, key=lambda r: r["k"])


def _save(fig, out: Path, name: str) -> None:
    fig.tight_layout()
    fig.savefig(out / name, dpi=150)
    plt.close(fig)
    print(f"wrote {out / name}")


def speedup_vs_k(model, rows, out):
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for i, (exp, label) in enumerate(RUNS):
        run = _by_k(rows, exp)
        base = next((r for r in run if r["k"] == 1), None)
        if not base:
            continue
        color = f"C{i}"
        ks = [r["k"] for r in run]
        ax.plot(ks, [base["latency_s"] / r["latency_s"] for r in run], "o", color=color,
                label=label)
        grid = np.arange(1, max(ks) + 1)
        pred = [model.predict(base["kind"], cm.items_of(base), int(k),
                              base["max_num_seqs"])["latency_s"] for k in grid]
        ax.plot(grid, [pred[0] / p for p in pred], "-", color=color, alpha=0.6)
    ax.axhline(1, color="grey", lw=0.8, ls="--")
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8, 16, 32])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("block size k (rows per LLM call)")
    ax.set_ylabel("speedup over per-tuple")
    ax.set_title("Blocking pays for short rows with short answers\n"
                 "dots = measured, lines = cost model")
    ax.legend(fontsize=8, loc="upper right")
    _save(fig, out, "speedup_vs_k.png")


def mechanics(rows, out):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for i, (exp, label) in enumerate(RUNS):
        if exp == "single/filter-n100":
            continue
        run = _by_k(rows, exp)
        base = next((r for r in run if r["k"] == 1), None)
        if not base:
            continue
        ks = [r["k"] for r in run]
        kw = dict(marker="o", color=f"C{i}", label=label)
        axes[0].plot(ks, [r["computed_prompt_tokens"] / base["computed_prompt_tokens"]
                          for r in run], **kw)
        axes[1].plot(ks, [r["gen_tokens_per_call"] for r in run], **kw)
        axes[2].plot(ks, [r["mean_decode_time_s"] for r in run], **kw)
    titles = ("Prompt tokens computed\n(relative to per-tuple)",
              "Answer tokens per call", "Time writing the answer per call (s)")
    for ax, title in zip(axes, titles):
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 2, 4, 8, 16])
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_xlabel("block size k")
        ax.set_title(title, fontsize=10)
    axes[0].axhline(1, color="grey", lw=0.8, ls="--")
    axes[1].set_yscale("log")
    axes[0].legend(fontsize=7)
    fig.suptitle("Why: blocking reads fewer tokens (except long rows and long-row joins) "
                 "but writes longer answers", fontsize=11)
    _save(fig, out, "mechanics_vs_k.png")


def predicted_vs_measured(evaluated, pipes, validation, out):
    fig, ax = plt.subplots(figsize=(5.8, 5.4))
    for i, kind in enumerate(("filter", "join", "map")):
        rs = [r for r in evaluated if r["kind"] == kind]
        ax.plot([r["latency_s"] for r in rs], [r["pred_full"] for r in rs], "o", ms=4,
                alpha=0.6, color=f"C{i}", label=f"{kind} (fit)")
    ax.plot([r["latency_s"] for r in pipes], [r["pred_full"] for r in pipes], "s", ms=4,
            alpha=0.6, color="C4", label="pipelines (fit)")
    held = {(r["experiment"], r["arm"]): r for r in validation
            if r["phase"] in cm.HELD_OUT and r["latency_s"] and r["predicted_s"]}
    ax.plot([r["latency_s"] for r in held.values()], [r["predicted_s"] for r in held.values()],
            "*", ms=11, color="black", label="held out (" + ", ".join(cm.HELD_OUT) + ")")
    lo, hi = 0.3, 200
    ax.plot([lo, hi], [lo, hi], color="grey", lw=0.8)
    ax.fill_between([lo, hi], [lo * 0.8, hi * 0.8], [lo * 1.2, hi * 1.2], color="grey",
                    alpha=0.12, label="±20%")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("measured latency (s)")
    ax.set_ylabel("predicted latency (s)")
    ax.set_title("Cost model: predicted vs measured")
    ax.legend(fontsize=8)
    _save(fig, out, "predicted_vs_measured.png")


def validation_bars(validation, out, phase: str, title: str):
    rows = [r for r in validation if r["phase"] == phase]
    exps = list(dict.fromkeys(r["experiment"] for r in rows))
    speed = {(r["experiment"], r["strategy"]): r["speedup"] for r in rows}
    groups = [e.split("/", 1)[1] for e in exps] + ["geomean"]
    x = np.arange(len(groups))
    width = 0.8 / len(SHOWN)
    fig, ax = plt.subplots(figsize=(10, 4.4))
    for i, name in enumerate(SHOWN):
        vals = [speed.get((e, name)) or 0 for e in exps]
        ok = [v for v in vals if v]
        vals.append(float(np.exp(np.mean(np.log(ok)))) if ok else 0)
        bars = ax.bar(x + (i - (len(SHOWN) - 1) / 2) * width, vals, width, label=LABELS[name],
                      color=("black" if name == "Algorithm" else f"C{i}"))
        if name == "Algorithm":
            ax.bar_label(bars, fmt="%.2fx", fontsize=7, padding=2)
    ax.axhline(1, color="grey", lw=0.8, ls="--")
    ax.set_xticks(x, groups, fontsize=8)
    ax.set_ylabel("speedup over per-tuple")
    ax.set_title(title)
    ax.legend(fontsize=8, ncol=3)
    _save(fig, out, f"validation_{phase.replace('-', '_')}.png")


def filter_pass(pipes, out):
    labels, truth, per_tuple, blocked = [], [], [], []
    for exp in FILTER_JOIN:
        run = [r for r in pipes if r["experiment"] == exp and r["passed"] is not None]
        pt = [r["passed"] for r in run if r["blocks_t"][0] == 1]
        bl = [r["passed"] for r in run if r["blocks_t"][0] > 1]
        if not run or not pt or not bl:
            continue
        pipe = run[0]["pipe"]
        labels.append(f"{pipe.selectivity:.0%} positive")
        truth.append(pipe.selectivity * pipe.rows)
        per_tuple.append(float(np.mean(pt)))
        blocked.append(float(np.mean(bl)))
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.bar(x - 0.27, truth, 0.27, label="truly positive", color="lightgrey")
    ax.bar(x, per_tuple, 0.27, label="passed, filter per-tuple", color="C0")
    ax.bar(x + 0.27, blocked, 0.27, label="passed, filter blocked (mean over k)", color="C1")
    ax.set_xticks(x, labels)
    ax.set_ylabel("rows passed to the join (of 100)")
    ax.set_title("Filter → join: a blocked filter passes more rows,\n"
                 "so the join (32 pairs per row) gets more work")
    ax.legend(fontsize=8)
    _save(fig, out, "filter_rows_passed.png")


def map_reruns(model, out):
    keys = (("first:1", "map first,\nper-tuple"), ("first:block", "map first,\nblocked"),
            ("after:1", "after a filter,\nper-tuple"), ("after:block", "after a filter,\nblocked"))
    vals = [model.retry.get(k, 0.0) * 100 for k, _ in keys]
    fig, ax = plt.subplots(figsize=(6, 3.6))
    bars = ax.bar([lbl for _, lbl in keys], vals, color="C3")
    ax.bar_label(bars, fmt="%.0f%%", fontsize=8)
    ax.set_ylabel("extra map calls (% of map rows)")
    ax.set_title("Map reruns in pipelines when another stage is blocked")
    _save(fig, out, "map_reruns.png")


def row_length(model, rows, held, out):
    """Q1: best predicted speedup of a 1000-row filter against row length,
    with the measured runs: the fitted ones at both ends and the held-out
    row-length sweep (validate4) between them."""
    lengths = np.unique(np.geomspace(20, 3000, 40).astype(int))
    best = []
    for t in lengths:
        items = cm.Items(1000, float(t), float(t), 0, 0.0)
        lat = [model.predict("filter", items, k)["latency_s"] for k in range(1, 17)]
        best.append(lat[0] / min(lat))
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(lengths, best, "-", color="C0", label="cost model, best k (1000 rows)")
    for exp in ("single/filter-n1000", "longrows/cnli-filter-s50", "longrows/cnli-filter-s10",
                "longrows/cnli-filter-s90"):
        run = _by_k(rows, exp)
        base = next((r for r in run if r["k"] == 1), None)
        if base:
            ax.plot(base["item_tokens"], max(base["latency_s"] / r["latency_s"] for r in run),
                    "o", color="black")
    ax.plot([], [], "o", color="black", label="measured, best k (fit)")
    for n in (128, 256, 512, 1024):
        run = _by_k(held, f"validate4/cnli-len{n}")
        base = next((r for r in run if r["k"] == 1), None)
        if base:
            ax.plot(base["item_tokens"], max(base["latency_s"] / r["latency_s"] for r in run
                                             if r["latency_s"]), "*", ms=11, color="C3")
    ax.plot([], [], "*", ms=11, color="C3", label="measured, best k (held out, 400 rows)")
    ax.axhline(1, color="grey", lw=0.8, ls="--")
    ax.set_xscale("log")
    ax.set_xlabel("tokens per row")
    ax.set_ylabel("best speedup over per-tuple")
    ax.set_title("Filter: blocking stops paying as rows get longer")
    ax.legend(fontsize=8)
    _save(fig, out, "speedup_vs_row_length.png")


# Per operator, the runs that make up the row-length sweep, shortest first.
LENGTH_SWEEP = {
    "filter": ("single/filter-n1000", "validate4/cnli-len128", "validate4/cnli-len256",
               "validate4/cnli-len512", "validate4/cnli-len1024", "longrows/cnli-filter-s50"),
    "join": ("single/join-fanout50", "validate4/cnli-join-len128", "validate4/cnli-join-len256",
             "validate4/cnli-join-len512", "validate4/cnli-join-len1024", "longrows/cnli-join-high"),
    "map": ("map/map-n500", "validate4/cnli-map-len128", "validate4/cnli-map-len256",
            "validate4/cnli-map-len512", "validate4/cnli-map-len1024"),
}


def length_sweep(rows):
    """Per operator and run: tokens per item and the best measured speedup of a
    block (k >= 2) over per-tuple, with its k; below 1 means blocking lost."""
    out = {}
    for op, exps in LENGTH_SWEEP.items():
        pts = []
        for exp in exps:
            run = [r for r in _by_k(rows, exp) if r["latency_s"]]
            base = next((r for r in run if r["k"] == 1), None)
            if not base:
                continue
            blocked = [r for r in run if r["k"] > 1]
            if not blocked:
                continue
            best = min(blocked, key=lambda r: r["latency_s"])
            pts.append((base["item_tokens"], base["latency_s"] / best["latency_s"], best["k"]))
        out[op] = pts
    return out


def operators_vs_row_length(rows, out):
    """Q1, all operators: best measured speedup against tokens per item."""
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    for i, (op, pts) in enumerate(length_sweep(rows).items()):
        if not pts:
            continue
        x, y = [p[0] for p in pts], [p[1] for p in pts]
        ax.plot(x, y, "o-", color=f"C{i}", label=op)
        for t, sp, k in pts:
            ax.annotate(f"k{k}", (t, sp), textcoords="offset points", xytext=(4, 4), fontsize=7,
                        color=f"C{i}")
    ax.axhline(1, color="grey", lw=0.8, ls="--")
    ax.set_xscale("log")
    ax.set_xlabel("tokens per item (join: left + right row)")
    ax.set_ylabel("best block (k >= 2) speedup over per-tuple")
    ax.set_title("When blocking pays, per operator (below 1: per-tuple is faster)")
    ax.legend(fontsize=8)
    _save(fig, out, "operators_vs_row_length.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", type=Path, default=cm.RESULTS)
    args = parser.parse_args()
    analysis = args.results / "analysis"
    out = analysis / "plots"
    out.mkdir(parents=True, exist_ok=True)
    cache_path = analysis / "item_tokens.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    everything = analyze.collect(args.results)
    collected = [r for r in everything if r["phase"] not in cm.HELD_OUT]
    rows = cm.attach_tokens(cm._single_operator_rows(collected), cache)
    pipeline_rows = cm.attach_tokens(cm._pipeline_rows(collected), cache)
    model = cm.fit(rows, 0.02)
    model.retry = cm.fit_retry(pipeline_rows)
    model.lam = cm.fit_lambda(model, cm.pipeline_runs(model, pipeline_rows, cache))
    evaluated = cm.evaluate(model, rows)
    pipes = cm.pipeline_runs(model, pipeline_rows, cache)
    all_pipes = cm.pipeline_runs(model, cm.attach_tokens(cm._pipeline_rows(everything), cache),
                                 cache)
    choices_path = analysis / "algorithm_choices.json"
    choices = json.loads(choices_path.read_text()) if choices_path.exists() else {}
    validated = [r for r in everything if r["phase"] in cm.VALIDATION_PHASES]
    validation = cm.score_validation(model, validated, choices, cache) if choices else []

    speedup_vs_k(model, rows, out)
    mechanics(rows, out)
    predicted_vs_measured(evaluated, pipes, validation, out)
    if validation:
        for phase, title in (
                ("validate3", "Held out (validate3): block sizes picked by v3 before the runs"),
                ("validate-jf", "Held out (validate-jf): join -> filter"),
                ("validate4", "Held out (validate4): long rows and row-length sweep")):
            validation_bars(validation, out, phase, title)
    filter_pass(all_pipes, out)
    map_reruns(model, out)
    held = cm.attach_tokens(cm._single_operator_rows(
        [r for r in everything if r["phase"] in cm.HELD_OUT]), cache)
    row_length(model, rows, held, out)
    operators_vs_row_length(cm.attach_tokens(cm._single_operator_rows(everything), cache), out)


if __name__ == "__main__":
    main()
