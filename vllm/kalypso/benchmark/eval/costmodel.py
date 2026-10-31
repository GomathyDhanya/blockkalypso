"""Cost model for operator batching, and the block-size choice built on it.

    python3 -m vllm.kalypso.benchmark.eval.costmodel [--results DIR]

Fits on every single-operator Kalypso arm under ``eval/results/`` (see
analyze.py) and writes ``results/analysis/costmodel.md`` (coefficients, fit
error, leave-one-workload-out error, and the algorithm's choice for each
measured run against the best measured arm) and ``costmodel.json``.

Model, per operator kind (filter, join, map), for N items of mean T tokens
(a join item is a pair: left plus right row) at block size k:

    calls        C = N            (k = 1, per-tuple)     C = ceil(N / k)
    decode steps D = d_pt         (k = 1)                D = d0 + d1 * k
    computed prompt tokens        P = N * (T + a_pt)     P = N * (T + e) + C * o
                 (prompt tokens not served from the prefix cache)
    waves        W = ceil(C / max_num_seqs)
    latency      L = g + alpha * P + beta * W * D

``alpha`` is prefill seconds per computed token, ``beta`` seconds per decode
step of one wave (all of a wave's calls decode together), ``g`` a fixed
overhead. ``a_pt``, ``e``, ``o``, ``d_pt``, ``d0`` and ``d1`` are fitted per kind:
``e`` is per-item prompt overhead inside a block (item numbering), ``o`` the
per-call overhead the prefix cache does not absorb.

The choice (``choose_k``): candidates k = 1..k_max, where k_max is the
smallest of the context-window limit and the accuracy cap (the largest size
whose F1 stayed within ``f1_slack`` of per-tuple in every measured run of
that kind). Pick the k with the lowest predicted latency; among sizes within
``tie`` of it (about the model's mean error), prefer fewer calls.
"""

import argparse
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from . import analyze

RESULTS = analyze.RESULTS
TOKENIZER = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_MAX_MODEL_LEN = 32768


@dataclass
class KindParams:
    a_pt: float = 0.0      # per-tuple: computed prompt tokens per item beyond its row
    d_pt: float = 0.0      # per-tuple: decode steps per call
    e: float = 0.0         # block: prompt tokens per item beyond its row
    o: float = 0.0         # block: computed prompt tokens per call
    d0: float = 0.0        # block: decode steps per call ...
    d1: float = 0.0        # ... plus d1 per item
    k_acc: int = 1         # accuracy cap (largest measured k that kept F1)
    measured_k: list = field(default_factory=list)


@dataclass
class Model:
    g: float
    alpha: float
    beta: float
    kinds: dict            # kind -> KindParams
    f1_slack: float = 0.02

    def predict(self, kind: str, n: int, tokens: float, k: int,
                max_num_seqs: int = analyze.DEFAULT_MAX_NUM_SEQS) -> dict:
        p = self.kinds[kind]
        if k == 1:
            calls, decode, computed = n, p.d_pt, n * (tokens + p.a_pt)
        else:
            calls = math.ceil(n / k)
            decode = p.d0 + p.d1 * k
            computed = n * (tokens + p.e) + calls * p.o
        waves = math.ceil(calls / max_num_seqs)
        latency = self.g + self.alpha * computed + self.beta * waves * decode
        return {"k": k, "calls": calls, "decode_steps": decode, "computed_prompt_tokens": computed,
                "waves": waves, "latency_s": latency}

    def k_ctx(self, kind: str, max_tokens: float, max_model_len: int = DEFAULT_MAX_MODEL_LEN) -> int:
        """Largest k whose block prompt and answer fit the context window."""
        p = self.kinds[kind]
        per_item = max_tokens + p.e + p.d1
        return max(1, int((max_model_len - p.o - p.d0) // per_item)) if per_item > 0 else 1

    def choose_k(self, kind: str, n: int, tokens: float, max_tokens: float,
                 max_num_seqs: int = analyze.DEFAULT_MAX_NUM_SEQS,
                 max_model_len: int = DEFAULT_MAX_MODEL_LEN, tie: float = 0.02,
                 candidates=None) -> dict:
        p = self.kinds[kind]
        k_max = min(self.k_ctx(kind, max_tokens, max_model_len), p.k_acc, max(n, 1))
        ks = [k for k in (candidates or range(1, k_max + 1)) if k <= k_max] or [1]
        preds = [self.predict(kind, n, tokens, k, max_num_seqs) for k in ks]
        best = min(x["latency_s"] for x in preds)
        close = [x for x in preds if x["latency_s"] <= best + tie * abs(best)]
        pick = min(close, key=lambda x: (x["calls"], x["latency_s"]))
        return {**pick, "k_max": k_max, "fastest_k": min(preds, key=lambda x: x["latency_s"])["k"]}


# --- Item tokens -------------------------------------------------------------

_tokenizer = None


def _count(texts: list[str]) -> list[int]:
    global _tokenizer
    if _tokenizer is None:
        from transformers import AutoTokenizer
        _tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    return [len(ids) for ids in _tokenizer(texts, add_special_tokens=False)["input_ids"]]


def item_tokens(workload: str, n: int | None, cache: dict) -> dict:
    """Mean and max tokens per item (a join item is left + right row)."""
    key = f"{workload}:{n or ''}"
    if key not in cache:
        from .workloads import WORKLOADS
        w = WORKLOADS[workload](n or None)
        left = _count([r.text for r in w.rows])
        if w.right_rows:
            right = _count([r.text for r in w.right_rows])
            mean = sum(left) / len(left) + sum(right) / len(right)
            biggest = max(left) + max(right)
        else:
            mean, biggest = sum(left) / len(left), max(left)
        cache[key] = {"mean": round(mean, 2), "max": biggest}
    return cache[key]


# --- Fitting -----------------------------------------------------------------

def _single_operator_rows(rows: list[dict]) -> list[dict]:
    keep = []
    for r in rows:
        if r["stages"] or not r["arm"].startswith("kalypso") or r["latency_s"] is None:
            continue
        if None in (r["computed_prompt_tokens"], r["gen_tokens_per_call"], r["llm_requests"]):
            continue
        keep.append({**r, "k": int(r["blocks"])})
    return keep


_part_n = None


def _n_arg(r: dict) -> int | None:
    """The --n a run used: the manifest's for an experiment part, else its
    row count (joins: rows per side)."""
    global _part_n
    if r.get("experiment"):
        if _part_n is None:
            _part_n = {(e["id"], p["name"]): p["n"]
                       for e in _manifest().EXPERIMENTS for p in e["parts"]}
        key = (r["experiment"], r["part"])
        if key in _part_n:
            return _part_n[key] or None
    return r["rows"] if r["rows"] else None


def quality_ok(r: dict, f1_slack: float, min_agreement: float = 0.9) -> bool | None:
    """Did the arm keep per-tuple's quality? F1 for labeled workloads, agreement
    with per-tuple for unlabeled maps; None when neither was measured."""
    if r.get("f1_delta") is not None:
        return r["f1_delta"] >= -f1_slack
    if r.get("agreement") is not None:
        return r["agreement"] >= min_agreement
    return None


def attach_tokens(rows: list[dict], cache: dict) -> list[dict]:
    out = []
    for r in rows:
        try:
            t = item_tokens(r["workload"], _n_arg(r), cache)
        except Exception as exc:  # noqa: BLE001 -- e.g. a workload not in this checkout
            print(f"no item tokens for {r['workload']}: {exc}")
            continue
        out.append({**r, "item_tokens": t["mean"], "item_tokens_max": t["max"]})
    return out


def _lstsq(x, y):
    coef, *_ = np.linalg.lstsq(np.asarray(x, float), np.asarray(y, float), rcond=None)
    return [float(c) for c in coef]


def fit_kind(rows: list[dict], f1_slack: float) -> KindParams:
    pt = [r for r in rows if r["k"] == 1]
    blk = [r for r in rows if r["k"] > 1]
    p = KindParams()
    if pt:
        p.a_pt = float(np.mean([r["computed_prompt_tokens"] / r["items"] - r["item_tokens"]
                                for r in pt]))
        p.d_pt = float(np.mean([r["gen_tokens_per_call"] for r in pt]))
    if len({r["k"] for r in blk}) >= 2:
        p.e, p.o = _lstsq([[r["items"], r["llm_requests"]] for r in blk],
                          [r["computed_prompt_tokens"] - r["items"] * r["item_tokens"]
                           for r in blk])
        p.d0, p.d1 = _lstsq([[1, r["k"]] for r in blk], [r["gen_tokens_per_call"] for r in blk])
    measured = sorted({r["k"] for r in rows})
    p.measured_k = measured
    # Accuracy cap: grow k while most measured runs at that size kept
    # quality. A majority, not every run: one-off F1 swings of a few points
    # on small labeled sets would otherwise cap every kind at k = 1.
    p.k_acc = 1
    for k in measured[1:]:
        at_k = [quality_ok(r, f1_slack) for r in rows if r["k"] == k]
        at_k = [ok for ok in at_k if ok is not None]
        if at_k and sum(at_k) > len(at_k) / 2:
            p.k_acc = k
        elif at_k:
            break
    return p


def fit(rows: list[dict], f1_slack: float = 0.02) -> Model:
    x = [[1, r["computed_prompt_tokens"], r["waves"] * r["gen_tokens_per_call"]] for r in rows]
    g, alpha, beta = _lstsq(x, [r["latency_s"] for r in rows])
    kinds = {kind: fit_kind([r for r in rows if r["kind"] == kind], f1_slack)
             for kind in sorted({r["kind"] for r in rows})}
    return Model(g, alpha, beta, kinds, f1_slack)


# --- Evaluation --------------------------------------------------------------

def _err(pred, meas):
    return abs(pred - meas) / meas


def evaluate(model: Model, rows: list[dict]) -> list[dict]:
    """Per arm: latency from measured tokens (latency model alone) and from
    predicted tokens (the whole model, as the algorithm uses it)."""
    out = []
    for r in rows:
        from_measured = (model.g + model.alpha * r["computed_prompt_tokens"]
                         + model.beta * r["waves"] * r["gen_tokens_per_call"])
        full = model.predict(r["kind"], r["items"], r["item_tokens"], r["k"],
                             r["max_num_seqs"])
        out.append({**r, "pred_from_tokens": from_measured, "pred_full": full["latency_s"],
                    "pred_computed": full["computed_prompt_tokens"],
                    "pred_decode": full["decode_steps"]})
    return out


def leave_one_workload_out(rows: list[dict], f1_slack: float) -> list[float]:
    errors = []
    for w in sorted({r["workload"] for r in rows}):
        train = [r for r in rows if r["workload"] != w]
        test = [r for r in rows if r["workload"] == w]
        if not {r["kind"] for r in test} <= {r["kind"] for r in train}:
            continue
        m = fit(train, f1_slack)
        errors += [_err(m.predict(r["kind"], r["items"], r["item_tokens"], r["k"],
                                  r["max_num_seqs"])["latency_s"], r["latency_s"]) for r in test]
    return errors


def choices_per_run(model: Model, rows: list[dict]) -> list[dict]:
    """For every measured run: the algorithm's k among the sizes that run
    measured, against the fastest measured arm that kept F1."""
    out = []
    for source in sorted({r["source"] for r in rows}):
        run = [r for r in rows if r["source"] == source]
        if len(run) < 2:
            continue
        first = run[0]
        pick = model.choose_k(first["kind"], first["items"], first["item_tokens"],
                              first["item_tokens_max"], first["max_num_seqs"],
                              candidates=sorted({r["k"] for r in run}))
        by_k = {r["k"]: r for r in run}
        ok = [r for r in run if quality_ok(r, model.f1_slack) is not False]
        best = min(ok, key=lambda r: r["latency_s"])
        if pick["k"] not in by_k:
            continue
        chosen = by_k[pick["k"]]
        # The best measured size is above the accuracy cap: the run kept F1 at a
        # size another run of this kind showed to be unsafe.
        capped = best["k"] > pick["k_max"]
        free = model.choose_k(first["kind"], first["items"], first["item_tokens"],
                              first["item_tokens_max"], first["max_num_seqs"])
        out.append({"source": source, "workload": first["workload"], "kind": first["kind"],
                    "items": first["items"], "chosen_k": pick["k"], "best_k": best["k"],
                    "regret": chosen["latency_s"] / best["latency_s"] - 1,
                    "chosen_f1_delta": chosen.get("f1_delta"), "unrestricted_k": free["k"],
                    "capped": capped})
    return out


# --- Report ------------------------------------------------------------------

def _pct(x):
    return f"{100 * x:.1f}%"


def write_report(model: Model, rows: list[dict], lowo: list[float], picks: list[dict],
                 out: Path) -> str:
    fit_err = [_err(r["pred_from_tokens"], r["latency_s"]) for r in rows]
    full_err = [_err(r["pred_full"], r["latency_s"]) for r in rows]
    lines = [
        "# Cost model", "",
        f"Fitted on {len(rows)} single-operator arms from "
        f"{len({r['source'] for r in rows})} runs ({', '.join(sorted({r['workload'] for r in rows}))}).",
        "",
        "Latency `L = g + alpha * P + beta * W * D` (P computed prompt tokens, W waves, "
        "D decode steps per call):", "",
        f"- g = {model.g:.3f} s, alpha = {model.alpha * 1e4:.3f} s per 10k computed tokens "
        f"({1 / model.alpha:,.0f} tokens/s), beta = {model.beta * 1e3:.1f} ms per decode step"
        if model.alpha > 0 else f"- g = {model.g:.3f}, alpha = {model.alpha:.2e}, beta = {model.beta:.4f}",
        "",
        "| Error | Mean | Max |", "|---|---|---|",
        f"| Latency from measured tokens | {_pct(np.mean(fit_err))} | {_pct(max(fit_err))} |",
        f"| Full model (tokens predicted too) | {_pct(np.mean(full_err))} | {_pct(max(full_err))} |",
    ]
    if lowo:
        lines.append(f"| Full model, leave one workload out | {_pct(np.mean(lowo))} | "
                     f"{_pct(max(lowo))} |")
    lines += ["", "Per operator kind:", "",
              "| Kind | a_pt | d_pt | e | o | d0 | d1 | Accuracy cap k | Measured k |",
              "|---|---|---|---|---|---|---|---|---|"]
    for kind, p in model.kinds.items():
        lines.append(f"| {kind} | {p.a_pt:.1f} | {p.d_pt:.2f} | {p.e:.1f} | {p.o:.1f} | "
                     f"{p.d0:.2f} | {p.d1:.2f} | {p.k_acc} | {p.measured_k} |")
    lines += ["", "## Choice per measured run", "",
              "Chosen among the sizes the run measured; best = fastest measured arm whose F1 "
              f"stayed within {model.f1_slack} of per-tuple. Free choice = the choice over "
              "every k up to the caps. Capped: the run's best size is above the accuracy "
              "cap (another run of this kind lost F1 there), so it is left out of the regret.",
              "", "| Run | Items | Chosen k | Best k | Regret | Chosen ΔF1 | Free choice k |",
              "|---|---|---|---|---|---|---|"]
    for c in picks:
        f1 = "–" if c["chosen_f1_delta"] is None else f"{c['chosen_f1_delta']:+.2f}"
        regret = "capped" if c["capped"] else _pct(c["regret"])
        lines.append(f"| {c['source']} | {c['items']} | {c['chosen_k']} | {c['best_k']} | "
                     f"{regret} | {f1} | {c['unrestricted_k']} |")
    scored = [c["regret"] for c in picks if not c["capped"]]
    if scored:
        lines += ["", f"Regret over {len(scored)} runs: mean {_pct(np.mean(scored))}, "
                      f"max {_pct(max(scored))}."]
    lines += ["", "## Every arm", "",
              "| Run | Arm | Measured (s) | From tokens (s) | Full (s) | Computed tokens | "
              "Predicted | Decode steps | Predicted |", "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(
            f"| {r['source']} | `{r['arm']}` | {r['latency_s']:.2f} | {r['pred_from_tokens']:.2f} | "
            f"{r['pred_full']:.2f} | {r['computed_prompt_tokens']:,} | {r['pred_computed']:,.0f} | "
            f"{r['gen_tokens_per_call']:.1f} | {r['pred_decode']:.1f} |")
    text = "\n".join(lines) + "\n"
    out.mkdir(parents=True, exist_ok=True)
    (out / "costmodel.md").write_text(text, encoding="utf-8")
    (out / "costmodel.json").write_text(json.dumps(
        {"g": model.g, "alpha": model.alpha, "beta": model.beta, "f1_slack": model.f1_slack,
         "kinds": {k: asdict(v) for k, v in model.kinds.items()}}, indent=2), encoding="utf-8")
    return text


def best_in_isolation(model: Model, rows: list[dict]) -> dict[str, int]:
    """Per kind, the block size with the highest mean speedup over per-tuple
    across the measured runs, among sizes that never lost F1 (the static
    "tune each operator alone" strategy)."""
    out = {}
    for kind, p in model.kinds.items():
        speedups = {}
        for r in rows:
            if r["kind"] == kind and r["k"] > 1 and r["k"] <= p.k_acc and r["speedup"]:
                speedups.setdefault(r["k"], []).append(r["speedup"])
        out[kind] = (max(speedups, key=lambda k: np.mean(speedups[k])) if speedups else 1)
    return out


def _manifest():
    import importlib.util
    path = Path(__file__).resolve().parents[4] / "cloud" / "experiments.py"
    spec = importlib.util.spec_from_file_location("experiments", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validation_choices(model: Model, rows: list[dict], cache: dict) -> dict:
    """The algorithm's pick, best-in-isolation and a grid around the pick for
    each single-operator validation experiment, as cloud/experiments.py reads
    them. Pipeline experiments are left out until the pipeline model exists."""
    from .workloads import WORKLOADS
    iso = best_in_isolation(model, rows)
    choices = {}
    for exp in _manifest().select("validate"):
        (part,) = exp["parts"]
        try:
            workload = WORKLOADS[part["workload"]](part["n"] or None)
        except KeyError:
            continue
        if workload.stages or workload.kind not in model.kinds:
            continue
        tokens = item_tokens(part["workload"], part["n"] or None, cache)
        items = len(workload.rows) * len(workload.right_rows or [None])
        pick = model.choose_k(workload.kind, items, tokens["mean"], tokens["max"])
        k = pick["k"]
        grid = sorted({max(1, k // 2), max(1, k - 1), k + 1, 2 * k} - {k, 1})
        choices[exp["id"]] = {
            "algorithm": "kalypso_per_tuple" if k == 1 else f"kalypso_block:{k}",
            "best_in_isolation": f"kalypso_block:{iso[workload.kind]}",
            "grid": [f"kalypso_block:{g}" for g in grid if g <= pick["k_max"]],
            "predicted": {"k": k, "latency_s": round(pick["latency_s"], 3),
                          "items": items, "item_tokens": tokens["mean"],
                          "k_max": pick["k_max"]},
        }
    return choices


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--out", type=Path, default=None, help="default: <results>/analysis")
    parser.add_argument("--f1-slack", type=float, default=0.02)
    parser.add_argument("--write-choices", action="store_true",
                        help="also write the validation experiments' arms "
                             "(analysis/algorithm_choices.json; unlocks them in the manifest)")
    args = parser.parse_args()
    out = args.out or args.results / "analysis"
    cache_path = out / "item_tokens.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    rows = attach_tokens(_single_operator_rows(analyze.collect(args.results)), cache)
    out.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache, indent=2, sort_keys=True), encoding="utf-8")
    if not rows:
        raise SystemExit("no single-operator Kalypso arms to fit")
    model = fit(rows, args.f1_slack)
    evaluated = evaluate(model, rows)
    picks = choices_per_run(model, evaluated)
    write_report(model, evaluated, leave_one_workload_out(rows, args.f1_slack), picks, out)
    print(f"fitted on {len(rows)} arms -> {out / 'costmodel.md'}")
    if args.write_choices:
        choices = validation_choices(model, rows, cache)
        (out / "algorithm_choices.json").write_text(json.dumps(choices, indent=2),
                                                    encoding="utf-8")
        cache_path.write_text(json.dumps(cache, indent=2, sort_keys=True), encoding="utf-8")
        for exp_id, c in choices.items():
            print(f"{exp_id}: {c['algorithm']} (predicted {c['predicted']['latency_s']} s), "
                  f"best in isolation {c['best_in_isolation']}, grid {c['grid']}")


if __name__ == "__main__":
    main()
