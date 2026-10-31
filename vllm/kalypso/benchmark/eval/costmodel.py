"""Cost model for operator batching, and the block-size choice built on it.

    python3 -m vllm.kalypso.benchmark.eval.costmodel [--results DIR] [--write-choices]

Fits on every Kalypso arm under ``eval/results/`` (see analyze.py): the
latency and token model on single-operator arms, the composition factor on
pipeline arms. Writes ``results/analysis/costmodel.md`` (coefficients, fit
and leave-one-workload-out error, accuracy caps, the choice for every measured
run against the best measured arm) and ``costmodel.json``. With
``--write-choices`` also ``algorithm_choices.json``, the arms the validation
experiments in cloud/experiments.py run.

Latency of one operator over N items (a join item is a pair) at block size k,
from C calls, W = ceil(C / max_num_seqs) waves, D decode steps per call, P
prompt tokens computed (not served from the prefix cache) and p prompt tokens
per call:

    L = g + alpha * P + (beta + tail[kind]) * W * D + gamma * C * D + delta * C * p^2

``alpha`` is prefill time per token; ``beta`` the time of one decode step of a
wave (its calls decode together, so a wave lasts its decode steps);
``tail[kind]`` the extra for operators whose outputs vary in length (maps: a
wave lasts as long as its longest answer, not its mean); ``gamma`` the
per-token cost of a larger running batch; ``delta`` attention, quadratic in
prompt length, which long rows make visible.

Tokens, per kind, for mean item tokens T (joins: left row T_L plus right row)
and, for joins, N_L distinct left rows:

    per-tuple (k = 1)  C = N   D = d_pt   p = T + q_pt
                       P = N (T + a_pt)                      (filter, map)
                       P = N (T - T_L + a_pt) + N_L T_L      (join: each left row
                                                              is computed once; the
                                                              prefix cache serves it
                                                              to its other pairs)
    block (k > 1)      C = ceil(N / k)   D = d0 + d1 k   p = q + k (T + e)
                       P = N (T + e) + C o - c_left max(0, C - N_L) T_L

A block join writes the left row into every pair, so it computes the left row
about once per pair rather than once per left row; only calls that start on
the same left row share it (``c_left``). That is why blocking a join over long
left rows is several times slower.

The choice (``choose_k``): k from 1 up to the smallest of the context-window
limit and the accuracy cap of the operator's regime (short or long rows): the
largest measured k whose mean F1 change against per-tuple across runs stayed
within ``f1_slack`` (maps: mean ROUGE-1 similarity to per-tuple's outputs at
least analyze.MIN_SIMILARITY). Lowest predicted latency wins; among sizes
within ``tie`` of it (about the model's error), fewer calls win.

Pipelines (``choose_pipeline``): stages stream into each other on one GPU, so
their work adds up: L = g + lambda * sum_i (L_i - g), lambda fitted on the
pipeline runs. A filter's output size depends on its block size: block filters
pass more rows (higher recall) than per-tuple, rho(k) times as many, measured
on the single-filter runs; that changes the work of every later stage.
"""

import argparse
import itertools
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from . import analyze

RESULTS = analyze.RESULTS
TOKENIZER = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_MAX_MODEL_LEN = 32768
# Items above this many tokens are "long rows" (ContractNLI, ~1.6k): a regime
# with its own accuracy cap.
LONG_ROW_TOKENS = 512
# Operators whose answers vary in length, so a wave lasts its longest call.
TAIL_KINDS = ("map",)
# Largest size considered when the accuracy cap is off and the regime has no
# measured sizes.
MAX_K = 32


@dataclass
class Items:
    n: float                  # items (joins: pairs)
    tokens: float             # mean tokens per item (joins: left + right row)
    max_tokens: float
    left_rows: float = 0      # joins: distinct left rows
    left_tokens: float = 0.0  # joins: mean left-row tokens

    @property
    def regime(self) -> str:
        return "long" if self.tokens > LONG_ROW_TOKENS else "short"


@dataclass
class KindParams:
    a_pt: float = 0.0      # per-tuple: computed prompt tokens per item beyond its row
    q_pt: float = 0.0      # per-tuple: prompt tokens per call beyond its row
    d_pt: float = 0.0      # per-tuple: decode steps per call
    e: float = 0.0         # block: prompt tokens per item beyond its row
    o: float = 0.0         # block: computed prompt tokens per call
    c_left: float = 0.0    # block join: share of a repeated left row the cache serves
    q: float = 0.0         # block: prompt tokens per call beyond its items
    d0: float = 0.0        # block: decode steps per call ...
    d1: float = 0.0        # ... plus d1 per item
    caps: dict = field(default_factory=dict)        # regime -> accuracy cap k
    measured_k: dict = field(default_factory=dict)  # regime -> measured sizes
    rho: dict = field(default_factory=dict)         # regime -> {k: output rows vs per-tuple}

    def cap(self, regime: str) -> int:
        return self.caps.get(regime, 1)

    def pass_ratio(self, regime: str, k: int) -> float:
        """Rows a block filter passes relative to per-tuple, interpolated
        between measured sizes (1 at k = 1, flat past the largest)."""
        points = sorted((int(x), y) for x, y in self.rho.get(regime, {}).items())
        if k <= 1 or not points:
            return 1.0
        xs, ys = [1] + [x for x, _ in points], [1.0] + [y for _, y in points]
        return float(np.interp(k, xs, ys))


@dataclass
class Pipeline:
    ops: tuple                 # e.g. ("filter", "join")
    rows: float                # input rows
    tokens: float              # mean row tokens
    max_tokens: float
    pass_share: float          # share of rows the filter passes at k = 1
    right_rows: float = 0      # joins
    right_tokens: float = 0.0
    right_max_tokens: float = 0.0


@dataclass
class Model:
    g: float
    alpha: float
    beta: float
    gamma: float
    delta: float
    tail: dict             # kind -> extra seconds per decode step of a wave
    kinds: dict            # kind -> KindParams
    f1_slack: float = 0.02
    lam: float = 1.0       # pipelines: share of the stages' summed time not overlapped

    def latency(self, kind, computed, waves, decode, calls, per_call) -> float:
        return (self.g + self.alpha * computed
                + (self.beta + self.tail.get(kind, 0.0)) * waves * decode
                + self.gamma * calls * decode + self.delta * calls * per_call ** 2 / 1e3)

    def predict(self, kind: str, items: Items, k: int,
                max_num_seqs: int = analyze.DEFAULT_MAX_NUM_SEQS) -> dict:
        p, n, t = self.kinds[kind], items.n, items.tokens
        if k == 1:
            calls, decode, per_call = n, p.d_pt, t + p.q_pt
            computed = n * (t - items.left_tokens + p.a_pt) + items.left_rows * items.left_tokens
        else:
            calls = math.ceil(n / k)
            decode, per_call = p.d0 + p.d1 * k, p.q + k * (t + p.e)
            computed = (n * (t + p.e) + calls * p.o
                        - p.c_left * max(0, calls - items.left_rows) * items.left_tokens)
        waves = math.ceil(calls / max_num_seqs)
        return {"k": k, "calls": calls, "decode_steps": decode, "computed_prompt_tokens": computed,
                "prompt_per_call": per_call, "waves": waves,
                "latency_s": self.latency(kind, computed, waves, decode, calls, per_call)}

    def k_ctx(self, kind: str, max_tokens: float, max_model_len: int = DEFAULT_MAX_MODEL_LEN) -> int:
        """Largest k whose block prompt and answer fit the context window."""
        p = self.kinds[kind]
        per_item = max_tokens + p.e + p.d1
        return max(1, int((max_model_len - p.q - p.d0) // per_item)) if per_item > 0 else 1

    def k_max(self, kind: str, items: Items, quality: bool = True,
              max_model_len: int = DEFAULT_MAX_MODEL_LEN) -> int:
        p = self.kinds[kind]
        # Without the accuracy cap, stay within the sizes measured for the
        # regime: the model is not checked beyond them.
        cap = (p.cap(items.regime) if quality
               else max(p.measured_k.get(items.regime) or [MAX_K]))
        return max(1, min(self.k_ctx(kind, items.max_tokens, max_model_len), cap,
                          max(int(items.n), 1)))

    def choose_k(self, kind: str, items: Items,
                 max_num_seqs: int = analyze.DEFAULT_MAX_NUM_SEQS,
                 max_model_len: int = DEFAULT_MAX_MODEL_LEN, tie: float = 0.02,
                 candidates=None, quality: bool = True) -> dict:
        k_max = self.k_max(kind, items, quality, max_model_len)
        ks = [k for k in (candidates or range(1, k_max + 1)) if k <= k_max] or [1]
        preds = [self.predict(kind, items, k, max_num_seqs) for k in ks]
        pick = _pick(preds, tie)
        return {**pick, "k_max": k_max,
                "fastest_k": min(preds, key=lambda x: x["latency_s"])["k"]}

    # --- Pipelines ------------------------------------------------------------

    def stage_items(self, pipe: Pipeline, rows: float) -> list[Items]:
        """Each stage's input for ``rows`` rows reaching every stage (the sizes
        are filled in by predict_pipeline; this gives tokens and regimes)."""
        out, tokens, max_tokens = [], pipe.tokens, pipe.max_tokens
        for op in pipe.ops:
            if op == "join":
                out.append(Items(rows * pipe.right_rows, tokens + pipe.right_tokens,
                                 max_tokens + pipe.right_max_tokens, rows, tokens))
            else:
                out.append(Items(rows, tokens, max_tokens))
            if op == "map":
                # The next stage sees each row with its map output.
                tokens += self.kinds["map"].d_pt
                max_tokens += self.kinds["map"].d_pt
        return out

    def predict_pipeline(self, pipe: Pipeline, blocks: tuple,
                         max_num_seqs: int = analyze.DEFAULT_MAX_NUM_SEQS,
                         passed: float | None = None) -> dict:
        """``passed``: rows the leading filter passed, when known (measured);
        otherwise per-tuple's pass share times rho(k)."""
        rows, stages = pipe.rows, []
        for i, (op, k) in enumerate(zip(pipe.ops, blocks)):
            items = self.stage_items(pipe, rows)[i]
            stages.append(self.predict(op, items, k, max_num_seqs))
            if op == "filter":
                rows = (passed if passed is not None else
                        rows * pipe.pass_share
                        * self.kinds["filter"].pass_ratio(items.regime, k))
        latency = self.g + self.lam * sum(s["latency_s"] - self.g for s in stages)
        return {"blocks": tuple(blocks), "stages": stages, "latency_s": latency,
                "calls": sum(s["calls"] for s in stages)}

    def choose_pipeline(self, pipe: Pipeline, max_num_seqs: int = analyze.DEFAULT_MAX_NUM_SEQS,
                        tie: float = 0.02, candidates=None, quality: bool = True) -> dict:
        k_max = [self.k_max(op, items, quality)
                 for op, items in zip(pipe.ops, self.stage_items(pipe, pipe.rows))]
        if candidates is None:
            candidates = itertools.product(*[range(1, k + 1) for k in k_max])
        candidates = [tuple(c) for c in candidates
                      if all(b <= m for b, m in zip(c, k_max))] or [(1,) * len(pipe.ops)]
        preds = [self.predict_pipeline(pipe, c, max_num_seqs) for c in candidates]
        pick = _pick(preds, tie)
        return {**pick, "fastest": min(preds, key=lambda x: x["latency_s"])["blocks"]}


def _pick(preds: list[dict], tie: float) -> dict:
    best = min(x["latency_s"] for x in preds)
    close = [x for x in preds if x["latency_s"] <= best + tie * abs(best)]
    return min(close, key=lambda x: (x["calls"], x["latency_s"]))


def arm_name(blocks) -> str:
    blocks = tuple(blocks)
    if all(b == 1 for b in blocks):
        return "kalypso_per_tuple"
    if len(blocks) == 1:
        return f"kalypso_block:{blocks[0]}"
    return "kalypso_pipe:" + ",".join(str(b) for b in blocks)


# --- Item tokens -------------------------------------------------------------

_tokenizer = None


def _count(texts: list[str]) -> list[int]:
    global _tokenizer
    if _tokenizer is None:
        from transformers import AutoTokenizer
        _tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    return [len(ids) for ids in _tokenizer(texts, add_special_tokens=False)["input_ids"]]


def item_tokens(workload: str, n: int | None, cache: dict) -> dict:
    """Token counts of a workload's rows: mean and max per item (a join item
    is left + right row), and per side."""
    key = f"{workload}:{n or ''}"
    if "left_mean" not in cache.get(key, {}):
        from .workloads import WORKLOADS
        w = WORKLOADS[workload](n or None)
        left = _count([r.text for r in w.rows])
        right = _count([r.text for r in w.right_rows]) if w.right_rows else [0]
        left_mean, right_mean = sum(left) / len(left), sum(right) / len(right)
        cache[key] = {"mean": round(left_mean + right_mean, 2), "max": max(left) + max(right),
                      "left_mean": round(left_mean, 2), "left_max": max(left),
                      "right_mean": round(right_mean, 2), "right_max": max(right),
                      "left_rows": len(left), "right_rows": len(right) if w.right_rows else 0}
    return cache[key]


# --- Rows ----------------------------------------------------------------------

def _kalypso_rows(rows: list[dict]) -> list[dict]:
    keep = []
    for r in rows:
        if not r["arm"].startswith("kalypso") or r["latency_s"] is None:
            continue
        if None in (r["computed_prompt_tokens"], r["gen_tokens_per_call"], r["llm_requests"],
                    r["prompt_tokens_per_call"]):
            continue
        keep.append({**r, "blocks_t": tuple(int(b) for b in str(r["blocks"]).split(","))})
    return keep


def _single_operator_rows(rows: list[dict]) -> list[dict]:
    return [{**r, "k": r["blocks_t"][0]} for r in _kalypso_rows(rows) if not r["stages"]]


def _pipeline_rows(rows: list[dict]) -> list[dict]:
    return [r for r in _kalypso_rows(rows) if r["stages"]]


def _manifest():
    import importlib.util
    path = Path(__file__).resolve().parents[4] / "cloud" / "experiments.py"
    spec = importlib.util.spec_from_file_location("experiments", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def attach_tokens(rows: list[dict], cache: dict) -> list[dict]:
    out = []
    for r in rows:
        try:
            t = item_tokens(r["workload"], _n_arg(r), cache)
        except Exception as exc:  # noqa: BLE001 -- e.g. a workload not in this checkout
            print(f"no item tokens for {r['workload']}: {exc}")
            continue
        joined = bool(t["right_rows"]) and r["kind"] == "join" and not r["stages"]
        out.append({**r, "item_tokens": t["mean"], "item_tokens_max": t["max"],
                    "left_tokens": t["left_mean"] if joined else 0.0,
                    "left_rows": t["left_rows"] if joined else 0, "tok": t})
    return out


def items_of(r: dict) -> Items:
    return Items(r["items"], r["item_tokens"], r["item_tokens_max"],
                 r.get("left_rows", 0), r.get("left_tokens", 0.0))


def quality_ok(r: dict, f1_slack: float) -> bool | None:
    """Did the arm keep per-tuple's quality? F1 for labeled workloads, output
    similarity for unlabeled maps; None when neither was measured."""
    if r.get("f1_delta") is not None:
        return r["f1_delta"] >= -f1_slack
    if r.get("similarity") is not None:
        return r["similarity"] >= analyze.MIN_SIMILARITY
    return None


def _regime(r: dict) -> str:
    return "long" if r["item_tokens"] > LONG_ROW_TOKENS else "short"


# --- Fitting -----------------------------------------------------------------

def _lstsq(x, y, w=None):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if w is not None:
        w = np.asarray(w, float)
        x, y = x * w[:, None], y * w
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    return [float(c) for c in coef]


def _cap(rows: list[dict], measured: list[int], f1_slack: float) -> int:
    """Largest measured k (growing from 1) whose runs kept quality on average."""
    cap = 1
    for k in measured:
        if k == 1:
            continue
        at_k = [r for r in rows if r["k"] == k]
        f1 = [r["f1_delta"] for r in at_k if r.get("f1_delta") is not None]
        sim = [r["similarity"] for r in at_k if r.get("similarity") is not None]
        if f1:
            ok = np.mean(f1) >= -f1_slack
        elif sim:
            ok = np.mean(sim) >= analyze.MIN_SIMILARITY
        else:
            continue
        if not ok:
            break
        cap = k
    return cap


def fit_kind(rows: list[dict], f1_slack: float) -> KindParams:
    pt = [r for r in rows if r["k"] == 1]
    blk = [r for r in rows if r["k"] > 1]
    p = KindParams()
    if pt:
        p.a_pt = float(np.mean([(r["computed_prompt_tokens"] - r["left_rows"] * r["left_tokens"])
                                / r["items"] - r["item_tokens"] + r["left_tokens"] for r in pt]))
        p.q_pt = float(np.mean([r["prompt_tokens_per_call"] - r["item_tokens"] for r in pt]))
        p.d_pt = float(np.mean([r["gen_tokens_per_call"] for r in pt]))
    if len({r["k"] for r in blk}) >= 2:
        joined = any(r["left_tokens"] for r in blk)
        x = [[r["items"], r["llm_requests"]]
             + ([-max(0, r["llm_requests"] - r["left_rows"]) * r["left_tokens"]] if joined else [])
             for r in blk]
        coef = _lstsq(x, [r["computed_prompt_tokens"] - r["items"] * r["item_tokens"] for r in blk])
        p.e, p.o = coef[:2]
        p.c_left = coef[2] if joined else 0.0
        p.q = float(np.mean([r["prompt_tokens_per_call"] - r["k"] * (r["item_tokens"] + p.e)
                             for r in blk]))
        p.d0, p.d1 = _lstsq([[1, r["k"]] for r in blk], [r["gen_tokens_per_call"] for r in blk])
    for regime in ("short", "long"):
        at = [r for r in rows if _regime(r) == regime]
        if not at:
            continue
        measured = sorted({r["k"] for r in at})
        p.measured_k[regime] = measured
        p.caps[regime] = _cap(at, measured, f1_slack)
        # Output rows against per-tuple in the same run (filters), pooled
        # (sum over sum) so a run where per-tuple passed 2 rows does not
        # swing the ratio.
        sums = {}
        for source in sorted({r["source"] for r in at}):
            run = {r["k"]: r for r in at if r["source"] == source}
            base = run.get(1, {}).get("selected_count")
            if not base:
                continue
            for k, r in run.items():
                if k > 1 and r.get("selected_count") is not None:
                    top, bottom = sums.get(k, (0.0, 0.0))
                    sums[k] = (top + r["selected_count"], bottom + base)
        if sums:
            p.rho[regime] = {str(k): round(a / b, 4) for k, (a, b) in sorted(sums.items())}
    return p


def latency_features(r: dict, tails=TAIL_KINDS) -> list[float]:
    """Measured features of one arm, in Model.latency's order."""
    wd = r["waves"] * r["gen_tokens_per_call"]
    return [1.0, r["computed_prompt_tokens"], wd, *[wd * (r["kind"] == kind) for kind in tails],
            r["llm_requests"] * r["gen_tokens_per_call"],
            r["llm_requests"] * r["prompt_tokens_per_call"] ** 2 / 1e3]


def fit(rows: list[dict], f1_slack: float = 0.02) -> Model:
    """Least squares on relative error, so short runs count as much as long ones."""
    tails = tuple(kind for kind in TAIL_KINDS if any(r["kind"] == kind for r in rows))
    coef = _lstsq([latency_features(r, tails) for r in rows], [r["latency_s"] for r in rows],
                  [1 / r["latency_s"] for r in rows])
    g, alpha, beta, *rest = coef
    tail = dict(zip(tails, rest[:len(tails)]))
    gamma, delta = rest[len(tails):]
    kinds = {kind: fit_kind([r for r in rows if r["kind"] == kind], f1_slack)
             for kind in sorted({r["kind"] for r in rows})}
    return Model(g, alpha, beta, gamma, delta, tail, kinds, f1_slack)


# --- Evaluation --------------------------------------------------------------

def _err(pred, meas):
    return abs(pred - meas) / meas


def evaluate(model: Model, rows: list[dict]) -> list[dict]:
    """Per arm: latency from measured tokens (latency model alone) and from
    predicted tokens (the whole model, as the algorithm uses it)."""
    out = []
    for r in rows:
        from_measured = model.latency(r["kind"], r["computed_prompt_tokens"], r["waves"],
                                      r["gen_tokens_per_call"], r["llm_requests"],
                                      r["prompt_tokens_per_call"])
        full = model.predict(r["kind"], items_of(r), r["k"], r["max_num_seqs"])
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
        errors += [_err(m.predict(r["kind"], items_of(r), r["k"],
                                  r["max_num_seqs"])["latency_s"], r["latency_s"]) for r in test]
    return errors


def choices_per_run(model: Model, rows: list[dict]) -> list[dict]:
    """For every measured run: the algorithm's k among the sizes that run
    measured, against the fastest measured arm that kept quality; and the
    latency-only choice against the fastest measured arm."""
    out = []
    for source in sorted({r["source"] for r in rows}):
        run = [r for r in rows if r["source"] == source]
        if len(run) < 2 or not any(r["k"] == 1 for r in run):
            continue
        first, by_k = run[0], {r["k"]: r for r in run}
        items, measured = items_of(first), sorted(by_k)
        pick = model.choose_k(first["kind"], items, first["max_num_seqs"], candidates=measured)
        fast = model.choose_k(first["kind"], items, first["max_num_seqs"], candidates=measured,
                              quality=False)
        ok = [r for r in run if quality_ok(r, model.f1_slack) is not False]
        best = min(ok, key=lambda r: r["latency_s"])
        fastest = min(run, key=lambda r: r["latency_s"])
        chosen = by_k[pick["k"]]
        free = model.choose_k(first["kind"], items, first["max_num_seqs"])
        out.append({"source": source, "workload": first["workload"], "kind": first["kind"],
                    "regime": items.regime, "items": first["items"],
                    "chosen_k": pick["k"], "best_k": best["k"],
                    "regret": chosen["latency_s"] / best["latency_s"] - 1,
                    "chosen_quality": quality_ok(chosen, model.f1_slack),
                    "latency_k": fast["k"], "fastest_k": fastest["k"],
                    "latency_regret": by_k[fast["k"]]["latency_s"] / fastest["latency_s"] - 1,
                    "unrestricted_k": free["k"]})
    return out


# --- Pipelines ---------------------------------------------------------------

def pipeline_of(r: dict, pass_share: float) -> Pipeline:
    t = r["tok"]
    return Pipeline(tuple(r["stages"].split(">")), t["left_rows"], t["left_mean"], t["left_max"],
                    pass_share, t["right_rows"], t["right_mean"], t["right_max"])


def passed_rows(r: dict, pipe: Pipeline) -> float | None:
    """Rows a pipeline's leading filter passed, from its call count: the
    filter makes ceil(rows / k) calls, every later call is the next stage's."""
    if pipe.ops[0] != "filter" or len(pipe.ops) != 2:
        return None
    k1, k2 = r["blocks_t"]
    items = max(r["llm_requests"] - math.ceil(pipe.rows / k1), 0) * k2
    return items / pipe.right_rows if pipe.ops[1] == "join" else items


def pipeline_runs(model: Model, rows: list[dict]) -> list[dict]:
    """Every pipeline arm with two predictions: from the rows its filter
    really passed (tests composition alone), and from per-tuple's pass share
    times rho(k) (the whole model, as the algorithm uses it)."""
    out = []
    for source in sorted({r["source"] for r in rows}):
        run = [r for r in rows if r["source"] == source]
        base = next((r for r in run if all(b == 1 for b in r["blocks_t"])), None)
        if base is None:
            continue
        probe = pipeline_of(base, 1.0)
        base_passed = passed_rows(base, probe)
        pipe = pipeline_of(base, base_passed / probe.rows if base_passed is not None else 1.0)
        for r in run:
            passed = passed_rows(r, pipe)
            comp = model.predict_pipeline(pipe, r["blocks_t"], r["max_num_seqs"], passed=passed)
            full = model.predict_pipeline(pipe, r["blocks_t"], r["max_num_seqs"])
            out.append({**r, "pipe": pipe, "passed": passed,
                        "stage_sum": sum(s["latency_s"] for s in comp["stages"]),
                        "stage_max": max(s["latency_s"] for s in comp["stages"]),
                        "pred_comp": comp["latency_s"], "pred_full": full["latency_s"]})
    return out


def fit_lambda(model: Model, runs: list[dict]) -> float:
    """lambda in L = g + lambda * sum(L_i - g), from the measured stage sizes."""
    if not runs:
        return 1.0
    x = [[r["stage_sum"] - len(r["pipe"].ops) * model.g] for r in runs]
    y = [r["latency_s"] - model.g for r in runs]
    return _lstsq(x, y, [1 / r["latency_s"] for r in runs])[0]


def pipeline_choices(model: Model, runs: list[dict]) -> list[dict]:
    out = []
    for source in sorted({r["source"] for r in runs}):
        run = [r for r in runs if r["source"] == source]
        pipe, seqs = run[0]["pipe"], run[0]["max_num_seqs"]
        cands = [r["blocks_t"] for r in run]
        by = {r["blocks_t"]: r for r in run}
        pick = model.choose_pipeline(pipe, seqs, candidates=cands)
        fast = model.choose_pipeline(pipe, seqs, candidates=cands, quality=False)
        ok = [r for r in run if quality_ok(r, model.f1_slack) is not False]
        best = min(ok, key=lambda r: r["latency_s"])
        fastest = min(run, key=lambda r: r["latency_s"])
        chosen = by[pick["blocks"]]
        out.append({"source": source, "workload": run[0]["workload"],
                    "chosen": pick["blocks"], "best": best["blocks_t"],
                    "regret": chosen["latency_s"] / best["latency_s"] - 1,
                    "chosen_quality": quality_ok(chosen, model.f1_slack),
                    "latency_choice": fast["blocks"], "fastest": fastest["blocks_t"],
                    "latency_regret": by[fast["blocks"]]["latency_s"] / fastest["latency_s"] - 1,
                    "unrestricted": model.choose_pipeline(pipe, seqs)["blocks"]})
    return out


# --- Report ------------------------------------------------------------------

def _pct(x):
    return f"{100 * x:.1f}%"


def _blocks(b) -> str:
    return ",".join(str(x) for x in b)


_YES = {True: "yes", False: "no", None: "–"}


def write_report(model: Model, rows: list[dict], lowo: list[float], picks: list[dict],
                 pipes: list[dict], pipe_picks: list[dict], out: Path) -> str:
    fit_err = [_err(r["pred_from_tokens"], r["latency_s"]) for r in rows]
    full_err = [_err(r["pred_full"], r["latency_s"]) for r in rows]
    tails = "; ".join(f"tail[{k}] = {v * 1e3:.1f} ms" for k, v in model.tail.items())
    lines = [
        "# Cost model", "",
        f"Fitted on {len(rows)} single-operator arms from "
        f"{len({r['source'] for r in rows})} runs ({', '.join(sorted({r['workload'] for r in rows}))}); "
        f"pipeline composition on {len(pipes)} pipeline arms.", "",
        "Latency `L = g + alpha P + (beta + tail[kind]) W D + gamma C D + delta C p^2` "
        "(P computed prompt tokens, W waves, D decode steps per call, C calls, p prompt "
        "tokens per call):", "",
        f"- g = {model.g:.3f} s; alpha = {model.alpha * 1e4:.3f} s per 10k computed tokens "
        f"({1 / model.alpha:,.0f} tokens/s); beta = {model.beta * 1e3:.1f} ms per decode step "
        f"of a wave; {tails}; gamma = {model.gamma * 1e3:.3f} ms per generated token; "
        f"delta = {model.delta * 1e6:.2f} ms per call per (1k prompt tokens)^2.",
        f"- Pipelines: L = g + lambda * sum over stages of (L_i - g), lambda = {model.lam:.2f}.",
        "",
        "| Error | Mean | Max |", "|---|---|---|",
        f"| Latency from measured tokens | {_pct(np.mean(fit_err))} | {_pct(max(fit_err))} |",
        f"| Full model (tokens predicted too) | {_pct(np.mean(full_err))} | {_pct(max(full_err))} |",
    ]
    if lowo:
        lines.append(f"| Full model, leave one workload out | {_pct(np.mean(lowo))} | "
                     f"{_pct(max(lowo))} |")
    if pipes:
        comp = [_err(r["pred_comp"], r["latency_s"]) for r in pipes]
        full = [_err(r["pred_full"], r["latency_s"]) for r in pipes]
        lines += [f"| Pipelines, measured filter output | {_pct(np.mean(comp))} | "
                  f"{_pct(max(comp))} |",
                  f"| Pipelines, full model (filter output from rho) | {_pct(np.mean(full))} | "
                  f"{_pct(max(full))} |"]
    lines += ["", "Per operator kind (tokens):", "",
              "| Kind | a_pt | q_pt | d_pt | e | o | c_left | q | d0 | d1 |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for kind, p in model.kinds.items():
        lines.append(f"| {kind} | {p.a_pt:.1f} | {p.q_pt:.1f} | {p.d_pt:.2f} | {p.e:.1f} | "
                     f"{p.o:.1f} | {p.c_left:.2f} | {p.q:.1f} | {p.d0:.2f} | {p.d1:.2f} |")
    lines += ["", f"Accuracy caps (largest k whose mean ΔF1 across runs stayed within "
                  f"{model.f1_slack}; maps: mean similarity at least {analyze.MIN_SIMILARITY}) "
                  f"and rows a block filter passes against per-tuple (rho). Long rows: items "
                  f"over {LONG_ROW_TOKENS} tokens.", "",
              "| Kind | Regime | Measured k | Cap k | rho by k |", "|---|---|---|---|---|"]
    for kind, p in model.kinds.items():
        for regime, measured in p.measured_k.items():
            rho = ", ".join(f"{k}: {v:.2f}" for k, v in p.rho.get(regime, {}).items()) or "–"
            lines.append(f"| {kind} | {regime} | {measured} | {p.cap(regime)} | {rho} |")
    lines += ["", "## Choice per measured single-operator run", "",
              "Chosen among the sizes the run measured. Best = fastest measured arm that kept "
              "quality. Latency-only = the choice with the accuracy cap off, scored against "
              "the fastest measured arm. Free = the choice over every k up to the caps.", "",
              "| Run | Regime | Items | Chosen k | Best k | Regret | Chosen kept quality | "
              "Latency-only k | Fastest k | Latency regret | Free k |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in picks:
        lines.append(f"| {c['source']} | {c['regime']} | {c['items']} | {c['chosen_k']} | "
                     f"{c['best_k']} | {_pct(c['regret'])} | {_YES[c['chosen_quality']]} | "
                     f"{c['latency_k']} | {c['fastest_k']} | {_pct(c['latency_regret'])} | "
                     f"{c['unrestricted_k']} |")
    if picks:
        lines += ["", f"Over {len(picks)} runs: regret mean "
                      f"{_pct(np.mean([c['regret'] for c in picks]))}, max "
                      f"{_pct(max(c['regret'] for c in picks))}; latency-only regret mean "
                      f"{_pct(np.mean([c['latency_regret'] for c in picks]))}, max "
                      f"{_pct(max(c['latency_regret'] for c in picks))}."]
    if pipes:
        lines += ["", "## Pipelines", "",
                  "Rows passed: by the leading filter, from the call count. Stage sum/max: "
                  "stage latencies predicted at those sizes. Composed: g + lambda * sum. Full: "
                  "filter output from per-tuple's pass share times rho(k).", "",
                  "| Run | Arm | Measured (s) | Rows passed | Stage sum (s) | Stage max (s) | "
                  "Composed (s) | Full (s) | ΔF1 |", "|---|---|---|---|---|---|---|---|---|"]
        for r in pipes:
            passed = "–" if r["passed"] is None else f"{r['passed']:.0f}"
            f1 = "–" if r.get("f1_delta") is None else f"{r['f1_delta']:+.2f}"
            lines.append(f"| {r['source']} | `{r['arm']}` | {r['latency_s']:.2f} | {passed} | "
                         f"{r['stage_sum']:.2f} | {r['stage_max']:.2f} | {r['pred_comp']:.2f} | "
                         f"{r['pred_full']:.2f} | {f1} |")
        lines += ["", "| Run | Chosen | Best | Regret | Chosen kept quality | Latency-only | "
                      "Fastest | Latency regret | Free |", "|---|---|---|---|---|---|---|---|---|"]
        for c in pipe_picks:
            lines.append(f"| {c['source']} | {_blocks(c['chosen'])} | {_blocks(c['best'])} | "
                         f"{_pct(c['regret'])} | {_YES[c['chosen_quality']]} | "
                         f"{_blocks(c['latency_choice'])} | {_blocks(c['fastest'])} | "
                         f"{_pct(c['latency_regret'])} | {_blocks(c['unrestricted'])} |")
    lines += ["", "## Every single-operator arm", "",
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
        {"g": model.g, "alpha": model.alpha, "beta": model.beta, "gamma": model.gamma,
         "delta": model.delta, "tail": model.tail, "lambda": model.lam,
         "f1_slack": model.f1_slack,
         "kinds": {k: asdict(v) for k, v in model.kinds.items()}}, indent=2), encoding="utf-8")
    return text


# --- Validation choices ------------------------------------------------------

def best_in_isolation(model: Model, rows: list[dict]) -> dict:
    """Per (kind, regime), the block size with the highest mean speedup over
    per-tuple across the measured runs, among sizes within the accuracy cap
    (the static "tune each operator alone" strategy); 1 if none is faster."""
    out = {}
    for kind, p in model.kinds.items():
        for regime in p.measured_k:
            speedups = {}
            for r in rows:
                if (r["kind"] == kind and _regime(r) == regime and 1 < r["k"] <= p.cap(regime)
                        and r.get("speedup")):
                    speedups.setdefault(r["k"], []).append(r["speedup"])
            best = max(speedups, key=lambda k: np.mean(speedups[k])) if speedups else 1
            out[(kind, regime)] = best if speedups and np.mean(speedups[best]) > 1 else 1
    return out


def per_tuple_pass_share(rows: list[dict]) -> float:
    """Per-tuple filter output rows over true positives, pooled over filter runs:
    turns a workload's selectivity into the share per-tuple passes."""
    runs = [r for r in rows if r["kind"] == "filter" and r["k"] == 1
            and r.get("selected_count") is not None and r.get("positives")]
    return (sum(r["selected_count"] for r in runs) / sum(r["positives"] for r in runs)
            if runs else 1.0)


def _grid(blocks: tuple, k_max: list[int]) -> list[tuple]:
    out = set()
    for i, k in enumerate(blocks):
        for g in {max(1, k // 2), max(1, k - 1), k + 1, 2 * k}:
            if g != k and g <= k_max[i]:
                out.add(blocks[:i] + (g,) + blocks[i + 1:])
    return sorted(out)


def validation_choices(model: Model, rows: list[dict], cache: dict) -> dict:
    """For each validation experiment: the algorithm's pick (quality-capped),
    its latency-only pick, best-in-isolation and a grid around both picks
    (for the oracle), as cloud/experiments.py reads them."""
    from .workloads import WORKLOADS
    iso = best_in_isolation(model, rows)
    calib = per_tuple_pass_share(rows)
    choices = {}
    for exp in _manifest().select("validate"):
        (part,) = exp["parts"]
        try:
            workload = WORKLOADS[part["workload"]](part["n"] or None)
        except KeyError:
            continue
        t = item_tokens(part["workload"], part["n"] or None, cache)
        if workload.stages:
            ops = tuple(s["op"] for s in workload.stages)
            if not set(ops) <= set(model.kinds):
                continue
            share = 1.0
            if ops[0] == "filter" and workload.positives is not None:
                left = {p[0] if isinstance(p, tuple) else p for p in workload.positives}
                share = len(left) / len(workload.rows) * calib
            pipe = Pipeline(ops, t["left_rows"], t["left_mean"], t["left_max"], share,
                            t["right_rows"], t["right_mean"], t["right_max"])
            stage_items = model.stage_items(pipe, pipe.rows)
            pick = model.choose_pipeline(pipe)
            fast = model.choose_pipeline(pipe, quality=False)
            blocks, fast_blocks = pick["blocks"], fast["blocks"]
            k_max = [model.k_max(op, it, quality=False) for op, it in zip(ops, stage_items)]
            iso_blocks = tuple(iso.get((op, it.regime), 1) for op, it in zip(ops, stage_items))
            predicted = {"pass_share": round(share, 3), "rows": pipe.rows}
        else:
            kind = workload.kind
            if kind not in model.kinds:
                continue
            joined = kind == "join"
            items = Items(len(workload.rows) * (t["right_rows"] or 1), t["mean"], t["max"],
                          t["left_rows"] if joined else 0, t["left_mean"] if joined else 0.0)
            pick = model.choose_k(kind, items)
            fast = model.choose_k(kind, items, quality=False)
            blocks, fast_blocks = (pick["k"],), (fast["k"],)
            k_max = [model.k_max(kind, items, quality=False)]
            iso_blocks = (iso.get((kind, items.regime), 1),)
            predicted = {"items": items.n, "item_tokens": items.tokens, "regime": items.regime}
        choices[exp["id"]] = {
            "algorithm": arm_name(blocks),
            "algorithm_latency": arm_name(fast_blocks),
            "best_in_isolation": arm_name(iso_blocks),
            "grid": [arm_name(g) for g in sorted(set(_grid(tuple(blocks), k_max))
                                                 | set(_grid(tuple(fast_blocks), k_max)))
                     if g not in (tuple(blocks), tuple(fast_blocks))],
            "predicted": {**predicted, "blocks": list(blocks),
                          "latency_s": round(pick["latency_s"], 3),
                          "latency_only": list(fast_blocks),
                          "latency_only_s": round(fast["latency_s"], 3)},
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
    # Validation runs test the model; they are not fitted on.
    collected = [r for r in analyze.collect(args.results) if r["phase"] != "validate"]
    rows = attach_tokens(_single_operator_rows(collected), cache)
    pipeline_rows = attach_tokens(_pipeline_rows(collected), cache)
    out.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache, indent=2, sort_keys=True), encoding="utf-8")
    if not rows:
        raise SystemExit("no single-operator Kalypso arms to fit")
    model = fit(rows, args.f1_slack)
    model.lam = fit_lambda(model, pipeline_runs(model, pipeline_rows))
    pipes = pipeline_runs(model, pipeline_rows)
    evaluated = evaluate(model, rows)
    write_report(model, evaluated, leave_one_workload_out(rows, args.f1_slack),
                 choices_per_run(model, evaluated), pipes, pipeline_choices(model, pipes), out)
    print(f"fitted on {len(rows)} arms and {len(pipes)} pipeline arms -> {out / 'costmodel.md'}")
    if args.write_choices:
        choices = validation_choices(model, rows, cache)
        (out / "algorithm_choices.json").write_text(json.dumps(choices, indent=2),
                                                    encoding="utf-8")
        cache_path.write_text(json.dumps(cache, indent=2, sort_keys=True), encoding="utf-8")
        for exp_id, c in choices.items():
            print(f"{exp_id}: {c['algorithm']} (predicted {c['predicted']['latency_s']} s), "
                  f"latency-only {c['algorithm_latency']}, best in isolation "
                  f"{c['best_in_isolation']}, grid {c['grid']}")


if __name__ == "__main__":
    main()
