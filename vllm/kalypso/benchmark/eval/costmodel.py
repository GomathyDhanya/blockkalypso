"""Cost model (v3) for operator batching, and the block-size choice built on it.

    python3 -m vllm.kalypso.benchmark.eval.costmodel [--results DIR] [--write-choices]

Fits on every Kalypso arm under ``eval/results/`` (see analyze.py) except the
held-out validation phase (``HELD_OUT``): the latency and token model on
single-operator arms, the composition factor and map reruns on pipeline arms.
Writes ``results/analysis/costmodel.md`` (coefficients, fit and
leave-one-workload-out error, the choice for every measured run against the
best measured arm), ``costmodel.json`` and, once validation runs exist,
``validation.md``. With ``--write-choices`` also ``algorithm_choices.json``:
the arms validation experiments not yet run will run (choices for experiments
already run are kept as they were).

Latency of one operator over N items (a join item is a pair) at block size k,
from C calls, W = ceil(C / max_num_seqs) waves, D decode steps per call, P
prompt tokens computed (not served from the prefix cache) and p prompt tokens
per call:

    L = g[kind] + alpha * P + (beta + tail[kind]) * W * D + gamma * C * D + delta * C * p^2

``g[kind]`` is a fixed cost per operator (it dominates small inputs);
``alpha`` prefill time per token; ``beta`` the time of one decode step of a
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

The choice (``choose_k``) is latency first: k from 1 up to the context-window
limit and the largest size measured for the operator's regime (short or long
rows); lowest predicted latency wins, and among sizes within ``tie`` of it
(about the model's error), the smallest blocks win. ``quality=True`` adds the
accuracy cap of the regime (the largest measured k whose mean F1 change
against per-tuple stayed within ``f1_slack``; maps: mean ROUGE-1 similarity
to per-tuple's outputs at least analyze.MIN_SIMILARITY), reported as the
guarded choice.

Pipelines (``choose_pipeline``): stages stream into each other on one GPU, so
their work adds up: L = lambda * sum_i L_i, lambda fitted on the pipeline
runs. A leading filter passes rows * (recall(k) s + fpr(k) (1 - s)) of its
rows for true selectivity s, recall and false-positive rate measured per
block size on the single-filter runs (block filters have higher recall, so
they pass more rows and every later stage does more work). A per-tuple map in
a pipeline with a blocked stage reruns a share of its rows (``retry``):
SemMap sizes each answer's budget from the answers finished so far, which
are the shortest, so when the map's rows arrive in waves many are cut off and
run again at 512 tokens.
"""

import argparse
import bisect
import copy
import hashlib
import itertools
import json
import math
from collections import OrderedDict
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np

from . import analyze

RESULTS = analyze.RESULTS
TOKENIZER = "Qwen/Qwen2.5-7B-Instruct"
# Hub commit of TOKENIZER that the Session 1 GPU runs resolved (all 22
# research/baseline/results/gpu/*/env.json records). Pinned so token counts
# do not move if the Hub's main branch does.
TOKENIZER_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
DEFAULT_MAX_MODEL_LEN = 32768
# Items above this many tokens are "long rows" (ContractNLI, ~1.6k): a regime
# with its own accuracy cap.
LONG_ROW_TOKENS = 512
# Operators whose answers vary in length, so a wave lasts its longest call.
TAIL_KINDS = ("map",)
# Largest size considered when the accuracy cap is off and the regime has no
# measured sizes.
MAX_K = 32
# Block-size candidate sets for choose_pipeline(candidate_set=...).
# "exhaustive" (the default) searches every k in 1..k_max per stage. "pow2"
# searches only these sizes; research/baseline/results/optimizer_scaling.md
# measured it against the exhaustive pick on 1-4 stage pipelines (regret
# -1.7% to +2.3%) and it plans 6 stages in under 2 s.
POW2_CANDIDATES = (1, 2, 4, 8, 16, 32)
CANDIDATE_SETS = ("exhaustive", "pow2")


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


# --- Join quality caps by selectivity band (Phase 2 step A) ---------------------
#
# The pooled per-(kind, regime) cap put joins at k = 1 because it averaged
# workloads whose block-join quality moves in opposite directions with join
# selectivity (research/docs/QUALITY_CAP_DECISION.md). Joins now also get a cap
# per (regime, selectivity band), fitted the same way as ``_cap`` (grow k from
# 2, stop at the first failing k) over per-workload mean ΔF1 within the band.
# Two guards against fitting noise (repeat-level ΔF1 variance is unknown):
# - one edge only, at 0.6: between the 0.47-0.50 joins (mixed, mostly failing)
#   and the 0.65-0.91 joins. No edge is fitted inside the 0.47-0.50 cluster;
# - a k counts only when at least MIN_BAND_WORKLOADS workloads in the band were
#   measured there; fewer stops the band at the previous k.
# Callers that pass no selectivity get the pooled cap, unchanged. Maps and
# filters have no band caps.
#
# Reverted (Phase 2, after P2-bandcap-resample): the fit above gives the
# selectivity >= 0.6 short-row band k = 6 from movie_fanout75/90. On 8 fresh,
# disjoint 32 x 32 samples of the same two workloads, k = 6 failed the F1
# check on 7 and k = 4 on 4, with negative mean ΔF1 in both. The original
# draws were favourable outliers. Every band cap is now held at
# BAND_CAP_CEILING = 1, the pooled join cap, pending better evidence. The
# machinery (bands, band_caps, the controller's selectivity-aware veto)
# stays. Raising the ceiling restores the raw fit.

JOIN_SELECTIVITY_EDGES = (0.6,)
MIN_BAND_WORKLOADS = 2
BAND_CAP_CEILING = 1


def selectivity_band(selectivity: float) -> str:
    return str(bisect.bisect_right(JOIN_SELECTIVITY_EDGES, selectivity))


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
    band_caps: dict = field(default_factory=dict)   # joins: regime -> {band: cap k}
    measured_k: dict = field(default_factory=dict)  # regime -> measured sizes
    recall: dict = field(default_factory=dict)      # filters: regime -> {k: recall}
    fpr: dict = field(default_factory=dict)         # filters: regime -> {k: false-positive rate}

    def cap(self, regime: str, selectivity: float | None = None) -> int:
        bands = self.band_caps.get(regime) if selectivity is not None else None
        if bands:
            return bands.get(selectivity_band(selectivity), 1)
        return self.caps.get(regime, 1)

    @staticmethod
    def _interp(points: dict, k: int, default: float) -> float:
        """Per-tuple's value at k = 1; for k > 1 interpolated between the
        measured block sizes, flat past the ends. Blocking changes recall as
        soon as rows share a prompt (k = 2 passes about what k = 4 does), so
        k = 2 takes the smallest measured block size's value rather than a
        blend with per-tuple."""
        pts = sorted((int(x), y) for x, y in points.items() if (int(x) == 1) == (k == 1))
        if not pts:
            return default
        return float(np.interp(k, [x for x, _ in pts], [y for _, y in pts]))

    def pass_share(self, regime: str, k: int, selectivity: float) -> float:
        """Share of rows a filter at block size k passes when a share
        ``selectivity`` of them truly match: recall * s + fpr * (1 - s)."""
        r = self._interp(self.recall.get(regime, {}), k, 1.0)
        f = self._interp(self.fpr.get(regime, {}), k, 0.0)
        return r * selectivity + f * (1 - selectivity)



@dataclass
class Pipeline:
    ops: tuple                 # e.g. ("filter", "join")
    rows: float                # input rows
    tokens: float              # mean row tokens
    max_tokens: float
    selectivity: float         # share of rows the leading filter truly matches
    right_rows: float = 0      # joins
    right_tokens: float = 0.0
    right_max_tokens: float = 0.0
    # True share of each stage's input that matches, when known (join ->
    # filter); else ``selectivity`` is the leading filter's.
    stage_selectivity: tuple = ()

    def true_share(self, i: int) -> float:
        if self.stage_selectivity:
            return self.stage_selectivity[i]
        return self.selectivity if i == 0 else 1.0


# --- Decision cache -----------------------------------------------------------
#
# A small bounded LRU of choose_k / choose_pipeline results. Activation
# (Phase 2 step D): choose_pipeline uses it automatically for pipelines with
# more than AUTO_CACHE_MIN_STAGES - 1 stages, keyed on the exact inputs so a
# result is bit-identical to an uncached call; 1-3 stages skip it (the key
# costs more than the search). ``cache=True`` still forces it with bucketed
# inputs (below) and ``cache=False`` turns it off; choose_k never uses it
# unless ``cache=True``. It covers
# the plan's "cache-based amortization" idea for completeness. It is NOT a
# fix for a measured bottleneck:
#
# - The search is only called offline (analysis, experiment choices), never
#   on a query's path (BASELINE_AUDIT.md finding A).
# - Measured search cost (research/baseline/results/optimizer_scaling.md):
#   choose_k and 1-stage pipelines 0.1-0.3 ms; 2 stages 9-11 ms; pow2 4-6 ms
#   at 3 stages and 28-38 ms at 4. Only exhaustive search at 3+ stages is
#   slow (0.35 s at 3, 8.6-20 s at 4), and candidate_set="pow2" already
#   addresses that.
# - A key includes the model's content fingerprint (~0.1 ms to compute),
#   which costs more than a single choose_k call (~15 us). A cached choose_k
#   is therefore slower than an uncached one. Only multi-stage searches can
#   come out ahead.
#
# Key: the model fingerprint (a refit, or any edit to its parameters, misses),
# the call's own arguments and the inputs, bucketed so near-identical calls
# hit: item counts and token figures rounded to whole numbers, selectivities
# to 3 decimals. The regime is in the key explicitly. On a miss the result is
# computed on those bucketed inputs, so what a key returns does not depend on
# which call filled it. Nothing about runtime scheduler or KV state is in the
# key or the value: it caches the model's prediction for its inputs only.

DECISION_CACHE_MAX_SIZE = 1024
AUTO_CACHE_MIN_STAGES = 4  # choose_pipeline caches automatically from 4 stages


@dataclass
class DecisionCacheStats:
    hits: int = 0
    misses: int = 0
    evictions: int = 0

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


class DecisionCache:
    def __init__(self, max_size: int = DECISION_CACHE_MAX_SIZE):
        self.max_size = max_size
        self.stats = DecisionCacheStats()
        self._entries: OrderedDict = OrderedDict()

    def __len__(self) -> int:
        return len(self._entries)

    def clear(self) -> None:
        self._entries.clear()
        self.stats = DecisionCacheStats()

    def get_or_compute(self, key, compute):
        if key in self._entries:
            self._entries.move_to_end(key)
            self.stats.hits += 1
            return copy.deepcopy(self._entries[key])
        self.stats.misses += 1
        value = compute()
        self._entries[key] = value
        if len(self._entries) > self.max_size:
            self._entries.popitem(last=False)
            self.stats.evictions += 1
        return copy.deepcopy(value)


DECISION_CACHE = DecisionCache()


def _bucket_items(items: Items) -> Items:
    return Items(n=round(items.n), tokens=round(items.tokens),
                 max_tokens=round(items.max_tokens), left_rows=round(items.left_rows),
                 left_tokens=round(items.left_tokens))


def _bucket_pipeline(pipe: Pipeline) -> Pipeline:
    return Pipeline(tuple(pipe.ops), round(pipe.rows), round(pipe.tokens),
                    round(pipe.max_tokens), round(pipe.selectivity, 3), round(pipe.right_rows),
                    round(pipe.right_tokens), round(pipe.right_max_tokens),
                    tuple(round(s, 3) for s in pipe.stage_selectivity))


# --- KV feasibility gate -------------------------------------------------------
#
# A sanity check, not a resource model. With ``kv_budget_tokens`` set,
# choose_k / choose_pipeline drop any block size k > 1 for which ONE call's
# KV footprint, predict()'s prompt_per_call + decode_steps, exceeds the
# budget. Rejected sizes are listed in the result's ``kv_rejected`` field.
# k = 1 is never rejected (per-tuple is the fallback; a row that cannot fit
# alone is not this gate's problem).
#
# Why it stops here: Session 1 found KV was never the binding constraint at
# this scale (peak KV <= 17.5%, 22% under load, 0 preemptions;
# BASELINE_RESULTS.md). Queueing was the visible pressure. So there is no
# KV-cost term in the latency model, and no concurrency, admission or
# double-counting modelling here. This only catches pathological inputs,
# e.g. huge rows with a large k against a small stage budget. It uses mean
# row tokens (predict's estimate), so it rejects only clear overflows. The
# context-window bound already exists separately as k_ctx (worst-case row
# tokens vs max_model_len) inside k_max. Default None: no gate, results
# unchanged.


def kv_budget_tokens(stage_id: int | None = None) -> float:
    """A KV budget in tokens from Kalypso's KVMemoryManager: the registered
    capacity of ``stage_id``, or the whole Kalypso budget when None. It reads
    the static capacity, not live usage."""
    from vllm.kalypso.budget import KVMemoryManager

    mgr = KVMemoryManager.get_instance()
    capacity = mgr.capacity() if stage_id is None else mgr.stage_usage(stage_id)[1]
    return capacity / mgr.bytes_per_token


# --- Adaptive selectivity (Phase 2 step C; OFF by default) -----------------------
#
# Plumbing only: Model.observe_selectivity nudges a stored per-stage match-rate
# estimate toward an observed value (EWMA, weight SELECTIVITY_EWMA_WEIGHT), and
# predict_pipeline / choose_pipeline use the stored estimates only when called
# with ``adaptive_selectivity`` set to True. Nothing in the repo sets it.
#
# Why it stays off: the model's current error (held-out validation,
# research/docs/RESOURCE_MODEL.md and validation.md) was measured with the
# offline selectivities. A model that updates its own inputs from live
# observations is a moving target, and its error needs its own evaluation
# before anyone relies on it. That evaluation is explicitly not done here.
# With the flag off, every call is bit-for-bit what it was without this code.

SELECTIVITY_EWMA_WEIGHT = 0.2


def selectivity_key(ops: tuple, stage: int) -> str:
    return f"{'+'.join(ops)}#{stage}"


@dataclass
class Model:
    g: dict                # kind -> fixed seconds per operator (stage)
    alpha: float
    beta: float
    gamma: float
    delta: float
    tail: dict             # kind -> extra seconds per decode step of a wave
    kinds: dict            # kind -> KindParams
    f1_slack: float = 0.02
    lam: float = 1.0       # pipelines: share of the stages' summed time not overlapped
    # Pipelines: extra per-row map calls (reruns of cut-off answers, repairs of
    # rows a block answer missed) per map row, by the map's position and
    # whether it is blocked: "first:1", "first:block", "after:1", "after:block".
    retry: dict = field(default_factory=dict)
    # Adaptive selectivity (off unless a call sets adaptive_selectivity to True):
    # selectivity_key(ops, stage) -> EWMA of observed match rates.
    selectivity_estimates: dict = field(default_factory=dict)

    def observe_selectivity(self, ops: tuple, stage: int, observed: float,
                            prior: float | None = None,
                            weight: float = SELECTIVITY_EWMA_WEIGHT) -> float:
        """Move the stored estimate for (ops, stage) a share ``weight`` of the
        way toward ``observed``; the first observation starts from ``prior``
        (the offline value), or from ``observed`` when no prior is given."""
        if not 0.0 < weight <= 1.0:
            raise ValueError("weight must be in (0, 1]")
        key = selectivity_key(tuple(ops), stage)
        old = self.selectivity_estimates.get(key, observed if prior is None else prior)
        new = old + weight * (observed - old)
        self.selectivity_estimates[key] = new
        return new

    def adapted_pipeline(self, pipe: Pipeline) -> Pipeline:
        """``pipe`` with each stage's true share replaced by its stored
        estimate where there is one; ``pipe`` itself when there is none."""
        ops = tuple(pipe.ops)
        keys = [selectivity_key(ops, i) for i in range(len(ops))]
        if not any(k in self.selectivity_estimates for k in keys):
            return pipe
        return replace(pipe, stage_selectivity=tuple(
            self.selectivity_estimates.get(k, pipe.true_share(i)) for i, k in enumerate(keys)))

    def fingerprint(self) -> str:
        """Identity of this fit: a hash of every parameter. Recomputed per
        call because fitting code mutates a Model in place (main() sets
        ``retry`` and ``lam`` after ``fit``)."""
        blob = json.dumps(asdict(self), sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    def latency(self, kind, computed, waves, decode, calls, per_call) -> float:
        return (self.g.get(kind, 0.0) + self.alpha * computed
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

    def k_max(self, kind: str, items: Items, quality: bool = False,
              max_model_len: int = DEFAULT_MAX_MODEL_LEN,
              selectivity: float | None = None) -> int:
        """``selectivity``: the operator's own match rate, when known; with
        ``quality`` it selects a join's selectivity-band cap."""
        p = self.kinds[kind]
        # Without the accuracy cap, stay within the sizes measured for the
        # regime: the model is not checked beyond them.
        cap = (p.cap(items.regime, selectivity) if quality
               else max(p.measured_k.get(items.regime) or [MAX_K]))
        return max(1, min(self.k_ctx(kind, items.max_tokens, max_model_len), cap,
                          max(int(items.n), 1)))

    def call_kv_tokens(self, kind: str, items: Items, k: int) -> float:
        """KV tokens one call at block size k holds: prompt plus answer."""
        pred = self.predict(kind, items, k)
        return pred["prompt_per_call"] + pred["decode_steps"]

    def kv_fits(self, kind: str, items: Items, k: int, kv_budget_tokens: float | None) -> bool:
        """The KV feasibility gate (see the note above Model)."""
        return (kv_budget_tokens is None or k == 1
                or self.call_kv_tokens(kind, items, k) <= kv_budget_tokens)

    def choose_k(self, kind: str, items: Items,
                 max_num_seqs: int = analyze.DEFAULT_MAX_NUM_SEQS,
                 max_model_len: int = DEFAULT_MAX_MODEL_LEN, tie: float = 0.02,
                 candidates=None, quality: bool = False, cache: bool | None = None,
                 kv_budget_tokens: float | None = None,
                 selectivity: float | None = None) -> dict:
        if cache:
            b = _bucket_items(items)
            cands = tuple(candidates) if candidates is not None else None
            band = None if selectivity is None else selectivity_band(selectivity)
            key = ("choose_k", self.fingerprint(), kind, b.regime, b.n, b.tokens,
                   b.max_tokens, b.left_rows, b.left_tokens, max_num_seqs, max_model_len,
                   tie, cands, quality, kv_budget_tokens, band)
            return DECISION_CACHE.get_or_compute(key, lambda: self.choose_k(
                kind, b, max_num_seqs, max_model_len, tie, cands, quality, cache=False,
                kv_budget_tokens=kv_budget_tokens, selectivity=selectivity))
        k_max = self.k_max(kind, items, quality, max_model_len, selectivity)
        ks = [k for k in (candidates or range(1, k_max + 1)) if k <= k_max] or [1]
        rejected = [k for k in ks if not self.kv_fits(kind, items, k, kv_budget_tokens)]
        ks = [k for k in ks if k not in rejected] or [1]
        preds = [self.predict(kind, items, k, max_num_seqs) for k in ks]
        pick = _pick(preds, tie)
        out = {**pick, "k_max": k_max,
               "fastest_k": min(preds, key=lambda x: x["latency_s"])["k"]}
        if kv_budget_tokens is not None:
            out["kv_rejected"] = rejected
        return out

    # --- Pipelines ------------------------------------------------------------

    def stage_items(self, pipe: Pipeline, rows: float) -> list[Items]:
        """Each stage's input for ``rows`` rows reaching every stage (the sizes
        are filled in by predict_pipeline; this gives tokens and regimes)."""
        out, tokens, max_tokens = [], pipe.tokens, pipe.max_tokens
        for op in pipe.ops:
            if op == "join":
                out.append(Items(rows * pipe.right_rows, tokens + pipe.right_tokens,
                                 max_tokens + pipe.right_max_tokens, rows, tokens))
                # Later stages see each pair as one row.
                tokens += pipe.right_tokens
                max_tokens += pipe.right_max_tokens
            else:
                out.append(Items(rows, tokens, max_tokens))
            if op == "map":
                # The next stage sees each row with its map output.
                tokens += self.kinds["map"].d_pt
                max_tokens += self.kinds["map"].d_pt
        return out

    def retry_share(self, pipe: Pipeline, blocks: tuple, i: int) -> float:
        """Extra per-row calls a pipeline's map makes per row (see retry_key)."""
        key = retry_key(pipe.ops, blocks, i)
        return self.retry.get(key, 0.0) if key else 0.0

    def predict_pipeline(self, pipe: Pipeline, blocks: tuple,
                         max_num_seqs: int = analyze.DEFAULT_MAX_NUM_SEQS,
                         passed: float | None = None, retried: float | None = None,
                         adaptive_selectivity: bool = False) -> dict:
        """``passed``/``retried``: rows the leading filter passed and map rows
        rerun, when known (measured); otherwise predicted from the filter's
        recall and false-positive rate at its block size and the retry share.
        ``adaptive_selectivity``: see the note above Model (default off)."""
        if adaptive_selectivity:
            pipe = self.adapted_pipeline(pipe)
        rows, stages = pipe.rows, []
        for i, (op, k) in enumerate(zip(pipe.ops, blocks)):
            items = self.stage_items(pipe, rows)[i]
            stage = self.predict(op, items, k, max_num_seqs)
            share = self.retry_share(pipe, blocks, i)
            again = retried if retried is not None and op == "map" else share * rows
            if again:
                # Reruns pay their work again, not the stage's fixed cost.
                extra = self.predict(op, Items(again, items.tokens, items.max_tokens), 1,
                                     max_num_seqs)
                stage = {**stage, "retried": again,
                         "latency_s": stage["latency_s"] + extra["latency_s"] - self.g[op]}
            stages.append(stage)
            if op == "filter":
                rows = (passed if passed is not None else
                        rows * self.kinds["filter"].pass_share(items.regime, k,
                                                               pipe.true_share(i)))
            elif op == "join" and i + 1 < len(pipe.ops):
                # Pairs the join keeps, from its recall and false-positive rate.
                rows = items.n * self.kinds["join"].pass_share(items.regime, k,
                                                                pipe.true_share(i))
        latency = self.lam * sum(s["latency_s"] for s in stages)
        return {"blocks": tuple(blocks), "stages": stages, "latency_s": latency,
                "calls": sum(s["calls"] for s in stages)}

    def choose_pipeline(self, pipe: Pipeline, max_num_seqs: int = analyze.DEFAULT_MAX_NUM_SEQS,
                        tie: float = 0.02, candidates=None, quality: bool = False,
                        candidate_set: str = "exhaustive", cache: bool | None = None,
                        kv_budget_tokens=None, adaptive_selectivity: bool = False) -> dict:
        """``candidate_set`` picks the generated per-stage block sizes when no
        explicit ``candidates`` are given: "exhaustive" (default, 1..k_max) or
        "pow2" (POW2_CANDIDATES up to k_max). ``cache``: None (default) caches
        only pipelines of AUTO_CACHE_MIN_STAGES or more, on exact inputs; True
        always, on bucketed inputs; False never. See DECISION_CACHE.
        ``kv_budget_tokens``: one budget for every stage, or one per stage
        (see the KV feasibility gate note above Model).
        ``adaptive_selectivity``: see the note above Model (default off)."""
        if adaptive_selectivity:
            pipe = self.adapted_pipeline(pipe)
        if cache is None:
            cache, bucket = len(pipe.ops) >= AUTO_CACHE_MIN_STAGES, False
        else:
            bucket = cache
        if cache:
            b = (_bucket_pipeline(pipe) if bucket
                 else replace(pipe, ops=tuple(pipe.ops),
                              stage_selectivity=tuple(pipe.stage_selectivity)))
            cands = tuple(tuple(c) for c in candidates) if candidates is not None else None
            regimes = tuple(it.regime for it in self.stage_items(b, b.rows))
            key = ("choose_pipeline", bucket, self.fingerprint(), b.ops, regimes, b.rows, b.tokens,
                   b.max_tokens, b.selectivity, b.right_rows, b.right_tokens,
                   b.right_max_tokens, b.stage_selectivity, max_num_seqs, tie, quality,
                   candidate_set, cands,
                   tuple(kv_budget_tokens) if isinstance(kv_budget_tokens, (list, tuple))
                   else kv_budget_tokens)
            return DECISION_CACHE.get_or_compute(key, lambda: self.choose_pipeline(
                b, max_num_seqs, tie, cands, quality, candidate_set, cache=False,
                kv_budget_tokens=kv_budget_tokens))
        if candidate_set not in CANDIDATE_SETS:
            raise ValueError(f"candidate_set must be one of {CANDIDATE_SETS}, got {candidate_set!r}")
        if candidates is not None and candidate_set != "exhaustive":
            raise ValueError("pass either explicit candidates or candidate_set, not both")
        stage_items = self.stage_items(pipe, pipe.rows)
        k_max = [self.k_max(op, items, quality) for op, items in zip(pipe.ops, stage_items)]
        if candidates is None and candidate_set == "pow2":
            # Same set (and order) as the full POW2 product with over-k_max
            # combinations dropped, which is how optimizer_scaling.py ran it.
            candidates = itertools.product(*[[k for k in POW2_CANDIDATES if k <= m]
                                             for m in k_max])
        if candidates is None:
            candidates = itertools.product(*[range(1, k + 1) for k in k_max])
        candidates = [tuple(c) for c in candidates
                      if all(b <= m for b, m in zip(c, k_max))] or [(1,) * len(pipe.ops)]
        rejected = []
        if kv_budget_tokens is not None:
            budgets = (list(kv_budget_tokens) if isinstance(kv_budget_tokens, (list, tuple))
                       else [kv_budget_tokens] * len(pipe.ops))
            # A call's prompt per call and decode steps depend on the stage's
            # row tokens and k only, not on cardinality, so each (stage, k)
            # is checked once, independent of the other stages' sizes.
            fits = {}
            for c in candidates:
                for i, k in enumerate(c):
                    if (i, k) not in fits:
                        fits[(i, k)] = self.kv_fits(pipe.ops[i], stage_items[i], k, budgets[i])
            rejected = sorted(ik for ik, ok in fits.items() if not ok)
            candidates = ([c for c in candidates if all(fits[(i, k)] for i, k in enumerate(c))]
                          or [(1,) * len(pipe.ops)])
        preds = [self.predict_pipeline(pipe, c, max_num_seqs) for c in candidates]
        pick = _pick(preds, tie)
        out = {**pick, "fastest": min(preds, key=lambda x: x["latency_s"])["blocks"]}
        if kv_budget_tokens is not None:
            out["kv_rejected"] = rejected  # (stage index, k) pairs dropped by the gate
        return out


def retry_key(ops: tuple, blocks: tuple, i: int) -> str | None:
    """Which extra-call share applies to stage i, None if none does. A
    per-tuple map reruns answers it cut off only when another stage is
    blocked: SemMap sizes each answer's budget from the answers finished so
    far (controller/map_estimator.py), the shortest ones, and its rows then
    arrive in waves. A block map also repairs rows its block answer missed."""
    if ops[i] != "map" or len(ops) < 2 or all(b == 1 for b in blocks):
        return None
    return f"{'first' if i == 0 else 'after'}:{'block' if blocks[i] > 1 else 1}"


def _pick(preds: list[dict], tie: float) -> dict:
    """Lowest predicted latency; among sizes within ``tie`` of it, the smallest
    blocks (measured latency rises with k past the best size faster than the
    model's flat optimum suggests, and smaller blocks stay nearer the data)."""
    best = min(x["latency_s"] for x in preds)
    close = [x for x in preds if x["latency_s"] <= best + tie * abs(best)]
    return min(close, key=lambda x: (sum(x["blocks"]) if "blocks" in x else x["k"],
                                     x["latency_s"]))


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
        _tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, revision=TOKENIZER_REVISION)
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
    """Did the arm keep per-tuple's quality? F1 for labeled workloads and output
    similarity for maps (a filter -> map pipeline needs both); None when
    neither was measured."""
    checks = []
    if r.get("f1_delta") is not None:
        checks.append(r["f1_delta"] >= -f1_slack)
    if r.get("similarity") is not None:
        checks.append(r["similarity"] >= analyze.MIN_SIMILARITY)
    return all(checks) if checks else None


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


def _band_cap(rows: list[dict], measured: list[int], f1_slack: float) -> int:
    """``_cap`` within one join selectivity band: the mean of per-workload mean
    ΔF1 at each k, and a k with fewer than MIN_BAND_WORKLOADS measured
    workloads ends the band (too little evidence to raise it further)."""
    cap = 1
    for k in measured:
        if k == 1:
            continue
        per = {}
        for r in rows:
            if r["k"] == k and r.get("f1_delta") is not None:
                per.setdefault(r["workload"], []).append(r["f1_delta"])
        if not per:
            continue
        if len(per) < MIN_BAND_WORKLOADS:
            break
        if np.mean([np.mean(v) for v in per.values()]) < -f1_slack:
            break
        cap = k
    return cap


def fit_band_caps(rows: list[dict], f1_slack: float) -> dict:
    """Join caps per (regime, selectivity band), held at BAND_CAP_CEILING; see
    the note above KindParams.
    A workload's selectivity is positives / items of its per-tuple run."""
    sel = {r["workload"]: r["positives"] / r["items"] for r in rows
           if r["k"] == 1 and r.get("positives") is not None and r.get("items")}
    out = {}
    for regime in ("short", "long"):
        at = [r for r in rows if _regime(r) == regime and r["workload"] in sel]
        if not at:
            continue
        measured = sorted({r["k"] for r in at})
        bands = sorted({selectivity_band(sel[r["workload"]]) for r in at})
        out[regime] = {b: min(BAND_CAP_CEILING,
                              _band_cap([r for r in at if selectivity_band(sel[r["workload"]]) == b],
                                        measured, f1_slack)) for b in bands}
    return out


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
        # Filters: recall and false-positive rate per k, pooled over runs
        # (true positives over positives, false positives over negatives), so
        # the rows a filter passes follow from any selectivity.
        sums = {}
        for r in at:
            if r.get("recall") is None or r.get("selected_count") is None or not r.get("positives"):
                continue
            tp = r["recall"] * r["positives"]
            fp = max(r["selected_count"] - tp, 0.0)
            a = sums.setdefault(r["k"], [0.0, 0.0, 0.0, 0.0])
            a[0] += tp
            a[1] += r["positives"]
            a[2] += fp
            a[3] += r["items"] - r["positives"]
        if sums:
            p.recall[regime] = {str(k): round(a[0] / a[1], 4) for k, a in sorted(sums.items())}
            p.fpr[regime] = {str(k): round(a[2] / a[3], 4) if a[3] else 0.0
                             for k, a in sorted(sums.items())}
    if rows and all(r.get("kind") == "join" for r in rows):
        p.band_caps = fit_band_caps(rows, f1_slack)
    return p


def latency_features(r: dict, kinds: tuple, tails=TAIL_KINDS) -> list[float]:
    """Measured features of one arm, in Model.latency's order: a fixed cost
    per operator kind, then the shared terms."""
    wd = r["waves"] * r["gen_tokens_per_call"]
    return [*[float(r["kind"] == kind) for kind in kinds], r["computed_prompt_tokens"], wd,
            *[wd * (r["kind"] == kind) for kind in tails],
            r["llm_requests"] * r["gen_tokens_per_call"],
            r["llm_requests"] * r["prompt_tokens_per_call"] ** 2 / 1e3]


def fit(rows: list[dict], f1_slack: float = 0.02) -> Model:
    """Least squares on relative error, so short runs count as much as long ones."""
    kinds = tuple(sorted({r["kind"] for r in rows}))
    tails = tuple(kind for kind in TAIL_KINDS if kind in kinds)
    coef = _lstsq([latency_features(r, kinds, tails) for r in rows],
                  [r["latency_s"] for r in rows], [1 / r["latency_s"] for r in rows])
    g = dict(zip(kinds, coef[:len(kinds)]))
    alpha, beta, *rest = coef[len(kinds):]
    tail = dict(zip(tails, rest[:len(tails)]))
    gamma, delta = rest[len(tails):]
    params = {kind: fit_kind([r for r in rows if r["kind"] == kind], f1_slack) for kind in kinds}
    return Model(g, alpha, beta, gamma, delta, tail, params, f1_slack)


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
    measured, against the fastest measured arm; and the accuracy-guarded
    choice against the fastest measured arm that kept quality."""
    out = []
    for source in sorted({r["source"] for r in rows}):
        run = [r for r in rows if r["source"] == source]
        if len(run) < 2 or not any(r["k"] == 1 for r in run):
            continue
        first, by_k = run[0], {r["k"]: r for r in run}
        items, measured = items_of(first), sorted(by_k)
        pick = model.choose_k(first["kind"], items, first["max_num_seqs"], candidates=measured,
                              quality=True)
        fast = model.choose_k(first["kind"], items, first["max_num_seqs"], candidates=measured)
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

def pipeline_of(r: dict, selectivity: float, stage_selectivity: tuple = ()) -> Pipeline:
    t = r["tok"]
    return Pipeline(tuple(r["stages"].split(">")), t["left_rows"], t["left_mean"], t["left_max"],
                    selectivity, t["right_rows"], t["right_mean"], t["right_max"],
                    tuple(stage_selectivity))


def passed_rows(r: dict, pipe: Pipeline) -> float | None:
    """Rows a pipeline's leading filter passed. Filter -> map: the rows that
    came out (the map keeps every row). Filter -> join: from the call count;
    the filter makes ceil(rows / k) calls and every later call is the join's."""
    if pipe.ops[0] != "filter" or len(pipe.ops) != 2:
        return None
    if pipe.ops[1] == "map":
        return r.get("selected_count")
    k1, k2 = r["blocks_t"]
    calls = r["llm_requests"] - math.ceil(pipe.rows / k1)
    items = max(calls, 0) * k2
    return items / pipe.right_rows if pipe.ops[1] == "join" else items


def expected_calls(r: dict, pipe: Pipeline, passed: float | None) -> float | None:
    """Calls a two-stage pipeline arm makes with no reruns or repairs."""
    if len(pipe.ops) != 2:
        return None
    k1, k2 = r["blocks_t"]
    second = pipe.rows if pipe.ops[0] != "filter" else passed
    if second is None:
        return None
    if pipe.ops[1] == "join":
        second *= pipe.right_rows
    return math.ceil(pipe.rows / k1) + math.ceil(second / k2)


def retried_rows(r: dict, pipe: Pipeline) -> float | None:
    """Extra per-row map calls: calls beyond those of the stages' blocks."""
    if "map" not in pipe.ops:
        return None
    expected = expected_calls(r, pipe, passed_rows(r, pipe))
    if expected is None:
        return r.get("truncated")
    return max(r["llm_requests"] - expected, 0)


def workload_selectivity(workload: str, n: int | None, cache: dict) -> float:
    """Share of a pipeline's rows its leading filter truly matches: the
    filter's positives (filter -> join: left rows of a matching pair, which
    are the filter's positives); 1 without labels."""
    key = f"{workload}:{n or ''}"
    entry = cache.setdefault(key, {})
    if "selectivity" not in entry:
        from .workloads import WORKLOADS
        w = WORKLOADS[workload](n or None)
        if w.positives is None:
            entry["selectivity"] = 1.0
        else:
            left = {p[0] if isinstance(p, tuple) else p for p in w.positives}
            entry["selectivity"] = round(len(left) / len(w.rows), 4)
    return entry["selectivity"]


def workload_stage_selectivity(workload: str, n: int | None, cache: dict) -> tuple:
    """Per-stage true selectivity when the workload records it, else ()."""
    entry = cache.setdefault(f"{workload}:{n or ''}", {})
    if "stage_selectivity" not in entry:
        from .workloads import WORKLOADS
        entry["stage_selectivity"] = [round(x, 4) for x in
                                      WORKLOADS[workload](n or None).stage_selectivity or ()]
    return tuple(entry["stage_selectivity"])


def pipeline_runs(model: Model, rows: list[dict], cache: dict) -> list[dict]:
    """Every pipeline arm with two predictions: from the rows its filter really
    passed and the map calls really repeated (tests composition alone), and
    from the workload's selectivity (the whole model, as the algorithm uses
    it)."""
    out = []
    for source in sorted({r["source"] for r in rows}):
        run = [r for r in rows if r["source"] == source]
        base = run[0]
        pipe = pipeline_of(base, workload_selectivity(base["workload"], _n_arg(base), cache),
                           workload_stage_selectivity(base["workload"], _n_arg(base), cache))
        for r in run:
            passed = passed_rows(r, pipe)
            comp = model.predict_pipeline(pipe, r["blocks_t"], r["max_num_seqs"], passed=passed,
                                          retried=retried_rows(r, pipe))
            full = model.predict_pipeline(pipe, r["blocks_t"], r["max_num_seqs"])
            out.append({**r, "pipe": pipe, "passed": passed,
                        "stage_sum": sum(s["latency_s"] for s in comp["stages"]),
                        "stage_max": max(s["latency_s"] for s in comp["stages"]),
                        "pred_comp": comp["latency_s"], "pred_full": full["latency_s"]})
    return out


def fit_lambda(model: Model, runs: list[dict]) -> float:
    """lambda in L = lambda * sum(L_i), from the measured stage sizes."""
    if not runs:
        return 1.0
    return _lstsq([[r["stage_sum"]] for r in runs], [r["latency_s"] for r in runs],
                  [1 / r["latency_s"] for r in runs])[0]


def fit_retry(rows: list[dict]) -> dict:
    """Extra per-row map calls per map row, pooled by retry_key."""
    sums = {}
    for r in rows:
        pipe = pipeline_of(r, 1.0)
        for i, op in enumerate(pipe.ops):
            key = retry_key(pipe.ops, r["blocks_t"], i)
            if key is None:
                continue
            mapped = pipe.rows if i == 0 else passed_rows(r, pipe)
            extra = retried_rows(r, pipe)
            if not mapped or extra is None:
                continue
            a = sums.setdefault(key, [0.0, 0.0])
            a[0] += extra
            a[1] += mapped
    return {key: round(a[0] / a[1], 4) for key, a in sorted(sums.items())}


def pipeline_choices(model: Model, runs: list[dict]) -> list[dict]:
    out = []
    for source in sorted({r["source"] for r in runs}):
        run = [r for r in runs if r["source"] == source]
        pipe, seqs = run[0]["pipe"], run[0]["max_num_seqs"]
        cands = [r["blocks_t"] for r in run]
        by = {r["blocks_t"]: r for r in run}
        pick = model.choose_pipeline(pipe, seqs, candidates=cands, quality=True)
        fast = model.choose_pipeline(pipe, seqs, candidates=cands)
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
    fixed = ", ".join(f"{k} {v:.3f} s" for k, v in model.g.items())
    retry = ", ".join(f"{k} {v:.0%}" for k, v in model.retry.items()) or "none measured"
    lines = [
        f"# Cost model ({MODEL_VERSION})", "",
        f"Fitted on {len(rows)} single-operator arms from "
        f"{len({r['source'] for r in rows})} runs ({', '.join(sorted({r['workload'] for r in rows}))}); "
        f"pipeline composition on {len(pipes)} pipeline arms. Held out: "
        f"{', '.join(HELD_OUT)}.", "",
        "Latency `L = g[kind] + alpha P + (beta + tail[kind]) W D + gamma C D + delta C p^2` "
        "(P computed prompt tokens, W waves, D decode steps per call, C calls, p prompt "
        "tokens per call):", "",
        f"- g (fixed cost per operator) = {fixed}; alpha = {model.alpha * 1e4:.3f} s per 10k "
        f"computed tokens ({1 / model.alpha:,.0f} tokens/s); beta = {model.beta * 1e3:.1f} ms per "
        f"decode step of a wave; {tails}; gamma = {model.gamma * 1e3:.3f} ms per generated token; "
        f"delta = {model.delta * 1e6:.2f} ms per call per (1k prompt tokens)^2.",
        f"- Pipelines: L = lambda * sum over stages of L_i, lambda = {model.lam:.2f}. A leading "
        "filter passes rows * (recall(k) s + fpr(k) (1 - s)) for true selectivity s. A "
        "per-tuple map reruns a share of its rows when another stage is blocked (by the "
        f"map's position and size: {retry}).",
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
        lines += [f"| Pipelines, measured filter output and reruns | {_pct(np.mean(comp))} | "
                  f"{_pct(max(comp))} |",
                  f"| Pipelines, full model | {_pct(np.mean(full))} | {_pct(max(full))} |"]
    lines += ["", "Per operator kind (tokens):", "",
              "| Kind | a_pt | q_pt | d_pt | e | o | c_left | q | d0 | d1 |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for kind, p in model.kinds.items():
        lines.append(f"| {kind} | {p.a_pt:.1f} | {p.q_pt:.1f} | {p.d_pt:.2f} | {p.e:.1f} | "
                     f"{p.o:.1f} | {p.c_left:.2f} | {p.q:.1f} | {p.d0:.2f} | {p.d1:.2f} |")
    lines += ["", f"Accuracy caps, used only by the guarded choice (largest k whose mean ΔF1 "
                  f"across runs stayed within {model.f1_slack}; maps: mean similarity at least "
                  f"{analyze.MIN_SIMILARITY}), and filters' recall and false-positive rate by k. "
                  f"Long rows: items over {LONG_ROW_TOKENS} tokens.", "",
              "| Kind | Regime | Measured k | Cap k | Recall by k | False-positive rate by k |",
              "|---|---|---|---|---|---|"]
    for kind, p in model.kinds.items():
        for regime, measured in p.measured_k.items():
            rec = ", ".join(f"{k}: {v:.2f}" for k, v in p.recall.get(regime, {}).items()) or "–"
            fpr = ", ".join(f"{k}: {v:.3f}" for k, v in p.fpr.get(regime, {}).items()) or "–"
            lines.append(f"| {kind} | {regime} | {measured} | {p.cap(regime)} | {rec} | {fpr} |")
    lines += ["", "## Choice per measured single-operator run", "",
              "Chosen among the sizes the run measured. Algorithm = lowest predicted latency, "
              "against the fastest measured arm. Guarded = with the accuracy caps, against the "
              "fastest measured arm that kept quality. Free = the algorithm over every size up "
              "to the measured ones.", "",
              "| Run | Regime | Items | Algorithm k | Fastest k | Regret | Guarded k | "
              "Best kept-quality k | Guarded regret | Guarded kept quality | Free k |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in picks:
        lines.append(f"| {c['source']} | {c['regime']} | {c['items']} | {c['latency_k']} | "
                     f"{c['fastest_k']} | {_pct(c['latency_regret'])} | {c['chosen_k']} | "
                     f"{c['best_k']} | {_pct(c['regret'])} | {_YES[c['chosen_quality']]} | "
                     f"{c['unrestricted_k']} |")
    if picks:
        lines += ["", f"Over {len(picks)} runs: algorithm regret mean "
                      f"{_pct(np.mean([c['latency_regret'] for c in picks]))}, max "
                      f"{_pct(max(c['latency_regret'] for c in picks))}; guarded regret mean "
                      f"{_pct(np.mean([c['regret'] for c in picks]))}, max "
                      f"{_pct(max(c['regret'] for c in picks))}."]
    if pipes:
        lines += ["", "## Pipelines", "",
                  "Rows passed: by the leading filter (filter -> map: rows out; otherwise from "
                  "the call count). Map reruns: per-row map calls beyond the stages' blocks "
                  "(reruns of cut-off answers, repairs). Stage sum/max: stage latencies "
                  "predicted at those sizes and measured reruns. Composed: lambda * sum. Full: "
                  "rows passed and reruns predicted too.", "",
                  "| Run | Arm | Measured (s) | Rows passed | Map reruns | Stage sum (s) | "
                  "Stage max (s) | Composed (s) | Full (s) | ΔF1 |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for r in pipes:
            passed = "–" if r["passed"] is None else f"{r['passed']:.0f}"
            f1 = "–" if r.get("f1_delta") is None else f"{r['f1_delta']:+.2f}"
            reruns = retried_rows(r, r["pipe"])
            lines.append(f"| {r['source']} | `{r['arm']}` | {r['latency_s']:.2f} | {passed} | "
                         f"{_fmt(reruns, '.0f')} | {r['stage_sum']:.2f} | {r['stage_max']:.2f} | "
                         f"{r['pred_comp']:.2f} | {r['pred_full']:.2f} | {f1} |")
        lines += ["", "| Run | Algorithm | Fastest | Regret | Guarded | Best kept quality | "
                      "Guarded regret | Guarded kept quality | Free |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for c in pipe_picks:
            lines.append(f"| {c['source']} | {_blocks(c['latency_choice'])} | "
                         f"{_blocks(c['fastest'])} | {_pct(c['latency_regret'])} | "
                         f"{_blocks(c['chosen'])} | {_blocks(c['best'])} | {_pct(c['regret'])} | "
                         f"{_YES[c['chosen_quality']]} | {_blocks(c['unrestricted'])} |")
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
        {"version": MODEL_VERSION, "g": model.g, "alpha": model.alpha, "beta": model.beta,
         "gamma": model.gamma, "delta": model.delta, "tail": model.tail, "lambda": model.lam,
         "retry": model.retry,
         "f1_slack": model.f1_slack,
         "kinds": {k: asdict(v) for k, v in model.kinds.items()}}, indent=2), encoding="utf-8")
    return text


# --- Validation choices ------------------------------------------------------

def best_in_isolation(model: Model, rows: list[dict]) -> dict:
    """Per (kind, regime), the block size with the highest mean speedup over
    per-tuple across the measured runs (the static "tune each operator alone"
    strategy); 1 if none is faster."""
    out = {}
    for kind, p in model.kinds.items():
        for regime in p.measured_k:
            speedups = {}
            for r in rows:
                if r["kind"] == kind and _regime(r) == regime and r["k"] > 1 and r.get("speedup"):
                    speedups.setdefault(r["k"], []).append(r["speedup"])
            best = max(speedups, key=lambda k: np.mean(speedups[k])) if speedups else 1
            out[(kind, regime)] = best if speedups and np.mean(speedups[best]) > 1 else 1
    return out


def _grid(blocks: tuple, k_max: list[int]) -> list[tuple]:
    out = set()
    for i, k in enumerate(blocks):
        for g in {max(1, k // 2), max(1, k - 1), k + 1, 2 * k}:
            if g != k and g <= k_max[i]:
                out.add(blocks[:i] + (g,) + blocks[i + 1:])
    return sorted(out)


VALIDATION_PHASES = ("validate", "validate3", "validate-jf", "validate4")
# Phases the current model is checked on and never fitted on. The first
# validation phase tested v2; v3 is fitted on it and checked on validate3.
HELD_OUT = ("validate3", "validate-jf", "validate4")
MODEL_VERSION = "v3"


def _validation_target(model: Model, exp: dict, cache: dict):
    """What the algorithm sees for a validation experiment before running it:
    ``("pipe", Pipeline)`` or ``("single", kind, Items)``; None if the model
    cannot plan it (an operator kind it was not fitted on). A leading filter's
    selectivity is the workload's true share of matching rows."""
    from .workloads import WORKLOADS
    # The first part is what is validated; later parts (an operator alone)
    # only share its container.
    part = exp["parts"][0]
    try:
        workload = WORKLOADS[part["workload"]](part["n"] or None)
    except KeyError:
        return None
    t = item_tokens(part["workload"], part["n"] or None, cache)
    if workload.stages:
        ops = tuple(s["op"] for s in workload.stages)
        if not set(ops) <= set(model.kinds):
            return None
        selectivity = 1.0
        if ops[0] == "filter" and workload.positives is not None:
            # Filter -> map: the filter's positives; filter -> join: left rows
            # of a matching pair, which are the filter's positives.
            left = {p[0] if isinstance(p, tuple) else p for p in workload.positives}
            selectivity = len(left) / len(workload.rows)
        return ("pipe", Pipeline(ops, t["left_rows"], t["left_mean"], t["left_max"], selectivity,
                                 t["right_rows"], t["right_mean"], t["right_max"],
                                 tuple(workload.stage_selectivity or ())))
    kind = workload.kind
    if kind not in model.kinds:
        return None
    joined = kind == "join"
    return ("single", kind, Items(len(workload.rows) * (t["right_rows"] or 1), t["mean"], t["max"],
                                  t["left_rows"] if joined else 0,
                                  t["left_mean"] if joined else 0.0))


def _predict_target(model: Model, target, blocks: tuple) -> float:
    if target[0] == "pipe":
        return model.predict_pipeline(target[1], blocks)["latency_s"]
    return model.predict(target[1], target[2], blocks[0])["latency_s"]


def _choose_target(model: Model, target, quality: bool = False, candidates=None) -> tuple:
    if target[0] == "pipe":
        return tuple(model.choose_pipeline(target[1], candidates=candidates,
                                           quality=quality)["blocks"])
    ks = [c[0] for c in candidates] if candidates else None
    return (model.choose_k(target[1], target[2], candidates=ks, quality=quality)["k"],)


def validation_choices(model: Model, rows: list[dict], cache: dict, frozen: dict) -> dict:
    """For each validation experiment: the algorithm's pick (lowest predicted
    latency), the pick with the accuracy guard, best-in-isolation and a grid
    around both (for the oracle), as cloud/experiments.py reads them. Choices
    in ``frozen`` (experiments already run) are kept as they were."""
    iso = best_in_isolation(model, rows)
    choices = dict(frozen)
    for exp in _manifest().EXPERIMENTS:
        if exp["phase"] not in VALIDATION_PHASES or exp["id"] in frozen:
            continue
        target = _validation_target(model, exp, cache)
        if target is None:
            continue
        if target[0] == "pipe":
            pipe = target[1]
            stage_items = model.stage_items(pipe, pipe.rows)
            k_max = [model.k_max(op, it) for op, it in zip(pipe.ops, stage_items)]
            iso_blocks = tuple(iso.get((op, it.regime), 1)
                               for op, it in zip(pipe.ops, stage_items))
            predicted = {"selectivity": round(pipe.selectivity, 3), "rows": pipe.rows}
        else:
            _, kind, items = target
            k_max = [model.k_max(kind, items)]
            iso_blocks = (iso.get((kind, items.regime), 1),)
            predicted = {"items": items.n, "item_tokens": items.tokens, "regime": items.regime}
        blocks = _choose_target(model, target)
        guarded = _choose_target(model, target, quality=True)
        choices[exp["id"]] = {
            "model": MODEL_VERSION,
            "algorithm": arm_name(blocks),
            # The latency pick under the name the first validation phase used.
            "algorithm_latency": arm_name(blocks),
            "algorithm_guarded": arm_name(guarded),
            "best_in_isolation": arm_name(iso_blocks),
            "grid": [arm_name(g) for g in sorted(set(_grid(blocks, k_max))
                                                 | set(_grid(guarded, k_max)))
                     if g not in (blocks, guarded)],
            "predicted": {**predicted, "blocks": list(blocks),
                          "latency_s": round(_predict_target(model, target, blocks), 3),
                          "guarded": list(guarded),
                          "guarded_s": round(_predict_target(model, target, guarded), 3)},
        }
    return choices


# --- Validation scoring ------------------------------------------------------

STRATEGIES = (
    ("Tuple-wise", lambda c: "kalypso_per_tuple"),
    ("Static 4", lambda c: "kalypso_block:4"),
    ("Static 16", lambda c: "kalypso_block:16"),
    ("Static 32", lambda c: "kalypso_block:32"),
    ("Best in isolation", lambda c: c["best_in_isolation"]),
    # v2 choices name the latency pick algorithm_latency and the guarded one
    # algorithm; v3 choices name both explicitly.
    ("Algorithm", lambda c: c["algorithm_latency"]),
    ("Algorithm with accuracy guard", lambda c: c.get("algorithm_guarded", c["algorithm"])),
)
ORACLE = "Oracle (fastest measured)"
RETRO = "Current model, among measured arms"


def _blocks_of(arm: str, n_stages: int) -> tuple:
    if arm == "kalypso_per_tuple":
        return (1,) * n_stages
    kind, _, size = arm.partition(":")
    sizes = tuple(int(x) for x in size.split(","))
    return sizes * n_stages if kind == "kalypso_block" else sizes


def score_validation(model: Model, collected: list[dict], choices: dict,
                     cache: dict) -> list[dict]:
    """Per validation experiment and strategy: the arm it runs, its measured
    latency and quality, speedup over tuple-wise, regret against the oracle
    (the fastest measured arm; accuracy is reported, not required) and the
    current model's prediction for that arm. Experiments chosen by an earlier
    model also get the current model's pick among the arms they measured."""
    out = []
    for exp in _manifest().EXPERIMENTS:
        if exp["phase"] not in VALIDATION_PHASES:
            continue
        run = {r["arm"]: r for r in collected if r.get("experiment") == exp["id"]
               and r.get("part") == exp["parts"][0]["name"]}
        c = choices.get(exp["id"])
        if not run or c is None:
            continue
        target = _validation_target(model, exp, cache)
        n_stages = len(target[1].ops) if target and target[0] == "pipe" else 1
        ok = {a: quality_ok(r, model.f1_slack) for a, r in run.items()}
        measured = {a: r for a, r in run.items() if r["latency_s"] is not None}
        base = measured.get("kalypso_per_tuple")
        oracle = min(measured.values(), key=lambda r: r["latency_s"])
        strategies = [*((name, f(c)) for name, f in STRATEGIES)]
        version = c.get("model", "v2")
        if version != MODEL_VERSION and target:
            by_blocks = {_blocks_of(a, n_stages): a for a in measured}
            pick = _choose_target(model, target, candidates=list(by_blocks))
            strategies.append((RETRO, by_blocks[pick]))
        strategies.append((ORACLE, oracle["arm"]))
        for name, arm in strategies:
            r = run.get(arm)
            lat = r["latency_s"] if r else None
            pred = (_predict_target(model, target, _blocks_of(arm, n_stages))
                    if target and arm else None)
            out.append({
                "experiment": exp["id"], "phase": exp["phase"], "chosen_by": version,
                "strategy": name, "arm": arm, "latency_s": lat,
                "failed": r is not None and lat is None,
                "speedup": base["latency_s"] / lat if base and lat else None,
                "regret": lat / oracle["latency_s"] - 1 if oracle and lat else None,
                "f1_delta": r.get("f1_delta") if r else None,
                "similarity": r.get("similarity") if r else None,
                "kept_quality": ok.get(arm), "predicted_s": pred,
                "pred_error": _err(pred, lat) if pred and lat else None,
            })
    return out


def _fmt(x, spec, missing="–"):
    return missing if x is None else format(x, spec)


def _summary(rows: list[dict], names: list[str]) -> list[str]:
    lines = ["| Strategy | Experiments | Kept quality | Failed | Speedup (geomean) | "
             "Regret (mean) | Regret (max) | Prediction error (mean) |",
             "|---|---|---|---|---|---|---|---|"]
    for name in names:
        rs = [r for r in rows if r["strategy"] == name and r["arm"]]
        if not rs:
            continue
        ran = [r for r in rs if r["latency_s"] is not None]
        sp = [r["speedup"] for r in ran if r["speedup"]]
        rg = [r["regret"] for r in ran if r["regret"] is not None]
        pe = [r["pred_error"] for r in ran if r["pred_error"] is not None]
        kept = sum(1 for r in ran if r["kept_quality"] is not False)
        lines.append(
            f"| {name} | {len(rs)} | {kept}/{len(ran)} | {sum(r['failed'] for r in rs)} | "
            f"{_fmt(float(np.exp(np.mean(np.log(sp)))) if sp else None, '.2f')}x | "
            f"{_pct(np.mean(rg)) if rg else '–'} | {_pct(max(rg)) if rg else '–'} | "
            f"{_pct(np.mean(pe)) if pe else '–'} |")
    return lines


def write_validation(rows: list[dict], out: Path) -> str:
    lines = ["# Validation", "",
             "Latency first: the algorithm picks the block sizes with the lowest predicted "
             "latency, and the oracle is the fastest measured arm. Accuracy is reported, not "
             "required: kept quality = F1 within the slack of tuple-wise (and for a pipeline "
             f"ending in a map, ROUGE-1 similarity at least {analyze.MIN_SIMILARITY}). The "
             "accuracy-guarded algorithm only blocks where the measured runs kept quality. "
             "Speedup over tuple-wise is a geometric mean over the experiments the strategy "
             "ran; regret = how much slower than the oracle. Predicted = the current model "
             f"({MODEL_VERSION}) for that arm, from the workload's true selectivity.", "",
             f"Each phase is scored with the choices made before it ran. `validate` was "
             f"chosen by v2 and is now part of {MODEL_VERSION}'s fit, so its "
             f"\"{RETRO}\" row is in-sample; {', '.join(f'`{p}`' for p in HELD_OUT)} "
             f"{'is' if len(HELD_OUT) == 1 else 'are'} held out from {MODEL_VERSION}."]
    names = [n for n, _ in STRATEGIES] + [RETRO, ORACLE]
    for phase in dict.fromkeys(r["phase"] for r in rows):
        at = [r for r in rows if r["phase"] == phase]
        held = "held out" if phase in HELD_OUT else "in the current fit"
        lines += ["", f"## Summary: `{phase}` (chosen by {at[0]['chosen_by']}; {held})", "",
                  *_summary(at, names)]
    for exp in dict.fromkeys(r["experiment"] for r in rows):
        lines += ["", f"## {exp}", "",
                  "| Strategy | Arm | Latency (s) | Speedup | Regret | ΔF1 | Sim | "
                  "Kept quality | Predicted (s) |", "|---|---|---|---|---|---|---|---|---|"]
        for r in rows:
            if r["experiment"] != exp:
                continue
            lat = "failed" if r["failed"] else _fmt(r["latency_s"], ".2f")
            lines.append(
                f"| {r['strategy']} | `{r['arm']}` | {lat} | {_fmt(r['speedup'], '.2f')}x | "
                f"{_pct(r['regret']) if r['regret'] is not None else '–'} | "
                f"{_fmt(r['f1_delta'], '+.2f')} | {_fmt(r['similarity'], '.2f')} | "
                f"{_YES[r['kept_quality']]} | {_fmt(r['predicted_s'], '.2f')} |")
    text = "\n".join(lines) + "\n"
    (out / "validation.md").write_text(text, encoding="utf-8")
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--out", type=Path, default=None, help="default: <results>/analysis")
    parser.add_argument("--f1-slack", type=float, default=0.02)
    parser.add_argument("--write-choices", action="store_true",
                        help="also write the choices for validation experiments not yet run "
                             "(analysis/algorithm_choices.json; unlocks them in the manifest)")
    args = parser.parse_args()
    out = args.out or args.results / "analysis"
    cache_path = out / "item_tokens.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    everything = analyze.collect(args.results)
    # Held-out validation runs test the model; they are not fitted on.
    collected = [r for r in everything if r["phase"] not in HELD_OUT]
    rows = attach_tokens(_single_operator_rows(collected), cache)
    pipeline_rows = attach_tokens(_pipeline_rows(collected), cache)
    out.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache, indent=2, sort_keys=True), encoding="utf-8")
    if not rows:
        raise SystemExit("no single-operator Kalypso arms to fit")
    model = fit(rows, args.f1_slack)
    model.retry = fit_retry(pipeline_rows)
    model.lam = fit_lambda(model, pipeline_runs(model, pipeline_rows, cache))
    pipes = pipeline_runs(model, pipeline_rows, cache)
    evaluated = evaluate(model, rows)
    write_report(model, evaluated, leave_one_workload_out(rows, args.f1_slack),
                 choices_per_run(model, evaluated), pipes, pipeline_choices(model, pipes), out)
    print(f"fitted on {len(rows)} arms and {len(pipes)} pipeline arms -> {out / 'costmodel.md'}")
    choices_path = out / "algorithm_choices.json"
    choices = json.loads(choices_path.read_text()) if choices_path.exists() else {}
    if args.write_choices:
        ran = {r["experiment"] for r in everything if r.get("experiment")}
        frozen = {k: v for k, v in choices.items() if k in ran}
        choices = validation_choices(model, rows, cache, frozen)
        choices_path.write_text(json.dumps(choices, indent=2), encoding="utf-8")
        cache_path.write_text(json.dumps(cache, indent=2, sort_keys=True), encoding="utf-8")
        for exp_id, c in choices.items():
            if exp_id in frozen:
                continue
            print(f"{exp_id}: {c['algorithm']} (predicted {c['predicted']['latency_s']} s), "
                  f"guarded {c['algorithm_guarded']}, best in isolation "
                  f"{c['best_in_isolation']}, grid {c['grid']}")
    validated = [r for r in everything if r["phase"] in VALIDATION_PHASES]
    if validated and choices:
        write_validation(score_validation(model, validated, choices, cache), out)
        print(f"scored {len({r['experiment'] for r in validated})} validation experiments -> "
              f"{out / 'validation.md'}")


if __name__ == "__main__":
    main()
