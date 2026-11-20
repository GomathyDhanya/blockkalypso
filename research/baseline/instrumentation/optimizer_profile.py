"""Measure the original (v3) block-size optimizer without changing it.

``profile_choose_pipeline`` runs ``costmodel.Model.choose_pipeline`` three
times on the same input and returns one record:

1. **plain**: the original call, timed with ``perf_counter`` only. This is the
   planning latency.
2. **components**: the same call with timers wrapped around the model's own
   methods (``stage_items``, ``k_max``, ``predict_pipeline``, ``predict``,
   ``retry_share``) and the module's ``_pick``. Gives the time split and the
   number of candidates actually evaluated. Timer overhead makes this run
   slower; its total is reported separately and never used as the latency.
3. **memory**: the same call under ``tracemalloc``, for peak Python heap.

The three picks must agree with each other (asserted), so wrapping does not
change what the optimizer chooses.

Also returns, per stage, which bound set ``k_max`` (context window, measured
or quality cap, item count): the optimizer's candidate-feasibility reasons.
The original search never rejects an enumerated candidate (it only enumerates
legal ones); a candidate list passed in explicitly is filtered by ``k_max``,
and those rejections are counted too.
"""

from __future__ import annotations

import contextlib
import itertools
import json
import math
import time
import tracemalloc
from dataclasses import fields
from pathlib import Path

from vllm.kalypso.benchmark.eval import costmodel as cm

REPO = Path(__file__).resolve().parents[3]
FITTED = REPO / "vllm/kalypso/benchmark/eval/results/analysis/costmodel.json"


def load_model(path: Path = FITTED) -> cm.Model:
    """The committed v3 fit (costmodel.json) as a ``costmodel.Model``."""
    d = json.loads(Path(path).read_text())
    names = {f.name for f in fields(cm.KindParams)}
    kinds = {k: cm.KindParams(**{n: v for n, v in p.items() if n in names})
             for k, p in d["kinds"].items()}
    return cm.Model(g=d["g"], alpha=d["alpha"], beta=d["beta"], gamma=d["gamma"],
                    delta=d["delta"], tail=d["tail"], kinds=kinds,
                    f1_slack=d["f1_slack"], lam=d["lambda"], retry=d["retry"])


class _Timers:
    def __init__(self):
        self.seconds: dict[str, float] = {}
        self.calls: dict[str, int] = {}

    def wrap(self, name, fn):
        def timed(*args, **kwargs):
            t = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                self.seconds[name] = self.seconds.get(name, 0.0) + time.perf_counter() - t
                self.calls[name] = self.calls.get(name, 0) + 1
        return timed


@contextlib.contextmanager
def _instrumented(model: cm.Model, timers: _Timers):
    """Wrap the instance's methods and the module's _pick; restore after."""
    names = ("stage_items", "k_max", "predict_pipeline", "predict", "retry_share")
    for n in names:
        setattr(model, n, timers.wrap(n, getattr(model, n)))
    original_pick = cm._pick
    cm._pick = timers.wrap("pick", original_pick)
    try:
        yield
    finally:
        cm._pick = original_pick
        for n in names:
            delattr(model, n)  # instance attribute removed: class method again


def k_max_reasons(model: cm.Model, pipe: cm.Pipeline, quality: bool = False) -> list[dict]:
    """Per stage: k_max and which bound set it (ties list every bound)."""
    out = []
    for op, items in zip(pipe.ops, model.stage_items(pipe, pipe.rows)):
        p = model.kinds[op]
        bounds = {
            "context_window": model.k_ctx(op, items.max_tokens),
            ("quality_cap" if quality else "measured_k_cap"):
                (p.cap(items.regime) if quality
                 else max(p.measured_k.get(items.regime) or [cm.MAX_K])),
            "items": max(int(items.n), 1),
        }
        k = model.k_max(op, items, quality)
        out.append({"op": op, "regime": items.regime, "k_max": k,
                    "binding": sorted(b for b, v in bounds.items() if v == k),
                    "bounds": bounds})
    return out


def legal_space(model: cm.Model, pipe: cm.Pipeline, quality: bool = False,
                candidate_set=None) -> tuple[int, int]:
    """(candidates enumerated, candidates legal) without running the search."""
    kmax = [r["k_max"] for r in k_max_reasons(model, pipe, quality)]
    if candidate_set is None:
        n = math.prod(kmax)
        return n, n
    per_stage = [len([k for k in candidate_set if k <= m]) or 1 for m in kmax]
    return len(candidate_set) ** len(kmax), math.prod(per_stage)


def candidates_from_set(model, pipe, candidate_set, quality=False):
    """Full product of a fixed per-stage candidate set (e.g. powers of 2);
    choose_pipeline itself drops the ones beyond k_max."""
    return list(itertools.product(candidate_set, repeat=len(pipe.ops)))


def profile_choose_pipeline(model: cm.Model, pipe: cm.Pipeline, candidates=None,
                            quality: bool = False, tie: float = 0.02,
                            memory: bool = True) -> dict:
    cand_list = list(candidates) if candidates is not None else None

    def call():
        return model.choose_pipeline(pipe, tie=tie, quality=quality,
                                     candidates=None if cand_list is None else iter(cand_list))

    t = time.perf_counter()
    plain = call()
    plain_s = time.perf_counter() - t

    timers = _Timers()
    with _instrumented(model, timers):
        t = time.perf_counter()
        timed = call()
        timed_s = time.perf_counter() - t

    peak = None
    if memory:
        tracemalloc.start()
        try:
            mem = call()
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        assert mem["blocks"] == plain["blocks"]
    assert timed["blocks"] == plain["blocks"] and timed["fastest"] == plain["fastest"]

    reasons = k_max_reasons(model, pipe, quality)
    evaluated = timers.calls.get("predict_pipeline", 0)
    submitted = len(cand_list) if cand_list is not None else evaluated
    component_s = {k: round(v, 6) for k, v in timers.seconds.items()}
    # Time in choose_pipeline itself: enumeration, list building, filtering.
    nested = {"predict", "retry_share"}  # inside predict_pipeline
    outer = sum(v for k, v in timers.seconds.items() if k not in nested and k != "stage_items")
    return {
        "ops": list(pipe.ops), "stages": len(pipe.ops), "quality": quality,
        "rows": pipe.rows, "selectivity": pipe.selectivity,
        "k_max": [r["k_max"] for r in reasons], "k_max_reasons": reasons,
        "candidates_submitted": submitted,
        "candidates_evaluated": evaluated,
        "candidates_rejected_k_max": submitted - evaluated if cand_list is not None else 0,
        "planning_s": plain_s,
        "instrumented_total_s": timed_s,
        "component_s": component_s,
        "component_calls": dict(timers.calls),
        "enumeration_other_s": max(timed_s - outer, 0.0),
        "peak_python_heap_bytes": peak,
        "chosen": list(plain["blocks"]), "fastest": list(plain["fastest"]),
        "predicted_latency_s": plain["latency_s"],
    }
