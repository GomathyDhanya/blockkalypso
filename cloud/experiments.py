"""Every batching-study experiment, as data for cloud/modal_block_size.py.

    modal run cloud/modal_block_size.py::status
    modal run --detach cloud/modal_block_size.py::experiments --phase single
    modal run cloud/modal_block_size.py::pull
    python3 -m vllm.kalypso.benchmark.eval.analyze

An experiment is one container: Kalypso on one GPU, the per-tuple baseline
plus the listed arms, each part's workload run in turn. Parts let a pipeline
and its operators run alone share a container, so their latencies compare
directly. Results go to the volume under ``experiments/<id>/<part>/``.

Phases follow the study's steps: single operators (``single``, ``map``,
``longrows``), pipelines (``pipeline``) and validation of the block-size
algorithm against static batching and tuple-wise execution (``validate``).
An experiment whose code is not in this checkout yet is listed as blocked
with the reason; it runs once that lands.

This file is plain Python with no imports from vllm or modal, so the local
entrypoints can load it without either installed.
"""

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EVAL = REPO / "vllm/kalypso/benchmark/eval"
RESULTS = EVAL / "results"
# Written by the cost model (eval/costmodel.py): per validation experiment,
# {"algorithm": arm, "best_in_isolation": arm, "grid": [arms]}.
ALGORITHM_CHOICES = RESULTS / "analysis" / "algorithm_choices.json"


def blocks(*sizes):
    return [f"kalypso_block:{k}" for k in sizes]


def pipe(*configs):
    """kalypso_pipe arms, one block size per stage (1 = per-tuple)."""
    return [f"kalypso_pipe:{','.join(str(k) for k in c)}" for c in configs]


def part(workload, arms, n=0, name=None):
    return {"name": name or (f"{workload}-n{n}" if n else workload),
            "workload": workload, "n": n, "arms": list(arms)}


def experiment(id, phase, parts, purpose, requires=()):
    return {"id": id, "phase": phase, "parts": parts, "purpose": purpose,
            "requires": list(requires)}


# What an experiment needs that may not be in this checkout yet.
REQUIREMENTS = {
    "block_map": ("vllm/kalypso/sem_ops/block_map.py",
                  "block map (merge the map branch, origin/map)"),
    "contractnli": ("vllm/kalypso/benchmark/eval/workloads.py",
                    "the ContractNLI workloads (cnli_sel*, cnli_join_*; PR #18)"),
    "algorithm": (str(ALGORITHM_CHOICES.relative_to(REPO)),
                  "the cost model's choices (eval/costmodel.py writes them)"),
}


def missing(exp) -> list[str]:
    reasons = []
    for req in exp["requires"]:
        path, what = REQUIREMENTS[req]
        target = REPO / path
        if req == "contractnli":
            ok = target.exists() and "CNLI_SELECTIVITIES" in target.read_text(encoding="utf-8")
        elif req == "algorithm":
            ok = target.exists() and exp["id"] in json.loads(target.read_text(encoding="utf-8"))
        else:
            ok = target.exists()
        if not ok:
            reasons.append(what)
    return reasons


# --- Step 2: single operators across regimes --------------------------------

SINGLE = [
    *[experiment(f"single/join-fanout{f}", "single",
                 [part(f"movie_fanout{f}", blocks(2, 4, 6, 11))],
                 f"Join, every left row matches {f}% of right rows: does speed or "
                 "recall depend on fanout?")
      for f in (10, 25, 50, 75, 90)],
    *[experiment(f"single/filter-n{n or 'all'}", "single",
                 [part("movie_positive", blocks(4, 6, 8), n=n)],
                 "Filter at a different workload size (n=500 is the block-size "
                 "search): concurrency, waves, and whether k* moves with N.")
      for n in (100, 250, 1000, 0)],
    *[experiment(f"single/join-n{n}", "single",
                 [part("movie_opposite", blocks(2, 4, 6), n=n)],
                 f"Join on {n}x{n} pairs (32x32 is the block-size search): k* vs N.")
      for n in (16, 48, 64)],
]

MAP = [
    experiment("map/map-n500", "map", [part("movie_map", blocks(2, 4, 6, 8, 16), n=500)],
               "Map batching on 500 reviews: latency, decode per call, agreement "
               "with per-tuple output.", requires=["block_map"]),
    experiment("map/map-n100", "map", [part("movie_map", blocks(2, 4, 8), n=100)],
               "Map at a small workload: does batching still pay?", requires=["block_map"]),
]

# ContractNLI (PR #18): contracts up to 12,500 characters (~3k tokens), so
# 8 rows still fit the 32k context. Reruns PR #18's sizes with this
# harness's fallback, waves and running/waiting metrics.
LONGROWS = [
    *[experiment(f"longrows/cnli-filter-s{s}", "longrows",
                 [part(f"cnli_sel{s}", blocks(2, 4, 8))],
                 f"Long-row filter (one contract + hypothesis per row, no shared "
                 f"prefix), s={s / 100:.2f}: does batching pay when rows are long?",
                 requires=["contractnli"])
      for s in (10, 50, 90)],
    *[experiment(f"longrows/cnli-join-{level}", "longrows",
                 [part(f"cnli_join_{level}", blocks(2, 4, 8))],
                 f"Long-row join (24 contracts x 8 hypotheses, {level} selectivity): "
                 "BlockSemJoin repeats the contract once per pair in a block, while "
                 "per-tuple pairs share it through the prefix cache.",
                 requires=["contractnli"])
      for level in ("low", "high")],
]

# --- Step 3: pipelines -------------------------------------------------------
# Each pipeline shares its container with its operators run alone, so the
# composition (sum vs max of stages) is measured on one server.

PIPELINE = [
    experiment("pipeline/filter-map", "pipeline", [
        part("pipe_filter_map", pipe((4, 1), (6, 1), (11, 1))),
        part("movie_positive", blocks(4, 6, 11), n=500, name="alone-filter"),
        part("pipe_filter_map_map", [], name="alone-map"),
    ], "Filter -> map: does a block filter (own stage, prefix pin chain ended) "
       "lose to the fused per-tuple pipeline; pipeline vs sum/max of stages."),
    experiment("pipeline/filter-map-blockmap", "pipeline", [
        part("pipe_filter_map", pipe((1, 4), (6, 4), (6, 8))),
        part("pipe_filter_map_map", blocks(4, 8), name="alone-map"),
    ], "Filter -> map with the map batched too.", requires=["block_map"]),
    *[experiment(f"pipeline/filter-join-s{s}", "pipeline", [
        part(f"pipe_filter_join_s{s}", pipe((4, 4), (6, 4), (11, 4), (6, 2), (6, 6),
                                             (1, 4), (4, 1))),
        part(f"movie_sel{s}", blocks(4, 6, 11), n=100, name="alone-filter"),
        part(f"pipe_filter_join_s{s}_join", blocks(2, 4, 6), name="alone-join"),
    ], f"Filter (s={s / 100:.2f}) -> join: is the best upstream k smaller inside the "
       "pipeline; does the join's k* move when its input trickles in?")
      for s in (10, 50, 90)],
]

# --- Step 4: validation on held-out configurations ---------------------------
# Static strategies and tuple-wise run beside the algorithm's pick, the
# best-in-isolation sizes and a grid around the pick (for the oracle).

STATIC = blocks(4, 16, 32)


def validation(id, workload, n, purpose, requires=()):
    return experiment(f"validate/{id}", "validate", [part(workload, STATIC, n=n)],
                      purpose, requires=["algorithm", *requires])


VALIDATE = [
    validation("filter-join-s25", "pipe_filter_join_s25", 0,
               "Filter -> join at an unseen selectivity."),
    validation("filter-map-n1000", "pipe_filter_map", 1000, "Filter -> map at an unseen size.",
               requires=["block_map"]),
    validation("map-filter", "pipe_map_filter", 500, "A pipeline shape not used for fitting.",
               requires=["block_map"]),
    validation("filter-map-n50", "pipe_filter_map", 50, "Small N, where tuple-wise may win.",
               requires=["block_map"]),
    validation("join-fanout25-n64", "movie_fanout25", 64,
               "A join outside the fitting grid (64x64, fanout 16)."),
    validation("cnli-filter-s25", "cnli_sel25", 200,
               "Long rows at an unseen selectivity: the algorithm should not batch.",
               requires=["contractnli"]),
]

EXPERIMENTS = SINGLE + MAP + LONGROWS + PIPELINE + VALIDATE
PHASES = ("single", "map", "longrows", "pipeline", "validate")


def arms_for(exp, part_) -> list[str]:
    """The part's arms, plus the cost model's picks for a validation experiment."""
    arms = list(part_["arms"])
    if exp["phase"] == "validate" and ALGORITHM_CHOICES.exists():
        picks = json.loads(ALGORITHM_CHOICES.read_text(encoding="utf-8")).get(exp["id"], {})
        arms += [picks[k] for k in ("algorithm", "best_in_isolation") if picks.get(k)]
        arms += picks.get("grid", [])
    return list(dict.fromkeys(arms))


def select(phase: str = "", only: str = "") -> list[dict]:
    """Experiments in ``phase`` (all when empty), or the ids in ``only``."""
    ids = [i for i in only.replace(" ", "").split(",") if i]
    if ids:
        unknown = set(ids) - {e["id"] for e in EXPERIMENTS}
        if unknown:
            raise SystemExit(f"unknown experiment ids: {sorted(unknown)}")
        return [e for e in EXPERIMENTS if e["id"] in ids]
    if phase and phase not in PHASES:
        raise SystemExit(f"unknown phase {phase!r}; use one of {', '.join(PHASES)}")
    return [e for e in EXPERIMENTS if not phase or e["phase"] == phase]
