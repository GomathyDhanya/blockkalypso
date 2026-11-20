"""Regression test: choose_pipeline(candidate_set="pow2") reproduces the
exhaustive-vs-powers-of-two comparison already measured in
research/baseline/experiments/optimizer_scaling.py
(results: research/baseline/results/optimizer_scaling.{jsonl,md}).

Same model (committed v3 costmodel.json), same synthetic pipelines (400 rows,
31 tokens/row, max 120, 32 right rows, selectivity 0.5), same families. Only
depths 1-3 are rerun: exhaustive search at depth 4 takes 8-20 s per case.
Expected picks and predicted latencies are the recorded values.
"""

import itertools
import json
from dataclasses import fields
from pathlib import Path

import pytest

from vllm.kalypso.benchmark.eval import costmodel as cm

FITTED = Path(cm.__file__).parent / "results/analysis/costmodel.json"

ROWS, TOKENS, MAX_TOKENS = 400, 31.0, 120.0
RIGHT_ROWS, RIGHT_TOKENS, RIGHT_MAX = 32, 31.0, 120.0
SELECTIVITY = 0.5
FAMILIES = {
    "filters": ["filter"] * 3,
    "filter_join": ["filter", "join", "filter"],
    "mixed": ["filter", "join", "map"],
}

# (family, depth) -> (exhaustive pick, its predicted s, pow2 pick, its predicted s),
# from research/baseline/results/optimizer_scaling.jsonl.
RECORDED = {
    ("filters", 1): ((3,), 1.8466613372709741, (4,), 1.8589977704051721),
    ("filters", 2): ((3, 2), 2.696928075725969, (4, 2), 2.709264508860167),
    ("filters", 3): ((3, 2, 2), 3.387645549748875, (2, 2, 2), 3.4642956253913777),
    ("filter_join", 1): ((3,), 1.8466613372709741, (4,), 1.8589977704051721),
    ("filter_join", 2): ((1, 2), 30.911238559501857, (1, 2), 30.911238559501857),
    ("filter_join", 3): ((1, 30, 2), 35.79240854796947, (1, 32, 2), 35.28575836803469),
    ("mixed", 1): ((3,), 1.8466613372709741, (4,), 1.8589977704051721),
    ("mixed", 2): ((1, 2), 30.911238559501857, (1, 2), 30.911238559501857),
    ("mixed", 3): ((1, 31, 1), 40.4785485790185, (1, 32, 1), 39.80181966260049),
}
# Largest |regret| baseline measured for pow2 vs exhaustive (filters, depth 3:
# +2.3%), with a little headroom.
MAX_ABS_REGRET = 0.025


@pytest.fixture(scope="module")
def model():
    # Same construction as research/baseline/instrumentation/optimizer_profile.load_model.
    d = json.loads(FITTED.read_text())
    names = {f.name for f in fields(cm.KindParams)}
    kinds = {k: cm.KindParams(**{n: v for n, v in p.items() if n in names})
             for k, p in d["kinds"].items()}
    return cm.Model(g=d["g"], alpha=d["alpha"], beta=d["beta"], gamma=d["gamma"],
                    delta=d["delta"], tail=d["tail"], kinds=kinds,
                    f1_slack=d["f1_slack"], lam=d["lambda"], retry=d["retry"])


def _pipe(family, depth):
    ops = tuple(FAMILIES[family][:depth])
    join = "join" in ops
    return cm.Pipeline(ops, ROWS, TOKENS, MAX_TOKENS, SELECTIVITY,
                       RIGHT_ROWS if join else 0, RIGHT_TOKENS if join else 0.0,
                       RIGHT_MAX if join else 0.0)


@pytest.mark.parametrize("family,depth", sorted(RECORDED))
def test_pow2_matches_recorded_comparison(model, family, depth):
    pipe = _pipe(family, depth)
    ex_blocks, ex_lat, p2_blocks, p2_lat = RECORDED[(family, depth)]

    exhaustive = model.choose_pipeline(pipe)  # default: unchanged exhaustive path
    assert exhaustive["blocks"] == ex_blocks
    assert exhaustive["latency_s"] == pytest.approx(ex_lat, rel=1e-9)

    pow2 = model.choose_pipeline(pipe, candidate_set="pow2")
    assert pow2["blocks"] == p2_blocks
    assert pow2["latency_s"] == pytest.approx(p2_lat, rel=1e-9)

    # The flag gives exactly what optimizer_scaling.py's explicit candidate
    # list (full pow2 product, filtered by k_max inside choose_pipeline) gave.
    explicit = model.choose_pipeline(pipe, candidates=list(
        itertools.product(cm.POW2_CANDIDATES, repeat=depth)))
    assert explicit["blocks"] == pow2["blocks"]

    regret = pow2["latency_s"] / exhaustive["latency_s"] - 1
    assert abs(regret) <= MAX_ABS_REGRET


def test_default_is_exhaustive(model):
    pipe = _pipe("filters", 2)
    assert (model.choose_pipeline(pipe)
            == model.choose_pipeline(pipe, candidate_set="exhaustive"))


def test_bad_candidate_set_rejected(model):
    pipe = _pipe("filters", 1)
    with pytest.raises(ValueError):
        model.choose_pipeline(pipe, candidate_set="beam")
    with pytest.raises(ValueError):
        model.choose_pipeline(pipe, candidates=[(1,)], candidate_set="pow2")
