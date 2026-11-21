"""Phase 2 research workloads, registered into eval/workloads.py's WORKLOADS.

- ``rpipe_s50_drift``: rpipe_s50 (filter -> join, 100 left x 32 right), with
  the SAME rows and labels in a different order. Its first DRIFT_NEGATIVES
  negative left rows come first, followed by every other row in its original
  order (48 positives among the remaining 76). The filter's pass rate starts
  near 0 and then recovers to about 63%, which drives the adaptive
  controller through drop (switch to per-tuple), recovery (switch back to
  block) and so its quality veto. No catalog workload does this: their rows
  are shuffled, and in label order none drops below 5% and recovers above
  20% (checked before this experiment).

Ordering is the only change: the same reviews, instructions, right table and
ground truth as rpipe_s50, re-indexed.

- ``movie_fanout{75,90}_r{1..4}``: resamples of movie_fanout75/90 for the
  band-cap check (P2-bandcap-resample). They are built the same way as
  ``load_movie_fanout`` (32 positive left reviews; 32 right reviews, a share f
  of them negative; right side shuffled; same instruction and labels). Each
  resample draws with seed r from the reviews NOT used by the original
  (seed 0) draw or by any earlier resample of the same fanout. So every
  sample is a disjoint set of 1,024 pairs.
"""

import random
from dataclasses import replace

from vllm.kalypso.benchmark.eval import workloads as W

from research.baseline import workloads as _baseline  # noqa: F401 -- registers rpipe_*

DRIFT_NEGATIVES = 24


def load_rpipe_drift(n=None, base="rpipe_s50", name="rpipe_s50_drift"):
    w = W.WORKLOADS[base](n)
    left = {i for i, _ in w.positives}
    negatives = [r for r in w.rows if r.idx not in left][:DRIFT_NEGATIVES]
    head = {r.idx for r in negatives}
    order = negatives + [r for r in w.rows if r.idx not in head]
    new_index = {r.idx: i for i, r in enumerate(order)}
    rows = [W.Row(i, r.text) for i, r in enumerate(order)]
    return replace(
        w, name=name,
        data_path=W._movie_csv(f"{name}_{len(rows)}",
                               [{"reviewText": r.text, "reviewId": ""} for r in rows]),
        rows=rows,
        positives=frozenset((new_index[i], j) for i, j in w.positives))


RESAMPLES = 4


def _fanout_draw(f, side, seed, positives, negatives):
    want = round(f * side)
    rng = random.Random(seed)
    pool = rng.sample(positives, 2 * side - want)
    left, right = pool[:side], pool[side:] + rng.sample(negatives, want)
    rng.shuffle(right)
    return left, right


def load_fanout_resample(base, r, n=None):
    side = W.FANOUT_SIDE if n is None else n
    f = W.FANOUTS[base]
    reviews = W._sembench_reviews()
    used = set()
    for seed in range(r + 1):  # seed 0 is the original draw
        positives = [x for x in reviews if W._positive(x) and id(x) not in used]
        negatives = [x for x in reviews if not W._positive(x) and id(x) not in used]
        left, right = _fanout_draw(f, side, seed if seed else W.FANOUT_SEED,
                                   positives, negatives)
        used |= {id(x) for x in left + right}
    name = f"{base}_r{r}"
    return W.Workload(
        name=name, kind=W.JOIN,
        data_path=W._movie_csv(f"{name}_{side}_left", left),
        right_path=W._movie_csv(f"{name}_{side}_right", right),
        instruction=W.MOVIE_OPPOSITE_INSTRUCTION,
        rows=[W.Row(i, x["reviewText"]) for i, x in enumerate(left)],
        right_rows=[W.Row(j, x["reviewText"]) for j, x in enumerate(right)],
        positives=frozenset((i, j) for i in range(side)
                            for j, x in enumerate(right) if not W._positive(x)))


def register() -> list[str]:
    W.WORKLOADS["rpipe_s50_drift"] = load_rpipe_drift
    names = ["rpipe_s50_drift"]
    for base in ("movie_fanout75", "movie_fanout90"):
        for r in range(1, RESAMPLES + 1):
            name = f"{base}_r{r}"
            W.WORKLOADS[name] = lambda n=None, base=base, r=r: load_fanout_resample(base, r, n)
            names.append(name)
    return names


NAMES = register()
