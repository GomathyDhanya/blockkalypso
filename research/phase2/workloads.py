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
"""

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


def register() -> list[str]:
    W.WORKLOADS["rpipe_s50_drift"] = load_rpipe_drift
    return ["rpipe_s50_drift"]


NAMES = register()
