# Resource model: do pipeline stages overlap? (Phase 2, step 01)

A decision record for whoever next changes the pipeline composition term in
`vllm/kalypso/benchmark/eval/costmodel.py` (`predict_pipeline`,
`L = λ · Σ_i L_i`, λ = 0.98). Nothing in the cost model was changed in this
step.

## Question

COST_MODEL_VALIDATION.md §2: held-out pipeline MAPE is 20.3%, and 20.2% when
the model is given the true cardinalities, with 108 of 143 held-out arms
over-predicted. If the cardinality is right but the total is wrong, the
suspect is the composition term. The specific question here: do stages run
concurrently, so that summing per-stage latencies double-counts time?

## What the executor allows (code)

There is no barrier between stages inside a segment
(`vllm/kalypso/execution/pipeline_execution.py`). The scheduling loop handles
each completed task as soon as `asyncio.wait(..., FIRST_COMPLETED)` returns
it. Its output rows are enqueued on the next stage right away: as one task
per row, through `enqueue_block_rows` for a downstream block stage (emitted
once `block_size` rows have accumulated), or via the fanout. The next loop
iteration launches them. The only wait for the whole upstream stage is the
final partial block (`flush_partial_block_tasks`). So overlap is
architecturally possible. The question is how much there is in practice.

## Measurement

Data: the Session 1 KALYPSO_TRACE files. `task_start` and `task_end` are
taken on one process clock (`t_mono`). A task's interval runs from Kalypso
dispatching it to the vLLM request returning, so it includes any wait in
vLLM's queue. Script: `research/baseline/analysis/stage_overlap.py`.
Per-query output for E1: `research/baseline/results/stage_overlap_E1.jsonl`.

```
python -m research.baseline.analysis.stage_overlap \
    research/baseline/results/gpu/research-session1/E1-s*/trace-on.jsonl.gz
```

Plans: E1 runs the 2-stage plan "filter (with Cartesian fanout) → join
predicate" in four arms (per-tuple or k=4 block on each side), at five join
selectivities (5–95%), with 1 warm-up and 5 repeats. That makes 120 queries
on a server with no other load. E3-load-v2 runs the same plans under
background load (48 queries).

### Result: stages overlap measurably, but only by a little

In **all 120** E1 queries, and all 48 E3 queries, stage 2's first task starts
before stage 1's last task ends. The overlap is short, though, because stage 1
is short. Medians per arm over the five selectivities:

| arm (E1, no load) | n | latency s | stage-2 start − stage-1 end (s) | stage-2 tasks started before stage 1 ended | latency with both stages running | (span₁ + span₂) / latency |
|---|---|---|---|---|---|---|
| per-tuple filter → per-tuple join | 30 | 10.14 | −1.20 | 85% | 12% | 1.11 |
| per-tuple filter → block join (k=4) | 30 | 7.45 | −0.92 | 93% | 12% | 1.11 |
| block filter (k=4) → per-tuple join | 30 | 13.98 | −0.49 | 40% | 4% | 1.03 |
| block filter (k=4) → block join (k=4) | 30 | 9.72 | −0.29 | 90% | 6% | 1.03 |

Over all 120 queries, the median of (span₁ + span₂) / latency is **1.070**
(p10 1.012, p90 1.190). The sum of the stage spans exceeds the wall time by a
median of 0.79 s. E3 under load looks the same: 1.02–1.09 per arm.

**Shape of the overlap.** One E1-s80 query, per-tuple → per-tuple, 18.6 s
total:

- Stage 1 (100 tasks): every task starts within 0.11–0.17 s. They finish
  between 0.52 s and 2.41 s.
- Stage 2 (1,600 tasks): tasks start between 0.74 s and 2.56 s, as stage-1
  results arrive. They finish between 3.06 s and 18.62 s.

So the overlap is a single window: stage 1's drain, when its completions trickle
in and stage-2 work is already being dispatched. After that, about 80% of the
query (`tail_after_s1`, median 0.80–0.87) is stage 2 running alone.

**Overlap as a share of stage 1.** Call ω the share of stage 1's span that
overlaps stage 2: ω = (stage-1 last end − stage-2 first start) / span₁.

- Per-tuple stage 1: ω is 0.60–0.82 from 20% selectivity upward (0.40–0.45 at
  5%).
- Block stage 1 (k=4): ω is lower and noisier, 0.11–0.74. Its 25 block tasks
  finish in clumps, so stage-2 dispatch starts later in stage 1's drain.

**How much the additive sum over-counts.** It depends on stage 1's share of
the query:

- Up to 1.27× when stage 1 is a large share: per-tuple stage 1 at 5%
  selectivity, where stage 1 is 73% of latency.
- About 1.01–1.07× when stage 2 dominates.

## Decision record

Stages do overlap measurably at this scale, so the additive term is not
exactly right. But the overlap is bounded by stage 1's drain window. On the
traced plans it accounts for a median 7% (p90 19%) over-count, compared with
+22.6%, the mean error of the 108 over-predicted held-out arms. It explains part of the pipeline
error, not all of it. Since overlap can only be a fraction of the upstream
stage's own time, the formula shape it implies is
`L ≈ (1 − ω_up) · L_up + L_down`. Here ω_up is the overlapped share of the
upstream stage. It depends on the upstream operator kind: about 0.60–0.82 for
per-tuple, lower and more variable for block. A single global λ multiplying
Σ L_i cannot represent this. λ = 0.98 removes 2% uniformly, but the measured
over-count ranges from 0% to 27% and grows with the upstream stage's share of
the query. Whoever next changes `predict_pipeline` should replace λ with a
per-upstream-kind overlap discount on the upstream stage only, fit from these
traces. They should not expect that change alone to close the 20% gap.

## Limits of this evidence

1. **Plan shape coverage.** The held-out pipeline error is concentrated in
   join → filter (`validate-jf`, +9% to +28%). Those runs are historical and
   have no task-level traces, so they cannot be checked this way. Every
   traced 2-stage plan is filter → join. A join-first plan has a long first
   stage, so the drain-window overlap could be a different size there. The
   next step is to trace a join → filter run.
2. **In-pipeline spans are not the cost model's L_i.** The cost model's L_i
   is fitted on single-operator runs. This analysis measures overlap between
   the stages as they ran inside the pipeline. It does not test whether each
   stage's in-pipeline span equals its isolated latency, for example because
   stage-2 requests share vLLM batches with stage 1's tail. That is a
   separate possible cause of the residual error.
3. Task intervals are measured at Kalypso dispatch and completion, not at
   vLLM scheduling, so "both running" means both had requests in flight on
   the server. That is the right notion for wall-time composition, but it is
   not GPU-level concurrency.

## Join-first plans (Phase 2, item 4)

Limit 1 above said the held-out pipeline error sits in join → filter
(`validate-jf`) and that no join-first plan had been traced. This section
takes one pass at it. It is read-only over committed results; there was no
new GPU run and no cost-model change. Script: `research/phase2/join_first.py`.
Output: `research/phase2/results/join_first/` (`arms.jsonl`,
`overlap.jsonl`, `summary.json`).

```
python -m research.phase2.join_first --out research/phase2/results/join_first
```

Coverage: the 4 held-out `pipe_join_filter_*` workloads (119 arms), their
`alone-join` runs (the join alone at k = 1, 2, 4, 6), and the only
join-first task traces that exist. Those are pipe_join_filter_j62_f90 in
the Phase 2 Modal runs: 12 queries at k = 10, 2 (the P2-compare and
P2-compare-v2 controller-off phases) and 2 at k = 4, 4 (P2-functional; any
query with a controller decision is dropped). The catalog has **no join →
filter → join plan**, so that shape is untested.

### 1. Cardinality: this is the main cause

`costmodel.passed_rows` returns None unless the plan starts with a filter.
So for join-first plans the "given cardinality" prediction in
COST_MODEL_VALIDATION.md is the full prediction, and the finding there that
"the error persists given the true cardinality, so it is composition" was
never tested on these plans. Here the filter stage is re-predicted from the
pairs the join really passed: (calls - the join stage's calls) × k2, which
agrees with `alone-join` `rows_out`. Every other term is left unchanged.

| arms | n | MAPE, as predicted | MAPE, true join cardinality | bias as predicted → true card | over-predicted (as predicted → true card) |
|---|---|---|---|---|---|
| all | 119 | 21.6% | **7.5%** | +17.2% → −5.5% | 95 → 25 |
| per-tuple join (k1 = 1) | 22 | 36.0% | 5.0% | +36.0% → +3.9% | 22 → 18 |
| k1 = 2-8 | 62 | 23.0% | 5.9% | +23.0% → −5.2% | 61 → 6 |
| k1 ≥ 9 | 35 | 10.2% | 12.1% | −4.9% → −12.0% | 12 → 1 |

**The model's join pass share is far off for this instruction.** The
join's pass share (recall and false-positive rate) is fitted on the movie
joins, and it predicts the join keeps 350-670 of 1,024 pairs. On these
workloads the per-tuple join kept 2-57, and the block joins kept 89-396
(`alone-join`). The over-prediction at k1 = 1 is almost entirely a filter
stage the model sizes at about 540 rows when it really sees about 20.

### 2. The join's standalone cost matches, but not inside the pipeline

**Run alone, the model's join stage is accurate.** For the join alone
(`alone-join`), the model predicts it within 2-6%: k = 1 is 10.4-11.1 s
measured against 10.6-11.0 s predicted, and k = 2-6 is 6.9-7.4 s measured
against 6.5-7.0 s predicted.

**Inside the pipeline, the join takes longer.** In the traced j62_f90 k = 4, 4
queries, the join stage spans 8.85 s, against 7.04 s alone (+26%), because the
filter's requests share the server with it (section 3).

### 3. Overlap is large, and it roughly cancels the contention

| traced arm (j62_f90) | n | latency s | join span s | filter span s | filter tasks started before the join ended | (span₁ + span₂) / latency | latency after the join ended |
|---|---|---|---|---|---|---|---|
| join k=10 → filter k=2 | 12 | 9.91 | 9.53 | 7.12 | 97% | **1.68** | 3% |
| join k=4 → filter k=4 | 2 | 9.54 | 8.85 | 7.32 | 97% | **1.70** | 6% |

This is the opposite of filter → join (additive ratio about 1.07, with
about 80% of the query after stage 1).
- **The filter runs inside the join's span.** Join results stream into the
  filter as they finish, so almost all filter work happens while the join is
  still running.
- **A span-overlap discount does not fit here.** The additive ratio is
  about 1.7, but once the cardinality is right, adding the stage costs
  lands 1-6% under the traced latency at these two arms (9.31 s against
  9.91 s, and 9.42 s against 9.54 s). The server is saturated, so
  the filter's work slows the join instead of being hidden. Overlap and
  in-pipeline contention roughly cancel.
- **Implication:** the `(1 − ω_up) · L_up + L_down` form proposed above for
  filter → join does not carry over to join-first plans. On a saturated
  server, summing the work (λ·Σ L_i) is close to right for join → filter at
  k1 ≤ 8.

### What is still open

1. **Large first-stage join blocks (k1 ≥ 9) are under-predicted** by a
   median 9% (pipe arms) to 20% (`kalypso_block:16/32`), even at the true
   cardinality. Examples (j62_f90): block:16 is 13.2 s against 11.6 s
   predicted, and block:32 is 18.6 s against 14.6 s.
   - **What is missing:** `alone-join` has no k ≥ 9 measurement. So this
     data cannot separate a wrong standalone join cost at large k from
     in-pipeline contention at large k.
   - **What it would take:** a traced run of the join alone at k = 9-32.
2. **Join → filter → join is untested:** no such workload exists.
3. **The overlap evidence is thin:** it comes from one workload (j62_f90),
   two arms (14 queries), on the Phase 2 controller-off runs. There are no
   per-tuple join traces. That makes the "they cancel" reading plausible,
   not established.

### Needs its own decision (not acted on)

- **The join cardinality model.** The fitted join pass share does not
  transfer to the same-sentiment instruction, and the error is large at
  k = 1. Fixing it, for example with per-instruction pass shares or by
  using observed selectivity, changes plans, so it is a cost-model change
  for a separate step.
- **COST_MODEL_VALIDATION.md finding 3** ("given true cardinality the error
  persists, so it is composition") does not hold for join-first plans,
  because the given-cardinality path never ran for them. That document and
  `passed_rows` should be corrected in a separate change. The 20.2%
  "given cardinality" held-out figure mixes real given-cardinality
  predictions for filter-first plans with plain predictions for join-first
  plans.
- **Replacing λ** with the per-upstream discount proposed above should be
  scoped to filter-first plans only, given section 3.
