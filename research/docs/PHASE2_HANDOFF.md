# Phase 2 handoff

Branch `research/phase2`, on top of the Session 1 baseline. One commit per
step, `phase2/00` (`5eea5f7`) to the join-first analysis (`e480d47`,
`16f5d5b`). This is the final handover state.

Phase 2 had three GPU runs of its own:
- a functional run, P2-functional;
- one static vs adaptive controller comparison, run twice: P2-compare, and
  P2-compare-v2 after the cold-start fix;
- one quality resample check, P2-bandcap-resample.

These are confirmation and characterization runs on 4 workloads with
min-max spread. They are not the evaluation matrix, and no paper claim
should rest on them alone.

## Correct the record first

The evaluation matrix should not assume mechanisms that do not exist:

- **Beam search was not built.** A **powers-of-two restriction** of the
  exhaustive search replaced it (`choose_pipeline(candidate_set="pow2")`,
  step 02). It kept regret within 2.3% of exhaustive on the recorded cases
  and cut the 4-stage search from 8.6-20 s to 28-38 ms.
- **"Stage-boundary replanning" was not built as a stage-boundary
  trigger.**
  - **Why:** the executor runs stages concurrently, so there is no single
    boundary.
  - **What exists instead:** continuous per-task reshaping. The adaptive
    controller re-decides one stage's block size after each upstream task,
    and a new size applies only to blocks that start forming afterwards.
  - **Its scope:** it never reorders stages, changes operators or touches
    other stages.
- **Cost-benefit replanning was never built.** The controller switches on
  pass-rate hysteresis plus a quality veto, not on predicted latency.
- **The optimizer still runs offline only.** No query calls
  `choose_k` / `choose_pipeline` (BASELINE_AUDIT finding A).

## Built and on by default

| what | where | notes |
|---|---|---|
| Seeded planner sampling, pinned model and tokenizer revision | step 00 | `SemanticPlan.DEFAULT_PLAN_SEED = 0` |
| Join quality caps per selectivity band: **machinery only, every band at 1** | steps A and revert, `costmodel.KindParams.band_caps`, `BAND_CAP_CEILING` | See "Join band cap: final state". `k_max` and `choose_k` with any selectivity equal the pooled join cap, which is tested. |
| Decision cache, automatic for pipelines of 4+ stages | step D | Exact keys, so results are bit-identical to uncached calls. No current workload has 4+ stages, so in practice it never engages. |

## Built but gated off

| what | how to enable | why it is off |
|---|---|---|
| Adaptive per-stage controller (steps 03, 07, A, B, cold-start fix) | `KALYPSO_ADAPTIVE_STAGE=1` | It changes executed block sizes. It has had one 4-workload comparison and not the evaluation matrix. |
| KV feasibility check (step 06) | `kv_budget_tokens=` on `choose_k` / `choose_pipeline` | A safety check for pathological inputs. Session 1 KV was never binding (peak <= 17.5%), and in P2-functional it rejected nothing. |
| Adaptive selectivity, EWMA (step C) | `adaptive_selectivity=True` on `predict_pipeline` / `choose_pipeline` | Turning it on makes the validated model a moving target, so it needs its own evaluation. A test checks that nothing sets it. |
| Pow2 search restriction (step 02) | `candidate_set="pow2"` | Exhaustive stays the default, so recorded results are unchanged. |

**How the controller behaves, as committed:**
- **Scope:** it manages block stages after the first. The candidates are the
  search's powers of two, up to the query's block size, starting at the
  query's own size.
- **Voting:** it votes on the upstream stage's per-task pass rate (window 16,
  low 0.05, high 0.20, 3 consecutive votes to switch).
- **Switching to per-tuple:** this is never gated.
- **Switching to blocks:** it asks the cost model's quality cap, passing
  the managed stage's own pass rate as the selectivity. On a veto it steps
  down through the smaller powers of two (step B). If every size above 1 is
  vetoed, the stage stays per-tuple.
- **Join cold start:** a switch to blocks waits until the join has 16
  finished tasks of its own. Filters and maps are not gated.
- **End-of-query telemetry:** an `adaptive_summary` event gives the
  deferrals, the veto checks, and per stage its own tasks and reading.
- **Net effect with the committed fit:** only short-row filters (cap 20)
  can switch to blocks. Joins (cap 1 in every band), long-row filters and
  maps can only go to per-tuple. On real data the controller has only ever
  made `static -> tuple` switches (P2-functional, P2-compare, -v2).

## Deliberately not built

- **Beam search:** the pow2 restriction already brought the search time
  down, with little regret.
- **Cost-benefit replanning:** there is no measured gain to weigh it
  against.
- **A literal stage-boundary trigger:** there is no boundary to hook (see
  above).
- **A predictive KV-cost term, and concurrency or admission modelling in
  the KV check:** KV was never the bottleneck; queueing was.

## No longer applicable

- **Wire band caps into the initial plan** (`choose_pipeline(quality=True)`
  passing a selectivity). Every join band now caps at the pooled value, so
  passing a selectivity cannot change a plan.
- **Fix the controller's switch-up trigger for joins.** P2-compare-v2
  found the join's deferred switch to blocks is never re-checked: the
  reading arrives after the last upstream vote. With a join cap of 1 the
  re-check would always end in a veto, so the trigger gap has no effect.
  Revisit both only if a future band cap above 1 is supported by evidence.

## Join band cap: final state

- **Before:** step A (`5c6c57e`) fitted k = 6 for short-row joins at
  selectivity >= 0.6. The fit used one draw each of movie_fanout75 and
  movie_fanout90. Greedy decoding makes repeats identical, so they could not
  show noise.
- **The check:** P2-bandcap-resample (`2eebf56`, `794dba1`) re-drew 4
  disjoint 32 x 32 samples of each.

| sample | per-tuple F1 | ΔF1 k=4 | ΔF1 k=6 |
|---|---|---|---|
| fanout75 original | 0.806 | +0.109 | +0.087 |
| fanout75 r1-r4 | 0.809-0.941 | +0.014, -0.015, -0.063, -0.085 | -0.036, -0.139, -0.139, -0.128 |
| fanout90 original | 0.819 | +0.125 | +0.071 |
| fanout90 r1-r4 | 0.848-0.938 | -0.066, -0.072, +0.036, +0.041 | -0.096, -0.131, -0.027, -0.010 |

- **The originals reproduced.** On fresh samples, k = 6 failed the -0.02
  check on 7 of 8 and k = 4 on 4 of 8, with negative mean ΔF1 at both.
- **The original draws were favourable.** They had the lowest per-tuple F1
  of their family.
- **Final state (`7f43749`):** `costmodel.BAND_CAP_CEILING = 1` holds every
  fitted band cap at the pooled join cap. The `costmodel.json` `band_caps`
  are all 1, and the machinery is kept (`selectivity_band`, `band_caps`,
  `fit_band_caps`, the selectivity-aware veto). Lifting the ceiling gives
  back the raw fit, which is tested.
- **For any future band split:** fit it over several independent samples
  per workload, not one draw (QUALITY_CAP_DECISION.md).

## Join-first composition error (item 4)

This was one read-only pass over committed results
(`research/phase2/join_first.py`, `research/phase2/results/join_first/`).
The full write-up is in RESOURCE_MODEL.md, section "Join-first plans".

**Explained:**
- **Most of the held-out join -> filter error is the join's predicted
  cardinality, not composition.**
  - **Why it was missed:** `costmodel.passed_rows` returns None for
    join-first plans, so COST_MODEL_VALIDATION's "given cardinality"
    prediction never used the real cardinality there.
  - **The effect:** with the join's real pass count, held-out MAPE over 119
    arms drops from 21.6% to 7.5%, and the signed bias goes from +17% to
    -5.5%.
  - **The cause:** the model predicts the join keeps 350-670 of 1,024
    pairs. It actually kept 2-57 per-tuple and 89-396 blocked.
- **The join's standalone cost is right.** The model matches the join run
  alone within 2-6% at k = 1-6.
- **Overlap differs from filter -> join.** In join-first plans the filter
  runs almost entirely inside the join's span: additive ratio 1.68-1.70,
  against 1.07 for filter -> join. In-pipeline contention slows the join
  about 26%. The two roughly cancel, so summing the work (λ·Σ L_i) is close
  to right here.

**Still open:**
- **Large first-stage blocks (k1 >= 9) are under-predicted** by 9-20% even
  at the true cardinality. There is no join-alone measurement at k >= 9 to
  tell standalone cost from contention.
- **Join -> filter -> join is untested:** no catalog workload has that
  shape.
- **The overlap evidence is thin:** one workload (j62_f90), two arms and 14
  queries, with no per-tuple join traces.

**Needs its own decision (not acted on):**
- **The join cardinality model.** The fitted pass share does not transfer
  across join instructions.
- **Correcting COST_MODEL_VALIDATION.md finding 3 and `passed_rows`.** The
  held-out "given cardinality" 20.2% mixes in plain predictions for
  join-first plans.
- **Replacing λ:** scope the per-upstream overlap discount from step 01 to
  filter-first plans only.

## Still open (unchanged)

1. **The maps quality metric.** ROUGE-1 against per-tuple output measures
   wording, not answers, and there is no prompt-matched control. Maps stay
   at k = 1 (QUALITY_CAP_DECISION.md).
2. **The 0.47-0.50 join conflict.** fanout50 and opposite pass at k = 4,
   while the pipe joins fail. It is unexplained; with every join band at 1
   it does not affect behaviour.

## GPU runs in Phase 2

All runs used an L40S with Qwen2.5-7B-Instruct at the Session 1 pinned
revision and server settings, with temperature 0. Each run's raw output
is in `research/phase2/results/<run>/`, with traces and logs gzipped.

| run | job / commit | what it showed |
|---|---|---|
| P2-functional | `fc-01M7XKD5TWFSQX7YK73FK8VP9T` / `48b3508` | 28 of 28 queries finished across 4 workloads x 3 arms. The KV check was live and rejected nothing. The cache engaged 0 times (all plans are 2 stages). The controller made 4 `static -> tuple` switches and no veto. |
| P2-compare | `fc-01M7XNZ7RFAFM216K8D3BKMYJT` / `6f7c976` | Static vs adaptive, off, on, off, on, 4 repeats each. The adaptive arm moved the drift and s05 joins to per-tuple, raising F1 (drift 0.505 → 0.627). Latency was faster on s05 (1.70 → 1.60 s) and overlapped on drift. The 11 vetoes all ran with no join reading (cold start). |
| P2-compare-v2 | `fc-01M7XSWKP3FFCD7FSAPGF31QDD` / `5d7ca4e` | The same comparison after the cold-start fix: 0 vetoes, 10 deferrals. The join's window filled (896 own tasks) but after the last upstream vote, so there was no switch up. Drift was 7.52 → 6.60 s and s05 1.65 → 1.56 s, both with ranges that don't overlap. The noise floor between phases is about 1%. |
| P2-bandcap-resample | `fc-01M7Y73769KHWWS7P01VWXEQC4` / `2eebf56` | The band cap of 6 failed on fresh samples, so it was reverted (above). |

The F1 gains in P2-compare and -v2 came from moving 0.47-selectivity joins
to per-tuple, which is consistent with the join cap of 1. They are not
evidence for blocking.

**Reproduce the summaries:**
- `python -m research.phase2.functional_summary research/phase2/results/P2-functional`
- `python -m research.phase2.compare_summary research/phase2/results/P2-compare-v2`
- `python -m research.phase2.join_first --out research/phase2/results/join_first`

The Modal entry points are in `research/phase2/gpu/modal_phase2.py`
(`launch`, `launch_compare`, `launch_bandcap`, and their `status*`
counterparts).
