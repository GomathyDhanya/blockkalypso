# Phase 2 handoff: what exists, what is off, what was not built

Branch `research/phase2`, on top of `research/session1-baseline`. One commit
per step. Phase 2 ran no evaluation: no comparison, benchmark, latency or
quality claim comes from it.

## Correct the record first

The evaluation matrix should not assume mechanisms that do not exist:

- **Beam search was not built.** It was replaced by a **powers-of-two
  restriction** of the exhaustive search (`choose_pipeline(candidate_set="pow2")`,
  step 02). That restriction kept regret within 2.3% of exhaustive on the
  recorded cases.
- **"Stage-boundary replanning" was not built as a stage-boundary trigger.**
  This executor runs stages concurrently, so there is no single moment when
  control passes from one stage to the next. What exists is **continuous
  per-task reshaping**: the adaptive controller re-decides one stage's block
  size after every upstream task. A new size applies only to blocks that start
  forming afterwards (step 03, verified with a test).
  - It chooses only that stage's block size, either per-tuple or a power of
    two.
  - It never reorders stages, changes operators or touches other stages.
- **Cost-benefit replanning was never built.** Nothing weighs the cost of
  changing a plan against its predicted gain. The controller switches on
  pass-rate hysteresis plus a quality veto, not on predicted latency.

## Built and on by default

| what | where | notes |
|---|---|---|
| Seeded planner sampling, pinned model/tokenizer revision | step 00 | `SemanticPlan.DEFAULT_PLAN_SEED = 0` |
| Join quality caps per selectivity band | step A, `costmodel.KindParams.band_caps` | Used only when a caller passes `selectivity`. Without one, `k_max` gives the old pooled cap. |
| Decision cache, automatic for pipelines of 4+ stages | step D | Keyed on exact inputs, so results are bit-identical to uncached calls. 1-3 stages skip it. No current workload has 4+ stages, so in practice it does not engage. |

The optimizer itself (`choose_k` / `choose_pipeline`) still runs **offline
only**; no query calls it (BASELINE_AUDIT finding A).

## Built but gated off

| what | how to enable | why it is off |
|---|---|---|
| Adaptive per-stage controller (steps 03, 07, A, B) | `KALYPSO_ADAPTIVE_STAGE=1` | No evaluation has measured it. It changes executed block sizes. |
| KV feasibility check (step 06) | `kv_budget_tokens=` on choose_k / choose_pipeline | Safety check for pathological inputs only. Session 1 KV was never binding (peak <= 17.5%). |
| Adaptive selectivity, EWMA (step C, item 9) | `adaptive_selectivity=True` on predict_pipeline / choose_pipeline | Turning it on makes the validated model a moving target, so it needs its own evaluation. Nothing sets it; a test checks the sources. |
| Pow2 search restriction (step 02) | `candidate_set="pow2"` | Exhaustive stays the default, so recorded results are unchanged. |

How the controller behaves:

- It manages block stages after the first.
- Its candidate sizes are the search's powers of two, up to the query's block
  size. It starts at the query's own size.
- It votes on the upstream stage's pass rate, needing 3 consecutive votes to
  switch.
- Before switching to blocks it asks the cost model's quality cap,
  passing the managed stage's **own** pass rate as the selectivity. If the
  preferred size is vetoed, it steps down through the smaller powers of two.
  If every size above 1 is vetoed, the stage stays per-tuple.
- With the committed fit it can block short-row filters (cap 20) and
  short-row joins at selectivity >= 0.6 (cap 6). Long-row filters, other
  joins and all maps stay per-tuple.

## Deliberately not built

- **Beam search:** the powers-of-two restriction already cuts the 4-stage
  search from 8.6-20 s to 28-38 ms, with little regret.
- **Cost-benefit replanning:** there was no measured gain to weigh it
  against.
- **A literal stage-boundary trigger:** the executor runs stages concurrently,
  so there is no boundary to hook (see above).
- **A predictive KV-cost term:** KV was never the bottleneck; queueing was.
- **Concurrency or admission modelling in the KV check:** same evidence.

## Still genuinely open

1. **The maps quality metric.** ROUGE-1 against per-tuple output measures
   wording, not answers, and there is no prompt-matched control (a block
   prompt with one row). Maps stay at k = 1 until that control exists
   (QUALITY_CAP_DECISION.md).
2. **The join composition error.** Stage overlap explains only part of the
   ~20% held-out pipeline error, and the join-first plans where it
   concentrates have never been traced (RESOURCE_MODEL.md). The Part 2 run
   includes `pipe_join_filter_j62_f90`, so its trace is a first look, not an
   analysis.
3. **The selectivity-band join caps rest on thin evidence.**
   - One fit row per (workload, k), and repeat-level ΔF1 noise is unknown.
   - The 0.47-0.50 conflict (fanout50/opposite pass at k = 4, the pipe joins
     fail) is unexplained.
   - movie_fanout50 stays at 1 because it fails at k = 2 first.
4. **`choose_pipeline(quality=True)` does not use the band caps.** It passes
   no selectivity, so pipeline search still gets the pooled join cap. Only
   `k_max`, `choose_k` and the controller take a selectivity.
5. **The controller's selectivity is cold at the start.** Until a managed
   stage's own tasks have finished, its selectivity is unknown and the pooled
   cap (1 for joins) applies.

## Part 2: functional run (prepared, not yet run)

`research/phase2/gpu/modal_phase2.py` runs the finished mechanisms on real
workloads, with the Session 1 image and settings and telemetry on:

- **Setup:** controller on, KV check and automatic cache through a real
  optimizer step, adaptive selectivity off.
- **Workloads:** four 2-stage workloads; arms are per-tuple, static (4,4)
  and the optimizer's choice; 1 warmup and 2 repeats.
- **Output:** a functional check only. The summary counts finished queries,
  controller decisions and arms with outputs.

```
modal run --detach research/phase2/gpu/modal_phase2.py::launch
modal run research/phase2/gpu/modal_phase2.py::status
modal volume get blockkalypso-results research-phase2 ./research/phase2/results/
python -m research.phase2.functional_summary research/phase2/results/P2-functional
```

Raw output lands under `research-phase2/P2-functional/`: traces, GPU and
`/metrics` samples, `env.json`, `tokens.json`, `optimizer.json`, per-part
`run-on.jsonl` and `functional_summary.json`. **Status: not yet launched.**
Record the result here once it has run.
