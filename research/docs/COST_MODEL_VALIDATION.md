# Cost-model validation (Session 1, Step 5)

**Question.** How well does the original cost model (v3,
`vllm/kalypso/benchmark/eval/costmodel.py`, coefficients in
`results/analysis/costmodel.json`) predict latency, rank plans and estimate
cardinality, on data it was and was not fitted to?

**Data.** Every committed GPU result under
`vllm/kalypso/benchmark/eval/results/` (L40S, Qwen2.5-7B-Instruct, 1 warm-up
+ 3 repeats, arm = median over repeats as `eval/analyze.py` computes it):
484 Kalypso arms, 266 single-operator and 218 pipeline. No new GPU runs are
used here; the new Session 1 runs (5 repeats, with telemetry) are analysed
in BASELINE_RESULTS.md once they finish.

**Split.** "fit" is every phase `costmodel.main` fits on; "held_out" is the
phases it excludes (`HELD_OUT`: `validate3`, `validate-jf`, `validate4`).
The refit reproduces the committed coefficients to within 3e-11 relative
difference, so these are the model's own numbers, not a re-implementation.

**Reproduce.**
`python -m research.baseline.analysis.costmodel_validation --out research/baseline/results/costmodel_validation`
(about one minute on CPU). Full tables:
`research/baseline/results/costmodel_validation/summary.md`; per-arm data:
`arms.jsonl` in the same folder.

## 1. Metrics and how zeros and direction are handled

- Latency: MAE, RMSE (seconds), MAPE, median and p90 APE, max APE.
  Under-prediction (pred < measured) and over-prediction are counted and
  averaged separately, because they have different consequences: under-
  prediction makes a plan look cheaper than it is, over-prediction makes the
  optimizer avoid it.
- Ranking, per run with at least 3 measured arms: Spearman correlation of
  predicted vs. measured latency, top-1 agreement (the model's pick is the
  measured fastest), and regret = measured latency of the model's pick /
  measured fastest − 1.
- Cardinality: q-error = max(pred/actual, actual/pred) on the rows a filter
  passes. If both are zero, q = 1; if exactly one is zero, the arm is
  reported as a "zero mismatch" and excluded from q statistics (none
  occurred in this data).
- Two prediction modes: **full** (the model predicts both cardinality and
  latency) and **given measured cardinality / tokens** (isolates the latency
  terms from the cardinality estimate).

## 2. Results

### Latency

| split | prediction | n | MAPE | median APE | p90 APE | under (n, mean) | over (n, mean) |
|---|---|---|---|---|---|---|---|
| fit | single operator | 173 | 4.3% | 2.8% | 6.7% | 97, −4.0% | 76, +4.6% |
| held out | single operator | 93 | 6.6% | 4.1% | 15.1% | **86**, −6.9% | 7, +2.9% |
| fit | pipeline | 75 | 13.2% | 7.3% | 33.1% | 26, −13.8% | 49, +12.9% |
| fit | pipeline, given cardinality | 75 | 10.9% | 7.4% | 28.0% | 45, −11.1% | 30, +10.5% |
| held out | pipeline | 143 | 20.3% | 17.5% | 40.3% | 35, −13.3% | **108**, +22.6% |
| held out | pipeline, given cardinality | 143 | 20.2% | 17.5% | 40.3% | 39, −13.1% | 104, +22.8% |

Findings:

1. **Single operators are well modelled** in-fit (4.3% MAPE, matching
   STUDY.md) but the held-out error is **one-sided**: 86 of 93 held-out arms
   are under-predicted. Maps are the weak operator (14.6% MAPE overall; 31.7%
   on the four long-output held-out maps, all under-predicted).
2. **Pipeline error is 3-5x the single-operator error** and on held-out data
   it is mostly **over-prediction** (108 of 143 arms).
3. **Given the true cardinality, held-out pipeline error does not improve**
   (20.3% → 20.2%). So the pipeline error is in the composition term
   (λ·ΣL_i, i.e. how stages overlap), not in the selectivity estimate.
4. The held-out pipeline error is concentrated in **join → filter** plans
   (`validate-jf`): median over-prediction +9% (join selectivity 18%), +19%
   (50%), +16-28% (62%). Filter → join and filter → map held-out arms are
   within ±9%; map → filter is under-predicted by 16%.

### Ranking

| split | level | runs | Spearman median (min) | top-1 | regret mean (max) |
|---|---|---|---|---|---|
| fit | single | 34 | 1.00 (0.40) | 82% | 0.5% (14.4%) |
| held out | single | 16 | 0.99 (0.80) | 69% | 0.7% (3.2%) |
| fit | pipeline | 9 | 0.70 (−0.60) | 67% | 2.7% (15.7%) |
| held out | pipeline | 7 | 0.54 (0.32) | **0%** | **10.2% (23.8%)** |

For single operators the model ranks block sizes almost perfectly and the
wrong picks cost under 1% on average. For held-out pipelines it never picks
the measured-fastest plan, and its pick is on average 10% (at worst 24%)
slower than the best measured arm.

### Cardinality (q-error)

| split | quantity | n | median q | p90 q | max q |
|---|---|---|---|---|---|
| fit | filter pass | 87 | 1.02 | 1.11 | 2.82 |
| fit | join pass | 72 | 1.10 | 1.36 | 2.29 |
| held out | filter pass | 41 | 1.32 | 1.80 | 5.94 |
| held out | join pass | 36 | **1.83** | **4.75** | **15.19** |
| held out | pipeline stage-1 out | 17 | 1.03 | 1.07 | 1.10 |

The cardinality model takes pass rates from recall and false-positive rate
fitted per (operator, k). On held-out data, join pass counts are off by
1.8x at the median and by 15x at worst; held-out filter pass counts are
mostly over-estimated (32 of 41).

### KV memory and queueing

The cost model has **no KV-memory, queueing or preemption term**. What the
historical runs measured: peak KV cache use median 6.9%, p90 14.7%, max
75.5% of the cache; peak waiting requests median 151, p90 847, max 2011;
**0 of 484 arms had any preemption**. So at these sizes (≤ 400 rows, 7B
model, one L40S) KV capacity was not the constraint and the scheduler's
waiting queue was; the model's latency terms absorb queueing implicitly.

## 3. What this means for the research claims

- RQ "the cost model is accurate": supported for single operators in the
  regime it was fitted on; **not supported** for multi-stage plans outside
  the fit (held-out MAPE 20%, top-1 0%, regret up to 24%).
- The dominant error is the **pipeline composition** term and the **join
  cardinality** estimate, not per-call latency. A Session 2 model that only
  refines per-call latency will not fix plan choice.
- Because KV was never binding and preemption never occurred in the
  historical data, any claim about KV-aware planning needs new runs that
  push KV use up (more rows, longer contexts, concurrent load). Session 1's
  E3 load experiment is the first step.

## 4. Threats to validity

- The historical data has 3 repeats per arm, and 53 of 66 L40S result files
  were recorded from a dirty tree (ENVIRONMENT.md §4), so the code behind
  each number is not exactly recoverable.
- Arms are medians; per-repeat variance is not used in these metrics.
- Held-out phases are few (7 pipeline runs) and dominated by join → filter;
  the held-out pipeline numbers should be read as "on these plan shapes".
- The model's ranking is evaluated among the measured arms only, not over
  its whole search space; regret against unmeasured plans is unknown.
