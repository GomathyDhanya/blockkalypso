# Baseline results (Session 1)

Everything measured in Session 1, in one place. Full tables:
`research/baseline/results/gpu_analysis/summary.md` (GPU),
`research/baseline/results/optimizer_scaling.md`,
`research/baseline/results/executor_simulation/summary.md`,
`research/baseline/results/costmodel_validation/summary.md`.
Protocol: BASELINE_METHODOLOGY.md. Hypotheses: RESEARCH_HYPOTHESES.md.

**GPU setup as run.** Modal, 10 containers × 1 L40S (46 GB), launched from
`98e03cb` (clean tree) on 2026-11-20 06:34 UTC; vLLM 0.29.0 + Kalypso overlay,
torch 2.13.0+cu130, driver 620.70.06, Python 3.12.3,
Qwen/Qwen2.5-7B-Instruct revision `a09a35458c70`, bf16. 1 warm-up + 5
repeats per arm, arms interleaved, prefix cache reset before every trial.
**82.3 container-minutes of L40S for the first 10 experiments, plus 38.5 for the E3 rerun (120.8 in total)** (plus image build and model load
before each container's start time). Raw data:
`research/baseline/results/gpu/research-session1/` (traces gzipped).

## 1. Headline numbers

| Finding | Evidence |
|---|---|
| Block filters are ~1.9x faster than per-tuple at every selectivity, with *higher* F1 | E1 rsel*: block:4 1.88-1.90x (✓ all repeats), F1 0.87-0.88 vs 0.70-0.77 |
| Filter block size has an interior optimum; very large blocks lose | E2-filter (1000 rows): k=2 1.75x, 4 2.11x, **8 2.23x**, 16 1.89x, 32 1.36x, 64 0.93x |
| Join blocks give ≤1.5x and degrade quality fast | E2-join: k=4 1.53x (F1 0.83); k=8 F1 0.70; k=16 0.54; k=32 0.40 with pass rate falling 0.56 → 0.17 |
| Map blocks do not help | E2-map: k=2 1.02x, k=4 0.71x, k=8 0.47x |
| In filter → join, only blocking the **join** pays off, and the gain grows with selectivity | rpipe pipe:1,4 vs per-tuple: 0.97x (5%), 1.25x (20%), 1.35x (50%), 1.47x (80%), 1.52x (95%) |
| Blocking the **filter** in a pipeline is always slower | pipe:4,1 0.63-0.84x at every selectivity |
| KV memory is not the binding resource at this scale | peak KV ≤ 17.5% without load (≤ 22% with), **0 preemptions** in every experiment |
| Queueing is | per-tuple join stage wait p95 rises 0.04 s → 3.2 s from 5% to 95% selectivity; vLLM waiting queue up to 1,896 |
| Telemetry overhead: ~5% on per-tuple (400 calls), ~0% on block:8 | E0: 3.77 → 3.96 s; 2.14 → 2.14 s |
| Background load slows every plan but does not change which plan is best | E3 rerun (v2): pipe:1,4 stays best at 0/32/128 streams; its speedup is 1.35x → 1.43x → 1.13x; first E3 run was invalid (§5) |

## 2. Cardinality: the block size changes the selectivity (H2)

Target selectivity is the labelled positive share; achieved is what the
model passes.

| target | per-tuple passes | block:4 passes | block:8 passes |
|---|---|---|---|
| 5% | 5.0% | 7.0% | 6.0% |
| 20% | 14.0% | 19.3% | 17.7% |
| 50% | 31.0% | 41.5% | 40.0% |
| 80% | 50.7% | 64.2% | 66.5% |
| 95% | 60.5% | 75.2% | 75.2% |

Two consequences:

1. The model under-passes at high target selectivity (per-tuple reaches 61%
   at a 95% target), so "selectivity" in any plan must be the **achieved**
   one, not the label share.
2. The block filter passes 25-40% more rows than the per-tuple filter on the
   same data, and its F1 is higher, so per-tuple is the less accurate
   operator here (it misses positives). In a pipeline, the extra rows feed
   the join: pipe:4,1 issues more join calls than per-tuple (e.g. 1,177 vs
   996 requests at 50%) and is slower even though its filter stage is
   faster. **Block size changes downstream cardinality**, and that, not
   per-call latency, decides whether blocking an upstream stage pays off.

Per-stage out/in from the traces (rpipe, stage 2 = join) is 0.48-0.50 for
per-tuple and 0.60-0.78 for block:4 joins: the block join also passes more
pairs than the per-tuple join.

**H2 verdict:** the measured-best pipeline configuration is pipe:1,4 at
20-95% selectivity, but at 5% no block configuration beats per-tuple
(0.97x), so the best choice *does* change with selectivity (reject criterion
not met → H2 weakly supported). The stronger effect is that k changes
cardinality, which the cost model represents only through per-(op, k)
recall/FPR fitted on labels (held-out join q-error up to 15x, see
COST_MODEL_VALIDATION.md).

## 3. Block-size sensitivity (E2)

- **Filter (1000 rows):** latency is U-shaped in k with the minimum at k=8
  (4.59 s vs 10.19 s per-tuple). F1 is flat at 0.85-0.88 for k = 2-64. One
  partial block at the end for k ∈ {16, 32, 64} (1000 is not a multiple);
  no fallbacks; 2 truncated per-tuple calls.
- **Join (32 × 32):** best latency at k = 2-4; beyond that quality
  collapses (F1 0.83 → 0.40) because the block join passes far fewer pairs
  (achieved 0.56 → 0.17 vs 0.50 true). Peak KV grows with k (0.8% → 17.5%).
- **Map (200 rows):** k=2 is neutral, larger k is slower; one block
  fallback at k=8.
- Plan time (Kalypso's own planner, per query) is 27-88 ms, under 1% of
  query latency in every experiment.

## 4. Resources (H3)

- Peak KV use: 0.8-1% per-tuple, 3-17.5% block (larger prompts held longer);
  22% max under background load. **0 preemptions in all 10 experiments.**
- Kalypso KV admission: `kv_admit_rejected` fired only in E1-s95 (200
  rejections, all at the join stage); no forced admissions anywhere. So
  admission control is reached only at the highest-cardinality pipeline.
- The pressure is the request queue: vLLM's waiting queue peaks at 346-381
  for 400-row per-tuple filters and 1,467-1,896 for per-tuple pipelines at
  80-95%; blocking cuts it 4-8x (e.g. 1,596 → 390 at 95%, pipe:1,4).

**H3 verdict:** at this scale (≤ 1,000 rows, 7B, one L40S) KV memory is not
the limiting resource; queueing is. Under 32 and 128 background streams the
best plan does not change (E3 rerun, §5), so H3 is **rejected at this scale**;
load does shrink the best plan's advantage (1.35x → 1.13x at 128 streams).

## 5. Load sensitivity (E3, rerun with confirmed cache resets)

**First run (E3-load, invalid).** vLLM refuses to reset the prefix cache
while background requests hold KV blocks ("Failed to reset prefix cache
because some blocks (961) are not freed yet") but still answers HTTP 200,
so `eval.run` did not notice. 30 of 48 resets failed (all L32/L128 trials);
the block arms then ran on a warm cache (prefix hit rate 0.94 vs 0.20) and
looked 3.7x faster under load. Those numbers are kept in
`gpu/research-session1/E3-load/` but must not be used.

**Rerun (E3-load-v2, commit `18da93f`, 38.5 L40S minutes).** Per trial: no
background load running, reset the prefix cache and confirm it in the
server log, start the background load and let it settle for 20 s, run one
arm once (eval.run's own reset disabled), stop the load. **45 of 45 resets
confirmed.** rpipe_s50 (filter → join, 100 × 32), 5 repeats per arm and
level:

| arm | 0 streams | 32 streams | 128 streams |
|---|---|---|---|
| per-tuple | 10.03 s [9.96, 10.63] | 10.93 s [10.58, 11.24] (+9%) | 11.65 s [11.00, 12.39] (+16%) |
| pipe:1,4 | 7.53 s [7.32, 7.99] · 1.35x ✓ | 7.77 s [7.52, 7.81] · 1.43x ✓ | 10.14 s [10.07, 10.92] · 1.13x ✓ |
| pipe:4,4 | 9.73 s [9.68, 10.09] · 1.03x | 11.16 s [10.55, 11.77] · 1.00x | 12.11 s [11.87, 12.54] · 0.97x ✗ |

(✓/✗ = faster/slower than per-tuple in all 5 paired repeats.) Peak KV
17.7%, 0 preemptions, no Kalypso KV admission rejections at any level.

- The ordering of plans is the same at every load level: pipe:1,4 is best,
  per-tuple and pipe:4,4 are close. Load does not change the best block
  configuration here.
- Load hurts the block-join plan most at 128 streams (+35% vs +16% for
  per-tuple), so the block plan's advantage shrinks from 1.35x to 1.13x.
  Larger blocks hold more KV per request and wait longer for a batch slot
  when the server is shared; a resource-aware model would need this.
- The rerun's L0 matches E1-s50 within 2% (per-tuple 10.03 vs 10.20 s;
  pipe:1,4 7.53 vs 7.45 s).

## 6. Telemetry overhead (H5)

| arm | trace off | trace on | change |
|---|---|---|---|
| per-tuple (400 calls) | 3.77 [3.68, 3.82] s | 3.96 [3.91, 4.04] s | **+5.0%** (ranges do not overlap) |
| block:8 (50 calls) | 2.14 [2.09, 2.16] s | 2.14 [2.14, 2.15] s | 0% |

Off and on ran on two consecutive server instances in one container, so
part of the difference may be instance-to-instance drift (E3-L0 vs E1-s50
suggests ~5% between instances). The CPU bound (~21 µs per event, ~10
events per LLM call) predicts ~2% for 400 calls. **H5 is rejected for
per-tuple workloads at the 3% threshold** (or at least not shown to hold);
compare traced arms only with traced arms. Results with tracing on are
identical (same rows out, same F1, same request counts).

## 7. Scheduler behaviour (CPU simulation, real scheduler)

- Call counts follow the block sizes exactly at every selectivity (e.g. 1x1
  → 1x4 cuts calls 3.4x at 50%, 3.65x at 95%); all 125 runs correct.
- No partial join blocks except where the row count is not a multiple of k;
  join block *sizes* were identical across repeats, so variation is in
  membership only (H4).
- Under a fixed per-call cost, blocking the join gives 2.5-3.3x and blocking
  the filter alone gives ~nothing; the GPU shows the same ordering.

## 8. Optimizer scaling (CPU)

See RESEARCH_HYPOTHESES.md H1 (supported): 4-stage exhaustive search takes
8.6-19.9 s and 297-795 MB; 5-6 stages were not run (extrapolated 3 min to
5 h). Kalypso's own runtime planner, by contrast, takes 27-88 ms per query
(GPU traces) because it does not search block sizes.

## 9. What is not measured

- Load above 128 streams, or load from other Kalypso queries rather than synthetic traffic.
- Cost-model predictions for the new GPU arms (the model was validated on
  historical data only; scoring E1/E2 against it is the first Session 2
  check).
- Any GPU other than L40S, any model other than Qwen2.5-7B, more than 1,000
  rows, or more than 2 semantic stages on the GPU.
- Quality effect of run-to-run block membership (H4) on the GPU.
