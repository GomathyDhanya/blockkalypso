# Baseline methodology (Session 1, Step 6)

The protocol under which every Session 1 number was produced, fixed before
the GPU runs were launched (commit `98e03cb`). Hypotheses and their
rejection criteria are in RESEARCH_HYPOTHESES.md; environments in
ENVIRONMENT.md.

## 1. System under test

The original BlockKalypso at `main` `21c9ee8`, unchanged except for the
opt-in telemetry added on this branch (`vllm/kalypso/telemetry.py` and its
hooks, off unless `KALYPSO_TRACE` is set; tested to give identical outputs
and LLM calls with it on or off). Baselines inside each experiment:

- `kalypso_per_tuple`: the original one-row-per-call operators (k = 1);
- `block:k` / `pipe:k1,k2`: block operators at fixed sizes;
- the cost model's own pick, evaluated offline against the measured arms.

## 2. Experiments

| id | where | question | independent variable | fixed |
|---|---|---|---|---|
| A | CPU | optimizer scaling (H1) | stages 1-6; family filters / filter-join / mixed; candidate set exhaustive / powers of two / 1-4 | cost model v3; 400 rows, 31 tokens, 32 right rows, selectivity 0.5 |
| Sim | CPU | scheduler behaviour vs. selectivity and k (H2, H4) | selectivity 5/20/50/80/95%; (k_filter, k_join) ∈ {1x1, 4x1, 1x4, 4x4, 8x8} | real planner/scheduler/KV ledger; simulated LLM 20 ms + 2 ms/item, 64 slots |
| E0 | GPU | telemetry overhead (H5) | tracing off / on | rsel50, per-tuple and block:8 |
| E1 | GPU | cardinality sensitivity (H2) | selectivity 5/20/50/80/95% | rsel* (400-row filter): per-tuple, block 4, 8; rpipe_s* (100x32 filter→join): per-tuple, pipe 4,1 / 1,4 / 4,4 |
| E2 | GPU | block-size sensitivity | k up to 64 (filter), 32 (join), 8 (map) | movie_positive n=1000, movie_fanout50, movie_map n=200 |
| E3 | GPU | load sensitivity (H3) | 0 / 32 / 128 background streams (unique ~400-token prompts, 128 output tokens, ignore_eos) | rpipe_s50: per-tuple, pipe 1,4 and 4,4 |
| Val | CPU on historical GPU data | cost-model accuracy | fit vs. held-out phases | committed results, 484 arms |

Selectivity is controlled through labels (share of truly positive rows,
seed 0); the achieved selectivity (rows the model actually passes) is
measured and reported separately from the target.

## 3. Controls

- Same server for every arm of one experiment; prefix cache reset before
  every trial; arms interleaved within each repeat (eval.run's order) so
  drift hits all arms equally.
- Greedy decoding (temperature 0, seed 42).
- One experiment per container and GPU; experiments do not share a GPU.
- E3: only the background stream count changes; load levels are interleaved
  per repeat, and the background client starts before and stops after the
  measured arms.
- Model weights: the resolved revision is recorded in `env.json` (not pinned
  in the runner; see ENVIRONMENT.md §7).

## 4. Repetitions, warm-up, statistics

- GPU: 1 warm-up (discarded) + 5 measured repeats per arm. CPU experiments:
  5 repeats (A, Sim), 15 interleaved pairs (CPU telemetry overhead).
- Report the median with min/max for every arm; 95% confidence intervals for
  the median by bootstrap (10,000 resamples over the 5 repeats) where a
  difference is claimed. With n = 5 these intervals are wide; differences
  inside the repeat range are not claimed.
- Comparisons between two arms of the same experiment: paired by repeat
  (they ran interleaved), Wilcoxon signed-rank on the per-repeat ratios;
  with n = 5 the smallest attainable two-sided p is 0.0625, so a
  "significant" result needs all 5 repeats in the same direction **and** a
  median effect larger than the run-to-run spread. Effect sizes are reported
  as median ratios, not p-values alone.
- Ranking: Spearman correlation, top-1 agreement and regret (as in
  COST_MODEL_VALIDATION.md).
- Cardinality: q-error with zero handling (both zero = 1; one zero =
  reported separately); under- and over-estimation reported separately.

## 5. Measurement and its overhead

- Executor telemetry: per query, stage, task and LLM call (wait and
  execution times, block fill, retries, tokens incl. cached, finish reason,
  KV admission counters, scheduler queue samples every 0.25 s).
- GPU: `nvidia-smi` every 0.5 s; vLLM `/metrics` every 0.5 s (KV use,
  running/waiting, preemptions), all wall-clock stamped.
- Optimizer: wall time per component, peak Python heap via tracemalloc
  (separate pass so the timed runs carry no tracemalloc cost), and a check
  that instrumentation does not change the pick.
- Overhead: CPU upper bound ~21-22 µs per event (H5); E0 measures it on
  the GPU. If E0 exceeds 3%, all traced GPU latencies are reported with that
  caveat and compared only with traced arms.

## 6. GPU budget

10 containers, one L40S each, launched once. Each container time is
recorded in its `status.json` and summed in BASELINE_RESULTS.md; nothing
beyond these 10 is launched in Session 1. A failed first launch (import
error before vLLM started) used no measurable GPU time.

## 7. Threats to validity

- **Internal:** downstream block membership varies run to run (H4); model
  revision not pinned; the planner's `random.sample` is unseeded; the KV
  budget factor is 1.0 on Modal vs. 0.5 in the fork's server.
- **Construct:** selectivity is a label share, not the model's achieved
  pass rate; "load" is synthetic background traffic, not other Kalypso
  queries.
- **External:** one GPU type (L40S), one model (Qwen2.5-7B), one dataset
  family (SemBench movie reviews; historical data adds ContractNLI etc.), at
  most 1000 rows. Conclusions about KV pressure in particular may not hold
  for larger inputs or models.
- **Statistical:** 5 repeats; historical data 3 repeats and mostly from
  dirty trees.
- **CPU simulation:** the LLM cost is a fixed linear model; simulated
  latency measures scheduler structure only.

## 8. Rejection criteria

Stated per hypothesis in RESEARCH_HYPOTHESES.md. In addition, a GPU
experiment is **invalid** (reported, not used for conclusions) if any
request failed with a server error, the server log shows OOM or a restart,
or fewer than 5 repeats completed for an arm.
