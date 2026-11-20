# Session 1 report: research baseline and experimental foundation

Branch `research/session1-baseline` (cut from `main` at `21c9ee8`). Session 1
only; Phase 2 has not started. Every number below links to a committed
artifact; anything not measured says so.

## A. Actual baseline findings

1. **Block operators work, but only for filters and joins at small k.**
   Filter blocks are 1.7-2.2x faster than per-tuple with higher F1 (0.87 vs
   0.75); latency is U-shaped in k (best k=8 on 1000 rows; k=64 is slower
   than per-tuple). Join blocks give ≤ 1.53x (k=4) and lose quality fast
   beyond k=4 (F1 0.83 → 0.40 at k=32). Map blocks never help (k=4: 0.71x;
   k=8: 0.47x). [BASELINE_RESULTS.md §1, §3]
2. **Block size changes cardinality.** Block filters pass 25-40% more rows
   than per-tuple on the same data (e.g. 41.5% vs 31.0% at a 50% target),
   and block joins pass more pairs. In filter → join this makes blocking the
   filter *slower* end to end (0.63-0.84x at every selectivity), while
   blocking the join is 0.97x at 5% selectivity rising to 1.52x at 95%.
   [§2]
3. **The model's achieved selectivity is far from the label share at high
   targets** (per-tuple passes 61% at a 95% target). Plans must use achieved
   cardinality. [§2]
4. **KV memory is not the bottleneck at this scale; the request queue is.**
   Peak KV ≤ 17.5% (22% under load), 0 preemptions in all runs, Kalypso KV
   admission rejected only in the 95% pipeline; vLLM waiting queues reach
   1,896 and per-tuple join-stage wait p95 grows from 0.04 s to 3.2 s with
   selectivity. [§4]
5. **The offline cost model is good for single operators and weak for
   pipelines.** Single-operator MAPE 4.3% in-fit / 6.6% held-out (held-out
   almost always under-predicted); held-out pipelines 20.3% MAPE, top-1 0%,
   regret up to 24%; supplying true cardinalities does not reduce the
   held-out pipeline error (20.3% → 20.2%), so the composition term is the
   problem; held-out join q-error median 1.83, max 15.2. The model has no
   resource term. [COST_MODEL_VALIDATION.md]
6. **Exhaustive block-size search scales exponentially.** 4 stages: 8.6 s /
   297 MB (filters), 19.9 s / 795 MB (filter-join); 5-6 stages not run
   (extrapolated 3 min to 5 h, 7-700 GB). Powers-of-two candidates match
   the exhaustive pick within ±2% and plan 6 stages in under 2 s; the
   {1..4} set loses 39-506% in predicted latency. [optimizer_scaling.md]
7. **Reproducibility hazards found:** downstream block membership varies
   run to run (asyncio completion order); the planner's KV estimate uses an
   unseeded `random.sample`; most historical results were recorded from
   dirty trees; the model revision is not pinned; the Modal overlay gives
   Kalypso 2x the KV budget the fork's server does. [BASELINE_AUDIT.md §4,
   ENVIRONMENT.md]
8. **Background load slows every plan but does not change the best one**
   (E3 rerun, 0/32/128 streams: pipe:1,4 best throughout; its speedup
   1.35x → 1.43x → 1.13x). The first E3 run was invalid because vLLM
   silently refuses prefix-cache resets under load (HTTP 200 plus a log
   warning); the rerun confirms every reset in the server log (45/45).
   [BASELINE_RESULTS.md §5]

## B. The three proposed criticisms

| Criticism | Verdict | Evidence |
|---|---|---|
| RQ1: exhaustive optimization is costly as depth and candidates grow | **Supported** for the offline optimizer. Not a runtime cost today: the search runs once per experiment, and Kalypso's runtime planner takes 27-88 ms per query because it takes k from the query. | optimizer_scaling.md; GPU traces `plan_s` |
| RQ2: static configurations cannot react to cardinality changes | **Partly supported.** Block *membership* already adapts at run time (rows are packed as they arrive; no partial blocks except the final one). What is static is k and the selectivity estimate, and both matter: the best pipeline configuration changes between 5% and ≥ 20% selectivity, and k itself shifts downstream cardinality by 25-40%. | BASELINE_RESULTS.md §2; audit §2.8 |
| RQ3: resource metrics (KV, queueing, preemption) are ignored | **Supported for the optimizer** (no resource term). **KV pressure itself was not observed** at ≤ 1,000 rows on one L40S (0 preemptions); queueing was. The executor already admits stages by reserved KV. Under 32/128 background streams the best plan does not change (E3 rerun), though its advantage shrinks at 128. | COST_MODEL_VALIDATION.md §2; BASELINE_RESULTS.md §4-5 |

## C. Files changed (vs `main`)

Original system, opt-in telemetry hooks only (no behaviour change; guarded by
`telemetry.enabled()`): `vllm/kalypso/budget.py`,
`vllm/kalypso/controller/plan.py`, `vllm/kalypso/controller/stage.py`,
`vllm/kalypso/execution/pipeline_execution.py`,
`vllm/kalypso/execution/vllm_executor.py`,
`vllm/kalypso/sem_ops/block_filter.py`, `vllm/kalypso/sem_ops/block_map.py`.
New: `vllm/kalypso/telemetry.py`, `vllm/kalypso/tests/test_telemetry.py`.

Research code (new): `research/baseline/instrumentation/optimizer_profile.py`,
`research/baseline/experiments/{optimizer_scaling,executor_simulation,telemetry_overhead_cpu}.py`,
`research/baseline/analysis/{costmodel_validation,trace_summary,gpu_results,optimizer_scaling_summary}.py`,
`research/baseline/gpu/{modal_baseline,run_eval}.py`, `research/baseline/workloads.py`.

Docs (new): `research/PROGRESS.md`, `research/EXPERIMENT_MANIFEST.json`,
`research/docs/{BASELINE_AUDIT,ENVIRONMENT,COST_MODEL_VALIDATION,BASELINE_METHODOLOGY,RESEARCH_HYPOTHESES,BASELINE_RESULTS,SESSION1_REPORT}.md`.

Results (new): `research/baseline/results/` (see F). `.gitignore` gains `._*`.
Historical results under `vllm/kalypso/benchmark/eval/results/` are untouched.

## D. Commands executed

```bash
python -m pytest -q vllm/kalypso/tests
python -m research.baseline.experiments.optimizer_scaling --out research/baseline/results/optimizer_scaling.jsonl
python -m research.baseline.analysis.optimizer_scaling_summary research/baseline/results/optimizer_scaling.jsonl > research/baseline/results/optimizer_scaling.md
python -m research.baseline.experiments.executor_simulation --out research/baseline/results/executor_simulation
python -m research.baseline.experiments.telemetry_overhead_cpu --out research/baseline/results/telemetry_overhead_cpu.json
python -m research.baseline.analysis.costmodel_validation --out research/baseline/results/costmodel_validation
modal run --detach research/baseline/gpu/modal_baseline.py::launch     # from Xen's Mac, commit 98e03cb
modal run research/baseline/gpu/modal_baseline.py::status
modal run --detach research/baseline/gpu/modal_baseline.py::launch --only E3-load-v2   # rerun, commit 18da93f
modal volume get blockkalypso-results research-session1 ./research/baseline/results/gpu/
python -m research.baseline.analysis.gpu_results research/baseline/results/gpu/research-session1 --out research/baseline/results/gpu_analysis
```

## E. Tests

- `vllm/kalypso/tests`: **141 passed, 0 failed** (includes 13 new telemetry
  tests: off by default; identical results, call counts and item multisets
  with tracing on/off on 4 pipelines; trace consistency; stage/task
  attribution; KV counters under forced admission; LLM-call records; file
  sink).
- `docker/kalypso/test_*.py`: **not run here**; they need an installed
  upstream vLLM 0.29.0 tree (collection fails with a path error in this
  container). Unchanged on this branch.
- First Modal launch (`fbf9dbc`) failed at import (repo-root lookup inside
  the container); fixed in `98e03cb` before any GPU work ran.

## F. Benchmark results available

| Result | Path |
|---|---|
| GPU E0-E3 raw (run records, traces, vLLM metrics, nvidia-smi, server logs, env), incl. the E3 rerun `E3-load-v2/` | `research/baseline/results/gpu/research-session1/` |
| GPU analysis | `research/baseline/results/gpu_analysis/summary.md`, `summary.json` |
| Optimizer scaling | `research/baseline/results/optimizer_scaling.jsonl`, `.md` |
| Executor simulation | `research/baseline/results/executor_simulation/` |
| CPU telemetry overhead | `research/baseline/results/telemetry_overhead_cpu.json` |
| Cost-model validation | `research/baseline/results/costmodel_validation/` |

GPU budget used: 120.8 L40S container-minutes (10 containers, 82.3 min, plus the E3 rerun, 38.5 min).

## G. Remaining limitations

- Load was tested only up to 128 synthetic background streams on one
  pipeline (E3 rerun); the first E3 run is invalid and kept only for the
  record.
- Telemetry adds ~5% on 400-call per-tuple runs (E0); traced vs untraced
  comparisons are not valid at that precision.
- One GPU type, one model, ≤ 1,000 rows, ≤ 2 semantic stages on GPU; KV
  pressure never reached; 5 repeats (CIs are tight but the between-server
  drift is ~5%).
- The cost model was validated on historical runs only; it has not yet been
  scored on the new E1/E2 arms.
- Block-membership nondeterminism is documented but its GPU quality effect
  is not isolated.
- Hazards listed in ENVIRONMENT.md §7 remain unfixed on purpose (Session 1
  does not change semantics).

## H. Recommendations for Session 2

Ordered by evidence strength.

1. **Make cardinality an observed, per-k quantity.** The largest plan
   errors come from k changing selectivity and from held-out join
   cardinality (q-error up to 15x). Session 2's first mechanism should
   measure achieved pass rate per (stage, k) online from the telemetry
   (`task_end.rows_out / rows_in` is already recorded) and feed it to the
   planner, instead of label-fitted recall/FPR.
2. **Replace the pipeline composition term before adding resource terms.**
   Held-out pipeline error is unchanged with true cardinalities, so λ·ΣL_i
   is wrong for overlapping stages. Model the critical path / queueing
   (per-stage wait times are in the traces) and validate on the E1 rpipe
   arms first.
3. **Model queueing, not KV, as the resource signal** at this scale: vLLM
   waiting count and stage wait p95 vary 10-80x across configurations while
   KV stays under 20%. Add a KV term only after an experiment that actually
   reaches KV pressure (more rows, longer contexts, a larger model).
4. **Restrict the candidate space to powers of two (≤ 32)** for any search
   that must run per query: same pick within ±2% at 1-4 stages, under 2 s at
   6 stages. Do not use {1..4}.
5. **Cap join and map blocks.** Join k > 4 and map k > 2 lose quality or
   speed in every measurement; a quality constraint (F1 drop vs per-tuple)
   must be part of the feasible set F_t.
6. **Fix measurement before the adaptive experiments:** verify every
   prefix-cache reset (as E3-load-v2 does; vLLM returns 200 on failure), seed the planner's sampling, pin the
   model revision, choose one KV budget factor, and report block-membership
   variance.
7. **Replanning trigger (for the proposed adaptive mechanism):** the data
   suggest the decision that matters is *which stage to block*, not fine k
   tuning (pipe:1,4 vs 4,1 differs by 2x; k=4 vs 8 by ~10%). A first
   adaptive policy can therefore switch a stage between per-tuple and a
   small block size when its observed pass rate crosses a threshold
   (here between 5% and 20%), with hysteresis.

## Commits on this branch

`41a3c20` branch + log · `ab03d8d` audit · `3d922dd` telemetry + tests ·
`a54c663` optimizer profiler, scaling, cost-model validation, GPU runner ·
`fbf9dbc` detached launcher · `98e03cb` Modal import fix, executor simulation,
trace summariser, full scaling results (GPU runs launched from here) ·
`71d72a1` ENVIRONMENT.md · `65d8f98` scaling table · `55548fc`
COST_MODEL_VALIDATION.md · `f598b7a` CPU overhead · `2044028` methodology +
hypotheses · `5b75c81` GPU analysis script · `9959240` manifest · `dd6208a`
progress · `9e9daf2` GPU results · `813ef24` BASELINE_RESULTS.md · `754c066` report · `18da93f` E3 rerun protocol (rerun launched from here) · (E3 rerun results: next commit).
