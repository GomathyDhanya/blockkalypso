# Baseline audit: BlockKalypso at `21c9ee8`

Session 1, Step 1. Every statement below was checked against the code at
`main` = `21c9ee862` ("updated Milestone 3 Report", 2026-11-04). Line numbers
refer to that commit. Each row is tagged:

- **[code]** implemented and executed on the request path,
- **[offline]** implemented, but only in the offline analysis/benchmark code
  (`vllm/kalypso/benchmark/eval/`), never called while a query runs,
- **[doc]** described only in a design document, comment or proposal.

## 0. Summary of the most important facts

| # | Finding | Evidence |
|---|---|---|
| A | **The block-size optimizer is offline only.** `costmodel.choose_k` / `choose_pipeline` are called from the analysis scripts and tests, never from the planner, scheduler or API server. At query time the block size is whatever the client put in each op's `args` (`block_size`, `pair_block_size`). | [offline] `benchmark/eval/costmodel.py:218-296`; only callers: `costmodel.py` itself, `eval/plots.py:20`, `tests/test_eval_harness.py:516-538` (grep, no hit in `controller/`, `execution/`, `query_processor.py`, `api_server.py`). [code] `controller/plan.py:319-329, 340-349, 392-409` |
| B | **Search is exhaustive**: `itertools.product(range(1, k_max_i + 1) ...)` over all stages, every candidate fully materialised in a list, every prediction kept in a list. Cost per candidate is O(S^2) in the number of stages S because `stage_items` is recomputed inside the per-stage loop. | [offline] `costmodel.py:286-296`, `:262-263` |
| C | **Optimization runs once, before execution, with a selectivity the caller supplies** (the workload's *true* label share in every validation). No runtime feedback exists. | [offline] `costmodel.py:657-671, 923-956`; [doc] STUDY.md "Selectivity given to the model … live re-estimation is proposed, not built" |
| D | **Block formation at runtime is dynamic, count-plus-token bounded, and greedy**: downstream block stages buffer rows across producer tasks until `block_size` rows (or the `max_block_tokens` limit) are reached; the final partial block is flushed only after the upstream stage has fully drained. | [code] `execution/pipeline_execution.py:153-209`, `controller/stage.py:137-158` |
| E | **KV memory is managed per stage by Kalypso's own reservation ledger, not by the cost model.** The cost model only bounds k by the context window (`k_ctx`). It has no KV, queueing or preemption term. | [code] `budget.py:77-272`, `controller/plan.py:61-169`; [offline] `costmodel.py:202-216` |
| F | **Quality enters only as an optional, offline per-(kind, regime) cap on k** learned from past runs (mean ΔF1 ≥ −0.02, or ROUGE-1 ≥ 0.8 for maps). It is not predicate-specific, and is off in the default (latency-first) choice. | [offline] `costmodel.py:208-216, 454-472` |
| G | **Recall/FPR used to propagate cardinality are pooled per operator kind and row regime**, so they do not transfer across predicates (validate-jf: per-tuple join recall assumed 0.69, measured ~0.05). | [offline] `costmodel.py:136-141, 502-519`; STUDY.md "Join first, then a filter" |
| H | Planning-time capacity sizing uses **an unseeded `random.sample`** of up to 100 rows, so a stage's minimum KV capacity can differ between runs of the same query. | [code] `controller/plan.py:99-105` |
| I | Executor's built-in `stage_stat["input"]` counts **tasks**, while `["output"]` counts **rows**; for block stages the two are not comparable. | [code] `execution/pipeline_execution.py:266-275, 401, 458` |
| J | The server-side telemetry (Prometheus deltas, 0.25 s gauge samples) is **per trial and server-wide**: no per-stage, per-task or per-call record exists. Token counts per call returned by vLLM (`usage`) are discarded by `VLLMExecutor`. | [offline] `benchmark/eval/metrics.py:24-243`; [code] `execution/vllm_executor.py:115-136` |

## 1. Execution lifecycle (input query → final result)

| Step | What happens | Where |
|---|---|---|
| 1. Request | `POST /v1/semantic/query` with `ops` (list of `{op, args}`) and `data_path` or inline `records` | [code] `vllm/entrypoints/openai/api_server.py:682-742`; on the Modal image (vLLM 0.29.0) the same routes are copied into `vllm/kalypso/container_api.py` by `docker/kalypso/prepare.py:73-79` |
| 2. Processor | One `QueryProcessor` per server; KV budget = `get_kv_cache_budget()` × TP (container) or × TP × 0.5 (fork's api_server) | [code] `docker/kalypso/init_api.py:17-22`; `api_server.py:1564-1580`; `query_processor.py:17-25, 47-53` |
| 3. Load rows | `data._data_source` turns each CSV row into a `SemContext` | [code] `controller/plan.py:574` |
| 4. Logical → physical ops | `parse_ops`: `sem_filter` → `BlockSemFilter` if `block_size > 1` else `SemFilter`; `sem_map` → `BlockSemMap` / `SemMap`; `join` with `pair_block_size` → `CartesianProduct` + `BlockSemJoin`, else `CartesianProduct` + `SemFilter` (or ICP/cascade variants) | [code] `controller/plan.py:236-560` |
| 5. Plan build (stages) | Tuple-independent ops fuse into one `Stage`; a block op always gets its own stage; a `CartesianProduct` becomes the previous stage's `fanout_op` (or a standalone plan item if first); BLOCKING ops (agg, top-k) split segments | [code] `controller/plan.py:21-209` |
| 6. Physical block config | `Stage.block_size` = the first block op's `block_size`; `block_tasks` true if > 1 | [code] `controller/stage.py:75-84` |
| 7. KV registration | Per segment: sample ≤100 contexts, estimate each stage's largest task budget (bytes), register stage capacities (single stage: 100%; multi-stage: earlier stages at their sampled minimum, last stage gets the rest; max fraction = initial + 0.5) | [code] `controller/plan.py:48-172`, `budget.py:77-94` |
| 8. Initial tasks | First stage: one `Task` per block (`make_blocks`) or per row | [code] `pipeline_execution.py:436-444` |
| 9. Scheduling loop | Repeat: flush partial blocks whose producer drained → rebalance capacities → launch every head-of-queue task whose budget fits (`try_allocate_stage`) → if nothing runs but work is queued, force-launch one per stage → wait for first completion | [code] `pipeline_execution.py:499-518`, `:388-431` |
| 10. vLLM request | `Stage.run_task` → operator → `VLLMExecutor.execute` → in-process `create_chat_completion` (no HTTP hop) with priority `-(stage offset + op index)` | [code] `controller/stage.py:247-303`, `sem_ops/block_utils.py:90-102`, `execution/vllm_executor.py:65-136` |
| 11. Intermediate tuples | Block result → surviving rows (and fanout children) forwarded; downstream block stage accumulates into `pending_block_rows`; per-tuple result → new `Task` per row | [code] `pipeline_execution.py:446-497, 576-669` |
| 12. Retries | `RETRY_TASK` (SemMap cut off by length) re-enqueued at the same op index with higher priority, budget released or retained | [code] `pipeline_execution.py:532-561`, `sem_ops/map.py:97-130` |
| 13. Result | Last stage's rows collected in `out`; the API serialises `ctx.output` per row | [code] `pipeline_execution.py:465-466, 657-659`; `api_server.py:708-742` |

## 2. The twelve questions

### 2.1 Where block-size optimization occurs

There are two different things called "block size":

- **Runtime block size** [code]: taken verbatim from the query
  (`plan.py:319-329` filter, `:340-349` map, `:392-409` join). Constructor
  defaults (`BlockSemFilter` 8, `block_filter.py:131`; `BlockSemMap` 8,
  `block_map.py:116`; `BlockSemJoin` 25, `block_join.py:60`) apply only when the
  op is built directly, because `parse_ops` passes the request's value. No
  component chooses it.
- **Recommended block size** [offline]: `Model.choose_k` (single operator,
  `costmodel.py:218-227`) and `Model.choose_pipeline` (`:286-296`). Used to
  write `results/analysis/algorithm_choices.json`, which `cloud/experiments.py:255-266`
  turns into benchmark arms. The pick reaches Kalypso only as a benchmark arm.

### 2.2 Cost-model equations and parameters

Per operator (`costmodel.py:181-200`):

    L = g[kind] + α·P + (β + tail[kind])·W·D + γ·C·D + δ·C·p²/1000

with C calls, W = ⌈C / max_num_seqs⌉ waves, D decode steps per call, P computed
(non-cached) prompt tokens, p prompt tokens per call. Token terms per kind
(`KindParams`, `:105-141`): per-tuple `a_pt, q_pt, d_pt`; block `e, o, q, d0,
d1, c_left`. Fitted by weighted least squares on relative error (`fit`,
`:533-544`; `fit_kind`, `:475-520`). Fitted values (v3) are in
`results/analysis/costmodel.json` and STUDY.md ("What the cost model computes").

Pipelines (`predict_pipeline`, `:255-284`): `L = λ · Σ_i L_i` (λ fitted, 0.98,
`fit_lambda` `:707-712`); a filter's output rows = rows · (recall(k)·s +
fpr(k)·(1−s)) (`pass_share`, `:136-141`); joins likewise on pairs
(`:278-281`); map reruns added as `retry_share` (`:250-253, 299-307`).

Feasible k per stage (`k_max`, `:208-216`): min(context-window limit `k_ctx`
(`:202-206`), quality cap or largest measured k for the regime, number of
items). `MAX_K` = 32 when nothing was measured (`:89`).

### 2.3 How candidates are generated and evaluated

`choose_pipeline` (`:286-296`): compute `k_max` per stage; if no candidate
list is given, `itertools.product(range(1, k+1) for k in k_max)`; materialise
all candidates (`:292-293`); call `predict_pipeline` on each (`:294`); `_pick`
takes the lowest latency and, among those within `tie` = 2% of it, the
smallest Σk (`:310-317`); the absolute fastest is reported too (`:296`).

### 2.4 Is search exhaustive? Complexity

Yes. Candidates = Π_i k_max_i. Each `predict_pipeline` loops over S stages and
calls `stage_items(pipe, rows)` (itself O(S)) **inside** the loop (`:263`), so
one evaluation is O(S²) and the whole search is O(S² · Π k_max_i) time and
O(Π k_max_i) memory (two Python lists of that length: candidates and prediction
dicts). With k_max = 20 (short-row filter cap) the space is 20^S:
8,000 at S = 3, 3.2 M at S = 5, 64 M at S = 6. Measured cost: see
`BASELINE_RESULTS.md` §Optimizer scaling.

### 2.5 Once per query or repeatedly?

Once, offline, per experiment (`validation_choices`, `:973-1015`; choices for
experiments already run are frozen, `:1185-1190`). Nothing re-plans during
execution [code: no call site]. The proposal's "Level 5 online hill climbing"
controller is **[doc] only** (`systems_for_ml_proposal.pdf`, lines ~165-290 of
its text).

### 2.6 Independent or coupled block sizes?

Coupled **in the cost model**: a filter's k changes recall/FPR and so the
next stage's input (`:274-281`), and map retries depend on whether any other
stage is blocked (`retry_key`, `:299-307`). Independent **at runtime**: each
stage's `block_size` comes from its own op args (`stage.py:80-84`); the
scheduler does not change it.

### 2.7 How filters and joins change cardinality

[code] Filters drop rows whose verdict is not `True` (block,
`block_filter.py:270-284`; undecided rows are dropped) or contains `"false"`
(per-tuple, `sem_ops/base.py:45-53`). Joins: `CartesianProduct` emits one
context per right row (`sem_ops/joins.py:26-53`), then the join predicate
filters pairs. [offline] The cost model predicts these counts from
recall/FPR and a given true selectivity (§2.2); there is no estimator of
selectivity itself.

### 2.8 Downstream block formation: static or dynamic?

Dynamic in membership, static in size. `enqueue_block_rows`
(`pipeline_execution.py:161-190`) appends arriving rows to a pending buffer and
emits a task when it reaches `block_size` rows or adding a row would exceed
`max_block_tokens` (via `make_blocks`, `stage.py:137-158`). The leftover
partial block is emitted only when the upstream stage has no waiting and no
running tasks (`flush_partial_block_tasks`, `:192-209`). There is no timeout:
a partial block waits for the whole upstream stage to drain
(STUDY.md "Holding rows vs sending early": untested).

### 2.9 Backpressure and pending blocks

[code] Admission control is the only backpressure: a task launches only if
`stage_used + budget ≤ stage_capacity` (`stage.py:208-220`,
`budget.py:212-222`); otherwise it waits at the head of its stage queue
(head-of-line: `launch_ready_tasks` breaks at the first non-admissible task,
`pipeline_execution.py:391-399`). If nothing is running and queues are
non-empty, one task per stage is force-admitted ignoring capacity
(`:412-431, 504-512`). Pending block rows are not budgeted until they become a
task. Stage queue thresholds (`tune_thresholds`, `stage.py:160-168`) drive
capacity borrowing (`rebalance_stage_capacity`, `pipeline_execution.py:282-338`;
`budget.py:117-203`) but never block size.

### 2.10 Memory reservations, scheduler and block formation

[code] A block task reserves `max over the stage's ops of estimate_tokens(block)
× bytes_per_token` (`stage.py:181-205`). For `BlockSemFilter` that is the block
prompt + `16 + 6·k` output tokens, or the largest single-row fallback if larger
(`block_filter.py:173-199`). The reservation ledger is Kalypso-side bookkeeping:
it is not checked against vLLM's real KV usage, and other requests on the
server are not visible to it (`api_server.py:1579` comment). `bytes_per_token`
comes from the HF config (`budget.py:9-52`). Block formation does not look at
the ledger, and the ledger does not change k.

### 2.11 Accuracy metrics and quality constraints

[offline] `eval/report.py` and `eval/analyze.py` compute accuracy, precision,
recall, F1 against labels (filters: rows; joins: candidate pairs), agreement
with per-tuple, self-agreement across repeats, and ROUGE-1 similarity for
maps (`analyze.py:75-97`, `MIN_SIMILARITY` = 0.8, `:45`). The only quality
*constraint* is the optional cap (§F). [code] Operators record parse outcome
counters: `BlockFilterStats` (`block_filter.py:94-112`: block/fallback calls,
rows answered by block vs fallback, undecided rows, truncated block calls) and
`BlockMapStats` (`block_map.py:84`). These counters are per operator
instance, never exported by the API.

### 2.12 Telemetry already collected

| Source | What | Granularity |
|---|---|---|
| `eval/metrics.py:24-243` [offline] | Prometheus counter deltas (requests by finish reason, prompt/generation tokens, prefix-cache hits/queries, prefill/decode time histograms' sum/count, preemptions, max_tokens histogram); gauges every 0.25 s (KV %, running, waiting); server RSS every 0.1 s | per trial, server-wide |
| `eval/run.py:113-152` [offline] | latency, rows in/out, rows/s, selected rows, every output, fallback calls | per trial |
| `pipeline_execution.py:234-260` [code] | `[scheduler]` stdout lines (used/cap/waiting/running per stage), throttled 1 s | stdout only |
| `pipeline_execution.py:266-275` [code] | `stage_stat` input (tasks) / output (rows) | per query, returned but discarded by `SemanticPlan.execute` (`plan.py:582`) |
| Operator stats [code] | `BlockFilterStats`, `BlockMapStats` | in memory |
| Not collected | per-stage wait/exec times, per-task rows, per-call tokens and latency, planning latency, predicted vs actual cardinality per stage, GPU utilisation, KV reservation failures, timestamps aligned across sources | — |

## 3. Proposed criticisms, as stated in the team's plan, against the code

| Criticism | Verdict from the code | What still needs measurement |
|---|---|---|
| RQ1: exhaustive optimization is costly as depth and candidates grow | **Supported structurally** (§2.4): O(S²·Π k_max) time and O(Π k_max) memory. But it currently runs offline once per experiment, on 1-3 stage pipelines, so its cost is not on any query's critical path today. | Wall time and peak memory vs depth (Step 4A, CPU). |
| RQ2: static configurations cannot react to filters/joins changing cardinalities | **Partly supported.** Block *membership* already adapts (rows are packed as they arrive, fill was 100% in every checked run, STUDY.md Q2). What is static is the *size* k and the *predicted* selectivity, which comes from labels. | How far predicted cardinalities are from observed ones (Step 5), and whether the k choice is sensitive to that error. |
| RQ3: resource metrics (KV, queueing, preemption) are ignored | **Supported for the optimizer** (no resource term in `L`). **Not for the executor**: Kalypso's stage ledger already admits by reserved KV bytes. Whether resource pressure matters for block queries on the measured setup is open: published peak KV use was low on short rows. | KV peak, waiting, preemptions under controlled load (Step 4D, GPU). |

## 4. Reproducibility hazards found in the code

1. Unseeded `random.sample` in planning (`plan.py:99-105`) — changes stage
   min capacities, not answers.
2. `VLLMExecutor.DEFAULT_SAMPLING` has `seed: 42`, temperature 0
   (`vllm_executor.py:32-38`); block ops override penalties
   (`block_utils.py:74-79`). Greedy decoding on a GPU is still not
   bit-deterministic across batch compositions, so outputs can change with
   concurrency (observed: block arms varied ~1 s in validate3).
3. Two server code paths: the fork's `api_server.py` (vLLM 0.13.0rc4, KV
   budget × 0.5) and the Modal overlay (`docker/kalypso/prepare.py`, vLLM
   0.29.0, budget × 1.0, `init_api.py:17-22`). All published GPU numbers use the
   overlay.
4. `stage_stat` input/output units differ (§I).
5. Map answer budgets depend on the order rows finish
   (`MapRatioEstimator`, `controller/map_estimator.py:17-27`, `map.py:52-65`),
   so map retries depend on scheduling (STUDY.md: 12–56% reruns).
6. **Downstream block membership is not reproducible run to run.** When a
   per-tuple (or block) stage feeds a block stage, rows are packed in the
   order their upstream tasks complete. The scheduler takes completions from
   `asyncio.wait(...)`'s `done` *set* (`pipeline_execution.py:515-520`), whose
   iteration order is not stable. With a deterministic fake LLM, the same
   query produced a different grouping of rows into downstream blocks in 9-10
   of 10 reruns (filter → block join, per-tuple filter → block filter), and
   0 of 10 when the upstream stage was itself blocked in fixed order
   (`vllm/kalypso/tests/test_telemetry.py` documents this). With a real LLM a
   row's verdict can depend on its block-mates, so block-stage answers are not
   guaranteed identical across repeats even at temperature 0.
