# Milestone 2: block-size experiments (E1-E11)

Every experiment in the Milestone 2 experiment plan, run on one model, one GPU
and one protocol. Raw results, per-run reports and diagnostics are in
`vllm/kalypso/benchmark/eval/results/m2-*-qwen14b*`; the harness is
`vllm/kalypso/benchmark/eval/` (see its README).

## Environment

| | |
|---|---|
| Compute | [Modal](https://modal.com), one NVIDIA L40S (48 GB) per run (`cloud/modal_eval.py`) |
| Model | `Qwen/Qwen2.5-14B-Instruct`, vLLM's default dtype for it (bf16) (the smallest Qwen model that answers the paper's arXiv join prompt: 7B answered "false" to every pair) |
| Serving | Kalypso on vLLM 0.29.0, the project container (`docker/Dockerfile.kalypso`, vLLM's official `vllm/vllm-openai:v0.29.0` image plus this branch's Kalypso package) |
| Server flags | `--enable-prefix-caching --no-async-scheduling --api-server-count 1 --max-model-len 16384` (32768 for E1, so long-row blocks can fit); `VLLM_SERVER_DEV_MODE=1` for prefix-cache resets |
| Protocol | Server and harness in the same container (no network in the timings); prefix cache reset before every trial; 1 warmup, then 3 repeats with arms interleaved; medians reported |
| Baseline | Per-tuple Kalypso (Kalypso's original operators, one LLM call per row or pair) |
| Metrics | The Kalypso paper's: query completion time, speedup over the baseline, LLM calls. F1 is added because blocking can change answers; peak KV-cache use is in each run's report |
| Code | Branch `m2-pipeline-instrumentation`; each results file records the exact commit, server command and GPU |

### Datasets

| Dataset | Source | Used for | Rows per run | Labels |
|---|---|---|---|---|
| arXiv 2023 abstracts | arXiv API (CC0); 1,513 papers: robotics (cs.RO) papers whose title names one task (aerial, driving, legged, grasping, SLAM) plus databases, security and programming-languages papers | E1 short rows, E2, E3, pipeline (E4-E11) | E1: 64; E2: 16 x 8; E3: 16-128; pipeline: 32 left x 32 right | Proxy: cs.RO (Filter1), same title task (join), AI/ML terms in either abstract (Filter2) |
| MEDEC-MS | abachaa/MEDEC (CC BY 4.0), 2,189 clinical notes | E1 medium rows | 64 (seeded sample) | Gold: note contains a medical error |
| ContractNLI | Stanford (CC BY 4.0), 423 train contracts | E1 long rows | first 64 contracts | Gold: every document is an NDA |

Prompts are the Kalypso paper's own (`vllm/kalypso/benchmark/scenarios.py`):
the arXiv pipeline is its FILTER -> JOIN -> FILTER case, with Filter1 "Is the
paper relevant to AI robotic research?", the join "Do these two papers study
similar topic?" and Filter2 "Is this abstract about AI or machine learning?".
Pipeline workloads are built from real papers to a target Filter1 selectivity
`s` and join fanout `f` (default s = 0.5, f = 1). Fetch everything with
`python -m vllm.kalypso.benchmark.eval.fetch_data`.

Block sizes are written `k` for a filter, `k x j` for a join (k left rows by
j right rows per call) and `F1 / join / F2` for the pipeline; `@ 4096` caps a
filter block at 4096 estimated tokens (Level 2).

## Fixes made along the way

Problems found while setting up E4-E11, and what changed. Each was found from
a measured run, not assumed.

| Problem | Evidence | Fix |
|---|---|---|
| **Model too small** | Qwen2.5-1.5B and 3B: blocked Filter1 passed 25-31 (1.5B) or 4-12 (3B) of 16 robotics papers while per-tuple passed exactly 16, so every blocked pipeline did different work. Qwen2.5-7B answered "false" to all 128 join pairs with the paper's prompt (and with three rewordings it answered all-no or all-yes) | Qwen2.5-14B on an L40S: the smallest model that discriminates with the paper's own join prompt (10 of 16 true pairs, 3 of 112 false). Qwen2.5-32B was no better (12 of 16, 9 false) and slower |
| **Join pairs too loose** | Pairs were "same topic" when two robotics papers shared a secondary arXiv category (e.g. both cs.LG); with 7B even per-pair found 1 of 16, e.g. a drone-navigation paper and a neuromorphic drone-control paper | Pair papers on the robotics task their titles name (aerial, driving, legged, grasping, SLAM), one task per paper |
| **Join wording** | Tested on 7B before changing models: the paper's question, a statement form, "related research topics" and "same type of robot or task" all gave all-no or all-yes answers | Kept the paper's wording; the fix was the model (above) |
| **Filter2 never passed a row** | With "Does this abstract mention using an image dataset?" (a single-abstract question) on joined pairs, 14B answered "false" to all 13 of per-tuple's pairs, so every pipeline output was empty | Filter2 is the paper's "Is this abstract about AI or machine learning?", which has a real mix of answers on pairs (per-tuple keeps 6 of 13) |
| **Join output invisible for per-tuple** | Per-tuple Kalypso runs the join check and Filter2 as one stage, so the trace could not show how many pairs the join kept; the drift check passed arms whose join output differed | The trace counts rows and answers per operator, and the drift check compares the join's output too |
| **Rows dropped by code?** | Checked: no unparseable or truncated block answers, every per-tuple answer began with true/false; the only code-level drops were 5-6 out-of-range pair ids in 4 x 32 join shapes | Nothing to fix; rows after Filter1 are decided by the model |

The first E4 and E5 runs (with the old Filter2) were discarded and rerun
(`results/m2-E4-qwen14b-v2`, `m2-E5-qwen14b-v2`).

## E1: block filter alone

64 rows per row length; block size k, Level 1 (rows only) and Level 2 (rows
and a 4096-token budget). Completion time (speedup over per-tuple):

| Block size (LLM calls) | Short: arXiv | Medium: MEDEC | Long: ContractNLI |
|---|---|---|---|
| per-tuple (64) | 3.65 s (1.00x) | 6.15 s (1.00x) | 19.81 s (1.00x) |
| 2 (32) | 3.05 s (1.20x) | 4.71 s (1.30x) | 19.54 s (1.01x) |
| 4 (16) | 3.31 s (1.10x) | 5.05 s (1.22x) | 21.57 s (0.92x) |
| 8 (8) | 4.02 s (0.91x) | 5.92 s (1.04x) | 26.54 s (0.75x) |
| 16 (4) | 5.89 s (0.62x) | 8.07 s (0.76x) | failed (over context) |
| 32 (2) | 9.77 s (0.37x) | 12.48 s (0.49x) | failed (over context) |
| 2 @ 4096 (32) | 3.13 s (1.17x) | 4.83 s (1.27x) | 19.55 s (1.01x) |
| 4 @ 4096 (16) | 3.44 s (1.06x) | 5.27 s (1.17x) | 19.73 s (1.00x) |
| 8 @ 4096 (8) | 4.28 s (0.85x) | 6.30 s (0.98x) | 20.22 s (0.98x) |
| 16 @ 4096 (5) | 6.23 s (0.59x) | 6.74 s (0.91x) | 20.14 s (0.98x) |
| 32 @ 4096 (5) | 6.25 s (0.58x) | 6.73 s (0.91x) | 19.86 s (1.00x) |

Small blocks win (k = 2 is fastest on short and medium rows, up to 1.30x), k >= 8
is slower than per-tuple, and on long contracts no block helps and k >= 16 does
not fit the context; the token budget makes every k runnable on long rows.

## E2: block join alone

16 robotics papers x 8 papers (f = 1, 16 true pairs), every shape k x j:

| Shape k x j | Completion time (speedup) | LLM calls | F1 |
|---|---|---|---|
| 8x4 | 2.03 s (4.15x) | 4 | 0.30 |
| 8x8 | 2.23 s (3.79x) | 2 | 0.18 |
| 8x2 | 2.45 s (3.45x) | 8 | 0.48 |
| 4x4 | 2.69 s (3.14x) | 8 | 0.55 |
| 4x2 | 2.73 s (3.09x) | 16 | 0.38 |
| 4x8 | 2.74 s (3.08x) | 4 | 0.43 |
| 4x1 | 3.36 s (2.52x) | 32 | 0.52 |
| 8x1 | 3.48 s (2.42x) | 16 | 0.46 |
| 2x4 | 4.04 s (2.09x) | 16 | 0.40 |
| 2x8 | 4.43 s (1.90x) | 8 | 0.57 |
| 2x2 | 4.48 s (1.88x) | 32 | 0.35 |
| 2x1 | 5.20 s (1.62x) | 64 | 0.48 |
| 1x8 | 6.63 s (1.27x) | 16 | 0.45 |
| 1x4 | 6.64 s (1.27x) | 32 | 0.38 |
| 1x2 | 7.15 s (1.18x) | 64 | 0.35 |
| per-tuple | 8.44 s (1.00x) | 128 | 0.69 |

More left rows per block is faster (8 x 4 is 4.15x), but every shape finds fewer
true pairs than per-pair (F1 0.18-0.57 vs 0.69).

## E3: input size vs block size

arXiv filter at N = 16 to 128 rows:

| N | per-tuple | k = 4 | k = 16 |
|---|---|---|---|
| 16 | 1.09 s (1.00x) | 1.28 s (0.85x) | 3.51 s (0.31x) |
| 32 | 1.96 s (1.00x) | 1.96 s (1.00x) | 4.40 s (0.44x) |
| 64 | 3.63 s (1.00x) | 3.29 s (1.10x) | 5.88 s (0.62x) |
| 128 | 6.99 s (1.00x) | 6.05 s (1.16x) | 8.82 s (0.79x) |

The best k grows with N: per-tuple wins up to 32 rows, k = 4 from 64 rows
(1.16x at 128); k = 16 is never best.

## E4: pipeline, one block size for all stages

Filter1 -> Join -> Filter2, s = 0.5, f = 1 (16 papers should pass Filter1; 13
join matches and 6 outputs for per-tuple):

| Block sizes (F1 / join / F2) | Completion time (speedup) | LLM calls | Join matches | Final outputs | F1 |
|---|---|---|---|---|---|
| per-tuple | 31.79 s (1.00x) | 557 | 13 | 6 | 0.80 |
| 2/2x2/2 | 16.52 s (1.92x) | 150 | 10 | 7 | 0.15 |
| 4/4x4/4 | 9.62 s (3.31x) | 42 | 2 | 1 | 0.00 |
| 8/8x8/8 | 10.13 s (3.14x) | 14 | 8 | 2 | 0.36 |
| 16/16x16/16 | 9.83 s (3.23x) | 4 | 0 | 0 | - |

Blocking the whole pipeline is 1.9-3.3x faster with up to 140x fewer calls, but
blocked joins keep far fewer matches (0-10 vs 13), so final F1 drops from 0.80
to 0-0.36.

## E5: pipeline, a block size per stage

The plan's grid: Filter1 k in {4, 16} x join in {2x2, 4x4, 4x32} x Filter2 k
in {4, 16}, s = 0.5, f = 1:

| Block sizes (F1 / join / F2) | Completion time (speedup) | LLM calls | Join matches | Final outputs | F1 |
|---|---|---|---|---|---|
| per-tuple | 31.37 s (1.00x) | 557 | 13 | 6 | 0.80 |
| 4/4x4/16 | 10.33 s (3.04x) | 42 | 4 | 1 | 0.00 |
| 4/4x4/4 | 10.52 s (2.98x) | 42 | 4 | 1 | 0.00 |
| 4/4x32/4 | 12.33 s (2.54x) | 14 | 2 | 1 | 0.00 |
| 16/4x4/4 | 15.77 s (1.99x) | 36 | 4 | 1 | 0.00 |
| 16/4x4/16 | 15.83 s (1.98x) | 36 | 4 | 1 | 0.00 |
| 4/2x2/4 | 16.39 s (1.91x) | 142 | 6 | 5 | 0.14 |
| 4/2x2/16 | 16.77 s (1.87x) | 141 | 6 | 3 | 0.00 |
| 4/4x32/16 | 17.29 s (1.81x) | 14 | 9 | 1 | 0.00 |
| 16/4x32/16 | 17.49 s (1.79x) | 8 | 6 | 2 | 0.00 |
| 16/4x32/4 | 17.59 s (1.78x) | 9 | 6 | 3 | 0.00 |
| 16/2x2/16 | 20.86 s (1.50x) | 135 | 6 | 3 | 0.00 |
| 16/2x2/4 | 21.39 s (1.47x) | 135 | 6 | 3 | 0.00 |

Filter1 at k = 4 with a 4 x 4 join is fastest (about 3x); Filter1 at k = 16 is
consistently slower, and Filter2's size barely matters.

## E6: selectivity sweep

E5's best settings at Filter1 selectivity s = 0.1 to 0.9 (f = 1):

| Condition | per-tuple | 4/4x4/16 | 4/4x4/4 | 4/2x2/4 |
|---|---|---|---|---|
| s = 0.1, f = 1.0 | 7.57 s (1.00x) | 5.32 s (1.42x) | 5.32 s (1.42x) | 6.69 s (1.13x) |
| s = 0.3, f = 1.0 | 20.86 s (1.00x) | 7.74 s (2.70x) | 8.04 s (2.59x) | 11.33 s (1.84x) |
| s = 0.6, f = 1.0 | 38.73 s (1.00x) | 11.39 s (3.40x) | 11.34 s (3.42x) | 19.05 s (2.03x) |
| s = 0.9, f = 1.0 | 51.79 s (1.00x) | 13.04 s (3.97x) | 12.96 s (4.00x) | 22.77 s (2.27x) |

The speedup grows with selectivity, from 1.4x at s = 0.1 to 4.0x at s = 0.9:
the more rows reach the join, the more blocking saves.

## E7: fanout sweep

E5's best settings at join fanout f = 0.2 to 5 (s = 0.5):

| Condition | per-tuple | 4/4x4/16 | 4/4x4/4 | 4/2x2/4 |
|---|---|---|---|---|
| s = 0.5, f = 0.2 | 33.15 s (1.00x) | 9.57 s (3.46x) | 9.61 s (3.45x) | 16.04 s (2.07x) |
| s = 0.5, f = 1.0 | 33.10 s (1.00x) | 10.64 s (3.11x) | 9.80 s (3.38x) | 16.59 s (1.99x) |
| s = 0.5, f = 3.0 | 32.74 s (1.00x) | 11.01 s (2.98x) | 11.11 s (2.95x) | 17.68 s (1.85x) |
| s = 0.5, f = 5.0 | 31.62 s (1.00x) | 18.49 s (1.71x) | 15.46 s (2.04x) | 18.46 s (1.71x) |

The speedup shrinks as fanout grows, from about 3.5x at f = 0.2 to 1.7-2.0x at
f = 5, where blocks have many matches to write out.

## E8: pipelined vs materialized

The same settings run pipelined (stages overlap) and materialized (each stage
finishes before the next starts), s = 0.5, f = 1:

| Block sizes | Pipelined | Materialized | Time saved by pipelining |
|---|---|---|---|
| per-tuple | 30.99 s | 41.68 s | 26% |
| 4/4x4/4 | 9.40 s | 16.49 s | 43% |
| 4/2x2/4 | 16.12 s | 28.44 s | 43% |

Pipelining saves 26% per-tuple and 43% with blocks: overlap matters more when
the stages are blocked.

## E9: block reforming off vs on

Coalescing Filter2's input across join tasks (Filter2 occupancy = rows per
block / block size), at s = 0.5, f = 1 and the two low-occupancy corners:

| Condition | Block sizes | Off: time, F2 occupancy | On: time, F2 occupancy |
|---|---|---|---|
| s = 0.5, f = 1.0 | 4/4x4/4 | 10.36 s, 0.50 | 10.76 s, 1.00 |
| s = 0.5, f = 1.0 | 4/2x2/4 | 16.37 s, 0.30 | 16.71 s, 0.75 |
| s = 0.1, f = 1.0 | 4/4x4/4 | 5.33 s, 0.25 | 5.36 s, 0.25 |
| s = 0.1, f = 1.0 | 4/2x2/4 | 7.14 s, 0.25 | 7.17 s, 0.25 |
| s = 0.5, f = 0.2 | 4/4x4/4 | 9.45 s, no F2 rows | 9.50 s, no F2 rows |
| s = 0.5, f = 0.2 | 4/2x2/4 | 16.10 s, no F2 rows | 16.05 s, no F2 rows |

Coalescing fills Filter2's blocks (0.50 to 1.00 occupancy) but never saves
time, because Filter2 receives only a few rows; in the low corners there is
nothing to merge.

## E10: accuracy vs block size

Collected from all runs above (`results/m2-E10-qwen14b.md` has every arm).
F1 against labels, F1 taking per-tuple's output as the truth, and fallback
calls (block answers redone per row or pair), for the blocked arms:

| Experiment | Per-tuple F1 | Blocked F1 | Blocked F1 vs per-tuple | Fallback calls (all blocked arms) |
|---|---|---|---|---|
| E1 | 0.71-1.00 | 0.40-1.00 | 0.33-1.00 | 101 |
| E2 | 0.69 | 0.18-0.57 | 0.40-0.72 | 0 |
| E3 | 0.99-1.00 | 0.96-1.00 | 0.97-1.00 | 0 |
| E4 | 0.80 | 0.00-0.36 | 0.00-0.50 | 6 |
| E5 | 0.80 | 0.00-0.14 | 0.00-0.18 | 30 |
| E6 | 0.18-0.67 | 0.00-0.50 | 0.00-0.50 | 26 |
| E7 | 0.62-1.00 | 0.00-0.30 | 0.00-0.45 | 21 |
| E8 | 0.80 | 0.00 | 0.00 | 6 |
| E9 | 0.80-1.00 | 0.00 | 0.00 | 9 |
| E11 | 0.69-0.80 | 0.00-0.30 | 0.00-0.47 | 2 |

Filters keep most of their accuracy in blocks (arXiv and contracts 0.95-1.00),
but block joins lose most of their matches, and that loss carries through the
whole pipeline.

## E11: Trummer's cost model vs measured

Level 3 (Trummer's token-cost model, `vllm/kalypso/controller/cost_model.py`)
picks k x j from measured token sizes; it runs next to the fastest measured
shape:

| Workload | Block sizes | Completion time (speedup) | LLM calls | F1 |
|---|---|---|---|---|
| Join alone, 16 x 8 | per-tuple | 8.05 s (1.00x) | 128 | 0.69 |
| Join alone, 16 x 8 | trummer (chose 16 x 8) | 2.16 s (3.73x) | 1 | 0.20 |
| Join alone, 16 x 8 | 8x4 | 1.96 s (4.10x) | 4 | 0.30 |
| Pipeline, s = 0.5, f = 1 | per-tuple | 31.45 s (1.00x) | 557 | 0.80 |
| Pipeline, s = 0.5, f = 1 | 4/trummer/4 (chose 31 x 30) | 5.28 s (5.96x) | 10 | - |
| Pipeline, s = 0.5, f = 1 | 4/4x4/4 | 9.43 s (3.34x) | 42 | 0.00 |

The token-cost optimum is not the latency optimum: Trummer picks the largest
block that fits, which is slower than 8 x 4 for the join alone and, in the
pipeline, fast only because its 31 x 30 join returns no matches.

## Static vs dynamic block size (regret)

The plan's test: for every condition c, k*(c) is the fastest arm and
`regret(c) = latency(k, c) / latency(k*(c), c)`; k_static is the arm with
the lowest worst-case regret. Regret at or below 1.1 everywhere means a static
k is enough. Computed from the results above with
`python -m vllm.kalypso.benchmark.eval.regret` (full output in
`results/m2-regret-filter-qwen14b.md` and `results/m2-regret-pipeline-qwen14b.md`).

### Filter (E1 row lengths and E3 input sizes)

| Arm | E1 long | E1 medium | E1 short | E3 N = 128 | E3 N = 16 | E3 N = 32 | E3 N = 64 | Worst | Mean |
|---|---|---|---|---|---|---|---|---|---|
| per-tuple | 1.01 | 1.30 | 1.20 | 1.16 | 1.00 | 1.00 | 1.10 | 1.30 | 1.11 |
| k = 4 | 1.10 | 1.07 | 1.09 | 1.00 | 1.17 | 1.00 | 1.00 | **1.17** | 1.06 |
| k = 16 | failed | 1.71 | 1.93 | 1.46 | 3.20 | 2.25 | 1.78 | failed | failed |

k_static = 4, worst-case regret 1.17 (at N = 16 and, just over 1.1, long
contracts): the filter's block size has to adapt to input size and row
length. (Only the arms both experiments ran are compared; E1's fastest arm,
k = 2, sets k*(c) for its conditions.)

### Pipeline (E6 selectivities and E7 fanouts)

| Arm | s = 0.1 | s = 0.3 | s = 0.6 | s = 0.9 | f = 0.2 | f = 1 | f = 3 | f = 5 | Worst | Mean |
|---|---|---|---|---|---|---|---|---|---|---|
| per-tuple | 1.42 | 2.70 | 3.42 | 4.00 | 3.46 | 3.38 | 2.98 | 2.04 | 4.00 | 2.92 |
| 4 / 4x4 / 16 | 1.00 | 1.00 | 1.00 | 1.01 | 1.00 | 1.09 | 1.00 | 1.20 | 1.20 | 1.04 |
| 4 / 4x4 / 4 | 1.00 | 1.04 | 1.00 | 1.00 | 1.00 | 1.00 | 1.01 | 1.00 | **1.04** | 1.01 |
| 4 / 2x2 / 4 | 1.26 | 1.46 | 1.68 | 1.76 | 1.68 | 1.69 | 1.61 | 1.19 | 1.76 | 1.54 |

k_static = 4 / 4x4 / 4, worst-case regret 1.04: one pipeline setting is within
4% of the best in every condition, so a static k is enough on latency. Every
blocked arm's work drifted from per-tuple (fewer join matches; see E4 and E10),
which this latency-only measure does not account for.

## Insights: which block sizes are good

- **Filters: small blocks win.** k = 2-4 is up to 1.30x faster than
  per-tuple (E1) and keeps F1 at 0.95-1.00 on arXiv and contracts; use
  per-tuple for small inputs (N <= 32, E3) and for long rows (contracts, E1);
  avoid k >= 8, which is slower and on long rows does not fit the context.
- **Joins alone: a moderate square block is the best balance.** 4 x 4 is 3.14x
  faster than per-pair with F1 0.55 (per-pair 0.69); the fastest shape, 8 x 4
  (4.15x), drops F1 to 0.30, and 2 x 8 keeps the most accuracy (F1 0.57) at
  only 1.90x (E2).
- **Pipelines: about 3x faster, but blocked joins lose answers.** 8 / 8x8 / 8 is
  the best measured balance (3.14x, 8 of per-tuple's 13 join matches, F1 0.36);
  the fastest, 4 / 4x4 / 4 (3.31x), keeps 2 matches and F1 0.00; 2 / 2x2 / 2
  keeps the most matches (10) at 1.92x (E4). Every blocked setting is far below
  per-tuple's F1 0.80 on Qwen2.5-14B.
- **Where blocking pays most:** high Filter1 selectivity (4.0x at s = 0.9, E6)
  and low join fanout (3.5x at f = 0.2, E7); with many matches per block
  (f = 5) the gain falls to 1.7-2.0x.
- **Pipelining matters more with blocks:** overlapping the stages saves 43% of
  the time for blocked settings versus 26% for per-tuple (E8).
- **Reforming did not pay off here:** coalescing filled Filter2's blocks but
  saved no time, because Filter2 receives only a few rows (E9).
- **Trummer's cost model (Level 3) picks poor block sizes:** it minimises tokens,
  so it chooses the largest block that fits. For the join alone that is 10%
  slower than 8 x 4 and less accurate (F1 0.20 vs 0.30); in the pipeline its
  31 x 30 join returned no matches, so its 5.96x "speedup" is a dropped result
  (E11). Adding decode cost and an accuracy floor is what Level 4 is for.
- **One static setting is fine for the pipeline, not for filters:** 4 / 4x4 / 4
  is within 4% of the best pipeline time in every selectivity and fanout tested,
  while the filter's best k depends on input size and row length (regret 1.17).
- **Not yet measured:** per-tuple joins with blocked filters (blocking only
  where it keeps accuracy) were not in the plan's grids; it is the obvious
  next configuration to test given the block join's accuracy loss.
