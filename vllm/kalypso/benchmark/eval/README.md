# Evaluation harness

Runs a workload several ways ("arms") against a live server and records the
same server-side metrics, plus accuracy where labels exist, for each. It covers
the Milestone 1 baseline (per-tuple Kalypso vs plain vLLM), the block filter
and block join, and the Milestone 2 experiments (E1-E3 of the experiment plan).
Results: [Milestone 1](#milestone-1-results), [Milestone 2](#milestone-2-results-e1-e3).

## Workloads

| Name | Kind | Data | Labels |
|---|---|---|---|
| `biodex` | map | 10 BioDEX articles, reaction extraction | none |
| `reviews` | filter | 20 restaurant reviews (`block_filter/reviews_20.csv`) | complains: 10 of 20 |
| `medec` | filter | 10 MEDEC clinical notes | has a medical error: 7 of 10 |
| `join5x5` | join | 5 complaints x 5 categories (`block_join/`) | 5 expected matches |
| `medec_full` | filter | seeded sample of MEDEC-MS notes (default 64) | has a medical error |
| `contractnli_filter` | filter | first n ContractNLI contracts (default 64), paper's valid-contract prompt | all are NDAs |
| `contractnli` | join | first n contracts (default 20) x 17 hypotheses | gold Entailment |
| `contractnli_fjm`, `contractnli_fj` | pipeline | the paper's ContractNLI filter -> join (-> map) | gold Entailment |
| `arxiv_filter` | filter | n 2023 abstracts (default 64), share `s` robotics; paper's arXiv filter | proxy: cs.RO |
| `arxiv_join` | join | n robotics x `right` abstracts with fanout `f` (default 16 x 8, f = 1) | proxy: same robotics topic |
| `arxiv_fjf` | pipeline | the paper's arXiv FILTER -> JOIN -> FILTER: n x `right` (default 32 x 32), `s`, `f` | proxy |

The first four are the Milestone 1 samples; the rest use the full public
datasets, which `python -m vllm.kalypso.benchmark.eval.fetch_data` writes to
`vllm/kalypso/benchmark/data/` (git-ignored; sources, checksums and arXiv
queries in its `manifest.json`). `s`, `f`, `right` and `seed` are set with
`--param KEY=VALUE`. The 10-row samples are labeled; one item moves accuracy
by 4-10 points.

## Arms

| Arm | What runs | Workloads |
|---|---|---|
| `kalypso_per_tuple` | Kalypso's original operators: one call per row (`sem_map`, `sem_filter`) or per pair (`join`) | all |
| `vllm_direct` | The same per-row prompts sent concurrently to `/v1/chat/completions`, no Kalypso | all |
| `kalypso_block:N` | `sem_filter` with `block_size: N` (BlockSemFilter) | filter |
| `kalypso_block:KxJ` | `join` with `left_block_size: K`, `right_block_size: J` (BlockSemJoin) | join |
| `kalypso_block:N@T` | as `N`, also capped at T estimated tokens per block (`max_block_tokens`, Level 2) | filter |
| `kalypso_block:N/KxJ/N` | one size per filter or join stage, in order (1 or 1x1 = per-tuple) | pipeline |

There is no block map operator: map outputs are long free text, so one call
would decode every row's answer in sequence, and answers are hard to separate
reliably.

`vllm_direct` sends exactly what Kalypso's per-tuple path sends: same messages
(Kalypso's `get_data_prompt` / `get_prompt`), `VLLMExecutor.DEFAULT_SAMPLING`
and chat-template arguments, and `max_tokens` 8 for filters and join pairs.
`test_direct_prompts_match_kalypso_operators` checks this against the real
operators. For maps it uses `max_tokens` 512, Kalypso's retry ceiling, where
Kalypso estimates a per-row limit.

## Server

Start it from this checkout (join arms need the `right_idx` that
`/v1/semantic/query` now returns):

```bash
export VLLM_SERVER_DEV_MODE=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_CPU_KVCACHE_SPACE=2
M=Qwen/Qwen2.5-0.5B-Instruct
python -m vllm.entrypoints.openai.api_server \
  --model $M --dtype float16 --port 8003 \
  --max-model-len 8192 --enable-prefix-caching \
  --block-size 16
```

- `VLLM_SERVER_DEV_MODE=1` exposes `/reset_prefix_cache`. The harness resets
  the cache before every trial; arms send identical prompts, so without it
  each arm inherits the previous arm's cache.
- `--max-model-len 8192`: with no history, `SemMap` asks for `max_tokens`
  equal to the prompt length, so BioDEX articles (1.9k-3.7k tokens) fail at
  4096. A 10-note MEDEC block is also ~5k tokens.
- `--block-size 16`: the CPU backend defaults to 128-token KV blocks, and the
  prefix cache only reuses whole blocks. Kalypso's prompts share a ~53-token
  prefix (chat template plus system prompt), so at 128 rows never share cache
  and the per-tuple baseline is penalised compared with a GPU.
- `--dtype bfloat16` on M2 and later; `float16` on M1.

## Run

Everything for Milestone 1, one results file per workload plus `summary.md`:

```bash
P=vllm.kalypso.benchmark.eval
R=vllm/kalypso/benchmark/eval/results
python -m $P.suite --out-dir $R/m1
```

One workload, chosen arms:

```bash
python -m $P.run --workload reviews \
  --arms kalypso_per_tuple,kalypso_block:5 \
  --out $R/reviews.jsonl
python -m $P.report $R/reviews.jsonl
```

Defaults: one warmup trial per arm, then 3 repeats with arms interleaved
(A, B, A, B, ...). Run nothing else against the server meanwhile: the counters
are server-wide. An arm that fails (e.g. a block too large for the context) is
recorded with its error and the run continues.

## Running on a GPU (Modal)

`cloud/modal_eval.py` builds the project's Kalypso container
(`docker/Dockerfile.kalypso`: vLLM 0.29.0 plus this checkout's Kalypso),
fetches the full datasets into the image, then starts the server and runs
`eval.suite` in one container on an L4 GPU. Results go to the Modal volume
`blockkalypso-results`.

```bash
pip install modal && python3 -m modal setup    # once
python3 -m modal run cloud/modal_eval.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --workloads contractnli_fj,medec_full --out m2-qwen7b
python3 -m modal volume get blockkalypso-results m2-qwen7b ./results
```

GPU functions need a payment method on the Modal account; set a spending
limit so runs stop at the free credit. On the GPU this is Kalypso on vLLM
0.29.0, not the repository's vLLM fork used for local CPU runs, so compare
arms within one setup. GPU batching is not bit-for-bit deterministic, so
per-tuple Kalypso and `vllm_direct` can disagree on a few rows there.

## Metrics (one JSON line per trial)

| Field | Source |
|---|---|
| `latency_s` | Client wall clock around the arm |
| `llm_requests`, `requests_by_reason` | Delta of `vllm:request_success_total`, by `finished_reason` |
| `truncated` | The `length` share of that delta |
| `prompt_tokens`, `generation_tokens` | Deltas of `vllm:prompt_tokens_total`, `vllm:generation_tokens_total` |
| `prefix_cache_hit_rate` | Delta hits / delta queries |
| `peak_kv_cache_pct` | Max of `vllm:kv_cache_usage_perc`, sampled every 0.25 s |
| `peak_server_rss_kb` | Max RSS of the server process tree, sampled every 0.1 s |
| `rows_per_s` | Items decided per second (rows, or pairs for a join) |
| `selected` | Rows that passed, or (left, right) pairs that matched |
| `outputs` | Model output per row or pair |
| `error` | Present instead of the above when the arm failed |

`report.py` prints the median and [min-max] per arm. For labeled workloads it
adds accuracy, precision, recall and F1 against the labels, over every row
(filter) or every candidate pair (join). Agreement is the share of items an
arm decides the same as `kalypso_per_tuple` in the same repeat (text equality
for maps), and self-agreement compares an arm with its own first repeat.

The first line of every results file records the git SHA and dirty flag,
model, server command line and start time, platform, CPU, workload, labels and
run settings.

## Milestone 1 results

Apple M4 CPU, server as above, prefix cache reset before every trial, 1 warmup
then 3 repeats per arm; numbers are medians. Full tables, including ranges,
memory and agreement, are in `results/m1-qwen0.5b/summary.md` (all workloads)
and `results/m1-qwen1.5b/summary.md` (labeled workloads).

All runs passed these checks: per-tuple Kalypso and `vllm_direct` processed
identical prompt tokens and agreed on every item; every arm gave the same
answers on every repeat (self-agreement 100%); no arm failed.

### Baseline: per-tuple Kalypso vs vLLM (BioDEX map, Qwen2.5-0.5B)

| Arm | Latency | LLM requests | Prompt tokens | Peak KV | Peak RSS |
|---|---|---|---|---|---|
| `kalypso_per_tuple` | 240.4 s | 10 | 28,872 | 13.3% | 1.7 GB |
| `vllm_direct` | 244.3 s | 10 | 28,872 | 12.7% | 1.6 GB |

For a single map operator Kalypso adds no measurable overhead and no gain over
vLLM's own batching; its scheduling wins are for multi-operator pipelines.

### Block filter

| Workload | Model | Per-tuple | Block 5 | Block 10 | Block 20 |
|---|---|---|---|---|---|
| reviews latency | 0.5B | 10.4 s | 3.1 s | 3.1 s | 3.6 s |
| reviews latency | 1.5B | 36.1 s | 9.8 s | 9.4 s | 11.1 s |
| reviews requests / prompt tokens | both | 20 / 2,818 | 4 / 582 | 2 / 426 | 1 / 352 |
| reviews F1 (accuracy) | 0.5B | 0.57 (0.40) | n/a (0.50) | n/a (0.50) | n/a (0.50) |
| reviews F1 (accuracy) | 1.5B | n/a (0.50) | 0.27 (0.20) | 0.11 (0.15) | 0.00 (0.05) |
| MEDEC latency | 0.5B | 40.0 s | 40.6 s | 51.7 s | 50.6 s |
| MEDEC latency | 1.5B | 139.1 s | 121.4 s | 144.7 s | 144.2 s |
| MEDEC requests / prompt tokens | both | 10 / 6,772 | 2 / 4,946 | 1 / 4,770 | 1 / 4,770 |
| MEDEC F1 (accuracy) | 1.5B | 0.60 (0.60) | 0.67 (0.60) | 0.82 (0.70) | 0.82 (0.70) |

"n/a" means the arm passed no rows. MEDEC has 10 rows, so blocks 10 and 20
are the same single block.

### Block join (5 x 5, 25 candidate pairs)

| Model | Metric | Per-pair | Block 2x2 | Block 5x5 |
|---|---|---|---|---|
| both | Requests / prompt tokens | 25 / 4,310 | 9 / 1,219 | 1 / 224 |
| 0.5B | Latency | 17.2 s | 5.9 s | 2.2 s |
| 1.5B | Latency | 66.5 s | 19.4 s | 6.4 s |
| 0.5B | F1 (precision / recall) | 0.80 (0.80 / 0.80) | 0.38 (0.25 / 0.80) | 0.00 (0 / 0) |
| 1.5B | F1 (precision / recall) | 0.33 (0.20 / 1.00) | 0.67 (0.50 / 1.00) | 0.60 (0.60 / 0.60) |

Block 2x2 makes 9 calls: 5 rows per side split into blocks of 2, 2 and 1,
so there are 3 x 3 = 9 block pairs (the 1x1 corner pair is a plain per-pair
call).

### What the numbers show

- **Short rows: large system wins.** Blocking cuts requests 3-25x and prompt
  tokens 3.5-19x. Latency drops 3-4x for the review filter and 3-10x for the
  join, on both models.
- **Long rows: little or no win.** MEDEC notes are long relative to the
  instruction, so blocking saves only ~27% of prompt tokens, and one large
  request loses the parallelism of many small ones: a single 10-note block is
  up to 29% slower than per-tuple. Blocks should be bounded by tokens, not
  just rows (the Level 2 `max_block_tokens` limit).
- **Memory.** Peak KV use is small throughout. Blocks of short rows use less
  peak KV than per-tuple (0.3% vs 1.1% on reviews); blocks of long rows use
  more (6.5% vs 2.9% on MEDEC at 1.5B). Server RSS is dominated by model
  weights (~1.3 GB for 0.5B, ~3.4 GB for 1.5B) and does not separate arms.
- **Accuracy is limited by the models, per-tuple included.** At 0.5B, block
  filters answer NO to every row and every MEDEC arm answers "no error". At
  1.5B, Kalypso's per-tuple filter answers `false` to all 20 reviews and its
  per-pair join answers `true` to all 25 pairs. Raw block output on reviews
  at 1.5B answers the question inverted (YES for the pleasant reviews); the
  operator's parsing was checked and is correct. Higher block F1 is not always
  a real gain either: on MEDEC (7 of 10 positive), blocks 10 and 20 say YES to
  every note, which alone gives F1 0.82.
- **Prompt wording matters as much as blocking at this size.** On 1.5B the
  block filter's own one-row prompt (`block_filter_standalone.py`) gets 17 of
  20 reviews right, while Kalypso's `SemFilter` prompt gets 10 of 20.

### Limitations

- Small labeled samples (10-25 items); differences of one or two items are
  noise.
- CPU only: absolute latencies do not transfer to GPUs; ratios between arms
  are the useful part.
- Models of 1.5B parameters and below cannot do these tasks reliably even one
  row at a time, so these runs cannot isolate the accuracy cost of blocking.

## Milestone 2 results (E1-E3)

E1-E3 of the Milestone 2 experiment plan, exactly as its grids specify,
defined in `experiments.py` and run with `cloud/modal_eval.py --experiment`.
Setup: one NVIDIA L4, Qwen2.5-1.5B-Instruct (the plan's E10 model; 7B
results [below](#repeated-on-qwen25-7b-nvidia-l40s)), Kalypso on vLLM 0.29.0
(the project container), prefix cache reset before every trial, 1 warmup and 3
repeats per arm, medians. Metrics follow the Kalypso paper: query completion
time, speedup over per-tuple Kalypso (the request-centric baseline), and LLM
calls; accuracy and peak memory are added from the proposal, because blocking,
unlike Kalypso's own optimizations, can change answers. Each experiment's
folder in `results/` holds the per-run JSONL and `summary.md` (main table,
then diagnostics: tokens, prefix reuse, stages). Runs were made from commit `d19f407a0` plus the then-uncommitted
experiment code, now in the commits that add `experiments.py`.

### E1: block filter alone (N = 64)

Query completion time (speedup over per-tuple) by block size, row length,
and Level 1 (rows only) vs Level 2 (also at most 4096 estimated tokens per
block); LLM calls in brackets (on contracts, Level 2 makes 39-42):

| Arm (LLM calls) | Short rows: arXiv (~235 tok) | Medium: MEDEC (~390) | Long: contracts (~1750) |
|---|---|---|---|
| per-tuple (64) | 1.57 s (1.00x) | 2.76 s (1.00x) | **9.48 s (1.00x)** |
| k = 2 (32) | **1.45 s (1.08x)** | **2.25 s (1.23x)** | 9.57 s (0.99x) |
| k = 4 (16) | 1.48 s (1.06x) | 2.36 s (1.17x) | 10.47 s (0.90x) |
| k = 8 (8) | 1.77 s (0.88x) | 2.69 s (1.03x) | 12.59 s (0.75x) |
| k = 16 (4) | 2.45 s (0.64x) | 3.56 s (0.78x) | failed (over the 32k context) |
| k = 32 (2) | 3.95 s (0.40x) | 5.36 s (0.51x) | failed (over the 32k context) |
| k = 2 @ 4096 (32) | 1.54 s (1.02x) | 2.41 s (1.14x) | 9.94 s (0.95x) |
| k = 4 @ 4096 (16) | 1.72 s (0.91x) | 2.70 s (1.02x) | 10.61 s (0.89x) |
| k = 8 @ 4096 (8) | 2.18 s (0.72x) | 3.30 s (0.84x) | 10.93 s (0.87x) |
| k = 16 @ 4096 (5) | 3.10 s (0.51x) | 3.60 s (0.77x) | 10.92 s (0.87x) |
| k = 32 @ 4096 (5) | 3.11 s (0.50x) | 3.60 s (0.77x) | 11.09 s (0.85x) |

- Small blocks win on the GPU: k = 2 is fastest for short and medium rows,
  and per-tuple is (just) fastest for long rows. Larger blocks are slower:
  fewer requests for vLLM to run in parallel, and one call decodes every
  row's answer in sequence. Prompt tokens fall by up to 33% (arXiv 24.3k to
  16.3k) but that does not pay for the lost parallelism here.
- Row length matters as predicted: on long contracts blocking saves only
  5-7% of prompt tokens and never helps; k = 16 and 32 cannot run at all.
- Level 2 makes every k runnable on long rows (the budget keeps blocks to one
  or two contracts; single-contract blocks run as plain calls, the 20
  fallback calls in the table). On short and medium rows it is slower than
  Level 1 at the same k even when it forms the same blocks (arXiv k = 8:
  2.18 s vs 1.77 s, 8 blocks each); not yet profiled, but the likely cost is
  tokenizing candidate blocks while forming them.
- Peak KV grows with block size on long rows (3.1% per-tuple to 13.8% at
  k = 8).
- The same small block, k = 2, is the best or within 1% of the best for all
  three row lengths.
- Accuracy (recorded, not used to choose k; F1): arXiv per-tuple 1.00 vs
  0.59-0.73 for blocks; contracts per-tuple 0.99 vs 0.84-1.00. On MEDEC the
  1.5B model fails per-tuple too (says "no error" to 63 of 64 notes, F1 0.00).

### E2: block join alone (16 x 8 arXiv abstracts, f = 1)

| Shape | Completion time (speedup) | LLM calls | Matches (16 expected) |
|---|---|---|---|
| per-pair (1x1) | 3.35 s (1.00x) | 128 | 128 (says yes to every pair) |
| 1x8 | 2.97 s (1.13x) | 16 | 18 |
| 2x2 | 2.38 s (1.41x) | 32 | 16 |
| 2x4 | 2.05 s (1.64x) | 16 | 11 |
| 4x8 | 1.81 s (1.85x) | 4 | 12 |
| 8x8 | **1.46 s (2.29x)** | 2 | 8 |
| 8x1 | 1.13 s (2.97x) | 16 | 0 |
| 8x4 | 0.70 s (4.78x) | 4 | 0 |

All 15 shapes are in `results/m2-E2-qwen1.5b/summary.md`. Every shape with
k >= 4 and j <= 4 answered NONE for every block (0 matches, about 2 output
tokens per call), so its speed partly reflects writing nothing. Among shapes
that did answer, blocking ranges from 0.9x (1x4, slightly slower) to 2.3x
(8x8) the speed of per-pair, with 2-64x fewer calls; more left rows per block
is faster. Skewed shapes are not symmetric: 8x1 (1.13 s) vs 1x8 (2.97 s) for the
same pairs and tokens, partly because right blocks run one after another
inside a join task and partly because 8x1 answered NONE. At 1.5B neither
side is accurate: per-pair matches everything (precision 0.12) and blocks
find 0-4 of the 16 true matches.

### E3: input size vs block size (arXiv filter)

Query completion time (speedup over per-tuple); LLM calls are N, N/4 and
N/16:

| N | per-tuple | k = 4 | k = 16 |
|---|---|---|---|
| 16 | **0.39 s (1.00x)** | 0.54 s (0.72x) | 1.44 s (0.27x) |
| 32 | **0.82 s (1.00x)** | 0.84 s (0.97x) | 1.77 s (0.46x) |
| 64 | 1.51 s (1.00x) | **1.46 s (1.04x)** | 2.44 s (0.62x) |
| 128 | 3.07 s (1.00x) | **2.69 s (1.14x)** | 3.77 s (0.81x) |

The best k grows with N, as the plan expected: with few rows separate
requests run in parallel and per-tuple wins; from about 64 rows the GPU is
busy anyway and fewer requests pay off, reaching 1.14x at N = 128. k = 16 is
never best. So the best k depends on N. Accuracy: per-tuple F1 0.98-1.00, blocks 0.59-0.70.

### Repeated on Qwen2.5-7B (NVIDIA L40S)

The same three experiments, grids and protocol on Qwen2.5-7B-Instruct, run
on an L40S (48 GB) so the 7B weights and 32k-token blocks fit
(`results/m2-E*-qwen7b/`).

| | Qwen2.5-1.5B (L4) | Qwen2.5-7B (L40S) |
|---|---|---|
| E1 best k: short / medium / long rows | 2 / 2 / per-tuple | 2 / 2 / per-tuple |
| E1 best speedup: short / medium / long rows | 1.08x / 1.23x / 1.00x | 1.09x / 1.05x / 1.00x |
| E1 arXiv filter F1: per-tuple vs blocks | 1.00 vs 0.59-0.73 | 1.00 vs 0.67-0.90 |
| E1 contracts F1: per-tuple vs blocks | 0.99 vs 0.84-1.00 | 1.00 vs 0.92-0.97 |
| E1 MEDEC F1: per-tuple vs blocks | 0.00 vs 0.11-0.64 | 0.60 vs 0.00-0.37 |
| E2 fastest shape, speedup over per-pair | 8x8, 2.3x | 8x8, 5.2x |
| E3 crossover (blocks start to win) | N = 64 | N = 64 |
| E3 speedup of k = 4 at N = 128 | 1.14x | 1.12x |
| E3 arXiv filter F1, k = 4 | 0.63-0.70 | 0.81-0.97 |

- The speed findings do not depend on model size: the same best k in E1,
  the same crossover at N = 64 in E3, and similar speedups.
- Joins gain more on the bigger model: up to 5.2x faster than per-pair, scaling
  cleanly with left rows per block. On 7B every E2 arm, per-pair included,
  writes about 2 tokens per call, so this speedup comes from fewer, shared
  calls and not from shorter answers.
- The accuracy gap narrows on arXiv and contracts but does not close: with
  the same 7B model, blocks stay 3-33 F1 points behind per-tuple. On MEDEC,
  7B is the first model that gets per-tuple partly right (F1 0.60), and
  blocks fall far behind it (0.00-0.37). So part of the accuracy cost is
  model size and part is blocking itself.
- E2's arXiv join labels are too loose to score: on 7B even per-pair finds
  only 1 of the 16 "same topic" pairs. The proxy (two robotics papers that
  share a secondary category, e.g. both cs.LG) is not what "Do these two
  papers study similar topic?" asks, and the stronger model says no. E2's
  speed results stand; its precision and recall should not be used.

### Limits

- Two models, one GPU each; ratios between arms are the result, not the
  absolute times (L4 and L40S runs are not comparable in seconds). 3 repeats;
  ranges are in each `summary.md`.
- arXiv labels are category proxies; contracts and MEDEC have gold labels.
  The arXiv join proxy is too loose (see above).
- One query at a time on an otherwise idle GPU, and N <= 128: the best case
  for per-tuple. Throughput under concurrent queries and larger N are not
  measured yet.

## Layout

| File | Role |
|---|---|
| `workloads.py` | Rows, instruction and labels per workload, read the way Kalypso reads them |
| `arms.py` | Arm parsing, Kalypso query ops, direct-vLLM prompts |
| `metrics.py` | `/metrics` parsing and deltas, KV and RSS samplers, `measure()` |
| `run.py` | CLI for one workload: warmup, interleaved repeats, JSONL output |
| `report.py` | JSONL to markdown tables: accuracy, agreement, stages, drift |
| `stages.py` | The server's per-query trace as per-stage numbers |
| `suite.py` | Several workloads with default arms, plus `summary.md` |
| `experiments.py` | The Milestone 2 experiment grids (E1-E3) |
| `fetch_data.py` | Downloads ContractNLI, MEDEC and the arXiv pool |
| `results/` | Results per milestone, experiment and model |
| `../../../../cloud/modal_eval.py` | Runs a suite or experiment on a Modal GPU |

Tests: `vllm/kalypso/tests/test_eval_harness.py`.

## Next

- E4-E7 on `arxiv_fjf` (Filter1 -> Join -> Filter2). At 1.5B and 3B the
  blocked arms change how many rows pass Filter1 (the drift table flags
  every blocked arm), so the plan's "held fixed s, f" does not hold; how to
  compare them is an open decision. The arXiv join also needs a tighter
  "same topic" pairing first: on 7B the planned fanout f = 1 is about 0.06.
- Larger N (E3 beyond 128) and throughput with several queries at once.
- Materialized mode (E8) and Filter2 coalescing (E9), then E10 from all runs.
- E11 needs the Level 3 cost model.
