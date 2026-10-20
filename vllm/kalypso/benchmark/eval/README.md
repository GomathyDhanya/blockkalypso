# Evaluation harness

Runs a workload several ways ("arms") against a live server and records the
same server-side metrics, plus accuracy where labels exist, for each. It covers
the Milestone 1 baseline (per-tuple Kalypso vs plain vLLM) and the block filter
and block join. Milestone 1 results are [below](#milestone-1-results).

## Workloads

| Name | Kind | Data | Labels |
|---|---|---|---|
| `biodex` | map | 10 BioDEX articles, reaction extraction | none |
| `reviews` | filter | 20 restaurant reviews (`block_filter/reviews_20.csv`) | complains: 10 of 20 |
| `medec` | filter | 10 MEDEC clinical notes | has a medical error: 7 of 10 |
| `join5x5` | join | 5 complaints x 5 categories (`block_join/`) | 5 expected matches |

All labeled sets are small: one item moves accuracy by 4-10 points.

## Arms

| Arm | What runs | Workloads |
|---|---|---|
| `kalypso_per_tuple` | Kalypso's original operators: one call per row (`sem_map`, `sem_filter`) or per pair (`join`) | all |
| `vllm_direct` | The same per-row prompts sent concurrently to `/v1/chat/completions`, no Kalypso | all |
| `kalypso_block:N` | `sem_filter` with `block_size: N` (BlockSemFilter) | filter |
| `kalypso_block:KxJ` | `join` with `left_block_size: K`, `right_block_size: J` (BlockSemJoin) | join |

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

## Layout

| File | Role |
|---|---|
| `workloads.py` | Rows, instruction and labels per workload, read the way Kalypso reads them |
| `arms.py` | Arm parsing, Kalypso query ops, direct-vLLM prompts |
| `metrics.py` | `/metrics` parsing and deltas, KV and RSS samplers, `measure()` |
| `run.py` | CLI for one workload: warmup, interleaved repeats, JSONL output |
| `report.py` | JSONL to markdown table with accuracy and agreement |
| `suite.py` | All Milestone 1 workloads plus `summary.md` |
| `results/` | Milestone 1 results per model |

Tests: `vllm/kalypso/tests/test_eval_harness.py`.

## Next

- M2: a stronger model (e.g. Qwen2.5-7B, on a GPU) so per-tuple accuracy is
  high enough to measure what blocking costs; full MEDEC (~2k labeled notes)
  and ContractNLI (gold labels per contract-hypothesis pair, the proposal's
  500 x 20 join) as workloads.
- M3: a Filter1 → Join → Filter2 workload with per-stage timestamps.
