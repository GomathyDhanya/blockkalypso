# Evaluation harness

Runs a workload several ways ("arms") against a live server and records the
same server-side metrics, plus accuracy where labels exist, for each. It covers
the Milestone 1 baseline and the block filter and block join.

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

There is no block map operator; see `docs/blockkalypso/milestone1.md`.

`vllm_direct` sends exactly what Kalypso's per-tuple path sends: same messages
(Kalypso's `get_data_prompt` / `get_prompt`), `VLLMExecutor.DEFAULT_SAMPLING`
and chat-template arguments, and `max_tokens` 8 for filters and join pairs.
`test_direct_prompts_match_kalypso_operators` checks this against the real
operators. For maps it uses `max_tokens` 512, Kalypso's retry ceiling, where
Kalypso estimates a per-row limit.

## Server

```bash
VLLM_SERVER_DEV_MODE=1 VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_CPU_KVCACHE_SPACE=2 \
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-0.5B-Instruct --dtype float16 \
  --max-model-len 8192 --enable-prefix-caching --port 8003
```

- `VLLM_SERVER_DEV_MODE=1` exposes `/reset_prefix_cache`. The harness resets
  the cache before every trial; arms send identical prompts, so without it
  each arm inherits the previous arm's cache.
- `--max-model-len 8192`: with no history, `SemMap` asks for `max_tokens`
  equal to the prompt length, so BioDEX articles (1.9k-3.7k tokens) fail at
  4096. A 10-note MEDEC block is also ~5k tokens.
- Join arms need a server running code whose `/v1/semantic/query` results
  include `right_idx`.
- Accuracy: Qwen2.5-0.5B answers NO to every row inside a block and leans
  towards true per row, so its accuracy says little about the operators. Use
  `--model Qwen/Qwen2.5-1.5B-Instruct` (about 3x slower) for accuracy runs.
- `--dtype bfloat16` on M2 and later; `float16` on M1.

## Run

Everything for Milestone 1, one results file per workload plus `summary.md`:

```bash
python -m vllm.kalypso.benchmark.eval.suite \
  --out-dir vllm/kalypso/benchmark/eval/results/m1
```

One workload, chosen arms:

```bash
python -m vllm.kalypso.benchmark.eval.run --workload reviews \
  --arms kalypso_per_tuple,kalypso_block:5,kalypso_block:20 \
  --out vllm/kalypso/benchmark/eval/results/reviews.jsonl
python -m vllm.kalypso.benchmark.eval.report \
  vllm/kalypso/benchmark/eval/results/reviews.jsonl
```

Defaults: one warmup trial per arm, then 3 repeats with arms interleaved
(A, B, A, B, ...). Run nothing else against the server meanwhile: the counters
are server-wide. An arm that fails (e.g. a block too large for the context) is
recorded with its error and the run continues.

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

## Layout

| File | Role |
|---|---|
| `workloads.py` | Rows, instruction and labels per workload, read the way Kalypso reads them |
| `arms.py` | Arm parsing, Kalypso query ops, direct-vLLM prompts |
| `metrics.py` | `/metrics` parsing and deltas, KV and RSS samplers, `measure()` |
| `run.py` | CLI for one workload: warmup, interleaved repeats, JSONL output |
| `report.py` | JSONL to markdown table with accuracy and agreement |
| `suite.py` | All Milestone 1 workloads plus `summary.md` |

Tests: `vllm/kalypso/tests/test_eval_harness.py`.

## Next

- M2: full MEDEC (~2k labeled notes) and ContractNLI (gold labels per
  contract-hypothesis pair, the proposal's 500 x 20 join) as workloads.
- M3: a Filter1 → Join → Filter2 workload with per-stage timestamps.
