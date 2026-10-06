# Evaluation harness

Runs one workload several ways ("arms") against a live server and records the
same server-side metrics for each, so per-tuple Kalypso, plain vLLM and, later,
the block operators are compared on equal terms.

## Milestone 1 baseline

Workload: BioDEX `sem_map` over `sample_data/articles_500` (10 articles) with
`scenarios.BIODEX_MAP_REACTIONS`.

| Arm | What runs |
|---|---|
| `kalypso_per_tuple` | `sem_map` posted to `/v1/semantic/query`; Kalypso plans, schedules and issues one call per article |
| `vllm_direct` | The same per-article messages sent concurrently to `/v1/chat/completions`, bypassing Kalypso's query processor |

Both arms hit the same server, so model, hardware and KV budget match. The
direct arm builds messages with Kalypso's own `get_data_prompt` and
`get_prompt`, and uses `VLLMExecutor.DEFAULT_SAMPLING` and the same chat
template arguments. One difference: Kalypso sizes `max_tokens` per row, the
direct arm uses 512, Kalypso's retry ceiling for maps.

### Server

```bash
VLLM_SERVER_DEV_MODE=1 VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_CPU_KVCACHE_SPACE=2 \
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-0.5B-Instruct --dtype float16 \
  --max-model-len 8192 --enable-prefix-caching --port 8003
```

- `VLLM_SERVER_DEV_MODE=1` exposes `/reset_prefix_cache`; the harness resets
  the cache before every trial so no arm inherits another's cached prompts.
- `--max-model-len 8192`: with no history, Kalypso's `SemMap` asks for
  `max_tokens` equal to the prompt length. BioDEX prompts are 1.9k-3.7k
  tokens, so at 4096 vLLM rejects 9 of 10 articles and the query returns 500.
- `--dtype bfloat16` on M2 and later; `float16` on M1.

### Run

```bash
python -m vllm.kalypso.benchmark.eval.run --workload biodex \
  --arms kalypso_per_tuple,vllm_direct --warmup 1 --repeats 3 \
  --out vllm/kalypso/benchmark/eval/results/m1_baseline.jsonl

python -m vllm.kalypso.benchmark.eval.report \
  vllm/kalypso/benchmark/eval/results/m1_baseline.jsonl
```

Run nothing else against the server meanwhile: the counters are server-wide.

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
| `rows_out`, `missing_rows`, `rows_per_s` | From the arm's outputs |
| `outputs` | Per-row text, used for agreement |

`report.py` prints the median and [min-max] per arm, plus agreement: the
share of rows matching the reference arm (`kalypso_per_tuple`) in the same
repeat, and each arm's agreement with its own first repeat. BioDEX has no
per-article labels here (`sample_data/reactions/` is the join vocabulary), so
agreement with per-tuple execution stands in for accuracy, as in the proposal.

The first line of every results file records the git SHA and dirty flag,
model, server command line and start time, platform, CPU and run settings.

## Layout

| File | Role |
|---|---|
| `workloads.py` | Rows and instruction per workload, read the way Kalypso reads them |
| `arms.py` | One async function per execution path |
| `metrics.py` | `/metrics` parsing and deltas, KV and RSS samplers, `measure()` |
| `run.py` | CLI: warmup, interleaved repeats, JSONL output |
| `report.py` | JSONL to markdown table |

Tests: `vllm/kalypso/tests/test_eval_harness.py`.

## Next milestones

- M2: arms for `BlockSemFilter` and `BlockSemJoin` over at least 4 block
  sizes, and the labeled `block_filter` / `block_join` workloads so accuracy
  is measured against labels.
- M3: a Filter1 → Join → Filter2 arm with per-stage timestamps to show stage
  overlap.
