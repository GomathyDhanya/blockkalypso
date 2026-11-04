Model `Qwen/Qwen2.5-7B-Instruct` on unknown (Linux-4.19.0-gvisor-x86_64-with-glibc2.39); workload `cnli_sel75` (filter), 300 rows, 223 positive; 3 repeats after 1 warmup; prefix cache reset per trial: yes; git `5891dbbb5` (dirty).
Server: `python3 -m vllm.entrypoints.cli.main serve Qwen/Qwen2.5-7B-Instruct --host 127.0.0.1 --port 8000 --enable-prefix-caching --no-async-scheduling --api-server-count 1 --max-model-len 32768 --max-num-seqs 256 --max-num-batched-tokens 2048` (started Tue Nov  3 22:06:23 2026).

| Arm | Latency (s) | LLM requests | Block calls | Fallback calls | Prompt tokens | Gen tokens | Prefix hit rate | Peak KV % | Peak running | Peak waiting | Preemptions | Peak RSS (MiB) | Items/s | Truncated | Mean prefill (s) | Mean decode (s) | Accuracy | Precision | Recall | F1 | Agrees w/ kalypso_per_tuple | Self-agreement |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 39.73 [39.65-39.80] | 300 | n/a | n/a | 486825 | 898 | 0.04 | 1.5 | 6 | 271 | 0 | 11519 [11519-11520] | 7.55 [7.54-7.57] | 0 | 0.290 [0.289-0.290] | 0.327 [0.327-0.327] | 0.71 | 0.85 | 0.73 | 0.79 | 100% | 100% |
| `kalypso_block:4` | 42.92 [42.56-42.94] | 75 | 75 | 0 | 451607 | 1200 | 0.01 | 8.4 | 8 | 70 [70-71] | 0 | 11519 [11519-11520] | 6.99 [6.99-7.05] | 0 | 0.707 [0.707-0.708] | 2.619 [2.618-2.622] | 0.60 | 0.85 | 0.56 | 0.68 | 69% | 100% |
| `kalypso_block:16` | 58.29 [57.74-58.31] | 19 | 19 | 0 | 446678 | 1329 | 0.00 | 37.9 [37.6-37.9] | 8 | 17 | 0 | 11519 [11519-11520] | 5.15 [5.14-5.20] | 0 | 3.039 [3.038-3.040] | 14.492 [14.482-14.497] | 0.53 | 0.82 | 0.48 | 0.60 | 61% | 100% |
| `kalypso_block:32` | failed in all 3 trials | | | | | | | | | | | | | | | | | | | | | |
| `kalypso_block:2` | 40.55 [40.50-40.55] | 150 | 150 | 0 | 458432 | 1200 | 0.02 | 4.3 [4.2-4.3] | 8 | 141 | 0 | 11520 [11519-11520] | 7.40 [7.40-7.41] | 0 | 0.417 [0.417-0.417] | 1.167 [1.167-1.169] | 0.53 | 0.88 | 0.43 | 0.57 | 67% | 100% |

- `kalypso_block:32` failed 3 trial(s): ArmError: /v1/semantic/query returned 400: {"error":{"message":"This model's maximum context length is 32768 tokens. However, you requested 208 output tokens and your prompt contains at least 32561 input tokens, for a total of at least 32769 tokens. Please reduce the length of the input prompt or th
