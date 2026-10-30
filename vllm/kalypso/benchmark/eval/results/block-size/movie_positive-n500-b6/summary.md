Model `Qwen/Qwen2.5-7B-Instruct` on unknown (Linux-4.19.0-gvisor-x86_64-with-glibc2.39); workload `movie_positive` (filter), 500 rows, 373 positive; 3 repeats after 1 warmup; prefix cache reset per trial: yes; git `db0f47ac7` (dirty).
Server: `python3 -m vllm.entrypoints.cli.main serve Qwen/Qwen2.5-7B-Instruct --host 127.0.0.1 --port 8000 --enable-prefix-caching --no-async-scheduling --api-server-count 1 --max-model-len 32768` (started Wed Oct 28 07:22:40 2026).

| Arm | Latency (s) | LLM requests | Prompt tokens | Gen tokens | Prefix hit rate | Peak KV % | Peak RSS (MiB) | Items/s | Truncated | Mean prefill (s) | Mean decode (s) | Accuracy | Precision | Recall | F1 | Agrees w/ kalypso_per_tuple | Self-agreement |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 4.85 [4.79-4.86] | 500 | 81698 | 1479 [1479-1480] | 0.29 | 1.0 | 11423 [11422-11423] | 103.01 [102.94-104.36] | 1 | 0.158 [0.156-0.158] | 0.285 [0.281-0.285] | 0.72 | 1.00 | 0.63 | 0.77 | 100% | 100% |
| `kalypso_block:6` | 2.43 [2.42-2.44] | 84 | 23662 | 2000 | 0.11 | 5.0 | 11423 [11423-11423] | 205.42 [205.34-206.53] | 0 | 0.165 [0.163-0.165] | 1.264 [1.257-1.265] | 0.82 | 0.99 | 0.77 | 0.87 | 85% | 100% |
