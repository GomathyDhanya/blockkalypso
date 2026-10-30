Model `Qwen/Qwen2.5-7B-Instruct` on unknown (Linux-4.19.0-gvisor-x86_64-with-glibc2.39); workload `movie_positive` (filter), 500 rows, 373 positive; 3 repeats after 1 warmup; prefix cache reset per trial: yes; git `db0f47ac7` (dirty).
Server: `python3 -m vllm.entrypoints.cli.main serve Qwen/Qwen2.5-7B-Instruct --host 127.0.0.1 --port 8000 --enable-prefix-caching --no-async-scheduling --api-server-count 1 --max-model-len 32768` (started Wed Oct 28 17:26:49 2026).

| Arm | Latency (s) | LLM requests | Prompt tokens | Gen tokens | Prefix hit rate | Peak KV % | Peak RSS (MiB) | Items/s | Truncated | Mean prefill (s) | Mean decode (s) | Accuracy | Precision | Recall | F1 | Agrees w/ kalypso_per_tuple | Self-agreement |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 4.71 [4.68-4.80] | 500 | 81698 | 1479 | 0.29 | 1.0 [1.0-1.0] | 11426 [11425-11426] | 106.09 [104.14-106.75] | 1 | 0.153 [0.152-0.154] | 0.274 [0.273-0.276] | 0.72 | 1.00 | 0.63 | 0.77 | 100% | 100% |
| `kalypso_block:20` | 3.91 [3.87-3.95] | 25 | 19090 | 2275 | 0.04 | 4.6 [4.6-4.6] | 11426 [11426-11426] | 127.91 [126.49-129.27] | 0 | 0.187 [0.184-0.187] | 2.936 [2.914-2.978] | 0.83 | 0.99 | 0.79 | 0.87 | 83% | 100% |
