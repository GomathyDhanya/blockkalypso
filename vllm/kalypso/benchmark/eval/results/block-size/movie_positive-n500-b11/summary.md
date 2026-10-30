Model `Qwen/Qwen2.5-7B-Instruct` on unknown (Linux-4.19.0-gvisor-x86_64-with-glibc2.39); workload `movie_positive` (filter), 500 rows, 373 positive; 3 repeats after 1 warmup; prefix cache reset per trial: yes; git `db0f47ac7` (dirty).
Server: `python3 -m vllm.entrypoints.cli.main serve Qwen/Qwen2.5-7B-Instruct --host 127.0.0.1 --port 8000 --enable-prefix-caching --no-async-scheduling --api-server-count 1 --max-model-len 32768` (started Wed Oct 28 06:47:54 2026).

| Arm | Latency (s) | LLM requests | Prompt tokens | Gen tokens | Prefix hit rate | Peak KV % | Peak RSS (MiB) | Items/s | Truncated | Mean prefill (s) | Mean decode (s) | Accuracy | Precision | Recall | F1 | Agrees w/ kalypso_per_tuple | Self-agreement |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 4.77 [4.67-4.77] | 500 | 81698 | 1481 | 0.29 | 1.0 | 11444 [11443-11444] | 104.87 [104.78-106.97] | 1 | 0.155 [0.152-0.155] | 0.279 [0.273-0.279] | 0.72 | 1.00 | 0.63 | 0.77 | 100% | 100% |
| `kalypso_block:11` | 2.83 [2.83-2.88] | 46 | 20688 | 2090 | 0.07 | 4.7 [4.7-4.7] | 11444 [11443-11444] | 176.37 [173.91-176.55] | 0 | 0.169 [0.168-0.173] | 1.791 [1.777-1.812] | 0.79 | 0.98 | 0.73 | 0.84 | 87% | 100% |
