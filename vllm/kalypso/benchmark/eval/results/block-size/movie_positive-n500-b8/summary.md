Model `Qwen/Qwen2.5-7B-Instruct` on unknown (Linux-4.19.0-gvisor-x86_64-with-glibc2.39); workload `movie_positive` (filter), 500 rows, 373 positive; 3 repeats after 1 warmup; prefix cache reset per trial: yes; git `db0f47ac7` (dirty).
Server: `python3 -m vllm.entrypoints.cli.main serve Qwen/Qwen2.5-7B-Instruct --host 127.0.0.1 --port 8000 --enable-prefix-caching --no-async-scheduling --api-server-count 1 --max-model-len 32768` (started Wed Oct 28 13:55:59 2026).

| Arm | Latency (s) | LLM requests | Prompt tokens | Gen tokens | Prefix hit rate | Peak KV % | Peak RSS (MiB) | Items/s | Truncated | Mean prefill (s) | Mean decode (s) | Accuracy | Precision | Recall | F1 | Agrees w/ kalypso_per_tuple | Self-agreement |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 4.56 [4.53-4.56] | 500 | 81698 | 1479 | 0.29 | 1.0 | 11431 [11431-11431] | 109.60 [109.55-110.30] | 1 | 0.150 [0.149-0.151] | 0.270 [0.268-0.271] | 0.72 | 1.00 | 0.63 | 0.77 | 100% | 100% |
| `kalypso_block:8` | 2.43 [2.43-2.44] | 63 | 21919 | 2000 | 0.09 | 4.8 [4.8-4.9] | 11431 [11431-11432] | 205.59 [204.50-205.76] | 0 | 0.160 [0.160-0.162] | 1.392 [1.389-1.394] | 0.83 | 0.99 | 0.78 | 0.87 | 85% | 100% |
