Model `Qwen/Qwen2.5-7B-Instruct` on unknown (Linux-4.19.0-gvisor-x86_64-with-glibc2.39); workload `movie_positive` (filter), 500 rows, 373 positive; 3 repeats after 1 warmup; prefix cache reset per trial: yes; git `db0f47ac7` (dirty).
Server: `python3 -m vllm.entrypoints.cli.main serve Qwen/Qwen2.5-7B-Instruct --host 127.0.0.1 --port 8000 --enable-prefix-caching --no-async-scheduling --api-server-count 1 --max-model-len 32768` (started Wed Oct 28 07:07:22 2026).

| Arm | Latency (s) | LLM requests | Prompt tokens | Gen tokens | Prefix hit rate | Peak KV % | Peak RSS (MiB) | Items/s | Truncated | Mean prefill (s) | Mean decode (s) | Accuracy | Precision | Recall | F1 | Agrees w/ kalypso_per_tuple | Self-agreement |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 4.69 [4.62-4.73] | 500 | 81698 | 1479 | 0.29 | 1.0 | 11430 [11430-11430] | 106.52 [105.75-108.18] | 1 | 0.153 [0.151-0.154] | 0.275 [0.271-0.277] | 0.72 | 1.00 | 0.63 | 0.77 | 100% | 100% |
| `kalypso_block:16` | 3.38 [3.37-3.40] | 32 | 19625 | 2217 | 0.05 | 4.6 | 11430 [11430-11430] | 148.15 [147.02-148.41] | 0 | 0.173 [0.172-0.175] | 2.350 [2.349-2.367] | 0.81 | 0.99 | 0.76 | 0.86 | 85% | 100% |
