Model `Qwen/Qwen2.5-7B-Instruct` on unknown (Linux-4.19.0-gvisor-x86_64-with-glibc2.39); workload `movie_positive` (filter), 500 rows, 373 positive; 3 repeats after 1 warmup; prefix cache reset per trial: yes; git `db0f47ac7` (dirty).
Server: `python3 -m vllm.entrypoints.cli.main serve Qwen/Qwen2.5-7B-Instruct --host 127.0.0.1 --port 8000 --enable-prefix-caching --no-async-scheduling --api-server-count 1 --max-model-len 32768` (started Wed Oct 28 07:40:30 2026).

| Arm | Latency (s) | LLM requests | Prompt tokens | Gen tokens | Prefix hit rate | Peak KV % | Peak RSS (MiB) | Items/s | Truncated | Mean prefill (s) | Mean decode (s) | Accuracy | Precision | Recall | F1 | Agrees w/ kalypso_per_tuple | Self-agreement |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 4.55 [4.50-4.62] | 500 | 81698 | 1481 | 0.29 | 1.0 | 11429 [11429-11429] | 109.84 [108.22-111.04] | 1 | 0.152 [0.151-0.155] | 0.274 [0.271-0.279] | 0.72 | 1.00 | 0.63 | 0.77 | 100% | 100% |
| `kalypso_block:4` | 2.28 [2.27-2.30] | 125 | 27065 | 2000 | 0.15 | 5.6 [5.6-5.6] | 11429 [11429-11429] | 219.11 [217.49-219.97] | 0 | 0.156 [0.155-0.157] | 1.077 [1.075-1.085] | 0.82 | 0.99 | 0.77 | 0.87 | 85% | 100% |
