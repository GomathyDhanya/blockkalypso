Model `Qwen/Qwen2.5-7B-Instruct` on unknown (Linux-4.19.0-gvisor-x86_64-with-glibc2.39); workload `movie_sel90` (filter), 400 rows, 360 positive; 3 repeats after 1 warmup; prefix cache reset per trial: yes; git `3cdb0b42d` (dirty).
Server: `python3 -m vllm.entrypoints.cli.main serve Qwen/Qwen2.5-7B-Instruct --host 127.0.0.1 --port 8000 --enable-prefix-caching --no-async-scheduling --api-server-count 1 --max-model-len 32768` (started Wed Oct 28 20:28:17 2026).

| Arm | Latency (s) | LLM requests | Prompt tokens | Gen tokens | Prefix hit rate | Peak KV % | Peak RSS (MiB) | Items/s | Truncated | Mean prefill (s) | Mean decode (s) | Accuracy | Precision | Recall | F1 | Agrees w/ kalypso_per_tuple | Self-agreement |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 3.75 [3.70-3.79] | 400 | 65195 | 1176 | 0.29 | 1.0 | 11419 [11419-11419] | 106.75 [105.57-108.22] | 0 | 0.157 [0.154-0.159] | 0.277 [0.271-0.280] | 0.68 | 1.00 | 0.64 | 0.78 | 100% | 100% |
| `kalypso_block:4` | 1.93 [1.91-1.95] | 100 | 21493 | 1600 | 0.15 | 4.5 [4.5-4.5] | 11419 [11419-11419] | 207.36 [204.71-208.99] | 0 | 0.160 [0.159-0.164] | 0.938 [0.928-0.951] | 0.83 | 1.00 | 0.82 | 0.90 | 82% | 100% |
| `kalypso_block:6` | 2.00 [1.98-2.01] | 67 | 18754 | 1600 | 0.11 | 4.0 | 11419 [11419-11419] | 200.50 [198.61-202.12] | 0 | 0.163 [0.161-0.165] | 1.090 [1.085-1.101] | 0.82 | 0.99 | 0.80 | 0.89 | 82% | 100% |
| `kalypso_block:8` | 2.14 [2.12-2.14] | 50 | 17343 | 1600 | 0.09 | 3.9 [3.9-3.9] | 11419 [11419-11419] | 187.18 [186.65-188.41] | 0 | 0.165 [0.163-0.166] | 1.303 [1.293-1.304] | 0.81 | 0.99 | 0.79 | 0.88 | 81% | 100% |
| `kalypso_block:16` | 3.02 [3.00-3.02] | 25 | 15493 | 1775 | 0.05 | 3.7 | 11419 [11419-11419] | 132.54 [132.23-133.16] | 0 | 0.180 [0.179-0.182] | 2.236 [2.236-2.237] | 0.80 | 1.00 | 0.78 | 0.87 | 80% | 100% |
