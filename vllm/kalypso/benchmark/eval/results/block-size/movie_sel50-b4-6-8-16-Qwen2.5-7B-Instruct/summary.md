Model `Qwen/Qwen2.5-7B-Instruct` on unknown (Linux-4.19.0-gvisor-x86_64-with-glibc2.39); workload `movie_sel50` (filter), 400 rows, 200 positive; 3 repeats after 1 warmup; prefix cache reset per trial: yes; git `3cdb0b42d` (dirty).
Server: `python3 -m vllm.entrypoints.cli.main serve Qwen/Qwen2.5-7B-Instruct --host 127.0.0.1 --port 8000 --enable-prefix-caching --no-async-scheduling --api-server-count 1 --max-model-len 32768` (started Wed Oct 28 20:21:39 2026).

| Arm | Latency (s) | LLM requests | Prompt tokens | Gen tokens | Prefix hit rate | Peak KV % | Peak RSS (MiB) | Items/s | Truncated | Mean prefill (s) | Mean decode (s) | Accuracy | Precision | Recall | F1 | Agrees w/ kalypso_per_tuple | Self-agreement |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 3.75 [3.71-3.76] | 400 | 65520 | 1182 | 0.29 | 1.0 | 11424 [11424-11424] | 106.64 [106.47-107.88] | 0 | 0.152 [0.151-0.153] | 0.270 [0.267-0.272] | 0.80 | 0.98 | 0.60 | 0.75 | 100% | 100% |
| `kalypso_block:4` | 1.96 [1.93-1.96] | 100 | 21816 | 1600 | 0.15 | 4.5 [4.5-4.5] | 11424 [11424-11424] | 204.50 [204.39-207.25] | 0 | 0.155 [0.153-0.155] | 0.940 [0.919-0.941] | 0.89 | 0.96 | 0.80 | 0.87 | 87% | 100% |
| `kalypso_block:6` | 2.01 [2.00-2.02] | 67 | 19077 | 1600 | 0.11 | 4.1 [4.1-4.1] | 11424 [11424-11424] | 198.81 [198.31-199.90] | 0 | 0.157 [0.154-0.158] | 1.095 [1.095-1.096] | 0.89 [0.89-0.89] | 0.97 [0.97-0.97] | 0.81 [0.81-0.81] | 0.88 [0.88-0.88] | 87% | 100% |
| `kalypso_block:8` | 2.15 [2.14-2.15] | 50 | 17666 | 1600 | 0.09 | 4.0 [4.0-4.0] | 11424 [11424-11424] | 185.79 [185.62-187.00] | 0 | 0.160 [0.159-0.161] | 1.297 [1.291-1.299] | 0.87 | 0.96 | 0.77 | 0.86 | 89% | 100% |
| `kalypso_block:16` | 3.04 [3.02-3.07] | 25 | 15816 | 1775 | 0.05 | 3.7 | 11424 [11424-11424] | 131.41 [130.25-132.32] | 0 | 0.178 [0.177-0.180] | 2.240 [2.217-2.247] | 0.85 | 0.96 | 0.72 | 0.83 | 90% | 100% |
