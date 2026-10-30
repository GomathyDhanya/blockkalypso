Model `Qwen/Qwen2.5-7B-Instruct` on unknown (Linux-4.19.0-gvisor-x86_64-with-glibc2.39); workload `movie_sel25` (filter), 400 rows, 100 positive; 3 repeats after 1 warmup; prefix cache reset per trial: yes; git `3cdb0b42d` (dirty).
Server: `python3 -m vllm.entrypoints.cli.main serve Qwen/Qwen2.5-7B-Instruct --host 127.0.0.1 --port 8000 --enable-prefix-caching --no-async-scheduling --api-server-count 1 --max-model-len 32768` (started Wed Oct 28 20:18:19 2026).

| Arm | Latency (s) | LLM requests | Prompt tokens | Gen tokens | Prefix hit rate | Peak KV % | Peak RSS (MiB) | Items/s | Truncated | Mean prefill (s) | Mean decode (s) | Accuracy | Precision | Recall | F1 | Agrees w/ kalypso_per_tuple | Self-agreement |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 3.76 [3.70-3.78] | 400 | 65450 | 1188 | 0.29 | 1.0 | 11429 [11429-11429] | 106.50 [105.90-108.22] | 0 | 0.153 [0.150-0.154] | 0.273 [0.268-0.274] | 0.89 | 0.91 | 0.62 | 0.74 | 100% | 100% |
| `kalypso_block:4` | 1.95 [1.94-1.95] | 100 | 21749 | 1600 | 0.15 | 4.5 [4.5-4.5] | 11429 [11429-11429] | 205.13 [204.81-205.87] | 0 | 0.155 [0.155-0.156] | 0.934 [0.934-0.941] | 0.94 | 0.92 | 0.82 | 0.87 | 93% | 100% |
| `kalypso_block:6` | 2.00 [2.00-2.02] | 67 | 19010 | 1600 | 0.11 | 4.1 [4.1-4.1] | 11429 [11429-11429] | 200.10 [198.51-200.40] | 0 | 0.158 [0.157-0.158] | 1.087 [1.085-1.095] | 0.94 | 0.89 | 0.86 | 0.87 | 92% | 100% |
| `kalypso_block:8` | 2.15 [2.14-2.16] | 50 | 17599 | 1600 | 0.09 | 3.9 [3.9-4.0] | 11429 [11429-11429] | 186.05 [185.36-186.92] | 0 | 0.159 [0.156-0.161] | 1.306 [1.300-1.307] | 0.93 | 0.94 | 0.78 | 0.85 | 92% | 100% |
| `kalypso_block:16` | 3.05 [3.05-3.07] | 25 | 15749 | 1775 | 0.05 | 3.7 | 11429 [11429-11429] | 131.06 [130.25-131.19] | 0 | 0.176 [0.176-0.177] | 2.253 [2.247-2.265] | 0.91 | 0.93 | 0.69 | 0.79 | 93% | 100% |
