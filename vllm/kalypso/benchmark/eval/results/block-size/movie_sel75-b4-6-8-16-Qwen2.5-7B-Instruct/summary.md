Model `Qwen/Qwen2.5-7B-Instruct` on unknown (Linux-4.19.0-gvisor-x86_64-with-glibc2.39); workload `movie_sel75` (filter), 400 rows, 300 positive; 3 repeats after 1 warmup; prefix cache reset per trial: yes; git `3cdb0b42d` (dirty).
Server: `python3 -m vllm.entrypoints.cli.main serve Qwen/Qwen2.5-7B-Instruct --host 127.0.0.1 --port 8000 --enable-prefix-caching --no-async-scheduling --api-server-count 1 --max-model-len 32768` (started Wed Oct 28 20:25:03 2026).

| Arm | Latency (s) | LLM requests | Prompt tokens | Gen tokens | Prefix hit rate | Peak KV % | Peak RSS (MiB) | Items/s | Truncated | Mean prefill (s) | Mean decode (s) | Accuracy | Precision | Recall | F1 | Agrees w/ kalypso_per_tuple | Self-agreement |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 3.81 [3.76-3.88] | 400 | 65163 | 1172 | 0.29 | 1.0 | 11414 [11414-11414] | 105.01 [102.99-106.36] | 0 | 0.155 [0.153-0.158] | 0.273 [0.269-0.277] | 0.71 | 0.99 | 0.61 | 0.76 | 100% | 100% |
| `kalypso_block:4` | 1.99 [1.96-2.00] | 100 | 21461 | 1600 | 0.15 | 4.5 [4.5-4.5] | 11414 [11414-11414] | 200.80 [199.50-203.98] | 0 | 0.161 [0.158-0.163] | 0.956 [0.942-0.967] | 0.84 | 0.98 | 0.80 | 0.88 | 83% | 100% |
| `kalypso_block:6` | 2.02 [2.01-2.03] | 67 | 18722 | 1600 | 0.11 | 4.0 [4.0-4.0] | 11414 [11414-11414] | 198.51 [196.66-198.91] | 0 | 0.161 [0.161-0.164] | 1.106 [1.102-1.116] | 0.84 | 0.99 | 0.80 | 0.89 | 84% | 100% |
| `kalypso_block:8` | 2.18 [2.16-2.19] | 50 | 17311 | 1600 | 0.09 | 3.9 [3.9-3.9] | 11414 [11414-11414] | 183.66 [182.81-185.27] | 0 | 0.163 [0.161-0.165] | 1.319 [1.303-1.319] | 0.85 | 1.00 | 0.80 | 0.89 | 84% | 100% |
| `kalypso_block:16` | 3.09 [3.07-3.09] | 25 | 15461 | 1775 | 0.05 | 3.6 | 11414 [11414-11414] | 129.58 [129.32-130.25] | 0 | 0.180 [0.179-0.181] | 2.285 [2.275-2.286] | 0.83 | 0.99 | 0.78 | 0.88 | 84% | 100% |
