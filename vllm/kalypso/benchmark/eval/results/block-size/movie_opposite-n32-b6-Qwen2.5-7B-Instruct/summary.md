Model `Qwen/Qwen2.5-7B-Instruct` on unknown (Linux-4.19.0-gvisor-x86_64-with-glibc2.39); workload `movie_opposite` (join), 32 x 32 = 1024 pairs, 510 positive; 3 repeats after 1 warmup; prefix cache reset per trial: yes; git `2ac9745e0` (dirty).
Server: `python3 -m vllm.entrypoints.cli.main serve Qwen/Qwen2.5-7B-Instruct --host 127.0.0.1 --port 8000 --enable-prefix-caching --no-async-scheduling --api-server-count 1 --max-model-len 32768` (started Fri Oct 30 15:53:32 2026).

| Arm | Latency (s) | LLM requests | Block calls | Fallback calls | Prompt tokens | Gen tokens | Prefix hit rate | Peak KV % | Peak RSS (MiB) | Items/s | Truncated | Mean prefill (s) | Mean decode (s) | Accuracy | Precision | Recall | F1 | Agrees w/ kalypso_per_tuple | Self-agreement |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 11.05 [10.99-11.24] | 1024 | n/a | n/a | 238976 | 2293 | 0.42 | 0.9 [0.8-0.9] | 11461 [11461-11462] | 92.64 [91.13-93.17] | 0 | 0.157 [0.155-0.159] | 0.180 [0.178-0.184] | 0.59 [0.59-0.59] | 0.58 [0.58-0.58] | 0.62 [0.62-0.63] | 0.60 [0.60-0.60] | 100% | 100% |
| `kalypso_block:6` | 8.28 [8.20-8.33] | 171 | 171 | 0 | 102654 | 4096 | 0.13 | 10.8 [10.7-10.8] | 11461 [11461-11462] | 123.75 [122.87-124.94] | 0 | 0.194 [0.192-0.196] | 2.838 [2.812-2.862] | 0.59 | 0.57 | 0.69 | 0.63 | 66% | 100% |
