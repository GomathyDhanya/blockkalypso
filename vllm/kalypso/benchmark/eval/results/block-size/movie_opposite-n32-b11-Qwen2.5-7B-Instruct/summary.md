Model `Qwen/Qwen2.5-7B-Instruct` on unknown (Linux-4.19.0-gvisor-x86_64-with-glibc2.39); workload `movie_opposite` (join), 32 x 32 = 1024 pairs, 510 positive; 3 repeats after 1 warmup; prefix cache reset per trial: yes; git `d16192471` (dirty).
Server: `python3 -m vllm.entrypoints.cli.main serve Qwen/Qwen2.5-7B-Instruct --host 127.0.0.1 --port 8000 --enable-prefix-caching --no-async-scheduling --api-server-count 1 --max-model-len 32768` (started Sat Oct 31 01:54:38 2026).

| Arm | Latency (s) | LLM requests | Block calls | Fallback calls | Prompt tokens | Gen tokens | Prefix hit rate | Peak KV % | Peak RSS (MiB) | Items/s | Truncated | Mean prefill (s) | Mean decode (s) | Accuracy | Precision | Recall | F1 | Agrees w/ kalypso_per_tuple | Self-agreement |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 10.73 [10.68-10.76] | 1024 | n/a | n/a | 238976 | 2294 | 0.42 | 0.8 [0.7-0.8] | 11436 [11436-11437] | 95.42 [95.15-95.86] | 0 | 0.154 [0.154-0.156] | 0.177 [0.177-0.180] | 0.59 | 0.58 | 0.62 | 0.60 | 100% | 100% |
| `kalypso_block:11` | 8.52 [8.50-8.62] | 94 | 93 | 1 | 96060 | 4280 | 0.07 | 18.1 [18.1-18.2] | 11436 [11436-11437] | 120.14 [118.77-120.51] | 0 | 0.225 [0.222-0.228] | 4.039 [4.003-4.094] | 0.56 | 0.58 | 0.46 | 0.51 | 62% | 100% |
