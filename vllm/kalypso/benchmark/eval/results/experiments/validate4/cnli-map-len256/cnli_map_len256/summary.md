Model `Qwen/Qwen2.5-7B-Instruct` on unknown (Linux-4.19.0-gvisor-x86_64-with-glibc2.39); workload `cnli_map_len256` (map), 400 rows; 3 repeats after 1 warmup; prefix cache reset per trial: yes; git `6d3c9b6c1`.
Server: `python3 -m vllm.entrypoints.cli.main serve Qwen/Qwen2.5-7B-Instruct --host 127.0.0.1 --port 8000 --enable-prefix-caching --no-async-scheduling --api-server-count 1 --max-model-len 32768 --max-num-seqs 256 --max-num-batched-tokens 2048` (started Tue Nov  3 23:57:21 2026).

| Arm | Latency (s) | LLM requests | Block calls | Fallback calls | Prompt tokens | Gen tokens | Prefix hit rate | Peak KV % | Peak running | Peak waiting | Preemptions | Peak RSS (MiB) | Items/s | Truncated | Mean prefill (s) | Mean decode (s) | Agrees w/ kalypso_per_tuple | Self-agreement |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 13.06 [12.79-13.10] | 400 | n/a | n/a | 147935 | 9784 | 0.13 | 11.1 [11.1-11.2] | 145 [145-147] | 379 [373-379] | 0 | 11427 [11427-11427] | 30.62 [30.52-31.28] | 0 | 0.186 [0.182-0.187] | 3.150 [3.097-3.177] | 100% | 100% |
| `kalypso_block:2` | 17.19 [16.97-17.30] | 219 | 219 | 0 | 133594 | 14236 | 0.10 | 26.2 [26.1-26.2] | 194 [193-194] | 188 | 0 | 11427 [11427-11427] | 23.27 [23.12-23.57] | 1 | 0.199 [0.195-0.201] | 6.169 [6.079-6.247] | 4% | 100% |
| `kalypso_block:4` | 16.23 [16.13-16.34] | 119 | 119 | 0 | 122087 | 12491 | 0.07 | 25.4 [25.3-25.4] | 100 | 95 | 0 | 11427 [11427-11427] | 24.64 [24.49-24.79] | 4 | 0.218 [0.216-0.219] | 6.750 [6.723-6.753] | 3% | 100% |
| `kalypso_block:8` | 20.59 [20.51-20.72] | 71 | 71 | 0 | 117227 | 11973 | 0.05 | 24.7 [24.7-24.8] | 50 | 47 | 0 | 11427 [11427-11427] | 19.42 [19.31-19.51] | 5 | 0.250 [0.249-0.251] | 8.169 [8.149-8.184] | 3% | 100% |
