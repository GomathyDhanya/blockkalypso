Model `Qwen/Qwen2.5-7B-Instruct` on unknown (Linux-4.19.0-gvisor-x86_64-with-glibc2.39); workload `movie_map` (map), 100 rows; 3 repeats after 1 warmup; prefix cache reset per trial: yes; git `2db189318` (dirty).
Server: `python3 -m vllm.entrypoints.cli.main serve Qwen/Qwen2.5-7B-Instruct --host 127.0.0.1 --port 8000 --enable-prefix-caching --no-async-scheduling --api-server-count 1 --max-model-len 32768 --max-num-seqs 256 --max-num-batched-tokens 2048` (started Sat Oct 31 16:22:29 2026).

| Arm | Latency (s) | LLM requests | Block calls | Fallback calls | Prompt tokens | Gen tokens | Prefix hit rate | Peak KV % | Peak running | Peak waiting | Preemptions | Peak RSS (MiB) | Items/s | Truncated | Mean prefill (s) | Mean decode (s) | Agrees w/ kalypso_per_tuple | Self-agreement |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 2.65 [2.60-2.66] | 100 | n/a | n/a | 15829 | 2101 | 0.30 | 2.6 [2.6-2.7] | 92 [92-96] | 43 [43-48] | 0 | 11424 [11424-11424] | 37.66 [37.59-38.42] | 0 | 0.144 [0.142-0.146] | 0.850 [0.849-0.857] | 100% | 100% |
| `kalypso_block:2` | 3.40 [3.32-3.40] | 50 | 50 | 0 | 9975 | 3354 | 0.31 | 2.1 [2.1-2.1] | 50 | 29 [8-33] | 0 | 11424 [11424-11424] | 29.44 [29.38-30.14] | 0 | 0.139 [0.137-0.142] | 2.021 [1.993-2.041] | 0% | 100% |
| `kalypso_block:4` | 5.09 [4.96-5.17] | 25 | 25 | 0 | 7050 | 2962 | 0.22 | 1.7 [1.7-1.7] | 25 | 14 [2-14] | 0 | 11424 [11424-11424] | 19.66 [19.32-20.18] | 0 | 0.137 [0.136-0.138] | 3.205 [3.147-3.277] | 0% | 100% |
| `kalypso_block:8` | 7.85 [7.80-7.92] | 15 | 15 | 0 | 5960 | 2889 | 0.14 | 1.5 | 13 | 6 | 0 | 11424 [11424-11424] | 12.74 [12.62-12.82] | 0 | 0.125 [0.123-0.127] | 4.549 [4.518-4.556] | 1% | 100% |
