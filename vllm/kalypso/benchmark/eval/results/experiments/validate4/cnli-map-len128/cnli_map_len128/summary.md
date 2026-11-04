Model `Qwen/Qwen2.5-7B-Instruct` on unknown (Linux-4.19.0-gvisor-x86_64-with-glibc2.39); workload `cnli_map_len128` (map), 400 rows; 3 repeats after 1 warmup; prefix cache reset per trial: yes; git `6d3c9b6c1`.
Server: `python3 -m vllm.entrypoints.cli.main serve Qwen/Qwen2.5-7B-Instruct --host 127.0.0.1 --port 8000 --enable-prefix-caching --no-async-scheduling --api-server-count 1 --max-model-len 32768 --max-num-seqs 256 --max-num-batched-tokens 2048` (started Tue Nov  3 23:57:20 2026).

| Arm | Latency (s) | LLM requests | Block calls | Fallback calls | Prompt tokens | Gen tokens | Prefix hit rate | Peak KV % | Peak running | Peak waiting | Preemptions | Peak RSS (MiB) | Items/s | Truncated | Mean prefill (s) | Mean decode (s) | Agrees w/ kalypso_per_tuple | Self-agreement |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 10.10 [9.88-10.15] | 400 | n/a | n/a | 100057 | 9251 | 0.19 | 10.4 [10.3-10.4] | 211 [209-211] | 368 [358-368] | 0 | 11427 [11427-11427] | 39.61 [39.42-40.48] | 0 | 0.185 [0.179-0.186] | 2.877 [2.788-2.889] | 100% | 100% |
| `kalypso_block:2` | 9.79 [9.61-9.80] | 211 | 211 | 0 | 80581 | 12718 | 0.17 | 15.9 | 200 | 180 [180-186] | 0 | 11427 [11427-11427] | 40.87 [40.81-41.64] | 4 | 0.187 [0.182-0.188] | 4.305 [4.207-4.307] | 1% | 100% |
| `kalypso_block:4` | 10.71 [10.57-10.73] | 113 | 113 | 0 | 69642 | 10965 | 0.11 | 14.9 [14.8-14.9] | 100 | 92 | 0 | 11427 [11427-11427] | 37.35 [37.27-37.86] | 3 | 0.194 [0.189-0.195] | 4.910 [4.844-4.921] | 2% | 100% |
| `kalypso_block:8` | 14.84 [14.75-14.89] | 72 | 72 | 0 | 66298 | 9992 | 0.06 | 13.6 [13.6-13.7] | 50 | 46 [45-46] | 0 | 11427 [11427-11427] | 26.95 [26.86-27.12] | 0 | 0.191 [0.187-0.192] | 5.602 [5.577-5.628] | 5% | 100% |
