Model `Qwen/Qwen2.5-7B-Instruct` on unknown (Linux-4.19.0-gvisor-x86_64-with-glibc2.39); workload `movie_map` (map), 500 rows; 3 repeats after 1 warmup; prefix cache reset per trial: yes; git `2db189318` (dirty).
Server: `python3 -m vllm.entrypoints.cli.main serve Qwen/Qwen2.5-7B-Instruct --host 127.0.0.1 --port 8000 --enable-prefix-caching --no-async-scheduling --api-server-count 1 --max-model-len 32768 --max-num-seqs 256 --max-num-batched-tokens 2048` (started Sat Oct 31 16:22:27 2026).

| Arm | Latency (s) | LLM requests | Block calls | Fallback calls | Prompt tokens | Gen tokens | Prefix hit rate | Peak KV % | Peak running | Peak waiting | Preemptions | Peak RSS (MiB) | Items/s | Truncated | Mean prefill (s) | Mean decode (s) | Agrees w/ kalypso_per_tuple | Self-agreement |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 7.36 [7.30-7.38] | 500 | n/a | n/a | 79698 | 9917 | 0.30 | 7.2 [7.2-7.2] | 249 [249-250] | 443 | 0 | 11428 [11428-11428] | 67.94 [67.70-68.45] | 0 | 0.152 [0.151-0.153] | 2.045 [2.022-2.052] | 100% | 100% |
| `kalypso_block:2` | 6.43 [6.39-6.44] | 250 | 250 | 0 | 50468 | 17133 | 0.32 | 10.1 [10.1-10.2] | 250 | 204 | 0 | 11428 [11428-11428] | 77.74 [77.59-78.26] | 0 | 0.164 [0.163-0.165] | 3.420 [3.402-3.430] | 0% | 100% |
| `kalypso_block:4` | 7.93 [7.89-7.93] | 125 | 125 | 0 | 35843 | 15654 | 0.22 | 8.7 [8.7-8.7] | 125 | 96 | 0 | 11428 [11428-11428] | 63.08 [63.03-63.38] | 0 | 0.168 [0.165-0.168] | 4.637 [4.614-4.640] | 0% | 100% |
| `kalypso_block:6` | 9.47 [9.46-9.49] | 87 | 87 | 0 | 31544 | 14707 | 0.17 | 8.2 [8.2-8.2] | 84 | 62 [62-64] | 0 | 11428 [11428-11428] | 52.79 [52.70-52.84] | 0 | 0.165 [0.165-0.167] | 5.716 [5.699-5.735] | 1% | 100% |
| `kalypso_block:8` | 10.74 [10.74-10.81] | 65 | 65 | 0 | 28923 | 13716 | 0.14 | 7.8 [7.8-7.8] | 63 | 56 | 0 | 11428 [11428-11428] | 46.53 [46.27-46.57] | 0 | 0.172 [0.171-0.173] | 6.797 [6.792-6.838] | 0% | 100% |
| `kalypso_block:16` | 19.80 [19.64-19.80] | 51 | 51 | 0 | 28003 | 13603 | 0.11 | 7.4 | 32 | 28 | 0 | 11428 [11428-11428] | 25.25 [25.25-25.46] | 3 | 0.136 [0.136-0.137] | 7.662 [7.616-7.665] | 3% | 100% |
