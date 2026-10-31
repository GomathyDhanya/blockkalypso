Model `Qwen/Qwen2.5-7B-Instruct` on unknown (Linux-4.19.0-gvisor-x86_64-with-glibc2.39); workload `pipe_filter_map_map` (map), 373 rows; 3 repeats after 1 warmup; prefix cache reset per trial: yes; git `2db189318` (dirty).
Server: `python3 -m vllm.entrypoints.cli.main serve Qwen/Qwen2.5-7B-Instruct --host 127.0.0.1 --port 8000 --enable-prefix-caching --no-async-scheduling --api-server-count 1 --max-model-len 32768 --max-num-seqs 256 --max-num-batched-tokens 2048` (started Sat Oct 31 16:40:36 2026).

| Arm | Latency (s) | LLM requests | Block calls | Fallback calls | Prompt tokens | Gen tokens | Prefix hit rate | Peak KV % | Peak running | Peak waiting | Preemptions | Peak RSS (MiB) | Items/s | Truncated | Mean prefill (s) | Mean decode (s) | Agrees w/ kalypso_per_tuple | Self-agreement |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 5.24 [5.20-5.27] | 373 | n/a | n/a | 59412 | 7401 | 0.30 | 7.2 | 251 [247-251] | 317 | 0 | 11408 [11408-11408] | 71.17 [70.72-71.70] | 0 | 0.155 [0.153-0.157] | 1.854 [1.835-1.863] | 100% | 100% |
| `kalypso_block:4` | 6.79 [6.79-6.80] | 94 | 94 | 0 | 26782 | 11718 | 0.22 | 6.5 [6.5-6.5] | 94 | 64 | 0 | 11408 [11408-11408] | 54.91 [54.84-54.97] | 0 | 0.165 [0.163-0.165] | 4.260 [4.256-4.265] | 0% | 100% |
| `kalypso_block:8` | 13.32 [13.26-13.35] | 55 | 55 | 0 | 22545 | 11191 | 0.15 | 5.8 [5.8-5.8] | 47 | 40 | 0 | 11408 [11408-11408] | 28.01 [27.93-28.12] | 0 | 0.154 [0.150-0.154] | 6.085 [6.069-6.104] | 2% | 100% |
