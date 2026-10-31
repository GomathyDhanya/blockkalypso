Model `Qwen/Qwen2.5-7B-Instruct` on unknown (Linux-4.19.0-gvisor-x86_64-with-glibc2.39); workload `pipe_filter_map_map` (map), 373 rows; 3 repeats after 1 warmup; prefix cache reset per trial: yes; git `2db189318` (dirty).
Server: `python3 -m vllm.entrypoints.cli.main serve Qwen/Qwen2.5-7B-Instruct --host 127.0.0.1 --port 8000 --enable-prefix-caching --no-async-scheduling --api-server-count 1 --max-model-len 32768 --max-num-seqs 256 --max-num-batched-tokens 2048` (started Sat Oct 31 16:40:35 2026).

| Arm | Latency (s) | LLM requests | Block calls | Fallback calls | Prompt tokens | Gen tokens | Prefix hit rate | Peak KV % | Peak running | Peak waiting | Preemptions | Peak RSS (MiB) | Items/s | Truncated | Mean prefill (s) | Mean decode (s) | Agrees w/ kalypso_per_tuple | Self-agreement |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 5.29 [5.29-5.30] | 373 | n/a | n/a | 59412 | 7401 | 0.30 | 7.2 | 252 | 317 | 0 | 11422 [11422-11422] | 70.48 [70.35-70.54] | 0 | 0.149 [0.149-0.150] | 1.804 [1.802-1.810] | 100% | 100% |
