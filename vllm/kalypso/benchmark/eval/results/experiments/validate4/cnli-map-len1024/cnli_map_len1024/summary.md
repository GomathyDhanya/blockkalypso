Model `Qwen/Qwen2.5-7B-Instruct` on unknown (Linux-4.19.0-gvisor-x86_64-with-glibc2.39); workload `cnli_map_len1024` (map), 400 rows; 3 repeats after 1 warmup; prefix cache reset per trial: yes; git `6d3c9b6c1`.
Server: `python3 -m vllm.entrypoints.cli.main serve Qwen/Qwen2.5-7B-Instruct --host 127.0.0.1 --port 8000 --enable-prefix-caching --no-async-scheduling --api-server-count 1 --max-model-len 32768 --max-num-seqs 256 --max-num-batched-tokens 2048` (started Tue Nov  3 23:57:27 2026).

| Arm | Latency (s) | LLM requests | Block calls | Fallback calls | Prompt tokens | Gen tokens | Prefix hit rate | Peak KV % | Peak running | Peak waiting | Preemptions | Peak RSS (MiB) | Items/s | Truncated | Mean prefill (s) | Mean decode (s) | Agrees w/ kalypso_per_tuple | Self-agreement |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 41.90 [40.48-42.12] | 412 | n/a | n/a | 427695 | 11720 | 0.07 | 13.5 [13.2-13.5] | 59 | 244 [242-244] | 0 | 11464 [11464-11464] | 9.55 [9.50-9.88] | 12 | 0.246 [0.236-0.248] | 4.117 [3.946-4.148] | 100% | 100% |
| `kalypso_block:2` | 52.67 [52.55-54.16] | 312 | 312 | 0 | 518354 | 18701 | 0.05 | 39.1 | 91 | 141 [140-141] | 0 | 11464 [11464-11464] | 7.59 [7.38-7.61] | 12 | 0.313 [0.312-0.323] | 9.851 [9.838-10.165] | 22% | 100% |
| `kalypso_block:4` | 58.77 [58.67-61.08] | 156 | 156 | 0 | 447907 | 16575 [16575-16607] | 0.03 | 68.8 | 82 | 72 | 0 | 11464 [11464-11464] | 6.81 [6.55-6.82] | 7 | 0.452 [0.451-0.467] | 16.488 [16.452-16.779] | 10% | 99% |
| `kalypso_block:8` | 76.83 [76.54-76.91] | 122 | 122 | 0 | 458862 | 17669 | 0.03 | 66.4 [66.2-66.4] | 41 | 36 | 0 | 11464 [11464-11464] | 5.21 [5.20-5.23] | 11 | 0.487 [0.484-0.489] | 13.786 [13.610-13.799] | 12% | 100% |
