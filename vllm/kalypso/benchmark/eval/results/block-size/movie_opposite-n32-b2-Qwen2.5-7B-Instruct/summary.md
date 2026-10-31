Model `Qwen/Qwen2.5-7B-Instruct` on unknown (Linux-4.19.0-gvisor-x86_64-with-glibc2.39); workload `movie_opposite` (join), 32 x 32 = 1024 pairs, 510 positive; 3 repeats after 1 warmup; prefix cache reset per trial: yes; git `44f36173b`.
Server: `python3 -m vllm.entrypoints.cli.main serve Qwen/Qwen2.5-7B-Instruct --host 127.0.0.1 --port 8000 --enable-prefix-caching --no-async-scheduling --api-server-count 1 --max-model-len 32768` (started Sat Oct 31 05:18:11 2026).

| Arm | Latency (s) | LLM requests | Block calls | Fallback calls | Prompt tokens | Gen tokens | Prefix hit rate | Peak KV % | Peak RSS (MiB) | Items/s | Truncated | Mean prefill (s) | Mean decode (s) | Accuracy | Precision | Recall | F1 | Agrees w/ kalypso_per_tuple | Self-agreement |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 11.24 [11.07-11.32] | 1024 | n/a | n/a | 238976 | 2293 | 0.42 | 0.8 [0.8-0.9] | 11473 [11473-11473] | 91.06 [90.42-92.46] | 0 | 0.159 [0.157-0.161] | 0.183 [0.181-0.185] | 0.59 | 0.58 | 0.62 | 0.60 | 100% | 100% |
| `kalypso_block:2` | 8.03 [7.99-8.05] | 512 | 512 | 0 | 133344 | 4096 | 0.33 | 3.4 [3.4-3.4] | 11473 [11473-11473] | 127.58 [127.13-128.16] | 0 | 0.167 [0.165-0.167] | 0.999 [0.987-1.001] | 0.61 [0.61-0.61] | 0.60 [0.60-0.60] | 0.67 [0.67-0.67] | 0.63 [0.63-0.63] | 59% | 96% |
