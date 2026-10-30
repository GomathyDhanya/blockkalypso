Model `Qwen/Qwen2.5-7B-Instruct` on unknown (Linux-4.19.0-gvisor-x86_64-with-glibc2.39); workload `movie_sel10` (filter), 400 rows, 40 positive; 3 repeats after 1 warmup; prefix cache reset per trial: yes; git `3cdb0b42d` (dirty).
Server: `python3 -m vllm.entrypoints.cli.main serve Qwen/Qwen2.5-7B-Instruct --host 127.0.0.1 --port 8000 --enable-prefix-caching --no-async-scheduling --api-server-count 1 --max-model-len 32768` (started Wed Oct 28 20:14:33 2026).

| Arm | Latency (s) | LLM requests | Prompt tokens | Gen tokens | Prefix hit rate | Peak KV % | Peak RSS (MiB) | Items/s | Truncated | Mean prefill (s) | Mean decode (s) | Accuracy | Precision | Recall | F1 | Agrees w/ kalypso_per_tuple | Self-agreement |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 3.95 [3.82-3.98] | 400 | 65259 | 1190 | 0.29 | 1.0 [1.0-1.0] | 11419 [11419-11419] | 101.37 [100.43-104.69] | 0 | 0.161 [0.156-0.163] | 0.289 [0.279-0.292] | 0.95 | 0.81 | 0.62 | 0.70 | 100% | 100% |
| `kalypso_block:4` | 2.03 [2.00-2.04] | 100 | 21557 | 1600 | 0.15 | 4.4 [4.4-4.5] | 11419 [11419-11419] | 197.14 [196.27-200.40] | 0 | 0.164 [0.161-0.166] | 0.973 [0.968-0.981] | 0.96 | 0.77 | 0.82 | 0.80 | 95% | 100% |
| `kalypso_block:6` | 2.08 [2.06-2.11] | 67 | 18818 | 1600 | 0.11 | 4.0 [4.0-4.0] | 11419 [11419-11419] | 192.59 [189.84-194.36] | 0 | 0.167 [0.164-0.170] | 1.124 [1.117-1.144] | 0.95 | 0.73 | 0.82 | 0.78 | 95% | 100% |
| `kalypso_block:8` | 2.21 [2.19-2.24] | 50 | 17407 | 1600 | 0.09 | 3.9 | 11419 [11419-11419] | 181.24 [178.89-182.73] | 0 | 0.167 [0.164-0.172] | 1.334 [1.322-1.339] | 0.95 | 0.76 | 0.72 | 0.74 | 97% | 100% |
| `kalypso_block:16` | 3.10 [3.09-3.12] | 25 | 15557 | 1775 | 0.05 | 3.7 [3.7-3.7] | 11419 [11419-11420] | 128.87 [128.16-129.32] | 0 | 0.185 [0.183-0.189] | 2.303 [2.291-2.303] | 0.94 | 0.77 | 0.60 | 0.68 | 96% | 100% |
