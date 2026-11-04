Model `Qwen/Qwen2.5-7B-Instruct` on unknown (Linux-4.19.0-gvisor-x86_64-with-glibc2.39); workload `cnli_map_len512` (map), 400 rows; 3 repeats after 1 warmup; prefix cache reset per trial: yes; git `6d3c9b6c1`.
Server: `python3 -m vllm.entrypoints.cli.main serve Qwen/Qwen2.5-7B-Instruct --host 127.0.0.1 --port 8000 --enable-prefix-caching --no-async-scheduling --api-server-count 1 --max-model-len 32768 --max-num-seqs 256 --max-num-batched-tokens 2048` (started Tue Nov  3 23:57:25 2026).

| Arm | Latency (s) | LLM requests | Block calls | Fallback calls | Prompt tokens | Gen tokens | Prefix hit rate | Peak KV % | Peak running | Peak waiting | Preemptions | Peak RSS (MiB) | Items/s | Truncated | Mean prefill (s) | Mean decode (s) | Agrees w/ kalypso_per_tuple | Self-agreement |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 23.13 [22.61-23.58] | 410 | n/a | n/a | 250244 | 10808 | 0.10 | 12.2 [12.2-12.2] | 93 | 351 | 0 | 11464 [11463-11464] | 17.29 [16.96-17.69] | 10 | 0.202 [0.196-0.207] | 3.566 [3.473-3.661] | 100% | 100% |
| `kalypso_block:2` | 26.41 [25.46-27.41] | 233 | 233 | 0 | 244817 | 15594 | 0.06 | 34.6 | 140 | 191 | 0 | 11464 [11463-11464] | 15.14 [14.59-15.71] | 2 | 0.254 [0.242-0.266] | 9.134 [8.760-9.542] | 6% | 100% |
| `kalypso_block:4` | 29.83 [29.22-30.95] | 141 | 141 | 0 | 237893 | 14259 | 0.04 | 43.5 [43.5-43.5] | 92 [92-94] | 96 [95-96] | 0 | 11464 [11463-11464] | 13.41 [12.92-13.69] | 4 | 0.288 [0.280-0.300] | 10.085 [9.903-10.537] | 7% | 100% |
| `kalypso_block:8` | 36.50 [36.42-38.62] | 103 | 103 | 0 | 239636 | 13871 [13850-13893] | 0.04 | 45.0 [45.0-45.0] | 50 | 48 | 0 | 11464 [11464-11464] | 10.96 [10.36-10.98] | 7 | 0.307 [0.306-0.327] | 9.383 [9.368-9.831] | 8% | 93% |
