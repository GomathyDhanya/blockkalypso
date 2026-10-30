Model `Qwen/Qwen2.5-7B-Instruct` on unknown (Linux-4.19.0-gvisor-x86_64-with-glibc2.39); workload `movie_positive` (filter), 500 rows, 373 positive; 3 repeats after 1 warmup; prefix cache reset per trial: yes; git `db0f47ac7` (dirty).
Server: `python3 -m vllm.entrypoints.cli.main serve Qwen/Qwen2.5-7B-Instruct --host 127.0.0.1 --port 8000 --enable-prefix-caching --no-async-scheduling --api-server-count 1 --max-model-len 32768` (started Wed Oct 28 17:49:07 2026).

| Arm | Latency (s) | LLM requests | Prompt tokens | Gen tokens | Prefix hit rate | Peak KV % | Peak RSS (MiB) | Items/s | Truncated | Mean prefill (s) | Mean decode (s) | Accuracy | Precision | Recall | F1 | Agrees w/ kalypso_per_tuple | Self-agreement |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 4.90 [4.81-5.94] | 500 | 81698 | 1479 | 0.29 | 1.0 [1.0-1.0] | 11428 [11427-11428] | 102.06 [84.17-103.89] | 1 | 0.161 [0.157-0.176] | 0.289 [0.283-0.316] | 0.72 [0.72-0.72] | 1.00 [1.00-1.00] | 0.63 [0.62-0.63] | 0.77 [0.77-0.77] | 100% | 100% |
| `vllm_direct` | 4.92 [4.62-5.67] | 500 | 81698 | 1480 | 0.29 | 1.0 [0.9-1.0] | 11428 [11427-11428] | 101.56 [88.22-108.16] | 1 | 0.160 [0.157-0.163] | 0.288 [0.284-0.294] | 0.72 | 1.00 | 0.63 | 0.77 | 100% | 100% |
| `vllm_direct:no_penalty` | 4.88 [4.68-5.10] | 500 | 81698 | 1475 [1474-1475] | 0.29 | 0.9 [0.9-0.9] | 11428 [11427-11428] | 102.50 [97.96-106.88] | 0 | 0.158 [0.157-0.164] | 0.283 [0.273-0.295] | 0.72 | 1.00 | 0.63 | 0.77 | 100% | 100% |
| `vllm_direct:block_prompt` | 3.84 [3.30-4.02] | 500 | 40190 | 1000 | 0.20 | 0.5 [0.5-0.5] | 11428 [11427-11428] | 130.14 [124.47-151.38] | 0 | 0.132 [0.131-0.140] | 0.111 [0.110-0.116] | 0.72 | 1.00 | 0.62 | 0.77 | 92% | 100% |
| `vllm_direct:block_prompt+no_penalty` | 3.46 [3.33-3.50] | 500 | 40190 | 1000 | 0.20 | 0.5 [0.5-0.5] | 11428 [11427-11428] | 144.43 [142.65-150.06] | 0 | 0.135 [0.132-0.137] | 0.114 [0.114-0.114] | 0.72 | 1.00 | 0.62 | 0.77 | 92% | 100% |
