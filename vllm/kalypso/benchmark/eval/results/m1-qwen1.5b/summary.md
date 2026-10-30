# Milestone 1 evaluation

## reviews

Model `Qwen/Qwen2.5-1.5B-Instruct` on Apple M4 (macOS-15.7.3-arm64-arm-64bit); workload `reviews` (filter), 20 rows, 10 positive; 3 repeats after 1 warmup; prefix cache reset per trial: yes; git `2bdf3b530`.
Server: `python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-1.5B-Instruct --dtype float16 --port 8003 --max-model-len 8192 --enable-prefix-caching --block-size 16` (started Tue Oct  6 21:09:32 2026).

| Arm | Latency (s) | LLM requests | Prompt tokens | Gen tokens | Prefix hit rate | Peak KV % | Peak RSS (MiB) | Items/s | Truncated | Accuracy | Precision | Recall | F1 | Agrees w/ kalypso_per_tuple | Self-agreement |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 36.08 [35.85-37.13] | 20 | 2818 | 160 | 0.32 | 3.0 | 3483 [3452-3493] | 0.55 [0.54-0.56] | 20 | 0.50 | n/a | 0.00 | n/a | 100% | 100% |
| `vllm_direct` | 36.37 [35.90-38.33] | 20 | 2818 | 160 | 0.32 | 3.0 | 3523 [3521-3578] | 0.55 [0.52-0.56] | 20 | 0.50 | n/a | 0.00 | n/a | 100% | 100% |
| `kalypso_block:5` | 9.80 [9.56-10.31] | 4 | 582 | 80 | 0.16 | 0.8 | 3486 [3483-3495] | 2.04 [1.94-2.09] | 0 | 0.20 | 0.25 | 0.30 | 0.27 | 40% | 100% |
| `kalypso_block:10` | 9.38 [9.04-9.46] | 2 | 426 | 82 | 0.08 | 0.6 | 3433 [3390-3490] | 2.13 [2.11-2.21] | 0 | 0.15 | 0.11 | 0.10 | 0.11 | 55% | 100% |
| `kalypso_block:20` | 11.07 [10.59-11.42] | 1 | 352 | 91 | 0.00 | 0.6 | 3388 [3295-3415] | 1.81 [1.75-1.89] | 0 | 0.05 | 0.00 | 0.00 | 0.00 | 55% | 100% |

## medec

Model `Qwen/Qwen2.5-1.5B-Instruct` on Apple M4 (macOS-15.7.3-arm64-arm-64bit); workload `medec` (filter), 10 rows, 7 positive; 3 repeats after 1 warmup; prefix cache reset per trial: yes; git `2bdf3b530`.
Server: `python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-1.5B-Instruct --dtype float16 --port 8003 --max-model-len 8192 --enable-prefix-caching --block-size 16` (started Tue Oct  6 21:09:32 2026).

| Arm | Latency (s) | LLM requests | Prompt tokens | Gen tokens | Prefix hit rate | Peak KV % | Peak RSS (MiB) | Items/s | Truncated | Accuracy | Precision | Recall | F1 | Agrees w/ kalypso_per_tuple | Self-agreement |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 139.14 [120.42-144.08] | 10 | 6772 | 20 | 0.06 | 2.9 | 3322 [3313-3475] | 0.07 [0.07-0.08] | 0 | 0.60 | 1.00 | 0.43 | 0.60 | 100% | 100% |
| `vllm_direct` | 138.12 [128.21-210.03] | 10 | 6772 | 20 | 0.06 | 3.2 [3.0-3.9] | 3401 [3364-3434] | 0.07 [0.05-0.08] | 0 | 0.60 | 1.00 | 0.43 | 0.60 | 100% | 100% |
| `kalypso_block:5` | 121.38 [117.11-124.77] | 2 | 4946 | 40 | 0.03 | 6.5 [6.5-6.5] | 3378 [3371-3426] | 0.08 [0.08-0.09] | 0 | 0.60 | 0.80 | 0.57 | 0.67 | 40% | 100% |
| `kalypso_block:10` | 144.70 [125.77-147.42] | 1 | 4770 | 41 | 0.00 | 6.4 | 3455 [3408-3490] | 0.07 [0.07-0.08] | 0 | 0.70 | 0.70 | 1.00 | 0.82 | 30% | 100% |
| `kalypso_block:20` | 144.23 [127.92-149.11] | 1 | 4770 | 41 | 0.00 | 6.4 | 3423 [3400-3448] | 0.07 [0.07-0.08] | 0 | 0.70 | 0.70 | 1.00 | 0.82 | 30% | 100% |

## join5x5

Model `Qwen/Qwen2.5-1.5B-Instruct` on Apple M4 (macOS-15.7.3-arm64-arm-64bit); workload `join5x5` (join), 5 x 5 = 25 pairs, 5 positive; 3 repeats after 1 warmup; prefix cache reset per trial: yes; git `2bdf3b530`.
Server: `python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-1.5B-Instruct --dtype float16 --port 8003 --max-model-len 8192 --enable-prefix-caching --block-size 16` (started Tue Oct  6 21:09:32 2026).

| Arm | Latency (s) | LLM requests | Prompt tokens | Gen tokens | Prefix hit rate | Peak KV % | Peak RSS (MiB) | Items/s | Truncated | Accuracy | Precision | Recall | F1 | Agrees w/ kalypso_per_tuple | Self-agreement |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 66.53 [64.49-69.23] | 25 | 4310 | 200 | 0.34 | 4.2 | 3386 [3357-3566] | 0.38 [0.36-0.39] | 25 | 0.20 | 0.20 | 1.00 | 0.33 | 100% | 100% |
| `vllm_direct` | 66.27 [66.25-66.61] | 25 | 4310 | 200 | 0.34 | 4.2 | 3503 [3452-3512] | 0.38 [0.38-0.38] | 25 | 0.20 | 0.20 | 1.00 | 0.33 | 100% | 100% |
| `kalypso_block:4` | 19.41 [19.14-20.47] | 9 | 1219 | 68 | 0.35 | 0.5 [0.5-0.5] | 3391 [3373-3486] | 1.29 [1.22-1.31] | 0 | 0.80 | 0.50 | 1.00 | 0.67 | 40% | 100% |
| `kalypso_block:25` | 6.41 [6.37-6.58] | 1 | 224 | 30 | 0.00 | 0.3 | 3289 [3167-3489] | 3.90 [3.80-3.92] | 0 | 0.84 | 0.60 | 0.60 | 0.60 | 20% | 100% |
