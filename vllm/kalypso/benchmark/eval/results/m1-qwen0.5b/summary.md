# Milestone 1 evaluation

## biodex

Model `Qwen/Qwen2.5-0.5B-Instruct` on Apple M4 (macOS-15.7.3-arm64-arm-64bit); workload `biodex` (map), 10 rows; 3 repeats after 1 warmup; prefix cache reset per trial: yes; git `0686f6eba` (dirty).
Server: `python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-0.5B-Instruct --dtype float16 --port 8003 --max-model-len 8192 --enable-prefix-caching --block-size 16` (started Tue Oct  6 19:23:31 2026).

| Arm | Latency (s) | LLM requests | Prompt tokens | Gen tokens | Prefix hit rate | Peak KV % | Peak RSS (MiB) | Items/s | Truncated | Agrees w/ kalypso_per_tuple | Self-agreement |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 240.44 [220.92-248.74] | 10 | 28872 | 246 | 0.01 | 13.3 [12.5-13.3] | 1701 [1546-1724] | 0.04 [0.04-0.04] | 0 | 100% | 100% |
| `vllm_direct` | 244.25 [230.81-406.13] | 10 | 28872 | 246 | 0.01 | 12.7 [11.8-14.9] | 1587 [1482-1717] | 0.04 [0.03-0.04] | 0 | 100% | 100% |

## reviews

Model `Qwen/Qwen2.5-0.5B-Instruct` on Apple M4 (macOS-15.7.3-arm64-arm-64bit); workload `reviews` (filter), 20 rows, 10 positive; 3 repeats after 1 warmup; prefix cache reset per trial: yes; git `0686f6eba` (dirty).
Server: `python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-0.5B-Instruct --dtype float16 --port 8003 --max-model-len 8192 --enable-prefix-caching --block-size 16` (started Tue Oct  6 19:23:31 2026).

| Arm | Latency (s) | LLM requests | Prompt tokens | Gen tokens | Prefix hit rate | Peak KV % | Peak RSS (MiB) | Items/s | Truncated | Accuracy | Precision | Recall | F1 | Agrees w/ kalypso_per_tuple | Self-agreement |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 10.43 [10.11-10.56] | 20 | 2818 | 40 | 0.32 | 1.1 [0.1-1.1] | 1249 [1248-1249] | 1.92 [1.89-1.98] | 0 | 0.40 | 0.44 | 0.80 | 0.57 | 100% | 100% |
| `vllm_direct` | 9.88 [9.68-9.97] | 20 | 2818 | 40 | 0.32 | 1.1 [0.1-1.1] | 1323 [1314-1326] | 2.02 [2.01-2.07] | 0 | 0.40 | 0.44 | 0.80 | 0.57 | 100% | 100% |
| `kalypso_block:5` | 3.14 [3.00-3.15] | 4 | 582 | 80 | 0.16 | 0.3 [0.3-0.3] | 1339 [1334-1342] | 6.37 [6.35-6.68] | 0 | 0.50 | n/a | 0.00 | n/a | 10% | 100% |
| `kalypso_block:10` | 3.05 [2.89-3.06] | 2 | 426 | 82 | 0.08 | 0.3 | 1338 [1300-1339] | 6.56 [6.53-6.91] | 0 | 0.50 | n/a | 0.00 | n/a | 10% | 100% |
| `kalypso_block:20` | 3.60 [3.59-3.74] | 1 | 352 | 91 | 0.00 | 0.2 [0.2-0.3] | 1271 [1191-1282] | 5.56 [5.35-5.57] | 0 | 0.50 | n/a | 0.00 | n/a | 10% | 100% |

## medec

Model `Qwen/Qwen2.5-0.5B-Instruct` on Apple M4 (macOS-15.7.3-arm64-arm-64bit); workload `medec` (filter), 10 rows, 7 positive; 3 repeats after 1 warmup; prefix cache reset per trial: yes; git `0686f6eba` (dirty).
Server: `python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-0.5B-Instruct --dtype float16 --port 8003 --max-model-len 8192 --enable-prefix-caching --block-size 16` (started Tue Oct  6 19:23:31 2026).

| Arm | Latency (s) | LLM requests | Prompt tokens | Gen tokens | Prefix hit rate | Peak KV % | Peak RSS (MiB) | Items/s | Truncated | Accuracy | Precision | Recall | F1 | Agrees w/ kalypso_per_tuple | Self-agreement |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 39.97 [39.57-41.45] | 10 | 6772 | 20 | 0.06 | 1.2 | 1304 [1301-1328] | 0.25 [0.24-0.25] | 0 | 0.30 | n/a | 0.00 | n/a | 100% | 100% |
| `vllm_direct` | 40.56 [39.75-41.41] | 10 | 6772 | 20 | 0.06 | 1.3 [1.3-1.6] | 1329 [1315-1342] | 0.25 [0.24-0.25] | 0 | 0.30 | n/a | 0.00 | n/a | 100% | 100% |
| `kalypso_block:5` | 40.58 [39.42-41.23] | 2 | 4946 | 40 | 0.03 | 2.8 | 1366 [1363-1379] | 0.25 [0.24-0.25] | 0 | 0.30 | n/a | 0.00 | n/a | 100% | 100% |
| `kalypso_block:10` | 51.70 [50.52-52.18] | 1 | 4770 | 41 | 0.00 | 2.8 | 1303 [1266-1329] | 0.19 [0.19-0.20] | 0 | 0.30 | n/a | 0.00 | n/a | 100% | 100% |
| `kalypso_block:20` | 50.64 [50.58-51.72] | 1 | 4770 | 41 | 0.00 | 2.8 | 1300 [1285-1303] | 0.20 [0.19-0.20] | 0 | 0.30 | n/a | 0.00 | n/a | 100% | 100% |

## join5x5

Model `Qwen/Qwen2.5-0.5B-Instruct` on Apple M4 (macOS-15.7.3-arm64-arm-64bit); workload `join5x5` (join), 5 x 5 = 25 pairs, 5 positive; 3 repeats after 1 warmup; prefix cache reset per trial: yes; git `0686f6eba` (dirty).
Server: `python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-0.5B-Instruct --dtype float16 --port 8003 --max-model-len 8192 --enable-prefix-caching --block-size 16` (started Tue Oct  6 19:23:31 2026).

| Arm | Latency (s) | LLM requests | Prompt tokens | Gen tokens | Prefix hit rate | Peak KV % | Peak RSS (MiB) | Items/s | Truncated | Accuracy | Precision | Recall | F1 | Agrees w/ kalypso_per_tuple | Self-agreement |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 17.24 [16.91-18.31] | 25 | 4310 | 50 | 0.34 | 1.2 | 1368 [1304-1413] | 1.45 [1.37-1.48] | 0 | 0.92 | 0.80 | 0.80 | 0.80 | 100% | 100% |
| `vllm_direct` | 18.29 [17.44-18.98] | 25 | 4310 | 50 | 0.34 | 1.2 | 1399 [1386-1425] | 1.37 [1.32-1.43] | 0 | 0.92 | 0.80 | 0.80 | 0.80 | 100% | 100% |
| `kalypso_block:4` | 5.87 [5.69-6.13] | 9 | 1219 | 134 | 0.35 | 0.2 [0.2-0.2] | 1376 [1318-1413] | 4.26 [4.08-4.39] | 0 | 0.48 | 0.25 | 0.80 | 0.38 | 40% | 100% |
| `kalypso_block:25` | 2.22 [2.02-2.28] | 1 | 224 | 30 | 0.00 | 0.1 | 1308 [1262-1385] | 11.25 [10.97-12.38] | 0 | 0.60 | 0.00 | 0.00 | 0.00 | 60% | 100% |
