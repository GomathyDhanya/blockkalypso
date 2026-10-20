# E1

## short_arxiv: short rows (arXiv)

Model `Qwen/Qwen2.5-1.5B-Instruct` on NVIDIA L4 (Linux-4.19.0-gvisor-x86_64-with-glibc2.39); workload `arxiv_filter` (filter), 64 rows, 32 positive; 3 repeats after 1 warmup; prefix cache reset per trial: yes; git `d19f407a0` (dirty).
Server: `python3 -m vllm.entrypoints.cli.main serve Qwen/Qwen2.5-1.5B-Instruct --host 127.0.0.1 --port 8000 --enable-prefix-caching --no-async-scheduling --api-server-count 1 --max-model-len 32768` (started Tue Oct 20 06:08:54 2026).

| Arm | Completion time (s) | Speedup vs kalypso_per_tuple | LLM calls [retries + fallbacks] | Agrees w/ kalypso_per_tuple | F1 vs labels | Peak KV % |
|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 1.57 [1.56-1.58] | 1.00x | 64 [0] | 100% | 1.00 | 0.5 |
| `kalypso_block:2` | 1.45 [1.43-1.45] | 1.08x | 32 [0] | 62% | 0.73 | 2.5 |
| `kalypso_block:4` | 1.48 [1.48-1.50] | 1.06x | 16 [0] | 59% | 0.70 | 2.9 [2.9-2.9] |
| `kalypso_block:8` | 1.77 [1.75-1.78] | 0.88x | 8 [0] | 55% | 0.69 | 2.8 |
| `kalypso_block:16` | 2.45 [2.45-2.48] | 0.64x | 4 [0] | 48% | 0.59 | 2.8 |
| `kalypso_block:32` | 3.95 [3.94-4.01] | 0.40x | 2 [0] | 50% | 0.67 | 2.8 |
| `kalypso_block:2@4096` | 1.54 [1.54-1.63] | 1.02x | 32 [0] | 62% | 0.73 | 2.5 |
| `kalypso_block:4@4096` | 1.72 [1.70-1.73] | 0.91x | 16 [0] | 59% | 0.70 | 2.9 [2.9-2.9] |
| `kalypso_block:8@4096` | 2.18 [2.15-2.19] | 0.72x | 8 [0] | 55% | 0.69 | 2.8 |
| `kalypso_block:16@4096` | 3.10 [3.10-3.14] | 0.51x | 5 [0] | 55% | 0.69 | 2.8 |
| `kalypso_block:32@4096` | 3.11 [3.09-3.13] | 0.50x | 5 [0] | 55% | 0.69 | 2.8 |


Diagnostics:

| Arm | Prompt tokens | Gen tokens | Prefix hit rate | Peak RSS (MiB) | Items/s | First output (s) | Truncated | Accuracy | Precision | Recall | Self-agreement |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 24293 | 140 | 0.12 | 11396 [11396-11397] | 40.82 [40.53-40.97] | 0.34 [0.34-0.34] | 2 | 1.00 | 1.00 | 1.00 | 100% |
| `kalypso_block:2` | 18662 | 256 | 0.05 | 11396 [11396-11397] | 44.26 [44.11-44.63] | 1.11 [1.10-1.12] | 0 | 0.62 | 0.57 | 1.00 | 100% |
| `kalypso_block:4` | 17351 | 256 | 0.03 | 11396 [11396-11397] | 43.16 [42.58-43.27] | 1.34 [1.34-1.36] | 0 | 0.59 | 0.56 | 0.94 | 100% |
| `kalypso_block:8` | 16695 | 256 | 0.01 | 11396 [11396-11397] | 36.06 [36.02-36.47] | 1.65 [1.64-1.66] | 0 | 0.55 | 0.52 | 1.00 | 100% |
| `kalypso_block:16` | 16403 | 284 | 0.01 | 11396 [11396-11397] | 26.11 [25.79-26.12] | 2.36 [2.36-2.39] | 0 | 0.48 | 0.49 | 0.75 | 100% |
| `kalypso_block:32` | 16253 | 302 | 0.00 | 11396 [11396-11398] | 16.22 [15.96-16.23] | 3.89 [3.88-3.95] | 0 | 0.50 | 0.50 | 1.00 | 100% |
| `kalypso_block:2@4096` | 18662 | 256 | 0.05 | 11396 [11396-11398] | 41.51 [39.22-41.53] | 1.22 [1.21-1.30] | 0 | 0.62 | 0.57 | 1.00 | 100% |
| `kalypso_block:4@4096` | 17351 | 256 | 0.03 | 11396 [11396-11398] | 37.12 [37.04-37.58] | 1.58 [1.56-1.59] | 0 | 0.59 | 0.56 | 0.94 | 100% |
| `kalypso_block:8@4096` | 16695 | 256 | 0.01 | 11396 [11396-11398] | 29.37 [29.24-29.73] | 2.06 [2.03-2.07] | 0 | 0.55 | 0.52 | 1.00 | 100% |
| `kalypso_block:16@4096` | 16480 | 279 | 0.01 | 11396 [11396-11398] | 20.65 [20.36-20.66] | 2.32 [2.32-2.35] | 0 | 0.55 | 0.52 | 1.00 | 100% |
| `kalypso_block:32@4096` | 16480 | 279 | 0.01 | 11396 [11396-11398] | 20.55 [20.47-20.73] | 2.32 [2.31-2.33] | 0 | 0.55 | 0.52 | 1.00 | 100% |


Stages (medians over repeats; times from query start):

| Arm | Stage | Operators | Block | Tasks | Rows in | Rows out | Out/in | Occupancy | Start (s) | End (s) | Overlap (s) | Fallbacks |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 0 | SemFilter | 1 | 64 | 64 | 32 | 0.50 | n/a | 0.08 [0.08-0.08] | 1.56 [1.55-1.57] | n/a | 0 |
| `kalypso_block:2` | 0 | BlockSemFilter | 2 | 32 | 64 | 56 | 0.88 | 1.00 | 0.06 [0.06-0.07] | 1.44 [1.43-1.44] | n/a | 0 |
| `kalypso_block:4` | 0 | BlockSemFilter | 4 | 16 | 64 | 54 | 0.84 | 1.00 | 0.06 [0.06-0.06] | 1.48 [1.47-1.50] | n/a | 0 |
| `kalypso_block:8` | 0 | BlockSemFilter | 8 | 8 | 64 | 61 | 0.95 | 1.00 | 0.06 [0.06-0.06] | 1.77 [1.75-1.77] | n/a | 0 |
| `kalypso_block:16` | 0 | BlockSemFilter | 16 | 4 | 64 | 49 | 0.77 | 1.00 | 0.06 [0.06-0.07] | 2.44 [2.44-2.47] | n/a | 0 |
| `kalypso_block:32` | 0 | BlockSemFilter | 32 | 2 | 64 | 64 | 1.00 | 1.00 | 0.08 [0.08-0.08] | 3.94 [3.93-4.00] | n/a | 0 |
| `kalypso_block:2@4096` | 0 | BlockSemFilter | 2 | 32 | 64 | 56 | 0.88 | 1.00 | 0.17 [0.17-0.20] | 1.53 [1.53-1.62] | n/a | 0 |
| `kalypso_block:4@4096` | 0 | BlockSemFilter | 4 | 16 | 64 | 54 | 0.84 | 1.00 | 0.29 [0.29-0.29] | 1.72 [1.70-1.72] | n/a | 0 |
| `kalypso_block:8@4096` | 0 | BlockSemFilter | 8 | 8 | 64 | 61 | 0.95 | 1.00 | 0.48 [0.47-0.49] | 2.17 [2.15-2.18] | n/a | 0 |
| `kalypso_block:16@4096` | 0 | BlockSemFilter | 16 | 5 | 64 | 61 | 0.95 | 0.80 | 0.84 [0.83-0.84] | 3.09 [3.09-3.14] | n/a | 0 |
| `kalypso_block:32@4096` | 0 | BlockSemFilter | 32 | 5 | 64 | 61 | 0.95 | 0.40 | 0.82 [0.81-0.84] | 3.11 [3.08-3.12] | n/a | 0 |


## medium_medec: medium rows (MEDEC)

Model `Qwen/Qwen2.5-1.5B-Instruct` on NVIDIA L4 (Linux-4.19.0-gvisor-x86_64-with-glibc2.39); workload `medec_full` (filter), 64 rows, 35 positive; 3 repeats after 1 warmup; prefix cache reset per trial: yes; git `d19f407a0` (dirty).
Server: `python3 -m vllm.entrypoints.cli.main serve Qwen/Qwen2.5-1.5B-Instruct --host 127.0.0.1 --port 8000 --enable-prefix-caching --no-async-scheduling --api-server-count 1 --max-model-len 32768` (started Tue Oct 20 06:08:54 2026).

| Arm | Completion time (s) | Speedup vs kalypso_per_tuple | LLM calls [retries + fallbacks] | Agrees w/ kalypso_per_tuple | F1 vs labels | Peak KV % |
|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 2.76 [2.75-2.77] | 1.00x | 64 [0] | 100% | 0.00 | 0.5 |
| `kalypso_block:2` | 2.25 [2.24-2.25] | 1.23x | 32 [0] | 25% | 0.64 | 2.7 |
| `kalypso_block:4` | 2.36 [2.36-2.37] | 1.17x | 16 [0] | 33% | 0.55 | 4.7 |
| `kalypso_block:8` | 2.69 [2.68-2.71] | 1.03x | 8 [0] | 64% | 0.51 | 4.7 |
| `kalypso_block:16` | 3.56 [3.54-3.63] | 0.78x | 4 [0] | 98% | n/a | 4.6 |
| `kalypso_block:32` | 5.36 [5.29-5.39] | 0.51x | 2 [0] | 95% | 0.11 | 4.6 |
| `kalypso_block:2@4096` | 2.41 [2.40-2.42] | 1.14x | 32 [0] | 25% | 0.64 | 2.7 |
| `kalypso_block:4@4096` | 2.70 [2.67-2.72] | 1.02x | 16 [0] | 33% | 0.55 | 4.7 [4.4-4.7] |
| `kalypso_block:8@4096` | 3.30 [3.30-3.62] | 0.84x | 9 [1] | 56% | 0.53 | 4.6 [4.6-4.7] |
| `kalypso_block:16@4096` | 3.60 [3.55-4.23] | 0.77x | 8 [0] | 58% | 0.60 | 4.7 |
| `kalypso_block:32@4096` | 3.60 [3.57-3.78] | 0.77x | 8 [0] | 58% | 0.60 | 4.7 |


Diagnostics:

| Arm | Prompt tokens | Gen tokens | Prefix hit rate | Peak RSS (MiB) | Items/s | First output (s) | Truncated | Accuracy | Precision | Recall | Self-agreement |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 40837 | 128 | 0.07 | 11404 [11403-11404] | 23.20 [23.09-23.28] | 1.10 [1.09-1.10] | 0 | 0.44 | 0.00 | 0.00 | 100% |
| `kalypso_block:2` | 32635 | 256 | 0.14 | 11404 [11403-11404] | 28.51 [28.43-28.61] | 1.23 [1.23-1.23] | 0 | 0.53 | 0.55 | 0.77 | 100% |
| `kalypso_block:4` | 29771 | 256 | 0.07 | 11404 [11403-11404] | 27.16 [27.03-27.16] | 2.12 [2.12-2.13] | 0 | 0.45 | 0.50 | 0.60 | 100% |
| `kalypso_block:8` | 28339 | 256 | 0.04 | 11404 [11403-11404] | 23.84 [23.62-23.90] | 2.51 [2.51-2.54] | 0 | 0.55 | 0.62 | 0.43 | 100% |
| `kalypso_block:16` | 27659 | 284 | 0.02 | 11404 [11403-11404] | 17.98 [17.63-18.09] | n/a | 0 | 0.45 | n/a | 0.00 | 100% |
| `kalypso_block:32` | 27315 | 302 | 0.01 | 11404 [11404-11404] | 11.94 [11.88-12.10] | 5.24 [5.17-5.26] | 0 | 0.48 | 1.00 | 0.06 | 100% |
| `kalypso_block:2@4096` | 32635 | 256 | 0.14 | 11404 [11404-11404] | 26.52 [26.46-26.62] | 1.39 [1.39-1.40] | 0 | 0.53 | 0.55 | 0.77 | 100% |
| `kalypso_block:4@4096` | 29771 | 256 | 0.07 | 11404 [11404-11404] | 23.70 [23.56-23.95] | 2.47 [2.44-2.48] | 0 | 0.45 | 0.50 | 0.60 | 100% |
| `kalypso_block:8@4096` | 28482 | 254 | 0.04 | 11404 [11404-11404] | 19.38 [17.69-19.41] | 2.75 [2.74-2.88] | 0 | 0.53 | 0.59 | 0.49 | 100% |
| `kalypso_block:16@4096` | 28342 | 257 | 0.04 | 11404 [11404-11404] | 17.79 [15.12-18.02] | 3.26 [3.21-3.77] | 0 | 0.61 | 0.68 | 0.54 | 100% |
| `kalypso_block:32@4096` | 28342 | 257 | 0.04 | 11404 [11404-11404] | 17.80 [16.94-17.95] | 3.26 [3.23-3.44] | 0 | 0.61 | 0.68 | 0.54 | 100% |


Stages (medians over repeats; times from query start):

| Arm | Stage | Operators | Block | Tasks | Rows in | Rows out | Out/in | Occupancy | Start (s) | End (s) | Overlap (s) | Fallbacks |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 0 | SemFilter | 1 | 64 | 64 | 1 | 0.02 | n/a | 0.17 [0.17-0.17] | 2.75 [2.74-2.77] | n/a | 0 |
| `kalypso_block:2` | 0 | BlockSemFilter | 2 | 32 | 64 | 49 | 0.77 | 1.00 | 0.15 [0.14-0.15] | 2.24 [2.23-2.24] | n/a | 0 |
| `kalypso_block:4` | 0 | BlockSemFilter | 4 | 16 | 64 | 42 | 0.66 | 1.00 | 0.14 [0.14-0.14] | 2.35 [2.35-2.36] | n/a | 0 |
| `kalypso_block:8` | 0 | BlockSemFilter | 8 | 8 | 64 | 24 | 0.38 | 1.00 | 0.13 [0.13-0.14] | 2.68 [2.67-2.70] | n/a | 0 |
| `kalypso_block:16` | 0 | BlockSemFilter | 16 | 4 | 64 | 0 | 0.00 | 1.00 | 0.14 [0.14-0.14] | 3.55 [3.53-3.63] | n/a | 0 |
| `kalypso_block:32` | 0 | BlockSemFilter | 32 | 2 | 64 | 2 | 0.03 | 1.00 | 0.15 [0.15-0.18] | 5.36 [5.29-5.38] | n/a | 0 |
| `kalypso_block:2@4096` | 0 | BlockSemFilter | 2 | 32 | 64 | 49 | 0.77 | 1.00 | 0.31 [0.31-0.31] | 2.40 [2.40-2.41] | n/a | 0 |
| `kalypso_block:4@4096` | 0 | BlockSemFilter | 4 | 16 | 64 | 42 | 0.66 | 1.00 | 0.47 [0.47-0.49] | 2.69 [2.67-2.71] | n/a | 0 |
| `kalypso_block:8@4096` | 0 | BlockSemFilter | 8 | 9 | 64 | 29 | 0.45 | 0.89 | 0.74 [0.74-0.74] | 3.30 [3.29-3.60] | n/a | 1 |
| `kalypso_block:16@4096` | 0 | BlockSemFilter | 16 | 8 | 64 | 28 | 0.44 | 0.50 | 0.92 [0.87-1.26] | 3.59 [3.55-4.22] | n/a | 0 |
| `kalypso_block:32@4096` | 0 | BlockSemFilter | 32 | 8 | 64 | 28 | 0.44 | 0.25 | 0.91 [0.88-1.08] | 3.59 [3.56-3.76] | n/a | 0 |


## long_contracts: long rows (ContractNLI)

Model `Qwen/Qwen2.5-1.5B-Instruct` on NVIDIA L4 (Linux-4.19.0-gvisor-x86_64-with-glibc2.39); workload `contractnli_filter` (filter), 64 rows, 64 positive; 3 repeats after 1 warmup; prefix cache reset per trial: yes; git `d19f407a0` (dirty).
Server: `python3 -m vllm.entrypoints.cli.main serve Qwen/Qwen2.5-1.5B-Instruct --host 127.0.0.1 --port 8000 --enable-prefix-caching --no-async-scheduling --api-server-count 1 --max-model-len 32768` (started Tue Oct 20 06:08:54 2026).

| Arm | Completion time (s) | Speedup vs kalypso_per_tuple | LLM calls [retries + fallbacks] | Agrees w/ kalypso_per_tuple | F1 vs labels | Peak KV % |
|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 9.48 [9.42-9.50] | 1.00x | 64 [0] | 100% | 0.99 | 3.1 |
| `kalypso_block:2` | 9.57 [9.48-9.57] | 0.99x | 32 [0] | 75% | 0.87 | 3.4 |
| `kalypso_block:4` | 10.47 [10.43-10.48] | 0.90x | 16 [0] | 73% | 0.84 | 6.9 |
| `kalypso_block:8` | 12.59 [12.58-12.65] | 0.75x | 8 [0] | 98% | 1.00 | 13.8 |
| `kalypso_block:16` | failed in all 3 trials | | | | | |
| `kalypso_block:32` | failed in all 3 trials | | | | | |
| `kalypso_block:2@4096` | 9.94 [9.93-9.96] | 0.95x | 42 [20] | 91% | 0.96 | 3.0 |
| `kalypso_block:4@4096` | 10.61 [10.61-11.13] | 0.89x | 39 [20] | 81% | 0.91 | 3.9 |
| `kalypso_block:8@4096` | 10.93 [10.73-11.03] | 0.87x | 39 [20] | 81% | 0.91 | 3.9 |
| `kalypso_block:16@4096` | 10.92 [10.87-11.35] | 0.87x | 39 [20] | 81% | 0.91 | 3.9 |
| `kalypso_block:32@4096` | 11.09 [10.71-11.79] | 0.85x | 39 [20] | 81% | 0.91 | 3.9 |

- `kalypso_block:16` failed 3 trial(s): ArmError: /v1/semantic/query returned 400: {"error":{"message":"This model's maximum context length is 32768 tokens. However, you requested 112 output tokens and your prompt contains at least 32657 input tokens, for a total of at least 32769 tokens. Please reduce the length of the input prompt or th
- `kalypso_block:32` failed 3 trial(s): ArmError: /v1/semantic/query returned 400: {"error":{"message":"This model's maximum context length is 32768 tokens. However, you requested 208 output tokens and your prompt contains at least 32561 input tokens, for a total of at least 32769 tokens. Please reduce the length of the input prompt or th


Diagnostics:

| Arm | Prompt tokens | Gen tokens | Prefix hit rate | Peak RSS (MiB) | Items/s | First output (s) | Truncated | Accuracy | Precision | Recall | Self-agreement |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 126767 | 506 | 0.03 | 11425 [11425-11426] | 6.75 [6.74-6.80] | 1.85 [1.81-1.86] | 59 | 0.98 | 1.00 | 0.98 | 100% |
| `kalypso_block:2` | 120386 | 256 | 0.01 | 11425 [11425-11426] | 6.69 [6.69-6.75] | 2.30 [2.26-2.33] | 0 | 0.77 | 1.00 | 0.77 | 100% |
| `kalypso_block:4` | 118933 | 256 | 0.01 | 11426 [11425-11429] | 6.11 [6.11-6.14] | 3.96 [3.95-3.96] | 0 | 0.72 | 1.00 | 0.72 | 100% |
| `kalypso_block:8` | 118205 | 256 | 0.00 | 11431 [11429-11431] | 5.08 [5.06-5.09] | 8.51 [8.50-8.54] | 0 | 1.00 | 1.00 | 1.00 | 100% |
| `kalypso_block:2@4096` | 120575 | 216 | 0.01 | 11458 [11457-11458] | 6.44 [6.43-6.44] | 3.02 [3.01-3.03] | 0 | 0.92 | 1.00 | 0.92 | 100% |
| `kalypso_block:4@4096` | 120303 | 216 | 0.01 | 11458 [11457-11458] | 6.03 [5.75-6.03] | 3.63 [3.59-4.15] | 0 | 0.83 | 1.00 | 0.83 | 100% |
| `kalypso_block:8@4096` | 120303 | 216 | 0.01 | 11425 [11424-11457] | 5.85 [5.80-5.97] | 3.71 [3.68-3.92] | 0 | 0.83 | 1.00 | 0.83 | 100% |
| `kalypso_block:16@4096` | 120303 | 216 | 0.01 | 11425 [11424-11457] | 5.86 [5.64-5.89] | 3.83 [3.78-4.25] | 0 | 0.83 | 1.00 | 0.83 | 100% |
| `kalypso_block:32@4096` | 120303 | 216 | 0.01 | 11425 [11424-11457] | 5.77 [5.43-5.97] | 4.00 [3.59-4.65] | 0 | 0.83 | 1.00 | 0.83 | 100% |


Stages (medians over repeats; times from query start):

| Arm | Stage | Operators | Block | Tasks | Rows in | Rows out | Out/in | Occupancy | Start (s) | End (s) | Overlap (s) | Fallbacks |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 0 | SemFilter | 1 | 64 | 64 | 63 | 0.98 | n/a | 0.37 [0.35-0.38] | 9.46 [9.40-9.48] | n/a | 0 |
| `kalypso_block:2` | 0 | BlockSemFilter | 2 | 32 | 64 | 49 | 0.77 | 1.00 | 0.41 [0.39-0.41] | 9.55 [9.47-9.56] | n/a | 0 |
| `kalypso_block:4` | 0 | BlockSemFilter | 4 | 16 | 64 | 46 | 0.72 | 1.00 | 0.39 [0.37-0.39] | 10.46 [10.41-10.46] | n/a | 0 |
| `kalypso_block:8` | 0 | BlockSemFilter | 8 | 8 | 64 | 64 | 1.00 | 1.00 | 0.42 [0.41-0.43] | 12.57 [12.56-12.63] | n/a | 0 |
| `kalypso_block:2@4096` | 0 | BlockSemFilter | 2 | 42 | 64 | 59 | 0.92 | 0.76 | 1.37 [1.35-1.38] | 9.92 [9.91-9.94] | n/a | 20 |
| `kalypso_block:4@4096` | 0 | BlockSemFilter | 4 | 39 | 64 | 53 | 0.83 | 0.41 | 1.99 [1.95-2.37] | 10.59 [10.59-11.10] | n/a | 20 |
| `kalypso_block:8@4096` | 0 | BlockSemFilter | 8 | 39 | 64 | 53 | 0.83 | 0.21 | 2.08 [2.01-2.26] | 10.91 [10.71-11.01] | n/a | 20 |
| `kalypso_block:16@4096` | 0 | BlockSemFilter | 16 | 39 | 64 | 53 | 0.83 | 0.10 | 2.15 [2.11-2.57] | 10.90 [10.85-11.33] | n/a | 20 |
| `kalypso_block:32@4096` | 0 | BlockSemFilter | 32 | 39 | 64 | 53 | 0.83 | 0.05 | 2.30 [1.96-2.74] | 11.07 [10.69-11.78] | n/a | 20 |

