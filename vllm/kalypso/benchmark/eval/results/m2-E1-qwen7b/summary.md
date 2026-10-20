# E1

## short_arxiv: short rows (arXiv)

Model `Qwen/Qwen2.5-7B-Instruct` on NVIDIA L40S (Linux-4.19.0-gvisor-x86_64-with-glibc2.39); workload `arxiv_filter` (filter), 64 rows, 32 positive; 3 repeats after 1 warmup; prefix cache reset per trial: yes; git `d19f407a0` (dirty).
Server: `python3 -m vllm.entrypoints.cli.main serve Qwen/Qwen2.5-7B-Instruct --host 127.0.0.1 --port 8000 --enable-prefix-caching --no-async-scheduling --api-server-count 1 --max-model-len 32768` (started Tue Oct 20 15:59:44 2026).

| Arm | Completion time (s) | Speedup vs kalypso_per_tuple | LLM calls [retries + fallbacks] | Agrees w/ kalypso_per_tuple | F1 vs labels | Peak KV % |
|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 1.78 [1.76-1.82] | 1.00x | 64 [0] | 100% | 1.00 | 1.0 |
| `kalypso_block:2` | 1.63 [1.60-1.65] | 1.09x | 32 [0] | 75% | 0.67 | 3.3 |
| `kalypso_block:4` | 1.74 [1.71-1.77] | 1.02x | 16 [0] | 83% | 0.81 | 3.8 |
| `kalypso_block:8` | 2.10 [2.08-2.12] | 0.85x | 8 [0] | 83% | 0.81 | 3.8 [3.7-3.8] |
| `kalypso_block:16` | 3.04 [3.03-3.05] | 0.59x | 4 [0] | 81% | 0.78 | 3.7 |
| `kalypso_block:32` | 4.97 [4.96-5.01] | 0.36x | 2 [0] | 75% | 0.71 | 3.7 |
| `kalypso_block:2@4096` | 1.69 [1.68-1.73] | 1.06x | 32 [0] | 75% | 0.67 | 3.3 |
| `kalypso_block:4@4096` | 1.89 [1.87-1.91] | 0.95x | 16 [0] | 83% | 0.81 | 3.8 |
| `kalypso_block:8@4096` | 2.36 [2.34-2.39] | 0.76x | 8 [0] | 83% | 0.81 | 3.7 [3.7-3.8] |
| `kalypso_block:16@4096` | 3.35 [3.33-3.39] | 0.53x | 5 [0] | 91% | 0.90 | 3.7 |
| `kalypso_block:32@4096` | 3.36 [3.35-3.39] | 0.53x | 5 [0] | 91% | 0.90 | 3.7 |


Diagnostics:

| Arm | Prompt tokens | Gen tokens | Prefix hit rate | Peak RSS (MiB) | Items/s | First output (s) | Truncated | Accuracy | Precision | Recall | Self-agreement |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 24293 | 192 | 0.12 | 12415 [12414-12415] | 35.85 [35.18-36.36] | 0.46 [0.45-0.46] | 0 | 1.00 | 1.00 | 1.00 | 100% |
| `kalypso_block:2` | 18662 | 256 | 0.05 | 12415 [12414-12415] | 39.24 [38.81-39.88] | 1.22 [1.20-1.24] | 0 | 0.75 | 1.00 | 0.50 | 100% |
| `kalypso_block:4` | 17351 | 256 | 0.03 | 12415 [12414-12415] | 36.72 [36.16-37.38] | 1.55 [1.52-1.58] | 0 | 0.83 | 0.89 | 0.75 | 100% |
| `kalypso_block:8` | 16695 | 256 | 0.01 | 12415 [12414-12415] | 30.53 [30.16-30.70] | 1.93 [1.93-1.96] | 0 | 0.83 | 0.89 | 0.75 | 100% |
| `kalypso_block:16` | 16403 | 284 | 0.01 | 12415 [12414-12415] | 21.09 [21.01-21.10] | 2.91 [2.91-2.93] | 0 | 0.81 | 0.95 | 0.66 | 100% |
| `kalypso_block:32` | 16253 | 302 | 0.00 | 12415 [12414-12416] | 12.89 [12.77-12.90] | 4.90 [4.89-4.94] | 0 | 0.75 | 0.83 | 0.62 | 100% |
| `kalypso_block:2@4096` | 18662 | 256 | 0.05 | 12415 [12414-12416] | 37.94 [36.97-38.12] | 1.28 [1.27-1.32] | 0 | 0.75 | 1.00 | 0.50 | 100% |
| `kalypso_block:4@4096` | 17351 | 256 | 0.03 | 12415 [12414-12416] | 33.90 [33.54-34.28] | 1.69 [1.68-1.72] | 0 | 0.83 | 0.89 | 0.75 | 100% |
| `kalypso_block:8@4096` | 16695 | 256 | 0.01 | 12415 [12414-12416] | 27.07 [26.78-27.32] | 2.20 [2.18-2.23] | 0 | 0.83 | 0.89 | 0.75 | 100% |
| `kalypso_block:16@4096` | 16480 | 279 | 0.01 | 12415 [12414-12416] | 19.13 [18.90-19.24] | 2.27 [2.25-2.32] | 0 | 0.91 | 0.96 | 0.84 | 100% |
| `kalypso_block:32@4096` | 16480 | 279 | 0.01 | 12415 [12414-12416] | 19.06 [18.90-19.10] | 2.29 [2.26-2.31] | 0 | 0.91 | 0.96 | 0.84 | 100% |


Stages (medians over repeats; times from query start):

| Arm | Stage | Operators | Block | Tasks | Rows in | Rows out | Out/in | Occupancy | Start (s) | End (s) | Overlap (s) | Fallbacks |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 0 | SemFilter | 1 | 64 | 64 | 32 | 0.50 | n/a | 0.05 [0.05-0.05] | 1.78 [1.75-1.81] | n/a | 0 |
| `kalypso_block:2` | 0 | BlockSemFilter | 2 | 32 | 64 | 16 | 0.25 | 1.00 | 0.04 [0.04-0.04] | 1.63 [1.60-1.64] | n/a | 0 |
| `kalypso_block:4` | 0 | BlockSemFilter | 4 | 16 | 64 | 27 | 0.42 | 1.00 | 0.04 [0.04-0.04] | 1.74 [1.71-1.77] | n/a | 0 |
| `kalypso_block:8` | 0 | BlockSemFilter | 8 | 8 | 64 | 27 | 0.42 | 1.00 | 0.04 [0.04-0.04] | 2.09 [2.08-2.12] | n/a | 0 |
| `kalypso_block:16` | 0 | BlockSemFilter | 16 | 4 | 64 | 22 | 0.34 | 1.00 | 0.04 [0.04-0.04] | 3.03 [3.03-3.04] | n/a | 0 |
| `kalypso_block:32` | 0 | BlockSemFilter | 32 | 2 | 64 | 24 | 0.38 | 1.00 | 0.05 [0.05-0.05] | 4.96 [4.96-5.01] | n/a | 0 |
| `kalypso_block:2@4096` | 0 | BlockSemFilter | 2 | 32 | 64 | 16 | 0.25 | 1.00 | 0.10 [0.10-0.10] | 1.68 [1.67-1.73] | n/a | 0 |
| `kalypso_block:4@4096` | 0 | BlockSemFilter | 4 | 16 | 64 | 27 | 0.42 | 1.00 | 0.17 [0.17-0.18] | 1.88 [1.86-1.90] | n/a | 0 |
| `kalypso_block:8@4096` | 0 | BlockSemFilter | 8 | 8 | 64 | 27 | 0.42 | 1.00 | 0.29 [0.29-0.30] | 2.36 [2.34-2.39] | n/a | 0 |
| `kalypso_block:16@4096` | 0 | BlockSemFilter | 16 | 5 | 64 | 28 | 0.44 | 0.80 | 0.50 [0.49-0.50] | 3.34 [3.32-3.38] | n/a | 0 |
| `kalypso_block:32@4096` | 0 | BlockSemFilter | 32 | 5 | 64 | 28 | 0.44 | 0.40 | 0.49 [0.49-0.50] | 3.35 [3.35-3.38] | n/a | 0 |


## medium_medec: medium rows (MEDEC)

Model `Qwen/Qwen2.5-7B-Instruct` on NVIDIA L40S (Linux-4.19.0-gvisor-x86_64-with-glibc2.39); workload `medec_full` (filter), 64 rows, 35 positive; 3 repeats after 1 warmup; prefix cache reset per trial: yes; git `d19f407a0` (dirty).
Server: `python3 -m vllm.entrypoints.cli.main serve Qwen/Qwen2.5-7B-Instruct --host 127.0.0.1 --port 8000 --enable-prefix-caching --no-async-scheduling --api-server-count 1 --max-model-len 32768` (started Tue Oct 20 15:59:44 2026).

| Arm | Completion time (s) | Speedup vs kalypso_per_tuple | LLM calls [retries + fallbacks] | Agrees w/ kalypso_per_tuple | F1 vs labels | Peak KV % |
|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 3.26 [3.19-3.30] | 1.00x | 64 [0] | 100% | 0.60 | 1.1 |
| `kalypso_block:2` | 3.11 [3.01-3.12] | 1.05x | 43 [11] | 61% | 0.21 | 3.6 |
| `kalypso_block:4` | 3.38 [3.28-3.39] | 0.97x | 29 [13] | 56% | 0.37 | 6.2 |
| `kalypso_block:8` | 3.17 [3.11-3.22] | 1.03x | 8 [0] | 61% | n/a | 6.2 |
| `kalypso_block:16` | 5.05 [5.02-5.12] | 0.65x | 4 [0] | 61% | n/a | 6.2 |
| `kalypso_block:32` | 6.36 [6.31-6.47] | 0.51x | 2 [0] | 61% | n/a | 6.2 |
| `kalypso_block:2@4096` | 3.12 [3.11-3.31] | 1.04x | 43 [11] | 61% | 0.21 | 3.6 |
| `kalypso_block:4@4096` | 3.49 [3.48-3.78] | 0.93x | 29 [13] | 56% | 0.37 | 5.9 [5.9-6.2] |
| `kalypso_block:8@4096` | 3.47 [3.47-3.68] | 0.94x | 9 [1] | 59% | 0.00 | 6.2 [6.1-6.2] |
| `kalypso_block:16@4096` | 3.83 [3.82-4.02] | 0.85x | 11 [3] | 61% | 0.15 | 6.2 |
| `kalypso_block:32@4096` | 3.84 [3.83-3.99] | 0.85x | 11 [3] | 61% | 0.15 | 6.2 |


Diagnostics:

| Arm | Prompt tokens | Gen tokens | Prefix hit rate | Peak RSS (MiB) | Items/s | First output (s) | Truncated | Accuracy | Precision | Recall | Self-agreement |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 40837 | 192 | 0.07 | 12422 [12422-12422] | 19.64 [19.39-20.09] | 0.55 [0.54-0.55] | 0 | 0.62 | 0.72 | 0.51 | 100% |
| `kalypso_block:2` | 39140 | 286 | 0.12 | 12422 [12422-12422] | 20.61 [20.51-21.28] | 2.21 [2.13-2.22] | 0 | 0.42 | 0.42 | 0.14 | 100% |
| `kalypso_block:4` | 37359 | 290 | 0.07 | 12422 [12422-12422] | 18.96 [18.89-19.50] | 2.61 [2.52-2.61] | 0 | 0.53 | 0.69 | 0.26 | 100% |
| `kalypso_block:8` | 28339 | 256 | 0.04 | 12422 [12422-12422] | 20.19 [19.89-20.58] | n/a | 0 | 0.45 | n/a | 0.00 | 100% |
| `kalypso_block:16` | 27659 | 325 | 0.02 | 12422 [12422-12422] | 12.67 [12.49-12.74] | n/a | 1 | 0.45 | n/a | 0.00 | 100% |
| `kalypso_block:32` | 27315 | 302 | 0.01 | 12422 [12422-12422] | 10.07 [9.89-10.14] | n/a | 0 | 0.45 | n/a | 0.00 | 100% |
| `kalypso_block:2@4096` | 39140 | 286 | 0.12 | 12422 [12422-12422] | 20.48 [19.31-20.56] | 2.24 [2.24-2.38] | 0 | 0.42 | 0.42 | 0.14 | 100% |
| `kalypso_block:4@4096` | 37359 | 290 | 0.07 | 12422 [12422-12422] | 18.36 [16.93-18.41] | 2.73 [2.72-2.98] | 0 | 0.53 | 0.69 | 0.26 | 100% |
| `kalypso_block:8@4096` | 28482 | 254 | 0.04 | 12422 [12422-12422] | 18.45 [17.40-18.45] | 2.75 [2.75-2.94] | 0 | 0.44 | 0.00 | 0.00 | 100% |
| `kalypso_block:16@4096` | 29905 | 264 | 0.04 | 12422 [12422-12422] | 16.69 [15.93-16.75] | 3.47 [3.46-3.65] | 0 | 0.48 | 0.75 | 0.09 | 100% |
| `kalypso_block:32@4096` | 29905 | 264 | 0.04 | 12422 [12422-12422] | 16.66 [16.04-16.73] | 3.48 [3.46-3.62] | 0 | 0.48 | 0.75 | 0.09 | 100% |


Stages (medians over repeats; times from query start):

| Arm | Stage | Operators | Block | Tasks | Rows in | Rows out | Out/in | Occupancy | Start (s) | End (s) | Overlap (s) | Fallbacks |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 0 | SemFilter | 1 | 64 | 64 | 25 | 0.39 | n/a | 0.10 [0.10-0.10] | 3.25 [3.18-3.29] | n/a | 0 |
| `kalypso_block:2` | 0 | BlockSemFilter | 2 | 32 | 64 | 12 | 0.19 | 1.00 | 0.09 [0.09-0.10] | 3.10 [3.00-3.12] | n/a | 11 |
| `kalypso_block:4` | 0 | BlockSemFilter | 4 | 16 | 64 | 13 | 0.20 | 1.00 | 0.09 [0.08-0.09] | 3.37 [3.28-3.39] | n/a | 13 |
| `kalypso_block:8` | 0 | BlockSemFilter | 8 | 8 | 64 | 0 | 0.00 | 1.00 | 0.08 [0.08-0.08] | 3.17 [3.11-3.21] | n/a | 0 |
| `kalypso_block:16` | 0 | BlockSemFilter | 16 | 4 | 64 | 0 | 0.00 | 1.00 | 0.09 [0.08-0.09] | 5.05 [5.02-5.12] | n/a | 0 |
| `kalypso_block:32` | 0 | BlockSemFilter | 32 | 2 | 64 | 0 | 0.00 | 1.00 | 0.09 [0.09-0.10] | 6.35 [6.31-6.47] | n/a | 0 |
| `kalypso_block:2@4096` | 0 | BlockSemFilter | 2 | 32 | 64 | 12 | 0.19 | 1.00 | 0.18 [0.18-0.19] | 3.12 [3.11-3.31] | n/a | 11 |
| `kalypso_block:4@4096` | 0 | BlockSemFilter | 4 | 16 | 64 | 13 | 0.20 | 1.00 | 0.29 [0.29-0.35] | 3.48 [3.47-3.78] | n/a | 13 |
| `kalypso_block:8@4096` | 0 | BlockSemFilter | 8 | 9 | 64 | 1 | 0.02 | 0.89 | 0.44 [0.44-0.46] | 3.47 [3.47-3.68] | n/a | 1 |
| `kalypso_block:16@4096` | 0 | BlockSemFilter | 16 | 8 | 64 | 4 | 0.06 | 0.50 | 0.53 [0.52-0.54] | 3.83 [3.82-4.01] | n/a | 3 |
| `kalypso_block:32@4096` | 0 | BlockSemFilter | 32 | 8 | 64 | 4 | 0.06 | 0.25 | 0.53 [0.53-0.56] | 3.84 [3.82-3.99] | n/a | 3 |


## long_contracts: long rows (ContractNLI)

Model `Qwen/Qwen2.5-7B-Instruct` on NVIDIA L40S (Linux-4.19.0-gvisor-x86_64-with-glibc2.39); workload `contractnli_filter` (filter), 64 rows, 64 positive; 3 repeats after 1 warmup; prefix cache reset per trial: yes; git `d19f407a0` (dirty).
Server: `python3 -m vllm.entrypoints.cli.main serve Qwen/Qwen2.5-7B-Instruct --host 127.0.0.1 --port 8000 --enable-prefix-caching --no-async-scheduling --api-server-count 1 --max-model-len 32768` (started Tue Oct 20 15:59:44 2026).

| Arm | Completion time (s) | Speedup vs kalypso_per_tuple | LLM calls [retries + fallbacks] | Agrees w/ kalypso_per_tuple | F1 vs labels | Peak KV % |
|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 10.39 [10.31-10.58] | 1.00x | 64 [0] | 100% | 1.00 | 2.1 |
| `kalypso_block:2` | 10.61 [10.48-10.63] | 0.98x | 32 [0] | 84% | 0.92 | 4.5 |
| `kalypso_block:4` | 11.33 [11.31-11.56] | 0.92x | 16 [0] | 86% | 0.92 | 9.2 |
| `kalypso_block:8` | 12.96 [12.89-13.37] | 0.80x | 8 [0] | 94% | 0.97 | 18.4 |
| `kalypso_block:16` | failed in all 3 trials | | | | | |
| `kalypso_block:32` | failed in all 3 trials | | | | | |
| `kalypso_block:2@4096` | 10.85 [10.74-11.35] | 0.96x | 42 [20] | 84% | 0.92 | 4.0 |
| `kalypso_block:4@4096` | 11.29 [11.19-12.02] | 0.92x | 39 [20] | 86% | 0.92 | 5.2 |
| `kalypso_block:8@4096` | 11.35 [11.30-13.49] | 0.92x | 39 [20] | 86% | 0.92 | 5.2 |
| `kalypso_block:16@4096` | 11.29 [11.02-12.13] | 0.92x | 39 [20] | 86% | 0.92 | 5.2 |
| `kalypso_block:32@4096` | 11.37 [11.08-11.54] | 0.91x | 39 [20] | 86% | 0.92 | 5.2 |

- `kalypso_block:16` failed 3 trial(s): ArmError: /v1/semantic/query returned 400: {"error":{"message":"This model's maximum context length is 32768 tokens. However, you requested 112 output tokens and your prompt contains at least 32657 input tokens, for a total of at least 32769 tokens. Please reduce the length of the input prompt or th
- `kalypso_block:32` failed 3 trial(s): ArmError: /v1/semantic/query returned 400: {"error":{"message":"This model's maximum context length is 32768 tokens. However, you requested 208 output tokens and your prompt contains at least 32561 input tokens, for a total of at least 32769 tokens. Please reduce the length of the input prompt or th


Diagnostics:

| Arm | Prompt tokens | Gen tokens | Prefix hit rate | Peak RSS (MiB) | Items/s | First output (s) | Truncated | Accuracy | Precision | Recall | Self-agreement |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 126767 | 193 | 0.03 | 12452 [12445-12476] | 6.16 [6.05-6.21] | 0.89 [0.88-1.10] | 1 | 1.00 | 1.00 | 1.00 | 100% |
| `kalypso_block:2` | 120386 | 256 | 0.01 | 12452 [12445-12476] | 6.04 [6.02-6.11] | 2.15 [2.14-2.23] | 0 | 0.84 | 1.00 | 0.84 | 100% |
| `kalypso_block:4` | 118933 | 256 | 0.01 | 12454 [12445-12476] | 5.65 [5.54-5.66] | 4.04 [4.04-4.09] | 0 | 0.86 | 1.00 | 0.86 | 100% |
| `kalypso_block:8` | 118205 | 256 | 0.00 | 12454 [12447-12476] | 4.94 [4.79-4.96] | 8.64 [8.62-8.90] | 0 | 0.94 | 1.00 | 0.94 | 100% |
| `kalypso_block:2@4096` | 120575 | 216 | 0.01 | 12484 [12477-12484] | 5.90 [5.64-5.96] | 2.54 [2.50-2.60] | 0 | 0.84 | 1.00 | 0.84 | 100% |
| `kalypso_block:4@4096` | 120303 | 216 | 0.01 | 12484 [12477-12484] | 5.67 [5.32-5.72] | 2.90 [2.87-3.00] | 0 | 0.86 | 1.00 | 0.86 | 100% |
| `kalypso_block:8@4096` | 120303 | 216 | 0.01 | 12451 [12445-12452] | 5.64 [4.75-5.66] | 3.04 [2.96-4.90] | 0 | 0.86 | 1.00 | 0.86 | 100% |
| `kalypso_block:16@4096` | 120303 | 216 | 0.01 | 12451 [12445-12452] | 5.67 [5.28-5.81] | 2.91 [2.84-3.71] | 0 | 0.86 | 1.00 | 0.86 | 100% |
| `kalypso_block:32@4096` | 120303 | 216 | 0.01 | 12451 [12445-12452] | 5.63 [5.54-5.78] | 2.94 [2.85-3.23] | 0 | 0.86 | 1.00 | 0.86 | 100% |


Stages (medians over repeats; times from query start):

| Arm | Stage | Operators | Block | Tasks | Rows in | Rows out | Out/in | Occupancy | Start (s) | End (s) | Overlap (s) | Fallbacks |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 0 | SemFilter | 1 | 64 | 64 | 64 | 1.00 | n/a | 0.21 [0.21-0.30] | 10.38 [10.30-10.56] | n/a | 0 |
| `kalypso_block:2` | 0 | BlockSemFilter | 2 | 32 | 64 | 54 | 0.84 | 1.00 | 0.23 [0.22-0.24] | 10.59 [10.47-10.62] | n/a | 0 |
| `kalypso_block:4` | 0 | BlockSemFilter | 4 | 16 | 64 | 55 | 0.86 | 1.00 | 0.23 [0.22-0.24] | 11.32 [11.30-11.55] | n/a | 0 |
| `kalypso_block:8` | 0 | BlockSemFilter | 8 | 8 | 64 | 60 | 0.94 | 1.00 | 0.24 [0.23-0.25] | 12.95 [12.88-13.35] | n/a | 0 |
| `kalypso_block:2@4096` | 0 | BlockSemFilter | 2 | 42 | 64 | 54 | 0.84 | 0.76 | 0.80 [0.78-0.81] | 10.84 [10.73-11.34] | n/a | 20 |
| `kalypso_block:4@4096` | 0 | BlockSemFilter | 4 | 39 | 64 | 55 | 0.86 | 0.41 | 1.17 [1.15-1.21] | 11.28 [11.17-11.97] | n/a | 20 |
| `kalypso_block:8@4096` | 0 | BlockSemFilter | 8 | 39 | 64 | 55 | 0.86 | 0.21 | 1.30 [1.21-2.74] | 11.34 [11.29-13.47] | n/a | 20 |
| `kalypso_block:16@4096` | 0 | BlockSemFilter | 16 | 39 | 64 | 55 | 0.86 | 0.10 | 1.17 [1.15-1.84] | 11.28 [11.00-12.12] | n/a | 20 |
| `kalypso_block:32@4096` | 0 | BlockSemFilter | 32 | 39 | 64 | 55 | 0.86 | 0.05 | 1.19 [1.15-1.49] | 11.36 [11.06-11.53] | n/a | 20 |

