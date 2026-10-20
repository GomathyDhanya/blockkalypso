# E1

## short_arxiv: short rows (arXiv)

Model `Qwen/Qwen2.5-14B-Instruct` on NVIDIA L40S (Linux-4.19.0-gvisor-x86_64-with-glibc2.39); workload `arxiv_filter` (filter), 64 rows, 32 positive; 3 repeats after 1 warmup; prefix cache reset per trial: yes; git `73d49d6a7`.
Server: `python3 -m vllm.entrypoints.cli.main serve Qwen/Qwen2.5-14B-Instruct --host 127.0.0.1 --port 8000 --enable-prefix-caching --no-async-scheduling --api-server-count 1 --max-model-len 32768` (started Tue Oct 20 19:54:49 2026).

| Arm | Completion time (s) | Speedup vs kalypso_per_tuple | LLM calls [retries + fallbacks] | Agrees w/ kalypso_per_tuple | F1 vs labels | Peak KV % |
|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 3.65 [3.64-3.67] | 1.00x | 64 [0] | 100% | 1.00 | 28.6 |
| `kalypso_block:2` | 3.05 [3.04-3.07] | 1.20x | 32 [0] | 95% | 0.95 | 28.5 |
| `kalypso_block:4` | 3.31 [3.29-3.31] | 1.10x | 16 [0] | 100% | 1.00 | 33.3 |
| `kalypso_block:8` | 4.02 [3.98-4.03] | 0.91x | 8 [0] | 100% | 1.00 | 32.6 |
| `kalypso_block:16` | 5.89 [5.88-5.92] | 0.62x | 4 [0] | 98% | 0.98 | 32.2 |
| `kalypso_block:32` | 9.77 [9.75-9.77] | 0.37x | 2 [0] | 95% | 0.95 | 32.1 |
| `kalypso_block:2@4096` | 3.13 [3.12-3.15] | 1.17x | 32 [0] | 95% | 0.95 | 28.5 |
| `kalypso_block:4@4096` | 3.44 [3.43-3.47] | 1.06x | 16 [0] | 100% | 1.00 | 33.3 |
| `kalypso_block:8@4096` | 4.28 [4.28-4.29] | 0.85x | 8 [0] | 100% | 1.00 | 32.6 |
| `kalypso_block:16@4096` | 6.23 [6.23-6.26] | 0.59x | 5 [0] | 100% | 1.00 | 32.1 [32.1-32.1] |
| `kalypso_block:32@4096` | 6.25 [6.24-6.25] | 0.58x | 5 [0] | 100% | 1.00 | 32.1 |


Diagnostics:

| Arm | Prompt tokens | Gen tokens | Prefix hit rate | Peak RSS (MiB) | Items/s | First output (s) | Truncated | Accuracy | Precision | Recall | Self-agreement |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 24503 | 512 | 0.12 | 11538 [11536-11539] | 17.52 [17.44-17.57] | 2.25 [2.24-2.27] | 64 | 1.00 | 1.00 | 1.00 | 100% |
| `kalypso_block:2` | 18873 | 256 | 0.05 | 11538 [11536-11539] | 21.01 [20.87-21.06] | 2.25 [2.24-2.27] | 0 | 0.95 | 1.00 | 0.91 | 100% |
| `kalypso_block:4` | 17561 | 256 | 0.03 | 11538 [11536-11539] | 19.34 [19.31-19.45] | 2.95 [2.93-2.96] | 0 | 1.00 | 1.00 | 1.00 | 100% |
| `kalypso_block:8` | 16905 | 256 | 0.01 | 11538 [11536-11539] | 15.92 [15.87-16.06] | 3.71 [3.67-3.72] | 0 | 1.00 | 1.00 | 1.00 | 100% |
| `kalypso_block:16` | 16613 | 284 | 0.01 | 11538 [11536-11539] | 10.87 [10.82-10.89] | 5.58 [5.57-5.60] | 0 | 0.98 | 1.00 | 0.97 | 100% |
| `kalypso_block:32` | 16463 | 302 | 0.00 | 11539 [11537-11539] | 6.55 [6.55-6.56] | 9.59 [9.57-9.59] | 0 | 0.95 | 1.00 | 0.91 | 100% |
| `kalypso_block:2@4096` | 18873 | 256 | 0.05 | 11539 [11537-11539] | 20.47 [20.31-20.55] | 2.33 [2.32-2.35] | 0 | 0.95 | 1.00 | 0.91 | 100% |
| `kalypso_block:4@4096` | 17561 | 256 | 0.03 | 11539 [11537-11539] | 18.59 [18.44-18.66] | 3.08 [3.07-3.11] | 0 | 1.00 | 1.00 | 1.00 | 100% |
| `kalypso_block:8@4096` | 16905 | 256 | 0.01 | 11539 [11537-11539] | 14.95 [14.90-14.96] | 3.97 [3.96-3.98] | 0 | 1.00 | 1.00 | 1.00 | 100% |
| `kalypso_block:16@4096` | 16690 | 279 | 0.01 | 11539 [11537-11539] | 10.27 [10.23-10.27] | 3.98 [3.97-4.00] | 0 | 1.00 | 1.00 | 1.00 | 100% |
| `kalypso_block:32@4096` | 16690 | 279 | 0.01 | 11539 [11537-11539] | 10.24 [10.23-10.25] | 4.00 [3.99-4.00] | 0 | 1.00 | 1.00 | 1.00 | 100% |


Stages (medians over repeats; times from query start):

| Arm | Stage | Operators | Block | Tasks | Rows in | Rows out | Out/in | Occupancy | Start (s) | End (s) | Overlap (s) | Fallbacks |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 0 | SemFilter | 1 | 64 | 64 | 32 | 0.50 | n/a | 0.05 [0.05-0.05] | 3.65 [3.64-3.67] | n/a | 0 |
| `kalypso_block:2` | 0 | BlockSemFilter | 2 | 32 | 64 | 29 | 0.45 | 1.00 | 0.04 [0.04-0.04] | 3.04 [3.03-3.06] | n/a | 0 |
| `kalypso_block:4` | 0 | BlockSemFilter | 4 | 16 | 64 | 32 | 0.50 | 1.00 | 0.04 [0.04-0.04] | 3.31 [3.29-3.31] | n/a | 0 |
| `kalypso_block:8` | 0 | BlockSemFilter | 8 | 8 | 64 | 32 | 0.50 | 1.00 | 0.04 [0.04-0.04] | 4.02 [3.98-4.03] | n/a | 0 |
| `kalypso_block:16` | 0 | BlockSemFilter | 16 | 4 | 64 | 31 | 0.48 | 1.00 | 0.04 [0.04-0.04] | 5.88 [5.87-5.91] | n/a | 0 |
| `kalypso_block:32` | 0 | BlockSemFilter | 32 | 2 | 64 | 29 | 0.45 | 1.00 | 0.05 [0.05-0.05] | 9.76 [9.75-9.77] | n/a | 0 |
| `kalypso_block:2@4096` | 0 | BlockSemFilter | 2 | 32 | 64 | 29 | 0.45 | 1.00 | 0.11 [0.11-0.11] | 3.12 [3.11-3.15] | n/a | 0 |
| `kalypso_block:4@4096` | 0 | BlockSemFilter | 4 | 16 | 64 | 32 | 0.50 | 1.00 | 0.18 [0.17-0.19] | 3.44 [3.42-3.47] | n/a | 0 |
| `kalypso_block:8@4096` | 0 | BlockSemFilter | 8 | 8 | 64 | 32 | 0.50 | 1.00 | 0.30 [0.29-0.31] | 4.28 [4.27-4.29] | n/a | 0 |
| `kalypso_block:16@4096` | 0 | BlockSemFilter | 16 | 5 | 64 | 32 | 0.50 | 0.80 | 0.49 [0.49-0.49] | 6.23 [6.23-6.25] | n/a | 0 |
| `kalypso_block:32@4096` | 0 | BlockSemFilter | 32 | 5 | 64 | 32 | 0.50 | 0.40 | 0.50 [0.50-0.51] | 6.25 [6.24-6.25] | n/a | 0 |


## medium_medec: medium rows (MEDEC)

Model `Qwen/Qwen2.5-14B-Instruct` on NVIDIA L40S (Linux-4.19.0-gvisor-x86_64-with-glibc2.39); workload `medec_full` (filter), 64 rows, 35 positive; 3 repeats after 1 warmup; prefix cache reset per trial: yes; git `73d49d6a7`.
Server: `python3 -m vllm.entrypoints.cli.main serve Qwen/Qwen2.5-14B-Instruct --host 127.0.0.1 --port 8000 --enable-prefix-caching --no-async-scheduling --api-server-count 1 --max-model-len 32768` (started Tue Oct 20 19:54:49 2026).

| Arm | Completion time (s) | Speedup vs kalypso_per_tuple | LLM calls [retries + fallbacks] | Agrees w/ kalypso_per_tuple | F1 vs labels | Peak KV % |
|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 6.15 [6.15-6.18] | 1.00x | 64 [0] | 100% | 0.71 | 29.3 |
| `kalypso_block:2` | 4.71 [4.71-4.73] | 1.30x | 32 [0] | 50% | 0.51 | 31.4 |
| `kalypso_block:4` | 5.05 [5.03-5.05] | 1.22x | 16 [0] | 47% | 0.55 | 53.4 |
| `kalypso_block:8` | 5.92 [5.86-5.92] | 1.04x | 8 [0] | 47% | 0.51 | 52.9 |
| `kalypso_block:16` | 8.07 [8.02-8.10] | 0.76x | 4 [0] | 41% | 0.49 | 52.8 |
| `kalypso_block:32` | 12.48 [12.41-12.54] | 0.49x | 2 [0] | 38% | 0.40 | 52.7 |
| `kalypso_block:2@4096` | 4.83 [4.81-5.00] | 1.27x | 32 [0] | 50% | 0.51 | 31.4 |
| `kalypso_block:4@4096` | 5.27 [5.25-5.45] | 1.17x | 16 [0] | 47% | 0.55 | 53.4 |
| `kalypso_block:8@4096` | 6.30 [6.29-6.42] | 0.98x | 9 [1] | 44% | 0.45 | 52.9 |
| `kalypso_block:16@4096` | 6.74 [6.72-6.80] | 0.91x | 8 [0] | 39% | 0.50 | 53.0 |
| `kalypso_block:32@4096` | 6.73 [6.72-6.74] | 0.91x | 8 [0] | 39% | 0.50 | 53.0 |


Diagnostics:

| Arm | Prompt tokens | Gen tokens | Prefix hit rate | Peak RSS (MiB) | Items/s | First output (s) | Truncated | Accuracy | Precision | Recall | Self-agreement |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 40837 | 512 | 0.07 | 11544 [11543-11544] | 10.41 [10.36-10.41] | 2.36 [2.36-2.36] | 64 | 0.61 | 0.60 | 0.86 | 100% |
| `kalypso_block:2` | 32635 | 259 | 0.14 | 11544 [11543-11544] | 13.58 [13.52-13.59] | 2.40 [2.38-2.41] | 0 | 0.58 | 0.70 | 0.40 | 100% |
| `kalypso_block:4` | 29771 | 256 | 0.07 | 11544 [11543-11544] | 12.68 [12.66-12.72] | 4.49 [4.47-4.50] | 0 | 0.61 | 0.75 | 0.43 | 100% |
| `kalypso_block:8` | 28339 | 256 | 0.04 | 11544 [11543-11544] | 10.81 [10.80-10.92] | 5.36 [5.31-5.38] | 0 | 0.58 | 0.70 | 0.40 | 100% |
| `kalypso_block:16` | 27659 | 284 | 0.02 | 11544 [11543-11544] | 7.93 [7.90-7.98] | 7.62 [7.57-7.64] | 0 | 0.61 | 0.86 | 0.34 | 100% |
| `kalypso_block:32` | 27315 | 301 | 0.01 | 11544 [11544-11544] | 5.13 [5.10-5.16] | 12.12 [12.06-12.19] | 0 | 0.58 | 0.90 | 0.26 | 100% |
| `kalypso_block:2@4096` | 32635 | 259 | 0.14 | 11544 [11544-11544] | 13.25 [12.80-13.31] | 2.50 [2.48-2.59] | 0 | 0.58 | 0.70 | 0.40 | 100% |
| `kalypso_block:4@4096` | 29771 | 256 | 0.07 | 11544 [11544-11544] | 12.14 [11.74-12.19] | 4.71 [4.70-4.89] | 0 | 0.61 | 0.75 | 0.43 | 100% |
| `kalypso_block:8@4096` | 28482 | 256 | 0.04 | 11544 [11544-11544] | 10.16 [9.97-10.17] | 5.76 [5.75-5.88] | 1 | 0.55 | 0.67 | 0.34 | 100% |
| `kalypso_block:16@4096` | 28342 | 257 | 0.04 | 11544 [11544-11544] | 9.50 [9.42-9.52] | 5.96 [5.93-6.01] | 0 | 0.59 | 0.76 | 0.37 | 100% |
| `kalypso_block:32@4096` | 28342 | 257 | 0.04 | 11544 [11544-11544] | 9.51 [9.50-9.53] | 5.95 [5.94-5.95] | 0 | 0.59 | 0.76 | 0.37 | 100% |


Stages (medians over repeats; times from query start):

| Arm | Stage | Operators | Block | Tasks | Rows in | Rows out | Out/in | Occupancy | Start (s) | End (s) | Overlap (s) | Fallbacks |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 0 | SemFilter | 1 | 64 | 64 | 50 | 0.78 | n/a | 0.10 [0.10-0.10] | 6.14 [6.14-6.17] | n/a | 0 |
| `kalypso_block:2` | 0 | BlockSemFilter | 2 | 32 | 64 | 20 | 0.31 | 1.00 | 0.09 [0.09-0.09] | 4.71 [4.70-4.73] | n/a | 0 |
| `kalypso_block:4` | 0 | BlockSemFilter | 4 | 16 | 64 | 20 | 0.31 | 1.00 | 0.08 [0.08-0.09] | 5.04 [5.03-5.05] | n/a | 0 |
| `kalypso_block:8` | 0 | BlockSemFilter | 8 | 8 | 64 | 20 | 0.31 | 1.00 | 0.09 [0.08-0.09] | 5.91 [5.86-5.92] | n/a | 0 |
| `kalypso_block:16` | 0 | BlockSemFilter | 16 | 4 | 64 | 14 | 0.22 | 1.00 | 0.09 [0.09-0.09] | 8.07 [8.02-8.09] | n/a | 0 |
| `kalypso_block:32` | 0 | BlockSemFilter | 32 | 2 | 64 | 10 | 0.16 | 1.00 | 0.10 [0.09-0.10] | 12.48 [12.41-12.54] | n/a | 0 |
| `kalypso_block:2@4096` | 0 | BlockSemFilter | 2 | 32 | 64 | 20 | 0.31 | 1.00 | 0.19 [0.18-0.19] | 4.83 [4.80-5.00] | n/a | 0 |
| `kalypso_block:4@4096` | 0 | BlockSemFilter | 4 | 16 | 64 | 20 | 0.31 | 1.00 | 0.29 [0.29-0.29] | 5.27 [5.25-5.44] | n/a | 0 |
| `kalypso_block:8@4096` | 0 | BlockSemFilter | 8 | 9 | 64 | 18 | 0.28 | 0.89 | 0.45 [0.45-0.45] | 6.30 [6.29-6.42] | n/a | 1 |
| `kalypso_block:16@4096` | 0 | BlockSemFilter | 16 | 8 | 64 | 17 | 0.27 | 0.50 | 0.55 [0.53-0.56] | 6.73 [6.72-6.79] | n/a | 0 |
| `kalypso_block:32@4096` | 0 | BlockSemFilter | 32 | 8 | 64 | 17 | 0.27 | 0.25 | 0.55 [0.54-0.55] | 6.73 [6.71-6.73] | n/a | 0 |


## long_contracts: long rows (ContractNLI)

Model `Qwen/Qwen2.5-14B-Instruct` on NVIDIA L40S (Linux-4.19.0-gvisor-x86_64-with-glibc2.39); workload `contractnli_filter` (filter), 64 rows, 64 positive; 3 repeats after 1 warmup; prefix cache reset per trial: yes; git `73d49d6a7`.
Server: `python3 -m vllm.entrypoints.cli.main serve Qwen/Qwen2.5-14B-Instruct --host 127.0.0.1 --port 8000 --enable-prefix-caching --no-async-scheduling --api-server-count 1 --max-model-len 32768` (started Tue Oct 20 19:54:49 2026).

| Arm | Completion time (s) | Speedup vs kalypso_per_tuple | LLM calls [retries + fallbacks] | Agrees w/ kalypso_per_tuple | F1 vs labels | Peak KV % |
|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 19.81 [19.54-20.30] | 1.00x | 64 [0] | 100% | 1.00 | 35.0 |
| `kalypso_block:2` | 19.54 [19.30-19.96] | 1.01x | 32 [0] | 100% | 1.00 | 38.8 |
| `kalypso_block:4` | 21.57 [21.43-21.91] | 0.92x | 16 [0] | 95% | 0.98 | 78.5 |
| `kalypso_block:8` | 26.54 [26.47-27.11] | 0.75x | 8 [0] | 95% | 0.98 [0.97-0.98] | 83.1 |
| `kalypso_block:16` | failed in all 3 trials | | | | | |
| `kalypso_block:32` | failed in all 3 trials | | | | | |
| `kalypso_block:2@4096` | 19.55 [19.43-19.84] | 1.01x | 42 [20] | 98% | 0.99 | 34.1 |
| `kalypso_block:4@4096` | 19.73 [19.69-20.37] | 1.00x | 39 [20] | 100% | 1.00 | 38.7 |
| `kalypso_block:8@4096` | 20.22 [20.00-20.81] | 0.98x | 39 [20] | 100% | 1.00 | 38.7 |
| `kalypso_block:16@4096` | 20.14 [20.11-20.19] | 0.98x | 39 [20] | 100% | 1.00 | 38.7 |
| `kalypso_block:32@4096` | 19.86 [19.82-20.01] | 1.00x | 39 [20] | 100% | 1.00 | 38.7 |

- `kalypso_block:16` failed 3 trial(s): ArmError: /v1/semantic/query returned 400: {"error":{"message":"This model's maximum context length is 32768 tokens. However, you requested 112 output tokens and your prompt contains at least 32657 input tokens, for a total of at least 32769 tokens. Please reduce the length of the input prompt or th
- `kalypso_block:32` failed 3 trial(s): ArmError: /v1/semantic/query returned 400: {"error":{"message":"This model's maximum context length is 32768 tokens. However, you requested 208 output tokens and your prompt contains at least 32561 input tokens, for a total of at least 32769 tokens. Please reduce the length of the input prompt or th


Diagnostics:

| Arm | Prompt tokens | Gen tokens | Prefix hit rate | Peak RSS (MiB) | Items/s | First output (s) | Truncated | Accuracy | Precision | Recall | Self-agreement |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 126767 | 512 | 0.03 | 11599 [11598-11599] | 3.23 [3.15-3.28] | 2.80 [2.77-2.88] | 64 | 1.00 | 1.00 | 1.00 | 100% |
| `kalypso_block:2` | 120386 | 256 | 0.01 | 11599 [11598-11599] | 3.27 [3.21-3.32] | 3.57 [3.55-3.64] | 0 | 1.00 | 1.00 | 1.00 | 100% |
| `kalypso_block:4` | 118933 | 256 | 0.01 | 11599 [11598-11599] | 2.97 [2.92-2.99] | 7.25 [7.24-7.42] | 0 | 0.95 | 1.00 | 0.95 | 100% |
| `kalypso_block:8` | 118205 | 256 | 0.00 | 11599 [11598-11599] | 2.41 [2.36-2.42] | 9.58 [9.56-9.82] | 0 | 0.95 [0.94-0.95] | 1.00 | 0.95 [0.94-0.95] | 99% |
| `kalypso_block:2@4096` | 120575 | 216 | 0.01 | 11599 [11599-11599] | 3.27 [3.23-3.29] | 3.82 [3.72-3.88] | 0 | 0.98 | 1.00 | 0.98 | 100% |
| `kalypso_block:4@4096` | 120303 | 216 | 0.01 | 11599 [11599-11599] | 3.24 [3.14-3.25] | 4.08 [4.03-4.10] | 0 | 1.00 | 1.00 | 1.00 | 100% |
| `kalypso_block:8@4096` | 120303 | 216 | 0.01 | 11599 [11599-11599] | 3.17 [3.08-3.20] | 4.18 [4.16-4.18] | 0 | 1.00 | 1.00 | 1.00 | 100% |
| `kalypso_block:16@4096` | 120303 | 216 | 0.01 | 11599 [11599-11599] | 3.18 [3.17-3.18] | 4.18 [4.14-4.18] | 0 | 1.00 | 1.00 | 1.00 | 100% |
| `kalypso_block:32@4096` | 120303 | 216 | 0.01 | 11599 [11599-11599] | 3.22 [3.20-3.23] | 4.09 [4.08-4.10] | 0 | 1.00 | 1.00 | 1.00 | 100% |


Stages (medians over repeats; times from query start):

| Arm | Stage | Operators | Block | Tasks | Rows in | Rows out | Out/in | Occupancy | Start (s) | End (s) | Overlap (s) | Fallbacks |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 0 | SemFilter | 1 | 64 | 64 | 64 | 1.00 | n/a | 0.20 [0.20-0.21] | 19.80 [19.52-20.29] | n/a | 0 |
| `kalypso_block:2` | 0 | BlockSemFilter | 2 | 32 | 64 | 64 | 1.00 | 1.00 | 0.23 [0.22-0.23] | 19.53 [19.29-19.95] | n/a | 0 |
| `kalypso_block:4` | 0 | BlockSemFilter | 4 | 16 | 64 | 61 | 0.95 | 1.00 | 0.24 [0.22-0.24] | 21.56 [21.42-21.89] | n/a | 0 |
| `kalypso_block:8` | 0 | BlockSemFilter | 8 | 8 | 64 | 61 [60-61] | 0.95 [0.94-0.95] | 1.00 | 0.25 [0.24-0.26] | 26.52 [26.46-27.10] | n/a | 0 |
| `kalypso_block:2@4096` | 0 | BlockSemFilter | 2 | 42 | 64 | 63 | 0.98 | 0.76 | 0.85 [0.81-0.95] | 19.54 [19.42-19.83] | n/a | 20 |
| `kalypso_block:4@4096` | 0 | BlockSemFilter | 4 | 39 | 64 | 64 | 1.00 | 0.41 | 1.16 [1.14-1.19] | 19.72 [19.67-20.36] | n/a | 20 |
| `kalypso_block:8@4096` | 0 | BlockSemFilter | 8 | 39 | 64 | 64 | 1.00 | 0.21 | 1.24 [1.16-1.27] | 20.21 [19.99-20.80] | n/a | 20 |
| `kalypso_block:16@4096` | 0 | BlockSemFilter | 16 | 39 | 64 | 64 | 1.00 | 0.10 | 1.21 [1.19-1.24] | 20.13 [20.10-20.18] | n/a | 20 |
| `kalypso_block:32@4096` | 0 | BlockSemFilter | 32 | 39 | 64 | 64 | 1.00 | 0.05 | 1.18 [1.18-1.19] | 19.85 [19.81-20.00] | n/a | 20 |

