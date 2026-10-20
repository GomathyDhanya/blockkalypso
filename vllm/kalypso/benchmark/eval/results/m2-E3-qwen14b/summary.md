# E3

## n16: N = 16

Model `Qwen/Qwen2.5-14B-Instruct` on NVIDIA L40S (Linux-4.19.0-gvisor-x86_64-with-glibc2.39); workload `arxiv_filter` (filter), 16 rows, 8 positive; 3 repeats after 1 warmup; prefix cache reset per trial: yes; git `73d49d6a7`.
Server: `python3 -m vllm.entrypoints.cli.main serve Qwen/Qwen2.5-14B-Instruct --host 127.0.0.1 --port 8000 --enable-prefix-caching --no-async-scheduling --api-server-count 1 --max-model-len 16384` (started Tue Oct 20 19:55:01 2026).

| Arm | Completion time (s) | Speedup vs kalypso_per_tuple | LLM calls [retries + fallbacks] | Agrees w/ kalypso_per_tuple | F1 vs labels | Peak KV % |
|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 1.09 [1.09-1.10] | 1.00x | 16 [0] | 100% | 1.00 | 10.6 |
| `kalypso_block:4` | 1.28 [1.28-1.28] | 0.85x | 4 [0] | 100% | 1.00 | 8.2 |
| `kalypso_block:16` | 3.51 [3.50-3.51] | 0.31x | 1 [0] | 100% | 1.00 | 8.0 |


Diagnostics:

| Arm | Prompt tokens | Gen tokens | Prefix hit rate | Peak RSS (MiB) | Items/s | First output (s) | Truncated | Accuracy | Precision | Recall | Self-agreement |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 6049 | 128 | 0.12 | 11527 [11527-11527] | 14.62 [14.57-14.66] | 1.01 [1.00-1.01] | 16 | 1.00 | 1.00 | 1.00 | 100% |
| `kalypso_block:4` | 4313 | 64 | 0.02 | 11527 [11527-11527] | 12.49 [12.47-12.51] | 1.19 [1.19-1.20] | 0 | 1.00 | 1.00 | 1.00 | 100% |
| `kalypso_block:16` | 4076 | 71 | 0.00 | 11527 [11527-11527] | 4.56 [4.56-4.57] | 3.50 [3.50-3.50] | 0 | 1.00 | 1.00 | 1.00 | 100% |


Stages (medians over repeats; times from query start):

| Arm | Stage | Operators | Block | Tasks | Rows in | Rows out | Out/in | Occupancy | Start (s) | End (s) | Overlap (s) | Fallbacks |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 0 | SemFilter | 1 | 16 | 16 | 8 | 0.50 | n/a | 0.01 [0.01-0.01] | 1.09 [1.09-1.10] | n/a | 0 |
| `kalypso_block:4` | 0 | BlockSemFilter | 4 | 4 | 16 | 8 | 0.50 | 1.00 | 0.01 [0.01-0.01] | 1.28 [1.28-1.28] | n/a | 0 |
| `kalypso_block:16` | 0 | BlockSemFilter | 16 | 1 | 16 | 8 | 0.50 | 1.00 | 0.02 [0.02-0.02] | 3.50 [3.50-3.50] | n/a | 0 |


## n32: N = 32

Model `Qwen/Qwen2.5-14B-Instruct` on NVIDIA L40S (Linux-4.19.0-gvisor-x86_64-with-glibc2.39); workload `arxiv_filter` (filter), 32 rows, 16 positive; 3 repeats after 1 warmup; prefix cache reset per trial: yes; git `73d49d6a7`.
Server: `python3 -m vllm.entrypoints.cli.main serve Qwen/Qwen2.5-14B-Instruct --host 127.0.0.1 --port 8000 --enable-prefix-caching --no-async-scheduling --api-server-count 1 --max-model-len 16384` (started Tue Oct 20 19:55:01 2026).

| Arm | Completion time (s) | Speedup vs kalypso_per_tuple | LLM calls [retries + fallbacks] | Agrees w/ kalypso_per_tuple | F1 vs labels | Peak KV % |
|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 1.96 [1.96-1.97] | 1.00x | 32 [0] | 100% | 1.00 | 21.3 [21.0-21.3] |
| `kalypso_block:4` | 1.96 [1.94-1.96] | 1.00x | 8 [0] | 100% | 1.00 | 16.8 |
| `kalypso_block:16` | 4.40 [4.38-4.40] | 0.44x | 2 [0] | 100% | 1.00 | 16.2 |


Diagnostics:

| Arm | Prompt tokens | Gen tokens | Prefix hit rate | Peak RSS (MiB) | Items/s | First output (s) | Truncated | Accuracy | Precision | Recall | Self-agreement |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 12276 | 256 | 0.12 | 11528 [11528-11528] | 16.36 [16.28-16.37] | 1.73 [1.73-1.74] | 32 | 1.00 | 1.00 | 1.00 | 100% |
| `kalypso_block:4` | 8805 | 128 | 0.03 | 11528 [11528-11528] | 16.37 [16.36-16.52] | 1.78 [1.76-1.78] | 0 | 1.00 | 1.00 | 1.00 | 100% |
| `kalypso_block:16` | 8331 | 142 | 0.00 | 11528 [11528-11529] | 7.28 [7.28-7.30] | 4.27 [4.26-4.27] | 0 | 1.00 | 1.00 | 1.00 | 100% |


Stages (medians over repeats; times from query start):

| Arm | Stage | Operators | Block | Tasks | Rows in | Rows out | Out/in | Occupancy | Start (s) | End (s) | Overlap (s) | Fallbacks |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 0 | SemFilter | 1 | 32 | 32 | 16 | 0.50 | n/a | 0.03 [0.03-0.03] | 1.95 [1.95-1.96] | n/a | 0 |
| `kalypso_block:4` | 0 | BlockSemFilter | 4 | 8 | 32 | 16 | 0.50 | 1.00 | 0.02 [0.02-0.02] | 1.95 [1.93-1.95] | n/a | 0 |
| `kalypso_block:16` | 0 | BlockSemFilter | 16 | 2 | 32 | 16 | 0.50 | 1.00 | 0.03 [0.03-0.03] | 4.39 [4.38-4.39] | n/a | 0 |


## n64: N = 64

Model `Qwen/Qwen2.5-14B-Instruct` on NVIDIA L40S (Linux-4.19.0-gvisor-x86_64-with-glibc2.39); workload `arxiv_filter` (filter), 64 rows, 32 positive; 3 repeats after 1 warmup; prefix cache reset per trial: yes; git `73d49d6a7`.
Server: `python3 -m vllm.entrypoints.cli.main serve Qwen/Qwen2.5-14B-Instruct --host 127.0.0.1 --port 8000 --enable-prefix-caching --no-async-scheduling --api-server-count 1 --max-model-len 16384` (started Tue Oct 20 19:55:01 2026).

| Arm | Completion time (s) | Speedup vs kalypso_per_tuple | LLM calls [retries + fallbacks] | Agrees w/ kalypso_per_tuple | F1 vs labels | Peak KV % |
|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 3.63 [3.63-3.65] | 1.00x | 64 [0] | 100% | 1.00 | 28.6 |
| `kalypso_block:4` | 3.29 [3.28-3.31] | 1.10x | 16 [0] | 100% | 1.00 | 33.3 |
| `kalypso_block:16` | 5.88 [5.87-5.88] | 0.62x | 4 [0] | 98% | 0.98 | 32.2 |


Diagnostics:

| Arm | Prompt tokens | Gen tokens | Prefix hit rate | Peak RSS (MiB) | Items/s | First output (s) | Truncated | Accuracy | Precision | Recall | Self-agreement |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 24503 | 512 | 0.12 | 11533 [11533-11533] | 17.63 [17.53-17.64] | 2.24 [2.22-2.26] | 64 | 1.00 | 1.00 | 1.00 | 100% |
| `kalypso_block:4` | 17561 | 256 | 0.03 | 11533 [11533-11533] | 19.42 [19.32-19.52] | 2.94 [2.92-2.96] | 0 | 1.00 | 1.00 | 1.00 | 100% |
| `kalypso_block:16` | 16613 | 284 | 0.01 | 11533 [11533-11533] | 10.89 [10.89-10.91] | 5.57 [5.56-5.57] | 0 | 0.98 | 1.00 | 0.97 | 100% |


Stages (medians over repeats; times from query start):

| Arm | Stage | Operators | Block | Tasks | Rows in | Rows out | Out/in | Occupancy | Start (s) | End (s) | Overlap (s) | Fallbacks |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 0 | SemFilter | 1 | 64 | 64 | 32 | 0.50 | n/a | 0.04 [0.04-0.04] | 3.63 [3.62-3.65] | n/a | 0 |
| `kalypso_block:4` | 0 | BlockSemFilter | 4 | 16 | 64 | 32 | 0.50 | 1.00 | 0.04 [0.04-0.04] | 3.29 [3.27-3.31] | n/a | 0 |
| `kalypso_block:16` | 0 | BlockSemFilter | 16 | 4 | 64 | 31 | 0.48 | 1.00 | 0.04 [0.04-0.04] | 5.87 [5.86-5.87] | n/a | 0 |


## n128: N = 128

Model `Qwen/Qwen2.5-14B-Instruct` on NVIDIA L40S (Linux-4.19.0-gvisor-x86_64-with-glibc2.39); workload `arxiv_filter` (filter), 128 rows, 64 positive; 3 repeats after 1 warmup; prefix cache reset per trial: yes; git `73d49d6a7`.
Server: `python3 -m vllm.entrypoints.cli.main serve Qwen/Qwen2.5-14B-Instruct --host 127.0.0.1 --port 8000 --enable-prefix-caching --no-async-scheduling --api-server-count 1 --max-model-len 16384` (started Tue Oct 20 19:55:01 2026).

| Arm | Completion time (s) | Speedup vs kalypso_per_tuple | LLM calls [retries + fallbacks] | Agrees w/ kalypso_per_tuple | F1 vs labels | Peak KV % |
|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 6.99 [6.95-7.04] | 1.00x | 128 [0] | 100% | 0.99 | 28.7 |
| `kalypso_block:4` | 6.05 [6.04-6.08] | 1.16x | 32 [0] | 98% | 0.98 | 59.4 |
| `kalypso_block:16` | 8.82 [8.81-8.87] | 0.79x | 8 [0] | 97% | 0.96 | 63.8 [63.7-63.8] |


Diagnostics:

| Arm | Prompt tokens | Gen tokens | Prefix hit rate | Peak RSS (MiB) | Items/s | First output (s) | Truncated | Accuracy | Precision | Recall | Self-agreement |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 48666 | 1022 | 0.13 | 11538 [11538-11538] | 18.31 [18.17-18.41] | 2.34 [2.32-2.36] | 127 | 0.99 | 1.00 | 0.98 | 100% |
| `kalypso_block:4` | 34781 | 512 | 0.03 | 11538 [11538-11538] | 21.16 [21.06-21.21] | 4.96 [4.95-4.99] | 0 | 0.98 | 1.00 | 0.95 | 100% |
| `kalypso_block:16` | 32885 | 568 | 0.01 | 11538 [11538-11538] | 14.50 [14.42-14.53] | 8.23 [8.21-8.27] | 0 | 0.96 | 1.00 | 0.92 | 100% |


Stages (medians over repeats; times from query start):

| Arm | Stage | Operators | Block | Tasks | Rows in | Rows out | Out/in | Occupancy | Start (s) | End (s) | Overlap (s) | Fallbacks |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 0 | SemFilter | 1 | 128 | 128 | 63 | 0.49 | n/a | 0.06 [0.06-0.06] | 6.98 [6.94-7.04] | n/a | 0 |
| `kalypso_block:4` | 0 | BlockSemFilter | 4 | 32 | 128 | 61 | 0.48 | 1.00 | 0.05 [0.05-0.05] | 6.04 [6.03-6.07] | n/a | 0 |
| `kalypso_block:16` | 0 | BlockSemFilter | 16 | 8 | 128 | 59 | 0.46 | 1.00 | 0.05 [0.05-0.06] | 8.82 [8.81-8.87] | n/a | 0 |

