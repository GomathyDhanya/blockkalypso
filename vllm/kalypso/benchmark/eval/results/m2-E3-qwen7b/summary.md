# E3

## n16: N = 16

Model `Qwen/Qwen2.5-7B-Instruct` on NVIDIA L40S (Linux-4.19.0-gvisor-x86_64-with-glibc2.39); workload `arxiv_filter` (filter), 16 rows, 8 positive; 3 repeats after 1 warmup; prefix cache reset per trial: yes; git `d19f407a0` (dirty).
Server: `python3 -m vllm.entrypoints.cli.main serve Qwen/Qwen2.5-7B-Instruct --host 127.0.0.1 --port 8000 --enable-prefix-caching --no-async-scheduling --api-server-count 1 --max-model-len 16384` (started Tue Oct 20 15:59:43 2026).

| Arm | Completion time (s) | Speedup vs kalypso_per_tuple | LLM calls [retries + fallbacks] | Agrees w/ kalypso_per_tuple | F1 vs labels | Peak KV % |
|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 0.46 [0.46-0.47] | 1.00x | 16 [0] | 100% | 1.00 | 0.9 |
| `kalypso_block:4` | 0.67 [0.67-0.67] | 0.69x | 4 [0] | 94% | 0.93 | 0.9 [0.9-0.9] |
| `kalypso_block:16` | 1.85 [1.84-1.86] | 0.25x | 1 [0] | 81% | 0.77 | 0.9 |


Diagnostics:

| Arm | Prompt tokens | Gen tokens | Prefix hit rate | Peak RSS (MiB) | Items/s | First output (s) | Truncated | Accuracy | Precision | Recall | Self-agreement |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 5913 | 48 | 0.12 | 12402 [12401-12402] | 34.71 [34.26-34.71] | 0.41 [0.41-0.42] | 0 | 1.00 | 1.00 | 1.00 | 100% |
| `kalypso_block:4` | 4177 | 64 | 0.02 | 12402 [12401-12402] | 23.92 [23.84-23.99] | 0.62 [0.62-0.62] | 0 | 0.94 | 1.00 | 0.88 | 100% |
| `kalypso_block:16` | 3940 | 71 | 0.00 | 12402 [12401-12402] | 8.63 [8.60-8.69] | 1.85 [1.84-1.86] | 0 | 0.81 | 1.00 | 0.62 | 100% |


Stages (medians over repeats; times from query start):

| Arm | Stage | Operators | Block | Tasks | Rows in | Rows out | Out/in | Occupancy | Start (s) | End (s) | Overlap (s) | Fallbacks |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 0 | SemFilter | 1 | 16 | 16 | 8 | 0.50 | n/a | 0.01 [0.01-0.01] | 0.46 [0.46-0.46] | n/a | 0 |
| `kalypso_block:4` | 0 | BlockSemFilter | 4 | 4 | 16 | 7 | 0.44 | 1.00 | 0.01 [0.01-0.01] | 0.67 [0.66-0.67] | n/a | 0 |
| `kalypso_block:16` | 0 | BlockSemFilter | 16 | 1 | 16 | 5 | 0.31 | 1.00 | 0.02 [0.02-0.02] | 1.85 [1.84-1.86] | n/a | 0 |


## n32: N = 32

Model `Qwen/Qwen2.5-7B-Instruct` on NVIDIA L40S (Linux-4.19.0-gvisor-x86_64-with-glibc2.39); workload `arxiv_filter` (filter), 32 rows, 16 positive; 3 repeats after 1 warmup; prefix cache reset per trial: yes; git `d19f407a0` (dirty).
Server: `python3 -m vllm.entrypoints.cli.main serve Qwen/Qwen2.5-7B-Instruct --host 127.0.0.1 --port 8000 --enable-prefix-caching --no-async-scheduling --api-server-count 1 --max-model-len 16384` (started Tue Oct 20 15:59:43 2026).

| Arm | Completion time (s) | Speedup vs kalypso_per_tuple | LLM calls [retries + fallbacks] | Agrees w/ kalypso_per_tuple | F1 vs labels | Peak KV % |
|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 0.90 [0.90-0.91] | 1.00x | 32 [0] | 100% | 1.00 | 1.0 |
| `kalypso_block:4` | 1.00 [1.00-1.02] | 0.90x | 8 [0] | 97% | 0.97 | 1.9 |
| `kalypso_block:16` | 2.26 [2.25-2.26] | 0.40x | 2 [0] | 75% | 0.69 | 1.8 |


Diagnostics:

| Arm | Prompt tokens | Gen tokens | Prefix hit rate | Peak RSS (MiB) | Items/s | First output (s) | Truncated | Accuracy | Precision | Recall | Self-agreement |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 12038 | 96 | 0.12 | 12404 [12404-12404] | 35.48 [35.01-35.59] | 0.53 [0.53-0.54] | 0 | 1.00 | 1.00 | 1.00 | 100% |
| `kalypso_block:4` | 8567 | 128 | 0.03 | 12404 [12404-12404] | 32.00 [31.46-32.03] | 0.91 [0.91-0.92] | 0 | 0.97 | 1.00 | 0.94 | 100% |
| `kalypso_block:16` | 8093 | 142 | 0.00 | 12404 [12404-12404] | 14.17 [14.13-14.20] | 2.23 [2.23-2.24] | 0 | 0.75 | 0.90 | 0.56 | 100% |


Stages (medians over repeats; times from query start):

| Arm | Stage | Operators | Block | Tasks | Rows in | Rows out | Out/in | Occupancy | Start (s) | End (s) | Overlap (s) | Fallbacks |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 0 | SemFilter | 1 | 32 | 32 | 16 | 0.50 | n/a | 0.03 [0.03-0.03] | 0.90 [0.90-0.91] | n/a | 0 |
| `kalypso_block:4` | 0 | BlockSemFilter | 4 | 8 | 32 | 15 | 0.47 | 1.00 | 0.02 [0.02-0.02] | 1.00 [1.00-1.01] | n/a | 0 |
| `kalypso_block:16` | 0 | BlockSemFilter | 16 | 2 | 32 | 10 | 0.31 | 1.00 | 0.02 [0.02-0.03] | 2.25 [2.25-2.26] | n/a | 0 |


## n64: N = 64

Model `Qwen/Qwen2.5-7B-Instruct` on NVIDIA L40S (Linux-4.19.0-gvisor-x86_64-with-glibc2.39); workload `arxiv_filter` (filter), 64 rows, 32 positive; 3 repeats after 1 warmup; prefix cache reset per trial: yes; git `d19f407a0` (dirty).
Server: `python3 -m vllm.entrypoints.cli.main serve Qwen/Qwen2.5-7B-Instruct --host 127.0.0.1 --port 8000 --enable-prefix-caching --no-async-scheduling --api-server-count 1 --max-model-len 16384` (started Tue Oct 20 15:59:43 2026).

| Arm | Completion time (s) | Speedup vs kalypso_per_tuple | LLM calls [retries + fallbacks] | Agrees w/ kalypso_per_tuple | F1 vs labels | Peak KV % |
|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 1.78 [1.77-1.79] | 1.00x | 64 [0] | 100% | 1.00 | 1.0 |
| `kalypso_block:4` | 1.73 [1.70-1.75] | 1.03x | 16 [0] | 83% | 0.81 | 3.8 |
| `kalypso_block:16` | 3.04 [3.01-3.06] | 0.58x | 4 [0] | 86% | 0.84 | 3.7 |


Diagnostics:

| Arm | Prompt tokens | Gen tokens | Prefix hit rate | Peak RSS (MiB) | Items/s | First output (s) | Truncated | Accuracy | Precision | Recall | Self-agreement |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 24293 | 192 | 0.12 | 12408 [12407-12408] | 36.04 [35.75-36.20] | 0.46 [0.45-0.46] | 0 | 1.00 | 1.00 | 1.00 | 100% |
| `kalypso_block:4` | 17351 | 256 | 0.03 | 12408 [12407-12408] | 37.10 [36.68-37.67] | 1.53 [1.51-1.56] | 0 | 0.83 | 0.89 | 0.75 | 100% |
| `kalypso_block:16` | 16403 | 284 | 0.01 | 12408 [12408-12408] | 21.05 [20.91-21.25] | 2.92 [2.89-2.94] | 0 | 0.86 | 0.96 | 0.75 | 100% |


Stages (medians over repeats; times from query start):

| Arm | Stage | Operators | Block | Tasks | Rows in | Rows out | Out/in | Occupancy | Start (s) | End (s) | Overlap (s) | Fallbacks |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 0 | SemFilter | 1 | 64 | 64 | 32 | 0.50 | n/a | 0.05 [0.05-0.05] | 1.77 [1.76-1.78] | n/a | 0 |
| `kalypso_block:4` | 0 | BlockSemFilter | 4 | 16 | 64 | 27 | 0.42 | 1.00 | 0.04 [0.04-0.04] | 1.72 [1.70-1.74] | n/a | 0 |
| `kalypso_block:16` | 0 | BlockSemFilter | 16 | 4 | 64 | 25 | 0.39 | 1.00 | 0.04 [0.04-0.04] | 3.04 [3.01-3.06] | n/a | 0 |


## n128: N = 128

Model `Qwen/Qwen2.5-7B-Instruct` on NVIDIA L40S (Linux-4.19.0-gvisor-x86_64-with-glibc2.39); workload `arxiv_filter` (filter), 128 rows, 64 positive; 3 repeats after 1 warmup; prefix cache reset per trial: yes; git `d19f407a0` (dirty).
Server: `python3 -m vllm.entrypoints.cli.main serve Qwen/Qwen2.5-7B-Instruct --host 127.0.0.1 --port 8000 --enable-prefix-caching --no-async-scheduling --api-server-count 1 --max-model-len 16384` (started Tue Oct 20 15:59:43 2026).

| Arm | Completion time (s) | Speedup vs kalypso_per_tuple | LLM calls [retries + fallbacks] | Agrees w/ kalypso_per_tuple | F1 vs labels | Peak KV % |
|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 3.49 [3.48-3.66] | 1.00x | 128 [0] | 100% | 0.98 | 1.0 |
| `kalypso_block:4` | 3.12 [3.10-3.48] | 1.12x | 32 [0] | 85% | 0.82 | 6.9 |
| `kalypso_block:16` | 4.43 [4.42-4.77] | 0.79x | 8 [0] | 80% | 0.76 | 7.3 |


Diagnostics:

| Arm | Prompt tokens | Gen tokens | Prefix hit rate | Peak RSS (MiB) | Items/s | First output (s) | Truncated | Accuracy | Precision | Recall | Self-agreement |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 47969 | 384 | 0.13 | 12412 [12412-12412] | 36.65 [34.94-36.75] | 0.70 [0.69-0.79] | 0 | 0.98 | 1.00 | 0.97 | 100% |
| `kalypso_block:4` | 34084 | 512 | 0.03 | 12412 [12412-12412] | 40.96 [36.78-41.30] | 2.74 [2.72-2.86] | 0 | 0.84 | 0.92 | 0.73 | 100% |
| `kalypso_block:16` | 32188 | 568 | 0.01 | 12412 [12412-12412] | 28.91 [26.82-28.93] | 4.12 [4.12-4.46] | 0 | 0.80 | 0.98 | 0.62 | 100% |


Stages (medians over repeats; times from query start):

| Arm | Stage | Operators | Block | Tasks | Rows in | Rows out | Out/in | Occupancy | Start (s) | End (s) | Overlap (s) | Fallbacks |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 0 | SemFilter | 1 | 128 | 128 | 62 | 0.48 | n/a | 0.08 [0.08-0.13] | 3.48 [3.48-3.63] | n/a | 0 |
| `kalypso_block:4` | 0 | BlockSemFilter | 4 | 32 | 128 | 51 | 0.40 | 1.00 | 0.06 [0.06-0.12] | 3.12 [3.09-3.46] | n/a | 0 |
| `kalypso_block:16` | 0 | BlockSemFilter | 16 | 8 | 128 | 41 | 0.32 | 1.00 | 0.06 [0.06-0.15] | 4.42 [4.42-4.76] | n/a | 0 |

