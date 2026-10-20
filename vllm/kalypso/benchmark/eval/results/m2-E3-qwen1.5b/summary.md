# E3

## n16: N = 16

Model `Qwen/Qwen2.5-1.5B-Instruct` on NVIDIA L4 (Linux-4.19.0-gvisor-x86_64-with-glibc2.39); workload `arxiv_filter` (filter), 16 rows, 8 positive; 3 repeats after 1 warmup; prefix cache reset per trial: yes; git `d19f407a0` (dirty).
Server: `python3 -m vllm.entrypoints.cli.main serve Qwen/Qwen2.5-1.5B-Instruct --host 127.0.0.1 --port 8000 --enable-prefix-caching --no-async-scheduling --api-server-count 1 --max-model-len 16384` (started Tue Oct 20 06:09:00 2026).

| Arm | Completion time (s) | Speedup vs kalypso_per_tuple | LLM calls [retries + fallbacks] | Agrees w/ kalypso_per_tuple | F1 vs labels | Peak KV % |
|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 0.39 [0.39-0.40] | 1.00x | 16 [0] | 100% | 1.00 | 0.4 |
| `kalypso_block:4` | 0.54 [0.53-0.56] | 0.72x | 4 [0] | 56% | 0.67 | 0.7 |
| `kalypso_block:16` | 1.44 [1.39-1.55] | 0.27x | 1 [0] | 50% | 0.67 | 0.7 |


Diagnostics:

| Arm | Prompt tokens | Gen tokens | Prefix hit rate | Peak RSS (MiB) | Items/s | First output (s) | Truncated | Accuracy | Precision | Recall | Self-agreement |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 5913 | 32 | 0.12 | 11362 [11362-11362] | 40.92 [40.51-41.45] | 0.32 [0.32-0.32] | 0 | 1.00 | 1.00 | 1.00 | 100% |
| `kalypso_block:4` | 4177 | 64 | 0.02 | 11362 [11362-11362] | 29.63 [28.83-30.07] | 0.50 [0.50-0.52] | 0 | 0.56 | 0.54 | 0.88 | 100% |
| `kalypso_block:16` | 3940 | 71 | 0.00 | 11362 | 11.10 [10.34-11.53] | 1.44 [1.38-1.54] | 0 | 0.50 | 0.50 | 1.00 | 100% |


Stages (medians over repeats; times from query start):

| Arm | Stage | Operators | Block | Tasks | Rows in | Rows out | Out/in | Occupancy | Start (s) | End (s) | Overlap (s) | Fallbacks |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 0 | SemFilter | 1 | 16 | 16 | 8 | 0.50 | n/a | 0.02 [0.02-0.02] | 0.39 [0.38-0.39] | n/a | 0 |
| `kalypso_block:4` | 0 | BlockSemFilter | 4 | 4 | 16 | 13 | 0.81 | 1.00 | 0.02 [0.02-0.03] | 0.54 [0.53-0.55] | n/a | 0 |
| `kalypso_block:16` | 0 | BlockSemFilter | 16 | 1 | 16 | 16 | 1.00 | 1.00 | 0.03 [0.03-0.04] | 1.44 [1.38-1.54] | n/a | 0 |


## n32: N = 32

Model `Qwen/Qwen2.5-1.5B-Instruct` on NVIDIA L4 (Linux-4.19.0-gvisor-x86_64-with-glibc2.39); workload `arxiv_filter` (filter), 32 rows, 16 positive; 3 repeats after 1 warmup; prefix cache reset per trial: yes; git `d19f407a0` (dirty).
Server: `python3 -m vllm.entrypoints.cli.main serve Qwen/Qwen2.5-1.5B-Instruct --host 127.0.0.1 --port 8000 --enable-prefix-caching --no-async-scheduling --api-server-count 1 --max-model-len 16384` (started Tue Oct 20 06:09:00 2026).

| Arm | Completion time (s) | Speedup vs kalypso_per_tuple | LLM calls [retries + fallbacks] | Agrees w/ kalypso_per_tuple | F1 vs labels | Peak KV % |
|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 0.82 [0.80-0.82] | 1.00x | 32 [0] | 100% | 1.00 | 0.4 |
| `kalypso_block:4` | 0.84 [0.83-0.85] | 0.97x | 8 [0] | 53% | 0.63 | 1.4 |
| `kalypso_block:16` | 1.77 [1.76-1.77] | 0.46x | 2 [0] | 53% | 0.68 | 1.4 |


Diagnostics:

| Arm | Prompt tokens | Gen tokens | Prefix hit rate | Peak RSS (MiB) | Items/s | First output (s) | Truncated | Accuracy | Precision | Recall | Self-agreement |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 12038 | 70 | 0.12 | 11363 [11363-11364] | 39.22 [38.84-40.15] | 0.37 [0.36-0.38] | 1 | 1.00 | 1.00 | 1.00 | 100% |
| `kalypso_block:4` | 8567 | 128 | 0.03 | 11363 [11363-11364] | 38.09 [37.78-38.37] | 0.77 [0.76-0.77] | 0 | 0.53 | 0.52 | 0.81 | 100% |
| `kalypso_block:16` | 8093 | 142 | 0.00 | 11363 [11363-11364] | 18.11 [18.07-18.22] | 1.74 [1.73-1.75] | 0 | 0.53 | 0.52 | 1.00 | 100% |


Stages (medians over repeats; times from query start):

| Arm | Stage | Operators | Block | Tasks | Rows in | Rows out | Out/in | Occupancy | Start (s) | End (s) | Overlap (s) | Fallbacks |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 0 | SemFilter | 1 | 32 | 32 | 16 | 0.50 | n/a | 0.04 [0.04-0.05] | 0.81 [0.79-0.82] | n/a | 0 |
| `kalypso_block:4` | 0 | BlockSemFilter | 4 | 8 | 32 | 25 | 0.78 | 1.00 | 0.04 [0.04-0.04] | 0.83 [0.83-0.84] | n/a | 0 |
| `kalypso_block:16` | 0 | BlockSemFilter | 16 | 2 | 32 | 31 | 0.97 | 1.00 | 0.05 [0.05-0.05] | 1.76 [1.75-1.76] | n/a | 0 |


## n64: N = 64

Model `Qwen/Qwen2.5-1.5B-Instruct` on NVIDIA L4 (Linux-4.19.0-gvisor-x86_64-with-glibc2.39); workload `arxiv_filter` (filter), 64 rows, 32 positive; 3 repeats after 1 warmup; prefix cache reset per trial: yes; git `d19f407a0` (dirty).
Server: `python3 -m vllm.entrypoints.cli.main serve Qwen/Qwen2.5-1.5B-Instruct --host 127.0.0.1 --port 8000 --enable-prefix-caching --no-async-scheduling --api-server-count 1 --max-model-len 16384` (started Tue Oct 20 06:09:00 2026).

| Arm | Completion time (s) | Speedup vs kalypso_per_tuple | LLM calls [retries + fallbacks] | Agrees w/ kalypso_per_tuple | F1 vs labels | Peak KV % |
|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 1.51 [1.50-1.55] | 1.00x | 64 [0] | 100% | 1.00 | 0.5 |
| `kalypso_block:4` | 1.46 [1.45-1.62] | 1.04x | 16 [0] | 59% | 0.70 | 2.9 |
| `kalypso_block:16` | 2.44 [2.39-2.50] | 0.62x | 4 [0] | 48% | 0.59 | 2.8 |


Diagnostics:

| Arm | Prompt tokens | Gen tokens | Prefix hit rate | Peak RSS (MiB) | Items/s | First output (s) | Truncated | Accuracy | Precision | Recall | Self-agreement |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 24293 | 140 | 0.12 | 11367 [11366-11367] | 42.38 [41.24-42.61] | 0.35 [0.35-0.36] | 2 | 1.00 | 1.00 | 1.00 | 100% |
| `kalypso_block:4` | 17351 | 256 | 0.03 | 11367 [11366-11367] | 43.96 [39.55-44.17] | 1.31 [1.31-1.46] | 0 | 0.59 | 0.56 | 0.94 | 100% |
| `kalypso_block:16` | 16403 | 284 | 0.01 | 11367 [11367-11367] | 26.25 [25.55-26.73] | 2.35 [2.31-2.42] | 0 | 0.48 | 0.49 | 0.75 | 100% |


Stages (medians over repeats; times from query start):

| Arm | Stage | Operators | Block | Tasks | Rows in | Rows out | Out/in | Occupancy | Start (s) | End (s) | Overlap (s) | Fallbacks |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 0 | SemFilter | 1 | 64 | 64 | 32 | 0.50 | n/a | 0.09 [0.09-0.09] | 1.50 [1.49-1.54] | n/a | 0 |
| `kalypso_block:4` | 0 | BlockSemFilter | 4 | 16 | 64 | 54 | 0.84 | 1.00 | 0.07 [0.07-0.09] | 1.45 [1.44-1.61] | n/a | 0 |
| `kalypso_block:16` | 0 | BlockSemFilter | 16 | 4 | 64 | 49 | 0.77 | 1.00 | 0.08 [0.07-0.12] | 2.43 [2.39-2.50] | n/a | 0 |


## n128: N = 128

Model `Qwen/Qwen2.5-1.5B-Instruct` on NVIDIA L4 (Linux-4.19.0-gvisor-x86_64-with-glibc2.39); workload `arxiv_filter` (filter), 128 rows, 64 positive; 3 repeats after 1 warmup; prefix cache reset per trial: yes; git `d19f407a0` (dirty).
Server: `python3 -m vllm.entrypoints.cli.main serve Qwen/Qwen2.5-1.5B-Instruct --host 127.0.0.1 --port 8000 --enable-prefix-caching --no-async-scheduling --api-server-count 1 --max-model-len 16384` (started Tue Oct 20 06:09:00 2026).

| Arm | Completion time (s) | Speedup vs kalypso_per_tuple | LLM calls [retries + fallbacks] | Agrees w/ kalypso_per_tuple | F1 vs labels | Peak KV % |
|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 3.07 [3.04-3.09] | 1.00x | 128 [0] | 100% | 0.98 | 0.5 |
| `kalypso_block:4` | 2.69 [2.67-2.71] | 1.14x | 32 [0] | 55% | 0.67 | 5.2 |
| `kalypso_block:16` | 3.77 [3.74-3.79] | 0.81x | 8 [0] | 49% | 0.67 | 5.5 |


Diagnostics:

| Arm | Prompt tokens | Gen tokens | Prefix hit rate | Peak RSS (MiB) | Items/s | First output (s) | Truncated | Accuracy | Precision | Recall | Self-agreement |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 47969 | 274 | 0.13 | 11371 [11371-11372] | 41.68 [41.37-42.06] | 0.62 [0.61-0.63] | 3 | 0.98 | 1.00 | 0.95 | 100% |
| `kalypso_block:4` | 34084 | 512 | 0.03 | 11371 [11371-11372] | 47.62 [47.23-47.99] | 2.30 [2.28-2.34] | 0 | 0.56 | 0.54 | 0.89 | 100% |
| `kalypso_block:16` | 32188 | 568 | 0.01 | 11372 [11371-11372] | 33.96 [33.76-34.22] | 3.54 [3.51-3.55] | 0 | 0.52 | 0.51 | 0.97 | 100% |


Stages (medians over repeats; times from query start):

| Arm | Stage | Operators | Block | Tasks | Rows in | Rows out | Out/in | Occupancy | Start (s) | End (s) | Overlap (s) | Fallbacks |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 0 | SemFilter | 1 | 128 | 128 | 61 | 0.48 | n/a | 0.13 [0.13-0.13] | 3.06 [3.03-3.08] | n/a | 0 |
| `kalypso_block:4` | 0 | BlockSemFilter | 4 | 32 | 128 | 106 | 0.83 | 1.00 | 0.10 [0.10-0.10] | 2.68 [2.66-2.70] | n/a | 0 |
| `kalypso_block:16` | 0 | BlockSemFilter | 16 | 8 | 128 | 122 | 0.95 | 1.00 | 0.10 [0.10-0.11] | 3.76 [3.73-3.78] | n/a | 0 |

