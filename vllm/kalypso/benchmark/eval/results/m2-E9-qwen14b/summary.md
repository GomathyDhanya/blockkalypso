# E9

## s05_f1: s = 0.5, f = 1.0

Model `Qwen/Qwen2.5-14B-Instruct` on NVIDIA L40S (Linux-4.19.0-gvisor-x86_64-with-glibc2.39); workload `arxiv_fjf` (pipeline), 32 x 32 = 1024 pairs, 9 positive; 3 repeats after 1 warmup; prefix cache reset per trial: yes; git `ba8ba983e` (dirty).
Server: `python3 -m vllm.entrypoints.cli.main serve Qwen/Qwen2.5-14B-Instruct --host 127.0.0.1 --port 8000 --enable-prefix-caching --no-async-scheduling --api-server-count 1 --max-model-len 16384` (started Tue Oct 20 21:40:46 2026).

| Arm | Completion time (s) | Speedup vs kalypso_per_tuple | LLM calls [retries + fallbacks] | Agrees w/ kalypso_per_tuple | F1 vs labels | Peak KV % |
|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 31.81 [31.52-32.38] | 1.00x | 557 [0] | 100% | 0.80 | 32.3 [32.2-32.5] |
| `kalypso_block:4/4x4/4` | 10.36 [9.68-10.57] | 3.07x | 42 [1] | 99% | 0.00 | 15.9 |
| `kalypso_block:4/4x4/4+coalesce` | 10.76 [9.79-10.81] | 2.95x | 41 [0] | 99% | 0.00 | 15.9 |
| `kalypso_block:4/2x2/4` | 16.37 [16.15-16.80] | 1.94x | 141 [4] | 99% | 0.00 | 16.0 |
| `kalypso_block:4/2x2/4+coalesce` | 16.71 [16.55-17.28] | 1.90x | 138 [0] | 99% | 0.00 | 16.0 [15.8-16.0] |


Diagnostics:

| Arm | Prompt tokens | Gen tokens | Prefix hit rate | Peak RSS (MiB) | Items/s | First output (s) | Truncated | Accuracy | Precision | Recall | Self-agreement |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 369422 | 4450 | 0.48 [0.48-0.48] | 11539 [11538-11539] | 32.20 [31.62-32.49] | 12.96 [8.86-15.06] | 474 | 1.00 | 1.00 | 0.67 | 100% |
| `kalypso_block:4/4x4/4` | 82447 [81554-82447] | 224 [204-224] | 0.39 [0.39-0.39] | 11539 [11538-11539] | 98.80 [96.84-105.77] | 7.91 [7.75-7.91] | 0 | 0.99 | 0.00 | 0.00 | 100% |
| `kalypso_block:4/4x4/4+coalesce` | 82401 [81544-82401] | 226 [208-226] | 0.39 [0.39-0.39] | 11539 [11538-11539] | 95.13 [94.71-104.55] | 9.79 | 0 | 0.99 [0.99-0.99] | 0.00 | 0.00 | 100% |
| `kalypso_block:4/2x2/4` | 161237 | 424 | 0.44 | 11539 [11538-11539] | 62.54 [60.96-63.39] | 13.67 [13.45-13.99] | 0 | 0.99 | 0.00 | 0.00 | 100% |
| `kalypso_block:4/2x2/4+coalesce` | 161135 | 432 | 0.44 | 11539 [11538-11539] | 61.28 [59.27-61.87] | 16.71 [16.54-17.27] | 0 | 0.99 | 0.00 | 0.00 | 100% |


Stages (medians over repeats; times from query start):

| Arm | Stage | Operators | Block | Tasks | Rows in | Rows out | Out/in | Occupancy | Start (s) | End (s) | Overlap (s) | Fallbacks |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 0 | SemFilter | 1 | 32 | 32 | 512 | 16.00 | n/a | 0.26 [0.24-0.26] | 21.28 [20.87-21.29] | n/a | 0 |
| `kalypso_per_tuple` | 1 | SemFilter+SemFilter | 1 | 512 | 512 | 6 | 0.01 | n/a | 0.98 [0.95-0.98] | 31.79 [31.50-32.37] | 20.30 [19.92-20.31] | 0 |
| `kalypso_block:4/4x4/4` | 0 | BlockSemFilter | 4 | 8 | 32 | 16 | 0.50 | 1.00 | 0.38 [0.35-0.40] | 4.80 [4.76-4.80] | n/a | 0 |
| `kalypso_block:4/4x4/4` | 1 | BlockSemJoin | 4 | 4 | 16 | 4 [2-4] | 0.25 [0.12-0.25] | 1.00 | 1.59 [1.56-1.59] | 9.69 [9.54-9.89] | 3.21 [3.17-3.24] | 0 |
| `kalypso_block:4/4x4/4` | 2 | BlockSemFilter | 4 | 2 | 4 [2-4] | 1 | 0.25 [0.25-0.50] | 0.50 [0.25-0.50] | 7.58 [7.44-7.59] | 10.36 [9.68-10.57] | 2.25 [1.95-2.31] | 1 [1-2] |
| `kalypso_block:4/4x4/4+coalesce` | 0 | BlockSemFilter | 4 | 8 | 32 | 16 | 0.50 | 1.00 | 0.36 [0.35-0.47] | 4.81 [4.73-4.86] | n/a | 0 |
| `kalypso_block:4/4x4/4+coalesce` | 1 | BlockSemJoin | 4 | 4 | 16 | 4 [2-4] | 0.25 [0.12-0.25] | 1.00 | 1.59 [1.56-1.68] | 9.84 [9.33-9.88] | 3.18 [3.17-3.22] | 0 |
| `kalypso_block:4/4x4/4+coalesce` | 2 | BlockSemFilter | 4 | 1 | 4 [2-4] | 0 [0-1] | 0.00 [0.00-0.50] | 1.00 [0.50-1.00] | 9.85 [9.33-9.88] | 10.76 [9.79-10.81] | 0.00 | 0 |
| `kalypso_block:4/2x2/4` | 0 | BlockSemFilter | 4 | 8 | 32 | 16 | 0.50 | 1.00 | 0.62 [0.59-0.68] | 5.59 [5.47-5.61] | n/a | 0 |
| `kalypso_block:4/2x2/4` | 1 | BlockSemJoin | 2 | 8 | 16 | 6 | 0.38 | 1.00 | 1.79 [1.75-1.85] | 15.89 [15.67-16.31] | 3.76 [3.71-3.80] | 0 |
| `kalypso_block:4/2x2/4` | 2 | BlockSemFilter | 4 | 5 | 6 | 3 | 0.50 | 0.30 | 13.03 [12.81-13.31] | 16.37 [16.15-16.79] | 2.86 [2.86-3.00] | 4 |
| `kalypso_block:4/2x2/4+coalesce` | 0 | BlockSemFilter | 4 | 8 | 32 | 16 | 0.50 | 1.00 | 0.62 [0.61-0.72] | 5.55 [5.42-5.64] | n/a | 0 |
| `kalypso_block:4/2x2/4+coalesce` | 1 | BlockSemJoin | 2 | 8 | 16 | 6 | 0.38 | 1.00 | 1.79 [1.79-1.88] | 15.89 [15.73-16.45] | 3.68 [3.63-3.86] | 0 |
| `kalypso_block:4/2x2/4+coalesce` | 2 | BlockSemFilter | 4 | 2 | 6 | 1 | 0.17 | 0.75 | 15.50 [15.34-16.04] | 16.71 [16.54-17.27] | 0.39 [0.39-0.41] | 0 |


Operators (first repeat):

| Arm | Query op | Kalypso operator | Rows in | Rows out | Answers began with | Other answers (samples) |
|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 0: sem_filter | SemFilter | 32 | 16 | false 16, true 16 | - |
| `kalypso_per_tuple` | 1: join | SemFilter | 512 | 13 | false 499, true 13 | - |
| `kalypso_per_tuple` | 2: sem_filter | SemFilter | 13 | 6 | false 7, true 6 | - |
| `kalypso_block:4/4x4/4` | 0: sem_filter | BlockSemFilter | 32 | 16 | false 16, true 16 | - |
| `kalypso_block:4/4x4/4` | 1: join | BlockSemJoin | 16 | 4 | - | - |
| `kalypso_block:4/4x4/4` | 2: sem_filter | BlockSemFilter | 4 | 1 | false 3, true 1 | - |
| `kalypso_block:4/4x4/4+coalesce` | 0: sem_filter | BlockSemFilter | 32 | 16 | false 16, true 16 | - |
| `kalypso_block:4/4x4/4+coalesce` | 1: join | BlockSemJoin | 16 | 4 | - | - |
| `kalypso_block:4/4x4/4+coalesce` | 2: sem_filter | BlockSemFilter | 4 | 0 | false 4 | - |
| `kalypso_block:4/2x2/4` | 0: sem_filter | BlockSemFilter | 32 | 16 | false 16, true 16 | - |
| `kalypso_block:4/2x2/4` | 1: join | BlockSemJoin | 16 | 6 | - | - |
| `kalypso_block:4/2x2/4` | 2: sem_filter | BlockSemFilter | 6 | 3 | false 3, true 3 | - |
| `kalypso_block:4/2x2/4+coalesce` | 0: sem_filter | BlockSemFilter | 32 | 16 | false 16, true 16 | - |
| `kalypso_block:4/2x2/4+coalesce` | 1: join | BlockSemJoin | 16 | 6 | - | - |
| `kalypso_block:4/2x2/4+coalesce` | 2: sem_filter | BlockSemFilter | 6 | 1 | false 5, true 1 | - |



Work passed downstream (medians). Gold: 16 rows should pass the filter, 9 final outputs are expected.

| Arm | Rows past first filter | Pairs past join | Final outputs | Comparable |
|---|---|---|---|---|
| `kalypso_per_tuple` | 16 | 13 | 6 | reference |
| `kalypso_block:4/4x4/4` | 16 | 4 | 1 | **no, work drifted** |
| `kalypso_block:4/4x4/4+coalesce` | 16 | 4 | 0 | **no, work drifted** |
| `kalypso_block:4/2x2/4` | 16 | 6 | 3 | **no, work drifted** |
| `kalypso_block:4/2x2/4+coalesce` | 16 | 6 | 1 | **no, work drifted** |

Drifted by more than 10% from `kalypso_per_tuple`: `kalypso_block:4/4x4/4`, `kalypso_block:4/4x4/4+coalesce`, `kalypso_block:4/2x2/4`, `kalypso_block:4/2x2/4+coalesce`. Their latency and requests reflect different work, not just blocking.

## s01_f1: s = 0.1, f = 1.0

Model `Qwen/Qwen2.5-14B-Instruct` on NVIDIA L40S (Linux-4.19.0-gvisor-x86_64-with-glibc2.39); workload `arxiv_fjf` (pipeline), 32 x 32 = 1024 pairs, 2 positive; 3 repeats after 1 warmup; prefix cache reset per trial: yes; git `ba8ba983e` (dirty).
Server: `python3 -m vllm.entrypoints.cli.main serve Qwen/Qwen2.5-14B-Instruct --host 127.0.0.1 --port 8000 --enable-prefix-caching --no-async-scheduling --api-server-count 1 --max-model-len 16384` (started Tue Oct 20 21:40:46 2026).

| Arm | Completion time (s) | Speedup vs kalypso_per_tuple | LLM calls [retries + fallbacks] | Agrees w/ kalypso_per_tuple | F1 vs labels | Peak KV % |
|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 7.68 [7.68-7.73] | 1.00x | 129 [0] | 100% | n/a | 29.9 [29.9-30.0] |
| `kalypso_block:4/4x4/4` | 5.33 [5.33-5.39] | 1.44x | 17 [1] | 100% | 0.00 | 10.2 [10.1-10.2] |
| `kalypso_block:4/4x4/4+coalesce` | 5.36 [5.28-5.44] | 1.43x | 17 [1] | 100% | 0.00 | 10.2 [10.1-10.2] |
| `kalypso_block:4/2x2/4` | 7.14 [7.08-7.15] | 1.08x | 41 [1] | 100% | 0.00 | 12.5 |
| `kalypso_block:4/2x2/4+coalesce` | 7.17 [7.17-7.20] | 1.07x | 41 [1] | 100% | 0.00 | 12.5 |


Diagnostics:

| Arm | Prompt tokens | Gen tokens | Prefix hit rate | Peak RSS (MiB) | Items/s | First output (s) | Truncated | Accuracy | Precision | Recall | Self-agreement |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 79552 | 1032 | 0.44 | 11539 [11539-11539] | 133.25 [132.50-133.38] | n/a | 123 | 1.00 | n/a | 0.00 | 100% |
| `kalypso_block:4/4x4/4` | 27462 | 152 | 0.31 | 11539 [11539-11539] | 192.01 [189.84-192.16] | 5.33 [5.33-5.39] | 0 | 1.00 | 0.00 | 0.00 | 100% |
| `kalypso_block:4/4x4/4+coalesce` | 27462 | 152 | 0.31 | 11539 [11539-11539] | 191.08 [188.24-194.12] | 5.36 [5.27-5.44] | 0 | 1.00 | 0.00 | 0.00 | 100% |
| `kalypso_block:4/2x2/4` | 47069 | 198 | 0.39 | 11539 [11539-11539] | 143.46 [143.28-144.67] | 6.71 [6.65-6.71] | 0 | 1.00 | 0.00 | 0.00 | 100% |
| `kalypso_block:4/2x2/4+coalesce` | 47069 | 198 | 0.39 | 11539 [11539-11539] | 142.82 [142.24-142.84] | 7.17 [7.17-7.19] | 0 | 1.00 | 0.00 | 0.00 | 100% |


Stages (medians over repeats; times from query start):

| Arm | Stage | Operators | Block | Tasks | Rows in | Rows out | Out/in | Occupancy | Start (s) | End (s) | Overlap (s) | Fallbacks |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 0 | SemFilter | 1 | 32 | 32 | 96 | 3.00 | n/a | 0.25 [0.24-0.26] | 6.17 [4.30-6.21] | n/a | 0 |
| `kalypso_per_tuple` | 1 | SemFilter+SemFilter | 1 | 96 | 96 | 0 | 0.00 | n/a | 0.96 [0.96-0.97] | 7.68 [7.67-7.72] | 5.21 [3.33-5.25] | 0 |
| `kalypso_block:4/4x4/4` | 0 | BlockSemFilter | 4 | 8 | 32 | 4 | 0.12 | 1.00 | 0.38 [0.38-0.41] | 2.93 [2.92-2.96] | n/a | 0 |
| `kalypso_block:4/4x4/4` | 1 | BlockSemJoin | 4 | 1 | 4 | 1 | 0.25 | 1.00 | 2.96 [2.95-3.00] | 5.19 [5.18-5.25] | 0.00 | 0 |
| `kalypso_block:4/4x4/4` | 2 | BlockSemFilter | 4 | 1 | 1 | 1 | 1.00 | 0.25 | 5.19 [5.18-5.25] | 5.33 [5.33-5.39] | 0.00 | 1 |
| `kalypso_block:4/4x4/4+coalesce` | 0 | BlockSemFilter | 4 | 8 | 32 | 4 | 0.12 | 1.00 | 0.38 [0.36-0.45] | 2.93 [2.90-3.01] | n/a | 0 |
| `kalypso_block:4/4x4/4+coalesce` | 1 | BlockSemJoin | 4 | 1 | 4 | 1 | 0.25 | 1.00 | 2.99 [2.93-3.04] | 5.22 [5.13-5.29] | 0.00 | 0 |
| `kalypso_block:4/4x4/4+coalesce` | 2 | BlockSemFilter | 4 | 1 | 1 | 1 | 1.00 | 0.25 | 5.22 [5.13-5.30] | 5.36 [5.27-5.44] | 0.00 | 1 |
| `kalypso_block:4/2x2/4` | 0 | BlockSemFilter | 4 | 8 | 32 | 4 | 0.12 | 1.00 | 0.59 [0.58-0.63] | 3.50 [3.48-3.51] | n/a | 0 |
| `kalypso_block:4/2x2/4` | 1 | BlockSemJoin | 2 | 2 | 4 | 1 | 0.25 | 1.00 | 1.76 [1.74-1.79] | 7.13 [7.07-7.14] | 1.74 [1.72-1.74] | 0 |
| `kalypso_block:4/2x2/4` | 2 | BlockSemFilter | 4 | 1 | 1 | 1 | 1.00 | 0.25 | 6.48 [6.41-6.48] | 6.71 [6.65-6.71] | 0.66 [0.66-0.67] | 1 |
| `kalypso_block:4/2x2/4+coalesce` | 0 | BlockSemFilter | 4 | 8 | 32 | 4 | 0.12 | 1.00 | 0.59 [0.57-0.63] | 3.50 [3.49-3.52] | n/a | 0 |
| `kalypso_block:4/2x2/4+coalesce` | 1 | BlockSemJoin | 2 | 2 | 4 | 1 | 0.25 | 1.00 | 1.76 [1.75-1.79] | 7.04 [7.03-7.07] | 1.74 [1.73-1.74] | 0 |
| `kalypso_block:4/2x2/4+coalesce` | 2 | BlockSemFilter | 4 | 1 | 1 | 1 | 1.00 | 0.25 | 7.04 [7.03-7.07] | 7.17 [7.17-7.19] | 0.00 | 1 |


Operators (first repeat):

| Arm | Query op | Kalypso operator | Rows in | Rows out | Answers began with | Other answers (samples) |
|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 0: sem_filter | SemFilter | 32 | 3 | false 29, true 3 | - |
| `kalypso_per_tuple` | 1: join | SemFilter | 96 | 1 | false 95, true 1 | - |
| `kalypso_per_tuple` | 2: sem_filter | SemFilter | 1 | 0 | false 1 | - |
| `kalypso_block:4/4x4/4` | 0: sem_filter | BlockSemFilter | 32 | 4 | false 28, true 4 | - |
| `kalypso_block:4/4x4/4` | 1: join | BlockSemJoin | 4 | 1 | - | - |
| `kalypso_block:4/4x4/4` | 2: sem_filter | BlockSemFilter | 1 | 1 | true 1 | - |
| `kalypso_block:4/4x4/4+coalesce` | 0: sem_filter | BlockSemFilter | 32 | 4 | false 28, true 4 | - |
| `kalypso_block:4/4x4/4+coalesce` | 1: join | BlockSemJoin | 4 | 1 | - | - |
| `kalypso_block:4/4x4/4+coalesce` | 2: sem_filter | BlockSemFilter | 1 | 1 | true 1 | - |
| `kalypso_block:4/2x2/4` | 0: sem_filter | BlockSemFilter | 32 | 4 | false 28, true 4 | - |
| `kalypso_block:4/2x2/4` | 1: join | BlockSemJoin | 4 | 1 | - | - |
| `kalypso_block:4/2x2/4` | 2: sem_filter | BlockSemFilter | 1 | 1 | true 1 | - |
| `kalypso_block:4/2x2/4+coalesce` | 0: sem_filter | BlockSemFilter | 32 | 4 | false 28, true 4 | - |
| `kalypso_block:4/2x2/4+coalesce` | 1: join | BlockSemJoin | 4 | 1 | - | - |
| `kalypso_block:4/2x2/4+coalesce` | 2: sem_filter | BlockSemFilter | 1 | 1 | true 1 | - |



Work passed downstream (medians). Gold: 3 rows should pass the filter, 2 final outputs are expected.

| Arm | Rows past first filter | Pairs past join | Final outputs | Comparable |
|---|---|---|---|---|
| `kalypso_per_tuple` | 3 | 1 | 0 | reference |
| `kalypso_block:4/4x4/4` | 4 | 1 | 1 | **no, work drifted** |
| `kalypso_block:4/4x4/4+coalesce` | 4 | 1 | 1 | **no, work drifted** |
| `kalypso_block:4/2x2/4` | 4 | 1 | 1 | **no, work drifted** |
| `kalypso_block:4/2x2/4+coalesce` | 4 | 1 | 1 | **no, work drifted** |

Drifted by more than 10% from `kalypso_per_tuple`: `kalypso_block:4/4x4/4`, `kalypso_block:4/4x4/4+coalesce`, `kalypso_block:4/2x2/4`, `kalypso_block:4/2x2/4+coalesce`. Their latency and requests reflect different work, not just blocking.

## s05_f02: s = 0.5, f = 0.2

Model `Qwen/Qwen2.5-14B-Instruct` on NVIDIA L40S (Linux-4.19.0-gvisor-x86_64-with-glibc2.39); workload `arxiv_fjf` (pipeline), 32 x 32 = 1024 pairs, 4 positive; 3 repeats after 1 warmup; prefix cache reset per trial: yes; git `ba8ba983e` (dirty).
Server: `python3 -m vllm.entrypoints.cli.main serve Qwen/Qwen2.5-14B-Instruct --host 127.0.0.1 --port 8000 --enable-prefix-caching --no-async-scheduling --api-server-count 1 --max-model-len 16384` (started Tue Oct 20 21:40:46 2026).

| Arm | Completion time (s) | Speedup vs kalypso_per_tuple | LLM calls [retries + fallbacks] | Agrees w/ kalypso_per_tuple | F1 vs labels | Peak KV % |
|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 31.56 [31.47-32.27] | 1.00x | 548 [0] | 100% | 1.00 | 31.0 [30.9-31.4] |
| `kalypso_block:4/4x4/4` | 9.45 [9.21-9.49] | 3.34x | 40 [0] | 100% | n/a | 15.9 |
| `kalypso_block:4/4x4/4+coalesce` | 9.50 [9.18-9.57] | 3.32x | 40 [0] | 100% | n/a | 15.9 |
| `kalypso_block:4/2x2/4` | 16.10 [15.35-16.12] | 1.96x | 136 [0] | 100% | n/a | 17.1 |
| `kalypso_block:4/2x2/4+coalesce` | 16.05 [15.35-16.20] | 1.97x | 136 [0] | 100% | n/a | 17.1 |


Diagnostics:

| Arm | Prompt tokens | Gen tokens | Prefix hit rate | Peak RSS (MiB) | Items/s | First output (s) | Truncated | Accuracy | Precision | Recall | Self-agreement |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 365006 | 4380 | 0.47 | 11540 [11540-11540] | 32.45 [31.73-32.53] | 12.78 [8.78-15.27] | 464 | 1.00 | 1.00 | 1.00 | 100% |
| `kalypso_block:4/4x4/4` | 80809 | 192 | 0.39 | 11540 [11540-11540] | 108.33 [107.89-111.12] | n/a | 0 | 1.00 | n/a | 0.00 | 100% |
| `kalypso_block:4/4x4/4+coalesce` | 80809 | 192 | 0.39 | 11540 [11540-11540] | 107.80 [106.96-111.53] | n/a | 0 | 1.00 | n/a | 0.00 | 100% |
| `kalypso_block:4/2x2/4` | 158573 | 384 | 0.44 | 11540 [11540-11540] | 63.60 [63.53-66.73] | n/a | 0 | 1.00 | n/a | 0.00 | 100% |
| `kalypso_block:4/2x2/4+coalesce` | 158573 | 384 | 0.44 | 11540 [11540-11540] | 63.80 [63.21-66.71] | n/a | 0 | 1.00 | n/a | 0.00 | 100% |


Stages (medians over repeats; times from query start):

| Arm | Stage | Operators | Block | Tasks | Rows in | Rows out | Out/in | Occupancy | Start (s) | End (s) | Overlap (s) | Fallbacks |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 0 | SemFilter | 1 | 32 | 32 | 512 | 16.00 | n/a | 0.25 [0.25-0.27] | 21.00 [20.84-21.44] | n/a | 0 |
| `kalypso_per_tuple` | 1 | SemFilter+SemFilter | 1 | 512 | 512 | 4 | 0.01 | n/a | 0.96 [0.96-0.99] | 31.55 [31.46-32.26] | 20.01 [19.88-20.47] | 0 |
| `kalypso_block:4/4x4/4` | 0 | BlockSemFilter | 4 | 8 | 32 | 16 | 0.50 | 1.00 | 0.34 [0.34-0.35] | 4.73 [4.61-4.74] | n/a | 0 |
| `kalypso_block:4/4x4/4` | 1 | BlockSemJoin | 4 | 4 | 16 | 0 | 0.00 | 1.00 | 1.54 [1.53-1.55] | 9.45 [9.21-9.49] | 3.18 [3.07-3.20] | 0 |
| `kalypso_block:4/4x4/4` | 2 | BlockSemFilter | 4 | 0 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | 0 |
| `kalypso_block:4/4x4/4+coalesce` | 0 | BlockSemFilter | 4 | 8 | 32 | 16 | 0.50 | 1.00 | 0.35 [0.34-0.35] | 4.74 [4.60-4.76] | n/a | 0 |
| `kalypso_block:4/4x4/4+coalesce` | 1 | BlockSemJoin | 4 | 4 | 16 | 0 | 0.00 | 1.00 | 1.55 [1.54-1.56] | 9.50 [9.18-9.57] | 3.18 [3.06-3.21] | 0 |
| `kalypso_block:4/4x4/4+coalesce` | 2 | BlockSemFilter | 4 | 0 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | 0 |
| `kalypso_block:4/2x2/4` | 0 | BlockSemFilter | 4 | 8 | 32 | 16 | 0.50 | 1.00 | 0.59 [0.57-0.59] | 5.34 [5.15-5.37] | n/a | 0 |
| `kalypso_block:4/2x2/4` | 1 | BlockSemJoin | 2 | 8 | 16 | 0 | 0.00 | 1.00 | 1.74 [1.73-1.76] | 16.10 [15.34-16.11] | 3.60 [3.41-3.61] | 0 |
| `kalypso_block:4/2x2/4` | 2 | BlockSemFilter | 4 | 0 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | 0 |
| `kalypso_block:4/2x2/4+coalesce` | 0 | BlockSemFilter | 4 | 8 | 32 | 16 | 0.50 | 1.00 | 0.59 [0.58-0.60] | 5.32 [5.17-5.39] | n/a | 0 |
| `kalypso_block:4/2x2/4+coalesce` | 1 | BlockSemJoin | 2 | 8 | 16 | 0 | 0.00 | 1.00 | 1.74 [1.74-1.76] | 16.05 [15.35-16.20] | 3.58 [3.43-3.63] | 0 |
| `kalypso_block:4/2x2/4+coalesce` | 2 | BlockSemFilter | 4 | 0 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | 0 |


Operators (first repeat):

| Arm | Query op | Kalypso operator | Rows in | Rows out | Answers began with | Other answers (samples) |
|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 0: sem_filter | SemFilter | 32 | 16 | false 16, true 16 | - |
| `kalypso_per_tuple` | 1: join | SemFilter | 512 | 4 | false 508, true 4 | - |
| `kalypso_per_tuple` | 2: sem_filter | SemFilter | 4 | 4 | true 4 | - |
| `kalypso_block:4/4x4/4` | 0: sem_filter | BlockSemFilter | 32 | 16 | false 16, true 16 | - |
| `kalypso_block:4/4x4/4` | 1: join | BlockSemJoin | 16 | 0 | - | - |
| `kalypso_block:4/4x4/4` | 2: sem_filter | BlockSemFilter | 0 | 0 | - | - |
| `kalypso_block:4/4x4/4+coalesce` | 0: sem_filter | BlockSemFilter | 32 | 16 | false 16, true 16 | - |
| `kalypso_block:4/4x4/4+coalesce` | 1: join | BlockSemJoin | 16 | 0 | - | - |
| `kalypso_block:4/4x4/4+coalesce` | 2: sem_filter | BlockSemFilter | 0 | 0 | - | - |
| `kalypso_block:4/2x2/4` | 0: sem_filter | BlockSemFilter | 32 | 16 | false 16, true 16 | - |
| `kalypso_block:4/2x2/4` | 1: join | BlockSemJoin | 16 | 0 | - | - |
| `kalypso_block:4/2x2/4` | 2: sem_filter | BlockSemFilter | 0 | 0 | - | - |
| `kalypso_block:4/2x2/4+coalesce` | 0: sem_filter | BlockSemFilter | 32 | 16 | false 16, true 16 | - |
| `kalypso_block:4/2x2/4+coalesce` | 1: join | BlockSemJoin | 16 | 0 | - | - |
| `kalypso_block:4/2x2/4+coalesce` | 2: sem_filter | BlockSemFilter | 0 | 0 | - | - |



Work passed downstream (medians). Gold: 16 rows should pass the filter, 4 final outputs are expected.

| Arm | Rows past first filter | Pairs past join | Final outputs | Comparable |
|---|---|---|---|---|
| `kalypso_per_tuple` | 16 | 4 | 4 | reference |
| `kalypso_block:4/4x4/4` | 16 | 0 | 0 | **no, work drifted** |
| `kalypso_block:4/4x4/4+coalesce` | 16 | 0 | 0 | **no, work drifted** |
| `kalypso_block:4/2x2/4` | 16 | 0 | 0 | **no, work drifted** |
| `kalypso_block:4/2x2/4+coalesce` | 16 | 0 | 0 | **no, work drifted** |

Drifted by more than 10% from `kalypso_per_tuple`: `kalypso_block:4/4x4/4`, `kalypso_block:4/4x4/4+coalesce`, `kalypso_block:4/2x2/4`, `kalypso_block:4/2x2/4+coalesce`. Their latency and requests reflect different work, not just blocking.
