# E6

## s10: s = 0.1, f = 1

Model `Qwen/Qwen2.5-14B-Instruct` on NVIDIA L40S (Linux-4.19.0-gvisor-x86_64-with-glibc2.39); workload `arxiv_fjf` (pipeline), 32 x 32 = 1024 pairs, 2 positive; 3 repeats after 1 warmup; prefix cache reset per trial: yes; git `6368c4b90` (dirty).
Server: `python3 -m vllm.entrypoints.cli.main serve Qwen/Qwen2.5-14B-Instruct --host 127.0.0.1 --port 8000 --enable-prefix-caching --no-async-scheduling --api-server-count 1 --max-model-len 16384` (started Tue Oct 20 21:16:06 2026).

| Arm | Completion time (s) | Speedup vs kalypso_per_tuple | LLM calls [retries + fallbacks] | Agrees w/ kalypso_per_tuple | F1 vs labels | Peak KV % |
|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 7.57 [7.43-7.69] | 1.00x | 129 [0] | 100% | n/a | 23.3 |
| `kalypso_block:4/4x4/16` | 5.32 [5.30-5.43] | 1.42x | 17 [0] | 100% | 0.50 | 6.4 |
| `kalypso_block:4/4x4/4` | 5.32 [5.29-5.36] | 1.42x | 17 [0] | 100% | 0.50 | 6.4 [6.4-6.4] |
| `kalypso_block:4/2x2/4` | 6.69 [6.66-6.74] | 1.13x | 41 [1] | 100% | 0.00 | 8.1 [7.6-8.1] |


Diagnostics:

| Arm | Prompt tokens | Gen tokens | Prefix hit rate | Peak RSS (MiB) | Items/s | First output (s) | Truncated | Accuracy | Precision | Recall | Self-agreement |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 79552 | 1032 | 0.44 | 11549 [11549-11549] | 135.36 [133.21-137.78] | n/a | 122 | 1.00 | n/a | 0.00 | 100% |
| `kalypso_block:4/4x4/16` | 27946 | 162 | 0.30 | 11549 [11549-11549] | 192.37 [188.58-193.13] | 5.32 [5.30-5.43] | 0 | 1.00 | 0.50 | 0.50 | 100% |
| `kalypso_block:4/4x4/4` | 27946 | 162 | 0.30 | 11549 [11549-11549] | 192.48 [191.22-193.54] | 5.32 [5.29-5.35] | 0 | 1.00 | 0.50 | 0.50 | 100% |
| `kalypso_block:4/2x2/4` | 47069 | 198 | 0.39 | 11549 [11549-11549] | 153.04 [151.84-153.69] | 6.29 [6.26-6.34] | 0 | 1.00 | 0.00 | 0.00 | 100% |


Stages (medians over repeats; times from query start):

| Arm | Stage | Operators | Block | Tasks | Rows in | Rows out | Out/in | Occupancy | Start (s) | End (s) | Overlap (s) | Fallbacks |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 0 | SemFilter | 1 | 32 | 32 | 96 | 3.00 | n/a | 0.19 [0.18-0.20] | 4.29 [4.23-4.34] | n/a | 0 |
| `kalypso_per_tuple` | 1 | SemFilter+SemFilter | 1 | 96 | 96 | 0 | 0.00 | n/a | 0.92 [0.91-0.92] | 7.56 [7.43-7.68] | 3.38 [3.30-3.42] | 0 |
| `kalypso_block:4/4x4/16` | 0 | BlockSemFilter | 4 | 8 | 32 | 4 | 0.12 | 1.00 | 0.28 [0.28-0.30] | 2.59 [2.58-2.64] | n/a | 0 |
| `kalypso_block:4/4x4/16` | 1 | BlockSemJoin | 4 | 1 | 4 | 2 | 0.50 | 1.00 | 2.61 [2.61-2.67] | 4.90 [4.88-5.00] | 0.00 | 0 |
| `kalypso_block:4/4x4/16` | 2 | BlockSemFilter | 16 | 1 | 2 | 2 | 1.00 | 0.12 | 4.90 [4.88-5.00] | 5.32 [5.30-5.43] | 0.00 | 0 |
| `kalypso_block:4/4x4/4` | 0 | BlockSemFilter | 4 | 8 | 32 | 4 | 0.12 | 1.00 | 0.28 [0.28-0.28] | 2.59 [2.58-2.60] | n/a | 0 |
| `kalypso_block:4/4x4/4` | 1 | BlockSemJoin | 4 | 1 | 4 | 2 | 0.50 | 1.00 | 2.62 [2.61-2.63] | 4.89 [4.87-4.93] | 0.00 | 0 |
| `kalypso_block:4/4x4/4` | 2 | BlockSemFilter | 4 | 1 | 2 | 2 | 1.00 | 0.50 | 4.90 [4.87-4.93] | 5.32 [5.29-5.35] | 0.00 | 0 |
| `kalypso_block:4/2x2/4` | 0 | BlockSemFilter | 4 | 8 | 32 | 4 | 0.12 | 1.00 | 0.46 [0.46-0.47] | 3.16 [3.15-3.18] | n/a | 0 |
| `kalypso_block:4/2x2/4` | 1 | BlockSemJoin | 2 | 2 | 4 | 1 | 0.25 | 1.00 | 1.66 [1.65-1.67] | 6.69 [6.66-6.74] | 1.50 [1.50-1.51] | 0 |
| `kalypso_block:4/2x2/4` | 2 | BlockSemFilter | 4 | 1 | 1 | 1 | 1.00 | 0.25 | 6.05 [6.02-6.10] | 6.29 [6.26-6.34] | 0.64 [0.64-0.64] | 1 |


Operators (first repeat):

| Arm | Query op | Kalypso operator | Rows in | Rows out | Answers began with | Other answers (samples) |
|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 0: sem_filter | SemFilter | 32 | 3 | false 29, true 3 | - |
| `kalypso_per_tuple` | 1: join | SemFilter | 96 | 1 | false 95, true 1 | - |
| `kalypso_per_tuple` | 2: sem_filter | SemFilter | 1 | 0 | false 1 | - |
| `kalypso_block:4/4x4/16` | 0: sem_filter | BlockSemFilter | 32 | 4 | false 28, true 4 | - |
| `kalypso_block:4/4x4/16` | 1: join | BlockSemJoin | 4 | 2 | - | - |
| `kalypso_block:4/4x4/16` | 2: sem_filter | BlockSemFilter | 2 | 2 | true 2 | - |
| `kalypso_block:4/4x4/4` | 0: sem_filter | BlockSemFilter | 32 | 4 | false 28, true 4 | - |
| `kalypso_block:4/4x4/4` | 1: join | BlockSemJoin | 4 | 2 | - | - |
| `kalypso_block:4/4x4/4` | 2: sem_filter | BlockSemFilter | 2 | 2 | true 2 | - |
| `kalypso_block:4/2x2/4` | 0: sem_filter | BlockSemFilter | 32 | 4 | false 28, true 4 | - |
| `kalypso_block:4/2x2/4` | 1: join | BlockSemJoin | 4 | 1 | - | - |
| `kalypso_block:4/2x2/4` | 2: sem_filter | BlockSemFilter | 1 | 1 | true 1 | - |


Work passed downstream (medians). Gold: 3 rows should pass the filter, 2 final outputs are expected.

| Arm | Rows past first filter | Pairs past join | Final outputs | Comparable |
|---|---|---|---|---|
| `kalypso_per_tuple` | 3 | 1 | 0 | reference |
| `kalypso_block:4/4x4/16` | 4 | 2 | 2 | **no, work drifted** |
| `kalypso_block:4/4x4/4` | 4 | 2 | 2 | **no, work drifted** |
| `kalypso_block:4/2x2/4` | 4 | 1 | 1 | **no, work drifted** |

Drifted by more than 10% from `kalypso_per_tuple`: `kalypso_block:4/4x4/16`, `kalypso_block:4/4x4/4`, `kalypso_block:4/2x2/4`. Their latency and requests reflect different work, not just blocking.

## s30: s = 0.3, f = 1

Model `Qwen/Qwen2.5-14B-Instruct` on NVIDIA L40S (Linux-4.19.0-gvisor-x86_64-with-glibc2.39); workload `arxiv_fjf` (pipeline), 32 x 32 = 1024 pairs, 4 positive; 3 repeats after 1 warmup; prefix cache reset per trial: yes; git `6368c4b90` (dirty).
Server: `python3 -m vllm.entrypoints.cli.main serve Qwen/Qwen2.5-14B-Instruct --host 127.0.0.1 --port 8000 --enable-prefix-caching --no-async-scheduling --api-server-count 1 --max-model-len 16384` (started Tue Oct 20 21:16:06 2026).

| Arm | Completion time (s) | Speedup vs kalypso_per_tuple | LLM calls [retries + fallbacks] | Agrees w/ kalypso_per_tuple | F1 vs labels | Peak KV % |
|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 20.86 [20.78-20.98] | 1.00x | 357 [0] | 100% | 0.57 | 24.3 [24.2-24.6] |
| `kalypso_block:4/4x4/16` | 7.74 [7.71-7.75] | 2.70x | 33 [1] | 100% | 0.40 | 9.9 [9.9-9.9] |
| `kalypso_block:4/4x4/4` | 8.04 [7.99-8.06] | 2.59x | 34 [1] | 100% | 0.40 | 10.0 [10.0-10.1] |
| `kalypso_block:4/2x2/4` | 11.33 [11.10-11.35] | 1.84x | 90 [1] | 100% | 0.40 | 8.4 [8.4-8.5] |


Diagnostics:

| Arm | Prompt tokens | Gen tokens | Prefix hit rate | Peak RSS (MiB) | Items/s | First output (s) | Truncated | Accuracy | Precision | Recall | Self-agreement |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 233761 | 2850 | 0.47 [0.47-0.47] | 11551 [11551-11551] | 49.10 [48.80-49.28] | 13.50 [11.30-16.49] | 308 [308-309] | 1.00 | 0.67 | 0.50 | 100% |
| `kalypso_block:4/4x4/16` | 58707 | 182 | 0.34 | 11551 [11551-11551] | 132.33 [132.16-132.76] | 6.55 [6.52-6.55] | 0 | 1.00 | 1.00 | 0.25 | 100% |
| `kalypso_block:4/4x4/4` | 59859 | 200 | 0.34 | 11551 [11551-11551] | 127.35 [127.05-128.21] | 8.04 [7.98-8.06] | 0 | 1.00 | 1.00 | 0.25 | 100% |
| `kalypso_block:4/2x2/4` | 103661 [103198-103661] | 310 [304-310] | 0.43 [0.43-0.43] | 11551 [11551-11551] | 90.38 [90.20-92.24] | 11.34 [11.33-11.35] | 0 | 1.00 [1.00-1.00] | 1.00 | 0.25 [0.00-0.25] | 100% |


Stages (medians over repeats; times from query start):

| Arm | Stage | Operators | Block | Tasks | Rows in | Rows out | Out/in | Occupancy | Start (s) | End (s) | Overlap (s) | Fallbacks |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 0 | SemFilter | 1 | 32 | 32 | 320 | 10.00 | n/a | 0.19 [0.19-0.20] | 9.97 [9.93-10.00] | n/a | 0 |
| `kalypso_per_tuple` | 1 | SemFilter+SemFilter | 1 | 320 | 320 | 3 | 0.01 | n/a | 0.93 [0.93-0.94] | 20.85 [20.77-20.98] | 9.03 [9.00-9.07] | 0 |
| `kalypso_block:4/4x4/16` | 0 | BlockSemFilter | 4 | 8 | 32 | 10 | 0.31 | 1.00 | 0.29 [0.28-0.30] | 3.40 [3.40-3.42] | n/a | 0 |
| `kalypso_block:4/4x4/16` | 1 | BlockSemJoin | 4 | 3 | 10 | 1 | 0.10 | 0.83 | 1.46 [1.46-1.47] | 7.74 [7.71-7.74] | 1.94 [1.94-1.94] | 0 |
| `kalypso_block:4/4x4/16` | 2 | BlockSemFilter | 16 | 1 | 1 | 1 | 1.00 | 0.06 | 5.95 [5.93-5.96] | 6.55 [6.52-6.55] | 1.79 [1.78-1.79] | 1 |
| `kalypso_block:4/4x4/4` | 0 | BlockSemFilter | 4 | 8 | 32 | 10 | 0.31 | 1.00 | 0.28 [0.28-0.31] | 3.63 [3.62-3.63] | n/a | 0 |
| `kalypso_block:4/4x4/4` | 1 | BlockSemJoin | 4 | 3 | 10 | 3 | 0.30 | 0.83 | 1.46 [1.45-1.48] | 7.91 [7.86-7.93] | 2.16 [2.16-2.17] | 0 |
| `kalypso_block:4/4x4/4` | 2 | BlockSemFilter | 4 | 2 | 3 | 1 | 0.33 | 0.38 | 6.33 [6.32-6.51] | 8.04 [7.98-8.06] | 1.60 [1.35-1.60] | 1 |
| `kalypso_block:4/2x2/4` | 0 | BlockSemFilter | 4 | 8 | 32 | 10 | 0.31 | 1.00 | 0.47 [0.46-0.47] | 3.94 [3.93-3.95] | n/a | 0 |
| `kalypso_block:4/2x2/4` | 1 | BlockSemJoin | 2 | 5 | 10 | 3 [2-3] | 0.30 [0.20-0.30] | 1.00 | 1.61 [1.60-1.62] | 11.20 [11.10-11.23] | 2.33 [2.33-2.33] | 0 |
| `kalypso_block:4/2x2/4` | 2 | BlockSemFilter | 4 | 2 [1-2] | 3 [2-3] | 1 [0-1] | 0.33 [0.00-0.33] | 0.38 [0.38-0.50] | 10.21 [10.19-10.21] | 11.33 [11.10-11.35] | 1.01 [0.89-1.02] | 1 [0-1] |


Operators (first repeat):

| Arm | Query op | Kalypso operator | Rows in | Rows out | Answers began with | Other answers (samples) |
|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 0: sem_filter | SemFilter | 32 | 10 | false 22, true 10 | - |
| `kalypso_per_tuple` | 1: join | SemFilter | 320 | 5 | false 315, true 5 | - |
| `kalypso_per_tuple` | 2: sem_filter | SemFilter | 5 | 3 | false 2, true 3 | - |
| `kalypso_block:4/4x4/16` | 0: sem_filter | BlockSemFilter | 32 | 10 | false 22, true 10 | - |
| `kalypso_block:4/4x4/16` | 1: join | BlockSemJoin | 10 | 1 | - | - |
| `kalypso_block:4/4x4/16` | 2: sem_filter | BlockSemFilter | 1 | 1 | true 1 | - |
| `kalypso_block:4/4x4/4` | 0: sem_filter | BlockSemFilter | 32 | 10 | false 22, true 10 | - |
| `kalypso_block:4/4x4/4` | 1: join | BlockSemJoin | 10 | 3 | - | - |
| `kalypso_block:4/4x4/4` | 2: sem_filter | BlockSemFilter | 3 | 1 | false 2, true 1 | - |
| `kalypso_block:4/2x2/4` | 0: sem_filter | BlockSemFilter | 32 | 10 | false 22, true 10 | - |
| `kalypso_block:4/2x2/4` | 1: join | BlockSemJoin | 10 | 2 | - | - |
| `kalypso_block:4/2x2/4` | 2: sem_filter | BlockSemFilter | 2 | 0 | false 2 | - |


Work passed downstream (medians). Gold: 10 rows should pass the filter, 4 final outputs are expected.

| Arm | Rows past first filter | Pairs past join | Final outputs | Comparable |
|---|---|---|---|---|
| `kalypso_per_tuple` | 10 | 5 | 3 | reference |
| `kalypso_block:4/4x4/16` | 10 | 1 | 1 | **no, work drifted** |
| `kalypso_block:4/4x4/4` | 10 | 3 | 1 | **no, work drifted** |
| `kalypso_block:4/2x2/4` | 10 | 3 | 1 | **no, work drifted** |

Drifted by more than 10% from `kalypso_per_tuple`: `kalypso_block:4/4x4/16`, `kalypso_block:4/4x4/4`, `kalypso_block:4/2x2/4`. Their latency and requests reflect different work, not just blocking.

## s60: s = 0.6, f = 1

Model `Qwen/Qwen2.5-14B-Instruct` on NVIDIA L40S (Linux-4.19.0-gvisor-x86_64-with-glibc2.39); workload `arxiv_fjf` (pipeline), 32 x 32 = 1024 pairs, 10 positive; 3 repeats after 1 warmup; prefix cache reset per trial: yes; git `6368c4b90` (dirty).
Server: `python3 -m vllm.entrypoints.cli.main serve Qwen/Qwen2.5-14B-Instruct --host 127.0.0.1 --port 8000 --enable-prefix-caching --no-async-scheduling --api-server-count 1 --max-model-len 16384` (started Tue Oct 20 21:16:06 2026).

| Arm | Completion time (s) | Speedup vs kalypso_per_tuple | LLM calls [retries + fallbacks] | Agrees w/ kalypso_per_tuple | F1 vs labels | Peak KV % |
|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 38.73 [38.70-38.73] | 1.00x | 653 [0] | 100% | 0.67 | 24.7 [24.4-24.9] |
| `kalypso_block:4/4x4/16` | 11.39 [11.36-11.49] | 3.40x | 52 [4] | 99% | 0.14 | 13.1 [13.0-13.6] |
| `kalypso_block:4/4x4/4` | 11.34 [11.33-11.48] | 3.42x | 52 [4] | 99% | 0.14 [0.14-0.29] | 13.1 [13.1-13.6] |
| `kalypso_block:4/2x2/4` | 19.05 [18.98-19.06] | 2.03x | 158 [4] | 99% | 0.12 [0.00-0.12] | 12.0 [12.0-12.2] |


Diagnostics:

| Arm | Prompt tokens | Gen tokens | Prefix hit rate | Peak RSS (MiB) | Items/s | First output (s) | Truncated | Accuracy | Precision | Recall | Self-agreement |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 438085 | 5214 [5211-5216] | 0.47 [0.47-0.47] | 11552 [11552-11552] | 26.44 [26.44-26.46] | 9.61 [9.59-9.91] | 546 [544-546] | 0.99 | 0.75 | 0.60 | 100% |
| `kalypso_block:4/4x4/16` | 96916 [96916-96962] | 232 [232-234] | 0.37 [0.37-0.37] | 11552 [11552-11552] | 89.92 [89.13-90.16] | 8.47 [8.32-8.67] | 0 | 0.99 | 0.25 | 0.10 | 100% |
| `kalypso_block:4/4x4/4` | 96916 [96820-96916] | 232 | 0.37 [0.37-0.37] | 11552 [11552-11552] | 90.30 [89.17-90.37] | 8.31 [8.12-8.66] | 0 | 0.99 [0.99-0.99] | 0.25 [0.25-0.50] | 0.10 [0.10-0.20] | 100% |
| `kalypso_block:4/2x2/4` | 183142 [183142-183854] | 506 [492-506] | 0.43 [0.43-0.43] | 11552 [11552-11552] | 53.74 [53.73-53.95] | 14.77 [14.57-15.08] | 1 [0-1] | 0.99 [0.98-0.99] | 0.14 [0.00-0.14] | 0.10 [0.00-0.10] | 100% |


Stages (medians over repeats; times from query start):

| Arm | Stage | Operators | Block | Tasks | Rows in | Rows out | Out/in | Occupancy | Start (s) | End (s) | Overlap (s) | Fallbacks |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 0 | SemFilter | 1 | 32 | 32 | 608 | 19.00 | n/a | 0.19 [0.19-0.20] | 19.37 [19.37-19.39] | n/a | 0 |
| `kalypso_per_tuple` | 1 | SemFilter+SemFilter | 1 | 608 | 608 | 8 | 0.01 | n/a | 0.92 [0.91-0.92] | 38.72 [38.69-38.72] | 18.45 [18.44-18.48] | 0 |
| `kalypso_block:4/4x4/16` | 0 | BlockSemFilter | 4 | 8 | 32 | 18 | 0.56 | 1.00 | 0.28 [0.28-0.34] | 4.61 [4.50-4.66] | n/a | 0 |
| `kalypso_block:4/4x4/16` | 1 | BlockSemJoin | 4 | 5 | 18 | 4 | 0.22 | 0.90 | 1.42 [1.41-1.49] | 11.22 [11.16-11.26] | 3.13 [3.08-3.25] | 0 |
| `kalypso_block:4/4x4/16` | 2 | BlockSemFilter | 16 | 4 | 4 | 4 | 1.00 | 0.06 | 8.18 [7.96-8.34] | 11.38 [11.36-11.49] | 2.98 [2.91-3.26] | 4 |
| `kalypso_block:4/4x4/4` | 0 | BlockSemFilter | 4 | 8 | 32 | 18 | 0.56 | 1.00 | 0.29 [0.28-0.35] | 4.66 [4.62-4.67] | n/a | 0 |
| `kalypso_block:4/4x4/4` | 1 | BlockSemJoin | 4 | 5 | 18 | 4 | 0.22 | 0.90 | 1.42 [1.42-1.50] | 11.14 [11.11-11.25] | 3.24 [3.12-3.24] | 0 |
| `kalypso_block:4/4x4/4` | 2 | BlockSemFilter | 4 | 4 | 4 | 4 | 1.00 | 0.25 | 8.17 [7.98-8.33] | 11.34 [11.33-11.48] | 2.97 [2.92-3.13] | 4 |
| `kalypso_block:4/2x2/4` | 0 | BlockSemFilter | 4 | 8 | 32 | 18 | 0.56 | 1.00 | 0.47 [0.47-0.58] | 5.02 [5.00-5.16] | n/a | 0 |
| `kalypso_block:4/2x2/4` | 1 | BlockSemJoin | 2 | 9 | 18 | 9 [9-10] | 0.50 [0.50-0.56] | 1.00 | 1.60 [1.59-1.73] | 18.38 [18.31-18.40] | 3.43 [3.40-3.43] | 0 |
| `kalypso_block:4/2x2/4` | 2 | BlockSemFilter | 4 | 6 | 9 [9-10] | 7 [7-9] | 0.78 [0.78-0.90] | 0.38 [0.38-0.42] | 14.07 [13.87-14.36] | 19.05 [18.98-19.05] | 4.32 [4.04-4.45] | 4 [3-4] |


Operators (first repeat):

| Arm | Query op | Kalypso operator | Rows in | Rows out | Answers began with | Other answers (samples) |
|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 0: sem_filter | SemFilter | 32 | 19 | false 13, true 19 | - |
| `kalypso_per_tuple` | 1: join | SemFilter | 608 | 13 | false 595, true 13 | - |
| `kalypso_per_tuple` | 2: sem_filter | SemFilter | 13 | 8 | false 5, true 8 | - |
| `kalypso_block:4/4x4/16` | 0: sem_filter | BlockSemFilter | 32 | 18 | false 14, true 18 | - |
| `kalypso_block:4/4x4/16` | 1: join | BlockSemJoin | 18 | 4 | - | - |
| `kalypso_block:4/4x4/16` | 2: sem_filter | BlockSemFilter | 4 | 4 | true 4 | - |
| `kalypso_block:4/4x4/4` | 0: sem_filter | BlockSemFilter | 32 | 18 | false 14, true 18 | - |
| `kalypso_block:4/4x4/4` | 1: join | BlockSemJoin | 18 | 4 | - | - |
| `kalypso_block:4/4x4/4` | 2: sem_filter | BlockSemFilter | 4 | 4 | true 4 | - |
| `kalypso_block:4/2x2/4` | 0: sem_filter | BlockSemFilter | 32 | 18 | false 14, true 18 | - |
| `kalypso_block:4/2x2/4` | 1: join | BlockSemJoin | 18 | 9 | - | - |
| `kalypso_block:4/2x2/4` | 2: sem_filter | BlockSemFilter | 9 | 7 | false 2, true 7 | - |


Work passed downstream (medians). Gold: 19 rows should pass the filter, 10 final outputs are expected.

| Arm | Rows past first filter | Pairs past join | Final outputs | Comparable |
|---|---|---|---|---|
| `kalypso_per_tuple` | 19 | 13 | 8 | reference |
| `kalypso_block:4/4x4/16` | 18 | 4 | 4 | **no, work drifted** |
| `kalypso_block:4/4x4/4` | 18 | 4 | 4 | **no, work drifted** |
| `kalypso_block:4/2x2/4` | 18 | 9 | 7 | **no, work drifted** |

Drifted by more than 10% from `kalypso_per_tuple`: `kalypso_block:4/4x4/16`, `kalypso_block:4/4x4/4`, `kalypso_block:4/2x2/4`. Their latency and requests reflect different work, not just blocking.

## s90: s = 0.9, f = 1

Model `Qwen/Qwen2.5-14B-Instruct` on NVIDIA L40S (Linux-4.19.0-gvisor-x86_64-with-glibc2.39); workload `arxiv_fjf` (pipeline), 32 x 32 = 1024 pairs, 9 positive; 3 repeats after 1 warmup; prefix cache reset per trial: yes; git `6368c4b90` (dirty).
Server: `python3 -m vllm.entrypoints.cli.main serve Qwen/Qwen2.5-14B-Instruct --host 127.0.0.1 --port 8000 --enable-prefix-caching --no-async-scheduling --api-server-count 1 --max-model-len 16384` (started Tue Oct 20 21:16:06 2026).

| Arm | Completion time (s) | Speedup vs kalypso_per_tuple | LLM calls [retries + fallbacks] | Agrees w/ kalypso_per_tuple | F1 vs labels | Peak KV % |
|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 51.79 [51.46-51.90] | 1.00x | 972 [0] | 100% | 0.18 | 24.5 [24.2-24.6] |
| `kalypso_block:4/4x4/16` | 13.04 [12.93-13.16] | 3.97x | 68 [3] | 100% | 0.19 [0.18-0.20] | 16.5 [14.1-16.5] |
| `kalypso_block:4/4x4/4` | 12.96 [12.93-12.99] | 4.00x | 67 [2] | 100% | 0.20 | 14.1 [14.1-16.5] |
| `kalypso_block:4/2x2/4` | 22.77 [22.50-22.82] | 2.27x | 238 [5] | 100% | 0.20 [0.17-0.31] | 12.3 [12.1-12.3] |


Diagnostics:

| Arm | Prompt tokens | Gen tokens | Prefix hit rate | Peak RSS (MiB) | Items/s | First output (s) | Truncated | Accuracy | Precision | Recall | Self-agreement |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 609467 | 7768 [7768-7770] | 0.49 [0.49-0.49] | 11553 [11553-11553] | 19.77 [19.73-19.90] | 15.81 [15.13-16.47] | 800 [800-802] | 0.99 | 0.50 | 0.11 | 100% |
| `kalypso_block:4/4x4/16` | 125230 [124781-125727] | 274 [268-280] | 0.41 [0.41-0.41] | 11553 [11553-11553] | 78.50 [77.79-79.18] | 10.67 [10.57-10.77] | 0 | 0.99 [0.99-0.99] | 0.75 [0.50-1.00] | 0.11 [0.00-0.11] | 100% |
| `kalypso_block:4/4x4/4` | 124781 [124781-125230] | 268 [268-274] | 0.41 [0.41-0.41] | 11553 [11553-11553] | 79.01 [78.80-79.20] | 10.72 | 0 | 0.99 [0.99-0.99] | 1.00 | 0.00 [0.00-0.11] | 100% |
| `kalypso_block:4/2x2/4` | 249915 [248882-250028] | 622 [616-622] | 0.46 [0.46-0.46] | 11553 [11553-11553] | 44.97 [44.87-45.51] | 20.19 [20.14-22.46] | 0 | 0.99 [0.99-0.99] | 0.50 [0.33-1.00] | 0.11 [0.11-0.22] | 100% |


Stages (medians over repeats; times from query start):

| Arm | Stage | Operators | Block | Tasks | Rows in | Rows out | Out/in | Occupancy | Start (s) | End (s) | Overlap (s) | Fallbacks |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 0 | SemFilter | 1 | 32 | 32 | 928 | 29.00 | n/a | 0.19 [0.18-0.20] | 10.27 [10.26-19.68] | n/a | 0 |
| `kalypso_per_tuple` | 1 | SemFilter+SemFilter | 1 | 928 | 928 | 2 | 0.00 | n/a | 0.92 [0.92-0.95] | 51.78 [51.45-51.88] | 9.34 [9.32-18.75] | 0 |
| `kalypso_block:4/4x4/16` | 0 | BlockSemFilter | 4 | 8 | 32 | 27 | 0.84 | 1.00 | 0.27 [0.26-0.27] | 5.32 [5.28-5.41] | n/a | 0 |
| `kalypso_block:4/4x4/16` | 1 | BlockSemJoin | 4 | 7 | 27 | 5 [4-6] | 0.19 [0.15-0.22] | 0.96 | 1.44 [1.43-1.44] | 12.92 [12.81-12.95] | 3.88 [3.85-3.97] | 0 |
| `kalypso_block:4/4x4/16` | 2 | BlockSemFilter | 16 | 4 [3-5] | 5 [4-6] | 1 [0-2] | 0.20 [0.00-0.33] | 0.08 [0.07-0.08] | 9.53 [9.50-9.62] | 13.04 [12.93-13.16] | 3.38 [3.19-3.45] | 3 [2-4] |
| `kalypso_block:4/4x4/4` | 0 | BlockSemFilter | 4 | 8 | 32 | 27 | 0.84 | 1.00 | 0.27 [0.27-0.28] | 5.39 [5.28-5.42] | n/a | 0 |
| `kalypso_block:4/4x4/4` | 1 | BlockSemJoin | 4 | 7 | 27 | 4 [4-5] | 0.15 [0.15-0.19] | 0.96 | 1.44 [1.43-1.46] | 12.83 [12.80-12.87] | 3.96 [3.84-3.96] | 0 |
| `kalypso_block:4/4x4/4` | 2 | BlockSemFilter | 4 | 3 [3-4] | 4 [4-5] | 0 [0-1] | 0.00 [0.00-0.20] | 0.33 [0.31-0.33] | 9.61 [9.48-9.63] | 12.96 [12.93-12.99] | 3.20 [3.19-3.38] | 2 [2-3] |
| `kalypso_block:4/2x2/4` | 0 | BlockSemFilter | 4 | 8 | 32 | 27 | 0.84 | 1.00 | 0.44 [0.44-0.46] | 6.27 [6.10-6.27] | n/a | 0 |
| `kalypso_block:4/2x2/4` | 1 | BlockSemJoin | 2 | 14 | 27 | 7 [5-7] | 0.26 [0.19-0.26] | 0.96 | 1.61 [1.60-1.62] | 22.50 [22.48-22.59] | 4.65 [4.50-4.66] | 0 |
| `kalypso_block:4/2x2/4` | 2 | BlockSemFilter | 4 | 6 [3-6] | 7 [5-7] | 3 [1-4] | 0.43 [0.20-0.57] | 0.29 [0.29-0.42] | 19.36 [19.15-19.81] | 22.77 [22.46-22.82] | 3.23 [2.69-3.32] | 5 [1-5] |


Operators (first repeat):

| Arm | Query op | Kalypso operator | Rows in | Rows out | Answers began with | Other answers (samples) |
|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 0: sem_filter | SemFilter | 32 | 29 | false 3, true 29 | - |
| `kalypso_per_tuple` | 1: join | SemFilter | 928 | 12 | false 916, true 12 | - |
| `kalypso_per_tuple` | 2: sem_filter | SemFilter | 12 | 2 | false 10, true 2 | - |
| `kalypso_block:4/4x4/16` | 0: sem_filter | BlockSemFilter | 32 | 27 | false 5, true 27 | - |
| `kalypso_block:4/4x4/16` | 1: join | BlockSemJoin | 27 | 5 | - | - |
| `kalypso_block:4/4x4/16` | 2: sem_filter | BlockSemFilter | 5 | 1 | false 4, true 1 | - |
| `kalypso_block:4/4x4/4` | 0: sem_filter | BlockSemFilter | 32 | 27 | false 5, true 27 | - |
| `kalypso_block:4/4x4/4` | 1: join | BlockSemJoin | 27 | 5 | - | - |
| `kalypso_block:4/4x4/4` | 2: sem_filter | BlockSemFilter | 5 | 1 | false 4, true 1 | - |
| `kalypso_block:4/2x2/4` | 0: sem_filter | BlockSemFilter | 32 | 27 | false 5, true 27 | - |
| `kalypso_block:4/2x2/4` | 1: join | BlockSemJoin | 27 | 5 | - | - |
| `kalypso_block:4/2x2/4` | 2: sem_filter | BlockSemFilter | 5 | 1 | false 4, true 1 | - |


Work passed downstream (medians). Gold: 29 rows should pass the filter, 9 final outputs are expected.

| Arm | Rows past first filter | Pairs past join | Final outputs | Comparable |
|---|---|---|---|---|
| `kalypso_per_tuple` | 29 | 12 | 2 | reference |
| `kalypso_block:4/4x4/16` | 27 | 5 | 1 | **no, work drifted** |
| `kalypso_block:4/4x4/4` | 27 | 4 | 0 | **no, work drifted** |
| `kalypso_block:4/2x2/4` | 27 | 7 | 3 | **no, work drifted** |

Drifted by more than 10% from `kalypso_per_tuple`: `kalypso_block:4/4x4/16`, `kalypso_block:4/4x4/4`, `kalypso_block:4/2x2/4`. Their latency and requests reflect different work, not just blocking.
