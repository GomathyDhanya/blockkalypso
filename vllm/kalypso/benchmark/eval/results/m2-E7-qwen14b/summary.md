# E7

## f02: s = 0.5, f = 0.2

Model `Qwen/Qwen2.5-14B-Instruct` on NVIDIA L40S (Linux-4.19.0-gvisor-x86_64-with-glibc2.39); workload `arxiv_fjf` (pipeline), 32 x 32 = 1024 pairs, 4 positive; 3 repeats after 1 warmup; prefix cache reset per trial: yes; git `6368c4b90` (dirty).
Server: `python3 -m vllm.entrypoints.cli.main serve Qwen/Qwen2.5-14B-Instruct --host 127.0.0.1 --port 8000 --enable-prefix-caching --no-async-scheduling --api-server-count 1 --max-model-len 16384` (started Tue Oct 20 21:16:06 2026).

| Arm | Completion time (s) | Speedup vs kalypso_per_tuple | LLM calls [retries + fallbacks] | Agrees w/ kalypso_per_tuple | F1 vs labels | Peak KV % |
|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 33.15 [32.77-33.18] | 1.00x | 548 [0] | 100% | 1.00 | 31.0 [30.3-31.4] |
| `kalypso_block:4/4x4/16` | 9.57 [9.55-9.78] | 3.46x | 40 [0] | 100% | n/a | 15.9 |
| `kalypso_block:4/4x4/4` | 9.61 [9.52-9.68] | 3.45x | 40 [0] | 100% | n/a | 15.9 |
| `kalypso_block:4/2x2/4` | 16.04 [15.96-16.18] | 2.07x | 136 [0] | 100% | n/a | 15.2 [15.2-17.1] |


Diagnostics:

| Arm | Prompt tokens | Gen tokens | Prefix hit rate | Peak RSS (MiB) | Items/s | First output (s) | Truncated | Accuracy | Precision | Recall | Self-agreement |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 365006 | 4376 | 0.47 | 11539 [11539-11539] | 30.89 [30.87-31.25] | 13.03 [11.23-15.55] | 463 | 1.00 | 1.00 | 1.00 | 100% |
| `kalypso_block:4/4x4/16` | 80809 | 192 | 0.39 | 11539 [11539-11539] | 106.96 [104.71-107.18] | n/a | 0 | 1.00 | n/a | 0.00 | 100% |
| `kalypso_block:4/4x4/4` | 80809 | 192 | 0.39 | 11539 [11539-11539] | 106.51 [105.83-107.53] | n/a | 0 | 1.00 | n/a | 0.00 | 100% |
| `kalypso_block:4/2x2/4` | 158573 | 384 | 0.44 [0.44-0.44] | 11539 [11539-11539] | 63.82 [63.28-64.18] | n/a | 0 | 1.00 | n/a | 0.00 | 100% |


Stages (medians over repeats; times from query start):

| Arm | Stage | Operators | Block | Tasks | Rows in | Rows out | Out/in | Occupancy | Start (s) | End (s) | Overlap (s) | Fallbacks |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 0 | SemFilter | 1 | 32 | 32 | 512 | 16.00 | n/a | 0.20 [0.19-0.20] | 21.95 [21.66-21.96] | n/a | 0 |
| `kalypso_per_tuple` | 1 | SemFilter+SemFilter | 1 | 512 | 512 | 4 | 0.01 | n/a | 0.91 [0.91-0.91] | 33.14 [32.76-33.16] | 21.04 [20.75-21.05] | 0 |
| `kalypso_block:4/4x4/16` | 0 | BlockSemFilter | 4 | 8 | 32 | 16 | 0.50 | 1.00 | 0.30 [0.29-0.43] | 4.74 [4.74-4.92] | n/a | 0 |
| `kalypso_block:4/4x4/16` | 1 | BlockSemJoin | 4 | 4 | 16 | 0 | 0.00 | 1.00 | 1.49 [1.48-1.67] | 9.57 [9.55-9.78] | 3.25 [3.25-3.26] | 0 |
| `kalypso_block:4/4x4/16` | 2 | BlockSemFilter | 16 | 0 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | 0 |
| `kalypso_block:4/4x4/4` | 0 | BlockSemFilter | 4 | 8 | 32 | 16 | 0.50 | 1.00 | 0.30 [0.29-0.33] | 4.76 [4.72-4.80] | n/a | 0 |
| `kalypso_block:4/4x4/4` | 1 | BlockSemJoin | 4 | 4 | 16 | 0 | 0.00 | 1.00 | 1.49 [1.48-1.53] | 9.61 [9.52-9.67] | 3.27 [3.24-3.27] | 0 |
| `kalypso_block:4/4x4/4` | 2 | BlockSemFilter | 4 | 0 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | 0 |
| `kalypso_block:4/2x2/4` | 0 | BlockSemFilter | 4 | 8 | 32 | 16 | 0.50 | 1.00 | 0.48 [0.48-0.52] | 5.34 [5.33-5.37] | n/a | 0 |
| `kalypso_block:4/2x2/4` | 1 | BlockSemJoin | 2 | 8 | 16 | 0 | 0.00 | 1.00 | 1.64 [1.64-1.70] | 16.04 [15.95-16.18] | 3.69 [3.67-3.71] | 0 |
| `kalypso_block:4/2x2/4` | 2 | BlockSemFilter | 4 | 0 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | 0 |


Operators (first repeat):

| Arm | Query op | Kalypso operator | Rows in | Rows out | Answers began with | Other answers (samples) |
|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 0: sem_filter | SemFilter | 32 | 16 | false 16, true 16 | - |
| `kalypso_per_tuple` | 1: join | SemFilter | 512 | 4 | false 508, true 4 | - |
| `kalypso_per_tuple` | 2: sem_filter | SemFilter | 4 | 4 | true 4 | - |
| `kalypso_block:4/4x4/16` | 0: sem_filter | BlockSemFilter | 32 | 16 | false 16, true 16 | - |
| `kalypso_block:4/4x4/16` | 1: join | BlockSemJoin | 16 | 0 | - | - |
| `kalypso_block:4/4x4/16` | 2: sem_filter | BlockSemFilter | 0 | 0 | - | - |
| `kalypso_block:4/4x4/4` | 0: sem_filter | BlockSemFilter | 32 | 16 | false 16, true 16 | - |
| `kalypso_block:4/4x4/4` | 1: join | BlockSemJoin | 16 | 0 | - | - |
| `kalypso_block:4/4x4/4` | 2: sem_filter | BlockSemFilter | 0 | 0 | - | - |
| `kalypso_block:4/2x2/4` | 0: sem_filter | BlockSemFilter | 32 | 16 | false 16, true 16 | - |
| `kalypso_block:4/2x2/4` | 1: join | BlockSemJoin | 16 | 0 | - | - |
| `kalypso_block:4/2x2/4` | 2: sem_filter | BlockSemFilter | 0 | 0 | - | - |


Work passed downstream (medians). Gold: 16 rows should pass the filter, 4 final outputs are expected.

| Arm | Rows past first filter | Pairs past join | Final outputs | Comparable |
|---|---|---|---|---|
| `kalypso_per_tuple` | 16 | 4 | 4 | reference |
| `kalypso_block:4/4x4/16` | 16 | 0 | 0 | **no, work drifted** |
| `kalypso_block:4/4x4/4` | 16 | 0 | 0 | **no, work drifted** |
| `kalypso_block:4/2x2/4` | 16 | 0 | 0 | **no, work drifted** |

Drifted by more than 10% from `kalypso_per_tuple`: `kalypso_block:4/4x4/16`, `kalypso_block:4/4x4/4`, `kalypso_block:4/2x2/4`. Their latency and requests reflect different work, not just blocking.

## f10: s = 0.5, f = 1.0

Model `Qwen/Qwen2.5-14B-Instruct` on NVIDIA L40S (Linux-4.19.0-gvisor-x86_64-with-glibc2.39); workload `arxiv_fjf` (pipeline), 32 x 32 = 1024 pairs, 9 positive; 3 repeats after 1 warmup; prefix cache reset per trial: yes; git `6368c4b90` (dirty).
Server: `python3 -m vllm.entrypoints.cli.main serve Qwen/Qwen2.5-14B-Instruct --host 127.0.0.1 --port 8000 --enable-prefix-caching --no-async-scheduling --api-server-count 1 --max-model-len 16384` (started Tue Oct 20 21:16:06 2026).

| Arm | Completion time (s) | Speedup vs kalypso_per_tuple | LLM calls [retries + fallbacks] | Agrees w/ kalypso_per_tuple | F1 vs labels | Peak KV % |
|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 33.10 [32.94-33.16] | 1.00x | 557 [0] | 100% | 0.80 | 32.6 [32.3-32.6] |
| `kalypso_block:4/4x4/16` | 10.64 [9.84-10.64] | 3.11x | 42 [1] | 99% | 0.00 | 15.9 |
| `kalypso_block:4/4x4/4` | 9.80 [9.77-9.80] | 3.38x | 42 [2] | 99% | 0.00 | 15.9 |
| `kalypso_block:4/2x2/4` | 16.59 [16.55-16.84] | 1.99x | 142 [6] | 99% | 0.14 [0.00-0.14] | 16.6 [16.0-16.6] |


Diagnostics:

| Arm | Prompt tokens | Gen tokens | Prefix hit rate | Peak RSS (MiB) | Items/s | First output (s) | Truncated | Accuracy | Precision | Recall | Self-agreement |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 369422 | 4446 | 0.48 [0.48-0.48] | 11540 [11539-11540] | 30.94 [30.88-31.09] | 9.19 [9.18-11.26] | 476 [475-476] | 1.00 | 1.00 | 0.67 | 100% |
| `kalypso_block:4/4x4/16` | 82447 [81554-82447] | 224 [204-224] | 0.39 [0.39-0.39] | 11540 [11539-11540] | 96.24 [96.20-104.08] | 7.96 [7.95-8.00] | 0 | 0.99 | 0.00 | 0.00 | 100% |
| `kalypso_block:4/4x4/4` | 81554 | 204 | 0.39 | 11540 [11539-11540] | 104.47 [104.47-104.77] | 7.97 [7.96-8.00] | 0 | 0.99 | 0.00 | 0.00 | 100% |
| `kalypso_block:4/2x2/4` | 161267 [161237-161267] | 420 [420-424] | 0.44 [0.44-0.44] | 11540 [11539-11540] | 61.72 [60.81-61.87] | 14.13 [14.00-14.16] | 0 | 0.99 | 0.20 [0.00-0.20] | 0.11 [0.00-0.11] | 100% |


Stages (medians over repeats; times from query start):

| Arm | Stage | Operators | Block | Tasks | Rows in | Rows out | Out/in | Occupancy | Start (s) | End (s) | Overlap (s) | Fallbacks |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 0 | SemFilter | 1 | 32 | 32 | 512 | 16.00 | n/a | 0.19 [0.19-0.19] | 21.96 [21.69-25.96] | n/a | 0 |
| `kalypso_per_tuple` | 1 | SemFilter+SemFilter | 1 | 512 | 512 | 6 | 0.01 | n/a | 0.90 [0.89-0.90] | 33.09 [32.93-33.15] | 21.06 [20.79-25.07] | 0 |
| `kalypso_block:4/4x4/16` | 0 | BlockSemFilter | 4 | 8 | 32 | 16 | 0.50 | 1.00 | 0.29 [0.29-0.30] | 4.80 [4.80-4.84] | n/a | 0 |
| `kalypso_block:4/4x4/16` | 1 | BlockSemJoin | 4 | 4 | 16 | 4 [2-4] | 0.25 [0.12-0.25] | 1.00 | 1.48 [1.48-1.51] | 9.97 [9.71-9.98] | 3.31 [3.31-3.33] | 0 |
| `kalypso_block:4/4x4/16` | 2 | BlockSemFilter | 16 | 2 | 4 [2-4] | 1 | 0.25 [0.25-0.50] | 0.12 [0.06-0.12] | 7.63 [7.62-7.67] | 10.64 [9.84-10.64] | 2.35 [2.03-2.35] | 1 [1-2] |
| `kalypso_block:4/4x4/4` | 0 | BlockSemFilter | 4 | 8 | 32 | 16 | 0.50 | 1.00 | 0.29 [0.29-0.29] | 4.79 [4.79-4.81] | n/a | 0 |
| `kalypso_block:4/4x4/4` | 1 | BlockSemJoin | 4 | 4 | 16 | 2 | 0.12 | 1.00 | 1.48 [1.48-1.49] | 9.67 [9.64-9.67] | 3.32 [3.31-3.32] | 0 |
| `kalypso_block:4/4x4/4` | 2 | BlockSemFilter | 4 | 2 | 2 | 1 | 0.50 | 0.25 | 7.64 [7.63-7.67] | 9.80 [9.77-9.80] | 2.00 [2.00-2.04] | 2 |
| `kalypso_block:4/2x2/4` | 0 | BlockSemFilter | 4 | 8 | 32 | 16 | 0.50 | 1.00 | 0.48 [0.48-0.48] | 5.31 [5.29-5.46] | n/a | 0 |
| `kalypso_block:4/2x2/4` | 1 | BlockSemJoin | 2 | 8 | 16 | 6 | 0.38 | 1.00 | 1.63 [1.63-1.63] | 16.30 [16.26-16.35] | 3.68 [3.65-3.83] | 0 |
| `kalypso_block:4/2x2/4` | 2 | BlockSemFilter | 4 | 6 [5-6] | 6 | 5 [3-5] | 0.83 [0.50-0.83] | 0.25 [0.25-0.30] | 13.46 [13.32-13.49] | 16.59 [16.55-16.84] | 2.80 [2.80-3.03] | 6 [4-6] |


Operators (first repeat):

| Arm | Query op | Kalypso operator | Rows in | Rows out | Answers began with | Other answers (samples) |
|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 0: sem_filter | SemFilter | 32 | 16 | false 16, true 16 | - |
| `kalypso_per_tuple` | 1: join | SemFilter | 512 | 13 | false 499, true 13 | - |
| `kalypso_per_tuple` | 2: sem_filter | SemFilter | 13 | 6 | false 7, true 6 | - |
| `kalypso_block:4/4x4/16` | 0: sem_filter | BlockSemFilter | 32 | 16 | false 16, true 16 | - |
| `kalypso_block:4/4x4/16` | 1: join | BlockSemJoin | 16 | 2 | - | - |
| `kalypso_block:4/4x4/16` | 2: sem_filter | BlockSemFilter | 2 | 1 | false 1, true 1 | - |
| `kalypso_block:4/4x4/4` | 0: sem_filter | BlockSemFilter | 32 | 16 | false 16, true 16 | - |
| `kalypso_block:4/4x4/4` | 1: join | BlockSemJoin | 16 | 2 | - | - |
| `kalypso_block:4/4x4/4` | 2: sem_filter | BlockSemFilter | 2 | 1 | false 1, true 1 | - |
| `kalypso_block:4/2x2/4` | 0: sem_filter | BlockSemFilter | 32 | 16 | false 16, true 16 | - |
| `kalypso_block:4/2x2/4` | 1: join | BlockSemJoin | 16 | 6 | - | - |
| `kalypso_block:4/2x2/4` | 2: sem_filter | BlockSemFilter | 6 | 3 | false 3, true 3 | - |


Work passed downstream (medians). Gold: 16 rows should pass the filter, 9 final outputs are expected.

| Arm | Rows past first filter | Pairs past join | Final outputs | Comparable |
|---|---|---|---|---|
| `kalypso_per_tuple` | 16 | 13 | 6 | reference |
| `kalypso_block:4/4x4/16` | 16 | 4 | 1 | **no, work drifted** |
| `kalypso_block:4/4x4/4` | 16 | 2 | 1 | **no, work drifted** |
| `kalypso_block:4/2x2/4` | 16 | 6 | 5 | **no, work drifted** |

Drifted by more than 10% from `kalypso_per_tuple`: `kalypso_block:4/4x4/16`, `kalypso_block:4/4x4/4`, `kalypso_block:4/2x2/4`. Their latency and requests reflect different work, not just blocking.

## f30: s = 0.5, f = 3.0

Model `Qwen/Qwen2.5-14B-Instruct` on NVIDIA L40S (Linux-4.19.0-gvisor-x86_64-with-glibc2.39); workload `arxiv_fjf` (pipeline), 32 x 32 = 1024 pairs, 23 positive; 3 repeats after 1 warmup; prefix cache reset per trial: yes; git `6368c4b90` (dirty).
Server: `python3 -m vllm.entrypoints.cli.main serve Qwen/Qwen2.5-14B-Instruct --host 127.0.0.1 --port 8000 --enable-prefix-caching --no-async-scheduling --api-server-count 1 --max-model-len 16384` (started Tue Oct 20 21:16:06 2026).

| Arm | Completion time (s) | Speedup vs kalypso_per_tuple | LLM calls [retries + fallbacks] | Agrees w/ kalypso_per_tuple | F1 vs labels | Peak KV % |
|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 32.74 [32.74-32.76] | 1.00x | 571 [0] | 100% | 0.65 | 33.4 [33.0-34.1] |
| `kalypso_block:4/4x4/16` | 11.01 [10.67-11.09] | 2.98x | 44 [1] | 99% | 0.08 [0.07-0.15] | 18.7 [18.6-18.7] |
| `kalypso_block:4/4x4/4` | 11.11 [11.07-11.13] | 2.95x | 44 [1] | 98% | 0.07 | 18.6 [18.6-18.7] |
| `kalypso_block:4/2x2/4` | 17.68 [17.63-17.89] | 1.85x | 144 [3] | 99% | 0.21 | 14.5 [14.5-16.7] |


Diagnostics:

| Arm | Prompt tokens | Gen tokens | Prefix hit rate | Peak RSS (MiB) | Items/s | First output (s) | Truncated | Accuracy | Precision | Recall | Self-agreement |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 374021 | 4558 | 0.49 | 11540 [11540-11540] | 31.28 [31.25-31.28] | 9.26 [9.22-11.22] | 509 | 0.99 | 0.86 | 0.52 | 100% |
| `kalypso_block:4/4x4/16` | 84469 [84019-84669] | 270 [262-270] | 0.38 [0.38-0.38] | 11540 [11540-11540] | 93.05 [92.31-95.96] | 10.16 [10.10-10.38] | 0 | 0.98 [0.98-0.98] | 0.50 [0.25-0.67] | 0.04 [0.04-0.09] | 100% |
| `kalypso_block:4/4x4/4` | 84732 | 270 | 0.38 | 11540 [11540-11540] | 92.21 [92.03-92.49] | 10.17 [10.13-10.19] | 0 | 0.98 | 0.25 | 0.04 | 100% |
| `kalypso_block:4/2x2/4` | 164206 | 504 | 0.43 | 11540 [11540-11540] | 57.92 [57.25-58.07] | 15.56 [15.46-15.76] | 0 | 0.98 | 0.50 | 0.13 | 100% |


Stages (medians over repeats; times from query start):

| Arm | Stage | Operators | Block | Tasks | Rows in | Rows out | Out/in | Occupancy | Start (s) | End (s) | Overlap (s) | Fallbacks |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 0 | SemFilter | 1 | 32 | 32 | 512 | 16.00 | n/a | 0.20 [0.19-0.23] | 21.36 [21.36-21.40] | n/a | 0 |
| `kalypso_per_tuple` | 1 | SemFilter+SemFilter | 1 | 512 | 512 | 14 | 0.03 | n/a | 0.90 [0.90-0.94] | 32.73 [32.73-32.75] | 20.46 [20.46-20.46] | 0 |
| `kalypso_block:4/4x4/16` | 0 | BlockSemFilter | 4 | 8 | 32 | 16 | 0.50 | 1.00 | 0.29 [0.29-0.29] | 4.58 [4.58-4.59] | n/a | 0 |
| `kalypso_block:4/4x4/16` | 1 | BlockSemJoin | 4 | 4 | 16 | 9 [8-9] | 0.56 [0.50-0.56] | 1.00 | 1.48 [1.48-1.48] | 10.31 [10.18-10.37] | 3.10 [3.09-3.11] | 0 |
| `kalypso_block:4/4x4/16` | 2 | BlockSemFilter | 16 | 4 | 9 [8-9] | 3 [2-4] | 0.33 [0.25-0.44] | 0.14 [0.12-0.14] | 8.12 [7.99-8.42] | 11.00 [10.67-11.09] | 2.19 [1.76-2.39] | 1 |
| `kalypso_block:4/4x4/4` | 0 | BlockSemFilter | 4 | 8 | 32 | 16 | 0.50 | 1.00 | 0.29 [0.29-0.29] | 4.60 [4.55-4.61] | n/a | 0 |
| `kalypso_block:4/4x4/4` | 1 | BlockSemJoin | 4 | 4 | 16 | 9 | 0.56 | 1.00 | 1.49 [1.47-1.49] | 10.39 [10.35-10.41] | 3.11 [3.08-3.12] | 0 |
| `kalypso_block:4/4x4/4` | 2 | BlockSemFilter | 4 | 4 | 9 | 4 | 0.44 | 0.56 | 8.14 [8.09-8.16] | 11.10 [11.07-11.12] | 2.25 [2.24-2.25] | 1 |
| `kalypso_block:4/2x2/4` | 0 | BlockSemFilter | 4 | 8 | 32 | 16 | 0.50 | 1.00 | 0.47 [0.47-0.58] | 5.35 [5.35-5.36] | n/a | 0 |
| `kalypso_block:4/2x2/4` | 1 | BlockSemJoin | 2 | 8 | 16 | 15 | 0.94 | 1.00 | 1.63 [1.62-1.75] | 16.93 [16.89-17.14] | 3.72 [3.60-3.73] | 0 |
| `kalypso_block:4/2x2/4` | 2 | BlockSemFilter | 4 | 8 | 15 | 6 | 0.40 | 0.47 | 12.82 [12.79-13.41] | 17.68 [17.63-17.88] | 4.10 [3.73-4.11] | 3 |


Operators (first repeat):

| Arm | Query op | Kalypso operator | Rows in | Rows out | Answers began with | Other answers (samples) |
|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 0: sem_filter | SemFilter | 32 | 16 | false 16, true 16 | - |
| `kalypso_per_tuple` | 1: join | SemFilter | 512 | 27 | false 485, true 27 | - |
| `kalypso_per_tuple` | 2: sem_filter | SemFilter | 27 | 14 | false 13, true 14 | - |
| `kalypso_block:4/4x4/16` | 0: sem_filter | BlockSemFilter | 32 | 16 | false 16, true 16 | - |
| `kalypso_block:4/4x4/16` | 1: join | BlockSemJoin | 16 | 9 | - | - |
| `kalypso_block:4/4x4/16` | 2: sem_filter | BlockSemFilter | 9 | 3 | false 6, true 3 | - |
| `kalypso_block:4/4x4/4` | 0: sem_filter | BlockSemFilter | 32 | 16 | false 16, true 16 | - |
| `kalypso_block:4/4x4/4` | 1: join | BlockSemJoin | 16 | 9 | - | - |
| `kalypso_block:4/4x4/4` | 2: sem_filter | BlockSemFilter | 9 | 4 | false 5, true 4 | - |
| `kalypso_block:4/2x2/4` | 0: sem_filter | BlockSemFilter | 32 | 16 | false 16, true 16 | - |
| `kalypso_block:4/2x2/4` | 1: join | BlockSemJoin | 16 | 15 | - | - |
| `kalypso_block:4/2x2/4` | 2: sem_filter | BlockSemFilter | 15 | 6 | false 9, true 6 | - |


Work passed downstream (medians). Gold: 16 rows should pass the filter, 23 final outputs are expected.

| Arm | Rows past first filter | Pairs past join | Final outputs | Comparable |
|---|---|---|---|---|
| `kalypso_per_tuple` | 16 | 27 | 14 | reference |
| `kalypso_block:4/4x4/16` | 16 | 9 | 3 | **no, work drifted** |
| `kalypso_block:4/4x4/4` | 16 | 9 | 4 | **no, work drifted** |
| `kalypso_block:4/2x2/4` | 16 | 15 | 6 | **no, work drifted** |

Drifted by more than 10% from `kalypso_per_tuple`: `kalypso_block:4/4x4/16`, `kalypso_block:4/4x4/4`, `kalypso_block:4/2x2/4`. Their latency and requests reflect different work, not just blocking.

## f50: s = 0.5, f = 5.0

Model `Qwen/Qwen2.5-14B-Instruct` on NVIDIA L40S (Linux-4.19.0-gvisor-x86_64-with-glibc2.39); workload `arxiv_fjf` (pipeline), 32 x 32 = 1024 pairs, 37 positive; 3 repeats after 1 warmup; prefix cache reset per trial: yes; git `6368c4b90` (dirty).
Server: `python3 -m vllm.entrypoints.cli.main serve Qwen/Qwen2.5-14B-Instruct --host 127.0.0.1 --port 8000 --enable-prefix-caching --no-async-scheduling --api-server-count 1 --max-model-len 16384` (started Tue Oct 20 21:16:06 2026).

| Arm | Completion time (s) | Speedup vs kalypso_per_tuple | LLM calls [retries + fallbacks] | Agrees w/ kalypso_per_tuple | F1 vs labels | Peak KV % |
|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 31.62 [31.55-31.87] | 1.00x | 582 [0] | 100% | 0.62 | 34.4 [33.8-34.4] |
| `kalypso_block:4/4x4/16` | 18.49 [17.87-18.51] | 1.71x | 45 [1] | 98% | 0.20 [0.20-0.27] | 36.5 [20.8-36.5] |
| `kalypso_block:4/4x4/4` | 15.46 [15.46-15.74] | 2.04x | 52 [3] | 98% | 0.24 [0.20-0.24] | 20.6 [20.6-27.6] |
| `kalypso_block:4/2x2/4` | 18.46 [18.34-18.53] | 1.71x | 146 [3] | 98% | 0.30 [0.22-0.30] | 18.8 [18.8-19.1] |


Diagnostics:

| Arm | Prompt tokens | Gen tokens | Prefix hit rate | Peak RSS (MiB) | Items/s | First output (s) | Truncated | Accuracy | Precision | Recall | Self-agreement |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 373583 | 4640 | 0.51 | 11544 [11544-11544] | 32.38 [32.13-32.46] | 9.53 [9.21-11.24] | 515 | 0.98 | 0.86 | 0.49 | 100% |
| `kalypso_block:4/4x4/16` | 98830 [96131-98830] | 604 [527-604] | 0.32 [0.32-0.33] | 11544 [11544-11544] | 55.37 [55.33-57.31] | 10.12 [10.05-10.13] | 0 | 0.96 [0.96-0.97] | 0.42 [0.42-0.75] | 0.14 [0.14-0.16] | 99% |
| `kalypso_block:4/4x4/4` | 98134 [98134-99400] | 546 [546-590] | 0.34 [0.33-0.34] | 11544 [11544-11544] | 66.21 [65.07-66.23] | 8.36 [8.34-8.38] | 0 | 0.96 [0.96-0.96] | 0.50 [0.38-0.50] | 0.16 [0.14-0.16] | 100% |
| `kalypso_block:4/2x2/4` | 166100 [165403-166100] | 590 [582-590] | 0.42 [0.42-0.43] | 11544 [11544-11544] | 55.47 [55.27-55.83] | 12.10 [12.05-12.11] | 0 | 0.97 [0.96-0.97] | 0.70 [0.56-0.70] | 0.19 [0.14-0.19] | 100% |


Stages (medians over repeats; times from query start):

| Arm | Stage | Operators | Block | Tasks | Rows in | Rows out | Out/in | Occupancy | Start (s) | End (s) | Overlap (s) | Fallbacks |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 0 | SemFilter | 1 | 32 | 32 | 512 | 16.00 | n/a | 0.19 [0.19-0.19] | 12.56 [8.86-12.90] | n/a | 0 |
| `kalypso_per_tuple` | 1 | SemFilter+SemFilter | 1 | 512 | 512 | 21 | 0.04 | n/a | 0.90 [0.90-0.91] | 31.61 [31.54-31.86] | 11.66 [7.95-11.99] | 0 |
| `kalypso_block:4/4x4/16` | 0 | BlockSemFilter | 4 | 8 | 32 | 16 | 0.50 | 1.00 | 0.28 [0.28-0.29] | 3.75 [3.74-3.76] | n/a | 0 |
| `kalypso_block:4/4x4/16` | 1 | BlockSemJoin | 4 | 4 | 16 | 43 [36-43] | 2.69 [2.25-2.69] | 1.00 | 1.48 [1.47-1.49] | 13.18 [13.15-13.30] | 2.27 [2.26-2.27] | 0 |
| `kalypso_block:4/4x4/16` | 2 | BlockSemFilter | 16 | 5 | 43 [36-43] | 12 [8-12] | 0.28 [0.22-0.28] | 0.54 [0.45-0.54] | 7.95 [7.95-8.01] | 18.49 [17.87-18.50] | 5.22 [5.20-5.29] | 1 |
| `kalypso_block:4/4x4/4` | 0 | BlockSemFilter | 4 | 8 | 32 | 16 | 0.50 | 1.00 | 0.28 [0.28-0.29] | 3.72 [3.72-3.74] | n/a | 0 |
| `kalypso_block:4/4x4/4` | 1 | BlockSemJoin | 4 | 4 | 16 | 39 [39-43] | 2.44 [2.44-2.69] | 1.00 | 1.47 [1.47-1.48] | 13.38 [13.07-13.38] | 2.25 [2.25-2.26] | 0 |
| `kalypso_block:4/4x4/4` | 2 | BlockSemFilter | 4 | 12 [12-13] | 39 [39-43] | 12 [12-13] | 0.31 [0.30-0.31] | 0.81 [0.81-0.83] | 7.90 [7.88-7.92] | 15.46 [15.46-15.73] | 5.48 [5.15-5.50] | 3 |
| `kalypso_block:4/2x2/4` | 0 | BlockSemFilter | 4 | 8 | 32 | 16 | 0.50 | 1.00 | 0.46 [0.46-0.48] | 4.83 [4.83-4.84] | n/a | 0 |
| `kalypso_block:4/2x2/4` | 1 | BlockSemJoin | 2 | 8 | 16 | 25 [24-25] | 1.56 [1.50-1.56] | 1.00 | 1.62 [1.62-1.63] | 17.32 [17.22-17.38] | 3.22 [3.20-3.23] | 0 |
| `kalypso_block:4/2x2/4` | 2 | BlockSemFilter | 4 | 10 | 25 [24-25] | 10 [9-10] | 0.40 [0.38-0.40] | 0.62 [0.60-0.62] | 11.64 [11.59-11.65] | 18.46 [18.34-18.52] | 5.73 [5.57-5.74] | 3 |


Operators (first repeat):

| Arm | Query op | Kalypso operator | Rows in | Rows out | Answers began with | Other answers (samples) |
|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 0: sem_filter | SemFilter | 32 | 16 | false 16, true 16 | - |
| `kalypso_per_tuple` | 1: join | SemFilter | 512 | 38 | false 474, true 38 | - |
| `kalypso_per_tuple` | 2: sem_filter | SemFilter | 38 | 21 | false 17, true 21 | - |
| `kalypso_block:4/4x4/16` | 0: sem_filter | BlockSemFilter | 32 | 16 | false 16, true 16 | - |
| `kalypso_block:4/4x4/16` | 1: join | BlockSemJoin | 16 | 36 | - | - |
| `kalypso_block:4/4x4/16` | 2: sem_filter | BlockSemFilter | 36 | 8 | false 28, true 8 | - |
| `kalypso_block:4/4x4/4` | 0: sem_filter | BlockSemFilter | 32 | 16 | false 16, true 16 | - |
| `kalypso_block:4/4x4/4` | 1: join | BlockSemJoin | 16 | 39 | - | - |
| `kalypso_block:4/4x4/4` | 2: sem_filter | BlockSemFilter | 39 | 12 | false 27, true 12 | - |
| `kalypso_block:4/2x2/4` | 0: sem_filter | BlockSemFilter | 32 | 16 | false 16, true 16 | - |
| `kalypso_block:4/2x2/4` | 1: join | BlockSemJoin | 16 | 25 | - | - |
| `kalypso_block:4/2x2/4` | 2: sem_filter | BlockSemFilter | 25 | 10 | false 15, true 10 | - |


Work passed downstream (medians). Gold: 16 rows should pass the filter, 37 final outputs are expected.

| Arm | Rows past first filter | Pairs past join | Final outputs | Comparable |
|---|---|---|---|---|
| `kalypso_per_tuple` | 16 | 38 | 21 | reference |
| `kalypso_block:4/4x4/16` | 16 | 43 | 12 | **no, work drifted** |
| `kalypso_block:4/4x4/4` | 16 | 39 | 12 | **no, work drifted** |
| `kalypso_block:4/2x2/4` | 16 | 25 | 10 | **no, work drifted** |

Drifted by more than 10% from `kalypso_per_tuple`: `kalypso_block:4/4x4/16`, `kalypso_block:4/4x4/4`, `kalypso_block:4/2x2/4`. Their latency and requests reflect different work, not just blocking.
