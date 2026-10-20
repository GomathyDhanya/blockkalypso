# E4

## fjf_s05_f1: s = 0.5, f = 1

Model `Qwen/Qwen2.5-14B-Instruct` on NVIDIA L40S (Linux-4.19.0-gvisor-x86_64-with-glibc2.39); workload `arxiv_fjf` (pipeline), 32 x 32 = 1024 pairs, 9 positive; 3 repeats after 1 warmup; prefix cache reset per trial: yes; git `6368c4b90`.
Server: `python3 -m vllm.entrypoints.cli.main serve Qwen/Qwen2.5-14B-Instruct --host 127.0.0.1 --port 8000 --enable-prefix-caching --no-async-scheduling --api-server-count 1 --max-model-len 16384` (started Tue Oct 20 20:52:47 2026).

| Arm | Completion time (s) | Speedup vs kalypso_per_tuple | LLM calls [retries + fallbacks] | Agrees w/ kalypso_per_tuple | F1 vs labels | Peak KV % |
|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 31.79 [31.41-31.91] | 1.00x | 557 [0] | 100% | 0.80 | 32.5 [32.5-32.6] |
| `kalypso_block:2/2x2/2` | 16.52 [15.61-16.65] | 1.92x | 150 [4] | 99% | 0.15 [0.12-0.22] | 16.5 [16.4-17.3] |
| `kalypso_block:4/4x4/4` | 9.62 [9.54-9.64] | 3.31x | 42 [2] | 99% | 0.00 | 15.9 |
| `kalypso_block:8/8x8/8` | 10.13 [10.13-10.19] | 3.14x | 14 [0] | 100% | 0.36 | 17.4 [17.3-17.4] |
| `kalypso_block:16/16x16/16` | 9.83 [9.81-9.84] | 3.23x | 4 [0] | 99% | n/a | 17.3 [15.8-17.3] |


Diagnostics:

| Arm | Prompt tokens | Gen tokens | Prefix hit rate | Peak RSS (MiB) | Items/s | First output (s) | Truncated | Accuracy | Precision | Recall | Self-agreement |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 369422 | 4450 | 0.48 | 11549 [11549-11549] | 32.21 [32.09-32.60] | 12.52 [10.84-12.78] | 474 | 1.00 | 1.00 | 0.67 | 100% |
| `kalypso_block:2/2x2/2` | 164011 [161374-164110] | 462 [414-466] | 0.43 [0.43-0.44] | 11549 [11549-11549] | 61.99 [61.51-65.58] | 14.37 [14.03-14.54] | 0 | 0.99 [0.99-0.99] | 0.22 [0.14-0.25] | 0.11 [0.11-0.22] | 100% |
| `kalypso_block:4/4x4/4` | 81554 | 204 | 0.39 | 11549 [11549-11549] | 106.50 [106.22-107.31] | 7.86 [7.78-7.87] | 0 | 0.99 | 0.00 | 0.00 | 100% |
| `kalypso_block:8/8x8/8` | 47713 | 222 | 0.28 | 11549 [11549-11549] | 101.09 [100.49-101.09] | 8.02 [8.02-8.07] | 0 | 0.99 | 1.00 | 0.22 | 100% |
| `kalypso_block:16/16x16/16` | 25740 | 146 | 0.17 | 11549 [11549-11549] | 104.17 [104.03-104.43] | n/a | 0 | 0.99 | n/a | 0.00 | 100% |


Stages (medians over repeats; times from query start):

| Arm | Stage | Operators | Block | Tasks | Rows in | Rows out | Out/in | Occupancy | Start (s) | End (s) | Overlap (s) | Fallbacks |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 0 | SemFilter | 1 | 32 | 32 | 512 | 16.00 | n/a | 0.24 [0.23-0.24] | 20.79 [20.77-21.17] | n/a | 0 |
| `kalypso_per_tuple` | 1 | SemFilter+SemFilter | 1 | 512 | 512 | 6 | 0.01 | n/a | 0.94 [0.94-0.95] | 31.78 [31.40-31.90] | 19.85 [19.83-20.23] | 0 |
| `kalypso_block:2/2x2/2` | 0 | BlockSemFilter | 2 | 16 | 32 | 16 | 0.50 | 1.00 | 0.59 [0.50-0.60] | 4.23 [3.49-4.28] | n/a | 0 |
| `kalypso_block:2/2x2/2` | 1 | BlockSemJoin | 2 | 8 | 16 | 10 [5-10] | 0.62 [0.31-0.62] | 1.00 | 1.37 [1.29-1.38] | 15.76 [15.44-15.91] | 2.86 [2.20-2.89] | 0 |
| `kalypso_block:2/2x2/2` | 2 | BlockSemFilter | 2 | 6 [5-7] | 10 [5-10] | 7 [4-9] | 0.80 [0.70-0.90] | 0.71 [0.50-0.83] | 13.85 [13.40-13.87] | 16.51 [15.61-16.64] | 2.06 [1.58-2.36] | 4 [2-5] |
| `kalypso_block:4/4x4/4` | 0 | BlockSemFilter | 4 | 8 | 32 | 16 | 0.50 | 1.00 | 0.36 [0.28-0.36] | 4.76 [4.69-4.76] | n/a | 0 |
| `kalypso_block:4/4x4/4` | 1 | BlockSemJoin | 4 | 4 | 16 | 2 | 0.12 | 1.00 | 1.55 [1.47-1.55] | 9.48 [9.41-9.51] | 3.21 [3.21-3.23] | 0 |
| `kalypso_block:4/4x4/4` | 2 | BlockSemFilter | 4 | 2 | 2 | 1 | 0.50 | 0.25 | 7.54 [7.46-7.55] | 9.61 [9.54-9.64] | 1.95 [1.94-1.95] | 2 |
| `kalypso_block:8/8x8/8` | 0 | BlockSemFilter | 8 | 4 | 32 | 16 | 0.50 | 1.00 | 0.23 [0.19-0.24] | 5.46 [5.44-5.48] | n/a | 0 |
| `kalypso_block:8/8x8/8` | 1 | BlockSemJoin | 8 | 2 | 16 | 8 | 0.50 | 1.00 | 2.27 [2.22-2.27] | 9.20 [9.20-9.26] | 3.21 [3.19-3.21] | 0 |
| `kalypso_block:8/8x8/8` | 2 | BlockSemFilter | 8 | 2 | 8 | 2 | 0.25 | 0.50 | 6.63 [6.62-6.67] | 10.13 [10.13-10.19] | 2.58 [2.57-2.59] | 0 |
| `kalypso_block:16/16x16/16` | 0 | BlockSemFilter | 16 | 2 | 32 | 16 | 0.50 | 1.00 | 0.18 [0.15-0.18] | 7.36 [7.34-7.36] | n/a | 0 |
| `kalypso_block:16/16x16/16` | 1 | BlockSemJoin | 16 | 1 | 16 | 0 | 0.00 | 1.00 | 7.39 [7.37-7.40] | 9.83 [9.80-9.84] | 0.00 | 0 |
| `kalypso_block:16/16x16/16` | 2 | BlockSemFilter | 16 | 0 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | 0 |


Operators (first repeat):

| Arm | Query op | Kalypso operator | Rows in | Rows out | Answers began with | Other answers (samples) |
|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 0: sem_filter | SemFilter | 32 | 16 | false 16, true 16 | - |
| `kalypso_per_tuple` | 1: join | SemFilter | 512 | 13 | false 499, true 13 | - |
| `kalypso_per_tuple` | 2: sem_filter | SemFilter | 13 | 6 | false 7, true 6 | - |
| `kalypso_block:2/2x2/2` | 0: sem_filter | BlockSemFilter | 32 | 16 | false 16, true 16 | - |
| `kalypso_block:2/2x2/2` | 1: join | BlockSemJoin | 16 | 10 | - | - |
| `kalypso_block:2/2x2/2` | 2: sem_filter | BlockSemFilter | 10 | 9 | false 1, true 9 | - |
| `kalypso_block:4/4x4/4` | 0: sem_filter | BlockSemFilter | 32 | 16 | false 16, true 16 | - |
| `kalypso_block:4/4x4/4` | 1: join | BlockSemJoin | 16 | 2 | - | - |
| `kalypso_block:4/4x4/4` | 2: sem_filter | BlockSemFilter | 2 | 1 | false 1, true 1 | - |
| `kalypso_block:8/8x8/8` | 0: sem_filter | BlockSemFilter | 32 | 16 | false 16, true 16 | - |
| `kalypso_block:8/8x8/8` | 1: join | BlockSemJoin | 16 | 8 | - | - |
| `kalypso_block:8/8x8/8` | 2: sem_filter | BlockSemFilter | 8 | 2 | false 6, true 2 | - |
| `kalypso_block:16/16x16/16` | 0: sem_filter | BlockSemFilter | 32 | 16 | false 16, true 16 | - |
| `kalypso_block:16/16x16/16` | 1: join | BlockSemJoin | 16 | 0 | - | - |
| `kalypso_block:16/16x16/16` | 2: sem_filter | BlockSemFilter | 0 | 0 | - | - |


Work passed downstream (medians). Gold: 16 rows should pass the filter, 9 final outputs are expected.

| Arm | Rows past first filter | Pairs past join | Final outputs | Comparable |
|---|---|---|---|---|
| `kalypso_per_tuple` | 16 | 13 | 6 | reference |
| `kalypso_block:2/2x2/2` | 16 | 10 | 7 | **no, work drifted** |
| `kalypso_block:4/4x4/4` | 16 | 2 | 1 | **no, work drifted** |
| `kalypso_block:8/8x8/8` | 16 | 8 | 2 | **no, work drifted** |
| `kalypso_block:16/16x16/16` | 16 | 0 | 0 | **no, work drifted** |

Drifted by more than 10% from `kalypso_per_tuple`: `kalypso_block:2/2x2/2`, `kalypso_block:4/4x4/4`, `kalypso_block:8/8x8/8`, `kalypso_block:16/16x16/16`. Their latency and requests reflect different work, not just blocking.
