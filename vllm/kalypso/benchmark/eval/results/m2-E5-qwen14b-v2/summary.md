# E5

## fjf_s05_f1: s = 0.5, f = 1

Model `Qwen/Qwen2.5-14B-Instruct` on NVIDIA L40S (Linux-4.19.0-gvisor-x86_64-with-glibc2.39); workload `arxiv_fjf` (pipeline), 32 x 32 = 1024 pairs, 9 positive; 3 repeats after 1 warmup; prefix cache reset per trial: yes; git `6368c4b90`.
Server: `python3 -m vllm.entrypoints.cli.main serve Qwen/Qwen2.5-14B-Instruct --host 127.0.0.1 --port 8000 --enable-prefix-caching --no-async-scheduling --api-server-count 1 --max-model-len 16384` (started Tue Oct 20 20:53:05 2026).

| Arm | Completion time (s) | Speedup vs kalypso_per_tuple | LLM calls [retries + fallbacks] | Agrees w/ kalypso_per_tuple | F1 vs labels | Peak KV % |
|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 31.37 [30.92-31.44] | 1.00x | 557 [0] | 100% | 0.80 | 32.5 [32.2-32.6] |
| `kalypso_block:4/2x2/4` | 16.39 [16.21-16.49] | 1.91x | 142 [6] | 99% | 0.14 [0.00-0.14] | 16.0 |
| `kalypso_block:4/2x2/16` | 16.77 [16.35-17.05] | 1.87x | 141 [4] | 99% | 0.00 [0.00-0.14] | 16.0 [13.0-16.6] |
| `kalypso_block:4/4x4/4` | 10.52 [10.21-10.86] | 2.98x | 42 [2] | 99% | 0.00 [0.00-0.17] | 15.9 [15.9-17.3] |
| `kalypso_block:4/4x4/16` | 10.33 [9.79-10.54] | 3.04x | 42 [2] | 99% | 0.00 [0.00-0.17] | 15.9 [15.9-17.3] |
| `kalypso_block:4/4x32/4` | 12.33 [11.62-16.99] | 2.54x | 14 [2] | 99% | 0.00 | 37.2 [37.2-56.0] |
| `kalypso_block:4/4x32/16` | 17.29 [11.75-20.36] | 1.81x | 14 [1] | 99% | 0.00 | 37.3 [37.2-56.0] |
| `kalypso_block:16/2x2/4` | 21.39 [20.58-21.67] | 1.47x | 135 [4] | 99% | 0.00 | 13.0 |
| `kalypso_block:16/2x2/16` | 20.86 [20.75-21.75] | 1.50x | 135 [4] | 99% | 0.00 | 13.0 [12.7-13.0] |
| `kalypso_block:16/4x4/4` | 15.77 [15.25-16.03] | 1.99x | 36 [1] | 99% | 0.00 | 16.7 |
| `kalypso_block:16/4x4/16` | 15.83 [15.17-15.89] | 1.98x | 36 [1] | 99% | 0.00 | 16.3 [15.8-16.7] |
| `kalypso_block:16/4x32/4` | 17.59 [17.51-18.55] | 1.78x | 9 [2] | 99% | 0.00 | 28.0 |
| `kalypso_block:16/4x32/16` | 17.49 [17.47-18.86] | 1.79x | 8 [1] | 99% | 0.00 | 28.0 |


Diagnostics:

| Arm | Prompt tokens | Gen tokens | Prefix hit rate | Peak RSS (MiB) | Items/s | First output (s) | Truncated | Accuracy | Precision | Recall | Self-agreement |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 369422 | 4450 [4448-4450] | 0.48 [0.48-0.48] | 11539 [11538-11539] | 32.65 [32.57-33.12] | 10.57 [10.53-10.74] | 474 [474-475] | 1.00 | 1.00 | 0.67 | 100% |
| `kalypso_block:4/2x2/4` | 161267 [161237-161267] | 420 [420-424] | 0.44 [0.44-0.44] | 11539 [11538-11539] | 62.50 [62.09-63.15] | 13.75 [13.73-13.82] | 0 | 0.99 | 0.20 [0.00-0.20] | 0.11 [0.00-0.11] | 100% |
| `kalypso_block:4/2x2/16` | 161237 [161237-161267] | 424 [420-424] | 0.44 [0.44-0.44] | 11539 [11538-11539] | 61.08 [60.05-62.64] | 13.62 [13.36-14.29] | 0 | 0.99 | 0.00 [0.00-0.20] | 0.00 [0.00-0.11] | 100% |
| `kalypso_block:4/4x4/4` | 82447 [81554-82754] | 222 [204-224] | 0.39 [0.38-0.39] | 11539 [11538-11539] | 97.34 [94.31-100.32] | 8.29 [7.81-8.30] | 0 | 0.99 | 0.00 [0.00-0.33] | 0.00 [0.00-0.11] | 100% |
| `kalypso_block:4/4x4/16` | 82447 [81554-82754] | 222 [204-224] | 0.39 [0.38-0.39] | 11539 [11538-11539] | 99.15 [97.20-104.64] | 7.86 [7.82-8.01] | 0 | 0.99 | 0.00 [0.00-0.33] | 0.00 [0.00-0.11] | 100% |
| `kalypso_block:4/4x32/4` | 48830 [48830-55926] | 205 [205-421] | 0.01 [0.01-0.01] | 11539 [11539-11539] | 83.08 [60.28-88.14] | 11.65 [11.61-12.32] | 0 | 0.99 [0.99-0.99] | 0.00 | 0.00 | 100% |
| `kalypso_block:4/4x32/16` | 52738 [48366-57448] | 348 [203-482] | 0.01 [0.01-0.01] | 11539 [11539-11539] | 59.23 [50.30-87.12] | 12.03 [11.75-12.72] | 0 | 0.99 [0.98-0.99] | 0.00 | 0.00 | 99% |
| `kalypso_block:16/2x2/4` | 160763 | 438 | 0.44 | 11539 [11539-11539] | 47.88 [47.26-49.77] | 10.38 [10.15-10.74] | 0 | 0.99 | 0.00 | 0.00 | 100% |
| `kalypso_block:16/2x2/16` | 160763 | 438 | 0.44 | 11539 [11539-11539] | 49.08 [47.08-49.34] | 10.18 [10.16-10.89] | 0 | 0.99 | 0.00 | 0.00 | 100% |
| `kalypso_block:16/4x4/4` | 81973 | 238 | 0.39 | 11539 [11539-11539] | 64.93 [63.89-67.17] | 8.97 [8.69-9.46] | 0 | 0.99 | 0.00 | 0.00 | 100% |
| `kalypso_block:16/4x4/16` | 81973 | 238 | 0.39 | 11539 [11539-11539] | 64.69 [64.44-67.51] | 9.03 [8.72-9.14] | 0 | 0.99 | 0.00 | 0.00 | 100% |
| `kalypso_block:16/4x32/4` | 50998 | 288 | 0.00 | 11539 [11539-11539] | 58.21 [55.19-58.49] | 11.36 [11.31-12.11] | 0 | 0.99 | 0.00 | 0.00 | 100% |
| `kalypso_block:16/4x32/16` | 50952 | 288 | 0.00 | 11539 [11539-11539] | 58.55 [54.28-58.62] | 12.14 [12.12-13.18] | 0 | 0.99 | 0.00 | 0.00 | 100% |


Stages (medians over repeats; times from query start):

| Arm | Stage | Operators | Block | Tasks | Rows in | Rows out | Out/in | Occupancy | Start (s) | End (s) | Overlap (s) | Fallbacks |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 0 | SemFilter | 1 | 32 | 32 | 512 | 16.00 | n/a | 0.25 [0.24-0.27] | 20.79 [20.24-23.13] | n/a | 0 |
| `kalypso_per_tuple` | 1 | SemFilter+SemFilter | 1 | 512 | 512 | 6 | 0.01 | n/a | 0.95 [0.94-1.00] | 31.35 [30.90-31.42] | 19.85 [19.24-22.18] | 0 |
| `kalypso_block:4/2x2/4` | 0 | BlockSemFilter | 4 | 8 | 32 | 16 | 0.50 | 1.00 | 0.61 [0.60-0.66] | 5.58 [5.56-5.64] | n/a | 0 |
| `kalypso_block:4/2x2/4` | 1 | BlockSemJoin | 2 | 8 | 16 | 6 | 0.38 | 1.00 | 1.78 [1.78-1.86] | 16.02 [15.93-16.10] | 3.78 [3.78-3.80] | 0 |
| `kalypso_block:4/2x2/4` | 2 | BlockSemFilter | 4 | 6 [5-6] | 6 | 5 [3-5] | 0.83 [0.50-0.83] | 0.25 [0.25-0.30] | 13.08 [13.08-13.16] | 16.38 [16.21-16.49] | 2.94 [2.84-2.94] | 6 [4-6] |
| `kalypso_block:4/2x2/16` | 0 | BlockSemFilter | 4 | 8 | 32 | 16 | 0.50 | 1.00 | 0.61 [0.57-0.66] | 5.49 [5.48-8.00] | n/a | 0 |
| `kalypso_block:4/2x2/16` | 1 | BlockSemJoin | 2 | 8 | 16 | 6 | 0.38 | 1.00 | 1.78 [1.72-1.90] | 16.47 [15.86-16.59] | 3.76 [3.70-6.10] | 0 |
| `kalypso_block:4/2x2/16` | 2 | BlockSemFilter | 16 | 5 [5-6] | 6 | 3 [3-5] | 0.50 [0.50-0.83] | 0.07 [0.06-0.07] | 12.97 [12.68-13.62] | 16.76 [16.34-17.04] | 2.89 [2.85-3.91] | 4 [4-6] |
| `kalypso_block:4/4x4/4` | 0 | BlockSemFilter | 4 | 8 | 32 | 16 | 0.50 | 1.00 | 0.50 [0.36-0.77] | 5.05 [4.78-5.31] | n/a | 0 |
| `kalypso_block:4/4x4/4` | 1 | BlockSemJoin | 4 | 4 | 16 | 4 [2-4] | 0.25 [0.12-0.25] | 1.00 | 1.77 [1.58-2.04] | 10.07 [9.82-10.38] | 3.28 [3.21-3.28] | 0 |
| `kalypso_block:4/4x4/4` | 2 | BlockSemFilter | 4 | 2 [2-3] | 4 [2-4] | 1 [1-3] | 0.50 [0.25-0.75] | 0.33 [0.25-0.50] | 7.97 [7.49-8.10] | 10.51 [10.20-10.85] | 2.28 [2.10-2.33] | 2 [1-2] |
| `kalypso_block:4/4x4/16` | 0 | BlockSemFilter | 4 | 8 | 32 | 16 | 0.50 | 1.00 | 0.37 [0.37-0.38] | 4.86 [4.76-4.90] | n/a | 0 |
| `kalypso_block:4/4x4/16` | 1 | BlockSemJoin | 4 | 4 | 16 | 4 [2-4] | 0.25 [0.12-0.25] | 1.00 | 1.58 [1.58-1.63] | 9.84 [9.66-9.87] | 3.23 [3.18-3.32] | 0 |
| `kalypso_block:4/4x4/16` | 2 | BlockSemFilter | 16 | 2 [2-3] | 4 [2-4] | 1 [1-3] | 0.50 [0.25-0.75] | 0.08 [0.06-0.12] | 7.64 [7.54-7.69] | 10.32 [9.78-10.53] | 2.23 [1.96-2.30] | 2 [1-2] |
| `kalypso_block:4/4x32/4` | 0 | BlockSemFilter | 4 | 8 | 32 | 16 | 0.50 | 1.00 | 0.29 [0.29-0.29] | 5.84 [4.47-6.11] | n/a | 0 |
| `kalypso_block:4/4x32/4` | 1 | BlockSemJoin | 4 | 4 | 16 | 2 [2-14] | 0.12 [0.12-0.88] | 1.00 | 1.82 [1.61-2.00] | 12.15 [11.44-15.61] | 4.02 [2.87-4.11] | 0 |
| `kalypso_block:4/4x32/4` | 2 | BlockSemFilter | 4 | 2 [2-5] | 2 [2-14] | 1 [1-5] | 0.50 [0.36-0.50] | 0.25 [0.25-0.70] | 10.50 [9.69-11.09] | 12.32 [11.61-16.98] | 1.76 [1.65-4.52] | 2 |
| `kalypso_block:4/4x32/16` | 0 | BlockSemFilter | 4 | 8 | 32 | 16 | 0.50 | 1.00 | 0.32 [0.26-0.33] | 5.92 [4.45-6.00] | n/a | 0 |
| `kalypso_block:4/4x32/16` | 1 | BlockSemJoin | 4 | 4 | 16 | 9 [1-17] | 0.56 [0.06-1.06] | 1.00 | 1.61 [1.60-1.83] | 15.13 [11.58-17.42] | 4.09 [2.86-4.39] | 0 |
| `kalypso_block:4/4x32/16` | 2 | BlockSemFilter | 16 | 2 [1-3] | 9 [1-17] | 1 [1-7] | 0.41 [0.11-1.00] | 0.28 [0.06-0.35] | 11.58 [11.50-11.83] | 17.28 [11.75-20.35] | 3.30 [0.00-5.92] | 1 |
| `kalypso_block:16/2x2/4` | 0 | BlockSemFilter | 16 | 2 | 32 | 16 | 0.50 | 1.00 | 0.67 [0.62-0.75] | 11.65 [11.08-11.99] | n/a | 0 |
| `kalypso_block:16/2x2/4` | 1 | BlockSemJoin | 2 | 8 | 16 | 6 | 0.38 | 1.00 | 4.40 [4.24-4.82] | 20.90 [20.09-21.19] | 7.17 [6.83-7.25] | 0 |
| `kalypso_block:16/2x2/4` | 2 | BlockSemFilter | 4 | 5 | 6 | 3 | 0.50 | 0.30 | 9.72 [9.51-10.12] | 21.38 [20.57-21.66] | 11.07 [10.58-11.18] | 4 |
| `kalypso_block:16/2x2/16` | 0 | BlockSemFilter | 16 | 2 | 32 | 16 | 0.50 | 1.00 | 0.62 [0.60-0.72] | 11.38 [11.35-12.27] | n/a | 0 |
| `kalypso_block:16/2x2/16` | 1 | BlockSemJoin | 2 | 8 | 16 | 6 | 0.38 | 1.00 | 4.31 [4.30-4.65] | 20.36 [20.26-21.29] | 7.07 [7.05-7.62] | 0 |
| `kalypso_block:16/2x2/16` | 2 | BlockSemFilter | 16 | 5 | 6 | 3 | 0.50 | 0.07 | 9.58 [9.51-10.25] | 20.85 [20.75-21.74] | 10.78 [10.75-11.04] | 4 |
| `kalypso_block:16/4x4/4` | 0 | BlockSemFilter | 16 | 2 | 32 | 16 | 0.50 | 1.00 | 0.38 [0.36-0.49] | 10.70 [10.42-11.13] | n/a | 0 |
| `kalypso_block:16/4x4/4` | 1 | BlockSemJoin | 4 | 4 | 16 | 4 | 0.25 | 1.00 | 4.18 [4.07-4.28] | 15.06 [14.56-15.34] | 6.52 [6.34-6.85] | 0 |
| `kalypso_block:16/4x4/4` | 2 | BlockSemFilter | 4 | 2 | 4 | 1 | 0.25 | 0.50 | 8.78 [8.51-9.28] | 15.76 [15.23-16.02] | 6.06 [6.05-6.28] | 1 |
| `kalypso_block:16/4x4/16` | 0 | BlockSemFilter | 16 | 2 | 32 | 16 | 0.50 | 1.00 | 0.40 [0.37-0.53] | 10.70 [10.40-10.85] | n/a | 0 |
| `kalypso_block:16/4x4/16` | 1 | BlockSemJoin | 4 | 4 | 16 | 4 | 0.25 | 1.00 | 4.07 [4.06-4.26] | 15.16 [14.49-15.20] | 6.59 [6.33-6.64] | 0 |
| `kalypso_block:16/4x4/16` | 2 | BlockSemFilter | 16 | 2 | 4 | 1 | 0.25 | 0.12 | 8.85 [8.54-8.95] | 15.83 [15.16-15.88] | 6.24 [5.95-6.31] | 1 |
| `kalypso_block:16/4x32/4` | 0 | BlockSemFilter | 16 | 2 | 32 | 16 | 0.50 | 1.00 | 0.26 [0.25-0.27] | 11.23 [11.18-11.85] | n/a | 0 |
| `kalypso_block:16/4x32/4` | 1 | BlockSemJoin | 4 | 4 | 16 | 6 | 0.38 | 1.00 | 3.93 [3.90-4.23] | 17.59 [17.50-18.55] | 7.33 [7.25-7.62] | 0 |
| `kalypso_block:16/4x32/4` | 2 | BlockSemFilter | 4 | 3 | 6 | 3 | 0.50 | 0.50 | 9.06 [9.05-9.50] | 15.88 [15.80-16.78] | 8.53 [8.45-9.05] | 2 |
| `kalypso_block:16/4x32/16` | 0 | BlockSemFilter | 16 | 2 | 32 | 16 | 0.50 | 1.00 | 0.27 [0.25-0.28] | 11.16 [11.15-12.20] | n/a | 0 |
| `kalypso_block:16/4x32/16` | 1 | BlockSemJoin | 4 | 4 | 16 | 6 | 0.38 | 1.00 | 3.96 [3.91-4.28] | 17.49 [17.46-18.86] | 7.25 [7.19-7.92] | 0 |
| `kalypso_block:16/4x32/16` | 2 | BlockSemFilter | 16 | 2 | 6 | 2 | 0.33 | 0.19 | 9.05 [9.04-9.93] | 15.79 [15.77-16.91] | 8.44 [8.42-8.93] | 1 |


Operators (first repeat):

| Arm | Query op | Kalypso operator | Rows in | Rows out | Answers began with | Other answers (samples) |
|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 0: sem_filter | SemFilter | 32 | 16 | false 16, true 16 | - |
| `kalypso_per_tuple` | 1: join | SemFilter | 512 | 13 | false 499, true 13 | - |
| `kalypso_per_tuple` | 2: sem_filter | SemFilter | 13 | 6 | false 7, true 6 | - |
| `kalypso_block:4/2x2/4` | 0: sem_filter | BlockSemFilter | 32 | 16 | false 16, true 16 | - |
| `kalypso_block:4/2x2/4` | 1: join | BlockSemJoin | 16 | 6 | - | - |
| `kalypso_block:4/2x2/4` | 2: sem_filter | BlockSemFilter | 6 | 3 | false 3, true 3 | - |
| `kalypso_block:4/2x2/16` | 0: sem_filter | BlockSemFilter | 32 | 16 | false 16, true 16 | - |
| `kalypso_block:4/2x2/16` | 1: join | BlockSemJoin | 16 | 6 | - | - |
| `kalypso_block:4/2x2/16` | 2: sem_filter | BlockSemFilter | 6 | 3 | false 3, true 3 | - |
| `kalypso_block:4/4x4/4` | 0: sem_filter | BlockSemFilter | 32 | 16 | false 16, true 16 | - |
| `kalypso_block:4/4x4/4` | 1: join | BlockSemJoin | 16 | 4 | - | - |
| `kalypso_block:4/4x4/4` | 2: sem_filter | BlockSemFilter | 4 | 3 | false 1, true 3 | - |
| `kalypso_block:4/4x4/16` | 0: sem_filter | BlockSemFilter | 32 | 16 | false 16, true 16 | - |
| `kalypso_block:4/4x4/16` | 1: join | BlockSemJoin | 16 | 4 | - | - |
| `kalypso_block:4/4x4/16` | 2: sem_filter | BlockSemFilter | 4 | 3 | false 1, true 3 | - |
| `kalypso_block:4/4x32/4` | 0: sem_filter | BlockSemFilter | 32 | 16 | false 16, true 16 | - |
| `kalypso_block:4/4x32/4` | 1: join | BlockSemJoin | 16 | 14 | - | - |
| `kalypso_block:4/4x32/4` | 2: sem_filter | BlockSemFilter | 14 | 5 | false 9, true 5 | - |
| `kalypso_block:4/4x32/16` | 0: sem_filter | BlockSemFilter | 32 | 16 | false 16, true 16 | - |
| `kalypso_block:4/4x32/16` | 1: join | BlockSemJoin | 16 | 17 | - | - |
| `kalypso_block:4/4x32/16` | 2: sem_filter | BlockSemFilter | 17 | 7 | false 10, true 7 | - |
| `kalypso_block:16/2x2/4` | 0: sem_filter | BlockSemFilter | 32 | 16 | false 16, true 16 | - |
| `kalypso_block:16/2x2/4` | 1: join | BlockSemJoin | 16 | 6 | - | - |
| `kalypso_block:16/2x2/4` | 2: sem_filter | BlockSemFilter | 6 | 3 | false 3, true 3 | - |
| `kalypso_block:16/2x2/16` | 0: sem_filter | BlockSemFilter | 32 | 16 | false 16, true 16 | - |
| `kalypso_block:16/2x2/16` | 1: join | BlockSemJoin | 16 | 6 | - | - |
| `kalypso_block:16/2x2/16` | 2: sem_filter | BlockSemFilter | 6 | 3 | false 3, true 3 | - |
| `kalypso_block:16/4x4/4` | 0: sem_filter | BlockSemFilter | 32 | 16 | false 16, true 16 | - |
| `kalypso_block:16/4x4/4` | 1: join | BlockSemJoin | 16 | 4 | - | - |
| `kalypso_block:16/4x4/4` | 2: sem_filter | BlockSemFilter | 4 | 1 | false 3, true 1 | - |
| `kalypso_block:16/4x4/16` | 0: sem_filter | BlockSemFilter | 32 | 16 | false 16, true 16 | - |
| `kalypso_block:16/4x4/16` | 1: join | BlockSemJoin | 16 | 4 | - | - |
| `kalypso_block:16/4x4/16` | 2: sem_filter | BlockSemFilter | 4 | 1 | false 3, true 1 | - |
| `kalypso_block:16/4x32/4` | 0: sem_filter | BlockSemFilter | 32 | 16 | false 16, true 16 | - |
| `kalypso_block:16/4x32/4` | 1: join | BlockSemJoin | 16 | 6 | - | - |
| `kalypso_block:16/4x32/4` | 2: sem_filter | BlockSemFilter | 6 | 3 | false 3, true 3 | - |
| `kalypso_block:16/4x32/16` | 0: sem_filter | BlockSemFilter | 32 | 16 | false 16, true 16 | - |
| `kalypso_block:16/4x32/16` | 1: join | BlockSemJoin | 16 | 6 | - | - |
| `kalypso_block:16/4x32/16` | 2: sem_filter | BlockSemFilter | 6 | 2 | false 4, true 2 | - |


Work passed downstream (medians). Gold: 16 rows should pass the filter, 9 final outputs are expected.

| Arm | Rows past first filter | Pairs past join | Final outputs | Comparable |
|---|---|---|---|---|
| `kalypso_per_tuple` | 16 | 13 | 6 | reference |
| `kalypso_block:4/2x2/4` | 16 | 6 | 5 | **no, work drifted** |
| `kalypso_block:4/2x2/16` | 16 | 6 | 3 | **no, work drifted** |
| `kalypso_block:4/4x4/4` | 16 | 4 | 1 | **no, work drifted** |
| `kalypso_block:4/4x4/16` | 16 | 4 | 1 | **no, work drifted** |
| `kalypso_block:4/4x32/4` | 16 | 2 | 1 | **no, work drifted** |
| `kalypso_block:4/4x32/16` | 16 | 9 | 1 | **no, work drifted** |
| `kalypso_block:16/2x2/4` | 16 | 6 | 3 | **no, work drifted** |
| `kalypso_block:16/2x2/16` | 16 | 6 | 3 | **no, work drifted** |
| `kalypso_block:16/4x4/4` | 16 | 4 | 1 | **no, work drifted** |
| `kalypso_block:16/4x4/16` | 16 | 4 | 1 | **no, work drifted** |
| `kalypso_block:16/4x32/4` | 16 | 6 | 3 | **no, work drifted** |
| `kalypso_block:16/4x32/16` | 16 | 6 | 2 | **no, work drifted** |

Drifted by more than 10% from `kalypso_per_tuple`: `kalypso_block:4/2x2/4`, `kalypso_block:4/2x2/16`, `kalypso_block:4/4x4/4`, `kalypso_block:4/4x4/16`, `kalypso_block:4/4x32/4`, `kalypso_block:4/4x32/16`, `kalypso_block:16/2x2/4`, `kalypso_block:16/2x2/16`, `kalypso_block:16/4x4/4`, `kalypso_block:16/4x4/16`, `kalypso_block:16/4x32/4`, `kalypso_block:16/4x32/16`. Their latency and requests reflect different work, not just blocking.
