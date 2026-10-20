# E11

## join_16x8: join alone, 16 x 8, f = 1

Model `Qwen/Qwen2.5-14B-Instruct` on NVIDIA L40S (Linux-4.19.0-gvisor-x86_64-with-glibc2.39); workload `arxiv_join` (join), 16 x 8 = 128 pairs, 16 positive; 3 repeats after 1 warmup; prefix cache reset per trial: yes; git `a6b08c4da` (dirty).
Server: `python3 -m vllm.entrypoints.cli.main serve Qwen/Qwen2.5-14B-Instruct --host 127.0.0.1 --port 8000 --enable-prefix-caching --no-async-scheduling --api-server-count 1 --max-model-len 16384` (started Tue Oct 20 22:00:43 2026).

| Arm | Completion time (s) | Speedup vs kalypso_per_tuple | LLM calls [retries + fallbacks] | Agrees w/ kalypso_per_tuple | F1 vs labels | Peak KV % |
|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 8.05 [8.03-8.20] | 1.00x | 128 [0] | 100% | 0.69 | 29.2 |
| `kalypso_block:trummer` | 2.16 [2.15-2.16] | 3.73x | 1 [0] | 93% | 0.20 | 12.9 |
| `kalypso_block:8x4` | 1.96 [1.95-1.98] | 4.10x | 4 [0] | 93% | 0.30 | 13.0 |


Diagnostics:

| Arm | Prompt tokens | Gen tokens | Prefix hit rate | Peak RSS (MiB) | Items/s | First output (s) | Truncated | Accuracy | Precision | Recall | Self-agreement |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 89232 | 1024 | 0.42 | 11564 [11563-11564] | 15.89 [15.61-15.94] | 2.56 [2.55-2.60] | 111 | 0.93 | 0.77 | 0.62 | 100% |
| `kalypso_block:trummer` | 6702 | 26 | 0.00 | 11564 [11563-11564] | 59.26 [59.18-59.40] | 2.16 [2.15-2.16] | 0 | 0.88 | 0.50 | 0.12 | 100% |
| `kalypso_block:8x4` | 13570 | 28 | 0.33 | 11564 [11563-11564] | 65.24 [64.65-65.71] | 1.73 [1.72-1.75] | 0 | 0.89 | 0.75 | 0.19 | 100% |


Stages (medians over repeats; times from query start):

| Arm | Stage | Operators | Block | Tasks | Rows in | Rows out | Out/in | Occupancy | Start (s) | End (s) | Overlap (s) | Fallbacks |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 0 | SemFilter | 1 | 128 | 128 | 13 | 0.10 | n/a | 0.01 [0.01-0.01] | 8.05 [8.03-8.20] | n/a | 0 |
| `kalypso_block:trummer` | 0 | BlockSemJoin | 16 | 1 | 16 | 4 | 0.25 | 1.00 | 0.04 [0.04-0.04] | 2.16 [2.15-2.16] | n/a | 0 |
| `kalypso_block:8x4` | 0 | BlockSemJoin | 8 | 2 | 16 | 4 | 0.25 | 1.00 | 0.04 [0.04-0.04] | 1.96 [1.95-1.98] | n/a | 0 |



Cost-model block sizes (Trummer, Level 3):

| Arm | Chosen k x j | Closed form | Calls | Tokens per call | Inputs |
|---|---|---|---|---|---|
| `kalypso_block:trummer` | 16 x 8 | 28.6 x 27.4 | 1 | 6820 | p 87, s1 272, s2 283, s3 8, sigma 0.125 |


## fjf_s05_f1: pipeline, s = 0.5, f = 1

Model `Qwen/Qwen2.5-14B-Instruct` on NVIDIA L40S (Linux-4.19.0-gvisor-x86_64-with-glibc2.39); workload `arxiv_fjf` (pipeline), 32 x 32 = 1024 pairs, 9 positive; 3 repeats after 1 warmup; prefix cache reset per trial: yes; git `a6b08c4da` (dirty).
Server: `python3 -m vllm.entrypoints.cli.main serve Qwen/Qwen2.5-14B-Instruct --host 127.0.0.1 --port 8000 --enable-prefix-caching --no-async-scheduling --api-server-count 1 --max-model-len 16384` (started Tue Oct 20 22:00:43 2026).

| Arm | Completion time (s) | Speedup vs kalypso_per_tuple | LLM calls [retries + fallbacks] | Agrees w/ kalypso_per_tuple | F1 vs labels | Peak KV % |
|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 31.45 [31.17-31.52] | 1.00x | 557 [0] | 100% | 0.80 | 32.6 [32.6-32.6] |
| `kalypso_block:4/trummer/4` | 5.28 [5.25-5.28] | 5.96x | 10 [0] | 99% | n/a | 23.8 [23.6-23.8] |
| `kalypso_block:4/4x4/4` | 9.43 [9.37-10.27] | 3.34x | 42 [2] | 99% | 0.00 | 15.9 |


Diagnostics:

| Arm | Prompt tokens | Gen tokens | Prefix hit rate | Peak RSS (MiB) | Items/s | First output (s) | Truncated | Accuracy | Precision | Recall | Self-agreement |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 369422 | 4450 | 0.48 [0.48-0.48] | 11571 [11571-11571] | 32.56 [32.49-32.85] | 12.50 [12.28-12.55] | 461 [460-461] | 1.00 | 1.00 | 0.67 | 100% |
| `kalypso_block:4/trummer/4` | 26221 | 132 | 0.18 | 11571 [11571-11571] | 194.05 [193.98-195.20] | n/a | 0 | 0.99 | n/a | 0.00 | 100% |
| `kalypso_block:4/4x4/4` | 81554 [81554-82447] | 204 [204-224] | 0.39 [0.39-0.39] | 11571 [11571-11571] | 108.61 [99.74-109.30] | 7.65 [7.63-7.69] | 0 | 0.99 | 0.00 | 0.00 | 100% |


Stages (medians over repeats; times from query start):

| Arm | Stage | Operators | Block | Tasks | Rows in | Rows out | Out/in | Occupancy | Start (s) | End (s) | Overlap (s) | Fallbacks |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 0 | SemFilter | 1 | 32 | 32 | 512 | 16.00 | n/a | 0.19 [0.19-0.19] | 20.83 [20.36-20.91] | n/a | 0 |
| `kalypso_per_tuple` | 1 | SemFilter+SemFilter | 1 | 512 | 512 | 6 | 0.01 | n/a | 0.89 [0.88-0.89] | 31.44 [31.16-31.51] | 19.95 [19.47-20.02] | 0 |
| `kalypso_block:4/trummer/4` | 0 | BlockSemFilter | 4 | 8 | 32 | 16 | 0.50 | 1.00 | 0.18 [0.18-0.18] | 2.72 [2.71-2.72] | n/a | 0 |
| `kalypso_block:4/trummer/4` | 1 | BlockSemJoin | 31 | 1 | 16 | 0 | 0.00 | 0.52 | 2.75 [2.74-2.76] | 5.27 [5.24-5.28] | 0.00 | 0 |
| `kalypso_block:4/trummer/4` | 2 | BlockSemFilter | 4 | 0 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | 0 |
| `kalypso_block:4/4x4/4` | 0 | BlockSemFilter | 4 | 8 | 32 | 16 | 0.50 | 1.00 | 0.29 [0.29-0.29] | 4.64 [4.62-4.65] | n/a | 0 |
| `kalypso_block:4/4x4/4` | 1 | BlockSemJoin | 4 | 4 | 16 | 2 [2-4] | 0.12 [0.12-0.25] | 1.00 | 1.46 [1.46-1.47] | 9.30 [9.24-9.60] | 3.18 [3.16-3.18] | 0 |
| `kalypso_block:4/4x4/4` | 2 | BlockSemFilter | 4 | 2 | 2 [2-4] | 1 | 0.50 [0.25-0.50] | 0.25 [0.25-0.50] | 7.33 [7.32-7.37] | 9.43 [9.37-10.26] | 1.93 [1.91-2.29] | 2 [1-2] |


Operators (first repeat):

| Arm | Query op | Kalypso operator | Rows in | Rows out | Answers began with | Other answers (samples) |
|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 0: sem_filter | SemFilter | 32 | 16 | false 16, true 16 | - |
| `kalypso_per_tuple` | 1: join | SemFilter | 512 | 13 | false 499, true 13 | - |
| `kalypso_per_tuple` | 2: sem_filter | SemFilter | 13 | 6 | false 7, true 6 | - |
| `kalypso_block:4/trummer/4` | 0: sem_filter | BlockSemFilter | 32 | 16 | false 16, true 16 | - |
| `kalypso_block:4/trummer/4` | 1: join | BlockSemJoin | 16 | 0 | - | - |
| `kalypso_block:4/trummer/4` | 2: sem_filter | BlockSemFilter | 0 | 0 | - | - |
| `kalypso_block:4/4x4/4` | 0: sem_filter | BlockSemFilter | 32 | 16 | false 16, true 16 | - |
| `kalypso_block:4/4x4/4` | 1: join | BlockSemJoin | 16 | 2 | - | - |
| `kalypso_block:4/4x4/4` | 2: sem_filter | BlockSemFilter | 2 | 1 | false 1, true 1 | - |


Cost-model block sizes (Trummer, Level 3):

| Arm | Chosen k x j | Closed form | Calls | Tokens per call | Inputs |
|---|---|---|---|---|---|
| `kalypso_block:4/trummer/4` | 31 x 30 | 31.4 x 30.2 | 4 | 16226 | p 87, s1 256, s2 266, s3 8, sigma 0.031 |


Work passed downstream (medians). Gold: 16 rows should pass the filter, 9 final outputs are expected.

| Arm | Rows past first filter | Pairs past join | Final outputs | Comparable |
|---|---|---|---|---|
| `kalypso_per_tuple` | 16 | 13 | 6 | reference |
| `kalypso_block:4/trummer/4` | 16 | 0 | 0 | **no, work drifted** |
| `kalypso_block:4/4x4/4` | 16 | 2 | 1 | **no, work drifted** |

Drifted by more than 10% from `kalypso_per_tuple`: `kalypso_block:4/trummer/4`, `kalypso_block:4/4x4/4`. Their latency and requests reflect different work, not just blocking.
