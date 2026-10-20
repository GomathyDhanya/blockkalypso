# E8

## fjf_s05_f1: s = 0.5, f = 1

Model `Qwen/Qwen2.5-14B-Instruct` on NVIDIA L40S (Linux-4.19.0-gvisor-x86_64-with-glibc2.39); workload `arxiv_fjf` (pipeline), 32 x 32 = 1024 pairs, 9 positive; 3 repeats after 1 warmup; prefix cache reset per trial: yes; git `6368c4b90` (dirty).
Server: `python3 -m vllm.entrypoints.cli.main serve Qwen/Qwen2.5-14B-Instruct --host 127.0.0.1 --port 8000 --enable-prefix-caching --no-async-scheduling --api-server-count 1 --max-model-len 16384` (started Tue Oct 20 21:16:10 2026).

| Arm | Completion time (s) | Speedup vs kalypso_per_tuple | LLM calls [retries + fallbacks] | Agrees w/ kalypso_per_tuple | F1 vs labels | Peak KV % |
|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 30.99 [30.47-31.26] | 1.00x | 557 [0] | 100% | 0.80 | 32.1 [32.0-32.9] |
| `kalypso_per_tuple+materialized` | 41.68 [40.31-41.84] | 0.74x | 557 [0] | 100% | 0.80 | 32.0 |
| `kalypso_block:4/4x4/4` | 9.40 [9.38-10.44] | 3.30x | 42 [2] | 99% | 0.00 | 15.9 |
| `kalypso_block:4/4x4/4+materialized` | 16.49 [16.42-16.55] | 1.88x | 41 [0] | 99% | n/a | 4.8 |
| `kalypso_block:4/2x2/4` | 16.12 [15.82-16.27] | 1.92x | 141 [4] | 99% | 0.00 [0.00-0.14] | 16.0 |
| `kalypso_block:4/2x2/4+materialized` | 28.44 [28.26-28.89] | 1.09x | 138 [0] | 99% | 0.00 | 6.1 |


Diagnostics:

| Arm | Prompt tokens | Gen tokens | Prefix hit rate | Peak RSS (MiB) | Items/s | First output (s) | Truncated | Accuracy | Precision | Recall | Self-agreement |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 369422 | 4450 | 0.48 | 11544 [11544-11544] | 33.04 [32.76-33.61] | 12.39 [8.80-14.67] | 474 [474-475] | 1.00 | 1.00 | 0.67 | 100% |
| `kalypso_per_tuple+materialized` | 369422 | 4450 | 0.48 | 11544 [11544-11544] | 24.57 [24.48-25.40] | 22.54 [22.14-22.75] | 472 | 1.00 | 1.00 | 0.67 | 100% |
| `kalypso_block:4/4x4/4` | 81554 [81554-82447] | 204 [204-224] | 0.39 [0.39-0.39] | 11544 [11544-11544] | 108.92 [98.08-109.22] | 7.67 [7.65-7.81] | 0 | 0.99 | 0.00 | 0.00 | 100% |
| `kalypso_block:4/4x4/4+materialized` | 82401 | 226 | 0.39 | 11544 [11544-11544] | 62.11 [61.87-62.35] | n/a | 0 | 0.99 | n/a | 0.00 | 100% |
| `kalypso_block:4/2x2/4` | 161237 [161237-161267] | 424 [420-424] | 0.44 [0.44-0.44] | 11544 [11544-11544] | 63.50 [62.92-64.72] | 13.42 [13.37-13.54] | 0 | 0.99 | 0.00 [0.00-0.20] | 0.00 [0.00-0.11] | 100% |
| `kalypso_block:4/2x2/4+materialized` | 161135 | 432 | 0.44 | 11544 [11544-11544] | 36.01 [35.44-36.23] | 28.10 [27.93-28.56] | 0 | 0.99 | 0.00 | 0.00 | 100% |


Stages (medians over repeats; times from query start):

| Arm | Stage | Operators | Block | Tasks | Rows in | Rows out | Out/in | Occupancy | Start (s) | End (s) | Overlap (s) | Fallbacks |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 0 | SemFilter | 1 | 32 | 32 | 512 | 16.00 | n/a | 0.25 [0.24-0.25] | 8.39 [8.38-8.50] | n/a | 0 |
| `kalypso_per_tuple` | 1 | SemFilter+SemFilter | 1 | 512 | 512 | 6 | 0.01 | n/a | 0.95 [0.94-0.96] | 30.98 [30.46-31.25] | 7.44 [7.43-7.56] | 0 |
| `kalypso_per_tuple+materialized` | 0 | SemFilter | 1 | 32 | 32 | 512 | 16.00 | n/a | 0.25 [0.24-0.25] | 11.74 [11.70-11.76] | n/a | 0 |
| `kalypso_per_tuple+materialized` | 1 | SemFilter+SemFilter | 1 | 512 | 512 | 6 | 0.01 | n/a | 11.76 [11.72-11.78] | 41.66 [40.30-41.82] | 0.00 | 0 |
| `kalypso_block:4/4x4/4` | 0 | BlockSemFilter | 4 | 8 | 32 | 16 | 0.50 | 1.00 | 0.34 [0.34-0.38] | 4.67 [4.66-4.77] | n/a | 0 |
| `kalypso_block:4/4x4/4` | 1 | BlockSemJoin | 4 | 4 | 16 | 2 [2-4] | 0.12 [0.12-0.25] | 1.00 | 1.53 [1.53-1.58] | 9.27 [9.24-9.77] | 3.15 [3.12-3.19] | 0 |
| `kalypso_block:4/4x4/4` | 2 | BlockSemFilter | 4 | 2 | 2 [2-4] | 1 | 0.50 [0.25-0.50] | 0.25 [0.25-0.50] | 7.36 [7.34-7.49] | 9.40 [9.37-10.44] | 1.91 [1.90-2.28] | 2 [1-2] |
| `kalypso_block:4/4x4/4+materialized` | 0 | BlockSemFilter | 4 | 8 | 32 | 16 | 0.50 | 1.00 | 0.36 [0.35-0.39] | 6.68 [6.66-6.70] | n/a | 0 |
| `kalypso_block:4/4x4/4+materialized` | 1 | BlockSemJoin | 4 | 4 | 16 | 4 | 0.25 | 1.00 | 6.71 [6.69-6.73] | 15.57 [15.51-15.63] | 0.00 | 0 |
| `kalypso_block:4/4x4/4+materialized` | 2 | BlockSemFilter | 4 | 1 | 4 | 0 | 0.00 | 1.00 | 15.58 [15.52-15.64] | 16.48 [16.42-16.55] | 0.00 | 0 |
| `kalypso_block:4/2x2/4` | 0 | BlockSemFilter | 4 | 8 | 32 | 16 | 0.50 | 1.00 | 0.61 [0.58-0.61] | 5.45 [5.43-5.47] | n/a | 0 |
| `kalypso_block:4/2x2/4` | 1 | BlockSemJoin | 2 | 8 | 16 | 6 | 0.38 | 1.00 | 1.77 [1.73-1.77] | 15.64 [15.54-15.79] | 3.70 [3.66-3.71] | 0 |
| `kalypso_block:4/2x2/4` | 2 | BlockSemFilter | 4 | 5 [5-6] | 6 | 3 [3-5] | 0.50 [0.50-0.83] | 0.30 [0.25-0.30] | 12.79 [12.73-12.89] | 16.12 [15.82-16.27] | 2.85 [2.81-2.90] | 4 [4-6] |
| `kalypso_block:4/2x2/4+materialized` | 0 | BlockSemFilter | 4 | 8 | 32 | 16 | 0.50 | 1.00 | 0.59 [0.57-0.60] | 6.88 [6.87-6.92] | n/a | 0 |
| `kalypso_block:4/2x2/4+materialized` | 1 | BlockSemJoin | 2 | 8 | 16 | 6 | 0.38 | 1.00 | 6.91 [6.91-6.95] | 27.32 [27.15-27.75] | 0.00 | 0 |
| `kalypso_block:4/2x2/4+materialized` | 2 | BlockSemFilter | 4 | 2 | 6 | 1 | 0.17 | 0.75 | 27.33 [27.16-27.75] | 28.43 [28.26-28.89] | 0.00 | 0 |


Operators (first repeat):

| Arm | Query op | Kalypso operator | Rows in | Rows out | Answers began with | Other answers (samples) |
|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 0: sem_filter | SemFilter | 32 | 16 | false 16, true 16 | - |
| `kalypso_per_tuple` | 1: join | SemFilter | 512 | 13 | false 499, true 13 | - |
| `kalypso_per_tuple` | 2: sem_filter | SemFilter | 13 | 6 | false 7, true 6 | - |
| `kalypso_per_tuple+materialized` | 0: sem_filter | SemFilter | 32 | 16 | false 16, true 16 | - |
| `kalypso_per_tuple+materialized` | 1: join | SemFilter | 512 | 13 | false 499, true 13 | - |
| `kalypso_per_tuple+materialized` | 2: sem_filter | SemFilter | 13 | 6 | false 7, true 6 | - |
| `kalypso_block:4/4x4/4` | 0: sem_filter | BlockSemFilter | 32 | 16 | false 16, true 16 | - |
| `kalypso_block:4/4x4/4` | 1: join | BlockSemJoin | 16 | 2 | - | - |
| `kalypso_block:4/4x4/4` | 2: sem_filter | BlockSemFilter | 2 | 1 | false 1, true 1 | - |
| `kalypso_block:4/4x4/4+materialized` | 0: sem_filter | BlockSemFilter | 32 | 16 | false 16, true 16 | - |
| `kalypso_block:4/4x4/4+materialized` | 1: join | BlockSemJoin | 16 | 4 | - | - |
| `kalypso_block:4/4x4/4+materialized` | 2: sem_filter | BlockSemFilter | 4 | 0 | false 4 | - |
| `kalypso_block:4/2x2/4` | 0: sem_filter | BlockSemFilter | 32 | 16 | false 16, true 16 | - |
| `kalypso_block:4/2x2/4` | 1: join | BlockSemJoin | 16 | 6 | - | - |
| `kalypso_block:4/2x2/4` | 2: sem_filter | BlockSemFilter | 6 | 5 | false 1, true 5 | - |
| `kalypso_block:4/2x2/4+materialized` | 0: sem_filter | BlockSemFilter | 32 | 16 | false 16, true 16 | - |
| `kalypso_block:4/2x2/4+materialized` | 1: join | BlockSemJoin | 16 | 6 | - | - |
| `kalypso_block:4/2x2/4+materialized` | 2: sem_filter | BlockSemFilter | 6 | 1 | false 5, true 1 | - |


Work passed downstream (medians). Gold: 16 rows should pass the filter, 9 final outputs are expected.

| Arm | Rows past first filter | Pairs past join | Final outputs | Comparable |
|---|---|---|---|---|
| `kalypso_per_tuple` | 16 | 13 | 6 | reference |
| `kalypso_per_tuple+materialized` | 16 | 13 | 6 | yes |
| `kalypso_block:4/4x4/4` | 16 | 2 | 1 | **no, work drifted** |
| `kalypso_block:4/4x4/4+materialized` | 16 | 4 | 0 | **no, work drifted** |
| `kalypso_block:4/2x2/4` | 16 | 6 | 3 | **no, work drifted** |
| `kalypso_block:4/2x2/4+materialized` | 16 | 6 | 1 | **no, work drifted** |

Drifted by more than 10% from `kalypso_per_tuple`: `kalypso_block:4/4x4/4`, `kalypso_block:4/4x4/4+materialized`, `kalypso_block:4/2x2/4`, `kalypso_block:4/2x2/4+materialized`. Their latency and requests reflect different work, not just blocking.
