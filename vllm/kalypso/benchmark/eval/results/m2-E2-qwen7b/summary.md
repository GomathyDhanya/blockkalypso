# E2

## join_16x8: 16 x 8, f = 1

Model `Qwen/Qwen2.5-7B-Instruct` on NVIDIA L40S (Linux-4.19.0-gvisor-x86_64-with-glibc2.39); workload `arxiv_join` (join), 16 x 8 = 128 pairs, 16 positive; 3 repeats after 1 warmup; prefix cache reset per trial: yes; git `d19f407a0` (dirty).
Server: `python3 -m vllm.entrypoints.cli.main serve Qwen/Qwen2.5-7B-Instruct --host 127.0.0.1 --port 8000 --enable-prefix-caching --no-async-scheduling --api-server-count 1 --max-model-len 16384` (started Tue Oct 20 15:59:43 2026).

| Arm | Completion time (s) | Speedup vs kalypso_per_tuple | LLM calls [retries + fallbacks] | Agrees w/ kalypso_per_tuple | F1 vs labels | Peak KV % |
|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 3.77 [3.67-3.91] | 1.00x | 128 [0] | 100% | 0.12 | 0.6 |
| `kalypso_block:1x2` | 3.21 [3.17-3.29] | 1.17x | 64 [0] | 100% | 0.12 | 0.9 |
| `kalypso_block:1x4` | 3.08 [3.02-3.16] | 1.22x | 32 [0] | 100% | 0.12 | 1.1 |
| `kalypso_block:1x8` | 3.10 [3.01-3.13] | 1.22x | 16 [0] | 100% | 0.12 | 1.4 |
| `kalypso_block:2x1` | 2.22 [2.19-2.24] | 1.70x | 64 [0] | 98% | 0.00 | 0.8 |
| `kalypso_block:2x2` | 1.85 [1.83-1.88] | 2.04x | 32 [0] | 99% | n/a | 1.0 |
| `kalypso_block:2x4` | 1.79 [1.77-1.82] | 2.11x | 16 [0] | 98% | 0.11 | 1.3 [1.3-1.4] |
| `kalypso_block:2x8` | 1.71 [1.67-1.71] | 2.21x | 8 [0] | 99% | n/a | 1.0 |
| `kalypso_block:4x1` | 1.47 [1.45-1.47] | 2.57x | 32 [0] | 99% | n/a | 0.6 [0.6-0.6] |
| `kalypso_block:4x2` | 1.19 [1.19-1.21] | 3.17x | 16 [0] | 99% | n/a | 0.7 |
| `kalypso_block:4x4` | 1.08 [1.06-1.08] | 3.50x | 8 [0] | 99% | n/a | 0.9 |
| `kalypso_block:4x8` | 1.04 [1.03-1.06] | 3.61x | 4 [0] | 99% | n/a | 1.1 |
| `kalypso_block:8x1` | 1.25 [1.24-1.25] | 3.02x | 16 [0] | 100% | 0.12 | 0.9 [0.9-1.1] |
| `kalypso_block:8x2` | 0.88 [0.87-0.89] | 4.31x | 8 [0] | 99% | n/a | 0.9 |
| `kalypso_block:8x4` | 0.77 [0.76-0.77] | 4.91x | 4 [0] | 99% | n/a | 0.9 |
| `kalypso_block:8x8` | 0.72 [0.71-0.74] | 5.21x | 2 [0] | 99% | n/a | 1.4 |


Diagnostics:

| Arm | Prompt tokens | Gen tokens | Prefix hit rate | Peak RSS (MiB) | Items/s | First output (s) | Truncated | Accuracy | Precision | Recall | Self-agreement |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 83888 | 256 | 0.43 | 12279 [12279-12279] | 33.95 [32.74-34.83] | 3.42 [3.32-3.54] | 0 | 0.88 | 1.00 | 0.06 | 100% |
| `kalypso_block:1x2` | 54916 | 132 | 0.27 | 12279 [12279-12279] | 39.88 [38.85-40.31] | 3.21 [3.17-3.29] | 0 | 0.88 | 1.00 | 0.06 | 100% |
| `kalypso_block:1x4` | 43826 | 68 | 0.12 | 12279 [12279-12279] | 41.51 [40.51-42.31] | 3.08 [3.02-3.16] | 0 | 0.88 | 1.00 | 0.06 | 100% |
| `kalypso_block:1x8` | 38281 | 36 | 0.01 | 12279 [12279-12279] | 41.34 [40.93-42.57] | 3.09 [3.00-3.12] | 0 | 0.88 | 1.00 | 0.06 | 100% |
| `kalypso_block:2x1` | 54976 | 132 | 0.56 | 12279 [12279-12279] | 57.68 [57.14-58.34] | 2.22 [2.19-2.24] | 0 | 0.87 | 0.00 | 0.00 | 100% |
| `kalypso_block:2x2` | 35668 | 64 | 0.38 | 12279 [12279-12279] | 69.23 [68.12-69.94] | n/a | 0 | 0.88 | n/a | 0.00 | 100% |
| `kalypso_block:2x4` | 26018 | 40 | 0.18 | 12279 [12279-12279] | 71.67 [70.37-72.11] | 1.76 [1.75-1.79] | 0 | 0.88 | 0.50 | 0.06 | 100% |
| `kalypso_block:2x8` | 21193 | 16 | 0.01 | 12279 [12279-12279] | 75.03 [74.94-76.60] | n/a | 0 | 0.88 | n/a | 0.00 | 100% |
| `kalypso_block:4x1` | 43916 | 64 | 0.68 | 12279 [12279-12279] | 87.37 [87.37-88.22] | n/a | 0 | 0.88 | n/a | 0.00 | 100% |
| `kalypso_block:4x2` | 26048 | 32 | 0.50 | 12279 [12279-12279] | 107.65 [106.22-107.74] | n/a | 0 | 0.88 | n/a | 0.00 | 100% |
| `kalypso_block:4x4` | 17116 | 16 | 0.26 | 12279 [12279-12279] | 118.85 [118.19-120.75] | n/a | 0 | 0.88 | n/a | 0.00 | 100% |
| `kalypso_block:4x8` | 12650 | 8 | 0.01 | 12279 [12279-12279] | 122.61 [120.41-124.76] | n/a | 0 | 0.88 | n/a | 0.00 | 100% |
| `kalypso_block:8x1` | 38382 | 36 | 0.77 | 12279 [12279-12279] | 102.56 [102.16-103.06] | 1.24 [1.24-1.25] | 0 | 0.88 | 1.00 | 0.06 | 100% |
| `kalypso_block:8x2` | 21236 | 16 | 0.59 | 12279 [12279-12279] | 146.29 [143.82-147.64] | n/a | 0 | 0.88 | n/a | 0.00 | 100% |
| `kalypso_block:8x4` | 12664 | 8 | 0.33 | 12279 [12279-12279] | 166.67 [166.45-168.20] | n/a | 0 | 0.88 | n/a | 0.00 | 100% |
| `kalypso_block:8x8` | 8378 | 4 | 0.00 | 12279 [12279-12279] | 177.04 [173.68-179.27] | n/a | 0 | 0.88 | n/a | 0.00 | 100% |


Stages (medians over repeats; times from query start):

| Arm | Stage | Operators | Block | Tasks | Rows in | Rows out | Out/in | Occupancy | Start (s) | End (s) | Overlap (s) | Fallbacks |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 0 | SemFilter | 1 | 128 | 128 | 1 | 0.01 | n/a | 0.02 [0.02-0.02] | 3.77 [3.67-3.90] | n/a | 0 |
| `kalypso_block:1x2` | 0 | BlockSemJoin | 1 | 16 | 16 | 1 | 0.06 | n/a | 0.10 [0.10-0.10] | 3.21 [3.17-3.29] | n/a | 0 |
| `kalypso_block:1x4` | 0 | BlockSemJoin | 1 | 16 | 16 | 1 | 0.06 | n/a | 0.08 [0.08-0.08] | 3.08 [3.02-3.16] | n/a | 0 |
| `kalypso_block:1x8` | 0 | BlockSemJoin | 1 | 16 | 16 | 1 | 0.06 | n/a | 0.07 [0.07-0.07] | 3.09 [3.00-3.12] | n/a | 0 |
| `kalypso_block:2x1` | 0 | BlockSemJoin | 2 | 8 | 16 | 1 | 0.06 | 1.00 | 0.11 [0.11-0.11] | 2.22 [2.19-2.24] | n/a | 0 |
| `kalypso_block:2x2` | 0 | BlockSemJoin | 2 | 8 | 16 | 0 | 0.00 | 1.00 | 0.07 [0.07-0.07] | 1.85 [1.83-1.88] | n/a | 0 |
| `kalypso_block:2x4` | 0 | BlockSemJoin | 2 | 8 | 16 | 2 | 0.12 | 1.00 | 0.05 [0.05-0.05] | 1.78 [1.77-1.82] | n/a | 0 |
| `kalypso_block:2x8` | 0 | BlockSemJoin | 2 | 8 | 16 | 0 | 0.00 | 1.00 | 0.04 [0.04-0.04] | 1.70 [1.67-1.70] | n/a | 0 |
| `kalypso_block:4x1` | 0 | BlockSemJoin | 4 | 4 | 16 | 0 | 0.00 | 1.00 | 0.09 [0.09-0.09] | 1.46 [1.45-1.46] | n/a | 0 |
| `kalypso_block:4x2` | 0 | BlockSemJoin | 4 | 4 | 16 | 0 | 0.00 | 1.00 | 0.06 [0.06-0.06] | 1.19 [1.18-1.20] | n/a | 0 |
| `kalypso_block:4x4` | 0 | BlockSemJoin | 4 | 4 | 16 | 0 | 0.00 | 1.00 | 0.04 [0.04-0.04] | 1.07 [1.06-1.08] | n/a | 0 |
| `kalypso_block:4x8` | 0 | BlockSemJoin | 4 | 4 | 16 | 0 | 0.00 | 1.00 | 0.03 [0.03-0.03] | 1.04 [1.02-1.06] | n/a | 0 |
| `kalypso_block:8x1` | 0 | BlockSemJoin | 8 | 2 | 16 | 1 | 0.06 | 1.00 | 0.10 [0.10-0.10] | 1.24 [1.24-1.25] | n/a | 0 |
| `kalypso_block:8x2` | 0 | BlockSemJoin | 8 | 2 | 16 | 0 | 0.00 | 1.00 | 0.06 [0.06-0.06] | 0.87 [0.86-0.89] | n/a | 0 |
| `kalypso_block:8x4` | 0 | BlockSemJoin | 8 | 2 | 16 | 0 | 0.00 | 1.00 | 0.04 [0.04-0.04] | 0.77 [0.76-0.77] | n/a | 0 |
| `kalypso_block:8x8` | 0 | BlockSemJoin | 8 | 2 | 16 | 0 | 0.00 | 1.00 | 0.02 [0.02-0.02] | 0.72 [0.71-0.73] | n/a | 0 |

