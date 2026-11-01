# Batching study results

Best arm: fastest with F1 at least per-tuple's minus 0.02 (unlabeled maps: ROUGE-1 similarity to per-tuple's outputs (Sim) at least 0.8). Medians over repeats; speedup is against per-tuple in the same container.

## block-size

### block-size/movie_opposite-n32-b11-Qwen2.5-7B-Instruct

**block-size/movie_opposite-n32-b11-Qwen2.5-7B-Instruct**: `movie_opposite` (join), 32 x 32 = 1024 pairs, 50% positive; GPU L40S (not recorded), max_num_seqs 256, git `d16192471`.

| Arm | Latency (s) | Speedup | Calls | Fallbacks | Waves | Peak running | Peak waiting | Computed prompt tok | Gen tok/call | Decode/call (s) | s/gen tok | Hit rate | Peak KV % | Preempt | Recall | F1 | ΔF1 | Agree | Self | Sim | Selected |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 10.73 | 1.00 | 1024 | – | 4 | – | – | 138845 | 2.2 | 0.18 | 0.079 | 0.42 | 0.8 | – | 0.62 | 0.60 | 0.00 | 1.00 | 1.00 | – | 546 |
| `kalypso_block:11` | 8.52 | 1.26 | 94 | 1 | 1 | – | – | 89144 | 45.5 | 4.04 | 0.089 | 0.07 | 18.1 | – | 0.46 | 0.51 | -0.09 | 0.62 | 1.00 | – | 402 |

Best: `kalypso_per_tuple` (1.00x).

### block-size/movie_opposite-n32-b2-Qwen2.5-7B-Instruct

**block-size/movie_opposite-n32-b2-Qwen2.5-7B-Instruct**: `movie_opposite` (join), 32 x 32 = 1024 pairs, 50% positive; GPU L40S (not recorded), max_num_seqs 256, git `44f36173b`.

| Arm | Latency (s) | Speedup | Calls | Fallbacks | Waves | Peak running | Peak waiting | Computed prompt tok | Gen tok/call | Decode/call (s) | s/gen tok | Hit rate | Peak KV % | Preempt | Recall | F1 | ΔF1 | Agree | Self | Sim | Selected |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 11.24 | 1.00 | 1024 | – | 4 | – | – | 138845 | 2.2 | 0.18 | 0.082 | 0.42 | 0.8 | – | 0.62 | 0.60 | 0.00 | 1.00 | 1.00 | – | 544 |
| `kalypso_block:2` | 8.03 | 1.40 | 512 | 0 | 2 | – | – | 89594 | 8.0 | 1.00 | 0.125 | 0.33 | 3.4 | – | 0.67 | 0.63 | 0.03 | 0.59 | 0.96 | – | 570 |

Best: `kalypso_block:2` (1.40x).

### block-size/movie_opposite-n32-b6-Qwen2.5-7B-Instruct

**block-size/movie_opposite-n32-b6-Qwen2.5-7B-Instruct**: `movie_opposite` (join), 32 x 32 = 1024 pairs, 50% positive; GPU L40S (not recorded), max_num_seqs 256, git `2ac9745e0`.

| Arm | Latency (s) | Speedup | Calls | Fallbacks | Waves | Peak running | Peak waiting | Computed prompt tok | Gen tok/call | Decode/call (s) | s/gen tok | Hit rate | Peak KV % | Preempt | Recall | F1 | ΔF1 | Agree | Self | Sim | Selected |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 11.05 | 1.00 | 1024 | – | 4 | – | – | 138845 | 2.2 | 0.18 | 0.081 | 0.42 | 0.9 | – | 0.62 | 0.60 | 0.00 | 1.00 | 1.00 | – | 544 |
| `kalypso_block:6` | 8.28 | 1.34 | 171 | 0 | 1 | – | – | 88939 | 23.9 | 2.84 | 0.118 | 0.13 | 10.8 | – | 0.69 | 0.63 | 0.02 | 0.66 | 1.00 | – | 615 |

Best: `kalypso_block:6` (1.34x).

### block-size/movie_positive-n500-b11

**block-size/movie_positive-n500-b11**: `movie_positive` (filter), 500 rows, 75% positive; GPU L40S (not recorded), max_num_seqs 256, git `db0f47ac7`.

| Arm | Latency (s) | Speedup | Calls | Fallbacks | Waves | Peak running | Peak waiting | Computed prompt tok | Gen tok/call | Decode/call (s) | s/gen tok | Hit rate | Peak KV % | Preempt | Recall | F1 | ΔF1 | Agree | Self | Sim | Selected |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 4.77 | 1.00 | 500 | – | 2 | – | – | 57728 | 3.0 | 0.28 | 0.094 | 0.29 | 1.0 | – | 0.63 | 0.77 | 0.00 | 1.00 | 1.00 | – | 235 |
| `kalypso_block:11` | 2.83 | 1.68 | 46 | – | 1 | – | – | 19248 | 45.4 | 1.79 | 0.039 | 0.07 | 4.7 | – | 0.73 | 0.84 | 0.07 | 0.87 | 1.00 | – | 280 |

Best: `kalypso_block:11` (1.68x).

### block-size/movie_positive-n500-b16

**block-size/movie_positive-n500-b16**: `movie_positive` (filter), 500 rows, 75% positive; GPU L40S (not recorded), max_num_seqs 256, git `db0f47ac7`.

| Arm | Latency (s) | Speedup | Calls | Fallbacks | Waves | Peak running | Peak waiting | Computed prompt tok | Gen tok/call | Decode/call (s) | s/gen tok | Hit rate | Peak KV % | Preempt | Recall | F1 | ΔF1 | Agree | Self | Sim | Selected |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 4.69 | 1.00 | 500 | – | 2 | – | – | 57728 | 3.0 | 0.28 | 0.093 | 0.29 | 1.0 | – | 0.63 | 0.77 | 0.00 | 1.00 | 1.00 | – | 235 |
| `kalypso_block:16` | 3.38 | 1.39 | 32 | – | 1 | – | – | 18634 | 69.3 | 2.35 | 0.034 | 0.05 | 4.6 | – | 0.76 | 0.86 | 0.09 | 0.85 | 1.00 | – | 285 |

Best: `kalypso_block:16` (1.39x).

### block-size/movie_positive-n500-b20

**block-size/movie_positive-n500-b20**: `movie_positive` (filter), 500 rows, 75% positive; GPU L40S (not recorded), max_num_seqs 256, git `db0f47ac7`.

| Arm | Latency (s) | Speedup | Calls | Fallbacks | Waves | Peak running | Peak waiting | Computed prompt tok | Gen tok/call | Decode/call (s) | s/gen tok | Hit rate | Peak KV % | Preempt | Recall | F1 | ΔF1 | Agree | Self | Sim | Selected |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 4.71 | 1.00 | 500 | – | 2 | – | – | 57728 | 3.0 | 0.27 | 0.093 | 0.29 | 1.0 | – | 0.63 | 0.77 | 0.00 | 1.00 | 1.00 | – | 235 |
| `kalypso_block:20` | 3.91 | 1.21 | 25 | – | 1 | – | – | 18323 | 91.0 | 2.94 | 0.032 | 0.04 | 4.6 | – | 0.79 | 0.87 | 0.10 | 0.83 | 1.00 | – | 297 |

Best: `kalypso_block:20` (1.21x).

### block-size/movie_positive-n500-b4

**block-size/movie_positive-n500-b4**: `movie_positive` (filter), 500 rows, 75% positive; GPU L40S (not recorded), max_num_seqs 256, git `db0f47ac7`.

| Arm | Latency (s) | Speedup | Calls | Fallbacks | Waves | Peak running | Peak waiting | Computed prompt tok | Gen tok/call | Decode/call (s) | s/gen tok | Hit rate | Peak KV % | Preempt | Recall | F1 | ΔF1 | Agree | Self | Sim | Selected |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 4.55 | 1.00 | 500 | – | 2 | – | – | 57728 | 3.0 | 0.27 | 0.093 | 0.29 | 1.0 | – | 0.63 | 0.77 | 0.00 | 1.00 | 1.00 | – | 235 |
| `kalypso_block:4` | 2.28 | 2.00 | 125 | – | 1 | – | – | 23065 | 16.0 | 1.08 | 0.067 | 0.15 | 5.6 | – | 0.77 | 0.87 | 0.10 | 0.85 | 1.00 | – | 291 |

Best: `kalypso_block:4` (2.00x).

### block-size/movie_positive-n500-b6

**block-size/movie_positive-n500-b6**: `movie_positive` (filter), 500 rows, 75% positive; GPU L40S (not recorded), max_num_seqs 256, git `db0f47ac7`.

| Arm | Latency (s) | Speedup | Calls | Fallbacks | Waves | Peak running | Peak waiting | Computed prompt tok | Gen tok/call | Decode/call (s) | s/gen tok | Hit rate | Peak KV % | Preempt | Recall | F1 | ΔF1 | Agree | Self | Sim | Selected |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 4.85 | 1.00 | 500 | – | 2 | – | – | 57728 | 3.0 | 0.28 | 0.096 | 0.29 | 1.0 | – | 0.63 | 0.77 | 0.00 | 1.00 | 1.00 | – | 235 |
| `kalypso_block:6` | 2.43 | 1.99 | 84 | – | 1 | – | – | 20991 | 23.8 | 1.26 | 0.053 | 0.11 | 5.0 | – | 0.77 | 0.87 | 0.10 | 0.85 | 1.00 | – | 290 |

Best: `kalypso_block:6` (1.99x).

### block-size/movie_positive-n500-b8

**block-size/movie_positive-n500-b8**: `movie_positive` (filter), 500 rows, 75% positive; GPU L40S (not recorded), max_num_seqs 256, git `db0f47ac7`.

| Arm | Latency (s) | Speedup | Calls | Fallbacks | Waves | Peak running | Peak waiting | Computed prompt tok | Gen tok/call | Decode/call (s) | s/gen tok | Hit rate | Peak KV % | Preempt | Recall | F1 | ΔF1 | Agree | Self | Sim | Selected |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 4.56 | 1.00 | 500 | – | 2 | – | – | 57728 | 3.0 | 0.27 | 0.091 | 0.29 | 1.0 | – | 0.63 | 0.77 | 0.00 | 1.00 | 1.00 | – | 235 |
| `kalypso_block:8` | 2.43 | 1.88 | 63 | – | 1 | – | – | 19935 | 31.8 | 1.39 | 0.044 | 0.09 | 4.8 | – | 0.78 | 0.87 | 0.10 | 0.85 | 1.00 | – | 293 |

Best: `kalypso_block:8` (1.88x).

### block-size/movie_positive-n500-controls

**block-size/movie_positive-n500-controls**: `movie_positive` (filter), 500 rows, 75% positive; GPU L40S (not recorded), max_num_seqs 256, git `db0f47ac7`.

| Arm | Latency (s) | Speedup | Calls | Fallbacks | Waves | Peak running | Peak waiting | Computed prompt tok | Gen tok/call | Decode/call (s) | s/gen tok | Hit rate | Peak KV % | Preempt | Recall | F1 | ΔF1 | Agree | Self | Sim | Selected |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 4.90 | 1.00 | 500 | – | 2 | – | – | 57728 | 3.0 | 0.29 | 0.098 | 0.29 | 1.0 | – | 0.63 | 0.77 | 0.00 | 1.00 | 1.00 | – | 235 |
| `vllm_direct` | 4.92 | 0.99 | 500 | – | 2 | – | – | 57728 | 3.0 | 0.29 | 0.097 | 0.29 | 1.0 | – | 0.63 | 0.77 | 0.00 | 1.00 | 1.00 | – | 235 |
| `vllm_direct:no_penalty` | 4.88 | 1.00 | 500 | – | 2 | – | – | 57728 | 3.0 | 0.28 | 0.096 | 0.29 | 0.9 | – | 0.63 | 0.77 | 0.00 | 1.00 | 1.00 | – | 236 |
| `vllm_direct:block_prompt` | 3.84 | 1.27 | 500 | – | 2 | – | – | 32112 | 2.0 | 0.11 | 0.056 | 0.20 | 0.5 | – | 0.62 | 0.77 | -0.00 | 0.92 | 1.00 | – | 233 |
| `vllm_direct:block_prompt+no_penalty` | 3.46 | 1.42 | 500 | – | 2 | – | – | 32112 | 2.0 | 0.11 | 0.057 | 0.20 | 0.5 | – | 0.62 | 0.77 | -0.00 | 0.92 | 1.00 | – | 233 |

Best: `vllm_direct:block_prompt+no_penalty` (1.42x).

### block-size/movie_sel10-b4-6-8-16-Qwen2.5-7B-Instruct

**block-size/movie_sel10-b4-6-8-16-Qwen2.5-7B-Instruct**: `movie_sel10` (filter), 400 rows, 10% positive; GPU L40S (not recorded), max_num_seqs 256, git `3cdb0b42d`.

| Arm | Latency (s) | Speedup | Calls | Fallbacks | Waves | Peak running | Peak waiting | Computed prompt tok | Gen tok/call | Decode/call (s) | s/gen tok | Hit rate | Peak KV % | Preempt | Recall | F1 | ΔF1 | Agree | Self | Sim | Selected |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 3.95 | 1.00 | 400 | – | 2 | – | – | 46027 | 3.0 | 0.29 | 0.097 | 0.29 | 1.0 | – | 0.62 | 0.70 | 0.00 | 1.00 | 1.00 | – | 31 |
| `kalypso_block:4` | 2.03 | 1.95 | 100 | – | 1 | – | – | 18388 | 16.0 | 0.97 | 0.061 | 0.15 | 4.4 | – | 0.82 | 0.80 | 0.09 | 0.95 | 1.00 | – | 43 |
| `kalypso_block:6` | 2.08 | 1.90 | 67 | – | 1 | – | – | 16690 | 23.9 | 1.12 | 0.047 | 0.11 | 4.0 | – | 0.82 | 0.78 | 0.07 | 0.95 | 1.00 | – | 45 |
| `kalypso_block:8` | 2.21 | 1.79 | 50 | – | 1 | – | – | 15839 | 32.0 | 1.33 | 0.042 | 0.09 | 3.9 | – | 0.72 | 0.74 | 0.04 | 0.97 | 1.00 | – | 38 |
| `kalypso_block:16` | 3.10 | 1.27 | 25 | – | 1 | – | – | 14788 | 71.0 | 2.30 | 0.032 | 0.05 | 3.7 | – | 0.60 | 0.68 | -0.03 | 0.96 | 1.00 | – | 31 |

Best: `kalypso_block:4` (1.95x).

### block-size/movie_sel25-b4-6-8-16-Qwen2.5-7B-Instruct

**block-size/movie_sel25-b4-6-8-16-Qwen2.5-7B-Instruct**: `movie_sel25` (filter), 400 rows, 25% positive; GPU L40S (not recorded), max_num_seqs 256, git `3cdb0b42d`.

| Arm | Latency (s) | Speedup | Calls | Fallbacks | Waves | Peak running | Peak waiting | Computed prompt tok | Gen tok/call | Decode/call (s) | s/gen tok | Hit rate | Peak KV % | Preempt | Recall | F1 | ΔF1 | Agree | Self | Sim | Selected |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 3.76 | 1.00 | 400 | – | 2 | – | – | 46221 | 3.0 | 0.27 | 0.092 | 0.29 | 1.0 | – | 0.62 | 0.74 | 0.00 | 1.00 | 1.00 | – | 68 |
| `kalypso_block:4` | 1.95 | 1.93 | 100 | – | 1 | – | – | 18565 | 16.0 | 0.93 | 0.058 | 0.15 | 4.5 | – | 0.82 | 0.87 | 0.13 | 0.93 | 1.00 | – | 89 |
| `kalypso_block:6` | 2.00 | 1.88 | 67 | – | 1 | – | – | 16898 | 23.9 | 1.09 | 0.045 | 0.11 | 4.1 | – | 0.86 | 0.87 | 0.14 | 0.92 | 1.00 | – | 97 |
| `kalypso_block:8` | 2.15 | 1.75 | 50 | – | 1 | – | – | 16015 | 32.0 | 1.31 | 0.041 | 0.09 | 3.9 | – | 0.78 | 0.85 | 0.11 | 0.92 | 1.00 | – | 83 |
| `kalypso_block:16` | 3.05 | 1.23 | 25 | – | 1 | – | – | 14980 | 71.0 | 2.25 | 0.032 | 0.05 | 3.7 | – | 0.69 | 0.79 | 0.06 | 0.93 | 1.00 | – | 74 |

Best: `kalypso_block:4` (1.93x).

### block-size/movie_sel50-b4-6-8-16-Qwen2.5-7B-Instruct

**block-size/movie_sel50-b4-6-8-16-Qwen2.5-7B-Instruct**: `movie_sel50` (filter), 400 rows, 50% positive; GPU L40S (not recorded), max_num_seqs 256, git `3cdb0b42d`.

| Arm | Latency (s) | Speedup | Calls | Fallbacks | Waves | Peak running | Peak waiting | Computed prompt tok | Gen tok/call | Decode/call (s) | s/gen tok | Hit rate | Peak KV % | Preempt | Recall | F1 | ΔF1 | Agree | Self | Sim | Selected |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 3.75 | 1.00 | 400 | – | 2 | – | – | 46369 | 3.0 | 0.27 | 0.091 | 0.29 | 1.0 | – | 0.60 | 0.75 | 0.00 | 1.00 | 1.00 | – | 123 |
| `kalypso_block:4` | 1.96 | 1.92 | 100 | – | 1 | – | – | 18648 | 16.0 | 0.94 | 0.059 | 0.15 | 4.5 | – | 0.80 | 0.87 | 0.13 | 0.87 | 1.00 | – | 166 |
| `kalypso_block:6` | 2.01 | 1.86 | 67 | – | 1 | – | – | 16965 | 23.9 | 1.10 | 0.046 | 0.11 | 4.1 | – | 0.81 | 0.88 | 0.13 | 0.87 | 1.00 | – | 166 |
| `kalypso_block:8` | 2.15 | 1.74 | 50 | – | 1 | – | – | 16097 | 32.0 | 1.30 | 0.041 | 0.09 | 4.0 | – | 0.77 | 0.86 | 0.11 | 0.89 | 1.00 | – | 160 |
| `kalypso_block:16` | 3.04 | 1.23 | 25 | – | 1 | – | – | 15047 | 71.0 | 2.24 | 0.032 | 0.05 | 3.7 | – | 0.72 | 0.83 | 0.08 | 0.90 | 1.00 | – | 151 |

Best: `kalypso_block:4` (1.92x).

### block-size/movie_sel75-b4-6-8-16-Qwen2.5-7B-Instruct

**block-size/movie_sel75-b4-6-8-16-Qwen2.5-7B-Instruct**: `movie_sel75` (filter), 400 rows, 75% positive; GPU L40S (not recorded), max_num_seqs 256, git `3cdb0b42d`.

| Arm | Latency (s) | Speedup | Calls | Fallbacks | Waves | Peak running | Peak waiting | Computed prompt tok | Gen tok/call | Decode/call (s) | s/gen tok | Hit rate | Peak KV % | Preempt | Recall | F1 | ΔF1 | Agree | Self | Sim | Selected |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 3.81 | 1.00 | 400 | – | 2 | – | – | 46012 | 2.9 | 0.27 | 0.093 | 0.29 | 1.0 | – | 0.61 | 0.76 | 0.00 | 1.00 | 1.00 | – | 185 |
| `kalypso_block:4` | 1.99 | 1.91 | 100 | – | 1 | – | – | 18293 | 16.0 | 0.96 | 0.060 | 0.15 | 4.5 | – | 0.80 | 0.88 | 0.12 | 0.83 | 1.00 | – | 246 |
| `kalypso_block:6` | 2.02 | 1.89 | 67 | – | 1 | – | – | 16610 | 23.9 | 1.11 | 0.046 | 0.11 | 4.0 | – | 0.80 | 0.89 | 0.13 | 0.84 | 1.00 | – | 242 |
| `kalypso_block:8` | 2.18 | 1.75 | 50 | – | 1 | – | – | 15743 | 32.0 | 1.32 | 0.041 | 0.09 | 3.9 | – | 0.80 | 0.89 | 0.13 | 0.83 | 1.00 | – | 241 |
| `kalypso_block:16` | 3.09 | 1.23 | 25 | – | 1 | – | – | 14693 | 71.0 | 2.28 | 0.032 | 0.05 | 3.6 | – | 0.78 | 0.88 | 0.12 | 0.84 | 1.00 | – | 237 |

Best: `kalypso_block:4` (1.91x).

### block-size/movie_sel90-b4-6-8-16-Qwen2.5-7B-Instruct

**block-size/movie_sel90-b4-6-8-16-Qwen2.5-7B-Instruct**: `movie_sel90` (filter), 400 rows, 90% positive; GPU L40S (not recorded), max_num_seqs 256, git `3cdb0b42d`.

| Arm | Latency (s) | Speedup | Calls | Fallbacks | Waves | Peak running | Peak waiting | Computed prompt tok | Gen tok/call | Decode/call (s) | s/gen tok | Hit rate | Peak KV % | Preempt | Recall | F1 | ΔF1 | Agree | Self | Sim | Selected |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 3.75 | 1.00 | 400 | – | 2 | – | – | 46041 | 2.9 | 0.28 | 0.094 | 0.29 | 1.0 | – | 0.64 | 0.78 | 0.00 | 1.00 | 1.00 | – | 232 |
| `kalypso_block:4` | 1.93 | 1.94 | 100 | – | 1 | – | – | 18325 | 16.0 | 0.94 | 0.059 | 0.15 | 4.5 | – | 0.82 | 0.90 | 0.12 | 0.81 | 1.00 | – | 296 |
| `kalypso_block:6` | 2.00 | 1.88 | 67 | – | 1 | – | – | 16642 | 23.9 | 1.09 | 0.046 | 0.11 | 4.0 | – | 0.80 | 0.89 | 0.11 | 0.82 | 1.00 | – | 291 |
| `kalypso_block:8` | 2.14 | 1.75 | 50 | – | 1 | – | – | 15775 | 32.0 | 1.30 | 0.041 | 0.09 | 3.9 | – | 0.79 | 0.88 | 0.10 | 0.81 | 1.00 | – | 288 |
| `kalypso_block:16` | 3.02 | 1.24 | 25 | – | 1 | – | – | 14725 | 71.0 | 2.24 | 0.032 | 0.05 | 3.7 | – | 0.78 | 0.87 | 0.09 | 0.80 | 1.00 | – | 281 |

Best: `kalypso_block:4` (1.94x).

## longrows

### longrows/cnli-filter-s10

**cnli_sel10**: `cnli_sel10` (filter), 400 rows, 10% positive; GPU NVIDIA L40S, 46068 MiB, max_num_seqs 256, git `2db189318`.

| Arm | Latency (s) | Speedup | Calls | Fallbacks | Waves | Peak running | Peak waiting | Computed prompt tok | Gen tok/call | Decode/call (s) | s/gen tok | Hit rate | Peak KV % | Preempt | Recall | F1 | ΔF1 | Agree | Self | Sim | Selected |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 51.21 | 1.00 | 400 | – | 2 | 5 | 272 | 631716 | 3.0 | 0.32 | 0.106 | 0.03 | 1.5 | 0 | 0.62 | 0.27 | 0.00 | 1.00 | 1.00 | – | 147 |
| `kalypso_block:2` | 51.82 | 0.99 | 200 | 0 | 1 | 8 | 141 | 606625 | 8.0 | 1.14 | 0.142 | 0.02 | 4.2 | 0 | 0.40 | 0.33 | 0.06 | 0.74 | 1.00 | – | 58 |
| `kalypso_block:4` | 54.66 | 0.94 | 100 | 0 | 1 | 7 | 71 | 602719 | 16.0 | 2.55 | 0.160 | 0.01 | 8.7 | 0 | 0.35 | 0.22 | -0.04 | 0.72 | 1.00 | – | 85 |
| `kalypso_block:8` | 60.86 | 0.84 | 50 | 0 | 1 | 7 | 36 | 600739 | 32.0 | 5.73 | 0.179 | 0.00 | 17.4 | 0 | 0.28 | 0.22 | -0.05 | 0.70 | 1.00 | – | 62 |

Best: `kalypso_per_tuple` (1.00x).

### longrows/cnli-filter-s50

**cnli_sel50**: `cnli_sel50` (filter), 400 rows, 50% positive; GPU NVIDIA L40S, 46068 MiB, max_num_seqs 256, git `2db189318`.

| Arm | Latency (s) | Speedup | Calls | Fallbacks | Waves | Peak running | Peak waiting | Computed prompt tok | Gen tok/call | Decode/call (s) | s/gen tok | Hit rate | Peak KV % | Preempt | Recall | F1 | ΔF1 | Agree | Self | Sim | Selected |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 52.24 | 1.00 | 400 | – | 2 | 6 | 272 | 631859 | 3.0 | 0.32 | 0.107 | 0.03 | 1.5 | 0 | 0.69 | 0.68 | 0.00 | 1.00 | 1.00 | – | 206 |
| `kalypso_block:2` | 51.36 | 1.02 | 200 | 0 | 1 | 8 | 141 | 606770 | 8.0 | 1.12 | 0.140 | 0.02 | 4.2 | 0 | 0.44 | 0.55 | -0.13 | 0.70 | 1.00 | – | 119 |
| `kalypso_block:4` | 55.30 | 0.94 | 100 | 0 | 1 | 7 | 71 | 602866 | 16.0 | 2.56 | 0.160 | 0.01 | 8.7 | 0 | 0.52 | 0.60 | -0.08 | 0.68 | 1.00 | – | 144 |
| `kalypso_block:8` | 61.85 | 0.84 | 50 | 0 | 1 | 7 | 35 | 600887 | 32.0 | 5.79 | 0.181 | 0.00 | 17.3 | 0 | 0.40 | 0.49 | -0.19 | 0.63 | 1.00 | – | 126 |

Best: `kalypso_per_tuple` (1.00x).

### longrows/cnli-filter-s90

**cnli_sel90**: `cnli_sel90` (filter), 400 rows, 90% positive; GPU NVIDIA L40S, 46068 MiB, max_num_seqs 256, git `2db189318`.

| Arm | Latency (s) | Speedup | Calls | Fallbacks | Waves | Peak running | Peak waiting | Computed prompt tok | Gen tok/call | Decode/call (s) | s/gen tok | Hit rate | Peak KV % | Preempt | Recall | F1 | ΔF1 | Agree | Self | Sim | Selected |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 53.04 | 1.00 | 400 | – | 2 | 5 | 268 | 632014 | 3.0 | 0.32 | 0.107 | 0.03 | 1.5 | 0 | 0.70 | 0.80 | 0.00 | 1.00 | 1.00 | – | 271 |
| `kalypso_block:2` | 52.40 | 1.01 | 200 | 0 | 1 | 7 | 140 | 605879 | 8.0 | 1.13 | 0.141 | 0.02 | 4.3 | 0 | 0.49 | 0.65 | -0.15 | 0.70 | 1.00 | – | 187 |
| `kalypso_block:4` | 56.28 | 0.94 | 102 | 2 | 1 | 8 | 70 | 606109 | 15.7 | 2.55 | 0.163 | 0.01 | 8.6 | 0 | 0.56 | 0.71 | -0.10 | 0.66 | 1.00 | – | 212 |
| `kalypso_block:8` | 61.76 | 0.86 | 50 | 0 | 1 | 7 | 35 | 600986 | 32.0 | 5.70 | 0.178 | 0.00 | 17.4 | 0 | 0.47 | 0.62 | -0.18 | 0.60 | 1.00 | – | 182 |

Best: `kalypso_per_tuple` (1.00x).

### longrows/cnli-join-high

**cnli_join_high**: `cnli_join_high` (join), 24 x 8 = 192 pairs, 65% positive; GPU NVIDIA L40S, 46068 MiB, max_num_seqs 256, git `2db189318`.

| Arm | Latency (s) | Speedup | Calls | Fallbacks | Waves | Peak running | Peak waiting | Computed prompt tok | Gen tok/call | Decode/call (s) | s/gen tok | Hit rate | Peak KV % | Preempt | Recall | F1 | ΔF1 | Agree | Self | Sim | Selected |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 5.58 | 1.00 | 192 | – | 1 | 17 | 151 | 61052 | 2.5 | 0.23 | 0.091 | 0.82 | 1.4 | 0 | 0.42 | 0.57 | 0.00 | 1.00 | 1.00 | – | 56 |
| `kalypso_block:2` | 18.41 | 0.30 | 96 | 0 | 1 | 12 | 91 | 200623 | 8.0 | 1.12 | 0.141 | 0.38 | 4.4 | 0 | 0.31 | 0.44 | -0.14 | 0.67 | 1.00 | – | 54 |
| `kalypso_block:4` | 26.30 | 0.21 | 48 | 0 | 1 | 9 | 46 | 275696 | 16.0 | 2.53 | 0.158 | 0.13 | 9.2 | 0 | 0.49 | 0.57 | -0.00 | 0.64 | 1.00 | – | 88 |
| `kalypso_block:8` | 34.51 | 0.16 | 24 | 0 | 1 | 7 | 23 | 313238 | 32.0 | 5.79 | 0.181 | 0.00 | 18.6 | 0 | 0.48 | 0.57 | -0.00 | 0.54 | 1.00 | – | 85 |

Best: `kalypso_per_tuple` (1.00x).

### longrows/cnli-join-low

**cnli_join_low**: `cnli_join_low` (join), 24 x 8 = 192 pairs, 21% positive; GPU NVIDIA L40S, 46068 MiB, max_num_seqs 256, git `2db189318`.

| Arm | Latency (s) | Speedup | Calls | Fallbacks | Waves | Peak running | Peak waiting | Computed prompt tok | Gen tok/call | Decode/call (s) | s/gen tok | Hit rate | Peak KV % | Preempt | Recall | F1 | ΔF1 | Agree | Self | Sim | Selected |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 5.65 | 1.00 | 192 | – | 1 | 17 | 151 | 60754 | 2.5 | 0.23 | 0.092 | 0.82 | 1.3 | 0 | 0.28 | 0.38 | 0.00 | 1.00 | 1.00 | – | 18 |
| `kalypso_block:2` | 18.57 | 0.30 | 96 | 0 | 1 | 12 | 91 | 200469 | 8.0 | 1.13 | 0.142 | 0.38 | 4.3 | 0 | 0.28 | 0.39 | 0.01 | 0.90 | 1.00 | – | 17 |
| `kalypso_block:4` | 26.39 | 0.21 | 48 | 0 | 1 | 9 | 46 | 275603 | 16.0 | 2.53 | 0.158 | 0.13 | 9.2 | 0 | 0.25 | 0.32 | -0.06 | 0.88 | 1.00 | – | 22 |
| `kalypso_block:8` | 33.30 | 0.17 | 24 | 0 | 1 | 7 | 23 | 313095 | 32.0 | 5.57 | 0.174 | 0.00 | 18.6 | 0 | 0.17 | 0.24 | -0.14 | 0.89 | 1.00 | – | 18 |

Best: `kalypso_per_tuple` (1.00x).

## map

### map/map-n100

**movie_map-n100**: `movie_map` (map), 100 rows; GPU NVIDIA L40S, 46068 MiB, max_num_seqs 256, git `2db189318`.

| Arm | Latency (s) | Speedup | Calls | Fallbacks | Waves | Peak running | Peak waiting | Computed prompt tok | Gen tok/call | Decode/call (s) | s/gen tok | Hit rate | Peak KV % | Preempt | Recall | F1 | ΔF1 | Agree | Self | Sim | Selected |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 2.65 | 1.00 | 100 | – | 1 | 92 | 43 | 11077 | 21.0 | 0.85 | 0.040 | 0.30 | 2.6 | 0 | – | – | – | 1.00 | 1.00 | 1.00 | – |
| `kalypso_block:2` | 3.40 | 0.78 | 50 | 0 | 1 | 50 | 29 | 6839 | 67.1 | 2.02 | 0.030 | 0.31 | 2.1 | 0 | – | – | – | 0.00 | 1.00 | 0.39 | – |
| `kalypso_block:4` | 5.09 | 0.52 | 25 | 0 | 1 | 25 | 14 | 5514 | 118.5 | 3.20 | 0.027 | 0.22 | 1.7 | 0 | – | – | – | 0.00 | 1.00 | 0.34 | – |
| `kalypso_block:8` | 7.85 | 0.34 | 15 | 0 | 1 | 13 | 6 | 5144 | 192.6 | 4.55 | 0.024 | 0.14 | 1.5 | 0 | – | – | – | 0.01 | 1.00 | 0.34 | – |

Best: `kalypso_per_tuple` (1.00x).

### map/map-n500

**movie_map-n500**: `movie_map` (map), 500 rows; GPU NVIDIA L40S, 46068 MiB, max_num_seqs 256, git `2db189318`.

| Arm | Latency (s) | Speedup | Calls | Fallbacks | Waves | Peak running | Peak waiting | Computed prompt tok | Gen tok/call | Decode/call (s) | s/gen tok | Hit rate | Peak KV % | Preempt | Recall | F1 | ΔF1 | Agree | Self | Sim | Selected |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 7.36 | 1.00 | 500 | – | 2 | 249 | 443 | 55733 | 19.8 | 2.04 | 0.103 | 0.30 | 7.2 | 0 | – | – | – | 1.00 | 1.00 | 1.00 | – |
| `kalypso_block:2` | 6.43 | 1.14 | 250 | 0 | 1 | 250 | 204 | 34515 | 68.5 | 3.42 | 0.050 | 0.32 | 10.1 | 0 | – | – | – | 0.00 | 1.00 | 0.37 | – |
| `kalypso_block:4` | 7.93 | 0.93 | 125 | 0 | 1 | 125 | 96 | 27889 | 125.2 | 4.64 | 0.037 | 0.22 | 8.7 | 0 | – | – | – | 0.00 | 1.00 | 0.35 | – |
| `kalypso_block:6` | 9.47 | 0.78 | 87 | 0 | 1 | 84 | 62 | 26137 | 169.1 | 5.72 | 0.034 | 0.17 | 8.2 | 0 | – | – | – | 0.01 | 1.00 | 0.34 | – |
| `kalypso_block:8` | 10.74 | 0.69 | 65 | 0 | 1 | 63 | 56 | 24906 | 211.0 | 6.80 | 0.032 | 0.14 | 7.8 | 0 | – | – | – | 0.00 | 1.00 | 0.32 | – |
| `kalypso_block:16` | 19.80 | 0.37 | 51 | 0 | 1 | 32 | 28 | 24914 | 266.7 | 7.66 | 0.029 | 0.11 | 7.4 | 0 | – | – | – | 0.03 | 1.00 | 0.33 | – |

Best: `kalypso_per_tuple` (1.00x).

## pipeline

### pipeline/filter-join-s10

**alone-filter**: `movie_sel10` (filter), 100 rows, 7% positive; GPU NVIDIA L40S, 49140 MiB, max_num_seqs 256, git `2db189318`.

| Arm | Latency (s) | Speedup | Calls | Fallbacks | Waves | Peak running | Peak waiting | Computed prompt tok | Gen tok/call | Decode/call (s) | s/gen tok | Hit rate | Peak KV % | Preempt | Recall | F1 | ΔF1 | Agree | Self | Sim | Selected |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 0.95 | 1.00 | 100 | – | 1 | 36 | 45 | 11412 | 3.0 | 0.21 | 0.071 | 0.29 | 0.9 | 0 | 0.29 | 0.44 | 0.00 | 1.00 | 1.00 | – | 2 |
| `kalypso_block:4` | 0.73 | 1.31 | 25 | 0 | 1 | 25 | 12 | 4472 | 16.0 | 0.40 | 0.025 | 0.15 | 1.0 | 0 | 0.71 | 0.63 | 0.18 | 0.93 | 1.00 | – | 9 |
| `kalypso_block:6` | 0.85 | 1.13 | 17 | 0 | 1 | 17 | 7 | 4064 | 23.5 | 0.54 | 0.023 | 0.11 | 0.9 | 0 | 1.00 | 0.74 | 0.29 | 0.90 | 1.00 | – | 12 |
| `kalypso_block:11` | 1.21 | 0.79 | 10 | 1 | 1 | 9 | 3 | 3723 | 41.6 | 0.85 | 0.021 | 0.07 | 0.8 | 0 | 0.71 | 0.71 | 0.27 | 0.95 | 1.00 | – | 7 |

Best: `kalypso_block:4` (1.31x).

**alone-join**: `pipe_filter_join_s10_join` (join), 7 x 32 = 224 pairs, 47% positive; GPU NVIDIA L40S, 49140 MiB, max_num_seqs 256, git `2db189318`.

| Arm | Latency (s) | Speedup | Calls | Fallbacks | Waves | Peak running | Peak waiting | Computed prompt tok | Gen tok/call | Decode/call (s) | s/gen tok | Hit rate | Peak KV % | Preempt | Recall | F1 | ΔF1 | Agree | Self | Sim | Selected |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 2.36 | 1.00 | 224 | – | 1 | 23 | 176 | 30104 | 2.3 | 0.17 | 0.074 | 0.41 | 0.7 | 0 | 0.86 | 0.85 | 0.00 | 1.00 | 1.00 | – | 107 |
| `kalypso_block:2` | 1.74 | 1.35 | 112 | 0 | 1 | 82 | 74 | 18875 | 8.0 | 0.66 | 0.082 | 0.32 | 3.0 | 0 | 0.74 | 0.67 | -0.18 | 0.67 | 1.00 | – | 129 |
| `kalypso_block:4` | 1.86 | 1.26 | 56 | 0 | 1 | 49 | 48 | 18491 | 16.0 | 0.86 | 0.053 | 0.19 | 3.5 | 0 | 0.90 | 0.71 | -0.14 | 0.67 | 1.00 | – | 160 |
| `kalypso_block:6` | 2.02 | 1.17 | 38 | 0 | 1 | 37 | 32 | 18376 | 23.6 | 1.01 | 0.043 | 0.13 | 3.8 | 0 | 0.80 | 0.67 | -0.17 | 0.67 | 1.00 | – | 144 |

Best: `kalypso_per_tuple` (1.00x).

**pipe_filter_join_s10**: `pipe_filter_join_s10` (join), 100 x 32 = 3200 pairs, 3% positive, stages filter>join; GPU NVIDIA L40S, 46068 MiB, max_num_seqs 256, git `51fee093a`.

| Arm | Latency (s) | Speedup | Calls | Fallbacks | Waves | Peak running | Peak waiting | Computed prompt tok | Gen tok/call | Decode/call (s) | s/gen tok | Hit rate | Peak KV % | Preempt | Recall | F1 | ΔF1 | Agree | Self | Sim | Selected |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 1.79 | 1.00 | 164 | – | 1 | 36 | 45 | 19873 | 2.7 | 0.19 | 0.070 | 0.34 | 1.0 | 0 | 0.25 | 0.39 | 0.00 | 1.00 | 1.00 | – | 30 |
| `kalypso_pipe:4,4` | 3.02 | 0.59 | 89 | 0 | 1 | 63 | 46 | 25448 | 16.0 | 0.85 | 0.053 | 0.18 | 4.8 | 0 | 0.67 | 0.49 | 0.11 | 0.95 | 1.00 | – | 178 |
| `kalypso_pipe:6,4` | 4.05 | 0.44 | 113 | 0 | 1 | 79 | 68 | 35486 | 17.1 | 1.13 | 0.066 | 0.18 | 6.1 | 0 | 0.88 | 0.51 | 0.12 | 0.93 | 1.00 | – | 257 |
| `kalypso_pipe:11,4` | 3.31 | 0.54 | 66 | 1 | 1 | 56 | 47 | 21222 | 19.9 | 0.89 | 0.045 | 0.17 | 4.1 | 0 | 0.68 | 0.53 | 0.14 | 0.96 | 1.00 | – | 165 |
| `kalypso_pipe:6,2` | 3.93 | 0.46 | 209 | 0 | 1 | 86 | 139 | 36271 | 9.3 | 0.78 | 0.085 | 0.30 | 3.4 | 0 | 0.73 | 0.49 | 0.10 | 0.94 | 1.00 | – | 214 |
| `kalypso_pipe:6,6` | 4.18 | 0.43 | 81 | 0 | 1 | 59 | 47 | 35215 | 23.9 | 1.34 | 0.056 | 0.13 | 6.7 | 0 | 0.82 | 0.49 | 0.11 | 0.93 | 0.98 | – | 238 |
| `kalypso_pipe:1,4` | 1.82 | 0.98 | 116 | 0 | 1 | 36 | 45 | 16412 | 4.8 | 0.25 | 0.053 | 0.26 | 1.2 | 0 | 0.28 | 0.38 | -0.01 | 0.99 | 1.00 | – | 48 |
| `kalypso_pipe:4,1` | 3.64 | 0.49 | 281 | 0 | 2 | 25 | 210 | 39141 | 3.5 | 0.21 | 0.060 | 0.38 | 1.1 | 0 | 0.62 | 0.55 | 0.17 | 0.97 | 1.00 | – | 130 |

Best: `kalypso_per_tuple` (1.00x).

Composition (same container):

| Pipeline arm | Pipeline (s) | filter alone (s) | join alone (s) | Sum (s) | Max (s) | Pipeline / sum | Pipeline / max |
|---|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 1.79 | 0.95 | 2.36 | 3.31 | 2.36 | 0.54 | 0.76 |
| `kalypso_pipe:4,4` | 3.02 | 0.73 | 1.86 | 2.59 | 1.86 | 1.17 | 1.62 |
| `kalypso_pipe:6,4` | 4.05 | 0.85 | 1.86 | 2.71 | 1.86 | 1.49 | 2.17 |
| `kalypso_pipe:11,4` | 3.31 | 1.21 | 1.86 | 3.08 | 1.86 | 1.07 | 1.77 |
| `kalypso_pipe:6,2` | 3.93 | 0.85 | 1.74 | 2.59 | 1.74 | 1.52 | 2.25 |
| `kalypso_pipe:6,6` | 4.18 | 0.85 | 2.02 | 2.87 | 2.02 | 1.46 | 2.07 |
| `kalypso_pipe:1,4` | 1.82 | 0.95 | 1.86 | 2.82 | 1.86 | 0.65 | 0.98 |
| `kalypso_pipe:4,1` | 3.64 | 0.73 | 2.36 | 3.08 | 2.36 | 1.18 | 1.55 |

### pipeline/filter-join-s50

**alone-filter**: `movie_sel50` (filter), 100 rows, 48% positive; GPU NVIDIA L40S, 49140 MiB, max_num_seqs 256, git `2db189318`.

| Arm | Latency (s) | Speedup | Calls | Fallbacks | Waves | Peak running | Peak waiting | Computed prompt tok | Gen tok/call | Decode/call (s) | s/gen tok | Hit rate | Peak KV % | Preempt | Recall | F1 | ΔF1 | Agree | Self | Sim | Selected |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 0.96 | 1.00 | 100 | – | 1 | 36 | 47 | 11641 | 3.0 | 0.21 | 0.070 | 0.29 | 0.9 | 0 | 0.58 | 0.74 | 0.00 | 1.00 | 1.00 | – | 28 |
| `kalypso_block:4` | 0.74 | 1.29 | 25 | 0 | 1 | 25 | 13 | 4701 | 16.0 | 0.40 | 0.025 | 0.14 | 1.0 | 0 | 0.75 | 0.86 | 0.12 | 0.92 | 1.00 | – | 36 |
| `kalypso_block:6` | 0.86 | 1.11 | 17 | 0 | 1 | 17 | 8 | 4293 | 23.5 | 0.54 | 0.023 | 0.11 | 0.9 | 0 | 0.77 | 0.87 | 0.13 | 0.91 | 1.00 | – | 37 |
| `kalypso_block:11` | 1.24 | 0.78 | 10 | 1 | 1 | 9 | 4 | 3952 | 41.6 | 0.85 | 0.021 | 0.06 | 0.9 | 0 | 0.75 | 0.86 | 0.12 | 0.92 | 1.00 | – | 36 |

Best: `kalypso_block:4` (1.29x).

**alone-join**: `pipe_filter_join_s50_join` (join), 48 x 32 = 1536 pairs, 47% positive; GPU NVIDIA L40S, 49140 MiB, max_num_seqs 256, git `2db189318`.

| Arm | Latency (s) | Speedup | Calls | Fallbacks | Waves | Peak running | Peak waiting | Computed prompt tok | Gen tok/call | Decode/call (s) | s/gen tok | Hit rate | Peak KV % | Preempt | Recall | F1 | ΔF1 | Agree | Self | Sim | Selected |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 16.37 | 1.00 | 1536 | – | 6 | 25 | 1414 | 208403 | 2.3 | 0.19 | 0.082 | 0.40 | 0.7 | 0 | 0.86 | 0.84 | 0.00 | 1.00 | 1.00 | – | 749 |
| `kalypso_block:2` | 11.09 | 1.48 | 768 | 0 | 3 | 90 | 704 | 129328 | 8.0 | 0.99 | 0.124 | 0.32 | 3.0 | 0 | 0.69 | 0.64 | -0.21 | 0.67 | 1.00 | – | 828 |
| `kalypso_block:4` | 11.02 | 1.49 | 384 | 0 | 2 | 94 | 363 | 126023 | 16.0 | 2.05 | 0.128 | 0.19 | 6.3 | 0 | 0.79 | 0.69 | -0.16 | 0.70 | 1.00 | – | 932 |
| `kalypso_block:6` | 11.22 | 1.46 | 256 | 0 | 1 | 96 | 241 | 124892 | 24.0 | 3.02 | 0.126 | 0.13 | 9.7 | 0 | 0.71 | 0.67 | -0.18 | 0.66 | 1.00 | – | 824 |

Best: `kalypso_per_tuple` (1.00x).

**pipe_filter_join_s50**: `pipe_filter_join_s50` (join), 100 x 32 = 3200 pairs, 22% positive, stages filter>join; GPU NVIDIA L40S, 46068 MiB, max_num_seqs 256, git `51fee093a`.

| Arm | Latency (s) | Speedup | Calls | Fallbacks | Waves | Peak running | Peak waiting | Computed prompt tok | Gen tok/call | Decode/call (s) | s/gen tok | Hit rate | Peak KV % | Preempt | Recall | F1 | ΔF1 | Agree | Self | Sim | Selected |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 10.64 | 1.00 | 996 | – | 4 | 36 | 772 | 132074 | 2.3 | 0.21 | 0.089 | 0.40 | 1.0 | 0 | 0.51 | 0.63 | 0.00 | 1.00 | 1.00 | – | 445 |
| `kalypso_pipe:4,4` | 10.12 | 1.05 | 313 | 0 | 2 | 94 | 261 | 100410 | 16.0 | 2.02 | 0.127 | 0.19 | 7.0 | 0 | 0.61 | 0.62 | -0.01 | 0.87 | 1.00 | – | 705 |
| `kalypso_pipe:6,4` | 10.30 | 1.03 | 313 | 0 | 2 | 95 | 267 | 102196 | 16.4 | 2.03 | 0.124 | 0.19 | 7.0 | 0 | 0.63 | 0.63 | -0.00 | 0.87 | 1.00 | – | 719 |
| `kalypso_pipe:11,4` | 10.68 | 1.00 | 298 | 1 | 2 | 95 | 253 | 99861 | 16.9 | 2.08 | 0.123 | 0.19 | 7.0 | 0 | 0.61 | 0.62 | -0.01 | 0.87 | 1.00 | – | 705 |
| `kalypso_pipe:6,2` | 10.42 | 1.02 | 609 | 0 | 3 | 88 | 506 | 104499 | 8.4 | 1.03 | 0.122 | 0.31 | 3.4 | 0 | 0.56 | 0.58 | -0.05 | 0.87 | 1.00 | – | 667 |
| `kalypso_pipe:6,6` | 10.59 | 1.00 | 215 | 0 | 1 | 96 | 173 | 101387 | 23.9 | 2.85 | 0.119 | 0.13 | 10.8 | 0 | 0.55 | 0.59 | -0.04 | 0.87 | 0.91 | – | 624 |
| `kalypso_pipe:1,4` | 7.78 | 1.37 | 324 | 0 | 2 | 93 | 194 | 86189 | 12.0 | 1.52 | 0.127 | 0.20 | 7.1 | 0 | 0.47 | 0.54 | -0.09 | 0.93 | 1.00 | – | 532 |
| `kalypso_pipe:4,1` | 14.54 | 0.73 | 1177 | 0 | 5 | 25 | 1005 | 161181 | 2.5 | 0.21 | 0.082 | 0.40 | 1.1 | 0 | 0.65 | 0.73 | 0.10 | 0.96 | 1.00 | – | 561 |

Best: `kalypso_pipe:4,4` (1.05x).

Composition (same container):

| Pipeline arm | Pipeline (s) | filter alone (s) | join alone (s) | Sum (s) | Max (s) | Pipeline / sum | Pipeline / max |
|---|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 10.64 | 0.96 | 16.37 | 17.33 | 16.37 | 0.61 | 0.65 |
| `kalypso_pipe:4,4` | 10.12 | 0.74 | 11.02 | 11.76 | 11.02 | 0.86 | 0.92 |
| `kalypso_pipe:6,4` | 10.30 | 0.86 | 11.02 | 11.88 | 11.02 | 0.87 | 0.93 |
| `kalypso_pipe:11,4` | 10.68 | 1.24 | 11.02 | 12.25 | 11.02 | 0.87 | 0.97 |
| `kalypso_pipe:6,2` | 10.42 | 0.86 | 11.09 | 11.95 | 11.09 | 0.87 | 0.94 |
| `kalypso_pipe:6,6` | 10.59 | 0.86 | 11.22 | 12.08 | 11.22 | 0.88 | 0.94 |
| `kalypso_pipe:1,4` | 7.78 | 0.96 | 11.02 | 11.98 | 11.02 | 0.65 | 0.71 |
| `kalypso_pipe:4,1` | 14.54 | 0.74 | 16.37 | 17.11 | 16.37 | 0.85 | 0.89 |

### pipeline/filter-join-s90

**alone-filter**: `movie_sel90` (filter), 100 rows, 92% positive; GPU NVIDIA L40S, 46068 MiB, max_num_seqs 256, git `2db189318`.

| Arm | Latency (s) | Speedup | Calls | Fallbacks | Waves | Peak running | Peak waiting | Computed prompt tok | Gen tok/call | Decode/call (s) | s/gen tok | Hit rate | Peak KV % | Preempt | Recall | F1 | ΔF1 | Agree | Self | Sim | Selected |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 1.01 | 1.00 | 100 | – | 1 | 35 | 46 | 11616 | 2.9 | 0.21 | 0.073 | 0.29 | 1.0 | 0 | 0.63 | 0.77 | 0.00 | 1.00 | 1.00 | – | 58 |
| `kalypso_block:4` | 0.82 | 1.23 | 25 | 0 | 1 | 25 | 12 | 4674 | 16.0 | 0.46 | 0.029 | 0.14 | 1.1 | 0 | 0.84 | 0.91 | 0.14 | 0.81 | 1.00 | – | 77 |
| `kalypso_block:6` | 0.97 | 1.04 | 17 | 0 | 1 | 17 | 7 | 4266 | 23.5 | 0.63 | 0.027 | 0.11 | 1.0 | 0 | 0.83 | 0.90 | 0.13 | 0.80 | 1.00 | – | 76 |
| `kalypso_block:11` | 1.38 | 0.73 | 10 | 1 | 1 | 9 | 3 | 3925 | 41.6 | 0.98 | 0.024 | 0.06 | 0.9 | 0 | 0.80 | 0.89 | 0.12 | 0.82 | 1.00 | – | 74 |

Best: `kalypso_block:4` (1.23x).

**alone-join**: `pipe_filter_join_s90_join` (join), 92 x 32 = 2944 pairs, 47% positive; GPU NVIDIA L40S, 46068 MiB, max_num_seqs 256, git `2db189318`.

| Arm | Latency (s) | Speedup | Calls | Fallbacks | Waves | Peak running | Peak waiting | Computed prompt tok | Gen tok/call | Decode/call (s) | s/gen tok | Hit rate | Peak KV % | Preempt | Recall | F1 | ΔF1 | Agree | Self | Sim | Selected |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 31.30 | 1.00 | 2944 | – | 12 | 24 | 1890 | 397666 | 2.3 | 0.19 | 0.082 | 0.40 | 0.8 | 0 | 0.86 | 0.84 | 0.00 | 1.00 | 1.00 | – | 1432 |
| `kalypso_block:2` | 21.55 | 1.45 | 1472 | 0 | 6 | 88 | 1373 | 247895 | 8.0 | 1.03 | 0.129 | 0.32 | 3.4 | 0 | 0.70 | 0.63 | -0.21 | 0.66 | 0.99 | – | 1675 |
| `kalypso_block:4` | 21.06 | 1.49 | 736 | 0 | 3 | 96 | 702 | 241277 | 16.0 | 2.19 | 0.137 | 0.19 | 7.0 | 0 | 0.82 | 0.71 | -0.14 | 0.72 | 1.00 | – | 1832 |
| `kalypso_block:6` | 21.34 | 1.47 | 491 | 0 | 2 | 96 | 467 | 238971 | 24.0 | 3.34 | 0.139 | 0.13 | 10.8 | 0 | 0.74 | 0.69 | -0.16 | 0.67 | 1.00 | – | 1609 |

Best: `kalypso_per_tuple` (1.00x).

**pipe_filter_join_s90**: `pipe_filter_join_s90` (join), 100 x 32 = 3200 pairs, 43% positive, stages filter>join; GPU NVIDIA L40S, 46068 MiB, max_num_seqs 256, git `51fee093a`.

| Arm | Latency (s) | Speedup | Calls | Fallbacks | Waves | Peak running | Peak waiting | Computed prompt tok | Gen tok/call | Decode/call (s) | s/gen tok | Hit rate | Peak KV % | Preempt | Recall | F1 | ΔF1 | Agree | Self | Sim | Selected |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 20.19 | 1.00 | 1956 | – | 8 | 34 | 1590 | 259181 | 2.3 | 0.20 | 0.086 | 0.41 | 0.9 | 0 | 0.55 | 0.66 | 0.00 | 1.00 | 1.00 | – | 923 |
| `kalypso_pipe:4,4` | 19.12 | 1.06 | 641 | 0 | 3 | 96 | 589 | 206145 | 16.0 | 2.18 | 0.136 | 0.19 | 7.1 | 0 | 0.69 | 0.66 | -0.00 | 0.72 | 0.98 | – | 1512 |
| `kalypso_pipe:6,4` | 17.62 | 1.15 | 625 | 0 | 3 | 96 | 519 | 202456 | 16.2 | 2.22 | 0.137 | 0.19 | 7.0 | 0 | 0.68 | 0.65 | -0.00 | 0.73 | 1.00 | – | 1487 |
| `kalypso_pipe:11,4` | 18.87 | 1.07 | 602 | 1 | 3 | 96 | 557 | 196760 | 16.4 | 2.21 | 0.134 | 0.19 | 7.1 | 0 | 0.67 | 0.65 | -0.01 | 0.73 | 1.00 | – | 1472 |
| `kalypso_pipe:6,2` | 17.63 | 1.15 | 1233 | 0 | 5 | 89 | 1016 | 208361 | 8.2 | 1.05 | 0.128 | 0.31 | 3.4 | 0 | 0.60 | 0.59 | -0.07 | 0.70 | 0.99 | – | 1430 |
| `kalypso_pipe:6,6` | 17.93 | 1.13 | 423 | 0 | 2 | 99 | 349 | 200455 | 23.9 | 3.29 | 0.137 | 0.13 | 10.8 | 0 | 0.61 | 0.63 | -0.03 | 0.71 | 0.82 | – | 1275 |
| `kalypso_pipe:1,4` | 14.08 | 1.43 | 564 | 0 | 3 | 96 | 416 | 163805 | 13.7 | 1.89 | 0.138 | 0.20 | 7.1 | 0 | 0.52 | 0.57 | -0.08 | 0.85 | 1.00 | – | 1107 |
| `kalypso_pipe:4,1` | 28.60 | 0.71 | 2489 | 0 | 10 | 25 | 1887 | 336924 | 2.4 | 0.20 | 0.083 | 0.40 | 1.1 | 0 | 0.72 | 0.77 | 0.11 | 0.91 | 0.99 | – | 1207 |

Best: `kalypso_pipe:6,4` (1.15x).

Composition (same container):

| Pipeline arm | Pipeline (s) | filter alone (s) | join alone (s) | Sum (s) | Max (s) | Pipeline / sum | Pipeline / max |
|---|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 20.19 | 1.01 | 31.30 | 32.31 | 31.30 | 0.62 | 0.65 |
| `kalypso_pipe:4,4` | 19.12 | 0.82 | 21.06 | 21.88 | 21.06 | 0.87 | 0.91 |
| `kalypso_pipe:6,4` | 17.62 | 0.97 | 21.06 | 22.02 | 21.06 | 0.80 | 0.84 |
| `kalypso_pipe:11,4` | 18.87 | 1.38 | 21.06 | 22.43 | 21.06 | 0.84 | 0.90 |
| `kalypso_pipe:6,2` | 17.63 | 0.97 | 21.55 | 22.52 | 21.55 | 0.78 | 0.82 |
| `kalypso_pipe:6,6` | 17.93 | 0.97 | 21.34 | 22.31 | 21.34 | 0.80 | 0.84 |
| `kalypso_pipe:1,4` | 14.08 | 1.01 | 21.06 | 22.07 | 21.06 | 0.64 | 0.67 |
| `kalypso_pipe:4,1` | 28.60 | 0.82 | 31.30 | 32.12 | 31.30 | 0.89 | 0.91 |

### pipeline/filter-map

**alone-filter**: `movie_positive` (filter), 500 rows, 75% positive; GPU NVIDIA L40S, 46068 MiB, max_num_seqs 256, git `2db189318`.

| Arm | Latency (s) | Speedup | Calls | Fallbacks | Waves | Peak running | Peak waiting | Computed prompt tok | Gen tok/call | Decode/call (s) | s/gen tok | Hit rate | Peak KV % | Preempt | Recall | F1 | ΔF1 | Agree | Self | Sim | Selected |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 4.67 | 1.00 | 500 | – | 2 | 37 | 445 | 57728 | 3.0 | 0.27 | 0.092 | 0.29 | 1.0 | 0 | 0.63 | 0.77 | 0.00 | 1.00 | 1.00 | – | 235 |
| `kalypso_block:4` | 2.43 | 1.92 | 125 | 0 | 1 | 125 | 89 | 23065 | 16.0 | 1.08 | 0.068 | 0.15 | 5.6 | 0 | 0.77 | 0.87 | 0.10 | 0.85 | 1.00 | – | 291 |
| `kalypso_block:6` | 2.48 | 1.88 | 84 | 0 | 1 | 83 | 58 | 20991 | 23.8 | 1.23 | 0.052 | 0.11 | 5.0 | 0 | 0.77 | 0.87 | 0.10 | 0.85 | 1.00 | – | 290 |
| `kalypso_block:11` | 2.90 | 1.61 | 46 | 0 | 1 | 46 | 40 | 19248 | 45.4 | 1.76 | 0.039 | 0.07 | 4.7 | 0 | 0.73 | 0.84 | 0.07 | 0.87 | 1.00 | – | 280 |

Best: `kalypso_block:4` (1.92x).

**alone-map**: `pipe_filter_map_map` (map), 373 rows; GPU NVIDIA L40S, 46068 MiB, max_num_seqs 256, git `2db189318`.

| Arm | Latency (s) | Speedup | Calls | Fallbacks | Waves | Peak running | Peak waiting | Computed prompt tok | Gen tok/call | Decode/call (s) | s/gen tok | Hit rate | Peak KV % | Preempt | Recall | F1 | ΔF1 | Agree | Self | Sim | Selected |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 5.29 | 1.00 | 373 | – | 2 | 252 | 317 | 41559 | 19.8 | 1.80 | 0.091 | 0.30 | 7.2 | 0 | – | – | – | 1.00 | 1.00 | 1.00 | – |

Best: `kalypso_per_tuple` (1.00x).

**pipe_filter_map**: `pipe_filter_map` (map), 500 rows, 75% positive, stages filter>map; GPU NVIDIA L40S, 46068 MiB, max_num_seqs 256, git `51fee093a`.

| Arm | Latency (s) | Speedup | Calls | Fallbacks | Waves | Peak running | Peak waiting | Computed prompt tok | Gen tok/call | Decode/call (s) | s/gen tok | Hit rate | Peak KV % | Preempt | Recall | F1 | ΔF1 | Agree | Self | Sim | Selected |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 7.71 | 1.00 | 735 | – | 3 | 222 | 445 | 74834 | 8.2 | 0.56 | 0.069 | 0.37 | 6.2 | 0 | 0.63 | 0.77 | 0.00 | 1.00 | 1.00 | 1.00 | 235 |
| `kalypso_pipe:4,1` | 7.29 | 1.06 | 459 | 0 | 2 | 239 | 152 | 55937 | 18.9 | 1.47 | 0.078 | 0.30 | 6.9 | 0 | 0.77 | 0.87 | 0.10 | 0.60 | 0.92 | 0.92 | 291 |
| `kalypso_pipe:6,1` | 6.85 | 1.12 | 380 | 0 | 2 | 239 | 116 | 53644 | 20.9 | 1.51 | 0.072 | 0.24 | 7.3 | 0 | 0.77 | 0.87 | 0.10 | 0.59 | 0.91 | 0.92 | 292 |
| `kalypso_pipe:11,1` | 6.90 | 1.12 | 325 | 0 | 2 | 240 | 98 | 50325 | 23.5 | 1.61 | 0.069 | 0.23 | 6.9 | 0 | 0.73 | 0.84 | 0.07 | 0.63 | 0.93 | 0.92 | 279 |

Best: `kalypso_pipe:6,1` (1.12x).

Composition (same container):

| Pipeline arm | Pipeline (s) | filter alone (s) | map alone (s) | Sum (s) | Max (s) | Pipeline / sum | Pipeline / max |
|---|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 7.71 | 4.67 | 5.29 | 9.96 | 5.29 | 0.77 | 1.46 |
| `kalypso_pipe:4,1` | 7.29 | 2.43 | 5.29 | 7.73 | 5.29 | 0.94 | 1.38 |
| `kalypso_pipe:6,1` | 6.85 | 2.48 | 5.29 | 7.77 | 5.29 | 0.88 | 1.29 |
| `kalypso_pipe:11,1` | 6.90 | 2.90 | 5.29 | 8.19 | 5.29 | 0.84 | 1.30 |

### pipeline/filter-map-blockmap

**alone-map**: `pipe_filter_map_map` (map), 373 rows; GPU NVIDIA L40S, 46068 MiB, max_num_seqs 256, git `2db189318`.

| Arm | Latency (s) | Speedup | Calls | Fallbacks | Waves | Peak running | Peak waiting | Computed prompt tok | Gen tok/call | Decode/call (s) | s/gen tok | Hit rate | Peak KV % | Preempt | Recall | F1 | ΔF1 | Agree | Self | Sim | Selected |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 5.24 | 1.00 | 373 | – | 2 | 251 | 317 | 41559 | 19.8 | 1.85 | 0.093 | 0.30 | 7.2 | 0 | – | – | – | 1.00 | 1.00 | 1.00 | – |
| `kalypso_block:4` | 6.79 | 0.77 | 94 | 0 | 1 | 94 | 64 | 20895 | 124.7 | 4.26 | 0.034 | 0.22 | 6.5 | 0 | – | – | – | 0.00 | 1.00 | 0.38 | – |
| `kalypso_block:8` | 13.32 | 0.39 | 55 | 0 | 1 | 47 | 40 | 19265 | 203.5 | 6.08 | 0.030 | 0.15 | 5.8 | 0 | – | – | – | 0.02 | 1.00 | 0.36 | – |

Best: `kalypso_per_tuple` (1.00x).

**pipe_filter_map**: `pipe_filter_map` (map), 500 rows, 75% positive, stages filter>map; GPU NVIDIA L40S, 46068 MiB, max_num_seqs 256, git `51fee093a`.

| Arm | Latency (s) | Speedup | Calls | Fallbacks | Waves | Peak running | Peak waiting | Computed prompt tok | Gen tok/call | Decode/call (s) | s/gen tok | Hit rate | Peak KV % | Preempt | Recall | F1 | ΔF1 | Agree | Self | Sim | Selected |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 7.90 | 1.00 | 735 | – | 3 | 226 | 445 | 74834 | 8.2 | 0.57 | 0.069 | 0.37 | 6.3 | 0 | 0.63 | 0.77 | 0.00 | 1.00 | 1.00 | 1.00 | 235 |
| `kalypso_pipe:1,4` | 10.88 | 0.73 | 559 | 0 | 3 | 59 | 395 | 70861 | 16.3 | 0.70 | 0.043 | 0.28 | 4.1 | 0 | 0.63 | 0.77 | 0.00 | 0.01 | 0.25 | 0.38 | 235 |
| `kalypso_pipe:6,4` | 8.94 | 0.88 | 157 | 0 | 1 | 74 | 47 | 37344 | 72.5 | 2.71 | 0.037 | 0.16 | 5.0 | 0 | 0.77 | 0.87 | 0.10 | 0.00 | 0.19 | 0.38 | 292 |
| `kalypso_pipe:6,8` | 12.71 | 0.62 | 121 | 0 | 1 | 68 | 47 | 35457 | 87.7 | 2.95 | 0.034 | 0.12 | 4.6 | 0 | 0.77 | 0.87 | 0.10 | 0.01 | 0.04 | 0.37 | 292 |

Best: `kalypso_per_tuple` (1.00x).

## single

### single/filter-n100

**movie_positive-n100**: `movie_positive` (filter), 100 rows, 72% positive; GPU NVIDIA L40S, 46068 MiB, max_num_seqs 256, git `2db189318`.

| Arm | Latency (s) | Speedup | Calls | Fallbacks | Waves | Peak running | Peak waiting | Computed prompt tok | Gen tok/call | Decode/call (s) | s/gen tok | Hit rate | Peak KV % | Preempt | Recall | F1 | ΔF1 | Agree | Self | Sim | Selected |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 0.98 | 1.00 | 100 | – | 1 | 36 | 45 | 11477 | 3.0 | 0.21 | 0.072 | 0.29 | 1.0 | 0 | 0.61 | 0.76 | 0.00 | 1.00 | 1.00 | – | 44 |
| `kalypso_block:4` | 0.80 | 1.23 | 25 | 0 | 1 | 25 | 12 | 4532 | 16.0 | 0.45 | 0.028 | 0.14 | 1.1 | 0 | 0.76 | 0.85 | 0.09 | 0.85 | 1.00 | – | 57 |
| `kalypso_block:6` | 0.94 | 1.04 | 17 | 0 | 1 | 17 | 7 | 4124 | 23.5 | 0.62 | 0.026 | 0.11 | 1.0 | 0 | 0.71 | 0.82 | 0.06 | 0.87 | 1.00 | – | 53 |
| `kalypso_block:8` | 1.05 | 0.93 | 13 | 0 | 1 | 13 | 5 | 3920 | 30.8 | 0.73 | 0.024 | 0.09 | 0.9 | 0 | 0.76 | 0.85 | 0.09 | 0.87 | 1.00 | – | 57 |

Best: `kalypso_block:4` (1.23x).

### single/filter-n1000

**movie_positive-n1000**: `movie_positive` (filter), 1000 rows, 75% positive; GPU NVIDIA L40S, 46068 MiB, max_num_seqs 256, git `2db189318`.

| Arm | Latency (s) | Speedup | Calls | Fallbacks | Waves | Peak running | Peak waiting | Computed prompt tok | Gen tok/call | Decode/call (s) | s/gen tok | Hit rate | Peak KV % | Preempt | Recall | F1 | ΔF1 | Agree | Self | Sim | Selected |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 9.36 | 1.00 | 1000 | – | 4 | 38 | 928 | 115130 | 3.0 | 0.29 | 0.099 | 0.29 | 1.0 | 0 | 0.63 | 0.77 | 0.00 | 1.00 | 1.00 | – | 479 |
| `kalypso_block:4` | 4.49 | 2.08 | 250 | 0 | 1 | 161 | 214 | 45798 | 16.0 | 1.73 | 0.108 | 0.15 | 7.1 | 0 | 0.79 | 0.88 | 0.10 | 0.84 | 1.00 | – | 595 |
| `kalypso_block:6` | 4.34 | 2.16 | 167 | 0 | 1 | 167 | 141 | 41627 | 23.9 | 2.02 | 0.084 | 0.11 | 10.0 | 0 | 0.80 | 0.88 | 0.11 | 0.84 | 1.00 | – | 606 |
| `kalypso_block:8` | 4.34 | 2.15 | 125 | 0 | 1 | 125 | 104 | 39520 | 32.0 | 2.16 | 0.067 | 0.09 | 9.5 | 0 | 0.79 | 0.87 | 0.10 | 0.85 | 1.00 | – | 601 |

Best: `kalypso_block:6` (2.16x).

### single/filter-n250

**movie_positive-n250**: `movie_positive` (filter), 250 rows, 74% positive; GPU NVIDIA L40S, 46068 MiB, max_num_seqs 256, git `2db189318`.

| Arm | Latency (s) | Speedup | Calls | Fallbacks | Waves | Peak running | Peak waiting | Computed prompt tok | Gen tok/call | Decode/call (s) | s/gen tok | Hit rate | Peak KV % | Preempt | Recall | F1 | ΔF1 | Agree | Self | Sim | Selected |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 2.34 | 1.00 | 250 | – | 1 | 37 | 195 | 28987 | 3.0 | 0.26 | 0.088 | 0.29 | 1.0 | 0 | 0.63 | 0.78 | 0.00 | 1.00 | 0.99 | – | 118 |
| `kalypso_block:4` | 1.38 | 1.69 | 63 | 0 | 1 | 63 | 47 | 11677 | 15.9 | 0.70 | 0.044 | 0.15 | 2.8 | 0 | 0.78 | 0.87 | 0.10 | 0.85 | 1.00 | – | 149 |
| `kalypso_block:6` | 1.49 | 1.56 | 42 | 0 | 1 | 42 | 32 | 10605 | 23.8 | 0.86 | 0.036 | 0.11 | 2.5 | 0 | 0.77 | 0.86 | 0.09 | 0.85 | 1.00 | – | 147 |
| `kalypso_block:8` | 1.63 | 1.43 | 32 | 0 | 1 | 32 | 24 | 10112 | 31.2 | 1.03 | 0.033 | 0.09 | 2.5 | 0 | 0.80 | 0.88 | 0.10 | 0.85 | 1.00 | – | 152 |

Best: `kalypso_block:4` (1.69x).

### single/filter-nall

**movie_positive**: `movie_positive` (filter), 1865 rows, 76% positive; GPU NVIDIA L40S, 46068 MiB, max_num_seqs 256, git `2db189318`.

| Arm | Latency (s) | Speedup | Calls | Fallbacks | Waves | Peak running | Peak waiting | Computed prompt tok | Gen tok/call | Decode/call (s) | s/gen tok | Hit rate | Peak KV % | Preempt | Recall | F1 | ΔF1 | Agree | Self | Sim | Selected |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 18.32 | 1.00 | 1865 | – | 8 | 37 | 1722 | 214980 | 3.0 | 0.31 | 0.104 | 0.29 | 0.9 | 0 | 0.64 | 0.78 | 0.00 | 1.00 | 1.00 | – | 913 |
| `kalypso_block:4` | 8.50 | 2.16 | 467 | 1 | 2 | 161 | 430 | 85825 | 16.0 | 2.13 | 0.133 | 0.15 | 6.7 | 0 | 0.79 | 0.88 | 0.10 | 0.84 | 1.00 | – | 1135 |
| `kalypso_block:6` | 8.09 | 2.27 | 311 | 0 | 2 | 181 | 285 | 77985 | 24.0 | 2.99 | 0.124 | 0.11 | 10.1 | 0 | 0.81 | 0.89 | 0.11 | 0.85 | 1.00 | – | 1159 |
| `kalypso_block:8` | 7.97 | 2.30 | 234 | 1 | 1 | 191 | 213 | 74071 | 31.9 | 3.54 | 0.111 | 0.09 | 13.5 | 0 | 0.79 | 0.88 | 0.10 | 0.85 | 1.00 | – | 1138 |

Best: `kalypso_block:8` (2.30x).

### single/join-fanout10

**movie_fanout10**: `movie_fanout10` (join), 32 x 32 = 1024 pairs, 9% positive; GPU NVIDIA L40S, 46068 MiB, max_num_seqs 256, git `2db189318`.

| Arm | Latency (s) | Speedup | Calls | Fallbacks | Waves | Peak running | Peak waiting | Computed prompt tok | Gen tok/call | Decode/call (s) | s/gen tok | Hit rate | Peak KV % | Preempt | Recall | F1 | ΔF1 | Agree | Self | Sim | Selected |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 11.05 | 1.00 | 1024 | – | 4 | 26 | 928 | 132090 | 2.2 | 0.19 | 0.084 | 0.42 | 0.8 | 0 | 1.00 | 0.50 | 0.00 | 1.00 | 1.00 | – | 286 |
| `kalypso_block:2` | 7.55 | 1.46 | 512 | 0 | 2 | 91 | 473 | 81891 | 8.0 | 1.03 | 0.129 | 0.33 | 3.4 | 0 | 0.66 | 0.25 | -0.25 | 0.58 | 1.00 | – | 400 |
| `kalypso_block:4` | 7.58 | 1.46 | 256 | 0 | 1 | 100 | 235 | 79499 | 16.0 | 2.05 | 0.128 | 0.20 | 7.1 | 0 | 0.75 | 0.29 | -0.21 | 0.70 | 1.00 | – | 397 |
| `kalypso_block:6` | 7.78 | 1.42 | 171 | 0 | 1 | 101 | 160 | 78687 | 23.9 | 2.89 | 0.121 | 0.14 | 10.8 | 0 | 0.51 | 0.24 | -0.26 | 0.66 | 1.00 | – | 309 |
| `kalypso_block:11` | 8.41 | 1.31 | 94 | 1 | 1 | 93 | 87 | 78360 | 45.5 | 4.03 | 0.088 | 0.08 | 18.1 | 0 | 0.17 | 0.19 | -0.32 | 0.73 | 1.00 | – | 75 |

Best: `kalypso_per_tuple` (1.00x).

### single/join-fanout25

**movie_fanout25**: `movie_fanout25` (join), 32 x 32 = 1024 pairs, 25% positive; GPU NVIDIA L40S, 46068 MiB, max_num_seqs 256, git `2db189318`.

| Arm | Latency (s) | Speedup | Calls | Fallbacks | Waves | Peak running | Peak waiting | Computed prompt tok | Gen tok/call | Decode/call (s) | s/gen tok | Hit rate | Peak KV % | Preempt | Recall | F1 | ΔF1 | Agree | Self | Sim | Selected |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 10.83 | 1.00 | 1024 | – | 4 | 25 | 928 | 131110 | 2.2 | 0.19 | 0.085 | 0.42 | 0.8 | 0 | 0.88 | 0.72 | 0.00 | 1.00 | 1.00 | – | 368 |
| `kalypso_block:2` | 7.41 | 1.46 | 512 | 0 | 2 | 91 | 473 | 80926 | 8.0 | 1.03 | 0.128 | 0.33 | 3.4 | 0 | 0.48 | 0.37 | -0.35 | 0.60 | 1.00 | – | 400 |
| `kalypso_block:4` | 7.40 | 1.46 | 256 | 0 | 1 | 101 | 235 | 78491 | 16.0 | 2.03 | 0.127 | 0.20 | 7.1 | 0 | 0.84 | 0.61 | -0.11 | 0.69 | 1.00 | – | 450 |
| `kalypso_block:6` | 7.51 | 1.44 | 171 | 0 | 1 | 102 | 156 | 77679 | 23.9 | 2.83 | 0.118 | 0.14 | 10.7 | 0 | 0.62 | 0.53 | -0.19 | 0.66 | 1.00 | – | 339 |
| `kalypso_block:11` | 8.11 | 1.33 | 94 | 1 | 1 | 93 | 87 | 77338 | 45.5 | 3.90 | 0.086 | 0.08 | 17.9 | 0 | 0.34 | 0.42 | -0.30 | 0.69 | 1.00 | – | 156 |

Best: `kalypso_per_tuple` (1.00x).

### single/join-fanout50

**movie_fanout50**: `movie_fanout50` (join), 32 x 32 = 1024 pairs, 50% positive; GPU NVIDIA L40S, 46068 MiB, max_num_seqs 256, git `2db189318`.

| Arm | Latency (s) | Speedup | Calls | Fallbacks | Waves | Peak running | Peak waiting | Computed prompt tok | Gen tok/call | Decode/call (s) | s/gen tok | Hit rate | Peak KV % | Preempt | Recall | F1 | ΔF1 | Agree | Self | Sim | Selected |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 10.43 | 1.00 | 1024 | – | 4 | 28 | 926 | 127656 | 2.3 | 0.19 | 0.084 | 0.43 | 0.8 | 0 | 0.79 | 0.79 | 0.00 | 1.00 | 1.00 | – | 515 |
| `kalypso_block:2` | 7.00 | 1.49 | 512 | 0 | 2 | 95 | 471 | 77472 | 8.0 | 0.98 | 0.122 | 0.34 | 3.4 | 0 | 0.80 | 0.74 | -0.05 | 0.60 | 1.00 | – | 603 |
| `kalypso_block:4` | 6.97 | 1.50 | 256 | 0 | 1 | 104 | 234 | 75068 | 16.0 | 1.94 | 0.121 | 0.21 | 7.1 | 0 | 0.88 | 0.83 | 0.05 | 0.69 | 1.00 | – | 571 |
| `kalypso_block:6` | 7.15 | 1.46 | 171 | 0 | 1 | 107 | 156 | 74257 | 23.9 | 2.71 | 0.113 | 0.15 | 10.8 | 0 | 0.73 | 0.74 | -0.05 | 0.67 | 1.00 | – | 505 |
| `kalypso_block:11` | 7.72 | 1.35 | 94 | 1 | 1 | 93 | 87 | 73912 | 45.5 | 3.69 | 0.081 | 0.08 | 17.1 | 0 | 0.39 | 0.51 | -0.28 | 0.58 | 1.00 | – | 279 |

Best: `kalypso_block:4` (1.50x).

### single/join-fanout75

**movie_fanout75**: `movie_fanout75` (join), 32 x 32 = 1024 pairs, 75% positive; GPU NVIDIA L40S, 46068 MiB, max_num_seqs 256, git `2db189318`.

| Arm | Latency (s) | Speedup | Calls | Fallbacks | Waves | Peak running | Peak waiting | Computed prompt tok | Gen tok/call | Decode/call (s) | s/gen tok | Hit rate | Peak KV % | Preempt | Recall | F1 | ΔF1 | Agree | Self | Sim | Selected |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 10.74 | 1.00 | 1024 | – | 4 | 27 | 926 | 127652 | 2.3 | 0.21 | 0.089 | 0.43 | 0.8 | 0 | 0.71 | 0.81 | 0.00 | 1.00 | 1.00 | – | 576 |
| `kalypso_block:2` | 7.14 | 1.50 | 512 | 0 | 2 | 97 | 470 | 77494 | 8.0 | 1.03 | 0.129 | 0.34 | 3.4 | 0 | 0.81 | 0.87 | 0.06 | 0.67 | 1.00 | – | 670 |
| `kalypso_block:4` | 7.14 | 1.50 | 256 | 0 | 1 | 105 | 234 | 75075 | 16.0 | 2.05 | 0.128 | 0.21 | 7.0 | 0 | 0.89 | 0.91 | 0.10 | 0.71 | 1.00 | – | 738 |
| `kalypso_block:6` | 7.35 | 1.46 | 171 | 0 | 1 | 107 | 160 | 74284 | 23.9 | 2.85 | 0.119 | 0.15 | 10.8 | 0 | 0.83 | 0.89 | 0.08 | 0.67 | 1.00 | – | 669 |
| `kalypso_block:11` | 7.95 | 1.35 | 94 | 1 | 1 | 93 | 87 | 73950 | 45.5 | 3.87 | 0.085 | 0.08 | 17.2 | 0 | 0.58 | 0.71 | -0.10 | 0.61 | 1.00 | – | 476 |

Best: `kalypso_block:2` (1.50x).

### single/join-fanout90

**movie_fanout90**: `movie_fanout90` (join), 32 x 32 = 1024 pairs, 91% positive; GPU NVIDIA L40S, 46068 MiB, max_num_seqs 256, git `2db189318`.

| Arm | Latency (s) | Speedup | Calls | Fallbacks | Waves | Peak running | Peak waiting | Computed prompt tok | Gen tok/call | Decode/call (s) | s/gen tok | Hit rate | Peak KV % | Preempt | Recall | F1 | ΔF1 | Agree | Self | Sim | Selected |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 10.54 | 1.00 | 1024 | – | 4 | 30 | 943 | 128550 | 2.4 | 0.22 | 0.089 | 0.43 | 0.9 | 0 | 0.71 | 0.82 | 0.00 | 1.00 | 1.00 | – | 692 |
| `kalypso_block:2` | 7.03 | 1.50 | 512 | 0 | 2 | 96 | 472 | 78347 | 8.0 | 1.01 | 0.126 | 0.34 | 3.4 | 0 | 0.85 | 0.91 | 0.09 | 0.63 | 1.00 | – | 817 |
| `kalypso_block:4` | 7.03 | 1.50 | 256 | 0 | 1 | 104 | 234 | 75932 | 16.0 | 1.98 | 0.124 | 0.20 | 7.0 | 0 | 0.91 | 0.94 | 0.12 | 0.71 | 1.00 | – | 855 |
| `kalypso_block:6` | 7.17 | 1.47 | 171 | 0 | 1 | 106 | 156 | 75137 | 23.9 | 2.76 | 0.115 | 0.14 | 10.8 | 0 | 0.82 | 0.89 | 0.07 | 0.67 | 1.00 | – | 786 |
| `kalypso_block:11` | 7.82 | 1.35 | 94 | 1 | 1 | 93 | 87 | 74809 | 45.5 | 3.81 | 0.084 | 0.08 | 17.3 | 0 | 0.60 | 0.74 | -0.08 | 0.60 | 1.00 | – | 574 |

Best: `kalypso_block:2` (1.50x).

### single/join-n16

**movie_opposite-n16**: `movie_opposite` (join), 16 x 16 = 256 pairs, 43% positive; GPU NVIDIA L40S, 46068 MiB, max_num_seqs 256, git `2db189318`.

| Arm | Latency (s) | Speedup | Calls | Fallbacks | Waves | Peak running | Peak waiting | Computed prompt tok | Gen tok/call | Decode/call (s) | s/gen tok | Hit rate | Peak KV % | Preempt | Recall | F1 | ΔF1 | Agree | Self | Sim | Selected |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 2.69 | 1.00 | 256 | – | 1 | 22 | 208 | 34210 | 2.3 | 0.17 | 0.077 | 0.41 | 0.7 | 0 | 0.71 | 0.68 | 0.00 | 1.00 | 1.00 | – | 119 |
| `kalypso_block:2` | 2.00 | 1.34 | 128 | 0 | 1 | 85 | 92 | 21729 | 8.0 | 0.73 | 0.091 | 0.32 | 3.1 | 0 | 0.55 | 0.55 | -0.13 | 0.59 | 1.00 | – | 113 |
| `kalypso_block:4` | 2.15 | 1.25 | 64 | 0 | 1 | 63 | 57 | 21344 | 16.0 | 0.98 | 0.061 | 0.18 | 4.5 | 0 | 0.67 | 0.59 | -0.09 | 0.67 | 1.00 | – | 140 |
| `kalypso_block:6` | 2.32 | 1.16 | 43 | 0 | 1 | 43 | 38 | 21246 | 23.8 | 1.17 | 0.049 | 0.12 | 4.6 | 0 | 0.68 | 0.62 | -0.06 | 0.65 | 1.00 | – | 132 |

Best: `kalypso_per_tuple` (1.00x).

### single/join-n48

**movie_opposite-n48**: `movie_opposite` (join), 48 x 48 = 2304 pairs, 50% positive; GPU NVIDIA L40S, 46068 MiB, max_num_seqs 256, git `2db189318`.

| Arm | Latency (s) | Speedup | Calls | Fallbacks | Waves | Peak running | Peak waiting | Computed prompt tok | Gen tok/call | Decode/call (s) | s/gen tok | Hit rate | Peak KV % | Preempt | Recall | F1 | ΔF1 | Agree | Self | Sim | Selected |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 25.56 | 1.00 | 2304 | – | 9 | 27 | 1972 | 310498 | 2.3 | 0.20 | 0.087 | 0.42 | 0.8 | 0 | 0.54 | 0.55 | 0.00 | 1.00 | 1.00 | – | 1127 |
| `kalypso_block:2` | 17.99 | 1.42 | 1152 | 0 | 5 | 88 | 1086 | 199274 | 8.0 | 1.08 | 0.135 | 0.33 | 3.1 | 0 | 0.61 | 0.60 | 0.05 | 0.58 | 1.00 | – | 1195 |
| `kalypso_block:4` | 18.03 | 1.42 | 576 | 0 | 3 | 93 | 553 | 198262 | 16.0 | 2.27 | 0.142 | 0.20 | 6.6 | 0 | 0.71 | 0.62 | 0.07 | 0.67 | 1.00 | – | 1484 |
| `kalypso_block:6` | 18.25 | 1.40 | 384 | 0 | 2 | 96 | 369 | 197929 | 24.0 | 3.43 | 0.143 | 0.14 | 10.0 | 0 | 0.60 | 0.56 | 0.01 | 0.61 | 1.00 | – | 1297 |

Best: `kalypso_block:2` (1.42x).

### single/join-n64

**movie_opposite-n64**: `movie_opposite` (join), 64 x 64 = 4096 pairs, 50% positive; GPU NVIDIA L40S, 46068 MiB, max_num_seqs 256, git `2db189318`.

| Arm | Latency (s) | Speedup | Calls | Fallbacks | Waves | Peak running | Peak waiting | Computed prompt tok | Gen tok/call | Decode/call (s) | s/gen tok | Hit rate | Peak KV % | Preempt | Recall | F1 | ΔF1 | Agree | Self | Sim | Selected |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 45.05 | 1.00 | 4096 | – | 16 | 28 | 2011 | 546811 | 2.3 | 0.20 | 0.089 | 0.42 | 0.8 | 0 | 0.55 | 0.54 | 0.00 | 1.00 | 1.00 | – | 2084 |
| `kalypso_block:2` | 30.73 | 1.47 | 2048 | 0 | 8 | 92 | 1621 | 347208 | 8.0 | 1.11 | 0.139 | 0.34 | 3.1 | 0 | 0.54 | 0.55 | 0.01 | 0.56 | 1.00 | – | 1974 |
| `kalypso_block:4` | 31.24 | 1.44 | 1024 | 0 | 4 | 96 | 991 | 344089 | 16.0 | 2.39 | 0.149 | 0.20 | 6.6 | 0 | 0.71 | 0.61 | 0.07 | 0.65 | 0.99 | – | 2704 |
| `kalypso_block:6` | 31.73 | 1.42 | 683 | 0 | 3 | 98 | 667 | 342961 | 24.0 | 3.65 | 0.152 | 0.14 | 10.0 | 0 | 0.62 | 0.57 | 0.03 | 0.62 | 1.00 | – | 2413 |

Best: `kalypso_block:2` (1.47x).

## validate

### validate/cnli-filter-s25

**cnli_sel25-n200**: `cnli_sel25` (filter), 200 rows, 24% positive; GPU NVIDIA L40S, 46068 MiB, max_num_seqs 256, git `5522cd270`.

| Arm | Latency (s) | Speedup | Calls | Fallbacks | Waves | Peak running | Peak waiting | Computed prompt tok | Gen tok/call | Decode/call (s) | s/gen tok | Hit rate | Peak KV % | Preempt | Recall | F1 | ΔF1 | Agree | Self | Sim | Selected |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 24.03 | 1.00 | 200 | – | 1 | 5 | 193 | 307490 | 3.0 | 0.30 | 0.099 | 0.03 | 1.4 | 0 | 0.79 | 0.54 | 0.00 | 1.00 | 1.00 | – | 94 |
| `kalypso_block:4` | 25.99 | 0.93 | 50 | 0 | 1 | 7 | 47 | 291901 | 16.0 | 2.33 | 0.146 | 0.01 | 8.4 | 0 | 0.46 | 0.42 | -0.11 | 0.72 | 1.00 | – | 56 |
| `kalypso_block:16` | 35.51 | 0.68 | 13 | 0 | 1 | 8 | 12 | 290503 | 68.0 | 11.26 | 0.166 | 0.00 | 37.3 | 0 | 0.35 | 0.35 | -0.18 | 0.65 | 1.00 | – | 49 |
| `kalypso_block:32` | – | – | – | – | – | – | – | – | – | – | – | – | – | – | – | – | – | – | – | – | – |
| `kalypso_block:2` | 24.11 | 1.00 | 100 | 0 | 1 | 7 | 96 | 293902 | 8.0 | 1.03 | 0.129 | 0.02 | 4.0 | 0 | 0.38 | 0.41 | -0.12 | 0.69 | 1.00 | – | 39 |
| `kalypso_block:3` | 24.68 | 0.97 | 67 | 0 | 1 | 7 | 64 | 292577 | 11.9 | 1.64 | 0.137 | 0.01 | 6.4 | 0 | 0.50 | 0.46 | -0.07 | 0.67 | 1.00 | – | 56 |

Best: `kalypso_per_tuple` (1.00x).
- `kalypso_block:32` failed 3 trial(s); see run.jsonl

### validate/filter-join-s25

**pipe_filter_join_s25**: `pipe_filter_join_s25` (join), 100 x 32 = 3200 pairs, 14% positive, stages filter>join; GPU NVIDIA L40S, 46068 MiB, max_num_seqs 256, git `5522cd270`.

| Arm | Latency (s) | Speedup | Calls | Fallbacks | Waves | Peak running | Peak waiting | Computed prompt tok | Gen tok/call | Decode/call (s) | s/gen tok | Hit rate | Peak KV % | Preempt | Recall | F1 | ΔF1 | Agree | Self | Sim | Selected |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 6.58 | 1.00 | 612 | – | 3 | 37 | 445 | 79855 | 2.3 | 0.21 | 0.091 | 0.40 | 1.0 | 0 | 0.46 | 0.59 | 0.00 | 1.00 | 0.99 | – | 252 |
| `kalypso_block:4` | 7.34 | 0.90 | 233 | 0 | 1 | 91 | 181 | 74106 | 16.0 | 1.82 | 0.114 | 0.19 | 7.0 | 0 | 0.67 | 0.61 | 0.02 | 0.89 | 1.00 | – | 532 |
| `kalypso_block:16` | 9.82 | 0.67 | 57 | 0 | 1 | 50 | 46 | 68955 | 70.0 | 3.96 | 0.057 | 0.05 | 15.2 | 0 | 0.44 | 0.49 | -0.10 | 0.90 | 1.00 | – | 366 |
| `kalypso_block:32` | 12.81 | 0.51 | 25 | 0 | 1 | 21 | 18 | 57855 | 145.6 | 5.56 | 0.038 | 0.02 | 12.8 | 0 | 0.30 | 0.34 | -0.25 | 0.86 | 1.00 | – | 335 |
| `kalypso_pipe:2,4` | 6.50 | 1.01 | 234 | 0 | 1 | 92 | 163 | 67284 | 14.3 | 1.59 | 0.112 | 0.19 | 7.0 | 0 | 0.62 | 0.61 | 0.02 | 0.90 | 1.00 | – | 462 |
| `kalypso_pipe:4,1` | 10.09 | 0.65 | 857 | 0 | 4 | 25 | 739 | 117716 | 2.6 | 0.20 | 0.076 | 0.40 | 1.1 | 0 | 0.69 | 0.73 | 0.14 | 0.95 | 1.00 | – | 402 |
| `kalypso_pipe:1,2` | 4.93 | 1.33 | 356 | 0 | 2 | 83 | 205 | 55344 | 6.6 | 0.79 | 0.120 | 0.31 | 3.4 | 0 | 0.41 | 0.49 | -0.10 | 0.95 | 1.00 | – | 312 |
| `kalypso_pipe:1,4` | 5.09 | 1.29 | 228 | 0 | 1 | 92 | 108 | 54495 | 10.3 | 1.10 | 0.107 | 0.21 | 7.1 | 0 | 0.44 | 0.52 | -0.07 | 0.96 | 1.00 | – | 319 |
| `kalypso_pipe:2,1` | 8.98 | 0.73 | 786 | 0 | 4 | 50 | 644 | 106153 | 2.6 | 0.20 | 0.077 | 0.40 | 1.4 | 0 | 0.63 | 0.71 | 0.12 | 0.96 | 1.00 | – | 354 |
| `kalypso_pipe:2,2` | 6.40 | 1.03 | 418 | 0 | 2 | 86 | 319 | 68307 | 8.0 | 0.92 | 0.115 | 0.32 | 3.4 | 0 | 0.55 | 0.56 | -0.03 | 0.91 | 1.00 | – | 424 |
| `kalypso_pipe:2,3` | 6.45 | 1.02 | 296 | 1 | 2 | 90 | 220 | 67613 | 11.3 | 1.30 | 0.115 | 0.24 | 5.2 | 0 | 0.57 | 0.59 | 0.00 | 0.90 | 0.93 | – | 426 |
| `kalypso_pipe:2,5` | 6.64 | 0.99 | 198 | 1 | 1 | 93 | 131 | 67098 | 16.9 | 1.81 | 0.107 | 0.16 | 8.9 | 0 | 0.59 | 0.61 | 0.02 | 0.90 | 0.94 | – | 440 |
| `kalypso_pipe:2,8` | 6.92 | 0.95 | 142 | 0 | 1 | 84 | 81 | 66754 | 23.6 | 2.06 | 0.087 | 0.11 | 13.0 | 0 | 0.54 | 0.57 | -0.02 | 0.91 | 1.00 | – | 406 |
| `kalypso_pipe:3,4` | 6.75 | 0.98 | 226 | 1 | 1 | 91 | 170 | 68522 | 15.3 | 1.70 | 0.111 | 0.19 | 7.1 | 0 | 0.65 | 0.62 | 0.04 | 0.90 | 1.00 | – | 491 |
| `kalypso_pipe:4,4` | 7.36 | 0.90 | 233 | 0 | 1 | 91 | 185 | 74106 | 16.0 | 1.80 | 0.113 | 0.19 | 7.0 | 0 | 0.67 | 0.61 | 0.02 | 0.89 | 1.00 | – | 533 |

Best: `kalypso_pipe:2,3` (1.02x).

### validate/filter-map-n1000

**pipe_filter_map-n1000**: `pipe_filter_map` (map), 1000 rows, 75% positive, stages filter>map; GPU NVIDIA L40S, 46068 MiB, max_num_seqs 256, git `5522cd270`.

| Arm | Latency (s) | Speedup | Calls | Fallbacks | Waves | Peak running | Peak waiting | Computed prompt tok | Gen tok/call | Decode/call (s) | s/gen tok | Hit rate | Peak KV % | Preempt | Recall | F1 | ΔF1 | Agree | Self | Sim | Selected |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 15.86 | 1.00 | 1477 | – | 6 | 253 | 905 | 149831 | 8.5 | 0.73 | 0.086 | 0.37 | 7.2 | 0 | 0.63 | 0.77 | 0.00 | 1.00 | 0.78 | 1.00 | 477 |
| `kalypso_block:4` | 12.90 | 1.23 | 399 | 0 | 2 | 160 | 177 | 79026 | 59.0 | 3.14 | 0.053 | 0.18 | 10.3 | 0 | 0.79 | 0.88 | 0.11 | 0.00 | 0.13 | 0.39 | 596 |
| `kalypso_block:16` | 26.53 | 0.60 | 101 | 0 | 1 | 49 | 41 | 64232 | 213.9 | 7.00 | 0.033 | 0.06 | 8.9 | 0 | 0.78 | 0.87 | 0.10 | 0.00 | 0.05 | 0.35 | 594 |
| `kalypso_block:32` | 43.41 | 0.36 | 53 | 0 | 1 | 25 | 21 | 62353 | 398.1 | 12.10 | 0.030 | 0.03 | 8.5 | 0 | 0.78 | 0.87 | 0.10 | 0.01 | 0.04 | 0.35 | 594 |
| `kalypso_pipe:3,1` | 13.59 | 1.17 | 1015 | 1 | 4 | 249 | 365 | 116557 | 17.0 | 1.82 | 0.107 | 0.31 | 7.3 | 0 | 0.78 | 0.87 | 0.10 | 0.62 | 0.89 | 0.93 | 589 |
| `kalypso_pipe:4,1` | 13.70 | 1.16 | 972 | 0 | 4 | 249 | 329 | 113364 | 18.6 | 1.92 | 0.103 | 0.33 | 7.2 | 0 | 0.79 | 0.88 | 0.11 | 0.61 | 0.91 | 0.93 | 596 |
| `kalypso_pipe:2,1` | 14.25 | 1.11 | 1079 | 0 | 5 | 250 | 447 | 122417 | 14.6 | 1.68 | 0.115 | 0.27 | 7.2 | 0 | 0.75 | 0.85 | 0.08 | 0.64 | 0.90 | 0.93 | 574 |
| `kalypso_pipe:3,2` | 13.45 | 1.18 | 629 | 1 | 3 | 256 | 288 | 90783 | 40.3 | 2.69 | 0.067 | 0.25 | 11.0 | 0 | 0.78 | 0.87 | 0.10 | 0.00 | 0.33 | 0.41 | 589 |
| `kalypso_pipe:6,1` | 13.97 | 1.14 | 886 | 0 | 4 | 249 | 316 | 109800 | 20.8 | 2.06 | 0.099 | 0.32 | 10.1 | 0 | 0.80 | 0.88 | 0.11 | 0.61 | 0.87 | 0.93 | 603 |

Best: `kalypso_block:4` (1.23x).

### validate/filter-map-n50

**pipe_filter_map-n50**: `pipe_filter_map` (map), 50 rows, 74% positive, stages filter>map; GPU NVIDIA L40S, 46068 MiB, max_num_seqs 256, git `5522cd270`.

| Arm | Latency (s) | Speedup | Calls | Fallbacks | Waves | Peak running | Peak waiting | Computed prompt tok | Gen tok/call | Decode/call (s) | s/gen tok | Hit rate | Peak KV % | Preempt | Recall | F1 | ΔF1 | Agree | Self | Sim | Selected |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 2.30 | 1.00 | 69 | – | 1 | 19 | 31 | 7152 | 8.3 | 0.27 | 0.033 | 0.36 | 0.5 | 0 | 0.51 | 0.68 | 0.00 | 1.00 | 1.00 | 1.00 | 19 |
| `kalypso_block:4` | 4.03 | 0.57 | 20 | 0 | 1 | 13 | 0 | 3811 | 50.4 | 1.11 | 0.022 | 0.17 | 0.5 | 0 | 0.65 | 0.76 | 0.08 | 0.00 | 0.15 | 0.39 | 26 |
| `kalypso_block:16` | 9.94 | 0.23 | 6 | 0 | 1 | 4 | 0 | 3049 | 130.2 | 2.86 | 0.022 | 0.05 | 0.4 | 0 | 0.59 | 0.72 | 0.04 | 0.00 | 0.50 | 0.35 | 24 |
| `kalypso_block:32` | 19.61 | 0.12 | 3 | 0 | 1 | 2 | 0 | 3104 | 328.0 | 7.03 | 0.021 | 0.02 | 0.5 | 0 | 0.70 | 0.80 | 0.12 | 0.00 | 1.00 | 0.33 | 28 |
| `kalypso_pipe:2,1` | 2.31 | 1.00 | 49 | 0 | 1 | 25 | 0 | 5637 | 14.8 | 0.37 | 0.025 | 0.25 | 0.7 | 0 | 0.59 | 0.72 | 0.04 | 0.71 | 1.00 | 0.99 | 24 |
| `kalypso_pipe:4,1` | 2.07 | 1.11 | 39 | 0 | 1 | 26 | 0 | 5233 | 18.9 | 0.46 | 0.025 | 0.23 | 0.7 | 0 | 0.65 | 0.76 | 0.08 | 0.55 | 0.96 | 0.95 | 26 |
| `kalypso_pipe:2,2` | 2.60 | 0.88 | 37 | 0 | 1 | 25 | 0 | 4618 | 28.2 | 0.65 | 0.023 | 0.24 | 0.7 | 0 | 0.59 | 0.72 | 0.04 | 0.00 | 0.62 | 0.41 | 24 |
| `kalypso_pipe:3,1` | 1.99 | 1.16 | 40 | 0 | 1 | 23 | 0 | 5069 | 17.2 | 0.42 | 0.025 | 0.24 | 0.6 | 0 | 0.59 | 0.73 | 0.05 | 0.48 | 1.00 | 0.93 | 23 |

Best: `kalypso_pipe:3,1` (1.16x).

### validate/join-fanout25-n64

**movie_fanout25-n64**: `movie_fanout25` (join), 64 x 64 = 4096 pairs, 25% positive; GPU NVIDIA L40S, 46068 MiB, max_num_seqs 256, git `5522cd270`.

| Arm | Latency (s) | Speedup | Calls | Fallbacks | Waves | Peak running | Peak waiting | Computed prompt tok | Gen tok/call | Decode/call (s) | s/gen tok | Hit rate | Peak KV % | Preempt | Recall | F1 | ΔF1 | Agree | Self | Sim | Selected |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 42.26 | 1.00 | 4096 | – | 16 | 26 | 1957 | 518047 | 2.2 | 0.18 | 0.084 | 0.42 | 0.8 | 0 | 0.66 | 0.62 | 0.00 | 1.00 | 1.00 | – | 1150 |
| `kalypso_block:4` | 27.59 | 1.53 | 1024 | 0 | 4 | 106 | 983 | 302978 | 16.0 | 2.31 | 0.144 | 0.21 | 7.0 | 0 | 0.79 | 0.55 | -0.07 | 0.67 | 1.00 | – | 1901 |
| `kalypso_block:16` | 31.34 | 1.35 | 256 | 0 | 1 | 121 | 247 | 298416 | 71.0 | 10.51 | 0.148 | 0.06 | 32.4 | 0 | 0.29 | 0.38 | -0.25 | 0.74 | 1.00 | – | 554 |
| `kalypso_block:32` | 39.52 | 1.07 | 128 | 0 | 1 | 128 | 123 | 298157 | 151.0 | 21.26 | 0.141 | 0.03 | 69.1 | 0 | 0.07 | 0.10 | -0.52 | 0.68 | 1.00 | – | 355 |
| `kalypso_block:10` | 28.50 | 1.48 | 410 | 0 | 2 | 112 | 392 | 298644 | 41.0 | 6.00 | 0.146 | 0.09 | 18.6 | 0 | 0.46 | 0.50 | -0.13 | 0.75 | 1.00 | – | 872 |
| `kalypso_block:2` | 27.83 | 1.52 | 2048 | 0 | 8 | 100 | 1636 | 312092 | 8.0 | 1.08 | 0.135 | 0.34 | 3.4 | 0 | 0.65 | 0.47 | -0.15 | 0.61 | 0.99 | – | 1778 |
| `kalypso_block:5` | 27.66 | 1.53 | 820 | 1 | 4 | 108 | 786 | 301209 | 20.0 | 2.93 | 0.146 | 0.18 | 8.9 | 0 | 0.65 | 0.50 | -0.12 | 0.68 | 1.00 | – | 1619 |
| `kalypso_block:9` | 28.08 | 1.50 | 456 | 1 | 2 | 110 | 436 | 297956 | 35.9 | 5.26 | 0.146 | 0.10 | 16.3 | 0 | 0.54 | 0.54 | -0.08 | 0.75 | 1.00 | – | 1011 |
| `kalypso_block:11` | 28.77 | 1.47 | 373 | 0 | 2 | 114 | 361 | 298642 | 45.9 | 6.69 | 0.146 | 0.09 | 20.9 | 0 | 0.42 | 0.50 | -0.13 | 0.77 | 1.00 | – | 727 |

Best: `kalypso_per_tuple` (1.00x).

### validate/map-filter

**pipe_map_filter-n500**: `pipe_map_filter` (filter), 500 rows, 75% positive, stages map>filter; GPU NVIDIA L40S, 46068 MiB, max_num_seqs 256, git `5522cd270`.

| Arm | Latency (s) | Speedup | Calls | Fallbacks | Waves | Peak running | Peak waiting | Computed prompt tok | Gen tok/call | Decode/call (s) | s/gen tok | Hit rate | Peak KV % | Preempt | Recall | F1 | ΔF1 | Agree | Self | Sim | Selected |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 11.93 | 1.00 | 1001 | – | 4 | 250 | 443 | 98163 | 11.0 | 1.44 | 0.131 | 0.53 | 7.3 | 0 | 0.68 | 0.81 | 0.00 | 1.00 | 1.00 | – | 253 |
| `kalypso_block:4` | 18.70 | 0.64 | 552 | 0 | 3 | 125 | 85 | 123958 | 39.3 | 2.03 | 0.052 | 0.18 | 8.0 | 0 | 0.78 | 0.86 | 0.05 | 0.84 | 0.97 | – | 308 |
| `kalypso_block:16` | 38.31 | 0.31 | 146 | 0 | 1 | 32 | 18 | 89718 | 116.9 | 3.69 | 0.032 | 0.07 | 8.3 | 0 | 0.64 | 0.78 | -0.02 | 0.78 | 0.90 | – | 242 |
| `kalypso_block:32` | 62.25 | 0.19 | 413 | 0 | 2 | 16 | 8 | 121479 | 51.3 | 1.34 | 0.026 | 0.15 | 9.1 | 0 | 0.61 | 0.75 | -0.06 | 0.80 | 0.93 | – | 238 |
| `kalypso_pipe:1,7` | 13.79 | 0.87 | 853 | 0 | 4 | 255 | 382 | 118998 | 18.9 | 2.14 | 0.113 | 0.37 | 11.5 | 0 | 0.75 | 0.85 | 0.04 | 0.85 | 0.86 | – | 285 |
| `kalypso_pipe:1,4` | 13.86 | 0.86 | 906 | 0 | 4 | 251 | 369 | 121642 | 17.8 | 2.08 | 0.117 | 0.37 | 10.3 | 0 | 0.78 | 0.86 | 0.05 | 0.83 | 0.85 | – | 301 |
| `kalypso_pipe:1,3` | 14.09 | 0.85 | 949 | 0 | 4 | 251 | 375 | 123835 | 17.1 | 2.02 | 0.118 | 0.37 | 9.5 | 0 | 0.77 | 0.85 | 0.04 | 0.82 | 0.86 | – | 302 |
| `kalypso_pipe:1,6` | 13.99 | 0.85 | 866 | 0 | 4 | 255 | 364 | 119569 | 18.7 | 2.13 | 0.114 | 0.37 | 11.1 | 0 | 0.72 | 0.82 | 0.01 | 0.84 | 0.84 | – | 279 |
| `kalypso_pipe:1,8` | 13.76 | 0.87 | 845 | 0 | 4 | 254 | 361 | 118431 | 19.1 | 2.17 | 0.114 | 0.37 | 11.5 | 0 | 0.72 | 0.83 | 0.02 | 0.85 | 0.85 | – | 272 |
| `kalypso_pipe:1,14` | 14.68 | 0.81 | 818 | 0 | 4 | 255 | 357 | 117472 | 20.0 | 2.26 | 0.113 | 0.37 | 13.4 | 0 | 0.69 | 0.81 | 0.00 | 0.83 | 0.78 | – | 269 |
| `kalypso_pipe:2,7` | 15.79 | 0.76 | 471 | 0 | 2 | 221 | 186 | 111327 | 45.9 | 3.29 | 0.072 | 0.19 | 10.7 | 0 | 0.75 | 0.85 | 0.04 | 0.85 | 0.87 | – | 282 |

Best: `kalypso_per_tuple` (1.00x).

