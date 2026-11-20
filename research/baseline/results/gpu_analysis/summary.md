# Session 1 GPU results (E0-E3)

L40S, Qwen2.5-7B-Instruct; 1 warm-up + 5 repeats per arm unless noted. Latency: median [min, max] over repeats; CI: 95% bootstrap of the median. Speedup: per-repeat paired baseline/arm, median; ✓ = faster in every repeat.

## E0-overhead

status: {'rsel50:off': 'done', 'rsel50:on': 'done'} 

### rsel50 (off)

| arm | n | latency s | CI95 | speedup | LLM req | F1 | achieved sel | rows out | peak KV % | peak waiting | preempt |
|---|---|---|---|---|---|---|---|---|---|---|---|
| kalypso_per_tuple | 5 | 3.77 [3.68, 3.82] | [3.68, 3.82] | - | 400 | 0.75 | 0.307 | 123 | 0.9 | 347 | 0 |
| kalypso_block:8 | 5 | 2.14 [2.09, 2.16] | [2.09, 2.16] | 1.77x ✓ | 50 | 0.86 | 0.400 | 160 | 3.5 | 43 | 0 |

### rsel50 (on)

| arm | n | latency s | CI95 | speedup | LLM req | F1 | achieved sel | rows out | peak KV % | peak waiting | preempt |
|---|---|---|---|---|---|---|---|---|---|---|---|
| kalypso_per_tuple | 5 | 3.96 [3.91, 4.04] | [3.91, 4.04] | - | 400 | 0.75 | 0.307 | 123 | 0.8 | 347 | 0 |
| kalypso_block:8 | 5 | 2.14 [2.14, 2.15] | [2.14, 2.15] | 1.85x ✓ | 50 | 0.86 | 0.400 | 160 | 3.3 | 42 | 0 |

### Trace trace-on

12 queries (12 complete); KV admission counters none; plan time median 0.0276 s; by order (counts match).

| stage:k | out/in median [min,max] | partial blocks max | forced launches max | retries max | wait p95 s median | truncated max | block fallbacks max |
|---|---|---|---|---|---|---|---|
| stage1:k1 | 0.307 [0.307, 0.307] | - | 0 | 0 | 0.187 | 0 | 0 |
| stage1:k8 | 0.400 [0.400, 0.400] | 0 | 0 | 0 | 0.104 | 0 | 0 |

- vllm-metrics-off: 91 samples, KV % max 3.5, waiting max 381, preemptions 0.0
- vllm-metrics-on: 102 samples, KV % max 3.3, waiting max 381, preemptions 0.0

## E1-s05

status: {'rsel05:on': 'done', 'rpipe_s05:on': 'done'} 

### rpipe_s05 (on)

| arm | n | latency s | CI95 | speedup | LLM req | F1 | achieved sel | rows out | peak KV % | peak waiting | preempt |
|---|---|---|---|---|---|---|---|---|---|---|---|
| kalypso_per_tuple | 5 | 1.76 [1.75, 1.77] | [1.75, 1.77] | - | 164 | 0.43 | 0.009 | 30 | 1.0 | 46 | 0 |
| kalypso_pipe:4,1 | 5 | 2.63 [2.62, 2.65] | [2.62, 2.65] | 0.67x ✗ | 185 | 0.47 | 0.025 | 80 | 1.1 | 113 | 0 |
| kalypso_pipe:1,4 | 5 | 1.80 [1.78, 1.82] | [1.78, 1.82] | 0.97x ✗ | 116 | 0.37 | 0.016 | 50 | 1.1 | 46 | 0 |
| kalypso_pipe:4,4 | 5 | 2.23 [2.23, 2.24] | [2.23, 2.24] | 0.79x ✗ | 65 | 0.41 | 0.036 | 116 | 3.1 | 20 | 0 |

### rsel05 (on)

| arm | n | latency s | CI95 | speedup | LLM req | F1 | achieved sel | rows out | peak KV % | peak waiting | preempt |
|---|---|---|---|---|---|---|---|---|---|---|---|
| kalypso_per_tuple | 5 | 3.84 [3.80, 3.86] | [3.80, 3.86] | - | 400 | 0.70 | 0.050 | 20 | 1.0 | 346 | 0 |
| kalypso_block:4 | 5 | 2.05 [2.02, 2.06] | [2.02, 2.06] | 1.88x ✓ | 100 | 0.71 | 0.070 | 28 | 4.5 | 66 | 0 |
| kalypso_block:8 | 5 | 2.22 [2.19, 2.25] | [2.19, 2.25] | 1.72x ✓ | 50 | 0.59 | 0.060 | 24 | 3.9 | 41 | 0 |

### Trace trace-on

42 queries (42 complete); KV admission counters none; plan time median 0.0811 s; not attributed: 42 queries vs 0 arms x 6 trials (each arm may issue more than one query).

| stage:k | out/in median [min,max] | partial blocks max | forced launches max | retries max | wait p95 s median | truncated max | block fallbacks max |
|---|---|---|---|---|---|---|---|
| stage1:k1 | 0.640 [0.050, 0.640] | - | 0 | 0 | 0.048 | 0 | 0 |
| stage1:k4 | 1.600 [0.070, 1.600] | 0 | 0 | 0 | 0.032 | 0 | 0 |
| stage1:k8 | 0.060 [0.060, 0.060] | 0 | 0 | 0 | 0.105 | 0 | 0 |
| stage2:k1 | 0.484 [0.469, 0.506] | - | 0 | 0 | 0.039 | 0 | 0 |
| stage2:k4 | 0.753 [0.706, 0.781] | 0 | 0 | 0 | 0.030 | 0 | 0 |

- vllm-metrics-on: 257 samples, KV % max 4.5, waiting max 381, preemptions 0.0

## E1-s20

status: {'rsel20:on': 'done', 'rpipe_s20:on': 'done'} 

### rpipe_s20 (on)

| arm | n | latency s | CI95 | speedup | LLM req | F1 | achieved sel | rows out | peak KV % | peak waiting | preempt |
|---|---|---|---|---|---|---|---|---|---|---|---|
| kalypso_per_tuple | 5 | 5.47 [5.37, 5.59] | [5.37, 5.59] | - | 516 | 0.55 | 0.063 | 202 | 1.0 | 366 | 0 |
| kalypso_pipe:4,1 | 5 | 8.74 [8.62, 8.95] | [8.62, 8.95] | 0.63x ✗ | 697 | 0.73 | 0.101 | 322 | 1.1 | 609 | 0 |
| kalypso_pipe:1,4 | 5 | 4.38 [4.30, 4.47] | [4.30, 4.47] | 1.25x ✓ | 204 | 0.47 | 0.080 | 256 | 7.1 | 83 | 0 |
| kalypso_pipe:4,4 | 5 | 6.20 [6.13, 6.47] | [6.13, 6.47] | 0.88x ✗ | 193 | 0.61 | 0.137 | 439 | 7.1 | 144 | 0 |

### rsel20 (on)

| arm | n | latency s | CI95 | speedup | LLM req | F1 | achieved sel | rows out | peak KV % | peak waiting | preempt |
|---|---|---|---|---|---|---|---|---|---|---|---|
| kalypso_per_tuple | 5 | 3.96 [3.92, 4.03] | [3.92, 4.03] | - | 400 | 0.74 | 0.140 | 56 | 1.0 | 345 | 0 |
| kalypso_block:4 | 5 | 2.11 [2.10, 2.14] | [2.10, 2.14] | 1.88x ✓ | 100 | 0.87 | 0.193 | 77 | 4.5 | 65 | 0 |
| kalypso_block:8 | 5 | 2.29 [2.28, 2.33] | [2.28, 2.33] | 1.72x ✓ | 50 | 0.87 | 0.177 | 71 | 4.0 | 42 | 0 |

### Trace trace-on

42 queries (42 complete); KV admission counters none; plan time median 0.0885 s; not attributed: 42 queries vs 0 arms x 6 trials (each arm may issue more than one query).

| stage:k | out/in median [min,max] | partial blocks max | forced launches max | retries max | wait p95 s median | truncated max | block fallbacks max |
|---|---|---|---|---|---|---|---|
| stage1:k1 | 4.160 [0.140, 4.160] | - | 0 | 0 | 0.057 | 0 | 0 |
| stage1:k4 | 6.720 [0.193, 6.720] | 0 | 0 | 0 | 0.036 | 0 | 0 |
| stage1:k8 | 0.177 [0.177, 0.177] | 0 | 0 | 0 | 0.118 | 0 | 0 |
| stage2:k1 | 0.483 [0.479, 0.486] | - | 0 | 0 | 0.146 | 0 | 0 |
| stage2:k4 | 0.635 [0.615, 0.656] | 0 | 0 | 0 | 0.110 | 0 | 0 |

- vllm-metrics-on: 443 samples, KV % max 7.1, waiting max 601, preemptions 0.0

## E1-s50

status: {'rsel50:on': 'done', 'rpipe_s50:on': 'done'} 

### rpipe_s50 (on)

| arm | n | latency s | CI95 | speedup | LLM req | F1 | achieved sel | rows out | peak KV % | peak waiting | preempt |
|---|---|---|---|---|---|---|---|---|---|---|---|
| kalypso_per_tuple | 5 | 10.20 [10.08, 10.58] | [10.08, 10.58] | - | 996 | 0.63 | 0.139 | 445 | 1.0 | 758 | 0 |
| kalypso_pipe:4,1 | 5 | 13.99 [13.85, 14.51] | [13.85, 14.51] | 0.73x ✗ | 1177 | 0.73 | 0.175 | 561 | 1.1 | 1035 | 0 |
| kalypso_pipe:1,4 | 5 | 7.45 [7.38, 7.89] | [7.38, 7.89] | 1.35x ✓ | 324 | 0.55 | 0.167 | 534 | 7.1 | 190 | 0 |
| kalypso_pipe:4,4 | 5 | 9.78 [9.62, 10.05] | [9.62, 10.05] | 1.05x ✓ | 313 | 0.62 | 0.222 | 709 | 7.0 | 267 | 0 |

### rsel50 (on)

| arm | n | latency s | CI95 | speedup | LLM req | F1 | achieved sel | rows out | peak KV % | peak waiting | preempt |
|---|---|---|---|---|---|---|---|---|---|---|---|
| kalypso_per_tuple | 5 | 4.00 [3.94, 4.05] | [3.94, 4.05] | - | 400 | 0.75 | 0.310 | 124 | 1.0 | 347 | 0 |
| kalypso_block:4 | 5 | 2.13 [2.11, 2.14] | [2.11, 2.14] | 1.88x ✓ | 100 | 0.87 | 0.415 | 166 | 4.5 | 66 | 0 |
| kalypso_block:8 | 5 | 2.30 [2.29, 2.30] | [2.29, 2.30] | 1.74x ✓ | 50 | 0.86 | 0.400 | 160 | 4.0 | 43 | 0 |

### Trace trace-on

42 queries (42 complete); KV admission counters none; plan time median 0.0846 s; not attributed: 42 queries vs 0 arms x 6 trials (each arm may issue more than one query).

| stage:k | out/in median [min,max] | partial blocks max | forced launches max | retries max | wait p95 s median | truncated max | block fallbacks max |
|---|---|---|---|---|---|---|---|
| stage1:k1 | 8.960 [0.310, 8.960] | - | 0 | 0 | 0.053 | 0 | 0 |
| stage1:k4 | 11.520 [0.415, 11.520] | 0 | 0 | 0 | 0.035 | 0 | 0 |
| stage1:k8 | 0.400 [0.400, 0.400] | 0 | 0 | 0 | 0.112 | 0 | 0 |
| stage2:k1 | 0.492 [0.487, 0.497] | - | 0 | 0 | 0.412 | 0 | 0 |
| stage2:k4 | 0.605 [0.595, 0.617] | 0 | 0 | 0 | 0.303 | 0 | 0 |

- vllm-metrics-on: 628 samples, KV % max 7.1, waiting max 1060, preemptions 0.0

## E1-s80

status: {'rsel80:on': 'done', 'rpipe_s80:on': 'done'} 

### rpipe_s80 (on)

| arm | n | latency s | CI95 | speedup | LLM req | F1 | achieved sel | rows out | peak KV % | peak waiting | preempt |
|---|---|---|---|---|---|---|---|---|---|---|---|
| kalypso_per_tuple | 5 | 18.99 [18.63, 19.57] | [18.63, 19.57] | - | 1700 | 0.67 | 0.248 | 794 | 1.0 | 1467 | 0 |
| kalypso_pipe:4,1 | 5 | 22.79 [22.46, 23.23] | [22.46, 23.23] | 0.84x ✗ | 1881 | 0.74 | 0.284 | 908 | 1.1 | 1719 | 0 |
| kalypso_pipe:1,4 | 5 | 13.12 [12.87, 13.27] | [12.87, 13.27] | 1.47x ✓ | 500 | 0.59 | 0.304 | 974 | 7.1 | 380 | 0 |
| kalypso_pipe:4,4 | 5 | 14.95 [14.71, 15.26] | [14.71, 15.26] | 1.29x ✓ | 489 | 0.63 | 0.355 | 1137 | 7.1 | 437 | 0 |

### rsel80 (on)

| arm | n | latency s | CI95 | speedup | LLM req | F1 | achieved sel | rows out | peak KV % | peak waiting | preempt |
|---|---|---|---|---|---|---|---|---|---|---|---|
| kalypso_per_tuple | 5 | 3.90 [3.84, 3.92] | [3.84, 3.92] | - | 400 | 0.77 | 0.507 | 203 | 1.0 | 346 | 0 |
| kalypso_block:4 | 5 | 2.05 [2.04, 2.07] | [2.04, 2.07] | 1.90x ✓ | 100 | 0.88 | 0.642 | 257 | 4.4 | 65 | 0 |
| kalypso_block:8 | 5 | 2.24 [2.23, 2.25] | [2.23, 2.25] | 1.74x ✓ | 50 | 0.89 | 0.665 | 266 | 3.9 | 42 | 0 |

### Trace trace-on

42 queries (42 complete); KV admission counters none; plan time median 0.0871 s; not attributed: 42 queries vs 0 arms x 6 trials (each arm may issue more than one query).

| stage:k | out/in median [min,max] | partial blocks max | forced launches max | retries max | wait p95 s median | truncated max | block fallbacks max |
|---|---|---|---|---|---|---|---|
| stage1:k1 | 16.000 [0.507, 16.000] | - | 0 | 0 | 0.058 | 0 | 0 |
| stage1:k4 | 18.560 [0.642, 18.560] | 0 | 0 | 0 | 0.039 | 0 | 0 |
| stage1:k8 | 0.665 [0.665, 0.665] | 0 | 0 | 0 | 0.116 | 0 | 0 |
| stage2:k1 | 0.493 [0.489, 0.496] | - | 0 | 0 | 0.920 | 0 | 0 |
| stage2:k4 | 0.612 [0.606, 0.614] | 0 | 0 | 0 | 0.403 | 0 | 0 |

- vllm-metrics-on: 918 samples, KV % max 7.1, waiting max 1733, preemptions 0.0

## E1-s95

status: {'rsel95:on': 'done', 'rpipe_s95:on': 'done'} 

### rpipe_s95 (on)

| arm | n | latency s | CI95 | speedup | LLM req | F1 | achieved sel | rows out | peak KV % | peak waiting | preempt |
|---|---|---|---|---|---|---|---|---|---|---|---|
| kalypso_per_tuple | 5 | 20.91 [20.57, 21.73] | [20.57, 21.73] | - | 1860 | 0.61 | 0.274 | 878 | 0.9 | 1596 | 0 |
| kalypso_pipe:4,1 | 5 | 29.33 [28.66, 30.01] | [28.66, 30.01] | 0.71x ✗ | 2457 | 0.75 | 0.372 | 1191 | 1.1 | 1889 | 0 |
| kalypso_pipe:1,4 | 5 | 13.75 [13.55, 14.45] | [13.55, 14.45] | 1.52x ✓ | 540 | 0.53 | 0.329 | 1054 | 7.1 | 390 | 0 |
| kalypso_pipe:4,4 | 5 | 19.28 [19.08, 19.86] | [19.08, 19.86] | 1.07x ✓ | 633 | 0.64 | 0.461 | 1474 | 7.1 | 575 | 0 |

### rsel95 (on)

| arm | n | latency s | CI95 | speedup | LLM req | F1 | achieved sel | rows out | peak KV % | peak waiting | preempt |
|---|---|---|---|---|---|---|---|---|---|---|---|
| kalypso_per_tuple | 5 | 4.02 [3.90, 4.08] | [3.90, 4.08] | - | 400 | 0.77 | 0.605 | 242 | 1.0 | 346 | 0 |
| kalypso_block:4 | 5 | 2.13 [2.11, 2.18] | [2.11, 2.18] | 1.88x ✓ | 100 | 0.88 | 0.752 | 301 | 4.5 | 65 | 0 |
| kalypso_block:8 | 5 | 2.31 [2.27, 2.33] | [2.27, 2.33] | 1.73x ✓ | 50 | 0.88 | 0.752 | 301 | 3.9 | 42 | 0 |

### Trace trace-on

42 queries (42 complete); KV admission counters {'kv_admit_rejected': 200, 'kv_admit_rejected:stage2': 200}; plan time median 0.0874 s; not attributed: 42 queries vs 0 arms x 6 trials (each arm may issue more than one query).

| stage:k | out/in median [min,max] | partial blocks max | forced launches max | retries max | wait p95 s median | truncated max | block fallbacks max |
|---|---|---|---|---|---|---|---|
| stage1:k1 | 17.600 [0.605, 17.600] | - | 0 | 0 | 0.057 | 0 | 0 |
| stage1:k4 | 24.320 [0.752, 24.320] | 0 | 0 | 0 | 0.038 | 0 | 0 |
| stage1:k8 | 0.752 [0.752, 0.752] | 0 | 0 | 0 | 0.119 | 0 | 0 |
| stage2:k1 | 0.494 [0.490, 0.499] | - | 0 | 0 | 3.196 | 0 | 0 |
| stage2:k4 | 0.602 [0.599, 0.607] | 0 | 0 | 0 | 0.971 | 0 | 0 |

- vllm-metrics-on: 1081 samples, KV % max 7.1, waiting max 1896, preemptions 0.0

## E2-filter

status: {'movie_positive-n1000:on': 'done'} 

### movie_positive-n1000 (on)

| arm | n | latency s | CI95 | speedup | LLM req | F1 | achieved sel | rows out | peak KV % | peak waiting | preempt |
|---|---|---|---|---|---|---|---|---|---|---|---|
| kalypso_per_tuple | 5 | 10.19 [10.05, 10.52] | [10.05, 10.52] | - | 1000 | 0.77 | 0.478 | 478 | 1.0 | 892 | 0 |
| kalypso_block:2 | 5 | 5.85 [5.77, 5.93] | [5.77, 5.93] | 1.75x ✓ | 500 | 0.85 | 0.574 | 574 | 3.4 | 446 | 0 |
| kalypso_block:4 | 5 | 4.82 [4.80, 4.89] | [4.80, 4.89] | 2.11x ✓ | 250 | 0.88 | 0.595 | 595 | 7.1 | 214 | 0 |
| kalypso_block:8 | 5 | 4.59 [4.55, 4.64] | [4.55, 4.64] | 2.23x ✓ | 125 | 0.87 | 0.599 | 599 | 9.5 | 104 | 0 |
| kalypso_block:16 | 5 | 5.43 [5.38, 5.48] | [5.38, 5.48] | 1.89x ✓ | 63 | 0.87 | 0.594 | 594 | 9.1 | 51 | 0 |
| kalypso_block:32 | 5 | 7.47 [7.44, 7.54] | [7.44, 7.54] | 1.36x ✓ | 32 | 0.87 | 0.594 | 594 | 8.9 | 29 | 0 |
| kalypso_block:64 | 5 | 10.98 [10.90, 11.07] | [10.90, 11.07] | 0.93x ✗ | 16 | 0.85 | 0.567 | 567 | 8.5 | 13 | 0 |

### Trace trace-on

42 queries (42 complete); KV admission counters none; plan time median 0.0274 s; by order (counts match).

| stage:k | out/in median [min,max] | partial blocks max | forced launches max | retries max | wait p95 s median | truncated max | block fallbacks max |
|---|---|---|---|---|---|---|---|
| stage1:k1 | 0.478 [0.478, 0.478] | - | 0 | 0 | 0.569 | 2 | 0 |
| stage1:k16 | 0.594 [0.594, 0.594] | 1 | 0 | 0 | 0.266 | 0 | 0 |
| stage1:k2 | 0.574 [0.574, 0.574] | 0 | 0 | 0 | 0.466 | 0 | 0 |
| stage1:k32 | 0.594 [0.594, 0.594] | 1 | 0 | 0 | 0.244 | 0 | 0 |
| stage1:k4 | 0.595 [0.595, 0.596] | 0 | 0 | 0 | 0.358 | 0 | 0 |
| stage1:k64 | 0.567 [0.567, 0.567] | 1 | 0 | 0 | 0.242 | 0 | 0 |
| stage1:k8 | 0.599 [0.599, 0.599] | 0 | 0 | 0 | 0.289 | 0 | 0 |

- vllm-metrics-on: 579 samples, KV % max 9.6, waiting max 928, preemptions 0.0

## E2-join

status: {'movie_fanout50:on': 'done'} 

### movie_fanout50 (on)

| arm | n | latency s | CI95 | speedup | LLM req | F1 | achieved sel | rows out | peak KV % | peak waiting | preempt |
|---|---|---|---|---|---|---|---|---|---|---|---|
| kalypso_per_tuple | 5 | 11.25 [10.79, 11.43] | [10.79, 11.43] | - | 1024 | 0.79 | 0.503 | 515 | 0.8 | 926 | 0 |
| kalypso_block:2 | 5 | 7.49 [7.21, 7.53] | [7.21, 7.53] | 1.50x ✓ | 512 | 0.74 | 0.589 | 603 | 3.4 | 471 | 0 |
| kalypso_block:4 | 5 | 7.27 [7.13, 7.45] | [7.13, 7.45] | 1.53x ✓ | 256 | 0.83 | 0.558 | 571 | 7.1 | 234 | 0 |
| kalypso_block:8 | 5 | 7.61 [7.44, 7.72] | [7.44, 7.72] | 1.47x ✓ | 128 | 0.70 | 0.436 | 446 | 14.4 | 116 | 0 |
| kalypso_block:16 | 5 | 8.72 [8.62, 8.81] | [8.62, 8.81] | 1.28x ✓ | 64 | 0.54 | 0.232 | 238 | 17.3 | 59 | 0 |
| kalypso_block:32 | 5 | 11.16 [11.05, 11.30] | [11.05, 11.30] | 1.00x | 32 | 0.40 | 0.172 | 176 | 17.5 | 29 | 0 |

### Trace trace-on

36 queries (36 complete); KV admission counters none; plan time median 0.0433 s; by order (counts match).

| stage:k | out/in median [min,max] | partial blocks max | forced launches max | retries max | wait p95 s median | truncated max | block fallbacks max |
|---|---|---|---|---|---|---|---|
| stage1:k1 | 0.503 [0.503, 0.503] | - | 0 | 0 | 0.654 | 0 | 0 |
| stage1:k16 | 0.232 [0.227, 0.232] | 0 | 0 | 0 | 0.387 | 0 | 0 |
| stage1:k2 | 0.589 [0.588, 0.589] | 0 | 0 | 0 | 0.600 | 0 | 0 |
| stage1:k32 | 0.172 [0.172, 0.172] | 0 | 0 | 0 | 0.365 | 0 | 0 |
| stage1:k4 | 0.558 [0.558, 0.561] | 0 | 0 | 0 | 0.486 | 0 | 0 |
| stage1:k8 | 0.436 [0.431, 0.436] | 0 | 0 | 0 | 0.423 | 0 | 0 |

- vllm-metrics-on: 618 samples, KV % max 17.5, waiting max 958, preemptions 0.0

## E2-map

status: {'movie_map-n200:on': 'done'} 

### movie_map-n200 (on)

| arm | n | latency s | CI95 | speedup | LLM req | F1 | achieved sel | rows out | peak KV % | peak waiting | preempt |
|---|---|---|---|---|---|---|---|---|---|---|---|
| kalypso_per_tuple | 5 | 4.14 [3.52, 4.15] | [3.52, 4.15] | - | 200 | - | - | 200 | 5.0 | 143 | 0 |
| kalypso_block:2 | 5 | 4.08 [4.06, 4.08] | [4.06, 4.08] | 1.02x | 100 | - | - | 200 | 3.9 | 54 | 0 |
| kalypso_block:4 | 5 | 5.84 [5.83, 5.85] | [5.83, 5.85] | 0.71x ✗ | 50 | - | - | 200 | 3.2 | 36 | 0 |
| kalypso_block:8 | 5 | 8.79 [8.77, 8.82] | [8.77, 8.82] | 0.47x ✗ | 26 | - | - | 200 | 2.9 | 18 | 0 |

### Trace trace-on

24 queries (24 complete); KV admission counters none; plan time median 0.0672 s; by order (counts match).

| stage:k | out/in median [min,max] | partial blocks max | forced launches max | retries max | wait p95 s median | truncated max | block fallbacks max |
|---|---|---|---|---|---|---|---|
| stage1:k1 | 1.000 [1.000, 1.000] | - | 0 | 0 | 0.089 | 0 | 0 |
| stage1:k2 | 1.000 [1.000, 1.000] | 0 | 0 | 0 | 0.162 | 0 | 0 |
| stage1:k4 | 1.000 [1.000, 1.000] | 0 | 0 | 0 | 0.142 | 0 | 0 |
| stage1:k8 | 1.000 [1.000, 1.000] | 0 | 0 | 0 | 0.131 | 0 | 1 |

- vllm-metrics-on: 301 samples, KV % max 4.9, waiting max 180, preemptions 0.0

## E3-load

status: {'rpipe_s50:on': 'done'} 

### rpipe_s50 (L0)

| arm | n | latency s | CI95 | speedup | LLM req | F1 | achieved sel | rows out | peak KV % | peak waiting | preempt |
|---|---|---|---|---|---|---|---|---|---|---|---|
| kalypso_per_tuple | 5 | 10.75 [10.27, 10.83] | [10.27, 10.83] | - | 996 | 0.63 | 0.139 | 445 | 1.0 | 785 | 0 |
| kalypso_pipe:1,4 | 5 | 7.90 [7.55, 7.92] | [7.55, 7.92] | 1.37x ✓ | 324 | 0.55 | 0.167 | 534 | 7.1 | 196 | 0 |
| kalypso_pipe:4,4 | 5 | 10.18 [9.86, 10.23] | [9.86, 10.23] | 1.05x ✓ | 313 | 0.62 | 0.221 | 708 | 7.0 | 273 | 0 |

### rpipe_s50 (L32)

| arm | n | latency s | CI95 | speedup | LLM req | F1 | achieved sel | rows out | peak KV % | peak waiting | preempt |
|---|---|---|---|---|---|---|---|---|---|---|---|
| kalypso_per_tuple | 5 | 11.39 [11.12, 11.46] | [11.12, 11.46] | - | 1028 | 0.63 | 0.139 | 445 | 4.4 | 786 | 0 |
| kalypso_pipe:1,4 | 5 | 2.12 [2.09, 2.17] | [2.09, 2.17] | 5.32x ✓ | 324 | 0.54 | 0.167 | 535 | 20.7 | 7 | 0 |
| kalypso_pipe:4,4 | 5 | 3.63 [3.61, 3.65] | [3.61, 3.65] | 3.12x ✓ | 345 | 0.62 | 0.222 | 711 | 21.0 | 64 | 0 |

### rpipe_s50 (L128)

| arm | n | latency s | CI95 | speedup | LLM req | F1 | achieved sel | rows out | peak KV % | peak waiting | preempt |
|---|---|---|---|---|---|---|---|---|---|---|---|
| kalypso_per_tuple | 5 | 12.53 [12.06, 12.66] | [12.06, 12.66] | - | 1022 | 0.63 | 0.139 | 445 | 10.9 | 832 | 0 |
| kalypso_pipe:1,4 | 5 | 10.28 [9.80, 10.36] | [9.80, 10.36] | 1.21x ✓ | 424 | 0.54 | 0.168 | 536 | 12.7 | 211 | 0 |
| kalypso_pipe:4,4 | 5 | 5.29 [5.26, 5.40] | [5.26, 5.40] | 2.35x ✓ | 313 | 0.62 | 0.222 | 709 | 21.7 | 185 | 0 |

### Trace trace-on

48 queries (48 complete); KV admission counters none; plan time median 0.0824 s; not attributed: 48 queries vs 0 arms x 6 trials (each arm may issue more than one query).

| stage:k | out/in median [min,max] | partial blocks max | forced launches max | retries max | wait p95 s median | truncated max | block fallbacks max |
|---|---|---|---|---|---|---|---|
| stage1:k1 | 8.960 [8.960, 8.960] | - | 0 | 0 | 0.047 | 0 | 0 |
| stage1:k4 | 11.520 [11.520, 11.520] | 0 | 0 | 0 | 0.031 | 0 | 0 |
| stage2:k1 | 0.497 [0.497, 0.499] | - | 0 | 0 | 0.146 | 0 | 0 |
| stage2:k4 | 0.607 [0.594, 0.618] | 0 | 0 | 0 | 0.367 | 0 | 0 |

- vllm-metrics-on: 1736 samples, KV % max 22.0, waiting max 832, preemptions 0.0

## E3-load-v2

status: {'rpipe_s50:on': 'done'} 

prefix-cache resets confirmed before 45 of 45 trials

### rpipe_s50 (L0)

| arm | n | latency s | CI95 | speedup | LLM req | F1 | achieved sel | rows out | peak KV % | peak waiting | preempt |
|---|---|---|---|---|---|---|---|---|---|---|---|
| kalypso_per_tuple | 5 | 10.03 [9.96, 10.63] | [9.96, 10.63] | - | 996 | 0.63 | 0.139 | 445 | 1.0 | 757 | 0 |
| kalypso_pipe:1,4 | 5 | 7.53 [7.32, 7.99] | [7.32, 7.99] | 1.35x ✓ | 324 | 0.54 | 0.167 | 534 | 7.1 | 191 | 0 |
| kalypso_pipe:4,4 | 5 | 9.73 [9.68, 10.09] | [9.68, 10.09] | 1.03x | 313 | 0.62 | 0.220 | 704 | 7.0 | 267 | 0 |

### rpipe_s50 (L32)

| arm | n | latency s | CI95 | speedup | LLM req | F1 | achieved sel | rows out | peak KV % | peak waiting | preempt |
|---|---|---|---|---|---|---|---|---|---|---|---|
| kalypso_per_tuple | 5 | 10.93 [10.58, 11.24] | [10.58, 11.24] | - | 996 | 0.63 | 0.139 | 445 | 4.3 | 763 | 0 |
| kalypso_pipe:1,4 | 5 | 7.77 [7.52, 7.81] | [7.52, 7.81] | 1.43x ✓ | 324 | 0.55 | 0.166 | 531 | 10.1 | 193 | 0 |
| kalypso_pipe:4,4 | 5 | 11.16 [10.55, 11.77] | [10.55, 11.77] | 1.00x | 345 | 0.62 | 0.219 | 702 | 10.4 | 263 | 0 |

### rpipe_s50 (L128)

| arm | n | latency s | CI95 | speedup | LLM req | F1 | achieved sel | rows out | peak KV % | peak waiting | preempt |
|---|---|---|---|---|---|---|---|---|---|---|---|
| kalypso_per_tuple | 5 | 11.65 [11.00, 12.39] | [11.00, 12.39] | - | 1096 | 0.63 | 0.139 | 445 | 11.7 | 785 | 0 |
| kalypso_pipe:1,4 | 5 | 10.14 [10.07, 10.92] | [10.07, 10.92] | 1.13x ✓ | 424 | 0.55 | 0.167 | 534 | 17.6 | 195 | 0 |
| kalypso_pipe:4,4 | 5 | 12.11 [11.87, 12.54] | [11.87, 12.54] | 0.97x ✗ | 413 | 0.62 | 0.219 | 702 | 12.4 | 269 | 0 |

### Trace trace-on

48 queries (48 complete); KV admission counters none; plan time median 0.0907 s; not attributed: 48 queries vs 0 arms x 6 trials (each arm may issue more than one query).

| stage:k | out/in median [min,max] | partial blocks max | forced launches max | retries max | wait p95 s median | truncated max | block fallbacks max |
|---|---|---|---|---|---|---|---|
| stage1:k1 | 8.960 [8.960, 8.960] | - | 0 | 0 | 0.054 | 0 | 0 |
| stage1:k4 | 11.520 [11.520, 11.520] | 0 | 0 | 0 | 0.036 | 0 | 0 |
| stage2:k1 | 0.497 [0.494, 0.498] | - | 0 | 0 | 0.167 | 0 | 0 |
| stage2:k4 | 0.603 [0.593, 0.613] | 0 | 0 | 0 | 0.230 | 0 | 0 |

- vllm-metrics-on: 4226 samples, KV % max 17.7, waiting max 844, preemptions 0.0

