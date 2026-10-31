# Cost model

Fitted on 159 single-operator arms from 43 runs (cnli_join_high, cnli_join_low, cnli_sel10, cnli_sel50, cnli_sel90, movie_fanout10, movie_fanout25, movie_fanout50, movie_fanout75, movie_fanout90, movie_map, movie_opposite, movie_positive, movie_sel10, movie_sel25, movie_sel50, movie_sel75, movie_sel90, pipe_filter_join_s10_join, pipe_filter_join_s50_join, pipe_filter_join_s90_join, pipe_filter_map_map); pipeline composition on 32 pipeline arms.

Latency `L = g + alpha P + (beta + tail[kind]) W D + gamma C D + delta C p^2` (P computed prompt tokens, W waves, D decode steps per call, C calls, p prompt tokens per call):

- g = -0.044 s; alpha = 0.789 s per 10k computed tokens (12,668 tokens/s); beta = 24.3 ms per decode step of a wave; tail[map] = 14.4 ms; gamma = 0.102 ms per generated token; delta = 1.83 ms per call per (1k prompt tokens)^2.
- Pipelines: L = g + lambda * sum over stages of (L_i - g), lambda = 1.00.

| Error | Mean | Max |
|---|---|---|
| Latency from measured tokens | 3.3% | 31.0% |
| Full model (tokens predicted too) | 4.1% | 34.3% |
| Full model, leave one workload out | 4.2% | 32.3% |
| Pipelines, measured filter output | 4.6% | 11.5% |
| Pipelines, full model (filter output from rho) | 13.8% | 57.8% |

Per operator kind (tokens):

| Kind | a_pt | q_pt | d_pt | e | o | c_left | q | d0 | d1 |
|---|---|---|---|---|---|---|---|---|---|
| filter | 84.0 | 133.0 | 2.96 | 3.6 | 40.4 | 0.00 | 67.9 | -2.86 | 4.48 |
| join | 96.0 | 156.8 | 2.30 | 8.0 | 40.5 | 1.00 | 90.9 | -0.53 | 4.14 |
| map | 80.2 | 128.0 | 20.13 | 12.4 | 49.7 | 0.00 | 71.6 | 63.32 | 14.74 |

Accuracy caps (largest k whose mean ΔF1 across runs stayed within 0.02; maps: mean similarity at least 0.8) and rows a block filter passes against per-tuple (rho). Long rows: items over 512 tokens.

| Kind | Regime | Measured k | Cap k | rho by k |
|---|---|---|---|---|
| filter | short | [1, 4, 6, 8, 11, 16, 20] | 20 | 4: 1.26, 6: 1.28, 8: 1.26, 11: 1.21, 16: 1.21, 20: 1.26 |
| filter | long | [1, 2, 4, 8] | 1 | 2: 0.58, 4: 0.71, 8: 0.59 |
| join | short | [1, 2, 4, 6, 11] | 1 | 2: 1.09, 4: 1.27, 6: 1.12, 11: 0.66 |
| join | long | [1, 2, 4, 8] | 1 | 2: 0.96, 4: 1.49, 8: 1.39 |
| map | short | [1, 2, 4, 6, 8, 16] | 1 | – |

## Choice per measured single-operator run

Chosen among the sizes the run measured. Best = fastest measured arm that kept quality. Latency-only = the choice with the accuracy cap off, scored against the fastest measured arm. Free = the choice over every k up to the caps.

| Run | Regime | Items | Chosen k | Best k | Regret | Chosen kept quality | Latency-only k | Fastest k | Latency regret | Free k |
|---|---|---|---|---|---|---|---|---|---|---|
| block-size/movie_opposite-n32-b11-Qwen2.5-7B-Instruct | short | 1024 | 1 | 1 | 0.0% | yes | 11 | 11 | 0.0% | 1 |
| block-size/movie_opposite-n32-b2-Qwen2.5-7B-Instruct | short | 1024 | 1 | 2 | 40.1% | yes | 2 | 2 | 0.0% | 1 |
| block-size/movie_opposite-n32-b6-Qwen2.5-7B-Instruct | short | 1024 | 1 | 6 | 33.6% | yes | 6 | 6 | 0.0% | 1 |
| block-size/movie_positive-n500-b11 | short | 500 | 11 | 11 | 0.0% | yes | 11 | 11 | 0.0% | 5 |
| block-size/movie_positive-n500-b16 | short | 500 | 16 | 16 | 0.0% | yes | 16 | 16 | 0.0% | 5 |
| block-size/movie_positive-n500-b20 | short | 500 | 20 | 20 | 0.0% | yes | 20 | 20 | 0.0% | 5 |
| block-size/movie_positive-n500-b4 | short | 500 | 4 | 4 | 0.0% | yes | 4 | 4 | 0.0% | 5 |
| block-size/movie_positive-n500-b6 | short | 500 | 6 | 6 | 0.0% | yes | 6 | 6 | 0.0% | 5 |
| block-size/movie_positive-n500-b8 | short | 500 | 8 | 8 | 0.0% | yes | 8 | 8 | 0.0% | 5 |
| block-size/movie_sel10-b4-6-8-16-Qwen2.5-7B-Instruct | short | 400 | 4 | 4 | 0.0% | yes | 4 | 4 | 0.0% | 4 |
| block-size/movie_sel25-b4-6-8-16-Qwen2.5-7B-Instruct | short | 400 | 4 | 4 | 0.0% | yes | 4 | 4 | 0.0% | 4 |
| block-size/movie_sel50-b4-6-8-16-Qwen2.5-7B-Instruct | short | 400 | 4 | 4 | 0.0% | yes | 4 | 4 | 0.0% | 4 |
| block-size/movie_sel75-b4-6-8-16-Qwen2.5-7B-Instruct | short | 400 | 4 | 4 | 0.0% | yes | 4 | 4 | 0.0% | 4 |
| block-size/movie_sel90-b4-6-8-16-Qwen2.5-7B-Instruct | short | 400 | 4 | 4 | 0.0% | yes | 4 | 4 | 0.0% | 4 |
| experiments/longrows/cnli-filter-s10/cnli_sel10 | long | 400 | 1 | 1 | 0.0% | yes | 2 | 1 | 1.2% | 1 |
| experiments/longrows/cnli-filter-s50/cnli_sel50 | long | 400 | 1 | 1 | 0.0% | yes | 2 | 2 | 0.0% | 1 |
| experiments/longrows/cnli-filter-s90/cnli_sel90 | long | 400 | 1 | 1 | 0.0% | yes | 2 | 2 | 0.0% | 1 |
| experiments/longrows/cnli-join-high/cnli_join_high | long | 192 | 1 | 1 | 0.0% | yes | 1 | 1 | 0.0% | 1 |
| experiments/longrows/cnli-join-low/cnli_join_low | long | 192 | 1 | 1 | 0.0% | yes | 1 | 1 | 0.0% | 1 |
| experiments/map/map-n100/movie_map-n100 | short | 100 | 1 | 1 | 0.0% | yes | 1 | 1 | 0.0% | 1 |
| experiments/map/map-n500/movie_map-n500 | short | 500 | 1 | 1 | 0.0% | yes | 1 | 2 | 14.4% | 1 |
| experiments/pipeline/filter-join-s10/alone-filter | short | 100 | 4 | 4 | 0.0% | yes | 4 | 4 | 0.0% | 2 |
| experiments/pipeline/filter-join-s10/alone-join | short | 224 | 1 | 1 | 0.0% | yes | 2 | 2 | 0.0% | 1 |
| experiments/pipeline/filter-join-s50/alone-filter | short | 100 | 4 | 4 | 0.0% | yes | 4 | 4 | 0.0% | 2 |
| experiments/pipeline/filter-join-s50/alone-join | short | 1536 | 1 | 1 | 0.0% | yes | 6 | 4 | 1.9% | 1 |
| experiments/pipeline/filter-join-s90/alone-filter | short | 100 | 4 | 4 | 0.0% | yes | 4 | 4 | 0.0% | 2 |
| experiments/pipeline/filter-join-s90/alone-join | short | 2944 | 1 | 1 | 0.0% | yes | 6 | 4 | 1.4% | 1 |
| experiments/pipeline/filter-map-blockmap/alone-map | short | 373 | 1 | 1 | 0.0% | yes | 1 | 1 | 0.0% | 1 |
| experiments/pipeline/filter-map/alone-filter | short | 500 | 4 | 4 | 0.0% | yes | 4 | 4 | 0.0% | 5 |
| experiments/single/filter-n100/movie_positive-n100 | short | 100 | 4 | 4 | 0.0% | yes | 4 | 4 | 0.0% | 2 |
| experiments/single/filter-n1000/movie_positive-n1000 | short | 1000 | 6 | 6 | 0.0% | yes | 6 | 6 | 0.0% | 7 |
| experiments/single/filter-n250/movie_positive-n250 | short | 250 | 4 | 4 | 0.0% | yes | 4 | 4 | 0.0% | 3 |
| experiments/single/filter-nall/movie_positive | short | 1865 | 8 | 8 | 0.0% | yes | 8 | 8 | 0.0% | 10 |
| experiments/single/join-fanout10/movie_fanout10 | short | 1024 | 1 | 1 | 0.0% | yes | 4 | 2 | 0.3% | 1 |
| experiments/single/join-fanout25/movie_fanout25 | short | 1024 | 1 | 1 | 0.0% | yes | 4 | 4 | 0.0% | 1 |
| experiments/single/join-fanout50/movie_fanout50 | short | 1024 | 1 | 4 | 49.7% | yes | 4 | 4 | 0.0% | 1 |
| experiments/single/join-fanout75/movie_fanout75 | short | 1024 | 1 | 2 | 50.4% | yes | 4 | 2 | 0.0% | 1 |
| experiments/single/join-fanout90/movie_fanout90 | short | 1024 | 1 | 2 | 50.0% | yes | 4 | 2 | 0.0% | 1 |
| experiments/single/join-n16/movie_opposite-n16 | short | 256 | 1 | 1 | 0.0% | yes | 2 | 2 | 0.0% | 1 |
| experiments/single/join-n48/movie_opposite-n48 | short | 2304 | 1 | 2 | 42.1% | yes | 6 | 2 | 1.5% | 1 |
| experiments/single/join-n64/movie_opposite-n64 | short | 4096 | 1 | 2 | 46.6% | yes | 6 | 2 | 3.2% | 1 |

Over 41 runs: regret mean 7.6%, max 50.4%; latency-only regret mean 0.6%, max 14.4%.

## Pipelines

Rows passed: by the leading filter, from the call count. Stage sum/max: stage latencies predicted at those sizes. Composed: g + lambda * sum. Full: filter output from per-tuple's pass share times rho(k).

| Run | Arm | Measured (s) | Rows passed | Stage sum (s) | Stage max (s) | Composed (s) | Full (s) | ΔF1 |
|---|---|---|---|---|---|---|---|---|
| experiments/pipeline/filter-join-s10/pipe_filter_join_s10 | `kalypso_per_tuple` | 1.79 | 2 | 1.68 | 0.96 | 1.72 | 1.72 | +0.00 |
| experiments/pipeline/filter-join-s10/pipe_filter_join_s10 | `kalypso_pipe:4,4` | 3.02 | 8 | 2.79 | 2.09 | 2.83 | 1.64 | +0.11 |
| experiments/pipeline/filter-join-s10/pipe_filter_join_s10 | `kalypso_pipe:6,4` | 4.05 | 12 | 3.86 | 2.96 | 3.89 | 1.84 | +0.12 |
| experiments/pipeline/filter-join-s10/pipe_filter_join_s10 | `kalypso_pipe:11,4` | 3.31 | 7 | 3.30 | 1.87 | 3.33 | 2.34 | +0.14 |
| experiments/pipeline/filter-join-s10/pipe_filter_join_s10 | `kalypso_pipe:6,2` | 3.93 | 12 | 3.73 | 2.83 | 3.76 | 1.65 | +0.10 |
| experiments/pipeline/filter-join-s10/pipe_filter_join_s10 | `kalypso_pipe:6,6` | 4.18 | 12 | 4.04 | 3.14 | 4.07 | 2.04 | +0.11 |
| experiments/pipeline/filter-join-s10/pipe_filter_join_s10 | `kalypso_pipe:1,4` | 1.82 | 2 | 1.74 | 0.96 | 1.78 | 1.78 | -0.01 |
| experiments/pipeline/filter-join-s10/pipe_filter_join_s10 | `kalypso_pipe:4,1` | 3.64 | 8 | 3.54 | 2.84 | 3.57 | 1.65 | +0.17 |
| experiments/pipeline/filter-join-s50/pipe_filter_join_s50 | `kalypso_per_tuple` | 10.64 | 28 | 11.06 | 10.08 | 11.06 | 11.06 | +0.00 |
| experiments/pipeline/filter-join-s50/pipe_filter_join_s50 | `kalypso_pipe:4,4` | 10.12 | 36 | 9.46 | 8.74 | 9.47 | 9.35 | -0.01 |
| experiments/pipeline/filter-join-s50/pipe_filter_join_s50 | `kalypso_pipe:6,4` | 10.30 | 37 | 9.88 | 8.96 | 9.89 | 9.61 | -0.00 |
| experiments/pipeline/filter-join-s50/pipe_filter_join_s50 | `kalypso_pipe:11,4` | 10.68 | 36 | 10.19 | 8.74 | 10.20 | 9.75 | -0.01 |
| experiments/pipeline/filter-join-s50/pipe_filter_join_s50 | `kalypso_pipe:6,2` | 10.42 | 37 | 9.83 | 8.91 | 9.84 | 9.55 | -0.05 |
| experiments/pipeline/filter-join-s50/pipe_filter_join_s50 | `kalypso_pipe:6,6` | 10.59 | 37 | 9.69 | 8.77 | 9.69 | 9.39 | -0.04 |
| experiments/pipeline/filter-join-s50/pipe_filter_join_s50 | `kalypso_pipe:1,4` | 7.78 | 28 | 7.55 | 6.57 | 7.57 | 7.57 | -0.09 |
| experiments/pipeline/filter-join-s50/pipe_filter_join_s50 | `kalypso_pipe:4,1` | 14.54 | 36 | 13.68 | 12.96 | 13.68 | 13.47 | +0.10 |
| experiments/pipeline/filter-join-s90/pipe_filter_join_s90 | `kalypso_per_tuple` | 20.19 | 58 | 21.88 | 20.90 | 21.84 | 21.84 | +0.00 |
| experiments/pipeline/filter-join-s90/pipe_filter_join_s90 | `kalypso_pipe:4,4` | 19.12 | 77 | 18.93 | 18.21 | 18.90 | 18.10 | -0.00 |
| experiments/pipeline/filter-join-s90/pipe_filter_join_s90 | `kalypso_pipe:6,4` | 17.62 | 76 | 18.90 | 17.98 | 18.87 | 18.44 | -0.00 |
| experiments/pipeline/filter-join-s90/pipe_filter_join_s90 | `kalypso_pipe:11,4` | 18.87 | 74 | 18.99 | 17.54 | 18.96 | 18.16 | -0.01 |
| experiments/pipeline/filter-join-s90/pipe_filter_join_s90 | `kalypso_pipe:6,2` | 17.63 | 76 | 19.02 | 18.10 | 18.99 | 18.55 | -0.07 |
| experiments/pipeline/filter-join-s90/pipe_filter_join_s90 | `kalypso_pipe:6,6` | 17.93 | 76 | 18.86 | 17.95 | 18.84 | 18.37 | -0.03 |
| experiments/pipeline/filter-join-s90/pipe_filter_join_s90 | `kalypso_pipe:1,4` | 14.08 | 58 | 14.58 | 13.60 | 14.57 | 14.57 | -0.08 |
| experiments/pipeline/filter-join-s90/pipe_filter_join_s90 | `kalypso_pipe:4,1` | 28.60 | 77 | 28.45 | 27.73 | 28.39 | 27.11 | +0.11 |
| experiments/pipeline/filter-map-blockmap/pipe_filter_map | `kalypso_per_tuple` | 7.90 | 235 | 8.13 | 4.83 | 8.14 | 8.14 | +0.00 |
| experiments/pipeline/filter-map-blockmap/pipe_filter_map | `kalypso_pipe:1,4` | 10.88 | 236 | 11.31 | 6.48 | 11.31 | 11.31 | +0.00 |
| experiments/pipeline/filter-map-blockmap/pipe_filter_map | `kalypso_pipe:6,4` | 8.94 | 292 | 9.32 | 6.90 | 9.32 | 9.38 | +0.10 |
| experiments/pipeline/filter-map-blockmap/pipe_filter_map | `kalypso_pipe:6,8` | 12.71 | 296 | 11.25 | 8.83 | 11.25 | 11.28 | +0.10 |
| experiments/pipeline/filter-map/pipe_filter_map | `kalypso_per_tuple` | 7.71 | 235 | 8.13 | 4.83 | 8.14 | 8.14 | +0.00 |
| experiments/pipeline/filter-map/pipe_filter_map | `kalypso_pipe:4,1` | 7.29 | 334 | 7.47 | 5.16 | 7.48 | 7.08 | +0.10 |
| experiments/pipeline/filter-map/pipe_filter_map | `kalypso_pipe:6,1` | 6.85 | 296 | 7.16 | 4.74 | 7.17 | 7.21 | +0.10 |
| experiments/pipeline/filter-map/pipe_filter_map | `kalypso_pipe:11,1` | 6.90 | 279 | 7.41 | 4.56 | 7.43 | 7.49 | +0.07 |

| Run | Chosen | Best | Regret | Chosen kept quality | Latency-only | Fastest | Latency regret | Free |
|---|---|---|---|---|---|---|---|---|
| experiments/pipeline/filter-join-s10/pipe_filter_join_s10 | 4,1 | 1,1 | 103.6% | yes | 4,4 | 1,1 | 68.9% | 2,1 |
| experiments/pipeline/filter-join-s50/pipe_filter_join_s50 | 1,1 | 4,4 | 5.1% | yes | 1,4 | 1,4 | 0.0% | 1,1 |
| experiments/pipeline/filter-join-s90/pipe_filter_join_s90 | 1,1 | 6,4 | 14.6% | yes | 1,4 | 1,4 | 0.0% | 1,1 |
| experiments/pipeline/filter-map-blockmap/pipe_filter_map | 1,1 | 1,1 | 0.0% | yes | 1,1 | 1,1 | 0.0% | 2,1 |
| experiments/pipeline/filter-map/pipe_filter_map | 6,1 | 6,1 | 0.0% | yes | 6,1 | 6,1 | 0.0% | 2,1 |

## Every single-operator arm

| Run | Arm | Measured (s) | From tokens (s) | Full (s) | Computed tokens | Predicted | Decode steps | Predicted |
|---|---|---|---|---|---|---|---|---|
| block-size/movie_opposite-n32-b11-Qwen2.5-7B-Instruct | `kalypso_per_tuple` | 10.73 | 11.47 | 11.51 | 138,845 | 139,214 | 2.2 | 2.3 |
| block-size/movie_opposite-n32-b11-Qwen2.5-7B-Instruct | `kalypso_block:11` | 8.52 | 8.72 | 8.68 | 89,144 | 88,912 | 45.5 | 45.0 |
| block-size/movie_opposite-n32-b2-Qwen2.5-7B-Instruct | `kalypso_per_tuple` | 11.24 | 11.47 | 11.51 | 138,845 | 139,214 | 2.2 | 2.3 |
| block-size/movie_opposite-n32-b2-Qwen2.5-7B-Instruct | `kalypso_block:2` | 8.03 | 7.90 | 7.88 | 89,594 | 89,705 | 8.0 | 7.8 |
| block-size/movie_opposite-n32-b6-Qwen2.5-7B-Instruct | `kalypso_per_tuple` | 11.05 | 11.47 | 11.51 | 138,845 | 139,214 | 2.2 | 2.3 |
| block-size/movie_opposite-n32-b6-Qwen2.5-7B-Instruct | `kalypso_block:6` | 8.28 | 8.09 | 8.12 | 88,939 | 89,058 | 23.9 | 24.3 |
| block-size/movie_positive-n500-b11 | `kalypso_per_tuple` | 4.77 | 4.83 | 4.83 | 57,728 | 57,726 | 3.0 | 3.0 |
| block-size/movie_positive-n500-b11 | `kalypso_block:11` | 2.83 | 2.81 | 2.85 | 19,248 | 19,400 | 45.4 | 46.5 |
| block-size/movie_positive-n500-b16 | `kalypso_per_tuple` | 4.69 | 4.83 | 4.83 | 57,728 | 57,726 | 3.0 | 3.0 |
| block-size/movie_positive-n500-b16 | `kalypso_block:16` | 3.38 | 3.36 | 3.37 | 18,634 | 18,834 | 69.3 | 68.9 |
| block-size/movie_positive-n500-b20 | `kalypso_per_tuple` | 4.71 | 4.83 | 4.83 | 57,728 | 57,726 | 3.0 | 3.0 |
| block-size/movie_positive-n500-b20 | `kalypso_block:20` | 3.91 | 3.87 | 3.78 | 18,323 | 18,551 | 91.0 | 86.8 |
| block-size/movie_positive-n500-b4 | `kalypso_per_tuple` | 4.55 | 4.83 | 4.83 | 57,728 | 57,726 | 3.0 | 3.0 |
| block-size/movie_positive-n500-b4 | `kalypso_block:4` | 2.28 | 2.38 | 2.31 | 23,065 | 22,594 | 16.0 | 15.1 |
| block-size/movie_positive-n500-b6 | `kalypso_per_tuple` | 4.85 | 4.83 | 4.83 | 57,728 | 57,726 | 3.0 | 3.0 |
| block-size/movie_positive-n500-b6 | `kalypso_block:6` | 2.43 | 2.41 | 2.41 | 20,991 | 20,936 | 23.8 | 24.0 |
| block-size/movie_positive-n500-b8 | `kalypso_per_tuple` | 4.56 | 4.83 | 4.83 | 57,728 | 57,726 | 3.0 | 3.0 |
| block-size/movie_positive-n500-b8 | `kalypso_block:8` | 2.43 | 2.52 | 2.57 | 19,935 | 20,087 | 31.8 | 33.0 |
| block-size/movie_positive-n500-controls | `kalypso_per_tuple` | 4.90 | 4.83 | 4.83 | 57,728 | 57,726 | 3.0 | 3.0 |
| block-size/movie_sel10-b4-6-8-16-Qwen2.5-7B-Instruct | `kalypso_per_tuple` | 3.95 | 3.88 | 3.88 | 46,027 | 46,069 | 3.0 | 3.0 |
| block-size/movie_sel10-b4-6-8-16-Qwen2.5-7B-Instruct | `kalypso_block:4` | 2.03 | 1.97 | 1.90 | 18,388 | 17,963 | 16.0 | 15.1 |
| block-size/movie_sel10-b4-6-8-16-Qwen2.5-7B-Instruct | `kalypso_block:6` | 2.08 | 2.03 | 2.03 | 16,690 | 16,629 | 23.9 | 24.0 |
| block-size/movie_sel10-b4-6-8-16-Qwen2.5-7B-Instruct | `kalypso_block:8` | 2.21 | 2.16 | 2.20 | 15,839 | 15,942 | 32.0 | 33.0 |
| block-size/movie_sel10-b4-6-8-16-Qwen2.5-7B-Instruct | `kalypso_block:16` | 3.10 | 3.05 | 3.00 | 14,788 | 14,931 | 71.0 | 68.9 |
| block-size/movie_sel25-b4-6-8-16-Qwen2.5-7B-Instruct | `kalypso_per_tuple` | 3.76 | 3.89 | 3.89 | 46,221 | 46,265 | 3.0 | 3.0 |
| block-size/movie_sel25-b4-6-8-16-Qwen2.5-7B-Instruct | `kalypso_block:4` | 1.95 | 1.98 | 1.92 | 18,565 | 18,159 | 16.0 | 15.1 |
| block-size/movie_sel25-b4-6-8-16-Qwen2.5-7B-Instruct | `kalypso_block:6` | 2.00 | 2.04 | 2.04 | 16,898 | 16,825 | 23.9 | 24.0 |
| block-size/movie_sel25-b4-6-8-16-Qwen2.5-7B-Instruct | `kalypso_block:8` | 2.15 | 2.17 | 2.21 | 16,015 | 16,138 | 32.0 | 33.0 |
| block-size/movie_sel25-b4-6-8-16-Qwen2.5-7B-Instruct | `kalypso_block:16` | 3.05 | 3.06 | 3.02 | 14,980 | 15,127 | 71.0 | 68.9 |
| block-size/movie_sel50-b4-6-8-16-Qwen2.5-7B-Instruct | `kalypso_per_tuple` | 3.75 | 3.90 | 3.90 | 46,369 | 46,337 | 3.0 | 3.0 |
| block-size/movie_sel50-b4-6-8-16-Qwen2.5-7B-Instruct | `kalypso_block:4` | 1.96 | 1.99 | 1.92 | 18,648 | 18,231 | 16.0 | 15.1 |
| block-size/movie_sel50-b4-6-8-16-Qwen2.5-7B-Instruct | `kalypso_block:6` | 2.01 | 2.05 | 2.05 | 16,965 | 16,897 | 23.9 | 24.0 |
| block-size/movie_sel50-b4-6-8-16-Qwen2.5-7B-Instruct | `kalypso_block:8` | 2.15 | 2.18 | 2.22 | 16,097 | 16,210 | 32.0 | 33.0 |
| block-size/movie_sel50-b4-6-8-16-Qwen2.5-7B-Instruct | `kalypso_block:16` | 3.04 | 3.07 | 3.03 | 15,047 | 15,199 | 71.0 | 68.9 |
| block-size/movie_sel75-b4-6-8-16-Qwen2.5-7B-Instruct | `kalypso_per_tuple` | 3.81 | 3.87 | 3.87 | 46,012 | 45,985 | 2.9 | 3.0 |
| block-size/movie_sel75-b4-6-8-16-Qwen2.5-7B-Instruct | `kalypso_block:4` | 1.99 | 1.96 | 1.90 | 18,293 | 17,879 | 16.0 | 15.1 |
| block-size/movie_sel75-b4-6-8-16-Qwen2.5-7B-Instruct | `kalypso_block:6` | 2.02 | 2.02 | 2.02 | 16,610 | 16,545 | 23.9 | 24.0 |
| block-size/movie_sel75-b4-6-8-16-Qwen2.5-7B-Instruct | `kalypso_block:8` | 2.18 | 2.15 | 2.19 | 15,743 | 15,858 | 32.0 | 33.0 |
| block-size/movie_sel75-b4-6-8-16-Qwen2.5-7B-Instruct | `kalypso_block:16` | 3.09 | 3.04 | 3.00 | 14,693 | 14,847 | 71.0 | 68.9 |
| block-size/movie_sel90-b4-6-8-16-Qwen2.5-7B-Instruct | `kalypso_per_tuple` | 3.75 | 3.87 | 3.87 | 46,041 | 46,017 | 2.9 | 3.0 |
| block-size/movie_sel90-b4-6-8-16-Qwen2.5-7B-Instruct | `kalypso_block:4` | 1.93 | 1.96 | 1.90 | 18,325 | 17,911 | 16.0 | 15.1 |
| block-size/movie_sel90-b4-6-8-16-Qwen2.5-7B-Instruct | `kalypso_block:6` | 2.00 | 2.02 | 2.02 | 16,642 | 16,577 | 23.9 | 24.0 |
| block-size/movie_sel90-b4-6-8-16-Qwen2.5-7B-Instruct | `kalypso_block:8` | 2.14 | 2.15 | 2.19 | 15,775 | 15,890 | 32.0 | 33.0 |
| block-size/movie_sel90-b4-6-8-16-Qwen2.5-7B-Instruct | `kalypso_block:16` | 3.02 | 3.04 | 3.00 | 14,725 | 14,879 | 71.0 | 68.9 |
| experiments/longrows/cnli-filter-s10/cnli_sel10 | `kalypso_per_tuple` | 51.21 | 52.05 | 52.05 | 631,716 | 631,949 | 3.0 | 3.0 |
| experiments/longrows/cnli-filter-s10/cnli_sel10 | `kalypso_block:2` | 51.82 | 51.68 | 51.66 | 606,625 | 607,885 | 8.0 | 6.1 |
| experiments/longrows/cnli-filter-s10/cnli_sel10 | `kalypso_block:4` | 54.66 | 54.85 | 54.88 | 602,719 | 603,843 | 16.0 | 15.1 |
| experiments/longrows/cnli-filter-s10/cnli_sel10 | `kalypso_block:8` | 60.86 | 61.64 | 61.76 | 600,739 | 601,822 | 32.0 | 33.0 |
| experiments/longrows/cnli-filter-s50/cnli_sel50 | `kalypso_per_tuple` | 52.24 | 52.06 | 52.06 | 631,859 | 632,097 | 3.0 | 3.0 |
| experiments/longrows/cnli-filter-s50/cnli_sel50 | `kalypso_block:2` | 51.36 | 51.70 | 51.67 | 606,770 | 608,033 | 8.0 | 6.1 |
| experiments/longrows/cnli-filter-s50/cnli_sel50 | `kalypso_block:4` | 55.30 | 54.86 | 54.89 | 602,866 | 603,991 | 16.0 | 15.1 |
| experiments/longrows/cnli-filter-s50/cnli_sel50 | `kalypso_block:8` | 61.85 | 61.66 | 61.77 | 600,887 | 601,970 | 32.0 | 33.0 |
| experiments/longrows/cnli-filter-s90/cnli_sel90 | `kalypso_per_tuple` | 53.04 | 52.07 | 52.08 | 632,014 | 632,257 | 3.0 | 3.0 |
| experiments/longrows/cnli-filter-s90/cnli_sel90 | `kalypso_block:2` | 52.40 | 51.63 | 51.69 | 605,879 | 608,193 | 8.0 | 6.1 |
| experiments/longrows/cnli-filter-s90/cnli_sel90 | `kalypso_block:4` | 56.28 | 55.05 | 54.91 | 606,109 | 604,151 | 15.7 | 15.1 |
| experiments/longrows/cnli-filter-s90/cnli_sel90 | `kalypso_block:8` | 61.76 | 61.67 | 61.79 | 600,986 | 602,130 | 32.0 | 33.0 |
| experiments/longrows/cnli-join-high/cnli_join_high | `kalypso_per_tuple` | 5.58 | 6.00 | 5.88 | 61,052 | 59,832 | 2.5 | 2.3 |
| experiments/longrows/cnli-join-high/cnli_join_high | `kalypso_block:2` | 18.41 | 18.04 | 18.02 | 200,623 | 200,662 | 8.0 | 7.8 |
| experiments/longrows/cnli-join-high/cnli_join_high | `kalypso_block:4` | 26.30 | 26.02 | 25.97 | 275,696 | 275,334 | 16.0 | 16.0 |
| experiments/longrows/cnli-join-high/cnli_join_high | `kalypso_block:8` | 34.51 | 33.09 | 33.03 | 313,238 | 312,671 | 32.0 | 32.6 |
| experiments/longrows/cnli-join-low/cnli_join_low | `kalypso_per_tuple` | 5.65 | 5.97 | 5.87 | 60,754 | 59,663 | 2.5 | 2.3 |
| experiments/longrows/cnli-join-low/cnli_join_low | `kalypso_block:2` | 18.57 | 18.02 | 18.00 | 200,469 | 200,493 | 8.0 | 7.8 |
| experiments/longrows/cnli-join-low/cnli_join_low | `kalypso_block:4` | 26.39 | 26.01 | 25.95 | 275,603 | 275,165 | 16.0 | 16.0 |
| experiments/longrows/cnli-join-low/cnli_join_low | `kalypso_block:8` | 33.30 | 33.07 | 33.00 | 313,095 | 312,502 | 32.0 | 32.6 |
| experiments/map/map-n100/movie_map-n100 | `kalypso_per_tuple` | 2.65 | 1.86 | 1.82 | 11,077 | 11,043 | 21.0 | 20.1 |
| experiments/map/map-n100/movie_map-n100 | `kalypso_block:2` | 3.40 | 3.44 | 4.56 | 6,839 | 6,755 | 67.1 | 92.8 |
| experiments/map/map-n100/movie_map-n100 | `kalypso_block:4` | 5.09 | 5.28 | 5.44 | 5,514 | 5,512 | 118.5 | 122.3 |
| experiments/map/map-n100/movie_map-n100 | `kalypso_block:8` | 7.85 | 8.11 | 7.60 | 5,144 | 4,916 | 192.6 | 181.2 |
| experiments/map/map-n500/movie_map-n500 | `kalypso_per_tuple` | 7.36 | 6.92 | 6.97 | 55,733 | 55,814 | 19.8 | 20.1 |
| experiments/map/map-n500/movie_map-n500 | `kalypso_block:2` | 6.43 | 7.10 | 8.64 | 34,515 | 34,377 | 68.5 | 92.8 |
| experiments/map/map-n500/movie_map-n500 | `kalypso_block:4` | 7.93 | 8.62 | 8.48 | 27,889 | 28,162 | 125.2 | 122.3 |
| experiments/map/map-n500/movie_map-n500 | `kalypso_block:6` | 9.47 | 10.08 | 9.21 | 26,137 | 26,124 | 169.1 | 151.7 |
| experiments/map/map-n500/movie_map-n500 | `kalypso_block:8` | 10.74 | 11.51 | 10.13 | 24,906 | 25,080 | 211.0 | 181.2 |
| experiments/map/map-n500/movie_map-n500 | `kalypso_block:16` | 19.80 | 13.66 | 14.40 | 24,914 | 23,539 | 266.7 | 299.1 |
| experiments/pipeline/filter-join-s10/alone-filter | `kalypso_per_tuple` | 0.95 | 0.96 | 0.96 | 11,412 | 11,366 | 3.0 | 3.0 |
| experiments/pipeline/filter-join-s10/alone-filter | `kalypso_block:4` | 0.73 | 0.74 | 0.71 | 4,472 | 4,340 | 16.0 | 15.1 |
| experiments/pipeline/filter-join-s10/alone-filter | `kalypso_block:6` | 0.85 | 0.89 | 0.90 | 4,064 | 4,016 | 23.5 | 24.0 |
| experiments/pipeline/filter-join-s10/alone-filter | `kalypso_block:11` | 1.21 | 1.31 | 1.43 | 3,723 | 3,733 | 41.6 | 46.5 |
| experiments/pipeline/filter-join-s10/alone-join | `kalypso_per_tuple` | 2.36 | 2.46 | 2.49 | 30,104 | 30,411 | 2.3 | 2.3 |
| experiments/pipeline/filter-join-s10/alone-join | `kalypso_block:2` | 1.74 | 1.74 | 1.74 | 18,875 | 18,906 | 8.0 | 7.8 |
| experiments/pipeline/filter-join-s10/alone-join | `kalypso_block:4` | 1.86 | 1.91 | 1.91 | 18,491 | 18,464 | 16.0 | 16.0 |
| experiments/pipeline/filter-join-s10/alone-join | `kalypso_block:6` | 2.02 | 2.09 | 2.11 | 18,376 | 18,322 | 23.6 | 24.3 |
| experiments/pipeline/filter-join-s50/alone-filter | `kalypso_per_tuple` | 0.96 | 0.98 | 0.98 | 11,641 | 11,599 | 3.0 | 3.0 |
| experiments/pipeline/filter-join-s50/alone-filter | `kalypso_block:4` | 0.74 | 0.76 | 0.72 | 4,701 | 4,573 | 16.0 | 15.1 |
| experiments/pipeline/filter-join-s50/alone-filter | `kalypso_block:6` | 0.86 | 0.91 | 0.92 | 4,293 | 4,249 | 23.5 | 24.0 |
| experiments/pipeline/filter-join-s50/alone-filter | `kalypso_block:11` | 1.24 | 1.33 | 1.45 | 3,952 | 3,966 | 41.6 | 46.5 |
| experiments/pipeline/filter-join-s50/alone-join | `kalypso_per_tuple` | 16.37 | 17.24 | 17.26 | 208,403 | 208,509 | 2.3 | 2.3 |
| experiments/pipeline/filter-join-s50/alone-join | `kalypso_block:2` | 11.09 | 11.46 | 11.42 | 129,328 | 129,235 | 8.0 | 7.8 |
| experiments/pipeline/filter-join-s50/alone-join | `kalypso_block:4` | 11.02 | 11.42 | 11.43 | 126,023 | 126,012 | 16.0 | 16.0 |
| experiments/pipeline/filter-join-s50/alone-join | `kalypso_block:6` | 11.22 | 11.17 | 11.19 | 124,892 | 124,938 | 24.0 | 24.3 |
| experiments/pipeline/filter-join-s90/alone-filter | `kalypso_per_tuple` | 1.01 | 0.98 | 0.98 | 11,616 | 11,573 | 2.9 | 3.0 |
| experiments/pipeline/filter-join-s90/alone-filter | `kalypso_block:4` | 0.82 | 0.76 | 0.72 | 4,674 | 4,547 | 16.0 | 15.1 |
| experiments/pipeline/filter-join-s90/alone-filter | `kalypso_block:6` | 0.97 | 0.91 | 0.92 | 4,266 | 4,223 | 23.5 | 24.0 |
| experiments/pipeline/filter-join-s90/alone-filter | `kalypso_block:11` | 1.38 | 1.32 | 1.45 | 3,925 | 3,940 | 41.6 | 46.5 |
| experiments/pipeline/filter-join-s90/alone-join | `kalypso_per_tuple` | 31.30 | 32.96 | 33.14 | 397,666 | 399,600 | 2.3 | 2.3 |
| experiments/pipeline/filter-join-s90/alone-join | `kalypso_block:2` | 21.55 | 22.06 | 21.95 | 247,895 | 247,451 | 8.0 | 7.8 |
| experiments/pipeline/filter-join-s90/alone-join | `kalypso_block:4` | 21.06 | 21.59 | 21.59 | 241,277 | 241,172 | 16.0 | 16.0 |
| experiments/pipeline/filter-join-s90/alone-join | `kalypso_block:6` | 21.34 | 21.47 | 21.51 | 238,971 | 239,082 | 24.0 | 24.3 |
| experiments/pipeline/filter-map/alone-filter | `kalypso_per_tuple` | 4.67 | 4.83 | 4.83 | 57,728 | 57,726 | 3.0 | 3.0 |
| experiments/pipeline/filter-map/alone-filter | `kalypso_block:4` | 2.43 | 2.38 | 2.31 | 23,065 | 22,594 | 16.0 | 15.1 |
| experiments/pipeline/filter-map/alone-filter | `kalypso_block:6` | 2.48 | 2.41 | 2.41 | 20,991 | 20,936 | 23.8 | 24.0 |
| experiments/pipeline/filter-map/alone-filter | `kalypso_block:11` | 2.90 | 2.81 | 2.85 | 19,248 | 19,400 | 45.4 | 46.5 |
| experiments/pipeline/filter-map/alone-map | `kalypso_per_tuple` | 5.29 | 5.54 | 5.58 | 41,559 | 41,593 | 19.8 | 20.1 |
| experiments/pipeline/filter-map-blockmap/alone-map | `kalypso_per_tuple` | 5.24 | 5.54 | 5.58 | 41,559 | 41,593 | 19.8 | 20.1 |
| experiments/pipeline/filter-map-blockmap/alone-map | `kalypso_block:4` | 6.79 | 7.64 | 7.53 | 20,895 | 21,002 | 124.7 | 122.3 |
| experiments/pipeline/filter-map-blockmap/alone-map | `kalypso_block:8` | 13.32 | 10.51 | 9.33 | 19,265 | 18,665 | 203.5 | 181.2 |
| experiments/single/filter-n100/movie_positive-n100 | `kalypso_per_tuple` | 0.98 | 0.97 | 0.97 | 11,477 | 11,425 | 3.0 | 3.0 |
| experiments/single/filter-n100/movie_positive-n100 | `kalypso_block:4` | 0.80 | 0.75 | 0.71 | 4,532 | 4,399 | 16.0 | 15.1 |
| experiments/single/filter-n100/movie_positive-n100 | `kalypso_block:6` | 0.94 | 0.90 | 0.91 | 4,124 | 4,075 | 23.5 | 24.0 |
| experiments/single/filter-n100/movie_positive-n100 | `kalypso_block:8` | 1.05 | 1.06 | 1.11 | 3,920 | 3,914 | 30.8 | 33.0 |
| experiments/single/filter-n1000/movie_positive-n1000 | `kalypso_per_tuple` | 9.36 | 9.68 | 9.69 | 115,130 | 115,173 | 3.0 | 3.0 |
| experiments/single/filter-n1000/movie_positive-n1000 | `kalypso_block:4` | 4.49 | 4.39 | 4.27 | 45,798 | 44,907 | 16.0 | 15.1 |
| experiments/single/filter-n1000/movie_positive-n1000 | `kalypso_block:6` | 4.34 | 4.26 | 4.25 | 41,627 | 41,552 | 23.9 | 24.0 |
| experiments/single/filter-n1000/movie_positive-n1000 | `kalypso_block:8` | 4.34 | 4.29 | 4.35 | 39,520 | 39,854 | 32.0 | 33.0 |
| experiments/single/filter-n250/movie_positive-n250 | `kalypso_per_tuple` | 2.34 | 2.41 | 2.40 | 28,987 | 28,963 | 3.0 | 3.0 |
| experiments/single/filter-n250/movie_positive-n250 | `kalypso_block:4` | 1.38 | 1.37 | 1.33 | 11,677 | 11,417 | 15.9 | 15.1 |
| experiments/single/filter-n250/movie_positive-n250 | `kalypso_block:6` | 1.49 | 1.48 | 1.48 | 10,605 | 10,568 | 23.8 | 24.0 |
| experiments/single/filter-n250/movie_positive-n250 | `kalypso_block:8` | 1.63 | 1.62 | 1.68 | 10,112 | 10,164 | 31.2 | 33.0 |
| experiments/single/filter-nall/movie_positive | `kalypso_per_tuple` | 18.32 | 18.15 | 18.17 | 214,980 | 215,170 | 3.0 | 3.0 |
| experiments/single/filter-nall/movie_positive | `kalypso_block:4` | 8.50 | 8.31 | 8.09 | 85,825 | 84,155 | 16.0 | 15.1 |
| experiments/single/filter-nall/movie_positive | `kalypso_block:6` | 8.09 | 8.09 | 8.08 | 77,985 | 77,849 | 24.0 | 24.0 |
| experiments/single/filter-nall/movie_positive | `kalypso_block:8` | 7.97 | 7.39 | 7.50 | 74,071 | 74,737 | 31.9 | 33.0 |
| experiments/single/join-fanout10/movie_fanout10 | `kalypso_per_tuple` | 11.05 | 10.92 | 11.07 | 132,090 | 133,683 | 2.2 | 2.3 |
| experiments/single/join-fanout10/movie_fanout10 | `kalypso_block:2` | 7.55 | 7.28 | 7.20 | 81,891 | 81,193 | 8.0 | 7.8 |
| experiments/single/join-fanout10/movie_fanout10 | `kalypso_block:4` | 7.58 | 7.11 | 7.09 | 79,499 | 79,223 | 16.0 | 16.0 |
| experiments/single/join-fanout10/movie_fanout10 | `kalypso_block:6` | 7.78 | 7.26 | 7.26 | 78,687 | 78,569 | 23.9 | 24.3 |
| experiments/single/join-fanout10/movie_fanout10 | `kalypso_block:11` | 8.41 | 7.83 | 7.78 | 78,360 | 77,977 | 45.5 | 45.0 |
| experiments/single/join-fanout25/movie_fanout25 | `kalypso_per_tuple` | 10.83 | 10.85 | 10.99 | 131,110 | 132,721 | 2.2 | 2.3 |
| experiments/single/join-fanout25/movie_fanout25 | `kalypso_block:2` | 7.41 | 7.20 | 7.12 | 80,926 | 80,230 | 8.0 | 7.8 |
| experiments/single/join-fanout25/movie_fanout25 | `kalypso_block:4` | 7.40 | 7.03 | 7.01 | 78,491 | 78,261 | 16.0 | 16.0 |
| experiments/single/join-fanout25/movie_fanout25 | `kalypso_block:6` | 7.51 | 7.18 | 7.19 | 77,679 | 77,607 | 23.9 | 24.3 |
| experiments/single/join-fanout25/movie_fanout25 | `kalypso_block:11` | 8.11 | 7.74 | 7.70 | 77,338 | 77,015 | 45.5 | 45.0 |
| experiments/single/join-fanout50/movie_fanout50 | `kalypso_per_tuple` | 10.43 | 10.59 | 10.71 | 127,656 | 129,167 | 2.3 | 2.3 |
| experiments/single/join-fanout50/movie_fanout50 | `kalypso_block:2` | 7.00 | 6.93 | 6.84 | 77,472 | 76,677 | 8.0 | 7.8 |
| experiments/single/join-fanout50/movie_fanout50 | `kalypso_block:4` | 6.97 | 6.75 | 6.73 | 75,068 | 74,708 | 16.0 | 16.0 |
| experiments/single/join-fanout50/movie_fanout50 | `kalypso_block:6` | 7.15 | 6.90 | 6.90 | 74,257 | 74,054 | 23.9 | 24.3 |
| experiments/single/join-fanout50/movie_fanout50 | `kalypso_block:11` | 7.72 | 7.46 | 7.41 | 73,912 | 73,461 | 45.5 | 45.0 |
| experiments/single/join-fanout75/movie_fanout75 | `kalypso_per_tuple` | 10.74 | 10.59 | 10.71 | 127,652 | 129,167 | 2.3 | 2.3 |
| experiments/single/join-fanout75/movie_fanout75 | `kalypso_block:2` | 7.14 | 6.93 | 6.84 | 77,494 | 76,677 | 8.0 | 7.8 |
| experiments/single/join-fanout75/movie_fanout75 | `kalypso_block:4` | 7.14 | 6.75 | 6.73 | 75,075 | 74,708 | 16.0 | 16.0 |
| experiments/single/join-fanout75/movie_fanout75 | `kalypso_block:6` | 7.35 | 6.90 | 6.90 | 74,284 | 74,054 | 23.9 | 24.3 |
| experiments/single/join-fanout75/movie_fanout75 | `kalypso_block:11` | 7.95 | 7.46 | 7.41 | 73,950 | 73,461 | 45.5 | 45.0 |
| experiments/single/join-fanout90/movie_fanout90 | `kalypso_per_tuple` | 10.54 | 10.68 | 10.77 | 128,550 | 129,997 | 2.4 | 2.3 |
| experiments/single/join-fanout90/movie_fanout90 | `kalypso_block:2` | 7.03 | 7.00 | 6.91 | 78,347 | 77,506 | 8.0 | 7.8 |
| experiments/single/join-fanout90/movie_fanout90 | `kalypso_block:4` | 7.03 | 6.82 | 6.79 | 75,932 | 75,537 | 16.0 | 16.0 |
| experiments/single/join-fanout90/movie_fanout90 | `kalypso_block:6` | 7.17 | 6.97 | 6.97 | 75,137 | 74,883 | 23.9 | 24.3 |
| experiments/single/join-fanout90/movie_fanout90 | `kalypso_block:11` | 7.82 | 7.53 | 7.48 | 74,809 | 74,291 | 45.5 | 45.0 |
| experiments/single/join-n16/movie_opposite-n16 | `kalypso_per_tuple` | 2.69 | 2.80 | 2.81 | 34,210 | 34,306 | 2.3 | 2.3 |
| experiments/single/join-n16/movie_opposite-n16 | `kalypso_block:2` | 2.00 | 1.99 | 1.96 | 21,729 | 21,547 | 8.0 | 7.8 |
| experiments/single/join-n16/movie_opposite-n16 | `kalypso_block:4` | 2.15 | 2.15 | 2.15 | 21,344 | 21,236 | 16.0 | 16.0 |
| experiments/single/join-n16/movie_opposite-n16 | `kalypso_block:6` | 2.32 | 2.34 | 2.35 | 21,246 | 21,134 | 23.8 | 24.3 |
| experiments/single/join-n48/movie_opposite-n48 | `kalypso_per_tuple` | 25.56 | 25.72 | 25.84 | 310,498 | 311,692 | 2.3 | 2.3 |
| experiments/single/join-n48/movie_opposite-n48 | `kalypso_block:2` | 17.99 | 17.74 | 17.74 | 199,274 | 199,997 | 8.0 | 7.8 |
| experiments/single/join-n48/movie_opposite-n48 | `kalypso_block:4` | 18.03 | 17.91 | 17.95 | 198,262 | 198,756 | 16.0 | 16.0 |
| experiments/single/join-n48/movie_opposite-n48 | `kalypso_block:6` | 18.25 | 17.94 | 18.00 | 197,929 | 198,342 | 24.0 | 24.3 |
| experiments/single/join-n64/movie_opposite-n64 | `kalypso_per_tuple` | 45.05 | 45.36 | 45.53 | 546,811 | 548,722 | 2.3 | 2.3 |
| experiments/single/join-n64/movie_opposite-n64 | `kalypso_block:2` | 30.73 | 30.84 | 30.79 | 347,208 | 347,898 | 8.0 | 7.8 |
| experiments/single/join-n64/movie_opposite-n64 | `kalypso_block:4` | 31.24 | 30.68 | 30.72 | 344,089 | 344,567 | 16.0 | 16.0 |
| experiments/single/join-n64/movie_opposite-n64 | `kalypso_block:6` | 31.73 | 30.88 | 30.97 | 342,961 | 343,458 | 24.0 | 24.3 |
