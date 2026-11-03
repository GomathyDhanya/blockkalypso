# Cost model (v3)

Fitted on 173 single-operator arms from 45 runs (cnli_join_high, cnli_join_low, cnli_sel10, cnli_sel25, cnli_sel50, cnli_sel90, movie_fanout10, movie_fanout25, movie_fanout50, movie_fanout75, movie_fanout90, movie_map, movie_opposite, movie_positive, movie_sel10, movie_sel25, movie_sel50, movie_sel75, movie_sel90, pipe_filter_join_s10_join, pipe_filter_join_s50_join, pipe_filter_join_s90_join, pipe_filter_map_map); pipeline composition on 75 pipeline arms. Held out: validate3, validate-jf, validate4.

Latency `L = g[kind] + alpha P + (beta + tail[kind]) W D + gamma C D + delta C p^2` (P computed prompt tokens, W waves, D decode steps per call, C calls, p prompt tokens per call):

- g (fixed cost per operator) = filter -0.057 s, join -0.048 s, map 0.286 s; alpha = 0.792 s per 10k computed tokens (12,634 tokens/s); beta = 24.8 ms per decode step of a wave; tail[map] = 11.0 ms; gamma = 0.100 ms per generated token; delta = 1.77 ms per call per (1k prompt tokens)^2.
- Pipelines: L = lambda * sum over stages of L_i, lambda = 0.98. A leading filter passes rows * (recall(k) s + fpr(k) (1 - s)) for true selectivity s. A per-tuple map reruns a share of its rows when another stage is blocked (by the map's position and size: after:1 12%, after:block 0%, first:1 56%, first:block 46%).

| Error | Mean | Max |
|---|---|---|
| Latency from measured tokens | 3.4% | 33.3% |
| Full model (tokens predicted too) | 4.3% | 35.8% |
| Full model, leave one workload out | 4.8% | 61.4% |
| Pipelines, measured filter output and reruns | 10.9% | 57.8% |
| Pipelines, full model | 13.2% | 87.2% |

Per operator kind (tokens):

| Kind | a_pt | q_pt | d_pt | e | o | c_left | q | d0 | d1 |
|---|---|---|---|---|---|---|---|---|---|
| filter | 84.1 | 133.3 | 2.96 | 3.7 | 39.7 | 0.00 | 52.2 | -2.64 | 4.46 |
| join | 96.0 | 156.8 | 2.30 | 8.2 | 40.0 | 1.00 | 89.8 | -2.61 | 4.57 |
| map | 80.2 | 128.0 | 20.13 | 12.4 | 49.7 | 0.00 | 71.6 | 63.32 | 14.74 |

Accuracy caps, used only by the guarded choice (largest k whose mean ΔF1 across runs stayed within 0.02; maps: mean similarity at least 0.8), and filters' recall and false-positive rate by k. Long rows: items over 512 tokens.

| Kind | Regime | Measured k | Cap k | Recall by k | False-positive rate by k |
|---|---|---|---|---|---|
| filter | short | [1, 4, 6, 8, 11, 16, 20] | 20 | 1: 0.63, 4: 0.79, 6: 0.80, 8: 0.79, 11: 0.74, 16: 0.75, 20: 0.79 | 1: 0.013, 4: 0.029, 6: 0.033, 8: 0.033, 11: 0.034, 16: 0.021, 20: 0.032 |
| filter | long | [1, 2, 3, 4, 8, 16] | 1 | 1: 0.70, 2: 0.46, 3: 0.50, 4: 0.53, 8: 0.43, 16: 0.35 | 1: 0.350, 2: 0.137, 3: 0.210, 4: 0.206, 8: 0.185, 16: 0.210 |
| join | short | [1, 2, 4, 5, 6, 9, 10, 11, 16, 32] | 1 | 1: 0.69, 2: 0.67, 4: 0.79, 5: 0.65, 6: 0.70, 9: 0.54, 10: 0.46, 11: 0.48, 16: 0.29, 32: 0.07 | 1: 0.273, 2: 0.394, 4: 0.432, 5: 0.310, 6: 0.410, 9: 0.150, 10: 0.131, 11: 0.117, 16: 0.084, 32: 0.092 |
| join | long | [1, 2, 4, 8] | 1 | 1: 0.38, 2: 0.30, 4: 0.43, 8: 0.41 | 1: 0.050, 2: 0.096, 4: 0.178, 8: 0.164 |
| map | short | [1, 2, 4, 6, 8, 16] | 1 | – | – |

## Choice per measured single-operator run

Chosen among the sizes the run measured. Algorithm = lowest predicted latency, against the fastest measured arm. Guarded = with the accuracy caps, against the fastest measured arm that kept quality. Free = the algorithm over every size up to the measured ones.

| Run | Regime | Items | Algorithm k | Fastest k | Regret | Guarded k | Best kept-quality k | Guarded regret | Guarded kept quality | Free k |
|---|---|---|---|---|---|---|---|---|---|---|
| block-size/movie_opposite-n32-b11-Qwen2.5-7B-Instruct | short | 1024 | 11 | 11 | 0.0% | 1 | 1 | 0.0% | yes | 2 |
| block-size/movie_opposite-n32-b2-Qwen2.5-7B-Instruct | short | 1024 | 2 | 2 | 0.0% | 1 | 2 | 40.1% | yes | 2 |
| block-size/movie_opposite-n32-b6-Qwen2.5-7B-Instruct | short | 1024 | 6 | 6 | 0.0% | 1 | 6 | 33.6% | yes | 2 |
| block-size/movie_positive-n500-b11 | short | 500 | 11 | 11 | 0.0% | 11 | 11 | 0.0% | yes | 3 |
| block-size/movie_positive-n500-b16 | short | 500 | 16 | 16 | 0.0% | 16 | 16 | 0.0% | yes | 3 |
| block-size/movie_positive-n500-b20 | short | 500 | 20 | 20 | 0.0% | 20 | 20 | 0.0% | yes | 3 |
| block-size/movie_positive-n500-b4 | short | 500 | 4 | 4 | 0.0% | 4 | 4 | 0.0% | yes | 3 |
| block-size/movie_positive-n500-b6 | short | 500 | 6 | 6 | 0.0% | 6 | 6 | 0.0% | yes | 3 |
| block-size/movie_positive-n500-b8 | short | 500 | 8 | 8 | 0.0% | 8 | 8 | 0.0% | yes | 3 |
| block-size/movie_sel10-b4-6-8-16-Qwen2.5-7B-Instruct | short | 400 | 4 | 4 | 0.0% | 4 | 4 | 0.0% | yes | 3 |
| block-size/movie_sel25-b4-6-8-16-Qwen2.5-7B-Instruct | short | 400 | 4 | 4 | 0.0% | 4 | 4 | 0.0% | yes | 3 |
| block-size/movie_sel50-b4-6-8-16-Qwen2.5-7B-Instruct | short | 400 | 4 | 4 | 0.0% | 4 | 4 | 0.0% | yes | 3 |
| block-size/movie_sel75-b4-6-8-16-Qwen2.5-7B-Instruct | short | 400 | 4 | 4 | 0.0% | 4 | 4 | 0.0% | yes | 3 |
| block-size/movie_sel90-b4-6-8-16-Qwen2.5-7B-Instruct | short | 400 | 4 | 4 | 0.0% | 4 | 4 | 0.0% | yes | 3 |
| experiments/longrows/cnli-filter-s10/cnli_sel10 | long | 400 | 1 | 1 | 0.0% | 1 | 1 | 0.0% | yes | 1 |
| experiments/longrows/cnli-filter-s50/cnli_sel50 | long | 400 | 1 | 2 | 1.7% | 1 | 1 | 0.0% | yes | 1 |
| experiments/longrows/cnli-filter-s90/cnli_sel90 | long | 400 | 1 | 2 | 1.2% | 1 | 1 | 0.0% | yes | 1 |
| experiments/longrows/cnli-join-high/cnli_join_high | long | 192 | 1 | 1 | 0.0% | 1 | 1 | 0.0% | yes | 1 |
| experiments/longrows/cnli-join-low/cnli_join_low | long | 192 | 1 | 1 | 0.0% | 1 | 1 | 0.0% | yes | 1 |
| experiments/map/map-n100/movie_map-n100 | short | 100 | 1 | 1 | 0.0% | 1 | 1 | 0.0% | yes | 1 |
| experiments/map/map-n500/movie_map-n500 | short | 500 | 1 | 2 | 14.4% | 1 | 1 | 0.0% | yes | 1 |
| experiments/pipeline/filter-join-s10/alone-filter | short | 100 | 4 | 4 | 0.0% | 4 | 4 | 0.0% | yes | 2 |
| experiments/pipeline/filter-join-s10/alone-join | short | 224 | 2 | 2 | 0.0% | 1 | 1 | 0.0% | yes | 2 |
| experiments/pipeline/filter-join-s50/alone-filter | short | 100 | 4 | 4 | 0.0% | 4 | 4 | 0.0% | yes | 2 |
| experiments/pipeline/filter-join-s50/alone-join | short | 1536 | 2 | 4 | 0.6% | 1 | 1 | 0.0% | yes | 2 |
| experiments/pipeline/filter-join-s90/alone-filter | short | 100 | 4 | 4 | 0.0% | 4 | 4 | 0.0% | yes | 2 |
| experiments/pipeline/filter-join-s90/alone-join | short | 2944 | 2 | 4 | 2.4% | 1 | 1 | 0.0% | yes | 2 |
| experiments/pipeline/filter-map-blockmap/alone-map | short | 373 | 1 | 1 | 0.0% | 1 | 1 | 0.0% | yes | 1 |
| experiments/pipeline/filter-map/alone-filter | short | 500 | 4 | 4 | 0.0% | 4 | 4 | 0.0% | yes | 3 |
| experiments/single/filter-n100/movie_positive-n100 | short | 100 | 4 | 4 | 0.0% | 4 | 4 | 0.0% | yes | 2 |
| experiments/single/filter-n1000/movie_positive-n1000 | short | 1000 | 4 | 6 | 3.5% | 4 | 6 | 3.5% | yes | 4 |
| experiments/single/filter-n250/movie_positive-n250 | short | 250 | 4 | 4 | 0.0% | 4 | 4 | 0.0% | yes | 2 |
| experiments/single/filter-nall/movie_positive | short | 1865 | 8 | 8 | 0.0% | 8 | 8 | 0.0% | yes | 8 |
| experiments/single/join-fanout10/movie_fanout10 | short | 1024 | 2 | 2 | 0.0% | 1 | 1 | 0.0% | yes | 2 |
| experiments/single/join-fanout25/movie_fanout25 | short | 1024 | 2 | 4 | 0.1% | 1 | 1 | 0.0% | yes | 2 |
| experiments/single/join-fanout50/movie_fanout50 | short | 1024 | 2 | 4 | 0.5% | 1 | 4 | 49.7% | yes | 2 |
| experiments/single/join-fanout75/movie_fanout75 | short | 1024 | 2 | 2 | 0.0% | 1 | 2 | 50.4% | yes | 2 |
| experiments/single/join-fanout90/movie_fanout90 | short | 1024 | 2 | 2 | 0.0% | 1 | 2 | 50.0% | yes | 2 |
| experiments/single/join-n16/movie_opposite-n16 | short | 256 | 2 | 2 | 0.0% | 1 | 1 | 0.0% | yes | 2 |
| experiments/single/join-n48/movie_opposite-n48 | short | 2304 | 2 | 2 | 0.0% | 1 | 2 | 42.1% | yes | 2 |
| experiments/single/join-n64/movie_opposite-n64 | short | 4096 | 2 | 2 | 0.0% | 1 | 2 | 46.6% | yes | 2 |
| experiments/validate/cnli-filter-s25/cnli_sel25-n200 | long | 200 | 1 | 1 | 0.0% | 1 | 1 | 0.0% | yes | 1 |
| experiments/validate/join-fanout25-n64/movie_fanout25-n64 | short | 4096 | 2 | 4 | 0.9% | 1 | 1 | 0.0% | yes | 2 |

Over 43 runs: algorithm regret mean 0.6%, max 14.4%; guarded regret mean 7.3%, max 50.4%.

## Pipelines

Rows passed: by the leading filter (filter -> map: rows out; otherwise from the call count). Map reruns: per-row map calls beyond the stages' blocks (reruns of cut-off answers, repairs). Stage sum/max: stage latencies predicted at those sizes and measured reruns. Composed: lambda * sum. Full: rows passed and reruns predicted too.

| Run | Arm | Measured (s) | Rows passed | Map reruns | Stage sum (s) | Stage max (s) | Composed (s) | Full (s) | ΔF1 |
|---|---|---|---|---|---|---|---|---|---|
| experiments/pipeline/filter-join-s10/pipe_filter_join_s10 | `kalypso_per_tuple` | 1.79 | 2 | – | 1.67 | 0.95 | 1.64 | 2.90 | +0.00 |
| experiments/pipeline/filter-join-s10/pipe_filter_join_s10 | `kalypso_pipe:4,4` | 3.02 | 8 | – | 2.79 | 2.08 | 2.74 | 2.80 | +0.11 |
| experiments/pipeline/filter-join-s10/pipe_filter_join_s10 | `kalypso_pipe:6,4` | 4.05 | 12 | – | 3.86 | 2.95 | 3.79 | 3.08 | +0.12 |
| experiments/pipeline/filter-join-s10/pipe_filter_join_s10 | `kalypso_pipe:11,4` | 3.31 | 7 | – | 3.31 | 1.86 | 3.25 | 3.55 | +0.14 |
| experiments/pipeline/filter-join-s10/pipe_filter_join_s10 | `kalypso_pipe:6,2` | 3.93 | 12 | – | 3.68 | 2.77 | 3.61 | 2.89 | +0.10 |
| experiments/pipeline/filter-join-s10/pipe_filter_join_s10 | `kalypso_pipe:6,6` | 4.18 | 12 | – | 4.07 | 3.17 | 4.00 | 3.29 | +0.11 |
| experiments/pipeline/filter-join-s10/pipe_filter_join_s10 | `kalypso_pipe:1,4` | 1.82 | 2 | – | 1.73 | 0.95 | 1.70 | 2.48 | -0.01 |
| experiments/pipeline/filter-join-s10/pipe_filter_join_s10 | `kalypso_pipe:4,1` | 3.64 | 8 | – | 3.54 | 2.84 | 3.48 | 3.63 | +0.17 |
| experiments/pipeline/filter-join-s50/pipe_filter_join_s50 | `kalypso_per_tuple` | 10.64 | 28 | – | 11.06 | 10.09 | 10.87 | 11.87 | +0.00 |
| experiments/pipeline/filter-join-s50/pipe_filter_join_s50 | `kalypso_pipe:4,4` | 10.12 | 36 | – | 9.46 | 8.74 | 9.30 | 10.07 | -0.01 |
| experiments/pipeline/filter-join-s50/pipe_filter_join_s50 | `kalypso_pipe:6,4` | 10.30 | 37 | – | 9.88 | 8.96 | 9.71 | 10.36 | -0.00 |
| experiments/pipeline/filter-join-s50/pipe_filter_join_s50 | `kalypso_pipe:11,4` | 10.68 | 36 | – | 10.20 | 8.74 | 10.02 | 10.34 | -0.01 |
| experiments/pipeline/filter-join-s50/pipe_filter_join_s50 | `kalypso_pipe:6,2` | 10.42 | 37 | – | 9.68 | 8.76 | 9.51 | 10.17 | -0.05 |
| experiments/pipeline/filter-join-s50/pipe_filter_join_s50 | `kalypso_pipe:6,6` | 10.59 | 37 | – | 9.74 | 8.81 | 9.57 | 10.19 | -0.04 |
| experiments/pipeline/filter-join-s50/pipe_filter_join_s50 | `kalypso_pipe:1,4` | 7.78 | 28 | – | 7.54 | 6.57 | 7.41 | 8.04 | -0.09 |
| experiments/pipeline/filter-join-s50/pipe_filter_join_s50 | `kalypso_pipe:4,1` | 14.54 | 36 | – | 13.70 | 12.98 | 13.46 | 14.69 | +0.10 |
| experiments/pipeline/filter-join-s90/pipe_filter_join_s90 | `kalypso_per_tuple` | 20.19 | 58 | – | 21.90 | 20.93 | 21.52 | 21.52 | +0.00 |
| experiments/pipeline/filter-join-s90/pipe_filter_join_s90 | `kalypso_pipe:4,4` | 19.12 | 77 | – | 18.93 | 18.21 | 18.60 | 17.74 | -0.00 |
| experiments/pipeline/filter-join-s90/pipe_filter_join_s90 | `kalypso_pipe:6,4` | 17.62 | 76 | – | 18.91 | 17.99 | 18.58 | 18.05 | -0.00 |
| experiments/pipeline/filter-join-s90/pipe_filter_join_s90 | `kalypso_pipe:11,4` | 18.87 | 74 | – | 19.00 | 17.54 | 18.67 | 17.49 | -0.01 |
| experiments/pipeline/filter-join-s90/pipe_filter_join_s90 | `kalypso_pipe:6,2` | 17.63 | 76 | – | 18.74 | 17.83 | 18.42 | 17.89 | -0.07 |
| experiments/pipeline/filter-join-s90/pipe_filter_join_s90 | `kalypso_pipe:6,6` | 17.93 | 76 | – | 18.97 | 18.05 | 18.64 | 18.09 | -0.03 |
| experiments/pipeline/filter-join-s90/pipe_filter_join_s90 | `kalypso_pipe:1,4` | 14.08 | 58 | – | 14.57 | 13.60 | 14.32 | 14.32 | -0.08 |
| experiments/pipeline/filter-join-s90/pipe_filter_join_s90 | `kalypso_pipe:4,1` | 28.60 | 77 | – | 28.49 | 27.77 | 28.00 | 26.63 | +0.11 |
| experiments/pipeline/filter-map-blockmap/pipe_filter_map | `kalypso_per_tuple` | 7.90 | 235 | 0 | 8.40 | 4.84 | 8.26 | 8.27 | +0.00 |
| experiments/pipeline/filter-map-blockmap/pipe_filter_map | `kalypso_pipe:1,4` | 10.88 | 235 | 0 | 11.28 | 6.44 | 11.09 | 11.82 | +0.00 |
| experiments/pipeline/filter-map-blockmap/pipe_filter_map | `kalypso_pipe:6,4` | 8.94 | 292 | 0 | 9.28 | 6.87 | 9.12 | 9.91 | +0.10 |
| experiments/pipeline/filter-map-blockmap/pipe_filter_map | `kalypso_pipe:6,8` | 12.71 | 292 | 0 | 11.03 | 8.62 | 10.84 | 11.61 | +0.10 |
| experiments/pipeline/filter-map/pipe_filter_map | `kalypso_per_tuple` | 7.71 | 235 | 0 | 8.40 | 4.84 | 8.26 | 8.27 | +0.00 |
| experiments/pipeline/filter-map/pipe_filter_map | `kalypso_pipe:4,1` | 7.29 | 291 | 43 | 8.39 | 6.09 | 8.25 | 8.25 | +0.10 |
| experiments/pipeline/filter-map/pipe_filter_map | `kalypso_pipe:6,1` | 6.85 | 292 | 4 | 8.08 | 5.67 | 7.95 | 8.38 | +0.10 |
| experiments/pipeline/filter-map/pipe_filter_map | `kalypso_pipe:11,1` | 6.90 | 279 | 0 | 7.63 | 4.77 | 7.50 | 8.58 | +0.07 |
| experiments/validate/filter-join-s25/pipe_filter_join_s25 | `kalypso_per_tuple` | 6.58 | 16 | – | 6.69 | 5.73 | 6.58 | 7.95 | +0.00 |
| experiments/validate/filter-join-s25/pipe_filter_join_s25 | `kalypso_block:4` | 7.34 | 26 | – | 6.81 | 6.09 | 6.69 | 6.65 | +0.02 |
| experiments/validate/filter-join-s25/pipe_filter_join_s25 | `kalypso_block:16` | 9.82 | 25 | – | 9.27 | 7.27 | 9.11 | 8.93 | -0.10 |
| experiments/validate/filter-join-s25/pipe_filter_join_s25 | `kalypso_block:32` | 12.81 | 21 | – | 12.08 | 8.30 | 11.87 | 12.95 | -0.25 |
| experiments/validate/filter-join-s25/pipe_filter_join_s25 | `kalypso_pipe:2,4` | 6.50 | 23 | – | 5.99 | 5.43 | 5.89 | 6.50 | +0.02 |
| experiments/validate/filter-join-s25/pipe_filter_join_s25 | `kalypso_pipe:4,1` | 10.09 | 26 | – | 10.10 | 9.38 | 9.92 | 9.86 | +0.14 |
| experiments/validate/filter-join-s25/pipe_filter_join_s25 | `kalypso_pipe:1,2` | 4.93 | 16 | – | 4.66 | 3.70 | 4.58 | 5.58 | -0.10 |
| experiments/validate/filter-join-s25/pipe_filter_join_s25 | `kalypso_pipe:1,4` | 5.09 | 16 | – | 4.84 | 3.88 | 4.76 | 5.59 | -0.07 |
| experiments/validate/filter-join-s25/pipe_filter_join_s25 | `kalypso_pipe:2,1` | 8.98 | 23 | – | 8.83 | 8.26 | 8.68 | 9.71 | +0.12 |
| experiments/validate/filter-join-s25/pipe_filter_join_s25 | `kalypso_pipe:2,2` | 6.40 | 23 | – | 5.99 | 5.43 | 5.89 | 6.51 | -0.03 |
| experiments/validate/filter-join-s25/pipe_filter_join_s25 | `kalypso_pipe:2,3` | 6.45 | 23 | – | 5.91 | 5.35 | 5.81 | 6.68 | +0.00 |
| experiments/validate/filter-join-s25/pipe_filter_join_s25 | `kalypso_pipe:2,5` | 6.64 | 23 | – | 6.13 | 5.56 | 6.02 | 6.61 | +0.02 |
| experiments/validate/filter-join-s25/pipe_filter_join_s25 | `kalypso_pipe:2,8` | 6.92 | 23 | – | 6.44 | 5.87 | 6.33 | 6.94 | -0.02 |
| experiments/validate/filter-join-s25/pipe_filter_join_s25 | `kalypso_pipe:3,4` | 6.75 | 24 | – | 6.28 | 5.65 | 6.17 | 6.57 | +0.04 |
| experiments/validate/filter-join-s25/pipe_filter_join_s25 | `kalypso_pipe:4,4` | 7.36 | 26 | – | 6.81 | 6.09 | 6.69 | 6.65 | +0.02 |
| experiments/validate/filter-map-n1000/pipe_filter_map-n1000 | `kalypso_per_tuple` | 15.86 | 477 | 0 | 16.62 | 9.71 | 16.33 | 16.29 | +0.00 |
| experiments/validate/filter-map-n1000/pipe_filter_map-n1000 | `kalypso_block:4` | 12.90 | 596 | 0 | 13.41 | 9.15 | 13.18 | 13.91 | +0.11 |
| experiments/validate/filter-map-n1000/pipe_filter_map-n1000 | `kalypso_block:16` | 26.53 | 594 | 0 | 19.47 | 14.38 | 19.13 | 19.69 | +0.10 |
| experiments/validate/filter-map-n1000/pipe_filter_map-n1000 | `kalypso_block:32` | 43.41 | 594 | 2 | 30.23 | 23.42 | 29.71 | 29.69 | +0.10 |
| experiments/validate/filter-map-n1000/pipe_filter_map-n1000 | `kalypso_pipe:3,1` | 13.59 | 589 | 92 | 15.23 | 10.57 | 14.97 | 14.83 | +0.10 |
| experiments/validate/filter-map-n1000/pipe_filter_map-n1000 | `kalypso_pipe:4,1` | 13.70 | 596 | 126 | 15.28 | 11.02 | 15.02 | 14.44 | +0.11 |
| experiments/validate/filter-map-n1000/pipe_filter_map-n1000 | `kalypso_pipe:2,1` | 14.25 | 574 | 5 | 14.38 | 9.46 | 14.13 | 15.08 | +0.08 |
| experiments/validate/filter-map-n1000/pipe_filter_map-n1000 | `kalypso_pipe:3,2` | 13.45 | 589 | 0 | 17.54 | 12.88 | 17.23 | 18.04 | +0.10 |
| experiments/validate/filter-map-n1000/pipe_filter_map-n1000 | `kalypso_pipe:6,1` | 13.97 | 603 | 116 | 15.24 | 10.99 | 14.97 | 14.49 | +0.11 |
| experiments/validate/filter-map-n50/pipe_filter_map-n50 | `kalypso_per_tuple` | 2.30 | 19 | 0 | 1.70 | 1.21 | 1.67 | 1.72 | +0.00 |
| experiments/validate/filter-map-n50/pipe_filter_map-n50 | `kalypso_block:4` | 4.03 | 26 | 0 | 5.39 | 4.87 | 5.29 | 6.03 | +0.08 |
| experiments/validate/filter-map-n50/pipe_filter_map-n50 | `kalypso_block:16` | 9.94 | 24 | 0 | 12.99 | 11.16 | 12.76 | 13.49 | +0.04 |
| experiments/validate/filter-map-n50/pipe_filter_map-n50 | `kalypso_block:32` | 19.61 | 28 | 0 | 23.21 | 19.62 | 22.81 | 23.53 | +0.12 |
| experiments/validate/filter-map-n50/pipe_filter_map-n50 | `kalypso_pipe:2,1` | 2.31 | 24 | 0 | 1.59 | 1.27 | 1.57 | 2.37 | +0.04 |
| experiments/validate/filter-map-n50/pipe_filter_map-n50 | `kalypso_pipe:4,1` | 2.07 | 26 | 0 | 1.80 | 1.29 | 1.77 | 2.56 | +0.08 |
| experiments/validate/filter-map-n50/pipe_filter_map-n50 | `kalypso_pipe:2,2` | 2.60 | 24 | 0 | 4.18 | 3.85 | 4.11 | 4.88 | +0.04 |
| experiments/validate/filter-map-n50/pipe_filter_map-n50 | `kalypso_pipe:3,1` | 1.99 | 23 | 0 | 1.67 | 1.26 | 1.64 | 2.46 | +0.05 |
| experiments/validate/map-filter/pipe_map_filter-n500 | `kalypso_per_tuple` | 11.93 | – | 1 | 13.54 | 7.91 | 13.31 | 12.59 | +0.00 |
| experiments/validate/map-filter/pipe_map_filter-n500 | `kalypso_block:4` | 18.70 | – | 302 | 16.28 | 13.17 | 15.99 | 14.50 | +0.05 |
| experiments/validate/map-filter/pipe_map_filter-n500 | `kalypso_block:16` | 38.31 | – | 82 | 19.68 | 15.48 | 19.34 | 20.91 | -0.02 |
| experiments/validate/map-filter/pipe_map_filter-n500 | `kalypso_block:32` | 62.25 | – | 381 | 33.75 | 27.77 | 33.17 | 30.83 | -0.06 |
| experiments/validate/map-filter/pipe_map_filter-n500 | `kalypso_pipe:1,7` | 13.79 | – | 281 | 14.97 | 11.68 | 14.72 | 14.72 | +0.04 |
| experiments/validate/map-filter/pipe_map_filter-n500 | `kalypso_pipe:1,4` | 13.86 | – | 281 | 14.78 | 11.68 | 14.53 | 14.54 | +0.05 |
| experiments/validate/map-filter/pipe_map_filter-n500 | `kalypso_pipe:1,3` | 14.09 | – | 282 | 14.80 | 11.69 | 14.55 | 14.54 | +0.04 |
| experiments/validate/map-filter/pipe_map_filter-n500 | `kalypso_pipe:1,6` | 13.99 | – | 282 | 14.91 | 11.69 | 14.65 | 14.65 | +0.01 |
| experiments/validate/map-filter/pipe_map_filter-n500 | `kalypso_pipe:1,8` | 13.76 | – | 282 | 15.07 | 11.69 | 14.81 | 14.81 | +0.02 |
| experiments/validate/map-filter/pipe_map_filter-n500 | `kalypso_pipe:1,14` | 14.68 | – | 282 | 15.68 | 11.69 | 15.40 | 15.40 | +0.00 |
| experiments/validate/map-filter/pipe_map_filter-n500 | `kalypso_pipe:2,7` | 15.79 | – | 149 | 14.30 | 11.00 | 14.05 | 14.90 | +0.04 |

| Run | Algorithm | Fastest | Regret | Guarded | Best kept quality | Guarded regret | Guarded kept quality | Free |
|---|---|---|---|---|---|---|---|---|
| experiments/pipeline/filter-join-s10/pipe_filter_join_s10 | 1,4 | 1,1 | 2.0% | 1,1 | 1,1 | 0.0% | yes | 1,2 |
| experiments/pipeline/filter-join-s50/pipe_filter_join_s50 | 1,4 | 1,4 | 0.0% | 1,1 | 4,4 | 5.1% | yes | 1,2 |
| experiments/pipeline/filter-join-s90/pipe_filter_join_s90 | 1,4 | 1,4 | 0.0% | 1,1 | 6,4 | 14.6% | yes | 1,2 |
| experiments/pipeline/filter-map-blockmap/pipe_filter_map | 1,1 | 1,1 | 0.0% | 1,1 | 1,1 | 0.0% | yes | 1,1 |
| experiments/pipeline/filter-map/pipe_filter_map | 1,1 | 6,1 | 12.5% | 1,1 | 6,1 | 12.5% | yes | 1,1 |
| experiments/validate/filter-join-s25/pipe_filter_join_s25 | 1,2 | 1,2 | 0.0% | 1,1 | 2,3 | 2.0% | yes | 1,2 |
| experiments/validate/filter-map-n1000/pipe_filter_map-n1000 | 4,4 | 4,4 | 0.0% | 4,1 | 3,1 | 0.8% | yes | 4,3 |
| experiments/validate/filter-map-n50/pipe_filter_map-n50 | 1,1 | 3,1 | 15.7% | 1,1 | 3,1 | 15.7% | yes | 1,1 |
| experiments/validate/map-filter/pipe_map_filter-n500 | 1,1 | 1,1 | 0.0% | 1,1 | 1,1 | 0.0% | yes | 1,1 |

## Every single-operator arm

| Run | Arm | Measured (s) | From tokens (s) | Full (s) | Computed tokens | Predicted | Decode steps | Predicted |
|---|---|---|---|---|---|---|---|---|
| block-size/movie_opposite-n32-b11-Qwen2.5-7B-Instruct | `kalypso_per_tuple` | 10.73 | 11.49 | 11.53 | 138,845 | 139,145 | 2.2 | 2.3 |
| block-size/movie_opposite-n32-b11-Qwen2.5-7B-Instruct | `kalypso_block:11` | 8.52 | 8.74 | 8.81 | 89,144 | 89,097 | 45.5 | 47.7 |
| block-size/movie_opposite-n32-b2-Qwen2.5-7B-Instruct | `kalypso_per_tuple` | 11.24 | 11.49 | 11.53 | 138,845 | 139,145 | 2.2 | 2.3 |
| block-size/movie_opposite-n32-b2-Qwen2.5-7B-Instruct | `kalypso_block:2` | 8.03 | 7.91 | 7.77 | 89,594 | 89,655 | 8.0 | 6.5 |
| block-size/movie_opposite-n32-b6-Qwen2.5-7B-Instruct | `kalypso_per_tuple` | 11.05 | 11.49 | 11.53 | 138,845 | 139,145 | 2.2 | 2.3 |
| block-size/movie_opposite-n32-b6-Qwen2.5-7B-Instruct | `kalypso_block:6` | 8.28 | 8.10 | 8.16 | 88,939 | 89,200 | 23.9 | 24.8 |
| block-size/movie_positive-n500-b11 | `kalypso_per_tuple` | 4.77 | 4.83 | 4.84 | 57,728 | 57,783 | 3.0 | 3.0 |
| block-size/movie_positive-n500-b11 | `kalypso_block:11` | 2.83 | 2.82 | 2.86 | 19,248 | 19,412 | 45.4 | 46.4 |
| block-size/movie_positive-n500-b16 | `kalypso_per_tuple` | 4.69 | 4.83 | 4.84 | 57,728 | 57,783 | 3.0 | 3.0 |
| block-size/movie_positive-n500-b16 | `kalypso_block:16` | 3.38 | 3.38 | 3.38 | 18,634 | 18,856 | 69.3 | 68.7 |
| block-size/movie_positive-n500-b20 | `kalypso_per_tuple` | 4.71 | 4.83 | 4.84 | 57,728 | 57,783 | 3.0 | 3.0 |
| block-size/movie_positive-n500-b20 | `kalypso_block:20` | 3.91 | 3.91 | 3.80 | 18,323 | 18,578 | 91.0 | 86.6 |
| block-size/movie_positive-n500-b4 | `kalypso_per_tuple` | 4.55 | 4.83 | 4.84 | 57,728 | 57,783 | 3.0 | 3.0 |
| block-size/movie_positive-n500-b4 | `kalypso_block:4` | 2.28 | 2.38 | 2.30 | 23,065 | 22,551 | 16.0 | 15.2 |
| block-size/movie_positive-n500-b6 | `kalypso_per_tuple` | 4.85 | 4.83 | 4.84 | 57,728 | 57,783 | 3.0 | 3.0 |
| block-size/movie_positive-n500-b6 | `kalypso_block:6` | 2.43 | 2.41 | 2.41 | 20,991 | 20,922 | 23.8 | 24.1 |
| block-size/movie_positive-n500-b8 | `kalypso_per_tuple` | 4.56 | 4.83 | 4.84 | 57,728 | 57,783 | 3.0 | 3.0 |
| block-size/movie_positive-n500-b8 | `kalypso_block:8` | 2.43 | 2.52 | 2.57 | 19,935 | 20,087 | 31.8 | 33.0 |
| block-size/movie_positive-n500-controls | `kalypso_per_tuple` | 4.90 | 4.83 | 4.84 | 57,728 | 57,783 | 3.0 | 3.0 |
| block-size/movie_sel10-b4-6-8-16-Qwen2.5-7B-Instruct | `kalypso_per_tuple` | 3.95 | 3.87 | 3.88 | 46,027 | 46,115 | 3.0 | 3.0 |
| block-size/movie_sel10-b4-6-8-16-Qwen2.5-7B-Instruct | `kalypso_block:4` | 2.03 | 1.96 | 1.90 | 18,388 | 17,929 | 16.0 | 15.2 |
| block-size/movie_sel10-b4-6-8-16-Qwen2.5-7B-Instruct | `kalypso_block:6` | 2.08 | 2.03 | 2.03 | 16,690 | 16,617 | 23.9 | 24.1 |
| block-size/movie_sel10-b4-6-8-16-Qwen2.5-7B-Instruct | `kalypso_block:8` | 2.21 | 2.16 | 2.20 | 15,839 | 15,942 | 32.0 | 33.0 |
| block-size/movie_sel10-b4-6-8-16-Qwen2.5-7B-Instruct | `kalypso_block:16` | 3.10 | 3.07 | 3.02 | 14,788 | 14,949 | 71.0 | 68.7 |
| block-size/movie_sel25-b4-6-8-16-Qwen2.5-7B-Instruct | `kalypso_per_tuple` | 3.76 | 3.89 | 3.89 | 46,221 | 46,311 | 3.0 | 3.0 |
| block-size/movie_sel25-b4-6-8-16-Qwen2.5-7B-Instruct | `kalypso_block:4` | 1.95 | 1.98 | 1.91 | 18,565 | 18,125 | 16.0 | 15.2 |
| block-size/movie_sel25-b4-6-8-16-Qwen2.5-7B-Instruct | `kalypso_block:6` | 2.00 | 2.04 | 2.04 | 16,898 | 16,813 | 23.9 | 24.1 |
| block-size/movie_sel25-b4-6-8-16-Qwen2.5-7B-Instruct | `kalypso_block:8` | 2.15 | 2.18 | 2.22 | 16,015 | 16,138 | 32.0 | 33.0 |
| block-size/movie_sel25-b4-6-8-16-Qwen2.5-7B-Instruct | `kalypso_block:16` | 3.05 | 3.09 | 3.04 | 14,980 | 15,145 | 71.0 | 68.7 |
| block-size/movie_sel50-b4-6-8-16-Qwen2.5-7B-Instruct | `kalypso_per_tuple` | 3.75 | 3.90 | 3.90 | 46,369 | 46,383 | 3.0 | 3.0 |
| block-size/movie_sel50-b4-6-8-16-Qwen2.5-7B-Instruct | `kalypso_block:4` | 1.96 | 1.98 | 1.92 | 18,648 | 18,197 | 16.0 | 15.2 |
| block-size/movie_sel50-b4-6-8-16-Qwen2.5-7B-Instruct | `kalypso_block:6` | 2.01 | 2.05 | 2.05 | 16,965 | 16,885 | 23.9 | 24.1 |
| block-size/movie_sel50-b4-6-8-16-Qwen2.5-7B-Instruct | `kalypso_block:8` | 2.15 | 2.18 | 2.22 | 16,097 | 16,210 | 32.0 | 33.0 |
| block-size/movie_sel50-b4-6-8-16-Qwen2.5-7B-Instruct | `kalypso_block:16` | 3.04 | 3.09 | 3.04 | 15,047 | 15,217 | 71.0 | 68.7 |
| block-size/movie_sel75-b4-6-8-16-Qwen2.5-7B-Instruct | `kalypso_per_tuple` | 3.81 | 3.87 | 3.87 | 46,012 | 46,031 | 2.9 | 3.0 |
| block-size/movie_sel75-b4-6-8-16-Qwen2.5-7B-Instruct | `kalypso_block:4` | 1.99 | 1.96 | 1.89 | 18,293 | 17,845 | 16.0 | 15.2 |
| block-size/movie_sel75-b4-6-8-16-Qwen2.5-7B-Instruct | `kalypso_block:6` | 2.02 | 2.02 | 2.02 | 16,610 | 16,533 | 23.9 | 24.1 |
| block-size/movie_sel75-b4-6-8-16-Qwen2.5-7B-Instruct | `kalypso_block:8` | 2.18 | 2.15 | 2.19 | 15,743 | 15,858 | 32.0 | 33.0 |
| block-size/movie_sel75-b4-6-8-16-Qwen2.5-7B-Instruct | `kalypso_block:16` | 3.09 | 3.06 | 3.01 | 14,693 | 14,865 | 71.0 | 68.7 |
| block-size/movie_sel90-b4-6-8-16-Qwen2.5-7B-Instruct | `kalypso_per_tuple` | 3.75 | 3.87 | 3.87 | 46,041 | 46,063 | 2.9 | 3.0 |
| block-size/movie_sel90-b4-6-8-16-Qwen2.5-7B-Instruct | `kalypso_block:4` | 1.93 | 1.96 | 1.89 | 18,325 | 17,877 | 16.0 | 15.2 |
| block-size/movie_sel90-b4-6-8-16-Qwen2.5-7B-Instruct | `kalypso_block:6` | 2.00 | 2.02 | 2.02 | 16,642 | 16,565 | 23.9 | 24.1 |
| block-size/movie_sel90-b4-6-8-16-Qwen2.5-7B-Instruct | `kalypso_block:8` | 2.14 | 2.16 | 2.20 | 15,775 | 15,890 | 32.0 | 33.0 |
| block-size/movie_sel90-b4-6-8-16-Qwen2.5-7B-Instruct | `kalypso_block:16` | 3.02 | 3.07 | 3.02 | 14,725 | 14,897 | 71.0 | 68.7 |
| experiments/longrows/cnli-filter-s10/cnli_sel10 | `kalypso_per_tuple` | 51.21 | 52.11 | 52.11 | 631,716 | 631,995 | 3.0 | 3.0 |
| experiments/longrows/cnli-filter-s10/cnli_sel10 | `kalypso_block:2` | 51.82 | 51.69 | 51.63 | 606,625 | 607,782 | 8.0 | 6.3 |
| experiments/longrows/cnli-filter-s10/cnli_sel10 | `kalypso_block:4` | 54.66 | 54.75 | 54.75 | 602,719 | 603,809 | 16.0 | 15.2 |
| experiments/longrows/cnli-filter-s10/cnli_sel10 | `kalypso_block:8` | 60.86 | 61.33 | 61.42 | 600,739 | 601,822 | 32.0 | 33.0 |
| experiments/longrows/cnli-filter-s50/cnli_sel50 | `kalypso_per_tuple` | 52.24 | 52.12 | 52.13 | 631,859 | 632,143 | 3.0 | 3.0 |
| experiments/longrows/cnli-filter-s50/cnli_sel50 | `kalypso_block:2` | 51.36 | 51.70 | 51.64 | 606,770 | 607,930 | 8.0 | 6.3 |
| experiments/longrows/cnli-filter-s50/cnli_sel50 | `kalypso_block:4` | 55.30 | 54.76 | 54.76 | 602,866 | 603,957 | 16.0 | 15.2 |
| experiments/longrows/cnli-filter-s50/cnli_sel50 | `kalypso_block:8` | 61.85 | 61.35 | 61.43 | 600,887 | 601,970 | 32.0 | 33.0 |
| experiments/longrows/cnli-filter-s90/cnli_sel90 | `kalypso_per_tuple` | 53.04 | 52.13 | 52.14 | 632,014 | 632,303 | 3.0 | 3.0 |
| experiments/longrows/cnli-filter-s90/cnli_sel90 | `kalypso_block:2` | 52.40 | 51.63 | 51.66 | 605,879 | 608,090 | 8.0 | 6.3 |
| experiments/longrows/cnli-filter-s90/cnli_sel90 | `kalypso_block:4` | 56.28 | 54.95 | 54.78 | 606,109 | 604,117 | 15.7 | 15.2 |
| experiments/longrows/cnli-filter-s90/cnli_sel90 | `kalypso_block:8` | 61.76 | 61.36 | 61.45 | 600,986 | 602,130 | 32.0 | 33.0 |
| experiments/longrows/cnli-join-high/cnli_join_high | `kalypso_per_tuple` | 5.58 | 5.97 | 5.86 | 61,052 | 59,819 | 2.5 | 2.3 |
| experiments/longrows/cnli-join-high/cnli_join_high | `kalypso_block:2` | 18.41 | 18.01 | 17.95 | 200,623 | 200,656 | 8.0 | 6.5 |
| experiments/longrows/cnli-join-high/cnli_join_high | `kalypso_block:4` | 26.30 | 25.95 | 25.90 | 275,696 | 275,353 | 16.0 | 15.7 |
| experiments/longrows/cnli-join-high/cnli_join_high | `kalypso_block:8` | 34.51 | 32.91 | 32.90 | 313,238 | 312,702 | 32.0 | 34.0 |
| experiments/longrows/cnli-join-low/cnli_join_low | `kalypso_per_tuple` | 5.65 | 5.94 | 5.84 | 60,754 | 59,650 | 2.5 | 2.3 |
| experiments/longrows/cnli-join-low/cnli_join_low | `kalypso_block:2` | 18.57 | 18.00 | 17.94 | 200,469 | 200,487 | 8.0 | 6.5 |
| experiments/longrows/cnli-join-low/cnli_join_low | `kalypso_block:4` | 26.39 | 25.94 | 25.88 | 275,603 | 275,184 | 16.0 | 15.7 |
| experiments/longrows/cnli-join-low/cnli_join_low | `kalypso_block:8` | 33.30 | 32.90 | 32.87 | 313,095 | 312,533 | 32.0 | 34.0 |
| experiments/map/map-n100/movie_map-n100 | `kalypso_per_tuple` | 2.65 | 2.13 | 2.09 | 11,077 | 11,043 | 21.0 | 20.1 |
| experiments/map/map-n100/movie_map-n100 | `kalypso_block:2` | 3.40 | 3.57 | 4.61 | 6,839 | 6,755 | 67.1 | 92.8 |
| experiments/map/map-n100/movie_map-n100 | `kalypso_block:4` | 5.09 | 5.27 | 5.41 | 5,514 | 5,512 | 118.5 | 122.3 |
| experiments/map/map-n100/movie_map-n100 | `kalypso_block:8` | 7.85 | 7.89 | 7.41 | 5,144 | 4,916 | 192.6 | 181.2 |
| experiments/map/map-n500/movie_map-n500 | `kalypso_per_tuple` | 7.36 | 7.13 | 7.17 | 55,733 | 55,814 | 19.8 | 20.1 |
| experiments/map/map-n500/movie_map-n500 | `kalypso_block:2` | 6.43 | 7.20 | 8.66 | 34,515 | 34,377 | 68.5 | 92.8 |
| experiments/map/map-n500/movie_map-n500 | `kalypso_block:4` | 7.93 | 8.56 | 8.44 | 27,889 | 28,162 | 125.2 | 122.3 |
| experiments/map/map-n500/movie_map-n500 | `kalypso_block:6` | 9.47 | 9.90 | 9.08 | 26,137 | 26,124 | 169.1 | 151.7 |
| experiments/map/map-n500/movie_map-n500 | `kalypso_block:8` | 10.74 | 11.21 | 9.93 | 24,906 | 25,080 | 211.0 | 181.2 |
| experiments/map/map-n500/movie_map-n500 | `kalypso_block:16` | 19.80 | 13.20 | 13.86 | 24,914 | 23,539 | 266.7 | 299.1 |
| experiments/pipeline/filter-join-s10/alone-filter | `kalypso_per_tuple` | 0.95 | 0.95 | 0.95 | 11,412 | 11,378 | 3.0 | 3.0 |
| experiments/pipeline/filter-join-s10/alone-filter | `kalypso_block:4` | 0.73 | 0.74 | 0.70 | 4,472 | 4,331 | 16.0 | 15.2 |
| experiments/pipeline/filter-join-s10/alone-filter | `kalypso_block:6` | 0.85 | 0.89 | 0.90 | 4,064 | 4,013 | 23.5 | 24.1 |
| experiments/pipeline/filter-join-s10/alone-filter | `kalypso_block:11` | 1.21 | 1.31 | 1.44 | 3,723 | 3,735 | 41.6 | 46.4 |
| experiments/pipeline/filter-join-s10/alone-join | `kalypso_per_tuple` | 2.36 | 2.46 | 2.49 | 30,104 | 30,396 | 2.3 | 2.3 |
| experiments/pipeline/filter-join-s10/alone-join | `kalypso_block:2` | 1.74 | 1.75 | 1.69 | 18,875 | 18,895 | 8.0 | 6.5 |
| experiments/pipeline/filter-join-s10/alone-join | `kalypso_block:4` | 1.86 | 1.92 | 1.91 | 18,491 | 18,485 | 16.0 | 15.7 |
| experiments/pipeline/filter-join-s10/alone-join | `kalypso_block:6` | 2.02 | 2.10 | 2.14 | 18,376 | 18,353 | 23.6 | 24.8 |
| experiments/pipeline/filter-join-s50/alone-filter | `kalypso_per_tuple` | 0.96 | 0.97 | 0.97 | 11,641 | 11,611 | 3.0 | 3.0 |
| experiments/pipeline/filter-join-s50/alone-filter | `kalypso_block:4` | 0.74 | 0.75 | 0.72 | 4,701 | 4,564 | 16.0 | 15.2 |
| experiments/pipeline/filter-join-s50/alone-filter | `kalypso_block:6` | 0.86 | 0.91 | 0.92 | 4,293 | 4,246 | 23.5 | 24.1 |
| experiments/pipeline/filter-join-s50/alone-filter | `kalypso_block:11` | 1.24 | 1.33 | 1.46 | 3,952 | 3,968 | 41.6 | 46.4 |
| experiments/pipeline/filter-join-s50/alone-join | `kalypso_per_tuple` | 16.37 | 17.28 | 17.28 | 208,403 | 208,405 | 2.3 | 2.3 |
| experiments/pipeline/filter-join-s50/alone-join | `kalypso_block:2` | 11.09 | 11.48 | 11.25 | 129,328 | 129,160 | 8.0 | 6.5 |
| experiments/pipeline/filter-join-s50/alone-join | `kalypso_block:4` | 11.02 | 11.45 | 11.43 | 126,023 | 126,153 | 16.0 | 15.7 |
| experiments/pipeline/filter-join-s50/alone-join | `kalypso_block:6` | 11.22 | 11.19 | 11.25 | 124,892 | 125,150 | 24.0 | 24.8 |
| experiments/pipeline/filter-join-s90/alone-filter | `kalypso_per_tuple` | 1.01 | 0.97 | 0.97 | 11,616 | 11,585 | 2.9 | 3.0 |
| experiments/pipeline/filter-join-s90/alone-filter | `kalypso_block:4` | 0.82 | 0.75 | 0.72 | 4,674 | 4,538 | 16.0 | 15.2 |
| experiments/pipeline/filter-join-s90/alone-filter | `kalypso_block:6` | 0.97 | 0.91 | 0.92 | 4,266 | 4,220 | 23.5 | 24.1 |
| experiments/pipeline/filter-join-s90/alone-filter | `kalypso_block:11` | 1.38 | 1.33 | 1.46 | 3,925 | 3,942 | 41.6 | 46.4 |
| experiments/pipeline/filter-join-s90/alone-join | `kalypso_per_tuple` | 31.30 | 33.03 | 33.19 | 397,666 | 399,401 | 2.3 | 2.3 |
| experiments/pipeline/filter-join-s90/alone-join | `kalypso_block:2` | 21.55 | 22.10 | 21.62 | 247,895 | 247,308 | 8.0 | 6.5 |
| experiments/pipeline/filter-join-s90/alone-join | `kalypso_block:4` | 21.06 | 21.63 | 21.59 | 241,277 | 241,441 | 16.0 | 15.7 |
| experiments/pipeline/filter-join-s90/alone-join | `kalypso_block:6` | 21.34 | 21.51 | 21.63 | 238,971 | 239,488 | 24.0 | 24.8 |
| experiments/pipeline/filter-map/alone-filter | `kalypso_per_tuple` | 4.67 | 4.83 | 4.84 | 57,728 | 57,783 | 3.0 | 3.0 |
| experiments/pipeline/filter-map/alone-filter | `kalypso_block:4` | 2.43 | 2.38 | 2.30 | 23,065 | 22,551 | 16.0 | 15.2 |
| experiments/pipeline/filter-map/alone-filter | `kalypso_block:6` | 2.48 | 2.41 | 2.41 | 20,991 | 20,922 | 23.8 | 24.1 |
| experiments/pipeline/filter-map/alone-filter | `kalypso_block:11` | 2.90 | 2.82 | 2.86 | 19,248 | 19,412 | 45.4 | 46.4 |
| experiments/pipeline/filter-map/alone-map | `kalypso_per_tuple` | 5.29 | 5.75 | 5.79 | 41,559 | 41,593 | 19.8 | 20.1 |
| experiments/pipeline/filter-map-blockmap/alone-map | `kalypso_per_tuple` | 5.24 | 5.75 | 5.79 | 41,559 | 41,593 | 19.8 | 20.1 |
| experiments/pipeline/filter-map-blockmap/alone-map | `kalypso_block:4` | 6.79 | 7.59 | 7.49 | 20,895 | 21,002 | 124.7 | 122.3 |
| experiments/pipeline/filter-map-blockmap/alone-map | `kalypso_block:8` | 13.32 | 10.24 | 9.12 | 19,265 | 18,665 | 203.5 | 181.2 |
| experiments/single/filter-n100/movie_positive-n100 | `kalypso_per_tuple` | 0.98 | 0.96 | 0.96 | 11,477 | 11,437 | 3.0 | 3.0 |
| experiments/single/filter-n100/movie_positive-n100 | `kalypso_block:4` | 0.80 | 0.74 | 0.71 | 4,532 | 4,390 | 16.0 | 15.2 |
| experiments/single/filter-n100/movie_positive-n100 | `kalypso_block:6` | 0.94 | 0.90 | 0.91 | 4,124 | 4,072 | 23.5 | 24.1 |
| experiments/single/filter-n100/movie_positive-n100 | `kalypso_block:8` | 1.05 | 1.06 | 1.12 | 3,920 | 3,913 | 30.8 | 33.0 |
| experiments/single/filter-n1000/movie_positive-n1000 | `kalypso_per_tuple` | 9.36 | 9.69 | 9.71 | 115,130 | 115,286 | 3.0 | 3.0 |
| experiments/single/filter-n1000/movie_positive-n1000 | `kalypso_block:4` | 4.49 | 4.38 | 4.26 | 45,798 | 44,821 | 16.0 | 15.2 |
| experiments/single/filter-n1000/movie_positive-n1000 | `kalypso_block:6` | 4.34 | 4.25 | 4.25 | 41,627 | 41,524 | 23.9 | 24.1 |
| experiments/single/filter-n1000/movie_positive-n1000 | `kalypso_block:8` | 4.34 | 4.29 | 4.35 | 39,520 | 39,855 | 32.0 | 33.0 |
| experiments/single/filter-n250/movie_positive-n250 | `kalypso_per_tuple` | 2.34 | 2.40 | 2.40 | 28,987 | 28,992 | 3.0 | 3.0 |
| experiments/single/filter-n250/movie_positive-n250 | `kalypso_block:4` | 1.38 | 1.37 | 1.32 | 11,677 | 11,395 | 15.9 | 15.2 |
| experiments/single/filter-n250/movie_positive-n250 | `kalypso_block:6` | 1.49 | 1.48 | 1.48 | 10,605 | 10,561 | 23.8 | 24.1 |
| experiments/single/filter-n250/movie_positive-n250 | `kalypso_block:8` | 1.63 | 1.63 | 1.68 | 10,112 | 10,164 | 31.2 | 33.0 |
| experiments/single/filter-nall/movie_positive | `kalypso_per_tuple` | 18.32 | 18.18 | 18.22 | 214,980 | 215,382 | 3.0 | 3.0 |
| experiments/single/filter-nall/movie_positive | `kalypso_block:4` | 8.50 | 8.31 | 8.08 | 85,825 | 83,995 | 16.0 | 15.2 |
| experiments/single/filter-nall/movie_positive | `kalypso_block:6` | 8.09 | 8.09 | 8.08 | 77,985 | 77,797 | 24.0 | 24.1 |
| experiments/single/filter-nall/movie_positive | `kalypso_block:8` | 7.97 | 7.39 | 7.50 | 74,071 | 74,738 | 31.9 | 33.0 |
| experiments/single/join-fanout10/movie_fanout10 | `kalypso_per_tuple` | 11.05 | 10.94 | 11.08 | 132,090 | 133,614 | 2.2 | 2.3 |
| experiments/single/join-fanout10/movie_fanout10 | `kalypso_block:2` | 7.55 | 7.29 | 7.08 | 81,891 | 81,143 | 8.0 | 6.5 |
| experiments/single/join-fanout10/movie_fanout10 | `kalypso_block:4` | 7.58 | 7.12 | 7.09 | 79,499 | 79,317 | 16.0 | 15.7 |
| experiments/single/join-fanout10/movie_fanout10 | `kalypso_block:6` | 7.78 | 7.27 | 7.31 | 78,687 | 78,711 | 23.9 | 24.8 |
| experiments/single/join-fanout10/movie_fanout10 | `kalypso_block:11` | 8.41 | 7.85 | 7.91 | 78,360 | 78,161 | 45.5 | 47.7 |
| experiments/single/join-fanout25/movie_fanout25 | `kalypso_per_tuple` | 10.83 | 10.87 | 11.00 | 131,110 | 132,651 | 2.2 | 2.3 |
| experiments/single/join-fanout25/movie_fanout25 | `kalypso_block:2` | 7.41 | 7.21 | 7.01 | 80,926 | 80,180 | 8.0 | 6.5 |
| experiments/single/join-fanout25/movie_fanout25 | `kalypso_block:4` | 7.40 | 7.04 | 7.01 | 78,491 | 78,354 | 16.0 | 15.7 |
| experiments/single/join-fanout25/movie_fanout25 | `kalypso_block:6` | 7.51 | 7.19 | 7.23 | 77,679 | 77,748 | 23.9 | 24.8 |
| experiments/single/join-fanout25/movie_fanout25 | `kalypso_block:11` | 8.11 | 7.76 | 7.83 | 77,338 | 77,199 | 45.5 | 47.7 |
| experiments/single/join-fanout50/movie_fanout50 | `kalypso_per_tuple` | 10.43 | 10.61 | 10.72 | 127,656 | 129,098 | 2.3 | 2.3 |
| experiments/single/join-fanout50/movie_fanout50 | `kalypso_block:2` | 7.00 | 6.94 | 6.72 | 77,472 | 76,627 | 8.0 | 6.5 |
| experiments/single/join-fanout50/movie_fanout50 | `kalypso_block:4` | 6.97 | 6.76 | 6.72 | 75,068 | 74,801 | 16.0 | 15.7 |
| experiments/single/join-fanout50/movie_fanout50 | `kalypso_block:6` | 7.15 | 6.91 | 6.94 | 74,257 | 74,195 | 23.9 | 24.8 |
| experiments/single/join-fanout50/movie_fanout50 | `kalypso_block:11` | 7.72 | 7.48 | 7.54 | 73,912 | 73,646 | 45.5 | 47.7 |
| experiments/single/join-fanout75/movie_fanout75 | `kalypso_per_tuple` | 10.74 | 10.61 | 10.72 | 127,652 | 129,098 | 2.3 | 2.3 |
| experiments/single/join-fanout75/movie_fanout75 | `kalypso_block:2` | 7.14 | 6.94 | 6.72 | 77,494 | 76,627 | 8.0 | 6.5 |
| experiments/single/join-fanout75/movie_fanout75 | `kalypso_block:4` | 7.14 | 6.76 | 6.72 | 75,075 | 74,801 | 16.0 | 15.7 |
| experiments/single/join-fanout75/movie_fanout75 | `kalypso_block:6` | 7.35 | 6.91 | 6.94 | 74,284 | 74,195 | 23.9 | 24.8 |
| experiments/single/join-fanout75/movie_fanout75 | `kalypso_block:11` | 7.95 | 7.48 | 7.54 | 73,950 | 73,646 | 45.5 | 47.7 |
| experiments/single/join-fanout90/movie_fanout90 | `kalypso_per_tuple` | 10.54 | 10.70 | 10.79 | 128,550 | 129,928 | 2.4 | 2.3 |
| experiments/single/join-fanout90/movie_fanout90 | `kalypso_block:2` | 7.03 | 7.01 | 6.79 | 78,347 | 77,456 | 8.0 | 6.5 |
| experiments/single/join-fanout90/movie_fanout90 | `kalypso_block:4` | 7.03 | 6.83 | 6.79 | 75,932 | 75,631 | 16.0 | 15.7 |
| experiments/single/join-fanout90/movie_fanout90 | `kalypso_block:6` | 7.17 | 6.98 | 7.01 | 75,137 | 75,024 | 23.9 | 24.8 |
| experiments/single/join-fanout90/movie_fanout90 | `kalypso_block:11` | 7.82 | 7.55 | 7.60 | 74,809 | 74,475 | 45.5 | 47.7 |
| experiments/single/join-n16/movie_opposite-n16 | `kalypso_per_tuple` | 2.69 | 2.80 | 2.80 | 34,210 | 34,288 | 2.3 | 2.3 |
| experiments/single/join-n16/movie_opposite-n16 | `kalypso_block:2` | 2.00 | 1.99 | 1.92 | 21,729 | 21,534 | 8.0 | 6.5 |
| experiments/single/join-n16/movie_opposite-n16 | `kalypso_block:4` | 2.15 | 2.16 | 2.14 | 21,344 | 21,259 | 16.0 | 15.7 |
| experiments/single/join-n16/movie_opposite-n16 | `kalypso_block:6` | 2.32 | 2.35 | 2.37 | 21,246 | 21,169 | 23.8 | 24.8 |
| experiments/single/join-n48/movie_opposite-n48 | `kalypso_per_tuple` | 25.56 | 25.77 | 25.88 | 310,498 | 311,536 | 2.3 | 2.3 |
| experiments/single/join-n48/movie_opposite-n48 | `kalypso_block:2` | 17.99 | 17.77 | 17.47 | 199,274 | 199,885 | 8.0 | 6.5 |
| experiments/single/join-n48/movie_opposite-n48 | `kalypso_block:4` | 18.03 | 17.94 | 17.96 | 198,262 | 198,966 | 16.0 | 15.7 |
| experiments/single/join-n48/movie_opposite-n48 | `kalypso_block:6` | 18.25 | 17.97 | 18.10 | 197,929 | 198,660 | 24.0 | 24.8 |
| experiments/single/join-n64/movie_opposite-n64 | `kalypso_per_tuple` | 45.05 | 45.46 | 45.60 | 546,811 | 548,445 | 2.3 | 2.3 |
| experiments/single/join-n64/movie_opposite-n64 | `kalypso_block:2` | 30.73 | 30.89 | 30.34 | 347,208 | 347,699 | 8.0 | 6.5 |
| experiments/single/join-n64/movie_opposite-n64 | `kalypso_block:4` | 31.24 | 30.73 | 30.73 | 344,089 | 344,942 | 16.0 | 15.7 |
| experiments/single/join-n64/movie_opposite-n64 | `kalypso_block:6` | 31.73 | 30.93 | 31.14 | 342,961 | 344,023 | 24.0 | 24.8 |
| experiments/validate/cnli-filter-s25/cnli_sel25-n200 | `kalypso_per_tuple` | 24.03 | 25.31 | 25.26 | 307,490 | 306,967 | 3.0 | 3.0 |
| experiments/validate/cnli-filter-s25/cnli_sel25-n200 | `kalypso_block:4` | 25.99 | 26.59 | 26.63 | 291,901 | 292,874 | 16.0 | 15.2 |
| experiments/validate/cnli-filter-s25/cnli_sel25-n200 | `kalypso_block:16` | 35.51 | 36.25 | 37.32 | 290,503 | 291,404 | 68.0 | 68.7 |
| experiments/validate/cnli-filter-s25/cnli_sel25-n200 | `kalypso_block:2` | 24.11 | 25.07 | 25.05 | 293,902 | 294,861 | 8.0 | 6.3 |
| experiments/validate/cnli-filter-s25/cnli_sel25-n200 | `kalypso_block:3` | 24.68 | 25.79 | 25.83 | 292,577 | 293,550 | 11.9 | 10.7 |
| experiments/validate/join-fanout25-n64/movie_fanout25-n64 | `kalypso_per_tuple` | 42.26 | 43.08 | 43.51 | 518,047 | 522,479 | 2.2 | 2.3 |
| experiments/validate/join-fanout25-n64/movie_fanout25-n64 | `kalypso_block:4` | 27.59 | 27.41 | 27.38 | 302,978 | 303,359 | 16.0 | 15.7 |
| experiments/validate/join-fanout25-n64/movie_fanout25-n64 | `kalypso_block:16` | 31.34 | 27.84 | 27.74 | 298,416 | 297,398 | 71.0 | 70.5 |
| experiments/validate/join-fanout25-n64/movie_fanout25-n64 | `kalypso_block:32` | 39.52 | 30.53 | 30.10 | 298,157 | 296,405 | 151.0 | 143.7 |
| experiments/validate/join-fanout25-n64/movie_fanout25-n64 | `kalypso_block:10` | 28.50 | 27.77 | 27.96 | 298,644 | 298,593 | 41.0 | 43.1 |
| experiments/validate/join-fanout25-n64/movie_fanout25-n64 | `kalypso_block:2` | 27.83 | 28.07 | 27.42 | 312,092 | 311,306 | 8.0 | 6.5 |
| experiments/validate/join-fanout25-n64/movie_fanout25-n64 | `kalypso_block:5` | 27.66 | 27.70 | 27.80 | 301,209 | 301,775 | 20.0 | 20.2 |
| experiments/validate/join-fanout25-n64/movie_fanout25-n64 | `kalypso_block:9` | 28.08 | 27.38 | 27.71 | 297,956 | 298,950 | 35.9 | 38.5 |
| experiments/validate/join-fanout25-n64/movie_fanout25-n64 | `kalypso_block:11` | 28.77 | 28.08 | 28.21 | 298,642 | 298,306 | 45.9 | 47.7 |
