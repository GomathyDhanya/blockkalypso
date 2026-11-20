# Cost-model validation (v3, committed GPU results)

Coefficient relative difference after refitting vs committed costmodel.json: alpha 3.1e-12, beta 1.6e-12, gamma 1.6e-11, delta 2.8e-11, lambda 1.6e-12

## Latency

| Split | Prediction | n | MAE (s) | RMSE (s) | MAPE | median APE | p90 APE | max APE | under (n, mean) | over (n, mean) |
|---|---|---|---|---|---|---|---|---|---|---|
| fit | single_full | 173 | 0.44 | 1.06 | 4.3% | 2.8% | 6.7% | 35.8% | 97, -4.0% | 76, 4.6% |
| fit | single_from_measured_tokens | 173 | 0.39 | 1.01 | 3.4% | 2.3% | 6.7% | 33.3% | 87, -3.6% | 86, 3.1% |
| fit | pipeline_full | 75 | 1.76 | 4.68 | 13.2% | 7.3% | 33.1% | 87.2% | 26, -13.8% | 49, 12.9% |
| fit | pipeline_given_cardinality | 75 | 1.66 | 4.51 | 10.9% | 7.4% | 28.0% | 57.8% | 45, -11.1% | 30, 10.5% |
| held_out | single_full | 93 | 1.79 | 4.88 | 6.6% | 4.1% | 15.1% | 45.0% | 86, -6.9% | 7, 2.9% |
| held_out | single_from_measured_tokens | 93 | 1.55 | 4.14 | 5.8% | 3.9% | 13.2% | 40.7% | 79, -6.5% | 14, 2.0% |
| held_out | pipeline_full | 143 | 2.30 | 3.69 | 20.3% | 17.5% | 40.3% | 67.3% | 35, -13.3% | 108, 22.6% |
| held_out | pipeline_given_cardinality | 143 | 2.30 | 3.70 | 20.2% | 17.5% | 40.3% | 67.3% | 39, -13.1% | 104, 22.8% |
| all | single_full | 266 | 0.91 | 3.01 | 5.1% | 3.3% | 10.1% | 45.0% | 183, -5.4% | 83, 4.5% |
| all | single_from_measured_tokens | 266 | 0.80 | 2.58 | 4.2% | 2.7% | 8.0% | 40.7% | 166, -5.0% | 100, 2.9% |
| all | pipeline_full | 218 | 2.11 | 4.06 | 17.9% | 13.4% | 39.3% | 87.2% | 61, -13.5% | 157, 19.6% |
| all | pipeline_given_cardinality | 218 | 2.08 | 4.00 | 17.0% | 12.4% | 37.7% | 67.3% | 84, -12.0% | 134, 20.1% |

### Single operators by kind and regime (full model)

| Split | Kind/regime | n | MAPE | median APE | max APE | under n | over n |
|---|---|---|---|---|---|---|---|
| fit | filter/short | 70 | 3.5% | 2.7% | 18.7% | 37 | 33 |
| fit | filter/long | 17 | 2.0% | 1.4% | 5.1% | 8 | 9 |
| fit | join/short | 64 | 3.5% | 2.8% | 23.8% | 39 | 25 |
| fit | join/long | 8 | 3.0% | 3.0% | 4.9% | 6 | 2 |
| fit | map/short | 14 | 15.4% | 9.8% | 35.8% | 7 | 7 |
| held_out | filter/short | 30 | 4.6% | 4.4% | 9.4% | 29 | 1 |
| held_out | filter/long | 11 | 2.9% | 2.6% | 5.3% | 10 | 1 |
| held_out | join/short | 29 | 4.6% | 4.1% | 14.9% | 29 | 0 |
| held_out | join/long | 7 | 2.4% | 1.7% | 5.7% | 3 | 4 |
| held_out | map/short | 12 | 13.6% | 12.9% | 31.3% | 11 | 1 |
| held_out | map/long | 4 | 31.7% | 32.4% | 45.0% | 4 | 0 |
| all | filter/short | 100 | 3.8% | 3.2% | 18.7% | 66 | 34 |
| all | filter/long | 28 | 2.3% | 2.0% | 5.3% | 18 | 10 |
| all | join/short | 93 | 3.9% | 3.2% | 23.8% | 68 | 25 |
| all | join/long | 15 | 2.7% | 2.5% | 5.7% | 9 | 6 |
| all | map/short | 26 | 14.6% | 10.4% | 35.8% | 18 | 8 |
| all | map/long | 4 | 31.7% | 32.4% | 45.0% | 4 | 0 |

## Ranking among measured arms (per run with >= 3 arms)

| Split | Level | Runs | Arms | Spearman median | Spearman min | Top-1 | Regret mean | Regret max |
|---|---|---|---|---|---|---|---|---|
| fit | single | 34 | 153 | 1.00 | 0.40 | 82.4% | 0.5% | 14.4% |
| fit | pipeline | 9 | 75 | 0.70 | -0.60 | 66.7% | 2.7% | 15.7% |
| held_out | single | 16 | 93 | 0.99 | 0.80 | 68.8% | 0.7% | 3.2% |
| held_out | pipeline | 7 | 143 | 0.54 | 0.32 | 0.0% | 10.2% | 23.8% |
| all | single | 50 | 246 | 1.00 | 0.40 | 78.0% | 0.6% | 14.4% |
| all | pipeline | 16 | 218 | 0.66 | -0.60 | 37.5% | 6.0% | 23.8% |

## Cardinality (q-error; zero cardinalities counted separately)

| Split | Quantity | n | median q | p90 q | max q | under n | over n | status |
|---|---|---|---|---|---|---|---|---|
| fit | filter_pass | 87 | 1.02 | 1.11 | 2.82 | 40 | 47 | {'ok': 87} |
| fit | join_pass | 72 | 1.10 | 1.36 | 2.29 | 42 | 30 | {'ok': 72} |
| fit | pipeline_stage1_out | 64 | 1.06 | 1.24 | 2.82 | 19 | 45 | {'ok': 64} |
| held_out | filter_pass | 41 | 1.32 | 1.80 | 5.94 | 9 | 32 | {'ok': 41} |
| held_out | join_pass | 36 | 1.83 | 4.75 | 15.19 | 18 | 18 | {'ok': 36} |
| held_out | pipeline_stage1_out | 17 | 1.03 | 1.07 | 1.10 | 4 | 13 | {'ok': 17} |
| all | filter_pass | 128 | 1.03 | 1.54 | 5.94 | 49 | 79 | {'ok': 128} |
| all | join_pass | 108 | 1.11 | 2.39 | 15.19 | 60 | 48 | {'ok': 108} |
| all | pipeline_stage1_out | 81 | 1.04 | 1.24 | 2.82 | 23 | 58 | {'ok': 81} |

## KV / resources

costmodel.py has no KV-memory, queueing or preemption term; only measured peaks are available (below).

- peak KV cache %: {'n': 484, 'median': 6.905, 'p90': 14.701, 'max': 75.49}
- peak waiting requests: {'n': 440, 'median': 151.0, 'p90': 847.1, 'max': 2011.0}
- arms with any preemption: 0 of 484
