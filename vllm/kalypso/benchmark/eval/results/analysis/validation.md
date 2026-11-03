# Validation

Latency first: the algorithm picks the block sizes with the lowest predicted latency, and the oracle is the fastest measured arm. Accuracy is reported, not required: kept quality = F1 within the slack of tuple-wise (and for a pipeline ending in a map, ROUGE-1 similarity at least 0.8). The accuracy-guarded algorithm only blocks where the measured runs kept quality. Speedup over tuple-wise is a geometric mean over the experiments the strategy ran; regret = how much slower than the oracle. Predicted = the current model (v3) for that arm, from the workload's true selectivity.

Each phase is scored with the choices made before it ran. `validate` was chosen by v2 and is now part of v3's fit, so its "Current model, among measured arms" row is in-sample; `validate3`, `validate-jf` are held out from v3.

## Summary: `validate` (chosen by v2; in the current fit)

| Strategy | Experiments | Kept quality | Failed | Speedup (geomean) | Regret (mean) | Regret (max) | Prediction error (mean) |
|---|---|---|---|---|---|---|---|
| Tuple-wise | 6 | 6/6 | 0 | 1.00x | 20.9% | 53.2% | 10.4% |
| Static 4 | 6 | 2/6 | 0 | 0.91x | 36.1% | 102.8% | 15.4% |
| Static 16 | 6 | 0/6 | 0 | 0.55x | 147.8% | 399.8% | 22.1% |
| Static 32 | 6 | 0/5 | 1 | 0.34x | 349.5% | 886.3% | 25.4% |
| Best in isolation | 6 | 6/6 | 0 | 0.95x | 30.7% | 104.6% | 7.4% |
| Algorithm | 6 | 4/6 | 0 | 1.07x | 12.1% | 31.8% | 4.1% |
| Algorithm with accuracy guard | 6 | 6/6 | 0 | 1.00x | 20.6% | 53.2% | 7.9% |
| Current model, among measured arms | 6 | 3/6 | 0 | 1.16x | 2.8% | 15.7% | 9.7% |
| Oracle (fastest measured) | 6 | 3/6 | 0 | 1.19x | 0.0% | 0.0% | 9.3% |

## Summary: `validate3` (chosen by v3; held out)

| Strategy | Experiments | Kept quality | Failed | Speedup (geomean) | Regret (mean) | Regret (max) | Prediction error (mean) |
|---|---|---|---|---|---|---|---|
| Tuple-wise | 5 | 5/5 | 0 | 1.00x | 50.6% | 126.0% | 7.9% |
| Static 4 | 5 | 2/5 | 0 | 1.08x | 39.0% | 89.4% | 6.8% |
| Static 16 | 5 | 1/5 | 0 | 0.69x | 162.8% | 437.7% | 16.6% |
| Static 32 | 5 | 1/5 | 0 | 0.47x | 346.4% | 857.7% | 22.1% |
| Best in isolation | 5 | 3/5 | 0 | 1.31x | 11.6% | 32.9% | 5.3% |
| Algorithm | 5 | 3/5 | 0 | 1.38x | 6.1% | 23.8% | 8.9% |
| Algorithm with accuracy guard | 5 | 5/5 | 0 | 1.17x | 25.6% | 53.8% | 9.1% |
| Oracle (fastest measured) | 5 | 3/5 | 0 | 1.45x | 0.0% | 0.0% | 7.4% |

## validate/filter-join-s25

| Strategy | Arm | Latency (s) | Speedup | Regret | ΔF1 | Sim | Kept quality | Predicted (s) |
|---|---|---|---|---|---|---|---|---|
| Tuple-wise | `kalypso_per_tuple` | 6.58 | 1.00x | 33.4% | +0.00 | – | yes | 7.95 |
| Static 4 | `kalypso_block:4` | 7.34 | 0.90x | 48.8% | +0.02 | – | yes | 6.65 |
| Static 16 | `kalypso_block:16` | 9.82 | 0.67x | 98.9% | -0.10 | – | no | 8.93 |
| Static 32 | `kalypso_block:32` | 12.81 | 0.51x | 159.6% | -0.25 | – | no | 12.95 |
| Best in isolation | `kalypso_pipe:4,1` | 10.09 | 0.65x | 104.6% | +0.14 | – | yes | 9.86 |
| Algorithm | `kalypso_pipe:2,4` | 6.50 | 1.01x | 31.8% | +0.02 | – | yes | 6.50 |
| Algorithm with accuracy guard | `kalypso_per_tuple` | 6.58 | 1.00x | 33.4% | +0.00 | – | yes | 7.95 |
| Current model, among measured arms | `kalypso_pipe:1,2` | 4.93 | 1.33x | 0.0% | -0.10 | – | no | 5.58 |
| Oracle (fastest measured) | `kalypso_pipe:1,2` | 4.93 | 1.33x | 0.0% | -0.10 | – | no | 5.58 |

## validate/filter-map-n1000

| Strategy | Arm | Latency (s) | Speedup | Regret | ΔF1 | Sim | Kept quality | Predicted (s) |
|---|---|---|---|---|---|---|---|---|
| Tuple-wise | `kalypso_per_tuple` | 15.86 | 1.00x | 23.0% | +0.00 | 1.00 | yes | 16.29 |
| Static 4 | `kalypso_block:4` | 12.90 | 1.23x | 0.0% | +0.11 | 0.39 | no | 13.91 |
| Static 16 | `kalypso_block:16` | 26.53 | 0.60x | 105.7% | +0.10 | 0.35 | no | 19.69 |
| Static 32 | `kalypso_block:32` | 43.41 | 0.37x | 236.6% | +0.10 | 0.35 | no | 29.69 |
| Best in isolation | `kalypso_pipe:4,1` | 13.70 | 1.16x | 6.2% | +0.11 | 0.93 | yes | 14.44 |
| Algorithm | `kalypso_pipe:3,1` | 13.59 | 1.17x | 5.4% | +0.10 | 0.93 | yes | 14.83 |
| Algorithm with accuracy guard | `kalypso_pipe:3,1` | 13.59 | 1.17x | 5.4% | +0.10 | 0.93 | yes | 14.83 |
| Current model, among measured arms | `kalypso_block:4` | 12.90 | 1.23x | 0.0% | +0.11 | 0.39 | no | 13.91 |
| Oracle (fastest measured) | `kalypso_block:4` | 12.90 | 1.23x | 0.0% | +0.11 | 0.39 | no | 13.91 |

## validate/map-filter

| Strategy | Arm | Latency (s) | Speedup | Regret | ΔF1 | Sim | Kept quality | Predicted (s) |
|---|---|---|---|---|---|---|---|---|
| Tuple-wise | `kalypso_per_tuple` | 11.93 | 1.00x | 0.0% | +0.00 | – | yes | 12.59 |
| Static 4 | `kalypso_block:4` | 18.70 | 0.64x | 56.7% | +0.05 | – | yes | 14.50 |
| Static 16 | `kalypso_block:16` | 38.31 | 0.31x | 221.1% | -0.02 | – | no | 20.91 |
| Static 32 | `kalypso_block:32` | 62.25 | 0.19x | 421.7% | -0.06 | – | no | 30.83 |
| Best in isolation | `kalypso_pipe:1,4` | 13.86 | 0.86x | 16.2% | +0.05 | – | yes | 14.54 |
| Algorithm | `kalypso_pipe:1,7` | 13.79 | 0.87x | 15.5% | +0.04 | – | yes | 14.72 |
| Algorithm with accuracy guard | `kalypso_pipe:1,7` | 13.79 | 0.87x | 15.5% | +0.04 | – | yes | 14.72 |
| Current model, among measured arms | `kalypso_per_tuple` | 11.93 | 1.00x | 0.0% | +0.00 | – | yes | 12.59 |
| Oracle (fastest measured) | `kalypso_per_tuple` | 11.93 | 1.00x | 0.0% | +0.00 | – | yes | 12.59 |

## validate/filter-map-n50

| Strategy | Arm | Latency (s) | Speedup | Regret | ΔF1 | Sim | Kept quality | Predicted (s) |
|---|---|---|---|---|---|---|---|---|
| Tuple-wise | `kalypso_per_tuple` | 2.30 | 1.00x | 15.7% | +0.00 | 1.00 | yes | 1.72 |
| Static 4 | `kalypso_block:4` | 4.03 | 0.57x | 102.8% | +0.08 | 0.39 | no | 6.03 |
| Static 16 | `kalypso_block:16` | 9.94 | 0.23x | 399.8% | +0.04 | 0.35 | no | 13.49 |
| Static 32 | `kalypso_block:32` | 19.61 | 0.12x | 886.3% | +0.12 | 0.33 | no | 23.53 |
| Best in isolation | `kalypso_pipe:4,1` | 2.07 | 1.11x | 4.1% | +0.08 | 0.95 | yes | 2.56 |
| Algorithm | `kalypso_pipe:2,1` | 2.31 | 1.00x | 16.2% | +0.04 | 0.99 | yes | 2.37 |
| Algorithm with accuracy guard | `kalypso_pipe:2,1` | 2.31 | 1.00x | 16.2% | +0.04 | 0.99 | yes | 2.37 |
| Current model, among measured arms | `kalypso_per_tuple` | 2.30 | 1.00x | 15.7% | +0.00 | 1.00 | yes | 1.72 |
| Oracle (fastest measured) | `kalypso_pipe:3,1` | 1.99 | 1.16x | 0.0% | +0.05 | 0.93 | yes | 2.46 |

## validate/join-fanout25-n64

| Strategy | Arm | Latency (s) | Speedup | Regret | ΔF1 | Sim | Kept quality | Predicted (s) |
|---|---|---|---|---|---|---|---|---|
| Tuple-wise | `kalypso_per_tuple` | 42.26 | 1.00x | 53.2% | +0.00 | – | yes | 43.51 |
| Static 4 | `kalypso_block:4` | 27.59 | 1.53x | 0.0% | -0.07 | – | no | 27.38 |
| Static 16 | `kalypso_block:16` | 31.34 | 1.35x | 13.6% | -0.25 | – | no | 27.74 |
| Static 32 | `kalypso_block:32` | 39.52 | 1.07x | 43.3% | -0.52 | – | no | 30.10 |
| Best in isolation | `kalypso_per_tuple` | 42.26 | 1.00x | 53.2% | +0.00 | – | yes | 43.51 |
| Algorithm | `kalypso_block:10` | 28.50 | 1.48x | 3.3% | -0.13 | – | no | 27.96 |
| Algorithm with accuracy guard | `kalypso_per_tuple` | 42.26 | 1.00x | 53.2% | +0.00 | – | yes | 43.51 |
| Current model, among measured arms | `kalypso_block:2` | 27.83 | 1.52x | 0.9% | -0.15 | – | no | 27.42 |
| Oracle (fastest measured) | `kalypso_block:4` | 27.59 | 1.53x | 0.0% | -0.07 | – | no | 27.38 |

## validate/cnli-filter-s25

| Strategy | Arm | Latency (s) | Speedup | Regret | ΔF1 | Sim | Kept quality | Predicted (s) |
|---|---|---|---|---|---|---|---|---|
| Tuple-wise | `kalypso_per_tuple` | 24.03 | 1.00x | 0.0% | +0.00 | – | yes | 25.26 |
| Static 4 | `kalypso_block:4` | 25.99 | 0.92x | 8.1% | -0.11 | – | no | 26.63 |
| Static 16 | `kalypso_block:16` | 35.51 | 0.68x | 47.8% | -0.18 | – | no | 37.32 |
| Static 32 | `kalypso_block:32` | failed | –x | – | – | – | – | 53.47 |
| Best in isolation | `kalypso_per_tuple` | 24.03 | 1.00x | 0.0% | +0.00 | – | yes | 25.26 |
| Algorithm | `kalypso_block:2` | 24.11 | 1.00x | 0.3% | -0.12 | – | no | 25.05 |
| Algorithm with accuracy guard | `kalypso_per_tuple` | 24.03 | 1.00x | 0.0% | +0.00 | – | yes | 25.26 |
| Current model, among measured arms | `kalypso_per_tuple` | 24.03 | 1.00x | 0.0% | +0.00 | – | yes | 25.26 |
| Oracle (fastest measured) | `kalypso_per_tuple` | 24.03 | 1.00x | 0.0% | +0.00 | – | yes | 25.26 |

## validate3/filter-n1500

| Strategy | Arm | Latency (s) | Speedup | Regret | ΔF1 | Sim | Kept quality | Predicted (s) |
|---|---|---|---|---|---|---|---|---|
| Tuple-wise | `kalypso_per_tuple` | 14.64 | 1.00x | 126.0% | +0.00 | – | yes | 14.61 |
| Static 4 | `kalypso_block:4` | 6.80 | 2.15x | 5.0% | +0.11 | – | yes | 6.64 |
| Static 16 | `kalypso_block:16` | 7.23 | 2.03x | 11.6% | +0.10 | – | yes | 6.82 |
| Static 32 | `kalypso_block:32` | 9.37 | 1.56x | 44.6% | +0.09 | – | yes | 8.51 |
| Best in isolation | `kalypso_block:4` | 6.80 | 2.15x | 5.0% | +0.11 | – | yes | 6.64 |
| Algorithm | `kalypso_block:6` | 6.55 | 2.23x | 1.1% | +0.11 | – | yes | 6.12 |
| Algorithm with accuracy guard | `kalypso_block:6` | 6.55 | 2.23x | 1.1% | +0.11 | – | yes | 6.12 |
| Oracle (fastest measured) | `kalypso_block:7` | 6.48 | 2.26x | 0.0% | +0.10 | – | yes | 6.14 |

## validate3/join-n48

| Strategy | Arm | Latency (s) | Speedup | Regret | ΔF1 | Sim | Kept quality | Predicted (s) |
|---|---|---|---|---|---|---|---|---|
| Tuple-wise | `kalypso_per_tuple` | 25.49 | 1.00x | 53.8% | +0.00 | – | yes | 24.89 |
| Static 4 | `kalypso_block:4` | 16.78 | 1.52x | 1.3% | -0.10 | – | no | 16.09 |
| Static 16 | `kalypso_block:16` | 19.14 | 1.33x | 15.5% | -0.45 | – | no | 16.78 |
| Static 32 | `kalypso_block:32` | 22.23 | 1.15x | 34.1% | -0.46 | – | no | 18.93 |
| Best in isolation | `kalypso_block:5` | 16.78 | 1.52x | 1.2% | -0.07 | – | no | 15.91 |
| Algorithm | `kalypso_block:2` | 16.73 | 1.52x | 0.9% | -0.18 | – | no | 15.91 |
| Algorithm with accuracy guard | `kalypso_per_tuple` | 25.49 | 1.00x | 53.8% | +0.00 | – | yes | 24.89 |
| Oracle (fastest measured) | `kalypso_block:3` | 16.57 | 1.54x | 0.0% | -0.07 | – | no | 15.80 |

## validate3/filter-join-s75

| Strategy | Arm | Latency (s) | Speedup | Regret | ΔF1 | Sim | Kept quality | Predicted (s) |
|---|---|---|---|---|---|---|---|---|
| Tuple-wise | `kalypso_per_tuple` | 15.04 | 1.00x | 44.7% | +0.00 | – | yes | 16.26 |
| Static 4 | `kalypso_block:4` | 13.74 | 1.10x | 32.1% | -0.02 | – | no | 13.31 |
| Static 16 | `kalypso_block:16` | 15.93 | 0.94x | 53.2% | -0.15 | – | no | 14.98 |
| Static 32 | `kalypso_block:32` | 21.05 | 0.71x | 102.5% | -0.27 | – | no | 19.34 |
| Best in isolation | `kalypso_pipe:4,5` | 13.82 | 1.09x | 32.9% | -0.03 | – | no | 13.51 |
| Algorithm | `kalypso_pipe:1,2` | 10.40 | 1.45x | 0.0% | -0.13 | – | no | 10.88 |
| Algorithm with accuracy guard | `kalypso_per_tuple` | 15.04 | 1.00x | 44.7% | +0.00 | – | yes | 16.26 |
| Oracle (fastest measured) | `kalypso_pipe:1,2` | 10.40 | 1.45x | 0.0% | -0.13 | – | no | 10.88 |

## validate3/filter-map-n250

| Strategy | Arm | Latency (s) | Speedup | Regret | ΔF1 | Sim | Kept quality | Predicted (s) |
|---|---|---|---|---|---|---|---|---|
| Tuple-wise | `kalypso_per_tuple` | 4.45 | 1.00x | 4.7% | +0.00 | 1.00 | yes | 4.61 |
| Static 4 | `kalypso_block:4` | 7.10 | 0.63x | 67.1% | +0.10 | 0.41 | no | 7.72 |
| Static 16 | `kalypso_block:16` | 16.84 | 0.26x | 296.3% | +0.08 | 0.36 | no | 14.81 |
| Static 32 | `kalypso_block:32` | 33.69 | 0.13x | 692.9% | +0.11 | 0.35 | no | 24.86 |
| Best in isolation | `kalypso_pipe:4,1` | 4.28 | 1.04x | 0.6% | +0.10 | 0.93 | yes | 4.79 |
| Algorithm | `kalypso_per_tuple` | 4.45 | 1.00x | 4.7% | +0.00 | 1.00 | yes | 4.61 |
| Algorithm with accuracy guard | `kalypso_per_tuple` | 4.45 | 1.00x | 4.7% | +0.00 | 1.00 | yes | 4.61 |
| Oracle (fastest measured) | `kalypso_pipe:2,1` | 4.25 | 1.05x | 0.0% | +0.07 | 0.93 | yes | 4.74 |

## validate3/map-filter-n200

| Strategy | Arm | Latency (s) | Speedup | Regret | ΔF1 | Sim | Kept quality | Predicted (s) |
|---|---|---|---|---|---|---|---|---|
| Tuple-wise | `kalypso_per_tuple` | 7.08 | 1.00x | 23.8% | +0.00 | – | yes | 5.31 |
| Static 4 | `kalypso_block:4` | 10.84 | 0.65x | 89.4% | +0.03 | – | yes | 9.14 |
| Static 16 | `kalypso_block:16` | 30.76 | 0.23x | 437.7% | -0.08 | – | no | 16.24 |
| Static 32 | `kalypso_block:32` | 54.78 | 0.13x | 857.7% | -0.04 | – | no | 26.28 |
| Best in isolation | `kalypso_pipe:1,4` | 6.76 | 1.05x | 18.2% | +0.05 | – | yes | 6.43 |
| Algorithm | `kalypso_per_tuple` | 7.08 | 1.00x | 23.8% | +0.00 | – | yes | 5.31 |
| Algorithm with accuracy guard | `kalypso_per_tuple` | 7.08 | 1.00x | 23.8% | +0.00 | – | yes | 5.31 |
| Oracle (fastest measured) | `kalypso_pipe:1,2` | 5.72 | 1.24x | 0.0% | +0.05 | – | yes | 6.35 |
