# Optimizer scaling (Experiment A)

Cost model v3 (`costmodel.json`), synthetic shape: 400 rows, 31 tokens/row, 32 right rows, selectivity 0.5. 5 timed repeats per configuration; cap 2,000,000 candidates. CPU: Linux-6.18.44-fc-v726-x86_64-with-glibc2.39, Python 3.13.18, git 3d922dd.

| family | stages | candidate set | legal plans | run? | planning s median [min,max] | peak heap MB | chosen blocks | predicted s | regret vs exhaustive |
|---|---|---|---|---|---|---|---|---|---|
| filters | 1 | exhaustive | 20 | yes | 0.0003 [0.0002, 0.0004] | 0.0 | [3] | 1.85 | (reference) |
| filters | 1 | pow2 | 5 | yes | 0.0001 [0.0001, 0.0001] | 0.0 | [4] | 1.86 | +0.7% |
| filters | 1 | small | 4 | yes | 0.0001 [0.0001, 0.0001] | 0.0 | [3] | 1.85 | +0.0% |
| filters | 2 | exhaustive | 400 | yes | 0.0113 [0.0096, 0.0166] | 0.4 | [3, 2] | 2.70 | (reference) |
| filters | 2 | pow2 | 25 | yes | 0.0006 [0.0006, 0.0006] | 0.0 | [4, 2] | 2.71 | +0.5% |
| filters | 2 | small | 16 | yes | 0.0004 [0.0004, 0.0004] | 0.0 | [3, 2] | 2.70 | +0.0% |
| filters | 3 | exhaustive | 8,000 | yes | 0.3575 [0.2619, 0.5440] | 11.7 | [3, 2, 2] | 3.39 | (reference) |
| filters | 3 | pow2 | 125 | yes | 0.0039 [0.0038, 0.0039] | 0.2 | [2, 2, 2] | 3.46 | +2.3% |
| filters | 3 | small | 64 | yes | 0.0021 [0.0020, 0.0022] | 0.1 | [3, 2, 2] | 3.39 | +0.0% |
| filters | 4 | exhaustive | 160,000 | yes | 8.6406 [8.2742, 9.4914] | 297.4 | [2, 2, 2, 2] | 4.03 | (reference) |
| filters | 4 | pow2 | 625 | yes | 0.0277 [0.0269, 0.0295] | 1.1 | [2, 2, 2, 2] | 4.03 | +0.0% |
| filters | 4 | small | 256 | yes | 0.0115 [0.0112, 0.0122] | 0.4 | [2, 2, 2, 2] | 4.03 | +0.0% |
| filters | 5 | exhaustive | 3,200,000 | no | NOT RUN (extrapolated ~185) | NOT RUN (extrapolated ~7,006) | - | - | - |
| filters | 5 | pow2 | 3,125 | yes | 0.1810 [0.1758, 0.3013] | 6.9 | [2, 2, 2, 2, 2] | 4.50 | n/a (exhaustive not run) |
| filters | 5 | small | 1,024 | yes | 0.0592 [0.0564, 0.1025] | 2.2 | [2, 2, 2, 2, 2] | 4.50 | n/a (exhaustive not run) |
| filters | 6 | exhaustive | 64,000,000 | no | NOT RUN (extrapolated ~5,745) | NOT RUN (extrapolated ~163,734) | - | - | - |
| filters | 6 | pow2 | 15,625 | yes | 1.3277 [1.1987, 1.4844] | 40.1 | [2, 2, 2, 2, 2, 2] | 4.90 | n/a (exhaustive not run) |
| filters | 6 | small | 4,096 | yes | 0.2801 [0.2646, 0.4031] | 10.5 | [2, 2, 2, 2, 2, 2] | 4.90 | n/a (exhaustive not run) |
| filter_join | 1 | exhaustive | 20 | yes | 0.0002 [0.0002, 0.0003] | 0.0 | [3] | 1.85 | (reference) |
| filter_join | 1 | pow2 | 5 | yes | 0.0001 [0.0001, 0.0001] | 0.0 | [4] | 1.86 | +0.7% |
| filter_join | 1 | small | 4 | yes | 0.0001 [0.0001, 0.0001] | 0.0 | [3] | 1.85 | +0.0% |
| filter_join | 2 | exhaustive | 640 | yes | 0.0094 [0.0088, 0.0098] | 0.7 | [1, 2] | 30.91 | (reference) |
| filter_join | 2 | pow2 | 30 | yes | 0.0004 [0.0004, 0.0006] | 0.0 | [1, 2] | 30.91 | +0.0% |
| filter_join | 2 | small | 16 | yes | 0.0002 [0.0002, 0.0002] | 0.0 | [1, 2] | 30.91 | +0.0% |
| filter_join | 3 | exhaustive | 12,800 | yes | 0.5327 [0.4812, 0.7286] | 19.5 | [1, 30, 2] | 35.79 | (reference) |
| filter_join | 3 | pow2 | 150 | yes | 0.0057 [0.0056, 0.0060] | 0.2 | [1, 32, 2] | 35.29 | -1.4% |
| filter_join | 3 | small | 64 | yes | 0.0024 [0.0023, 0.0024] | 0.1 | [1, 2, 3] | 49.87 | +39.3% |
| filter_join | 4 | exhaustive | 409,600 | yes | 19.8507 [19.1845, 20.3735] | 794.9 | [1, 32, 1, 2] | 82.54 | (reference) |
| filter_join | 4 | pow2 | 900 | yes | 0.0376 [0.0346, 0.0408] | 1.7 | [1, 32, 1, 2] | 82.54 | +0.0% |
| filter_join | 4 | small | 256 | yes | 0.0104 [0.0102, 0.0107] | 0.5 | [1, 2, 1, 2] | 500.41 | +506.3% |
| filter_join | 5 | exhaustive | 8,192,000 | no | NOT RUN (extrapolated ~554) | NOT RUN (extrapolated ~18,812) | - | - | - |
| filter_join | 5 | pow2 | 4,500 | yes | 0.2908 [0.2703, 0.4056] | 10.3 | [1, 32, 1, 32, 2] | 96.51 | n/a (exhaustive not run) |
| filter_join | 5 | small | 1,024 | yes | 0.0622 [0.0596, 0.1032] | 2.3 | [1, 2, 1, 2, 3] | 845.77 | n/a (exhaustive not run) |
| filter_join | 6 | exhaustive | 262,144,000 | no | NOT RUN (extrapolated ~19,071) | NOT RUN (extrapolated ~707,996) | - | - | - |
| filter_join | 6 | pow2 | 27,000 | yes | 1.9119 [1.8568, 2.2340] | 73.1 | [1, 32, 1, 32, 1, 2] | 174.26 | n/a (exhaustive not run) |
| filter_join | 6 | small | 4,096 | yes | 0.2762 [0.2639, 0.3926] | 11.2 | [1, 2, 1, 2, 1, 2] | 7900.00 | n/a (exhaustive not run) |
| mixed | 1 | exhaustive | 20 | yes | 0.0002 [0.0002, 0.0003] | 0.0 | [3] | 1.85 | (reference) |
| mixed | 1 | pow2 | 5 | yes | 0.0001 [0.0001, 0.0001] | 0.0 | [4] | 1.86 | +0.7% |
| mixed | 1 | small | 4 | yes | 0.0001 [0.0001, 0.0001] | 0.0 | [3] | 1.85 | +0.0% |
| mixed | 2 | exhaustive | 640 | yes | 0.0091 [0.0086, 0.0095] | 0.7 | [1, 2] | 30.91 | (reference) |
| mixed | 2 | pow2 | 30 | yes | 0.0005 [0.0004, 0.0005] | 0.0 | [1, 2] | 30.91 | +0.0% |
| mixed | 2 | small | 16 | yes | 0.0002 [0.0002, 0.0003] | 0.0 | [1, 2] | 30.91 | +0.0% |
| mixed | 3 | exhaustive | 10,240 | yes | 0.3533 [0.3250, 0.5184] | 15.8 | [1, 31, 1] | 40.48 | (reference) |
| mixed | 3 | pow2 | 150 | yes | 0.0049 [0.0044, 0.0053] | 0.2 | [1, 32, 1] | 39.80 | -1.7% |
| mixed | 3 | small | 64 | yes | 0.0019 [0.0019, 0.0021] | 0.1 | [1, 2, 4] | 71.84 | +77.5% |
| mixed | 4 | exhaustive | 204,800 | yes | 10.7102 [10.4046, 10.9822] | 398.5 | [1, 32, 1, 2] | 42.38 | (reference) |
| mixed | 4 | pow2 | 750 | yes | 0.0325 [0.0311, 0.0560] | 1.4 | [1, 32, 1, 2] | 42.38 | +0.0% |
| mixed | 4 | small | 256 | yes | 0.0112 [0.0101, 0.0119] | 0.5 | [1, 2, 4, 2] | 96.29 | +127.2% |
| mixed | 5 | exhaustive | 6,553,600 | no | NOT RUN (extrapolated ~353) | NOT RUN (extrapolated ~15,104) | - | - | - |
| mixed | 5 | pow2 | 4,500 | yes | 0.2206 [0.2160, 0.2523] | 10.4 | [1, 32, 1, 1, 2] | 94.60 | n/a (exhaustive not run) |
| mixed | 5 | small | 1,024 | yes | 0.0502 [0.0477, 0.0598] | 2.4 | [1, 2, 1, 1, 2] | 601.88 | n/a (exhaustive not run) |
| mixed | 6 | exhaustive | 104,857,600 | no | NOT RUN (extrapolated ~7,797) | NOT RUN (extrapolated ~285,617) | - | - | - |
| mixed | 6 | pow2 | 22,500 | yes | 1.7346 [1.6613, 1.8271] | 61.4 | [1, 32, 1, 1, 32, 1] | 120.47 | n/a (exhaustive not run) |
| mixed | 6 | small | 4,096 | yes | 0.2792 [0.2647, 0.2821] | 11.2 | [1, 2, 1, 1, 2, 4] | 1276.86 | n/a (exhaustive not run) |
