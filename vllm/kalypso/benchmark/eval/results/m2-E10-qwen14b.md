# E10: accuracy cost of larger blocks

F1 against labels, agreement with per-tuple Kalypso (the proposal's accuracy measure) and fallback calls, medians over repeats. Drifted = the pipeline's work differed from per-tuple by more than 10% at Filter1, the join or the output.

F1 vs per-tuple takes per-tuple's output as the truth; agreement counts every candidate, so it stays high when outputs are sparse.

| Experiment | Condition | Blocks | F1 vs labels | F1 vs per-tuple | Agrees w/ per-tuple | Fallback calls | Drifted |
|---|---|---|---|---|---|---|---|
| m2-E1-qwen14b | long_contracts | per-tuple | 1.00 | 1.00 | 100% | 0 |  |
| m2-E1-qwen14b | long_contracts | 2 | 1.00 | 1.00 | 100% | 0 |  |
| m2-E1-qwen14b | long_contracts | 4 | 0.98 | 0.98 | 95% | 0 |  |
| m2-E1-qwen14b | long_contracts | 8 | 0.98 | 0.98 | 95% | 0 |  |
| m2-E1-qwen14b | long_contracts | 16 | failed (3 trials) | | | | |
| m2-E1-qwen14b | long_contracts | 32 | failed (3 trials) | | | | |
| m2-E1-qwen14b | long_contracts | 2@4096 | 0.99 | 0.99 | 98% | 20 |  |
| m2-E1-qwen14b | long_contracts | 4@4096 | 1.00 | 1.00 | 100% | 20 |  |
| m2-E1-qwen14b | long_contracts | 8@4096 | 1.00 | 1.00 | 100% | 20 |  |
| m2-E1-qwen14b | long_contracts | 16@4096 | 1.00 | 1.00 | 100% | 20 |  |
| m2-E1-qwen14b | long_contracts | 32@4096 | 1.00 | 1.00 | 100% | 20 |  |
| m2-E1-qwen14b | medium_medec | per-tuple | 0.71 | 1.00 | 100% | 0 |  |
| m2-E1-qwen14b | medium_medec | 2 | 0.51 | 0.54 | 50% | 0 |  |
| m2-E1-qwen14b | medium_medec | 4 | 0.55 | 0.51 | 47% | 0 |  |
| m2-E1-qwen14b | medium_medec | 8 | 0.51 | 0.51 | 47% | 0 |  |
| m2-E1-qwen14b | medium_medec | 16 | 0.49 | 0.41 | 41% | 0 |  |
| m2-E1-qwen14b | medium_medec | 32 | 0.40 | 0.33 | 38% | 0 |  |
| m2-E1-qwen14b | medium_medec | 2@4096 | 0.51 | 0.54 | 50% | 0 |  |
| m2-E1-qwen14b | medium_medec | 4@4096 | 0.55 | 0.51 | 47% | 0 |  |
| m2-E1-qwen14b | medium_medec | 8@4096 | 0.45 | 0.47 | 44% | 1 |  |
| m2-E1-qwen14b | medium_medec | 16@4096 | 0.50 | 0.42 | 39% | 0 |  |
| m2-E1-qwen14b | medium_medec | 32@4096 | 0.50 | 0.42 | 39% | 0 |  |
| m2-E1-qwen14b | short_arxiv | per-tuple | 1.00 | 1.00 | 100% | 0 |  |
| m2-E1-qwen14b | short_arxiv | 2 | 0.95 | 0.95 | 95% | 0 |  |
| m2-E1-qwen14b | short_arxiv | 4 | 1.00 | 1.00 | 100% | 0 |  |
| m2-E1-qwen14b | short_arxiv | 8 | 1.00 | 1.00 | 100% | 0 |  |
| m2-E1-qwen14b | short_arxiv | 16 | 0.98 | 0.98 | 98% | 0 |  |
| m2-E1-qwen14b | short_arxiv | 32 | 0.95 | 0.95 | 95% | 0 |  |
| m2-E1-qwen14b | short_arxiv | 2@4096 | 0.95 | 0.95 | 95% | 0 |  |
| m2-E1-qwen14b | short_arxiv | 4@4096 | 1.00 | 1.00 | 100% | 0 |  |
| m2-E1-qwen14b | short_arxiv | 8@4096 | 1.00 | 1.00 | 100% | 0 |  |
| m2-E1-qwen14b | short_arxiv | 16@4096 | 1.00 | 1.00 | 100% | 0 |  |
| m2-E1-qwen14b | short_arxiv | 32@4096 | 1.00 | 1.00 | 100% | 0 |  |
| m2-E2-qwen14b | join_16x8 | per-tuple | 0.69 | 1.00 | 100% | 0 |  |
| m2-E2-qwen14b | join_16x8 | 1x2 | 0.35 | 0.60 | 94% | 0 |  |
| m2-E2-qwen14b | join_16x8 | 1x4 | 0.38 | 0.56 | 94% | 0 |  |
| m2-E2-qwen14b | join_16x8 | 1x8 | 0.45 | 0.53 | 93% | 0 |  |
| m2-E2-qwen14b | join_16x8 | 2x1 | 0.48 | 0.64 | 94% | 0 |  |
| m2-E2-qwen14b | join_16x8 | 2x2 | 0.35 | 0.40 | 91% | 0 |  |
| m2-E2-qwen14b | join_16x8 | 2x4 | 0.40 | 0.45 | 91% | 0 |  |
| m2-E2-qwen14b | join_16x8 | 2x8 | 0.57 | 0.72 | 95% | 0 |  |
| m2-E2-qwen14b | join_16x8 | 4x1 | 0.52 | 0.60 | 94% | 0 |  |
| m2-E2-qwen14b | join_16x8 | 4x2 | 0.38 | 0.44 | 92% | 0 |  |
| m2-E2-qwen14b | join_16x8 | 4x4 | 0.55 | 0.53 | 93% | 0 |  |
| m2-E2-qwen14b | join_16x8 | 4x8 | 0.43 | 0.50 | 92% | 0 |  |
| m2-E2-qwen14b | join_16x8 | 8x1 | 0.46 | 0.70 | 95% | 0 |  |
| m2-E2-qwen14b | join_16x8 | 8x2 | 0.48 | 0.56 | 94% | 0 |  |
| m2-E2-qwen14b | join_16x8 | 8x4 | 0.30 | 0.47 | 93% | 0 |  |
| m2-E2-qwen14b | join_16x8 | 8x8 | 0.18 | 0.42 | 91% | 0 |  |
| m2-E3-qwen14b | n128 | per-tuple | 0.99 | 1.00 | 100% | 0 |  |
| m2-E3-qwen14b | n128 | 4 | 0.98 | 0.98 | 98% | 0 |  |
| m2-E3-qwen14b | n128 | 16 | 0.96 | 0.97 | 97% | 0 |  |
| m2-E3-qwen14b | n16 | per-tuple | 1.00 | 1.00 | 100% | 0 |  |
| m2-E3-qwen14b | n16 | 4 | 1.00 | 1.00 | 100% | 0 |  |
| m2-E3-qwen14b | n16 | 16 | 1.00 | 1.00 | 100% | 0 |  |
| m2-E3-qwen14b | n32 | per-tuple | 1.00 | 1.00 | 100% | 0 |  |
| m2-E3-qwen14b | n32 | 4 | 1.00 | 1.00 | 100% | 0 |  |
| m2-E3-qwen14b | n32 | 16 | 1.00 | 1.00 | 100% | 0 |  |
| m2-E3-qwen14b | n64 | per-tuple | 1.00 | 1.00 | 100% | 0 |  |
| m2-E3-qwen14b | n64 | 4 | 1.00 | 1.00 | 100% | 0 |  |
| m2-E3-qwen14b | n64 | 16 | 0.98 | 0.98 | 98% | 0 |  |
| m2-E4-qwen14b-v2 | fjf_s05_f1 | per-tuple | 0.80 | 1.00 | 100% | 0 | no |
| m2-E4-qwen14b-v2 | fjf_s05_f1 | 2/2x2/2 | 0.15 | 0.20 | 99% | 4 | yes |
| m2-E4-qwen14b-v2 | fjf_s05_f1 | 4/4x4/4 | 0.00 | 0.00 | 99% | 2 | yes |
| m2-E4-qwen14b-v2 | fjf_s05_f1 | 8/8x8/8 | 0.36 | 0.50 | 100% | 0 | yes |
| m2-E4-qwen14b-v2 | fjf_s05_f1 | 16/16x16/16 | n/a | 0.00 | 99% | 0 | yes |
| m2-E5-qwen14b-v2 | fjf_s05_f1 | per-tuple | 0.80 | 1.00 | 100% | 0 | no |
| m2-E5-qwen14b-v2 | fjf_s05_f1 | 4/2x2/4 | 0.14 | 0.18 | 99% | 6 | yes |
| m2-E5-qwen14b-v2 | fjf_s05_f1 | 4/2x2/16 | 0.00 | 0.00 | 99% | 4 | yes |
| m2-E5-qwen14b-v2 | fjf_s05_f1 | 4/4x4/4 | 0.00 | 0.00 | 99% | 2 | yes |
| m2-E5-qwen14b-v2 | fjf_s05_f1 | 4/4x4/16 | 0.00 | 0.00 | 99% | 2 | yes |
| m2-E5-qwen14b-v2 | fjf_s05_f1 | 4/4x32/4 | 0.00 | 0.00 | 99% | 2 | yes |
| m2-E5-qwen14b-v2 | fjf_s05_f1 | 4/4x32/16 | 0.00 | 0.00 | 99% | 1 | yes |
| m2-E5-qwen14b-v2 | fjf_s05_f1 | 16/2x2/4 | 0.00 | 0.00 | 99% | 4 | yes |
| m2-E5-qwen14b-v2 | fjf_s05_f1 | 16/2x2/16 | 0.00 | 0.00 | 99% | 4 | yes |
| m2-E5-qwen14b-v2 | fjf_s05_f1 | 16/4x4/4 | 0.00 | 0.00 | 99% | 1 | yes |
| m2-E5-qwen14b-v2 | fjf_s05_f1 | 16/4x4/16 | 0.00 | 0.00 | 99% | 1 | yes |
| m2-E5-qwen14b-v2 | fjf_s05_f1 | 16/4x32/4 | 0.00 | 0.00 | 99% | 2 | yes |
| m2-E5-qwen14b-v2 | fjf_s05_f1 | 16/4x32/16 | 0.00 | 0.00 | 99% | 1 | yes |
| m2-E6-qwen14b | s10 | per-tuple | n/a | 1.00 | 100% | 0 | no |
| m2-E6-qwen14b | s10 | 4/4x4/16 | 0.50 | 0.00 | 100% | 0 | yes |
| m2-E6-qwen14b | s10 | 4/4x4/4 | 0.50 | 0.00 | 100% | 0 | yes |
| m2-E6-qwen14b | s10 | 4/2x2/4 | 0.00 | 0.00 | 100% | 1 | yes |
| m2-E6-qwen14b | s30 | per-tuple | 0.57 | 1.00 | 100% | 0 | no |
| m2-E6-qwen14b | s30 | 4/4x4/16 | 0.40 | 0.50 | 100% | 1 | yes |
| m2-E6-qwen14b | s30 | 4/4x4/4 | 0.40 | 0.50 | 100% | 1 | yes |
| m2-E6-qwen14b | s30 | 4/2x2/4 | 0.40 | 0.50 | 100% | 1 | yes |
| m2-E6-qwen14b | s60 | per-tuple | 0.67 | 1.00 | 100% | 0 | no |
| m2-E6-qwen14b | s60 | 4/4x4/16 | 0.14 | 0.50 | 99% | 4 | yes |
| m2-E6-qwen14b | s60 | 4/4x4/4 | 0.14 | 0.50 | 99% | 4 | yes |
| m2-E6-qwen14b | s60 | 4/2x2/4 | 0.12 | 0.40 | 99% | 4 | yes |
| m2-E6-qwen14b | s90 | per-tuple | 0.18 | 1.00 | 100% | 0 | no |
| m2-E6-qwen14b | s90 | 4/4x4/16 | 0.19 | 0.00 | 100% | 3 | yes |
| m2-E6-qwen14b | s90 | 4/4x4/4 | 0.20 | 0.00 | 100% | 2 | yes |
| m2-E6-qwen14b | s90 | 4/2x2/4 | 0.20 | 0.00 | 100% | 5 | yes |
| m2-E7-qwen14b | f02 | per-tuple | 1.00 | 1.00 | 100% | 0 | no |
| m2-E7-qwen14b | f02 | 4/4x4/16 | n/a | 0.00 | 100% | 0 | yes |
| m2-E7-qwen14b | f02 | 4/4x4/4 | n/a | 0.00 | 100% | 0 | yes |
| m2-E7-qwen14b | f02 | 4/2x2/4 | n/a | 0.00 | 100% | 0 | yes |
| m2-E7-qwen14b | f10 | per-tuple | 0.80 | 1.00 | 100% | 0 | no |
| m2-E7-qwen14b | f10 | 4/4x4/16 | 0.00 | 0.00 | 99% | 1 | yes |
| m2-E7-qwen14b | f10 | 4/4x4/4 | 0.00 | 0.00 | 99% | 2 | yes |
| m2-E7-qwen14b | f10 | 4/2x2/4 | 0.14 | 0.18 | 99% | 6 | yes |
| m2-E7-qwen14b | f30 | per-tuple | 0.65 | 1.00 | 100% | 0 | no |
| m2-E7-qwen14b | f30 | 4/4x4/16 | 0.08 | 0.12 | 99% | 1 | yes |
| m2-E7-qwen14b | f30 | 4/4x4/4 | 0.07 | 0.11 | 98% | 1 | yes |
| m2-E7-qwen14b | f30 | 4/2x2/4 | 0.21 | 0.30 | 99% | 3 | yes |
| m2-E7-qwen14b | f50 | per-tuple | 0.62 | 1.00 | 100% | 0 | no |
| m2-E7-qwen14b | f50 | 4/4x4/16 | 0.20 | 0.30 | 98% | 1 | yes |
| m2-E7-qwen14b | f50 | 4/4x4/4 | 0.24 | 0.36 | 98% | 3 | yes |
| m2-E7-qwen14b | f50 | 4/2x2/4 | 0.30 | 0.45 | 98% | 3 | yes |
| m2-E8-qwen14b | fjf_s05_f1 | per-tuple | 0.80 | 1.00 | 100% | 0 | no |
| m2-E8-qwen14b | fjf_s05_f1 | per-tuple+materialized | 0.80 | 1.00 | 100% | 0 | no |
| m2-E8-qwen14b | fjf_s05_f1 | 4/4x4/4 | 0.00 | 0.00 | 99% | 2 | yes |
| m2-E8-qwen14b | fjf_s05_f1 | 4/4x4/4+materialized | n/a | 0.00 | 99% | 0 | yes |
| m2-E8-qwen14b | fjf_s05_f1 | 4/2x2/4 | 0.00 | 0.00 | 99% | 4 | yes |
| m2-E8-qwen14b | fjf_s05_f1 | 4/2x2/4+materialized | 0.00 | 0.00 | 99% | 0 | yes |
| m2-E9-qwen14b | s01_f1 | per-tuple | n/a | 1.00 | 100% | 0 | no |
| m2-E9-qwen14b | s01_f1 | 4/4x4/4 | 0.00 | 0.00 | 100% | 1 | yes |
| m2-E9-qwen14b | s01_f1 | 4/4x4/4+coalesce | 0.00 | 0.00 | 100% | 1 | yes |
| m2-E9-qwen14b | s01_f1 | 4/2x2/4 | 0.00 | 0.00 | 100% | 1 | yes |
| m2-E9-qwen14b | s01_f1 | 4/2x2/4+coalesce | 0.00 | 0.00 | 100% | 1 | yes |
| m2-E9-qwen14b | s05_f02 | per-tuple | 1.00 | 1.00 | 100% | 0 | no |
| m2-E9-qwen14b | s05_f02 | 4/4x4/4 | n/a | 0.00 | 100% | 0 | yes |
| m2-E9-qwen14b | s05_f02 | 4/4x4/4+coalesce | n/a | 0.00 | 100% | 0 | yes |
| m2-E9-qwen14b | s05_f02 | 4/2x2/4 | n/a | 0.00 | 100% | 0 | yes |
| m2-E9-qwen14b | s05_f02 | 4/2x2/4+coalesce | n/a | 0.00 | 100% | 0 | yes |
| m2-E9-qwen14b | s05_f1 | per-tuple | 0.80 | 1.00 | 100% | 0 | no |
| m2-E9-qwen14b | s05_f1 | 4/4x4/4 | 0.00 | 0.00 | 99% | 1 | yes |
| m2-E9-qwen14b | s05_f1 | 4/4x4/4+coalesce | 0.00 | 0.00 | 99% | 0 | yes |
| m2-E9-qwen14b | s05_f1 | 4/2x2/4 | 0.00 | 0.00 | 99% | 4 | yes |
| m2-E9-qwen14b | s05_f1 | 4/2x2/4+coalesce | 0.00 | 0.00 | 99% | 0 | yes |
| m2-E11-qwen14b | fjf_s05_f1 | per-tuple | 0.80 | 1.00 | 100% | 0 | no |
| m2-E11-qwen14b | fjf_s05_f1 | 4/trummer/4 | n/a | 0.00 | 99% | 0 | yes |
| m2-E11-qwen14b | fjf_s05_f1 | 4/4x4/4 | 0.00 | 0.00 | 99% | 2 | yes |
| m2-E11-qwen14b | join_16x8 | per-tuple | 0.69 | 1.00 | 100% | 0 |  |
| m2-E11-qwen14b | join_16x8 | trummer | 0.20 | 0.47 | 93% | 0 |  |
| m2-E11-qwen14b | join_16x8 | 8x4 | 0.30 | 0.47 | 93% | 0 |  |
