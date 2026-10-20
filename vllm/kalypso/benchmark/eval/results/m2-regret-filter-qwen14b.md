# Static vs dynamic block size

regret(c) = latency(arm, c) / latency(k*(c), c); k_static = the arm with the lowest worst-case regret; threshold 1.1. `*` = the arm's pipeline work drifted from per-tuple in that condition.

| Condition | k*(c) | Latency (s) |
|---|---|---|
| m2-E1-qwen14b/long_contracts | `kalypso_block:2` | 19.54 |
| m2-E1-qwen14b/medium_medec | `kalypso_block:2` | 4.71 |
| m2-E1-qwen14b/short_arxiv | `kalypso_block:2` | 3.05 |
| m2-E3-qwen14b/n128 | `kalypso_block:4` | 6.05 |
| m2-E3-qwen14b/n16 | `kalypso_per_tuple` | 1.09 |
| m2-E3-qwen14b/n32 | `kalypso_block:4` | 1.96 |
| m2-E3-qwen14b/n64 | `kalypso_block:4` | 3.29 |

| Arm | m2-E1-qwen14b/long_contracts | m2-E1-qwen14b/medium_medec | m2-E1-qwen14b/short_arxiv | m2-E3-qwen14b/n128 | m2-E3-qwen14b/n16 | m2-E3-qwen14b/n32 | m2-E3-qwen14b/n64 | Worst | Mean |
|---|---|---|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 1.01 | 1.30 | 1.20 | 1.16 | 1.00 | 1.00 | 1.10 | 1.30 | 1.11 |
| `kalypso_block:4` | 1.10 | 1.07 | 1.09 | 1.00 | 1.17 | 1.00 | 1.00 | 1.17 | 1.06 |
| `kalypso_block:16` | failed | 1.71 | 1.93 | 1.46 | 3.20 | 2.25 | 1.78 | failed | failed |

k_static: `kalypso_block:4` (worst-case regret 1.17, mean 1.06).
Verdict: dynamic k / reforming is needed; regret exceeds 1.1 for: m2-E1-qwen14b/long_contracts, m2-E3-qwen14b/n16.
