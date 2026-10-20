# Static vs dynamic block size

regret(c) = latency(arm, c) / latency(k*(c), c); k_static = the arm with the lowest worst-case regret; threshold 1.1. `*` = the arm's pipeline work drifted from per-tuple in that condition.

| Condition | k*(c) | Latency (s) |
|---|---|---|
| m2-E6-qwen14b/s10 (s=0.1, f=1.0) | `kalypso_block:4/4x4/4` * | 5.32 |
| m2-E6-qwen14b/s30 (s=0.3, f=1.0) | `kalypso_block:4/4x4/16` * | 7.74 |
| m2-E6-qwen14b/s60 (s=0.6, f=1.0) | `kalypso_block:4/4x4/4` * | 11.34 |
| m2-E6-qwen14b/s90 (s=0.9, f=1.0) | `kalypso_block:4/4x4/4` * | 12.96 |
| m2-E7-qwen14b/f02 (s=0.5, f=0.2) | `kalypso_block:4/4x4/16` * | 9.57 |
| m2-E7-qwen14b/f10 (s=0.5, f=1.0) | `kalypso_block:4/4x4/4` * | 9.80 |
| m2-E7-qwen14b/f30 (s=0.5, f=3.0) | `kalypso_block:4/4x4/16` * | 11.01 |
| m2-E7-qwen14b/f50 (s=0.5, f=5.0) | `kalypso_block:4/4x4/4` * | 15.46 |

| Arm | m2-E6-qwen14b/s10 (s=0.1, f=1.0) | m2-E6-qwen14b/s30 (s=0.3, f=1.0) | m2-E6-qwen14b/s60 (s=0.6, f=1.0) | m2-E6-qwen14b/s90 (s=0.9, f=1.0) | m2-E7-qwen14b/f02 (s=0.5, f=0.2) | m2-E7-qwen14b/f10 (s=0.5, f=1.0) | m2-E7-qwen14b/f30 (s=0.5, f=3.0) | m2-E7-qwen14b/f50 (s=0.5, f=5.0) | Worst | Mean |
|---|---|---|---|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 1.42 | 2.70 | 3.42 | 4.00 | 3.46 | 3.38 | 2.98 | 2.04 | 4.00 | 2.92 |
| `kalypso_block:4/4x4/16` | 1.00 * | 1.00 * | 1.00 * | 1.01 * | 1.00 * | 1.09 * | 1.00 * | 1.20 * | 1.20 | 1.04 |
| `kalypso_block:4/4x4/4` | 1.00 * | 1.04 * | 1.00 * | 1.00 * | 1.00 * | 1.00 * | 1.01 * | 1.00 * | 1.04 | 1.01 |
| `kalypso_block:4/2x2/4` | 1.26 * | 1.46 * | 1.68 * | 1.76 * | 1.68 * | 1.69 * | 1.61 * | 1.19 * | 1.76 | 1.54 |

k_static: `kalypso_block:4/4x4/4` (worst-case regret 1.04, mean 1.01).
Verdict: a static k is enough (regret <= 1.1 in every condition).
