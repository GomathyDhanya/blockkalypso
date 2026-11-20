# Session 1 progress: research baseline

Branch: `research/session1-baseline`, cut from `main` at `21c9ee8` (2026-11-04,
"updated Milestone 3 Report"). Scope: **Session 1 only** (baseline,
instrumentation, cost-model validation, protocol). Phase 2 does not start
until the team reviews this branch.

Ground rules for this branch:

- The original optimizer and executor semantics are not changed. All new
  instrumentation is opt-in (off by default) and lives beside the original
  code.
- Historical results under `vllm/kalypso/benchmark/eval/results/` are never
  edited. New results go under `research/baseline/results/`.
- Any number not measured is marked **NOT EXECUTED**, never estimated.

## Checkpoints

| # | Step | Status | Commit |
|---|---|---|---|
| 0 | Branch created, this log | done | (this commit) |
| 1 | Repository audit (`research/docs/BASELINE_AUDIT.md`) | in progress | |
| 2 | Environment and reproducibility audit | not started | |
| 3 | Opt-in instrumentation + tests | not started | |
| 4 | CPU-only experiments (optimizer scaling, simulations) | not started | |
| 5 | Cost-model validation on existing GPU results | not started | |
| 6 | New GPU baseline runs (Modal) | not started | |
| 7 | Protocol, hypotheses, final report | not started | |

## Missing measurements

(filled in as steps finish)
