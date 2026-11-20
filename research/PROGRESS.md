# Session 1 progress and handoff: research baseline

Branch: `research/session1-baseline`, cut from `main` at `21c9ee8` (2026-11-04,
"updated Milestone 3 Report"). Scope: **Session 1 only** (baseline,
instrumentation, cost-model validation, protocol). Phase 2 does not start
until the team reviews this branch.

Ground rules for this branch:

- The original optimizer and executor semantics are not changed. All new
  instrumentation is opt-in (off unless `KALYPSO_TRACE` is set) and is tested
  to give identical query results with it on or off.
- Historical results under `vllm/kalypso/benchmark/eval/results/` are never
  edited. New results go under `research/baseline/results/`.
- Any number not measured is marked **NOT EXECUTED**, never estimated.

## Status

| # | Step | Status | Where |
|---|---|---|---|
| 0 | Branch + this log | done | `research/PROGRESS.md` |
| 1 | Repository audit | done | `research/docs/BASELINE_AUDIT.md` |
| 2 | Environment and reproducibility audit | in progress | `research/docs/ENVIRONMENT.md` (next) |
| 3 | Opt-in instrumentation + tests | done: executor telemetry + optimizer profiler (`research/baseline/instrumentation/optimizer_profile.py`) | `vllm/kalypso/telemetry.py`, hooks in `controller/`, `execution/`, `sem_ops/block_*`, `budget.py`; tests `vllm/kalypso/tests/test_telemetry.py` |
| 4 | CPU-only experiments (optimizer scaling, executor simulation) | optimizer scaling running; simulation next | `research/baseline/experiments/` |
| 5 | Cost-model validation on existing GPU results | analysis done, write-up next | `research/baseline/analysis/costmodel_validation.py` → `research/baseline/results/costmodel_validation/summary.md` |
| 6 | New GPU baseline runs (Modal, via Xen's Mac) | runner written, launching | `research/baseline/gpu/modal_baseline.py` → volume `blockkalypso-results:research-session1/` → `research/baseline/results/gpu/` |
| 7 | Protocol, hypotheses, final report | not started | `research/docs/` |

## How to pick this up

```bash
git fetch origin && git checkout research/session1-baseline
python -m venv .venv && .venv/bin/pip install torch -r requirements/common.txt pytest
.venv/bin/python -m pytest -q vllm/kalypso/tests      # 141 pass at the last checkpoint
```

Telemetry: set `KALYPSO_TRACE=/path/trace.jsonl` (and optionally
`KALYPSO_RUN_ID`) in the server's environment; every query appends events
(`query_start`, `plan_built`, `task_start/end`, `llm_call`, `block_eval`,
`scheduler_sample`, `query_end`, ...). See the module docstring.

## Open issues / notes for the next person

- GitHub is not linked in the cloud session that did this work; pushes go
  through Xen's Mac (git bundle → push). Modal runs also go through the Mac
  (`modal` CLI, profile on that machine).
- The cloud container cannot reach huggingface.co or stanfordnlp.github.io,
  so tokenizer-dependent and ContractNLI-dependent steps run on Modal only.
- Found while auditing: downstream block membership is not reproducible run
  to run (audit §4.6); planning uses an unseeded `random.sample` (§4.1).

## Commands (all from the repo root)

```bash
# CPU: optimizer scaling (about 15 min on 4 cores)
python -m research.baseline.experiments.optimizer_scaling --out research/baseline/results/optimizer_scaling.jsonl
# CPU: cost-model validation on the committed GPU results (about 1 min)
python -m research.baseline.analysis.costmodel_validation --out research/baseline/results/costmodel_validation
# GPU (Modal): every Session 1 experiment, detached; then download
modal run --detach research/baseline/gpu/modal_baseline.py::launch
modal run research/baseline/gpu/modal_baseline.py::status
modal volume get blockkalypso-results research-session1 ./research/baseline/results/gpu/
```

## Missing measurements

- GPU experiments E0-E3: launched from Xen's Mac on Modal (see status below once they return).

## Commit log

| Commit | What |
|---|---|
| `41a3c20` | Branch + progress log |
| `ab03d8d` | Audit |
| `3d922dd` | Opt-in telemetry + tests |
| (next) | Optimizer profiler, scaling experiment, cost-model validation, research workloads, Modal runner |
