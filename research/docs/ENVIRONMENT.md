# Environment and reproducibility audit (Session 1, Step 2)

Covers where each Session 1 number comes from, which versions produced it,
and what is (and is not) pinned. Sources are file:line at `21c9ee8` unless
stated otherwise.

## 1. Three execution environments

| | CPU container (this branch's CPU work) | Modal GPU (historical results + new runs) | Team laptops (early milestone runs) |
|---|---|---|---|
| Used for | audit, unit tests, telemetry tests, optimizer scaling, executor simulation, cost-model validation | every GPU number in STUDY.md and `research/baseline/results/gpu/` | `results/m1-qwen0.5b`, `results/m1-qwen1.5b` |
| Hardware | 4 vCPU, no GPU | NVIDIA L40S (46068 MiB reported in 62 records, 49140 MiB in 4) | Apple M4 (macOS 15.7.3) |
| Python | 3.13.18 | 3.12.3 (image `vllm/vllm-openai:v0.29.0`) | 3.11.9 |
| torch / transformers | 2.15.0+cu132 / 4.57.6 (CPU use only) | as shipped in the vLLM 0.29.0 image | n/a |
| vLLM | this repo's fork, imported from source (no compiled ops; `vllm._version` missing warning is harmless) | upstream wheel 0.29.0 + the Kalypso overlay (`docker/kalypso/prepare.py`) | fork |
| Model | none (simulated / rule-based LLM) | Qwen/Qwen2.5-7B-Instruct | Qwen2.5-0.5B / 1.5B-Instruct |

The CPU container cannot reach huggingface.co or the ContractNLI host, so
anything that needs the real tokenizer or that dataset runs on Modal only.

## 2. The GPU server configuration

From `cloud/modal_block_size.py` and copied unchanged into
`research/baseline/gpu/modal_baseline.py`: L40S, `max_num_seqs 256`,
`max_num_batched_tokens 2048`, `max_model_len 32768`, prefix caching on
(required, `docker/kalypso/init_api.py:15`), `--no-async-scheduling`
(required, `init_api.py:11`), prefix cache reset before every trial.

**Two different KV budgets exist.** The fork's own server
(`vllm/entrypoints/openai/api_server.py:1570`) gives Kalypso
`per_gpu_budget * tp_size * 0.5`; the Modal overlay (`init_api.py:17-22`)
gives `per_gpu * tp_size` (factor 1.0). All historical GPU results and all
Session 1 GPU runs use the overlay, so the planner's KV ledger there is twice
what a fork-server run would see. Results are not comparable across the two
without accounting for this.

## 3. Determinism and seeds

| Source of variation | Pinned? | Where |
|---|---|---|
| Sampling | yes: temperature 0.0, seed 42 | `vllm/kalypso/execution/vllm_executor.py:35-36` |
| Workload row selection | yes: `SELECTIVITY_SEED` 0, `FANOUT_SEED` 0 | `eval/workloads.py:282,317` |
| Arm order | fixed interleaving per repeat | `eval/run.py` |
| Planner sampling for KV estimates | yes since phase2/00: `SemanticPlan(plan_seed=0)` (was unseeded `random.sample`) | `controller/plan.py` `DEFAULT_PLAN_SEED`; `tests/test_plan_seed.py` |
| Downstream block membership | **no**: depends on task completion order (`asyncio.wait` returns a set). Changes *which rows share a block*, not just order; final results change only if a row's verdict depends on its block-mates | audit §4.6; `tests/test_block_membership_order.py` |
| GPU batch composition | no (inherent to continuous batching); greedy decoding still varies at the margin with batch shape | vLLM |
| Model weights revision | pinned since phase2/00 to `a09a3545…8bc28` (the commit all Session 1 runs resolved): `--revision` in both Modal runners, the KV manager's tokenizer/config follow `ModelConfig.revision`, cost-model tokenizer pinned. Session 1 itself ran unpinned (`revision=main`) | `modal_baseline.py` `MODEL_REVISION`; `modal_block_size.py` `MODEL_REVISIONS`; `budget.py`; `costmodel.py` `TOKENIZER_REVISION` |

## 4. Provenance of historical results

`git_dirty` as recorded in each result file's meta line:

| results folder | dirty | clean |
|---|---|---|
| `experiments/` (L40S, 7B) | 53 | 13 |
| `block-size/` (7B) | 14 | 1 |
| `m1-qwen0.5b/` | 4 | 0 |
| `m1-qwen1.5b/` | 0 | 3 |

So most historical numbers were produced from a working tree with
uncommitted changes, and the exact code that produced them cannot be
recovered from git. Session 1 treats them as given data for the cost-model
validation (that is what the committed `costmodel.json` was fit on) but does
not use them as the baseline for new claims.

## 5. What Session 1 records per GPU run

`modal_baseline.py` writes to each experiment's folder: `env.json` (package
versions, CUDA, driver, GPU name and memory, model revision, vLLM's KV cache
size, seeds, git sha and dirty flag of the launching tree), `tokens.json`
(tokenizer row counts per workload), `gpu-*.csv` (nvidia-smi every 0.5 s),
`vllm-metrics-*.jsonl` (`/metrics` every 0.5 s), `trace-*.jsonl`
(KALYPSO_TRACE), `server-*.log` (committed gzipped), `run-*.jsonl` (eval.run output), and
`status.json`.

## 6. Reproduction

```bash
# CPU (any machine with Python >= 3.11)
python -m venv .venv && .venv/bin/pip install torch -r requirements/common.txt pytest
.venv/bin/python -m pytest -q vllm/kalypso/tests
.venv/bin/python -m research.baseline.experiments.optimizer_scaling --out research/baseline/results/optimizer_scaling.jsonl
.venv/bin/python -m research.baseline.experiments.executor_simulation --out research/baseline/results/executor_simulation
.venv/bin/python -m research.baseline.analysis.costmodel_validation --out research/baseline/results/costmodel_validation
# GPU (Modal account with the HF cache volume used by cloud/modal_block_size.py)
modal run --detach research/baseline/gpu/modal_baseline.py::launch
modal volume get blockkalypso-results research-session1 ./research/baseline/results/gpu/
```

## 7. Hazards to fix before Session 2 comparisons

1. ~~Pin the model revision in both Modal runners.~~ Done (phase2/00).
2. ~~Seed the planner's `random.sample` (or record its draw).~~ Done (phase2/00).
3. Decide on one KV budget factor (0.5 vs 1.0) and record it in result meta.
   phase2/00: **not reconciled**; both are now named constants
   (`init_api.py` `KV_BUDGET_FACTOR` = 1.0, `api_server.py`
   `FORK_KV_BUDGET_FACTOR` = 0.5) and `modal_baseline.py` records
   `kalypso_kv_budget_factor` in `env.json`. Session 1 traces confirm 1.0
   (Kalypso capacity = vLLM's available KV memory in every run).
4. Refuse to record results from a dirty tree, or store the diff with them.
5. Report block-composition variance explicitly; it is a real source of
   run-to-run variation in downstream block quality and latency.
