# BlockKalypso

Extending [Kalypso](https://github.com/goodluck-hojae/kalypso), a relational
LLM serving system, with **block-level semantic operators**: one LLM call
evaluates a filter, join or map over a whole block of tuples instead of one
tuple (or tuple pair) at a time.

**Course:** COMPSCI 692S F26 Systems for ML
**Proposal:** [`systems_for_ml_proposal.pdf`](./systems_for_ml_proposal.pdf)

**Team**
- Anjali Kanvinde
- Bhavana Anand
- Gomathy Dhanya Sankara Subramanian
- Sameeksha Bhatia

## Repository layout

This repo contains the full Kalypso source tree (vLLM `v0.13.0rc4` plus the
semantic query processor), merged with its git history. Our work goes on top
of it.

| Path | What it is |
|---|---|
| `vllm/kalypso/` | Kalypso's semantic query processor. **Most of our changes go here.** |
| `vllm/kalypso/sem_ops/` | Semantic operators (filter, join, map, ...). Block operators go here. |
| `vllm/kalypso/controller/` | `Task`, `Stage`, plan builder. Block tasks and block-granularity scheduler signals go here. |
| `vllm/kalypso/budget.py` | `KVMemoryManager`, per-stage KV memory reservations. |
| `vllm/kalypso/execution/` | Pipelined stage scheduler and the vLLM executor. |
| `vllm/kalypso/benchmark/` | Workload clients from the Kalypso paper (ContractNLI, BioDEX, ...). |
| `KALYPSO_README.md` | Upstream Kalypso README (serving and ICP service instructions). |

## Milestone 1

Block filter and block join, how to test and run them, and first results:
[`docs/blockkalypso/milestone1.md`](./docs/blockkalypso/milestone1.md).

## Git workflow

- `origin` is this repo. `upstream` is Kalypso:

  ```bash
  git remote add upstream https://github.com/goodluck-hojae/kalypso.git
  ```

- Branch off `main` for each piece of work (e.g. `block-filter`,
  `block-join`, `scheduler-signals`, `eval-baseline`) and open a pull request.
- To pick up upstream Kalypso fixes:

  ```bash
  git fetch upstream
  git merge upstream/main
  ```

## Setup

Kalypso is a fork of vLLM, so it is installed from source. Follow
[`KALYPSO_README.md`](./KALYPSO_README.md) on a CUDA machine:

```bash
pip install -U pip setuptools wheel ninja cmake packaging
pip install -r requirements/build.txt
pip install -r requirements/common.txt
pip install -e . --no-build-isolation
```

The course asks for a local vLLM instance on a laptop with the smallest
viable model. Instructions for laptops without an NVIDIA GPU (vLLM CPU
backend) will be added once verified.
