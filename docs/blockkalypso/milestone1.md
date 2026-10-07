# Milestone 1: block filter and block join

Brute-force block operators on top of Kalypso. Accuracy is out of scope for
this milestone; every check still reports agreement so it can be studied later.

## What is implemented

| Piece | Where | Notes |
|---|---|---|
| Shared helpers | `vllm/kalypso/sem_ops/block_utils.py` | Row text, fixed-size and k-by-j blocking, single-answer parsing, LLM call wrapper |
| Block filter | `vllm/kalypso/sem_ops/block_filter.py` | One call per block, `<id>: YES/NO` lines, one-row fallback for unusable answers |
| Block tasks | `vllm/kalypso/controller/stage.py` | A `Task` can hold a list of rows; one KV reservation per block |
| Block scheduling | `vllm/kalypso/execution/pipeline_execution.py` | Block stages get one task per block; block boundaries kept between block stages |
| Planner hook | `vllm/kalypso/controller/plan.py` | `sem_filter` with `"block_size" > 1` becomes a `BlockSemFilter` |
| Sampling override | `vllm/kalypso/execution/vllm_executor.py` | Block calls turn off Kalypso's repetition penalties |
| Join id fix | `vllm/kalypso/sem_ops/joins.py` | Join pairs keep the left row id and record the right row id |
| Block join | `vllm/kalypso/sem_ops/block_join.py` | One call per k-by-j block, `L<i>, R<j>` lines, id validation, per-pair fallback; also callable as a pipeline stage with join block sizes |

## Run the unit tests

No model or server needed. From the repo root:

```bash
python -m pytest vllm/kalypso/tests -q -p no:cacheprovider
```

## Start Kalypso's server on a laptop

`vllm serve` currently fails on a circular import inside Kalypso. Start the
server module directly instead:

```bash
VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_CPU_KVCACHE_SPACE=2 \
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-0.5B-Instruct --dtype bfloat16 \
  --max-model-len 4096 --enable-prefix-caching --port 8003
```

Use `--dtype float16` on an M1, which has no bfloat16 support.

## Standalone checks

Both scripts work against any OpenAI-compatible server.

```bash
# Block filter: 20 labeled reviews, one call per row vs one call per block
python vllm/kalypso/benchmark/block_filter_standalone.py \
  --base-url http://localhost:8003/v1 --block-size 20 --show-raw

# Block join: 5 complaints x 5 categories, 25 pair calls vs one 5x5 block
python vllm/kalypso/benchmark/block_join_standalone.py \
  --base-url http://localhost:8003/v1 --k 5 --j 5 --show-raw
```

## Block filter inside a Kalypso query

Add `block_size` to a `sem_filter` op and post the query to
`/v1/semantic/query`:

```json
{
  "data_path": "vllm/kalypso/benchmark/sample_data/block_filter/reviews_20.csv",
  "ops": [
    {"op": "sem_filter", "args": {"prompt": "Does the review complain about the restaurant?", "block_size": 5}},
    {"op": "sem_filter", "args": {"prompt": "Is the complaint about the staff or the service?", "block_size": 5}}
  ]
}
```

## Results on Apple M4 CPU

From the evaluation harness (`vllm/kalypso/benchmark/eval/`): same server and
prompts for every arm, prefix cache cleared before each trial, median of 3
runs. Full tables are in `vllm/kalypso/benchmark/eval/results/`.

Block filter over 20 reviews (10 complaints):

| Block size | LLM requests | Prompt tokens | Latency 0.5B | Latency 1.5B | F1 1.5B |
|---|---|---|---|---|---|
| 1 (Kalypso's SemFilter) | 20 | 2,818 | 10.4 s | 36.1 s | n/a |
| 5 | 4 | 582 | 3.1 s | 9.8 s | 0.27 |
| 10 | 2 | 426 | 3.1 s | 9.4 s | 0.11 |
| 20 | 1 | 352 | 3.6 s | 11.1 s | 0.00 |

5x5 block join (5 expected matches):

| Mode | LLM calls | Prompt tokens | Latency 1.5B | F1 1.5B |
|---|---|---|---|---|
| One call per pair (Kalypso) | 25 | 4,310 | 66.5 s | 0.33 |
| One 5x5 block | 1 | 224 | 6.4 s | 0.60 |

Blocking is 3-4x faster for the filter and 10x for the join. On long rows
(MEDEC clinical notes) it gives no speedup. Accuracy cannot be judged yet:
at 0.5B and 1.5B even Kalypso's per-row operators fail (the 1.5B per-row
filter says no to every review, the per-pair join says yes to every pair).
Measuring the accuracy cost of blocking needs a larger model (Milestone 2).

## Edge cases handled

- Missing, out-of-range, or conflicting answer ids: the row gets its own call.
- Model numbers items from 0: shifted, but only when all n answers are there.
- Markdown, brackets, `<1: NO>`, TRUE/FALSE instead of YES/NO: all parsed.
- A row containing newlines or `2. ...` text cannot pose as another item.
- Empty rows are shown as `(empty)` so their number still appears.
- Truncated outputs (`finish_reason == "length"`) are counted.
- Join pairs with ids outside the blocks are dropped and counted; repeats too.
- A join answer with no pairs and no `NONE` falls back to per-pair calls.
- Blocks of one row, or a 1x1 join block, skip the block prompt.
- Empty inputs make no calls.

## Not done yet (Milestone 2)

- Recursive sub-blocking repair for the filter, and candidate sub-blocking for the join.
- KV pinning for block requests.
