Model `Qwen/Qwen2.5-7B-Instruct` on unknown (Linux-4.19.0-gvisor-x86_64-with-glibc2.39); workload `cnli_join_high` (join), 40 x 8 = 320 pairs, 219 positive; 3 repeats after 1 warmup; prefix cache reset per trial: yes; git `5891dbbb5` (dirty).
Server: `python3 -m vllm.entrypoints.cli.main serve Qwen/Qwen2.5-7B-Instruct --host 127.0.0.1 --port 8000 --enable-prefix-caching --no-async-scheduling --api-server-count 1 --max-model-len 32768 --max-num-seqs 256 --max-num-batched-tokens 2048` (started Tue Nov  3 22:06:24 2026).

| Arm | Latency (s) | LLM requests | Block calls | Fallback calls | Prompt tokens | Gen tokens | Prefix hit rate | Peak KV % | Peak running | Peak waiting | Preemptions | Peak RSS (MiB) | Items/s | Truncated | Mean prefill (s) | Mean decode (s) | Accuracy | Precision | Recall | F1 | Agrees w/ kalypso_per_tuple | Self-agreement |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 9.43 [9.22-9.54] | 320 | n/a | n/a | 578728 | 795 | 0.82 | 1.4 | 17 [16-17] | 217 [217-218] | 0 | 11515 [11506-11515] | 33.93 [33.56-34.70] | 0 | 0.193 [0.188-0.194] | 0.241 [0.235-0.241] | 0.59 | 0.95 | 0.42 | 0.58 | 100% | 100% |
| `kalypso_block:4` | 45.23 [44.62-45.24] | 80 | 80 | 0 | 537696 | 1280 | 0.13 | 9.2 [9.1-9.2] | 9 | 62 | 0 | 11515 [11506-11515] | 7.07 [7.07-7.17] | 0 | 0.721 [0.710-0.721] | 2.713 [2.672-2.713] | 0.54 | 0.74 | 0.51 | 0.60 | 67% | 100% |
| `kalypso_block:16` | failed in all 3 trials | | | | | | | | | | | | | | | | | | | | | |
| `kalypso_block:32` | failed in all 3 trials | | | | | | | | | | | | | | | | | | | | | |
| `kalypso_block:2` | 31.30 [31.22-31.41] | 160 | 160 | 0 | 545456 | 1280 | 0.38 | 4.4 | 12 | 123 | 0 | 11515 [11514-11515] | 10.22 [10.19-10.25] | 0 | 0.355 [0.353-0.356] | 1.193 [1.186-1.194] | 0.48 | 0.79 | 0.33 | 0.46 | 71% | 100% |

- `kalypso_block:16` failed 3 trial(s): ArmError: /v1/semantic/query returned 400: {"error":{"message":"This model's maximum context length is 32768 tokens. However, you requested 112 output tokens and your prompt contains at least 32657 input tokens, for a total of at least 32769 tokens. Please reduce the length of the input prompt or th
- `kalypso_block:32` failed 3 trial(s): ArmError: /v1/semantic/query returned 400: {"error":{"message":"This model's maximum context length is 32768 tokens. However, you requested 208 output tokens and your prompt contains at least 32561 input tokens, for a total of at least 32769 tokens. Please reduce the length of the input prompt or th
