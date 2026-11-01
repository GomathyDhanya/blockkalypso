Model `Qwen/Qwen2.5-7B-Instruct` on unknown (Linux-4.19.0-gvisor-x86_64-with-glibc2.39); workload `cnli_sel25` (filter), 200 rows, 48 positive; 3 repeats after 1 warmup; prefix cache reset per trial: yes; git `5522cd270` (dirty).
Server: `python3 -m vllm.entrypoints.cli.main serve Qwen/Qwen2.5-7B-Instruct --host 127.0.0.1 --port 8000 --enable-prefix-caching --no-async-scheduling --api-server-count 1 --max-model-len 32768 --max-num-seqs 256 --max-num-batched-tokens 2048` (started Sat Oct 31 21:11:47 2026).

| Arm | Latency (s) | LLM requests | Block calls | Fallback calls | Prompt tokens | Gen tokens | Prefix hit rate | Peak KV % | Peak running | Peak waiting | Preemptions | Peak RSS (MiB) | Items/s | Truncated | Mean prefill (s) | Mean decode (s) | Accuracy | Precision | Recall | F1 | Agrees w/ kalypso_per_tuple | Self-agreement |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `kalypso_per_tuple` | 24.03 [23.86-26.95] | 200 | n/a | n/a | 318148 | 599 | 0.03 | 1.4 [1.4-1.5] | 5 | 193 [178-193] | 0 | 11522 [11520-11522] | 8.32 [7.42-8.38] | 0 | 0.262 [0.260-0.271] | 0.297 [0.295-0.306] | 0.67 | 0.40 | 0.79 | 0.54 | 100% | 100% |
| `kalypso_block:4` | 25.99 [25.70-26.62] | 50 | 50 | 0 | 294344 | 800 | 0.01 | 8.4 | 7 | 47 [47-48] | 0 | 11522 [11520-11522] | 7.70 [7.51-7.78] | 0 | 0.635 [0.627-0.646] | 2.333 [2.304-2.375] | 0.70 | 0.39 | 0.46 | 0.42 | 72% | 100% |
| `kalypso_block:16` | 35.51 [34.97-35.62] | 13 | 13 | 0 | 291085 | 884 | 0.00 | 37.3 | 8 | 12 | 0 | 11522 [11520-11522] | 5.63 [5.61-5.72] | 0 | 2.619 [2.599-2.649] | 11.264 [11.133-11.508] | 0.69 | 0.35 | 0.35 | 0.35 | 64% | 100% |
| `kalypso_block:32` | failed in all 3 trials | | | | | | | | | | | | | | | | | | | | | |
| `kalypso_block:2` | 24.11 [24.00-25.07] | 100 | 100 | 0 | 298894 | 800 | 0.02 | 4.0 [4.0-4.1] | 7 | 96 | 0 | 11522 [11522-11522] | 8.30 [7.98-8.34] | 0 | 0.368 [0.367-0.385] | 1.033 [1.028-1.081] | 0.74 | 0.46 | 0.38 | 0.41 | 68% | 100% |
| `kalypso_block:3` | 24.68 [24.15-26.19] | 67 | 67 | 0 | 295891 | 800 | 0.01 | 6.4 | 7 | 64 | 0 | 11522 [11522-11522] | 8.10 [7.64-8.28] | 0 | 0.489 [0.478-0.517] | 1.639 [1.602-1.747] | 0.72 | 0.43 | 0.50 | 0.46 | 67% | 100% |

- `kalypso_block:32` failed 3 trial(s): ArmError: /v1/semantic/query returned 400: {"error":{"message":"This model's maximum context length is 32768 tokens. However, you requested 208 output tokens and your prompt contains at least 32561 input tokens, for a total of at least 32769 tokens. Please reduce the length of the input prompt or th
