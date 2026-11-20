# Decision: is the k = 1 quality cap for joins and maps real? (Phase 2, step 04)

**Verdict: (b), the cap is miscalibrated, for two different identified
reasons.** The k = 1 result is not a measured "every block size loses
quality" fact for either operator:

- **Joins:** `_cap` averages workloads whose block-join quality moves in
  opposite directions. Six of the eleven join workloads kept F1 within the
  slack at some k > 1.
- **Maps:** the similarity bar compares block output with per-tuple output
  word for word. It measures "same wording", not "same answer", and the data
  has no prompt-matched reference that would tell the two apart.

Nothing is changed in this step. The adaptive controller (step 03) stays
filter-only, but that is a consequence of this calibration, not a
data-grounded property of joins or maps.

Reproduce every number below with
`python -m research.baseline.analysis.quality_cap_audit` (read-only). It
rebuilds the 173 single-operator rows `costmodel.fit` uses and asserts that
the refit caps equal `results/analysis/costmodel.json`.

## How the cap is computed

`costmodel._cap(rows, measured, f1_slack)`, per (kind, regime), with regime =
long if item tokens > 512:

1. Walk the measured k values upward from 2.
2. At each k, average the quality over **all rows at that k, pooled across
   workloads**. For joins and filters that is ΔF1 vs per-tuple, which must be
   ≥ −0.02. For maps it is ROUGE-1 vs per-tuple's output, which must be ≥ 0.8.
3. Stop at the first k that fails. The cap is the last k that passed.

Two possible artifacts were checked and ruled out:

- **Stopping at the first failure:** with that rule removed, there is still
  no pooled k > 1 that passes, for joins (short or long) or maps. It changes
  nothing.
- **The 512-token regime boundary:** join items are 62–~100 tokens (movie)
  or about 1,615 (ContractNLI), and map items about 30. Nothing sits near 512.

## Joins: numbers

ΔF1 vs per-tuple at the block sizes the controller would use. "Sel" is true
join selectivity (positives / pairs). There is one fit row per (workload, k),
each the median over repeats.

| workload | sel | k=2 | k=4 | k=6 | k=8 |
|---|---|---|---|---|---|
| movie_fanout10 | 0.09 | −0.249 | −0.210 | −0.261 | – |
| movie_fanout25 | 0.25 | −0.249 | −0.091 | −0.187 | – |
| movie_fanout50 | 0.50 | −0.048 | **+0.048** | −0.049 | – |
| movie_fanout75 | 0.75 | **+0.060** | **+0.104** | **+0.077** | – |
| movie_fanout90 | 0.91 | **+0.090** | **+0.125** | **+0.071** | – |
| movie_opposite | 0.50 | **−0.010** | **+0.016** | **+0.000** | – |
| pipe_filter_join_s10_join | 0.47 | −0.182 | −0.140 | −0.174 | – |
| pipe_filter_join_s50_join | 0.47 | −0.206 | −0.157 | −0.178 | – |
| pipe_filter_join_s90_join | 0.47 | −0.214 | −0.140 | −0.160 | – |
| cnli_join_low (long) | 0.21 | **+0.007** | −0.057 | – | −0.138 |
| cnli_join_high (long) | 0.65 | −0.139 | **−0.002** | – | **−0.003** |
| **pooled mean, short** | | −0.099 | −0.042 | −0.072 | |
| **pooled mean, long** | | −0.066 | −0.029 | | −0.071 |

Bold = within the 0.02 slack.

**Did any measured join block size keep quality? Yes.** At k = 4:
movie_fanout50 (+0.048), movie_fanout75 (+0.104), movie_fanout90 (+0.125),
movie_opposite (+0.016) and cnli_join_high (−0.002, and −0.003 at k = 8).
cnli_join_low passed at k = 2 (+0.007).

**Why the cap misses them.** Within the controlled fanout family, block-join
ΔF1 at k = 4 rises steadily with join selectivity: −0.210, −0.091, +0.048,
+0.104, +0.125 at 9%, 25%, 50%, 75%, 91%. The long-row pair agrees: 65%
passes at k = 4, 21% does not. `_cap` keys only on (kind, regime) and
averages across selectivities, so a spread from −0.21 to +0.13 collapses into
one failing mean. The three `pipe_filter_join_*_join` rows weigh three times
in that mean, because they are three nearly identical joins: the same 32
right rows, the same 0.47 selectivity, per-tuple F1 0.84–0.85.

**What the data does not settle.** Selectivity is not the whole story. At
nearly the same selectivity, movie_fanout50 (0.50, +0.048) and
movie_opposite (0.50, +0.016) pass, while the pipe joins (0.47, −0.14 to
−0.16) fail clearly. That difference is unexplained. One candidate is that
the pipe joins start from a much higher per-tuple F1 (0.84 vs 0.60–0.79). With one
row per (workload, k), the repeat-to-repeat noise in ΔF1 is also unknown;
cnli_join_high's −0.139 at k = 2 next to −0.002 at k = 4 suggests it could
be around 0.1.

## Maps: numbers

| | k=2 | k=4 | k=6 | k=8 | k=16 |
|---|---|---|---|---|---|
| ROUGE-1 vs per-tuple (pooled) | 0.384 | 0.354 | 0.340 | 0.342 | 0.334 |

**Did any measured map block size keep quality? No:** 0.33–0.40 against a
0.8 bar, on both map workloads, at every k. But:

- **The reference is verbatim.** Per-tuple vs per-tuple scores 1.000 within
  every run and 0.979–1.000 across runs: identical prompts at temperature 0
  give identical text. The 0.8 bar is therefore a bar on reproducing
  per-tuple's wording.
- **The score does not move with k.** It is 0.38–0.40 at k = 2 and 0.33–0.34
  at k = 16. Damage that builds up with block size would fall with k, as join
  F1 does (−0.04 at k = 4, −0.18 at k = 11, −0.52 at k = 32). A flat ~0.35
  from k = 2 looks like a one-time change caused by the block prompt itself.
- **It is not language drift.** Block maps return non-Latin text (for
  example Chinese) on 0–7% of rows, against 2–4% for per-tuple. Excluding
  those rows leaves ROUGE-1 at 0.33–0.40.
- **The paraphrase case is real.** In movie_map n=100 there are rows where
  per-tuple and k = 2 give the same answer in different words (both say the
  context is not a movie review and the reason cannot be determined), yet
  score about 0.3.
- **No control exists to separate the two.** Map runs have no "block prompt,
  one row" arm. (The filter experiments have such controls: the
  `vllm_direct:block_prompt` arms in `cloud/modal_block_size.py`.)

So the map data neither shows that block maps lose answer quality nor that
they keep it. Separately from quality, STUDY.md measured block maps as
slower (0.5–0.9x at k = 4, 0.37x at k = 16; only k = 2 on 500 rows was
1.14x). There is therefore a latency reason not to block maps whatever the
cap says.

## Decision and fixes (not implemented)

1. **Join cap: condition it on join selectivity instead of pooling.** Fit
   `caps` per (regime, selectivity band) from per-workload means, not one
   pooled mean per k. This is justified by the steady sign change within the
   fanout family. Before adopting a band edge, two things are needed:
   - an explanation of the pipe-join vs fanout50 conflict at ~0.5;
   - repeat-level ΔF1 variance, so a band edge is not fitted to noise.

   Also: the controller in step 03 observes the pass rate of the stage
   *feeding* the join, but a selectivity-banded join cap would need the
   join's *own* pass rate. That needs a different signal, which the join
   stage's own `task_end.rows_passed` already provides.
2. **Map cap: change the reference, not the threshold.** Calibrate map
   similarity against a prompt-matched reference: per-tuple run with the
   block prompt (k = 1, block wording), or a semantic or LLM-judge
   comparison. Then the bar measures answer change rather than wording
   change. Until that control exists, keep maps at k = 1. The latency data
   already argues against block maps.
3. **Controller (step 03): no change.** It stays filter-only under the
   current fit. Treat that as "not yet measured correctly for joins", not as
   "joins must not be blocked". Do not widen the controller's join behaviour
   until fix 1 has data behind it.

No evaluation was run for this step; all numbers come from existing results
under `vllm/kalypso/benchmark/eval/results/`.
