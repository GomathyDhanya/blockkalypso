# Research hypotheses (Session 1, Step 6)

Each hypothesis is stated so that the Session 1 measurements can reject it.
"Status" is what the evidence on this branch says now; GPU-dependent parts
stay open until the Modal runs (E0-E3) are analysed in BASELINE_RESULTS.md.
Nothing here proposes a Phase 2 design.

## H1. Exhaustive block-size search does not scale with pipeline depth

**Claim.** `choose_pipeline` (costmodel.py:286-296) enumerates
Π k_max(stage) plans and evaluates each with an O(S) prediction, so planning
time and memory grow exponentially with depth and become impractical beyond
3-4 semantic stages.

**Test.** Experiment A (CPU): planning time, peak Python heap and chosen plan
for 1-6 stages, three operator families, exhaustive vs. two restricted
candidate sets, 5 repeats each.

**Reject if** exhaustive planning at 5-6 stages stays under 1 s and 100 MB on
the reference CPU, or if planning time grows sub-exponentially in depth.

**Status: supported.** 55-90 µs and 1.5-1.9 KB per candidate. At 4 stages
exhaustive search takes 8.6 s / 297 MB (filters) and 19.9 s / 795 MB
(filter-join). 5-6 stages (3.2M-262M plans) were not run; extrapolation gives
3 min to 5 h and 7 GB to 700 GB. Restricted sets (powers of two; 1-4) plan
6 stages in under 2 s. Caveat: today the search runs offline once per
experiment, not per query, so this is a scalability limit for deeper
pipelines, not a measured latency cost in current use.

**Secondary finding (H1b).** The obvious restriction is not safe: the {1..4}
set picks plans with 39-506% higher *predicted* latency on filter-join and
mixed pipelines (it cannot reach k=32 for joins), while the powers-of-two set
matches the exhaustive pick within ±2%. The loss comes from what the set
excludes, not from search depth.

## H2. Static block sizes and static selectivity estimates mis-handle cardinality change

**Claim.** k is fixed per stage before execution and the expected rows per
stage come from labelled recall/FPR, so when the real selectivity differs,
the chosen k is wrong for the rows that actually arrive.

**What the code already adapts.** Block membership: downstream blocks are
packed from whatever rows arrive (pipeline_execution.py:161-209), and the
CPU simulation shows 0 partial join blocks in every
configuration. So "static blocks" in the strong sense (blocks formed before
filtering) is **rejected by the code**; the open question is the static *k*.

**Test.** E1 (GPU): five selectivities (5/20/50/80/95%) × block configs for a
filter and a filter → join pipeline; compare predicted vs. achieved
selectivity and whether the best measured k changes with selectivity.
Cost-model validation (Step 5) gives the historical q-error.

**Reject if** the measured-best k is the same at all five selectivities and
the model's pick is within 5% of it everywhere.

**Status after E1 (GPU): weakly supported.** pipe:1,4 is best at 20-95% (1.25-1.52x) but nothing beats per-tuple at 5% (0.97x), so the best configuration changes with selectivity. Stronger finding: k itself changes cardinality (block filters pass 25-40% more rows than per-tuple; pipe:4,1 is 0.63-0.84x because of the extra join work). See BASELINE_RESULTS.md §2. Earlier evidence: Evidence so far: held-out join pass q-error
median 1.83, max 15.2; held-out pipeline top-1 0%, regret up to 24%; and
supplying the true cardinality does *not* reduce held-out pipeline error
(20.3% → 20.2%), which points at the composition term rather than the
selectivity estimate. The simulation shows k's effect on call count is
proportional at every selectivity (e.g. 1x1 → 1x4 cuts calls ~3.4x at 50%
and ~3.6x at 95%), so any selectivity-dependence on the GPU must come from
latency per call, not call counts.

## H3. Resource state (KV, queueing) matters but is ignored by the optimizer

**Claim.** The optimizer has no KV, queue or preemption term, and under
concurrent load the best block size shifts.

**Test.** E3 (GPU): the same filter → join plans with 0, 32 and 128
concurrent background streams; record KV use, waiting requests and
preemptions (vLLM metrics, 0.5 s) and Kalypso's admission counters
(telemetry). E0-E2 also record KV and queue state.

**Reject if** the best measured block configuration and the relative
ordering of plans are the same at all three load levels, and KV use stays
under 50% with zero preemptions.

**Status after GPU runs: rejected at this scale.** The E3 rerun (confirmed cache resets) keeps the same best plan (pipe:1,4) and plan ordering at 0, 32 and 128 streams, with peak KV 17.7% and 0 preemptions; load shrinks the best plan's advantage from 1.35x to 1.13x. The first E3 run was invalid (failed cache resets). Peak KV ≤ 17.5% (22% under load), 0 preemptions in all runs, KV admission rejections only in E1-s95; the queue is the visible pressure. E3 under load is invalid (prefix-cache resets failed under load; BASELINE_RESULTS.md §5). Earlier evidence: Historical evidence leans towards rejection at
the scale measured so far: peak KV median 6.9%, max 75%, 0 preemptions in
484 arms; queueing (median 151, max 2011 waiting requests) was the visible
pressure.

## H4. Run-to-run variation in downstream block composition is a measurement confound

**Claim.** Which rows share a downstream block depends on upstream task
completion order (`asyncio.wait` returns a set), so the same query yields
different blocks across runs, which changes block-prompt quality and latency.

**Status: confirmed on CPU** for the unit-test pipelines (9-10 of 10 reruns
differ in grouping with tracing off; audit §4.6). In the simulation the
*sizes* of join blocks are identical across repeats (all full), so the
variation is in membership only. Its effect on GPU quality/latency is
measured indirectly by the repeat spread in E1/E2.

**Consequence for methodology.** Report the spread across repeats for every
block arm, and do not attribute differences smaller than that spread to the
block size.

## H5. Telemetry does not perturb what it measures

**Claim.** Opt-in tracing changes neither results nor timing materially.

**Status:** results-identical: **confirmed** (unit tests, 4 pipelines,
identical outputs and call multisets). Timing: on CPU with zero-cost LLM
calls, tracing adds ~21-22 µs per event (37-63% of an otherwise near-empty
run, 1.7-5.1k events per query); with real calls of 0.1-10 s this is
expected to be ~1% or less. E0 on the GPU measures it directly; **reject
H5 if E0 shows > 3% median latency increase with tracing on.** E0 result: +5.0% on per-tuple (400 calls), 0% on block:8; H5 **rejected for per-tuple** at the 3% threshold (part may be server-instance drift). Compare traced arms only with traced arms.
