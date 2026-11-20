"""Why the cost model's quality cap puts joins and maps at k = 1 (read-only).

Rebuilds the exact single-operator rows ``costmodel.fit`` uses (same filter on
held-out phases, cached item tokens; never fetches a tokenizer, never writes),
refits the caps to confirm they match the committed costmodel.json, then prints:

1. per (kind, regime, k): the pooled quality ``costmodel._cap`` tests and the
   per-workload values behind it, with each join workload's true selectivity;
2. for maps: per-tuple vs per-tuple ROUGE-1 (within and across runs), the
   share of non-Latin (language-drift) outputs per arm, and ROUGE-1 on the
   rows where neither side drifted.

    python -m research.baseline.analysis.quality_cap_audit

Decision record: research/docs/QUALITY_CAP_DECISION.md.
"""

from __future__ import annotations

import itertools
import json
import re
import statistics as st
from collections import defaultdict

from vllm.kalypso.benchmark.eval import analyze
import vllm.kalypso.benchmark.eval.costmodel as cm

SLACK = 0.02
NON_LATIN = re.compile(r"[　-鿿가-힯Ѐ-ӿ]")
MAP_WORKLOADS = ("movie_map", "pipe_filter_map_map")


def fit_rows() -> list[dict]:
    cache = json.loads((analyze.RESULTS / "analysis/item_tokens.json").read_text())

    def no_fetch(*_a, **_k):
        raise RuntimeError("item tokens missing from cache; refusing to fetch a tokenizer")

    cm._count = no_fetch
    everything = analyze.collect(analyze.RESULTS)
    collected = [r for r in everything if r["phase"] not in cm.HELD_OUT]
    return cm.attach_tokens(cm._single_operator_rows(collected), cache)


def regime(r) -> str:
    return "long" if r["item_tokens"] > cm.LONG_ROW_TOKENS else "short"


def quality(r):
    return r["f1_delta"] if r.get("f1_delta") is not None else r.get("similarity")


def cap_table(rows) -> None:
    sel = {r["workload"]: r["positives"] / r["items"] for r in rows
           if r["kind"] == "join" and r["k"] == 1 and r.get("positives") is not None}
    for kind in ("join", "map"):
        for rg in ("short", "long"):
            at = [r for r in rows if r["kind"] == kind and regime(r) == rg]
            if not at:
                continue
            print(f"\n{kind} / {rg}")
            if kind == "join":
                print("  true selectivity: " + ", ".join(
                    f"{w} {sel[w]:.2f}" for w in sorted({r['workload'] for r in at}) if w in sel))
            for k in sorted({r["k"] for r in at}):
                ak = [r for r in at if r["k"] == k]
                f1 = [r["f1_delta"] for r in ak if r.get("f1_delta") is not None]
                sim = [r["similarity"] for r in ak if r.get("similarity") is not None]
                if f1:
                    pooled = f"mean dF1 {st.mean(f1):+.3f} {'ok' if st.mean(f1) >= -SLACK else 'FAIL'}"
                else:
                    pooled = f"mean sim {st.mean(sim):.3f} {'ok' if st.mean(sim) >= analyze.MIN_SIMILARITY else 'FAIL'}"
                per = defaultdict(list)
                for r in ak:
                    if quality(r) is not None:
                        per[r["workload"]].append(quality(r))
                cells = "; ".join(f"{w} {st.mean(v):+.3f}" for w, v in sorted(per.items()))
                print(f"  k={k:>2} n={len(ak):>2} {pooled:22s} | {cells}")


def map_checks() -> None:
    runs = []
    for path in sorted(analyze.RESULTS.rglob("run.jsonl")):
        if "analysis" in path.relative_to(analyze.RESULTS).parts:
            continue
        lines = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
        meta = next((x for x in lines if x.get("type") == "meta"), {})
        if meta.get("workload") not in MAP_WORKLOADS:
            continue
        trials = [x for x in lines if x.get("type") != "meta" and x.get("outputs")]
        runs.append((meta["workload"], path.relative_to(analyze.RESULTS).parent, trials))

    def sim(a, b, skip_drift=False):
        ks = [k for k in a if k in b and not (skip_drift and (
            NON_LATIN.search(str(a[k])) or NON_LATIN.search(str(b[k]))))]
        return st.mean(analyze.rouge1(a[k], b[k]) for k in ks) if ks else None

    print("\nmaps: per-tuple vs per-tuple ROUGE-1")
    base_first = {}
    for w, where, trials in runs:
        pt = [t["outputs"] for t in trials if t["arm"] == analyze.BASELINE]
        if pt:
            within = [sim(a, b) for a, b in itertools.combinations(pt, 2)]
            print(f"  {where}: within run {st.mean(within):.3f}")
            base_first.setdefault(w, []).append(pt[0])
    for w, outs in base_first.items():
        for a, b in itertools.combinations(outs, 2):
            print(f"  {w}: across runs {sim(a, b):.3f}")

    print("\nmaps: language drift and ROUGE-1 vs per-tuple on non-drifted rows")
    for _w, where, trials in runs:
        by = defaultdict(list)
        for t in trials:
            by[t["arm"]].append(t)
        base = {t["repeat"]: t["outputs"] for t in by.get(analyze.BASELINE, [])}
        print(f"  {where}")
        for arm, ts in sorted(by.items()):
            outs = [str(v) for t in ts for v in t["outputs"].values()]
            drift = sum(bool(NON_LATIN.search(o)) for o in outs) / len(outs)
            scores = [s for t in ts if t["repeat"] in base
                      for s in [sim(base[t["repeat"]], t["outputs"], skip_drift=True)] if s is not None]
            print(f"    {arm:22s} non-Latin {drift:5.1%}  ROUGE-1 {st.mean(scores):.3f}")


def main() -> None:
    rows = fit_rows()
    model = cm.fit(rows, SLACK)
    committed = json.loads((analyze.RESULTS / "analysis/costmodel.json").read_text())["kinds"]
    refit = {k: p.caps for k, p in model.kinds.items()}
    assert refit == {k: p["caps"] for k, p in committed.items()}, (refit, committed)
    print(f"{len(rows)} fit rows; caps reproduce costmodel.json: {refit}")
    cap_table(rows)
    map_checks()


if __name__ == "__main__":
    main()
