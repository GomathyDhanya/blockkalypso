"""The Phase 2 functional run's optimizer step (research/phase2/gpu/modal_phase2.py).

Turns on the two offline mechanisms for real workloads. It reads Kalypso's
real KV capacity from the run's trace (the first ``kv_stage_registered``
event), converts it to tokens, and calls ``choose_pipeline(quality=True,
candidate_set="pow2", kv_budget_tokens=...)`` with the default (automatic)
decision cache on each workload's Pipeline. It writes ``optimizer.json``:
the choice, the KV-rejected sizes and the cache counters. The choice is then
run as an extra arm. This is a functional check, not a comparison.

    python -m research.phase2.optimizer_step <run dir> '<parts json>' <model> <revision>
"""

from __future__ import annotations

import gzip
import json
import sys
from dataclasses import fields
from pathlib import Path

from vllm.kalypso.benchmark.eval import costmodel as cm


def load_model(path: Path = cm.RESULTS / "analysis/costmodel.json") -> cm.Model:
    d = json.loads(Path(path).read_text())
    names = {f.name for f in fields(cm.KindParams)}
    kinds = {k: cm.KindParams(**{n: v for n, v in p.items() if n in names})
             for k, p in d["kinds"].items()}
    return cm.Model(g=d["g"], alpha=d["alpha"], beta=d["beta"], gamma=d["gamma"],
                    delta=d["delta"], tail=d["tail"], kinds=kinds, f1_slack=d["f1_slack"],
                    lam=d["lambda"], retry=d["retry"])


def kv_capacity_bytes(root: Path) -> int | None:
    for trace in sorted(Path(root).glob("trace-*.jsonl*")):
        opener = gzip.open if trace.suffix == ".gz" else open
        with opener(trace, "rt") as f:
            for line in f:
                r = json.loads(line) if line.strip() else {}
                if r.get("event") == "kv_stage_registered" and r.get("total_capacity_bytes"):
                    return int(r["total_capacity_bytes"])
    return None


def item_tokens_from_run(root: Path) -> dict:
    """``costmodel.item_tokens`` cache entries from the run's own tokens.json
    (modal_baseline._tokens: the same tokenizer, revision and
    add_special_tokens=False as costmodel._count), so the container never
    has to tokenize again."""
    path = Path(root, "tokens.json")
    if not path.exists():
        return {}
    out = {}
    for key, t in json.loads(path.read_text()).items():
        left, right = t["left"], t["right"] or [0]
        lm, rm = sum(left) / len(left), sum(right) / len(right)
        out[key] = {"mean": round(lm + rm, 2), "max": max(left) + max(right),
                    "left_mean": round(lm, 2), "left_max": max(left),
                    "right_mean": round(rm, 2), "right_max": max(right),
                    "left_rows": len(left), "right_rows": len(t["right"]) if t["right"] else 0}
    return out


def run(root: Path, parts: list[dict], bytes_per_token: float,
        model: cm.Model | None = None, item_tokens: dict | None = None) -> dict:
    from research.baseline import workloads  # noqa: F401 -- registers research workloads

    model = model or load_model()
    if item_tokens is None:
        item_tokens = json.loads((cm.RESULTS / "analysis/item_tokens.json").read_text())
        item_tokens.update(item_tokens_from_run(root))
    capacity = kv_capacity_bytes(root)
    budget = capacity / bytes_per_token if capacity else None
    cm.DECISION_CACHE.clear()
    out = {"kv_capacity_bytes": capacity, "bytes_per_token": bytes_per_token,
           "kv_budget_tokens": budget, "parts": {}}
    for part in parts:
        target = cm._validation_target(model, {"parts": [part]}, item_tokens)
        if target is None or target[0] != "pipe":
            out["parts"][part["name"]] = {"error": "no pipeline target"}
            continue
        pipe = target[1]
        res = model.choose_pipeline(pipe, quality=True, candidate_set="pow2",
                                    kv_budget_tokens=budget)
        out["parts"][part["name"]] = {
            "ops": list(pipe.ops), "blocks": list(res["blocks"]),
            "predicted_latency_s": res["latency_s"], "kv_rejected": res.get("kv_rejected"),
            "arm": "kalypso_pipe:" + ",".join(str(k) for k in res["blocks"])}
    out["decision_cache"] = cm.DECISION_CACHE.stats.as_dict() | {"entries": len(cm.DECISION_CACHE)}
    Path(root, "optimizer.json").write_text(json.dumps(out, indent=2))
    return out


def main(argv=None) -> None:
    root, parts, model_name, revision = (argv if argv is not None else sys.argv[1:])
    from vllm.kalypso.budget import compute_bytes_per_token
    bpt = compute_bytes_per_token(model_name, revision=revision)
    print(json.dumps(run(Path(root), json.loads(parts), bpt)))


if __name__ == "__main__":
    main()
