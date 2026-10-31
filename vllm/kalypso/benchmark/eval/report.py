"""Summarize a results JSONL from run.py into a markdown table.

    python -m vllm.kalypso.benchmark.eval.report results/reviews.jsonl

Each cell is the median over repeats, with min-max in brackets when repeats
differ. For labeled filter and join workloads it adds accuracy, precision,
recall and F1 against the labels. Agreement is the share of items (rows, or
pairs for a join) on which an arm decides the same as the reference arm in the
same repeat; for maps it compares output text exactly.
"""

import argparse
import json
import statistics
from pathlib import Path

SYSTEM_COLUMNS = (
    ("latency_s", "Latency (s)"),
    ("llm_requests", "LLM requests"),
    ("block_calls", "Block calls"),
    ("fallback_calls", "Fallback calls"),
    ("prompt_tokens", "Prompt tokens"),
    ("generation_tokens", "Gen tokens"),
    ("prefix_cache_hit_rate", "Prefix hit rate"),
    ("peak_kv_cache_pct", "Peak KV %"),
    ("peak_server_rss_kb", "Peak RSS (MiB)"),
    ("rows_per_s", "Items/s"),
    ("truncated", "Truncated"),
    ("mean_prefill_time_s", "Mean prefill (s)"),
    ("mean_decode_time_s", "Mean decode (s)"),
)
QUALITY_COLUMNS = (
    ("accuracy", "Accuracy"),
    ("precision", "Precision"),
    ("recall", "Recall"),
    ("f1", "F1"),
)
_FORMATS = {
    "latency_s": "{:.2f}", "prefix_cache_hit_rate": "{:.2f}",
    "peak_kv_cache_pct": "{:.1f}", "peak_server_rss_kb": "{:.0f}",
    "rows_per_s": "{:.2f}", "accuracy": "{:.2f}", "precision": "{:.2f}",
    "recall": "{:.2f}", "f1": "{:.2f}",
    "mean_prefill_time_s": "{:.3f}", "mean_decode_time_s": "{:.3f}",
}


def load(path: Path) -> tuple[dict, list[dict]]:
    meta, trials = None, []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("type") == "meta":
            meta = record
        elif record.get("type") == "trial":
            trials.append(record)

    if meta is None:
        raise ValueError(f"{path} has no metadata line")
    return meta, trials


def _key(item):
    return tuple(item) if isinstance(item, list) else item


def universe(meta: dict) -> list:
    """Every item an arm decides on: row idxs, or (left, right) pairs."""
    if meta.get("kind") == "join":
        return [(l, r) for l in range(meta["rows"]) for r in range(meta["right_rows"])]
    return list(range(meta["rows"]))


def quality(meta: dict, trial: dict) -> dict:
    """Accuracy, precision, recall and F1 of one trial against the labels."""
    if meta.get("positives") is None or trial.get("selected") is None:
        return {}
    truth = {_key(x) for x in meta["positives"]}
    chosen = {_key(x) for x in trial["selected"]}
    items = universe(meta)
    tp = len(chosen & truth)
    fp = len(chosen - truth)
    fn = len(truth - chosen)
    tn = len(items) - tp - fp - fn
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    if precision is None or recall is None:
        f1 = None
    elif precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return {"accuracy": (tp + tn) / len(items), "precision": precision,
            "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def by_arm(trials: list[dict]) -> dict[str, list[dict]]:
    """Successful trials grouped per arm."""
    groups = {}
    for t in trials:
        if "error" not in t:
            groups.setdefault(t["arm"], []).append(t)
    return groups


def failures(trials: list[dict]) -> dict[str, list[str]]:
    out = {}
    for t in trials:
        if "error" in t:
            out.setdefault(t["arm"], []).append(t["error"])
    return out


def summarize(meta: dict, trials: list[dict]) -> dict:
    """Per arm and column: median, min and max over repeats (None-safe)."""
    columns = [key for key, _ in SYSTEM_COLUMNS + QUALITY_COLUMNS]
    summary = {}
    for arm, group in by_arm(trials).items():
        rows = [{**t, **quality(meta, t)} for t in group]
        summary[arm] = {}
        for key in columns:
            values = [r[key] for r in rows if r.get(key) is not None]
            if key == "peak_server_rss_kb":
                values = [v / 1024 for v in values]
            summary[arm][key] = (
                {"median": statistics.median(values), "min": min(values), "max": max(values)}
                if values else None)
    return summary


def _decisions(meta: dict, trial: dict) -> dict:
    if meta.get("kind", "map") == "map":
        return trial["outputs"]
    chosen = {_key(x) for x in trial["selected"]}
    return {item: item in chosen for item in universe(meta)}


def _rate(pairs) -> float | None:
    pairs = list(pairs)
    return sum(a == b for a, b in pairs) / len(pairs) if pairs else None


def agreement(meta: dict, trials: list[dict], reference: str) -> dict:
    """Per arm: agreement with ``reference`` and with the arm's own first repeat.

    ``self`` below 1.0 means an arm's results are not reproducible run to run
    even at temperature 0.
    """
    groups = by_arm(trials)
    ref = {t["repeat"]: _decisions(meta, t) for t in groups.get(reference, [])}
    result = {}
    for arm, group in groups.items():
        group = sorted(group, key=lambda t: t["repeat"])
        first = _decisions(meta, group[0])
        vs_ref, own = [], []
        for t in group:
            decided = _decisions(meta, t)
            if t["repeat"] in ref:
                vs_ref += [(v, ref[t["repeat"]].get(k)) for k, v in decided.items()]
            if t is not group[0]:
                own += [(v, first.get(k)) for k, v in decided.items()]
        result[arm] = {"vs_reference": _rate(vs_ref), "self": _rate(own)}
    return result


def _cell(stat, key) -> str:
    if stat is None:
        return "n/a"
    fmt = _FORMATS.get(key, "{:.0f}")
    text = fmt.format(stat["median"])
    if stat["min"] != stat["max"]:
        text += f" [{fmt.format(stat['min'])}-{fmt.format(stat['max'])}]"
    return text


def _pct(value) -> str:
    return "n/a" if value is None else f"{value:.0%}"


def to_markdown(meta: dict, summary: dict, agree: dict, reference: str,
                failed: dict | None = None) -> str:
    server = meta.get("server") or {}
    items = len(universe(meta))
    size = (f"{meta['rows']} x {meta['right_rows']} = {items} pairs"
            if meta.get("kind") == "join" else f"{meta['rows']} rows")
    labeled = meta.get("positives") is not None
    positives = f", {len(meta['positives'])} positive" if labeled else ""
    lines = [
        f"Model `{meta['model']}` on {meta['cpu']} ({meta['platform']}); "
        f"workload `{meta['workload']}` ({meta.get('kind', 'map')}), {size}"
        f"{positives}; "
        f"{meta['repeats']} repeats after {meta['warmup']} warmup; "
        f"prefix cache reset per trial: {'yes' if meta['prefix_cache_reset'] else 'no'}; "
        f"git `{(meta['git_sha'] or '?')[:9]}`{' (dirty)' if meta['git_dirty'] else ''}.",
    ]
    if server.get("command"):
        lines.append(f"Server: `{server['command']}` (started {server.get('started')}).")
    lines.append("")
    columns = SYSTEM_COLUMNS + (QUALITY_COLUMNS if labeled else ())
    header = (["Arm"] + [label for _, label in columns]
              + [f"Agrees w/ {reference}", "Self-agreement"])
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))
    failed = failed or {}
    for arm in meta["arms"]:
        if arm not in summary:
            if arm in failed:
                lines.append(f"| `{arm}` | failed in all {len(failed[arm])} trials |"
                             + " |" * (len(header) - 2))
            continue
        cells = [f"`{arm}`"] + [_cell(summary[arm][key], key) for key, _ in columns]
        cells += [_pct(agree[arm]["vs_reference"]), _pct(agree[arm]["self"])]
        lines.append("| " + " | ".join(cells) + " |")
    if failed:
        lines.append("")
        for arm, errors in failed.items():
            lines.append(f"- `{arm}` failed {len(errors)} trial(s): {errors[0][:300]}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("results", type=Path)
    parser.add_argument("--reference", default="kalypso_per_tuple")
    args = parser.parse_args()
    meta, trials = load(args.results)
    if not trials:
        raise SystemExit(f"{args.results} has no trials")
    print(to_markdown(meta, summarize(meta, trials), agreement(meta, trials, args.reference),
                      args.reference, failures(trials)), end="")


if __name__ == "__main__":
    main()
