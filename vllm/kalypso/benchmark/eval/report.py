"""Summarize a results JSONL from run.py into a markdown table.

    python -m vllm.kalypso.benchmark.eval.report results/m1_baseline.jsonl

Each cell is the median over repeats, with min-max in brackets when the
repeats differ. Agreement is the share of rows whose output exactly matches
the reference arm's output in the same repeat.
"""

import argparse
import json
import statistics
from pathlib import Path

COLUMNS = (
    ("latency_s", "Latency (s)"),
    ("llm_requests", "LLM requests"),
    ("prompt_tokens", "Prompt tokens"),
    ("generation_tokens", "Gen tokens"),
    ("prefix_cache_hit_rate", "Prefix hit rate"),
    ("peak_kv_cache_pct", "Peak KV %"),
    ("peak_server_rss_kb", "Peak RSS (MiB)"),
    ("rows_per_s", "Rows/s"),
    ("truncated", "Truncated"),
)


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


def by_arm(trials: list[dict]) -> dict[str, list[dict]]:
    groups = {}
    for t in trials:
        groups.setdefault(t["arm"], []).append(t)
    return groups


def summarize(trials: list[dict]) -> dict:
    """Per arm and column: median, min and max over repeats (None-safe)."""
    summary = {}
    for arm, group in by_arm(trials).items():
        summary[arm] = {}
        for key, _ in COLUMNS:
            values = [t[key] for t in group if t.get(key) is not None]
            if key == "peak_server_rss_kb":
                values = [v / 1024 for v in values]
            summary[arm][key] = (
                {"median": statistics.median(values), "min": min(values), "max": max(values)}
                if values else None)
    return summary


def _rate(pairs) -> float | None:
    pairs = list(pairs)
    return sum(a == b for a, b in pairs) / len(pairs) if pairs else None


def agreement(trials: list[dict], reference: str) -> dict:
    """Agreement with ``reference`` per arm, plus each arm's own stability.

    ``vs_reference``: rows matching the reference arm's output in the same
    repeat. ``self``: rows matching the arm's own first-repeat output; at
    temperature 0 this should be 1.0, and less means outputs are not
    reproducible run to run.
    """
    groups = by_arm(trials)
    ref = {t["repeat"]: t["outputs"] for t in groups.get(reference, [])}
    result = {}
    for arm, group in groups.items():
        group = sorted(group, key=lambda t: t["repeat"])
        first = group[0]["outputs"]
        vs_ref = [(out, ref[t["repeat"]].get(idx))
                  for t in group if t["repeat"] in ref
                  for idx, out in t["outputs"].items()]
        own = [(out, first.get(idx)) for t in group[1:] for idx, out in t["outputs"].items()]
        result[arm] = {"vs_reference": _rate(vs_ref), "self": _rate(own)}
    return result


def _cell(stat, key) -> str:
    if stat is None:
        return "n/a"
    fmt = {"latency_s": "{:.2f}", "prefix_cache_hit_rate": "{:.2f}",
           "peak_kv_cache_pct": "{:.1f}", "peak_server_rss_kb": "{:.0f}",
           "rows_per_s": "{:.2f}"}.get(key, "{:.0f}")
    text = fmt.format(stat["median"])
    if stat["min"] != stat["max"]:
        text += f" [{fmt.format(stat['min'])}-{fmt.format(stat['max'])}]"
    return text


def _pct(value) -> str:
    return "n/a" if value is None else f"{value:.0%}"


def to_markdown(meta: dict, summary: dict, agree: dict, reference: str) -> str:
    server = meta.get("server") or {}
    lines = [
        f"Model `{meta['model']}` on {meta['cpu']} ({meta['platform']}); "
        f"workload `{meta['workload']}`, {meta['rows']} rows; "
        f"{meta['repeats']} repeats after {meta['warmup']} warmup; "
        f"prefix cache reset per trial: {'yes' if meta['prefix_cache_reset'] else 'no'}; "
        f"git `{(meta['git_sha'] or '?')[:9]}`{' (dirty)' if meta['git_dirty'] else ''}.",
    ]
    if server.get("command"):
        lines.append(f"Server: `{server['command']}` (started {server.get('started')}).")
    lines.append("")
    header = ["Arm"] + [label for _, label in COLUMNS] + [f"Agrees w/ {reference}", "Self-agreement"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))
    for arm in meta["arms"]:
        if arm not in summary:
            continue
        cells = [f"`{arm}`"] + [_cell(summary[arm][key], key) for key, _ in COLUMNS]
        cells += [_pct(agree[arm]["vs_reference"]), _pct(agree[arm]["self"])]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("results", type=Path)
    parser.add_argument("--reference", default="kalypso_per_tuple")
    args = parser.parse_args()
    meta, trials = load(args.results)
    if not trials:
        raise SystemExit(f"{args.results} has no trials")
    print(to_markdown(meta, summarize(trials), agreement(trials, args.reference),
                      args.reference), end="")


if __name__ == "__main__":
    main()
