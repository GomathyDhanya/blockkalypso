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

# Main table: the Kalypso paper's metrics (query completion time, speedup over
# the request-centric baseline, LLM calls with extra calls in brackets) plus
# the proposal's accuracy and peak memory. Everything else is a diagnostic.
SYSTEM_COLUMNS = (
    ("latency_s", "Latency (s)"),
    ("llm_requests", "LLM requests"),
    ("extra_calls", "Retries + fallbacks"),
    ("prompt_tokens", "Prompt tokens"),
    ("generation_tokens", "Gen tokens"),
    ("prefix_cache_hit_rate", "Prefix hit rate"),
    ("peak_kv_cache_pct", "Peak KV %"),
    ("peak_server_rss_kb", "Peak RSS (MiB)"),
    ("rows_per_s", "Items/s"),
    ("first_output_s", "First output (s)"),
    ("fallback_calls", "Fallback calls"),
    ("truncated", "Truncated"),
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
    "recall": "{:.2f}", "f1": "{:.2f}", "first_output_s": "{:.2f}",
    "selectivity": "{:.2f}", "occupancy": "{:.2f}", "start_s": "{:.2f}",
    "end_s": "{:.2f}", "overlap_s": "{:.2f}",
}
STAGE_COLUMNS = (
    ("tasks", "Tasks"),
    ("rows_in", "Rows in"),
    ("rows_out", "Rows out"),
    ("selectivity", "Out/in"),
    ("occupancy", "Occupancy"),
    ("start_s", "Start (s)"),
    ("end_s", "End (s)"),
    ("overlap_s", "Overlap (s)"),
    ("fallback_calls", "Fallbacks"),
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


def _key(item):
    return tuple(item) if isinstance(item, list) else item


def universe(meta: dict) -> list:
    """Every item an arm decides on: row idxs, or (left, right) pairs.

    Uses the row ids run.py records; older results files without them had
    ids 0..n-1.
    """
    left = meta.get("row_ids") or list(range(meta["rows"]))
    if meta.get("kind") in ("join", "pipeline"):
        right = meta.get("right_row_ids") or list(range(meta["right_rows"]))
        return [(l, r) for l in left for r in right]
    return list(left)


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
        rows = [{**t, **quality(meta, t), "extra_calls": _extra_calls(t)} for t in group]
        summary[arm] = {}
        for key in columns:
            values = [r[key] for r in rows if r.get(key) is not None]
            if key == "peak_server_rss_kb":
                values = [v / 1024 for v in values]
            summary[arm][key] = (
                {"median": statistics.median(values), "min": min(values), "max": max(values)}
                if values else None)
    return summary


def _extra_calls(trial: dict) -> int | None:
    """Calls beyond the plan: Kalypso retries plus block fallbacks."""
    if trial.get("retries") is None and trial.get("fallback_calls") is None:
        return None
    return (trial.get("retries") or 0) + (trial.get("fallback_calls") or 0)


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


def _median(summary, arm, key):
    stat = summary.get(arm, {}).get(key)
    return stat["median"] if stat else None


def _header(meta: dict) -> list[str]:
    server = meta.get("server") or {}
    items = len(universe(meta))
    size = (f"{meta['rows']} x {meta['right_rows']} = {items} pairs"
            if meta.get("kind") in ("join", "pipeline") else f"{meta['rows']} rows")
    labeled = meta.get("positives") is not None
    positives = f", {len(meta['positives'])} positive" if labeled else ""
    lines = [
        f"Model `{meta['model']}` on {meta.get('gpu') or meta['cpu']} ({meta['platform']}); "
        f"workload `{meta['workload']}` ({meta.get('kind', 'map')}), {size}"
        f"{positives}; "
        f"{meta['repeats']} repeats after {meta['warmup']} warmup; "
        f"prefix cache reset per trial: {'yes' if meta['prefix_cache_reset'] else 'no'}; "
        f"git `{(meta['git_sha'] or '?')[:9]}`{' (dirty)' if meta['git_dirty'] else ''}.",
    ]
    if server.get("command"):
        lines.append(f"Server: `{server['command']}` (started {server.get('started')}).")
    return lines


def to_markdown(meta: dict, summary: dict, agree: dict, reference: str,
                failed: dict | None = None) -> str:
    """Main table: query completion time, speedup over ``reference``, LLM
    calls [retries + fallbacks], accuracy (agreement with ``reference``, plus
    F1 against labels when the workload has them) and peak KV-cache use."""
    labeled = meta.get("positives") is not None
    header = (["Arm", "Completion time (s)", f"Speedup vs {reference}",
               "LLM calls [retries + fallbacks]", f"Agrees w/ {reference}"]
              + (["F1 vs labels"] if labeled else []) + ["Peak KV %"])
    lines = _header(meta) + ["", "| " + " | ".join(header) + " |",
                             "|" + "---|" * len(header)]
    ref_latency = _median(summary, reference, "latency_s")
    failed = failed or {}
    for arm in meta["arms"]:
        if arm not in summary:
            if arm in failed:
                lines.append(f"| `{arm}` | failed in all {len(failed[arm])} trials |"
                             + " |" * (len(header) - 2))
            continue
        latency = _median(summary, arm, "latency_s")
        speedup = (f"{ref_latency / latency:.2f}x" if ref_latency and latency else "n/a")
        calls = _median(summary, arm, "llm_requests")
        extra = _median(summary, arm, "extra_calls")
        calls_cell = ("n/a" if calls is None else
                      f"{calls:.0f}" + ("" if extra is None else f" [{extra:.0f}]"))
        cells = [f"`{arm}`", _cell(summary[arm]["latency_s"], "latency_s"), speedup,
                 calls_cell, _pct(agree[arm]["vs_reference"])]
        if labeled:
            cells.append(_cell(summary[arm]["f1"], "f1"))
        cells.append(_cell(summary[arm]["peak_kv_cache_pct"], "peak_kv_cache_pct"))
        lines.append("| " + " | ".join(cells) + " |")
    if failed:
        lines.append("")
        for arm, errors in failed.items():
            lines.append(f"- `{arm}` failed {len(errors)} trial(s): {errors[0][:300]}")
    return "\n".join(lines) + "\n"


DIAGNOSTIC_COLUMNS = (
    ("prompt_tokens", "Prompt tokens"),
    ("generation_tokens", "Gen tokens"),
    ("prefix_cache_hit_rate", "Prefix hit rate"),
    ("peak_server_rss_kb", "Peak RSS (MiB)"),
    ("rows_per_s", "Items/s"),
    ("first_output_s", "First output (s)"),
    ("truncated", "Truncated"),
)


def diagnostics_table(meta: dict, summary: dict, agree: dict) -> str:
    """Measurements outside the paper's metrics, kept for diagnosis."""
    labeled = meta.get("positives") is not None
    columns = DIAGNOSTIC_COLUMNS + (tuple(c for c in QUALITY_COLUMNS if c[0] != "f1")
                                    if labeled else ())
    header = ["Arm"] + [label for _, label in columns] + ["Self-agreement"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for arm in meta["arms"]:
        if arm not in summary:
            continue
        cells = [f"`{arm}`"] + [_cell(summary[arm][key], key) for key, _ in columns]
        cells.append(_pct(agree[arm]["self"]))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def full_report(meta: dict, trials: list[dict], reference: str = "kalypso_per_tuple") -> str:
    """Main table, then a Diagnostics section (other measurements, stages,
    and for pipelines the drift check)."""
    summary = summarize(meta, trials)
    agree = agreement(meta, trials, reference)
    parts = [to_markdown(meta, summary, agree, reference, failures(trials)),
             "", "Diagnostics:", "", diagnostics_table(meta, summary, agree),
             stage_table(meta, trials), operator_table(meta, trials),
             drift_table(meta, trials, reference)]
    return "\n".join(part for part in parts if part is not None)


def _stat(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    return {"median": statistics.median(values), "min": min(values), "max": max(values)}


def stage_table(meta: dict, trials: list[dict]) -> str:
    """Per arm and stage (by position): medians over repeats.

    Empty when no trial carries stage data (e.g. only direct-vLLM arms).
    """
    groups = by_arm(trials)
    header = ["Arm", "Stage", "Operators", "Block"] + [label for _, label in STAGE_COLUMNS]
    lines = []
    for arm in meta["arms"]:
        group = groups.get(arm, [])
        depth = max((len(t.get("stages") or []) for t in group), default=0)
        for pos in range(depth):
            rows = [t["stages"][pos] for t in group if len(t.get("stages") or []) > pos]
            first = rows[0]
            cells = [f"`{arm}`", str(pos), "+".join(first["operators"]) or "-",
                     str(first.get("block_size"))]
            cells += [_cell(_stat(r.get(key) for r in rows), key) for key, _ in STAGE_COLUMNS]
            lines.append("| " + " | ".join(cells) + " |")
    if not lines:
        return ""
    return "\n".join(["", "Stages (medians over repeats; times from query start):", "",
                      "| " + " | ".join(header) + " |", "|" + "---|" * len(header),
                      *lines]) + "\n"


DRIFT_TOLERANCE = 0.10


def _drifted(value, ref, tolerance):
    if value is None or ref is None:
        return False
    return abs(value - ref) > tolerance * max(ref, 1)


def drift(meta: dict, trials: list[dict], reference: str,
          tolerance: float = DRIFT_TOLERANCE) -> dict:
    """Per arm: medians of the work a pipeline passed downstream, and
    whether it differs from the reference arm by more than ``tolerance``.

    A blocked stage that answers differently changes how much work the later
    stages get, so its latency is not comparable with the reference.
    """
    groups = by_arm(trials)
    out = {}
    for arm in meta["arms"]:
        group = groups.get(arm, [])
        stats = {key: _stat(t.get(key) for t in group)
                 for key in ("filter_pass", "join_out", "final_out")}
        out[arm] = {key: (v["median"] if v else None) for key, v in stats.items()}
    ref = out.get(reference, {})
    for arm, row in out.items():
        row["drifted"] = arm != reference and any(
            _drifted(row[key], ref.get(key), tolerance)
            for key in ("filter_pass", "join_out", "final_out"))
    return out


def operator_table(meta: dict, trials: list[dict]) -> str:
    """Per arm and query operator (first repeat): rows in and out, and how
    the model's answers began; empty when no trial recorded operators."""
    stages = meta.get("stages") or []
    lines = []
    for arm in meta["arms"]:
        group = sorted(by_arm(trials).get(arm, []), key=lambda t: t.get("repeat", 0))
        if not group or not group[0].get("operators"):
            continue
        for pos, op in sorted(group[0]["operators"].items(), key=lambda kv: int(kv[0])):
            name = stages[int(pos)]["op"] if int(pos) < len(stages) else pos
            answers = ", ".join(f"{k} {v}" for k, v in sorted(op["answers"].items())) or "-"
            other = "; ".join(op.get("other_samples") or []) or "-"
            lines.append(f"| `{arm}` | {pos}: {name} | {'+'.join(op['operators'])} | "
                         f"{op['rows_in']} | {op['rows_out']} | {answers} | {other} |")
    if not lines:
        return ""
    header = "| Arm | Query op | Kalypso operator | Rows in | Rows out | Answers began with | Other answers (samples) |"
    return "\n".join(["", "Operators (first repeat):", "", header, "|---|---|---|---|---|---|---|",
                      *lines]) + "\n"


def drift_table(meta: dict, trials: list[dict], reference: str,
                tolerance: float = DRIFT_TOLERANCE) -> str:
    """Markdown for drift(); empty for workloads that are not pipelines."""
    if meta.get("kind") != "pipeline" or not by_arm(trials):
        return ""
    rows = drift(meta, trials, reference, tolerance)
    gold_pairs = len(meta["positives"]) if meta.get("positives") is not None else "n/a"
    lines = ["", "Work passed downstream (medians). Gold: "
             f"{meta.get('gold_filter_pass', 'n/a')} rows should pass the filter, "
             f"{gold_pairs} final outputs are expected.", "",
             "| Arm | Rows past first filter | Pairs past join | Final outputs | Comparable |",
             "|---|---|---|---|---|"]
    flagged = []
    for arm in meta["arms"]:
        if arm not in rows or not by_arm(trials).get(arm):
            continue
        r = rows[arm]
        fmt = lambda v: "n/a" if v is None else f"{v:g}"
        note = "reference" if arm == reference else ("**no, work drifted**" if r["drifted"] else "yes")
        if r["drifted"]:
            flagged.append(arm)
        lines.append(f"| `{arm}` | {fmt(r['filter_pass'])} | {fmt(r['join_out'])} | "
                     f"{fmt(r['final_out'])} | {note} |")
    if flagged:
        lines += ["", f"Drifted by more than {tolerance:.0%} from `{reference}`: "
                  + ", ".join(f"`{a}`" for a in flagged)
                  + ". Their latency and requests reflect different work, not just blocking."]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("results", type=Path)
    parser.add_argument("--reference", default="kalypso_per_tuple")
    args = parser.parse_args()
    meta, trials = load(args.results)
    if not trials:
        raise SystemExit(f"{args.results} has no trials")
    print(full_report(meta, trials, args.reference), end="")


if __name__ == "__main__":
    main()
