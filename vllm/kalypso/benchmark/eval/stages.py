"""Turn the server's per-query ``trace`` into per-stage numbers.

A stage is what Kalypso's planner schedules as one unit: a block operator is
its own stage, while per-tuple operators can be fused into one stage, so the
``operators`` list says what each stage ran.
"""


def _ratio(num, den):
    return round(num / den, 4) if den else None


def summarize_trace(trace: dict | None) -> dict:
    """Flat per-trial fields plus one entry per stage, in pipeline order.

    Per stage: rows in and out, selectivity (rows out / rows in, i.e. fanout
    for a join), tasks, mean block occupancy, first start and last finish, and
    ``overlap_s``: how long the stage was already running before the previous
    stage finished (0 when it started only afterwards).
    """
    if not trace:
        return {"first_output_s": None, "fallback_calls": None, "retries": None,
                "stages": []}
    stages = []
    fallbacks = retries = 0
    for stage_id in sorted(trace.get("stages", {}), key=int):
        s = trace["stages"][stage_id]
        stats = s.get("operator_stats") or {}
        fallbacks += sum(v.get("fallback_calls", 0) for v in stats.values())
        retries += s.get("retries") or 0
        stages.append({
            "stage_id": int(stage_id),
            "operators": s.get("operators", []),
            "block_size": s.get("block_size"),
            "tasks": s.get("tasks"),
            "rows_in": s.get("rows_in"),
            "rows_out": s.get("rows_out"),
            "selectivity": _ratio(s.get("rows_out") or 0, s.get("rows_in")),
            "occupancy": s.get("mean_occupancy"),
            "start_s": s.get("first_start"),
            "end_s": s.get("last_end"),
            "fallback_calls": sum(v.get("fallback_calls", 0) for v in stats.values()),
        })
    for prev, cur in zip(stages, stages[1:]):
        if prev["end_s"] is None or cur["start_s"] is None:
            cur["overlap_s"] = None
        else:
            cur["overlap_s"] = round(max(0.0, prev["end_s"] - cur["start_s"]), 4)
    if stages:
        stages[0]["overlap_s"] = None
    return {
        "first_output_s": trace.get("first_output_s"),
        "fallback_calls": fallbacks,
        # Kalypso's own task retries (e.g. a map re-run with more tokens).
        "retries": retries,
        "stages": stages,
    }


FILTER_OPERATORS = ("SemFilter", "BlockSemFilter")


def filter_pass(trace: dict | None, right_rows: int | None) -> int | None:
    """Rows that passed a pipeline's first filter.

    Read from the first stage that runs a filter. In Kalypso's per-tuple
    layout that stage also fans each passing row out into one pair per right
    row (``fanout == "CartesianProduct"``), so its output is divided back.
    """
    if not trace:
        return None
    for stage_id in sorted(trace.get("stages", {}), key=int):
        stage = trace["stages"][stage_id]
        if not set(stage.get("operators", [])) & set(FILTER_OPERATORS):
            continue
        out = stage.get("rows_out") or 0
        if stage.get("fanout") == "CartesianProduct":
            return out // right_rows if right_rows else None
        return out
    return None


def rows_by_position(trace: dict | None) -> dict:
    """Per query operator (its position in the ops list): rows in and out,
    summed over stages, and how its answers began.

    Positions are the same whether Kalypso fuses operators into one stage
    (per-tuple) or runs each as its own stage (blocked), so arms compare.
    """
    out = {}
    for stage in (trace or {}).get("stages", {}).values():
        for op in stage.get("operator_rows", []) or []:
            pos = op.get("position")
            if pos is None:
                continue
            row = out.setdefault(pos, {"operators": [], "rows_in": 0, "rows_out": 0,
                                       "answers": {}, "other_samples": []})
            row["operators"].append(op["operator"])
            row["rows_in"] += op.get("rows_in", 0)
            row["rows_out"] += op.get("rows_out", 0)
            for head, n in (op.get("answers") or {}).items():
                row["answers"][head] = row["answers"].get(head, 0) + n
            row["other_samples"] += op.get("other_samples", [])[:3 - len(row["other_samples"])]
    return out
