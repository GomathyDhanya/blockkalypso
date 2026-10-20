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
        return {"first_output_s": None, "fallback_calls": None, "stages": []}
    stages = []
    fallbacks = 0
    for stage_id in sorted(trace.get("stages", {}), key=int):
        s = trace["stages"][stage_id]
        stats = s.get("operator_stats") or {}
        fallbacks += sum(v.get("fallback_calls", 0) for v in stats.values())
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
        "stages": stages,
    }
