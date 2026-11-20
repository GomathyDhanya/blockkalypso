"""Opt-in structured telemetry for Kalypso queries (research baseline).

Off by default. When off, every hook is a single ``if`` on a module global and
nothing else runs, so query results and scheduling are unchanged.

Turn it on with an environment variable before the server starts::

    KALYPSO_TRACE=/path/to/trace.jsonl      # append one JSON object per line
    KALYPSO_RUN_ID=my-run                   # optional; default: a random id

or from code (tests, simulations)::

    from vllm.kalypso import telemetry
    telemetry.configure("/tmp/trace.jsonl", run_id="t1")   # or a list sink
    ...
    telemetry.configure(None)                              # off again

Every record has ``event``, ``run_id``, ``query_id`` (None outside a query),
``t_wall`` (Unix seconds, comparable with other samplers such as nvidia-smi)
and ``t_mono`` (monotonic seconds, for durations). Hooks add ``stage_id``,
``task_id`` and ``op`` from context variables set by the stage that runs the
task, so an LLM call can be attributed to its stage without passing ids
through every operator.

Events (see research/docs/BASELINE_METHODOLOGY.md for every field):
``query_start``, ``plan_built``, ``kv_stage_registered``, ``task_enqueue``,
``task_start``, ``task_end``, ``task_retry``, ``scheduler_sample``,
``llm_call``, ``block_eval``, ``kv_admit_rejected``, ``query_end``.
"""

from __future__ import annotations

import contextvars
import json
import os
import threading
import time
import uuid
from typing import Any

_lock = threading.Lock()
_sink = None          # file object, list, or None (off)
_run_id: str | None = None
_counters: dict[str, int] = {}

query_id: contextvars.ContextVar = contextvars.ContextVar("kalypso_query_id", default=None)
stage_id: contextvars.ContextVar = contextvars.ContextVar("kalypso_stage_id", default=None)
task_id: contextvars.ContextVar = contextvars.ContextVar("kalypso_task_id", default=None)
op_name: contextvars.ContextVar = contextvars.ContextVar("kalypso_op", default=None)

# Minimum seconds between two scheduler_sample events of one query.
SCHEDULER_SAMPLE_INTERVAL_S = float(os.environ.get("KALYPSO_TRACE_SAMPLE_S", "0.25"))


def configure(target=None, run_id: str | None = None) -> None:
    """Send events to ``target``: a path (appended), a list, or None (off)."""
    global _sink, _run_id
    with _lock:
        if _sink is not None and hasattr(_sink, "close"):
            _sink.close()
        if target is None:
            _sink = None
        elif isinstance(target, list):
            _sink = target
        else:
            path = os.fspath(target)
            parent = os.path.dirname(path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            _sink = open(path, "a", encoding="utf-8")
        _run_id = run_id or os.environ.get("KALYPSO_RUN_ID") or uuid.uuid4().hex[:12]
        _counters.clear()


def enabled() -> bool:
    return _sink is not None


def new_query_id() -> str:
    return uuid.uuid4().hex[:12]


def _jsonable(value: Any):
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return repr(value)


def emit(event: str, **fields) -> None:
    """Write one event. Never raises: telemetry must not break a query."""
    sink = _sink
    if sink is None:
        return
    record = {
        "event": event,
        "run_id": _run_id,
        "query_id": query_id.get(),
        "t_wall": time.time(),
        "t_mono": time.monotonic(),
    }
    for key, var in (("stage_id", stage_id), ("task_id", task_id), ("op", op_name)):
        if key not in fields and var.get() is not None:
            record[key] = var.get()
    record.update({k: _jsonable(v) for k, v in fields.items()})
    try:
        with _lock:
            if isinstance(sink, list):
                sink.append(record)
            else:
                sink.write(json.dumps(record) + "\n")
                sink.flush()
    except Exception:  # noqa: BLE001 -- best effort by design
        pass


def count(name: str, n: int = 1) -> None:
    """Increment a named counter (reported in ``query_end``)."""
    if _sink is None:
        return
    with _lock:
        _counters[name] = _counters.get(name, 0) + n


def take_counters() -> dict[str, int]:
    with _lock:
        out = dict(_counters)
        _counters.clear()
    return out


def op_label(op) -> str:
    return type(op).__name__


# Configure from the environment at import, so a server only needs the variable.
if os.environ.get("KALYPSO_TRACE"):
    configure(os.environ["KALYPSO_TRACE"])
