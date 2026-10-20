"""Run arms over a workload and append one JSON line per trial.

    python -m vllm.kalypso.benchmark.eval.run --workload reviews \\
        --arms kalypso_per_tuple,vllm_direct,kalypso_block:5,kalypso_block:10 \\
        --warmup 1 --repeats 3 --out vllm/kalypso/benchmark/eval/results/reviews.jsonl

``--arms`` defaults to every arm that fits the workload (see arms.DEFAULT_ARMS).

The first line of the output is run metadata; every later line is a trial.
Trials run arm by arm within each repeat (A, B, A, B, ...) so slow drift on
the machine affects every arm alike. The prefix cache is reset before each
trial, which needs the server started with VLLM_SERVER_DEV_MODE=1.
"""

import argparse
import asyncio
import datetime as dt
import json
import platform
import subprocess
import sys
from pathlib import Path

import httpx

from . import metrics
from .stages import filter_pass, summarize_trace
from .arms import DEFAULT_ARMS, ArmError, check_arm, parse_arm, run_arm
from .workloads import WORKLOADS

REPO_ROOT = Path(__file__).resolve().parents[4]
SERVER_MODULE = "vllm.entrypoints.openai.api_server"


def _run(cmd: list[str]) -> str | None:
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def find_server_pid() -> int | None:
    out = _run(["pgrep", "-f", SERVER_MODULE])
    pids = [int(p) for p in out.split()] if out else []
    if len(pids) > 1:
        raise SystemExit(f"several {SERVER_MODULE} processes ({pids}); pass --server-pid")
    return pids[0] if pids else None


def cpu_name() -> str:
    if sys.platform == "darwin":
        return _run(["sysctl", "-n", "machdep.cpu.brand_string"]) or platform.processor()
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def run_metadata(args, model: str, workload, server_pid: int | None) -> dict:
    git = ["git", "-C", str(REPO_ROOT)]
    server = {"pid": server_pid}
    if server_pid:
        server["command"] = _run(["ps", "-o", "command=", "-p", str(server_pid)])
        server["started"] = _run(["ps", "-o", "lstart=", "-p", str(server_pid)])
    return {
        "type": "meta",
        "timestamp": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "git_sha": _run(git + ["rev-parse", "HEAD"]),
        "git_dirty": bool(_run(git + ["status", "--porcelain", "--untracked-files=no"])),
        "model": model,
        "base_url": args.base_url,
        "workload": workload.name,
        "kind": workload.kind,
        "rows": len(workload.rows),
        "right_rows": len(workload.right_rows) if workload.right_rows else None,
        "row_ids": [r.idx for r in workload.rows],
        "right_row_ids": [r.idx for r in workload.right_rows] if workload.right_rows else None,
        "stages": list(workload.stages) if workload.stages else None,
        "gold_filter_pass": workload.gold_filter_pass,
        "data_path": str(workload.data_path),
        "right_path": str(workload.right_path) if workload.right_path else None,
        "positives": _jsonable(workload.positives),
        "instruction": workload.instruction,
        "source_ids": list(workload.source_ids) if workload.source_ids else None,
        "arms": args.arms,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "prefix_cache_reset": not args.allow_warm_cache,
        "server": server,
        "platform": platform.platform(),
        "cpu": cpu_name(),
        "python": platform.python_version(),
    }


async def reset_prefix_cache(client: httpx.AsyncClient, base_url: str) -> bool:
    response = await client.post(f"{base_url}/reset_prefix_cache")
    return response.status_code < 400


async def resolve_model(client: httpx.AsyncClient, base_url: str, model: str | None) -> str:
    if model:
        return model
    response = await client.get(f"{base_url}/v1/models")
    response.raise_for_status()
    return response.json()["data"][0]["id"]


def _jsonable(selected):
    if selected is None:
        return None
    return sorted(list(x) if isinstance(x, tuple) else x for x in selected)


async def trial(client, args, spec, workload, model, server_pid) -> dict:
    if not args.allow_warm_cache and not await reset_prefix_cache(client, args.base_url):
        raise SystemExit(
            "could not reset the prefix cache (POST /reset_prefix_cache failed). "
            "Restart the server with VLLM_SERVER_DEV_MODE=1, or pass "
            "--allow-warm-cache to measure with a warm cache on purpose.")
    try:
        result, measured = await metrics.measure(
            lambda: run_arm(spec, client, args.base_url, workload, model),
            client, args.base_url, server_pid)
    except (ArmError, httpx.HTTPError) as exc:
        # One failing arm (e.g. a block too large for the context window)
        # is a result worth recording, not a reason to stop the run.
        return {"type": "trial", "arm": spec.name, "error": f"{type(exc).__name__}: {exc}"}
    rows_in = len(workload.rows) * len(workload.right_rows or [None])
    record = {
        "type": "trial",
        "arm": spec.name,
        **measured,
        "rows_in": rows_in,
        "rows_out": len(result.outputs),
        "rows_per_s": round(rows_in / measured["latency_s"], 3)
        if measured["latency_s"] else None,
        **summarize_trace(result.trace),
        "selected": _jsonable(result.selected),
        "outputs": dict(sorted(result.outputs.items())),
        "trace": result.trace,
    }
    if workload.kind == "pipeline":
        # The work the pipeline actually passed downstream; compared across
        # arms in report.drift_table.
        record["filter_pass"] = filter_pass(result.trace, len(workload.right_rows))
        record["join_out"] = len(result.selected)
    if workload.kind == "map":
        record["missing_rows"] = sorted({str(r.idx) for r in workload.rows} - set(result.outputs))
    return record


def _summary(record: dict) -> str:
    if "error" in record:
        return f"FAILED {record['error'][:200]}"
    return f"{record['latency_s']}s {record['llm_requests']} requests"


async def run(args) -> None:
    workload = WORKLOADS[args.workload](args.n)
    try:
        names = args.arms or workload.default_arms or DEFAULT_ARMS[workload.kind]
        specs = [parse_arm(a) for a in names]
        for spec in specs:
            check_arm(spec, workload)
    except ValueError as exc:
        raise SystemExit(str(exc))
    args.arms = [s.name for s in specs]
    server_pid = args.server_pid or find_server_pid()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    async with httpx.AsyncClient(timeout=args.timeout) as client:
        model = await resolve_model(client, args.base_url, args.model)
        meta = run_metadata(args, model, workload, server_pid)
        print(json.dumps({k: meta[k] for k in ("model", "git_sha", "rows", "arms")}))
        with out.open("w", encoding="utf-8") as f:
            f.write(json.dumps(meta) + "\n")
            for w in range(args.warmup):
                for spec in specs:
                    record = await trial(client, args, spec, workload, model, server_pid)
                    print(f"warmup {w + 1} {spec.name}: {_summary(record)}", flush=True)
            for repeat in range(1, args.repeats + 1):
                for spec in specs:
                    record = await trial(client, args, spec, workload, model, server_pid)
                    record["repeat"] = repeat
                    f.write(json.dumps(record) + "\n")
                    f.flush()
                    print(f"repeat {repeat} {spec.name}: {_summary(record)}", flush=True)
    print(f"wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-url", default="http://localhost:8003",
                        help="server root, without /v1")
    parser.add_argument("--model", default=None, help="defaults to the server's first model")
    parser.add_argument("--workload", choices=sorted(WORKLOADS), default="biodex")
    parser.add_argument("--n", type=int, default=None, help="rows to use; default all")
    parser.add_argument("--arms", default=None, type=lambda s: s.split(","),
                        help="comma-separated; default: every arm that fits the workload")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--server-pid", type=int, default=None,
                        help="for memory sampling; found with pgrep when omitted")
    parser.add_argument("--allow-warm-cache", action="store_true",
                        help="do not require resetting the prefix cache between trials")
    parser.add_argument("--timeout", type=float, default=3600)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    args.base_url = args.base_url.rstrip("/").removesuffix("/v1")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
