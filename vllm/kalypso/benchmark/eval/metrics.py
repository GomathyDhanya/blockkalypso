"""Server-side measurements shared by every arm.

Counters come from vLLM's Prometheus endpoint (``<base>/metrics``) as a
before/after delta, so requests Kalypso issues internally (e.g. length
retries) are counted too. Gauges are sampled while the arm runs: KV-cache use
from ``/metrics``, and resident memory of the server's process tree via
``ps``.

The counters are server-wide: run nothing else against the server while the
harness is measuring.
"""

import asyncio
import re
import subprocess
import time

import httpx

_SAMPLE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+(\S+)$")
_REASON = re.compile(r'finished_reason="([^"]*)"')

COUNTERS = {
    "vllm:prompt_tokens_total": "prompt_tokens",
    "vllm:generation_tokens_total": "generation_tokens",
    "vllm:prefix_cache_hits_total": "prefix_cache_hits",
    "vllm:prefix_cache_queries_total": "prefix_cache_queries",
}
REQUESTS = "vllm:request_success_total"
KV_GAUGE = "vllm:kv_cache_usage_perc"


def parse_metrics(text: str) -> dict:
    """Sum each counter we track across its label sets.

    ``request_success_total`` is kept per ``finished_reason`` under
    ``"requests"`` (stop, length, abort, error).
    """
    out = {key: 0.0 for key in COUNTERS.values()}
    out["requests"] = {}
    out["kv_cache_pct"] = None
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        match = _SAMPLE.match(line)
        if not match:
            continue
        name, labels, value = match.group(1), match.group(2) or "", float(match.group(3))
        if name in COUNTERS:
            out[COUNTERS[name]] += value
        elif name == REQUESTS:
            reason = _REASON.search(labels)
            key = reason.group(1) if reason else "unknown"
            out["requests"][key] = out["requests"].get(key, 0.0) + value
        elif name == KV_GAUGE:
            out["kv_cache_pct"] = max(out["kv_cache_pct"] or 0.0, value * 100)
    return out


async def snapshot(client: httpx.AsyncClient, base_url: str) -> dict:
    response = await client.get(f"{base_url}/metrics", timeout=10.0)
    response.raise_for_status()
    return parse_metrics(response.text)


def delta(before: dict, after: dict) -> dict:
    reasons = set(before["requests"]) | set(after["requests"])
    by_reason = {r: int(after["requests"].get(r, 0) - before["requests"].get(r, 0))
                 for r in sorted(reasons)}
    by_reason = {r: n for r, n in by_reason.items() if n}
    queries = after["prefix_cache_queries"] - before["prefix_cache_queries"]
    hits = after["prefix_cache_hits"] - before["prefix_cache_hits"]
    return {
        "llm_requests": sum(by_reason.values()),
        "requests_by_reason": by_reason,
        "truncated": by_reason.get("length", 0),
        "prompt_tokens": int(after["prompt_tokens"] - before["prompt_tokens"]),
        "generation_tokens": int(after["generation_tokens"] - before["generation_tokens"]),
        "prefix_cache_hit_rate": round(hits / queries, 4) if queries else None,
    }


def process_tree_rss_kb(root_pid: int) -> int | None:
    """Resident memory of ``root_pid`` plus all its descendants, in KiB."""
    try:
        out = subprocess.check_output(["ps", "-A", "-o", "pid=,ppid=,rss="], text=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    children, rss = {}, {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) != 3:
            continue
        pid, ppid, kb = map(int, parts)
        children.setdefault(ppid, []).append(pid)
        rss[pid] = kb
    if root_pid not in rss:
        return None
    total, stack = 0, [root_pid]
    while stack:
        pid = stack.pop()
        total += rss.get(pid, 0)
        stack.extend(children.get(pid, ()))
    return total


async def _sample_kv(client, base_url, stop, samples, interval=0.25):
    while not stop.is_set():
        try:
            value = (await snapshot(client, base_url))["kv_cache_pct"]
            if value is not None:
                samples.append(value)
        except (httpx.HTTPError, ValueError):
            pass
        try:
            await asyncio.wait_for(stop.wait(), timeout=interval)
        except asyncio.TimeoutError:
            pass


async def _sample_rss(pid, stop, samples, interval=0.1):
    while not stop.is_set():
        value = await asyncio.to_thread(process_tree_rss_kb, pid)
        if value is not None:
            samples.append(value)
        try:
            await asyncio.wait_for(stop.wait(), timeout=interval)
        except asyncio.TimeoutError:
            pass


async def measure(work, client: httpx.AsyncClient, base_url: str,
                  server_pid: int | None = None):
    """Run ``await work()`` and return ``(result, metrics)``."""
    before = await snapshot(client, base_url)
    stop = asyncio.Event()
    kv_samples, rss_samples = [], []
    samplers = [asyncio.create_task(_sample_kv(client, base_url, stop, kv_samples))]
    if server_pid:
        samplers.append(asyncio.create_task(_sample_rss(server_pid, stop, rss_samples)))
    start = time.perf_counter()
    try:
        result = await work()
    finally:
        latency = time.perf_counter() - start
        stop.set()
        await asyncio.gather(*samplers)
    after = await snapshot(client, base_url)
    return result, {
        "latency_s": round(latency, 3),
        **delta(before, after),
        "peak_kv_cache_pct": round(max(kv_samples), 2) if kv_samples else None,
        "kv_samples": len(kv_samples),
        "peak_server_rss_kb": max(rss_samples) if rss_samples else None,
    }
