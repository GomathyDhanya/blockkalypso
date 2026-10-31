"""Server-side measurements shared by every arm.

Counters come from vLLM's Prometheus endpoint (``<base>/metrics``) as a
before/after delta, so requests Kalypso issues internally (e.g. length
retries) are counted too. Gauges are sampled while the arm runs: KV-cache use
from ``/metrics`` (KV-cache use, requests running and waiting), and resident
memory of the server's process tree via ``ps``.

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
_LE = re.compile(r'le="([^"]*)"')

COUNTERS = {
    "vllm:prompt_tokens_total": "prompt_tokens",
    "vllm:generation_tokens_total": "generation_tokens",
    "vllm:prefix_cache_hits_total": "prefix_cache_hits",
    "vllm:prefix_cache_queries_total": "prefix_cache_queries",
    # Histogram _sum/_count pairs: summed like any other counter, then
    # turned into a mean of (after - before) in delta(). vLLM exposes these
    # over the same /metrics HTTP endpoint, so they work against a remote
    # server (e.g. a Modal-hosted one) exactly like the counters above do.
    "vllm:request_prefill_time_seconds_sum": "prefill_time_sum_s",
    "vllm:request_prefill_time_seconds_count": "prefill_time_count",
    "vllm:request_decode_time_seconds_sum": "decode_time_sum_s",
    "vllm:request_decode_time_seconds_count": "decode_time_count",
    # Requests the scheduler evicted to free KV cache (recomputed later).
    "vllm:num_preemptions_total": "preemptions",
}
REQUESTS = "vllm:request_success_total"
KV_GAUGE = "vllm:kv_cache_usage_perc"
# Requests in the running batch and in the queue; their peaks show how many
# calls actually decode together and whether calls wait for a slot.
RUNNING_GAUGE = "vllm:num_requests_running"
WAITING_GAUGE = "vllm:num_requests_waiting"
# Requests by their max_tokens. Block operators send a single-row call (a
# repair, or a block of one row) with max_tokens <= 4 (BlockSemFilter's
# single_max_tokens, BlockSemJoin's pair_max_tokens); a block call asks for
# 16 + 6 per row and the per-tuple filter for 8. The le="5" bucket therefore
# counts exactly a block arm's single-row calls.
MAX_TOKENS_BUCKET = "vllm:request_params_max_tokens_bucket"
SINGLE_ROW_MAX_TOKENS = 5.0


def parse_metrics(text: str) -> dict:
    """Sum each counter we track across its label sets.

    ``request_success_total`` is kept per ``finished_reason`` under
    ``"requests"`` (stop, length, abort, error).
    """
    out = {key: 0.0 for key in COUNTERS.values()}
    out["requests"] = {}
    out["kv_cache_pct"] = None
    out["single_row_calls"] = 0.0
    out["max_tokens_buckets"] = {}
    out["running"] = None
    out["waiting"] = None
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
        elif name in (RUNNING_GAUGE, WAITING_GAUGE):
            key = "running" if name == RUNNING_GAUGE else "waiting"
            out[key] = (out[key] or 0.0) + value
        elif name == MAX_TOKENS_BUCKET:
            le = _LE.search(labels)
            if not le:
                continue
            bound = le.group(1)
            buckets = out["max_tokens_buckets"]
            buckets[bound] = buckets.get(bound, 0.0) + value
            if bound != "+Inf" and float(bound) == SINGLE_ROW_MAX_TOKENS:
                out["single_row_calls"] += value
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
    prefill_n = after["prefill_time_count"] - before["prefill_time_count"]
    decode_n = after["decode_time_count"] - before["decode_time_count"]
    prefill_sum = after["prefill_time_sum_s"] - before["prefill_time_sum_s"]
    decode_sum = after["decode_time_sum_s"] - before["decode_time_sum_s"]
    return {
        "llm_requests": sum(by_reason.values()),
        "requests_by_reason": by_reason,
        "truncated": by_reason.get("length", 0),
        # Requests with max_tokens <= 5; see MAX_TOKENS_BUCKET.
        "single_row_calls": int(after["single_row_calls"] - before["single_row_calls"]),
        "prompt_tokens": int(after["prompt_tokens"] - before["prompt_tokens"]),
        "generation_tokens": int(after["generation_tokens"] - before["generation_tokens"]),
        "prefix_cache_hit_rate": round(hits / queries, 4) if queries else None,
        # Mean over this trial's requests, from the histogram's _sum/_count
        # delta -- the standard way to read a mean out of a Prometheus
        # histogram without needing the bucket boundaries.
        "mean_prefill_time_s": round(prefill_sum / prefill_n, 4) if prefill_n else None,
        "mean_decode_time_s": round(decode_sum / decode_n, 4) if decode_n else None,
        "preemptions": int(after["preemptions"] - before["preemptions"]),
        "requests_by_max_tokens": requests_by_max_tokens(before, after),
    }


def requests_by_max_tokens(before: dict, after: dict) -> dict:
    """Requests per max_tokens range, e.g. {"<=5": 3, "6-10": 500, ">1000": 2}.

    vLLM's histogram is cumulative, so each range is the difference of two
    adjacent buckets. Operators ask for different max_tokens (block single-row
    calls 4, per-tuple predicates 8, block calls 16 + 6 per row, maps more),
    so in a pipeline this splits requests roughly by stage.
    """
    def cum(bound):
        return int(after["max_tokens_buckets"].get(bound, 0)
                   - before["max_tokens_buckets"].get(bound, 0))

    bounds = sorted((b for b in after["max_tokens_buckets"] if b != "+Inf"), key=float)
    out, below, low = {}, 0, None
    for bound in bounds:
        top = int(float(bound))
        n = cum(bound) - below
        if n:
            out[f"<={top}" if low is None else f"{low}-{top}"] = n
        below, low = cum(bound), top + 1
    if "+Inf" in after["max_tokens_buckets"] and cum("+Inf") - below:
        out[f">{low - 1}" if low else "any"] = cum("+Inf") - below
    return out


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


async def _sample_gauges(client, base_url, stop, samples, start, interval=0.25):
    """Append (seconds since start, KV %, running, waiting) every ``interval``."""
    while not stop.is_set():
        try:
            snap = await snapshot(client, base_url)
            samples.append((round(time.perf_counter() - start, 2), snap["kv_cache_pct"],
                            snap["running"], snap["waiting"]))
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
    gauge_samples, rss_samples = [], []
    start = time.perf_counter()
    samplers = [asyncio.create_task(
        _sample_gauges(client, base_url, stop, gauge_samples, start))]
    if server_pid:
        samplers.append(asyncio.create_task(_sample_rss(server_pid, stop, rss_samples)))
    try:
        result = await work()
    finally:
        latency = time.perf_counter() - start
        stop.set()
        await asyncio.gather(*samplers)
    after = await snapshot(client, base_url)

    def peak(i):
        values = [s[i] for s in gauge_samples if s[i] is not None]
        return max(values) if values else None

    kv = peak(1)
    return result, {
        "latency_s": round(latency, 3),
        **delta(before, after),
        "peak_kv_cache_pct": round(kv, 2) if kv is not None else None,
        "kv_samples": len(gauge_samples),
        "peak_running": int(peak(2)) if peak(2) is not None else None,
        "peak_waiting": int(peak(3)) if peak(3) is not None else None,
        "peak_server_rss_kb": max(rss_samples) if rss_samples else None,
        # [seconds, KV %, running, waiting] every 0.25 s while the arm ran.
        "timeline": [list(s) for s in gauge_samples],
    }
