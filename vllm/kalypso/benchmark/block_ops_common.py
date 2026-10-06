"""Shared pieces for the standalone block-operator checks.

``HTTPExecutor`` has the same ``execute`` signature as Kalypso's
``VLLMExecutor`` but talks to any OpenAI-compatible server over HTTP, so the
block operators can be run without starting Kalypso's query pipeline.
"""

import sys
import time
from pathlib import Path

import httpx

# Make `vllm.kalypso...` importable when the script is run from a checkout.
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SAMPLE_DATA = Path(__file__).resolve().parent / "sample_data"


class Completion:
    def __init__(self, text, finish_reason, request_id):
        self.text = text
        self.finish_reason = finish_reason
        self.request_id = request_id


class HTTPExecutor:
    """Counts calls and prompt/completion tokens for one experiment arm."""

    def __init__(self, client: httpx.AsyncClient, base_url: str, model: str, seed=42):
        self.client = client
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.seed = seed
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    async def execute(self, raw_request, prompt, max_tokens, pin=False, priority=0):
        self.calls += 1
        r = await self.client.post(
            f"{self.base_url}/chat/completions",
            json={
                "model": self.model,
                "messages": prompt,
                "max_tokens": max_tokens,
                "temperature": 0.0,
                "seed": self.seed,
            },
        )
        r.raise_for_status()
        data = r.json()
        usage = data.get("usage") or {}
        self.prompt_tokens += int(usage.get("prompt_tokens") or 0)
        self.completion_tokens += int(usage.get("completion_tokens") or 0)
        choice = data["choices"][0]
        return Completion(
            (choice.get("message") or {}).get("content") or "",
            choice.get("finish_reason"),
            data.get("id"),
        )

    def summary(self, seconds: float) -> dict:
        return {
            "llm_calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "latency_s": round(seconds, 2),
        }


async def resolve_model(client: httpx.AsyncClient, base_url: str, model: str | None) -> str:
    if model:
        return model
    r = await client.get(f"{base_url.rstrip('/')}/models")
    r.raise_for_status()
    return r.json()["data"][0]["id"]


class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.seconds = time.perf_counter() - self.start
