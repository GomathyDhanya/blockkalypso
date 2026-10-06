"""Execution paths compared by the harness.

Each arm runs one workload end to end and returns the model output per row.
Arms do not measure anything; ``metrics.measure`` wraps them.
"""

import asyncio
from dataclasses import dataclass

import httpx

# Kalypso's controller package has an import cycle; importing the query
# processor first loads the modules in an order that works.
import vllm.kalypso.query_processor  # noqa: F401
from vllm.kalypso.execution.vllm_executor import VLLMExecutor
from vllm.kalypso.sem_ops.base import OpName
from vllm.kalypso.sem_ops.prompt_utils import get_data_prompt, get_prompt

from .workloads import Workload

# Kalypso retries a truncated map at this limit (SemMap.MAX_TOKEN_LIMIT), so
# it is the most a Kalypso map output can be; the direct arm uses it up front.
DIRECT_MAX_TOKENS = 512

_OP_NAMES = {"sem_map": OpName.SEM_MAP, "sem_filter": OpName.SEM_FILTER}


@dataclass
class ArmOutput:
    outputs: dict[int, str]   # row idx -> final model output for that row


class ArmError(RuntimeError):
    pass


async def kalypso_per_tuple(client: httpx.AsyncClient, base_url: str,
                            workload: Workload, model: str) -> ArmOutput:
    """Kalypso's unmodified per-tuple operator via /v1/semantic/query."""
    payload = {
        "data_path": str(workload.data_path),
        "ops": [{"op": workload.op, "args": {"prompt": workload.instruction}}],
        "model_name": model,
    }
    response = await client.post(f"{base_url}/v1/semantic/query", json=payload)
    if response.status_code >= 400:
        raise ArmError(f"/v1/semantic/query returned {response.status_code}: "
                       f"{response.text[:500]}")
    outputs = {}
    for result in response.json().get("results", []):
        values = [o.get("value") for o in result.get("outputs", [])]
        outputs[int(result["idx"])] = "" if not values else str(values[-1])
    return ArmOutput(outputs)


async def vllm_direct(client: httpx.AsyncClient, base_url: str,
                      workload: Workload, model: str) -> ArmOutput:
    """The same per-row prompts, all sent at once to vLLM's chat endpoint.

    Messages, sampling and chat-template arguments match what Kalypso's
    VLLMExecutor sends; only ``max_tokens`` differs (see DIRECT_MAX_TOKENS).
    """
    op = _OP_NAMES[workload.op]

    async def one(row):
        body = {
            "model": model,
            "messages": get_prompt(workload.instruction, get_data_prompt(row.text), op=op),
            "max_tokens": DIRECT_MAX_TOKENS,
            **VLLMExecutor.DEFAULT_SAMPLING,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        response = await client.post(f"{base_url}/v1/chat/completions", json=body)
        if response.status_code >= 400:
            raise ArmError(f"/v1/chat/completions returned {response.status_code} "
                           f"for row {row.idx}: {response.text[:500]}")
        message = response.json()["choices"][0].get("message") or {}
        return row.idx, message.get("content") or ""

    pairs = await asyncio.gather(*(one(row) for row in workload.rows))
    return ArmOutput(dict(pairs))


ARMS = {
    "kalypso_per_tuple": kalypso_per_tuple,
    "vllm_direct": vllm_direct,
}
