"""Execution paths compared by the harness.

Each arm runs one workload end to end. Arms do not measure anything;
``metrics.measure`` wraps them.

Arm names:
  kalypso_per_tuple     Kalypso's original one-row (one-pair) operators
  vllm_direct           the same per-row prompts sent straight to vLLM
  kalypso_block:N       BlockSemFilter with block_size N (filter workloads)
  kalypso_block:KxJ     BlockSemJoin with K left x J right rows (join workloads)
"""

import asyncio
from dataclasses import dataclass

import httpx

# Kalypso's controller package has an import cycle; importing the query
# processor first loads the modules in an order that works.
import vllm.kalypso.query_processor  # noqa: F401
from vllm.kalypso.execution.vllm_executor import VLLMExecutor
from vllm.kalypso.sem_ops.base import OpName
from vllm.kalypso.sem_ops.filters import SemFilter
from vllm.kalypso.sem_ops.prompt_utils import get_data_prompt, get_prompt

from .workloads import FILTER, JOIN, MAP, Workload

# Kalypso retries a truncated map at this limit (SemMap.MAX_TOKEN_LIMIT), so
# it is the most a Kalypso map output can be; the direct arm uses it up front.
DIRECT_MAP_MAX_TOKENS = 512
# SemFilter's default, used for filters and for the per-pair join check.
DIRECT_PREDICATE_MAX_TOKENS = 8

_PREDICATE = SemFilter(instruction="")


@dataclass
class ArmOutput:
    # Map: row idx -> output. Filter: row idx -> verdict. Join: "l,r" -> verdict.
    outputs: dict[str, str]
    # Filter: idxs that passed. Join: (left, right) pairs that matched.
    # None for maps.
    selected: frozenset | None = None
    # The server's per-query execution trace (Kalypso arms only).
    trace: dict | None = None


class ArmError(RuntimeError):
    pass


@dataclass(frozen=True)
class ArmSpec:
    name: str                          # as given on the command line
    kind: str                          # "kalypso_per_tuple", "vllm_direct", "kalypso_block"
    block: tuple[int, ...] | None = None


def parse_arm(name: str) -> ArmSpec:
    kind, _, size = name.partition(":")
    if kind in ("kalypso_per_tuple", "vllm_direct") and not size:
        return ArmSpec(name, kind)
    if kind == "kalypso_block" and size:
        try:
            block = tuple(int(part) for part in size.lower().split("x"))
        except ValueError:
            block = ()
        if len(block) in (1, 2) and all(b >= 1 for b in block):
            return ArmSpec(name, kind, block)
    raise ValueError(
        f"bad arm {name!r}; use kalypso_per_tuple, vllm_direct, "
        "kalypso_block:N (filter) or kalypso_block:KxJ (join)")


def check_arm(spec: ArmSpec, workload: Workload) -> None:
    if spec.kind != "kalypso_block":
        return
    if workload.kind == MAP:
        raise ValueError(f"{spec.name}: there is no block operator for maps")
    if workload.kind == FILTER and len(spec.block) != 1:
        raise ValueError(f"{spec.name}: filter blocks take one size, e.g. kalypso_block:10")
    if workload.kind == JOIN and len(spec.block) != 2:
        raise ValueError(f"{spec.name}: join blocks take KxJ, e.g. kalypso_block:5x5")


def kalypso_ops(workload: Workload, block: tuple[int, ...] | None = None) -> list[dict]:
    """The /v1/semantic/query ops for a workload, per-tuple or blocked."""
    if workload.kind == MAP:
        return [{"op": OpName.SEM_MAP, "args": {"prompt": workload.instruction}}]
    if workload.kind == FILTER:
        args = {"prompt": workload.instruction}
        if block:
            args["block_size"] = block[0]
        return [{"op": OpName.SEM_FILTER, "args": args}]
    args = {"instruction": workload.instruction, "right_table": str(workload.right_path)}
    if block:
        args["left_block_size"], args["right_block_size"] = block
    return [{"op": OpName.JOIN, "args": args}]


def parse_query_response(workload: Workload, body: dict) -> ArmOutput:
    outputs, selected = {}, set()
    for result in body.get("results", []):
        values = [o.get("value") for o in result.get("outputs", [])]
        value = "" if not values else str(values[-1])
        idx = int(result["idx"])
        if workload.kind == JOIN:
            if result.get("right_idx") is None:
                raise ArmError("join result has no right_idx; restart the server on "
                               "code that reports it (api_server.py)")
            key = (idx, int(result["right_idx"]))
            outputs[f"{key[0]},{key[1]}"] = value
            selected.add(key)
        else:
            outputs[str(idx)] = value
            selected.add(idx)
    return ArmOutput(outputs, None if workload.kind == MAP else frozenset(selected),
                     trace=body.get("trace"))


async def run_kalypso(client: httpx.AsyncClient, base_url: str, workload: Workload,
                      model: str, block: tuple[int, ...] | None = None) -> ArmOutput:
    payload = {
        "data_path": str(workload.data_path),
        "ops": kalypso_ops(workload, block),
        "model_name": model,
    }
    response = await client.post(f"{base_url}/v1/semantic/query", json=payload)
    if response.status_code >= 400:
        raise ArmError(f"/v1/semantic/query returned {response.status_code}: "
                       f"{response.text[:500]}")
    return parse_query_response(workload, response.json())


def direct_requests(workload: Workload) -> list[tuple[str, list, int]]:
    """(key, messages, max_tokens) for each call Kalypso's per-tuple path makes."""
    if workload.kind == MAP:
        return [(str(r.idx),
                 get_prompt(workload.instruction, get_data_prompt(r.text), op=OpName.SEM_MAP),
                 DIRECT_MAP_MAX_TOKENS) for r in workload.rows]
    if workload.kind == FILTER:
        return [(str(r.idx),
                 get_prompt(workload.instruction, get_data_prompt(r.text), op=OpName.SEM_FILTER),
                 DIRECT_PREDICATE_MAX_TOKENS) for r in workload.rows]
    # CartesianProduct + SemFilter: left messages followed by right messages.
    return [(f"{left.idx},{right.idx}",
             get_prompt(workload.instruction,
                        get_data_prompt(left.text) + get_data_prompt(right.text),
                        op=OpName.SEM_JOIN),
             DIRECT_PREDICATE_MAX_TOKENS)
            for left in workload.rows for right in workload.right_rows]


async def run_direct(client: httpx.AsyncClient, base_url: str, workload: Workload,
                     model: str) -> ArmOutput:
    """All per-row prompts at once to vLLM's chat endpoint, no Kalypso.

    Messages, sampling and chat-template arguments match what Kalypso's
    VLLMExecutor sends; for maps ``max_tokens`` is DIRECT_MAP_MAX_TOKENS
    instead of Kalypso's per-row estimate.
    """
    async def one(key, messages, max_tokens):
        body = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            **VLLMExecutor.DEFAULT_SAMPLING,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        response = await client.post(f"{base_url}/v1/chat/completions", json=body)
        if response.status_code >= 400:
            raise ArmError(f"/v1/chat/completions returned {response.status_code} "
                           f"for {key}: {response.text[:500]}")
        message = response.json()["choices"][0].get("message") or {}
        return key, message.get("content") or ""

    replies = await asyncio.gather(*(one(*req) for req in direct_requests(workload)))
    if workload.kind == MAP:
        return ArmOutput(dict(replies))
    # Same verdict parsing as SemFilter: the row passes unless "false" appears.
    outputs = {key: text.strip().lower() for key, text in replies}
    passed = [key for key, verdict in outputs.items() if _PREDICATE.predicate_passed(verdict)]
    if workload.kind == FILTER:
        selected = frozenset(int(k) for k in passed)
    else:
        selected = frozenset(tuple(int(x) for x in k.split(",")) for k in passed)
    return ArmOutput(outputs, selected)


async def run_arm(spec: ArmSpec, client: httpx.AsyncClient, base_url: str,
                  workload: Workload, model: str) -> ArmOutput:
    if spec.kind == "vllm_direct":
        return await run_direct(client, base_url, workload, model)
    return await run_kalypso(client, base_url, workload, model, spec.block)


DEFAULT_ARMS = {
    MAP: ["kalypso_per_tuple", "vllm_direct"],
    FILTER: ["kalypso_per_tuple", "vllm_direct",
             "kalypso_block:5", "kalypso_block:10", "kalypso_block:20"],
    JOIN: ["kalypso_per_tuple", "vllm_direct", "kalypso_block:2x2", "kalypso_block:5x5"],
}
