"""Execution paths compared by the harness.

Each arm runs one workload end to end. Arms do not measure anything;
``metrics.measure`` wraps them.

Arm names:
  kalypso_per_tuple     Kalypso's original one-row (one-pair) operators
  vllm_direct           the same per-row prompts sent straight to vLLM
  kalypso_block:N       BlockSemFilter with block_size N (filter workloads)
  kalypso_block:N@T     the same, also capped at T estimated tokens per block
                        (max_block_tokens; Kalypso's Level 2 sizing)
  kalypso_block:KxJ     BlockSemJoin with K left x J right rows (join workloads)
  kalypso_block:N/KxJ   pipelines: one size per filter or join stage, in order,
                        separated by "/" (1 or 1x1 keeps that stage per-tuple);
                        map stages always run per-tuple
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

from .workloads import FILTER, JOIN, MAP, PIPELINE, Workload

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
    # One size per blockable stage: (N,) for a filter, (K, J) for a join.
    groups: tuple[tuple[int, ...], ...] | None = None
    # Per stage, the filter's token budget (N@T), or None.
    budgets: tuple[int | None, ...] | None = None

    @property
    def block(self) -> tuple[int, ...] | None:
        """The single block size of a one-operator arm."""
        return self.groups[0] if self.groups else None


def _size(text: str) -> tuple[tuple[int, ...], int | None]:
    """``"16@4096"`` -> ``((16,), 4096)``; ``"2x8"`` -> ``((2, 8), None)``.

    Returns ``((), None)`` when the text is not a valid size. A token budget
    only applies to a filter block of more than one row.
    """
    size_text, sep, budget_text = text.partition("@")
    try:
        size = tuple(int(part) for part in size_text.lower().split("x"))
        budget = int(budget_text) if sep else None
    except ValueError:
        return (), None
    if len(size) not in (1, 2) or any(b < 1 for b in size):
        return (), None
    if budget is not None and (budget < 1 or len(size) != 1 or size[0] < 2):
        return (), None
    return size, budget


def parse_arm(name: str) -> ArmSpec:
    kind, _, sizes = name.partition(":")
    if kind in ("kalypso_per_tuple", "vllm_direct") and not sizes:
        return ArmSpec(name, kind)
    if kind == "kalypso_block" and sizes:
        parsed = [_size(part) for part in sizes.split("/")]
        if all(size for size, _ in parsed):
            return ArmSpec(name, kind, tuple(size for size, _ in parsed),
                           tuple(budget for _, budget in parsed))
    raise ValueError(
        f"bad arm {name!r}; use kalypso_per_tuple, vllm_direct, "
        "kalypso_block:N or N@T (filter), kalypso_block:KxJ (join) or "
        "kalypso_block:N/KxJ (pipeline)")


def _blockable(workload: Workload) -> list[dict]:
    return [s for s in workload.stages if s["op"] in ("sem_filter", "join")]


def check_arm(spec: ArmSpec, workload: Workload) -> None:
    if workload.kind == PIPELINE:
        if spec.kind == "vllm_direct":
            raise ValueError("vllm_direct runs single operators only, not pipelines")
        if spec.kind != "kalypso_block":
            return
        stages = _blockable(workload)
        if len(spec.groups) != len(stages):
            raise ValueError(f"{spec.name}: give one size per filter/join stage "
                             f"({len(stages)}), separated by '/'")
        for size, stage in zip(spec.groups, stages):
            if len(size) != (2 if stage["op"] == "join" else 1):
                raise ValueError(f"{spec.name}: {stage['op']} takes "
                                 f"{'KxJ' if stage['op'] == 'join' else 'N'}, got {size}")
        return
    if spec.kind != "kalypso_block":
        return
    if len(spec.groups) != 1:
        raise ValueError(f"{spec.name}: '/' sizes are for pipeline workloads")
    if workload.kind == MAP:
        raise ValueError(f"{spec.name}: there is no block operator for maps")
    if workload.kind == FILTER and len(spec.block) != 1:
        raise ValueError(f"{spec.name}: filter blocks take one size, e.g. kalypso_block:10")
    if workload.kind == JOIN and len(spec.block) != 2:
        raise ValueError(f"{spec.name}: join blocks take KxJ, e.g. kalypso_block:5x5")


def _filter_args(prompt: str, size, budget) -> dict:
    args = {"prompt": prompt}
    if size and size[0] > 1:
        args["block_size"] = size[0]
        if budget is not None:
            args["max_block_tokens"] = budget
    return args


def pipeline_ops(workload: Workload, groups=None, budgets=None) -> list[dict]:
    """Ops for a pipeline; ``groups`` gives each filter/join stage's size."""
    sizes = iter(zip(groups or (), budgets or (None,) * len(groups or ())))
    ops = []
    for stage in workload.stages:
        if stage["op"] == "sem_map":
            ops.append({"op": OpName.SEM_MAP, "args": {"prompt": stage["prompt"]}})
            continue
        size, budget = next(sizes, (None, None))
        if stage["op"] == "sem_filter":
            ops.append({"op": OpName.SEM_FILTER,
                        "args": _filter_args(stage["prompt"], size, budget)})
        else:
            args = {"instruction": stage["instruction"],
                    "right_table": str(workload.right_path)}
            if size and size != (1, 1):
                args["left_block_size"], args["right_block_size"] = size
            ops.append({"op": OpName.JOIN, "args": args})
    return ops


def kalypso_ops(workload: Workload, block: tuple[int, ...] | None = None,
                budget: int | None = None) -> list[dict]:
    """The /v1/semantic/query ops for a one-operator workload."""
    if workload.kind == MAP:
        return [{"op": OpName.SEM_MAP, "args": {"prompt": workload.instruction}}]
    if workload.kind == FILTER:
        return [{"op": OpName.SEM_FILTER,
                 "args": _filter_args(workload.instruction, block, budget)}]
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
        if workload.kind in (JOIN, PIPELINE):
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
                      model: str, spec: ArmSpec | None = None) -> ArmOutput:
    groups = spec.groups if spec else None
    budgets = spec.budgets if spec else None
    payload = {
        "data_path": str(workload.data_path),
        "ops": (pipeline_ops(workload, groups, budgets) if workload.kind == PIPELINE
                else kalypso_ops(workload, groups[0] if groups else None,
                                 budgets[0] if budgets else None)),
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
    return await run_kalypso(client, base_url, workload, model, spec)


DEFAULT_ARMS = {
    MAP: ["kalypso_per_tuple", "vllm_direct"],
    FILTER: ["kalypso_per_tuple", "vllm_direct",
             "kalypso_block:5", "kalypso_block:10", "kalypso_block:20"],
    JOIN: ["kalypso_per_tuple", "vllm_direct", "kalypso_block:2x2", "kalypso_block:5x5"],
    PIPELINE: ["kalypso_per_tuple"],
}
