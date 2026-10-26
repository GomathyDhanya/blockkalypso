# execution/executor.py
from abc import ABC, abstractmethod
from typing import Optional


class CompletionResult:
    def __init__(self, text: str, request_id: str, finish_reason):
        self.text = text
        self.request_id = request_id
        self.finish_reason = finish_reason


class LLMRequestRejected(RuntimeError):
    """The LLM engine rejected a request outright (e.g. the prompt plus
    max_tokens exceeds the model's context length), so no CompletionResult
    is available. Distinct from a parse failure of a well-formed reply, so
    callers (e.g. a block operator) can tell the two apart and, in the
    future, react by retrying with a smaller block instead of crashing.
    """


class LLMExecutor(ABC):
    @abstractmethod
    async def complete(
        self,
        raw_request,
        prompt,
        max_tokens: int,
        pin: bool = False,
        priority: int = 0
    ) -> CompletionResult:
        pass

    async def chatcomplete(
        self,
        raw_request,
        prompt,
        max_tokens: int,
        pin: bool = False,
        priority: int = 0
    ) -> CompletionResult:
        pass

    @abstractmethod
    async def unpin(self, request_id: str):
        pass

    @abstractmethod
    async def abort(self, request_id: str):
        pass
