"""Fixtures for Kalypso unit tests that need the scheduler.

``fake_kv_manager`` installs a KVMemoryManager without loading a tokenizer:
token counts are approximated as ``len(text) // 4``.
"""

import pytest

# Kalypso's controller package has an import cycle; importing the query
# processor first loads the modules in an order that works.
import vllm.kalypso.query_processor  # noqa: F401
from vllm.kalypso.budget import KVMemoryManager


class _FakeTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        return "\n".join(str(m.get("content", "")) for m in messages)

    def encode(self, text, add_special_tokens=False):
        return [0] * max(1, len(text) // 4)


@pytest.fixture
def fake_kv_manager():
    import asyncio

    mgr = KVMemoryManager.__new__(KVMemoryManager)
    mgr.tokenizer = _FakeTokenizer()
    mgr.bytes_per_token = 1
    mgr._capacity = 10_000_000
    mgr._global_used = 0
    mgr._stage_capacity, mgr._stage_min_capacity = {}, {}
    mgr._stage_max_capacity, mgr._stage_used, mgr._stage_inflight = {}, {}, {}
    mgr._cond = asyncio.Condition()
    saved = KVMemoryManager._instance
    KVMemoryManager._instance = mgr
    yield mgr
    KVMemoryManager._instance = saved
