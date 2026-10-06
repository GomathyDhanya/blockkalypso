"""Kalypso's CartesianProduct keeps the ids of both sides of each pair."""

from vllm.kalypso.context import ExecutionState, SemanticInput, SemContext
from vllm.kalypso.sem_ops.joins import CartesianProduct


def ctx(text, idx):
    return SemContext(
        input=SemanticInput(data=text),
        state=ExecutionState(executor="ex", raw_request="req", idx=idx),
    )


def test_pairs_keep_left_and_right_ids():
    right = [ctx("r0", 10), ctx("r1", 11)]
    pairs = CartesianProduct(right_table=right)(ctx("left", 3))
    assert [(p.state.idx, p.state.right_idx) for p in pairs] == [(3, 10), (3, 11)]
    assert all(p.state.executor == "ex" and p.state.raw_request == "req" for p in pairs)
