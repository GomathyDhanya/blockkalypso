"""Level 3: Trummer's block-join cost model (arXiv:2510.08489)."""

import math

import pytest

from vllm.kalypso.controller import cost_model as cm
from vllm.kalypso.controller.plan import SemanticPlan

from .test_block_filter import row


def inputs(**kw):
    base = dict(p=100, s1=50, s2=40, s3=5, sigma=0.1, t=4000, r1=64, r2=64, g=1.0)
    return cm.JoinCostInputs(**{**base, **kw})


def test_formulas_match_the_paper():
    x = inputs()
    # Lemma 4.1: p + b1*s1 + b2*s2 + b1*b2*σ*s3
    assert cm.tokens_per_call(x, 4, 8) == 100 + 4 * 50 + 8 * 40 + 4 * 8 * 0.1 * 5
    # Corollary 4.4 with g = 3
    y = inputs(g=3.0)
    assert cm.cost(y, 4, 8) == pytest.approx(
        (64 / 4) * (64 / 8) * (100 + 200 + 320 + 4 * 8 * 0.1 * 5 * 3))
    # Equation 1
    assert cm.feasible(x, 4, 8) and not cm.feasible(x, 64, 64)


def test_closed_form_is_the_continuous_optimum_on_the_constraint():
    x = inputs(r1=10_000, r2=10_000)
    b1, b2 = cm.closed_form(x)
    # Theorem 5.2: the optimum uses the whole budget.
    assert b1 * x.s1 + b2 * x.s2 + b1 * b2 * x.s3 * x.sigma == pytest.approx(x.t)
    # The integer choice costs within 1% of the continuous optimum (the cost
    # is flat near it, so rounding along the constraint can move b1, b2 a bit).
    choice = cm.optimal_block(x)
    assert choice.cost <= cm.cost(x, b1, b2) * 1.01
    # and nothing on the integer grid is cheaper.
    assert all(cm.cost(x, i, j) >= choice.cost - 1e-9
               for i in range(1, 60) for j in range(1, 100) if cm.feasible(x, i, j))


def test_choice_respects_table_sizes_and_edge_cases():
    small = cm.optimal_block(inputs(r1=3, r2=2))
    assert (small.b1, small.b2) == (3, 2) and small.calls == 1
    no_matches = cm.optimal_block(inputs(sigma=0.0, r1=10_000, r2=10_000))
    assert no_matches.b1_closed_form is None
    assert cm.feasible(inputs(sigma=0.0), no_matches.b1, no_matches.b2)
    with pytest.raises(ValueError):
        cm.optimal_block(inputs(t=10))              # one pair does not fit


def test_planner_sizes_a_block_join_with_the_cost_model(fake_kv_manager, monkeypatch):
    right = [row(f"category {i} " * 20, 100 + i) for i in range(8)]
    left = [row(f"complaint {i} " * 30, i) for i in range(16)]
    monkeypatch.setattr("vllm.kalypso.controller.plan.data._data_source",
                        lambda *args: iter(right))
    plan = SemanticPlan(executor=None)
    plan._input_ctxs = left
    (op,) = plan.parse_ops([{"op": "join", "args": {
        "instruction": "same", "right_table": "r.csv", "block_sizing": "trummer",
        "selectivity": 0.125, "context_tokens": 4096}}])
    assert type(op).__name__ == "BlockSemJoin"
    assert (op.left_block_size, op.right_block_size) == (op.sizing["b1"], op.sizing["b2"])
    x = cm.JoinCostInputs(**op.sizing["inputs"])
    assert x.r1 == 16 and x.r2 == 8 and x.sigma == 0.125
    assert cm.feasible(x, op.left_block_size, op.right_block_size)
    with pytest.raises(ValueError):
        plan.parse_ops([{"op": "join", "args": {"instruction": "x", "right_table": "r",
                                                 "block_sizing": "magic"}}])
