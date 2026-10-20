"""Level 3 block sizing: Trummer's cost model for block joins.

I. Trummer, "Implementing Semantic Join Operators Efficiently",
arXiv:2510.08489. A block join compares b1 left rows with b2 right rows in
one LLM call. With
    p   static prompt tokens,
    s1  tokens per left row,   s2 tokens per right row,
    s3  output tokens per matching pair,
    σ   join selectivity (fraction of pairs that match),
    g   weight of an output token relative to an input token,
    t   tokens available per call besides the static prompt,
    r1, r2  rows on each side,
the paper's model is
    tokens per call      p + b1*s1 + b2*s2 + b1*b2*σ*s3          (Lemma 4.1)
    total cost  c(b1, b2) = (r1/b1)(r2/b2)(p + b1*s1 + b2*s2
                                           + b1*b2*σ*s3*g)        (Cor. 4.4)
    subject to  b1*s1 + b2*s2 + b1*b2*σ*s3 <= t                   (Eq. 1)
and the optimum lies on the constraint (Thm. 5.2) with
    b2 = (t - b1*s1) / (s2 + b1*s3*σ)                             (Lemma 5.4)
    b1* = (-s1*s2 + sqrt(s1²s2² + s1*s2*s3*σ*t)) / (s1*s3*σ)       (Thm. 5.6)

Block sizes are integers no larger than the tables, so ``optimal_block``
searches the feasible integer grid for the lowest c; the closed form is
returned alongside for reference. Like the paper, the model counts tokens
only: it has no accuracy term and no prefill/decode distinction (that is
Level 4).
"""

import math
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class JoinCostInputs:
    p: float
    s1: float
    s2: float
    s3: float
    sigma: float
    t: float
    r1: int
    r2: int
    g: float = 1.0


@dataclass(frozen=True)
class BlockChoice:
    b1: int
    b2: int
    cost: float
    tokens_per_call: float
    calls: int
    b1_closed_form: float | None
    b2_closed_form: float | None

    def as_dict(self) -> dict:
        return asdict(self)


def tokens_per_call(x: JoinCostInputs, b1: int, b2: int) -> float:
    return x.p + b1 * x.s1 + b2 * x.s2 + b1 * b2 * x.sigma * x.s3


def cost(x: JoinCostInputs, b1: int, b2: int) -> float:
    """Total cost of the join, Corollary 4.4 (fractional batch counts)."""
    per_call = x.p + b1 * x.s1 + b2 * x.s2 + b1 * b2 * x.sigma * x.s3 * x.g
    return (x.r1 / b1) * (x.r2 / b2) * per_call


def feasible(x: JoinCostInputs, b1: int, b2: int) -> bool:
    """Equation 1: the call's variable tokens fit in the available budget."""
    return b1 * x.s1 + b2 * x.s2 + b1 * b2 * x.s3 * x.sigma <= x.t


def closed_form(x: JoinCostInputs) -> tuple[float | None, float | None]:
    """Theorem 5.6 and Lemma 5.4 (continuous); None when σ = 0."""
    if x.sigma <= 0 or x.s1 <= 0 or x.s3 <= 0:
        return None, None
    a = x.s1 * x.s2
    b1 = (-a + math.sqrt(a * a + x.s1 * x.s2 * x.s3 * x.sigma * x.t)) / (x.s1 * x.s3 * x.sigma)
    b2 = (x.t - b1 * x.s1) / (x.s2 + b1 * x.s3 * x.sigma)
    return b1, b2


def optimal_block(x: JoinCostInputs) -> BlockChoice:
    """Lowest-cost feasible integer (b1, b2) with 1 <= b1 <= r1, 1 <= b2 <= r2.

    Raises ValueError when not even a 1 x 1 block fits the budget.
    """
    best = None
    for b1 in range(1, x.r1 + 1):
        for b2 in range(1, x.r2 + 1):
            if not feasible(x, b1, b2):
                break                      # larger b2 only uses more tokens
            c = cost(x, b1, b2)
            if best is None or c < best[0]:
                best = (c, b1, b2)
    if best is None:
        raise ValueError("no block fits: a single pair exceeds the token budget")
    c, b1, b2 = best
    cf1, cf2 = closed_form(x)
    return BlockChoice(b1=b1, b2=b2, cost=c, tokens_per_call=tokens_per_call(x, b1, b2),
                       calls=math.ceil(x.r1 / b1) * math.ceil(x.r2 / b2),
                       b1_closed_form=cf1, b2_closed_form=cf2)
