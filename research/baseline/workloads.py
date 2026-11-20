"""Research workloads, registered into eval/workloads.py's tables at import.

Nothing in eval/workloads.py is edited: the loaders are the original
``load_movie_selectivity`` and ``load_pipe_filter_join``; this module only
adds new names to the dictionaries they read (``SELECTIVITIES``,
``PIPE_FILTER_JOIN``) and to ``WORKLOADS``.

Added (Session 1 cardinality sensitivity, target selectivity 5/20/50/80/95%):

- ``rsel05``, ``rsel20``, ``rsel50``, ``rsel80``, ``rsel95``: SemBench movie
  "clearly positive" filter, 400 rows, that share truly positive (same seed
  and sampling as movie_sel*). ``rsel50`` equals ``movie_sel50`` by
  construction.
- ``rpipe_s05`` ... ``rpipe_s95``: filter (that selectivity) -> join with 32
  Ant-Man reviews ("opposite sentiment"), 100 left rows by default, the shape
  of the existing pipe_filter_join_s*.

Target selectivity is a label share; achieved selectivity (rows the LLM
actually passes) is measured per arm and reported separately.
"""

from vllm.kalypso.benchmark.eval import workloads as W

RESEARCH_SELECTIVITIES = {"rsel05": 0.05, "rsel20": 0.20, "rsel50": 0.50,
                          "rsel80": 0.80, "rsel95": 0.95}
RESEARCH_PIPELINES = {f"rpipe_s{name[-2:]}": name for name in RESEARCH_SELECTIVITIES}


def register() -> list[str]:
    W.SELECTIVITIES.update(RESEARCH_SELECTIVITIES)
    W.PIPE_FILTER_JOIN.update(RESEARCH_PIPELINES)
    for name in RESEARCH_SELECTIVITIES:
        W.WORKLOADS[name] = (lambda n=None, name=name: W.load_movie_selectivity(name, n))
    for name in RESEARCH_PIPELINES:
        W.WORKLOADS[name] = (lambda n=None, name=name: W.load_pipe_filter_join(name, n))
    return sorted([*RESEARCH_SELECTIVITIES, *RESEARCH_PIPELINES])


NAMES = register()
