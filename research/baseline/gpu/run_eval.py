"""``eval.run`` with the research workloads registered.

    python -m research.baseline.gpu.run_eval --workload rpipe_s20 --arms ... --out ...

Same CLI and output as ``python -m vllm.kalypso.benchmark.eval.run``.
"""

import sys

from research.baseline import workloads  # noqa: F401  (registers rsel*, rpipe_*)
from research.phase2 import workloads as _phase2  # noqa: F401  (registers rpipe_s50_drift)
from vllm.kalypso.benchmark.eval import run

if __name__ == "__main__":
    # eval.run builds --workload choices from WORKLOADS at parse time, after
    # the import above, so the research names are accepted.
    sys.exit(run.main())
