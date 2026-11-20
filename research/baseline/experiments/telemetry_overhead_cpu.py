"""CPU upper bound on telemetry overhead: the executor simulation with tracing
off vs. on, with LLM calls made near-instant so scheduler and tracing code
dominate wall time (on a GPU, each call takes 10-1000x longer, so the
relative overhead there is smaller; E0 on Modal measures it directly).

    python -m research.baseline.experiments.telemetry_overhead_cpu \
        --out research/baseline/results/telemetry_overhead_cpu.json

Arms alternate off/on within each repeat, so drift affects both equally.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
from pathlib import Path

from research.baseline.experiments import executor_simulation as sim
from vllm.kalypso import telemetry


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--repeats", type=int, default=15)
    ap.add_argument("--configs", default="1x1,4x4")
    ap.add_argument("--selectivity", type=float, default=0.5)
    args = ap.parse_args(argv)
    simargs = argparse.Namespace(base_ms=0.0, per_item_ms=0.0, slots=64, kv_tokens=2_000_000)
    pass_rows = set(random.Random(0).sample(range(100), round(args.selectivity * 100)))
    result = {"selectivity": args.selectivity, "repeats": args.repeats,
              "llm_call_cost_s": 0.0, "configs": {}}
    for cfg in args.configs.split(","):
        kf, kj = (int(x) for x in cfg.split("x"))
        ops = sim.pipeline(kf, kj)
        times = {"off": [], "on": []}
        events_per_query = []
        for rep in range(-1, args.repeats):
            for mode in ("off", "on") if rep % 2 else ("on", "off"):
                events = []
                telemetry.configure(events if mode == "on" else None, run_id="overhead")
                try:
                    wall, _, _ = asyncio.run(sim._run_once(ops, 100, 32, pass_rows, simargs))
                finally:
                    telemetry.configure(None)
                if rep >= 0:
                    times[mode].append(wall)
                    if mode == "on":
                        events_per_query.append(len(events))
        off, on = statistics.median(times["off"]), statistics.median(times["on"])
        result["configs"][cfg] = {
            "off_s": times["off"], "on_s": times["on"],
            "off_median_s": off, "on_median_s": on,
            "overhead_rel": on / off - 1, "events_per_query": statistics.median(events_per_query),
        }
        print(f"{cfg}: off {off:.3f}s, on {on:.3f}s, overhead {100 * (on / off - 1):+.1f}%, "
              f"{statistics.median(events_per_query):g} events/query")
    args.out.write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
