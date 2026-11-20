"""Phase 2 functional run on Modal: the finished mechanisms, on, on real data.

This checks that the stack runs end to end and produces output. It is NOT a
comparison or a benchmark, and no latency, throughput or quality claim
should be drawn from it.

    modal run --detach research/phase2/gpu/modal_phase2.py::launch
    modal run research/phase2/gpu/modal_phase2.py::status
    modal volume get blockkalypso-results research-phase2 ./research/phase2/results/
    python -m research.phase2.functional_summary research/phase2/results/P2-functional

Same image, model, revision, GPU and server settings as the Session 1
baseline (research/baseline/gpu/modal_baseline.py, whose helpers this
reuses), with telemetry on. What is on:

- Adaptive controller: ``KALYPSO_ADAPTIVE_STAGE=1`` for the server, with
  power-of-two sizes, step-down on veto and selectivity-band join caps.
  Every arm with a block stage after the first is managed. ``costmodel.json``
  is added to the image, because the controller's veto reads it.
  Telemetry swallows subscriber errors, so the run checks first that the
  veto loads.
- KV feasibility check and decision cache: both live in the offline
  optimizer, which no query calls. So after a warmup the run reads
  Kalypso's real KV capacity from the trace (``kv_stage_registered``),
  converts it to tokens, and runs ``choose_pipeline(quality=True,
  kv_budget_tokens=...)`` on each workload's Pipeline. That choice is run as
  an extra arm, and everything goes into ``optimizer.json``. The cache uses
  its automatic activation. None of these workloads has 4+ stages, so it is
  expected not to engage; ``optimizer.json`` records its counters as
  evidence.
- Adaptive selectivity (step C): left off, its default.

Parts: two filter -> join pipelines (managed join stage, at a low and a
middling filter pass rate), one join -> filter and one map -> filter
(managed filter stage). Arms: per-tuple, a static (4, 4) block plan, and the
optimizer's choice. 1 warmup, 2 repeats. That is enough to show it runs, and
not meant to measure anything.

Results: volume ``blockkalypso-results`` under ``research-phase2/P2-functional/``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import modal

# Locally, put the repo root on the path for the baseline helpers. Inside the
# container they come from PYTHONPATH (the image's copy of the repo).
_HERE = Path(__file__).resolve()
if len(_HERE.parents) > 3 and (_HERE.parents[3] / "research/baseline").is_dir():
    sys.path.insert(0, str(_HERE.parents[3]))

from research.baseline.gpu import modal_baseline as mb  # noqa: E402

OUT_ROOT = "research-phase2"
REPEATS = 2
COSTMODEL_JSON = "vllm/kalypso/benchmark/eval/results/analysis/costmodel.json"
ITEM_TOKENS_JSON = "vllm/kalypso/benchmark/eval/results/analysis/item_tokens.json"
ANALYSIS = "kalypso/benchmark/eval/results/analysis"

image = (
    mb.image
    .add_local_file(mb.REPO / COSTMODEL_JSON, f"{mb.SOURCE}/{COSTMODEL_JSON}", copy=True)
    .add_local_file(mb.REPO / ITEM_TOKENS_JSON, f"{mb.SOURCE}/{ITEM_TOKENS_JSON}", copy=True)
    .run_commands(f"mkdir -p {mb.VLLM_PKG}/{ANALYSIS} && "
                  f"cp {mb.SOURCE}/{COSTMODEL_JSON} {mb.SOURCE}/{ITEM_TOKENS_JSON} "
                  f"{mb.VLLM_PKG}/{ANALYSIS}/")
)
app = modal.App("blockkalypso-phase2-functional", image=image)

PARTS = [
    {"name": "rpipe_s05", "workload": "rpipe_s05", "n": 0},
    {"name": "rpipe_s50", "workload": "rpipe_s50", "n": 0},
    {"name": "pipe_join_filter_j62_f90", "workload": "pipe_join_filter_j62_f90", "n": 0},
    {"name": "pipe_map_filter", "workload": "pipe_map_filter", "n": 0},
]
STATIC_ARMS = [mb.BASELINE, *mb.pipe((4, 4))]
EXPERIMENT = {"id": "P2-functional", "parts": PARTS, "static_arms": STATIC_ARMS,
              "repeats": REPEATS, "env": {"KALYPSO_ADAPTIVE_STAGE": "1"}}

PREFLIGHT = """
from vllm.kalypso.controller.adaptive_stage import AdaptiveStageConfig, costmodel_quality_veto
veto = costmodel_quality_veto()
assert AdaptiveStageConfig.from_env().enabled, "KALYPSO_ADAPTIVE_STAGE not set"
assert veto("join", 4, 100.0, 0.8) is False and veto("join", 4, 100.0, 0.3) is True
print("preflight ok: controller enabled, veto loads, band caps present")
"""



@app.function(gpu=mb.GPU, timeout=4 * 3600,
              volumes={"/results": mb.results, "/cache": mb.hf_cache})
def run_functional(exp: dict, git_sha: str, git_dirty: bool) -> dict:
    os.environ.update(exp["env"])
    root = Path("/results") / OUT_ROOT / exp["id"]
    root.mkdir(parents=True, exist_ok=True)
    status = {"id": exp["id"], "git_sha": git_sha, "git_dirty": git_dirty,
              "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "parts": {}}
    (root / "plan.json").write_text(json.dumps(exp, indent=2))
    try:
        mb._environment(root)
        mb._tokens(root, exp["parts"])
        pre = subprocess.run(["python3", "-c", PREFLIGHT], cwd=mb.SOURCE,
                             capture_output=True, text=True)
        status["preflight"] = (pre.stdout + pre.stderr).strip()[-2000:]
        if pre.returncode != 0:
            raise RuntimeError("preflight failed: " + status["preflight"])
        with mb._server(root, True, "on") as proc, mb._samplers(root, "on"):
            for part in exp["parts"]:  # warmup; also registers Kalypso's KV stages
                mb._eval(root, part, [mb.BASELINE], 1, 0, "warmup.jsonl", proc.pid)
            opt = subprocess.run(["python3", "-m", "research.phase2.optimizer_step", str(root),
                                  json.dumps(exp["parts"]), mb.MODEL, mb.MODEL_REVISION],
                                 cwd=mb.SOURCE, capture_output=True, text=True)
            status["optimizer"] = (opt.stdout + opt.stderr).strip()[-4000:]
            chosen = {}
            if opt.returncode == 0:
                chosen = {name: p["arm"] for name, p in
                          json.loads((root / "optimizer.json").read_text())["parts"].items()
                          if "arm" in p}
            for part in exp["parts"]:
                arms = list(dict.fromkeys(exp["static_arms"] + (
                    [chosen[part["name"]]] if part["name"] in chosen else [])))
                key = f"{part['name']}:on"
                try:
                    mb._eval(root, part, arms, 0, exp["repeats"], "run-on.jsonl", proc.pid)
                    status["parts"][key] = {"arms": arms, "result": "done"}
                except subprocess.CalledProcessError as exc:
                    status["parts"][key] = {"arms": arms, "result": f"failed: {exc}"}
        summary = subprocess.run(["python3", "-m", "research.phase2.functional_summary",
                                  str(root)], cwd=mb.SOURCE, capture_output=True, text=True)
        (root / "functional_summary.json").write_text(summary.stdout or summary.stderr)
    except Exception as exc:  # noqa: BLE001 -- record and return
        status["error"] = f"{type(exc).__name__}: {exc}"
    status["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    (root / "status.json").write_text(json.dumps(status, indent=2))
    mb.results.commit()
    return status


@app.local_entrypoint()
def launch(wait: bool = False):
    sha = mb._git("rev-parse", "HEAD")
    dirty = bool(mb._git("status", "--porcelain", "--untracked-files=no"))
    print(f"launching {EXPERIMENT['id']} at {sha[:9]}{' (dirty)' if dirty else ''}")
    call = run_functional.spawn(EXPERIMENT, sha, dirty)
    print(EXPERIMENT["id"], "spawned", call.object_id)
    if wait:
        print(json.dumps(call.get(), indent=2))
    print(f"Download: modal volume get blockkalypso-results {OUT_ROOT} ./research/phase2/results/")


@app.local_entrypoint()
def status():
    try:
        data = b"".join(mb.results.read_file(f"{OUT_ROOT}/{EXPERIMENT['id']}/status.json"))
        print(json.dumps(json.loads(data), indent=2))
    except Exception:  # noqa: BLE001
        print(EXPERIMENT["id"], "no status yet")
