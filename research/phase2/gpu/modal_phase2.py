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


# --- Comparison: static vs adaptive (P2-compare) ------------------------------
#
#     modal run --detach research/phase2/gpu/modal_phase2.py::launch_compare
#     modal run research/phase2/gpu/modal_phase2.py::status_compare
#     python -m research.phase2.compare_summary research/phase2/results/P2-compare
#
# Each part runs ONE arm, the cost model's default choice (choose_pipeline,
# exhaustive, unguarded, real Session 1 token counts; computed before the run
# and fixed here). It runs under two server configurations: static
# (KALYPSO_ADAPTIVE_STAGE unset, so the controller is off) and adaptive
# (controller on). The controller is a server-level setting, so the server
# restarts between configurations, in the order off, on, off, on. Each phase
# runs 1 warmup and 2 repeats per part, giving 4 measured repeats per
# configuration, interleaved against drift. The model, revision, GPU, plan seed
# (DEFAULT_PLAN_SEED 0), temperature and prefix-cache reset per trial are all
# the same.

COMPARE_PARTS = [
    # Drop then recover: rpipe_s50's own rows, negatives first (research/phase2/workloads.py).
    {"name": "rpipe_s50_drift", "workload": "rpipe_s50_drift", "n": 0, "arm": "kalypso_pipe:1,2"},
    # Drop with no recovery (2% true filter selectivity).
    {"name": "rpipe_s05", "workload": "rpipe_s05", "n": 0, "arm": "kalypso_pipe:1,2"},
    # No drift (48%, shuffled): the controller should stay static.
    {"name": "rpipe_s50", "workload": "rpipe_s50", "n": 0, "arm": "kalypso_pipe:1,2"},
    # Join first, managed filter after a block join (where the model's error concentrates).
    {"name": "pipe_join_filter_j62_f90", "workload": "pipe_join_filter_j62_f90", "n": 0,
     "arm": "kalypso_pipe:10,2"},
]
COMPARE = {"id": "P2-compare", "parts": COMPARE_PARTS, "warmup": 1, "repeats": 2,
           "phases": [("off-1", False), ("on-1", True), ("off-2", False), ("on-2", True)]}


@app.function(gpu=mb.GPU, timeout=4 * 3600,
              volumes={"/results": mb.results, "/cache": mb.hf_cache})
def run_compare(exp: dict, git_sha: str, git_dirty: bool) -> dict:
    root = Path("/results") / OUT_ROOT / exp["id"]
    root.mkdir(parents=True, exist_ok=True)
    status = {"id": exp["id"], "git_sha": git_sha, "git_dirty": git_dirty,
              "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "phases": {}}
    (root / "plan.json").write_text(json.dumps(exp, indent=2))
    try:
        mb._environment(root)
        # tokens.json for the catalog workloads (the drift one has rpipe_s50's rows).
        mb._tokens(root, [p for p in exp["parts"] if p["workload"] != "rpipe_s50_drift"])
        for tag, adaptive in exp["phases"]:
            if adaptive:
                os.environ["KALYPSO_ADAPTIVE_STAGE"] = "1"
            else:
                os.environ.pop("KALYPSO_ADAPTIVE_STAGE", None)
            pre = subprocess.run(["python3", "-c", PREFLIGHT], cwd=mb.SOURCE,
                                 capture_output=True, text=True)
            if adaptive and pre.returncode != 0:
                raise RuntimeError("preflight failed: " + (pre.stdout + pre.stderr)[-2000:])
            phase = status["phases"][tag] = {"adaptive": adaptive, "parts": {}}
            with mb._server(root, True, tag) as proc, mb._samplers(root, tag):
                for part in exp["parts"]:
                    try:
                        mb._eval(root, part, [part["arm"]], exp["warmup"], exp["repeats"],
                                 f"run-{tag}.jsonl", proc.pid)
                        phase["parts"][part["name"]] = "done"
                    except subprocess.CalledProcessError as exc:
                        phase["parts"][part["name"]] = f"failed: {exc}"
            (root / "status.json").write_text(json.dumps(status, indent=2))
            mb.results.commit()
    except Exception as exc:  # noqa: BLE001 -- record and return
        status["error"] = f"{type(exc).__name__}: {exc}"
    status["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    (root / "status.json").write_text(json.dumps(status, indent=2))
    mb.results.commit()
    return status


@app.local_entrypoint()
def launch_compare(wait: bool = False, run_id: str = ""):
    """``--run-id`` writes to research-phase2/<run_id>/ (default P2-compare)."""
    exp = {**COMPARE, "id": run_id or COMPARE["id"]}
    sha = mb._git("rev-parse", "HEAD")
    dirty = bool(mb._git("status", "--porcelain", "--untracked-files=no"))
    print(f"launching {exp['id']} at {sha[:9]}{' (dirty)' if dirty else ''}")
    call = run_compare.spawn(exp, sha, dirty)
    print(exp["id"], "spawned", call.object_id)
    if wait:
        print(json.dumps(call.get(), indent=2))


@app.local_entrypoint()
def status_compare(run_id: str = ""):
    rid = run_id or COMPARE["id"]
    try:
        data = b"".join(mb.results.read_file(f"{OUT_ROOT}/{rid}/status.json"))
        print(json.dumps(json.loads(data), indent=2))
    except Exception:  # noqa: BLE001
        print(rid, "no status yet")


# --- Join band-cap resample check (P2-bandcap-resample) -----------------------
#
#     modal run --detach research/phase2/gpu/modal_phase2.py::launch_bandcap
#     modal run research/phase2/gpu/modal_phase2.py::status_bandcap
#
# Measurement only. movie_fanout75/90 (the two workloads behind the band cap
# at selectivity >= 0.6) and 4 disjoint resamples of each
# (research/phase2/workloads.py), at per-tuple, block:4 and block:6. The
# controller is off (KALYPSO_ADAPTIVE_STAGE unset), so the sizes are fixed.
# Same model, revision, GPU, server settings and temperature 0 as the
# original single/join-fanout* runs. The originals come first as a
# reproduction control. 1 warmup and 2 repeats (greedy decoding, so the
# repeats are a determinism check).

BANDCAP_PARTS = [{"name": f"{b}{s}", "workload": f"{b}{s}", "n": 0}
                 for b in ("movie_fanout75", "movie_fanout90")
                 for s in ("", "_r1", "_r2", "_r3", "_r4")]
BANDCAP = {"id": "P2-bandcap-resample", "parts": BANDCAP_PARTS, "warmup": 1, "repeats": 2,
           "arms": [mb.BASELINE, "kalypso_block:4", "kalypso_block:6"]}


@app.function(gpu=mb.GPU, timeout=4 * 3600,
              volumes={"/results": mb.results, "/cache": mb.hf_cache})
def run_bandcap(exp: dict, git_sha: str, git_dirty: bool) -> dict:
    os.environ.pop("KALYPSO_ADAPTIVE_STAGE", None)
    root = Path("/results") / OUT_ROOT / exp["id"]
    root.mkdir(parents=True, exist_ok=True)
    status = {"id": exp["id"], "git_sha": git_sha, "git_dirty": git_dirty,
              "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "parts": {}}
    (root / "plan.json").write_text(json.dumps(exp, indent=2))
    try:
        mb._environment(root)
        with mb._server(root, False, "bandcap") as proc:
            for part in exp["parts"]:
                try:
                    mb._eval(root, part, exp["arms"], exp["warmup"], exp["repeats"],
                             "run.jsonl", proc.pid)
                    status["parts"][part["name"]] = "done"
                except subprocess.CalledProcessError as exc:
                    status["parts"][part["name"]] = f"failed: {exc}"
                (root / "status.json").write_text(json.dumps(status, indent=2))
                mb.results.commit()
    except Exception as exc:  # noqa: BLE001 -- record and return
        status["error"] = f"{type(exc).__name__}: {exc}"
    status["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    (root / "status.json").write_text(json.dumps(status, indent=2))
    mb.results.commit()
    return status


@app.local_entrypoint()
def launch_bandcap(wait: bool = False):
    sha = mb._git("rev-parse", "HEAD")
    dirty = bool(mb._git("status", "--porcelain", "--untracked-files=no"))
    print(f"launching {BANDCAP['id']} at {sha[:9]}{' (dirty)' if dirty else ''}")
    call = run_bandcap.spawn(BANDCAP, sha, dirty)
    print(BANDCAP["id"], "spawned", call.object_id)
    if wait:
        print(json.dumps(call.get(), indent=2))


@app.local_entrypoint()
def status_bandcap():
    try:
        data = b"".join(mb.results.read_file(f"{OUT_ROOT}/{BANDCAP['id']}/status.json"))
        print(json.dumps(json.loads(data), indent=2))
    except Exception:  # noqa: BLE001
        print(BANDCAP["id"], "no status yet")
