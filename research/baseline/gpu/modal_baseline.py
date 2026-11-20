"""Session 1 GPU baseline on Modal: the original system, measured with the
opt-in telemetry, GPU sampling and controlled background load.

    modal run --detach research/baseline/gpu/modal_baseline.py::launch            # every experiment
    modal run --detach research/baseline/gpu/modal_baseline.py::launch --only E1-s20,E3-load
    modal run research/baseline/gpu/modal_baseline.py::status
    modal volume get blockkalypso-results research-session1 ./research/baseline/results/gpu/

Same image, model and server settings as cloud/modal_block_size.py (vLLM
0.29.0 + the Kalypso overlay, Qwen2.5-7B-Instruct, L40S, max_num_seqs 256,
max_num_batched_tokens 2048, max_model_len 32768, prefix caching on and reset
before every trial). Differences, all additive:

- the server runs with ``KALYPSO_TRACE`` (per-query JSONL telemetry), except
  in the overhead experiment's "off" half;
- ``nvidia-smi`` is sampled every 0.5 s and vLLM's ``/metrics`` every 0.5 s,
  both with wall-clock timestamps, so they line up with the trace;
- the environment (package versions, CUDA, driver, model revision, vLLM's KV
  cache size) is written to ``env.json``;
- every workload's row token counts (the model's tokenizer) go to
  ``tokens.json``;
- 1 warmup and 5 repeats per arm (eval.run's interleaving is unchanged);
- the load experiment runs a separate background client at a fixed number of
  concurrent streams (each request a unique ~400-token prompt, 128 output
  tokens with ignore_eos) while the measured arms run.

Results: volume ``blockkalypso-results`` under ``research-session1/<id>/``.
"""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import modal

# The repo root on the launching machine. Inside a Modal container this file
# is /root/modal_baseline.py (no repo around it); the image steps that use
# REPO only run locally, so any placeholder path works there.
_HERE = Path(__file__).resolve()
REPO = _HERE.parents[3] if len(_HERE.parents) > 3 else _HERE.parent
SOURCE = "/opt/kalypso-source"
VLLM_PKG = ("$(python3 -c 'import importlib.util; print(next(iter(importlib.util"
            ".find_spec(\"vllm\").submodule_search_locations)))')")
MODEL = "Qwen/Qwen2.5-7B-Instruct"
GPU = "L40S"
PORT = 8000
MAX_NUM_SEQS = 256
MAX_NUM_BATCHED_TOKENS = 2048
MAX_MODEL_LEN = 32768
WARMUP, REPEATS = 1, 5
BASELINE = "kalypso_per_tuple"
OUT_ROOT = "research-session1"

image = (
    modal.Image.from_registry("vllm/vllm-openai:v0.29.0")
    .entrypoint([])
    .run_commands("ln -sf \"$(command -v python3)\" /usr/local/bin/python")
    .add_local_dir(REPO / "vllm/kalypso", f"{SOURCE}/vllm/kalypso", copy=True,
                   ignore=["**/__pycache__", "benchmark/data", "benchmark/data/**",
                           "benchmark/eval/results", "benchmark/eval/results/**"])
    .add_local_file(REPO / "vllm/entrypoints/openai/api_server.py",
                    f"{SOURCE}/vllm/entrypoints/openai/api_server.py", copy=True)
    .add_local_dir(REPO / "docker/kalypso", f"{SOURCE}/docker/kalypso", copy=True)
    .add_local_dir(REPO / "research", f"{SOURCE}/research", copy=True,
                   ignore=["**/__pycache__", "baseline/results", "baseline/results/**"])
    .run_commands(
        "python3 -c \"from importlib.metadata import version; "
        "assert version('vllm').split('+')[0] == '0.29.0', version('vllm')\"",
        f"python3 {SOURCE}/docker/kalypso/prepare.py --source {SOURCE} --target {VLLM_PKG}",
    )
    .env({"HF_HOME": "/cache/huggingface", "VLLM_SERVER_DEV_MODE": "1",
          "PYTHONPATH": SOURCE})
)

app = modal.App("blockkalypso-session1-baseline", image=image)
results = modal.Volume.from_name("blockkalypso-results", create_if_missing=True)
hf_cache = modal.Volume.from_name("blockkalypso-hf-cache", create_if_missing=True)


def pipe(*configs):
    return [f"kalypso_pipe:{','.join(str(k) for k in c)}" for c in configs]


def blocks(*ks):
    return [f"kalypso_block:{k}" for k in ks]


# --- The experiment plan (fixed before any run; see BASELINE_METHODOLOGY.md) --

SELECTIVITY_IDS = ("05", "20", "50", "80", "95")
EXPERIMENTS = [
    # E1: cardinality sensitivity. Filter alone (400 rows) and filter -> join
    # (100 left x 32 right) at 5/20/50/80/95% true selectivity.
    *[{"id": f"E1-s{s}", "parts": [
        {"name": f"rsel{s}", "workload": f"rsel{s}", "n": 0,
         "arms": [BASELINE, *blocks(4, 8)]},
        {"name": f"rpipe_s{s}", "workload": f"rpipe_s{s}", "n": 0,
         "arms": [BASELINE, *pipe((4, 1), (1, 4), (4, 4))]},
    ]} for s in SELECTIVITY_IDS],
    # E2: block-size sensitivity, including the boundaries the code accepts.
    {"id": "E2-filter", "parts": [
        {"name": "movie_positive-n1000", "workload": "movie_positive", "n": 1000,
         "arms": [BASELINE, *blocks(2, 4, 8, 16, 32, 64)]}]},
    {"id": "E2-join", "parts": [
        {"name": "movie_fanout50", "workload": "movie_fanout50", "n": 0,
         "arms": [BASELINE, *blocks(2, 4, 8, 16, 32)]}]},
    {"id": "E2-map", "parts": [
        {"name": "movie_map-n200", "workload": "movie_map", "n": 200,
         "arms": [BASELINE, *blocks(2, 4, 8)]}]},
    # E3: load sensitivity on one pipeline; only the background load varies.
    {"id": "E3-load", "load_levels": [0, 32, 128], "parts": [
        {"name": "rpipe_s50", "workload": "rpipe_s50", "n": 0,
         "arms": [BASELINE, *pipe((1, 4), (4, 4))]}]},
    # E3 v2 (rerun after E3's prefix-cache resets failed under load): the
    # cache is reset and the reset confirmed before the load starts, per trial.
    {"id": "E3-load-v2", "load_levels": [0, 32, 128], "load_protocol": "v2", "parts": [
        {"name": "rpipe_s50", "workload": "rpipe_s50", "n": 0,
         "arms": [BASELINE, *pipe((1, 4), (4, 4))]}]},
    # E0: telemetry overhead: same arms with KALYPSO_TRACE off, then on.
    {"id": "E0-overhead", "overhead": True, "parts": [
        {"name": "rsel50", "workload": "rsel50", "n": 0,
         "arms": [BASELINE, *blocks(8)]}]},
]


# --- Container side -------------------------------------------------------

def _sh(cmd) -> str | None:
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT,
                                       shell=isinstance(cmd, str)).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        return f"ERROR: {exc}"


def _environment(root: Path) -> dict:
    code = ("import json, importlib.metadata as m, torch, sys;"
            "print(json.dumps({p: (m.version(p) if p in {d.metadata['Name'] for d in m.distributions()} else None)"
            " for p in ['vllm','torch','transformers','tokenizers','numpy','httpx','huggingface-hub']}"
            " | {'torch_cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),"
            " 'python': sys.version}))")
    env = json.loads(_sh(["python3", "-c", code]) or "{}")
    env["nvidia_smi"] = _sh(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                             "--format=csv,noheader"])
    env["nvidia_smi_processes"] = _sh(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                                       "--format=csv,noheader"])
    ref = Path("/cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/refs/main")
    env["model"] = MODEL
    env["model_revision"] = ref.read_text().strip() if ref.exists() else None
    env["server"] = {"max_num_seqs": MAX_NUM_SEQS, "max_num_batched_tokens": MAX_NUM_BATCHED_TOKENS,
                     "max_model_len": MAX_MODEL_LEN, "prefix_caching": True,
                     "dtype": "auto (model config: bfloat16)", "gpu": GPU}
    env["seeds"] = {"VLLMExecutor.DEFAULT_SAMPLING.seed": 42, "workload SELECTIVITY_SEED": 0,
                    "temperature": 0.0}
    (root / "env.json").write_text(json.dumps(env, indent=2))
    return env


def _start_server(root: Path, trace: bool, tag: str):
    env = os.environ.copy()
    if trace:
        env["KALYPSO_TRACE"] = str(root / f"trace-{tag}.jsonl")
        env["KALYPSO_RUN_ID"] = f"{root.name}-{tag}"
    else:
        env.pop("KALYPSO_TRACE", None)
    log = open(root / f"server-{tag}.log", "w")
    proc = subprocess.Popen([
        "python3", "-m", "vllm.entrypoints.cli.main", "serve", MODEL,
        "--host", "127.0.0.1", "--port", str(PORT),
        "--enable-prefix-caching", "--no-async-scheduling", "--api-server-count", "1",
        "--max-model-len", str(MAX_MODEL_LEN), "--max-num-seqs", str(MAX_NUM_SEQS),
        "--max-num-batched-tokens", str(MAX_NUM_BATCHED_TOKENS),
    ], env=env, stdout=log, stderr=subprocess.STDOUT)
    return proc, log


def _wait_ready(proc, timeout_s=1800):
    import urllib.request
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited with code {proc.returncode}")
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/models", timeout=5)
            return
        except OSError:
            time.sleep(5)
    raise TimeoutError("server did not become ready")


@contextlib.contextmanager
def _server(root: Path, trace: bool, tag: str):
    proc, log = _start_server(root, trace, tag)
    try:
        _wait_ready(proc)
        hf_cache.commit()
        yield proc
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()


@contextlib.contextmanager
def _samplers(root: Path, tag: str):
    """nvidia-smi and /metrics, every 0.5 s, wall-clock stamped."""
    gpu_csv = open(root / f"gpu-{tag}.csv", "w")
    gpu_csv.write(f"# t_wall_start={time.time()}\n")
    gpu_csv.flush()
    smi = subprocess.Popen(
        ["nvidia-smi", "--query-gpu=timestamp,utilization.gpu,utilization.memory,"
         "memory.used,memory.total,power.draw", "--format=csv,noheader,nounits",
         "-lms", "500"], stdout=gpu_csv, stderr=subprocess.DEVNULL)
    stop = threading.Event()

    def metrics_loop():
        import urllib.request
        from vllm.kalypso.benchmark.eval.metrics import parse_metrics
        with open(root / f"vllm-metrics-{tag}.jsonl", "w") as f:
            while not stop.is_set():
                try:
                    text = urllib.request.urlopen(f"http://127.0.0.1:{PORT}/metrics",
                                                  timeout=5).read().decode()
                    m = parse_metrics(text)
                    f.write(json.dumps({
                        "t_wall": time.time(), "kv_cache_pct": m["kv_cache_pct"],
                        "running": m["running"], "waiting": m["waiting"],
                        "preemptions": m["preemptions"], "prompt_tokens": m["prompt_tokens"],
                        "generation_tokens": m["generation_tokens"],
                        "prefix_cache_hits": m["prefix_cache_hits"],
                        "prefix_cache_queries": m["prefix_cache_queries"],
                        "requests": m["requests"]}) + "\n")
                    f.flush()
                except Exception:  # noqa: BLE001 -- keep sampling
                    pass
                stop.wait(0.5)

    thread = threading.Thread(target=metrics_loop, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=10)
        smi.terminate()
        gpu_csv.close()


def _tokens(root: Path, parts: list[dict]) -> None:
    code = f"""
import json, statistics, sys
sys.path.insert(0, {SOURCE!r})
from research.baseline import workloads  # registers research workloads
from vllm.kalypso.benchmark.eval.workloads import WORKLOADS
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained({MODEL!r})
out = {{}}
for name, n in {[(p["workload"], p["n"]) for p in parts]!r}:
    w = WORKLOADS[name](n or None)
    left = [len(x) for x in tok([r.text for r in w.rows], add_special_tokens=False)["input_ids"]]
    right = ([len(x) for x in tok([r.text for r in w.right_rows], add_special_tokens=False)["input_ids"]]
             if w.right_rows else [])
    out[f"{{name}}:{{n or ''}}"] = {{"left": left, "right": right,
        "left_mean": statistics.mean(left), "left_max": max(left),
        "right_mean": statistics.mean(right) if right else None}}
open({str(root / 'tokens.json')!r}, "w").write(json.dumps(out))
"""
    print(_sh(["python3", "-c", code]))


def _eval(root: Path, part: dict, arms, warmup, repeats, out_name, server_pid,
          extra=()) -> None:
    out = root / part["name"] / out_name
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        "python3", "-m", "research.baseline.gpu.run_eval",
        "--base-url", f"http://127.0.0.1:{PORT}", "--server-pid", str(server_pid),
        "--workload", part["workload"], *(["--n", str(part["n"])] if part["n"] else []),
        "--arms", ",".join(arms), "--warmup", str(warmup), "--repeats", str(repeats),
        "--out", str(out), *extra], check=True, cwd=SOURCE)
    table = _sh(["python3", "-m", "vllm.kalypso.benchmark.eval.report", str(out),
                 "--reference", BASELINE])
    out.with_suffix(".summary.md").write_text(table or "")
    results.commit()


LOAD_CLIENT = r'''
import asyncio, json, sys, time, random, httpx
streams, out = int(sys.argv[1]), sys.argv[2]
words = "the quick brown fox jumps over a lazy dog while reading long contracts about data".split()
async def stream(i, client, log):
    rng = random.Random(i)
    n = 0
    while True:
        prompt = f"[{i}-{n}-{time.time()}] " + " ".join(rng.choice(words) for _ in range(330))
        t = time.time()
        try:
            r = await client.post("http://127.0.0.1:8000/v1/chat/completions", json={
                "model": "Qwen/Qwen2.5-7B-Instruct", "max_tokens": 128, "ignore_eos": True,
                "temperature": 0.0, "messages": [{"role": "user", "content": "Summarize: " + prompt}]})
            ok = r.status_code
        except Exception as exc:
            ok = repr(exc)[:80]
        log.write(json.dumps({"t_start": t, "t_end": time.time(), "stream": i, "status": ok}) + "\n")
        log.flush()
        n += 1
async def main():
    async with httpx.AsyncClient(timeout=600) as client:
        with open(out, "a") as log:
            await asyncio.gather(*(stream(i, client, log) for i in range(streams)))
asyncio.run(main())
'''


@contextlib.contextmanager
def _load(root: Path, streams: int, tag: str):
    if streams <= 0:
        yield
        return
    proc = subprocess.Popen(["python3", "-c", LOAD_CLIENT, str(streams),
                             str(root / f"load-{tag}.jsonl")])
    time.sleep(20)  # let the background load reach steady state
    try:
        yield
    finally:
        proc.terminate()
        proc.wait(timeout=30)
        time.sleep(5)  # drain in-flight load requests before the next trial


RESET_OK = "Successfully reset prefix cache"


def _reset_verified(root: Path, tag: str, tries: int = 10) -> bool:
    """POST /reset_prefix_cache and confirm it in the server log.

    vLLM answers 200 even when it refuses the reset ("Failed to reset prefix
    cache because some blocks ... are not freed yet"), which invalidated the
    first E3 run; the server log is the only reliable signal.
    """
    import urllib.request
    log = root / f"server-{tag}.log"
    for _ in range(tries):
        before = log.read_text(errors="ignore").count(RESET_OK)
        req = urllib.request.Request(f"http://127.0.0.1:{PORT}/reset_prefix_cache",
                                     method="POST")
        urllib.request.urlopen(req, timeout=30).read()
        for _ in range(20):
            time.sleep(0.25)
            if log.read_text(errors="ignore").count(RESET_OK) > before:
                return True
        time.sleep(2)  # blocks still held by in-flight requests: wait and retry
    return False


def _run_load_v2(root: Path, part: dict, levels, tag: str, server_pid: int) -> None:
    """E3 protocol v2: per trial, with no background load running, reset the
    prefix cache and confirm it; then start the load at the level under test,
    let it reach steady state, run one arm once with eval.run's own reset
    disabled (it would fail under load), and stop the load. Every trial's
    reset outcome is logged to resets.jsonl; trials whose reset was not
    confirmed are kept but marked."""
    _eval(root, part, part["arms"], WARMUP, 0, "warmup.jsonl", server_pid)
    resets = open(root / "resets.jsonl", "a")
    for rep in range(1, REPEATS + 1):
        for level in levels:
            for arm in part["arms"]:
                ok = _reset_verified(root, tag)
                safe = arm.replace(":", "_").replace(",", "-")
                resets.write(json.dumps({"rep": rep, "level": level, "arm": arm,
                                         "reset_confirmed": ok, "t_wall": time.time()}) + "\n")
                resets.flush()
                with _load(root, level, f"L{level}-r{rep}-{safe}"):
                    _eval(root, part, [arm], 0, 1, f"run-L{level}-r{rep}-{safe}.jsonl",
                          server_pid, extra=("--allow-warm-cache",))
    resets.close()


@app.function(gpu=GPU, timeout=4 * 3600, volumes={"/results": results, "/cache": hf_cache})
def run_experiment(exp: dict, git_sha: str, git_dirty: bool) -> dict:
    root = Path("/results") / OUT_ROOT / exp["id"]
    root.mkdir(parents=True, exist_ok=True)
    status = {"id": exp["id"], "git_sha": git_sha, "git_dirty": git_dirty,
              "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "parts": {}}
    (root / "plan.json").write_text(json.dumps(exp, indent=2))
    try:
        _environment(root)
        _tokens(root, exp["parts"])
        phases = ([("off", False), ("on", True)] if exp.get("overhead") else [("on", True)])
        for tag, trace in phases:
            with _server(root, trace, tag) as proc, _samplers(root, tag):
                env = json.loads((root / "env.json").read_text())
                env["kv_cache_log"] = _sh(f"grep -iE 'KV cache|kv_cache|GPU blocks' "
                                          f"{root}/server-{tag}.log | head -20")
                (root / "env.json").write_text(json.dumps(env, indent=2))
                for part in exp["parts"]:
                    key = f"{part['name']}:{tag}"
                    try:
                        levels = exp.get("load_levels")
                        if levels and exp.get("load_protocol") == "v2":
                            _run_load_v2(root, part, levels, tag, proc.pid)
                        elif not levels:
                            _eval(root, part, part["arms"], WARMUP, REPEATS,
                                  f"run-{tag}.jsonl", proc.pid)
                        else:
                            # Warm up once without load, then each repeat runs
                            # every load level in turn (drift hits all levels).
                            _eval(root, part, part["arms"], WARMUP, 0, "warmup.jsonl", proc.pid)
                            for rep in range(1, REPEATS + 1):
                                for level in levels:
                                    with _load(root, level, f"L{level}-r{rep}"):
                                        _eval(root, part, part["arms"], 0, 1,
                                              f"run-L{level}-r{rep}.jsonl", proc.pid)
                        status["parts"][key] = "done"
                    except subprocess.CalledProcessError as exc:
                        status["parts"][key] = f"failed: {exc}"
    except Exception as exc:  # noqa: BLE001 -- record and return
        status["error"] = f"{type(exc).__name__}: {exc}"
    status["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    (root / "status.json").write_text(json.dumps(status, indent=2))
    results.commit()
    return status


# --- Local side -----------------------------------------------------------

def _git(*args):
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True,
                          text=True).stdout.strip()


@app.local_entrypoint()
def launch(only: str = "", wait: bool = False):
    """Spawn every experiment (or the ids in --only) in its own container.

    With ``modal run --detach`` and no ``--wait`` this returns as soon as all
    are spawned; the containers keep running. Check with ::status.
    """
    todo = [e for e in EXPERIMENTS if not only or e["id"] in only.split(",")]
    sha = _git("rev-parse", "HEAD")
    dirty = bool(_git("status", "--porcelain", "--untracked-files=no"))
    print(f"launching {len(todo)} experiment(s) at {sha[:9]}{' (dirty)' if dirty else ''}")
    calls = [(e, run_experiment.spawn(e, sha, dirty)) for e in todo]
    for e, call in calls:
        print(e["id"], "spawned", call.object_id)
    if wait:
        for e, call in calls:
            try:
                print(e["id"], json.dumps(call.get()))
            except Exception as exc:  # noqa: BLE001
                print(e["id"], "FAILED", type(exc).__name__, exc)
    print(f"Download: modal volume get blockkalypso-results {OUT_ROOT} "
          "./research/baseline/results/gpu/")


@app.local_entrypoint()
def status():
    for e in EXPERIMENTS:
        try:
            data = b"".join(results.read_file(f"{OUT_ROOT}/{e['id']}/status.json"))
            s = json.loads(data)
            print(e["id"], s.get("finished"), s.get("parts"), s.get("error", ""))
        except Exception:  # noqa: BLE001
            print(e["id"], "no status yet")
