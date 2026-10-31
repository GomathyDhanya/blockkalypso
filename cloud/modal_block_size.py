"""Run block-size experiments with the eval harness on a Modal GPU.

    modal run cloud/modal_block_size.py --workload movie_positive --n 500 --blocks 4,6,8
    modal run cloud/modal_block_size.py --workload medec --blocks 2,4 --model Qwen/Qwen2.5-14B-Instruct
    modal run cloud/modal_block_size.py --workload movie_positive --n 500 --controls
    modal volume get blockkalypso-results block-size/<out> ./vllm/kalypso/benchmark/eval/results/block-size/

The batching study's experiments (cloud/experiments.py) run from here too:

    modal run cloud/modal_block_size.py::status
    modal run --detach cloud/modal_block_size.py::experiments --phase single
    modal run cloud/modal_block_size.py::pull

Each run starts Kalypso on vLLM 0.29.0 inside one container and runs
``eval.run`` against it on localhost, so no network hop is in the latency
numbers. Arms are Kalypso's per-tuple operator (always, as the baseline) plus
``kalypso_block:<k>`` for every ``--blocks`` size (pairs per call for joins), or the
arms in ``--arms``. Arms are interleaved over the repeats, with the prefix
cache reset before every trial, so all block sizes in one run share a
container and can be compared directly.

Results go to the ``blockkalypso-results`` volume under ``block-size/<out>/``:
``run.jsonl`` (metadata line, then one line per trial with every metric and
each row's output) and ``summary.md`` (``eval.report``'s table). Model weights
are cached in ``blockkalypso-hf-cache``. Any workload in eval/workloads.py
works; the harness takes rows from this checkout.
"""

import contextlib
import importlib.util
import json
import os
import subprocess
import time
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parents[1]
SOURCE = "/opt/kalypso-source"
VLLM_PKG = ("$(python3 -c 'import importlib.util; print(next(iter(importlib.util"
            ".find_spec(\"vllm\").submodule_search_locations)))')")
DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_GPU = "L40S"
PORT = 8000
# vLLM 0.29.0's own defaults for an API server on a GPU under 70 GiB (L40S):
# set explicitly so the batch limits are known and fixed across GPUs.
MAX_NUM_SEQS = 256
MAX_NUM_BATCHED_TOKENS = 2048
BASELINE = "kalypso_per_tuple"
# One-row accuracy controls (eval/arms.py): prompt x sampling, no blocking.
CONTROLS = ("vllm_direct,vllm_direct:no_penalty,vllm_direct:block_prompt,"
            "vllm_direct:block_prompt+no_penalty")

image = (
    modal.Image.from_registry("vllm/vllm-openai:v0.29.0")
    .entrypoint([])
    # Modal looks for `python`; the vLLM image only ships `python3`.
    .run_commands("ln -sf \"$(command -v python3)\" /usr/local/bin/python")
    .add_local_dir(REPO / "vllm/kalypso", f"{SOURCE}/vllm/kalypso", copy=True,
                   ignore=["**/__pycache__", "benchmark/data", "benchmark/data/**",
                           "benchmark/eval/results", "benchmark/eval/results/**"])
    .add_local_file(REPO / "vllm/entrypoints/openai/api_server.py",
                    f"{SOURCE}/vllm/entrypoints/openai/api_server.py", copy=True)
    .add_local_dir(REPO / "docker/kalypso", f"{SOURCE}/docker/kalypso", copy=True)
    .run_commands(
        "python3 -c \"from importlib.metadata import version; "
        "assert version('vllm').split('+')[0] == '0.29.0', version('vllm')\"",
        f"python3 {SOURCE}/docker/kalypso/prepare.py --source {SOURCE} --target {VLLM_PKG}",
    )
    .env({"HF_HOME": "/cache/huggingface", "VLLM_SERVER_DEV_MODE": "1"})
)

app = modal.App("blockkalypso-block-size", image=image)
results = modal.Volume.from_name("blockkalypso-results", create_if_missing=True)
hf_cache = modal.Volume.from_name("blockkalypso-hf-cache", create_if_missing=True)


def _start_server(model: str, max_model_len: int, max_num_seqs: int = MAX_NUM_SEQS,
                  max_num_batched_tokens: int = MAX_NUM_BATCHED_TOKENS) -> subprocess.Popen:
    # Flags Kalypso's container init requires (docker/kalypso/init_api.py).
    return subprocess.Popen([
        "python3", "-m", "vllm.entrypoints.cli.main", "serve", model,
        "--host", "127.0.0.1", "--port", str(PORT),
        "--enable-prefix-caching", "--no-async-scheduling", "--api-server-count", "1",
        "--max-model-len", str(max_model_len),
        "--max-num-seqs", str(max_num_seqs),
        "--max-num-batched-tokens", str(max_num_batched_tokens),
    ])


def _wait_ready(server: subprocess.Popen, timeout_s: float = 1800) -> None:
    import urllib.request
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if server.poll() is not None:
            raise RuntimeError(f"server exited with code {server.returncode}")
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/models", timeout=5)
            return
        except OSError:
            time.sleep(5)
    raise TimeoutError("server did not become ready")


def _stamp(path: Path, **fields) -> None:
    """Add to the run's metadata line what the container cannot see itself:
    the local checkout's commit (there is no .git here), GPU and server limits."""
    lines = path.read_text(encoding="utf-8").splitlines()
    meta = json.loads(lines[0])
    meta.update(fields)
    path.write_text("\n".join([json.dumps(meta), *lines[1:]]) + "\n", encoding="utf-8")


def _gpu_name() -> str | None:
    try:
        return subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


@contextlib.contextmanager
def _server(model: str, max_model_len: int, max_num_seqs: int, max_num_batched_tokens: int):
    server = _start_server(model, max_model_len, max_num_seqs, max_num_batched_tokens)
    try:
        _wait_ready(server)
        hf_cache.commit()  # keep downloaded weights for the next run
        yield server
    finally:
        server.terminate()
        try:
            server.wait(timeout=60)
        except subprocess.TimeoutExpired:
            server.kill()


def _bench(server, arms: list[str], workload: str, n: int, warmup: int, repeats: int,
           out_dir: Path, stamp: dict) -> str:
    """Run eval.run against the local server; write run.jsonl and summary.md."""
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl = out_dir / "run.jsonl"
    try:
        subprocess.run([
            "python3", "-m", "vllm.kalypso.benchmark.eval.run",
            "--base-url", f"http://127.0.0.1:{PORT}", "--server-pid", str(server.pid),
            "--workload", workload, *(["--n", str(n)] if n else []),
            "--arms", ",".join(arms),
            "--warmup", str(warmup), "--repeats", str(repeats), "--out", str(jsonl),
        ], check=True, env=os.environ.copy())
        _stamp(jsonl, **stamp)
        table = subprocess.run(
            ["python3", "-m", "vllm.kalypso.benchmark.eval.report", str(jsonl),
             "--reference", BASELINE],
            check=True, capture_output=True, text=True).stdout
        (out_dir / "summary.md").write_text(table, encoding="utf-8")
        return table
    finally:
        results.commit()  # keep whatever finished


@app.function(gpu=DEFAULT_GPU, timeout=6 * 3600,
              volumes={"/results": results, "/cache": hf_cache})
def run(arms: list[str], workload: str, n: int, warmup: int, repeats: int, model: str,
        max_model_len: int, out: str, git_sha: str, git_dirty: bool,
        max_num_seqs: int = MAX_NUM_SEQS,
        max_num_batched_tokens: int = MAX_NUM_BATCHED_TOKENS) -> str:
    stamp = dict(git_sha=git_sha, git_dirty=git_dirty, gpu=_gpu_name(),
                 max_num_seqs=max_num_seqs, max_num_batched_tokens=max_num_batched_tokens,
                 max_model_len=max_model_len)
    with _server(model, max_model_len, max_num_seqs, max_num_batched_tokens) as server:
        return _bench(server, arms, workload, n, warmup, repeats, Path("/results") / out, stamp)


@app.function(gpu=DEFAULT_GPU, timeout=6 * 3600,
              volumes={"/results": results, "/cache": hf_cache})
def run_experiment(exp: dict, warmup: int, repeats: int, model: str, max_model_len: int,
                   git_sha: str, git_dirty: bool, max_num_seqs: int = MAX_NUM_SEQS,
                   max_num_batched_tokens: int = MAX_NUM_BATCHED_TOKENS) -> dict:
    """One experiment from cloud/experiments.py: its parts in turn on one server.

    A part that fails is recorded in the experiment's status.json and the
    next part still runs.
    """
    root = Path("/results") / "experiments" / exp["id"]
    stamp = dict(git_sha=git_sha, git_dirty=git_dirty, gpu=_gpu_name(),
                 max_num_seqs=max_num_seqs, max_num_batched_tokens=max_num_batched_tokens,
                 max_model_len=max_model_len, experiment=exp["id"], phase=exp["phase"],
                 purpose=exp["purpose"])
    status = {"id": exp["id"], "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "parts": {}}
    with _server(model, max_model_len, max_num_seqs, max_num_batched_tokens) as server:
        for p in exp["parts"]:
            arms = [BASELINE, *[a for a in p["arms"] if a != BASELINE]]
            try:
                _bench(server, arms, p["workload"], p["n"], warmup, repeats,
                       root / p["name"], {**stamp, "part": p["name"]})
                status["parts"][p["name"]] = "done"
            except subprocess.CalledProcessError as exc:
                status["parts"][p["name"]] = f"failed: {exc}"
    status["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    root.mkdir(parents=True, exist_ok=True)
    (root / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    results.commit()
    return status


def _git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True,
                          text=True).stdout.strip()


def _clean(text: str) -> str:
    # Terminals wrap long lines and the break can be pasted along: drop all
    # whitespace, so "vllm_direct:\n  block_prompt" still names one arm.
    return "".join(text.split())


@app.local_entrypoint()
def main(workload: str = "movie_positive", n: int = 0, blocks: str = "", arms: str = "",
         controls: bool = False, warmup: int = 1, repeats: int = 3,
         model: str = DEFAULT_MODEL, gpu: str = DEFAULT_GPU, max_model_len: int = 32768,
         out: str = ""):
    blocks, arms = _clean(blocks), _clean(CONTROLS if controls else arms)
    others = ([f"kalypso_block:{b}" for b in blocks.split(",") if b]
              + [a for a in arms.split(",") if a])
    if not others:
        raise SystemExit("give --blocks (e.g. 4,6,8; for joins pairs per call, e.g. 4,25), "
                         "--arms or --controls")
    # Kalypso builds a plain SemFilter for block_size 1, so kalypso_block:1 is
    # just per-tuple; use --controls for a one-row block.
    if "1" in blocks.split(","):
        raise SystemExit("block size 1 is Kalypso's per-tuple path; use --controls")
    all_arms = [BASELINE, *dict.fromkeys(a for a in others if a != BASELINE)]
    sha = _git("rev-parse", "HEAD")
    dirty = bool(_git("status", "--porcelain", "--untracked-files=no"))
    tag = "controls" if controls else ("b" + "-".join(blocks.split(",")) if blocks else "arms")
    out = out or f"{workload}{f'-n{n}' if n else ''}-{tag}-{model.split('/')[-1]}"
    out = out if out.startswith("block-size/") else f"block-size/{out}"
    print(f"model={model} gpu={gpu} workload={workload} n={n or 'all'} "
          f"arms={','.join(all_arms)} out={out} git={sha[:9]}{' (dirty)' if dirty else ''}")
    function = run if gpu == DEFAULT_GPU else run.with_options(gpu=gpu)
    print(function.remote(all_arms, workload, n, warmup, repeats, model, max_model_len,
                          out, sha, dirty))
    print(f"Download: modal volume get blockkalypso-results {out} "
          f"./vllm/kalypso/benchmark/eval/results/block-size/")


# --- The batching study: every experiment in cloud/experiments.py ------------

def _manifest():
    """cloud/experiments.py, loaded by path (it needs neither vllm nor modal)."""
    spec = importlib.util.spec_from_file_location("experiments", REPO / "cloud/experiments.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _remote_files(path: str) -> list[str]:
    """Files under ``path`` on the results volume; [] when it does not exist."""
    try:
        return [e.path for e in results.listdir(path, recursive=True)
                if e.type == modal.volume.FileEntryType.FILE]
    except Exception:  # noqa: BLE001 -- a missing path raises; treat it as empty
        return []


def _state(m, exp) -> tuple[str, str]:
    """(state, detail): blocked, done, partial or pending, from the volume."""
    reasons = m.missing(exp)
    if reasons:
        return "blocked", "needs " + "; ".join(reasons)
    files = _remote_files(f"experiments/{exp['id']}")
    if any(f.endswith("status.json") for f in files):
        local = all((m.RESULTS / "experiments" / exp["id"] / p["name"] / "summary.md").exists()
                    for p in exp["parts"])
        return "done", "downloaded" if local else "on the volume; run ::pull"
    if files:
        return "partial", "started (running, or stopped part-way)"
    return "pending", ""


@app.local_entrypoint()
def status(phase: str = ""):
    """What has run, what can run now, and what is blocked and why."""
    m = _manifest()
    for exp in m.select(phase):
        state, detail = _state(m, exp)
        parts = ", ".join(p["name"] for p in exp["parts"])
        print(f"{state:8} {exp['id']:34} [{parts}] {detail}")


@app.local_entrypoint()
def experiments(phase: str = "", only: str = "", rerun: bool = False, parallel: int = 8,
                warmup: int = 1, repeats: int = 3, model: str = DEFAULT_MODEL,
                gpu: str = DEFAULT_GPU, max_model_len: int = 32768):
    """Run every runnable experiment in ``--phase`` (all phases when empty) or
    the ids in ``--only``, ``--parallel`` containers at a time. Finished ones
    are skipped unless ``--rerun``; blocked ones are listed with the reason."""
    m = _manifest()
    todo = []
    for exp in m.select(phase, only):
        state, detail = _state(m, exp)
        if state == "blocked" or (state == "done" and not rerun):
            print(f"skip {exp['id']}: {state} {detail}")
            continue
        exp = {**exp, "parts": [{**p, "arms": m.arms_for(exp, p)} for p in exp["parts"]]}
        todo.append(exp)
    if not todo:
        print("nothing to run")
        return
    sha = _git("rev-parse", "HEAD")
    dirty = bool(_git("status", "--porcelain", "--untracked-files=no"))
    function = run_experiment if gpu == DEFAULT_GPU else run_experiment.with_options(gpu=gpu)
    print(f"running {len(todo)} experiment(s), {parallel} at a time, on {gpu}, "
          f"git={sha[:9]}{' (dirty)' if dirty else ''}")
    for start in range(0, len(todo), parallel):
        batch = todo[start:start + parallel]
        calls = [(exp, function.spawn(exp, warmup, repeats, model, max_model_len, sha, dirty))
                 for exp in batch]
        for exp, call in calls:
            try:
                done = call.get()
                print(f"{exp['id']}: {done['parts']}")
            except Exception as exc:  # noqa: BLE001 -- report and keep going
                print(f"{exp['id']}: FAILED {type(exc).__name__}: {exc}")
    print("Download: modal run cloud/modal_block_size.py::pull")


@app.local_entrypoint()
def pull(phase: str = "", only: str = ""):
    """Download finished experiments into eval/results/experiments/."""
    m = _manifest()
    for exp in m.select(phase, only):
        files = _remote_files(f"experiments/{exp['id']}")
        if not files:
            continue
        fetched = 0
        for path in files:
            local = m.RESULTS / path
            if local.exists() and not path.endswith("status.json"):
                continue
            local.parent.mkdir(parents=True, exist_ok=True)
            with local.open("wb") as f:
                for chunk in results.read_file(path):
                    f.write(chunk)
            fetched += 1
        print(f"{exp['id']}: {fetched} new file(s)")
    print("Analyze: python3 -m vllm.kalypso.benchmark.eval.analyze")
