"""Run block-size experiments with the eval harness on a Modal GPU.

    modal run cloud/modal_block_size.py --workload movie_positive --n 500 --blocks 4,6,8
    modal run cloud/modal_block_size.py --workload medec --blocks 2,4 --model Qwen/Qwen2.5-14B-Instruct
    modal run cloud/modal_block_size.py --workload movie_positive --n 500 --controls
    modal volume get blockkalypso-results block-size/<out> ./vllm/kalypso/benchmark/eval/results/block-size/

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


def _start_server(model: str, max_model_len: int) -> subprocess.Popen:
    # Flags Kalypso's container init requires (docker/kalypso/init_api.py).
    return subprocess.Popen([
        "python3", "-m", "vllm.entrypoints.cli.main", "serve", model,
        "--host", "127.0.0.1", "--port", str(PORT),
        "--enable-prefix-caching", "--no-async-scheduling", "--api-server-count", "1",
        "--max-model-len", str(max_model_len),
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


def _stamp_git(path: Path, sha: str, dirty: bool) -> None:
    """The container has no .git, so record the local checkout's commit."""
    lines = path.read_text(encoding="utf-8").splitlines()
    meta = json.loads(lines[0])
    meta.update(git_sha=sha, git_dirty=dirty)
    path.write_text("\n".join([json.dumps(meta), *lines[1:]]) + "\n", encoding="utf-8")


@app.function(gpu=DEFAULT_GPU, timeout=6 * 3600,
              volumes={"/results": results, "/cache": hf_cache})
def run(arms: list[str], workload: str, n: int, warmup: int, repeats: int, model: str,
        max_model_len: int, out: str, git_sha: str, git_dirty: bool) -> str:
    out_dir = Path("/results") / out
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl = out_dir / "run.jsonl"
    server = _start_server(model, max_model_len)
    try:
        _wait_ready(server)
        hf_cache.commit()  # keep downloaded weights for the next run
        try:
            subprocess.run([
                "python3", "-m", "vllm.kalypso.benchmark.eval.run",
                "--base-url", f"http://127.0.0.1:{PORT}", "--server-pid", str(server.pid),
                "--workload", workload, *(["--n", str(n)] if n else []),
                "--arms", ",".join(arms),
                "--warmup", str(warmup), "--repeats", str(repeats), "--out", str(jsonl),
            ], check=True, env=os.environ.copy())
            _stamp_git(jsonl, git_sha, git_dirty)
            table = subprocess.run(
                ["python3", "-m", "vllm.kalypso.benchmark.eval.report", str(jsonl),
                 "--reference", BASELINE],
                check=True, capture_output=True, text=True).stdout
            (out_dir / "summary.md").write_text(table, encoding="utf-8")
            return table
        finally:
            results.commit()  # keep whatever finished
    finally:
        server.terminate()
        try:
            server.wait(timeout=60)
        except subprocess.TimeoutExpired:
            server.kill()


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
