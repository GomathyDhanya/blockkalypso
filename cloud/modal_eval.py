"""Run the evaluation suite on a Modal GPU.

    modal run cloud/modal_eval.py --workloads reviews --repeats 1 --warmup 0
    modal run cloud/modal_eval.py --experiment E1 --model Qwen/Qwen2.5-1.5B-Instruct \
        --max-model-len 32768 --out m2-E1
    modal run cloud/modal_eval.py --experiment E1 --model Qwen/Qwen2.5-7B-Instruct \
        --gpu L40S --max-model-len 32768 --out m2-E1-qwen7b
    modal run cloud/modal_eval.py --model Qwen/Qwen2.5-7B-Instruct \
        --workloads contractnli_fj,medec_full --out m2-qwen7b
    modal volume get blockkalypso-results m2-qwen7b ./results-m2-qwen7b

The image follows docker/Dockerfile.kalypso: the official vLLM 0.29.0 image
with this checkout's Kalypso package applied by docker/kalypso/prepare.py.
The full datasets are fetched while the image builds. In the container one
function starts the Kalypso server and runs eval.suite against it on
localhost, so no network hop is in the latency numbers; results go to the
``blockkalypso-results`` volume, model weights to ``blockkalypso-hf-cache``.

Note: this is Kalypso on vLLM 0.29.0. Local CPU runs use the repository's own
vLLM fork, so compare arms within one setup, not across the two.
"""

import subprocess
import time
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parents[1]
SOURCE = "/opt/kalypso-source"
VLLM_PKG = ("$(python3 -c 'import importlib.util; print(next(iter(importlib.util"
            ".find_spec(\"vllm\").submodule_search_locations)))')")

image = (
    modal.Image.from_registry("vllm/vllm-openai:v0.29.0")
    .entrypoint([])
    # Modal looks for `python`; the vLLM image only ships `python3`.
    .run_commands("ln -sf \"$(command -v python3)\" /usr/local/bin/python")
    .add_local_dir(REPO / "vllm/kalypso", f"{SOURCE}/vllm/kalypso", copy=True,
                   ignore=["**/__pycache__", "benchmark/data", "benchmark/data/**"])
    .add_local_file(REPO / "vllm/entrypoints/openai/api_server.py",
                    f"{SOURCE}/vllm/entrypoints/openai/api_server.py", copy=True)
    .add_local_dir(REPO / "docker/kalypso", f"{SOURCE}/docker/kalypso", copy=True)
    .run_commands(
        "python3 -c \"from importlib.metadata import version; "
        "assert version('vllm').split('+')[0] == '0.29.0', version('vllm')\"",
        f"python3 {SOURCE}/docker/kalypso/prepare.py --source {SOURCE} --target {VLLM_PKG}",
        "python3 -m vllm.kalypso.benchmark.eval.fetch_data",
    )
    .env({"HF_HOME": "/cache/huggingface", "VLLM_SERVER_DEV_MODE": "1"})
)

app = modal.App("blockkalypso-eval", image=image)
results = modal.Volume.from_name("blockkalypso-results", create_if_missing=True)
hf_cache = modal.Volume.from_name("blockkalypso-hf-cache", create_if_missing=True)

PORT = 8000


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


DEFAULT_GPU = "L4"


@app.function(gpu=DEFAULT_GPU, timeout=6 * 3600,
              volumes={"/results": results, "/cache": hf_cache})
def run_suite(model: str, workloads: str, out: str, warmup: int, repeats: int,
              max_model_len: int, git_sha: str, git_dirty: bool,
              experiment: str = "") -> str:
    """Run eval.experiments <experiment>, or eval.suite over ``workloads``."""
    import os
    server = _start_server(model, max_model_len)
    try:
        _wait_ready(server)
        hf_cache.commit()  # keep downloaded weights for the next run
        out_dir = Path("/results") / out
        target = (["vllm.kalypso.benchmark.eval.experiments", experiment] if experiment
                  else ["vllm.kalypso.benchmark.eval.suite", "--workloads", workloads])
        try:
            subprocess.run([
                "python3", "-m", *target,
                "--base-url", f"http://127.0.0.1:{PORT}", "--server-pid", str(server.pid),
                "--warmup", str(warmup), "--repeats", str(repeats),
                "--out-dir", str(out_dir),
            ], check=True, env={**os.environ, "EVAL_GIT_SHA": git_sha,
                                "EVAL_GIT_DIRTY": "1" if git_dirty else "0"})
        finally:
            results.commit()  # keep whatever finished, even if a later run failed
        summary = out_dir / "summary.md"
        return summary.read_text() if summary.exists() else ""
    finally:
        server.terminate()
        try:
            server.wait(timeout=60)
        except subprocess.TimeoutExpired:
            server.kill()


def _git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True,
                          text=True).stdout.strip()


@app.local_entrypoint()
def main(model: str = "Qwen/Qwen2.5-0.5B-Instruct", workloads: str = "reviews",
         experiment: str = "", out: str = "", warmup: int = 1, repeats: int = 3,
         max_model_len: int = 16384, gpu: str = DEFAULT_GPU):
    sha = _git("rev-parse", "HEAD")
    dirty = bool(_git("status", "--porcelain", "--untracked-files=no"))
    out = out or f"{time.strftime('%Y%m%d-%H%M%S')}-{model.split('/')[-1]}"
    what = f"experiment={experiment}" if experiment else f"workloads={workloads}"
    print(f"model={model} {what} gpu={gpu} out={out} git={sha[:9]}"
          f"{' (dirty)' if dirty else ''}")
    # A bigger model may need more GPU memory than the default L4 (24 GB).
    function = run_suite if gpu == DEFAULT_GPU else run_suite.with_options(gpu=gpu)
    summary = function.remote(model, workloads, out, warmup, repeats, max_model_len,
                              sha, dirty, experiment)
    print(summary)
    print(f"\nDownload: modal volume get blockkalypso-results {out} ./results-{out}")
