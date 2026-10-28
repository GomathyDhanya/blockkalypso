"""Serve a model with Kalypso on vLLM 0.29.0 on a Modal GPU, over HTTPS.

    modal deploy cloud/modal_serve.py
    modal app stop blockkalypso-serve

The image is the official vLLM 0.29.0 image with this checkout's Kalypso
package applied by docker/kalypso/prepare.py (as in docker/Dockerfile.kalypso).
Unlike an in-container eval run, the server is exposed at
``https://<workspace>--blockkalypso-serve-serve.<region>.modal.run``, so the
eval harness can drive it from elsewhere. VLLM_SERVER_DEV_MODE=1 enables
/reset_prefix_cache; /metrics is always on.

Requests need a workspace proxy auth token, sent as the Modal-Key and
Modal-Secret headers. The first request after idle waits for a cold start
(weight download into the ``blockkalypso-hf-cache`` volume, then vLLM start).
"""

import subprocess
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parents[1]
SOURCE = "/opt/kalypso-source"
VLLM_PKG = ("$(python3 -c 'import importlib.util; print(next(iter(importlib.util"
            ".find_spec(\"vllm\").submodule_search_locations)))')")
MODEL = "Qwen/Qwen2.5-7B-Instruct"
GPU = "L40S"
MAX_MODEL_LEN = 32768
PORT = 8000

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
    )
    .env({"HF_HOME": "/cache/huggingface", "VLLM_SERVER_DEV_MODE": "1"})
)

app = modal.App("blockkalypso-serve", image=image)
hf_cache = modal.Volume.from_name("blockkalypso-hf-cache", create_if_missing=True)


# One container only: the harness's /metrics deltas are server-wide, so every
# request must reach the same vLLM process. Scales to zero after 15 idle min.
@app.function(gpu=GPU, volumes={"/cache": hf_cache}, timeout=24 * 3600,
              max_containers=1, scaledown_window=15 * 60)
@modal.concurrent(max_inputs=256)
@modal.web_server(PORT, startup_timeout=30 * 60, requires_proxy_auth=True)
def serve():
    # Flags Kalypso's container init requires (docker/kalypso/init_api.py).
    subprocess.Popen([
        "python3", "-m", "vllm.entrypoints.cli.main", "serve", MODEL,
        "--host", "0.0.0.0", "--port", str(PORT),
        "--enable-prefix-caching", "--no-async-scheduling", "--api-server-count", "1",
        "--max-model-len", str(MAX_MODEL_LEN),
    ])
