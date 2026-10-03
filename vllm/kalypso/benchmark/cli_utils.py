import argparse


# DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-14B-Instruct"
# DEFAULT_MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
DEFAULT_MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct"
DEFAULT_PORT = 8003
DEFAULT_ROUTE = "/v1/semantic/query"


def parse_query_args() -> tuple[str, str]:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help="Model name to include with the request or local example.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help="Server port for the semantic query endpoint.",
    )
    parser.add_argument(
        "--endpoint",
        default=None,
        help="Optional full endpoint URL. Overrides --port when provided.",
    )
    args = parser.parse_args()
    endpoint = args.endpoint or f"http://localhost:{args.port}{DEFAULT_ROUTE}"
    return args.model_name, endpoint
