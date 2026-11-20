# Share of vLLM's per-GPU KV-cache budget given to Kalypso's reservation
# ledger. The fork's own server (vllm/entrypoints/openai/api_server.py,
# FORK_KV_BUDGET_FACTOR) uses 0.5; this overlay uses 1.0. Every Modal GPU
# result (historical and Session 1) ran with this overlay value. Not
# reconciled on purpose: see research/docs/ENVIRONMENT.md section 2.
KV_BUDGET_FACTOR = 1.0


async def init_kalypso(engine_client, state, args):
    from vllm.kalypso.execution.vllm_executor import VLLMExecutor

    config = engine_client.vllm_config
    if config.parallel_config.data_parallel_size != 1:
        raise ValueError("Kalypso currently requires data_parallel_size=1")
    if getattr(args, "api_server_count", 1) not in (None, 1):
        raise ValueError("Kalypso currently requires one API server process")
    if config.kv_transfer_config is not None:
        raise ValueError("Kalypso does not yet support KV transfer connectors")
    if config.scheduler_config.async_scheduling:
        raise ValueError("Kalypso currently requires --no-async-scheduling")
    if config.speculative_config is not None:
        raise ValueError("Kalypso speculative decoding has not been validated")
    if not config.cache_config.enable_prefix_caching:
        raise ValueError("Kalypso requires --enable-prefix-caching")
    per_gpu = await engine_client.engine_core.call_utility_async("get_kv_cache_budget")
    if per_gpu <= 0:
        raise ValueError("Kalypso requires a positive GPU KV-cache budget")
    state.query_processor = query_processor.QueryProcessor(
        model_name=config.model_config.model,
        budget=per_gpu * config.parallel_config.tensor_parallel_size * KV_BUDGET_FACTOR,
        # Tokenizer / KV-size lookups use the revision the engine serves.
        revision=config.model_config.revision,
        tokenizer_revision=config.model_config.tokenizer_revision,
    )
    # Tokenization uses the model path; API requests use its public alias.
    aliases = config.model_config.served_model_name
    model = aliases[0] if isinstance(aliases, list) else aliases
    state.query_processor.executor = VLLMExecutor(model or config.model_config.model)
    state.query_processor.start_stuck_monitor(engine_client)
