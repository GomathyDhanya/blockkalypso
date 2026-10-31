"""Server-free parts of the evaluation harness (benchmark/eval)."""

import json

import pytest

from vllm.kalypso.benchmark.eval import metrics, report, workloads
from vllm.kalypso.benchmark.eval.arms import (ArmError, check_arm, direct_requests,
                                              kalypso_ops, parse_arm,
                                              parse_query_response)
from vllm.kalypso.benchmark.eval.workloads import (BIODEX_ARTICLES, FILTER, JOIN,
                                                   load_biodex, load_join5x5,
                                                   load_medec, load_reviews)
from vllm.kalypso.data import data
from vllm.kalypso.sem_ops.filters import SemFilter
from vllm.kalypso.sem_ops.joins import CartesianProduct
from vllm.kalypso.sem_ops.map import SemMap

METRICS_TEXT = """\
# HELP vllm:request_success_total Count of successfully processed requests.
# TYPE vllm:request_success_total counter
vllm:request_success_total{engine="0",finished_reason="stop",model_name="m"} 7.0
vllm:request_success_total{engine="0",finished_reason="length",model_name="m"} 2.0
vllm:prompt_tokens_total{engine="0",model_name="m"} 1000.0
vllm:generation_tokens_total{engine="0",model_name="m"} 50.0
vllm:prefix_cache_hits_total{engine="0",model_name="m"} 300.0
vllm:prefix_cache_queries_total{engine="0",model_name="m"} 1000.0
vllm:kv_cache_usage_perc{engine="0",model_name="m"} 0.125
"""


def test_parse_metrics_sums_and_keeps_reasons():
    parsed = metrics.parse_metrics(METRICS_TEXT)
    assert parsed["requests"] == {"stop": 7.0, "length": 2.0}
    assert parsed["prompt_tokens"] == 1000.0
    assert parsed["kv_cache_pct"] == 12.5


def test_delta_counts_requests_truncation_and_hit_rate():
    before = metrics.parse_metrics(METRICS_TEXT)
    after = metrics.parse_metrics(
        METRICS_TEXT.replace('"stop",model_name="m"} 7.0', '"stop",model_name="m"} 10.0')
        .replace('"length",model_name="m"} 2.0', '"length",model_name="m"} 3.0')
        .replace("prompt_tokens_total{engine=\"0\",model_name=\"m\"} 1000.0",
                 "prompt_tokens_total{engine=\"0\",model_name=\"m\"} 1400.0")
        .replace("hits_total{engine=\"0\",model_name=\"m\"} 300.0",
                 "hits_total{engine=\"0\",model_name=\"m\"} 400.0")
        .replace("queries_total{engine=\"0\",model_name=\"m\"} 1000.0",
                 "queries_total{engine=\"0\",model_name=\"m\"} 1400.0"))
    d = metrics.delta(before, after)
    assert d["llm_requests"] == 4
    assert d["requests_by_reason"] == {"length": 1, "stop": 3}
    assert d["truncated"] == 1
    assert d["prompt_tokens"] == 400
    assert d["prefix_cache_hit_rate"] == 0.25


def test_delta_hit_rate_is_none_without_queries():
    snap = metrics.parse_metrics(METRICS_TEXT)
    assert metrics.delta(snap, snap)["prefix_cache_hit_rate"] is None


def test_biodex_subset_matches_kalypso_row_order(tmp_path):
    full = load_biodex()
    assert full.data_path == BIODEX_ARTICLES.resolve()
    assert [r.idx for r in full.rows] == sorted(r.idx for r in full.rows)
    sub = load_biodex(3)
    assert [r.idx for r in sub.rows] == [1, 2, 3]
    assert sub.data_path != full.data_path
    assert sorted(p.name for p in sub.data_path.glob("*.txt")) == ["1.txt", "2.txt", "3.txt"]


def _trial(arm, repeat, outputs, latency, selected=None):
    return {"type": "trial", "arm": arm, "repeat": repeat, "outputs": outputs,
            "selected": selected,
            "latency_s": latency, "llm_requests": 2, "prompt_tokens": 10,
            "generation_tokens": 4, "prefix_cache_hit_rate": None,
            "peak_kv_cache_pct": 5.0, "peak_server_rss_kb": 2048,
            "rows_per_s": 1.0, "truncated": 0}


def _meta(**extra):
    return {"type": "meta", "model": "m", "cpu": "c", "platform": "p",
            "workload": "w", "kind": "map", "rows": 2, "right_rows": None,
            "positives": None, "repeats": 2, "warmup": 1,
            "prefix_cache_reset": True, "git_sha": "abcdef0123", "git_dirty": False,
            "arms": ["kalypso_per_tuple", "vllm_direct"], "server": {}, **extra}


def test_report_map_summary_and_agreement(tmp_path):
    meta = _meta()
    trials = [
        _trial("kalypso_per_tuple", 1, {"1": "a", "2": "b"}, 4.0),
        _trial("vllm_direct", 1, {"1": "a", "2": "x"}, 2.0),
        _trial("kalypso_per_tuple", 2, {"1": "a", "2": "b"}, 6.0),
        _trial("vllm_direct", 2, {"1": "a", "2": "b"}, 3.0),
    ]
    summary = report.summarize(meta, trials)
    assert summary["kalypso_per_tuple"]["latency_s"] == {"median": 5.0, "min": 4.0, "max": 6.0}
    assert summary["vllm_direct"]["peak_server_rss_kb"]["median"] == 2.0
    assert summary["vllm_direct"]["prefix_cache_hit_rate"] is None
    assert summary["vllm_direct"]["accuracy"] is None

    agree = report.agreement(meta, trials, "kalypso_per_tuple")
    assert agree["kalypso_per_tuple"] == {"vs_reference": 1.0, "self": 1.0}
    assert agree["vllm_direct"] == {"vs_reference": 0.75, "self": 0.5}

    path = tmp_path / "r.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in [meta, *trials]) + "\n")
    loaded_meta, loaded = report.load(path)
    table = report.to_markdown(loaded_meta, report.summarize(loaded_meta, loaded),
                               report.agreement(loaded_meta, loaded, "kalypso_per_tuple"),
                               "kalypso_per_tuple")
    assert "| `vllm_direct` | 2.50 [2.00-3.00] |" in table
    assert "Accuracy" not in table
    assert table.rstrip().endswith("| 75% | 50% |")


def test_report_filter_quality_against_labels():
    meta = _meta(kind="filter", rows=4, positives=[0, 1],
                 arms=["kalypso_per_tuple", "kalypso_block:2"])
    trials = [
        _trial("kalypso_per_tuple", 1, {}, 4.0, selected=[0, 1]),
        _trial("kalypso_block:2", 1, {}, 1.0, selected=[0, 2]),
    ]
    q = report.quality(meta, trials[1])
    assert (q["tp"], q["fp"], q["fn"], q["tn"]) == (1, 1, 1, 1)
    assert q["accuracy"] == 0.5 and q["precision"] == 0.5 and q["recall"] == 0.5
    assert q["f1"] == 0.5
    agree = report.agreement(meta, trials, "kalypso_per_tuple")
    assert agree["kalypso_block:2"]["vs_reference"] == 0.5
    table = report.to_markdown(meta, report.summarize(meta, trials), agree,
                               "kalypso_per_tuple")
    assert "4 rows, 2 positive" in table
    assert "| 1.00 | 1.00 | 1.00 | 1.00 | 100% |" in table


def test_report_join_quality_counts_every_pair():
    meta = _meta(kind="join", rows=2, right_rows=3, positives=[[0, 2], [1, 0]])
    trial = _trial("kalypso_block:6", 1, {}, 1.0, selected=[[0, 2]])
    q = report.quality(meta, trial)
    assert (q["tp"], q["fp"], q["fn"], q["tn"]) == (1, 0, 1, 4)
    assert q["recall"] == 0.5 and q["precision"] == 1.0
    assert report.quality(meta, _trial("x", 1, {}, 1.0, selected=[]))["f1"] is None


def test_single_row_calls_from_max_tokens_histogram():
    """Block arms' fallback calls are the requests with max_tokens <= 5."""
    def text(small, total):
        lines = []
        for engine, share in (("0", small // 2), ("1", small - small // 2)):
            for le, count in (("1.0", 0), ("2.0", 0), ("5.0", share), ("10.0", share),
                              ("20.0", share), ("50.0", share + total), ("+Inf", share + total)):
                lines.append(f'vllm:request_params_max_tokens_bucket{{engine="{engine}",'
                             f'le="{le}",model_name="m"}} {count}.0')
        return METRICS_TEXT + "\n".join(lines) + "\n"

    before, after = metrics.parse_metrics(text(2, 10)), metrics.parse_metrics(text(7, 40))
    assert before["single_row_calls"] == 2 and after["single_row_calls"] == 7
    assert metrics.delta(before, after)["single_row_calls"] == 5
    assert metrics.parse_metrics(METRICS_TEXT)["single_row_calls"] == 0


def test_filter_and_join_workloads_carry_labels():
    reviews = load_reviews()
    assert reviews.kind == FILTER and len(reviews.rows) == 20
    assert reviews.rows[0].idx == 0
    assert reviews.positives and reviews.positives < {r.idx for r in reviews.rows}
    medec = load_medec()
    assert len(medec.positives) == 7 and len(medec.rows) == 10
    join = load_join5x5()
    assert join.kind == JOIN and len(join.positives) == 5
    assert all(0 <= l < 5 and 0 <= r < 5 for l, r in join.positives)


SEMBENCH_FIXTURE = """\
id,reviewId,reviewText,scoreSentiment
taken_3,1,Tense&#44; fun.,POSITIVE
taken_3,2,Dull.,NEGATIVE
ant_man_and_the_wasp_quantumania,3,Loved it.,POSITIVE
ant_man_and_the_wasp_quantumania,3,Loved it.,POSITIVE
ant_man_and_the_wasp_quantumania,4,A mess.,NEGATIVE
ant_man_and_the_wasp_quantumania,5,Great fun.,POSITIVE
"""


def test_sembench_movie_workloads(tmp_path, monkeypatch, fake_kv_manager):
    monkeypatch.setattr(workloads, "SEMBENCH_MOVIE", tmp_path)
    (tmp_path / "Reviews.csv").write_text(SEMBENCH_FIXTURE, encoding="utf-8")

    taken3 = workloads.load_movie_taken3()
    assert taken3.kind == FILTER and [r.text for r in taken3.rows] == ["Tense, fun.", "Dull."]
    assert taken3.positives == {0}
    ctxs = list(data._data_source(None, str(taken3.data_path), None))
    assert [c.state.idx for c in ctxs] == [0, 1]
    assert "Tense, fun." in str(ctxs[0].input.data)

    every = workloads.load_movie_positive()
    assert len(every.rows) == 5 and every.positives == {0, 2, 4}  # duplicate id 3 dropped
    assert len(workloads.load_movie_positive(2).rows) == 2

    join = workloads.load_movie_opposite()
    assert join.kind == JOIN and join.data_path == join.right_path
    assert len(join.rows) == 3
    assert join.positives == {(0, 1), (1, 0), (1, 2), (2, 1)}
    with pytest.raises(ValueError):
        workloads.load_movie_opposite(4)


def test_movie_selectivity_workloads(tmp_path, monkeypatch, fake_kv_manager):
    monkeypatch.setattr(workloads, "SEMBENCH_MOVIE", tmp_path)
    monkeypatch.setattr(workloads, "SELECTIVITY_ROWS", 10)
    rows = ["id,reviewId,reviewText,scoreSentiment"]
    rows += [f"m,p{i},good {i},POSITIVE" for i in range(12)]
    rows += [f"m,n{i},bad {i},NEGATIVE" for i in range(12)]
    (tmp_path / "Reviews.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")

    low, high = workloads.WORKLOADS["movie_sel10"](), workloads.WORKLOADS["movie_sel90"]()
    assert len(low.rows) == len(high.rows) == 10
    assert len(low.positives) == 1 and len(high.positives) == 9
    assert all(("good" in r.text) == (r.idx in high.positives) for r in high.rows)
    again = workloads.load_movie_selectivity("movie_sel10")
    assert [r.text for r in again.rows] == [r.text for r in low.rows]  # fixed seed
    assert len(list(data._data_source(None, str(low.data_path), None))) == 10
    assert len(workloads.WORKLOADS["movie_sel50"](4).rows) == 4


def test_movie_fanout_workloads(tmp_path, monkeypatch, fake_kv_manager):
    monkeypatch.setattr(workloads, "SEMBENCH_MOVIE", tmp_path)
    monkeypatch.setattr(workloads, "FANOUT_SIDE", 10)
    rows = ["id,reviewId,reviewText,scoreSentiment"]
    rows += [f"m,p{i},good {i},POSITIVE" for i in range(20)]
    rows += [f"m,n{i},bad {i},NEGATIVE" for i in range(10)]
    (tmp_path / "Reviews.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")

    for name, fanout in (("movie_fanout10", 1), ("movie_fanout90", 9)):
        join = workloads.WORKLOADS[name]()
        assert join.kind == JOIN and len(join.rows) == len(join.right_rows) == 10
        assert all("good" in r.text for r in join.rows)
        assert len(join.positives) == 10 * fanout
        for left in join.rows:  # every left row matches the same right rows
            assert {j for i, j in join.positives if i == left.idx} == {
                r.idx for r in join.right_rows if "bad" in r.text}
        assert not {r.text for r in join.rows} & {r.text for r in join.right_rows}
        right = list(data._data_source(None, str(join.right_path), None))
        assert len(right) == 10 and join.data_path != join.right_path
    again = workloads.load_movie_fanout("movie_fanout90")
    assert [r.text for r in again.right_rows] == [r.text for r in join.right_rows]  # fixed seed
    assert len(workloads.WORKLOADS["movie_fanout50"](4).positives) == 4 * 2
    with pytest.raises(ValueError):
        workloads.WORKLOADS["movie_fanout90"](12)  # needs 11 negatives, has 10


def test_parse_and_check_arms():
    assert parse_arm("kalypso_block:10").block == (10,)
    for bad in ("kalypso_block", "kalypso_block:0", "kalypso_block:2x5", "kalypso_block:axb", "nope"):
        with pytest.raises(ValueError):
            parse_arm(bad)
    with pytest.raises(ValueError):
        check_arm(parse_arm("kalypso_block:5"), load_biodex(1))
    check_arm(parse_arm("kalypso_block:5"), load_join5x5())


def test_query_ops_and_join_response_parsing():
    join = load_join5x5()
    ops = kalypso_ops(join, (10,))
    assert ops == [{"op": "join", "args": {
        "instruction": join.instruction, "right_table": str(join.right_path),
        "pair_block_size": 10}}]
    assert "block_size" not in kalypso_ops(load_reviews())[0]["args"]
    out = parse_query_response(join, {"results": [
        {"idx": 0, "right_idx": 2, "outputs": [{"op": "f", "value": "true"}]}]})
    assert out.selected == {(0, 2)} and out.outputs == {"0,2": "true"}
    with pytest.raises(ArmError):
        parse_query_response(join, {"results": [{"idx": 0, "outputs": []}]})


def test_direct_prompts_match_kalypso_operators(fake_kv_manager):
    """The direct arm must send exactly what Kalypso's per-tuple ops send."""
    filt = load_reviews()
    ctxs = list(data._data_source(None, str(filt.data_path), None))
    direct = direct_requests(filt)
    for ctx, (_, messages, max_tokens) in zip(ctxs, direct):
        assert SemFilter(filt.instruction)._build_prompts(ctx)[1] == messages
        assert max_tokens == SemFilter(filt.instruction).max_tokens

    join = load_join5x5()
    left = list(data._data_source(None, str(join.data_path), None))
    right = list(data._data_source(None, str(join.right_path), None))
    pairs = [p for ctx in left for p in CartesianProduct(right)(ctx)]
    by_key = {key: messages for key, messages, _ in direct_requests(join)}
    assert len(pairs) == len(by_key) == 25
    for pair in pairs:
        key = f"{pair.state.idx},{pair.state.right_idx}"
        assert SemFilter(join.instruction)._build_prompts(pair)[1] == by_key[key]

    biodex = load_biodex(2)
    ctxs = list(data._data_source(None, str(biodex.data_path), None))
    for ctx, (_, messages, _) in zip(ctxs, direct_requests(biodex)):
        assert SemMap(biodex.instruction)._build_prompt(ctx) == messages


def test_direct_controls_match_block_filter_one_row():
    """vllm_direct:block_prompt+no_penalty sends what BlockSemFilter does for one row."""
    import asyncio
    from types import SimpleNamespace

    from vllm.kalypso.benchmark.eval.arms import block_prompt_requests, direct_sampling
    from vllm.kalypso.execution.vllm_executor import VLLMExecutor
    from vllm.kalypso.sem_ops.block_filter import BlockSemFilter

    sent = []

    class FakeExecutor:
        async def execute(self, **kwargs):
            sent.append(kwargs)
            return SimpleNamespace(text="YES", finish_reason="stop")

    filt = load_reviews()
    op = BlockSemFilter(filt.instruction, block_size=2)
    result = asyncio.run(op.evaluate_block(FakeExecutor(), None, [filt.rows[0].text]))
    assert result.verdicts == [True] and len(sent) == 1
    key, messages, max_tokens = block_prompt_requests(filt)[0]
    assert key == "0" and messages == sent[0]["prompt"]
    assert max_tokens == sent[0]["max_tokens"]
    assert direct_sampling(frozenset({"no_penalty"})) == {
        **VLLMExecutor.DEFAULT_SAMPLING, **sent[0]["sampling"]}
    assert direct_sampling() == VLLMExecutor.DEFAULT_SAMPLING

    spec = parse_arm("vllm_direct:block_prompt+no_penalty")
    assert spec.kind == "vllm_direct" and spec.options == {"block_prompt", "no_penalty"}
    check_arm(spec, filt)
    for bad in ("vllm_direct:nope", "vllm_direct:no_penalty+x"):
        with pytest.raises(ValueError):
            parse_arm(bad)
    with pytest.raises(ValueError):
        check_arm(parse_arm("vllm_direct:no_penalty"), load_join5x5())


def test_report_lists_failed_arms():
    meta = _meta(arms=["kalypso_per_tuple", "kalypso_block:10"])
    trials = [
        _trial("kalypso_per_tuple", 1, {"1": "a"}, 1.0),
        {"type": "trial", "arm": "kalypso_block:10", "repeat": 1,
         "error": "ArmError: /v1/semantic/query returned 500"},
    ]
    failed = report.failures(trials)
    assert failed == {"kalypso_block:10": ["ArmError: /v1/semantic/query returned 500"]}
    table = report.to_markdown(meta, report.summarize(meta, trials),
                               report.agreement(meta, trials, "kalypso_per_tuple"),
                               "kalypso_per_tuple", failed)
    assert "| `kalypso_block:10` | failed in all 1 trials |" in table
    assert "returned 500" in table


def test_requests_by_max_tokens_and_gauges():
    def text(buckets, running=0):
        lines = [f'vllm:request_params_max_tokens_bucket{{engine="0",le="{le}",model_name="m"}} '
                 f'{count}.0' for le, count in buckets]
        lines.append(f'vllm:num_requests_running{{engine="0",model_name="m"}} {running}.0')
        lines.append('vllm:num_preemptions_total{engine="0",model_name="m"} 1.0')
        return METRICS_TEXT + "\n".join(lines) + "\n"

    before = metrics.parse_metrics(text([("5.0", 1), ("10.0", 1), ("50.0", 1), ("+Inf", 1)]))
    after = metrics.parse_metrics(text([("5.0", 4), ("10.0", 504), ("50.0", 530),
                                        ("+Inf", 532)], running=7))
    assert after["running"] == 7
    assert metrics.requests_by_max_tokens(before, after) == {
        "<=5": 3, "6-10": 500, "11-50": 26, ">50": 2}
    d = metrics.delta(before, after)
    assert d["single_row_calls"] == 3 and d["preemptions"] == 0


def _movie_fixture(tmp_path, monkeypatch, positives=12, negatives=12):
    monkeypatch.setattr(workloads, "SEMBENCH_MOVIE", tmp_path)
    rows = ["id,reviewId,reviewText,scoreSentiment"]
    rows += [f"m,p{i},good {i},POSITIVE" for i in range(positives)]
    rows += [f"m,n{i},bad {i},NEGATIVE" for i in range(negatives)]
    rows += [f"ant_man_and_the_wasp_quantumania,a{i},ant {i},"
             f"{'POSITIVE' if i % 2 else 'NEGATIVE'}" for i in range(6)]
    (tmp_path / "Reviews.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")


def test_pipeline_workloads_ops_and_arms(tmp_path, monkeypatch, fake_kv_manager):
    from vllm.kalypso.benchmark.eval import arms

    _movie_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(workloads, "SELECTIVITY_ROWS", 10)
    monkeypatch.setattr(workloads, "PIPELINE_RIGHT_ROWS", 6)

    fj = workloads.WORKLOADS["pipe_filter_join_s50"](10)
    assert [s["op"] for s in fj.stages] == [FILTER, JOIN] and fj.kind == JOIN
    left_pos = {i for i, _ in fj.positives}
    assert left_pos == workloads.load_movie_selectivity("movie_sel50", 10).positives
    assert {j for _, j in fj.positives} == {r.idx for r in fj.right_rows if "ant" in r.text
                                            and int(r.text.split()[1]) % 2 == 0}
    alone = workloads.WORKLOADS["pipe_filter_join_s50_join"](10)
    assert len(alone.rows) == len(left_pos) and not alone.stages
    assert len(alone.positives) == len(fj.positives)

    spec = parse_arm("kalypso_pipe:6,4")
    check_arm(spec, fj)
    ops = kalypso_ops(fj, arms.stage_blocks(spec, fj))
    assert ops[0] == {"op": "sem_filter", "args": {"prompt": fj.stages[0]["instruction"],
                                                   "block_size": 6}}
    assert ops[1]["op"] == "join" and ops[1]["args"]["pair_block_size"] == 4
    per_tuple = kalypso_ops(fj, arms.stage_blocks(parse_arm("kalypso_per_tuple"), fj))
    assert "block_size" not in per_tuple[0]["args"]
    assert "pair_block_size" not in per_tuple[1]["args"]
    assert arms.stage_blocks(parse_arm("kalypso_block:16"), fj) == (16, 16)
    with pytest.raises(ValueError):
        check_arm(parse_arm("kalypso_pipe:6"), fj)          # one size for two stages
    with pytest.raises(ValueError):
        check_arm(parse_arm("vllm_direct"), fj)
    with pytest.raises(ValueError):
        check_arm(parse_arm("kalypso_pipe:6,4"), load_reviews())

    fm = workloads.WORKLOADS["pipe_filter_map"](20)
    assert fm.kind == "map" and fm.positives and [s["op"] for s in fm.stages] == [FILTER, "map"]
    check_arm(parse_arm("kalypso_pipe:6,1"), fm)
    if not arms.HAS_BLOCK_MAP:
        with pytest.raises(ValueError, match="block_map"):
            check_arm(parse_arm("kalypso_pipe:6,4"), fm)
    # A filtered map's rows are a selection, scored against the filter labels.
    out = parse_query_response(fm, {"results": [
        {"idx": 0, "outputs": [{"op": "f", "value": "true"}, {"op": "m", "value": "x"}]}]})
    assert out.selected == {0} and out.outputs == {"0": "x"}
    assert parse_query_response(workloads.WORKLOADS["movie_map"](3), {"results": [
        {"idx": 0, "outputs": [{"op": "m", "value": "x"}]}]}).selected is None


def test_analyze_rows_and_best_arm(tmp_path):
    from vllm.kalypso.benchmark.eval import analyze

    meta = _meta(kind="filter", positives=[0], arms=["kalypso_per_tuple", "kalypso_block:4"],
                 max_num_seqs=2)
    trials = [{**_trial("kalypso_per_tuple", r, {}, 4.0, selected=[0]),
               "prefix_cache_hit_rate": 0.5, "mean_decode_time_s": 0.2} for r in (1, 2)]
    trials += [{**_trial("kalypso_block:4", r, {}, 2.0, selected=[0, 1]),
                "llm_requests": 1, "mean_decode_time_s": 1.0} for r in (1, 2)]
    run = tmp_path / "experiments" / "single" / "x" / "part"
    run.mkdir(parents=True)
    (run / "run.jsonl").write_text("\n".join(json.dumps(x) for x in [meta, *trials]) + "\n")

    rows = analyze.collect(tmp_path)
    base, block = rows
    assert block["speedup"] == 2.0 and block["blocks"] == "4"
    assert base["computed_prompt_tokens"] == 5 and base["waves"] == 1
    assert block["gen_tokens_per_call"] == 4.0 and block["decode_s_per_gen_token"] == 0.25
    assert block["f1_delta"] < 0          # a false positive
    assert analyze.best_arm(rows, 0.02, 0.9)["arm"] == "kalypso_per_tuple"
    assert analyze.best_arm(rows, 1.0, 0.9)["arm"] == "kalypso_block:4"
    assert "### " in analyze.summary(rows, 0.02, 0.9)


def test_experiment_manifest_is_consistent():
    import importlib.util
    from pathlib import Path

    from vllm.kalypso.benchmark.eval.arms import check_arm as check
    path = Path(__file__).resolve().parents[3] / "cloud" / "experiments.py"
    spec = importlib.util.spec_from_file_location("experiments", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    ids = [e["id"] for e in m.EXPERIMENTS]
    assert len(ids) == len(set(ids)) and {e["phase"] for e in m.EXPERIMENTS} == set(m.PHASES)
    for exp in m.EXPERIMENTS:
        names = [p["name"] for p in exp["parts"]]
        assert len(names) == len(set(names)), exp["id"]
        if m.missing(exp):
            continue  # its workload or operator is not in this checkout yet
        for p in exp["parts"]:
            assert p["workload"] in workloads.WORKLOADS, (exp["id"], p["workload"])
            for arm in m.arms_for(exp, p):
                parse_arm(arm)
    assert m.select("single") and all(e["phase"] == "single" for e in m.select("single"))
    with pytest.raises(SystemExit):
        m.select("nope")
