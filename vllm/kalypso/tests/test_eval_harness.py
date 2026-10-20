"""Server-free parts of the evaluation harness (benchmark/eval)."""

import csv
import io
import json

import pytest

from vllm.kalypso.benchmark.eval import fetch_data, metrics, report
from vllm.kalypso.benchmark.eval.stages import summarize_trace
from vllm.kalypso.benchmark.eval.arms import (ArmError, check_arm, direct_requests,
                                              kalypso_ops, parse_arm, pipeline_ops,
                                              parse_query_response)
from vllm.kalypso.benchmark.eval.workloads import (BIODEX_ARTICLES, FILTER, FULL_DATA,
                                                   JOIN, PIPELINE, Row, Workload,
                                                   load_biodex, load_contractnli,
                                                   load_contractnli_pipeline,
                                                   load_join5x5, load_medec,
                                                   load_medec_full, load_reviews)
from vllm.kalypso.controller.plan import SemanticPlan
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
    trial = _trial("kalypso_block:2x3", 1, {}, 1.0, selected=[[0, 2]])
    q = report.quality(meta, trial)
    assert (q["tp"], q["fp"], q["fn"], q["tn"]) == (1, 0, 1, 4)
    assert q["recall"] == 0.5 and q["precision"] == 1.0
    assert report.quality(meta, _trial("x", 1, {}, 1.0, selected=[]))["f1"] is None


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


def test_parse_and_check_arms():
    assert parse_arm("kalypso_block:10").block == (10,)
    assert parse_arm("kalypso_block:2x5").block == (2, 5)
    for bad in ("kalypso_block", "kalypso_block:0", "kalypso_block:axb", "nope"):
        with pytest.raises(ValueError):
            parse_arm(bad)
    with pytest.raises(ValueError):
        check_arm(parse_arm("kalypso_block:5"), load_biodex(1))
    with pytest.raises(ValueError):
        check_arm(parse_arm("kalypso_block:5"), load_join5x5())
    check_arm(parse_arm("kalypso_block:5x5"), load_join5x5())


def test_query_ops_and_join_response_parsing():
    join = load_join5x5()
    ops = kalypso_ops(join, (2, 5))
    assert ops == [{"op": "join", "args": {
        "instruction": join.instruction, "right_table": str(join.right_path),
        "left_block_size": 2, "right_block_size": 5}}]
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


def _trace():
    stats = {"BlockSemFilter": {"fallback_calls": 1, "llm_calls": 5}}
    return {
        "elapsed_s": 9.0, "first_output_s": 4.0, "outputs": 3,
        "stages": {
            "10": {"operators": ["BlockSemJoin"], "block_size": 2, "tasks": 2,
                   "rows_in": 4, "rows_out": 6, "mean_occupancy": 1.0,
                   "first_start": 2.0, "last_end": 7.0,
                   "operator_stats": {"BlockSemJoin": {"fallback_calls": 2}}},
            "9": {"operators": ["BlockSemFilter"], "block_size": 4, "tasks": 2,
                  "rows_in": 8, "rows_out": 4, "mean_occupancy": 1.0,
                  "first_start": 0.0, "last_end": 3.0, "operator_stats": stats},
        },
        "tasks": [],
    }


def test_summarize_trace_orders_stages_and_measures_overlap():
    d = summarize_trace(_trace())
    assert d["first_output_s"] == 4.0 and d["fallback_calls"] == 3
    f1, join = d["stages"]
    assert f1["operators"] == ["BlockSemFilter"] and f1["selectivity"] == 0.5
    assert f1["overlap_s"] is None
    assert join["selectivity"] == 1.5            # fanout
    assert join["overlap_s"] == 1.0              # started at 2.0, F1 ended at 3.0
    assert summarize_trace(None) == {"first_output_s": None, "fallback_calls": None,
                                     "stages": []}


def test_stage_table_reports_medians_per_stage():
    meta = _meta(arms=["kalypso_block:4"])
    trials = []
    for repeat, end in ((1, 3.0), (2, 5.0)):
        t = _trace()
        t["stages"]["9"]["last_end"] = end
        trials.append({**_trial("kalypso_block:4", repeat, {}, 1.0), **summarize_trace(t)})
    table = report.stage_table(meta, trials)
    assert "| `kalypso_block:4` | 0 | BlockSemFilter | 4 | 2 | 8 | 4 | 0.50 | 1.00 | 0.00 | 4.00 [3.00-5.00] |" in table
    assert "| `kalypso_block:4` | 1 | BlockSemJoin | 2 |" in table
    assert report.stage_table(meta, [_trial("kalypso_block:4", 1, {}, 1.0)]) == ""


SAMPLE = BIODEX_ARTICLES.parent


def test_fetch_writes_contractnli_like_the_paper_sample(tmp_path):
    hyp = (SAMPLE / "contract-nli" / "hypotheses" / "1.txt").read_text()
    labels = {"nda-11": {"short_description": "No reverse engineering",
                         "hypothesis": hyp.split("HYPOTHESIS:\n", 1)[1].strip()},
              "nda-2": {"short_description": "x", "hypothesis": "y"}}
    split = {"labels": labels, "documents": [
        {"id": 7, "text": "Contract A", "annotation_sets": [{"annotations": {
            "nda-11": {"choice": "Entailment"}, "nda-2": {"choice": "NotMentioned"}}}]}]}
    counts = fetch_data.write_contractnli_split(split, tmp_path)
    assert counts == {"contracts": 1, "hypotheses": 2}
    assert (tmp_path / "hypotheses" / "1.txt").read_text() == hyp
    assert (tmp_path / "contracts" / "1.txt").read_text() == "Contract A"
    rows = (tmp_path / "labels.csv").read_text().splitlines()
    assert rows[1:] == ["1,1,nda-11,Entailment,7", "1,2,nda-2,NotMentioned,7"]


def test_fetch_builds_medec_data_column_like_the_paper_sample(tmp_path):
    with (SAMPLE / "MEDEC-TrainingSet-1000.csv").open(newline="", encoding="utf-8") as f:
        sample = list(csv.DictReader(f))
    raw = io.StringIO()
    writer = csv.DictWriter(raw, fieldnames=[k for k in sample[0] if k != "data"])
    writer.writeheader()
    writer.writerows({k: v for k, v in r.items() if k != "data"} for r in sample)
    assert fetch_data.write_medec(raw.getvalue().encode(), tmp_path) == len(sample)
    with (tmp_path / "medec_train.csv").open(newline="", encoding="utf-8") as f:
        assert list(csv.DictReader(f)) == sample


needs_full_data = pytest.mark.skipif(
    not (FULL_DATA / "manifest.json").exists(),
    reason="full datasets not fetched (python -m vllm.kalypso.benchmark.eval.fetch_data)")


@needs_full_data
def test_contractnli_workload_matches_labels_and_kalypso_prompts(fake_kv_manager):
    w = load_contractnli(3)
    assert w.kind == JOIN and [r.idx for r in w.rows] == [1, 2, 3]
    assert len(w.right_rows) == 17 and w.source_ids == (1, 2, 3)
    assert w.positives and all(l in (1, 2, 3) and 1 <= r <= 17 for l, r in w.positives)
    left = list(data._data_source(None, str(w.data_path), None))
    right = list(data._data_source(None, str(w.right_path), None))
    by_key = {key: messages for key, messages, _ in direct_requests(w)}
    pairs = [p for ctx in left for p in CartesianProduct(right)(ctx)]
    assert len(pairs) == len(by_key) == 51
    for pair in pairs:
        key = f"{pair.state.idx},{pair.state.right_idx}"
        assert SemFilter(w.instruction)._build_prompts(pair)[1] == by_key[key]


@needs_full_data
def test_medec_full_workload_is_seeded_and_matches_kalypso_prompts(fake_kv_manager):
    a, b = load_medec_full(16), load_medec_full(16)
    assert a.source_ids == b.source_ids and len(a.rows) == 16
    assert load_medec_full(16, seed=1).source_ids != a.source_ids
    assert a.positives <= {r.idx for r in a.rows}
    ctxs = list(data._data_source(None, str(a.data_path), None))
    for ctx, (_, messages, _) in zip(ctxs, direct_requests(a)):
        assert SemFilter(a.instruction)._build_prompts(ctx)[1] == messages


def _sample_pipeline():
    base = SAMPLE / "contract-nli"
    rows = [Row(i, f"c{i}") for i in (1, 2)]
    right = [Row(i, f"h{i}") for i in (1, 2, 3)]
    return Workload(
        name="p", kind=PIPELINE, data_path=base / "contracts", instruction="j",
        rows=rows, right_path=base / "hypotheses", right_rows=right,
        positives=frozenset({(1, 2), (2, 3)}),
        stages=({"op": "sem_filter", "prompt": "valid?"},
                {"op": "join", "instruction": "entails?"},
                {"op": "sem_map", "prompt": "explain"}))


def test_pipeline_arms_parse_and_check():
    w = _sample_pipeline()
    spec = parse_arm("kalypso_block:4/1x17")
    assert spec.groups == ((4,), (1, 17)) and spec.block == (4,)
    check_arm(spec, w)
    check_arm(parse_arm("kalypso_per_tuple"), w)
    for bad in ("kalypso_block:4", "kalypso_block:1x17/4", "kalypso_block:4/1x17/2"):
        with pytest.raises(ValueError):
            check_arm(parse_arm(bad), w)
    with pytest.raises(ValueError):
        check_arm(parse_arm("vllm_direct"), w)
    with pytest.raises(ValueError):
        check_arm(parse_arm("kalypso_block:4/2"), load_reviews())


def test_pipeline_ops_build_the_intended_kalypso_plan(fake_kv_manager):
    w = _sample_pipeline()
    assert pipeline_ops(w) == [
        {"op": "sem_filter", "args": {"prompt": "valid?"}},
        {"op": "join", "args": {"instruction": "entails?", "right_table": str(w.right_path)}},
        {"op": "sem_map", "args": {"prompt": "explain"}},
    ]
    assert pipeline_ops(w, parse_arm("kalypso_block:1/1x1").groups) == pipeline_ops(w)
    blocked = pipeline_ops(w, parse_arm("kalypso_block:4/1x17").groups)
    assert blocked[0]["args"]["block_size"] == 4
    assert (blocked[1]["args"]["left_block_size"], blocked[1]["args"]["right_block_size"]) == (1, 17)
    physical = SemanticPlan(executor=None).parse_ops(blocked)
    assert [type(op).__name__ for op in physical] == ["BlockSemFilter", "BlockSemJoin", "SemMap"]
    assert physical[1].right_block_size == 17 and len(physical[1].right_table) == 10
    per_tuple = SemanticPlan(executor=None).parse_ops(pipeline_ops(w))
    assert [type(op).__name__ for op in per_tuple] == ["SemFilter", "CartesianProduct",
                                                      "SemFilter", "SemMap"]


def test_pipeline_scoring_uses_real_row_ids():
    w = _sample_pipeline()
    out = parse_query_response(w, {"results": [
        {"idx": 1, "right_idx": 2, "outputs": [{"op": "m", "value": "because"}]}]})
    assert out.selected == {(1, 2)} and out.outputs == {"1,2": "because"}
    meta = _meta(kind="pipeline", rows=2, right_rows=3, row_ids=[1, 2],
                 right_row_ids=[1, 2, 3], positives=[[1, 2], [2, 3]],
                 arms=["kalypso_per_tuple", "kalypso_block:2/2x3"])
    trials = [_trial("kalypso_per_tuple", 1, {}, 1.0, selected=[[1, 2], [2, 3]]),
              _trial("kalypso_block:2/2x3", 1, {}, 1.0, selected=[[1, 2]])]
    q = report.quality(meta, trials[1])
    assert (q["tp"], q["fp"], q["fn"], q["tn"]) == (1, 0, 1, 4)
    assert report.agreement(meta, trials, "kalypso_per_tuple")["kalypso_block:2/2x3"][
        "vs_reference"] == 5 / 6


@needs_full_data
def test_contractnli_pipeline_workloads():
    fjm, fj = load_contractnli_pipeline(2), load_contractnli_pipeline(2, with_map=False)
    assert [s["op"] for s in fjm.stages] == ["sem_filter", "join", "sem_map"]
    assert [s["op"] for s in fj.stages] == ["sem_filter", "join"]
    assert fjm.positives == load_contractnli(2).positives
    for arm in fjm.default_arms:
        check_arm(parse_arm(arm), fjm)
