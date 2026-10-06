"""Server-free parts of the evaluation harness (benchmark/eval)."""

import json

from vllm.kalypso.benchmark.eval import metrics, report
from vllm.kalypso.benchmark.eval.workloads import BIODEX_ARTICLES, load_biodex

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


def _trial(arm, repeat, outputs, latency):
    return {"type": "trial", "arm": arm, "repeat": repeat, "outputs": outputs,
            "latency_s": latency, "llm_requests": 2, "prompt_tokens": 10,
            "generation_tokens": 4, "prefix_cache_hit_rate": None,
            "peak_kv_cache_pct": 5.0, "peak_server_rss_kb": 2048,
            "rows_per_s": 1.0, "truncated": 0}


def test_report_summary_and_agreement(tmp_path):
    trials = [
        _trial("kalypso_per_tuple", 1, {"1": "a", "2": "b"}, 4.0),
        _trial("vllm_direct", 1, {"1": "a", "2": "x"}, 2.0),
        _trial("kalypso_per_tuple", 2, {"1": "a", "2": "b"}, 6.0),
        _trial("vllm_direct", 2, {"1": "a", "2": "b"}, 3.0),
    ]
    summary = report.summarize(trials)
    assert summary["kalypso_per_tuple"]["latency_s"] == {"median": 5.0, "min": 4.0, "max": 6.0}
    assert summary["vllm_direct"]["peak_server_rss_kb"]["median"] == 2.0
    assert summary["vllm_direct"]["prefix_cache_hit_rate"] is None

    agree = report.agreement(trials, "kalypso_per_tuple")
    assert agree["kalypso_per_tuple"] == {"vs_reference": 1.0, "self": 1.0}
    assert agree["vllm_direct"] == {"vs_reference": 0.75, "self": 0.5}

    meta = {"type": "meta", "model": "m", "cpu": "c", "platform": "p",
            "workload": "biodex", "rows": 2, "repeats": 2, "warmup": 1,
            "prefix_cache_reset": True, "git_sha": "abcdef0123", "git_dirty": False,
            "arms": ["kalypso_per_tuple", "vllm_direct"], "server": {}}
    path = tmp_path / "r.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in [meta, *trials]) + "\n")
    loaded_meta, loaded = report.load(path)
    table = report.to_markdown(loaded_meta, report.summarize(loaded),
                               report.agreement(loaded, "kalypso_per_tuple"),
                               "kalypso_per_tuple")
    assert "| `vllm_direct` | 2.50 [2.00-3.00] |" in table
    assert table.rstrip().endswith("| 75% | 50% |")
