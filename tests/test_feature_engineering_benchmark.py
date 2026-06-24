from __future__ import annotations

import json

from tools.benchmark_feature_engineering import (
    BenchmarkConfig,
    _json_default,
    format_report,
    run_benchmark,
)


def test_feature_engineering_benchmark_smoke():
    report = run_benchmark(BenchmarkConfig(rows=40, symbols=2, repeats=1, warmups=0))

    assert report.baseline.feature_count > 0
    assert report.baseline.total_rows == 80
    assert report.candidate is None
    assert "current_pandas" in format_report(report)
    assert json.loads(json.dumps(report, default=_json_default))["baseline"]["total_rows"] == 80
