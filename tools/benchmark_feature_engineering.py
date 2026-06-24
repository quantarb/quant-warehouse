from __future__ import annotations

import argparse
import importlib
import json
import statistics
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from quant_warehouse.feature_engineering import build_price_technical_features  # noqa: E402
from quant_warehouse.feature_engineering.specs import BuiltFeatureSet  # noqa: E402


FeatureBuilder = Callable[[str, pd.DataFrame], BuiltFeatureSet]


@dataclass(frozen=True)
class BenchmarkConfig:
    rows: int = 5000
    symbols: int = 100
    repeats: int = 5
    warmups: int = 1
    seed: int = 7


@dataclass(frozen=True)
class BenchmarkStats:
    name: str
    rows: int
    symbols: int
    repeats: int
    total_rows: int
    min_seconds: float
    median_seconds: float
    rows_per_second: float
    feature_count: int


@dataclass(frozen=True)
class BenchmarkReport:
    baseline: BenchmarkStats
    candidate: BenchmarkStats | None
    speedup: float | None
    min_speedup: float | None
    passed_speed_gate: bool | None


def make_price_frame(rows: int, *, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(int(seed))
    index = pd.date_range("1980-01-01", periods=int(rows), freq="B")
    close = 100.0 + np.cumsum(rng.normal(0.02, 1.0, int(rows)))
    return pd.DataFrame(
        {
            "open": close + rng.normal(0.0, 0.2, int(rows)),
            "high": close + rng.uniform(0.1, 1.5, int(rows)),
            "low": close - rng.uniform(0.1, 1.5, int(rows)),
            "close": close,
            "volume": rng.integers(1_000, 1_000_000, int(rows)).astype(float),
        },
        index=index,
    )


def make_price_frames(config: BenchmarkConfig) -> list[tuple[str, pd.DataFrame]]:
    return [
        (f"SYM{idx:04d}", make_price_frame(config.rows, seed=config.seed + idx))
        for idx in range(int(config.symbols))
    ]


def load_builder(import_path: str) -> FeatureBuilder:
    module_name, sep, attr_name = str(import_path).partition(":")
    if not sep or not module_name or not attr_name:
        raise ValueError("Candidate must be in 'module:function' form.")
    module = importlib.import_module(module_name)
    builder = getattr(module, attr_name)
    if not callable(builder):
        raise TypeError(f"{import_path} is not callable.")
    return builder


def validate_candidate(
    baseline_builder: FeatureBuilder,
    candidate_builder: FeatureBuilder,
    sample_symbol: str,
    sample_prices: pd.DataFrame,
) -> None:
    baseline = baseline_builder(sample_symbol, sample_prices)
    candidate = candidate_builder(sample_symbol, sample_prices)
    if list(candidate.feature_cols) != list(baseline.feature_cols):
        raise AssertionError("Candidate feature_cols differ from baseline.")
    pd.testing.assert_frame_equal(
        candidate.df[baseline.feature_cols],
        baseline.df[baseline.feature_cols],
        check_dtype=False,
        check_exact=False,
        rtol=1e-10,
        atol=1e-10,
    )


def benchmark_builder(
    name: str,
    builder: FeatureBuilder,
    frames: Sequence[tuple[str, pd.DataFrame]],
    config: BenchmarkConfig,
) -> BenchmarkStats:
    for _ in range(max(int(config.warmups), 0)):
        for symbol, prices in frames:
            builder(symbol, prices)

    elapsed: list[float] = []
    feature_count = 0
    for _ in range(max(int(config.repeats), 1)):
        start = time.perf_counter()
        for symbol, prices in frames:
            built = builder(symbol, prices)
            feature_count = max(feature_count, len(built.feature_cols))
        elapsed.append(time.perf_counter() - start)

    min_seconds = min(elapsed)
    total_rows = int(config.rows) * int(config.symbols)
    return BenchmarkStats(
        name=name,
        rows=int(config.rows),
        symbols=int(config.symbols),
        repeats=int(config.repeats),
        total_rows=total_rows,
        min_seconds=float(min_seconds),
        median_seconds=float(statistics.median(elapsed)),
        rows_per_second=float(total_rows / min_seconds) if min_seconds > 0.0 else float("inf"),
        feature_count=int(feature_count),
    )


def run_benchmark(
    config: BenchmarkConfig,
    *,
    candidate_builder: FeatureBuilder | None = None,
    candidate_name: str | None = None,
    min_speedup: float | None = None,
    validate: bool = True,
) -> BenchmarkReport:
    frames = make_price_frames(config)
    baseline_builder = build_price_technical_features
    if candidate_builder is not None and validate:
        validate_candidate(baseline_builder, candidate_builder, frames[0][0], frames[0][1])

    baseline = benchmark_builder("current_pandas", baseline_builder, frames, config)
    candidate = None
    speedup = None
    passed = None
    if candidate_builder is not None:
        candidate = benchmark_builder(candidate_name or "candidate", candidate_builder, frames, config)
        speedup = baseline.min_seconds / candidate.min_seconds if candidate.min_seconds > 0.0 else float("inf")
        passed = speedup >= float(min_speedup if min_speedup is not None else 1.0)

    return BenchmarkReport(
        baseline=baseline,
        candidate=candidate,
        speedup=speedup,
        min_speedup=min_speedup,
        passed_speed_gate=passed,
    )


def _json_default(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable.")


def format_report(report: BenchmarkReport) -> str:
    lines = [
        (
            f"{report.baseline.name}: rows={report.baseline.rows:,} "
            f"symbols={report.baseline.symbols:,} features={report.baseline.feature_count:,} "
            f"min={report.baseline.min_seconds:.4f}s "
            f"median={report.baseline.median_seconds:.4f}s "
            f"throughput={report.baseline.rows_per_second:,.0f} rows/s"
        )
    ]
    if report.candidate is not None:
        lines.append(
            f"{report.candidate.name}: min={report.candidate.min_seconds:.4f}s "
            f"median={report.candidate.median_seconds:.4f}s "
            f"throughput={report.candidate.rows_per_second:,.0f} rows/s"
        )
        lines.append(
            f"speedup={report.speedup:.2f}x "
            f"min_required={report.min_speedup:.2f}x "
            f"passed={bool(report.passed_speed_gate)}"
        )
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark dense price feature engineering.")
    parser.add_argument("--rows", type=int, default=5000, help="Business-day rows per symbol.")
    parser.add_argument("--symbols", type=int, default=100, help="Synthetic symbols to benchmark.")
    parser.add_argument("--repeats", type=int, default=5, help="Measured repeats.")
    parser.add_argument("--warmups", type=int, default=1, help="Warmup runs before timing.")
    parser.add_argument("--seed", type=int, default=7, help="Synthetic data RNG seed.")
    parser.add_argument(
        "--candidate",
        default="",
        help="Optional candidate builder in module:function form. Callable must accept (symbol, df_prices).",
    )
    parser.add_argument(
        "--min-speedup",
        type=float,
        default=1.05,
        help="Minimum candidate speedup required over current builder.",
    )
    parser.add_argument("--no-validate", action="store_true", help="Skip candidate output parity check.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = BenchmarkConfig(
        rows=args.rows,
        symbols=args.symbols,
        repeats=args.repeats,
        warmups=args.warmups,
        seed=args.seed,
    )
    candidate_builder = load_builder(args.candidate) if args.candidate else None
    report = run_benchmark(
        config,
        candidate_builder=candidate_builder,
        candidate_name=args.candidate or None,
        min_speedup=args.min_speedup if candidate_builder is not None else None,
        validate=not args.no_validate,
    )
    if args.json:
        print(json.dumps(report, default=_json_default, indent=2, sort_keys=True))
    else:
        print(format_report(report))
    if report.passed_speed_gate is False:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
