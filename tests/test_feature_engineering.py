from datetime import datetime, timedelta

import polars as pl

from quant_warehouse.platforms.data_providers.fmp.feature_engineering import (
    build_historical_price_eod_features,
    build_preferred_stock_features,
    build_time_features,
    compute_features_worldclass,
)
from quant_warehouse.platforms.data_providers.fmp.feature_engineering.fundamental_features import merge_feature_sets
from quant_warehouse.platforms.data_providers.fmp.feature_engineering.specs import BuiltFeatureSet
from quant_warehouse.platforms.data_providers.thetadata.feature_engineering import filter_option_instrument_rows


def _price_frame(rows: int = 260) -> pl.DataFrame:
    dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(rows)]
    trend = [100.0 + 30.0 * i / (rows - 1) for i in range(rows)]
    return pl.DataFrame({"date": dates, "open": trend, "high": [v + 1.0 for v in trend], "low": [v - 1.0 for v in trend], "close": [v + 0.25 for v in trend], "volume": [float(i) + 1000.0 for i in range(rows)]})


def test_filter_option_instrument_rows_uses_change_percent_not_raw_change():
    chain = pl.DataFrame({"bid": [1.0, 1.0, 0.0, 1.0], "ask": [1.1, 1.1, 1.1, 0.9], "change": [0.0, 0.0, 0.0, 1.0], "change_percent": [0.1, 0.0, 0.2, None]})
    filtered = filter_option_instrument_rows(chain)
    assert filtered.height == 1
    assert filtered.item(0, "change") == 0.0


def test_build_historical_price_eod_features_is_endpoint_native():
    built = build_historical_price_eod_features("aapl", _price_frame())
    assert not built.df.is_empty()
    assert set(built.df["symbol"]) == {"AAPL"}
    assert {"date", "symbol"}.issubset(built.df.columns)
    assert "eod__adjusted_open" in built.feature_cols
    assert built.family_name == "equity-historical-price-eod"
    assert built.endpoint_name == "historical-price-eod"


def test_passthrough_feature_set_can_remain_sparse():
    from quant_warehouse.platforms.data_providers.fmp.feature_engineering.fundamental_features import build_passthrough_section_features
    target = pl.DataFrame({"date": [datetime(2024, 1, 1), datetime(2024, 1, 2)], "symbol": ["AAPL", "AAPL"]})
    source = pl.DataFrame({"date": [datetime(2024, 1, 1)], "symbol": ["AAPL"], "income__revenue": [10.0]})
    built = build_passthrough_section_features("AAPL", target, section_key="income", prefix="income__", sparse_loader=lambda *args, **kwargs: source, broadcast_to_target=False)
    assert built.df.height == 1
    assert built.df.item(0, "income__revenue") == 10.0
    assert built.endpoint_name == "income"
    assert built.presence is not None


def test_build_preferred_stock_features_keeps_raw_family_separate():
    preferred = pl.DataFrame({"date": ["2025-01-02", "2025-01-02", "2025-01-03"], "symbol": ["ABC-PA", "ABC-PB", "ABC-PA"], "open": [10.0, 12.0, 11.0], "high": [11.0, 13.0, 12.0], "low": [9.0, 11.0, 10.0], "close": [10.5, 12.5, 11.5], "volume": [100.0, 200.0, 150.0]})
    built = build_preferred_stock_features("ABC", preferred)
    assert {"date", "symbol"}.issubset(built.df.columns)
    assert built.df.filter((pl.col("date") == datetime(2025, 1, 2)) & (pl.col("symbol") == "ABC")).item(0, "preferred__issue_count") == 2.0
    assert all(column.startswith("preferred__") for column in built.feature_cols)


def test_merge_feature_sets_preserves_endpoint_union_and_presence_masks():
    prices = BuiltFeatureSet(pl.DataFrame({"date": [datetime(2025, 1, 1)], "symbol": ["A"], "px__close": [10.0]}), ["px__close"], family_name="equity_ohlcv", endpoint_name="prices", source_asset_class="equity")
    options = BuiltFeatureSet(pl.DataFrame({"date": [datetime(2025, 1, 2)], "symbol": ["A"], "opt__delta": [0.5]}), ["opt__delta"], family_name="option_greeks", endpoint_name="option_history_greeks_eod", source_asset_class="option")
    merged = merge_feature_sets([prices, options])
    assert merged.df.height == 2
    assert set(merged.feature_cols) == {"px__close", "opt__delta"}


def test_historical_price_eod_ignores_ta_cuda_setting(monkeypatch):
    monkeypatch.setenv("QW_FEATURE_ENGINEERING_CUDA", "always")
    built = build_historical_price_eod_features("MSFT", _price_frame())
    assert not built.df.is_empty()
    assert built.feature_cols == ["eod__adjusted_open", "eod__adjusted_high", "eod__adjusted_low", "eod__adjusted_close", "eod__volume"]


def test_price_engine_produces_core_features(monkeypatch):
    monkeypatch.setenv("QW_FEATURE_ENGINEERING_CUDA", "never")
    actual = compute_features_worldclass(_price_frame())
    assert {"Ret20d", "DistSMA20", "MACD", "MACDSignal", "Vol20", "BreakoutUp20", "DollarVolZ20"}.issubset(actual.columns)
    assert actual.height == 260


def test_build_time_features_matches_target_index():
    target = pl.DataFrame({"date": [datetime(2024, 1, 1), datetime(2024, 1, 2), datetime(2024, 1, 3)], "symbol": ["AAPL"] * 3})
    frame = build_time_features(target_index=target)
    assert frame.select(["date", "symbol"]).equals(target)
    assert {"day_of_week", "month", "is_month_1"}.issubset(frame.columns)
