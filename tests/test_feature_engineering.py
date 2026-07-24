from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_warehouse.platforms.data_providers.fmp.feature_engineering import (
    TA_CLASSIC_FAMILY_PREFIXES,
    build_price_ta_classic_feature_families,
    build_price_technical_features,
    build_preferred_stock_features,
    build_time_features,
    compute_features_worldclass,
)
from quant_warehouse.platforms.data_providers.thetadata.feature_engineering import (
    filter_option_instrument_rows,
)


def _price_frame(rows: int = 260) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=rows, freq="B")
    trend = np.linspace(100.0, 130.0, rows)
    return pd.DataFrame(
        {
            "open": trend,
            "high": trend + 1.0,
            "low": trend - 1.0,
            "close": trend + 0.25,
            "volume": np.arange(rows, dtype=float) + 1000.0,
        },
        index=index,
    )


def test_filter_option_instrument_rows_uses_change_percent_not_raw_change():
    chain = pd.DataFrame(
        {
            "bid": [1.0, 1.0, 0.0, 1.0],
            "ask": [1.1, 1.1, 1.1, 0.9],
            "change": [0.0, 0.0, 0.0, 1.0],
            "change_percent": [0.1, 0.0, 0.2, None],
        }
    )
    filtered = filter_option_instrument_rows(chain)
    assert len(filtered) == 1
    assert filtered.iloc[0]["change"] == 0.0


def test_build_price_technical_features_is_standalone_and_prefixed():
    built = build_price_technical_features("aapl", _price_frame())

    assert not built.df.empty
    assert built.df.index.names == ["date", "symbol"]
    assert built.df.index.get_level_values("symbol").unique().tolist() == ["AAPL"]
    assert "px__ret_1d" in built.feature_cols
    assert "px__macd" in built.feature_cols
    assert all(column.startswith("px__") for column in built.feature_cols)


def test_build_preferred_stock_features_keeps_raw_family_separate():
    preferred = pd.DataFrame(
        {
            "date": ["2025-01-02", "2025-01-02", "2025-01-03"],
            "symbol": ["ABC-PA", "ABC-PB", "ABC-PA"],
            "open": [10.0, 12.0, 11.0],
            "high": [11.0, 13.0, 12.0],
            "low": [9.0, 11.0, 10.0],
            "close": [10.5, 12.5, 11.5],
            "volume": [100.0, 200.0, 150.0],
        }
    )

    built = build_preferred_stock_features("ABC", preferred)

    assert built.df.index.names == ["date", "symbol"]
    assert built.df.loc[(pd.Timestamp("2025-01-02"), "ABC"), "preferred__issue_count"] == 2.0
    assert built.df.loc[(pd.Timestamp("2025-01-02"), "ABC"), "preferred__close_mean"] == 11.5
    assert built.df.loc[(pd.Timestamp("2025-01-03"), "ABC"), "preferred__has_data"] == 1.0
    assert all(column.startswith("preferred__") for column in built.feature_cols)


def test_price_technical_cuda_setting_falls_back_without_cudf(monkeypatch):
    monkeypatch.setenv("QW_FEATURE_ENGINEERING_CUDA", "always")

    built = build_price_technical_features("MSFT", _price_frame())

    assert not built.df.empty
    assert "px__dollar_vol" in built.feature_cols


def test_price_engine_matches_pandas_reference_for_core_features(monkeypatch):
    monkeypatch.setenv("QW_FEATURE_ENGINEERING_CUDA", "never")
    prices = _price_frame(260)

    actual = compute_features_worldclass(prices)

    close = prices["close"]
    high = prices["high"]
    low = prices["low"]
    volume = prices["volume"]
    ret_1d = close.pct_change()
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    hh20 = high.rolling(20).max()
    ll20 = low.rolling(20).min()
    dollar_vol = close * volume
    expected = pd.DataFrame(
        {
            "Ret20d": close.pct_change(20),
            "DistSMA20": (close - close.rolling(20).mean()) / (close.rolling(20).mean() + 1e-12),
            "MACD": macd,
            "MACDSignal": signal,
            "Vol20": ret_1d.rolling(20).std(),
            "BreakoutUp20": (close > hh20.shift(1)).astype(float),
            "PosInChannel20": (close - ll20) / ((hh20 - ll20) + 2e-12),
            "DollarVolZ20": (
                dollar_vol - dollar_vol.rolling(20).mean()
            ) / (dollar_vol.rolling(20).std() + 2e-12),
        },
        index=prices.index,
    )

    pd.testing.assert_frame_equal(
        actual[list(expected.columns)],
        expected,
        check_dtype=False,
        check_exact=False,
        rtol=1e-10,
        atol=1e-10,
    )


def test_ta_classic_feature_families_are_split_and_prefixed():
    built_by_family = build_price_ta_classic_feature_families("AAPL", _price_frame(90))

    assert set(built_by_family) == set(TA_CLASSIC_FAMILY_PREFIXES)
    assert any(built.feature_cols for built in built_by_family.values())
    for family_name, built in built_by_family.items():
        assert all(
            column.startswith(TA_CLASSIC_FAMILY_PREFIXES[family_name])
            for column in built.feature_cols
        )


def test_ta_classic_exposes_only_the_bounded_v1_indicator_set():
    from quant_warehouse.platforms.data_providers.fmp.feature_engineering.ta_classic_technical import (
        _import_pandas_ta_classic,
        _indicator_specs,
    )

    ta = _import_pandas_ta_classic()
    curated_names = {spec.fn_name for specs in _indicator_specs(ta).values() for spec in specs}

    assert {"rsi", "macd", "stoch", "bbands", "drawdown"}.issubset(curated_names)
    assert {"macdext", "macdfix", "stochf", "dema", "tema", "alma"}.isdisjoint(curated_names)
    with pytest.raises(TypeError, match="unexpected keyword argument 'mode'"):
        build_price_ta_classic_feature_families("AAPL", _price_frame(90), mode="all")


def test_build_time_features_matches_target_index():
    target_index = pd.MultiIndex.from_product(
        [pd.date_range("2024-01-01", periods=3), ["AAPL"]],
        names=["date", "symbol"],
    )

    frame = build_time_features(target_index=target_index)

    assert frame.index.equals(target_index)
    assert {"day_of_week", "month", "is_month_1"}.issubset(frame.columns)
