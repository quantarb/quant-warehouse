from __future__ import annotations

import numpy as np
import pandas as pd

from quant_warehouse.platforms.data_providers.fmp.feature_engineering import (
    build_historical_price_eod_features,
    build_preferred_stock_features,
    build_time_features,
    compute_features_worldclass,
)
from quant_warehouse.platforms.data_providers.fmp.feature_engineering.fundamental_features import merge_feature_sets
from quant_warehouse.platforms.data_providers.fmp.feature_engineering.specs import BuiltFeatureSet
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


def test_build_historical_price_eod_features_is_endpoint_native():
    built = build_historical_price_eod_features("aapl", _price_frame())

    assert not built.df.empty
    assert built.df.index.names == ["date", "symbol"]
    assert built.df.index.get_level_values("symbol").unique().tolist() == ["AAPL"]
    assert "eod__adjusted_open" in built.feature_cols
    assert "eod__adjusted_close" in built.feature_cols
    assert "eod__volume" in built.feature_cols
    assert all(column.startswith("eod__") for column in built.feature_cols)
    assert built.family_name == "equity-historical-price-eod"
    assert built.endpoint_name == "historical-price-eod"
    assert not any("macd" in column or "ret" in column for column in built.feature_cols)


def test_passthrough_feature_set_can_remain_sparse():
    from quant_warehouse.platforms.data_providers.fmp.feature_engineering.fundamental_features import (
        build_passthrough_section_features,
    )

    target = pd.MultiIndex.from_tuples(
        [("2024-01-01", "AAPL"), ("2024-01-02", "AAPL")],
        names=["date", "symbol"],
    )
    source = pd.DataFrame(
        {"revenue": [10.0]},
        index=pd.MultiIndex.from_tuples(
            [("2024-01-01", "AAPL")], names=["date", "symbol"]
        ),
    )
    built = build_passthrough_section_features(
        "AAPL",
        target,
        section_key="income",
        prefix="income__",
        sparse_loader=lambda *args, **kwargs: source.rename(columns={"revenue": "income__revenue"}),
        broadcast_to_target=False,
    )
    assert len(built.df) == 1
    assert built.df.index.equals(source.index)
    assert built.endpoint_name == "income"
    assert built.source_asset_class == "equity"
    assert built.presence is not None


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


def test_merge_feature_sets_preserves_endpoint_union_and_presence_masks():
    index_a = pd.MultiIndex.from_tuples(
        [(pd.Timestamp("2025-01-01"), "A")], names=["date", "symbol"]
    )
    index_b = pd.MultiIndex.from_tuples(
        [(pd.Timestamp("2025-01-02"), "A")], names=["date", "symbol"]
    )
    prices = BuiltFeatureSet(
        pd.DataFrame({"px__close": [10.0]}, index=index_a),
        ["px__close"],
        family_name="equity_ohlcv",
        endpoint_name="prices",
        source_asset_class="equity",
    )
    options = BuiltFeatureSet(
        pd.DataFrame({"opt__delta": [0.5]}, index=index_b),
        ["opt__delta"],
        family_name="option_greeks",
        endpoint_name="option_history_greeks_eod",
        source_asset_class="option",
    )

    merged = merge_feature_sets([prices, options])

    assert set(merged.df.index) == set(index_a) | set(index_b)
    assert merged.family_columns == {
        "equity_ohlcv": ["px__close"],
        "option_greeks": ["opt__delta"],
    }
    assert merged.family_presence.loc[index_a[0], "equity_ohlcv"]
    assert not merged.family_presence.loc[index_a[0], "option_greeks"]
    assert merged.family_presence.loc[index_b[0], "option_greeks"]


def test_historical_price_eod_ignores_ta_cuda_setting(monkeypatch):
    monkeypatch.setenv("QW_FEATURE_ENGINEERING_CUDA", "always")

    built = build_historical_price_eod_features("MSFT", _price_frame())

    assert not built.df.empty
    assert built.feature_cols == [
        "eod__adjusted_open",
        "eod__adjusted_high",
        "eod__adjusted_low",
        "eod__adjusted_close",
        "eod__volume",
    ]


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


def test_build_time_features_matches_target_index():
    target_index = pd.MultiIndex.from_product(
        [pd.date_range("2024-01-01", periods=3), ["AAPL"]],
        names=["date", "symbol"],
    )

    frame = build_time_features(target_index=target_index)

    assert frame.index.equals(target_index)
    assert {"day_of_week", "month", "is_month_1"}.issubset(frame.columns)
