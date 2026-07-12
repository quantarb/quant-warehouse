from __future__ import annotations

import pandas as pd
import pytest

from quant_warehouse.platforms.data_providers.thetadata.target_engineering import (
    OracleOptionLabelPanelSpec,
    build_oracle_option_label_panel,
)
from quant_warehouse.platforms.data_providers.thetadata.target_engineering import oracle_option_labels


def _chain(snapshot_date: str, *, call_bid: float, call_ask: float) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "snapshot_date": snapshot_date,
                "underlying_symbol": "AAPL",
                "contract_symbol": "AAPL260220C00200000",
                "expiration": "2026-02-20",
                "strike": 200.0,
                "option_type": "call",
                "bid": call_bid,
                "ask": call_ask,
                "mid": (call_bid + call_ask) / 2.0,
                "underlying_price": 195.0,
            },
            {
                "snapshot_date": snapshot_date,
                "underlying_symbol": "AAPL",
                "contract_symbol": "AAPL260220P00190000",
                "expiration": "2026-02-20",
                "strike": 190.0,
                "option_type": "put",
                "bid": 3.0,
                "ask": 4.0,
                "mid": 3.5,
                "underlying_price": 195.0,
            },
        ]
    )


def _trades() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "trade_id": "t1",
                "symbol": "aapl",
                "side": "oracle_long",
                "entry_date": "2026-01-02",
                "exit_date": "2026-01-05",
                "freq": "YE",
                "k": 1,
                "ret_dec": 0.10,
            }
        ]
    )


def test_build_oracle_option_label_panel_uses_entry_ask_and_exit_bid(monkeypatch):
    entry = _chain("2026-01-02", call_bid=4.0, call_ask=5.0)
    exit_chain = _chain("2026-01-05", call_bid=7.0, call_ask=8.0)
    monkeypatch.setattr(
        oracle_option_labels,
        "_normalized_cached_snapshots",
        lambda _symbol, _dates: {
            pd.Timestamp("2026-01-02"): entry,
            pd.Timestamp("2026-01-05"): exit_chain,
        },
    )

    result = build_oracle_option_label_panel(
        _trades(),
        spec=OracleOptionLabelPanelSpec(max_dte=90, target_dte=45),
    )

    assert result.summary["trades_labeled"] == 1
    assert result.summary["long_call_rows"] == 1
    assert list(result.panel["contract_symbol"]) == ["AAPL260220C00200000"]
    assert result.panel.iloc[0]["entry_ask"] == 5.0
    assert result.panel.iloc[0]["exit_bid"] == 7.0
    assert result.panel.iloc[0]["option_return"] == pytest.approx(0.4)
    assert result.panel.iloc[0]["pricing_convention"] == "buy_ask_sell_bid_entry_exit_only"


def test_build_oracle_option_label_panel_reports_missing_endpoint(monkeypatch):
    monkeypatch.setattr(
        oracle_option_labels,
        "_normalized_cached_snapshots",
        lambda _symbol, _dates: {pd.Timestamp("2026-01-02"): _chain("2026-01-02", call_bid=4.0, call_ask=5.0)},
    )

    result = build_oracle_option_label_panel(_trades())

    assert result.panel.empty
    assert result.summary["status"] == "no_option_rows"
    assert result.summary["trades_skipped_missing_historical_options"] == 1
    assert result.summary["skipped_missing_options"][0]["exit_option_data"] is False


def test_build_oracle_option_label_panel_requires_trade_columns():
    with pytest.raises(KeyError, match="exit_date"):
        build_oracle_option_label_panel(pd.DataFrame([{"symbol": "AAPL", "side": "long", "entry_date": "2026-01-02"}]))


def test_entry_candidates_must_survive_oracle_exit_without_default_dte_ceiling():
    chain = pd.DataFrame(
        [
            {
                "option_type": "call",
                "bid": 1.0,
                "ask": 1.2,
                "expiration": "2026-03-01",
                "contract_symbol": "EXPIRES_BEFORE_EXIT",
            },
            {
                "option_type": "call",
                "bid": 2.0,
                "ask": 2.2,
                "expiration": "2026-12-31",
                "contract_symbol": "COVERS_LONG_TRADE",
            },
        ]
    )

    result = oracle_option_labels._filter_entry_candidates(
        chain,
        option_type="call",
        entry_date=pd.Timestamp("2026-01-02"),
        exit_date=pd.Timestamp("2026-06-30"),
        max_dte=None,
    )

    assert result["contract_symbol"].tolist() == ["COVERS_LONG_TRADE"]
    assert result["dte"].iloc[0] > 90
