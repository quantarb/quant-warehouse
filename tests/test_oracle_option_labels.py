from datetime import datetime

import polars as pl
import pytest

from quant_warehouse.platforms.data_providers.thetadata.target_engineering import OracleOptionLabelPanelSpec, build_oracle_option_label_panel
from quant_warehouse.platforms.data_providers.thetadata.target_engineering import oracle_option_labels


def _chain(snapshot_date: str, *, call_bid: float, call_ask: float) -> pl.DataFrame:
    return pl.DataFrame([
        {"snapshot_date": snapshot_date, "underlying_symbol": "AAPL", "contract_symbol": "AAPL260220C00200000", "expiration": "2026-02-20", "strike": 200.0, "option_type": "call", "bid": call_bid, "ask": call_ask, "mid": (call_bid + call_ask) / 2.0, "underlying_price": 195.0},
        {"snapshot_date": snapshot_date, "underlying_symbol": "AAPL", "contract_symbol": "AAPL260220P00190000", "expiration": "2026-02-20", "strike": 190.0, "option_type": "put", "bid": 3.0, "ask": 4.0, "mid": 3.5, "underlying_price": 195.0},
    ])


def _trades() -> pl.DataFrame:
    return pl.DataFrame([{"trade_id": "t1", "symbol": "aapl", "side": "oracle_long", "entry_date": "2026-01-02", "exit_date": "2026-01-05", "freq": "YE", "k": 1, "ret_dec": 0.10}])


def test_build_oracle_option_label_panel_uses_entry_ask_and_exit_bid(monkeypatch):
    monkeypatch.setattr(oracle_option_labels, "_normalized_cached_snapshots", lambda _symbol, _dates: {datetime(2026, 1, 2): _chain("2026-01-02", call_bid=4.0, call_ask=5.0), datetime(2026, 1, 5): _chain("2026-01-05", call_bid=7.0, call_ask=8.0)})
    result = build_oracle_option_label_panel(_trades(), spec=OracleOptionLabelPanelSpec(max_dte=90, target_dte=45))
    assert result.summary["trades_labeled"] == 1
    assert result.panel.height == 1
    row = result.panel.row(0, named=True)
    assert row["entry_quote"] == 5.0
    assert row["exit_quote"] == 7.0
    assert row["option_return_pct"] == pytest.approx(0.4)


def test_oracle_panel_is_polars_and_handles_missing_cache(monkeypatch):
    monkeypatch.setattr(oracle_option_labels, "_normalized_cached_snapshots", lambda _symbol, _dates: {})
    result = build_oracle_option_label_panel(_trades())
    assert isinstance(result.panel, pl.DataFrame)
    assert result.panel.is_empty()
    assert result.summary["trades_skipped_missing_historical_options"] == 1


def test_entry_candidates_have_no_default_dte_ceiling():
    chain = pl.DataFrame({"option_type": ["call", "call"], "bid": [1.0, 2.0], "ask": [1.2, 2.2], "expiration": ["2026-03-01", "2026-12-31"], "contract_symbol": ["EXPIRES_BEFORE_EXIT", "COVERS_LONG_TRADE"]})
    result = oracle_option_labels._filter_entry_candidates(chain, option_type="call", entry_date=datetime(2026, 1, 2), max_dte=None)
    assert result["contract_symbol"].to_list() == ["EXPIRES_BEFORE_EXIT", "COVERS_LONG_TRADE"]
    assert result["dte"].max() > 90


def test_oracle_panel_requires_trade_columns():
    with pytest.raises(KeyError, match="exit_date"):
        build_oracle_option_label_panel(pl.DataFrame([{"symbol": "AAPL", "side": "long", "entry_date": "2026-01-02"}]))
