from __future__ import annotations

import polars as pl
from datetime import datetime

from quant_warehouse.platforms.data_providers.fmp.feature_engineering import fundamentals


def test_warehouse_section_mapping_excludes_removed_ttm_sections():
    assert fundamentals.warehouse_section_for_legacy_key("key_metrics") == "metrics"
    assert fundamentals.warehouse_section_for_legacy_key("income_statement_ttm") is None


def test_warehouse_section_to_indexed_frame_prefixes_columns(monkeypatch):
    class _FakeWarehouse:
        def read_fundamentals(self, symbol, *, section, provider, start=None, end=None):
            assert symbol == "AAPL"
            assert section == "metrics"
            assert provider == "fmp"
            return pl.DataFrame({"pe_ratio": [20.0], "market_cap": [1e12], "date": [datetime(2024, 12, 31)]})

    monkeypatch.setattr(fundamentals, "get_warehouse", lambda: _FakeWarehouse())

    out = fundamentals.warehouse_section_to_indexed_frame(
        "AAPL",
        "key_metrics",
        prefix="km__",
    )

    assert not out.is_empty()
    assert {"date", "symbol"}.issubset(out.columns)
    assert "km__pe_ratio" in out.columns


def test_fetch_fundamentals_data_reads_metrics_and_ratios(monkeypatch):
    class _FakeWarehouse:
        def read_fundamentals(self, symbol, *, section, provider, start=None, end=None):
            assert symbol == "AAPL"
            if section == "metrics":
                return pl.DataFrame({"market_cap": [1e12], "date": [datetime(2024, 12, 31)]})
            if section == "ratios":
                return pl.DataFrame({"price_to_earnings_ratio": [20.0], "date": [datetime(2024, 12, 31)]})
            return pl.DataFrame()

    monkeypatch.setattr(fundamentals, "get_warehouse", lambda: _FakeWarehouse())

    out = fundamentals.fetch_fundamentals_data(["AAPL"], verbose=False, filing_lag_days=0)

    assert not out.is_empty()
    assert {"date", "symbol"}.issubset(out.columns)
    assert {"km__market_cap", "rt__price_to_earnings_ratio"}.issubset(out.columns)
