from __future__ import annotations

from datetime import datetime, timedelta
from types import SimpleNamespace

import polars as pl

from quant_warehouse.research_tools.security_context import (
    SecurityContextSpec,
    build_security_context_panel,
)


class _Warehouse:
    def read_prices(self, symbol, **kwargs):
        del symbol, kwargs
        dates = [datetime(2024, 12, 20) + timedelta(days=i) for i in range(28)]
        dates = pl.Series("date", dates).to_frame().filter(pl.col("date").dt.weekday() <= 5)["date"][:20]
        return pl.DataFrame(
            {
                "date": dates,
                "close": [100.0 + 20.0 * i / (len(dates) - 1) for i in range(len(dates))],
                "volume": [2_000_000] * len(dates),
            },
        )

    def read_fundamentals(self, symbol, **kwargs):
        del symbol, kwargs
        return pl.DataFrame({"date": ["2024-12-20", "2025-01-02"], "market_cap": [40e9, 60e9]}).with_columns(
            pl.col("date").str.to_datetime()
        )

    def read_profile(self, symbol, **kwargs):
        del symbol, kwargs
        return SimpleNamespace(
            sector="Technology",
            industry="Software",
            exchange="NASDAQ",
            country="US",
            fetched_at="2025-01-20T12:00:00Z",
        )


def test_build_security_context_panel_adds_point_in_time_and_profile_dimensions():
    panel = build_security_context_panel(
        ["aapl"],
        spec=SecurityContextSpec(volatility_window=4, liquidity_window=4),
        warehouse=_Warehouse(),
    )

    assert panel["symbol"].eq("AAPL").all()
    assert set(panel["sector"]) == {"Technology"}
    assert set(panel["industry"]) == {"Software"}
    assert set(panel["classification_temporality"]) == {"latest_known_applied_historically"}
    assert panel.filter(pl.col("date") < pl.datetime(2025, 1, 2))["market_cap_bucket"].eq("mid").all()
    assert panel.filter(pl.col("date") >= pl.datetime(2025, 1, 2))["market_cap_bucket"].eq("large").all()
    assert panel["calendar_year"].n_unique() == 2
    assert panel["adv_dollars"].drop_nulls().gt(0).all()
    assert panel["volatility_regime"].is_not_null().any()


def test_build_security_context_panel_returns_defined_empty_schema():
    class EmptyWarehouse(_Warehouse):
        def read_prices(self, symbol, **kwargs):
            return pl.DataFrame()

    panel = build_security_context_panel(["NONE"], warehouse=EmptyWarehouse())

    assert panel.is_empty()
    assert {"sector", "industry", "calendar_year", "market_cap_bucket"}.issubset(panel.columns)
