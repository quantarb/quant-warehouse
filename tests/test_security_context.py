from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from quant_warehouse.research_tools.security_context import (
    SecurityContextSpec,
    build_security_context_panel,
)


class _Warehouse:
    def read_prices(self, symbol, **kwargs):
        del symbol, kwargs
        index = pd.date_range("2024-12-20", periods=20, freq="B")
        return pd.DataFrame(
            {
                "close": np.linspace(100.0, 120.0, len(index)),
                "volume": np.full(len(index), 2_000_000),
            },
            index=index,
        )

    def read_fundamentals(self, symbol, **kwargs):
        del symbol, kwargs
        return pd.DataFrame(
            {"market_cap": [40e9, 60e9]},
            index=pd.to_datetime(["2024-12-20", "2025-01-02"]),
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
    assert panel.loc[panel["date"].lt("2025-01-02"), "market_cap_bucket"].eq("mid").all()
    assert panel.loc[panel["date"].ge("2025-01-02"), "market_cap_bucket"].eq("large").all()
    assert panel["calendar_year"].nunique() == 2
    assert panel["adv_dollars"].dropna().gt(0).all()
    assert panel["volatility_regime"].notna().any()


def test_build_security_context_panel_returns_defined_empty_schema():
    class EmptyWarehouse(_Warehouse):
        def read_prices(self, symbol, **kwargs):
            return pd.DataFrame()

    panel = build_security_context_panel(["NONE"], warehouse=EmptyWarehouse())

    assert panel.empty
    assert {"sector", "industry", "calendar_year", "market_cap_bucket"}.issubset(panel.columns)
