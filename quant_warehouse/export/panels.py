from __future__ import annotations

from typing import Sequence

import pandas as pd

from quant_warehouse.warehouse.api import Warehouse


def price_panel(
    warehouse: Warehouse,
    symbols: Sequence[str],
    *,
    provider: str = "yfinance",
    field: str = "close",
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Wide daily price matrix suitable for VectorBT or similar engines."""
    columns: dict[str, pd.Series] = {}
    for symbol in symbols:
        prices = warehouse.read_prices(symbol, provider=provider, start=start, end=end)
        if prices.empty or field not in prices.columns:
            continue
        columns[symbol.upper()] = prices[field]
    if not columns:
        return pd.DataFrame()
    return pd.DataFrame(columns).sort_index()