from __future__ import annotations

from typing import Sequence

import polars as pl

from quant_warehouse.warehouse.api import Warehouse


def price_panel(
    warehouse: Warehouse,
    symbols: Sequence[str],
    *,
    provider: str = "yfinance",
    field: str = "close",
    start: str | None = None,
    end: str | None = None,
 ) -> pl.DataFrame:
    """Wide daily price matrix suitable for VectorBT or similar engines."""
    polars_frames: list[pl.DataFrame] = []
    for symbol in symbols:
        prices = warehouse.read_prices(symbol, provider=provider, start=start, end=end)
        if prices.is_empty() or field not in prices.columns:
            continue
        polars_frames.append(
            prices.select([pl.col("date"), pl.col(field).alias(symbol.upper())])
        )
    if not polars_frames:
        return pl.DataFrame()
    out = polars_frames[0]
    for frame in polars_frames[1:]:
        out = out.join(frame, on="date", how="full", coalesce=True)
    return out.sort("date")
