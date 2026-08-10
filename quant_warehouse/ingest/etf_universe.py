from __future__ import annotations

import polars as pl


from quant_warehouse.ingest.credentials import configure_openbb_credentials


def fetch_etf_universe(*, provider: str = "fmp", query: str = "") -> list[str]:
    """Return ETF and mutual-fund symbols from OpenBB etf.search."""
    configure_openbb_credentials()
    from openbb import obb

    result = obb.etf.search(query=query, provider=str(provider or "fmp").strip().lower())
    frame = result.to_polars()
    if frame is None or frame.is_empty() or "symbol" not in frame.columns:
        return []
    symbols = frame.select(pl.col("symbol").cast(pl.String).str.strip_chars().str.to_uppercase().alias("symbol")).filter(pl.col("symbol") != "")["symbol"].unique().to_list()
    return sorted(symbols)
