from __future__ import annotations

import pandas as pd

from quant_warehouse.ingest.credentials import configure_openbb_credentials


def fetch_etf_universe(*, provider: str = "fmp", query: str = "") -> list[str]:
    """Return ETF and mutual-fund symbols from OpenBB etf.search."""
    configure_openbb_credentials()
    from openbb import obb

    result = obb.etf.search(query=query, provider=str(provider or "fmp").strip().lower())
    frame = result.to_df()
    if frame is None or frame.empty or "symbol" not in frame.columns:
        return []
    symbols = (
        frame["symbol"]
        .astype(str)
        .str.strip()
        .str.upper()
        .replace("", pd.NA)
        .dropna()
        .unique()
        .tolist()
    )
    return sorted(symbols)