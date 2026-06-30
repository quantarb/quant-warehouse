from __future__ import annotations

import pandas as pd

from quant_warehouse.platforms.data_providers.thetadata.options import (
    OPTIONS_THETADATA_EOD_LIBRARY,
    OPTIONS_THETADATA_PROVIDER,
    download_option_snapshots_for_range,
)
from quant_warehouse.warehouse.storage import provider_library


def test_default_thetadata_option_download_uses_arctic_paths(monkeypatch) -> None:
    def _fake_fetch(symbol, start_date, end_date, **kwargs):
        return pd.DataFrame(
            [
                {
                    "symbol": symbol,
                    "expiration": "2025-01-24",
                    "strike": 230.0,
                    "right": "PUT",
                    "created": f"{pd.Timestamp(start_date).date()} 17:00:00-05:00",
                    "bid": 1.0,
                    "ask": 1.2,
                }
            ]
        )

    written: list[tuple[str, pd.DataFrame]] = []

    class _Backend:
        def read(self, library: str, symbol: str) -> pd.DataFrame | None:
            assert library in {
                OPTIONS_THETADATA_EOD_LIBRARY,
                provider_library(OPTIONS_THETADATA_EOD_LIBRARY, OPTIONS_THETADATA_PROVIDER),
            }
            return None

        def write(self, library: str, symbol: str, df: pd.DataFrame, **kwargs) -> None:
            assert library == provider_library(OPTIONS_THETADATA_EOD_LIBRARY, OPTIONS_THETADATA_PROVIDER)
            written.append((symbol, df))

    monkeypatch.setattr(
        "quant_warehouse.platforms.data_providers.thetadata.options.fetch_option_history_eod",
        _fake_fetch,
    )
    monkeypatch.setattr(
        "quant_warehouse.platforms.data_providers.thetadata.options.open_backend",
        lambda *args, **kwargs: _Backend(),
    )

    manifest = download_option_snapshots_for_range("AAPL", "2025-01-06", "2025-01-06")

    expected_library = provider_library(OPTIONS_THETADATA_EOD_LIBRARY, OPTIONS_THETADATA_PROVIDER)
    assert manifest["paths"] == [f"arctic://{expected_library}/AAPL"]
    assert written
    assert written[0][0] == "AAPL"
    assert "snapshot_date" in written[0][1].columns
