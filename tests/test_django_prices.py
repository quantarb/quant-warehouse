from pathlib import Path

import pandas as pd

from quant_warehouse.ingest.django_prices import payloads_to_price_frame
from quant_warehouse.ingest.normalize import normalize_prices


def test_django_fmp_payload_normalizes_to_shared_schema():
    raw = payloads_to_price_frame(
        [
            {
                "symbol": "AAPL",
                "date": "2023-03-06",
                "adjOpen": 151.44,
                "adjHigh": 153.91,
                "adjLow": 151.12,
                "adjClose": 151.48,
                "volume": 87558028,
            }
        ]
    )
    out = normalize_prices(raw, provider="fmp")
    assert len(out) == 1
    assert out.index[0] == pd.Timestamp("2023-03-06")
    assert "close" in out.columns
    assert float(out.loc[pd.Timestamp("2023-03-06"), "close"]) == 151.48