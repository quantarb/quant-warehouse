from pathlib import Path

import pandas as pd

from quant_warehouse.config import WarehouseConfig
from quant_warehouse.warehouse.api import Warehouse


def test_company_news_store_upserts_and_reads_point_in_time(tmp_path: Path):
    config = WarehouseConfig(
        home=tmp_path / "home",
        arctic_uri=f"lmdb://{tmp_path / 'arctic'}",
        catalog_path=tmp_path / "catalog.sqlite",
    )
    warehouse = Warehouse(config)
    frame = pd.DataFrame(
        {
            "symbol": ["AAPL", "AAPL", "MSFT"],
            "observation_date": ["2024-01-02", "2024-01-03", "2024-01-02"],
            "published_at": [
                "2024-01-02T12:00:00Z",
                "2024-01-03T13:00:00Z",
                "2024-01-02T14:00:00Z",
            ],
            "title": ["one", "two", "three"],
            "url": ["https://one", "https://two", "https://three"],
            "provider": ["fmp", "fmp", "fmp"],
            "fetched_at": [
                "2026-01-01T00:00:00Z",
                "2026-01-02T00:00:00+00:00",
                "2026-01-03T00:00:00Z",
            ],
        }
    )

    counts = warehouse.news.import_frame(frame)
    warehouse.news.import_frame(frame)

    assert counts == {"AAPL": 2, "MSFT": 1}
    assert len(warehouse.read_news("AAPL")) == 2
    selected = warehouse.news.read("AAPL", observation_dates=["2024-01-03"])
    assert selected["title"].tolist() == ["two"]
    state = warehouse.catalog.get(symbol="AAPL", section="company_news", provider="fmp")
    assert state is not None
    assert state.row_count == 2
