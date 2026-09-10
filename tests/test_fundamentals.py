from pathlib import Path

import polars as pl
from datetime import datetime

from quant_warehouse.catalog.store import CatalogStore
from quant_warehouse.config import WarehouseConfig
from quant_warehouse.ingest.normalize import normalize_vendor_frame
from quant_warehouse.warehouse.backend import ArcticBackend
from quant_warehouse.warehouse.fundamentals import FundamentalsStore
from quant_warehouse.warehouse.sections import fundamental_library
from quant_warehouse.warehouse.storage import provider_library


def test_normalize_vendor_frame_without_provider_prefix():
    raw = pl.DataFrame(
        {
            "period_ending": [datetime(2023, 12, 31), datetime(2024, 12, 31)],
            "total_revenue": [100.0, 120.0],
            "symbol": ["AAPL", "AAPL"],
        }
    )
    out = normalize_vendor_frame(raw, provider="fmp", vendor_only_prefix=None)
    assert "total_revenue" in out.columns
    assert "fmp__total_revenue" not in out.columns
    assert "period_ending" in out.columns
    assert len(out) == 2


def test_fundamentals_store_per_section_libraries(tmp_path: Path):
    config = WarehouseConfig(
        home=tmp_path / "home",
        arctic_uri=f"lmdb://{tmp_path / 'arctic'}",
        catalog_path=tmp_path / "catalog.sqlite",
    )
    backend = ArcticBackend(config.arctic_uri)
    catalog = CatalogStore(config.catalog_path)
    store = FundamentalsStore(config, backend=backend, catalog=catalog)

    income = pl.DataFrame({"total_revenue": [100.0, 120.0], "period_ending": [datetime(2023, 12, 31), datetime(2024, 12, 31)]})
    store.ingest_frame("AAPL", section="income", provider="fmp", frame=income, merge=False)

    balance = pl.DataFrame({"total_assets": [500.0, 550.0], "period_ending": [datetime(2023, 12, 31), datetime(2024, 12, 31)]})
    store.ingest_frame("AAPL", section="balance", provider="fmp", frame=balance, merge=False)

    income_lib = fundamental_library("income")
    balance_lib = fundamental_library("balance")
    income_vendor_lib = provider_library(income_lib, "fmp")
    balance_vendor_lib = provider_library(balance_lib, "fmp")
    assert backend.read(income_vendor_lib, "AAPL__fmp") is not None
    assert backend.read(balance_vendor_lib, "AAPL__fmp") is not None
    assert backend.read(income_lib, "AAPL__fmp") is None
    assert backend.read(balance_lib, "AAPL__fmp") is None
    assert backend.read(income_vendor_lib, "AAPL__fmp").shape[1] == 2
    assert backend.read(balance_vendor_lib, "AAPL__fmp").shape[1] == 2

    out = store.read("AAPL", section="income", provider="fmp")
    assert out.filter(pl.col("period_ending") == datetime(2024, 12, 31)).item(0, "total_revenue") == 120.0

    rows = catalog.list_symbol("AAPL")
    sections = {row.section for row in rows}
    assert "income" in sections
    assert "balance" in sections


def test_fundamental_library_names():
    assert fundamental_library("income") == "fundamental_income"
    assert fundamental_library("income", "quarter") == "fundamental_income_quarter"
    assert fundamental_library("income", "annual") == "fundamental_income_annual"
    assert fundamental_library("etf_holdings") == "etf_holdings"


def test_fundamentals_store_keeps_quarterly_and_annual_refreshes_separate(tmp_path: Path, monkeypatch):
    config = WarehouseConfig(
        home=tmp_path / "home",
        arctic_uri=f"lmdb://{tmp_path / 'arctic'}",
        catalog_path=tmp_path / "catalog.sqlite",
    )
    backend = ArcticBackend(config.arctic_uri)
    catalog = CatalogStore(config.catalog_path)
    store = FundamentalsStore(config, backend=backend, catalog=catalog)

    def fake_fetch(section, *, symbol, provider, **kwargs):
        assert section == "income"
        assert kwargs["limit"] == 1000
        period = kwargs["period"]
        value = 10.0 if period == "quarter" else 100.0
        frame = pl.DataFrame({"total_revenue": [value], "period_ending": [datetime(2024, 3, 31) if period == "quarter" else datetime(2024, 12, 31)]})
        return frame

    monkeypatch.setattr("quant_warehouse.warehouse.fundamentals.fetch_dataframe", fake_fetch)
    stats = store.refresh(
        "AAPL",
        sections=["income"],
        providers=["fmp"],
        period=("quarter", "annual"),
    )

    assert stats == {"income_quarter:fmp": 1, "income_annual:fmp": 1}
    quarter = store.read("AAPL", section="income", provider="fmp", period="quarter")
    annual = store.read("AAPL", section="income", provider="fmp", period="annual")
    assert quarter.item(0, "total_revenue") == 10.0
    assert annual.item(0, "total_revenue") == 100.0
    assert {row.section for row in catalog.list_symbol("AAPL")} == {"income_quarter", "income_annual"}


def test_statement_refresh_preserves_history_and_filters_source_dates(tmp_path, monkeypatch):
    config = WarehouseConfig(home=tmp_path / 'home', arctic_uri=f"lmdb://{tmp_path / 'arctic'}", catalog_path=tmp_path / 'catalog.sqlite')
    store = FundamentalsStore(config)
    dates = [datetime(2023, 3, 31), datetime(2024, 3, 31)]
    monkeypatch.setattr('quant_warehouse.warehouse.fundamentals.fetch_dataframe',
                        lambda *args, **kwargs: pl.DataFrame({'period_ending': dates, 'revenue': [1.] * len(dates)}))
    store.refresh('A', sections=['income'], providers=['fmp'], period='quarter')
    dates[:] = [datetime(2025, 3, 31)]
    store.refresh('A', sections=['income'], providers=['fmp'], period='quarter')
    assert store.read('A', section='income', period='quarter').height == 3
    selected = store.read('A', section='income', period='quarter', start='2024-01-01', end='2024-12-31')
    assert selected['period_ending'].to_list() == [datetime(2024, 3, 31)]


def test_statement_merge_preserves_dates_when_source_column_changes():
    from quant_warehouse.warehouse.fundamentals import _merge_observations
    old = pl.DataFrame({'date': [datetime(2023, 3, 31), datetime(2024, 3, 31)], 'revenue': [1., 2.]})
    new = pl.DataFrame({'period_ending': [datetime(2024, 3, 31)], 'revenue': [3.]})
    merged = _merge_observations(old, new)
    assert merged['period_ending'].to_list() == [datetime(2023, 3, 31), datetime(2024, 3, 31)]
    assert merged['revenue'].to_list() == [1., 3.]
    assert 'date' not in merged.columns
    assert _merge_observations(merged, new.head(0)).equals(merged)


def test_ingest_revision_uses_source_dates_and_updates_catalog(tmp_path):
    config = WarehouseConfig(home=tmp_path / 'home', arctic_uri=f"lmdb://{tmp_path / 'arctic'}", catalog_path=tmp_path / 'catalog.sqlite')
    store = FundamentalsStore(config)
    for date, value in [(datetime(2023, 3, 31), 1.), (datetime(2024, 3, 31), 2.), (datetime(2024, 3, 31), 3.)]:
        stats = store.ingest_frame('A', section='income', period='quarter', provider='fmp',
                                  frame=pl.DataFrame({'period_ending': [date], 'revenue': [value]}))
    assert stats['rows'] == 2
    assert stats['min_date'] == '2023-03-31'
    assert stats['max_date'] == '2024-03-31'
    assert store.read('A', section='income', period='quarter', start='2024-03-31', end='2024-03-31')['revenue'].to_list() == [3.]
