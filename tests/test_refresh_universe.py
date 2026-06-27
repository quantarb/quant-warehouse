from __future__ import annotations

from datetime import date

from quant_warehouse.catalog.store import SectionState
from quant_warehouse.refresh.universe import refresh_universe_prices


class FakeWarehouse:
    def __init__(self, catalog: FakeCatalog) -> None:
        self.catalog = catalog
        self.refresh_calls: list[tuple[str, str]] = []

    def refresh_prices(self, symbol, *, providers, **kwargs):
        provider = providers[0]
        self.refresh_calls.append((symbol, provider))
        max_dates = {
            ("AAPL", "fmp"): "2026-06-15",
            ("AAPL", "yfinance"): "2026-06-17",
            ("MSFT", "fmp"): "2026-06-17",
            ("MSFT", "yfinance"): "2026-06-17",
            ("STALE", "fmp"): "2026-06-10",
            ("STALE", "yfinance"): "2026-06-10",
        }
        max_date = max_dates.get((symbol, provider), "2026-06-17")
        self.catalog.states[(symbol, "prices", provider)] = SectionState(
            symbol=symbol,
            section="prices",
            provider=provider,
            min_date="2020-01-02",
            max_date=max_date,
            row_count=100,
            columns_present=("close",),
            last_fetched_at="2026-06-18T00:00:00+00:00",
        )
        return {
            provider: {
                "rows": 100,
                "fetched_rows": 4,
                "max_date": max_date,
                "fetch_start": "2026-06-01",
            }
        }


class FakeCatalog:
    def __init__(self, states: dict[tuple[str, str, str], SectionState] | None = None) -> None:
        self.states = dict(states or {})

    def get(self, *, symbol: str, section: str, provider: str) -> SectionState | None:
        return self.states.get((symbol.upper(), section, provider))

    def equity_historical_start(self, symbol: str) -> str:
        return "1900-01-01"

    def resolve_equity_ipo_date(self, symbol: str):
        return None


def test_refresh_universe_prices_skips_when_any_provider_fresh():
    catalog = FakeCatalog(
        {
            ("AAPL", "prices", "fmp"): SectionState(
                symbol="AAPL",
                section="prices",
                provider="fmp",
                min_date="2020-01-02",
                max_date="2026-06-15",
                row_count=100,
                columns_present=("close",),
                last_fetched_at="2026-06-18T00:00:00+00:00",
            ),
            ("AAPL", "prices", "yfinance"): SectionState(
                symbol="AAPL",
                section="prices",
                provider="yfinance",
                min_date="2020-01-02",
                max_date="2026-06-17",
                row_count=100,
                columns_present=("close",),
                last_fetched_at="2026-06-18T00:00:00+00:00",
            ),
        }
    )
    warehouse = FakeWarehouse(catalog)
    target = date(2026, 6, 17)
    results = refresh_universe_prices(
        warehouse,  # type: ignore[arg-type]
        ["AAPL"],
        providers=["fmp", "yfinance"],
        target_end_date=target,
        skip_recent_hours=0,
    )
    assert warehouse.refresh_calls == []
    assert len(results) == 1
    assert results[0]["status"] == "skipped_fresh"


def test_refresh_universe_prices_falls_back_to_second_provider():
    catalog = FakeCatalog(
        {
            ("STALE", "prices", "fmp"): SectionState(
                symbol="STALE",
                section="prices",
                provider="fmp",
                min_date="2020-01-02",
                max_date="2026-06-10",
                row_count=100,
                columns_present=("close",),
                last_fetched_at="2026-06-18T00:00:00+00:00",
            ),
            ("STALE", "prices", "yfinance"): SectionState(
                symbol="STALE",
                section="prices",
                provider="yfinance",
                min_date="2020-01-02",
                max_date="2026-06-10",
                row_count=100,
                columns_present=("close",),
                last_fetched_at="2026-06-18T00:00:00+00:00",
            ),
        }
    )
    warehouse = FakeWarehouse(catalog)
    target = date(2026, 6, 17)
    results = refresh_universe_prices(
        warehouse,  # type: ignore[arg-type]
        ["STALE"],
        providers=["fmp", "yfinance"],
        target_end_date=target,
        skip_recent_hours=0,
    )
    assert warehouse.refresh_calls == [("STALE", "fmp"), ("STALE", "yfinance")]
    assert results[-1]["status"] == "still_stale"