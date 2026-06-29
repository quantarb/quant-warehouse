import pandas as pd

from quant_warehouse.ingest.equity_calendar_fetch import normalize_equity_calendar_frame
from quant_warehouse.platforms.data_providers.fmp.sections import (
    FMP_ALL_EQUITY_SECTIONS,
    FMP_EXTENDED_EQUITY_SECTIONS,
    FMP_HISTORICAL_ETF_SECTIONS,
)


def test_normalize_equity_calendar_earnings_frame():
    raw = pd.DataFrame(
        {
            "report_date": ["2024-01-31", "2024-01-30"],
            "symbol": ["AAPL", "MSFT"],
            "eps_actual": [1.2, 2.1],
        }
    )
    out = normalize_equity_calendar_frame(raw, section="equity_calendar_earnings")
    assert len(out) == 2
    assert out.index.name == "report_date"
    assert "symbol" in out.columns


def test_fmp_all_equity_sections_include_extended():
    assert "historical_market_cap" in FMP_ALL_EQUITY_SECTIONS
    assert "ownership_insider_trading" in FMP_EXTENDED_EQUITY_SECTIONS
    assert "transcript" not in FMP_EXTENDED_EQUITY_SECTIONS


def test_fmp_historical_etf_sections_cover_composition():
    assert "etf_holdings" in FMP_HISTORICAL_ETF_SECTIONS
    assert "etf_nport_disclosure" in FMP_HISTORICAL_ETF_SECTIONS
