from __future__ import annotations

from quant_warehouse.warehouse.sections import ETF_FUNDAMENTAL_SECTIONS

LEGACY_FMP_SECTION_MAP: dict[str, str] = {
    "income_statement": "income",
    "balance_sheet": "balance",
    "cash_flow": "cash",
    "key_metrics": "metrics",
    "ratios": "ratios",
    "income_statement_growth": "income_growth",
    "balance_sheet_growth": "balance_growth",
    "cash_flow_growth": "cash_growth",
    "dividends": "dividends",
    "splits": "historical_splits",
    "earnings": "earnings",
    "financial_growth": "financial_growth",
    "senate_trading": "senate_trading",
}

FMP_HISTORICAL_EQUITY_SECTIONS: tuple[str, ...] = (
    "income",
    "balance",
    "cash",
    "metrics",
    "ratios",
    "income_growth",
    "balance_growth",
    "cash_growth",
    "dividends",
    "historical_eps",
    "historical_splits",
    "revenue_per_geography",
    "revenue_per_segment",
    "employee_count",
)

FMP_EXTENDED_EQUITY_SECTIONS: tuple[str, ...] = (
    "historical_market_cap",
    "esg_score",
    "management_compensation",
    "management",
    "filings",
    "compare_peers",
    "estimates_historical",
    "estimates_consensus",
    "estimates_forward_eps",
    "estimates_forward_ebitda",
    "estimates_price_target",
    "ownership_insider_trading",
    "ownership_government_trades",
    "ownership_institutional",
    "ownership_share_statistics",
)

FMP_ALL_EQUITY_SECTIONS: tuple[str, ...] = (
    *FMP_HISTORICAL_EQUITY_SECTIONS,
    *FMP_EXTENDED_EQUITY_SECTIONS,
)

FMP_HISTORICAL_ETF_SECTIONS: tuple[str, ...] = ETF_FUNDAMENTAL_SECTIONS
