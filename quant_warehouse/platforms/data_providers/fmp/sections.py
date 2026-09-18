from __future__ import annotations

from quant_warehouse.warehouse.sections import ETF_FUNDAMENTAL_SECTIONS, fundamental_period_for_section

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
    "ownership_institutional",
    "estimates_historical",
    "ratings_historical",
)

FMP_EXTENDED_EQUITY_SECTIONS: tuple[str, ...] = (
    "historical_market_cap",
    "esg_score",
    "management_compensation",
    "management",
    "filings",
    "compare_peers",
    "estimates_consensus",
    "estimates_forward_eps",
    "estimates_forward_ebitda",
    "estimates_price_target",
    "ownership_insider_trading",
    "ownership_government_trades",
    "ownership_share_statistics",
)

FMP_ALL_EQUITY_SECTIONS: tuple[str, ...] = (
    *FMP_HISTORICAL_EQUITY_SECTIONS,
    *FMP_EXTENDED_EQUITY_SECTIONS,
)

FMP_HISTORICAL_ETF_SECTIONS: tuple[str, ...] = ETF_FUNDAMENTAL_SECTIONS


# Fundamental sources read by the issuer feature families and sparse events.
FMP_ISSUER_MODEL_SECTIONS: tuple[str, ...] = (
    *FMP_HISTORICAL_EQUITY_SECTIONS,
    "historical_market_cap",
    "esg_score",
    "ownership_insider_trading",
    "ownership_government_trades",
    "estimates_price_target",
)


def fmp_issuer_model_sections(period: str) -> tuple[str, ...]:
    """Refresh all sources quarterly, then only annual datasets on the annual pass."""
    if period not in {"quarter", "annual"}:
        raise ValueError("Issuer model refresh period must be quarter or annual")
    return tuple(section for section in FMP_ISSUER_MODEL_SECTIONS
                 if period == "quarter" or fundamental_period_for_section(section, preferred=period) == "annual")
