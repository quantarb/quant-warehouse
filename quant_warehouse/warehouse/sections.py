from __future__ import annotations

EQUITY_PRICES_LIBRARY = "prices"
ETF_PRICES_LIBRARY = "etf_prices"

EQUITY_PRICES_SECTION = "prices"
ETF_PRICES_SECTION = "etf_prices"
EQUITY_PROFILE_SECTION = "profile"
ETF_PROFILE_SECTION = "etf_profile"

# OpenBB equity.fundamental.* historical routes (one Arctic library per section).
EQUITY_FUNDAMENTAL_SECTIONS: tuple[str, ...] = (
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
    "reported_financials",
    "revenue_per_geography",
    "revenue_per_segment",
    "trailing_dividend_yield",
    "employee_count",
    "historical_market_cap",
    "esg_score",
    "management_compensation",
    "management",
    "filings",
    "transcript",
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

# Market-wide equity event calendars (OpenBB equity.calendar.*).
EQUITY_CALENDAR_SECTIONS: tuple[str, ...] = (
    "equity_calendar_earnings",
    "equity_calendar_dividend",
    "equity_calendar_splits",
    "equity_calendar_ipo",
)

# OpenBB etf.* composition / disclosure routes (separate from etf_prices / etf_profile).
ETF_FUNDAMENTAL_SECTIONS: tuple[str, ...] = (
    "etf_holdings",
    "etf_sectors",
    "etf_countries",
    "etf_equity_exposure",
    "etf_nport_disclosure",
    "etf_price_performance",
)

# Cross-sectional snapshots stored as dated panels (as-of index for Arctic).
DATED_SNAPSHOT_SECTIONS: frozenset[str] = frozenset(
    {
        "etf_holdings",
        "etf_sectors",
        "etf_countries",
        "etf_equity_exposure",
        "etf_price_performance",
        "management",
        "compare_peers",
        "estimates_consensus",
        "ownership_institutional",
        "ownership_share_statistics",
    }
)

# Backward-compatible alias.
ETF_COMPOSITION_SECTIONS = DATED_SNAPSHOT_SECTIONS

# FMP historical sections without a distinct OpenBB route name.
EXTENDED_FUNDAMENTAL_SECTIONS: tuple[str, ...] = (
    "earnings",
    "financial_growth",
    "senate_trading",
)

DJANGO_HISTORICAL_SECTION_MAP: dict[str, str] = {
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

# No Django-only sections are currently retained.
DJANGO_ONLY_FUNDAMENTAL_SECTIONS: frozenset[str] = frozenset()

ALL_FUNDAMENTAL_SECTIONS: frozenset[str] = frozenset(
    (
        *EQUITY_FUNDAMENTAL_SECTIONS,
        *ETF_FUNDAMENTAL_SECTIONS,
        *EXTENDED_FUNDAMENTAL_SECTIONS,
    )
)

# Sections that accept OpenBB `period` (annual / quarter / quarterly).
PERIOD_FUNDAMENTAL_SECTIONS: frozenset[str] = frozenset(
    {
        "income",
        "balance",
        "cash",
        "metrics",
        "ratios",
        "income_growth",
        "balance_growth",
        "cash_growth",
        "reported_financials",
        "revenue_per_geography",
        "revenue_per_segment",
    }
)

# Cross-sectional snapshots — full replace on refresh, not gap-fill merge.
SNAPSHOT_FUNDAMENTAL_SECTIONS: frozenset[str] = frozenset()

# Repeated cross-sections keyed by filing/as-of date (multiple rows per date).
PANEL_FUNDAMENTAL_SECTIONS: frozenset[str] = frozenset(
    {
        "etf_nport_disclosure",
        "revenue_per_geography",
        "revenue_per_segment",
        "ownership_insider_trading",
        "ownership_government_trades",
        "estimates_price_target",
        "filings",
        "transcript",
        "equity_calendar_earnings",
        "equity_calendar_dividend",
        "equity_calendar_splits",
        "equity_calendar_ipo",
    }
)

# Sections where OpenBB/FMP accepts start_date/end_date for bounded fetches.
DATE_WINDOW_SECTIONS: frozenset[str] = frozenset(
    {
        "prices",
        "etf_prices",
        "crypto_prices",
        "currency_prices",
        "index_prices",
        "historical_market_cap",
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
        "employee_count",
        "esg_score",
        "management_compensation",
        "estimates_historical",
        "estimates_forward_eps",
        "estimates_forward_ebitda",
        "ownership_insider_trading",
        "ownership_government_trades",
        "filings",
        "estimates_price_target",
        "revenue_per_geography",
        "revenue_per_segment",
    }
)

LEGACY_FUNDAMENTALS_LIBRARY = "fundamentals"

# Absolute historical floor. Equities use max(MIN_HISTORICAL_DATE, ipo_date); macro/ETF use this date.
MIN_HISTORICAL_DATE = "1900-01-01"

# OpenBB/FMP historical equity routes (excludes snapshots and non-FMP-only routes).
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

# Extended per-symbol FMP equity routes (estimates, ownership, ESG, etc.).
# Transcript is stored under EQUITY_FUNDAMENTAL_SECTIONS but fetched via refresh_transcripts().
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

# All per-symbol FMP equity routes for comprehensive backfill.
FMP_ALL_EQUITY_SECTIONS: tuple[str, ...] = (
    *FMP_HISTORICAL_EQUITY_SECTIONS,
    *FMP_EXTENDED_EQUITY_SECTIONS,
)

# OpenBB/FMP historical ETF routes.
FMP_HISTORICAL_ETF_SECTIONS: tuple[str, ...] = ETF_FUNDAMENTAL_SECTIONS

EQUITY_CALENDAR_EARNINGS_SECTION = "equity_calendar_earnings"
EQUITY_CALENDAR_DIVIDEND_SECTION = "equity_calendar_dividend"
EQUITY_CALENDAR_SPLITS_SECTION = "equity_calendar_splits"
EQUITY_CALENDAR_IPO_SECTION = "equity_calendar_ipo"

EQUITY_CALENDAR_EARNINGS_LIBRARY = "equity_calendar_earnings"
EQUITY_CALENDAR_DIVIDEND_LIBRARY = "equity_calendar_dividend"
EQUITY_CALENDAR_SPLITS_LIBRARY = "equity_calendar_splits"
EQUITY_CALENDAR_IPO_LIBRARY = "equity_calendar_ipo"

EQUITY_CALENDAR_BUNDLE_SYMBOL = "EQUITY_CALENDAR"

# Non-equity market price sections (OpenBB crypto/currency/index routes).
CRYPTO_PRICES_SECTION = "crypto_prices"
CURRENCY_PRICES_SECTION = "currency_prices"
INDEX_PRICES_SECTION = "index_prices"

CRYPTO_PRICES_LIBRARY = "crypto_prices"
CURRENCY_PRICES_LIBRARY = "currency_prices"
INDEX_PRICES_LIBRARY = "index_prices"

MARKET_PRICE_SECTIONS: dict[str, str] = {
    CRYPTO_PRICES_SECTION: CRYPTO_PRICES_LIBRARY,
    CURRENCY_PRICES_SECTION: CURRENCY_PRICES_LIBRARY,
    INDEX_PRICES_SECTION: INDEX_PRICES_LIBRARY,
}

DEFAULT_CRYPTO_SYMBOLS: tuple[str, ...] = (
    "BTCUSD",
    "ETHUSD",
    "SOLUSD",
    "XRPUSD",
    "ADAUSD",
    "DOGEUSD",
    "BNBUSD",
    "LTCUSD",
)

DEFAULT_CURRENCY_SYMBOLS: tuple[str, ...] = (
    "EURUSD",
    "GBPUSD",
    "USDJPY",
    "AUDUSD",
    "USDCAD",
    "USDCHF",
    "NZDUSD",
)

DEFAULT_INDEX_SYMBOLS: tuple[str, ...] = (
    "^GSPC",
    "^DJI",
    "^IXIC",
    "^VIX",
    "^RUT",
)

# Extended FMP macro via economic-indicators + OpenBB economy/fixedincome routes.
MACRO_ECONOMIC_SECTION = "macro_economic"
MACRO_TREASURY_SECTION = "macro_treasury"
MACRO_YIELD_CURVE_SECTION = "macro_yield_curve"
MACRO_CALENDAR_SECTION = "macro_calendar"
MACRO_RISK_PREMIUM_SECTION = "macro_risk_premium"

MACRO_ECONOMIC_LIBRARY = "macro_economic"
MACRO_TREASURY_LIBRARY = "macro_treasury"
MACRO_YIELD_CURVE_LIBRARY = "macro_yield_curve"
MACRO_CALENDAR_LIBRARY = "macro_calendar"
MACRO_RISK_PREMIUM_LIBRARY = "macro_risk_premium"

TREASURY_BUNDLE_SYMBOL = "TREASURY_CURVE"
YIELD_CURVE_BUNDLE_SYMBOL = "YIELD_CURVE"
MACRO_CALENDAR_SYMBOL = "MACRO_CALENDAR"
RISK_PREMIUM_SYMBOL = "RISK_PREMIUM"

DEFAULT_ECONOMIC_SERIES: tuple[str, ...] = (
    "GDP",
    "CPI",
    "unemploymentRate",
    "inflationRate",
    "federalFunds",
    "retailSales",
    "industrialProductionTotalIndex",
    "totalNonfarmPayroll",
    "initialClaims",
    "newPrivatelyOwnedHousingUnitsStartedTotalUnits",
)

EXTENDED_ECONOMIC_SERIES: tuple[str, ...] = DEFAULT_ECONOMIC_SERIES

_PERIOD_ALIASES: dict[str, str] = {
    "quarter": "quarter",
    "quarterly": "quarter",
    "annual": "annual",
    "year": "annual",
    "yearly": "annual",
}


def normalize_fundamental_period(period: str) -> str:
    key = str(period or "quarter").strip().lower()
    return _PERIOD_ALIASES.get(key, "quarter")


def fundamental_period_for_section(section: str, *, preferred: str = "quarter") -> str | None:
    """Return the OpenBB period for a section, preferring quarterly when supported."""
    if section not in PERIOD_FUNDAMENTAL_SECTIONS:
        return None
    return normalize_fundamental_period(preferred)


def fundamental_library(section: str) -> str:
    """Arctic library name mirroring the OpenBB route grouping."""
    if section.startswith("etf_"):
        return section
    return f"fundamental_{section}"
