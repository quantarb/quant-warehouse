from __future__ import annotations

# Providers supported by OpenBB equity.price.historical (SEC has no daily OHLCV route).
PRICE_PROVIDERS: tuple[str, ...] = ("fmp", "yfinance", "tiingo", "intrinio")
DEFAULT_PRICE_PROVIDERS: tuple[str, ...] = ("fmp", "yfinance", "tiingo")

FUNDAMENTAL_PROVIDERS: tuple[str, ...] = ("fmp", "yfinance", "sec", "intrinio", "tiingo")
DEFAULT_FUNDAMENTAL_PROVIDERS: tuple[str, ...] = ("fmp", "yfinance", "sec")

# Required OpenBB extension packages (equity + macro + alt data).
OPENBB_EXTENSIONS: tuple[str, ...] = (
    "openbb-fmp",
    "openbb-yfinance",
    "openbb-sec",
    "openbb-tiingo",
    "openbb-intrinio",
    "openbb-equity",
    "openbb-etf",
    "openbb-crypto",
    "openbb-currency",
    "openbb-derivatives",
    "openbb-economy",
    "openbb-index",
    "openbb-news",
    "openbb-regulators",
    "openbb-fred",
    "openbb-bls",
    "openbb-oecd",
    "openbb-tradingeconomics",
    "openbb-government-us",
    "openbb-federal-reserve",
    "openbb-fixedincome",
    "openbb-commodity",
    "openbb-benzinga",
    "openbb-congress-gov",
    "openbb-cftc",
    "openbb-imf",
    "openbb-us-eia",
    "openbb-econdb",
    "openbb-platform-api",
)


def validate_price_provider(provider: str) -> str:
    key = provider.strip().lower()
    if key not in PRICE_PROVIDERS:
        allowed = ", ".join(PRICE_PROVIDERS)
        raise ValueError(f"Unknown price provider '{provider}'. Supported: {allowed}")
    return key


def validate_fundamental_provider(provider: str) -> str:
    key = provider.strip().lower()
    if key not in FUNDAMENTAL_PROVIDERS:
        allowed = ", ".join(FUNDAMENTAL_PROVIDERS)
        raise ValueError(f"Unknown fundamental provider '{provider}'. Supported: {allowed}")
    return key
