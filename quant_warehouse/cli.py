from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from quant_warehouse.migrate.backfill_macro_alt import backfill_fmp_macro_alt, write_backfill_log as write_macro_alt_log
from quant_warehouse.migrate.backfill_fixes import (
    backfill_calendar_and_etf_composition,
    write_backfill_log as write_backfill_fixes_log,
)
from quant_warehouse.migrate.backfill_fmp_all import backfill_fmp_all, write_backfill_log as write_fmp_all_log
from quant_warehouse.migrate.backfill_missing_fmp import backfill_missing_fmp_historical, write_backfill_log
from quant_warehouse.migrate.backfill_thetadata_options import (
    backfill_thetadata_options,
    log_progress as log_thetadata_options_progress,
    write_backfill_log as write_thetadata_options_log,
)
from quant_warehouse.migrate.separate_fundamentals import separate_legacy_fundamentals
from quant_warehouse.platforms.data_providers.fmp.sections import FMP_HISTORICAL_EQUITY_SECTIONS
from quant_warehouse.warehouse.sections import (
    EQUITY_FUNDAMENTAL_SECTIONS,
    ETF_FUNDAMENTAL_SECTIONS,
    DEFAULT_CRYPTO_SYMBOLS,
    DEFAULT_CURRENCY_SYMBOLS,
    DEFAULT_ECONOMIC_SERIES,
    DEFAULT_INDEX_SYMBOLS,
    MIN_HISTORICAL_DATE,
)
from quant_warehouse.warehouse.api import Warehouse


def _parse_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _profile_payload(row) -> dict[str, object]:
    return {
        "provider": row.provider,
        "source_provider": row.source_provider,
        "company_name": row.company_name,
        "sector": row.sector,
        "industry": row.industry,
        "cik": row.cik,
        "fetched_at": row.fetched_at,
    }


def cmd_refresh(args: argparse.Namespace) -> int:
    wh = Warehouse()
    stats = wh.refresh(
        args.symbol,
        sections=_parse_csv(args.sections),
        providers=_parse_csv(args.providers),
        period=args.period,
    )
    print(json.dumps({"symbol": args.symbol.upper(), "rows": stats}, indent=2))
    return 0


def cmd_refresh_prices(args: argparse.Namespace) -> int:
    wh = Warehouse()
    stats = wh.refresh_prices(
        args.symbol,
        providers=_parse_csv(args.providers),
        start_date=args.start_date,
        end_date=args.end_date,
        full_refresh=args.full_refresh,
    )
    print(json.dumps({"symbol": args.symbol.upper(), "prices": stats}, indent=2))
    return 0


def cmd_refresh_etf_prices(args: argparse.Namespace) -> int:
    wh = Warehouse()
    stats = wh.etf.refresh_prices(
        args.symbol,
        providers=_parse_csv(args.providers),
        start_date=args.start_date,
        end_date=args.end_date,
        full_refresh=args.full_refresh,
    )
    print(json.dumps({"symbol": args.symbol.upper(), "etf_prices": stats}, indent=2))
    return 0


def cmd_refresh_profile(args: argparse.Namespace) -> int:
    wh = Warehouse()
    stats = wh.refresh_profile(args.symbol, provider=args.provider)
    profile = wh.read_profile(args.symbol, provider=args.provider)
    print(
        json.dumps(
            {
                "refresh": stats,
                "profile": _profile_payload(profile) if profile is not None else None,
            },
            indent=2,
        )
    )
    return 0


def cmd_refresh_etf_profile(args: argparse.Namespace) -> int:
    wh = Warehouse()
    stats = wh.etf.refresh_profile(args.symbol, provider=args.provider)
    profile = wh.etf.read_profile(args.symbol, provider=args.provider)
    print(
        json.dumps(
            {
                "refresh": stats,
                "etf_profile": _profile_payload(profile) if profile is not None else None,
            },
            indent=2,
        )
    )
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    wh = Warehouse()
    rows = wh.status(args.symbol)
    storage = {
        "backend": "arctic",
        "arctic_uri": wh.config.arctic_uri,
        "qw_home": str(wh.config.home),
    }
    print(
        json.dumps(
            {
                "symbol": args.symbol.upper(),
                "storage": storage,
                "sections": [
                    {
                        "section": row.section,
                        "provider": row.provider,
                        "min_date": row.min_date,
                        "max_date": row.max_date,
                        "row_count": row.row_count,
                        "columns_present": list(row.columns_present),
                        "last_fetched_at": row.last_fetched_at,
                    }
                    for row in rows
                ],
                "equity_profiles": [_profile_payload(row) for row in wh.profiles.list(args.symbol)],
                "etf_profiles": [_profile_payload(row) for row in wh.catalog.list_etf_profiles(args.symbol)],
            },
            indent=2,
        )
    )
    return 0


def cmd_refresh_fundamentals(args: argparse.Namespace) -> int:
    wh = Warehouse()
    sections = _parse_csv(args.sections) if args.sections else None
    stats = wh.refresh_fundamentals(
        args.symbol,
        sections=sections,
        providers=_parse_csv(args.providers),
        period=args.period,
    )
    print(json.dumps({"symbol": args.symbol.upper(), "fundamentals": stats}, indent=2))
    return 0


def cmd_refresh_etf_fundamentals(args: argparse.Namespace) -> int:
    wh = Warehouse()
    sections = _parse_csv(args.sections) if args.sections else None
    stats = wh.etf.refresh_fundamentals(
        args.symbol,
        sections=sections,
        providers=_parse_csv(args.providers),
        period=args.period,
    )
    print(json.dumps({"symbol": args.symbol.upper(), "etf_fundamentals": stats}, indent=2))
    return 0


def cmd_backfill_macro_alt(args: argparse.Namespace) -> int:
    log_path = Path(args.log).expanduser().resolve()

    def _log(message: str) -> None:
        print(message, flush=True)

    summary = backfill_fmp_macro_alt(
        macro_start_date=args.macro_start_date,
        economic_series=_parse_csv(args.economic_series) if args.economic_series else None,
        include_treasury_rates=not args.skip_treasury,
        include_yield_curve=not args.skip_yield_curve,
        include_calendar=not args.skip_calendar,
        include_risk_premium=not args.skip_risk_premium,
        include_crypto=not args.skip_crypto,
        include_currency=not args.skip_currency,
        include_index=not args.skip_index,
        crypto_symbols=_parse_csv(args.crypto_symbols) if args.crypto_symbols else None,
        currency_symbols=_parse_csv(args.currency_symbols) if args.currency_symbols else None,
        index_symbols=_parse_csv(args.index_symbols) if args.index_symbols else None,
        yield_curve_step_days=int(args.yield_curve_step_days),
        progress_logger=_log,
    )
    write_macro_alt_log(summary, log_path=log_path)
    print(json.dumps(summary, indent=2, default=str))
    return 0


def cmd_backfill_fmp_all(args: argparse.Namespace) -> int:
    log_path = Path(args.log).expanduser().resolve()

    def _log(message: str) -> None:
        print(message, flush=True)

    summary = backfill_fmp_all(
        equity_provider=args.provider,
        etf_provider=args.etf_provider,
        period=args.period,
        calendar_start_date=args.calendar_start_date,
        nport_start_year=int(args.nport_start_year),
        transcript_start_year=int(args.transcript_start_year),
        include_macro=args.include_macro,
        include_prices=not args.skip_prices,
        include_profiles=not args.skip_profiles,
        include_calendars=not args.skip_calendars,
        include_transcripts=args.include_transcripts,
        include_etf_universe=not args.skip_etf_universe,
        skip_equity_core=args.skip_equity_core,
        max_equity_symbols=args.limit,
        max_etf_symbols=args.etf_limit,
        staleness_days=int(args.staleness_days),
        skip_recent_hours=float(args.skip_recent_hours),
        request_sleep_seconds=float(args.request_sleep),
        max_workers=int(args.workers),
        progress_logger=_log,
    )
    write_fmp_all_log(summary, log_path=log_path)
    print(json.dumps(summary, indent=2, default=str))
    return 0


def cmd_backfill_missing_fmp(args: argparse.Namespace) -> int:
    log_path = Path(args.log).expanduser().resolve()

    def _log(message: str) -> None:
        print(message, flush=True)

    summary = backfill_missing_fmp_historical(
        equity_sections=_parse_csv(args.sections) if args.sections else None,
        period=args.period,
        nport_start_year=int(args.nport_start_year),
        include_prices=not args.skip_prices,
        force_macro=args.force_macro,
        macro_start_date=args.macro_start_date,
        max_equity_symbols=args.limit,
        max_etf_symbols=args.etf_limit,
        staleness_days=int(args.staleness_days),
        skip_recent_hours=float(args.skip_recent_hours),
        max_workers=int(args.workers),
        progress_logger=_log,
    )
    write_backfill_log(summary, log_path=log_path)
    print(json.dumps(summary, indent=2, default=str))
    return 0


def cmd_backfill_fixes(args: argparse.Namespace) -> int:
    log_path = Path(args.log).expanduser().resolve()

    def _log(message: str) -> None:
        print(message, flush=True)

    summary = backfill_calendar_and_etf_composition(
        calendar_start_date=args.calendar_start_date,
        full_refresh_earnings=args.full_refresh_earnings,
        calendar_sections=_parse_csv(args.calendar_sections) if args.calendar_sections else None,
        include_calendars=not args.skip_calendars,
        include_etf_composition=not args.skip_etf_composition,
        etf_retry_missing_holdings=args.etf_retry_missing_holdings,
        max_workers=int(args.workers),
        progress_logger=_log,
    )
    write_backfill_fixes_log(summary, log_path=log_path)
    print(json.dumps(summary, indent=2, default=str))
    return 0


def cmd_backfill_thetadata_options(args: argparse.Namespace) -> int:
    log_path = Path(args.log).expanduser().resolve()
    symbols = _parse_csv(args.symbols) if args.symbols else None
    summary = backfill_thetadata_options(
        symbols=symbols,
        source=args.source,
        start_date=args.start_date,
        end_date=args.end_date or None,
        max_dte=int(args.max_dte),
        strike_range=int(args.strike_range),
        limit=args.limit,
        offset=int(args.offset),
        skip_existing=not args.overwrite,
        overwrite=bool(args.overwrite),
        request_sleep=float(args.request_sleep),
        us_only=not args.include_non_us,
        progress_logger=log_thetadata_options_progress,
    )
    write_thetadata_options_log(summary, log_path=log_path)
    print(json.dumps(summary, indent=2, default=str))
    return 0 if int(summary.get("symbols_failed") or 0) == 0 else 1


def cmd_separate_fundamentals(args: argparse.Namespace) -> int:
    symbols = _parse_csv(args.symbols) if args.symbols else None
    sections = _parse_csv(args.sections) if args.sections else None
    stats = separate_legacy_fundamentals(symbols=symbols, sections=sections, dry_run=args.dry_run)
    print(json.dumps({"migrated": len(stats), "results": stats}, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="quant-warehouse")
    sub = parser.add_subparsers(dest="command", required=True)

    refresh_prices = sub.add_parser("refresh-prices", help="Fetch equity OHLCV via OpenBB equity.price.historical")
    refresh_prices.add_argument("symbol")
    refresh_prices.add_argument("--providers", default="fmp,yfinance,tiingo")
    refresh_prices.add_argument("--start-date", default=None)
    refresh_prices.add_argument("--end-date", default=None)
    refresh_prices.add_argument("--full-refresh", action="store_true")
    refresh_prices.set_defaults(func=cmd_refresh_prices)

    refresh_etf_prices = sub.add_parser("refresh-etf-prices", help="Fetch ETF OHLCV via OpenBB etf.historical")
    refresh_etf_prices.add_argument("symbol")
    refresh_etf_prices.add_argument("--providers", default="fmp,yfinance,tiingo")
    refresh_etf_prices.add_argument("--start-date", default=None)
    refresh_etf_prices.add_argument("--end-date", default=None)
    refresh_etf_prices.add_argument("--full-refresh", action="store_true")
    refresh_etf_prices.set_defaults(func=cmd_refresh_etf_prices)

    refresh_profile = sub.add_parser("refresh-profile", help="Fetch equity profile via OpenBB equity.profile")
    refresh_profile.add_argument("symbol")
    refresh_profile.add_argument("--provider", default="yfinance")
    refresh_profile.set_defaults(func=cmd_refresh_profile)

    refresh_etf_profile = sub.add_parser("refresh-etf-profile", help="Fetch ETF profile via OpenBB etf.info")
    refresh_etf_profile.add_argument("symbol")
    refresh_etf_profile.add_argument("--provider", default="yfinance")
    refresh_etf_profile.set_defaults(func=cmd_refresh_etf_profile)

    refresh = sub.add_parser("refresh", help="Fetch and upsert equity data for a symbol")
    refresh.add_argument("symbol")
    refresh.add_argument("--sections", default="prices")
    refresh.add_argument("--providers", default="fmp,yfinance,tiingo")
    refresh.add_argument("--period", default="annual", choices=["annual", "quarter", "quarterly"])
    refresh.set_defaults(func=cmd_refresh)

    default_equity_fundamentals = ",".join(EQUITY_FUNDAMENTAL_SECTIONS)
    refresh_fundamentals = sub.add_parser(
        "refresh-fundamentals",
        help="Fetch equity fundamental routes (one Arctic library per OpenBB section)",
    )
    refresh_fundamentals.add_argument("symbol")
    refresh_fundamentals.add_argument("--sections", default=default_equity_fundamentals)
    refresh_fundamentals.add_argument("--providers", default="fmp,yfinance,sec")
    refresh_fundamentals.add_argument("--period", default="annual", choices=["annual", "quarter", "quarterly"])
    refresh_fundamentals.set_defaults(func=cmd_refresh_fundamentals)

    default_etf_fundamentals = ",".join(ETF_FUNDAMENTAL_SECTIONS)
    refresh_etf_fundamentals = sub.add_parser(
        "refresh-etf-fundamentals",
        help="Fetch ETF composition/disclosure routes (etf.holdings, etf.sectors, ...)",
    )
    refresh_etf_fundamentals.add_argument("symbol")
    refresh_etf_fundamentals.add_argument("--sections", default=default_etf_fundamentals)
    refresh_etf_fundamentals.add_argument("--providers", default="fmp")
    refresh_etf_fundamentals.add_argument("--period", default="annual", choices=["annual", "quarter", "quarterly"])
    refresh_etf_fundamentals.set_defaults(func=cmd_refresh_etf_fundamentals)

    separate_fundamentals = sub.add_parser(
        "separate-fundamentals",
        help="Migrate legacy merged fundamentals library into per-route libraries",
    )
    separate_fundamentals.add_argument("--symbols", default="")
    separate_fundamentals.add_argument("--sections", default="")
    separate_fundamentals.add_argument("--dry-run", action="store_true")
    separate_fundamentals.set_defaults(func=cmd_separate_fundamentals)

    status = sub.add_parser("status", help="Show catalog state for a symbol")
    status.add_argument("symbol")
    status.set_defaults(func=cmd_status)

    backfill_missing = sub.add_parser(
        "backfill-missing-fmp",
        help="Backfill missing OpenBB/FMP historical sections into quant-warehouse",
    )
    backfill_missing.add_argument(
        "--sections",
        default=",".join(FMP_HISTORICAL_EQUITY_SECTIONS),
    )
    backfill_missing.add_argument("--period", default="quarter", choices=["annual", "quarter", "quarterly"])
    backfill_missing.add_argument("--nport-start-year", type=int, default=2019)
    backfill_missing.add_argument("--macro-start-date", default=MIN_HISTORICAL_DATE)
    backfill_missing.add_argument(
        "--skip-prices",
        action="store_true",
        help="Skip FMP equity/ETF price gap-fill during backfill",
    )
    backfill_missing.add_argument(
        "--force-macro",
        action="store_true",
        help="Force a full macro refresh even when catalog data is already current",
    )
    backfill_missing.add_argument("--staleness-days", type=int, default=90)
    backfill_missing.add_argument("--skip-recent-hours", type=float, default=24.0)
    backfill_missing.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Parallel symbol workers for price/fundamental/N-PORT backfill (default: 8)",
    )
    backfill_missing.add_argument("--limit", type=int, default=None, help="Max equity symbols")
    backfill_missing.add_argument("--etf-limit", type=int, default=None, help="Max ETF symbols")
    backfill_missing.add_argument(
        "--log",
        default="~/.quant-warehouse/logs/backfill-missing-fmp-historical.json",
    )
    backfill_missing.set_defaults(func=cmd_backfill_missing_fmp)

    backfill_all = sub.add_parser(
        "backfill-fmp-all",
        help="Comprehensive OpenBB/FMP backfill: equities, ETFs, mutual funds, calendars",
    )
    backfill_all.add_argument("--provider", default="fmp", help="Equity data provider")
    backfill_all.add_argument("--etf-provider", default="fmp", help="ETF/mutual-fund provider")
    backfill_all.add_argument("--period", default="quarter", choices=["annual", "quarter", "quarterly"])
    backfill_all.add_argument("--calendar-start-date", default="2005-01-01")
    backfill_all.add_argument("--nport-start-year", type=int, default=2019)
    backfill_all.add_argument("--transcript-start-year", type=int, default=2005)
    backfill_all.add_argument("--include-macro", action="store_true")
    backfill_all.add_argument("--skip-prices", action="store_true")
    backfill_all.add_argument(
        "--skip-equity-core",
        action="store_true",
        help="Skip equity prices/fundamentals core phase; run ETF expansion with --workers",
    )
    backfill_all.add_argument("--skip-profiles", action="store_true")
    backfill_all.add_argument("--skip-calendars", action="store_true")
    backfill_all.add_argument(
        "--include-transcripts",
        action="store_true",
        help="Fetch earnings call transcripts (large text payloads; off by default)",
    )
    backfill_all.add_argument("--skip-etf-universe", action="store_true")
    backfill_all.add_argument("--staleness-days", type=int, default=90)
    backfill_all.add_argument("--skip-recent-hours", type=float, default=24.0)
    backfill_all.add_argument("--request-sleep", type=float, default=0.05)
    backfill_all.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Parallel symbol workers for universe backfill phases (default: 8)",
    )
    backfill_all.add_argument("--limit", type=int, default=None, help="Max equity symbols")
    backfill_all.add_argument("--etf-limit", type=int, default=None, help="Max ETF/mutual-fund symbols")
    backfill_all.add_argument(
        "--log",
        default="~/.quant-warehouse/logs/backfill-fmp-all.json",
    )
    backfill_all.set_defaults(func=cmd_backfill_fmp_all)

    backfill_fixes = sub.add_parser(
        "backfill-fixes",
        help="Backfill failed equity calendars and ETF composition sections",
    )
    backfill_fixes.add_argument("--calendar-start-date", default="2005-01-01")
    backfill_fixes.add_argument(
        "--calendar-sections",
        default="",
        help="Comma-separated equity calendar sections (default: all four)",
    )
    backfill_fixes.add_argument(
        "--full-refresh-earnings",
        action="store_true",
        help="Rebuild equity_calendar_earnings from calendar-start-date",
    )
    backfill_fixes.add_argument("--skip-calendars", action="store_true")
    backfill_fixes.add_argument("--skip-etf-composition", action="store_true")
    backfill_fixes.add_argument(
        "--etf-retry-missing-holdings",
        action="store_true",
        help="Only retry ETF symbols still missing etf_holdings rows",
    )
    backfill_fixes.add_argument("--workers", type=int, default=8)
    backfill_fixes.add_argument(
        "--log",
        default="~/.quant-warehouse/logs/backfill-fixes.json",
    )
    backfill_fixes.set_defaults(func=cmd_backfill_fixes)

    backfill_macro_alt = sub.add_parser(
        "backfill-macro-alt",
        help="Backfill extended FMP macro, yield curve, calendar, and crypto/FX/index prices",
    )
    backfill_macro_alt.add_argument(
        "--macro-start-date",
        default="2005-01-01",
        help="History start for calendar/yield-curve/alt prices (FMP calendar begins ~2005)",
    )
    backfill_macro_alt.add_argument(
        "--economic-series",
        default=",".join(DEFAULT_ECONOMIC_SERIES),
    )
    backfill_macro_alt.add_argument("--skip-treasury", action="store_true")
    backfill_macro_alt.add_argument("--skip-yield-curve", action="store_true")
    backfill_macro_alt.add_argument("--skip-calendar", action="store_true")
    backfill_macro_alt.add_argument("--skip-risk-premium", action="store_true")
    backfill_macro_alt.add_argument("--skip-crypto", action="store_true")
    backfill_macro_alt.add_argument("--skip-currency", action="store_true")
    backfill_macro_alt.add_argument("--skip-index", action="store_true")
    backfill_macro_alt.add_argument("--crypto-symbols", default=",".join(DEFAULT_CRYPTO_SYMBOLS))
    backfill_macro_alt.add_argument("--currency-symbols", default=",".join(DEFAULT_CURRENCY_SYMBOLS))
    backfill_macro_alt.add_argument("--index-symbols", default=",".join(DEFAULT_INDEX_SYMBOLS))
    backfill_macro_alt.add_argument(
        "--yield-curve-step-days",
        type=int,
        default=5,
        help="Fetch every N business days when building yield curve history (default: weekly)",
    )
    backfill_macro_alt.add_argument(
        "--log",
        default="~/.quant-warehouse/logs/backfill-macro-alt.json",
    )
    backfill_macro_alt.set_defaults(func=cmd_backfill_macro_alt)

    backfill_thetadata = sub.add_parser(
        "backfill-thetadata-options",
        help="Batch download daily ThetaData EOD option chains for FMP underlyings in Arctic",
    )
    backfill_thetadata.add_argument(
        "--source",
        default="arctic-fmp",
        choices=["arctic-fmp", "catalog"],
        help="Symbol universe: Arctic prices library (FMP) or warehouse catalog metadata",
    )
    backfill_thetadata.add_argument("--symbols", default="", help="Optional comma-separated symbol override")
    backfill_thetadata.add_argument("--start-date", default="2024-01-01")
    backfill_thetadata.add_argument("--end-date", default="", help="Default: today (UTC)")
    backfill_thetadata.add_argument("--max-dte", type=int, default=60)
    backfill_thetadata.add_argument("--strike-range", type=int, default=10)
    backfill_thetadata.add_argument("--limit", type=int, default=None)
    backfill_thetadata.add_argument("--offset", type=int, default=0)
    backfill_thetadata.add_argument("--overwrite", action="store_true", help="Re-download even if daily cache exists")
    backfill_thetadata.add_argument(
        "--include-non-us",
        action="store_true",
        help="Include international tickers (ThetaData options are US-only; default filters them out)",
    )
    backfill_thetadata.add_argument(
        "--request-sleep",
        type=float,
        default=1.0,
        help="Seconds to sleep between symbols (default: 1.0)",
    )
    backfill_thetadata.add_argument(
        "--log",
        default="~/.quant-warehouse/logs/backfill-thetadata-options.json",
    )
    backfill_thetadata.set_defaults(func=cmd_backfill_thetadata_options)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
