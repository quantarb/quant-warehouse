from __future__ import annotations

import polars as pl

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date as date_type, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, MutableSet, Sequence


from quant_warehouse.config import WarehouseConfig
from quant_warehouse.platforms.data_providers.thetadata.options import (
    THETADATA_OPTION_HISTORY_ENDPOINT,
    THETADATA_RICH_OPTION_COLUMNS,
    ThetaDataDownloadSpec,
    download_option_snapshots_for_range,
    option_chain_cached_date_summary_bulk,
    option_chain_range_cached,
)
from quant_warehouse.warehouse.api import Warehouse
from quant_warehouse.warehouse.prices import list_arctic_price_underlyings

ProgressLogger = Callable[[str], None] | None
SymbolSource = Literal["arctic-fmp", "catalog", "market-cap"]
OPTION_SECTION = "options_eod"
OPTION_PROVIDER = "thetadata"
_NON_US_SUFFIXES = (".SZ", ".SS", ".HK", ".TO", ".L", ".PA", ".DE", ".AX", ".KS", ".TW", ".T")
MARKET_CAP_TIERS: dict[str, float] = {
    "1t": 1_000_000_000_000,
    "100b": 100_000_000_000,
    "10b": 10_000_000_000,
}

def _day(value: object) -> datetime:
    if isinstance(value, datetime):
        return value.replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=None)
    if isinstance(value, date_type):
        return datetime.combine(value, datetime.min.time())
    return datetime.fromisoformat(str(value)[:10])


def fmp_trading_days_for_year(
    year: int,
    *,
    warehouse: Warehouse | None = None,
    calendar_symbol: str = "SPY",
) -> tuple[datetime, ...]:
    """Return distinct FMP price dates for a calendar year.

    FMP's stored daily price panel is the source of truth for valid US market
    trading dates. This avoids treating exchange holidays as ThetaData dates.
    """

    warehouse = warehouse or Warehouse()
    start = f"{int(year):04d}-01-01"
    end = f"{int(year):04d}-12-31"
    frame = warehouse.read_prices(calendar_symbol, provider="fmp", start=start, end=end)
    if frame is None or frame.is_empty():
        raise ValueError(f"FMP price history is missing for calendar symbol {calendar_symbol!r} in {year}")
    if "date" in frame.columns:
        dates = frame["date"].cast(pl.Datetime, strict=False)
    else:
        raise ValueError("FMP price history must contain an explicit date column")
    result = tuple(sorted({value.replace(hour=0, minute=0, second=0, microsecond=0) for value in dates.drop_nulls().to_list()}))
    if not result:
        raise ValueError(f"FMP price history contains no valid dates for {calendar_symbol!r} in {year}")
    return result


def _is_us_option_symbol(symbol: str) -> bool:
    text = str(symbol).strip().upper()
    if not text:
        return False
    return not any(text.endswith(suffix) for suffix in _NON_US_SUFFIXES)


def _filter_us_symbols(symbols: Sequence[str]) -> list[str]:
    return [symbol for symbol in symbols if _is_us_option_symbol(symbol)]


def list_catalog_price_symbols(
    warehouse: Warehouse,
    *,
    providers: Sequence[str] = ("fmp",),
) -> list[str]:
    """Return warehouse catalog symbols that already have stored equity price history."""

    symbols: set[str] = set()
    for provider in providers:
        for state in warehouse.catalog.list_section("prices", provider=str(provider).strip().lower()):
            if int(state.row_count) > 0 and str(state.symbol).strip():
                symbols.add(str(state.symbol).strip().upper())
    return sorted(symbols)


def list_arctic_fmp_underlyings(warehouse: Warehouse) -> list[str]:
    """Return FMP underlyings stored in the Arctic prices library."""

    return list_arctic_price_underlyings(warehouse.prices.backend, provider="fmp")


def list_market_cap_symbols(
    warehouse: Warehouse,
    *,
    provider: str = "fmp",
    min_market_cap: float | None = None,
    max_market_cap: float | None = None,
    require_prices: bool = True,
    us_only: bool = True,
) -> list[str]:
    """Return profile symbols ordered by descending market cap."""

    profiles = warehouse.catalog.query_symbol_profiles(
        provider=str(provider).strip().lower(),
        min_market_cap=min_market_cap,
        max_market_cap=max_market_cap,
        country="US" if us_only else None,
        exclude_etf=True,
        exclude_fund=True,
    )
    symbols = [profile.symbol for profile in profiles if str(profile.symbol).strip()]
    if require_prices:
        price_symbols = set(list_arctic_fmp_underlyings(warehouse))
        symbols = [symbol for symbol in symbols if symbol in price_symbols]
    return symbols


def resolve_backfill_symbols(
    warehouse: Warehouse,
    *,
    source: SymbolSource = "arctic-fmp",
    symbols: Sequence[str] | None = None,
    min_market_cap: float | None = None,
    max_market_cap: float | None = None,
    require_prices: bool = True,
    limit: int | None = None,
    offset: int = 0,
    us_only: bool = True,
) -> list[str]:
    """Resolve the symbol batch from explicit args or warehouse storage."""

    if symbols:
        resolved = [str(symbol).strip().upper() for symbol in symbols if str(symbol).strip()]
    elif source == "catalog":
        resolved = list_catalog_price_symbols(warehouse, providers=("fmp",))
    elif source == "market-cap":
        resolved = list_market_cap_symbols(
            warehouse,
            min_market_cap=min_market_cap,
            max_market_cap=max_market_cap,
            require_prices=require_prices,
            us_only=us_only,
        )
    else:
        resolved = list_arctic_fmp_underlyings(warehouse)

    if offset and not symbols:
        resolved = resolved[offset:]
    if limit is not None and not symbols:
        resolved = resolved[: max(0, int(limit))]

    if us_only:
        resolved = _filter_us_symbols(resolved)
    return resolved


def _business_days(start: datetime, end: datetime) -> list[datetime]:
    current = start.replace(hour=0, minute=0, second=0, microsecond=0)
    final = end.replace(hour=0, minute=0, second=0, microsecond=0)
    return [current + timedelta(days=offset) for offset in range((final - current).days + 1)
            if (current + timedelta(days=offset)).weekday() < 5]


def _oracle_trade_endpoint_dates(start: datetime, end: datetime) -> list[datetime]:
    dates = [start.replace(hour=0, minute=0, second=0, microsecond=0), end.replace(hour=0, minute=0, second=0, microsecond=0)]
    return list(dict.fromkeys(dates))


def normalize_oracle_trade_windows(
    trades: Sequence[Mapping[str, Any]] | pl.DataFrame,
    *,
    max_trades: int | None = None,
    symbols: Sequence[str] | None = None,
) -> pl.DataFrame:
    """Return unique oracle trade endpoint windows sorted newest entry first.

    The ThetaData backfill for option ML does not need every date for every
    large-cap symbol. It only needs each unique symbol entry/exit endpoint pair
    once because the stored chain is full-chain by symbol/date.
    """

    if isinstance(trades, pl.DataFrame):
        frame = trades.clone()
    else:
        frame = pl.DataFrame([dict(row) for row in trades])
    if frame.is_empty():
        return pl.DataFrame({column: [] for column in ["trade_id", "symbol", "entry_date", "exit_date"]})

    rename_map = {}
    if "entry_date" not in frame.columns:
        for column in ("start_date", "entry_dt", "open_date"):
            if column in frame.columns:
                rename_map[column] = "entry_date"
                break
    if "exit_date" not in frame.columns:
        for column in ("end_date", "exit_dt", "close_date"):
            if column in frame.columns:
                rename_map[column] = "exit_date"
                break
    if "symbol" not in frame.columns and "underlying_symbol" in frame.columns:
        rename_map["underlying_symbol"] = "symbol"
    if rename_map:
        frame = frame.rename(rename_map)

    required = {"symbol", "entry_date", "exit_date"}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"oracle trades missing required columns: {sorted(missing)}")

    out = frame.with_columns(
        pl.col("symbol").cast(pl.String).str.strip_chars().str.to_uppercase(),
        (pl.col("entry_date").str.to_datetime(strict=False, time_zone="UTC") if frame.schema["entry_date"] == pl.String else pl.col("entry_date").cast(pl.Datetime, strict=False)).dt.replace_time_zone(None).dt.truncate("1d"),
        (pl.col("exit_date").str.to_datetime(strict=False, time_zone="UTC") if frame.schema["exit_date"] == pl.String else pl.col("exit_date").cast(pl.Datetime, strict=False)).dt.replace_time_zone(None).dt.truncate("1d"),
    ).drop_nulls(["symbol", "entry_date", "exit_date"]).filter(
        (pl.col("symbol") != "") & (pl.col("exit_date") >= pl.col("entry_date"))
    )
    if symbols:
        wanted = {str(symbol).strip().upper() for symbol in symbols if str(symbol).strip()}
        out = out.filter(pl.col("symbol").is_in(list(wanted)))
    if "trade_id" not in out.columns:
        out = out.with_row_index("_idx").with_columns(
            (pl.col("symbol") + "|" + pl.col("entry_date").dt.strftime("%Y-%m-%d") + "|" + pl.col("exit_date").dt.strftime("%Y-%m-%d") + "|" + pl.col("_idx").cast(pl.String)).alias("trade_id")
        ).drop("_idx")
    out = (out.with_columns(pl.col("trade_id").cast(pl.String))
           .unique(["symbol", "entry_date", "exit_date"], keep="first")
           .sort(["entry_date", "symbol", "trade_id"], descending=[True, False, False]))
    if max_trades is not None:
        out = out.head(max(0, int(max_trades)))
    return out


def _options_range_cached(
    symbol: str,
    start_date: datetime,
    end_date: datetime,
) -> bool:
    return option_chain_range_cached(
        symbol,
        start_date,
        end_date,
        required_columns=THETADATA_RICH_OPTION_COLUMNS,
    )


def _cached_endpoint_dates_by_symbol(trade_windows: pl.DataFrame) -> dict[str, set[datetime]]:
    cached: dict[str, set[datetime]] = {}
    if trade_windows.is_empty():
        return cached
    requested_ranges: dict[str, tuple[datetime, datetime]] = {}
    for symbol_key, group in trade_windows.group_by("symbol", maintain_order=True):
        symbol = symbol_key[0] if isinstance(symbol_key, tuple) else symbol_key
        endpoints = pl.concat([group["entry_date"], group["exit_date"]])
        dates = sorted({value.replace(hour=0, minute=0, second=0, microsecond=0) for value in endpoints.drop_nulls().to_list()})
        business_dates = [ts for ts in dates if _business_days(ts, ts)]
        if not business_dates:
            cached[str(symbol).upper()] = set()
            continue
        requested_ranges[str(symbol).upper()] = (min(business_dates), max(business_dates))
    if requested_ranges:
        global_start = min(start for start, _end in requested_ranges.values())
        global_end = max(end for _start, end in requested_ranges.values())
        bulk = option_chain_cached_date_summary_bulk(
            requested_ranges.keys(),
            global_start,
            global_end,
            required_columns=THETADATA_RICH_OPTION_COLUMNS,
        )
        for symbol in requested_ranges:
            cached[symbol] = {value.replace(hour=0, minute=0, second=0, microsecond=0) if isinstance(value, datetime) else datetime.fromisoformat(str(value)[:10]) for value in bulk.get(symbol, (set(), 0))[0]}
    return cached


def _endpoint_is_cached(
    symbol: str,
    snapshot_date: datetime,
    cached_dates_by_symbol: Mapping[str, set[datetime]],
    *,
    skip_existing: bool,
    overwrite: bool,
) -> bool:
    if not skip_existing or overwrite:
        return False
    normalized = snapshot_date.replace(hour=0, minute=0, second=0, microsecond=0)
    if not _business_days(normalized, normalized):
        return True
    return normalized in cached_dates_by_symbol.get(str(symbol).upper(), set())


def _upsert_options_catalog_state(
    warehouse: Warehouse,
    *,
    symbol: str,
    start_date: str,
    end_date: str,
    snapshot_days: int,
    contracts_total: int,
) -> None:
    warehouse.catalog.upsert(
        symbol=symbol,
        section=OPTION_SECTION,
        provider=OPTION_PROVIDER,
        min_date=start_date,
        max_date=end_date,
        row_count=int(contracts_total),
        columns_present=(
            "bid",
            "ask",
            "mid",
            "snapshot_date",
            "contract_symbol",
            "data_interval",
            *THETADATA_RICH_OPTION_COLUMNS,
        ),
    )


def backfill_thetadata_options(
    *,
    warehouse: Warehouse | None = None,
    config: WarehouseConfig | None = None,
    symbols: Sequence[str] | None = None,
    source: SymbolSource = "arctic-fmp",
    start_date: str = "2024-01-01",
    end_date: str | None = None,
    backfill_window_days: int = 7,
    fallback_window_days: int = 1,
    min_market_cap: float | None = None,
    max_market_cap: float | None = None,
    require_prices: bool = True,
    limit: int | None = None,
    offset: int = 0,
    skip_existing: bool = True,
    overwrite: bool = False,
    request_sleep: float = 1.0,
    max_workers: int = 1,
    us_only: bool = True,
    progress_logger: ProgressLogger = None,
) -> dict[str, object]:
    """Download full daily ThetaData EOD option chains for FMP underlyings in Arctic."""

    warehouse = warehouse or Warehouse(config=config)
    start = _day(start_date)
    end = _day(end_date or datetime.now(timezone.utc).date())
    if end < start:
        raise ValueError(f"end_date {end.date()} must be on or after start_date {start.date()}")

    target_symbols = resolve_backfill_symbols(
        warehouse,
        source=source,
        symbols=symbols,
        min_market_cap=min_market_cap,
        max_market_cap=max_market_cap,
        require_prices=require_prices,
        limit=limit,
        offset=offset,
        us_only=us_only,
    )
    download_spec = ThetaDataDownloadSpec(
        backfill_window_days=backfill_window_days,
        fallback_window_days=fallback_window_days,
    )
    started_at = datetime.now(timezone.utc).isoformat()
    results: list[dict[str, object]] = []
    total = len(target_symbols)

    def _run_symbol(index: int, symbol: str) -> dict[str, object]:
        row: dict[str, object] = {"symbol": symbol}
        try:
            if skip_existing and not overwrite and _options_range_cached(symbol, start, end):
                row.update({"skipped": True, "reason": "cached_range"})
                return row

            manifest = download_option_snapshots_for_range(
                symbol,
                start,
                end,
                spec=download_spec,
                overwrite=overwrite,
            )
            row.update({"skipped": False, **manifest})
        except Exception as exc:
            row.update({"skipped": False, "error": str(exc)})
        finally:
            row["index"] = index
            row["total"] = total
        return row

    def _record(row: dict[str, object]) -> None:
        symbol = str(row["symbol"])
        results.append(row)
        if not row.get("error") and not row.get("skipped"):
            _upsert_options_catalog_state(
                warehouse,
                symbol=symbol,
                start_date=str(row["start_date"]),
                end_date=str(row["end_date"]),
                snapshot_days=int(row.get("snapshot_days") or 0),
                contracts_total=int(row.get("contracts_total") or 0),
            )
        if callable(progress_logger):
            index = int(row.get("index") or len(results))
            if row.get("error"):
                progress_logger(f"[thetadata-options] {index}/{total} error {symbol}: {row.get('error')}")
            elif row.get("skipped"):
                progress_logger(f"[thetadata-options] {index}/{total} skipped cached {symbol}")
            else:
                progress_logger(
                    f"[thetadata-options] {index}/{total} {symbol} "
                    f"days={row.get('snapshot_days')} contracts={row.get('contracts_total')}"
                )

    workers = max(1, int(max_workers))
    if workers == 1:
        for index, symbol in enumerate(target_symbols, start=1):
            row = _run_symbol(index, symbol)
            _record(row)
            if request_sleep > 0 and index < total:
                time.sleep(float(request_sleep))
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_run_symbol, index, symbol): (index, symbol)
                for index, symbol in enumerate(target_symbols, start=1)
            }
            completed_count = 0
            for future in as_completed(futures):
                completed_count += 1
                row = future.result()
                _record(row)
                if request_sleep > 0 and completed_count < total:
                    time.sleep(float(request_sleep))

    completed = [row for row in results if not row.get("error")]
    skipped = [row for row in results if row.get("skipped")]
    failed = [row for row in results if row.get("error")]
    return {
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "start_date": start.date().isoformat(),
        "end_date": end.date().isoformat(),
        "min_market_cap": min_market_cap,
        "max_market_cap": max_market_cap,
        "require_prices": require_prices,
        "symbols_requested": total,
        "symbols_completed": len(completed),
        "symbols_skipped": len(skipped),
        "symbols_failed": len(failed),
        "us_only": us_only,
        "download_spec": {
            "endpoint": THETADATA_OPTION_HISTORY_ENDPOINT,
            "data_interval": download_spec.data_interval,
            "max_dte": None,
            "strike_range": None,
            "require_bid_ask": False,
            "min_ask": 0.0,
            "backfill_window_days": download_spec.backfill_window_days,
            "fallback_window_days": download_spec.fallback_window_days,
        },
        "storage_backend": "arctic",
        "max_workers": workers,
        "results": results,
    }


def backfill_thetadata_options_for_oracle_trades(
    trades: Sequence[Mapping[str, Any]] | pl.DataFrame,
    *,
    warehouse: Warehouse | None = None,
    config: WarehouseConfig | None = None,
    symbols: Sequence[str] | None = None,
    max_trades: int | None = None,
    backfill_window_days: int = 7,
    fallback_window_days: int = 1,
    skip_existing: bool = True,
    overwrite: bool = False,
    request_sleep: float = 1.0,
    trading_days: Sequence[date_type | str | datetime] | None = None,
    empty_symbol_probe_limit: int = 1,
    probed_symbols: MutableSet[str] | None = None,
    progress_logger: ProgressLogger = None,
) -> dict[str, object]:
    """Download ThetaData EOD option chains for oracle trade entry/exit dates."""

    warehouse = warehouse or Warehouse(config=config)
    trade_windows = normalize_oracle_trade_windows(trades, max_trades=max_trades, symbols=symbols)
    download_spec = ThetaDataDownloadSpec(
        backfill_window_days=backfill_window_days,
        fallback_window_days=fallback_window_days,
    )
    started_at = datetime.now(timezone.utc).isoformat()
    total = len(trade_windows)
    cached_dates_by_symbol = (
        _cached_endpoint_dates_by_symbol(trade_windows) if skip_existing and not overwrite else {}
    )
    today = _day(datetime.now(timezone.utc).date())

    records = trade_windows.to_dicts()
    endpoint_keys: list[tuple[str, datetime]] = []
    valid_trading_days = (
        {_day(value) for value in trading_days}
        if trading_days is not None
        else None
    )
    for trade in records:
        symbol = str(trade["symbol"]).upper()
        start = _day(trade["entry_date"])
        end = _day(trade["exit_date"])
        for snapshot_date in _oracle_trade_endpoint_dates(start, end):
            if valid_trading_days is not None and _day(snapshot_date) not in valid_trading_days:
                continue
            endpoint_keys.append((symbol, _day(snapshot_date)))
    unique_endpoint_keys = list(dict.fromkeys(endpoint_keys))
    if callable(progress_logger):
        progress_logger(
            f"[thetadata-oracle-options] planned trades={total:,} "
            f"unique_endpoint_dates={len(unique_endpoint_keys):,} order=trade_entry_desc"
        )

    endpoint_results: dict[tuple[str, datetime], dict[str, object]] = {}
    for symbol, snapshot_date in unique_endpoint_keys:
        cached = _endpoint_is_cached(
            symbol,
            snapshot_date,
            cached_dates_by_symbol,
            skip_existing=skip_existing,
            overwrite=overwrite,
        )
        date_result: dict[str, object] = {
            "snapshot_date": snapshot_date.date().isoformat(),
            "cached": bool(cached),
            "skipped": False,
        }
        if cached:
            date_result.update({"skipped": True, "reason": "cached_snapshot"})
        elif snapshot_date >= today:
            date_result.update({"skipped": True, "reason": "current_or_future_eod_snapshot"})
        endpoint_results[(symbol, snapshot_date)] = date_result

    # Always probe every symbol, even when its endpoint dates appear cached.
    # This makes zero-contract symbols explicit and prevents wasting calls on
    # later endpoint dates that cannot return option chains.
    no_cache_symbols = sorted({symbol for symbol, _snapshot_date in unique_endpoint_keys})
    if probed_symbols is not None:
        no_cache_symbols = [symbol for symbol in no_cache_symbols if symbol not in probed_symbols]

    unavailable_symbols: set[str] = set()
    for probe_index, symbol in enumerate(no_cache_symbols, start=1):
        probe_key = next(
            (
                (key_symbol, key_date)
                for key_symbol, key_date in unique_endpoint_keys
                if key_symbol == symbol and key_date < today
            ),
            None,
        )
        if probe_key is None:
            continue
        _probe_symbol, probe_date = probe_key
        try:
            manifest = download_option_snapshots_for_range(
                symbol,
                probe_date,
                probe_date,
                spec=download_spec,
                overwrite=overwrite,
            )
            is_empty = (
                int(manifest.get("snapshot_days") or 0) <= 0
                and int(manifest.get("contracts_total") or 0) <= 0
                and int(manifest.get("fetched_rows") or 0) <= 0
            )
            endpoint_results[probe_key].update(
                {
                    "downloaded": not is_empty,
                    "snapshot_days": int(manifest.get("snapshot_days") or 0),
                    "contracts_total": int(manifest.get("contracts_total") or 0),
                    "fetched_rows": int(manifest.get("fetched_rows") or 0),
                    "provider_call": True,
                    "manifest": manifest,
                }
            )
            if is_empty:
                unavailable_symbols.add(symbol)
                endpoint_results[probe_key].update(
                    {"skipped": True, "reason": "empty_symbol_after_probe"}
                )
            else:
                _upsert_options_catalog_state(
                    warehouse,
                    symbol=symbol,
                    start_date=str(manifest["start_date"]),
                    end_date=str(manifest["end_date"]),
                    snapshot_days=int(manifest.get("snapshot_days") or 0),
                    contracts_total=int(manifest.get("contracts_total") or 0),
                )
            if probed_symbols is not None:
                probed_symbols.add(symbol)
            if callable(progress_logger):
                status_text = "empty probe" if is_empty else "probe"
                progress_logger(
                    f"[thetadata-oracle-options] {status_text} {probe_index}/{len(no_cache_symbols)} "
                    f"({'db' if manifest.get('cached_only') else 'api'}) {symbol} "
                    f"{probe_date.date()} contracts={manifest.get('contracts_total')}"
                )
        except Exception as exc:
            endpoint_results[probe_key].update(
                {"skipped": False, "error": str(exc), "provider_call": True}
            )
            if callable(progress_logger):
                progress_logger(
                    f"[thetadata-oracle-options] probe {probe_index}/{len(no_cache_symbols)} "
                    f"error {symbol} {probe_date.date()}: {exc}"
                )
        if request_sleep > 0 and probe_index < len(no_cache_symbols):
            time.sleep(float(request_sleep))

    for symbol, snapshot_date in unique_endpoint_keys:
        if symbol in unavailable_symbols and not endpoint_results[(symbol, snapshot_date)].get("provider_call"):
            endpoint_results[(symbol, snapshot_date)].update(
                {"skipped": True, "reason": "empty_symbol_after_probe", "empty_probe_skipped": True}
            )

    missing_endpoint_keys = [
        key for key in unique_endpoint_keys
        if not endpoint_results[key].get("skipped") and not endpoint_results[key].get("provider_call")
    ]
    if callable(progress_logger):
        cached_count = sum(1 for row in endpoint_results.values() if row.get("reason") == "cached_snapshot")
        unavailable_count = sum(
            1 for row in endpoint_results.values()
            if row.get("reason") == "current_or_future_eod_snapshot"
        )
        empty_probe_count = sum(1 for row in endpoint_results.values() if row.get("reason") == "empty_symbol_after_probe")
        progress_logger(
            f"[thetadata-oracle-options] cache preflight: trades={total:,} "
            f"unique_endpoint_dates={len(unique_endpoint_keys):,} cached={cached_count:,} "
            f"current_or_future={unavailable_count:,} empty_probe_skipped={empty_probe_count:,} "
            f"missing={len(missing_endpoint_keys):,}"
        )

    empty_symbol_counts: dict[str, int] = {}
    for download_index, (symbol, snapshot_date) in enumerate(missing_endpoint_keys, start=1):
        if int(empty_symbol_probe_limit) > 0 and empty_symbol_counts.get(symbol, 0) >= int(empty_symbol_probe_limit):
            endpoint_results[(symbol, snapshot_date)].update(
                {"skipped": True, "reason": "empty_symbol_after_probe", "empty_probe_skipped": True}
            )
            continue
        try:
            if callable(progress_logger):
                progress_logger(
                    f"[thetadata-oracle-options] downloading {download_index}/{len(missing_endpoint_keys)} "
                    f"{symbol} {snapshot_date.date()}"
                )
            manifest = download_option_snapshots_for_range(
                symbol,
                snapshot_date,
                snapshot_date,
                spec=download_spec,
                overwrite=overwrite,
            )
            endpoint_results[(symbol, snapshot_date)].update(
                {
                    "downloaded": int(manifest.get("snapshot_days") or 0) > 0,
                    "snapshot_days": int(manifest.get("snapshot_days") or 0),
                    "contracts_total": int(manifest.get("contracts_total") or 0),
                    "fetched_rows": int(manifest.get("fetched_rows") or 0),
                    "provider_call": True,
                    "manifest": manifest,
                }
            )
            if (
                int(manifest.get("snapshot_days") or 0) <= 0
                and int(manifest.get("contracts_total") or 0) <= 0
                and int(manifest.get("fetched_rows") or 0) <= 0
            ):
                empty_symbol_counts[symbol] = empty_symbol_counts.get(symbol, 0) + 1
            _upsert_options_catalog_state(
                warehouse,
                symbol=symbol,
                start_date=str(manifest["start_date"]),
                end_date=str(manifest["end_date"]),
                snapshot_days=int(manifest.get("snapshot_days") or 0),
                contracts_total=int(manifest.get("contracts_total") or 0),
            )
            if callable(progress_logger):
                status_text = "empty" if int(manifest.get("contracts_total") or 0) <= 0 else "download"
                progress_logger(
                    f"[thetadata-oracle-options] {status_text} {download_index}/{len(missing_endpoint_keys)} "
                    f"{symbol} {snapshot_date.date()} contracts={manifest.get('contracts_total')}"
                )
        except Exception as exc:
            endpoint_results[(symbol, snapshot_date)].update(
                {"skipped": False, "error": str(exc), "provider_call": True}
            )
            if callable(progress_logger):
                progress_logger(
                    f"[thetadata-oracle-options] download {download_index}/{len(missing_endpoint_keys)} "
                    f"error {symbol} {snapshot_date.date()}: {exc}"
                )
        if request_sleep > 0 and download_index < len(missing_endpoint_keys):
            time.sleep(float(request_sleep))

    results: list[dict[str, object]] = []
    for index, trade in enumerate(records, start=1):
        symbol = str(trade["symbol"]).upper()
        start = _day(trade["entry_date"])
        end = _day(trade["exit_date"])
        row: dict[str, object] = {
            "index": index,
            "total": total,
            "trade_id": str(trade.get("trade_id") or ""),
            "symbol": symbol,
            "entry_date": start.date().isoformat(),
            "exit_date": end.date().isoformat(),
            "snapshot_mode": "entry_exit",
        }
        snapshot_dates = [
            _day(ts)
            for ts in _oracle_trade_endpoint_dates(start, end)
            if valid_trading_days is None or _day(ts) in valid_trading_days
        ]
        row["snapshot_dates"] = [ts.date().isoformat() for ts in snapshot_dates]
        date_results = []
        manifests: list[dict[str, Any]] = []
        errors: list[str] = []
        provider_calls = 0
        for snapshot_date in snapshot_dates:
            date_result = dict(endpoint_results[(symbol, snapshot_date)])
            manifest = date_result.pop("manifest", None)
            if manifest is not None:
                manifests.append(manifest)
            if date_result.get("provider_call"):
                provider_calls += 1
            if date_result.get("error"):
                errors.append(str(date_result["error"]))
            date_results.append(date_result)
        if errors:
            row["error"] = "; ".join(errors)
        all_dates_skipped = bool(date_results) and all(bool(date_result.get("skipped")) for date_result in date_results)
        row.update(
            {
                "skipped": all_dates_skipped and not errors,
                "reason": "cached_or_unavailable_snapshots" if all_dates_skipped and not errors else None,
                "provider_calls": provider_calls,
                "snapshot_days": int(sum(int(manifest.get("snapshot_days") or 0) for manifest in manifests)),
                "contracts_total": int(sum(int(manifest.get("contracts_total") or 0) for manifest in manifests)),
                "fetched_rows": int(sum(int(manifest.get("fetched_rows") or 0) for manifest in manifests)),
                "date_results": date_results,
            }
        )
        results.append(row)

        if callable(progress_logger):
            if row.get("error"):
                progress_logger(f"[thetadata-oracle-options] {index}/{total} error {symbol}: {row.get('error')}")
            elif not row.get("skipped"):
                progress_logger(
                    f"[thetadata-oracle-options] {index}/{total} {symbol} "
                    f"entry_exit={row.get('snapshot_dates')} days={row.get('snapshot_days')} "
                    f"contracts={row.get('contracts_total')} trade_id={row.get('trade_id')}"
                )

    completed = [row for row in results if not row.get("error")]
    skipped = [row for row in results if row.get("skipped")]
    failed = [row for row in results if row.get("error")]
    if callable(progress_logger) and skipped:
        progress_logger(
            f"[thetadata-oracle-options] skipped {len(skipped):,}/{total:,} trade windows "
            "with cached or unavailable entry/exit snapshots"
        )
    return {
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "source": "oracle-trades",
        "trade_windows_requested": total,
        "trade_windows_completed": len(completed),
        "trade_windows_skipped": len(skipped),
        "trade_windows_failed": len(failed),
        "symbols_requested": int(trade_windows["symbol"].n_unique()) if not trade_windows.is_empty() else 0,
        "max_trades": max_trades,
        "download_spec": {
            "endpoint": THETADATA_OPTION_HISTORY_ENDPOINT,
            "data_interval": download_spec.data_interval,
            "max_dte": None,
            "strike_range": None,
            "require_bid_ask": False,
            "min_ask": 0.0,
            "backfill_window_days": download_spec.backfill_window_days,
            "fallback_window_days": download_spec.fallback_window_days,
            "empty_symbol_probe_limit": int(empty_symbol_probe_limit),
        },
        "storage_backend": "arctic",
        "sort_order": "entry_date_desc",
        "snapshot_mode": "entry_exit",
        "results": results,
    }


def write_backfill_log(summary: dict[str, object], *, log_path: str | Path) -> Path:
    path = Path(log_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return path


def backfill_thetadata_options_for_oracle_trade_ranges(
    trades: Sequence[Mapping[str, Any]] | pl.DataFrame,
    *,
    mode: Literal["oracle_entry_exit", "all"] = "all",
    warehouse: Warehouse | None = None,
    config: WarehouseConfig | None = None,
    symbols: Sequence[str] | None = None,
    max_trades: int | None = None,
    trading_days: Sequence[date_type | str | datetime] | None = None,
    skip_existing: bool = True,
    overwrite: bool = False,
    request_sleep: float = 1.0,
    empty_symbol_probe_limit: int = 1,
    probed_symbols: MutableSet[str] | None = None,
    progress_logger: ProgressLogger = None,
) -> dict[str, object]:
    """Backfill ThetaData using an explicit oracle snapshot policy.

    ``oracle_entry_exit`` downloads only each trade's entry and exit dates.
    ``all`` downloads every missing business date in each oracle window,
    ordered newest-first so repeated runs progressively work backward.

    This is the progressive data-completion pass for option labels. It writes
    full chains to the warehouse and is safe to rerun: cached symbol/date
    snapshots are skipped and the next run resumes at the newest remaining
    missing date.
    """

    if mode not in {"oracle_entry_exit", "all"}:
        raise ValueError("mode must be 'oracle_entry_exit' or 'all'")
    if mode == "oracle_entry_exit":
        return backfill_thetadata_options_for_oracle_trades(
            trades,
            warehouse=warehouse,
            config=config,
            symbols=symbols,
            max_trades=max_trades,
            trading_days=trading_days,
            skip_existing=skip_existing,
            overwrite=overwrite,
            request_sleep=request_sleep,
            empty_symbol_probe_limit=empty_symbol_probe_limit,
            probed_symbols=probed_symbols,
            progress_logger=progress_logger,
        )

    warehouse = warehouse or Warehouse(config=config)
    windows = normalize_oracle_trade_windows(trades, max_trades=max_trades, symbols=symbols)
    if windows.is_empty():
        return {"status": "empty_trades", "dates_requested": 0, "dates_downloaded": 0}

    today = _day(datetime.now(timezone.utc).date())
    valid_trading_days = (
        {_day(value) for value in trading_days}
        if trading_days is not None
        else None
    )
    keys: set[tuple[str, datetime]] = set()
    for row in windows.to_dicts():
        symbol = str(row["symbol"]).upper()
        start = _day(row["entry_date"])
        end = _day(row["exit_date"])
        for date in _business_days(start, end):
            if (valid_trading_days is None or date in valid_trading_days) and date < today:
                keys.add((symbol, date))

    cached_by_symbol: dict[str, set[datetime]] = {}
    if skip_existing and not overwrite:
        requested_dates_by_symbol: dict[str, list[datetime]] = {}
        for symbol, date in keys:
            requested_dates_by_symbol.setdefault(symbol, []).append(date)
        if requested_dates_by_symbol:
            bulk = option_chain_cached_date_summary_bulk(
                requested_dates_by_symbol.keys(),
                min(date for dates in requested_dates_by_symbol.values() for date in dates),
                max(date for dates in requested_dates_by_symbol.values() for date in dates),
                required_columns=THETADATA_RICH_OPTION_COLUMNS,
            )
            for symbol, requested_dates in requested_dates_by_symbol.items():
                cached_dates = bulk.get(symbol, (set(), 0))[0]
                requested_set = set(requested_dates)
                cached_by_symbol[symbol] = cached_dates.intersection(requested_set)

    missing = [
        key for key in keys
        if overwrite or key[1] not in cached_by_symbol.get(key[0], set())
    ]
    missing.sort(key=lambda item: (item[1], item[0]), reverse=True)
    # Always probe every symbol before the date loop. A zero-contract probe
    # suppresses the symbol for this run, while a non-empty probe is cached and
    # the normal loop proceeds over the remaining dates.
    unavailable_symbols: set[str] = set()
    probe_symbols = sorted({symbol for symbol, _date in keys})
    if probed_symbols is not None:
        probe_symbols = [symbol for symbol in probe_symbols if symbol not in probed_symbols]
    for probe_index, symbol in enumerate(probe_symbols, start=1):
        symbol_dates = sorted(date for key_symbol, date in keys if key_symbol == symbol and date < today)
        if not symbol_dates:
            continue
        probe_date = symbol_dates[-1]
        if callable(progress_logger):
            progress_logger(
                f"[thetadata-oracle-ranges] probe {probe_index}/{len(probe_symbols)} "
                f"{symbol} {probe_date.date()}"
            )
        try:
            probe_manifest = download_option_snapshots_for_range(
                symbol,
                probe_date,
                probe_date,
                overwrite=overwrite,
            )
            probe_empty = (
                int(probe_manifest.get("snapshot_days") or 0) <= 0
                and int(probe_manifest.get("contracts_total") or 0) <= 0
                and int(probe_manifest.get("fetched_rows") or 0) <= 0
            )
            if probe_empty:
                unavailable_symbols.add(symbol)
                if callable(progress_logger):
                    progress_logger(
                        f"[thetadata-oracle-ranges] probe empty {symbol} {probe_date.date()} "
                        "; skipping remaining dates for this run"
                    )
            elif callable(progress_logger):
                progress_logger(
                    f"[thetadata-oracle-ranges] probe ok ({'db' if probe_manifest.get('cached_only') else 'api'}) "
                    f"{symbol} {probe_date.date()} contracts={probe_manifest.get('contracts_total')}"
                )
            if probed_symbols is not None:
                probed_symbols.add(symbol)
        except Exception as exc:
            if callable(progress_logger):
                progress_logger(
                    f"[thetadata-oracle-ranges] probe error {symbol} {probe_date.date()}: {exc}"
                )
    if callable(progress_logger):
        progress_logger(
            f"[thetadata-oracle-ranges] planned windows={len(windows):,} "
            f"unique_dates={len(keys):,} cached={len(keys) - len(missing):,} "
            f"missing={len(missing):,} order=newest_first"
        )
    results: list[dict[str, object]] = []
    for index, (symbol, date) in enumerate(missing, start=1):
        if symbol in unavailable_symbols:
            results.append({"symbol": symbol, "snapshot_date": date.date().isoformat(), "skipped": True, "reason": "empty_symbol"})
            continue
        try:
            if callable(progress_logger):
                progress_logger(
                    f"[thetadata-oracle-ranges] downloading {index}/{len(missing)} "
                    f"{symbol} {date.date()}"
                )
            manifest = download_option_snapshots_for_range(
                symbol,
                date,
                date,
                overwrite=overwrite,
            )
            empty = int(manifest.get("snapshot_days") or 0) <= 0 and int(manifest.get("fetched_rows") or 0) <= 0
            result = {
                "symbol": symbol,
                "snapshot_date": date.date().isoformat(),
                "downloaded": not empty,
                "skipped": False,
                "manifest": manifest,
            }
            if empty:
                unavailable_symbols.add(symbol)
                result.update({"skipped": True, "reason": "empty_symbol"})
            else:
                _upsert_options_catalog_state(
                    warehouse,
                    symbol=symbol,
                    start_date=str(manifest["start_date"]),
                    end_date=str(manifest["end_date"]),
                    snapshot_days=int(manifest.get("snapshot_days") or 0),
                    contracts_total=int(manifest.get("contracts_total") or 0),
                )
            results.append(result)
            if callable(progress_logger):
                progress_logger(
                    f"[thetadata-oracle-ranges] {index}/{len(missing)} {symbol} {date.date()} "
                    f"contracts={manifest.get('contracts_total')}"
                )
        except Exception as exc:  # noqa: BLE001
            results.append({"symbol": symbol, "snapshot_date": date.date().isoformat(), "skipped": False, "error": str(exc)})
            if callable(progress_logger):
                progress_logger(f"[thetadata-oracle-ranges] error {symbol} {date.date()}: {exc}")
        if request_sleep > 0 and index < len(missing):
            time.sleep(float(request_sleep))

    return {
        "status": "ok",
        "source": "oracle-trade-ranges",
        "mode": mode,
        "sort_order": "snapshot_date_desc",
        "trade_windows": int(len(windows)),
        "symbols": int(windows["symbol"].nunique()),
        "dates_requested": int(len(keys)),
        "dates_cached": int(len(keys) - len(missing)),
        "dates_missing": int(len(missing)),
        "dates_downloaded": int(sum(bool(row.get("downloaded")) for row in results)),
        "dates_skipped": int(sum(bool(row.get("skipped")) for row in results)),
        "dates_failed": int(sum(bool(row.get("error")) for row in results)),
        "results": results,
    }


def log_progress(message: str) -> None:
    print(message, file=sys.stderr, flush=True)
