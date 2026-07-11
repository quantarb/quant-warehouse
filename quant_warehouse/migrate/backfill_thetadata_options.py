from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Sequence

import pandas as pd

from quant_warehouse.config import WarehouseConfig
from quant_warehouse.platforms.data_providers.thetadata.options import (
    THETADATA_OPTION_HISTORY_ENDPOINT,
    THETADATA_RICH_OPTION_COLUMNS,
    ThetaDataDownloadSpec,
    download_option_snapshots_for_range,
    option_chain_cached_date_summary,
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


def _business_days(start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
    return [ts.normalize() for ts in pd.date_range(start, end, freq="B")]


def _oracle_trade_endpoint_dates(start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
    dates = [pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()]
    return list(dict.fromkeys(dates))


def normalize_oracle_trade_windows(
    trades: Sequence[Mapping[str, Any]] | pd.DataFrame,
    *,
    max_trades: int | None = None,
    symbols: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Return unique oracle trade endpoint windows sorted newest entry first.

    The ThetaData backfill for option ML does not need every date for every
    large-cap symbol. It only needs each unique symbol entry/exit endpoint pair
    once because the stored chain is full-chain by symbol/date.
    """

    if isinstance(trades, pd.DataFrame):
        frame = trades.copy()
    else:
        frame = pd.DataFrame([dict(row) for row in trades])
    if frame.empty:
        return pd.DataFrame(columns=["trade_id", "symbol", "entry_date", "exit_date"])

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
        frame = frame.rename(columns=rename_map)

    required = {"symbol", "entry_date", "exit_date"}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"oracle trades missing required columns: {sorted(missing)}")

    out = frame.copy()
    out["symbol"] = out["symbol"].astype(str).str.strip().str.upper()
    out["entry_date"] = pd.to_datetime(out["entry_date"], errors="coerce").dt.normalize()
    out["exit_date"] = pd.to_datetime(out["exit_date"], errors="coerce").dt.normalize()
    out = out.dropna(subset=["symbol", "entry_date", "exit_date"])
    out = out.loc[out["symbol"].ne("") & out["exit_date"].ge(out["entry_date"])].copy()
    if symbols:
        wanted = {str(symbol).strip().upper() for symbol in symbols if str(symbol).strip()}
        out = out.loc[out["symbol"].isin(wanted)].copy()
    if "trade_id" not in out.columns:
        out["trade_id"] = [
            f"{row.symbol}|{row.entry_date.date()}|{row.exit_date.date()}|{idx}"
            for idx, row in enumerate(out.itertuples(index=False), start=1)
        ]
    out["trade_id"] = out["trade_id"].astype(str)
    out = out.sort_values(["entry_date", "symbol", "trade_id"], ascending=[False, True, True], kind="stable")
    out = out.drop_duplicates(subset=["symbol", "entry_date", "exit_date"], keep="first")
    if max_trades is not None:
        out = out.head(max(0, int(max_trades)))
    return out.reset_index(drop=True)


def _options_range_cached(
    symbol: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> bool:
    return option_chain_range_cached(
        symbol,
        start_date,
        end_date,
        required_columns=THETADATA_RICH_OPTION_COLUMNS,
    )


def _cached_endpoint_dates_by_symbol(trade_windows: pd.DataFrame) -> dict[str, set[pd.Timestamp]]:
    cached: dict[str, set[pd.Timestamp]] = {}
    if trade_windows.empty:
        return cached
    for symbol, group in trade_windows.groupby("symbol", sort=True):
        endpoints = pd.concat([group["entry_date"], group["exit_date"]], ignore_index=True)
        dates = sorted({pd.Timestamp(value).normalize() for value in endpoints.dropna()})
        business_dates = [ts for ts in dates if _business_days(ts, ts)]
        if not business_dates:
            cached[str(symbol).upper()] = set()
            continue
        cached_dates, _row_count = option_chain_cached_date_summary(
            str(symbol).upper(),
            min(business_dates),
            max(business_dates),
            required_columns=THETADATA_RICH_OPTION_COLUMNS,
        )
        cached[str(symbol).upper()] = {pd.Timestamp(value).normalize() for value in cached_dates}
    return cached


def _endpoint_is_cached(
    symbol: str,
    snapshot_date: pd.Timestamp,
    cached_dates_by_symbol: Mapping[str, set[pd.Timestamp]],
    *,
    skip_existing: bool,
    overwrite: bool,
) -> bool:
    if not skip_existing or overwrite:
        return False
    normalized = pd.Timestamp(snapshot_date).normalize()
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
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date or datetime.now(timezone.utc).date()).normalize()
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
    trades: Sequence[Mapping[str, Any]] | pd.DataFrame,
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
    empty_symbol_probe_limit: int = 1,
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
    today = pd.Timestamp(datetime.now(timezone.utc).date()).normalize()

    records = trade_windows.to_dict("records")
    endpoint_keys: list[tuple[str, pd.Timestamp]] = []
    for trade in records:
        symbol = str(trade["symbol"]).upper()
        start = pd.Timestamp(trade["entry_date"]).normalize()
        end = pd.Timestamp(trade["exit_date"]).normalize()
        for snapshot_date in _oracle_trade_endpoint_dates(start, end):
            endpoint_keys.append((symbol, pd.Timestamp(snapshot_date).normalize()))
    unique_endpoint_keys = list(dict.fromkeys(endpoint_keys))

    endpoint_results: dict[tuple[str, pd.Timestamp], dict[str, object]] = {}
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

    no_cache_symbols: list[str] = []
    if skip_existing and not overwrite:
        seen_symbols: set[str] = set()
        for symbol, _snapshot_date in unique_endpoint_keys:
            if symbol in seen_symbols:
                continue
            seen_symbols.add(symbol)
            if not cached_dates_by_symbol.get(symbol):
                no_cache_symbols.append(symbol)

    unavailable_symbols: set[str] = set()
    for probe_index, symbol in enumerate(no_cache_symbols, start=1):
        probe_key = next(
            (
                (key_symbol, key_date)
                for key_symbol, key_date in unique_endpoint_keys
                if key_symbol == symbol and not endpoint_results[(key_symbol, key_date)].get("skipped")
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
            if callable(progress_logger):
                status_text = "empty probe" if is_empty else "probe"
                progress_logger(
                    f"[thetadata-oracle-options] {status_text} {probe_index}/{len(no_cache_symbols)} "
                    f"{symbol} {probe_date.date()} contracts={manifest.get('contracts_total')}"
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
        start = pd.Timestamp(trade["entry_date"]).normalize()
        end = pd.Timestamp(trade["exit_date"]).normalize()
        row: dict[str, object] = {
            "index": index,
            "total": total,
            "trade_id": str(trade.get("trade_id") or ""),
            "symbol": symbol,
            "entry_date": start.date().isoformat(),
            "exit_date": end.date().isoformat(),
            "snapshot_mode": "entry_exit",
            "snapshot_dates": [ts.date().isoformat() for ts in _oracle_trade_endpoint_dates(start, end)],
        }
        date_results = []
        manifests: list[dict[str, Any]] = []
        errors: list[str] = []
        provider_calls = 0
        for snapshot_date in _oracle_trade_endpoint_dates(start, end):
            date_result = dict(endpoint_results[(symbol, pd.Timestamp(snapshot_date).normalize())])
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
        "symbols_requested": int(trade_windows["symbol"].nunique()) if not trade_windows.empty else 0,
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


def log_progress(message: str) -> None:
    print(message, file=sys.stderr, flush=True)
