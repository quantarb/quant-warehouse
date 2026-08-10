from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Iterator, Literal, Mapping, Sequence

import polars as pl

from quant_warehouse.config import WarehouseConfig
from quant_warehouse.ingest.openbb_fetch import fetch_openbb
from quant_warehouse.warehouse.backend import ArcticBackend, open_backend
from quant_warehouse.warehouse.storage import read_provider_frame, provider_library

Frame = pl.DataFrame

def _day(value: object) -> datetime:
    if isinstance(value, datetime):
        return value.replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=None)
    if isinstance(value, date):
        return datetime.combine(value, datetime.min.time())
    return datetime.fromisoformat(str(value)[:10])


def _partition_day(value: object) -> datetime:
    """Normalize Polars partition keys across Polars versions."""
    if isinstance(value, tuple):
        value = value[0]
    if isinstance(value, pl.Series):
        value = value[0]
    return _day(value)

def _business_days(start: datetime, end: datetime) -> list[datetime]:
    start = _day(start); end = _day(end)
    return [start + timedelta(days=offset) for offset in range((end - start).days + 1)
            if (start + timedelta(days=offset)).weekday() < 5]

def _datetime_expr(frame: pl.DataFrame, column: str) -> pl.Expr:
    return ((pl.col(column).str.to_datetime(strict=False, time_zone="UTC") if frame.schema[column] == pl.String
             else pl.col(column).cast(pl.Datetime, strict=False)).dt.replace_time_zone(None).dt.truncate("1d"))

# ThetaData EOD history rejects spans longer than 365 calendar days.
THETADATA_MAX_EOD_SPAN_DAYS = 364
THETADATA_BACKFILL_WINDOW_DAYS = 7
THETADATA_FALLBACK_WINDOW_DAYS = 1
OPTIONS_THETADATA_EOD_LIBRARY = "options_thetadata_eod"
OPTIONS_THETADATA_PROVIDER = "thetadata"
THETADATA_OPTION_HISTORY_ENDPOINT = "option_history_greeks_eod"
THETADATA_RICH_OPTION_COLUMNS: tuple[str, ...] = (
    "underlying_price",
    "delta",
    "gamma",
    "theta",
    "vega",
    "rho",
    "iv",
)
THETADATA_EOD_OPTION_REQUIRED_COLUMNS: tuple[str, ...] = (
    "snapshot_date",
    "underlying_symbol",
    "contract_symbol",
    "expiration",
    "strike",
    "option_type",
    "bid",
    "ask",
    "mid",
)
THETADATA_EOD_OPTION_OPTIONAL_COLUMNS: tuple[str, ...] = (
    "volume",
    "open_interest",
    *THETADATA_RICH_OPTION_COLUMNS,
)
THETADATA_EOD_OPTION_CONTRACT_COLUMNS: tuple[str, ...] = tuple(
    dict.fromkeys([*THETADATA_EOD_OPTION_REQUIRED_COLUMNS, *THETADATA_EOD_OPTION_OPTIONAL_COLUMNS])
)
THETADATA_UNSUPPORTED_DOWNLOAD_FILTERS: tuple[str, ...] = (
    "dte",
    "min_dte",
    "max_dte",
    "expiration",
    "right",
    "option_type",
    "strike",
    "strike_range",
    "min_strike",
    "max_strike",
    "moneyness",
    "min_moneyness",
    "max_moneyness",
    "max_abs_moneyness",
    "delta",
    "min_delta",
    "max_delta",
    "bid",
    "ask",
    "require_bid_ask",
    "min_bid",
    "min_ask",
    "volume",
    "min_volume",
    "open_interest",
    "min_open_interest",
    "liquidity",
)


@dataclass(frozen=True)
class ThetaDataOptionSnapshot:
    snapshot_date: datetime
    frame: pl.DataFrame


@dataclass(frozen=True, init=False)
class ThetaDataDownloadSpec:
    """Parameters for daily ThetaData EOD option chain downloads.

    Contract selection is intentionally not configurable here. Warehouse
    downloads must preserve the full chain for each requested symbol/date, and
    research filters belong after the complete chain is stored.
    """

    data_interval: Literal["eod"] = "eod"
    annual_dividend: float | None = None
    rate_type: str | None = "sofr"
    rate_value: float | None = None
    version: str | None = "latest"
    underlyer_use_nbbo: bool = False
    backfill_window_days: int = THETADATA_BACKFILL_WINDOW_DAYS
    fallback_window_days: int = THETADATA_FALLBACK_WINDOW_DAYS

    def __init__(
        self,
        *,
        data_interval: Literal["eod"] = "eod",
        annual_dividend: float | None = None,
        rate_type: str | None = "sofr",
        rate_value: float | None = None,
        version: str | None = "latest",
        underlyer_use_nbbo: bool = False,
        backfill_window_days: int = THETADATA_BACKFILL_WINDOW_DAYS,
        fallback_window_days: int = THETADATA_FALLBACK_WINDOW_DAYS,
        **download_filters: Any,
    ) -> None:
        _reject_thetadata_download_filters(download_filters)
        object.__setattr__(self, "data_interval", data_interval)
        object.__setattr__(self, "annual_dividend", annual_dividend)
        object.__setattr__(self, "rate_type", rate_type)
        object.__setattr__(self, "rate_value", rate_value)
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "underlyer_use_nbbo", underlyer_use_nbbo)
        object.__setattr__(self, "backfill_window_days", backfill_window_days)
        object.__setattr__(self, "fallback_window_days", fallback_window_days)


def _iter_eod_date_chunks(
    start_date: date | str | datetime,
    end_date: date | str | datetime,
    *,
    max_span_days: int = THETADATA_MAX_EOD_SPAN_DAYS,
) -> Iterator[tuple[date, date]]:
    """Yield [start, end] date pairs that respect ThetaData's max EOD span."""

    start = _day(start_date)
    end = _day(end_date)
    cursor = start
    while cursor <= end:
        chunk_end = min(cursor + timedelta(days=max_span_days), end)
        yield cursor.date(), chunk_end.date()
        cursor = chunk_end + timedelta(days=1)


def fetch_option_history_eod(
    symbol: str,
    start_date: date | str | datetime,
    end_date: date | str | datetime,
    *,
    api_key: str | None = None,
    spec: ThetaDataDownloadSpec | None = None,
    **download_filters: Any,
) -> Frame:
    """Download normalized EOD option chains for a symbol over a date range."""

    _reject_thetadata_download_filters(download_filters)
    download_spec = spec or ThetaDataDownloadSpec()
    frames: list[Frame] = []
    for chunk_start, chunk_end in _iter_eod_date_chunks(start_date, end_date):
        frame = _fetch_option_history_eod_openbb(
            symbol,
            chunk_start,
            chunk_end,
            api_key=api_key,
            spec=download_spec,
        )
        if not frame.is_empty():
            frames.append(frame)

    if not frames:
        return pl.DataFrame()
    return normalize_thetadata_option_chain(pl.concat(frames, how="diagonal_relaxed"))


def _fetch_option_history_eod_openbb(
    symbol: str,
    start_date: date | str | datetime,
    end_date: date | str | datetime,
    *,
    api_key: str | None = None,
    spec: ThetaDataDownloadSpec,
) -> Frame:
    result = fetch_openbb(
        "options_eod",
        symbol=str(symbol).upper(),
        provider="thetadata",
        start_date=start_date,
        end_date=end_date,
        expiration="*",
        strike="*",
        right="both",
        max_dte=None,
        strike_range=None,
        require_bid_ask=False,
        min_ask=0.0,
        include_greeks=True,
        annual_dividend=spec.annual_dividend,
        rate_type=spec.rate_type,
        rate_value=spec.rate_value,
        version=spec.version,
        underlyer_use_nbbo=bool(spec.underlyer_use_nbbo),
    )
    return result.df.clone()


def split_snapshots_by_date(df: Frame) -> dict[datetime, Frame]:
    """Split a multi-day ThetaData frame into one chain per snapshot date."""

    if df is None or df.is_empty():
        return {}
    out = df
    source_col = "snapshot_date" if "snapshot_date" in out.columns else next(
        (col for col in ("eod_date", "created", "created_at", "quote_timestamp", "timestamp") if col in out.columns), None)
    if source_col is None:
        return {}
    out = out.with_columns(_datetime_expr(out, source_col).alias("snapshot_date"))
    return {_partition_day(key): group for key, group in out.drop_nulls("snapshot_date").partition_by("snapshot_date", as_dict=True).items()}


def option_chain_storage_symbol(symbol: str) -> str:
    return str(symbol).strip().upper()


def read_option_chain_arctic(
    symbol: str,
    *,
    start_date: date | str | datetime | None = None,
    end_date: date | str | datetime | None = None,
    columns: Sequence[str] | None = None,
    fallback_legacy: bool = False,
    backend: ArcticBackend | None = None,
    config: WarehouseConfig | None = None,
) -> Frame:
    backend = backend or open_backend(config or WarehouseConfig.from_env())
    start, end = _option_chain_read_bounds(start_date, end_date)
    projected_columns = _option_chain_projected_columns(columns)
    frame = read_provider_frame(
        backend,
        base_library=OPTIONS_THETADATA_EOD_LIBRARY,
        provider=OPTIONS_THETADATA_PROVIDER,
        symbol=option_chain_storage_symbol(symbol),
        fallback_legacy=fallback_legacy,
        start_date=start,
        end_date=end,
        columns=projected_columns,
        output_format="polars",
    )
    if frame is None or frame.is_empty():
        return pl.DataFrame()
    out = frame
    if "snapshot_date" not in out.columns:
        return pl.DataFrame()
    snapshot = pl.col("snapshot_date")
    if out.schema["snapshot_date"] == pl.String:
        snapshot = snapshot.str.to_datetime(strict=False, time_zone="UTC")
    else:
        snapshot = snapshot.cast(pl.Datetime, strict=False)
    out = out.with_columns(snapshot.dt.replace_time_zone(None).dt.truncate("1d").alias("snapshot_date"))
    if start_date is not None:
        out = out.filter(pl.col("snapshot_date") >= _day(start_date))
    if end_date is not None:
        out = out.filter(pl.col("snapshot_date") <= _day(end_date))
    if {"snapshot_date", "contract_symbol"}.issubset(out.columns):
        out = out.unique(subset=["snapshot_date", "contract_symbol"], keep="last", maintain_order=True)
    if columns is not None:
        out = out.select([column for column in columns if column in out.columns])
    return out.sort("snapshot_date") if "snapshot_date" in out.columns else out


def read_thetadata_eod_option_chain(
    symbol: str,
    *,
    start_date: date | str | datetime | None = None,
    end_date: date | str | datetime | None = None,
    columns: Sequence[str] | None = None,
    require_rich_columns: bool = False,
    fallback_legacy: bool = False,
    backend: ArcticBackend | None = None,
    config: WarehouseConfig | None = None,
) -> Frame:
    """Read a normalized ThetaData EOD option chain from warehouse storage.

    This is the provider-specific contract consumed by options research code.
    It guarantees one row per ``(snapshot_date, contract_symbol)`` after
    warehouse upsert/deduplication, normalizes ThetaData column aliases, and
    optionally requires the richer greeks endpoint fields.
    """

    requested_columns = tuple(str(column) for column in columns) if columns is not None else None
    read_columns = _thetadata_eod_contract_read_columns(requested_columns, require_rich_columns=require_rich_columns)
    frame = read_option_chain_arctic(
        symbol,
        start_date=start_date,
        end_date=end_date,
        columns=read_columns,
        fallback_legacy=fallback_legacy,
        backend=backend,
        config=config,
    )
    out = validate_thetadata_eod_option_chain_contract(frame, require_rich_columns=require_rich_columns)
    if requested_columns is not None:
        for column in requested_columns:
            if column not in out.columns:
                out = out.with_columns(pl.lit(None).alias(column))
        out = out.select(list(requested_columns))
    return out


def validate_thetadata_eod_option_chain_contract(
    frame: Frame,
    *,
    require_rich_columns: bool = False,
) -> Frame:
    """Validate and normalize the ThetaData EOD option-chain read contract."""

    if frame is None or frame.is_empty():
        return pl.DataFrame()
    return _validate_thetadata_eod_option_chain_polars(frame, require_rich_columns=require_rich_columns)


def write_option_chain_arctic(
    symbol: str,
    frame: pl.DataFrame,
    *,
    backend: ArcticBackend | None = None,
    config: WarehouseConfig | None = None,
    merge: bool = True,
) -> str:
    if frame is None or frame.is_empty():
        return _arctic_ref(symbol)
    backend = backend or open_backend(config or WarehouseConfig.from_env())
    storage_symbol = option_chain_storage_symbol(symbol)
    incoming = _prepare_option_chain_for_arctic(frame)
    library = provider_library(OPTIONS_THETADATA_EOD_LIBRARY, OPTIONS_THETADATA_PROVIDER)
    existing = (
        read_provider_frame(
            backend,
            base_library=OPTIONS_THETADATA_EOD_LIBRARY,
            provider=OPTIONS_THETADATA_PROVIDER,
            symbol=storage_symbol,
        )
        if merge
        else None
    )
    merged = _merge_option_chain_upsert(existing, incoming)
    if not merged.is_empty():
        backend.write(library, storage_symbol, merged, prune_previous_versions=True)
    return _arctic_ref(symbol)


def option_chain_coverage(
    symbols: Sequence[str] | None = None,
    *,
    backend: ArcticBackend | None = None,
    config: WarehouseConfig | None = None,
    fallback_legacy: bool = False,
) -> pl.DataFrame:
    """Return lightweight cached ThetaData option coverage by symbol."""

    backend = backend or open_backend(config or WarehouseConfig.from_env())
    if symbols is None:
        library = provider_library(OPTIONS_THETADATA_EOD_LIBRARY, OPTIONS_THETADATA_PROVIDER)
        symbols = sorted(str(symbol) for symbol in backend.list_symbols(library))
    rows: list[dict[str, Any]] = []
    for raw_symbol in symbols:
        symbol = option_chain_storage_symbol(raw_symbol)
        frame = read_option_chain_arctic(
            symbol,
            columns=["snapshot_date"],
            backend=backend,
            fallback_legacy=fallback_legacy,
        )
        if frame.is_empty() or "snapshot_date" not in frame.columns:
            rows.append({"symbol": symbol, "row_count": 0, "snapshot_day_count": 0})
            continue
        dates = frame["snapshot_date"].drop_nulls()
        rows.append(
            {
                "symbol": symbol,
                "row_count": int(len(frame)),
                "snapshot_day_count": int(dates.n_unique()),
                "min_snapshot_date": None if dates.is_empty() else dates.min().date().isoformat(),
                "max_snapshot_date": None if dates.is_empty() else dates.max().date().isoformat(),
            }
        )
    return pl.DataFrame(rows).sort(["row_count", "symbol"], descending=[True, False])


def deduplicate_option_chain_arctic(
    symbols: Sequence[str] | None = None,
    *,
    dry_run: bool = True,
    backend: ArcticBackend | None = None,
    config: WarehouseConfig | None = None,
) -> pl.DataFrame:
    """Remove duplicate cached option-chain rows by (snapshot_date, contract_symbol).

    Dry-run mode reports what would be rewritten without mutating Arctic. Set
    ``dry_run=False`` to rewrite each affected symbol with duplicate rows
    removed and previous versions pruned.
    """

    backend = backend or open_backend(config or WarehouseConfig.from_env())
    library = provider_library(OPTIONS_THETADATA_EOD_LIBRARY, OPTIONS_THETADATA_PROVIDER)
    if symbols is None:
        symbols = sorted(str(symbol) for symbol in backend.list_symbols(library))
    rows: list[dict[str, Any]] = []
    for raw_symbol in symbols:
        storage_symbol = option_chain_storage_symbol(raw_symbol)
        frame = read_provider_frame(
            backend,
            base_library=OPTIONS_THETADATA_EOD_LIBRARY,
            provider=OPTIONS_THETADATA_PROVIDER,
            symbol=storage_symbol,
        )
        rows_before = 0 if frame is None or frame.is_empty() else int(len(frame))
        if frame is None or frame.is_empty():
            rows.append(
                {
                    "symbol": storage_symbol,
                    "rows_before": rows_before,
                    "rows_after": 0,
                    "duplicate_rows": 0,
                    "rewritten": False,
                }
            )
            continue
        prepared = _prepare_option_chain_for_arctic(frame)
        rows_after = int(len(prepared))
        duplicate_rows = max(0, rows_before - rows_after)
        rewritten = bool(duplicate_rows and not dry_run)
        if rewritten:
            backend.write(library, storage_symbol, prepared, prune_previous_versions=True)
        rows.append(
            {
                "symbol": storage_symbol,
                "rows_before": rows_before,
                "rows_after": rows_after,
                "duplicate_rows": duplicate_rows,
                "rewritten": rewritten,
            }
        )
    return pl.DataFrame(rows).sort(["duplicate_rows", "symbol"], descending=[True, False])


def _option_chain_read_bounds(
    start_date: date | str | datetime | None,
    end_date: date | str | datetime | None,
) -> tuple[datetime | None, datetime | None]:
    start = None if start_date is None else _day(start_date)
    end = None
    if end_date is not None:
        end = _day(end_date) + timedelta(days=1) - timedelta(microseconds=1)
    return start, end


def _option_chain_projected_columns(columns: Sequence[str] | None) -> list[str] | None:
    if columns is None:
        return None
    requested = [str(column) for column in columns]
    if "snapshot_date" not in requested:
        requested.append("snapshot_date")
    if "contract_symbol" not in requested:
        requested.append("contract_symbol")
    return requested


def _thetadata_eod_contract_read_columns(
    columns: Sequence[str] | None,
    *,
    require_rich_columns: bool,
) -> list[str] | None:
    if columns is None:
        return None
    required = list(THETADATA_EOD_OPTION_REQUIRED_COLUMNS)
    if require_rich_columns:
        required.extend(THETADATA_RICH_OPTION_COLUMNS)
    return list(dict.fromkeys([*columns, *required]))


def option_chain_snapshots_cached(
    symbol: str,
    snapshot_dates: Sequence[date | str | datetime],
    *,
    required_columns: Sequence[str] | None = THETADATA_RICH_OPTION_COLUMNS,
    backend: ArcticBackend | None = None,
    config: WarehouseConfig | None = None,
) -> dict[datetime, Frame]:
    dates = [_day(value) for value in snapshot_dates]
    if not dates:
        return {}
    frame = read_option_chain_arctic(
        symbol,
        start_date=min(dates),
        end_date=max(dates),
        backend=backend,
        config=config,
    )
    if frame.is_empty():
        return {}
    snapshots: dict[datetime, Frame] = {}
    for key, group in frame.partition_by("snapshot_date", as_dict=True).items():
        normalized = _partition_day(key)
        if normalized not in set(dates):
            continue
        required = [str(column) for column in required_columns or ()]
        if any(column not in group.columns for column in required) or not all(group.select(pl.col(column).is_not_null().any()).item() for column in required):
            continue
        snapshots[normalized] = _normalize_thetadata_option_chain_polars(group)
    return snapshots


def option_chain_range_cached(
    symbol: str,
    start_date: date | str | datetime,
    end_date: date | str | datetime,
    *,
    required_columns: Sequence[str] | None = THETADATA_RICH_OPTION_COLUMNS,
    backend: ArcticBackend | None = None,
    config: WarehouseConfig | None = None,
) -> bool:
    dates = _business_days(_day(start_date), _day(end_date))
    if not dates:
        return False
    cached_dates, _row_count = option_chain_cached_date_summary(
        symbol,
        min(dates),
        max(dates),
        required_columns=required_columns,
        backend=backend,
        config=config,
    )
    return set(dates).issubset(cached_dates)


def option_chain_cached_date_summary(
    symbol: str,
    start_date: date | str | datetime,
    end_date: date | str | datetime,
    *,
    required_columns: Sequence[str] | None = None,
    backend: ArcticBackend | None = None,
    config: WarehouseConfig | None = None,
) -> tuple[set[datetime], int]:
    """Return cached snapshot dates and row count without loading full chains.

    When ``required_columns`` is provided, a snapshot is considered cached only
    if every required column exists and has at least one non-null value that day.
    """

    requested_columns = ["snapshot_date"]
    if required_columns is not None:
        requested_columns.extend(str(column) for column in required_columns)

    frame = read_option_chain_arctic(
        symbol,
        start_date=start_date,
        end_date=end_date,
        columns=requested_columns,
        backend=backend,
        config=config,
    )
    if frame.is_empty() or "snapshot_date" not in frame.columns:
        return set(), 0
    dates = frame["snapshot_date"].drop_nulls()
    if required_columns is None:
        return {_day(ts) for ts in dates.unique().to_list()}, int(len(frame))

    required = [str(column) for column in required_columns]
    if any(column not in frame.columns for column in required):
        return set(), 0

    work = frame.with_columns(pl.col("snapshot_date").alias("_snapshot_date"))
    rich_dates: set[datetime] = set()
    rich_row_count = 0
    for key, group in work.drop_nulls("_snapshot_date").partition_by("_snapshot_date", as_dict=True).items():
        if all(group.select(pl.col(column).is_not_null().any()).item() for column in required):
            rich_dates.add(_partition_day(key))
            rich_row_count += int(len(group))
    return rich_dates, rich_row_count


def option_chain_cached_date_summary_bulk(
    symbols: Sequence[str],
    start_date: date | str | datetime,
    end_date: date | str | datetime,
    *,
    required_columns: Sequence[str] | None = None,
    backend: Any | None = None,
    config: WarehouseConfig | None = None,
    batch_size: int = 100,
) -> dict[str, tuple[set[datetime], int]]:
    """Return cached date summaries for many symbols with one ArcticDB batch read.

    The result uses the same rich-date contract as
    :func:`option_chain_cached_date_summary`, but submits all symbol reads to
    ArcticDB's ``read_batch`` API instead of opening one read per symbol.
    """

    normalized_symbols = sorted({str(symbol).strip().upper() for symbol in symbols if str(symbol).strip()})
    result = {symbol: (set(), 0) for symbol in normalized_symbols}
    if not normalized_symbols:
        return result
    backend = backend or open_backend(config or WarehouseConfig.from_env())
    for symbol in normalized_symbols:
        result[symbol] = option_chain_cached_date_summary(symbol, start_date, end_date,
                                                           required_columns=required_columns, backend=backend, config=config)
    return result


def load_thetadata_option_snapshots(
    symbol: str,
    snapshot_dates: Sequence[date | str | datetime],
    *,
    api_key: str | None = None,
    use_cache: bool = True,
    download_spec: ThetaDataDownloadSpec | None = None,
    download_missing: bool = True,
    **download_filters: Any,
) -> dict[datetime, Frame]:
    """Load EOD option snapshots keyed by date, using ArcticDB as the cache/store."""

    _reject_thetadata_download_filters(download_filters)
    spec = download_spec or ThetaDataDownloadSpec()
    normalized_dates = [_day(value) for value in snapshot_dates]
    arctic_backend = open_backend(WarehouseConfig.from_env()) if use_cache else None
    snapshots: dict[datetime, Frame] = {}
    missing: list[datetime] = []

    if arctic_backend is not None:
        snapshots.update(
            option_chain_snapshots_cached(
                symbol,
                normalized_dates,
                backend=arctic_backend,
            )
        )

    for ts in normalized_dates:
        if ts in snapshots:
            continue
        missing.append(ts)

    if missing and download_missing:
        fetched = fetch_option_history_eod(
            symbol,
            min(missing),
            max(missing),
            api_key=api_key,
            spec=spec,
        )
        for ts, frame in split_snapshots_by_date(fetched).items():
            snapshots[ts] = frame
            if use_cache and not frame.is_empty() and arctic_backend is not None:
                write_option_chain_arctic(symbol, frame, backend=arctic_backend)

    return {ts: snapshots[ts] for ts in normalized_dates if ts in snapshots}


def download_option_snapshots_for_range(
    symbol: str,
    start_date: date | str | datetime,
    end_date: date | str | datetime,
    *,
    api_key: str | None = None,
    spec: ThetaDataDownloadSpec | None = None,
    overwrite: bool = False,
    **download_filters: Any,
) -> dict[str, Any]:
    """Download and cache full daily ThetaData EOD option chains in ArcticDB."""

    _reject_thetadata_download_filters(download_filters)
    download_spec = spec or ThetaDataDownloadSpec()
    arctic_backend = open_backend(WarehouseConfig.from_env())
    start = _day(start_date)
    end = _day(end_date)

    requested_dates = _business_days(start, end)
    if overwrite or not requested_dates:
        existing_cached_dates, cached_dates, cached_row_count = set(), set(), 0
    else:
        existing_cached_dates, _existing_row_count = option_chain_cached_date_summary(
            symbol,
            min(requested_dates),
            max(requested_dates),
            backend=arctic_backend,
        )
        cached_dates, cached_row_count = option_chain_cached_date_summary(
            symbol,
            min(requested_dates),
            max(requested_dates),
            required_columns=THETADATA_RICH_OPTION_COLUMNS,
            backend=arctic_backend,
        )
    missing_dates = [ts for ts in requested_dates if ts not in cached_dates]
    stale_cached_dates = [ts for ts in requested_dates if ts in existing_cached_dates and ts not in cached_dates]

    if not requested_dates or not missing_dates:
        paths = [_arctic_ref(symbol)] if cached_dates else []
        return {
            "symbol": str(symbol).upper(),
            "start_date": start.date().isoformat(),
            "end_date": end.date().isoformat(),
            "snapshot_days": len(cached_dates),
            "contracts_total": int(cached_row_count),
            "cached_days": len(cached_dates),
            "existing_cached_days": len(existing_cached_dates),
            "stale_cached_days": len(stale_cached_dates),
            "fetched_rows": 0,
            "cached_only": True,
            "paths": paths,
            "spec": _download_spec_manifest(download_spec),
        }

    downloaded_dates, fetched_rows, _written_paths = _download_and_cache_snapshots(
        symbol,
        missing_dates,
        api_key=api_key,
        spec=download_spec,
        backend=arctic_backend,
        overwrite=overwrite,
    )
    paths = [_arctic_ref(symbol)] if downloaded_dates or cached_dates else []

    manifest = {
        "symbol": str(symbol).upper(),
        "start_date": start.date().isoformat(),
        "end_date": end.date().isoformat(),
        "snapshot_days": len(cached_dates) + len(downloaded_dates),
        "contracts_total": int(cached_row_count + fetched_rows),
        "cached_days": len(cached_dates),
        "existing_cached_days": len(existing_cached_dates),
        "stale_cached_days": len(stale_cached_dates),
        "fetched_rows": int(fetched_rows),
        "cached_only": False,
        "paths": paths,
        "spec": _download_spec_manifest(download_spec),
    }
    return manifest


def _read_cached_snapshots(
    symbol: str,
    requested_dates: Sequence[datetime],
    *,
    backend: ArcticBackend | None = None,
) -> dict[datetime, pl.DataFrame]:
    if backend is not None:
        arctic = option_chain_snapshots_cached(
            symbol,
            requested_dates,
            required_columns=THETADATA_RICH_OPTION_COLUMNS,
            backend=backend,
        )
        if arctic:
            return arctic
    return {}


def _iter_contiguous_business_date_ranges(
    requested_dates: Sequence[datetime],
) -> Iterator[tuple[datetime, datetime]]:
    dates = sorted({_day(ts) for ts in requested_dates})
    if not dates:
        return

    range_start = dates[0]
    previous = dates[0]
    for current in dates[1:]:
        next_business_day = previous + timedelta(days=1)
        while next_business_day.weekday() >= 5: next_business_day += timedelta(days=1)
        if current != next_business_day:
            yield range_start, previous
            range_start = current
        previous = current
    yield range_start, previous


def _iter_bounded_business_date_ranges(
    requested_dates: Sequence[datetime],
    *,
    max_calendar_days: int = THETADATA_BACKFILL_WINDOW_DAYS,
) -> Iterator[tuple[datetime, datetime]]:
    dates = sorted({_day(ts) for ts in requested_dates})
    if not dates:
        return

    range_start = dates[0]
    previous = dates[0]
    for current in dates[1:]:
        next_business_day = previous + timedelta(days=1)
        while next_business_day.weekday() >= 5: next_business_day += timedelta(days=1)
        window_too_wide = (current - range_start).days >= int(max_calendar_days)
        if current != next_business_day or window_too_wide:
            yield range_start, previous
            range_start = current
        previous = current
    yield range_start, previous


def _download_and_cache_snapshots(
    symbol: str,
    requested_dates: Sequence[datetime],
    *,
    api_key: str | None,
    spec: ThetaDataDownloadSpec,
    backend: ArcticBackend | None,
    overwrite: bool,
) -> tuple[set[datetime], int, list[str]]:
    downloaded_dates: set[datetime] = set()
    paths: list[str] = []
    fetched_rows = 0
    requested = {_day(ts) for ts in requested_dates}

    window_days = max(1, min(int(spec.backfill_window_days), THETADATA_MAX_EOD_SPAN_DAYS))
    fallback_days = max(1, min(int(spec.fallback_window_days), window_days))
    for start, end in _iter_bounded_business_date_ranges(requested_dates, max_calendar_days=window_days):
        fetched = _fetch_option_history_with_window_fallback(
            symbol,
            start,
            end,
            spec=spec,
            fallback_window_days=fallback_days,
        )
        if fetched.is_empty():
            continue
        chunk_frames: list[pl.DataFrame] = []
        for ts, frame in split_snapshots_by_date(fetched).items():
            if ts not in requested or frame.is_empty():
                continue
            downloaded_dates.add(ts)
            chunk_frames.append(frame)
        if backend is not None and chunk_frames:
            combined = pl.concat(chunk_frames, how="diagonal_relaxed")
            fetched_rows += len(combined)
            paths.append(write_option_chain_arctic(symbol, combined, backend=backend, merge=True))

    missing_dates = [ts for ts in requested_dates if ts not in downloaded_dates]
    for ts in missing_dates:
        day_frame = fetch_option_history_eod(symbol, ts, ts, spec=spec)
        if day_frame.is_empty():
            continue
        day_frames: list[pl.DataFrame] = []
        for day_ts, frame in split_snapshots_by_date(day_frame).items():
            if day_ts not in requested or frame.is_empty():
                continue
            downloaded_dates.add(day_ts)
            day_frames.append(frame)
        if backend is not None and day_frames:
            combined = pl.concat(day_frames, how="diagonal_relaxed")
            fetched_rows += len(combined)
            paths.append(write_option_chain_arctic(symbol, combined, backend=backend, merge=True))

    return downloaded_dates, fetched_rows, list(dict.fromkeys(paths))


def _fetch_option_history_with_window_fallback(
    symbol: str,
    start: datetime,
    end: datetime,
    *,
    spec: ThetaDataDownloadSpec,
    fallback_window_days: int,
) -> pl.DataFrame:
    try:
        return fetch_option_history_eod(symbol, start, end, spec=spec)
    except Exception:
        if int(fallback_window_days) <= 1 or _day(start) >= _day(end):
            raise

        frames: list[pl.DataFrame] = []
        errors: list[Exception] = []
        dates = _business_days(start, end)
        for chunk_start, chunk_end in _iter_bounded_business_date_ranges(
            dates,
            max_calendar_days=int(fallback_window_days),
        ):
            try:
                frame = fetch_option_history_eod(symbol, chunk_start, chunk_end, spec=spec)
            except Exception as exc:
                errors.append(exc)
                continue
            if not frame.is_empty():
                frames.append(frame)

        if errors:
            raise RuntimeError(
                f"ThetaData fallback failed for {symbol} {_day(start).date()} to {_day(end).date()}"
            ) from errors[0]
        if not frames:
            return pl.DataFrame()
        return pl.concat(frames, how="diagonal_relaxed")


def load_cached_snapshots_for_trade_window(
    symbol: str,
    entry_date: date | str | datetime,
    exit_date: date | str | datetime,
    *,
    api_key: str | None = None,
    spec: ThetaDataDownloadSpec | None = None,
    download_missing: bool = True,
    **download_filters: Any,
) -> dict[datetime, pl.DataFrame]:
    """Load per-day chains for a trade window, optionally downloading missing days."""

    _reject_thetadata_download_filters(download_filters)
    start = _day(entry_date)
    end = _day(exit_date)
    dates = _business_days(start, end)
    if download_missing:
        download_option_snapshots_for_range(
            symbol,
            start,
            end,
            api_key=api_key,
            spec=spec,
        )
    return load_thetadata_option_snapshots(
        symbol,
        dates,
        api_key=api_key,
        download_spec=spec,
        use_cache=True,
        download_missing=download_missing,
    )


def _reject_thetadata_download_filters(download_filters: Mapping[str, Any]) -> None:
    if not download_filters:
        return
    keys = sorted(str(key) for key in download_filters)
    known_filters = [key for key in keys if key in THETADATA_UNSUPPORTED_DOWNLOAD_FILTERS]
    detail = ", ".join(known_filters or keys)
    raise ValueError(
        "ThetaData option downloads are full-chain-only in quant-warehouse. "
        f"Remove provider-side query filter(s): {detail}. "
        "Filter contracts only after reading the complete chain from the warehouse."
    )


def _download_spec_manifest(spec: ThetaDataDownloadSpec) -> dict[str, Any]:
    return {
        "endpoint": THETADATA_OPTION_HISTORY_ENDPOINT,
        "data_interval": spec.data_interval,
        "expiration": "*",
        "right": "both",
        "max_dte": None,
        "strike_range": None,
        "require_bid_ask": False,
        "min_ask": 0.0,
        "annual_dividend": spec.annual_dividend,
        "rate_type": spec.rate_type,
        "rate_value": spec.rate_value,
        "version": spec.version,
        "underlyer_use_nbbo": spec.underlyer_use_nbbo,
        "backfill_window_days": spec.backfill_window_days,
        "fallback_window_days": spec.fallback_window_days,
    }


def _add_quote_columns(frame: pl.DataFrame) -> pl.DataFrame:
    out = frame.clone()
    if "bid" in out.columns:
        out = out.with_columns(pl.col("bid").cast(pl.Float64, strict=False))
    if "ask" in out.columns:
        out = out.with_columns(pl.col("ask").cast(pl.Float64, strict=False))
    if "bid" in out.columns and "ask" in out.columns:
        out = out.with_columns(((pl.col("bid") + pl.col("ask")) / 2.0).alias("mid"))
    return out


def _prepare_option_chain_for_arctic(frame: pl.DataFrame) -> pl.DataFrame:
    normalized = normalize_thetadata_option_chain(frame)
    if normalized.is_empty():
        return normalized
    out = normalized.unique(maintain_order=True)
    out = out.drop_nulls(["snapshot_date", "contract_symbol"])
    out = _deduplicate_option_chain_rows(out)
    out = out.with_row_index("_row")
    out = out.with_columns((pl.col("snapshot_date") + pl.duration(nanoseconds=pl.col("_row"))).alias("date")).drop("_row")
    return _sanitize_option_chain_for_arctic(out.sort("date"))


def _merge_option_chain_upsert(
    existing: pl.DataFrame | None,
    incoming: pl.DataFrame,
) -> pl.DataFrame:
    if incoming is None or incoming.is_empty():
        return pl.DataFrame() if existing is None else existing
    if existing is None or existing.is_empty():
        return incoming.sort("date") if "date" in incoming.columns else incoming
    combined = pl.concat([existing, incoming], how="diagonal_relaxed")
    combined = combined.unique(["snapshot_date", "contract_symbol"], keep="last", maintain_order=True)
    combined = combined.drop_nulls(["snapshot_date"])
    return _prepare_option_chain_for_arctic(combined)


def _deduplicate_option_chain_rows(frame: pl.DataFrame) -> pl.DataFrame:
    if frame is None or frame.is_empty() or not {"snapshot_date", "contract_symbol"}.issubset(frame.columns):
        return pl.DataFrame() if frame is None else frame.clone()
    out = frame.drop_nulls(["snapshot_date", "contract_symbol"])
    if "created_at" in out.columns:
        out = out.with_columns(_datetime_expr(out, "created_at").alias("_created_at_sort")).sort(["snapshot_date", "contract_symbol", "_created_at_sort"])
    return out.unique(["snapshot_date", "contract_symbol"], keep="last", maintain_order=True).drop("_created_at_sort", strict=False)


def _sanitize_option_chain_for_arctic(frame: pl.DataFrame) -> pl.DataFrame:
    out = frame.clone()
    out = out.select([column for column in out.columns if not out.select(pl.col(column).is_not_null().any()).item() is False])
    date_columns = {
        "snapshot_date",
        "eod_date",
        "expiration",
        "created_at",
        "quote_timestamp",
        "last_trade_time",
        "underlying_timestamp",
        "bid_time",
        "ask_time",
        "close_time",
        "close_bid_time",
        "close_ask_time",
    }
    for column in list(out.columns):
        if column in date_columns:
            out = out.with_columns(_datetime_expr(out, column).alias(column))
        elif out.schema[column] == pl.String:
            out = out.with_columns(pl.col(column).fill_null(""))
    return out


def _arctic_ref(symbol: str) -> str:
    library = provider_library(OPTIONS_THETADATA_EOD_LIBRARY, OPTIONS_THETADATA_PROVIDER)
    return f"arctic://{library}/{option_chain_storage_symbol(symbol)}"


def _normalize_snapshot_dates(values: pl.Series) -> pl.Series:
    frame = values.to_frame()
    return frame.select(_datetime_expr(frame, values.name).alias(values.name)).to_series()


def normalize_thetadata_option_chain(df: Frame) -> Frame:
    """Normalize daily ThetaData EOD chains."""

    if df is None or df.is_empty():
        return pl.DataFrame()
    return _normalize_thetadata_option_chain_polars(df)


def _normalize_thetadata_option_chain_polars(df: pl.DataFrame) -> pl.DataFrame:
    """Polars-native equivalent of the ThetaData chain normalizer."""
    if df.is_empty():
        return pl.DataFrame()
    rename_map = {
        "symbol": "underlying_symbol", "right": "option_type", "created": "created_at",
        "timestamp": "quote_timestamp", "last_trade": "last_trade_time", "close": "last_trade_price",
        "open": "open_price", "high": "high_price", "low": "low_price",
        "implied_vol": "iv", "implied_volatility": "iv",
    }
    out = df.rename({column: column.strip().lower() for column in df.columns})
    out = out.rename({source: target for source, target in rename_map.items() if source in out.columns and target not in out.columns})
    if "snapshot_date" not in out.columns:
        source = next((column for column in ("eod_date", "created_at", "quote_timestamp") if column in out.columns), None)
        if source is not None:
            out = out.with_columns(_polars_datetime(out, source).alias("snapshot_date"))
    if "underlying_symbol" not in out.columns or "option_type" not in out.columns or "expiration" not in out.columns or "strike" not in out.columns:
        missing = [column for column in ("underlying_symbol", "option_type", "expiration", "strike") if column not in out.columns]
        raise KeyError(", ".join(missing))
    out = out.with_columns([
        pl.col("underlying_symbol").cast(pl.String).str.to_uppercase(),
        pl.col("option_type").cast(pl.String).str.strip_chars().str.to_lowercase().replace({"c": "call", "p": "put"}),
        _polars_datetime(out, "expiration").alias("expiration"),
        pl.col("strike").cast(pl.Float64, strict=False).alias("strike"),
    ])
    numeric = [column for column in ("open_price", "high_price", "low_price", "last_trade_price", "volume", "count", "bid_size", "ask_size", *THETADATA_RICH_OPTION_COLUMNS, "iv_error") if column in out.columns]
    if numeric:
        out = out.with_columns([pl.col(column).cast(pl.Float64, strict=False).alias(column) for column in numeric])
    if "snapshot_date" in out.columns:
        out = out.with_columns(_polars_datetime(out, "snapshot_date").alias("snapshot_date"))
    else:
        out = out.with_columns(pl.lit(None, dtype=pl.Datetime).alias("snapshot_date"))
    if "contract_symbol" not in out.columns:
        out = out.with_columns(
            (pl.col("underlying_symbol").fill_null("") + "_" + pl.col("option_type").fill_null("") + "_" + pl.col("expiration").dt.strftime("%Y%m%d").fill_null("") + "_" + pl.col("strike").map_elements(lambda value: f"{float(value):g}" if value is not None else "", return_dtype=pl.String)).alias("contract_symbol")
        )
    if "bid" in out.columns and "ask" in out.columns:
        out = out.with_columns([
            pl.col("bid").cast(pl.Float64, strict=False).alias("bid"),
            pl.col("ask").cast(pl.Float64, strict=False).alias("ask"),
        ]).with_columns(((pl.col("bid") + pl.col("ask")) / 2.0).alias("mid"))
    out = out.with_columns(pl.lit("eod").alias("data_interval")).drop_nulls(["snapshot_date", "contract_symbol", "expiration", "strike"])
    if {"snapshot_date", "contract_symbol"}.issubset(out.columns):
        out = out.unique(subset=["snapshot_date", "contract_symbol"], keep="last", maintain_order=True)
    return out.sort(["snapshot_date", "contract_symbol"])


def _validate_thetadata_eod_option_chain_polars(frame: pl.DataFrame, *, require_rich_columns: bool) -> pl.DataFrame:
    out = _normalize_thetadata_option_chain_polars(frame)
    missing_required = [column for column in THETADATA_EOD_OPTION_REQUIRED_COLUMNS if column not in out.columns]
    if missing_required:
        raise ValueError(f"ThetaData EOD option chain is missing required columns: {', '.join(missing_required)}")
    for column in THETADATA_EOD_OPTION_CONTRACT_COLUMNS:
        if column not in out.columns:
            out = out.with_columns(pl.lit(None).alias(column))
    out = out.with_columns([pl.col(column).cast(pl.Float64, strict=False).alias(column) for column in ("bid", "ask", "mid", *THETADATA_EOD_OPTION_OPTIONAL_COLUMNS)])
    if require_rich_columns:
        missing_rich = [column for column in THETADATA_RICH_OPTION_COLUMNS if column not in out.columns or not out.select(pl.col(column).is_not_null().any()).item()]
        if missing_rich:
            raise ValueError(f"ThetaData EOD option chain is missing required rich endpoint columns: {', '.join(missing_rich)}")
    return out.unique(subset=["snapshot_date", "contract_symbol"], keep="last", maintain_order=True).sort(["snapshot_date", "contract_symbol"])


def _polars_datetime(frame: pl.DataFrame, column: str) -> pl.Expr:
    source = pl.col(column)
    if frame.schema[column] == pl.String:
        source = source.str.to_datetime(strict=False, time_zone="UTC")
    else:
        source = source.cast(pl.Datetime, strict=False)
    return source.dt.replace_time_zone(None).dt.truncate("1d")
