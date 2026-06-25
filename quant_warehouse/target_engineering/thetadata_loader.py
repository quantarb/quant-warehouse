from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Iterator, Literal, Sequence

import pandas as pd
from pandas.api.types import is_object_dtype, is_string_dtype

from quant_warehouse.config import WarehouseConfig
from quant_warehouse.ingest.credentials import resolve_thetadata_api_key
from quant_warehouse.ingest.openbb_fetch import fetch_openbb
from quant_warehouse.warehouse.backend import ArcticBackend, open_backend

# ThetaData EOD history rejects spans longer than 365 calendar days.
THETADATA_MAX_EOD_SPAN_DAYS = 364
THETADATA_BACKFILL_WINDOW_DAYS = 31
OPTIONS_THETADATA_EOD_LIBRARY = "options_thetadata_eod"
OPTIONS_THETADATA_PROVIDER = "thetadata"


@dataclass(frozen=True)
class ThetaDataOptionSnapshot:
    snapshot_date: pd.Timestamp
    frame: pd.DataFrame


@dataclass(frozen=True)
class ThetaDataDownloadSpec:
    """Parameters for daily ThetaData EOD option chain downloads."""

    data_interval: Literal["eod"] = "eod"
    max_dte: int = 60
    strike_range: int = 10
    expiration: str = "*"
    right: str = "both"
    dataframe_type: str = "pandas"
    require_bid_ask: bool = True
    min_ask: float = 0.01


def _iter_eod_date_chunks(
    start_date: date | str | pd.Timestamp,
    end_date: date | str | pd.Timestamp,
    *,
    max_span_days: int = THETADATA_MAX_EOD_SPAN_DAYS,
) -> Iterator[tuple[date, date]]:
    """Yield [start, end] date pairs that respect ThetaData's max EOD span."""

    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    cursor = start
    while cursor <= end:
        chunk_end = min(cursor + pd.Timedelta(days=max_span_days), end)
        yield cursor.date(), chunk_end.date()
        cursor = chunk_end + pd.Timedelta(days=1)


def fetch_option_history_eod(
    symbol: str,
    start_date: date | str | pd.Timestamp,
    end_date: date | str | pd.Timestamp,
    *,
    api_key: str | None = None,
    spec: ThetaDataDownloadSpec | None = None,
) -> pd.DataFrame:
    """Download normalized EOD option chains for a symbol over a date range."""

    download_spec = spec or ThetaDataDownloadSpec()
    frames: list[pd.DataFrame] = []
    for chunk_start, chunk_end in _iter_eod_date_chunks(start_date, end_date):
        try:
            result = fetch_openbb(
                "options_eod",
                symbol=str(symbol).upper(),
                provider="thetadata",
                start_date=chunk_start,
                end_date=chunk_end,
                expiration=download_spec.expiration,
                max_dte=int(download_spec.max_dte),
                strike_range=int(download_spec.strike_range),
                right=download_spec.right,
                dataframe_type=download_spec.dataframe_type,
                require_bid_ask=download_spec.require_bid_ask,
                min_ask=download_spec.min_ask,
            )
            frame = result.df.copy()
        except Exception:
            frame = _fetch_option_history_eod_sdk(
                symbol,
                chunk_start,
                chunk_end,
                api_key=api_key,
                spec=download_spec,
            )
        if not frame.empty:
            frames.append(frame)

    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    return normalize_thetadata_option_chain(
        combined,
        require_bid_ask=download_spec.require_bid_ask,
        min_ask=download_spec.min_ask,
    )


def _fetch_option_history_eod_sdk(
    symbol: str,
    start_date: date,
    end_date: date,
    *,
    api_key: str | None,
    spec: ThetaDataDownloadSpec,
) -> pd.DataFrame:
    """Fetch ThetaData EOD option history directly when OpenBB has no provider."""

    try:
        from thetadata import ThetaClient
    except ImportError as exc:
        raise ImportError("Install quant-warehouse[thetadata] to download ThetaData options") from exc

    resolved_api_key = api_key or resolve_thetadata_api_key(required=True)
    client = ThetaClient(
        api_key=resolved_api_key,
        dataframe_type=spec.dataframe_type,
        dotenv_path=".env",
    )
    frame = client.option_history_eod(
        symbol=str(symbol).upper(),
        expiration=spec.expiration,
        start_date=start_date,
        end_date=end_date,
        max_dte=int(spec.max_dte),
        strike_range=int(spec.strike_range),
        right=spec.right,
    )
    if frame is None:
        return pd.DataFrame()
    return frame.to_pandas() if hasattr(frame, "to_pandas") else pd.DataFrame(frame)


def split_snapshots_by_date(df: pd.DataFrame) -> dict[pd.Timestamp, pd.DataFrame]:
    """Split a multi-day ThetaData frame into one chain per snapshot date."""

    if df is None or df.empty:
        return {}
    out = df.copy()
    if "snapshot_date" not in out.columns:
        source_col = "created" if "created" in out.columns else "snapshot_date"
        out["snapshot_date"] = _normalize_snapshot_dates(out[source_col])
    else:
        out["snapshot_date"] = _normalize_snapshot_dates(out["snapshot_date"])
    snapshots: dict[pd.Timestamp, pd.DataFrame] = {}
    for snapshot_date, group in out.groupby("snapshot_date", dropna=True):
        ts = pd.Timestamp(snapshot_date).normalize()
        snapshots[ts] = group.copy()
    return dict(sorted(snapshots.items(), key=lambda item: item[0]))


def option_chain_storage_symbol(symbol: str) -> str:
    return str(symbol).strip().upper()


def read_option_chain_arctic(
    symbol: str,
    *,
    start_date: date | str | pd.Timestamp | None = None,
    end_date: date | str | pd.Timestamp | None = None,
    backend: ArcticBackend | None = None,
    config: WarehouseConfig | None = None,
) -> pd.DataFrame:
    backend = backend or open_backend(config or WarehouseConfig.from_env())
    frame = backend.read(OPTIONS_THETADATA_EOD_LIBRARY, option_chain_storage_symbol(symbol))
    if frame is None or frame.empty:
        return pd.DataFrame()
    out = frame.copy()
    if "snapshot_date" in out.columns:
        snapshot_dates = _normalize_snapshot_dates(out["snapshot_date"])
    else:
        snapshot_dates = _normalize_snapshot_dates(pd.Series(out.index, index=out.index))
        out["snapshot_date"] = snapshot_dates
    if start_date is not None:
        out = out.loc[snapshot_dates >= pd.Timestamp(start_date).normalize()]
    if end_date is not None:
        snapshot_dates = _normalize_snapshot_dates(out["snapshot_date"])
        out = out.loc[snapshot_dates <= pd.Timestamp(end_date).normalize()]
    return out.sort_index()


def write_option_chain_arctic(
    symbol: str,
    frame: pd.DataFrame,
    *,
    backend: ArcticBackend | None = None,
    config: WarehouseConfig | None = None,
    merge: bool = True,
) -> str:
    if frame is None or frame.empty:
        return _arctic_ref(symbol)
    backend = backend or open_backend(config or WarehouseConfig.from_env())
    storage_symbol = option_chain_storage_symbol(symbol)
    incoming = _prepare_option_chain_for_arctic(frame)
    existing = backend.read(OPTIONS_THETADATA_EOD_LIBRARY, storage_symbol) if merge else None
    merged = _merge_option_chain_upsert(existing, incoming)
    if not merged.empty:
        backend.write(OPTIONS_THETADATA_EOD_LIBRARY, storage_symbol, merged, prune_previous_versions=True)
    return _arctic_ref(symbol)


def option_chain_snapshots_cached(
    symbol: str,
    snapshot_dates: Sequence[date | str | pd.Timestamp],
    *,
    backend: ArcticBackend | None = None,
    config: WarehouseConfig | None = None,
) -> dict[pd.Timestamp, pd.DataFrame]:
    dates = [pd.Timestamp(value).normalize() for value in snapshot_dates]
    if not dates:
        return {}
    frame = read_option_chain_arctic(
        symbol,
        start_date=min(dates),
        end_date=max(dates),
        backend=backend,
        config=config,
    )
    if frame.empty:
        return {}
    requested = set(dates)
    snapshots: dict[pd.Timestamp, pd.DataFrame] = {}
    snapshot_dates = _normalize_snapshot_dates(frame["snapshot_date"])
    for ts, group in frame.groupby(snapshot_dates, sort=True):
        normalized = pd.Timestamp(ts).normalize()
        if normalized in requested:
            out = group.reset_index(drop=True)
            snapshots[normalized] = normalize_thetadata_option_chain(out, require_bid_ask=True)
    return snapshots


def option_chain_range_cached(
    symbol: str,
    start_date: date | str | pd.Timestamp,
    end_date: date | str | pd.Timestamp,
    *,
    backend: ArcticBackend | None = None,
    config: WarehouseConfig | None = None,
) -> bool:
    dates = [ts.normalize() for ts in pd.date_range(start_date, end_date, freq="B")]
    if not dates:
        return False
    cached = option_chain_snapshots_cached(symbol, dates, backend=backend, config=config)
    return len(cached) == len(dates)


def load_thetadata_option_snapshots(
    symbol: str,
    snapshot_dates: Sequence[date | str | pd.Timestamp],
    *,
    api_key: str | None = None,
    max_dte: int = 60,
    strike_range: int = 10,
    dataframe_type: str = "pandas",
    use_cache: bool = True,
    download_spec: ThetaDataDownloadSpec | None = None,
    download_missing: bool = True,
) -> dict[pd.Timestamp, pd.DataFrame]:
    """Load EOD option snapshots keyed by date, using ArcticDB as the cache/store."""

    spec = download_spec or ThetaDataDownloadSpec(
        max_dte=max_dte,
        strike_range=strike_range,
        dataframe_type=dataframe_type,
    )
    normalized_dates = [pd.Timestamp(value).normalize() for value in snapshot_dates]
    arctic_backend = open_backend(WarehouseConfig.from_env()) if use_cache else None
    snapshots: dict[pd.Timestamp, pd.DataFrame] = {}
    missing: list[pd.Timestamp] = []

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
            if use_cache and not frame.empty and arctic_backend is not None:
                write_option_chain_arctic(symbol, frame, backend=arctic_backend)

    return {ts: snapshots[ts] for ts in normalized_dates if ts in snapshots}


def download_option_snapshots_for_range(
    symbol: str,
    start_date: date | str | pd.Timestamp,
    end_date: date | str | pd.Timestamp,
    *,
    api_key: str | None = None,
    spec: ThetaDataDownloadSpec | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Download and cache daily ThetaData EOD option chains in ArcticDB."""

    download_spec = spec or ThetaDataDownloadSpec()
    arctic_backend = open_backend(WarehouseConfig.from_env())
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()

    requested_dates = [ts.normalize() for ts in pd.date_range(start, end, freq="B")]
    cached = (
        {}
        if overwrite
        else _read_cached_snapshots(
            symbol,
            requested_dates,
            backend=arctic_backend,
        )
    )
    missing_dates = [ts for ts in requested_dates if ts not in cached]

    if not requested_dates or not missing_dates:
        paths = [_arctic_ref(symbol) for _ts in cached]
        return {
            "symbol": str(symbol).upper(),
            "start_date": start.date().isoformat(),
            "end_date": end.date().isoformat(),
            "snapshot_days": len(cached),
            "contracts_total": int(sum(len(frame) for frame in cached.values())),
            "cached_days": len(cached),
            "fetched_rows": 0,
            "cached_only": True,
            "paths": paths,
            "spec": _download_spec_manifest(download_spec),
        }

    snapshots, fetched_rows, _written_paths = _download_and_cache_snapshots(
        symbol,
        missing_dates,
        api_key=api_key,
        spec=download_spec,
        backend=arctic_backend,
        overwrite=overwrite,
    )
    snapshots = {**cached, **snapshots}
    paths = [_arctic_ref(symbol) for _ts in snapshots]

    manifest = {
        "symbol": str(symbol).upper(),
        "start_date": start.date().isoformat(),
        "end_date": end.date().isoformat(),
        "snapshot_days": len(snapshots),
        "contracts_total": int(sum(len(frame) for frame in snapshots.values())),
        "cached_days": len(cached),
        "fetched_rows": int(fetched_rows),
        "cached_only": False,
        "paths": paths,
        "spec": _download_spec_manifest(download_spec),
    }
    return manifest


def _read_cached_snapshots(
    symbol: str,
    requested_dates: Sequence[pd.Timestamp],
    *,
    backend: ArcticBackend | None = None,
) -> dict[pd.Timestamp, pd.DataFrame]:
    if backend is not None:
        arctic = option_chain_snapshots_cached(symbol, requested_dates, backend=backend)
        if arctic:
            return arctic
    return {}


def _iter_contiguous_business_date_ranges(
    requested_dates: Sequence[pd.Timestamp],
) -> Iterator[tuple[pd.Timestamp, pd.Timestamp]]:
    dates = sorted({pd.Timestamp(ts).normalize() for ts in requested_dates})
    if not dates:
        return

    range_start = dates[0]
    previous = dates[0]
    for current in dates[1:]:
        next_business_day = pd.bdate_range(previous, periods=2)[-1].normalize()
        if current != next_business_day:
            yield range_start, previous
            range_start = current
        previous = current
    yield range_start, previous


def _iter_bounded_business_date_ranges(
    requested_dates: Sequence[pd.Timestamp],
    *,
    max_calendar_days: int = THETADATA_BACKFILL_WINDOW_DAYS,
) -> Iterator[tuple[pd.Timestamp, pd.Timestamp]]:
    dates = sorted({pd.Timestamp(ts).normalize() for ts in requested_dates})
    if not dates:
        return

    range_start = dates[0]
    previous = dates[0]
    for current in dates[1:]:
        next_business_day = pd.bdate_range(previous, periods=2)[-1].normalize()
        window_too_wide = (current - range_start).days >= int(max_calendar_days)
        if current != next_business_day or window_too_wide:
            yield range_start, previous
            range_start = current
        previous = current
    yield range_start, previous


def _download_and_cache_snapshots(
    symbol: str,
    requested_dates: Sequence[pd.Timestamp],
    *,
    api_key: str | None,
    spec: ThetaDataDownloadSpec,
    backend: ArcticBackend | None,
    overwrite: bool,
) -> tuple[dict[pd.Timestamp, pd.DataFrame], int, list[str]]:
    snapshots: dict[pd.Timestamp, pd.DataFrame] = {}
    paths: list[str] = []
    fetched_rows = 0
    requested = {pd.Timestamp(ts).normalize() for ts in requested_dates}

    for start, end in _iter_bounded_business_date_ranges(requested_dates):
        try:
            fetched = fetch_option_history_eod(symbol, start, end, spec=spec)
        except Exception:
            continue
        fetched_rows += len(fetched)
        for ts, frame in split_snapshots_by_date(fetched).items():
            if ts not in requested or frame.empty:
                continue
            snapshots[ts] = frame

    missing_dates = [ts for ts in requested_dates if ts not in snapshots]
    for ts in missing_dates:
        try:
            day_frame = fetch_option_history_eod(symbol, ts, ts, spec=spec)
        except Exception:
            continue
        if day_frame.empty:
            continue
        fetched_rows += len(day_frame)
        for day_ts, frame in split_snapshots_by_date(day_frame).items():
            if day_ts not in requested or frame.empty:
                continue
            snapshots[day_ts] = frame

    if backend is not None and snapshots:
        combined = pd.concat(snapshots.values(), ignore_index=True, sort=False)
        paths.append(write_option_chain_arctic(symbol, combined, backend=backend, merge=not overwrite))

    return snapshots, fetched_rows, paths


def load_cached_snapshots_for_trade_window(
    symbol: str,
    entry_date: date | str | pd.Timestamp,
    exit_date: date | str | pd.Timestamp,
    *,
    api_key: str | None = None,
    spec: ThetaDataDownloadSpec | None = None,
    download_missing: bool = True,
) -> dict[pd.Timestamp, pd.DataFrame]:
    """Load per-day chains for a trade window, optionally downloading missing days."""

    start = pd.Timestamp(entry_date).normalize()
    end = pd.Timestamp(exit_date).normalize()
    dates = list(pd.date_range(start, end, freq="B"))
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


def _download_spec_manifest(spec: ThetaDataDownloadSpec) -> dict[str, Any]:
    return {
        "data_interval": spec.data_interval,
        "max_dte": spec.max_dte,
        "strike_range": spec.strike_range,
        "expiration": spec.expiration,
        "right": spec.right,
        "require_bid_ask": spec.require_bid_ask,
        "min_ask": spec.min_ask,
    }


def _add_quote_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "bid" in out.columns:
        out["bid"] = pd.to_numeric(out["bid"], errors="coerce")
    if "ask" in out.columns:
        out["ask"] = pd.to_numeric(out["ask"], errors="coerce")
    if "bid" in out.columns and "ask" in out.columns:
        out["mid"] = (out["bid"] + out["ask"]) / 2.0
    return out


def _prepare_option_chain_for_arctic(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = normalize_thetadata_option_chain(frame, require_bid_ask=True)
    if normalized.empty:
        return normalized
    out = normalized.loc[:, ~normalized.columns.duplicated()].copy()
    out["snapshot_date"] = _normalize_snapshot_dates(out["snapshot_date"])
    out = out.dropna(subset=["snapshot_date", "contract_symbol"])
    offsets = out.groupby("snapshot_date").cumcount()
    index = out["snapshot_date"] + pd.to_timedelta(offsets, unit="ns")
    out.index = pd.DatetimeIndex(index)
    out.index.name = "timestamp"
    return _sanitize_option_chain_for_arctic(out.sort_index())


def _merge_option_chain_upsert(
    existing: pd.DataFrame | None,
    incoming: pd.DataFrame,
) -> pd.DataFrame:
    if incoming is None or incoming.empty:
        if existing is None or existing.empty:
            return pd.DataFrame()
        return existing.copy()
    if existing is None or existing.empty:
        return incoming.sort_index()
    combined = pd.concat([existing, incoming]).reset_index()
    dedupe_cols = ["snapshot_date", "contract_symbol"]
    combined = combined.drop_duplicates(subset=dedupe_cols, keep="last")
    combined["snapshot_date"] = pd.to_datetime(combined["snapshot_date"], errors="coerce")
    combined = combined.dropna(subset=["snapshot_date"]).drop(columns=["timestamp"], errors="ignore")
    return _prepare_option_chain_for_arctic(combined)


def _sanitize_option_chain_for_arctic(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out = out.dropna(axis=1, how="all")
    date_columns = {
        "snapshot_date",
        "eod_date",
        "expiration",
        "last_trade_time",
        "bid_time",
        "ask_time",
        "close_time",
        "close_bid_time",
        "close_ask_time",
    }
    for column in list(out.columns):
        if column in date_columns:
            out[column] = pd.to_datetime(out[column], errors="coerce", utc=True).dt.tz_localize(None)
        elif is_object_dtype(out[column]) or is_string_dtype(out[column]):
            out[column] = out[column].where(out[column].notna(), "").astype(str).astype(object)
    return out


def _arctic_ref(symbol: str) -> str:
    return f"arctic://{OPTIONS_THETADATA_EOD_LIBRARY}/{option_chain_storage_symbol(symbol)}"


def _normalize_snapshot_dates(values: pd.Series) -> pd.Series:
    converted = pd.to_datetime(values, errors="coerce", utc=True)
    if isinstance(converted, pd.Series):
        return converted.dt.tz_localize(None).dt.normalize()
    return pd.Series(converted).dt.tz_localize(None).dt.normalize()


def _filter_quoteable_rows(
    frame: pd.DataFrame,
    *,
    require_bid_ask: bool,
    min_ask: float,
) -> pd.DataFrame:
    if frame.empty or not require_bid_ask:
        return frame
    if "bid" not in frame.columns or "ask" not in frame.columns:
        raise ValueError("Daily EOD option chains must include bid and ask columns")

    out = _add_quote_columns(frame)
    bid = pd.to_numeric(out["bid"], errors="coerce")
    ask = pd.to_numeric(out["ask"], errors="coerce")
    quoteable = bid.notna() & ask.notna() & (bid > 0.0) & (ask >= float(min_ask)) & (ask >= bid)
    return out.loc[quoteable].copy()


def normalize_thetadata_option_chain(
    df: pd.DataFrame,
    *,
    require_bid_ask: bool = True,
    min_ask: float = 0.01,
) -> pd.DataFrame:
    """Normalize daily ThetaData EOD chains; keep rows with usable bid/ask quotes."""

    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    out.columns = [str(col).strip().lower() for col in out.columns]
    rename_map = {
        "symbol": "underlying_symbol",
        "right": "option_type",
        "created": "created_at",
        "last_trade": "last_trade_time",
        "close": "last_trade_price",
        "open": "open_price",
        "high": "high_price",
        "low": "low_price",
    }
    rename_map = {
        source: target
        for source, target in rename_map.items()
        if source in out.columns and (target not in out.columns or source == target)
    }
    out = out.rename(columns=rename_map)
    out = out.loc[:, ~out.columns.duplicated()].copy()
    if "snapshot_date" not in out.columns:
        if "eod_date" in out.columns:
            out["snapshot_date"] = out["eod_date"]
        elif "created_at" in out.columns:
            out["snapshot_date"] = _normalize_snapshot_dates(out["created_at"])
    out["underlying_symbol"] = out["underlying_symbol"].astype(str).str.upper()
    out["option_type"] = out["option_type"].astype(str).str.strip().str.lower()
    out["expiration"] = pd.to_datetime(out["expiration"], errors="coerce").dt.normalize()
    out["strike"] = pd.to_numeric(out["strike"], errors="coerce")
    if "snapshot_date" in out.columns:
        out["snapshot_date"] = _normalize_snapshot_dates(out["snapshot_date"])
    else:
        out["snapshot_date"] = pd.NaT
    if "contract_symbol" not in out.columns:
        out["contract_symbol"] = (
            out["underlying_symbol"].fillna("")
            + "_"
            + out["option_type"].fillna("")
            + "_"
            + out["expiration"].dt.strftime("%Y%m%d").fillna("")
            + "_"
            + out["strike"].fillna(0).map(lambda v: f"{float(v):g}")
        )
    out = _filter_quoteable_rows(out, require_bid_ask=require_bid_ask, min_ask=min_ask)
    if out.empty:
        return out
    out["data_interval"] = "eod"
    return out.reset_index(drop=True)
