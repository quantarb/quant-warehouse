from __future__ import annotations

from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from time import sleep
from typing import Any

import pandas as pd

from quant_warehouse.ingest.openbb_fetch import fetch_dataframe

NewsFetcher = Callable[..., pd.DataFrame]


@dataclass(frozen=True)
class OracleNewsRefreshResult:
    requests: int
    boundary_rows: int
    news_rows: int
    output_path: Path


def oracle_trade_boundaries(trades: pd.DataFrame) -> pd.DataFrame:
    """Return unique symbol/date oracle entry and exit observations.

    A boundary can be both an exit and a new entry. It remains one provider request,
    while ``boundary_kind`` preserves both uses for downstream point-in-time joins.
    """
    required = {"symbol", "entry_date", "exit_date"}
    missing = sorted(required.difference(trades.columns))
    if missing:
        raise ValueError(f"Missing oracle trade columns: {', '.join(missing)}")

    pieces: list[pd.DataFrame] = []
    for column, kind in (("entry_date", "entry"), ("exit_date", "exit")):
        piece = trades.loc[:, ["symbol", column]].rename(columns={column: "observation_date"})
        piece["boundary_kind"] = kind
        pieces.append(piece)
    out = pd.concat(pieces, ignore_index=True)
    out["symbol"] = out["symbol"].astype(str).str.strip().str.upper()
    out["observation_date"] = pd.to_datetime(out["observation_date"], errors="coerce").dt.normalize()
    out = out.loc[out["symbol"].ne("") & out["observation_date"].notna()]
    out = (
        out.groupby(["symbol", "observation_date"], as_index=False, sort=True)["boundary_kind"]
        .agg(lambda values: ",".join(sorted(set(values))))
    )
    return out


def _symbols_from_value(value: Any) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        values: Iterable[Any] = value
    else:
        values = str(value or "").replace(";", ",").split(",")
    return [str(item).strip().upper() for item in values if str(item).strip()]


def _normalize_news(frame: pd.DataFrame, boundaries: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    out = frame.copy()
    published_column = next(
        (name for name in ("published_at", "date", "published_date", "publishedDate") if name in out.columns),
        None,
    )
    symbol_column = next((name for name in ("symbols", "symbol") if name in out.columns), None)
    if published_column is None and isinstance(out.index, pd.DatetimeIndex):
        out["published_at"] = pd.to_datetime(out.index, errors="coerce", utc=True)
    elif published_column is not None:
        out["published_at"] = pd.to_datetime(out[published_column], errors="coerce", utc=True)
    if "published_at" not in out or symbol_column is None:
        raise ValueError("FMP company news response lacks symbol or publication timestamp")
    out["observation_date"] = out["published_at"].dt.tz_convert(None).dt.normalize()
    out["symbol"] = out[symbol_column].map(_symbols_from_value)
    out = out.explode("symbol", ignore_index=True)
    out = out.merge(boundaries, on=["symbol", "observation_date"], how="inner")
    out["provider"] = "fmp"
    out["fetched_at"] = pd.Timestamp.now(tz="UTC")
    drop_columns = [symbol_column]
    if published_column != "published_at":
        drop_columns.append(published_column)
    out = out.drop(columns=drop_columns, errors="ignore")
    preferred = [
        "symbol", "observation_date", "boundary_kind", "published_at", "title",
        "url", "source", "author", "excerpt", "provider", "fetched_at",
    ]
    columns = [column for column in preferred if column in out.columns]
    columns.extend(column for column in out.columns if column not in columns)
    return out.loc[:, columns]


def refresh_fmp_news_for_oracle_trades(
    trades: pd.DataFrame,
    *,
    output_path: str | Path,
    fetcher: NewsFetcher = fetch_dataframe,
    page_limit: int = 250,
    max_pages: int = 101,
    checkpoint_dir: str | Path | None = None,
    max_workers: int = 4,
    max_retries: int = 3,
) -> OracleNewsRefreshResult:
    """Download FMP news only on oracle entry/exit dates and write one parquet panel."""
    boundaries = oracle_trade_boundaries(trades)
    path = Path(output_path).expanduser().resolve()
    parts_path = (
        Path(checkpoint_dir).expanduser().resolve()
        if checkpoint_dir is not None
        else path.parent / f".{path.stem}.parts"
    )
    parts_path.mkdir(parents=True, exist_ok=True)
    normalized_frames: list[pd.DataFrame] = []
    requests = 0

    def fetch_day(item: tuple[pd.Timestamp, pd.DataFrame]) -> tuple[pd.DataFrame, int]:
        observation_date, group = item
        day = pd.Timestamp(observation_date).date().isoformat()
        part_path = parts_path / f"{day}.parquet"
        if part_path.exists():
            part = pd.read_parquet(part_path)
            return part, 0
        symbols = ",".join(sorted(group["symbol"].unique()))
        day_frames: list[pd.DataFrame] = []
        day_requests = 0
        for page in range(max_pages):
            frame = None
            for attempt in range(max(1, int(max_retries) + 1)):
                try:
                    frame = fetcher(
                        "company_news",
                        symbol=symbols,
                        provider="fmp",
                        start_date=day,
                        end_date=day,
                        page=page,
                        limit=page_limit,
                    )
                    day_requests += 1
                    break
                except Exception:
                    day_requests += 1
                    if attempt >= max_retries:
                        raise
                    sleep(0.5 * (2**attempt))
            if frame is None or frame.empty:
                break
            if (isinstance(frame.index, pd.DatetimeIndex) or frame.index.name == "date") and not any(
                name in frame.columns for name in ("date", "published_date", "publishedDate", "published_at")
            ):
                frame = frame.copy()
                frame["published_at"] = pd.to_datetime(frame.index, errors="coerce", utc=True)
            day_frames.append(frame)
            if len(frame) < page_limit:
                break
        raw = pd.concat(day_frames, ignore_index=True) if day_frames else pd.DataFrame()
        part = _normalize_news(raw, group)
        part.to_parquet(part_path, index=False)
        return part, day_requests

    items = list(boundaries.groupby("observation_date", sort=True))
    with ThreadPoolExecutor(max_workers=max(1, int(max_workers))) as executor:
        for part, day_requests in executor.map(fetch_day, items):
            requests += day_requests
            if not part.empty:
                normalized_frames.append(part)

    news = pd.concat(normalized_frames, ignore_index=True) if normalized_frames else pd.DataFrame()
    if not news.empty:
        dedupe = [column for column in ("symbol", "published_at", "url", "title") if column in news]
        news = news.drop_duplicates(dedupe).sort_values(["observation_date", "symbol", "published_at"])
    path.parent.mkdir(parents=True, exist_ok=True)
    news.to_parquet(path, index=False)
    return OracleNewsRefreshResult(requests, len(boundaries), len(news), path)


def select_oracle_trades(
    trades: pd.DataFrame,
    *,
    symbols: Iterable[str],
    k_values: Iterable[int] = (1, 2, 3),
    frequency: str = "YE",
    start_date: str | date | None = None,
    end_date: str | date | None = None,
) -> pd.DataFrame:
    out = trades.copy()
    wanted_symbols = {str(symbol).strip().upper() for symbol in symbols}
    out = out.loc[out["symbol"].astype(str).str.upper().isin(wanted_symbols)]
    if "freq" in out:
        out = out.loc[out["freq"].astype(str).eq(frequency)]
    if "k" in out:
        out = out.loc[pd.to_numeric(out["k"], errors="coerce").isin(list(k_values))]
    if start_date is not None:
        out = out.loc[pd.to_datetime(out["entry_date"]) >= pd.Timestamp(start_date)]
    if end_date is not None:
        out = out.loc[pd.to_datetime(out["exit_date"]) <= pd.Timestamp(end_date)]
    return out.reset_index(drop=True)
