from __future__ import annotations

from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from time import sleep
from typing import Any

import polars as pl

from quant_warehouse.ingest.openbb_fetch import fetch_dataframe

NewsFetcher = Callable[..., pl.DataFrame]


@dataclass(frozen=True)
class OracleNewsRefreshResult:
    requests: int
    boundary_rows: int
    news_rows: int
    output_path: Path


def _date_expr(frame: pl.DataFrame, name: str) -> pl.Expr:
    return ((pl.col(name).str.to_datetime(strict=False, time_zone="UTC") if frame.schema[name] == pl.String
             else pl.col(name).cast(pl.Datetime, strict=False)).dt.replace_time_zone(None).dt.truncate("1d"))


def oracle_trade_boundaries(trades: pl.DataFrame) -> pl.DataFrame:
    required = {"symbol", "entry_date", "exit_date"}
    missing = sorted(required.difference(trades.columns))
    if missing:
        raise ValueError(f"Missing oracle trade columns: {', '.join(missing)}")
    pieces = [trades.select("symbol", pl.col(column).alias("observation_date"),
                            pl.lit(kind).alias("boundary_kind"))
              for column, kind in (("entry_date", "entry"), ("exit_date", "exit"))]
    out = pl.concat(pieces).with_columns(
        pl.col("symbol").cast(pl.String, strict=False).str.strip_chars().str.to_uppercase(),
        _date_expr(pl.concat(pieces), "observation_date"),
    ).filter((pl.col("symbol") != "") & pl.col("observation_date").is_not_null())
    return (out.group_by(["symbol", "observation_date"], maintain_order=True)
            .agg(pl.col("boundary_kind").unique().sort().str.join(","))
            .sort(["symbol", "observation_date"]))


def _symbols_from_value(value: Any) -> list[str]:
    values: Iterable[Any] = value if isinstance(value, (list, tuple, set)) else str(value or "").replace(";", ",").split(",")
    return [str(item).strip().upper() for item in values if str(item).strip()]


def _normalize_news(frame: pl.DataFrame, boundaries: pl.DataFrame) -> pl.DataFrame:
    if frame.is_empty():
        return pl.DataFrame()
    published_column = next((name for name in ("published_at", "date", "published_date", "publishedDate") if name in frame.columns), None)
    symbol_column = next((name for name in ("symbols", "symbol") if name in frame.columns), None)
    if published_column is None or symbol_column is None:
        raise ValueError("FMP company news response lacks symbol or publication timestamp")
    published = _date_expr(frame, published_column)
    out = frame.with_columns(published.alias("published_at"), published.alias("observation_date"))
    out = out.with_columns(pl.col(symbol_column).map_elements(_symbols_from_value, return_dtype=pl.List(pl.String)).alias("symbol"))
    out = out.explode("symbol").with_columns(pl.col("symbol").str.to_uppercase())
    out = out.join(boundaries, on=["symbol", "observation_date"], how="inner").with_columns(
        pl.lit("fmp").alias("provider"), pl.lit(datetime.now()).alias("fetched_at"))
    drop = [name for name in (symbol_column, published_column) if name != "published_at" and name in out.columns]
    if drop:
        out = out.drop(drop)
    preferred = ["symbol", "observation_date", "boundary_kind", "published_at", "title", "url", "source", "author", "excerpt", "provider", "fetched_at"]
    return out.select([name for name in preferred if name in out.columns] + [name for name in out.columns if name not in preferred])


def refresh_fmp_news_for_oracle_trades(
    trades: pl.DataFrame, *, output_path: str | Path, fetcher: NewsFetcher = fetch_dataframe,
    page_limit: int = 250, max_pages: int = 101, checkpoint_dir: str | Path | None = None,
    max_workers: int = 4, max_retries: int = 3,
) -> OracleNewsRefreshResult:
    boundaries = oracle_trade_boundaries(trades)
    path = Path(output_path).expanduser().resolve()
    parts_path = Path(checkpoint_dir).expanduser().resolve() if checkpoint_dir else path.parent / f".{path.stem}.parts"
    parts_path.mkdir(parents=True, exist_ok=True)
    requests = 0

    def fetch_day(item: tuple[tuple[Any, ...], pl.DataFrame]) -> tuple[pl.DataFrame, int]:
        (observation_date,), group = item
        day = observation_date.date().isoformat()
        part_path = parts_path / f"{day}.parquet"
        if part_path.exists():
            return pl.read_parquet(part_path), 0
        symbols = ",".join(sorted(group["symbol"].unique().to_list()))
        frames: list[pl.DataFrame] = []
        day_requests = 0
        for page in range(max_pages):
            frame = None
            for attempt in range(max(1, int(max_retries) + 1)):
                try:
                    frame = fetcher("company_news", symbol=symbols, provider="fmp", start_date=day, end_date=day, page=page, limit=page_limit)
                    day_requests += 1
                    break
                except Exception:
                    day_requests += 1
                    if attempt >= max_retries:
                        raise
                    sleep(0.5 * (2 ** attempt))
            if frame is None or frame.is_empty():
                break
            frames.append(frame)
            if frame.height < page_limit:
                break
        part = _normalize_news(pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame(), group)
        part.write_parquet(part_path)
        return part, day_requests

    items = list(boundaries.group_by("observation_date", maintain_order=True))
    parts: list[pl.DataFrame] = []
    with ThreadPoolExecutor(max_workers=max(1, int(max_workers))) as executor:
        for part, day_requests in executor.map(fetch_day, items):
            requests += day_requests
            if not part.is_empty():
                parts.append(part)
    news = pl.concat(parts, how="diagonal_relaxed") if parts else pl.DataFrame()
    if not news.is_empty():
        dedupe = [name for name in ("symbol", "published_at", "url", "title") if name in news.columns]
        news = news.unique(dedupe, keep="last").sort([name for name in ("observation_date", "symbol", "published_at") if name in news.columns])
    path.parent.mkdir(parents=True, exist_ok=True)
    news.write_parquet(path)
    return OracleNewsRefreshResult(requests, boundaries.height, news.height, path)


def select_oracle_trades(trades: pl.DataFrame, *, symbols: Iterable[str], k_values: Iterable[int] = (1, 2, 3),
                         frequency: str = "YE", start_date: str | date | None = None,
                         end_date: str | date | None = None) -> pl.DataFrame:
    out = trades.with_columns(pl.col("symbol").cast(pl.String, strict=False).str.to_uppercase())
    wanted = [str(symbol).strip().upper() for symbol in symbols]
    out = out.filter(pl.col("symbol").is_in(wanted))
    if "freq" in out.columns:
        out = out.filter(pl.col("freq").cast(pl.String, strict=False) == frequency)
    if "k" in out.columns:
        out = out.filter(pl.col("k").cast(pl.Int64, strict=False).is_in(list(k_values)))
    if start_date is not None:
        out = out.filter(_date_expr(out, "entry_date") >= datetime.fromisoformat(str(start_date)[:10]))
    if end_date is not None:
        out = out.filter(_date_expr(out, "exit_date") <= datetime.fromisoformat(str(end_date)[:10]))
    return out
