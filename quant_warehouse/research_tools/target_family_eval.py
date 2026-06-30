from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from quant_warehouse.platforms.data_providers.fmp.target_engineering.event_pairs import (
    EVENT_PAIR_COLUMNS,
    EVENT_PAIR_TAXONOMY,
    EventPairStore,
    build_event_pairs_from_historical_data,
)
from quant_warehouse.platforms.data_providers.fmp.target_engineering.strategy_solver import solve_side_trades_by_frequency_batched_multi_k
from quant_warehouse.warehouse.api import Warehouse


@dataclass(frozen=True)
class BinaryTargetConfig:
    provider: str = "fmp"
    start_date: str = "2018-01-01"
    end_date: str | None = None
    event_families: tuple[str, ...] = ("congress", "insider", "analyst_rating", "price_target", "guidance", "earnings")
    event_windows: tuple[int, ...] = (20, 60)
    oracle_trade_k_by_frequency: dict[str, tuple[int, ...]] | None = None
    oracle_trade_min_profit_pct: float = 0.01
    oracle_trade_long_entry_price_col: str = "high"
    oracle_trade_long_exit_price_col: str = "low"
    oracle_trade_short_entry_price_col: str = "low"
    oracle_trade_short_exit_price_col: str = "high"
    event_alignment_tolerance_days: int = 7


def load_fmp_event_pairs(
    symbols: Iterable[str],
    config: BinaryTargetConfig,
    *,
    event_store: EventPairStore | None = None,
    include_historical: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """Load normalized FMP event pairs without refreshing remote data."""

    start = perf_counter()
    store = event_store or EventPairStore()
    frames: list[pd.DataFrame] = []
    diagnostics: list[dict[str, object]] = []
    families = _normalize_event_families(config.event_families)

    for symbol in _normalize_symbols(symbols):
        cached = store.read(
            symbol,
            provider=config.provider,
            event_families=families,
            start_date=config.start_date,
            end_date=config.end_date,
        )
        cached_families = set(cached["event_family"].dropna().astype(str)) if not cached.empty else set()
        missing = tuple(family for family in families if family not in cached_families)
        historical = pd.DataFrame(columns=EVENT_PAIR_COLUMNS)
        if include_historical and missing:
            historical = build_event_pairs_from_historical_data(
                symbol,
                fundamentals=store.fundamentals,
                equity_calendar=store.equity_calendar,
                event_families=missing,
                start_date=config.start_date,
                end_date=config.end_date,
                provider=config.provider,
            )
        candidates = [frame for frame in (cached, historical) if frame is not None and not frame.empty]
        combined = _dedupe_events(pd.concat(candidates, ignore_index=True)) if candidates else pd.DataFrame(columns=EVENT_PAIR_COLUMNS)
        diagnostics.append(
            {
                "symbol": symbol,
                "cached_rows": len(cached),
                "historical_rows": len(historical),
                "combined_rows": len(combined),
                "event_families": tuple(sorted(combined["event_family"].dropna().unique())) if not combined.empty else (),
            }
        )
        if not combined.empty:
            frames.append(combined)

    events = _dedupe_events(pd.concat(frames, ignore_index=True)) if frames else pd.DataFrame(columns=EVENT_PAIR_COLUMNS)
    return events, pd.DataFrame(diagnostics), perf_counter() - start


def build_event_target_panel(
    feature_panel: pd.DataFrame,
    events: pd.DataFrame,
    config: BinaryTargetConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create same-day and forward-window binary event targets on feature-panel dates."""

    base = _base_panel_dates(feature_panel)
    event_types = _event_types_for_families(config.event_families)
    target_columns = [f"target_event_on__{event_type}" for event_type in event_types]
    if base.empty:
        return base, _target_metadata(target_columns, "event")

    out = base.copy()
    for column in target_columns:
        out[column] = 0

    aligned_events = _align_events_to_panel_dates(base, events, tolerance_days=config.event_alignment_tolerance_days)
    if not aligned_events.empty:
        indicators = (
            aligned_events.assign(value=1)
            .pivot_table(index=["symbol", "date"], columns="event_type", values="value", aggfunc="max", fill_value=0)
            .reset_index()
        )
        indicators.columns = [
            f"target_event_on__{column}" if column not in {"symbol", "date"} else column
            for column in indicators.columns
        ]
        out = out.merge(indicators, on=["symbol", "date"], how="left", suffixes=("", "_event"))
        for column in target_columns:
            event_column = f"{column}_event"
            if event_column in out.columns:
                out[column] = out[event_column].fillna(out[column]).fillna(0).astype("int8")
                out = out.drop(columns=[event_column])
            else:
                out[column] = out[column].fillna(0).astype("int8")

    forward_columns: list[str] = []
    for window in sorted(set(int(value) for value in config.event_windows if int(value) > 0)):
        for source_column in target_columns:
            forward_column = source_column.replace("target_event_on__", f"target_event_next_{window}d__")
            out[forward_column] = _future_binary_by_symbol(out, source_column, window)
            forward_columns.append(forward_column)

    metadata = _target_metadata(target_columns + forward_columns, "event")
    return out, metadata


def build_oracle_trade_target_panel(
    symbols: Iterable[str],
    config: BinaryTargetConfig,
    *,
    warehouse: Warehouse | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """Create sparse oracle-trade entry labels using the batched top-k solver."""

    start = perf_counter()
    wh = warehouse or Warehouse()
    price_frames: dict[str, pd.DataFrame] = {}
    for symbol in _normalize_symbols(symbols):
        prices = wh.read_prices(symbol, provider=config.provider, start=config.start_date, end=config.end_date)
        if prices is None or prices.empty:
            continue
        price_frames[symbol] = prices
    if not price_frames:
        empty = pd.DataFrame(columns=["symbol", "date"])
        return empty, _target_metadata([], "oracle_trade"), perf_counter() - start

    base = _price_base_panel(price_frames)
    if base.empty:
        empty = pd.DataFrame(columns=["symbol", "date"])
        return empty, _target_metadata([], "oracle_trade"), perf_counter() - start

    out = base.copy()
    target_columns: list[str] = []
    k_by_frequency = config.oracle_trade_k_by_frequency or {"YE": tuple(range(1, 13))}
    row_lookup = _target_row_lookup(out)
    for freq, raw_ks in k_by_frequency.items():
        frequency = str(freq or "").strip().upper()
        ks = tuple(dict.fromkeys(int(k) for k in raw_ks if int(k) > 0))
        if not frequency or not ks:
            continue
        trades_by_k = solve_side_trades_by_frequency_batched_multi_k(
            price_frames,
            ks=ks,
            freq=frequency,
            min_profit_pct=float(config.oracle_trade_min_profit_pct),
            long_entry_price_col=config.oracle_trade_long_entry_price_col,
            long_exit_price_col=config.oracle_trade_long_exit_price_col,
            short_entry_price_col=config.oracle_trade_short_entry_price_col,
            short_exit_price_col=config.oracle_trade_short_exit_price_col,
        )
        for k in ks:
            long_col = f"target_oracle_trade_entry__{frequency}_k{k}_long"
            short_col = f"target_oracle_trade_entry__{frequency}_k{k}_short"
            any_col = f"target_oracle_trade_entry__{frequency}_k{k}_any"
            for column in (long_col, short_col, any_col):
                out[column] = 0
            target_columns.extend([long_col, short_col, any_col])
            _mark_oracle_trade_entries(
                out,
                trades_by_k.get(k, {}),
                long_col=long_col,
                short_col=short_col,
                any_col=any_col,
                row_lookup=row_lookup,
            )

    metadata = _target_metadata(target_columns, "oracle_trade")
    return out, metadata, perf_counter() - start


def combine_target_panels(*panels: pd.DataFrame) -> pd.DataFrame:
    """Outer-join target panels on symbol/date."""

    cleaned = []
    for panel in panels:
        if panel is None or panel.empty:
            continue
        out = panel.copy()
        out["symbol"] = out["symbol"].astype(str).str.upper()
        out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
        cleaned.append(out.dropna(subset=["symbol", "date"]))
    if not cleaned:
        return pd.DataFrame(columns=["symbol", "date"])
    combined = cleaned[0]
    for panel in cleaned[1:]:
        combined = combined.merge(panel, on=["symbol", "date"], how="outer")
    target_cols = [column for column in combined.columns if column.startswith("target_")]
    combined[target_cols] = combined[target_cols].fillna(0).astype("int8")
    return combined.sort_values(["symbol", "date"]).reset_index(drop=True)


def summarize_binary_targets(target_panel: pd.DataFrame, target_metadata: pd.DataFrame) -> pd.DataFrame:
    """Summarize target sparsity, date coverage, and symbol coverage."""

    rows = []
    for column in target_metadata.get("target", []):
        if column not in target_panel.columns:
            continue
        values = pd.to_numeric(target_panel[column], errors="coerce").fillna(0)
        positive = values.gt(0)
        positive_frame = target_panel.loc[positive, ["symbol", "date"]]
        rows.append(
            {
                "target": column,
                "target_family": target_metadata.set_index("target").loc[column, "target_family"],
                "rows": int(values.notna().sum()),
                "positive_rows": int(positive.sum()),
                "positive_rate": float(positive.mean()) if len(values) else np.nan,
                "positive_symbols": int(positive_frame["symbol"].nunique()) if not positive_frame.empty else 0,
                "min_positive_date": positive_frame["date"].min() if not positive_frame.empty else pd.NaT,
                "max_positive_date": positive_frame["date"].max() if not positive_frame.empty else pd.NaT,
            }
        )
    return pd.DataFrame(rows).sort_values(["target_family", "positive_rows"], ascending=[True, False]).reset_index(drop=True)


def evaluate_feature_target_matrix(
    feature_panel: pd.DataFrame,
    feature_metadata: pd.DataFrame,
    target_panel: pd.DataFrame,
    target_metadata: pd.DataFrame,
    *,
    min_rows: int = 120,
    min_positive_rows: int = 10,
    min_feature_coverage: float = 0.5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Measure which feature families can be joined to which binary targets."""

    merged = _merge_features_targets(feature_panel, target_panel)
    rows: list[dict[str, object]] = []
    family_groups = feature_metadata.groupby(["source", "family"], sort=True)
    target_lookup = target_metadata.set_index("target")["target_family"].to_dict()
    target_columns = [column for column in target_metadata["target"].tolist() if column in merged.columns]
    for (source, family), family_meta in family_groups:
        features = [feature for feature in family_meta["feature"].tolist() if feature in merged.columns]
        if not features:
            continue
        feature_values = merged[features].apply(pd.to_numeric, errors="coerce")
        feature_coverage = feature_values.notna().mean(axis=1)
        feature_mask = feature_coverage.ge(float(min_feature_coverage))
        for target in target_columns:
            target_values = pd.to_numeric(merged[target], errors="coerce").fillna(0).astype("int8")
            mask = feature_mask & target_values.notna()
            n_rows = int(mask.sum())
            positives = int(target_values.loc[mask].gt(0).sum())
            if n_rows < int(min_rows) or positives < int(min_positive_rows):
                status = "sparse"
            else:
                status = "usable"
            smd = _mean_abs_standardized_difference(feature_values.loc[mask], target_values.loc[mask])
            rows.append(
                {
                    "source": source,
                    "feature_family": family,
                    "target_family": target_lookup.get(target, ""),
                    "target": target,
                    "feature_count": len(features),
                    "rows": n_rows,
                    "positive_rows": positives,
                    "positive_rate": float(positives / n_rows) if n_rows else np.nan,
                    "mean_feature_coverage": float(feature_coverage.loc[mask].mean()) if n_rows else np.nan,
                    "mean_abs_smd": smd,
                    "status": status,
                }
            )
    matrix = pd.DataFrame(rows)
    if matrix.empty:
        return matrix, merged
    matrix = matrix.sort_values(
        ["status", "positive_rows", "mean_abs_smd", "rows"],
        ascending=[True, False, False, False],
    ).reset_index(drop=True)
    return matrix, merged


def _normalize_symbols(symbols: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(str(symbol).strip().upper() for symbol in symbols if str(symbol).strip()))


def _normalize_event_families(families: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(dict.fromkeys(str(family).strip().lower() for family in families if str(family).strip()))
    unknown = sorted(set(normalized) - set(EVENT_PAIR_TAXONOMY))
    if unknown:
        raise ValueError(f"Unsupported event families: {unknown}")
    return normalized


def _event_types_for_families(families: Sequence[str]) -> list[str]:
    event_types: list[str] = []
    for family in _normalize_event_families(families):
        pair = EVENT_PAIR_TAXONOMY[family]
        event_types.extend([pair["positive"], pair["negative"]])
    return event_types


def _dedupe_events(events: pd.DataFrame) -> pd.DataFrame:
    if events is None or events.empty:
        return pd.DataFrame(columns=EVENT_PAIR_COLUMNS)
    out = events.copy()
    for column in EVENT_PAIR_COLUMNS:
        if column not in out.columns:
            out[column] = np.nan
    out = out[EVENT_PAIR_COLUMNS]
    out["symbol"] = out["symbol"].astype(str).str.upper()
    out["event_date"] = pd.to_datetime(out["event_date"], errors="coerce").dt.normalize()
    out = out.dropna(subset=["symbol", "event_date", "event_family", "event_type"])
    return out.drop_duplicates(["symbol", "event_date", "event_family", "event_type", "actor_name", "strength"]).reset_index(drop=True)


def _base_panel_dates(feature_panel: pd.DataFrame) -> pd.DataFrame:
    if feature_panel is None or feature_panel.empty:
        return pd.DataFrame(columns=["symbol", "date"])
    base = feature_panel[["symbol", "date"]].copy()
    base["symbol"] = base["symbol"].astype(str).str.upper()
    base["date"] = pd.to_datetime(base["date"], errors="coerce").dt.normalize()
    return base.dropna(subset=["symbol", "date"]).drop_duplicates().sort_values(["symbol", "date"]).reset_index(drop=True)


def _align_events_to_panel_dates(base: pd.DataFrame, events: pd.DataFrame, *, tolerance_days: int) -> pd.DataFrame:
    events = _dedupe_events(events)
    if base.empty or events.empty:
        return pd.DataFrame(columns=["symbol", "date", "event_type"])
    aligned_frames: list[pd.DataFrame] = []
    tolerance = pd.Timedelta(days=int(tolerance_days))
    for symbol, symbol_events in events.groupby("symbol", sort=False):
        dates = base.loc[base["symbol"].eq(symbol), ["date"]].sort_values("date")
        if dates.empty:
            continue
        symbol_events = symbol_events.sort_values("event_date")
        aligned = pd.merge_asof(
            symbol_events,
            dates,
            left_on="event_date",
            right_on="date",
            direction="forward",
            tolerance=tolerance,
        )
        aligned = aligned.dropna(subset=["date"])
        if not aligned.empty:
            aligned_frames.append(aligned[["symbol", "date", "event_type"]])
    if not aligned_frames:
        return pd.DataFrame(columns=["symbol", "date", "event_type"])
    return pd.concat(aligned_frames, ignore_index=True).drop_duplicates()


def _future_binary_by_symbol(panel: pd.DataFrame, source_column: str, window: int) -> pd.Series:
    pieces: list[pd.Series] = []
    for _, group in panel.sort_values(["symbol", "date"]).groupby("symbol", sort=False):
        values = pd.to_numeric(group[source_column], errors="coerce").fillna(0).astype("int8")
        future = pd.concat([values.shift(-offset).fillna(0) for offset in range(1, int(window) + 1)], axis=1).max(axis=1)
        pieces.append(future.astype("int8"))
    if not pieces:
        return pd.Series(index=panel.index, dtype="int8")
    return pd.concat(pieces).sort_index().astype("int8")


def _target_metadata(columns: Sequence[str], family: str) -> pd.DataFrame:
    rows = []
    for column in columns:
        target_family = family
        if column.startswith("target_event_"):
            target_family = "event"
        elif column.startswith("target_oracle_trade_"):
            target_family = "oracle_trade"
        rows.append({"target": column, "target_family": target_family, "target_type": "binary"})
    return pd.DataFrame(rows, columns=["target", "target_family", "target_type"])


def _price_base_panel(price_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for symbol, frame in price_frames.items():
        if frame is None or frame.empty:
            continue
        dates = pd.to_datetime(frame.index, errors="coerce").normalize()
        rows.append(pd.DataFrame({"symbol": str(symbol).upper(), "date": dates}).dropna())
    if not rows:
        return pd.DataFrame(columns=["symbol", "date"])
    return pd.concat(rows, ignore_index=True).drop_duplicates().sort_values(["symbol", "date"]).reset_index(drop=True)


def _mark_oracle_trade_entries(
    target_panel: pd.DataFrame,
    trades_by_symbol: dict[str, list[dict[str, object]]],
    *,
    long_col: str,
    short_col: str,
    any_col: str,
    row_lookup: dict[tuple[str, pd.Timestamp], int] | None = None,
) -> None:
    if not trades_by_symbol:
        return
    lookup = row_lookup or _target_row_lookup(target_panel)
    for symbol, trades in trades_by_symbol.items():
        symbol_key = str(symbol).strip().upper()
        for trade in trades or []:
            entry_row = trade.get("entry_row")
            entry_date = getattr(entry_row, "name", None)
            date = pd.to_datetime(entry_date, errors="coerce")
            if pd.isna(date):
                continue
            target_date = pd.Timestamp(date).normalize()
            row_index = lookup.get((symbol_key, target_date))
            if row_index is None:
                continue
            side = str(trade.get("side") or "").strip().lower()
            if side == "long":
                target_panel.at[row_index, long_col] = 1
            elif side == "short":
                target_panel.at[row_index, short_col] = 1
            target_panel.at[row_index, any_col] = 1


def _target_row_lookup(target_panel: pd.DataFrame) -> dict[tuple[str, pd.Timestamp], int]:
    symbols = target_panel["symbol"].astype(str).str.upper()
    dates = pd.to_datetime(target_panel["date"], errors="coerce").dt.normalize()
    return {
        (symbol, pd.Timestamp(date)): int(index)
        for index, symbol, date in zip(target_panel.index, symbols, dates, strict=False)
        if symbol and not pd.isna(date)
    }


def _merge_features_targets(feature_panel: pd.DataFrame, target_panel: pd.DataFrame) -> pd.DataFrame:
    left = feature_panel.copy()
    right = target_panel.copy()
    left["symbol"] = left["symbol"].astype(str).str.upper()
    right["symbol"] = right["symbol"].astype(str).str.upper()
    left["date"] = pd.to_datetime(left["date"], errors="coerce").dt.normalize()
    right["date"] = pd.to_datetime(right["date"], errors="coerce").dt.normalize()
    return left.merge(right, on=["symbol", "date"], how="left")


def _mean_abs_standardized_difference(features: pd.DataFrame, target: pd.Series) -> float:
    if features.empty or target.empty or target.nunique(dropna=True) < 2:
        return np.nan
    positives = target.gt(0)
    if positives.sum() == 0 or (~positives).sum() == 0:
        return np.nan
    smds = []
    for column in features.columns:
        values = pd.to_numeric(features[column], errors="coerce")
        pos = values.loc[positives]
        neg = values.loc[~positives]
        if pos.notna().sum() < 2 or neg.notna().sum() < 2:
            continue
        pooled_std = np.sqrt((float(pos.var()) + float(neg.var())) / 2.0)
        if not np.isfinite(pooled_std) or pooled_std == 0.0:
            continue
        smds.append(abs(float(pos.mean()) - float(neg.mean())) / pooled_std)
    return float(np.nanmean(smds)) if smds else np.nan
