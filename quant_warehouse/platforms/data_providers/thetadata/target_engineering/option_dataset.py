from __future__ import annotations

import polars as pl

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence


from quant_warehouse.platforms.data_providers.thetadata.target_engineering.option_labels import (
    OptionLabelSpec,
    build_option_label_panel,
)
from quant_warehouse.platforms.data_providers.thetadata.options import (
    ThetaDataDownloadSpec,
    load_cached_snapshots_for_trade_window,
)


@dataclass(frozen=True)
class OptionMlDatasetSpec:
    """Build ML rows for single-leg rank and multi-leg MV targets."""

    rank_spec: OptionLabelSpec = field(default_factory=OptionLabelSpec)
    mv_spec: OptionLabelSpec = field(default_factory=OptionLabelSpec.diversified_mean_variance)
    hybrid_spec: OptionLabelSpec = field(default_factory=OptionLabelSpec.diversified_hybrid)
    thetadata: ThetaDataDownloadSpec = field(default_factory=ThetaDataDownloadSpec)
    download_missing: bool = True


@dataclass(frozen=True)
class OptionMlDatasetResult:
    rows: list[dict[str, Any]] = field(default_factory=list)
    statistics: dict[str, Any] = field(default_factory=dict)


def build_option_ml_dataset(
    trades: Sequence[Mapping[str, Any]] | pl.DataFrame,
    *,
    dataset_spec: OptionMlDatasetSpec | None = None,
    label_specs: Sequence[OptionLabelSpec] | None = None,
) -> OptionMlDatasetResult:
    """Build per-contract ML rows with rank and MV portfolio labels for each trade."""

    dataset_spec = dataset_spec or OptionMlDatasetSpec()
    specs = list(label_specs) if label_specs is not None else [
        dataset_spec.rank_spec,
        dataset_spec.mv_spec,
        dataset_spec.hybrid_spec,
    ]
    trade_rows = _normalize_trade_rows(trades)
    if not trade_rows:
        return OptionMlDatasetResult()

    combined_frames: list[pl.DataFrame] = []
    for trade in trade_rows:
        symbol = str(trade.get("symbol") or trade.get("underlying_symbol") or "").strip().upper()
        entry_dt = datetime.fromisoformat(str(trade["entry_date"])[:10])
        exit_dt = datetime.fromisoformat(str(trade["exit_date"])[:10])
        if not symbol:
            continue

        snapshots = load_cached_snapshots_for_trade_window(
            symbol,
            entry_dt,
            exit_dt,
            spec=dataset_spec.thetadata,
            download_missing=dataset_spec.download_missing,
        )
        if not snapshots:
            continue

        for label_spec in specs:
            panel = build_option_label_panel([trade], snapshots, spec=label_spec)
            if panel.is_empty():
                continue
            target_column = _target_col_for_spec(label_spec)
            tagged = panel.with_columns(
                pl.lit(label_spec.label_method).alias("label_method"),
                pl.lit(_task_name_for_spec(label_spec)).alias("task_name"),
                pl.lit(target_column).alias("target_col"),
                pl.col(target_column).alias("target_value"),
            )
            combined_frames.append(tagged)

    if not combined_frames:
        return OptionMlDatasetResult()

    dataset = pl.concat(combined_frames, how="diagonal_relaxed")
    identity = [col for col in ("trade_id", "contract_symbol", "label_method") if col in dataset.columns]
    dataset = dataset.unique(identity, keep="first") if identity else dataset
    rows = dataset.to_dicts()
    stats = _build_dataset_statistics(dataset)
    return OptionMlDatasetResult(rows=rows, statistics=stats)


def save_option_ml_dataset(
    result: OptionMlDatasetResult,
    output_path: str | Path,
    *,
    file_format: str = "parquet",
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pl.DataFrame(result.rows)
    if file_format == "parquet":
        frame.write_parquet(path)
    elif file_format == "csv":
        frame.write_csv(path)
    else:
        raise ValueError("file_format must be 'parquet' or 'csv'")
    return path


def _normalize_trade_rows(trades: Sequence[Mapping[str, Any]] | pl.DataFrame) -> list[dict[str, Any]]:
    if isinstance(trades, pl.DataFrame):
        rows = trades.to_dicts()
    else:
        rows = [dict(row) for row in trades]
    out: list[dict[str, Any]] = []
    for row in rows:
        try:
            entry = datetime.fromisoformat(str(row.get("entry_date"))[:10])
            exit_ = datetime.fromisoformat(str(row.get("exit_date"))[:10])
        except (TypeError, ValueError):
            continue
        normalized = dict(row)
        normalized["entry_date"] = entry
        normalized["exit_date"] = exit_
        out.append(normalized)
    return out


def _task_name_for_spec(spec: OptionLabelSpec) -> str:
    if spec.label_method == "rank":
        return "option_rank"
    if spec.label_method == "hybrid":
        return "option_mv_hybrid"
    return "option_mv"


def _target_col_for_spec(spec: OptionLabelSpec) -> str:
    if spec.label_method == "rank":
        return "rank_y"
    return "label"


def _build_dataset_statistics(dataset: pl.DataFrame) -> dict[str, Any]:
    stats: dict[str, Any] = {
        "rows": int(len(dataset)),
        "trades": int(dataset["trade_id"].n_unique()) if "trade_id" in dataset.columns else 0,
        "tasks": [],
    }
    if "task_name" not in dataset.columns:
        return stats

    grouped = dataset.group_by("task_name").agg(
        pl.len().alias("rows"), pl.col("trade_id").n_unique().alias("trades"), pl.col("target_value").mean().alias("avg_target")
    )
    stats["tasks"] = grouped.to_dicts()
    return stats
