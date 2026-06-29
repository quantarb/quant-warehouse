from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from quant_warehouse.target_engineering.option_labels import (
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
    trades: Sequence[Mapping[str, Any]] | pd.DataFrame,
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

    combined_frames: list[pd.DataFrame] = []
    for trade in trade_rows:
        symbol = str(trade.get("symbol") or trade.get("underlying_symbol") or "").strip().upper()
        entry_dt = pd.Timestamp(trade["entry_date"]).normalize()
        exit_dt = pd.Timestamp(trade["exit_date"]).normalize()
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
            if panel.empty:
                continue
            tagged = panel.copy()
            tagged["label_method"] = label_spec.label_method
            tagged["task_name"] = _task_name_for_spec(label_spec)
            tagged["target_col"] = _target_col_for_spec(label_spec)
            tagged["target_value"] = tagged[_target_col_for_spec(label_spec)]
            combined_frames.append(tagged)

    if not combined_frames:
        return OptionMlDatasetResult()

    dataset = pd.concat(combined_frames, ignore_index=True, sort=False)
    dataset = dataset.drop_duplicates(
        subset=[col for col in ("trade_id", "contract_symbol", "label_method") if col in dataset.columns],
        keep="first",
    )
    rows = dataset.to_dict(orient="records")
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
    frame = pd.DataFrame(result.rows)
    if file_format == "parquet":
        frame.to_parquet(path, index=False)
    elif file_format == "csv":
        frame.to_csv(path, index=False)
    else:
        raise ValueError("file_format must be 'parquet' or 'csv'")
    return path


def _normalize_trade_rows(trades: Sequence[Mapping[str, Any]] | pd.DataFrame) -> list[dict[str, Any]]:
    if isinstance(trades, pd.DataFrame):
        rows = trades.to_dict(orient="records")
    else:
        rows = [dict(row) for row in trades]
    out: list[dict[str, Any]] = []
    for row in rows:
        entry = pd.to_datetime(row.get("entry_date"), errors="coerce")
        exit_ = pd.to_datetime(row.get("exit_date"), errors="coerce")
        if pd.isna(entry) or pd.isna(exit_):
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


def _build_dataset_statistics(dataset: pd.DataFrame) -> dict[str, Any]:
    stats: dict[str, Any] = {
        "rows": int(len(dataset)),
        "trades": int(dataset["trade_id"].nunique()) if "trade_id" in dataset.columns else 0,
        "tasks": [],
    }
    if "task_name" not in dataset.columns:
        return stats

    grouped = (
        dataset.groupby("task_name", dropna=False)
        .agg(
            rows=("contract_symbol", "count"),
            trades=("trade_id", "nunique"),
            avg_target=("target_value", "mean"),
        )
        .reset_index()
    )
    stats["tasks"] = grouped.to_dict(orient="records")
    return stats
