from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
import math
import re
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from quant_warehouse.platforms.data_providers.fmp.target_engineering.event_pairs import EVENT_PAIR_TAXONOMY
from quant_warehouse.research_tools.target_family_eval import BinaryTargetConfig
from quant_warehouse.warehouse.api import Warehouse

FMP_EVENT_CONTEXT_FEATURE_FAMILIES: dict[str, tuple[str, tuple[str, ...]]] = {
    "congress": (
        "fmp_congress_event_context",
        (
            "actor_name",
            "actor_type",
            "actor_chamber",
            "transaction_value",
            "reported_date",
            "disclosure_lag_days",
        ),
    ),
    "insider": (
        "fmp_insider_event_context",
        (
            "actor_name",
            "actor_type",
            "actor_role",
            "actor_title",
            "transaction_shares",
            "transaction_price",
            "transaction_value",
            "reported_date",
            "disclosure_lag_days",
        ),
    ),
    "analyst_rating": (
        "fmp_analyst_rating_event_context",
        (
            "actor_name",
            "actor_firm",
            "actor_role",
            "reported_date",
            "disclosure_lag_days",
        ),
    ),
    "price_target": (
        "fmp_price_target_event_context",
        (
            "actor_name",
            "actor_firm",
            "actor_role",
            "reported_date",
            "disclosure_lag_days",
        ),
    ),
}

FMP_EQUITY_PROFILE_FEATURE_SOURCE = "fmp"
FMP_EQUITY_PROFILE_FEATURE_FAMILY = "fmp_equity_profile"
FMP_EQUITY_PROFILE_FEATURE_COLUMNS = (
    "date",
    "symbol",
    "company_name",
    "exchange",
    "country",
    "sector",
    "industry",
)


@dataclass(frozen=True)
class EventFeatureDatasetConfig:
    min_feature_coverage: float = 0.50
    max_rows_per_task_split: int | None = None


EXCLUDED_TEXT_FEATURES = frozenset(
    {
        "date",
        "year",
        "year_label",
        "actor_name",
        "actor_type",
        "actor_role",
        "actor_chamber",
        "actor_firm",
        "actor_title",
        "analyst_actor",
        "congress_chamber",
        "insider_role",
    }
)


@dataclass(frozen=True)
class EventFeatureDatasetResult:
    rows: pd.DataFrame
    task_inventory: pd.DataFrame
    diagnostics: dict[str, object]


def sanitize_task_name(value: str) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    if not text:
        raise ValueError(f"Invalid task name from {value!r}")
    return f"task_{text}"


def event_pair_task_specs(
    target_config: BinaryTargetConfig,
    available_targets: Iterable[str],
) -> list[dict[str, object]]:
    available = set(str(target) for target in available_targets)
    specs: list[dict[str, object]] = []
    for family in target_config.event_families:
        pair = EVENT_PAIR_TAXONOMY[str(family)]
        positive_col = f"target_event_on__{pair['positive']}"
        negative_col = f"target_event_on__{pair['negative']}"
        if positive_col not in available or negative_col not in available:
            continue
        specs.append(
            {
                "target_task": f"event_pair__{family}",
                "task_id": sanitize_task_name(f"event_pair__{family}"),
                "positive_col": positive_col,
                "negative_col": negative_col,
                "positive_label": pair["positive"],
                "negative_label": pair["negative"],
            }
        )
    return specs


def oracle_side_task_specs(available_targets: Iterable[str]) -> list[dict[str, object]]:
    available = set(str(target) for target in available_targets)
    positive_cols = sorted(target for target in available if re.match(r"^target_oracle_trade_entry__.+_long$", target))
    negative_cols = sorted(
        re.sub(r"_long$", "_short", column)
        for column in positive_cols
        if re.sub(r"_long$", "_short", column) in available
    )
    positive_cols = [re.sub(r"_short$", "_long", column) for column in negative_cols]
    if not positive_cols or not negative_cols:
        return []
    return [
        {
            "target_task": "target_oracle_trade_entry__buy_sell",
            "task_id": sanitize_task_name("target_oracle_trade_entry__buy_sell"),
            "positive_cols": positive_cols,
            "negative_cols": negative_cols,
            "positive_label": "buy",
            "negative_label": "sell",
        }
    ]


def build_event_feature_text_dataset(
    feature_target_panel: pd.DataFrame,
    feature_metadata: pd.DataFrame,
    task_specs: Sequence[dict[str, object]],
    *,
    config: EventFeatureDatasetConfig | None = None,
    allowed_feature_families: set[tuple[str, str]] | None = None,
    allowed_feature_families_by_task: dict[str, set[tuple[str, str]]] | None = None,
) -> EventFeatureDatasetResult:
    """Build long-form text rows from actual event/oracle rows and covered feature families.

    The invariant is structural: task side columns select actual event/oracle rows first,
    then each feature family is inner-joined by row index through coverage filtering.
    No feature-family text is materialized for no-event rows.
    """

    cfg = config or EventFeatureDatasetConfig()
    rows: list[pd.DataFrame] = []
    diagnostics = {
        "candidate_task_specs": len(task_specs),
        "feature_families": 0,
        "event_feature_rows": 0,
    }
    if feature_target_panel.empty or feature_metadata.empty or not task_specs:
        empty = pd.DataFrame()
        return EventFeatureDatasetResult(empty, _task_inventory(empty), diagnostics)

    panel = _normalize_panel(feature_target_panel)
    for (source, family), family_meta in feature_metadata.groupby(["source", "family"], sort=True):
        source_key = str(source)
        family_key = str(family)
        if allowed_feature_families is not None and (source_key, family_key) not in allowed_feature_families:
            continue
        features = [feature for feature in family_meta["feature"].drop_duplicates().tolist() if feature in panel.columns]
        if not features:
            continue
        coverage_index = panel.index[feature_coverage_mask(panel, features, cfg.min_feature_coverage)]
        if coverage_index.empty:
            continue
        diagnostics["feature_families"] = int(diagnostics["feature_families"]) + 1
        base_columns = list(dict.fromkeys(["symbol", "date", *features]))
        for spec in task_specs:
            if not _task_allows_feature_family(
                str(spec["task_id"]),
                str(spec["target_task"]),
                source_key,
                family_key,
                allowed_feature_families_by_task,
            ):
                continue
            selected = _select_task_index(panel, coverage_index, spec)
            if selected.empty:
                continue
            base = panel.loc[selected, base_columns].copy()
            if base.empty:
                continue
            text_values = base.apply(
                lambda row: feature_family_text(row, features, source=source_key, family=family_key),
                axis=1,
            )
            positive_cols = spec.get("positive_cols", spec.get("positive_col"))
            negative_cols = spec.get("negative_cols", spec.get("negative_col"))
            positive = _side_values(panel, selected, positive_cols)
            task_frame = pd.DataFrame(
                {
                    "symbol": base["symbol"].astype(str).str.upper().to_numpy(),
                    "date": pd.to_datetime(base["date"], errors="coerce").dt.normalize().to_numpy(),
                    "source": source_key,
                    "feature_family": family_key,
                    "text": text_values.to_numpy(),
                    "target_task": str(spec["target_task"]),
                    "task_id": str(spec["task_id"]),
                    "label_type": str(spec["task_id"]),
                    "label": np.where(positive.gt(0), spec["positive_label"], spec["negative_label"]),
                    "positive_target_col": _lineage_value(positive_cols),
                    "negative_target_col": _lineage_value(negative_cols),
                },
                index=selected,
            )
            rows.append(task_frame)

    if not rows:
        empty = pd.DataFrame()
        return EventFeatureDatasetResult(empty, _task_inventory(empty), diagnostics)
    out = pd.concat(rows, ignore_index=True).dropna(subset=["date", "text", "label"])
    out = out.loc[out["text"].astype(str).str.len().gt(0)].copy()
    out = out.sort_values(["date", "symbol", "feature_family", "target_task"]).reset_index(drop=True)
    diagnostics["event_feature_rows"] = len(out)
    return EventFeatureDatasetResult(out, _task_inventory(out), diagnostics)


def add_fmp_event_context_feature_families(
    feature_target_panel: pd.DataFrame,
    feature_metadata: pd.DataFrame,
    events: pd.DataFrame,
    *,
    families: dict[str, tuple[str, tuple[str, ...]]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attach FMP event metadata as sparse feature families on a symbol/date panel.

    The returned panel remains keyed by ``(symbol, date)``. Each event source gets a
    separate feature family, so congress metadata is not represented as analyst
    metadata. Pair this with ``fmp_event_context_allowed_feature_families_by_task``
    when building text rows to prevent task/family mismatches.
    """

    if feature_target_panel is None or feature_target_panel.empty or events is None or events.empty:
        return feature_target_panel, feature_metadata

    panel = _normalize_panel(feature_target_panel)
    event_frame = events.copy()
    event_frame["symbol"] = event_frame["symbol"].astype(str).str.upper()
    event_frame["date"] = pd.to_datetime(event_frame["event_date"], errors="coerce").dt.tz_localize(None).dt.normalize()
    event_frame = event_frame.dropna(subset=["symbol", "date", "event_family"])
    if event_frame.empty:
        return panel, feature_metadata

    metadata_rows: list[dict[str, object]] = []
    context_frames: list[pd.DataFrame] = []
    family_map = families or FMP_EVENT_CONTEXT_FEATURE_FAMILIES
    for event_family, (feature_family, columns) in family_map.items():
        family_events = event_frame.loc[event_frame["event_family"].astype(str).eq(event_family)].copy()
        if family_events.empty:
            continue
        available_columns = [column for column in columns if column in family_events.columns]
        if not available_columns:
            continue
        renamed = {column: f"{feature_family}__{column}" for column in available_columns}
        family_values = family_events[["symbol", "date", *available_columns]].rename(columns=renamed)
        aggregated = (
            family_values.groupby(["symbol", "date"], as_index=False)
            .agg({renamed[column]: _aggregate_event_context_values for column in available_columns})
        )
        context_frames.append(aggregated)
        for column in available_columns:
            metadata_rows.append(
                {
                    "feature": renamed[column],
                    "family": feature_family,
                    "source": "fmp",
                    "source_column": column,
                    "expected_direction": _event_context_expected_direction(family_events[column]),
                }
            )

    out = panel
    for context_frame in context_frames:
        out = out.merge(context_frame, on=["symbol", "date"], how="left")
    if not metadata_rows:
        return out, feature_metadata
    context_metadata = pd.DataFrame(metadata_rows)
    metadata = (
        pd.concat([feature_metadata, context_metadata], ignore_index=True)
        .drop_duplicates(["source", "family", "feature"])
        .sort_values(["source", "family", "feature"])
        .reset_index(drop=True)
    )
    return out, metadata


def add_fmp_equity_profile_feature_family(
    feature_panel: pd.DataFrame,
    feature_metadata: pd.DataFrame,
    *,
    warehouse: Warehouse,
    provider: str = "fmp",
    columns: Sequence[str] = FMP_EQUITY_PROFILE_FEATURE_COLUMNS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attach FMP issuer profile fields as a global symbol/date feature family."""

    if feature_panel is None or feature_panel.empty:
        return feature_panel, feature_metadata
    out = feature_panel.copy()
    profile_frame = _profile_lookup(warehouse, out["symbol"], provider=provider)
    if not profile_frame.empty:
        out = out.merge(profile_frame, on="symbol", how="left", suffixes=("", "_profile"))
        for column in columns:
            fallback = f"{column}_profile"
            if fallback in out.columns:
                if column in out.columns:
                    out[column] = out[column].where(out[column].notna(), out[fallback])
                else:
                    out[column] = out[fallback]
                out = out.drop(columns=[fallback])
    metadata_rows = [
        {
            "feature": column,
            "family": FMP_EQUITY_PROFILE_FEATURE_FAMILY,
            "source": FMP_EQUITY_PROFILE_FEATURE_SOURCE,
            "source_column": column,
            "expected_direction": "categorical",
        }
        for column in columns
    ]
    profile_metadata = pd.DataFrame(metadata_rows)
    metadata = (
        pd.concat([feature_metadata, profile_metadata], ignore_index=True)
        .drop_duplicates(["source", "family", "feature"])
        .sort_values(["source", "family", "feature"])
        .reset_index(drop=True)
    )
    return out, metadata


def fmp_event_context_allowed_feature_families_by_task(
    task_specs: Sequence[dict[str, object]],
    allowed_feature_families: set[tuple[str, str]] | None = None,
    *,
    event_context_families: dict[str, tuple[str, tuple[str, ...]]] | None = None,
) -> dict[str, set[tuple[str, str]]]:
    """Return per-task feature-family allowlists for FMP event context families."""

    family_map = event_context_families or FMP_EVENT_CONTEXT_FEATURE_FAMILIES
    base_allowed = set(allowed_feature_families) if allowed_feature_families is not None else set()
    out: dict[str, set[tuple[str, str]]] = {}
    for spec in task_specs:
        target_task = str(spec["target_task"])
        task_id = str(spec["task_id"])
        task_allowed: set[tuple[str, str]] = set()
        for event_family, (feature_family, _) in family_map.items():
            if target_task == f"event_pair__{event_family}":
                family_key = ("fmp", feature_family)
                if allowed_feature_families is None or family_key in base_allowed:
                    task_allowed.add(family_key)
        if task_allowed:
            out[task_id] = task_allowed
            out[target_task] = task_allowed
    return out


def build_event_context(
    events: pd.DataFrame,
    feature_panel: pd.DataFrame,
    *,
    warehouse: Warehouse | None = None,
    provider: str = "fmp",
    event_families: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Inner join normalized event rows to feature rows on ``(symbol, date)``."""

    if events is None or events.empty or feature_panel is None or feature_panel.empty:
        return pd.DataFrame()
    event_frame = events.copy()
    event_frame["date"] = pd.to_datetime(event_frame["event_date"], errors="coerce").dt.tz_localize(None).dt.normalize()
    event_frame["symbol"] = event_frame["symbol"].astype(str).str.upper()
    if event_families is not None:
        families = set(str(family) for family in event_families)
        event_frame = event_frame.loc[event_frame["event_family"].astype(str).isin(families)]
    event_frame = event_frame.loc[event_frame["event_type"].notna() & event_frame["date"].notna()].copy()
    keep = [
        "symbol",
        "date",
        "event_family",
        "event_type",
        "actor_name",
        "actor_type",
        "actor_role",
        "actor_chamber",
        "actor_firm",
        "actor_title",
        "event_side",
        "strength",
        "transaction_shares",
        "transaction_price",
        "transaction_value",
        "reported_date",
        "disclosure_lag_days",
    ]
    event_frame = event_frame[[column for column in keep if column in event_frame.columns]].drop_duplicates()
    if event_frame.empty:
        return pd.DataFrame()
    if warehouse is not None:
        profile_frame = _profile_lookup(warehouse, event_frame["symbol"], provider=provider)
        event_frame = event_frame.merge(profile_frame, on="symbol", how="left")
    panel = _normalize_panel(feature_panel)
    base = panel.merge(event_frame, on=["symbol", "date"], how="inner", suffixes=("", "_event"))
    if base.empty:
        return base
    base["year_label"] = pd.to_datetime(base["date"], errors="coerce").dt.year.astype("Int64").astype("string")
    for column in (
        "sector",
        "industry",
        "exchange",
        "actor_name",
        "actor_type",
        "actor_role",
        "actor_chamber",
        "actor_firm",
        "actor_title",
        "event_type",
        "event_family",
    ):
        if column in base.columns:
            base[column] = base[column].astype("string").str.strip()
    return base


def feature_coverage_mask(frame: pd.DataFrame, features: list[str], min_feature_coverage: float) -> pd.Series:
    if not features:
        return pd.Series(False, index=frame.index)
    return frame[features].notna().mean(axis=1).ge(float(min_feature_coverage))


def feature_family_text(
    row: pd.Series,
    features: list[str],
    *,
    source: str,
    family: str,
) -> str:
    pairs = [f"source={source}", f"feature_family={family}"]
    for feature in features:
        if feature in EXCLUDED_TEXT_FEATURES:
            continue
        value = format_feature_value(row.get(feature))
        if value is not None:
            pairs.append(f"{compact_feature_key(feature, family)}={value}")
    return " ".join(pairs)


def format_feature_value(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        text = str(value).strip()
        return text if text else None
    if not math.isfinite(number):
        return None
    rounded = round(number, 2)
    if rounded == 0:
        rounded = 0.0
    return f"{rounded:.2f}"


def compact_feature_key(feature: str, family: str) -> str:
    key = str(feature)
    family_prefix = f"{family}__"
    if key.startswith(family_prefix):
        key = key[len(family_prefix):]
    replacements = {
        "market_cap": "mcap",
        "enterprise_value": "ev",
        "total_": "tot_",
        "current_": "cur_",
        "operating_": "op_",
        "stockholders": "sh",
        "liabilities": "liab",
        "receivables": "recv",
        "inventory": "inv",
        "depreciation": "depr",
        "amortization": "amort",
    }
    for old, short in replacements.items():
        key = key.replace(old, short)
    return key.strip("_") or str(feature)


def _select_task_index(panel: pd.DataFrame, candidate_index: pd.Index, spec: dict[str, object]) -> pd.Index:
    positive_cols = spec.get("positive_cols", spec.get("positive_col"))
    negative_cols = spec.get("negative_cols", spec.get("negative_col"))
    positive = _side_values(panel, candidate_index, positive_cols)
    negative = _side_values(panel, candidate_index, negative_cols)
    event_mask = positive.gt(0) | negative.gt(0)
    ambiguous_mask = positive.gt(0) & negative.gt(0)
    return candidate_index[event_mask.to_numpy() & ~ambiguous_mask.to_numpy()]


def _side_values(panel: pd.DataFrame, index: pd.Index, columns: Any) -> pd.Series:
    if isinstance(columns, str):
        return pd.to_numeric(panel.loc[index, columns], errors="coerce").fillna(0).astype("int8")
    values = panel.loc[index, list(columns)].apply(pd.to_numeric, errors="coerce").fillna(0)
    return values.gt(0).any(axis=1).astype("int8")


def _lineage_value(columns: Any) -> str:
    if isinstance(columns, str):
        return columns
    return "|".join(str(column) for column in columns)


def _task_allows_feature_family(
    task_id: str,
    target_task: str,
    source: str,
    family: str,
    allowed_feature_families_by_task: dict[str, set[tuple[str, str]]] | None,
) -> bool:
    if allowed_feature_families_by_task is None:
        return True
    family_key = (source, family)
    event_context_keys = {("fmp", feature_family) for feature_family, _ in FMP_EVENT_CONTEXT_FEATURE_FAMILIES.values()}
    if family_key not in event_context_keys:
        return True
    allowed = allowed_feature_families_by_task.get(task_id)
    if allowed is None:
        allowed = allowed_feature_families_by_task.get(target_task)
    if allowed is None:
        return False
    return family_key in allowed


def _aggregate_event_context_values(values: pd.Series) -> object:
    clean = values.dropna()
    if clean.empty:
        return pd.NA
    numeric = pd.to_numeric(clean, errors="coerce")
    if numeric.notna().all():
        return float(numeric.sum())
    normalized = clean.astype(str).str.strip()
    normalized = normalized.loc[normalized.ne("")]
    if normalized.empty:
        return pd.NA
    return "|".join(dict.fromkeys(normalized.tolist()))


def _event_context_expected_direction(values: pd.Series) -> str:
    numeric = pd.to_numeric(values.dropna(), errors="coerce")
    if not numeric.empty and numeric.notna().all():
        return "unknown"
    return "categorical"


def _normalize_panel(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["symbol"] = out["symbol"].astype(str).str.upper()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    return out.dropna(subset=["symbol", "date"])


def _profile_lookup(warehouse: Warehouse, symbols: pd.Series, *, provider: str) -> pd.DataFrame:
    rows = []
    for symbol in symbols.astype(str).str.upper().drop_duplicates().sort_values():
        profile = warehouse.catalog.get_profile(symbol=symbol, provider=provider)
        rows.append(
            {
                "symbol": symbol,
                "company_name": profile.company_name if profile is not None else None,
                "exchange": profile.exchange if profile is not None else None,
                "country": profile.country if profile is not None else None,
                "sector": profile.sector if profile is not None else None,
                "industry": profile.industry if profile is not None else None,
            }
        )
    return pd.DataFrame(rows)


def _task_inventory(rows: pd.DataFrame) -> pd.DataFrame:
    if rows is None or rows.empty:
        return pd.DataFrame(columns=["target_task", "task_id", "rows", "labels", "feature_families"])
    return (
        rows.groupby(["target_task", "task_id"])
        .agg(rows=("label", "size"), labels=("label", "nunique"), feature_families=("feature_family", "nunique"))
        .reset_index()
        .sort_values(["target_task", "task_id"])
        .reset_index(drop=True)
    )
