from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import pandas as pd


@dataclass(frozen=True)
class BuiltFeatureSet:
    """A named set of engineered features aligned on a shared index."""

    df: pd.DataFrame
    feature_cols: list[str]


@dataclass(frozen=True)
class FeatureToggleSpec:
    """Feature-family toggles for a research panel build."""

    include_price_technicals: bool = True
    include_ta_classic_technicals: bool = False
    include_time_calendar_features: bool = True
    include_fundamental_change: bool = True
    include_statement_quality: bool = True
    include_ttm_financial_statements: bool = False
    include_event_features: bool = True
    include_ownership_features: bool = True
    include_economic_indicators: bool = True
    include_treasury_rates: bool = True
    include_sector_performance: bool = False
    include_industry_performance: bool = False
    include_sector_pe: bool = False
    include_industry_pe: bool = False
    include_representation_embedding: bool = False

    @classmethod
    def from_mapping(cls, source: Mapping[str, Any] | None = None) -> "FeatureToggleSpec":
        raw = dict(source or {})
        defaults = cls()
        return cls(
            include_price_technicals=_as_bool(raw.get("include_price_technicals"), defaults.include_price_technicals),
            include_ta_classic_technicals=_as_bool(
                raw.get("include_ta_classic_technicals"),
                defaults.include_ta_classic_technicals,
            ),
            include_time_calendar_features=_as_bool(
                raw.get("include_time_calendar_features"),
                defaults.include_time_calendar_features,
            ),
            include_fundamental_change=_as_bool(raw.get("include_fundamental_change"), defaults.include_fundamental_change),
            include_statement_quality=_as_bool(raw.get("include_statement_quality"), defaults.include_statement_quality),
            include_ttm_financial_statements=_as_bool(
                raw.get("include_ttm_financial_statements"),
                defaults.include_ttm_financial_statements,
            ),
            include_event_features=_as_bool(raw.get("include_event_features"), defaults.include_event_features),
            include_ownership_features=_as_bool(raw.get("include_ownership_features"), defaults.include_ownership_features),
            include_economic_indicators=_as_bool(raw.get("include_economic_indicators"), defaults.include_economic_indicators),
            include_treasury_rates=_as_bool(raw.get("include_treasury_rates"), defaults.include_treasury_rates),
            include_sector_performance=_as_bool(
                raw.get("include_sector_performance"), defaults.include_sector_performance
            ),
            include_industry_performance=_as_bool(
                raw.get("include_industry_performance"), defaults.include_industry_performance
            ),
            include_sector_pe=_as_bool(raw.get("include_sector_pe"), defaults.include_sector_pe),
            include_industry_pe=_as_bool(raw.get("include_industry_pe"), defaults.include_industry_pe),
            include_representation_embedding=_as_bool(
                raw.get("include_representation_embedding"),
                defaults.include_representation_embedding,
            ),
        )

    def to_dict(self) -> dict[str, bool]:
        return {
            "include_price_technicals": bool(self.include_price_technicals),
            "include_ta_classic_technicals": bool(self.include_ta_classic_technicals),
            "include_time_calendar_features": bool(self.include_time_calendar_features),
            "include_fundamental_change": bool(self.include_fundamental_change),
            "include_statement_quality": bool(self.include_statement_quality),
            "include_ttm_financial_statements": bool(self.include_ttm_financial_statements),
            "include_event_features": bool(self.include_event_features),
            "include_ownership_features": bool(self.include_ownership_features),
            "include_economic_indicators": bool(self.include_economic_indicators),
            "include_treasury_rates": bool(self.include_treasury_rates),
            "include_sector_performance": bool(self.include_sector_performance),
            "include_industry_performance": bool(self.include_industry_performance),
            "include_sector_pe": bool(self.include_sector_pe),
            "include_industry_pe": bool(self.include_industry_pe),
            "include_representation_embedding": bool(self.include_representation_embedding),
        }


@dataclass(frozen=True)
class RepresentationEmbeddingSpec:
    """Semantic representation-embedding settings for feature panels."""

    enabled: bool = False
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    model_version: str = "semantic_grouped_v2"
    store_dir: str = ""
    column_prefix: str = "embedding_"
    local_files_only: bool = False
    device: str | None = None

    @classmethod
    def from_mapping(
        cls,
        source: Mapping[str, Any] | None = None,
        *,
        default_store_dir: str = "",
        default_model_version: str = "semantic_grouped_v2",
    ) -> "RepresentationEmbeddingSpec":
        raw = dict(source or {})
        column_prefix = str(raw.get("representation_embedding_column_prefix") or "embedding_").strip() or "embedding_"
        device = str(raw.get("representation_embedding_device") or "").strip() or None
        return cls(
            enabled=_as_bool(raw.get("include_representation_embedding"), False),
            model_name=str(raw.get("representation_embedding_model_name") or "sentence-transformers/all-MiniLM-L6-v2").strip()
            or "sentence-transformers/all-MiniLM-L6-v2",
            model_version=str(raw.get("representation_embedding_model_version") or default_model_version).strip()
            or default_model_version,
            store_dir=str(raw.get("representation_embedding_store_dir") or default_store_dir),
            column_prefix=column_prefix,
            local_files_only=_as_bool(raw.get("representation_embedding_local_files_only"), False),
            device=device,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "model_name": str(self.model_name),
            "model_version": str(self.model_version),
            "store_dir": str(self.store_dir),
            "column_prefix": str(self.column_prefix),
            "local_files_only": bool(self.local_files_only),
            "device": self.device,
        }


@dataclass(frozen=True)
class FeatureBuildSpec:
    """Typed feature-panel configuration used by workflows."""

    toggles: FeatureToggleSpec = field(default_factory=FeatureToggleSpec)
    representation_embedding: RepresentationEmbeddingSpec = field(default_factory=RepresentationEmbeddingSpec)
    start_date: str | None = None
    end_date: str | None = None
    raw_config: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(
        cls,
        source: Mapping[str, Any] | None = None,
        *,
        default_store_dir: str = "",
        default_model_version: str = "semantic_grouped_v2",
    ) -> "FeatureBuildSpec":
        raw = dict(source or {})
        start_date = str(raw.get("feature_start_date") or raw.get("start_date") or "").strip() or None
        end_date = str(raw.get("feature_end_date") or raw.get("end_date") or "").strip() or None
        return cls(
            toggles=FeatureToggleSpec.from_mapping(raw),
            representation_embedding=RepresentationEmbeddingSpec.from_mapping(
                raw,
                default_store_dir=default_store_dir,
                default_model_version=default_model_version,
            ),
            start_date=start_date,
            end_date=end_date,
            raw_config=raw,
        )

    def to_toggle_dict(self) -> dict[str, Any]:
        return self.toggles.to_dict()


def _as_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return bool(default)
