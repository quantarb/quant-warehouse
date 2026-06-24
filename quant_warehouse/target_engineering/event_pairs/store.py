from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Sequence

import pandas as pd

from quant_warehouse.catalog.store import CatalogStore
from quant_warehouse.config import WarehouseConfig
from quant_warehouse.ingest.normalize import symbol_provider_key
from quant_warehouse.target_engineering.event_pairs.event_pair_schema import EVENT_PAIR_COLUMNS
from quant_warehouse.target_engineering.event_pairs.event_pair_taxonomy import EVENT_PAIR_TAXONOMY
from quant_warehouse.target_engineering.event_pairs.fmp_fetch import (
    _combine_names,
    _congress_event_type,
    _dividend_event_type,
    _earnings_event_type,
    _ensure_symbol,
    _filter_dates,
    _first_numeric_column,
    _first_present_column,
    _insider_event_type,
    _institutional_event_type,
    _normalize_family_frame,
    _price_target_event_type,
    _raw_records,
    _split_event_type,
    _split_ratio,
    fetch_fmp_event_pair_family,
)
from quant_warehouse.warehouse.backend import ArcticBackend, StorageBackend, open_backend
from quant_warehouse.warehouse.equity_calendar import EquityCalendarStore
from quant_warehouse.warehouse.fundamentals import FundamentalsStore

EVENT_PAIR_LIBRARY = "target_event_pairs"
EVENT_PAIR_SECTION = "event_pairs"

_SUPPORTED_FAMILIES = tuple(EVENT_PAIR_TAXONOMY)
_HISTORICAL_SECTIONS: dict[str, tuple[str, ...]] = {
    "insider": ("ownership_insider_trading",),
    "congress": ("ownership_government_trades",),
    "price_target": ("estimates_price_target",),
    "institutional": ("ownership_institutional",),
    "dividend": ("dividends", "equity_calendar_dividend"),
    "split": ("historical_splits", "equity_calendar_splits"),
    "earnings": ("earnings", "equity_calendar_earnings"),
}
_FUNDAMENTAL_SECTIONS = {
    section
    for sections in _HISTORICAL_SECTIONS.values()
    for section in sections
    if not section.startswith("equity_calendar_")
}


@dataclass(frozen=True)
class EventPairLoadResult:
    symbol: str
    provider: str
    frame: pd.DataFrame
    source: str
    refreshed_families: tuple[str, ...] = ()


class EventPairStore:
    """Persist and load normalized FMP event-pair history for target labels."""

    def __init__(
        self,
        config: WarehouseConfig | None = None,
        *,
        backend: StorageBackend | None = None,
        catalog: CatalogStore | None = None,
        fundamentals: FundamentalsStore | None = None,
        equity_calendar: EquityCalendarStore | None = None,
    ) -> None:
        self.config = config or WarehouseConfig.from_env()
        self.config.ensure_dirs()
        self.backend: ArcticBackend = backend or open_backend(self.config)
        self.catalog = catalog or CatalogStore(self.config.catalog_path)
        self.fundamentals = fundamentals or FundamentalsStore(
            self.config,
            backend=self.backend,
            catalog=self.catalog,
        )
        self.equity_calendar = equity_calendar or EquityCalendarStore(
            self.config,
            backend=self.backend,
            catalog=self.catalog,
        )

    def read(
        self,
        symbol: str,
        *,
        provider: str = "fmp",
        event_families: Sequence[str] = _SUPPORTED_FAMILIES,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        storage_symbol = symbol_provider_key(symbol, provider)
        frame = self.backend.read(EVENT_PAIR_LIBRARY, storage_symbol)
        if frame is None or frame.empty:
            return pd.DataFrame(columns=EVENT_PAIR_COLUMNS)
        out = _restore_event_pair_frame(frame)
        return _slice_event_pairs(out, event_families=event_families, start_date=start_date, end_date=end_date)

    def load_or_refresh(
        self,
        symbol: str,
        *,
        provider: str = "fmp",
        event_families: Sequence[str] = _SUPPORTED_FAMILIES,
        start_date: str | None = None,
        end_date: str | None = None,
        refresh_missing: bool = True,
        refresh_source_history: bool = True,
        use_historical: bool = True,
    ) -> EventPairLoadResult:
        symbol = str(symbol or "").strip().upper()
        provider = str(provider or "fmp").strip().lower()
        families = _normalize_families(event_families)

        cached = self.read(
            symbol,
            provider=provider,
            event_families=families,
            start_date=start_date,
            end_date=end_date,
        )
        missing = tuple(family for family in families if family not in set(cached.get("event_family", [])))
        if not missing:
            return EventPairLoadResult(symbol, provider, cached, "cache")

        frames = [cached] if not cached.empty else []
        historical = pd.DataFrame(columns=EVENT_PAIR_COLUMNS)
        if use_historical:
            historical = build_event_pairs_from_historical_data(
                symbol,
                fundamentals=self.fundamentals,
                equity_calendar=self.equity_calendar,
                event_families=missing,
                start_date=start_date,
                end_date=end_date,
            )
            if not historical.empty:
                frames.append(historical)
                self.ingest(symbol, historical, provider=provider, merge=True)

        present = set(pd.concat(frames, ignore_index=True)["event_family"]) if frames else set()
        still_missing = tuple(family for family in missing if family not in present)
        if use_historical and refresh_source_history and refresh_missing and still_missing:
            self.refresh_source_sections(
                symbol,
                provider=provider,
                event_families=still_missing,
                start_date=start_date,
                end_date=end_date,
            )
            refreshed_historical = build_event_pairs_from_historical_data(
                symbol,
                fundamentals=self.fundamentals,
                equity_calendar=self.equity_calendar,
                event_families=still_missing,
                start_date=start_date,
                end_date=end_date,
                provider=provider,
            )
            if not refreshed_historical.empty:
                frames.append(refreshed_historical)
                self.ingest(symbol, refreshed_historical, provider=provider, merge=True)
                present.update(refreshed_historical["event_family"].unique())
                still_missing = tuple(family for family in still_missing if family not in present)

        if refresh_missing and still_missing:
            fetched_frames = []
            for family in still_missing:
                fetched = fetch_fmp_event_pair_family(
                    symbol,
                    event_family=family,
                    start_date=start_date,
                    end_date=end_date,
                )
                if not fetched.empty:
                    fetched_frames.append(fetched)
            if fetched_frames:
                fetched = pd.concat(fetched_frames, ignore_index=True)
                frames.append(fetched)
                self.ingest(symbol, fetched, provider=provider, merge=True)
                present.update(fetched["event_family"].unique())

        combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=EVENT_PAIR_COLUMNS)
        combined = _dedupe_event_pairs(combined)
        source = "cache"
        if not historical.empty:
            source = "historical"
        if any(family in present for family in still_missing):
            source = "fmp"
        return EventPairLoadResult(
            symbol,
            provider,
            _slice_event_pairs(combined, event_families=families, start_date=start_date, end_date=end_date),
            source,
            refreshed_families=tuple(family for family in still_missing if family in present),
        )

    def refresh_source_sections(
        self,
        symbol: str,
        *,
        provider: str = "fmp",
        event_families: Sequence[str] = _SUPPORTED_FAMILIES,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> dict[str, object]:
        """Refresh OpenBB/FMP source sections that can be reused for event-pair labels."""

        refreshed: dict[str, object] = {}
        for family in _normalize_families(event_families):
            for section in _HISTORICAL_SECTIONS.get(family, ()):
                if section.startswith("equity_calendar_"):
                    try:
                        refreshed[section] = self.equity_calendar.refresh_section(
                            section,
                            provider=provider,
                            start_date=start_date,
                            end_date=end_date,
                        )
                    except Exception as exc:
                        refreshed[section] = {"section": section, "error": str(exc)}
                elif section in _FUNDAMENTAL_SECTIONS:
                    try:
                        refreshed[section] = self.fundamentals.refresh_section(
                            symbol,
                            section,
                            provider=provider,
                            start_date=start_date,
                            end_date=end_date,
                        )
                    except Exception as exc:
                        refreshed[section] = {"section": section, "error": str(exc)}
        return refreshed

    def ingest(
        self,
        symbol: str,
        frame: pd.DataFrame,
        *,
        provider: str = "fmp",
        merge: bool = True,
    ) -> dict[str, object]:
        symbol = str(symbol or "").strip().upper()
        provider = str(provider or "fmp").strip().lower()
        normalized = _prepare_event_pairs_for_storage(frame)
        storage_symbol = symbol_provider_key(symbol, provider)
        existing = self.backend.read(EVENT_PAIR_LIBRARY, storage_symbol) if merge else None
        merged = _merge_event_pair_storage(existing, normalized)
        if not merged.empty:
            self.backend.write(EVENT_PAIR_LIBRARY, storage_symbol, merged)

        min_date = merged.index.min().strftime("%Y-%m-%d") if not merged.empty else None
        max_date = merged.index.max().strftime("%Y-%m-%d") if not merged.empty else None
        self.catalog.upsert(
            symbol=symbol,
            section=EVENT_PAIR_SECTION,
            provider=provider,
            min_date=min_date,
            max_date=max_date,
            row_count=int(len(merged)),
            columns_present=[str(column) for column in merged.columns],
        )
        return {
            "section": EVENT_PAIR_SECTION,
            "provider": provider,
            "rows": int(len(merged)),
            "min_date": min_date,
            "max_date": max_date,
            "library": EVENT_PAIR_LIBRARY,
            "storage_symbol": storage_symbol,
        }


def build_event_pairs_from_historical_data(
    symbol: str,
    *,
    fundamentals: FundamentalsStore,
    equity_calendar: EquityCalendarStore | None = None,
    event_families: Sequence[str] = _SUPPORTED_FAMILIES,
    start_date: str | None = None,
    end_date: str | None = None,
    provider: str = "fmp",
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for family in _normalize_families(event_families):
        built = _build_family_from_historical(
            symbol,
            family=family,
            fundamentals=fundamentals,
            equity_calendar=equity_calendar,
            start_date=start_date,
            end_date=end_date,
            provider=provider,
        )
        if not built.empty:
            frames.append(built)
    if not frames:
        return pd.DataFrame(columns=EVENT_PAIR_COLUMNS)
    return _dedupe_event_pairs(pd.concat(frames, ignore_index=True))


def _build_family_from_historical(
    symbol: str,
    *,
    family: str,
    fundamentals: FundamentalsStore,
    equity_calendar: EquityCalendarStore | None,
    start_date: str | None,
    end_date: str | None,
    provider: str,
) -> pd.DataFrame:
    if family == "insider":
        frame = fundamentals.read(symbol, section="ownership_insider_trading", provider=provider, start=start_date, end=end_date)
        return _build_insider(symbol, frame, start_date=start_date, end_date=end_date)
    if family == "congress":
        frame = fundamentals.read(symbol, section="ownership_government_trades", provider=provider, start=start_date, end=end_date)
        return _build_congress(symbol, frame, start_date=start_date, end_date=end_date)
    if family == "price_target":
        frame = fundamentals.read(symbol, section="estimates_price_target", provider=provider, start=start_date, end=end_date)
        return _build_price_target(symbol, frame, start_date=start_date, end_date=end_date)
    if family == "institutional":
        frame = fundamentals.read(symbol, section="ownership_institutional", provider=provider, start=start_date, end=end_date)
        return _build_institutional(symbol, frame, start_date=start_date, end_date=end_date)
    if family == "dividend":
        frame = fundamentals.read(symbol, section="dividends", provider=provider, start=start_date, end=end_date)
        if frame.empty and equity_calendar is not None:
            frame = _filter_calendar_symbol(equity_calendar.read("equity_calendar_dividend", provider=provider, start=start_date, end=end_date), symbol)
        return _build_dividend(symbol, frame, start_date=start_date, end_date=end_date)
    if family == "split":
        frame = fundamentals.read(symbol, section="historical_splits", provider=provider, start=start_date, end=end_date)
        if frame.empty and equity_calendar is not None:
            frame = _filter_calendar_symbol(equity_calendar.read("equity_calendar_splits", provider=provider, start=start_date, end=end_date), symbol)
        return _build_split(symbol, frame, start_date=start_date, end_date=end_date)
    if family == "earnings" and equity_calendar is not None:
        frame = _filter_calendar_symbol(equity_calendar.read("equity_calendar_earnings", provider=provider, start=start_date, end=end_date), symbol)
        return _build_earnings(symbol, frame, start_date=start_date, end_date=end_date)
    return pd.DataFrame(columns=EVENT_PAIR_COLUMNS)


def _build_insider(symbol: str, frame: pd.DataFrame, *, start_date: str | None, end_date: str | None) -> pd.DataFrame:
    frame = _prepare_historical_frame(symbol, frame)
    if frame.empty:
        return pd.DataFrame(columns=EVENT_PAIR_COLUMNS)
    frame = _filter_dates(frame, start_date=start_date, end_date=end_date)
    frame["event_type"] = frame.apply(_insider_event_type, axis=1)
    frame = frame.dropna(subset=["event_type"]).copy()
    if frame.empty:
        return pd.DataFrame(columns=EVENT_PAIR_COLUMNS)
    frame["event_date"] = _first_present_column(frame, ("transactionDate", "transaction_date", "filingDate", "filing_date", "date"))
    frame["actor_type"] = _first_present_column(frame, ("typeOfOwner", "type_of_owner", "relationship", "officerTitle", "officer_title"))
    frame["actor_name"] = _first_present_column(frame, ("reportingName", "reporting_name", "ownerName", "owner_name", "name"))
    frame["strength"] = _first_present_column(frame, ("securitiesTransacted", "securities_transacted", "shares", "transactionShares", "transaction_shares"))
    frame["raw_json"] = _raw_records(frame)
    return _normalize_family_frame(frame, event_family="insider", source="warehouse:ownership_insider_trading")


def _build_congress(symbol: str, frame: pd.DataFrame, *, start_date: str | None, end_date: str | None) -> pd.DataFrame:
    frame = _prepare_historical_frame(symbol, frame)
    if frame.empty:
        return pd.DataFrame(columns=EVENT_PAIR_COLUMNS)
    frame = _filter_dates(frame, start_date=start_date, end_date=end_date)
    frame["event_type"] = frame.apply(_congress_event_type, axis=1)
    frame = frame.dropna(subset=["event_type"]).copy()
    if frame.empty:
        return pd.DataFrame(columns=EVENT_PAIR_COLUMNS)
    frame["event_date"] = _first_present_column(frame, ("transactionDate", "transaction_date", "disclosureDate", "disclosure_date", "date"))
    frame["actor_type"] = _first_present_column(frame, ("chamber", "office", "representative", "senator")).fillna("congress")
    actor_name = _first_present_column(frame, ("representative", "senator", "firstName", "first_name", "name"))
    frame["actor_name"] = _combine_names(actor_name, _first_present_column(frame, ("lastName", "last_name")))
    frame["strength"] = _first_present_column(frame, ("amount", "amountRange", "amount_range", "assetDescription", "asset_description"))
    frame["raw_json"] = _raw_records(frame)
    return _normalize_family_frame(frame, event_family="congress", source="warehouse:ownership_government_trades")


def _build_price_target(symbol: str, frame: pd.DataFrame, *, start_date: str | None, end_date: str | None) -> pd.DataFrame:
    frame = _prepare_historical_frame(symbol, frame)
    if frame.empty:
        return pd.DataFrame(columns=EVENT_PAIR_COLUMNS)
    frame = _filter_dates(frame, start_date=start_date, end_date=end_date)
    frame["event_date"] = _first_present_column(frame, ("publishedDate", "published_date", "date"))
    frame["target_value"] = _first_numeric_column(frame, ("priceTarget", "price_target", "adjPriceTarget", "adj_price_target", "target"))
    frame = frame.dropna(subset=["event_date", "target_value"]).copy()
    group_key = _first_present_column(frame, ("analystCompany", "analyst_company", "analystName", "analyst_name", "publisher")).fillna("all")
    frame["_group_key"] = group_key
    frame = frame.sort_values(["_group_key", "event_date"])
    frame["previous_target_value"] = frame.groupby("_group_key")["target_value"].shift(1)
    frame["event_type"] = frame.apply(_price_target_event_type, axis=1)
    frame = frame.dropna(subset=["event_type"]).copy()
    if frame.empty:
        return pd.DataFrame(columns=EVENT_PAIR_COLUMNS)
    frame["actor_type"] = "analyst"
    frame["actor_name"] = _first_present_column(frame, ("analystName", "analyst_name", "analystCompany", "analyst_company", "publisher"))
    frame["strength"] = frame["target_value"]
    frame["raw_json"] = _raw_records(frame)
    return _normalize_family_frame(frame, event_family="price_target", source="warehouse:estimates_price_target")


def _build_institutional(symbol: str, frame: pd.DataFrame, *, start_date: str | None, end_date: str | None) -> pd.DataFrame:
    frame = _prepare_historical_frame(symbol, frame)
    if frame.empty:
        return pd.DataFrame(columns=EVENT_PAIR_COLUMNS)
    frame = _filter_dates(frame, start_date=start_date, end_date=end_date)
    frame["event_date"] = _first_present_column(frame, ("event_date", "date", "period_ending", "filing_date"))
    frame["delta"] = _first_numeric_column(frame, ("changeInShares", "change_in_shares", "sharesChange", "shares_change", "sharesHeldChange", "shares_held_change"))
    if frame["delta"].isna().all():
        shares = _first_numeric_column(frame, ("sharesHeld", "shares_held", "shares", "totalShares", "total_shares"))
        frame = frame.sort_values("event_date")
        frame["delta"] = shares.diff()
    frame["event_type"] = frame["delta"].map(_institutional_event_type)
    frame = frame.dropna(subset=["event_type"]).copy()
    if frame.empty:
        return pd.DataFrame(columns=EVENT_PAIR_COLUMNS)
    frame["actor_type"] = "institution"
    frame["actor_name"] = _first_present_column(frame, ("holder", "holder_name", "investor", "name")).fillna("aggregate")
    frame["strength"] = frame["delta"]
    frame["raw_json"] = _raw_records(frame)
    return _normalize_family_frame(frame, event_family="institutional", source="warehouse:ownership_institutional")


def _build_dividend(symbol: str, frame: pd.DataFrame, *, start_date: str | None, end_date: str | None) -> pd.DataFrame:
    frame = _prepare_historical_frame(symbol, frame)
    if frame.empty:
        return pd.DataFrame(columns=EVENT_PAIR_COLUMNS)
    frame = _filter_dates(frame, start_date=start_date, end_date=end_date)
    frame["event_date"] = _first_present_column(frame, ("exDividendDate", "ex_dividend_date", "declarationDate", "declaration_date", "date", "paymentDate", "payment_date"))
    frame["dividend_value"] = _first_numeric_column(frame, ("adjDividend", "adj_dividend", "dividend", "amount", "cashAmount", "cash_amount"))
    frame = frame.dropna(subset=["event_date", "dividend_value"]).copy()
    frame = frame.sort_values("event_date")
    frame["previous_dividend_value"] = frame["dividend_value"].shift(1)
    frame["event_type"] = frame.apply(_dividend_event_type, axis=1)
    frame = frame.dropna(subset=["event_type"]).copy()
    if frame.empty:
        return pd.DataFrame(columns=EVENT_PAIR_COLUMNS)
    frame["actor_type"] = "issuer"
    frame["actor_name"] = symbol
    frame["strength"] = frame["dividend_value"]
    frame["raw_json"] = _raw_records(frame)
    return _normalize_family_frame(frame, event_family="dividend", source="warehouse:dividends")


def _build_split(symbol: str, frame: pd.DataFrame, *, start_date: str | None, end_date: str | None) -> pd.DataFrame:
    frame = _prepare_historical_frame(symbol, frame)
    if frame.empty:
        return pd.DataFrame(columns=EVENT_PAIR_COLUMNS)
    frame = _filter_dates(frame, start_date=start_date, end_date=end_date)
    frame["event_date"] = _first_present_column(frame, ("date", "splitDate", "split_date"))
    frame["split_ratio"] = frame.apply(_split_ratio, axis=1)
    frame["event_type"] = frame["split_ratio"].map(_split_event_type)
    frame = frame.dropna(subset=["event_type"]).copy()
    if frame.empty:
        return pd.DataFrame(columns=EVENT_PAIR_COLUMNS)
    frame["actor_type"] = "issuer"
    frame["actor_name"] = symbol
    frame["strength"] = frame["split_ratio"]
    frame["raw_json"] = _raw_records(frame)
    return _normalize_family_frame(frame, event_family="split", source="warehouse:historical_splits")


def _build_earnings(symbol: str, frame: pd.DataFrame, *, start_date: str | None, end_date: str | None) -> pd.DataFrame:
    frame = _prepare_historical_frame(symbol, frame)
    if frame.empty:
        return pd.DataFrame(columns=EVENT_PAIR_COLUMNS)
    frame = _filter_dates(frame, start_date=start_date, end_date=end_date)
    frame["event_date"] = _first_present_column(frame, ("date", "reportDate", "report_date", "reportedDate", "reported_date", "fiscalDateEnding", "fiscal_date_ending"))
    actual = _first_numeric_column(frame, ("epsActual", "eps_actual", "actualEps", "actual_eps", "eps", "reportedEPS", "reported_eps"))
    estimated = _first_numeric_column(frame, ("epsEstimated", "eps_estimated", "estimatedEps", "estimated_eps", "epsEstimate", "eps_estimate"))
    frame["surprise"] = actual - estimated
    frame["event_type"] = frame["surprise"].map(_earnings_event_type)
    frame = frame.dropna(subset=["event_type"]).copy()
    if frame.empty:
        return pd.DataFrame(columns=EVENT_PAIR_COLUMNS)
    frame["actor_type"] = "issuer"
    frame["actor_name"] = symbol
    frame["strength"] = frame["surprise"]
    frame["raw_json"] = _raw_records(frame)
    return _normalize_family_frame(frame, event_family="earnings", source="warehouse:equity_calendar_earnings")


def _prepare_historical_frame(symbol: str, frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    out = frame.copy()
    if isinstance(out.index, pd.DatetimeIndex):
        index_name = out.index.name or "date"
        out = out.reset_index().rename(columns={index_name: index_name})
    return _ensure_symbol(out, symbol)


def _filter_calendar_symbol(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if frame is None or frame.empty or "symbol" not in frame.columns:
        return pd.DataFrame()
    return frame.loc[frame["symbol"].astype(str).str.upper() == symbol.strip().upper()].copy()


def _normalize_families(event_families: Sequence[str]) -> tuple[str, ...]:
    families = tuple(str(family).strip().lower() for family in event_families if str(family).strip())
    unknown = [family for family in families if family not in EVENT_PAIR_TAXONOMY]
    if unknown:
        raise ValueError(f"Unsupported event pair families: {unknown}")
    return families


def _slice_event_pairs(
    frame: pd.DataFrame,
    *,
    event_families: Sequence[str],
    start_date: str | None,
    end_date: str | None,
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=EVENT_PAIR_COLUMNS)
    out = frame.copy()
    out["event_date"] = pd.to_datetime(out["event_date"], errors="coerce").dt.normalize()
    out = out.dropna(subset=["event_date"])
    families = set(_normalize_families(event_families))
    out = out.loc[out["event_family"].isin(families)]
    if start_date is not None:
        out = out.loc[out["event_date"] >= pd.Timestamp(start_date)]
    if end_date is not None:
        out = out.loc[out["event_date"] <= pd.Timestamp(end_date)]
    return _dedupe_event_pairs(out)


def _dedupe_event_pairs(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=EVENT_PAIR_COLUMNS)
    out = frame.copy()
    out["event_date"] = pd.to_datetime(out["event_date"], errors="coerce").dt.normalize()
    out = out.dropna(subset=["event_date"])
    out = out.drop_duplicates(
        subset=["symbol", "event_date", "event_family", "event_type", "actor_type", "actor_name", "strength"],
        keep="last",
    )
    return out.sort_values(["symbol", "event_date", "event_family", "event_type"], ignore_index=True)[EVENT_PAIR_COLUMNS]


def _prepare_event_pairs_for_storage(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = _dedupe_event_pairs(frame)
    if normalized.empty:
        return pd.DataFrame(columns=[column for column in EVENT_PAIR_COLUMNS if column != "event_date"])
    out = normalized.copy()
    out["raw_json"] = out["raw_json"].map(_json_text)
    out = out.set_index("event_date")
    out.index.name = "event_date"
    return out


def _restore_event_pair_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if isinstance(out.index, pd.DatetimeIndex):
        out = out.reset_index()
    if "raw_json" in out.columns:
        out["raw_json"] = out["raw_json"].map(_json_load)
    return _dedupe_event_pairs(out)


def _merge_event_pair_storage(existing: pd.DataFrame | None, incoming: pd.DataFrame) -> pd.DataFrame:
    if incoming is None or incoming.empty:
        return existing.copy() if existing is not None and not existing.empty else pd.DataFrame()
    if existing is None or existing.empty:
        return incoming.sort_index()
    combined = pd.concat([existing, incoming]).reset_index()
    combined = combined.drop_duplicates(
        subset=["event_date", "symbol", "event_family", "event_type", "actor_type", "actor_name", "strength"],
        keep="last",
    )
    combined["event_date"] = pd.to_datetime(combined["event_date"], errors="coerce")
    combined = combined.dropna(subset=["event_date"]).set_index("event_date")
    combined.index = pd.DatetimeIndex(combined.index)
    combined.index.name = "event_date"
    return combined.sort_index()


def _json_text(value: object) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, str):
        return value
    return json.dumps(value, default=str, sort_keys=True)


def _json_load(value: object) -> object:
    if not isinstance(value, str) or not value:
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value
