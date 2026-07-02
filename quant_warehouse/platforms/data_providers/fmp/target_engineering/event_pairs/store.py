from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Sequence

import pandas as pd

from quant_warehouse.catalog.store import CatalogStore
from quant_warehouse.config import WarehouseConfig
from quant_warehouse.ingest.normalize import symbol_provider_key
from quant_warehouse.platforms.data_providers.fmp.target_engineering.event_pairs.classifiers import (
    analyst_rating_event_type,
    combine_names,
    congress_event_type,
    dividend_event_type,
    earnings_event_type,
    ensure_symbol,
    filter_dates,
    first_numeric_column,
    first_present_column,
    insider_event_type,
    institutional_event_type,
    normalize_family_frame,
    price_target_event_type,
    raw_records,
    split_event_type,
    split_ratio,
)
from quant_warehouse.platforms.data_providers.fmp.target_engineering.event_pairs.event_pair_schema import EVENT_PAIR_COLUMNS
from quant_warehouse.platforms.data_providers.fmp.target_engineering.event_pairs.event_pair_taxonomy import EVENT_PAIR_TAXONOMY
from quant_warehouse.warehouse.backend import ArcticBackend, StorageBackend, open_backend
from quant_warehouse.warehouse.equity_calendar import EquityCalendarStore
from quant_warehouse.warehouse.fundamentals import FundamentalsStore

EVENT_PAIR_LIBRARY = "target_event_pairs"
EVENT_PAIR_SECTION = "event_pairs"

_SUPPORTED_FAMILIES = tuple(EVENT_PAIR_TAXONOMY)
_HISTORICAL_SECTIONS: dict[str, tuple[str, ...]] = {
    "analyst_rating": ("estimates_price_target",),
    "capital_action": ("cash",),
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

        present_frame = _concat_event_pair_frames(frames)
        present = set(present_frame["event_family"]) if not present_frame.empty else set()
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

        combined = _dedupe_event_pairs(_concat_event_pair_frames(frames))
        source = "cache"
        if not historical.empty:
            source = "historical"
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
    return _dedupe_event_pairs(_concat_event_pair_frames(frames))


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
    if family == "analyst_rating":
        frame = fundamentals.read(symbol, section="estimates_price_target", provider=provider, start=start_date, end=end_date)
        return _build_analyst_rating(symbol, frame, start_date=start_date, end_date=end_date)
    if family == "capital_action":
        frame = fundamentals.read(symbol, section="cash", provider=provider, start=start_date, end=end_date)
        return _build_capital_action(symbol, frame, start_date=start_date, end_date=end_date)
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
    if family == "earnings":
        frame = fundamentals.read(symbol, section="earnings", provider=provider, start=start_date, end=end_date)
        if frame.empty and equity_calendar is not None:
            frame = _filter_calendar_symbol(equity_calendar.read("equity_calendar_earnings", provider=provider, start=start_date, end=end_date), symbol)
        return _build_earnings(symbol, frame, start_date=start_date, end_date=end_date)
    return pd.DataFrame(columns=EVENT_PAIR_COLUMNS)


def _build_analyst_rating(symbol: str, frame: pd.DataFrame, *, start_date: str | None, end_date: str | None) -> pd.DataFrame:
    frame = _prepare_historical_frame(symbol, frame)
    if frame.empty:
        return pd.DataFrame(columns=EVENT_PAIR_COLUMNS)
    frame = filter_dates(frame, start_date=start_date, end_date=end_date)
    if frame.empty:
        return pd.DataFrame(columns=EVENT_PAIR_COLUMNS)
    if "action" not in frame.columns:
        frame["action"] = first_present_column(frame, ("action", "news_title", "title"))
    frame["event_type"] = frame.apply(analyst_rating_event_type, axis=1)
    frame = frame.dropna(subset=["event_type"]).copy()
    if frame.empty:
        return pd.DataFrame(columns=EVENT_PAIR_COLUMNS)
    frame["event_date"] = first_present_column(frame, ("date", "publishedDate", "published_date", "gradingDate", "grading_date"))
    frame["actor_type"] = "analyst"
    analyst_firm = first_present_column(
        frame,
        ("gradingCompany", "grading_company", "company", "analystCompany", "analyst_company", "analystFirm", "analyst_firm", "firm", "analystName", "analyst_name"),
    )
    frame["actor_name"] = analyst_firm
    frame["actor_firm"] = analyst_firm
    frame["actor_role"] = "analyst"
    frame["strength"] = first_present_column(frame, ("newGrade", "new_grade", "newRating", "new_rating", "action", "news_title", "title"))
    frame["reported_date"] = first_present_column(frame, ("publishedDate", "published_date", "date", "gradingDate", "grading_date"))
    frame["raw_json"] = raw_records(frame)
    return normalize_family_frame(frame, event_family="analyst_rating", source="warehouse:estimates_price_target")


def _build_capital_action(symbol: str, frame: pd.DataFrame, *, start_date: str | None, end_date: str | None) -> pd.DataFrame:
    frame = _prepare_historical_frame(symbol, frame)
    if frame.empty:
        return pd.DataFrame(columns=EVENT_PAIR_COLUMNS)
    frame = filter_dates(frame, start_date=start_date, end_date=end_date)
    if frame.empty:
        return pd.DataFrame(columns=EVENT_PAIR_COLUMNS)

    event_date = first_present_column(frame, ("filing_date", "date", "accepted_date", "period_ending"))
    repurchased = first_numeric_column(
        frame,
        ("common_stock_repurchased", "commonStockRepurchased", "repurchases_of_common_stock", "stock_repurchased"),
    )
    issued = first_numeric_column(
        frame,
        ("common_stock_issuance", "commonStockIssuance", "issuance_of_common_stock", "stock_issued", "net_common_stock_issuance"),
    )

    rows: list[pd.DataFrame] = []
    buyback = frame.loc[repurchased.fillna(0).abs() > 0].copy()
    if not buyback.empty:
        buyback["event_date"] = event_date.loc[buyback.index]
        buyback["event_type"] = "buyback_authorization"
        buyback["actor_type"] = "issuer"
        buyback["actor_name"] = symbol
        buyback["strength"] = repurchased.loc[buyback.index]
        buyback["raw_json"] = raw_records(buyback)
        rows.append(buyback)

    offering = frame.loc[issued.fillna(0) > 0].copy()
    if not offering.empty:
        offering["event_date"] = event_date.loc[offering.index]
        offering["event_type"] = "equity_offering"
        offering["actor_type"] = "issuer"
        offering["actor_name"] = symbol
        offering["strength"] = issued.loc[offering.index]
        offering["raw_json"] = raw_records(offering)
        rows.append(offering)

    if not rows:
        return pd.DataFrame(columns=EVENT_PAIR_COLUMNS)
    out = pd.concat(rows, ignore_index=True)
    return normalize_family_frame(out, event_family="capital_action", source="warehouse:cash")


def _build_insider(symbol: str, frame: pd.DataFrame, *, start_date: str | None, end_date: str | None) -> pd.DataFrame:
    frame = _prepare_historical_frame(symbol, frame)
    if frame.empty:
        return pd.DataFrame(columns=EVENT_PAIR_COLUMNS)
    frame = filter_dates(frame, start_date=start_date, end_date=end_date)
    frame["event_type"] = frame.apply(insider_event_type, axis=1)
    frame = frame.dropna(subset=["event_type"]).copy()
    if frame.empty:
        return pd.DataFrame(columns=EVENT_PAIR_COLUMNS)
    frame["event_date"] = first_present_column(frame, ("transactionDate", "transaction_date", "filingDate", "filing_date", "date"))
    actor_title = first_present_column(frame, ("officerTitle", "officer_title", "typeOfOwner", "type_of_owner", "relationship"))
    shares = first_numeric_column(frame, ("securitiesTransacted", "securities_transacted", "shares", "transactionShares", "transaction_shares"))
    price = first_numeric_column(frame, ("price", "transactionPrice", "transaction_price", "securityPrice", "security_price"))
    frame["actor_type"] = actor_title
    frame["actor_name"] = first_present_column(frame, ("reportingName", "reporting_name", "ownerName", "owner_name", "name"))
    frame["actor_title"] = actor_title
    frame["actor_role"] = actor_title.map(_normalize_insider_role)
    frame["strength"] = shares
    frame["transaction_shares"] = shares
    frame["transaction_price"] = price
    frame["transaction_value"] = shares.abs() * price
    frame["reported_date"] = first_present_column(frame, ("filingDate", "filing_date", "reportedDate", "reported_date", "date"))
    frame["raw_json"] = raw_records(frame)
    return normalize_family_frame(frame, event_family="insider", source="warehouse:ownership_insider_trading")


def _build_congress(symbol: str, frame: pd.DataFrame, *, start_date: str | None, end_date: str | None) -> pd.DataFrame:
    frame = _prepare_historical_frame(symbol, frame)
    if frame.empty:
        return pd.DataFrame(columns=EVENT_PAIR_COLUMNS)
    frame = filter_dates(frame, start_date=start_date, end_date=end_date)
    frame["event_type"] = frame.apply(congress_event_type, axis=1)
    frame = frame.dropna(subset=["event_type"]).copy()
    if frame.empty:
        return pd.DataFrame(columns=EVENT_PAIR_COLUMNS)
    frame["event_date"] = first_present_column(frame, ("transactionDate", "transaction_date", "disclosureDate", "disclosure_date", "date"))
    chamber = first_present_column(frame, ("chamber", "office")).fillna("congress")
    frame["actor_type"] = chamber
    frame["actor_chamber"] = chamber.map(_normalize_congress_chamber)
    actor_name = first_present_column(frame, ("representative", "senator", "firstName", "first_name", "name"))
    frame["actor_name"] = combine_names(actor_name, first_present_column(frame, ("lastName", "last_name")))
    frame["reported_date"] = first_present_column(frame, ("disclosureDate", "disclosure_date", "filingDate", "filing_date", "date"))
    frame["strength"] = first_present_column(frame, ("amount", "amountRange", "amount_range", "assetDescription", "asset_description"))
    frame["raw_json"] = raw_records(frame)
    return normalize_family_frame(frame, event_family="congress", source="warehouse:ownership_government_trades")


def _build_price_target(symbol: str, frame: pd.DataFrame, *, start_date: str | None, end_date: str | None) -> pd.DataFrame:
    frame = _prepare_historical_frame(symbol, frame)
    if frame.empty:
        return pd.DataFrame(columns=EVENT_PAIR_COLUMNS)
    frame = filter_dates(frame, start_date=start_date, end_date=end_date)
    frame["event_date"] = first_present_column(frame, ("publishedDate", "published_date", "date"))
    frame["target_value"] = first_numeric_column(frame, ("priceTarget", "price_target", "adjPriceTarget", "adj_price_target", "target"))
    frame = frame.dropna(subset=["event_date", "target_value"]).copy()
    group_key = first_present_column(
        frame,
        ("analystCompany", "analyst_company", "analystFirm", "analyst_firm", "analystName", "analyst_name", "publisher"),
    ).fillna("all")
    frame["_group_key"] = group_key
    frame = frame.sort_values(["_group_key", "event_date"])
    frame["previous_target_value"] = frame.groupby("_group_key")["target_value"].shift(1)
    frame["event_type"] = frame.apply(price_target_event_type, axis=1)
    frame = frame.dropna(subset=["event_type"]).copy()
    if frame.empty:
        return pd.DataFrame(columns=EVENT_PAIR_COLUMNS)
    analyst_actor = first_present_column(
        frame,
        ("analystName", "analyst_name", "analystFirm", "analyst_firm", "analystCompany", "analyst_company", "publisher"),
    )
    analyst_firm = first_present_column(
        frame,
        ("analystFirm", "analyst_firm", "analystCompany", "analyst_company", "publisher", "analystName", "analyst_name"),
    )
    frame["actor_type"] = "analyst"
    frame["actor_name"] = analyst_actor
    frame["actor_firm"] = analyst_firm
    frame["actor_role"] = "analyst"
    frame["strength"] = frame["target_value"]
    frame["reported_date"] = first_present_column(frame, ("publishedDate", "published_date", "date"))
    frame["raw_json"] = raw_records(frame)
    return normalize_family_frame(frame, event_family="price_target", source="warehouse:estimates_price_target")


def _build_institutional(symbol: str, frame: pd.DataFrame, *, start_date: str | None, end_date: str | None) -> pd.DataFrame:
    frame = _prepare_historical_frame(symbol, frame)
    if frame.empty:
        return pd.DataFrame(columns=EVENT_PAIR_COLUMNS)
    frame = filter_dates(frame, start_date=start_date, end_date=end_date)
    frame["event_date"] = first_present_column(frame, ("event_date", "date", "as_of", "period_ending", "filing_date"))
    frame["delta"] = first_numeric_column(
        frame,
        (
            "changeInShares",
            "change_in_shares",
            "sharesChange",
            "shares_change",
            "sharesHeldChange",
            "shares_held_change",
            "numberOf13fSharesChange",
            "number_of_13f_shares_change",
            "investorsHoldingChange",
            "investors_holding_change",
            "totalInvestedChange",
            "total_invested_change",
            "ownershipPercentChange",
            "ownership_percent_change",
        ),
    )
    if frame["delta"].isna().all():
        shares = first_numeric_column(frame, ("sharesHeld", "shares_held", "shares", "totalShares", "total_shares"))
        frame = frame.sort_values("event_date")
        frame["delta"] = shares.diff()
    frame["event_type"] = frame["delta"].map(institutional_event_type)
    frame = frame.dropna(subset=["event_type"]).copy()
    if frame.empty:
        return pd.DataFrame(columns=EVENT_PAIR_COLUMNS)
    frame["actor_type"] = "institution"
    frame["actor_name"] = first_present_column(frame, ("holder", "holder_name", "investor", "name")).fillna("aggregate")
    frame["strength"] = frame["delta"]
    frame["raw_json"] = raw_records(frame)
    return normalize_family_frame(frame, event_family="institutional", source="warehouse:ownership_institutional")


def _build_dividend(symbol: str, frame: pd.DataFrame, *, start_date: str | None, end_date: str | None) -> pd.DataFrame:
    frame = _prepare_historical_frame(symbol, frame)
    if frame.empty:
        return pd.DataFrame(columns=EVENT_PAIR_COLUMNS)
    frame = filter_dates(frame, start_date=start_date, end_date=end_date)
    frame["event_date"] = first_present_column(frame, ("exDividendDate", "ex_dividend_date", "declarationDate", "declaration_date", "date", "paymentDate", "payment_date"))
    frame["dividend_value"] = first_numeric_column(frame, ("adjDividend", "adj_dividend", "dividend", "amount", "cashAmount", "cash_amount"))
    frame = frame.dropna(subset=["event_date", "dividend_value"]).copy()
    frame = frame.sort_values("event_date")
    frame["previous_dividend_value"] = frame["dividend_value"].shift(1)
    frame["event_type"] = frame.apply(dividend_event_type, axis=1)
    frame = frame.dropna(subset=["event_type"]).copy()
    if frame.empty:
        return pd.DataFrame(columns=EVENT_PAIR_COLUMNS)
    frame["actor_type"] = "issuer"
    frame["actor_name"] = symbol
    frame["strength"] = frame["dividend_value"]
    frame["raw_json"] = raw_records(frame)
    return normalize_family_frame(frame, event_family="dividend", source="warehouse:dividends")


def _build_split(symbol: str, frame: pd.DataFrame, *, start_date: str | None, end_date: str | None) -> pd.DataFrame:
    frame = _prepare_historical_frame(symbol, frame)
    if frame.empty:
        return pd.DataFrame(columns=EVENT_PAIR_COLUMNS)
    frame = filter_dates(frame, start_date=start_date, end_date=end_date)
    frame["event_date"] = first_present_column(frame, ("date", "splitDate", "split_date"))
    frame["split_ratio"] = frame.apply(split_ratio, axis=1)
    frame["event_type"] = frame["split_ratio"].map(split_event_type)
    frame = frame.dropna(subset=["event_type"]).copy()
    if frame.empty:
        return pd.DataFrame(columns=EVENT_PAIR_COLUMNS)
    frame["actor_type"] = "issuer"
    frame["actor_name"] = symbol
    frame["strength"] = frame["split_ratio"]
    frame["raw_json"] = raw_records(frame)
    return normalize_family_frame(frame, event_family="split", source="warehouse:historical_splits")


def _build_earnings(symbol: str, frame: pd.DataFrame, *, start_date: str | None, end_date: str | None) -> pd.DataFrame:
    frame = _prepare_historical_frame(symbol, frame)
    if frame.empty:
        return pd.DataFrame(columns=EVENT_PAIR_COLUMNS)
    frame = filter_dates(frame, start_date=start_date, end_date=end_date)
    frame["event_date"] = first_present_column(frame, ("date", "reportDate", "report_date", "reportedDate", "reported_date", "fiscalDateEnding", "fiscal_date_ending"))
    actual = first_numeric_column(frame, ("epsActual", "eps_actual", "actualEps", "actual_eps", "eps", "reportedEPS", "reported_eps"))
    estimated = first_numeric_column(
        frame,
        (
            "epsEstimated",
            "eps_estimated",
            "estimatedEps",
            "estimated_eps",
            "epsEstimate",
            "eps_estimate",
            "epsConsensus",
            "eps_consensus",
        ),
    )
    frame["surprise"] = actual - estimated
    frame["event_type"] = frame["surprise"].map(earnings_event_type)
    frame = frame.dropna(subset=["event_type"]).copy()
    if frame.empty:
        return pd.DataFrame(columns=EVENT_PAIR_COLUMNS)
    frame["actor_type"] = "issuer"
    frame["actor_name"] = symbol
    frame["strength"] = frame["surprise"]
    frame["raw_json"] = raw_records(frame)
    return normalize_family_frame(frame, event_family="earnings", source="warehouse:equity_calendar_earnings")


def _prepare_historical_frame(symbol: str, frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    out = frame.copy()
    if isinstance(out.index, pd.DatetimeIndex):
        index_name = out.index.name or "date"
        out = out.reset_index()
        if "index" in out.columns and index_name not in out.columns:
            out = out.rename(columns={"index": index_name})
    return ensure_symbol(out, symbol)


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
    out["event_date"] = pd.to_datetime(out["event_date"], errors="coerce", utc=True).dt.tz_convert(None).dt.normalize()
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
    for column in EVENT_PAIR_COLUMNS:
        if column not in out.columns:
            out[column] = None
    out["event_date"] = pd.to_datetime(out["event_date"], errors="coerce", utc=True).dt.tz_convert(None).dt.normalize()
    out = out.dropna(subset=["event_date"])
    out = out.drop_duplicates(
        subset=["symbol", "event_date", "event_family", "event_type", "actor_type", "actor_name", "strength"],
        keep="last",
    )
    return out.sort_values(["symbol", "event_date", "event_family", "event_type"], ignore_index=True)[EVENT_PAIR_COLUMNS]


def _concat_event_pair_frames(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    usable: list[pd.DataFrame] = []
    for frame in frames:
        if frame is None or frame.empty:
            continue
        out = frame.copy()
        for column in EVENT_PAIR_COLUMNS:
            if column not in out.columns:
                out[column] = None
        out = out[EVENT_PAIR_COLUMNS]
        if out.dropna(how="all").empty:
            continue
        usable.append(out.dropna(axis=1, how="all"))
    if not usable:
        return pd.DataFrame(columns=EVENT_PAIR_COLUMNS)
    combined = pd.concat(usable, ignore_index=True)
    for column in EVENT_PAIR_COLUMNS:
        if column not in combined.columns:
            combined[column] = None
    return combined[EVENT_PAIR_COLUMNS]


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
    combined["event_date"] = pd.to_datetime(combined["event_date"], errors="coerce", utc=True).dt.tz_convert(None).dt.normalize()
    combined = combined.dropna(subset=["event_date"]).set_index("event_date")
    combined.index = pd.DatetimeIndex(combined.index)
    combined.index.name = "event_date"
    return combined.sort_index()


def _normalize_congress_chamber(value: object) -> str | None:
    text = str(value).strip().lower() if pd.notna(value) else ""
    if not text:
        return None
    if "senate" in text or "senator" in text:
        return "senate"
    if "house" in text or "representative" in text or "rep." in text:
        return "house"
    return "congress_other"


def _normalize_insider_role(value: object) -> str | None:
    text = str(value).strip().lower() if pd.notna(value) else ""
    if not text:
        return None
    if "chief executive" in text or re.search(r"\bceo\b", text):
        return "ceo"
    if "chief financial" in text or re.search(r"\bcfo\b", text):
        return "cfo"
    if "director" in text:
        return "director"
    if "president" in text:
        return "president"
    if "officer" in text:
        return "officer"
    if "10%" in text or "ten percent" in text or "beneficial owner" in text:
        return "large_holder"
    return "other_insider"


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
