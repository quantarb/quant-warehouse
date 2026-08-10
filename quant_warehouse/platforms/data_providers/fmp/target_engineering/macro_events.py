"""Polars-only macro event normalization and response targets."""
from __future__ import annotations

import re
from dataclasses import dataclass
import polars as pl

MACRO_RESPONSE_CLASSES = ("strong_negative", "negative", "neutral", "positive", "strong_positive")

@dataclass(frozen=True)
class MacroEventSpec:
    horizons: tuple[int, ...] = (1, 5, 20)
    minimum_cross_section: int = 5
    require_actual: bool = True

def _slug(value: object) -> str:
    text = re.sub(r"[^a-z0-9]+", "_", str(value or "").lower()).strip("_")
    return text or "unknown_event"

def _canonical_event_type(event_type: str) -> str:
    months = "jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec"
    value = re.sub(rf"_(?:{months})(?:_\d{{1,2}})?$", "", str(event_type))
    value = re.sub(r"_q[1-4]$", "", value)
    return re.sub(r"_(?:mom|qoq|yoy)$", "", value) or str(event_type)

def _compact_directional_event_type(event_type: str, groups: set[str] | None = None) -> str:
    value = _canonical_event_type(event_type)
    groups = groups or {"treasury", "mortgage", "consumer"}
    if "treasury" in groups and value.endswith("_auction") and any(x in value for x in ("_bill_", "_note_", "_bond_", "_tips_", "_frn_")):
        return "treasury_auction"
    if "mortgage" in groups and value.endswith("_mortgage_rate"):
        return "mortgage_rate"
    if "consumer" in groups and value in {"all_car_sales", "all_truck_sales", "total_vehicle_sales"}:
        return "vehicle_sales"
    if "consumer" in groups and value == "consumer_credit_change": return "consumer_credit"
    if "consumer" in groups and value in {"cb_consumer_confidence", "consumer_inflation_expectation", "michigan_consumer_expectations", "michigan_consumer_sentiment"}: return "consumer_sentiment"
    if "consumer" in groups and value in {"personal_spending", "real_consumer_spending", "retail_sales", "retail_sales_ex_autos", "retail_sales_ex_gas_autos"}: return "consumer_spending"
    if "consumer" in groups and value == "retail_inventories_ex_autos": return "retail_inventory"
    if "aliases" in groups and value in {"core_pce_price_index", "core_pce_prices"}: return "core_pce"
    return value

def _date_expr(frame: pl.DataFrame, name: str) -> pl.Expr:
    return ((pl.col(name).str.to_datetime(strict=False, time_zone="UTC") if frame.schema[name] == pl.String else pl.col(name).cast(pl.Datetime, strict=False))
            .dt.replace_time_zone(None).dt.truncate("1d"))

def normalize_macro_events(events: pl.DataFrame) -> pl.DataFrame:
    columns = ["macro_event_id", "date", "country", "currency", "event", "event_type", "impact", "previous", "estimate", "actual", "unit", "surprise", "surprise_pct"]
    if events is None or events.is_empty():
        return pl.DataFrame(schema={column: pl.String for column in columns})
    if "date" not in events.columns or "event" not in events.columns:
        raise ValueError("macro events require date and event columns")
    out = events.with_columns(_date_expr(events, "date").alias("date"))
    for column in ("previous", "estimate", "actual", "change", "changePercentage"):
        if column in out.columns: out = out.with_columns(pl.col(column).cast(pl.Float64, strict=False))
    for column in ("country", "currency", "impact", "unit"):
        if column not in out.columns: out = out.with_columns(pl.lit("", dtype=pl.String).alias(column))
    for column in ("previous", "estimate", "actual"):
        if column not in out.columns: out = out.with_columns(pl.lit(None, dtype=pl.Float64).alias(column))
    out = out.with_columns(
        pl.col("country").cast(pl.String, strict=False).fill_null("").str.to_uppercase(),
        pl.col("currency").cast(pl.String, strict=False).fill_null("").str.to_uppercase(),
        pl.col("event").map_elements(_slug, return_dtype=pl.String).alias("event_type"),
        pl.col("impact").cast(pl.String, strict=False).fill_null("").str.to_lowercase(),
        pl.col("unit").cast(pl.String, strict=False).fill_null(""),
    ).filter(pl.col("date").is_not_null()).with_columns(
        (pl.col("actual") - pl.col("estimate")).alias("surprise"),
        pl.when(pl.col("estimate").abs() != 0).then((pl.col("actual") - pl.col("estimate")) / pl.col("estimate").abs()).otherwise(None).alias("surprise_pct"),
    ).with_columns(pl.concat_str([pl.col("date").cast(pl.String), pl.col("country"), pl.col("currency"), pl.col("event_type")], separator="|").alias("macro_event_id"))
    return out.select([column for column in columns if column in out.columns]).sort("date")

def _macro_target_name_from_mapping(row: dict[str, object]) -> str:
    country = str(row.get("country") or "").lower()
    event_type = _canonical_event_type(str(row.get("event_type") or "unknown_event"))
    try:
        actual, previous = float(row["actual"]), float(row["previous"])
        rate = country == "us" and any(x in event_type for x in ("interest_rate_decision", "federal_funds_rate", "fed_funds_rate"))
        if rate: return "fed_rate_cut" if actual < previous else "fed_rate_hike" if actual > previous else "fed_rate_hold"
        direction = "increase" if actual > previous else "decrease" if actual < previous else "unchanged"
        return f"macro_{country or 'global'}_{event_type}_{direction}"
    except (TypeError, ValueError):
        return f"macro_{country or 'global'}_{event_type}"

def build_macro_event_targets(events: pl.DataFrame) -> pl.DataFrame:
    macro = normalize_macro_events(events)
    if macro.is_empty():
        return macro.with_columns(pl.lit(None, dtype=pl.String).alias("target_name"), pl.lit(None, dtype=pl.String).alias("target_column"))
    names = [_macro_target_name_from_mapping(row) for row in macro.iter_rows(named=True)]
    return macro.with_columns(pl.Series("target_name", names), pl.Series("target_column", [f"is_{name}" for name in names]))

def build_macro_event_label_panel(tokens: pl.DataFrame, events: pl.DataFrame, *, date_column: str = "date",
                                  directional_only: bool = False, compact_directional: bool = False,
                                  compact_groups: tuple[str, ...] | None = None, deduplicate_identical: bool = False) -> pl.DataFrame:
    if date_column not in tokens.columns: raise ValueError(f"tokens must contain {date_column!r}")
    out = tokens.with_columns(_date_expr(tokens, date_column).alias(date_column))
    macro = build_macro_event_targets(events)
    if macro.is_empty(): return out
    if compact_directional:
        if not directional_only: raise ValueError("compact_directional requires directional_only=True")
        macro = macro.with_columns(pl.struct(macro.columns).map_elements(lambda row: row["target_name"] if row["target_name"] in {"fed_rate_cut", "fed_rate_hike", "fed_rate_hold"} else f"macro_{str(row.get('country','')).lower() or 'global'}_{_compact_directional_event_type(str(row.get('event_type','unknown_event')), set(compact_groups) if compact_groups else None)}_{str(row['target_name']).rsplit('_',1)[-1]}", return_dtype=pl.String).alias("target_name")).with_columns(pl.concat_str([pl.lit("is_"), pl.col("target_name")]).alias("target_column"))
    if directional_only:
        macro = macro.filter(pl.col("target_name").str.ends_with("_increase") | pl.col("target_name").str.ends_with("_decrease") | pl.col("target_name").is_in(["fed_rate_cut", "fed_rate_hike", "fed_rate_hold"]))
        if macro.is_empty(): return out
    wide = macro.select(["date", "target_column"]).unique().with_columns(pl.lit(1.0).alias("value")).pivot(on="target_column", index="date", values="value", aggregate_function="max").fill_null(0.0)
    out = out.join(wide, left_on=date_column, right_on="date", how="left")
    label_cols = [c for c in wide.columns if c != "date"]
    return out.with_columns([pl.col(c).fill_null(0.0) for c in label_cols])

def deduplicate_binary_label_columns(panel: pl.DataFrame, *, prefix: str = "is_") -> tuple[pl.DataFrame, dict[str, str]]:
    labels = [c for c in panel.columns if str(c).startswith(prefix)]
    signatures: dict[tuple[int, ...], list[str]] = {}
    for column in labels:
        signature = tuple(panel.get_column(column).cast(pl.Float64, strict=False).fill_null(0).cast(pl.Int8).to_list())
        signatures.setdefault(signature, []).append(column)
    mapping: dict[str, str] = {}; out = panel
    for columns in signatures.values():
        if len(columns) < 2: continue
        keep = columns[0]
        for duplicate in columns[1:]: mapping[duplicate] = keep
        out = out.with_columns(pl.max_horizontal(columns).alias(keep)).drop(columns[1:])
    return out, mapping

def build_macro_family_label_panel(tokens: pl.DataFrame, events: pl.DataFrame, *, date_column: str = "date") -> pl.DataFrame:
    if date_column not in tokens.columns: raise ValueError(f"tokens must contain {date_column!r}")
    out = tokens.with_columns(_date_expr(tokens, date_column).alias(date_column))
    macro = normalize_macro_events(events)
    if macro.is_empty(): return out
    macro = macro.with_columns(pl.concat_str([pl.lit("macro_"), pl.col("country").str.to_lowercase().fill_null("global"), pl.lit("_"), pl.col("event_type")]).alias("event_family"), pl.when(pl.col("actual") > pl.col("previous")).then(0).when(pl.col("actual") < pl.col("previous")).then(1).when(pl.col("actual") == pl.col("previous")).then(2).otherwise(-1).cast(pl.Int8).alias("direction_code"))
    presence = macro.select(["date", "event_family"]).unique().with_columns(pl.lit(1.0).alias("value")).pivot(on="event_family", index="date", values="value", aggregate_function="max").fill_null(0.0)
    presence = presence.rename({c: f"is_{c}" for c in presence.columns if c != "date"})
    directions = macro.filter(pl.col("direction_code") >= 0).group_by(["date", "event_family"]).agg(pl.col("direction_code").first()).pivot(on="event_family", index="date", values="direction_code")
    directions = directions.rename({c: f"macro_direction_{c}" for c in directions.columns if c != "date"})
    surprises = macro.filter(pl.col("surprise_pct").is_not_null()).group_by(["date", "event_family"]).agg(pl.col("surprise_pct").mean()).pivot(on="event_family", index="date", values="surprise_pct")
    surprises = surprises.rename({c: f"macro_surprise_{c}" for c in surprises.columns if c != "date"})
    result = out.join(presence, left_on=date_column, right_on="date", how="left")
    result = result.join(directions, left_on=date_column, right_on="date", how="left") if directions.width > 1 else result
    return result.join(surprises, left_on=date_column, right_on="date", how="left") if surprises.width > 1 else result

def build_macro_response_labels(prices: pl.DataFrame, events: pl.DataFrame, spec: MacroEventSpec | None = None, *, symbol_column: str = "symbol", date_column: str = "date", price_column: str = "close") -> pl.DataFrame:
    spec = spec or MacroEventSpec()
    if not {symbol_column, date_column, price_column}.issubset(prices.columns): raise ValueError("prices lacks required columns")
    macro = normalize_macro_events(events)
    if macro.is_empty(): return pl.DataFrame()
    if spec.require_actual: macro = macro.filter(pl.col("actual").is_not_null())
    panel = prices.select([symbol_column, date_column, price_column]).with_columns(_date_expr(prices, date_column).alias(date_column), pl.col(symbol_column).cast(pl.String).str.to_uppercase(), pl.col(price_column).cast(pl.Float64, strict=False)).drop_nulls().sort([symbol_column, date_column])
    outputs: list[pl.DataFrame] = []
    for horizon in spec.horizons:
        future = panel.with_columns(pl.col(date_column).alias("event_date"), pl.col(date_column).shift(-int(horizon)).over(symbol_column).alias("future_date"), pl.col(price_column).shift(-int(horizon)).over(symbol_column).alias("future_price")).select([symbol_column, "event_date", "future_date", "future_price"])
        current = panel.rename({date_column: "event_date", price_column: "event_price"})
        joined = current.join(future, on=[symbol_column, "event_date"], how="left").with_columns((pl.col("future_price") / pl.col("event_price") - 1).alias("forward_return")).join(macro, left_on="event_date", right_on="date", how="inner").drop_nulls("forward_return")
        if joined.is_empty(): continue
        joined = joined.with_columns(pl.col("forward_return").rank(method="average").over("macro_event_id").alias("rank"), pl.len().over("macro_event_id").alias("count"))
        joined = joined.with_columns(pl.when(pl.col("count") < spec.minimum_cross_section).then(2).otherwise((pl.col("rank") / pl.col("count") * 5).ceil().clip(upper_bound=4).cast(pl.Int8)).alias("response_class"), pl.lit(int(horizon)).alias("horizon"))
        outputs.append(joined)
    return pl.concat(outputs, how="diagonal_relaxed").sort(["event_date", "macro_event_id", symbol_column, "horizon"]) if outputs else pl.DataFrame()
