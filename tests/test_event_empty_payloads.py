from datetime import datetime
import polars as pl
import pytest
from polars.testing import assert_frame_equal
from quant_warehouse.platforms.data_providers.fmp.feature_engineering.event_observations import (
    _required_event_payloads,
    issuer_event_observations,
)

SCHEMAS = {
    "ownership_government_trades": {"transaction_type": pl.String, "amount": pl.String},
    "ownership_insider_trading": {
        "transaction_type": pl.String,
        "acquisition_or_disposition": pl.String,
        "securities_owned": pl.Float64,
        "securities_transacted": pl.Float64,
        "transaction_price": pl.Float64,
    },
}


def source(section):
    if section == "ownership_government_trades":
        return pl.DataFrame(
            {
                "transaction_date": [datetime(2024, 1, 2)],
                "disclosure_date": [datetime(2024, 1, 9)],
                "transaction_type": ["Purchase"],
                "amount": ["$1,001 - $15,000"],
            }
        )
    return pl.DataFrame(
        {
            "transaction_date": [datetime(2024, 1, 2)],
            "filing_date": [datetime(2024, 1, 9)],
            "transaction_type": ["P-Purchase"],
            "acquisition_or_disposition": ["A"],
            "securities_owned": [30.0],
            "securities_transacted": [10.0],
            "transaction_price": [42.0],
        }
    )


def output(frame, section):
    class Warehouse:
        def read_fundamentals(self, symbol, *, section: str, **kwargs):
            return frame if section == requested else pl.DataFrame()

    requested = section
    return list(issuer_event_observations(Warehouse(), "TEST"))


@pytest.mark.parametrize("section,field", [(s, f) for s, fields in SCHEMAS.items() for f in fields])
def test_missing_event_field_matches_typed_null_source(section, field):
    populated = source(section)
    missing = populated.drop(field)
    null_source = populated.with_columns(pl.lit(None, dtype=SCHEMAS[section][field]).alias(field))
    reconstructed = _required_event_payloads(missing, section)
    assert reconstructed.schema[field] == SCHEMAS[section][field]
    assert reconstructed[field].null_count() == reconstructed.height
    absent_output, null_output = output(missing, section), output(null_source, section)
    assert len(absent_output) == len(null_output) == 1
    assert_frame_equal(absent_output[0][1], null_output[0][1])
    result = absent_output[0][1]
    assert result["date"].to_list() == [datetime(2024, 1, 9)]
    assert result["event_date"].to_list() == [datetime(2024, 1, 2)]
    channels = {
        "transaction_type": ["signal_value", "text_0"],
        "amount": ["text_1", "text_2"],
        "acquisition_or_disposition": ["text_4", "text_5"],
        "securities_owned": ["text_1"],
        "securities_transacted": ["text_2"],
        "transaction_price": ["text_3"],
    }
    for channel in channels[field]:
        assert result[channel].to_list() == [None]


@pytest.mark.parametrize("section", list(SCHEMAS))
def test_populated_event_values_and_dates_unchanged(section):
    frame = source(section)
    assert_frame_equal(_required_event_payloads(frame, section), frame)
    result = output(frame, section)[0][1]
    expected = (
        [1.0, 0.0, 1001.0, 15000.0]
        if section.endswith("government_trades")
        else [1.0, 0.0, 30.0, 10.0, 42.0, 1.0, 0.0]
    )
    assert result.select(["signal_value", *[f"text_{i}" for i in range(len(expected) - 1)]]).row(
        0
    ) == tuple(expected)
    assert result["date"].to_list() == [datetime(2024, 1, 9)]
    assert result["event_date"].to_list() == [datetime(2024, 1, 2)]


def test_government_wholly_unknown_payload_has_no_event():
    frame = source("ownership_government_trades").drop("transaction_type", "amount")
    assert output(frame, "ownership_government_trades") == []


def test_government_missing_disclosure_does_not_fabricate_date():
    frame = source("ownership_government_trades").drop("disclosure_date")
    with pytest.warns(RuntimeWarning, match="without a disclosure date"):
        assert output(frame, "ownership_government_trades") == []


@pytest.mark.parametrize(
    "section,date_field,fields",
    [
        (
            "estimates_price_target",
            "published_date",
            ["price_target", "adj_price_target", "price_when_posted"],
        ),
        ("dividends", "ex_dividend_date", ["amount", "adjusted_amount"]),
        ("historical_splits", "date", ["numerator", "denominator"]),
    ],
)
def test_other_event_sections_missing_payloads_remain_unknown(section, date_field, fields):
    frame = pl.DataFrame({date_field: [datetime(2024, 1, 9)]})
    result = output(frame, section)[0][1]
    for channel in ["signal_value", *[f"text_{i}" for i in range(len(fields) - 1)]]:
        assert result[channel].to_list() == [None]
    assert result["date"].to_list() == [datetime(2024, 1, 9)]


@pytest.mark.parametrize("section,field", [(s, f) for s, fields in SCHEMAS.items() for f in fields])
@pytest.mark.parametrize("sentinel", [float("nan"), ""])
def test_empty_raw_sentinel_matches_unknown_null_payload(section, field, sentinel):
    frame = source(section)
    raw_empty = frame.with_columns(pl.lit(sentinel).alias(field))
    typed_null = frame.with_columns(pl.lit(None, dtype=SCHEMAS[section][field]).alias(field))
    reconstructed = _required_event_payloads(raw_empty, section)
    assert reconstructed[field].to_list() == [None]
    actual, expected = output(raw_empty, section), output(typed_null, section)
    assert len(actual) == len(expected) == 1
    assert_frame_equal(actual[0][1], expected[0][1])
