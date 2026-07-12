import pandas as pd

from quant_warehouse.ingest.oracle_news import (
    oracle_trade_boundaries,
    refresh_fmp_news_for_oracle_trades,
)


def _trades():
    return pd.DataFrame(
        {
            "symbol": ["aapl", "AAPL", "MSFT"],
            "entry_date": ["2024-01-02", "2024-01-03", "2024-01-02"],
            "exit_date": ["2024-01-03", "2024-01-04", "2024-01-03"],
        }
    )


def test_oracle_boundaries_dedupe_and_preserve_entry_exit_role():
    result = oracle_trade_boundaries(_trades())
    row = result.loc[
        result["symbol"].eq("AAPL") & result["observation_date"].eq(pd.Timestamp("2024-01-03"))
    ].iloc[0]
    assert row["boundary_kind"] == "entry,exit"
    assert len(result) == 5


def test_refresh_requests_only_exact_boundary_dates(tmp_path):
    calls = []

    def fetcher(section, **kwargs):
        calls.append((section, kwargs))
        day = kwargs["start_date"]
        if kwargs["page"]:
            return pd.DataFrame()
        return pd.DataFrame(
            {
                "symbols": [kwargs["symbol"].split(",")[0]],
                "date": [f"{day}T12:00:00Z"],
                "title": ["news"],
                "url": [f"https://example.test/{day}"],
            }
        )

    output = tmp_path / "news.parquet"
    result = refresh_fmp_news_for_oracle_trades(_trades(), output_path=output, fetcher=fetcher)
    assert output.exists()
    assert result.requests == 3
    assert {call[1]["start_date"] for call in calls} == {"2024-01-02", "2024-01-03", "2024-01-04"}
    assert all(call[1]["start_date"] == call[1]["end_date"] for call in calls)
    assert all(call[0] == "company_news" and call[1]["provider"] == "fmp" for call in calls)


def test_refresh_accepts_openbb_datetime_index(tmp_path):
    def fetcher(_section, **kwargs):
        return pd.DataFrame(
            {"symbols": [kwargs["symbol"].split(",")[0]], "title": ["indexed"]},
            index=pd.DatetimeIndex([f"{kwargs['start_date']}T12:00:00"], name="date"),
        )

    result = refresh_fmp_news_for_oracle_trades(
        _trades().iloc[:1], output_path=tmp_path / "indexed.parquet", fetcher=fetcher
    )
    assert result.news_rows == 2

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("completed dates should resume from checkpoints")

    resumed = refresh_fmp_news_for_oracle_trades(
        _trades().iloc[:1], output_path=tmp_path / "indexed.parquet", fetcher=fail_if_called
    )
    assert resumed.requests == 0
    assert resumed.news_rows == 2
