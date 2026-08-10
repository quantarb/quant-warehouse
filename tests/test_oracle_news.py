import polars as pl

from quant_warehouse.ingest.oracle_news import (
    oracle_trade_boundaries,
    refresh_fmp_news_for_oracle_trades,
)


def _trades():
    return pl.DataFrame(
        {
            "symbol": ["aapl", "AAPL", "MSFT"],
            "entry_date": ["2024-01-02", "2024-01-03", "2024-01-02"],
            "exit_date": ["2024-01-03", "2024-01-04", "2024-01-03"],
        }
    )


def test_oracle_boundaries_dedupe_and_preserve_entry_exit_role():
    result = oracle_trade_boundaries(_trades())
    row = result.filter((pl.col("symbol") == "AAPL") & (pl.col("observation_date") == pl.datetime(2024, 1, 3)))[0]
    assert row[0, "boundary_kind"] == "entry,exit"
    assert result.height == 5


def test_refresh_requests_only_exact_boundary_dates(tmp_path):
    calls = []

    def fetcher(section, **kwargs):
        calls.append((section, kwargs))
        day = kwargs["start_date"]
        if kwargs["page"]:
            return pl.DataFrame()
        return pl.DataFrame(
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
        return pl.DataFrame({"symbols": [kwargs["symbol"].split(",")[0]],
                             "date": [f"{kwargs['start_date']}T12:00:00"], "title": ["indexed"]})

    result = refresh_fmp_news_for_oracle_trades(
        _trades().head(1), output_path=tmp_path / "indexed.parquet", fetcher=fetcher
    )
    assert result.news_rows == 2

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("completed dates should resume from checkpoints")

    resumed = refresh_fmp_news_for_oracle_trades(
        _trades().head(1), output_path=tmp_path / "indexed.parquet", fetcher=fail_if_called
    )
    assert resumed.requests == 0
    assert resumed.news_rows == 2
