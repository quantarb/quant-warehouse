from __future__ import annotations

from datetime import date, datetime, timedelta

import polars as pl
import pytest

from quant_warehouse.platforms.data_providers.thetadata.options import (
    OPTIONS_THETADATA_EOD_LIBRARY,
    OPTIONS_THETADATA_PROVIDER,
    ThetaDataDownloadSpec,
    _iter_eod_date_chunks,
    download_option_snapshots_for_range,
    deduplicate_option_chain_arctic,
    fetch_option_history_eod,
    option_chain_cached_date_summary,
    write_option_chain_arctic,
    read_option_chain_arctic,
    read_thetadata_eod_option_chain,
    normalize_thetadata_option_chain,
    split_snapshots_by_date,
    load_thetadata_option_snapshots,
)
from quant_warehouse.warehouse.storage import provider_library


def _ts(value: str | date | datetime) -> datetime:
    if isinstance(value, datetime):
        return value.replace(hour=0, minute=0, second=0, microsecond=0)
    if isinstance(value, date):
        return datetime.combine(value, datetime.min.time())
    return datetime.fromisoformat(str(value)[:10])


def _business_days(start: str | date | datetime, end: str | date | datetime) -> list[datetime]:
    first, last = _ts(start), _ts(end)
    return [first + timedelta(days=offset) for offset in range((last - first).days + 1)
            if (first + timedelta(days=offset)).weekday() < 5]


def _raw_frame() -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "symbol": "AAPL",
                "expiration": "2025-01-24",
                "strike": 230.0,
                "right": "PUT",
                "created": "2025-01-06 17:21:40-05:00",
                "bid": 0.66,
                "ask": 0.81,
            },
            {
                "symbol": "AAPL",
                "expiration": "2025-01-24",
                "strike": 235.0,
                "right": "PUT",
                "created": "2025-01-07 17:21:40-05:00",
                "bid": 0.70,
                "ask": 0.85,
            },
        ]
    )


def _with_rich_option_fields(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.with_columns([
        pl.lit(value).alias(column)
        for column, value in {
        "underlying_price": 230.0,
        "delta": -0.4,
        "gamma": 0.02,
        "theta": -0.03,
        "vega": 0.1,
        "rho": -0.01,
        "iv": 0.25,
        }.items()
    ])


def test_normalize_thetadata_option_chain_builds_contract_symbol() -> None:
    frame = normalize_thetadata_option_chain(_raw_frame())
    assert "contract_symbol" in frame.columns
    assert frame["contract_symbol"][0] == "AAPL_put_20250124_230"
    assert "snapshot_date" in frame.columns
    assert "mid" in frame.columns
    assert frame["data_interval"][0] == "eod"


def test_normalize_thetadata_option_chain_keeps_rows_without_bid_ask() -> None:
    raw = _raw_frame()
    raw = raw.with_row_index("_row").with_columns(
        pl.when(pl.col("_row") == 0).then(0.0).otherwise(pl.col("bid")).alias("bid")
    ).drop("_row")
    frame = normalize_thetadata_option_chain(raw)
    assert frame.height == 2
    assert frame["contract_symbol"][0] == "AAPL_put_20250124_230"


def test_split_snapshots_by_date_groups_rows() -> None:
    normalized = normalize_thetadata_option_chain(_raw_frame())
    snapshots = split_snapshots_by_date(normalized)
    assert len(snapshots) == 2


class _MemoryBackend:
    def __init__(self, initial: pl.DataFrame | None = None) -> None:
        self.frame = initial
        self.writes: list[tuple[str, str, pl.DataFrame, bool]] = []

    def read(self, library: str, symbol: str, **kwargs) -> pl.DataFrame | None:
        assert library in {
            provider_library(OPTIONS_THETADATA_EOD_LIBRARY, OPTIONS_THETADATA_PROVIDER),
        }
        if self.frame is None:
            return None
        out = self.frame.clone()
        date_range = kwargs.get("date_range")
        if date_range is not None and "date" in out.columns:
            start, end = date_range
            if start is not None:
                out = out.filter(pl.col("date") >= start)
            if end is not None:
                out = out.filter(pl.col("date") <= end)
        columns = kwargs.get("columns")
        if columns is not None:
            keep = [column for column in columns if column in out.columns]
            out = out.select(keep)
        return out

    def write(
        self,
        library: str,
        symbol: str,
        df: pl.DataFrame,
        *,
        prune_previous_versions: bool = False,
    ) -> None:
        assert library == provider_library(OPTIONS_THETADATA_EOD_LIBRARY, OPTIONS_THETADATA_PROVIDER)
        self.frame = df.clone()
        self.writes.append((library, symbol, df.clone(), prune_previous_versions))


def test_arctic_option_chain_roundtrip() -> None:
    frame = normalize_thetadata_option_chain(_raw_frame().head(1))
    backend = _MemoryBackend()
    expected_library = provider_library(OPTIONS_THETADATA_EOD_LIBRARY, OPTIONS_THETADATA_PROVIDER)
    assert write_option_chain_arctic("AAPL", frame, backend=backend) == f"arctic://{expected_library}/AAPL"
    assert backend.writes[0][0] == expected_library
    assert backend.writes[0][3] is True
    loaded = read_option_chain_arctic("AAPL", start_date="2025-01-06", end_date="2025-01-06", backend=backend)
    assert len(loaded) == 1
    assert loaded["contract_symbol"][0] == "AAPL_put_20250124_230"


def test_read_option_chain_arctic_deduplicates_existing_snapshot_contract_rows() -> None:
    duplicate = pl.DataFrame(
        [
            {
                "symbol": "AAPL",
                "expiration": "2025-01-24",
                "strike": 230.0,
                "right": "PUT",
                "created": "2025-01-06 17:21:40-05:00",
                "bid": 0.66,
                "ask": 0.81,
            },
            {
                "symbol": "AAPL",
                "expiration": "2025-01-24",
                "strike": 230.0,
                "right": "PUT",
                "created": "2025-01-06 18:21:40-05:00",
                "bid": 0.70,
                "ask": 0.85,
            },
        ]
    )
    cached = pl.concat([
        normalize_thetadata_option_chain(duplicate.head(1)),
        normalize_thetadata_option_chain(duplicate.tail(1)),
    ], how="diagonal_relaxed")
    cached = cached.with_columns(
        pl.Series("date", [datetime(2025, 1, 6), datetime(2025, 1, 6, 0, 0, 0, 1)])
    )
    backend = _MemoryBackend(cached)

    loaded = read_option_chain_arctic("AAPL", start_date="2025-01-06", end_date="2025-01-06", backend=backend)

    assert loaded.height == 1
    assert loaded["contract_symbol"][0] == "AAPL_put_20250124_230"
    assert loaded["bid"][0] == 0.70


def test_read_thetadata_eod_option_chain_enforces_contract_and_projects_columns() -> None:
    duplicate = pl.DataFrame(
        [
            {
                "symbol": "AAPL",
                "expiration": "2025-01-24",
                "strike": 230.0,
                "right": "PUT",
                "created": "2025-01-06 17:21:40-05:00",
                "bid": 0.66,
                "ask": 0.81,
            },
            {
                "symbol": "AAPL",
                "expiration": "2025-01-24",
                "strike": 230.0,
                "right": "PUT",
                "created": "2025-01-06 18:21:40-05:00",
                "bid": 0.70,
                "ask": 0.85,
            },
        ]
    )
    cached = _with_rich_option_fields(pl.concat([
        normalize_thetadata_option_chain(duplicate.head(1)),
        normalize_thetadata_option_chain(duplicate.tail(1)),
    ], how="diagonal_relaxed"))
    cached = cached.with_columns(
        pl.Series("date", [datetime(2025, 1, 6), datetime(2025, 1, 6, 0, 0, 0, 1)])
    )
    backend = _MemoryBackend(cached)

    loaded = read_thetadata_eod_option_chain(
        "AAPL",
        start_date="2025-01-06",
        end_date="2025-01-06",
        columns=["snapshot_date", "contract_symbol", "bid"],
        require_rich_columns=True,
        backend=backend,
    )

    assert list(loaded.columns) == ["snapshot_date", "contract_symbol", "bid"]
    assert loaded.height == 1
    assert loaded["contract_symbol"][0] == "AAPL_put_20250124_230"
    assert loaded["bid"][0] == 0.70


def test_read_thetadata_eod_option_chain_requires_rich_endpoint_columns_when_requested() -> None:
    quote_only = normalize_thetadata_option_chain(_raw_frame().head(1))
    backend = _MemoryBackend(quote_only)

    with pytest.raises(ValueError, match="underlying_price"):
        read_thetadata_eod_option_chain(
            "AAPL",
            start_date="2025-01-06",
            end_date="2025-01-06",
            require_rich_columns=True,
            backend=backend,
        )


def test_deduplicate_option_chain_arctic_rewrites_existing_cached_duplicates() -> None:
    duplicate = pl.DataFrame(
        [
            {
                "symbol": "AAPL",
                "expiration": "2025-01-24",
                "strike": 230.0,
                "right": "PUT",
                "created": "2025-01-06 17:21:40-05:00",
                "bid": 0.66,
                "ask": 0.81,
            },
            {
                "symbol": "AAPL",
                "expiration": "2025-01-24",
                "strike": 230.0,
                "right": "PUT",
                "created": "2025-01-06 18:21:40-05:00",
                "bid": 0.70,
                "ask": 0.85,
            },
        ]
    )
    cached = pl.concat([
        normalize_thetadata_option_chain(duplicate.head(1)),
        normalize_thetadata_option_chain(duplicate.tail(1)),
    ], how="diagonal_relaxed")
    cached = cached.with_columns(
        pl.Series("date", [datetime(2025, 1, 6), datetime(2025, 1, 6, 0, 0, 0, 1)])
    )
    backend = _MemoryBackend(cached)

    dry_run = deduplicate_option_chain_arctic(["AAPL"], backend=backend)
    assert dry_run["duplicate_rows"][0] == 1
    assert bool(dry_run["rewritten"][0]) is False
    assert backend.frame.height == 2

    rewritten = deduplicate_option_chain_arctic(["AAPL"], backend=backend, dry_run=False)

    assert rewritten["duplicate_rows"][0] == 1
    assert bool(rewritten["rewritten"][0]) is True
    assert backend.frame.height == 1
    assert backend.writes[-1][3] is True


def test_cached_date_summary_requires_rich_greeks_columns() -> None:
    quote_only = normalize_thetadata_option_chain(_raw_frame().head(1))
    rich = _with_rich_option_fields(quote_only)

    quote_backend = _MemoryBackend(quote_only)
    rich_backend = _MemoryBackend(rich)

    quote_dates, quote_rows = option_chain_cached_date_summary(
        "AAPL",
        "2025-01-06",
        "2025-01-06",
        required_columns=("underlying_price", "delta", "gamma", "theta", "vega", "rho", "iv"),
        backend=quote_backend,
    )
    rich_dates, rich_rows = option_chain_cached_date_summary(
        "AAPL",
        "2025-01-06",
        "2025-01-06",
        required_columns=("underlying_price", "delta", "gamma", "theta", "vega", "rho", "iv"),
        backend=rich_backend,
    )

    assert quote_dates == set()
    assert quote_rows == 0
    assert rich_dates == {datetime(2025, 1, 6)}
    assert rich_rows == 1


def test_iter_eod_date_chunks_yields_one_day_requests() -> None:
    chunks = list(_iter_eod_date_chunks("2024-01-01", "2026-06-20"))
    assert len(chunks) > 500
    assert all(start == end for start, end in chunks)
    assert chunks[0] == (date(2024, 1, 1), date(2024, 1, 1))
    assert chunks[-1][1] == date(2026, 6, 20)


def test_fetch_option_history_eod_chunks_requests(monkeypatch) -> None:
    calls: list[tuple] = []

    class FakeResult:
        def __init__(self, frame: pl.DataFrame):
            self.df = frame

    def fake_fetch_openbb(section, *, symbol, provider, **kwargs):
        assert section == "options_eod"
        assert provider == "thetadata"
        assert kwargs["include_greeks"] is True
        assert kwargs["require_bid_ask"] is False
        assert kwargs["min_ask"] == 0.0
        calls.append((kwargs["start_date"], kwargs["end_date"]))
        return FakeResult(
            pl.DataFrame(
                [
                    {
                        "underlying_symbol": symbol,
                        "contract_symbol": "AAPL250124P00230000",
                        "eod_date": kwargs["start_date"],
                        "expiration": "2025-01-24",
                        "strike": 230.0,
                        "option_type": "put",
                        "created": f"{kwargs['start_date']} 17:21:40-05:00",
                        "bid": 0.66,
                        "ask": 0.81,
                    }
                ]
            )
        )

    monkeypatch.setattr(
        "quant_warehouse.platforms.data_providers.thetadata.options.fetch_openbb",
        fake_fetch_openbb,
    )
    frame = fetch_option_history_eod(
        "AAPL",
        "2024-01-01",
        "2025-06-01",
        api_key="test-key",
        spec=ThetaDataDownloadSpec(),
    )
    assert not frame.is_empty()
    assert len(calls) >= 2
    for start, end in calls:
        assert start == end


def test_thetadata_download_apis_reject_contract_filters() -> None:
    with pytest.raises(ValueError, match="full-chain-only"):
        ThetaDataDownloadSpec(max_dte=90)

    with pytest.raises(ValueError, match="full-chain-only"):
        fetch_option_history_eod("AAPL", "2025-01-06", "2025-01-06", min_dte=30)

    with pytest.raises(ValueError, match="full-chain-only"):
        load_thetadata_option_snapshots("AAPL", ["2025-01-06"], strike_range=10)

    with pytest.raises(ValueError, match="full-chain-only"):
        download_option_snapshots_for_range("AAPL", "2025-01-06", "2025-01-06", max_dte=90)


def test_normalize_thetadata_option_chain_preserves_greeks_endpoint_fields() -> None:
    frame = normalize_thetadata_option_chain(
        pl.DataFrame(
            [
                {
                    "underlying_symbol": "GOOG",
                    "expiration": "2021-02-05",
                    "strike": 2075.0,
                    "option_type": "call",
                    "eod_date": "2021-02-03",
                    "bid": 10.0,
                    "ask": 11.0,
                    "underlying_price": 2070.07,
                    "implied_volatility": 0.42,
                    "delta": 0.51,
                    "gamma": 0.02,
                    "theta": -0.1,
                    "vega": 0.3,
                    "rho": 0.04,
                }
            ]
        )
    )

    row = frame.row(0, named=True)
    assert row["snapshot_date"] == datetime(2021, 2, 3)
    assert row["contract_symbol"] == "GOOG_call_20210205_2075"
    assert row["underlying_price"] == 2070.07
    assert row["iv"] == 0.42
    assert row["delta"] == 0.51


def test_load_thetadata_option_snapshots_uses_cache_without_fetch(monkeypatch) -> None:
    frame = _with_rich_option_fields(normalize_thetadata_option_chain(_raw_frame().head(1)))
    backend = _MemoryBackend(frame)

    def _fail_fetch(*args, **kwargs):
        raise AssertionError("fetch should not be called when cache is warm")

    monkeypatch.setattr(
        "quant_warehouse.platforms.data_providers.thetadata.options.fetch_option_history_eod",
        _fail_fetch,
    )
    monkeypatch.setattr(
        "quant_warehouse.platforms.data_providers.thetadata.options.open_backend",
        lambda *args, **kwargs: backend,
    )
    snapshots = load_thetadata_option_snapshots(
        "AAPL",
        ["2025-01-06"],
        api_key="test-key",
        use_cache=True,
    )
    assert len(snapshots) == 1


def test_download_option_snapshots_for_range_returns_cached_manifest(
    monkeypatch,
) -> None:
    frame = _with_rich_option_fields(normalize_thetadata_option_chain(_raw_frame().head(1)))
    backend = _MemoryBackend(frame)

    def _fail_fetch(*args, **kwargs):
        raise AssertionError("fetch should not be called when every business day is cached")

    monkeypatch.setattr(
        "quant_warehouse.platforms.data_providers.thetadata.options.fetch_option_history_eod",
        _fail_fetch,
    )
    monkeypatch.setattr(
        "quant_warehouse.platforms.data_providers.thetadata.options.open_backend",
        lambda *args, **kwargs: backend,
    )
    manifest = download_option_snapshots_for_range(
        "AAPL",
        "2025-01-06",
        "2025-01-06",
    )
    assert manifest["cached_only"] is True
    assert manifest["snapshot_days"] == 1
    assert manifest["contracts_total"] == 1
    assert manifest["cached_days"] == 1
    expected_library = provider_library(OPTIONS_THETADATA_EOD_LIBRARY, OPTIONS_THETADATA_PROVIDER)
    assert manifest["paths"] == [f"arctic://{expected_library}/AAPL"]


def test_download_option_snapshots_for_range_fetches_only_missing_business_ranges(
    monkeypatch,
) -> None:
    cached = _with_rich_option_fields(normalize_thetadata_option_chain(
        pl.DataFrame(
            [
                {
                    "symbol": "AAPL",
                    "expiration": "2025-01-24",
                    "strike": 230.0,
                    "right": "PUT",
                    "created": "2025-01-07 17:21:40-05:00",
                    "bid": 0.66,
                    "ask": 0.81,
                }
            ]
        )
    ))
    backend = _MemoryBackend(cached)
    calls: list[tuple[datetime, datetime]] = []

    def _fake_fetch(symbol, start_date, end_date, **kwargs):
        start = _ts(start_date)
        end = _ts(end_date)
        calls.append((start, end))
        return normalize_thetadata_option_chain(
            pl.DataFrame(
                [
                    {
                        "symbol": symbol,
                        "expiration": "2025-01-24",
                        "strike": 230.0,
                        "right": "PUT",
                        "created": f"{start.date().isoformat()} 17:21:40-05:00",
                        "bid": 0.66,
                        "ask": 0.81,
                    }
                ]
            )
        )

    monkeypatch.setattr(
        "quant_warehouse.platforms.data_providers.thetadata.options.fetch_option_history_eod",
        _fake_fetch,
    )
    monkeypatch.setattr(
        "quant_warehouse.platforms.data_providers.thetadata.options.open_backend",
        lambda *args, **kwargs: backend,
    )
    manifest = download_option_snapshots_for_range(
        "AAPL",
        "2025-01-06",
        "2025-01-08",
    )
    assert calls == [
        (datetime(2025, 1, 6), datetime(2025, 1, 6)),
        (datetime(2025, 1, 8), datetime(2025, 1, 8)),
    ]
    assert manifest["snapshot_days"] == 3
    assert manifest["contracts_total"] == 3
    assert manifest["cached_days"] == 1
    assert manifest["fetched_rows"] == 2
    assert len(backend.writes) == 2


def test_download_option_snapshots_for_range_uses_large_backfill_window(monkeypatch) -> None:
    backend = _MemoryBackend()
    calls: list[tuple[datetime, datetime]] = []

    def _fake_fetch(symbol, start_date, end_date, **kwargs):
        start = _ts(start_date)
        end = _ts(end_date)
        calls.append((start, end))
        rows = []
        for ts in _business_days(start, end):
            rows.append(
                {
                    "symbol": symbol,
                    "expiration": "2025-03-21",
                    "strike": 230.0,
                    "right": "PUT",
                    "created": f"{ts.date().isoformat()} 17:21:40-05:00",
                    "bid": 0.66,
                    "ask": 0.81,
                }
            )
        return normalize_thetadata_option_chain(
            pl.DataFrame(rows)
        )

    monkeypatch.setattr(
        "quant_warehouse.platforms.data_providers.thetadata.options.fetch_option_history_eod",
        _fake_fetch,
    )
    monkeypatch.setattr(
        "quant_warehouse.platforms.data_providers.thetadata.options.open_backend",
        lambda *args, **kwargs: backend,
    )

    manifest = download_option_snapshots_for_range(
        "AAPL",
        "2025-01-02",
        "2025-02-28",
        spec=ThetaDataDownloadSpec(backfill_window_days=180),
    )

    assert all(start == end for start, end in calls)
    assert [start for start, _end in calls] == [
        datetime.combine(day, datetime.min.time())
        for day in _business_days("2025-01-02", "2025-02-28")
    ]
    assert manifest["fetched_rows"] == len(_business_days("2025-01-02", "2025-02-28"))


def test_download_option_snapshots_for_range_never_requests_multiple_days(monkeypatch) -> None:
    backend = _MemoryBackend()
    calls: list[tuple[datetime, datetime]] = []

    def _fake_fetch(symbol, start_date, end_date, **kwargs):
        start = _ts(start_date)
        end = _ts(end_date)
        calls.append((start, end))
        rows = []
        for ts in _business_days(start, end):
            rows.append(
                {
                    "symbol": symbol,
                    "expiration": "2025-03-21",
                    "strike": 230.0,
                    "right": "PUT",
                    "created": f"{ts.date().isoformat()} 17:21:40-05:00",
                    "bid": 0.66,
                    "ask": 0.81,
                }
            )
        return normalize_thetadata_option_chain(
            pl.DataFrame(rows)
        )

    monkeypatch.setattr(
        "quant_warehouse.platforms.data_providers.thetadata.options.fetch_option_history_eod",
        _fake_fetch,
    )
    monkeypatch.setattr(
        "quant_warehouse.platforms.data_providers.thetadata.options.open_backend",
        lambda *args, **kwargs: backend,
    )

    manifest = download_option_snapshots_for_range(
        "AAPL",
        "2025-01-02",
        "2025-01-24",
        spec=ThetaDataDownloadSpec(backfill_window_days=180, fallback_window_days=7),
    )

    assert calls
    assert all(start == end for start, end in calls)
    assert len(calls) == len(_business_days("2025-01-02", "2025-01-24"))
    assert manifest["fetched_rows"] > 0


def test_download_option_snapshots_for_range_does_not_swallow_provider_errors(monkeypatch) -> None:
    backend = _MemoryBackend()

    def _fake_fetch(*args, **kwargs):
        raise RuntimeError("missing credential")

    monkeypatch.setattr(
        "quant_warehouse.platforms.data_providers.thetadata.options.fetch_option_history_eod",
        _fake_fetch,
    )
    monkeypatch.setattr(
        "quant_warehouse.platforms.data_providers.thetadata.options.open_backend",
        lambda *args, **kwargs: backend,
    )

    with pytest.raises(RuntimeError, match="missing credential"):
        download_option_snapshots_for_range(
            "AAPL",
            "2025-01-06",
            "2025-01-06",
            spec=ThetaDataDownloadSpec(backfill_window_days=180, fallback_window_days=7),
        )


def test_current_day_wildcard_falls_back_to_actual_expirations(monkeypatch) -> None:
    from quant_warehouse.platforms.data_providers.thetadata import options as options_module

    class _Client:
        def __init__(self) -> None:
            self.expiration_calls = []

        def option_list_expirations(self, symbol):
            assert symbol == "AAPL"
            return pl.DataFrame({"symbol": ["AAPL", "AAPL"], "expiration": ["2026-08-21", "2026-09-18"]})

        def option_history_greeks_eod(self, **kwargs):
            self.expiration_calls.append(kwargs)
            return pl.DataFrame(
                [{
                    "symbol": "AAPL",
                    "expiration": kwargs["expiration"].isoformat(),
                    "strike": 230.0,
                    "right": "PUT",
                    "created": "2026-08-13 17:21:40-05:00",
                    "bid": 0.66,
                    "ask": 0.81,
                }]
            )

    client = _Client()

    def _raise_current_day(*args, **kwargs):
        raise RuntimeError("Cannot fetch current-day data without specifying an expiration")

    monkeypatch.setattr(options_module, "fetch_openbb", _raise_current_day)
    monkeypatch.setattr(options_module, "_thetadata_client", lambda api_key: client)
    monkeypatch.setenv("THETADATA_API_KEY", "test-key")

    frame = fetch_option_history_eod("AAPL", "2026-08-13", "2026-08-13")

    assert frame.height == 2
    assert [call["expiration"].isoformat() for call in client.expiration_calls] == [
        "2026-08-21",
        "2026-09-18",
    ]
    assert all(call["start_date"] == call["end_date"] for call in client.expiration_calls)
