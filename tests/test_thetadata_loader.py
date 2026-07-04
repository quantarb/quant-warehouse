from __future__ import annotations

import pandas as pd
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


def _raw_frame() -> pd.DataFrame:
    return pd.DataFrame(
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


def _with_rich_option_fields(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for column, value in {
        "underlying_price": 230.0,
        "delta": -0.4,
        "gamma": 0.02,
        "theta": -0.03,
        "vega": 0.1,
        "rho": -0.01,
        "iv": 0.25,
    }.items():
        out[column] = value
    return out


def test_normalize_thetadata_option_chain_builds_contract_symbol() -> None:
    frame = normalize_thetadata_option_chain(_raw_frame())
    assert "contract_symbol" in frame.columns
    assert frame["contract_symbol"].iloc[0] == "AAPL_put_20250124_230"
    assert "snapshot_date" in frame.columns
    assert "mid" in frame.columns
    assert frame["data_interval"].iloc[0] == "eod"


def test_normalize_thetadata_option_chain_keeps_rows_without_bid_ask() -> None:
    raw = _raw_frame()
    raw.loc[0, "bid"] = 0.0
    frame = normalize_thetadata_option_chain(raw)
    assert len(frame) == 2
    assert frame["contract_symbol"].iloc[0] == "AAPL_put_20250124_230"


def test_split_snapshots_by_date_groups_rows() -> None:
    normalized = normalize_thetadata_option_chain(_raw_frame())
    snapshots = split_snapshots_by_date(normalized)
    assert len(snapshots) == 2


class _MemoryBackend:
    def __init__(self, initial: pd.DataFrame | None = None) -> None:
        self.frame = initial
        self.writes: list[tuple[str, str, pd.DataFrame, bool]] = []

    def read(self, library: str, symbol: str, **kwargs) -> pd.DataFrame | None:
        assert library in {
            provider_library(OPTIONS_THETADATA_EOD_LIBRARY, OPTIONS_THETADATA_PROVIDER),
        }
        if self.frame is None:
            return None
        out = self.frame.copy()
        date_range = kwargs.get("date_range")
        if date_range is not None and isinstance(out.index, pd.DatetimeIndex):
            start, end = date_range
            if start is not None:
                out = out.loc[out.index >= start]
            if end is not None:
                out = out.loc[out.index <= end]
        columns = kwargs.get("columns")
        if columns is not None:
            keep = [column for column in columns if column in out.columns]
            out = out.loc[:, keep]
        return out

    def write(
        self,
        library: str,
        symbol: str,
        df: pd.DataFrame,
        *,
        prune_previous_versions: bool = False,
    ) -> None:
        assert library == provider_library(OPTIONS_THETADATA_EOD_LIBRARY, OPTIONS_THETADATA_PROVIDER)
        self.frame = df.copy()
        self.writes.append((library, symbol, df.copy(), prune_previous_versions))


def test_arctic_option_chain_roundtrip() -> None:
    frame = normalize_thetadata_option_chain(_raw_frame().iloc[[0]])
    backend = _MemoryBackend()
    expected_library = provider_library(OPTIONS_THETADATA_EOD_LIBRARY, OPTIONS_THETADATA_PROVIDER)
    assert write_option_chain_arctic("AAPL", frame, backend=backend) == f"arctic://{expected_library}/AAPL"
    assert backend.writes[0][0] == expected_library
    assert backend.writes[0][3] is True
    loaded = read_option_chain_arctic("AAPL", start_date="2025-01-06", end_date="2025-01-06", backend=backend)
    assert len(loaded) == 1
    assert loaded["contract_symbol"].iloc[0] == "AAPL_put_20250124_230"


def test_read_option_chain_arctic_deduplicates_existing_snapshot_contract_rows() -> None:
    duplicate = pd.DataFrame(
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
    cached = normalize_thetadata_option_chain(duplicate)
    cached.index = pd.DatetimeIndex(
        [pd.Timestamp("2025-01-06"), pd.Timestamp("2025-01-06") + pd.Timedelta(nanoseconds=1)]
    )
    backend = _MemoryBackend(cached)

    loaded = read_option_chain_arctic("AAPL", start_date="2025-01-06", end_date="2025-01-06", backend=backend)

    assert len(loaded) == 1
    assert loaded["contract_symbol"].iloc[0] == "AAPL_put_20250124_230"
    assert loaded["bid"].iloc[0] == 0.70


def test_read_thetadata_eod_option_chain_enforces_contract_and_projects_columns() -> None:
    duplicate = pd.DataFrame(
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
    cached = _with_rich_option_fields(normalize_thetadata_option_chain(duplicate))
    cached.index = pd.DatetimeIndex(
        [pd.Timestamp("2025-01-06"), pd.Timestamp("2025-01-06") + pd.Timedelta(nanoseconds=1)]
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
    assert len(loaded) == 1
    assert loaded["contract_symbol"].iloc[0] == "AAPL_put_20250124_230"
    assert loaded["bid"].iloc[0] == 0.70


def test_read_thetadata_eod_option_chain_requires_rich_endpoint_columns_when_requested() -> None:
    quote_only = normalize_thetadata_option_chain(_raw_frame().iloc[[0]])
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
    duplicate = pd.DataFrame(
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
    cached = normalize_thetadata_option_chain(duplicate)
    cached.index = pd.DatetimeIndex(
        [pd.Timestamp("2025-01-06"), pd.Timestamp("2025-01-06") + pd.Timedelta(nanoseconds=1)]
    )
    backend = _MemoryBackend(cached)

    dry_run = deduplicate_option_chain_arctic(["AAPL"], backend=backend)
    assert dry_run.loc[0, "duplicate_rows"] == 1
    assert bool(dry_run.loc[0, "rewritten"]) is False
    assert len(backend.frame) == 2

    rewritten = deduplicate_option_chain_arctic(["AAPL"], backend=backend, dry_run=False)

    assert rewritten.loc[0, "duplicate_rows"] == 1
    assert bool(rewritten.loc[0, "rewritten"]) is True
    assert len(backend.frame) == 1
    assert backend.writes[-1][3] is True


def test_cached_date_summary_requires_rich_greeks_columns() -> None:
    quote_only = normalize_thetadata_option_chain(_raw_frame().iloc[[0]])
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
    assert rich_dates == {pd.Timestamp("2025-01-06")}
    assert rich_rows == 1


def test_iter_eod_date_chunks_splits_long_ranges() -> None:
    chunks = list(_iter_eod_date_chunks("2024-01-01", "2026-06-20"))
    assert len(chunks) >= 2
    assert chunks[0] == (pd.Timestamp("2024-01-01").date(), pd.Timestamp("2024-12-30").date())
    assert chunks[-1][1] == pd.Timestamp("2026-06-20").date()


def test_fetch_option_history_eod_chunks_requests(monkeypatch) -> None:
    calls: list[tuple] = []

    class FakeResult:
        def __init__(self, frame: pd.DataFrame):
            self.df = frame

    def fake_fetch_openbb(section, *, symbol, provider, **kwargs):
        assert section == "options_eod"
        assert provider == "thetadata"
        assert kwargs["include_greeks"] is True
        assert kwargs["require_bid_ask"] is False
        assert kwargs["min_ask"] == 0.0
        calls.append((kwargs["start_date"], kwargs["end_date"]))
        return FakeResult(
            pd.DataFrame(
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
    assert not frame.empty
    assert len(calls) >= 2
    for start, end in calls:
        assert (end - start).days <= 364


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
        pd.DataFrame(
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

    row = frame.iloc[0]
    assert row["snapshot_date"] == pd.Timestamp("2021-02-03")
    assert row["contract_symbol"] == "GOOG_call_20210205_2075"
    assert row["underlying_price"] == 2070.07
    assert row["iv"] == 0.42
    assert row["delta"] == 0.51


def test_load_thetadata_option_snapshots_uses_cache_without_fetch(monkeypatch) -> None:
    frame = _with_rich_option_fields(normalize_thetadata_option_chain(_raw_frame().iloc[[0]]))
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
    frame = _with_rich_option_fields(normalize_thetadata_option_chain(_raw_frame().iloc[[0]]))
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
        pd.DataFrame(
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
    calls: list[tuple[pd.Timestamp, pd.Timestamp]] = []

    def _fake_fetch(symbol, start_date, end_date, **kwargs):
        start = pd.Timestamp(start_date).normalize()
        end = pd.Timestamp(end_date).normalize()
        calls.append((start, end))
        return normalize_thetadata_option_chain(
            pd.DataFrame(
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
        (pd.Timestamp("2025-01-06"), pd.Timestamp("2025-01-06")),
        (pd.Timestamp("2025-01-08"), pd.Timestamp("2025-01-08")),
    ]
    assert manifest["snapshot_days"] == 3
    assert manifest["contracts_total"] == 3
    assert manifest["cached_days"] == 1
    assert manifest["fetched_rows"] == 2
    assert len(backend.writes) == 2


def test_download_option_snapshots_for_range_uses_large_backfill_window(monkeypatch) -> None:
    backend = _MemoryBackend()
    calls: list[tuple[pd.Timestamp, pd.Timestamp]] = []

    def _fake_fetch(symbol, start_date, end_date, **kwargs):
        start = pd.Timestamp(start_date).normalize()
        end = pd.Timestamp(end_date).normalize()
        calls.append((start, end))
        rows = []
        for ts in pd.date_range(start, end, freq="B"):
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
            pd.DataFrame(rows)
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

    assert calls == [(pd.Timestamp("2025-01-02"), pd.Timestamp("2025-02-28"))]
    assert manifest["fetched_rows"] == len(pd.date_range("2025-01-02", "2025-02-28", freq="B"))


def test_download_option_snapshots_for_range_falls_back_after_large_request_error(monkeypatch) -> None:
    backend = _MemoryBackend()
    calls: list[tuple[pd.Timestamp, pd.Timestamp]] = []

    def _fake_fetch(symbol, start_date, end_date, **kwargs):
        start = pd.Timestamp(start_date).normalize()
        end = pd.Timestamp(end_date).normalize()
        calls.append((start, end))
        if (end - start).days > 10:
            raise RuntimeError("range too large")
        rows = []
        for ts in pd.date_range(start, end, freq="B"):
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
            pd.DataFrame(rows)
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

    assert calls[0] == (pd.Timestamp("2025-01-02"), pd.Timestamp("2025-01-24"))
    assert len(calls) > 1
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
