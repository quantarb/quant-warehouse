from __future__ import annotations

from pathlib import Path

import pandas as pd

from quant_warehouse.target_engineering.thetadata_loader import (
    _iter_eod_date_chunks,
    download_option_snapshots_for_range,
    fetch_option_history_eod,
    normalize_thetadata_option_chain,
    split_snapshots_by_date,
    write_snapshot_cache,
    read_snapshot_cache,
    load_thetadata_option_snapshots,
)


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


def test_normalize_thetadata_option_chain_builds_contract_symbol() -> None:
    frame = normalize_thetadata_option_chain(_raw_frame())
    assert "contract_symbol" in frame.columns
    assert frame["contract_symbol"].iloc[0] == "AAPL_put_20250124_230"
    assert "snapshot_date" in frame.columns
    assert "mid" in frame.columns
    assert frame["data_interval"].iloc[0] == "eod"


def test_normalize_thetadata_option_chain_drops_rows_without_bid_ask() -> None:
    raw = _raw_frame()
    raw.loc[0, "bid"] = 0.0
    frame = normalize_thetadata_option_chain(raw)
    assert len(frame) == 1


def test_split_snapshots_by_date_groups_rows() -> None:
    normalized = normalize_thetadata_option_chain(_raw_frame())
    snapshots = split_snapshots_by_date(normalized)
    assert len(snapshots) == 2


def test_snapshot_cache_roundtrip(tmp_path: Path) -> None:
    frame = normalize_thetadata_option_chain(_raw_frame().iloc[[0]])
    write_snapshot_cache("AAPL", "2025-01-06", frame, options_dir=tmp_path)
    loaded = read_snapshot_cache("AAPL", "2025-01-06", options_dir=tmp_path)
    assert loaded is not None
    assert len(loaded) == 1
    assert loaded["contract_symbol"].iloc[0] == "AAPL_put_20250124_230"


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
        "quant_warehouse.target_engineering.thetadata_loader.fetch_openbb",
        fake_fetch_openbb,
    )
    frame = fetch_option_history_eod("AAPL", "2024-01-01", "2025-06-01", api_key="test-key")
    assert not frame.empty
    assert len(calls) >= 2
    for start, end in calls:
        assert (end - start).days <= 364


def test_load_thetadata_option_snapshots_uses_cache_without_fetch(tmp_path: Path, monkeypatch) -> None:
    frame = normalize_thetadata_option_chain(_raw_frame().iloc[[0]])
    write_snapshot_cache("AAPL", "2025-01-06", frame, options_dir=tmp_path)

    def _fail_fetch(*args, **kwargs):
        raise AssertionError("fetch should not be called when cache is warm")

    monkeypatch.setattr(
        "quant_warehouse.target_engineering.thetadata_loader.fetch_option_history_eod",
        _fail_fetch,
    )
    snapshots = load_thetadata_option_snapshots(
        "AAPL",
        ["2025-01-06"],
        api_key="test-key",
        use_cache=True,
        options_dir=tmp_path,
    )
    assert len(snapshots) == 1


def test_download_option_snapshots_for_range_returns_cached_manifest(
    tmp_path: Path,
    monkeypatch,
) -> None:
    frame = normalize_thetadata_option_chain(_raw_frame().iloc[[0]])
    write_snapshot_cache("AAPL", "2025-01-06", frame, options_dir=tmp_path)

    def _fail_fetch(*args, **kwargs):
        raise AssertionError("fetch should not be called when every business day is cached")

    monkeypatch.setattr(
        "quant_warehouse.target_engineering.thetadata_loader.fetch_option_history_eod",
        _fail_fetch,
    )
    manifest = download_option_snapshots_for_range(
        "AAPL",
        "2025-01-06",
        "2025-01-06",
        options_dir=tmp_path,
    )
    assert manifest["cached_only"] is True
    assert manifest["snapshot_days"] == 1
    assert manifest["contracts_total"] == 1
    assert manifest["cached_days"] == 1


def test_download_option_snapshots_for_range_fetches_only_missing_business_ranges(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cached = normalize_thetadata_option_chain(
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
    )
    write_snapshot_cache("AAPL", "2025-01-07", cached, options_dir=tmp_path)
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
        "quant_warehouse.target_engineering.thetadata_loader.fetch_option_history_eod",
        _fake_fetch,
    )
    manifest = download_option_snapshots_for_range(
        "AAPL",
        "2025-01-06",
        "2025-01-08",
        options_dir=tmp_path,
    )
    assert calls == [
        (pd.Timestamp("2025-01-06"), pd.Timestamp("2025-01-06")),
        (pd.Timestamp("2025-01-08"), pd.Timestamp("2025-01-08")),
    ]
    assert manifest["snapshot_days"] == 3
    assert manifest["contracts_total"] == 3
    assert manifest["cached_days"] == 1
    assert manifest["fetched_rows"] == 2
