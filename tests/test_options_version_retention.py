from datetime import datetime
from types import SimpleNamespace

import polars as pl
import pytest

from quant_warehouse.platforms.data_providers.thetadata import options as opt
from quant_warehouse.warehouse.backend import ArcticBackend
from quant_warehouse.warehouse.storage import provider_library
from quant_warehouse.migrate import backfill_thetadata_options as migration


def frame(bid=1.0, strike=100.0):
    return opt.normalize_thetadata_option_chain(pl.DataFrame({
        'symbol': ['AAPL'], 'expiration': ['2026-11-20'], 'strike': [strike],
        'right': ['CALL'], 'created': ['2026-10-09 16:00:00-04:00'],
        'bid': [bid], 'ask': [bid + 1.0]}))


def test_real_temporary_arctic_retains_versions_and_merge(tmp_path):
    backend = ArcticBackend('lmdb://' + str(tmp_path / 'arctic'))
    opt.write_option_chain_arctic('AAPL', frame(), backend=backend)
    library = backend._library(provider_library(opt.OPTIONS_THETADATA_EOD_LIBRARY, opt.OPTIONS_THETADATA_PROVIDER))
    versions_before = library.list_versions('AAPL')
    first = next(iter(versions_before)).version
    opt.write_option_chain_arctic('AAPL', pl.concat([frame(3.0), frame(2.0, 110.0)]), backend=backend)
    assert len(library.list_versions('AAPL')) == 2
    assert library.read('AAPL', as_of=first).data['bid'].tolist() == [1.0]
    latest = opt.read_option_chain_arctic('AAPL', backend=backend)
    assert latest.height == 2
    assert sorted(latest['bid'].to_list()) == [2.0, 3.0]


@pytest.mark.parametrize('prune', [False, True])
def test_downloader_propagates_explicit_flag(monkeypatch, prune):
    seen = []
    monkeypatch.setattr(opt, 'open_backend', lambda _: object())
    monkeypatch.setattr(opt, 'option_chain_cached_date_summary', lambda *a, **k: (set(), 0))
    monkeypatch.setattr(opt, '_fetch_option_history_with_window_fallback', lambda *a, **k: frame())
    monkeypatch.setattr(opt, 'write_option_chain_arctic', lambda *a, **k: seen.append(k['prune_previous_versions']) or 'arctic://test')
    opt.download_option_snapshots_for_range('AAPL', '2026-10-09', '2026-10-09', prune_previous_versions=prune)
    assert seen == [prune]


def test_backfill_default_preserves_versions(monkeypatch):
    seen = []
    monkeypatch.setattr(migration, 'option_chain_cached_date_summary_bulk', lambda *a, **k: {})
    monkeypatch.setattr(migration, '_upsert_options_catalog_state', lambda *a, **k: None)
    def download(symbol, start, end, **kwargs):
        seen.append(kwargs['prune_previous_versions'])
        return {'snapshot_days': 1, 'contracts_total': 1, 'fetched_rows': 1,
                'start_date': '2026-10-09', 'end_date': '2026-10-09'}
    monkeypatch.setattr(migration, 'download_option_snapshots_for_range', download)
    migration.backfill_thetadata_options(warehouse=SimpleNamespace(), symbols=['AAPL'],
                                        start_date='2026-10-09', end_date='2026-10-09')
    assert seen and all(flag is False for flag in seen)
