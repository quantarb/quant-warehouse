from pathlib import Path
from datetime import datetime

import polars as pl
import pytest

from quant_warehouse.config import WarehouseConfig
from quant_warehouse.warehouse.backend import ArcticBackend, ProviderRoutingBackend, open_backend
from quant_warehouse.warehouse.storage import provider_from_library, provider_library


def _config(tmp_path: Path) -> WarehouseConfig:
    return WarehouseConfig(
        home=tmp_path / "home",
        arctic_uri=f"lmdb://{tmp_path / 'arctic'}",
        catalog_path=tmp_path / "catalog.sqlite",
    )


def _sample_frame() -> pl.DataFrame:
    return pl.DataFrame({"date": [datetime(2024, 1, 1), datetime(2024, 1, 2)], "close": [100.0, 101.0], "volume": [1000, 1100]})


def test_arctic_backend_roundtrip(tmp_path: Path):
    backend = ArcticBackend(_config(tmp_path).arctic_uri)
    frame = _sample_frame()
    backend.write("prices", "AAPL__yfinance", frame)
    out = backend.read("prices", "AAPL__yfinance")
    assert out is not None
    assert len(out) == 2
    assert out.filter(pl.col("date") == datetime(2024, 1, 2))["close"][0] == 101.0


def test_partially_dated_panel_preserves_undated_rows_and_supports_date_reads(tmp_path):
    backend = ArcticBackend(_config(tmp_path).arctic_uri)
    frame = pl.DataFrame({'date': [None, datetime(2024, 1, 2), datetime(2024, 2, 1)],
                          'amount': [10., 20., 30.]})
    backend.write('events', 'ABC__fmp', frame)
    out = backend.read('events', 'ABC__fmp')
    assert out.height == 3 and out['date'].null_count() == 1
    selected = backend.read('events', 'ABC__fmp',
                            date_range=(datetime(2024, 1, 1), datetime(2024, 1, 31)),
                            columns=['amount'])
    assert selected.to_dicts() == [{'amount': 20.}]


@pytest.mark.parametrize('date_dtype', [pl.Null, pl.String, pl.Datetime('ns')])
def test_all_missing_filing_dates_are_stored_without_object_inference(tmp_path, monkeypatch, date_dtype):
    import arcticdb.version_store._normalization as normalization
    original = normalization.get_sample_from_non_empty_arr
    def sample(array, name):
        assert name != 'filing_date', 'Missing filing dates must not use object inference'
        return original(array, name)
    monkeypatch.setattr(normalization, 'get_sample_from_non_empty_arr', sample)
    backend = ArcticBackend(_config(tmp_path).arctic_uri)
    frame = _sample_frame().with_columns(pl.Series('filing_date', [None, None], dtype=date_dtype))
    backend.write('statements', 'AAPL__fmp', frame)
    out = backend.read('statements', 'AAPL__fmp')
    assert out.height == frame.height
    assert out['filing_date'].to_list() == [None, None]
    assert out['close'].to_list() == frame['close'].to_list()
    assert out['date'].to_list() == frame['date'].to_list()
    # A later refresh can populate the same date field without losing rows.
    updated = frame.with_columns(pl.Series('filing_date', [None, datetime(2024, 1, 3)]))
    backend.write('statements', 'AAPL__fmp', updated)
    assert backend.read('statements', 'AAPL__fmp')['filing_date'].to_list() == [None, datetime(2024, 1, 3)]


def test_all_missing_payload_columns_bypass_object_inference_and_restore_schema(tmp_path, monkeypatch):
    import arcticdb.version_store._normalization as normalization
    original = normalization.get_sample_from_non_empty_arr
    def sample(array, name):
        if name in {'filing_url', 'action'}:
            assert any(value is not None for value in array), 'Empty payloads must not use object inference'
        return original(array, name)
    monkeypatch.setattr(normalization, 'get_sample_from_non_empty_arr', sample)
    backend = ArcticBackend(_config(tmp_path).arctic_uri)
    frame = _sample_frame().with_columns(
        pl.lit(None).alias('filing_url'),
        pl.Series('action', [None, None], dtype=pl.String),
    )
    backend.write('events', 'HON__fmp', frame)
    out = backend.read('events', 'HON__fmp')
    assert out.columns == frame.columns
    assert out.select('filing_url', 'action').null_count().row(0) == (2, 2)
    selected = backend.read('events', 'HON__fmp', columns=['action', 'close'])
    assert selected.columns == ['action', 'close']
    assert selected['action'].null_count() == 2
    populated = frame.with_columns(pl.Series('action', [None, 'Buy']))
    backend.write('events', 'HON__fmp', populated)
    assert backend.read('events', 'HON__fmp')['action'].to_list() == [None, 'Buy']


def test_open_backend_uses_arctic(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("QW_HOME", str(tmp_path / "home"))
    backend = open_backend(WarehouseConfig.from_env())
    assert backend.kind == "arctic"
    assert isinstance(backend, ProviderRoutingBackend)


def test_provider_library_names_are_provider_scoped_without_duplicate_provider():
    assert provider_library("prices", "yfinance") == "yfinance_equity_prices"
    assert provider_library("prices_unadjusted", "fmp") == "fmp_equity_prices_unadjusted"
    assert provider_library("etf_prices", "fmp") == "fmp_etf_prices"
    assert provider_library("fund_prices", "yfinance") == "yfinance_fund_prices"
    assert provider_library("fundamental_income", "fmp") == "fmp_equity_fundamental_income"
    assert provider_library("options_thetadata_eod", "thetadata") == "thetadata_derivatives_options_eod"


def test_provider_from_library_understands_route_family_names():
    assert provider_from_library("fmp_equity_prices") == "fmp"
    assert provider_from_library("thetadata_derivatives_options_eod") == "thetadata"
    assert provider_from_library("federal_reserve_macro_treasury") == "federal_reserve"
    assert provider_from_library("prices") is None


def test_provider_arctic_uri_defaults_to_separate_lmdb_roots(tmp_path: Path):
    config = _config(tmp_path)
    assert config.provider_arctic_uri("fmp") == f"lmdb://{tmp_path / 'arctic' / 'providers' / 'fmp'}"
    assert config.provider_arctic_uri("thetadata") == (
        f"lmdb://{tmp_path / 'arctic' / 'providers' / 'thetadata'}"
    )


def test_provider_routing_backend_separates_provider_roots(tmp_path: Path):
    config = _config(tmp_path)
    backend = ProviderRoutingBackend(config)
    frame = _sample_frame()

    backend.write("fmp_equity_prices", "AAPL__fmp", frame)
    backend.write("yfinance_equity_prices", "AAPL__yfinance", frame)

    fmp_backend = ArcticBackend(config.provider_arctic_uri("fmp"))
    yfinance_backend = ArcticBackend(config.provider_arctic_uri("yfinance"))
    default_backend = ArcticBackend(config.arctic_uri)

    assert fmp_backend.read("fmp_equity_prices", "AAPL__fmp") is not None
    assert yfinance_backend.read("yfinance_equity_prices", "AAPL__yfinance") is not None
    assert default_backend.read("fmp_equity_prices", "AAPL__fmp") is None


def test_unnamed_legacy_time_index_restores_observation_dates(tmp_path):
    backend = ArcticBackend(_config(tmp_path).arctic_uri)
    stored = _sample_frame().to_pandas().set_index('date')
    stored.index.name = None
    backend._library('dividends').write('TEST__fmp', stored)
    out = backend.read('dividends', 'TEST__fmp')
    assert out['date'].to_list() == _sample_frame()['date'].to_list()
    assert out['close'].to_list() == [100., 101.]


@pytest.mark.parametrize('dated', [True, False])
def test_column_counts_include_nan_null_and_omitted_columns(tmp_path, dated):
    backend = ArcticBackend(_config(tmp_path).arctic_uri)
    frame = pl.DataFrame({'date': [datetime(2024, 1, 1), datetime(2024, 1, 2)] if dated else [None, datetime(2024, 1, 2)],
        'value': [1., float('nan')], 'nan_only': [float('nan'), float('nan')],
        'null_only': [None, None], 'text': ['hello', None]})
    backend.write('events', 'TEST', frame)
    counts = backend.column_non_null_counts('events', 'TEST')
    assert counts['value'] == 1
    assert counts['nan_only'] == counts['null_only'] == 0
    assert counts['text'] == 1
    assert counts['date'] == (2 if dated else 1)


def test_empty_payload_detection_preserves_meaningful_values_and_keys():
    from quant_warehouse.warehouse.backend import empty_payload_columns
    frame = pl.DataFrame({
        'null': [None, None], 'nan': [float('nan'), None], 'blank': ['', None],
        'zero': [0, 0], 'false': [False, False], 'infinity': [float('inf'), None],
        'space': [' ', ''], 'partial': ['', 'real'], 'mixed_float': [float('nan'), 1.],
        'date': [None, None], 'symbol': [None, None], 'contract_symbol': ['', ''],
        'custom_key': [None, None], 'nested': [[], []],
    })
    assert empty_payload_columns(frame, key_columns=('custom_key',)) == ['null', 'nan', 'blank']
    assert empty_payload_columns(frame.head(0)) == []


@pytest.mark.parametrize('dated', [True, False])
def test_clean_schema_and_explicit_selection_preserve_compatibility(tmp_path, dated):
    backend = ArcticBackend(_config(tmp_path).arctic_uri)
    frame = pl.DataFrame({
        'date': [datetime(2024, 1, 1), datetime(2024, 1, 2)] if dated else [None, datetime(2024, 1, 2)],
        'value': [0., float('inf')], 'null': [None, None],
        'nan': [float('nan'), None], 'blank': ['', None], 'space': [' ', ' '],
    })
    backend.write('events', 'TEST', frame, prune_previous_versions=False)
    lib = backend._library('events')
    assert not {'null', 'nan', 'blank'} & {c.name for c in lib.get_description('TEST').columns}
    metadata = lib.read_metadata('TEST').metadata
    assert metadata['quant_warehouse_empty_payload_details']['nan']['nan_count'] == 1
    assert metadata['quant_warehouse_empty_payload_details']['blank']['empty_string_count'] == 1
    restored = backend.read('events', 'TEST')
    assert restored.columns == frame.columns
    assert restored.schema['nan'] == pl.Float64 and restored.schema['blank'] == pl.String
    assert backend.read('events', 'TEST', include_empty_columns=False).columns == ['date', 'value', 'space']
    selected = backend.read('events', 'TEST', columns=['blank', 'nan', 'value'], include_empty_columns=False)
    assert selected.columns == ['blank', 'nan', 'value'] and selected.height == 2
    assert selected['blank'].null_count() == 2
    filtered = backend.read('events', 'TEST', date_range=(datetime(2024, 1, 2), datetime(2024, 1, 2)),
                            columns=['blank'], include_empty_columns=False)
    assert filtered.height == 1 and filtered.columns == ['blank']


def test_all_empty_payload_with_missing_keys_preserves_rows(tmp_path):
    backend = ArcticBackend(_config(tmp_path).arctic_uri)
    frame = pl.DataFrame({'date': [None, None], 'symbol': [None, None],
                          'custom_key': [None, None], 'payload': ['', None]})
    backend.write('events', 'TEST', frame, key_columns=('custom_key',), prune_previous_versions=False)
    out = backend.read('events', 'TEST', include_empty_columns=False)
    assert out.height == 2 and out.columns == ['date', 'symbol', 'custom_key']


def test_clean_reader_only_hides_metadata_known_empty_fields(tmp_path):
    backend = ArcticBackend(_config(tmp_path).arctic_uri)
    backend._library('events').write('LEGACY', _sample_frame().with_columns(pl.lit(float('nan')).alias('nan')).to_pandas())
    assert 'nan' in backend.read('events', 'LEGACY', include_empty_columns=False).columns


def test_clean_schema_routes_through_provider_backend(tmp_path):
    backend = ProviderRoutingBackend(_config(tmp_path))
    backend.write('fmp_equity_prices', 'TEST', _sample_frame().with_columns(pl.lit('').alias('blank')),
                  prune_previous_versions=False)
    assert 'blank' not in backend.read('fmp_equity_prices', 'TEST', include_empty_columns=False).columns
    assert backend.read('fmp_equity_prices', 'TEST', columns=['blank'], include_empty_columns=False).height == 2


def test_cleanup_writes_can_preserve_previous_versions(tmp_path):
    backend = ArcticBackend(_config(tmp_path).arctic_uri)
    lib = backend._library('events')
    original = _sample_frame().with_columns(pl.lit('').alias('blank')).to_pandas()
    lib.write('TEST', original, metadata={'provenance': 'original'}, prune_previous_versions=False)
    backend.write('events', 'TEST', _sample_frame().with_columns(pl.lit('').alias('blank')),
                  prune_previous_versions=False)
    assert lib.read_metadata('TEST').metadata['provenance'] == 'original'
    assert len(lib.list_versions('TEST')) == 2
    assert lib.read('TEST', as_of=0).metadata == {'provenance': 'original'}
    assert lib.read('TEST', as_of=0).data['blank'].to_list() == ['', '']


def test_clean_empty_payload_can_later_receive_values(tmp_path):
    backend = ArcticBackend(_config(tmp_path).arctic_uri)
    empty = _sample_frame().with_columns(pl.Series('text', ['', None]), pl.Series('number', [float('nan'), None]))
    backend.write('events', 'TEST', empty, prune_previous_versions=False)
    updated = empty.with_columns(pl.Series('text', ['', 'value']), pl.Series('number', [float('nan'), 0.]))
    backend.write('events', 'TEST', updated, prune_previous_versions=False)
    out = backend.read('events', 'TEST', include_empty_columns=False)
    assert out['text'].to_list() == ['', 'value'] and out['number'][1] == 0.
    assert backend._library('events').read_metadata('TEST').metadata['quant_warehouse_empty_columns'] == []


def test_empty_frame_and_nullable_temporal_payload_detection():
    from quant_warehouse.warehouse.backend import empty_payload_columns
    frame = pl.DataFrame({'date': [None, None], 'optional_timestamp': [None, None]},
                         schema={'date': pl.Datetime('ns'), 'optional_timestamp': pl.Datetime('ns')})
    assert empty_payload_columns(frame) == ['optional_timestamp']
    assert empty_payload_columns(pl.DataFrame(schema={'empty': pl.String, 'number': pl.Float64})) == []


def test_empty_identity_and_availability_fields_are_protected():
    from quant_warehouse.warehouse.backend import empty_payload_columns
    frame = pl.DataFrame({"id": [None], "cik": [None], "created_at": [None],
                          "snapshot_date": [None], "period": [None], "url": [""]})
    assert empty_payload_columns(frame) == ["url"]


def test_dictionary_encoded_empty_strings_follow_same_policy(tmp_path):
    from quant_warehouse.warehouse.backend import empty_payload_columns
    frame = _sample_frame().with_columns(
        pl.Series("category_empty", ["", None], dtype=pl.Categorical),
        pl.Series("enum_empty", ["", None], dtype=pl.Enum(["", "real"])),
    )
    assert empty_payload_columns(frame) == ["category_empty", "enum_empty"]
    backend = ArcticBackend(_config(tmp_path).arctic_uri)
    backend.write("events", "TEST", frame, prune_previous_versions=False)
    restored = backend.read("events", "TEST", columns=["category_empty", "enum_empty"])
    assert restored.height == 2
    assert restored["category_empty"].null_count() == restored["enum_empty"].null_count() == 2
    assert "category_empty" not in backend.read("events", "TEST", include_empty_columns=False).columns
    assert empty_payload_columns(pl.DataFrame({"text": pl.Series(["", "real"], dtype=pl.Categorical)})) == []
