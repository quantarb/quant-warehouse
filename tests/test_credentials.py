import os
from pathlib import Path

from quant_warehouse.ingest import credentials


def test_resolve_fmp_api_key_from_quant_warehouse_dotenv(tmp_path: Path, monkeypatch):
    qw_home = tmp_path / "quant-warehouse"
    qw_home.mkdir()
    (qw_home / ".env").write_text('FMP_API_KEY="test-key-from-warehouse"\n', encoding="utf-8")

    monkeypatch.delenv("FMP_API_KEY", raising=False)
    monkeypatch.setenv("QW_HOME", str(qw_home))
    credentials.resolve_fmp_api_key.cache_clear()

    assert credentials.resolve_fmp_api_key() == "test-key-from-warehouse"
    assert os.environ["FMP_API_KEY"] == "test-key-from-warehouse"


def test_existing_env_takes_precedence_over_dotenv(tmp_path: Path, monkeypatch):
    qw_home = tmp_path / "quant-warehouse"
    qw_home.mkdir()
    (qw_home / ".env").write_text("FMP_API_KEY=from-file\n", encoding="utf-8")

    monkeypatch.setenv("FMP_API_KEY", "from-env")
    monkeypatch.setenv("QW_HOME", str(qw_home))
    credentials.resolve_fmp_api_key.cache_clear()

    assert credentials.resolve_fmp_api_key() == "from-env"
