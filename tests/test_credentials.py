import os
from pathlib import Path

from quant_warehouse.ingest import credentials


def test_resolve_fmp_api_key_from_optimal_trader_dotenv(tmp_path: Path, monkeypatch):
    ot_home = tmp_path / "optimal_trader"
    ot_home.mkdir()
    (ot_home / ".env").write_text('FMP_API_KEY="test-key-from-ot"\n', encoding="utf-8")

    monkeypatch.delenv("FMP_API_KEY", raising=False)
    monkeypatch.setenv("OPTIMAL_TRADER_HOME", str(ot_home))
    credentials.resolve_fmp_api_key.cache_clear()

    assert credentials.resolve_fmp_api_key() == "test-key-from-ot"
    assert os.environ["FMP_API_KEY"] == "test-key-from-ot"


def test_existing_env_takes_precedence_over_dotenv(tmp_path: Path, monkeypatch):
    ot_home = tmp_path / "optimal_trader"
    ot_home.mkdir()
    (ot_home / ".env").write_text("FMP_API_KEY=from-file\n", encoding="utf-8")

    monkeypatch.setenv("FMP_API_KEY", "from-env")
    monkeypatch.setenv("OPTIMAL_TRADER_HOME", str(ot_home))
    credentials.resolve_fmp_api_key.cache_clear()

    assert credentials.resolve_fmp_api_key() == "from-env"