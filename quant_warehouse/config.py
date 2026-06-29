from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path


def _expand(path: str) -> Path:
    return Path(path).expanduser().resolve()


@dataclass(frozen=True)
class WarehouseConfig:
    home: Path
    arctic_uri: str
    catalog_path: Path

    @classmethod
    def from_env(cls) -> WarehouseConfig:
        from quant_warehouse.ingest.credentials import load_shared_env

        load_shared_env()
        home = _expand(os.environ.get("QW_HOME", "~/.quant-warehouse"))
        arctic_uri = os.environ.get("QW_ARCTIC_URI", f"lmdb://{home}/arctic")
        catalog_path = _expand(os.environ.get("QW_CATALOG_PATH", str(home / "catalog.sqlite")))
        return cls(home=home, arctic_uri=arctic_uri, catalog_path=catalog_path)

    def provider_arctic_uri(self, provider: str) -> str:
        provider_key = re.sub(r"[^A-Z0-9]+", "_", str(provider or "").strip().upper()).strip("_")
        if not provider_key:
            raise ValueError("provider is required")
        override = os.environ.get(f"QW_ARCTIC_URI_{provider_key}")
        if override:
            return override

        provider_path = re.sub(r"[^a-z0-9]+", "_", str(provider or "").strip().lower()).strip("_")
        if self.arctic_uri.startswith("lmdb://"):
            base_path = Path(self.arctic_uri.removeprefix("lmdb://")).expanduser().resolve()
            return f"lmdb://{base_path / 'providers' / provider_path}"
        return self.arctic_uri

    @property
    def staging_dir(self) -> Path:
        return self.home / "staging"

    @property
    def prices_dir(self) -> Path:
        return self.home / "prices"

    @property
    def fundamentals_dir(self) -> Path:
        return self.home / "fundamentals"

    @property
    def features_dir(self) -> Path:
        return self.home / "features"

    def ensure_dirs(self) -> None:
        for path in (
            self.home,
            self.staging_dir,
            self.prices_dir,
            self.fundamentals_dir,
            self.features_dir,
            Path(self.arctic_uri.removeprefix("lmdb://")),
        ):
            path.mkdir(parents=True, exist_ok=True)
