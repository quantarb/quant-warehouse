from __future__ import annotations

from typing import Sequence

from quant_warehouse.catalog.store import CatalogStore, SymbolProfile
from quant_warehouse.config import WarehouseConfig
from quant_warehouse.ingest.openbb_fetch import fetch_openbb
from quant_warehouse.ingest.providers import validate_price_provider


class ProfileStore:
    def __init__(
        self,
        config: WarehouseConfig | None = None,
        *,
        catalog: CatalogStore | None = None,
    ) -> None:
        self.config = config or WarehouseConfig.from_env()
        self.config.ensure_dirs()
        self.catalog = catalog or CatalogStore(self.config.catalog_path)

    def refresh(
        self,
        symbol: str,
        *,
        provider: str,
    ) -> dict[str, object]:
        symbol = symbol.strip().upper()
        provider = validate_price_provider(provider)
        result = fetch_openbb("profile", symbol=symbol, provider=provider)
        record = dict(result.records[0]) if result.records else {}
        if not record and not result.df.empty:
            record = result.df.iloc[0].to_dict()
        self.catalog.upsert_profile(
            symbol=symbol,
            provider=provider,
            source_provider=result.provider_used,
            payload=record,
        )
        return {
            "symbol": symbol,
            "provider_requested": result.provider_requested,
            "source_provider": result.provider_used,
            "fields_populated": len([key for key, value in record.items() if value is not None]),
        }

    def read(self, symbol: str, *, provider: str) -> SymbolProfile | None:
        return self.catalog.get_profile(symbol=symbol, provider=validate_price_provider(provider))

    def list(self, symbol: str) -> list[SymbolProfile]:
        return self.catalog.list_profiles(symbol.strip().upper())

    def refresh_many(
        self,
        symbols: Sequence[str],
        *,
        providers: Sequence[str],
    ) -> list[dict[str, object]]:
        stats: list[dict[str, object]] = []
        for symbol in symbols:
            symbol = str(symbol).strip().upper()
            if not symbol:
                continue
            for provider in providers:
                provider = validate_price_provider(provider)
                try:
                    stats.append(self.refresh(symbol, provider=provider))
                except Exception as exc:
                    stats.append(
                        {
                            "symbol": symbol,
                            "provider_requested": provider,
                            "error": str(exc),
                        }
                    )
        return stats