from __future__ import annotations

import re

import pandas as pd

from quant_warehouse.warehouse.backend import StorageBackend


def normalize_storage_provider(provider: str) -> str:
    text = str(provider or "").strip().lower()
    if not text:
        raise ValueError("provider is required for provider-scoped storage")
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_")


def normalize_provider_dataset(base_library: str, provider: str) -> str:
    """Dataset token for provider-scoped physical libraries.

    Logical library names predate provider-isolated storage, so normalize them
    into route-family-style dataset names before adding the provider prefix.
    For example, prices is equity_prices and options_thetadata_eod is
    derivatives_options_eod.
    """

    base = re.sub(r"[^a-z0-9]+", "_", str(base_library).strip().lower()).strip("_")
    if not base:
        raise ValueError("base_library is required")
    provider_name = normalize_storage_provider(provider)
    parts = [part for part in base.split("_") if part and part != provider_name]
    dataset = "_".join(parts)
    if dataset == "prices":
        return "equity_prices"
    if dataset.startswith("fundamental_"):
        return f"equity_{dataset}"
    if dataset == "options_eod":
        return "derivatives_options_eod"
    return dataset


def provider_library(base_library: str, provider: str) -> str:
    """Physical Arctic library for provider-owned warehouse data."""

    return f"{normalize_storage_provider(provider)}_{normalize_provider_dataset(base_library, provider)}"


def provider_from_library(library: str) -> str | None:
    """Infer the provider prefix from a provider-scoped physical library name."""

    text = str(library or "").strip().lower()
    if not text:
        return None
    known_prefixes = (
        "federal_reserve",
        "government_us",
        "tradingeconomics",
        "congress_gov",
        "yfinance",
        "thetadata",
        "intrinio",
        "tiingo",
        "econdb",
        "fred",
        "fmp",
        "sec",
        "bls",
        "oecd",
        "imf",
        "cftc",
    )
    for prefix in known_prefixes:
        if text == prefix or text.startswith(f"{prefix}_"):
            return prefix
    return None


def read_provider_frame(
    backend: StorageBackend,
    *,
    base_library: str,
    provider: str,
    symbol: str,
    fallback_legacy: bool = True,
) -> pd.DataFrame | None:
    """Read from provider-scoped storage, optionally falling back to the legacy shared library."""

    frame = backend.read(provider_library(base_library, provider), symbol)
    if frame is not None and not frame.empty:
        return frame
    if fallback_legacy:
        return backend.read(base_library, symbol)
    return frame
