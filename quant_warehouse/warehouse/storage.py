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
    from quant_warehouse.platforms.data_providers.registry import PROVIDER_PREFIXES

    for prefix in PROVIDER_PREFIXES:
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
    start_date: pd.Timestamp | None = None,
    end_date: pd.Timestamp | None = None,
    columns: list[str] | None = None,
) -> pd.DataFrame | None:
    """Read from provider-scoped storage, optionally falling back to the legacy shared library."""

    date_range = (start_date, end_date) if start_date is not None or end_date is not None else None
    frame = _read_backend(
        backend,
        provider_library(base_library, provider),
        symbol,
        date_range=date_range,
        columns=columns,
    )
    if frame is not None and not frame.empty:
        return frame
    if fallback_legacy:
        return _read_backend(
            backend,
            base_library,
            symbol,
            date_range=date_range,
            columns=columns,
        )
    return frame


def _read_backend(
    backend: StorageBackend,
    library: str,
    symbol: str,
    *,
    date_range: tuple[pd.Timestamp | None, pd.Timestamp | None] | None = None,
    columns: list[str] | None = None,
) -> pd.DataFrame | None:
    try:
        return backend.read(library, symbol, date_range=date_range, columns=columns)
    except TypeError:
        frame = backend.read(library, symbol)
        if frame is None or frame.empty:
            return frame
        out = frame.copy()
        if date_range is not None:
            start, end = date_range
            if isinstance(out.index, pd.DatetimeIndex):
                if start is not None:
                    out = out.loc[out.index >= start]
                if end is not None:
                    out = out.loc[out.index <= end]
        if columns is not None:
            keep = [column for column in columns if column in out.columns]
            out = out.loc[:, keep]
        return out
