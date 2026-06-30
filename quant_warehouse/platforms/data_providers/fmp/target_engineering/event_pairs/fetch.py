from __future__ import annotations

from typing import Sequence

import pandas as pd

from quant_warehouse.platforms.data_providers.fmp.target_engineering.event_pairs.event_pair_taxonomy import (
    EVENT_PAIR_TAXONOMY,
)

_SUPPORTED_FAMILIES = tuple(EVENT_PAIR_TAXONOMY)
_DIRECT_FETCH_ERROR = (
    "Direct FMP event-pair fetches are disabled. Refresh the corresponding "
    "warehouse source sections and build event pairs from provider-local historical data."
)


def fetch_fmp_event_pairs(
    symbol: str,
    *,
    event_families: Sequence[str] = _SUPPORTED_FAMILIES,
    start_date: str | None = None,
    end_date: str | None = None,
    page: int = 0,
    limit: int = 100,
) -> pd.DataFrame:
    """Raise until FMP event-pair downloads are routed through warehouse sections."""

    raise RuntimeError(_DIRECT_FETCH_ERROR)


def fetch_fmp_event_pair_family(
    symbol: str,
    *,
    event_family: str,
    start_date: str | None = None,
    end_date: str | None = None,
    page: int = 0,
    limit: int = 100,
) -> pd.DataFrame:
    """Raise until FMP event-pair downloads are routed through warehouse sections."""

    raise RuntimeError(_DIRECT_FETCH_ERROR)
