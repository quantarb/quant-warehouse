from __future__ import annotations

import polars as pl

from dataclasses import dataclass
from datetime import datetime
from typing import Literal



OptionExitPriceSource = Literal[
    "contract_quote",
    "last_contract_quote",
    "expiration_intrinsic",
]


@dataclass(frozen=True)
class OptionQuote:
    snapshot_date: datetime
    bid: float
    ask: float
    mid: float


@dataclass(frozen=True)
class OptionSettlement:
    snapshot_date: datetime
    bid: float
    ask: float
    mid: float
    price_source: OptionExitPriceSource
