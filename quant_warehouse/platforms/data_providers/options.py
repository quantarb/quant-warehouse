from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd


OptionExitPriceSource = Literal[
    "contract_quote",
    "last_contract_quote",
    "expiration_intrinsic",
]


@dataclass(frozen=True)
class OptionQuote:
    snapshot_date: pd.Timestamp
    bid: float
    ask: float
    mid: float


@dataclass(frozen=True)
class OptionSettlement:
    snapshot_date: pd.Timestamp
    bid: float
    ask: float
    mid: float
    price_source: OptionExitPriceSource
