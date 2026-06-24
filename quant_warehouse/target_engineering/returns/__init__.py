"""Return-based target label builders."""

from quant_warehouse.target_engineering.returns.cross_sectional_rank import build_cross_sectional_rank_labels
from quant_warehouse.target_engineering.returns.forward_returns import build_forward_return_labels

__all__ = [
    "build_cross_sectional_rank_labels",
    "build_forward_return_labels",
]
