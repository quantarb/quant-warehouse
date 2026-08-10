from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime
import math
from typing import Any, Literal, Mapping, Sequence

import polars as pl
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LabelMethod = Literal["rank", "mean_variance", "hybrid"]
MvProfile = Literal["unconstrained", "diversified", "hedged"]


@dataclass(frozen=True)
class OptionLabelSpec:
    """Configuration for ranking options inside an underlying trade window."""

    entry_quote_col: str = "ask"
    exit_quote_col: str = "bid"
    price_fallback_cols: tuple[str, ...] = ("mid", "last_trade_price", "close", "open")
    option_id_cols: tuple[str, ...] = ("contract_symbol",)
    option_fallback_id_cols: tuple[str, ...] = ("option_type", "expiration", "strike")
    trade_id_col: str = "trade_id"
    snapshot_date_col: str = "snapshot_date"
    underlying_symbol_col: str = "underlying_symbol"
    option_type_col: str = "option_type"
    rank_method: str = "average"
    sort_descending: bool = True
    label_method: LabelMethod = "rank"
    include_equity: bool | None = None
    risk_aversion: float = 1.0
    worthless_exit_threshold: float = 0.01
    equity_contract_suffix: str = "_EQUITY"
    equity_annual_vol: float = 0.25
    variance_floor: float = 1e-8
    trade_entry_price_col: str = "entry_px"
    trade_exit_price_col: str = "exit_px"
    covariance_quote_col: str = "mid"
    covariance_min_observations: int = 2
    covariance_shrinkage: float = 0.1
    underlying_price_snapshots: Mapping[Any, float] | None = None
    allow_short_selling: bool = False
    max_weight: float | None = None
    max_gross_exposure: float | None = None
    min_weight: float = 0.0
    mv_profile: MvProfile | None = None
    hybrid_rank_weight: float = 0.5

    @classmethod
    def diversified_mean_variance(cls, **overrides: Any) -> OptionLabelSpec:
        """Long-only MV labels: rank as return, snapshot cov as risk."""

        return cls(
            label_method="mean_variance",
            allow_short_selling=False,
            max_weight=0.15,
            max_gross_exposure=1.0,
            risk_aversion=3.0,
            mv_profile="diversified",
            **overrides,
        )

    @classmethod
    def hedged_mean_variance(cls, **overrides: Any) -> OptionLabelSpec:
        """Long/short MV labels with gross exposure and per-leg caps."""

        return cls(
            label_method="mean_variance",
            allow_short_selling=True,
            max_weight=0.10,
            max_gross_exposure=2.0,
            risk_aversion=3.0,
            mv_profile="hedged",
            **overrides,
        )

    @classmethod
    def diversified_hybrid(cls, **overrides: Any) -> OptionLabelSpec:
        """Long-only MV labels using a blend of rank and normalized return."""

        return cls(
            label_method="hybrid",
            allow_short_selling=False,
            max_weight=0.15,
            max_gross_exposure=1.0,
            risk_aversion=3.0,
            hybrid_rank_weight=0.5,
            mv_profile="diversified",
            **overrides,
        )


@dataclass(frozen=True)
class OptionLabelResult:
    """Rows plus summary stats for ranked option labels."""

    option_rows: list[dict[str, Any]] = field(default_factory=list)
    statistics: dict[str, Any] = field(default_factory=dict)


def _day(value: datetime) -> datetime:
    return value.replace(hour=0, minute=0, second=0, microsecond=0)


def build_option_label_panel(
    trades: Sequence[Mapping[str, Any]] | pl.DataFrame,
    option_chains: Mapping[Any, pl.DataFrame] | pl.DataFrame,
    *,
    spec: OptionLabelSpec | None = None,
) -> pl.DataFrame:
    """Build a per-trade, per-contract ranking panel for option candidates."""

    result = build_option_labels(trades, option_chains, spec=spec)
    if not result.option_rows:
        return pl.DataFrame()
    panel = pl.DataFrame(result.option_rows)
    spec = spec or OptionLabelSpec()
    if spec.label_method in ("mean_variance", "hybrid") and "mv_weight" in panel.columns:
        panel = panel.sort(["trade_id", "mv_weight", "option_return_pct"], descending=[False, True, True])
    else:
        panel = panel.sort(["trade_id", "rank_y", "option_return_pct"], descending=[False, True, True])
    return panel


def _build_trade_window_price_panel(
    snapshots: Mapping[Any, pl.DataFrame],
    *,
    contract_symbols: Sequence[str],
    trade: Mapping[str, Any],
    entry_dt: Any,
    exit_dt: Any,
    underlying_symbol: str,
    spec: OptionLabelSpec,
) -> pl.DataFrame:
    """Build an aligned mid-price panel for covariance estimation."""
    del trade, underlying_symbol, spec
    frames: list[pl.DataFrame] = []
    for snapshot, frame in snapshots.items():
        if frame is None or frame.is_empty() or "contract_symbol" not in frame.columns:
            continue
        quote = "mid" if "mid" in frame.columns else next((c for c in ("last_trade_price", "close") if c in frame.columns), None)
        if quote is None:
            continue
        frames.append(frame.select(["contract_symbol", pl.col(quote).cast(pl.Float64, strict=False).alias("price")]).with_columns(pl.lit(_day(snapshot)).alias("date")))
    if not frames:
        return pl.DataFrame()
    panel = pl.concat(frames, how="diagonal_relaxed").filter(pl.col("contract_symbol").is_in(list(contract_symbols)))
    panel = panel.filter((pl.col("date") >= _day(entry_dt)) & (pl.col("date") <= _day(exit_dt)))
    return panel.pivot("price", index="date", on="contract_symbol", aggregate_function="last").sort("date")


def build_option_labels(
    trades: Sequence[Mapping[str, Any]] | pl.DataFrame,
    option_chains: Mapping[Any, pl.DataFrame] | pl.DataFrame,
    *,
    spec: OptionLabelSpec | None = None,
) -> OptionLabelResult:
    """Create realized-return labels for options across each underlying trade window."""

    spec = spec or OptionLabelSpec()
    trade_rows = _normalize_trades(trades, trade_id_col=spec.trade_id_col)
    snapshots = _normalize_option_snapshots(option_chains, spec=spec)
    if not trade_rows or not snapshots:
        return OptionLabelResult()

    option_rows: list[dict[str, Any]] = []
    for trade in trade_rows:
        trade_id = str(trade.get(spec.trade_id_col) or "").strip() or _trade_id(trade)
        entry_dt = _to_timestamp(trade.get("entry_date"))
        exit_dt = _to_timestamp(trade.get("exit_date"))
        underlying_symbol = str(trade.get("symbol") or trade.get("underlying_symbol") or "").strip().upper()
        if entry_dt is None or exit_dt is None:
            continue

        entry_snapshot_date, entry_chain = _lookup_snapshot(snapshots, entry_dt)
        exit_snapshot_date, exit_chain = _lookup_snapshot(snapshots, exit_dt)
        if entry_chain.is_empty() or exit_chain.is_empty():
            continue

        if underlying_symbol:
            entry_chain = _filter_underlying(entry_chain, underlying_symbol, spec.underlying_symbol_col)
            exit_chain = _filter_underlying(exit_chain, underlying_symbol, spec.underlying_symbol_col)

        if entry_chain.is_empty() or exit_chain.is_empty():
            continue

        entry_norm = _normalize_chain(entry_chain, snapshot_date=entry_dt, spec=spec)
        exit_norm = _normalize_chain(exit_chain, snapshot_date=exit_dt, spec=spec)

        join_cols = _resolve_join_cols(entry_norm, exit_norm, spec=spec)
        merged = entry_norm.join(exit_norm, on=join_cols, how="inner", suffix="_exit")
        if merged.is_empty():
            continue

        merged = merged.with_columns([
            _quote_expr(merged, spec.entry_quote_col, spec.price_fallback_cols, suffix="").alias("entry_quote"),
            _quote_expr(merged, spec.exit_quote_col, spec.price_fallback_cols, suffix="_exit").alias("exit_quote"),
        ]).filter(pl.col("entry_quote") > 0)
        if merged.is_empty():
            continue

        merged = merged.with_columns(pl.col("exit_quote").clip(lower_bound=0.0).alias("exit_quote"))
        merged = merged.with_columns([
            ((pl.col("exit_quote") - pl.col("entry_quote")) / pl.col("entry_quote")).alias("option_return_pct"),
            pl.lit(False).alias("expires_worthless"),
        ])

        equity_row = None
        include_equity = _resolve_include_equity(spec)
        if include_equity and underlying_symbol:
            equity_row = _build_equity_candidate_row(
                trade,
                underlying_symbol=underlying_symbol,
                entry_dt=entry_dt,
                exit_dt=exit_dt,
                trade_id=trade_id,
                entry_snapshot_date=entry_snapshot_date,
                exit_snapshot_date=exit_snapshot_date,
                spec=spec,
            )

        merged = merged.with_columns([
            pl.lit(trade_id).alias("trade_id"), pl.lit(entry_dt).alias("trade_entry_date"), pl.lit(exit_dt).alias("trade_exit_date"),
            pl.lit(int((exit_dt - entry_dt).days)).alias("trade_duration_days"), pl.lit(underlying_symbol).alias("underlying_symbol"),
            pl.lit(_float(trade.get("trade_return"))).alias("underlying_return_pct"), pl.lit(entry_snapshot_date).alias("entry_snapshot_date"),
            pl.lit(exit_snapshot_date).alias("exit_snapshot_date"), pl.lit(False).alias("is_equity"),
        ])

        rank_frame = merged
        if equity_row is not None:
            rank_frame = pl.concat([merged, pl.DataFrame([equity_row])], how="diagonal_relaxed")

        rank_frame = rank_frame.with_columns([
            pl.col("option_return_pct").rank(method="average", descending=spec.sort_descending).truediv(pl.len()).alias("rank_y"),
            pl.col("option_return_pct").rank(method="ordinal", descending=not spec.sort_descending).cast(pl.Int64).alias("rank_order"),
        ])

        if spec.label_method in ("mean_variance", "hybrid"):
            rank_frame = rank_frame.with_columns(pl.Series("mv_mu", _resolve_mv_expected_returns(rank_frame, spec=spec)))
            weights = _assign_mean_variance_weights(
                rank_frame,
                snapshots=snapshots,
                trade=trade,
                entry_dt=entry_dt,
                exit_dt=exit_dt,
                spec=spec,
            )
            rank_frame = rank_frame.with_columns(pl.Series("mv_weight", weights), pl.Series("label", weights))
        else:
            rank_frame = rank_frame.with_columns(pl.lit(0.0).alias("mv_mu"), pl.lit(0.0).alias("mv_weight"), pl.col("rank_y").alias("label"))

        rank_frame = rank_frame.with_columns(pl.lit(rank_frame.height).alias("trade_option_count"), pl.lit(trade_id).alias("trade_id"))

        option_rows.extend(rank_frame.to_dicts())

    if not option_rows:
        return OptionLabelResult()

    option_rows = _postprocess_option_rows(option_rows)
    statistics = _build_option_statistics(option_rows)
    return OptionLabelResult(option_rows=option_rows, statistics=statistics)


def _normalize_trades(
    trades: Sequence[Mapping[str, Any]] | pl.DataFrame,
    *,
    trade_id_col: str,
) -> list[dict[str, Any]]:
    if trades is None:
        return []
    if isinstance(trades, pl.DataFrame):
        rows = trades.to_dicts()
    else:
        rows = [dict(row) for row in trades]
    out: list[dict[str, Any]] = []
    for idx, row in enumerate(rows, start=1):
        entry_date = _to_timestamp(row.get("entry_date"))
        exit_date = _to_timestamp(row.get("exit_date"))
        if entry_date is None or exit_date is None:
            continue
        trade = dict(row)
        trade["entry_date"] = entry_date
        trade["exit_date"] = exit_date
        trade.setdefault(trade_id_col, _trade_id(trade, fallback=str(idx)))
        out.append(trade)
    return out


def _normalize_option_snapshots(
    option_chains: Mapping[Any, pl.DataFrame] | pl.DataFrame,
    *,
    spec: OptionLabelSpec,
) -> dict[datetime, pl.DataFrame]:
    if option_chains is None:
        return {}
    snapshots: dict[datetime, pl.DataFrame] = {}
    if isinstance(option_chains, pl.DataFrame):
        if spec.snapshot_date_col not in option_chains.columns:
            raise ValueError(f"Option chain frame must include '{spec.snapshot_date_col}' or be provided as a mapping")
        for snapshot_date, group in option_chains.partition_by(spec.snapshot_date_col, as_dict=True).items():
            if isinstance(snapshot_date, tuple): snapshot_date = snapshot_date[0]
            ts = _to_timestamp(snapshot_date)
            if ts is None:
                continue
            snapshots[_day(ts)] = group
        return dict(sorted(snapshots.items(), key=lambda item: item[0]))

    for key, frame in option_chains.items():
        ts = _to_timestamp(key)
        if ts is None or frame is None or frame.is_empty():
            continue
        snapshots[_day(ts)] = frame
    return dict(sorted(snapshots.items(), key=lambda item: item[0]))


def _lookup_snapshot(
    snapshots: Mapping[datetime, pl.DataFrame],
    target: datetime,
) -> tuple[datetime | None, pl.DataFrame]:
    if not snapshots:
        return None, pl.DataFrame()
    target = _day(target)
    if target in snapshots:
        return target, snapshots[target].clone()
    prior = [snapshot for snapshot in snapshots if snapshot <= target]
    if prior:
        chosen = max(prior)
        return chosen, snapshots[chosen].clone()
    return None, pl.DataFrame()


def _filter_underlying(df: pl.DataFrame, symbol: str, col: str) -> pl.DataFrame:
    if col not in df.columns:
        return df
    return df.filter(pl.col(col).cast(pl.String).str.to_uppercase() == symbol.upper())


def _normalize_chain(df: pl.DataFrame, *, snapshot_date: datetime, spec: OptionLabelSpec) -> pl.DataFrame:
    out = df.clone().rename({col: str(col).strip().lower() for col in df.columns})
    if "right" in out.columns and spec.option_type_col not in out.columns:
        out = out.with_columns(pl.col("right").cast(pl.String).str.strip_chars().str.to_lowercase().alias(spec.option_type_col))
    if "optiontype" in out.columns and spec.option_type_col not in out.columns:
        out = out.with_columns(pl.col("optiontype").cast(pl.String).str.strip_chars().str.to_lowercase().alias(spec.option_type_col))
    if "expiration" in out.columns:
        out = out.with_columns(pl.col("expiration").cast(pl.String).str.to_datetime(strict=False).dt.truncate("1d").alias("expiration"))
    if "strike" in out.columns:
        out = out.with_columns(pl.col("strike").cast(pl.Float64, strict=False).alias("strike"))
    if spec.underlying_symbol_col in out.columns:
        out = out.with_columns(pl.col(spec.underlying_symbol_col).cast(pl.String).str.to_uppercase().alias(spec.underlying_symbol_col))
    out = out.with_columns(pl.lit(snapshot_date.replace(hour=0, minute=0, second=0, microsecond=0)).alias("snapshot_date"))
    return out


def _resolve_join_cols(entry: pl.DataFrame, exit: pl.DataFrame, *, spec: OptionLabelSpec) -> list[str]:
    preferred = [col for col in spec.option_id_cols if col in entry.columns and col in exit.columns]
    if preferred:
        return preferred
    fallback = [col for col in spec.option_fallback_id_cols if col in entry.columns and col in exit.columns]
    if fallback:
        return fallback
    raise ValueError("No shared option identity columns found between entry and exit snapshots")


def _quote_expr(
    df: pl.DataFrame,
    primary: str,
    fallbacks: Sequence[str],
    *,
    suffix: str,
) -> pl.Expr:
    candidates = [primary, *fallbacks]
    for col in candidates:
        actual = f"{col}{suffix}"
        if actual in df.columns:
            return pl.col(actual).cast(pl.Float64, strict=False)
    raise ValueError(f"Could not resolve an executable option price column with suffix {suffix}")


def _resolve_include_equity(spec: OptionLabelSpec) -> bool:
    if spec.include_equity is not None:
        return bool(spec.include_equity)
    return spec.label_method in ("mean_variance", "hybrid")


def solve_mean_variance_weights(
    expected_returns: Sequence[float] | torch.Tensor,
    variances: Sequence[float] | torch.Tensor | None = None,
    *,
    covariance: Sequence[Sequence[float]] | torch.Tensor | None = None,
    risk_aversion: float = 1.0,
    eligible: Sequence[bool] | torch.Tensor | None = None,
    long_only: bool = True,
    max_weight: float | None = None,
    max_gross_exposure: float | None = None,
    min_weight: float = 0.0,
    return_shrinkage: float = 0.0,
) -> torch.Tensor:
    """Return mean-variance portfolio weights with net budget equal to one."""

    solved = _solve_mean_variance_weights_torch(
        expected_returns,
        variances,
        covariance=covariance,
        risk_aversion=risk_aversion,
        eligible=eligible,
        long_only=long_only,
        max_weight=max_weight,
        max_gross_exposure=max_gross_exposure,
        min_weight=min_weight,
        return_shrinkage=return_shrinkage,
    )
    return solved


def _solve_mean_variance_weights_torch(
    expected_returns: Sequence[float] | torch.Tensor,
    variances: Sequence[float] | torch.Tensor | None = None,
    *,
    covariance: Sequence[Sequence[float]] | torch.Tensor | None = None,
    risk_aversion: float = 1.0,
    eligible: Sequence[bool] | torch.Tensor | None = None,
    long_only: bool = True,
    max_weight: float | None = None,
    max_gross_exposure: float | None = None,
    min_weight: float = 0.0,
    return_shrinkage: float = 0.0,
) -> torch.Tensor:
    """Torch implementation of the portfolio solver."""
    mu = torch.as_tensor(expected_returns, dtype=torch.float64, device=DEVICE)
    n = int(mu.numel())
    if n == 0:
        return torch.empty(0, dtype=torch.float64, device=DEVICE)
    mask = torch.ones(n, dtype=torch.bool, device=DEVICE) if eligible is None else torch.as_tensor(eligible, dtype=torch.bool, device=DEVICE)
    if mask.numel() != n:
        raise ValueError("eligible must have the same length as expected_returns")
    alpha = min(max(float(return_shrinkage), 0.0), 1.0)
    if alpha > 0.0 and bool(mask.any()):
        mu = mu.clone()
        target = mu[mask].mean()
        mu[mask] = (1.0 - alpha) * mu[mask] + alpha * target
    if covariance is not None:
        cov = torch.as_tensor(covariance, dtype=torch.float64, device=DEVICE)
        if tuple(cov.shape) != (n, n):
            raise ValueError(f"covariance must be ({n}, {n}); got {tuple(cov.shape)}")
        cov = _torch_psd(cov, 1e-8)
    else:
        if variances is None:
            raise ValueError("variances or covariance must be provided")
        variance = torch.clamp(torch.as_tensor(variances, dtype=torch.float64, device=DEVICE), min=0.0)
        if variance.numel() != n:
            raise ValueError("variances must have the same length as expected_returns")
        cov = torch.diag(torch.clamp(variance, min=1e-12))
    active = torch.nonzero(mask, as_tuple=False).flatten()
    weights = torch.zeros(n, dtype=torch.float64, device=DEVICE)
    if active.numel() == 0:
        return weights
    active_mu = mu[active]
    active_cov = cov.index_select(0, active).index_select(1, active)
    constrained = max_weight is not None or max_gross_exposure is not None or min_weight > 0.0
    if covariance is None and long_only and not constrained:
        scores = torch.clamp(active_mu, min=0.0) / (max(float(risk_aversion), 1e-12) * torch.diagonal(active_cov))
        solved = torch.full_like(scores, 1.0 / active.numel()) if float(scores.sum()) <= 0.0 else scores / scores.sum()
    elif not long_only and not constrained:
        solved = _torch_budget_solution(active_mu, active_cov, risk_aversion)
    else:
        solved = _torch_projected_gradient(active_mu, active_cov, risk_aversion, long_only, max_weight, max_gross_exposure, min_weight)
    weights[active] = solved
    return weights


def _torch_psd(matrix: torch.Tensor, floor: float) -> torch.Tensor:
    symmetric = (matrix + matrix.T) * 0.5
    values, vectors = torch.linalg.eigh(symmetric)
    return (vectors * torch.clamp(values, min=floor)) @ vectors.T


def _torch_budget_solution(mu: torch.Tensor, covariance: torch.Tensor, risk_aversion: float) -> torch.Tensor:
    if mu.numel() == 1:
        return torch.ones(1, dtype=torch.float64, device=DEVICE)
    lam = max(float(risk_aversion), 1e-12)
    inv_cov = torch.linalg.pinv(covariance)
    ones = torch.ones_like(mu)
    inv_mu, inv_ones = inv_cov @ mu, inv_cov @ ones
    denom = ones @ inv_ones
    if abs(float(denom)) <= 1e-12:
        return ones / mu.numel()
    nu = ((ones @ inv_mu) - lam) / denom
    return (inv_mu - nu * inv_ones) / lam


def _torch_simplex(values: torch.Tensor) -> torch.Tensor:
    if values.numel() == 0:
        return values
    ordered, _ = torch.sort(values, descending=True)
    cumulative = torch.cumsum(ordered, 0)
    indices = torch.arange(1, values.numel() + 1, dtype=torch.float64, device=DEVICE)
    valid = ordered * indices > (cumulative - 1.0)
    if not bool(valid.any()):
        return torch.full_like(values, 1.0 / values.numel())
    rho = int(torch.nonzero(valid, as_tuple=False)[-1].item())
    theta = (cumulative[rho] - 1.0) / (rho + 1)
    projected = torch.clamp(values - theta, min=0.0)
    total = projected.sum()
    return projected / total if float(total) > 0.0 else torch.full_like(values, 1.0 / values.numel())


def _torch_project_feasible(values: torch.Tensor, long_only: bool, max_weight: float | None, max_gross: float | None, min_weight: float) -> torch.Tensor:
    out = values.clone()
    if long_only:
        out = torch.clamp(out, min=0.0)
        if min_weight > 0.0:
            out = torch.where(out > 0.0, torch.maximum(out, torch.tensor(float(min_weight), dtype=out.dtype, device=DEVICE)), out)
        out = _torch_simplex(out)
        if max_weight is not None:
            cap = max(float(max_weight), 1.0 / out.numel())
            out = torch.clamp(out, max=cap)
            out = out / out.sum()
    elif max_weight is not None:
        out = torch.clamp(out, min=-float(max_weight), max=float(max_weight))
    if not long_only and max_gross is not None:
        gross = out.abs().sum()
        if float(gross) > float(max_gross) and float(gross) > 0.0:
            out = out * (float(max_gross) / gross)
    elif not long_only and max_gross is None:
        total = out.sum()
        if abs(float(total)) > 1e-12:
            out = out / total
    return out


def _torch_projected_gradient(mu: torch.Tensor, covariance: torch.Tensor, risk_aversion: float, long_only: bool, max_weight: float | None, max_gross: float | None, min_weight: float, max_iter: int = 1000) -> torch.Tensor:
    if mu.numel() == 1:
        return torch.ones(1, dtype=torch.float64, device=DEVICE)
    lam = max(float(risk_aversion), 1e-12)
    weights = torch.full_like(mu, 1.0 / mu.numel())
    step = 0.25 / max(float(torch.diagonal(covariance).max()), 1e-6)
    for iteration in range(max_iter):
        proposal = _torch_project_feasible(weights + step * (mu - lam * (covariance @ weights)), long_only, max_weight, max_gross, min_weight)
        if float(torch.abs(proposal - weights).sum()) <= 1e-9:
            weights = proposal
            break
        weights = proposal
        if iteration > 50:
            step *= 0.995
    return _torch_project_feasible(weights, long_only, max_weight, max_gross, min_weight)

def solve_long_only_mean_variance_weights(
    expected_returns: Sequence[float] | torch.Tensor,
    variances: Sequence[float] | torch.Tensor | None = None,
    *,
    covariance: Sequence[Sequence[float]] | torch.Tensor | None = None,
    risk_aversion: float = 1.0,
    eligible: Sequence[bool] | torch.Tensor | None = None,
) -> torch.Tensor:
    """Return long-only portfolio weights that sum to one (no short selling)."""

    return solve_mean_variance_weights(
        expected_returns,
        variances,
        covariance=covariance,
        risk_aversion=risk_aversion,
        eligible=eligible,
        long_only=True,
    )


def compute_return_covariance_matrix(
    returns: pl.DataFrame,
    *,
    shrinkage: float = 0.1,
    variance_floor: float = 1e-8,
) -> torch.Tensor:
    """Estimate a PSD return covariance matrix from an aligned return panel."""

    if returns is None or returns.is_empty():
        return torch.empty((0, 0), dtype=torch.float64, device=DEVICE)

    values = torch.tensor(returns.cast(pl.Float64, strict=False).fill_null(0.0).fill_nan(0.0).rows(), dtype=torch.float64, device=DEVICE)
    sample = torch.cov(values.T)
    if sample.ndim == 0:
        sample = sample.reshape(1, 1)
    sample = torch.nan_to_num(sample, nan=0.0, posinf=0.0, neginf=0.0)
    diag = torch.diag(torch.diagonal(sample))
    alpha = min(max(float(shrinkage), 0.0), 1.0)
    cov = (1.0 - alpha) * sample + alpha * diag
    cov = _ensure_positive_semidefinite(cov, floor=variance_floor)
    return cov


def _ensure_positive_semidefinite(matrix: torch.Tensor, *, floor: float) -> torch.Tensor:
    """Clamp covariance eigenvalues to a small positive floor."""
    symmetric = (matrix + matrix.T) * 0.5
    values, vectors = torch.linalg.eigh(symmetric)
    return (vectors * torch.clamp(values, min=float(floor))) @ vectors.T


def _resolve_mv_spec(spec: OptionLabelSpec) -> OptionLabelSpec:
    if spec.mv_profile in (None, "unconstrained"):
        return spec

    profile = (
        OptionLabelSpec.diversified_mean_variance()
        if spec.mv_profile == "diversified"
        else OptionLabelSpec.hedged_mean_variance()
    )
    baseline = OptionLabelSpec()

    def _pick(field_name: str, profile_value: Any) -> Any:
        current = getattr(spec, field_name)
        default = getattr(baseline, field_name)
        return current if current != default else profile_value

    return replace(
        profile,
        include_equity=spec.include_equity,
        worthless_exit_threshold=spec.worthless_exit_threshold,
        covariance_shrinkage=spec.covariance_shrinkage,
        covariance_min_observations=spec.covariance_min_observations,
        covariance_quote_col=spec.covariance_quote_col,
        underlying_price_snapshots=spec.underlying_price_snapshots,
        allow_short_selling=_pick("allow_short_selling", profile.allow_short_selling),
        max_weight=_pick("max_weight", profile.max_weight),
        max_gross_exposure=_pick("max_gross_exposure", profile.max_gross_exposure),
        min_weight=_pick("min_weight", profile.min_weight),
        risk_aversion=_pick("risk_aversion", profile.risk_aversion),
    )


def _normalize_trade_returns(frame: pl.DataFrame, *, eligible: torch.Tensor) -> torch.Tensor:
    """Scale realized returns to [0, 1] within a trade's eligible legs."""

    returns = torch.tensor(frame["option_return_pct"].cast(pl.Float64, strict=False).fill_null(-1.0).to_list(), dtype=torch.float64, device=DEVICE)
    active = eligible.to(torch.bool)
    normalized = torch.zeros_like(returns)
    if not bool(active.any()):
        return normalized

    active_returns = returns[active]
    lo = float(active_returns.min())
    hi = float(active_returns.max())
    if hi <= lo:
        normalized[active] = 1.0
        return normalized

    normalized[active] = (returns[active] - lo) / (hi - lo)
    return normalized


def _resolve_mv_expected_returns(frame: pl.DataFrame, *, spec: OptionLabelSpec) -> torch.Tensor:
    """Build MV expected-return vector from rank, return, or a hybrid blend."""

    rank_y = torch.tensor(frame["rank_y"].cast(pl.Float64, strict=False).fill_null(0.0).to_list(), dtype=torch.float64, device=DEVICE)
    if spec.label_method == "mean_variance":
        return rank_y

    eligible = ~torch.tensor(frame["expires_worthless"].to_list(), dtype=torch.bool, device=DEVICE)
    return_norm = _normalize_trade_returns(frame, eligible=eligible)
    rank_weight = min(max(float(spec.hybrid_rank_weight), 0.0), 1.0)
    mu = rank_weight * rank_y + (1.0 - rank_weight) * return_norm
    return mu


def _assign_mean_variance_weights(
    frame: pl.DataFrame,
    *,
    snapshots: Mapping[datetime, pl.DataFrame],
    trade: Mapping[str, Any],
    entry_dt: datetime,
    exit_dt: datetime,
    spec: OptionLabelSpec,
) -> torch.Tensor:
    spec = _resolve_mv_spec(spec)
    eligible = ~torch.tensor(frame["expires_worthless"].to_list(), dtype=torch.bool, device=DEVICE)
    if "mv_mu" in frame.columns:
        mu = torch.tensor(frame["mv_mu"].cast(pl.Float64, strict=False).fill_null(0.0).to_list(), dtype=torch.float64, device=DEVICE)
    else:
        mu = _resolve_mv_expected_returns(frame, spec=spec)

    long_only = not spec.allow_short_selling
    eligible_mask = eligible
    solver_kwargs = {
        "risk_aversion": spec.risk_aversion,
        "eligible": eligible_mask,
        "long_only": long_only,
        "max_weight": spec.max_weight,
        "max_gross_exposure": spec.max_gross_exposure,
        "min_weight": spec.min_weight,
    }
    weights = solve_mean_variance_weights(mu, torch.ones(len(mu), dtype=torch.float64, device=DEVICE), **solver_kwargs)

    return weights


def _postprocess_option_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized_rows = _normalize_rows_for_polars(rows)
    columns = sorted({key for row in normalized_rows for key in row})
    df = pl.DataFrame({column: [row.get(column) for row in normalized_rows] for column in columns}, strict=False)
    if df.is_empty():
        return []
    id_cols = [col for col in ("trade_id", "contract_symbol", "option_type", "expiration", "strike") if col in df.columns]
    numeric = [column for column in ("rank_y", "option_return_pct", "mv_mu", "mv_weight") if column in df.columns]
    if numeric:
        df = df.with_columns([pl.col(column).cast(pl.Float64, strict=False).fill_null(0.0).alias(column) if column in {"mv_mu", "mv_weight"} else pl.col(column).cast(pl.Float64, strict=False).alias(column) for column in numeric])
    bools = [column for column in ("expires_worthless", "is_equity") if column in df.columns]
    if bools:
        df = df.with_columns([pl.col(column).cast(pl.Boolean, strict=False).fill_null(False).alias(column) for column in bools])
    sort_cols = [col for col in ("trade_id", "rank_order", "option_return_pct") if col in df.columns]
    if sort_cols:
        df = df.sort(sort_cols)
    if id_cols:
        df = df.unique(subset=id_cols, keep="first", maintain_order=True)
    return df.to_dicts()


def _build_option_statistics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"trade_stats": {}, "option_group_stats": []}
    normalized_rows = _normalize_rows_for_polars(rows)
    columns = sorted({key for row in normalized_rows for key in row})
    df = pl.DataFrame({column: [row.get(column) for row in normalized_rows] for column in columns}, strict=False)
    avg_return = df.select(pl.col("option_return_pct").cast(pl.Float64, strict=False).mean()).item() if "option_return_pct" in df.columns else 0.0
    stats = {
        "trade_stats": {
            "trades": int(df.select(pl.col("trade_id").n_unique()).item()) if "trade_id" in df.columns else 0,
            "contracts": int(df.height),
            "avg_option_return_pct": round(float(avg_return or 0.0) * 100.0, 4),
        },
        "option_group_stats": [],
    }
    if "mv_weight" in df.columns:
        avg_weight = df.select(pl.col("mv_weight").cast(pl.Float64, strict=False).mean()).item()
        stats["trade_stats"]["avg_mv_weight"] = round(float(avg_weight or 0.0), 6)
        stats["trade_stats"]["worthless_contracts"] = int(df.select(pl.col("expires_worthless").cast(pl.Int64).sum()).item()) if "expires_worthless" in df.columns else 0
    if "option_type" in df.columns:
        stats["option_group_stats"] = df.group_by(["trade_id", "option_type"], maintain_order=True).agg(
            pl.col("option_return_pct").count().alias("count"),
            pl.col("option_return_pct").cast(pl.Float64, strict=False).mean().alias("mean"),
            pl.col("option_return_pct").cast(pl.Float64, strict=False).median().alias("median"),
        ).to_dicts()
    return stats


def _normalize_rows_for_polars(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    string_columns = {
        "trade_id", "contract_symbol", "underlying_symbol", "underlying_symbol_entry",
        "underlying_symbol_exit", "option_type", "option_type_entry", "option_type_exit",
    }
    normalized: list[dict[str, Any]] = []
    for row in rows:
        clean: dict[str, Any] = {}
        for key, value in row.items():
            if isinstance(value, datetime):
                value = value
            elif value is None or (isinstance(value, float) and not math.isfinite(value)):
                value = None
            if isinstance(value, float) and not math.isfinite(value):
                value = None
            if key in string_columns and value is not None and not isinstance(value, str):
                value = None
            clean[key] = value
        normalized.append(clean)
    return normalized


def _trade_id(trade: Mapping[str, Any], *, fallback: str | None = None) -> str:
    symbol = str(trade.get("symbol") or trade.get("underlying_symbol") or "").strip().upper()
    entry_date = _to_timestamp(trade.get("entry_date"))
    exit_date = _to_timestamp(trade.get("exit_date"))
    side = str(trade.get("side") or "").strip().lower()
    if symbol and entry_date is not None and exit_date is not None:
        return f"T|{symbol}|E{entry_date.date().isoformat()}|X{exit_date.date().isoformat()}|S{side or 'na'}"
    return fallback or "trade"


def _to_timestamp(value: Any) -> datetime | None:
    if value is None or value == "":
        return None
    try:
        ts = value if isinstance(value, datetime) else datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except Exception:
        return None
    return ts.replace(tzinfo=None)


def _float(value: Any) -> float:
    try:
        return float(value if value not in (None, "") else 0.0)
    except Exception:
        return 0.0


def _float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        parsed = float(str(value).replace(",", "").replace("%", "").strip())
    except Exception:
        return None
    if not math.isfinite(parsed):
        return None
    return float(parsed)
