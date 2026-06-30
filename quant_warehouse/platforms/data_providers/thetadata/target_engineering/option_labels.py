from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import pandas as pd

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


def build_option_label_panel(
    trades: Sequence[Mapping[str, Any]] | pd.DataFrame,
    option_chains: Mapping[Any, pd.DataFrame] | pd.DataFrame,
    *,
    spec: OptionLabelSpec | None = None,
) -> pd.DataFrame:
    """Build a per-trade, per-contract ranking panel for option candidates."""

    result = build_option_labels(trades, option_chains, spec=spec)
    if not result.option_rows:
        return pd.DataFrame()
    panel = pd.DataFrame(result.option_rows)
    spec = spec or OptionLabelSpec()
    if spec.label_method in ("mean_variance", "hybrid") and "mv_weight" in panel.columns:
        return panel.sort_values(["trade_id", "mv_weight", "option_return_pct"], ascending=[True, False, False])
    return panel.sort_values(["trade_id", "rank_y", "option_return_pct"], ascending=[True, False, False])


def build_option_labels(
    trades: Sequence[Mapping[str, Any]] | pd.DataFrame,
    option_chains: Mapping[Any, pd.DataFrame] | pd.DataFrame,
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
        if entry_chain.empty or exit_chain.empty:
            continue

        if underlying_symbol:
            entry_chain = _filter_underlying(entry_chain, underlying_symbol, spec.underlying_symbol_col)
            exit_chain = _filter_underlying(exit_chain, underlying_symbol, spec.underlying_symbol_col)

        if entry_chain.empty or exit_chain.empty:
            continue

        entry_norm = _normalize_chain(entry_chain, snapshot_date=entry_dt, spec=spec)
        exit_norm = _normalize_chain(exit_chain, snapshot_date=exit_dt, spec=spec)

        join_cols = _resolve_join_cols(entry_norm, exit_norm, spec=spec)
        merged = entry_norm.merge(exit_norm, on=join_cols, suffixes=("_entry", "_exit"), how="inner")
        if merged.empty:
            continue

        merged["entry_quote"] = _pick_price_series(merged, spec.entry_quote_col, spec.price_fallback_cols, suffix="_entry")
        merged["exit_quote"] = _pick_price_series(merged, spec.exit_quote_col, spec.price_fallback_cols, suffix="_exit")
        merged = merged[merged["entry_quote"] > 0].copy()
        if merged.empty:
            continue

        merged["exit_quote"] = merged["exit_quote"].clip(lower=0.0)
        underlying_exit_px = _resolve_underlying_exit_price(trade, spec=spec)
        merged["expires_worthless"] = _expires_worthless_mask(
            merged,
            trade_exit_date=exit_dt,
            underlying_exit_price=underlying_exit_px,
            spec=spec,
        )
        merged["option_return_pct"] = np.where(
            merged["expires_worthless"],
            -1.0,
            (merged["exit_quote"] - merged["entry_quote"]) / merged["entry_quote"],
        )

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

        merged["trade_id"] = trade_id
        merged["trade_entry_date"] = entry_dt
        merged["trade_exit_date"] = exit_dt
        merged["trade_duration_days"] = int((exit_dt - entry_dt).days)
        merged["underlying_symbol"] = underlying_symbol
        merged["underlying_return_pct"] = _float(trade.get("trade_return"))
        merged["entry_snapshot_date"] = entry_snapshot_date
        merged["exit_snapshot_date"] = exit_snapshot_date
        merged["is_equity"] = False

        rank_frame = merged
        if equity_row is not None:
            rank_frame = pd.concat([merged, pd.DataFrame([equity_row])], ignore_index=True)

        rank_frame["rank_y"] = rank_frame["option_return_pct"].rank(
            method=spec.rank_method,
            pct=True,
            ascending=spec.sort_descending,
        )
        rank_frame["rank_order"] = rank_frame["option_return_pct"].rank(
            method="first",
            ascending=not spec.sort_descending,
        ).astype(int)

        if spec.label_method in ("mean_variance", "hybrid"):
            rank_frame["mv_mu"] = _resolve_mv_expected_returns(rank_frame, spec=spec)
            rank_frame["mv_weight"] = _assign_mean_variance_weights(
                rank_frame,
                snapshots=snapshots,
                trade=trade,
                entry_dt=entry_dt,
                exit_dt=exit_dt,
                spec=spec,
            )
            rank_frame["label"] = rank_frame["mv_weight"]
        else:
            rank_frame["mv_mu"] = 0.0
            rank_frame["mv_weight"] = 0.0
            rank_frame["label"] = rank_frame["rank_y"]

        rank_frame["trade_option_count"] = int(len(rank_frame))
        rank_frame["trade_id"] = trade_id

        option_rows.extend(rank_frame.to_dict(orient="records"))

    if not option_rows:
        return OptionLabelResult()

    option_rows = _postprocess_option_rows(option_rows)
    statistics = _build_option_statistics(option_rows)
    return OptionLabelResult(option_rows=option_rows, statistics=statistics)


def _normalize_trades(
    trades: Sequence[Mapping[str, Any]] | pd.DataFrame,
    *,
    trade_id_col: str,
) -> list[dict[str, Any]]:
    if trades is None:
        return []
    if isinstance(trades, pd.DataFrame):
        rows = trades.to_dict(orient="records")
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
    option_chains: Mapping[Any, pd.DataFrame] | pd.DataFrame,
    *,
    spec: OptionLabelSpec,
) -> dict[pd.Timestamp, pd.DataFrame]:
    if option_chains is None:
        return {}
    snapshots: dict[pd.Timestamp, pd.DataFrame] = {}
    if isinstance(option_chains, pd.DataFrame):
        if spec.snapshot_date_col not in option_chains.columns:
            raise ValueError(f"Option chain frame must include '{spec.snapshot_date_col}' or be provided as a mapping")
        for snapshot_date, group in option_chains.groupby(pd.to_datetime(option_chains[spec.snapshot_date_col], errors="coerce")):
            ts = _to_timestamp(snapshot_date)
            if ts is None:
                continue
            snapshots[ts.normalize()] = group.copy()
        return dict(sorted(snapshots.items(), key=lambda item: item[0]))

    for key, frame in option_chains.items():
        ts = _to_timestamp(key)
        if ts is None or frame is None or len(frame) == 0:
            continue
        snapshots[ts.normalize()] = frame.copy()
    return dict(sorted(snapshots.items(), key=lambda item: item[0]))


def _lookup_snapshot(
    snapshots: Mapping[pd.Timestamp, pd.DataFrame],
    target: pd.Timestamp,
) -> tuple[pd.Timestamp | None, pd.DataFrame]:
    if not snapshots:
        return None, pd.DataFrame()
    target = target.normalize()
    if target in snapshots:
        return target, snapshots[target].copy()
    prior = [snapshot for snapshot in snapshots if snapshot <= target]
    if prior:
        chosen = max(prior)
        return chosen, snapshots[chosen].copy()
    return None, pd.DataFrame()


def _filter_underlying(df: pd.DataFrame, symbol: str, col: str) -> pd.DataFrame:
    if col not in df.columns:
        return df
    return df.loc[df[col].astype(str).str.upper() == symbol.upper()].copy()


def _normalize_chain(df: pd.DataFrame, *, snapshot_date: pd.Timestamp, spec: OptionLabelSpec) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(col).strip().lower() for col in out.columns]
    if "right" in out.columns and spec.option_type_col not in out.columns:
        out[spec.option_type_col] = out["right"].astype(str).str.strip().str.lower()
    if "optiontype" in out.columns and spec.option_type_col not in out.columns:
        out[spec.option_type_col] = out["optiontype"].astype(str).str.strip().str.lower()
    if "expiration" in out.columns:
        out["expiration"] = pd.to_datetime(out["expiration"], errors="coerce").dt.normalize()
    if "strike" in out.columns:
        out["strike"] = pd.to_numeric(out["strike"], errors="coerce")
    if spec.underlying_symbol_col in out.columns:
        out[spec.underlying_symbol_col] = out[spec.underlying_symbol_col].astype(str).str.upper()
    out["snapshot_date"] = snapshot_date.normalize()
    return out


def _resolve_join_cols(entry: pd.DataFrame, exit: pd.DataFrame, *, spec: OptionLabelSpec) -> list[str]:
    preferred = [col for col in spec.option_id_cols if col in entry.columns and col in exit.columns]
    if preferred:
        return preferred
    fallback = [col for col in spec.option_fallback_id_cols if col in entry.columns and col in exit.columns]
    if fallback:
        return fallback
    raise ValueError("No shared option identity columns found between entry and exit snapshots")


def _pick_price_series(
    df: pd.DataFrame,
    primary: str,
    fallbacks: Sequence[str],
    *,
    suffix: str,
) -> pd.Series:
    candidates = [primary, *fallbacks]
    for col in candidates:
        actual = f"{col}{suffix}"
        if actual in df.columns:
            series = pd.to_numeric(df[actual], errors="coerce")
            if series.notna().any():
                return series
    raise ValueError(f"Could not resolve an executable option price column with suffix {suffix}")


def _resolve_include_equity(spec: OptionLabelSpec) -> bool:
    if spec.include_equity is not None:
        return bool(spec.include_equity)
    return spec.label_method in ("mean_variance", "hybrid")


def solve_mean_variance_weights(
    expected_returns: Sequence[float] | np.ndarray,
    variances: Sequence[float] | np.ndarray | None = None,
    *,
    covariance: Sequence[Sequence[float]] | np.ndarray | None = None,
    risk_aversion: float = 1.0,
    eligible: Sequence[bool] | np.ndarray | None = None,
    long_only: bool = True,
    max_weight: float | None = None,
    max_gross_exposure: float | None = None,
    min_weight: float = 0.0,
    return_shrinkage: float = 0.0,
) -> np.ndarray:
    """Return mean-variance portfolio weights with net budget equal to one."""

    mu = np.asarray(expected_returns, dtype=float)
    n = len(mu)
    if n == 0:
        return np.array([], dtype=float)

    mask = np.ones(n, dtype=bool) if eligible is None else np.asarray(eligible, dtype=bool)
    mu = _shrink_expected_returns(mu, mask, return_shrinkage)
    constraints = _mv_constraints_active(
        max_weight=max_weight,
        max_gross_exposure=max_gross_exposure,
        min_weight=min_weight,
    )

    if covariance is not None:
        cov = np.asarray(covariance, dtype=float)
        if cov.shape != (n, n):
            raise ValueError(f"covariance must be ({n}, {n}); got {cov.shape}")
        return _solve_mean_variance_covariance(
            mu,
            cov,
            risk_aversion=risk_aversion,
            eligible=mask,
            long_only=long_only,
            max_weight=max_weight,
            max_gross_exposure=max_gross_exposure,
            min_weight=min_weight,
            constraints=constraints,
        )

    if variances is None:
        raise ValueError("variances or covariance must be provided")
    return _solve_mean_variance_diagonal(
        mu,
        np.maximum(np.asarray(variances, dtype=float), 0.0),
        risk_aversion=risk_aversion,
        eligible=mask,
        long_only=long_only,
        max_weight=max_weight,
        max_gross_exposure=max_gross_exposure,
        min_weight=min_weight,
        constraints=constraints,
    )


def solve_long_only_mean_variance_weights(
    expected_returns: Sequence[float] | np.ndarray,
    variances: Sequence[float] | np.ndarray | None = None,
    *,
    covariance: Sequence[Sequence[float]] | np.ndarray | None = None,
    risk_aversion: float = 1.0,
    eligible: Sequence[bool] | np.ndarray | None = None,
) -> np.ndarray:
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
    returns: pd.DataFrame,
    *,
    shrinkage: float = 0.1,
    variance_floor: float = 1e-8,
) -> np.ndarray:
    """Estimate a PSD return covariance matrix from an aligned return panel."""

    if returns is None or returns.empty:
        return np.array([[]], dtype=float)

    sample = returns.astype(float).cov(min_periods=1).to_numpy(dtype=float)
    sample = np.nan_to_num(sample, nan=0.0, posinf=0.0, neginf=0.0)
    diag = np.diag(np.diag(sample))
    alpha = float(np.clip(shrinkage, 0.0, 1.0))
    cov = (1.0 - alpha) * sample + alpha * diag
    cov = _ensure_positive_semidefinite(cov, floor=variance_floor)
    return cov


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


def _normalize_trade_returns(frame: pd.DataFrame, *, eligible: pd.Series) -> np.ndarray:
    """Scale realized returns to [0, 1] within a trade's eligible legs."""

    returns = pd.to_numeric(frame["option_return_pct"], errors="coerce").fillna(-1.0).to_numpy(dtype=float)
    active = eligible.to_numpy(dtype=bool)
    normalized = np.zeros_like(returns, dtype=float)
    if not active.any():
        return normalized

    active_returns = returns[active]
    lo = float(np.min(active_returns))
    hi = float(np.max(active_returns))
    if hi <= lo:
        normalized[active] = 1.0
        return normalized

    normalized[active] = (returns[active] - lo) / (hi - lo)
    return normalized


def _resolve_mv_expected_returns(frame: pd.DataFrame, *, spec: OptionLabelSpec) -> pd.Series:
    """Build MV expected-return vector from rank, return, or a hybrid blend."""

    rank_y = pd.to_numeric(frame["rank_y"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    if spec.label_method == "mean_variance":
        return pd.Series(rank_y, index=frame.index, dtype=float)

    eligible = ~frame["expires_worthless"].astype(bool)
    return_norm = _normalize_trade_returns(frame, eligible=eligible)
    rank_weight = float(np.clip(spec.hybrid_rank_weight, 0.0, 1.0))
    mu = rank_weight * rank_y + (1.0 - rank_weight) * return_norm
    return pd.Series(mu, index=frame.index, dtype=float)


def _assign_mean_variance_weights(
    frame: pd.DataFrame,
    *,
    snapshots: Mapping[pd.Timestamp, pd.DataFrame],
    trade: Mapping[str, Any],
    entry_dt: pd.Timestamp,
    exit_dt: pd.Timestamp,
    spec: OptionLabelSpec,
) -> pd.Series:
    spec = _resolve_mv_spec(spec)
    eligible = ~frame["expires_worthless"].astype(bool)
    if "mv_mu" in frame.columns:
        mu = pd.to_numeric(frame["mv_mu"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    else:
        mu = _resolve_mv_expected_returns(frame, spec=spec).to_numpy(dtype=float)
    contract_symbols = frame["contract_symbol"].astype(str).tolist()
    underlying_symbol = str(frame["underlying_symbol"].iloc[0] if "underlying_symbol" in frame.columns else "").strip().upper()

    covariance = _resolve_return_covariance(
        snapshots,
        contract_symbols=contract_symbols,
        trade=trade,
        entry_dt=entry_dt,
        exit_dt=exit_dt,
        underlying_symbol=underlying_symbol,
        spec=spec,
    )

    long_only = not spec.allow_short_selling
    eligible_mask = eligible.to_numpy(dtype=bool)
    solver_kwargs = {
        "risk_aversion": spec.risk_aversion,
        "eligible": eligible_mask,
        "long_only": long_only,
        "max_weight": spec.max_weight,
        "max_gross_exposure": spec.max_gross_exposure,
        "min_weight": spec.min_weight,
    }
    if covariance is not None and covariance.shape == (len(mu), len(mu)):
        weights = solve_mean_variance_weights(mu, covariance=covariance, **solver_kwargs)
    else:
        variances = _resolve_return_variances(frame, spec=spec)
        weights = solve_mean_variance_weights(mu, variances, **solver_kwargs)

    weights = _finalize_mean_variance_weights(
        weights,
        eligible=eligible_mask,
        long_only=long_only,
        max_weight=spec.max_weight,
        max_gross_exposure=spec.max_gross_exposure,
        min_weight=spec.min_weight,
    )
    return pd.Series(weights, index=frame.index, dtype=float)


def _resolve_return_covariance(
    snapshots: Mapping[pd.Timestamp, pd.DataFrame],
    *,
    contract_symbols: Sequence[str],
    trade: Mapping[str, Any],
    entry_dt: pd.Timestamp,
    exit_dt: pd.Timestamp,
    underlying_symbol: str,
    spec: OptionLabelSpec,
) -> np.ndarray | None:
    price_panel = _build_trade_window_price_panel(
        snapshots,
        contract_symbols=contract_symbols,
        trade=trade,
        entry_dt=entry_dt,
        exit_dt=exit_dt,
        underlying_symbol=underlying_symbol,
        spec=spec,
    )
    if price_panel.shape[0] < 2 or price_panel.shape[1] == 0:
        return None

    returns = price_panel.pct_change().replace([np.inf, -np.inf], np.nan).dropna(how="all")
    if len(returns) < int(spec.covariance_min_observations):
        return None

    return compute_return_covariance_matrix(
        returns,
        shrinkage=spec.covariance_shrinkage,
        variance_floor=spec.variance_floor,
    )


def _build_trade_window_price_panel(
    snapshots: Mapping[pd.Timestamp, pd.DataFrame],
    *,
    contract_symbols: Sequence[str],
    trade: Mapping[str, Any],
    entry_dt: pd.Timestamp,
    exit_dt: pd.Timestamp,
    underlying_symbol: str,
    spec: OptionLabelSpec,
) -> pd.DataFrame:
    window = _snapshots_in_trade_window(snapshots, entry_dt, exit_dt)
    if not window:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for snap_dt, chain in window:
        norm = _normalize_chain(chain, snapshot_date=snap_dt, spec=spec)
        if underlying_symbol:
            norm = _filter_underlying(norm, underlying_symbol, spec.underlying_symbol_col)
        for contract_symbol in contract_symbols:
            if contract_symbol.endswith(spec.equity_contract_suffix):
                quote = _resolve_underlying_price_at_date(
                    trade,
                    snap_dt,
                    entry_dt=entry_dt,
                    exit_dt=exit_dt,
                    spec=spec,
                )
            else:
                quote = _lookup_contract_quote(norm, contract_symbol, spec=spec)
            rows.append(
                {
                    "snapshot_date": snap_dt.normalize(),
                    "contract_symbol": contract_symbol,
                    "quote": quote,
                }
            )

    if not rows:
        return pd.DataFrame()

    frame = pd.DataFrame(rows)
    if frame.duplicated(subset=["snapshot_date", "contract_symbol"]).any():
        frame = (
            frame.groupby(["snapshot_date", "contract_symbol"], as_index=False)["quote"]
            .mean(numeric_only=True)
        )
    panel = frame.pivot(index="snapshot_date", columns="contract_symbol", values="quote").sort_index()
    panel = panel.reindex(columns=list(contract_symbols))
    numeric = panel.apply(pd.to_numeric, errors="coerce")
    return numeric.ffill().bfill()


def _snapshots_in_trade_window(
    snapshots: Mapping[pd.Timestamp, pd.DataFrame],
    entry_dt: pd.Timestamp,
    exit_dt: pd.Timestamp,
) -> list[tuple[pd.Timestamp, pd.DataFrame]]:
    start = entry_dt.normalize()
    end = exit_dt.normalize()
    return [
        (ts, frame.copy())
        for ts, frame in sorted(snapshots.items(), key=lambda item: item[0])
        if start <= ts.normalize() <= end
    ]


def _lookup_contract_quote(
    chain: pd.DataFrame,
    contract_symbol: str,
    *,
    spec: OptionLabelSpec,
) -> float | None:
    if chain.empty or "contract_symbol" not in chain.columns:
        return None
    match = chain.loc[chain["contract_symbol"].astype(str) == str(contract_symbol)]
    if match.empty:
        return None
    return _resolve_row_quote(match.iloc[0], spec.covariance_quote_col, spec.price_fallback_cols)


def _resolve_row_quote(row: pd.Series, primary: str, fallbacks: Sequence[str]) -> float | None:
    candidates = [primary, *fallbacks]
    for col in candidates:
        if col not in row.index:
            continue
        value = pd.to_numeric(row[col], errors="coerce")
        if pd.notna(value) and float(value) > 0.0:
            return float(value)
    return None


def _resolve_underlying_price_at_date(
    trade: Mapping[str, Any],
    snapshot_date: pd.Timestamp,
    *,
    entry_dt: pd.Timestamp,
    exit_dt: pd.Timestamp,
    spec: OptionLabelSpec,
) -> float | None:
    snap = snapshot_date.normalize()
    if spec.underlying_price_snapshots:
        mapped = _normalize_price_snapshot_map(spec.underlying_price_snapshots)
        if snap in mapped:
            return mapped[snap]
        prior = [ts for ts in mapped if ts <= snap]
        if prior:
            return mapped[max(prior)]

    if snap == entry_dt.normalize():
        return _resolve_underlying_entry_price(trade, spec=spec)
    if snap == exit_dt.normalize():
        return _resolve_underlying_exit_price(trade, spec=spec)
    return None


def _normalize_price_snapshot_map(values: Mapping[Any, float]) -> dict[pd.Timestamp, float]:
    out: dict[pd.Timestamp, float] = {}
    for key, value in values.items():
        ts = _to_timestamp(key)
        parsed = _float_or_none(value)
        if ts is None or parsed is None or parsed <= 0.0:
            continue
        out[ts.normalize()] = parsed
    return dict(sorted(out.items(), key=lambda item: item[0]))


def _shrink_expected_returns(
    expected_returns: np.ndarray,
    eligible: np.ndarray,
    shrinkage: float,
) -> np.ndarray:
    alpha = float(np.clip(shrinkage, 0.0, 1.0))
    if alpha <= 0.0 or not eligible.any():
        return expected_returns
    active = expected_returns[eligible]
    target = float(np.mean(active))
    shrunk = expected_returns.copy()
    shrunk[eligible] = (1.0 - alpha) * active + alpha * target
    return shrunk


def _mv_constraints_active(
    *,
    max_weight: float | None,
    max_gross_exposure: float | None,
    min_weight: float,
) -> bool:
    return max_weight is not None or max_gross_exposure is not None or float(min_weight) > 0.0


def _finalize_mean_variance_weights(
    weights: np.ndarray,
    *,
    eligible: np.ndarray,
    long_only: bool,
    max_weight: float | None,
    max_gross_exposure: float | None,
    min_weight: float,
) -> np.ndarray:
    return _project_feasible_weights(
        weights,
        eligible=eligible,
        long_only=long_only,
        max_weight=max_weight,
        max_gross_exposure=max_gross_exposure,
        min_weight=min_weight,
    )


def _effective_max_weight(max_weight: float | None, eligible_count: int) -> float | None:
    if max_weight is None or eligible_count <= 0:
        return max_weight
    required = 1.0 / float(eligible_count)
    return max(float(max_weight), required)


def _project_feasible_weights(
    weights: np.ndarray,
    *,
    eligible: np.ndarray,
    long_only: bool,
    max_weight: float | None,
    max_gross_exposure: float | None,
    min_weight: float,
) -> np.ndarray:
    vector = np.asarray(weights, dtype=float).copy()
    eligible_idx = np.flatnonzero(eligible)
    eligible_count = int(eligible_idx.size)
    if eligible_count == 0:
        return np.zeros_like(vector)

    weight_cap = _effective_max_weight(max_weight, eligible_count)
    for _ in range(20):
        raw = np.zeros_like(vector)
        source = vector.copy()
        source[~eligible] = 0.0
        if long_only:
            source = np.maximum(source, 0.0)
        if min_weight > 0.0 and long_only:
            source[eligible_idx] = np.where(
                source[eligible_idx] > 0.0,
                np.maximum(source[eligible_idx], float(min_weight)),
                source[eligible_idx],
            )
        if weight_cap is not None:
            cap = float(weight_cap)
            if long_only:
                source[eligible_idx] = np.minimum(source[eligible_idx], cap)
            else:
                source[eligible_idx] = np.clip(source[eligible_idx], -cap, cap)

        if long_only:
            active = source[eligible_idx]
            if float(active.sum()) <= 0.0:
                active = np.full(eligible_count, 1.0 / eligible_count, dtype=float)
            else:
                active = _project_to_simplex(active)
                if weight_cap is not None:
                    active = np.minimum(active, float(weight_cap))
                    total = float(active.sum())
                    if total > 1e-12:
                        active = active / total
            raw[eligible_idx] = active
        else:
            active = source[eligible_idx]
            if max_gross_exposure is not None:
                gross = float(np.abs(active).sum())
                if gross > 1e-12:
                    active = active * (float(max_gross_exposure) / gross)
            else:
                total = float(active.sum())
                if abs(total) > 1e-12:
                    active = active / total
            raw[eligible_idx] = active

        vector = raw
        gross_ok = max_gross_exposure is None or float(np.abs(vector).sum()) <= float(max_gross_exposure) + 1e-9
        cap_ok = weight_cap is None or float(np.max(np.abs(vector[eligible_idx]))) <= float(weight_cap) + 1e-9
        if gross_ok and cap_ok:
            break

    return vector


def _solve_mean_variance_diagonal(
    expected_returns: np.ndarray,
    variances: np.ndarray,
    *,
    risk_aversion: float,
    eligible: np.ndarray,
    long_only: bool,
    max_weight: float | None,
    max_gross_exposure: float | None,
    min_weight: float,
    constraints: bool,
) -> np.ndarray:
    n = len(expected_returns)
    idx = np.flatnonzero(eligible)
    if idx.size == 0:
        return np.zeros(n, dtype=float)

    if constraints or not long_only:
        cov = np.diag(np.maximum(variances[idx], 1e-12))
        solved = _projected_gradient_mean_variance(
            expected_returns[idx],
            cov,
            risk_aversion=risk_aversion,
            long_only=long_only,
            max_weight=max_weight,
            max_gross_exposure=max_gross_exposure,
            min_weight=min_weight,
        )
        weights = np.zeros(n, dtype=float)
        weights[idx] = solved
        return weights

    lam = max(float(risk_aversion), 1e-12)
    var = np.maximum(variances[idx], 1e-12)
    scores = np.zeros(idx.size, dtype=float)
    positive = expected_returns[idx] > 0.0
    scores[positive] = expected_returns[idx][positive] / (lam * var[positive])
    if float(scores.sum()) <= 0.0:
        solved = np.full(idx.size, 1.0 / idx.size, dtype=float)
    else:
        solved = scores / scores.sum()
    weights = np.zeros(n, dtype=float)
    weights[idx] = solved
    return weights


def _solve_mean_variance_covariance(
    expected_returns: np.ndarray,
    covariance: np.ndarray,
    *,
    risk_aversion: float = 1.0,
    eligible: np.ndarray,
    long_only: bool,
    max_weight: float | None,
    max_gross_exposure: float | None,
    min_weight: float,
    constraints: bool,
    max_iter: int = 1000,
    tol: float = 1e-9,
) -> np.ndarray:
    n = len(expected_returns)
    weights = np.zeros(n, dtype=float)
    idx = np.flatnonzero(eligible)
    if idx.size == 0:
        return weights

    mu = expected_returns[idx]
    cov = _ensure_positive_semidefinite(covariance[np.ix_(idx, idx)], floor=1e-8)
    if constraints or not long_only:
        solved = _projected_gradient_mean_variance(
            mu,
            cov,
            risk_aversion=risk_aversion,
            long_only=long_only,
            max_weight=max_weight,
            max_gross_exposure=max_gross_exposure,
            min_weight=min_weight,
            max_iter=max_iter,
            tol=tol,
        )
    elif long_only:
        solved = _projected_gradient_mean_variance(
            mu,
            cov,
            risk_aversion=risk_aversion,
            long_only=True,
            max_weight=None,
            max_gross_exposure=None,
            min_weight=0.0,
            max_iter=max_iter,
            tol=tol,
        )
    else:
        solved = _solve_budget_constrained_mean_variance(mu, cov, risk_aversion=risk_aversion)
    weights[idx] = solved
    return weights


def _solve_budget_constrained_mean_variance(
    expected_returns: np.ndarray,
    covariance: np.ndarray,
    *,
    risk_aversion: float,
) -> np.ndarray:
    """Solve max w'mu - (lambda/2) w'Sigma w s.t. sum(w) = 1, shorts allowed."""

    n = len(expected_returns)
    if n == 1:
        return np.array([1.0], dtype=float)

    lam = max(float(risk_aversion), 1e-12)
    ones = np.ones(n, dtype=float)
    inv_cov = np.linalg.pinv(covariance)
    inv_mu = inv_cov @ expected_returns
    inv_ones = inv_cov @ ones
    denom = float(ones @ inv_ones)
    if abs(denom) <= 1e-12:
        return ones / n

    nu = (float(ones @ inv_mu) - lam) / denom
    weights = (inv_mu - nu * inv_ones) / lam
    return weights


def _projected_gradient_mean_variance(
    expected_returns: np.ndarray,
    covariance: np.ndarray,
    *,
    risk_aversion: float,
    long_only: bool,
    max_weight: float | None,
    max_gross_exposure: float | None,
    min_weight: float,
    max_iter: int = 1000,
    tol: float = 1e-9,
) -> np.ndarray:
    n = len(expected_returns)
    if n == 1:
        return np.array([1.0], dtype=float)

    eligible = np.ones(n, dtype=bool)
    lam = max(float(risk_aversion), 1e-12)
    w = np.full(n, 1.0 / n, dtype=float)
    step = 0.25 / max(float(np.max(np.diag(covariance))), 1e-6)

    for iteration in range(max_iter):
        gradient = expected_returns - lam * (covariance @ w)
        proposal = _project_feasible_weights(
            w + step * gradient,
            eligible=eligible,
            long_only=long_only,
            max_weight=max_weight,
            max_gross_exposure=max_gross_exposure,
            min_weight=min_weight,
        )
        if float(np.linalg.norm(proposal - w, ord=1)) <= tol:
            w = proposal
            break
        w = proposal
        step *= 0.995 if iteration > 50 else 1.0

    return _project_feasible_weights(
        w,
        eligible=eligible,
        long_only=long_only,
        max_weight=max_weight,
        max_gross_exposure=max_gross_exposure,
        min_weight=min_weight,
    )


def _project_to_simplex(values: np.ndarray) -> np.ndarray:
    """Euclidean projection onto the probability simplex."""

    vector = np.asarray(values, dtype=float)
    if vector.size == 0:
        return vector
    sorted_values = np.sort(vector)[::-1]
    cumulative = np.cumsum(sorted_values)
    rho = np.nonzero(sorted_values * np.arange(1, vector.size + 1) > (cumulative - 1.0))[0]
    if rho.size == 0:
        return np.full(vector.size, 1.0 / vector.size, dtype=float)
    rho_idx = int(rho[-1])
    theta = (cumulative[rho_idx] - 1.0) / float(rho_idx + 1)
    projected = np.maximum(vector - theta, 0.0)
    total = float(projected.sum())
    if total <= 0.0:
        return np.full(vector.size, 1.0 / vector.size, dtype=float)
    return projected / total


def _ensure_positive_semidefinite(matrix: np.ndarray, *, floor: float) -> np.ndarray:
    arr = np.asarray(matrix, dtype=float)
    if arr.size == 0:
        return arr
    sym = 0.5 * (arr + arr.T)
    eigvals, eigvecs = np.linalg.eigh(sym)
    clipped = np.maximum(eigvals, float(floor))
    return eigvecs @ np.diag(clipped) @ eigvecs.T


def _resolve_return_variances(frame: pd.DataFrame, *, spec: OptionLabelSpec) -> np.ndarray:
    duration_days = int(frame["trade_duration_days"].iloc[0]) if "trade_duration_days" in frame.columns else 1
    duration_scale = max(duration_days, 1) / 252.0
    entry_quotes = frame["entry_quote"].to_numpy(dtype=float)
    variances = np.maximum(entry_quotes**2, spec.variance_floor)
    if "is_equity" in frame.columns:
        equity_mask = frame["is_equity"].astype(bool).to_numpy()
        equity_var = max(float(spec.equity_annual_vol), 1e-6) ** 2 * duration_scale
        variances = np.where(equity_mask, equity_var, variances)
    return variances


def _resolve_underlying_exit_price(trade: Mapping[str, Any], *, spec: OptionLabelSpec) -> float | None:
    for key in (spec.trade_exit_price_col, "exit_px", "exit_price"):
        value = _float_or_none(trade.get(key))
        if value is not None and value > 0.0:
            return value
    return None


def _resolve_underlying_entry_price(trade: Mapping[str, Any], *, spec: OptionLabelSpec) -> float | None:
    for key in (spec.trade_entry_price_col, "entry_px", "entry_price"):
        value = _float_or_none(trade.get(key))
        if value is not None and value > 0.0:
            return value
    return None


def _build_equity_candidate_row(
    trade: Mapping[str, Any],
    *,
    underlying_symbol: str,
    entry_dt: pd.Timestamp,
    exit_dt: pd.Timestamp,
    trade_id: str,
    entry_snapshot_date: pd.Timestamp | None,
    exit_snapshot_date: pd.Timestamp | None,
    spec: OptionLabelSpec,
) -> dict[str, Any] | None:
    entry_px = _resolve_underlying_entry_price(trade, spec=spec)
    exit_px = _resolve_underlying_exit_price(trade, spec=spec)
    if entry_px is None or exit_px is None or entry_px <= 0.0:
        return None

    contract_symbol = f"{underlying_symbol}{spec.equity_contract_suffix}"
    return {
        "contract_symbol": contract_symbol,
        "option_type": "equity",
        "expiration": pd.NaT,
        "strike": np.nan,
        "entry_quote": float(entry_px),
        "exit_quote": float(max(exit_px, 0.0)),
        "option_return_pct": float((exit_px - entry_px) / entry_px),
        "expires_worthless": False,
        "is_equity": True,
        "trade_id": trade_id,
        "trade_entry_date": entry_dt,
        "trade_exit_date": exit_dt,
        "trade_duration_days": int((exit_dt - entry_dt).days),
        "underlying_symbol": underlying_symbol,
        "underlying_return_pct": _float(trade.get("trade_return")),
        "entry_snapshot_date": entry_snapshot_date,
        "exit_snapshot_date": exit_snapshot_date,
    }


def _expires_worthless_mask(
    frame: pd.DataFrame,
    *,
    trade_exit_date: pd.Timestamp,
    underlying_exit_price: float | None,
    spec: OptionLabelSpec,
) -> pd.Series:
    exit_quotes = pd.to_numeric(frame["exit_quote"], errors="coerce").fillna(0.0)
    worthless = exit_quotes <= float(spec.worthless_exit_threshold)

    expiration_col = "expiration_exit" if "expiration_exit" in frame.columns else "expiration"
    strike_col = "strike_exit" if "strike_exit" in frame.columns else "strike"
    option_type_col = f"{spec.option_type_col}_exit" if f"{spec.option_type_col}_exit" in frame.columns else spec.option_type_col

    if expiration_col in frame.columns and strike_col in frame.columns and option_type_col in frame.columns:
        expirations = pd.to_datetime(frame[expiration_col], errors="coerce").dt.normalize()
        strikes = pd.to_numeric(frame[strike_col], errors="coerce")
        option_types = frame[option_type_col].astype(str).str.strip().str.lower()
        expired = expirations.notna() & (expirations <= trade_exit_date.normalize())
        if underlying_exit_price is not None and underlying_exit_price > 0.0:
            spot = float(underlying_exit_price)
            call_otm = option_types.str.startswith("c") & (strikes >= spot)
            put_otm = option_types.str.startswith("p") & (strikes <= spot)
            worthless = worthless | (expired & (call_otm | put_otm))
        else:
            worthless = worthless | expired

    return worthless.astype(bool)


def _postprocess_option_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    df = pd.DataFrame(rows).copy()
    if df.empty:
        return []
    id_cols = [col for col in ("trade_id", "contract_symbol", "option_type", "expiration", "strike") if col in df.columns]
    if "rank_y" in df.columns:
        df["rank_y"] = pd.to_numeric(df["rank_y"], errors="coerce")
    if "option_return_pct" in df.columns:
        df["option_return_pct"] = pd.to_numeric(df["option_return_pct"], errors="coerce")
    if "mv_mu" in df.columns:
        df["mv_mu"] = pd.to_numeric(df["mv_mu"], errors="coerce").fillna(0.0)
    if "mv_weight" in df.columns:
        df["mv_weight"] = pd.to_numeric(df["mv_weight"], errors="coerce").fillna(0.0)
    if "expires_worthless" in df.columns:
        df["expires_worthless"] = df["expires_worthless"].astype(bool)
    if "is_equity" in df.columns:
        df["is_equity"] = df["is_equity"].astype(bool)
    sort_cols = [col for col in ("trade_id", "rank_order", "option_return_pct") if col in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols, ascending=[True] * len(sort_cols))
    if id_cols:
        df = df.drop_duplicates(subset=id_cols, keep="first")
    return df.to_dict(orient="records")


def _build_option_statistics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"trade_stats": {}, "option_group_stats": []}
    df = pd.DataFrame(rows)
    stats = {
        "trade_stats": {
            "trades": int(df["trade_id"].nunique()) if "trade_id" in df.columns else 0,
            "contracts": int(len(df)),
            "avg_option_return_pct": round(float(pd.to_numeric(df["option_return_pct"], errors="coerce").mean() or 0.0) * 100.0, 4),
        },
        "option_group_stats": [],
    }
    if "mv_weight" in df.columns:
        stats["trade_stats"]["avg_mv_weight"] = round(float(pd.to_numeric(df["mv_weight"], errors="coerce").mean() or 0.0), 6)
        stats["trade_stats"]["worthless_contracts"] = int(df["expires_worthless"].sum()) if "expires_worthless" in df.columns else 0
    if "option_type" in df.columns:
        grouped = (
            df.groupby(["trade_id", "option_type"], dropna=False)["option_return_pct"]
            .agg(["count", "mean", "median"])
            .reset_index()
        )
        stats["option_group_stats"] = grouped.to_dict(orient="records")
    return stats


def _trade_id(trade: Mapping[str, Any], *, fallback: str | None = None) -> str:
    symbol = str(trade.get("symbol") or trade.get("underlying_symbol") or "").strip().upper()
    entry_date = _to_timestamp(trade.get("entry_date"))
    exit_date = _to_timestamp(trade.get("exit_date"))
    side = str(trade.get("side") or "").strip().lower()
    if symbol and entry_date is not None and exit_date is not None:
        return f"T|{symbol}|E{entry_date.date().isoformat()}|X{exit_date.date().isoformat()}|S{side or 'na'}"
    return fallback or "trade"


def _to_timestamp(value: Any) -> pd.Timestamp | None:
    if value is None or value == "":
        return None
    try:
        ts = pd.Timestamp(value)
    except Exception:
        return None
    if pd.isna(ts):
        return None
    return ts


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
    if pd.isna(parsed):
        return None
    return float(parsed)
