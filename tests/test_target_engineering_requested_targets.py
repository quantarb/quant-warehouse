from __future__ import annotations

import pandas as pd
import pytest

from quant_warehouse.platforms.data_providers.fmp.target_engineering import (
    EventPairStore,
    build_event_pairs_from_historical_data,
    fetch_fmp_event_pair_family,
    fetch_fmp_event_pairs,
    get_event_side,
    get_mirror_event_type,
    normalize_event_pairs,
)
from quant_warehouse.platforms.data_providers.thetadata.target_engineering import (
    build_option_best_return_labels,
    build_option_mean_variance_labels,
    build_option_return_rank_labels,
)
from quant_warehouse.platforms.data_providers.fmp.target_engineering import (
    build_cross_sectional_rank_labels,
    build_forward_return_labels,
)


class _FakeFundamentals:
    def __init__(self, frames: dict[str, pd.DataFrame]):
        self.frames = frames

    def read(self, symbol, *, section, provider="fmp", start=None, end=None):
        return self.frames.get(section, pd.DataFrame()).copy()


class _FakeBackend:
    kind = "arctic"

    def __init__(self):
        self.frames = {}

    def read(self, library, symbol):
        frame = self.frames.get((library, symbol))
        return None if frame is None else frame.copy()

    def write(self, library, symbol, df):
        self.frames[(library, symbol)] = df.copy()


class _FakeCatalog:
    def __init__(self):
        self.rows = []

    def upsert(self, **kwargs):
        self.rows.append(kwargs)


def test_build_forward_return_labels() -> None:
    prices = pd.DataFrame(
        {
            "symbol": ["AAPL", "AAPL", "AAPL"],
            "date": pd.date_range("2024-01-01", periods=3),
            "close": [100.0, 110.0, 121.0],
        }
    )

    labels = build_forward_return_labels(prices, [1, 2])

    one_day = labels[(labels["date"] == pd.Timestamp("2024-01-01")) & (labels["horizon"] == 1)].iloc[0]
    two_day = labels[(labels["date"] == pd.Timestamp("2024-01-01")) & (labels["horizon"] == 2)].iloc[0]
    assert round(one_day["target_value"], 6) == 0.10
    assert round(two_day["target_value"], 6) == 0.21
    assert one_day["target_name"] == "forward_return_1d"


def test_build_cross_sectional_rank_labels_direction() -> None:
    forward = pd.DataFrame(
        {
            "symbol": ["AAA", "BBB", "CCC"],
            "date": [pd.Timestamp("2024-01-01")] * 3,
            "horizon": [5, 5, 5],
            "target_name": ["forward_return_5d"] * 3,
            "target_value": [0.01, 0.10, -0.03],
        }
    )

    ranked = build_cross_sectional_rank_labels(forward)

    best = ranked.loc[ranked["symbol"] == "BBB"].iloc[0]
    worst = ranked.loc[ranked["symbol"] == "CCC"].iloc[0]
    assert best["rank"] == 1.0
    assert best["rank_pct"] == 1.0
    assert worst["rank_pct"] < best["rank_pct"]


def _option_returns() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "underlying_symbol": ["AAPL", "AAPL", "AAPL"],
            "date": [pd.Timestamp("2024-01-02")] * 3,
            "option_symbol": ["AAPL_C_100", "AAPL_C_110", "AAPL_P_90"],
            "entry_price": [1.0, 2.0, 4.0],
            "exit_price": [1.5, 2.2, 3.0],
            "option_type": ["call", "call", "put"],
            "strike": [100.0, 110.0, 90.0],
        }
    )


def test_build_option_best_return_labels_selects_best_contract() -> None:
    labels = build_option_best_return_labels(_option_returns())

    row = labels.iloc[0]
    assert row["best_option_symbol"] == "AAPL_C_100"
    assert row["best_option_return"] == 0.5
    assert row["target_value"] == 0.5
    assert row["option_type"] == "call"


def test_build_option_return_rank_labels_direction() -> None:
    labels = build_option_return_rank_labels(_option_returns())

    best = labels.loc[labels["option_symbol"] == "AAPL_C_100"].iloc[0]
    worst = labels.loc[labels["option_symbol"] == "AAPL_P_90"].iloc[0]
    assert best["option_return_rank"] == 1.0
    assert best["option_return_percentile"] == 1.0
    assert worst["option_return_percentile"] < best["option_return_percentile"]


def test_build_option_mean_variance_labels_score_rank_selection() -> None:
    candidates = pd.DataFrame(
        {
            "underlying_symbol": ["AAPL", "AAPL", "AAPL"],
            "date": [pd.Timestamp("2024-01-02")] * 3,
            "option_symbol": ["low", "best", "risky"],
            "expected_return": [0.04, 0.12, 0.20],
            "risk": [0.02, 0.03, 0.20],
        }
    )

    labels = build_option_mean_variance_labels(candidates, risk_aversion=1.0)

    best = labels.loc[labels["option_symbol"] == "best"].iloc[0]
    assert round(best["mv_score"], 6) == 0.09
    assert best["mv_rank"] == 1.0
    assert bool(best["mv_selected"])
    assert abs(float(labels["mv_weight"].sum()) - 1.0) < 1e-9
    assert (labels["target_value"] == labels["mv_weight"]).all()


def test_event_pair_mirror_lookup() -> None:
    assert get_mirror_event_type("congress", "congress_buy") == "congress_sell"
    assert get_mirror_event_type("analyst_rating", "analyst_downgrade") == "analyst_upgrade"
    assert get_event_side("institutional", "institutional_add") == 1
    assert get_event_side("institutional", "institutional_reduce") == -1


def test_normalize_event_pairs_exact_dates() -> None:
    raw = pd.DataFrame(
        {
            "ticker": ["aapl", "MSFT"],
            "event_dt": ["2024-01-03 14:30:00", "2024-01-04"],
            "kind": ["insider_buy", "insider_sell"],
            "actor": ["CEO", "CFO"],
            "person": ["Jane", "John"],
            "role": ["ceo", "cfo"],
            "firm": ["Unit Firm", "Unit Firm"],
            "shares": [10, 20],
            "price": [100.0, 50.0],
            "reported": ["2024-01-05", "2024-01-06"],
            "score": [0.8, 0.4],
            "payload": [{"id": 1}, {"id": 2}],
        }
    )

    normalized = normalize_event_pairs(
        raw,
        event_family="insider",
        event_type_col="kind",
        symbol_col="ticker",
        event_date_col="event_dt",
        source="unit",
        actor_type_col="actor",
        actor_name_col="person",
        actor_role_col="role",
        actor_firm_col="firm",
        transaction_shares_col="shares",
        transaction_price_col="price",
        reported_date_col="reported",
        strength_col="score",
        raw_json_col="payload",
    )

    assert list(normalized["symbol"]) == ["AAPL", "MSFT"]
    assert normalized.loc[0, "event_date"] == pd.Timestamp("2024-01-03")
    assert normalized.loc[0, "event_side"] == 1
    assert normalized.loc[0, "mirror_event_type"] == "insider_sell"
    assert normalized.loc[0, "actor_role"] == "ceo"
    assert normalized.loc[0, "actor_firm"] == "Unit Firm"
    assert normalized.loc[0, "transaction_shares"] == 10
    assert normalized.loc[0, "transaction_price"] == 100.0
    assert normalized.loc[0, "reported_date"] == pd.Timestamp("2024-01-05")
    assert normalized.loc[0, "disclosure_lag_days"] == 2
    assert normalized.loc[1, "event_side"] == -1
    assert "horizon" not in normalized.columns


def test_fetch_fmp_insider_event_pairs_uses_real_payload_shape() -> None:
    with pytest.raises(RuntimeError, match="Direct FMP event-pair fetches are disabled"):
        fetch_fmp_event_pair_family("AAPL", event_family="insider")


def test_fetch_fmp_congress_event_pairs_combines_house_and_senate() -> None:
    with pytest.raises(RuntimeError, match="Direct FMP event-pair fetches are disabled"):
        fetch_fmp_event_pairs("AAPL", event_families=("congress",))


def test_build_event_pairs_from_existing_historical_sections() -> None:
    fundamentals = _FakeFundamentals(
        {
            "ownership_insider_trading": pd.DataFrame(
                {
                    "symbol": ["AAPL", "AAPL"],
                    "transaction_date": ["2024-01-03", "2024-01-04"],
                    "transaction_type": ["P-Purchase", "S-Sale"],
                    "reporting_name": ["Jane CEO", "John CFO"],
                    "type_of_owner": ["officer", "officer"],
                    "officer_title": ["Chief Executive Officer", "Chief Financial Officer"],
                    "securities_transacted": [100, 50],
                    "price": [10.0, 20.0],
                    "filing_date": ["2024-01-05", "2024-01-06"],
                }
            )
        }
    )

    events = build_event_pairs_from_historical_data(
        "AAPL",
        fundamentals=fundamentals,
        event_families=("insider",),
    )

    assert list(events["event_type"]) == ["insider_buy", "insider_sell"]
    assert list(events["actor_role"]) == ["ceo", "cfo"]
    assert list(events["actor_title"]) == ["Chief Executive Officer", "Chief Financial Officer"]
    assert list(events["transaction_value"]) == [1000.0, 1000.0]
    assert list(events["disclosure_lag_days"]) == [2, 2]
    assert list(events["source"]) == [
        "warehouse:ownership_insider_trading",
        "warehouse:ownership_insider_trading",
    ]


def test_build_event_pairs_preserves_congress_chamber_and_analyst_firm() -> None:
    fundamentals = _FakeFundamentals(
        {
            "ownership_government_trades": pd.DataFrame(
                {
                    "symbol": ["AAPL", "AAPL"],
                    "transaction_date": ["2024-01-03", "2024-01-04"],
                    "transaction_type": ["Purchase", "Sale"],
                    "representative": ["Jane House", None],
                    "senator": [None, "John Senate"],
                    "chamber": ["House", "Senate"],
                    "disclosure_date": ["2024-01-10", "2024-01-12"],
                    "amount": ["$1,001 - $15,000", "$15,001 - $50,000"],
                }
            ),
            "estimates_price_target": pd.DataFrame(
                {
                    "symbol": ["AAPL", "AAPL"],
                    "date": ["2024-01-03", "2024-01-04"],
                    "action": ["upgraded", "downgraded"],
                    "grading_company": ["Firm A", "Firm B"],
                    "new_grade": ["Buy", "Sell"],
                }
            ),
        }
    )

    congress = build_event_pairs_from_historical_data(
        "AAPL",
        fundamentals=fundamentals,
        event_families=("congress",),
    )
    analyst = build_event_pairs_from_historical_data(
        "AAPL",
        fundamentals=fundamentals,
        event_families=("analyst_rating",),
    )

    assert list(congress["actor_chamber"]) == ["house", "senate"]
    assert list(congress["disclosure_lag_days"]) == [7, 8]
    assert list(analyst["actor_firm"]) == ["Firm A", "Firm B"]
    assert list(analyst["actor_role"]) == ["analyst", "analyst"]


def test_event_pair_store_uses_cached_labels_without_refetch(monkeypatch) -> None:
    store = EventPairStore(
        backend=_FakeBackend(),
        catalog=_FakeCatalog(),
        fundamentals=_FakeFundamentals({}),
        equity_calendar=None,
    )
    cached = pd.DataFrame(
        {
            "symbol": ["AAPL"],
            "event_date": [pd.Timestamp("2024-01-03")],
            "event_family": ["insider"],
            "event_type": ["insider_buy"],
            "event_side": [1],
            "mirror_event_type": ["insider_sell"],
            "actor_type": ["officer"],
            "actor_name": ["Jane CEO"],
            "source": ["unit"],
            "strength": [100],
            "raw_json": [{"id": 1}],
        }
    )
    store.ingest("AAPL", cached)

    result = store.load_or_refresh("AAPL", event_families=("insider",))

    assert result.source == "cache"
    assert len(result.frame) == 1
    assert result.frame.loc[0, "raw_json"] == {"id": 1}
