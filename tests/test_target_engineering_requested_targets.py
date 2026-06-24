from __future__ import annotations

import pandas as pd

from quant_warehouse.target_engineering import (
    build_cross_sectional_rank_labels,
    build_forward_return_labels,
    build_optimal_trade_labels,
    build_option_best_return_labels,
    build_option_mean_variance_labels,
    build_option_return_rank_labels,
    build_event_pairs_from_historical_data,
    EventPairStore,
    fetch_fmp_event_pair_family,
    fetch_fmp_event_pairs,
    get_event_side,
    get_mirror_event_type,
    normalize_event_pairs,
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


def test_build_optimal_trade_labels_tiny_path() -> None:
    prices = pd.DataFrame(
        {
            "symbol": ["AAPL"] * 4,
            "date": pd.date_range("2024-01-01", periods=4),
            "close": [10.0, 9.5, 12.0, 7.0],
        }
    )

    labels = build_optimal_trade_labels(prices, [2], allow_short=True)
    first = labels.loc[labels["date"] == pd.Timestamp("2024-01-01")].iloc[0]
    third = labels.loc[labels["date"] == pd.Timestamp("2024-01-03")].iloc[0]

    assert first["optimal_side"] == "long"
    assert round(first["optimal_return"], 6) == 0.20
    assert first["optimal_exit_date"] == pd.Timestamp("2024-01-03")
    assert third["optimal_side"] == "short"
    assert round(third["short_best_return"], 6) == round((12.0 / 7.0) - 1.0, 6)


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
        strength_col="score",
        raw_json_col="payload",
    )

    assert list(normalized["symbol"]) == ["AAPL", "MSFT"]
    assert normalized.loc[0, "event_date"] == pd.Timestamp("2024-01-03")
    assert normalized.loc[0, "event_side"] == 1
    assert normalized.loc[0, "mirror_event_type"] == "insider_sell"
    assert normalized.loc[1, "event_side"] == -1
    assert "horizon" not in normalized.columns


def test_fetch_fmp_insider_event_pairs_uses_real_payload_shape(monkeypatch) -> None:
    payloads = {
        "insider-trading/search": [
            {
                "symbol": "aapl",
                "transactionDate": "2024-01-03",
                "transactionType": "P-Purchase",
                "reportingName": "Jane CEO",
                "typeOfOwner": "officer",
                "securitiesTransacted": 100,
            },
            {
                "symbol": "AAPL",
                "transactionDate": "2024-01-04",
                "transactionType": "S-Sale",
                "reportingName": "John CFO",
                "typeOfOwner": "officer",
                "securitiesTransacted": 50,
            },
        ]
    }

    def fake_get_json(endpoint, *, params):
        assert params["symbol"] == "AAPL"
        return payloads[endpoint]

    monkeypatch.setattr(
        "quant_warehouse.target_engineering.event_pairs.fmp_fetch._fmp_get_json",
        fake_get_json,
    )

    events = fetch_fmp_event_pair_family("AAPL", event_family="insider")

    assert list(events["event_type"]) == ["insider_buy", "insider_sell"]
    assert list(events["event_side"]) == [1, -1]
    assert list(events["actor_name"]) == ["Jane CEO", "John CFO"]
    assert events.loc[0, "source"] == "fmp:insider-trading/search"
    assert events.loc[0, "raw_json"]["transactionType"] == "P-Purchase"


def test_fetch_fmp_congress_event_pairs_combines_house_and_senate(monkeypatch) -> None:
    payloads = {
        "senate-trades": [
            {
                "symbol": "AAPL",
                "transactionDate": "2024-02-01",
                "transactionType": "Purchase",
                "senator": "Senator One",
                "amount": "$1,001 - $15,000",
            }
        ],
        "house-trades": [
            {
                "symbol": "AAPL",
                "transactionDate": "2024-02-02",
                "transactionType": "Sale (Full)",
                "representative": "Rep Two",
                "amount": "$15,001 - $50,000",
            }
        ],
    }

    def fake_get_json(endpoint, *, params):
        assert params["symbol"] == "AAPL"
        return payloads[endpoint]

    monkeypatch.setattr(
        "quant_warehouse.target_engineering.event_pairs.fmp_fetch._fmp_get_json",
        fake_get_json,
    )

    events = fetch_fmp_event_pairs("AAPL", event_families=("congress",))

    assert list(events["event_type"]) == ["congress_buy", "congress_sell"]
    assert list(events["event_side"]) == [1, -1]
    assert list(events["actor_type"]) == ["senate", "house"]
    assert list(events["mirror_event_type"]) == ["congress_sell", "congress_buy"]


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
                    "securities_transacted": [100, 50],
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
    assert list(events["source"]) == [
        "warehouse:ownership_insider_trading",
        "warehouse:ownership_insider_trading",
    ]


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

    def fail_fetch(*args, **kwargs):
        raise AssertionError("FMP should not be called when cached event pairs cover the request")

    monkeypatch.setattr(
        "quant_warehouse.target_engineering.event_pairs.store.fetch_fmp_event_pair_family",
        fail_fetch,
    )

    result = store.load_or_refresh("AAPL", event_families=("insider",))

    assert result.source == "cache"
    assert len(result.frame) == 1
    assert result.frame.loc[0, "raw_json"] == {"id": 1}
