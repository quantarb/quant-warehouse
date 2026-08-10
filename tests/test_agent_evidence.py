import polars as pl
from datetime import datetime, timedelta

from quant_warehouse.export.agent_evidence import build_agent_evidence


class FakeWarehouse:
    class News:
        def ensure_date(self, *args, **kwargs):
            return pl.DataFrame()

    news = News()
    def read_prices(self, symbol, **kwargs):
        return pl.DataFrame({"date": [datetime(2026, 4, 1) + timedelta(days=i) for i in range(70)],
                             "close": range(100, 170), "open": range(99, 169), "volume": [1000] * 70})

    def read_fundamentals(self, symbol, *, section, **kwargs):
        return pl.DataFrame({"date": ["2026-03-31"], "value": [1.0]})

    def read_news(self, symbol, **kwargs):
        return pl.DataFrame({"published_at": ["2026-06-09T12:00:00"], "title": ["Headline"],
                             "source": ["FMP"], "excerpt": ["Evidence"]})


def test_build_agent_evidence_is_compact_and_point_in_time():
    packet = build_agent_evidence(FakeWarehouse(), "aapl", "2026-06-10")

    assert packet["symbol"] == "AAPL"
    assert packet["sufficient"] is True
    assert packet["price_summary"]["latest_date"] == "2026-06-09"
    assert "return_20d" in packet["price_summary"]
    assert packet["fundamentals"]["income"]["period"] == "2026-03-31"
    assert packet["news"][0]["title"] == "Headline"
    assert packet["roles"]["market"]["price_summary"]["close"] == 169.0
    assert "price_summary" not in packet["roles"]["fundamentals"]
