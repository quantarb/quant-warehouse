import pandas as pd

from quant_warehouse.platforms.data_providers.fmp.target_engineering.corporate_events import (
    CORPORATE_EVENT_COLUMNS,
    build_corporate_event_label_panel,
)


def test_build_corporate_event_labels_maps_symbol_delisting_and_ma_roles() -> None:
    events = [
        {"corporate_event_type": "symbol_change", "date": "2025-01-02", "oldSymbol": "OLD", "newSymbol": "NEW"},
        {"corporate_event_type": "delisted", "date": "2025-01-03", "symbol": "OLD"},
        {
            "corporate_event_type": "merger_acquisition",
            "transactionDate": "2025-01-04",
            "acquirerSymbol": "ACQ",
            "targetSymbol": "TGT",
        },
    ]
    panel = build_corporate_event_label_panel(
        pd.DataFrame({"date": pd.date_range("2025-01-02", periods=3)}),
        events,
        symbols=("OLD", "NEW", "ACQ", "TGT"),
    )
    assert set(CORPORATE_EVENT_COLUMNS).issubset(panel.columns)
    assert panel.loc[(panel.symbol == "OLD") & (panel.date == pd.Timestamp("2025-01-02")), "is_symbol_change"].iat[0] == 1.0
    assert panel.loc[(panel.symbol == "NEW") & (panel.date == pd.Timestamp("2025-01-02")), "is_symbol_change"].iat[0] == 1.0
    assert panel.loc[(panel.symbol == "ACQ") & (panel.date == pd.Timestamp("2025-01-04")), "is_ma_acquirer"].iat[0] == 1.0
    assert panel.loc[(panel.symbol == "TGT") & (panel.date == pd.Timestamp("2025-01-04")), "is_ma_target"].iat[0] == 1.0
