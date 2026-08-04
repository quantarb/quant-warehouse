from quant_warehouse.platforms.data_providers.fmp.related_assets import (
    classify_related_security,
    parse_related_maturity_date,
)


def test_classify_related_security_uses_fmp_names_and_symbol_suffixes() -> None:
    assert classify_related_security("Acme Preferred Stock", "ACME-P", "ACME") == "preferred"
    assert classify_related_security("Acme Warrant", "ACME-W", "ACME") == "warrant"
    assert classify_related_security("Acme Senior Note", "ACME-N", "ACME") == "note_bond"
    assert classify_related_security("Acme Corporation", "ACME", "ACME") is None


def test_parse_related_maturity_date() -> None:
    assert parse_related_maturity_date("Acme warrant expiring 01/02/2027") == "2027-01-02"
    assert parse_related_maturity_date("Acme common stock") is None
