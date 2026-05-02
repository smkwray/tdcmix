from pathlib import Path

import pytest

from tdcmix.contracts import load_source_contracts, panel_columns, validate_source_contracts


CONTRACT_PATH = Path("config/source_contracts.yml")


def test_source_contracts_load() -> None:
    contracts = load_source_contracts(CONTRACT_PATH)

    assert "sector_tx_banks" in panel_columns(contracts)
    assert "mmf_treasury" in panel_columns(contracts)


def test_source_contracts_require_unique_output_columns() -> None:
    contracts = load_source_contracts(CONTRACT_PATH)
    contracts["sources"][1]["output_column"] = contracts["sources"][0]["output_column"]

    with pytest.raises(ValueError, match="mapped more than once"):
        validate_source_contracts(contracts)


def test_source_contracts_reject_revaluation_leakage() -> None:
    contracts = load_source_contracts(CONTRACT_PATH)
    contracts["sources"][3]["source_columns"] = ["fed_tsy_revaluation"]

    with pytest.raises(ValueError, match="revaluation"):
        validate_source_contracts(contracts)


def test_source_contracts_require_all_holder_buckets() -> None:
    contracts = load_source_contracts(CONTRACT_PATH)
    for record in contracts["sources"]:
        if record.get("holder_bucket") == "fed":
            record.pop("holder_bucket")
            break

    with pytest.raises(ValueError, match="missing buckets"):
        validate_source_contracts(contracts)
