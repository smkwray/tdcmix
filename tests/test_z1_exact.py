from pathlib import Path

import pandas as pd

from tdcmix.z1_exact import (
    build_z1_exact_bucket_definition_contract,
    build_z1_exact_bucket_definition_report,
    build_z1_exact_bucket_migration_policy,
    build_z1_exact_bucket_migration_policy_report,
    build_z1_exact_gate_review,
    build_z1_exact_holder_panel,
    build_z1_exact_holder_share_diagnostics,
    build_z1_exact_primary_migration_gate,
    build_z1_exact_reconciliation,
    build_z1_exact_report,
    build_z1_exact_transaction_panel,
    load_z1_exact_config,
)


def _config() -> dict:
    return {
        "schema_version": "0.1.0",
        "defaults": {},
        "anchors": {
            "all_sectors": {
                "fred_id": "ALL",
                "fof_code": "FUALL.Q",
                "output_column": "exact_all_holder_transactions_bn",
                "description": "All sectors.",
            },
            "issuer_marketable": {
                "fred_id": "ISSUER",
                "fof_code": "FUISSUER.Q",
                "output_column": "issuer_marketable_treasury_liability_transactions_bn",
                "description": "Issuer.",
            },
        },
        "components": [
            {
                "key": "fed",
                "fred_id": "FED",
                "fof_code": "FUFED.Q",
                "bucket": "fed",
                "description": "Fed.",
            },
            {
                "key": "mmf",
                "fred_id": "MMF",
                "fof_code": "FUMMF.Q",
                "bucket": "mmf",
                "description": "MMF.",
            },
        ],
    }


def _fetcher(series_id: str) -> pd.DataFrame:
    values = {
        "ALL": [3000.0, 5000.0],
        "ISSUER": [3000.0, 5000.0],
        "FED": [1000.0, 2000.0],
        "MMF": [2000.0, 3000.0],
    }[series_id]
    return pd.DataFrame(
        {
            "observation_date": ["2020-01-01", "2020-04-01"],
            series_id: values,
        }
    )


def test_load_z1_exact_config() -> None:
    config = load_z1_exact_config(Path("config/z1_exact_transaction_components.yml"))

    assert config["anchors"]["all_sectors"]["fred_id"] == "BOGZ1FU893061105Q"
    assert any(component["bucket"] == "mmf" for component in config["components"])


def test_build_z1_exact_transaction_panel_and_gate() -> None:
    exact = build_z1_exact_transaction_panel(_config(), fetcher=_fetcher)
    holder = pd.DataFrame(
        {
            "quarter": ["2020-03-31", "2020-06-30"],
            "sector_tx_fed": [1.0, 2.0],
            "sector_tx_mmf": [2.0, 3.0],
            "sector_tx_unmapped": [0.0, 0.0],
        }
    )

    reconciliation = build_z1_exact_reconciliation(exact, holder)
    exact_holder = build_z1_exact_holder_panel(exact, holder)
    share_diag = build_z1_exact_holder_share_diagnostics(exact_holder)
    gate = build_z1_exact_gate_review(exact, reconciliation, min_exact_rows=2)
    report = build_z1_exact_report(gate, reconciliation)

    assert exact["z1_exact_official_unmapped_bn"].abs().max() == 0
    assert exact_holder["holder_source_layer"].eq("z1_exact_fu_transactions").all()
    assert share_diag["share_sum_passes"].all()
    assert gate["status"].iloc[0] == "ready"
    assert "Z.1 Exact Transaction Reconciliation" in report


def test_build_z1_exact_bucket_definition_contract_flags_review_buckets() -> None:
    config = load_z1_exact_config(Path("config/z1_exact_transaction_components.yml"))

    contract = build_z1_exact_bucket_definition_contract(config)
    report = build_z1_exact_bucket_definition_report(contract)

    assert contract["contract_name"].eq("tdcmix_z1_exact_bucket_definition_contract").all()
    assert {"banks", "domestic_nonbank", "other_financial"}.issubset(set(contract["bucket"]))
    banks = contract.loc[contract["bucket"].eq("banks")].iloc[0]
    domestic = contract.loc[contract["bucket"].eq("domestic_nonbank")].iloc[0]
    other_financial = contract.loc[contract["bucket"].eq("other_financial")].iloc[0]
    assert banks["primary_panel_migration_status"] == "requires_perimeter_freeze"
    assert domestic["primary_panel_migration_status"] == "requires_residual_holder_label"
    assert other_financial["primary_panel_migration_status"] == "requires_subbucket_review"
    assert not banks["behavioral_interpretation_allowed"]
    assert "residual-holder" in domestic["component_caveats"]
    assert "Z.1 Exact Bucket Definition Contract" in report


def test_build_z1_exact_bucket_migration_policy_resolves_review_flags() -> None:
    config = load_z1_exact_config(Path("config/z1_exact_transaction_components.yml"))
    contract = build_z1_exact_bucket_definition_contract(config)
    exact_gate = pd.DataFrame({"status": ["ready"]})

    policy = build_z1_exact_bucket_migration_policy(contract)
    gate = build_z1_exact_primary_migration_gate(exact_gate, contract, policy)
    report = build_z1_exact_bucket_migration_policy_report(policy, gate)

    assert set(policy["bucket"]) == {"banks", "domestic_nonbank", "other_financial"}
    assert policy["decision_status"].str.startswith("resolved").all()
    assert policy["primary_panel_migration_allowed"].all()
    assert not policy["behavioral_interpretation_allowed"].any()
    assert gate["status"].iloc[0] == "ready"
    assert gate["primary_panel_migration_allowed"].iloc[0]
    assert not gate["main_empirical_allowed"].iloc[0]
    assert not gate["residual_sensitive_headline_allowed"].iloc[0]
    assert "Z.1 Exact Bucket Migration Policy" in report
