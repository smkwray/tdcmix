from pathlib import Path

import pandas as pd
import pytest

from tdcmix.audits import (
    build_sector_map_audit,
    build_source_overlap_audit,
    build_source_upgrade_recommendations,
    build_source_upgrade_recommendations_report,
    build_z1_candidate_audit,
)
from tdcmix.contracts import load_source_contracts
from tdcmix.schema import load_sector_map


def test_sector_map_audit_covers_all_buckets() -> None:
    sector_map = load_sector_map(Path("config/holder_sector_map.yml"))
    contracts = load_source_contracts(Path("config/source_contracts.yml"))

    audit = build_sector_map_audit(sector_map, contracts)

    assert sorted(audit["bucket"]) == sorted(sector_map["buckets"])
    assert (audit["status"] != "missing_contract").all()


@pytest.mark.parametrize("source_kind", ["synthetic", pytest.param("retained", marks=pytest.mark.integration)])
def test_source_overlap_audit_reports_existing_sources(source_kind: str) -> None:
    contracts = load_source_contracts(Path("config/source_contracts.yml"))

    audit = build_source_overlap_audit(contracts)

    tdc_components = audit.loc[audit["key"] == "sector_tx_banks"].iloc[0]
    assert bool(tdc_components["exists"])
    assert tdc_components["missing_columns"] == ""
    assert tdc_components["row_count"] > 0


def test_z1_candidate_audit_maps_local_candidate_schemas(tmp_path: Path) -> None:
    holdings = tmp_path / "z1_holdings_long.csv"
    transactions = tmp_path / "sector_transaction_groups.csv"
    series_panel = tmp_path / "z1_series_panel_full.csv"
    pd.DataFrame(
        {
            "date": ["2024-03-31", "2024-06-30"],
            "sector": ["money_market_funds", "money_market_funds"],
            "instrument": ["all_treasuries", "all_treasuries"],
            "holdings": [100.0, 110.0],
        }
    ).to_csv(holdings, index=False)
    pd.DataFrame(
        {
            "date": ["2024-03-31", "2024-06-30", "2024-03-31", "2024-06-30"],
            "sector": ["Money funds", "Money funds", "State/local governments", "State/local governments"],
            "transactions_bn": [1.0, 2.0, 3.0, 4.0],
        }
    ).to_csv(transactions, index=False)
    pd.DataFrame(
        {
            "date": ["2024-03-31", "2024-06-30"],
            "series_key": ["asset_backed_securities_issuers_treasuries", "asset_backed_securities_issuers_treasuries"],
            "description": ["ABS issuer Treasury securities", "ABS issuer Treasury securities"],
            "level": [3.0, 4.0],
            "transactions": [0.1, 0.2],
            "revaluation": [0.0, 0.0],
        }
    ).to_csv(series_panel, index=False)

    audit = build_z1_candidate_audit(
        project_root=tmp_path,
        sources=[
            {
                "source_project": "test",
                "source_path": "z1_holdings_long.csv",
                "source_kind": "long_holdings",
                "date_column": "date",
            },
            {
                "source_project": "test",
                "source_path": "sector_transaction_groups.csv",
                "source_kind": "grouped_transactions",
                "date_column": "date",
            },
            {
                "source_project": "test",
                "source_path": "z1_series_panel_full.csv",
                "source_kind": "series_panel",
                "date_column": "date",
            },
        ],
    )

    assert {"holder_bucket_candidate", "review_grade", "non_null_observation_count"}.issubset(audit.columns)
    assert "mmf" in set(audit["holder_bucket_candidate"])
    assert "other_financial" in set(audit["holder_bucket_candidate"])
    assert "domestic_nonbank" in set(audit["holder_bucket_candidate"])
    assert audit["non_null_observation_count"].min() == 2


@pytest.mark.parametrize("source_kind", ["synthetic", pytest.param("retained", marks=pytest.mark.integration)])
def test_source_upgrade_recommendations_keep_grouped_mmf_and_other_financial(source_kind: str) -> None:
    contracts = load_source_contracts(Path("config/source_contracts.yml"))
    candidates = build_z1_candidate_audit()

    recommendations = build_source_upgrade_recommendations(candidates, contracts)
    report = build_source_upgrade_recommendations_report(recommendations)

    mmf = recommendations.loc[recommendations["holder_bucket"].eq("mmf")].iloc[0]
    other_financial = recommendations.loc[recommendations["holder_bucket"].eq("other_financial")].iloc[0]
    assert mmf["recommended_action"] == "keep_grouped_flow_primary_validate_with_series_panel"
    assert other_financial["recommended_action"] == "keep_grouped_flow_primary_validate_with_series_panel"
    assert "Recommendations do not edit source contracts automatically" in report
