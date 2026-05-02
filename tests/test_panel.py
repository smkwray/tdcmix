from pathlib import Path

import pandas as pd
import pytest
import yaml

from tdcmix.contracts import load_source_contracts
from tdcmix.panel import (
    add_holder_absorption_shares,
    build_exact_fu_primary_holder_panel,
    build_exact_fu_primary_migration_comparison,
    build_exact_fu_primary_migration_report,
    build_holder_panel,
    build_holder_panel_skeleton,
    write_exact_fu_primary_panel_outputs,
)


def _mix_config() -> dict:
    with Path("config/holder_mix_variables.yml").open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def test_holder_panel_skeleton_has_contract_and_share_columns() -> None:
    contracts = load_source_contracts(Path("config/source_contracts.yml"))

    panel = build_holder_panel_skeleton(contracts, _mix_config(), quarters=["2020-03-31"])

    assert panel.columns[0] == "quarter"
    assert "sector_tx_mmf" in panel.columns
    assert "positive_absorption_total" in panel.columns
    assert "pos_abs_share_unmapped" in panel.columns
    assert panel.loc[0, "quarter"] == "2020-03-31"


def test_add_holder_absorption_shares_uses_configured_buckets() -> None:
    mix_config = _mix_config()
    frame = pd.DataFrame(
        {
            "quarter": ["2020-03-31"],
            "sector_tx_fed": [1.0],
            "sector_tx_banks": [1.0],
            "sector_tx_row": [-1.0],
            "sector_tx_mmf": [2.0],
            "sector_tx_domestic_nonbank": [0.0],
            "sector_tx_dealer_bridge": [1.0],
            "sector_tx_other_financial": [0.0],
            "sector_tx_unmapped": [0.0],
        }
    )

    out = add_holder_absorption_shares(frame, mix_config)

    assert out.loc[0, "positive_absorption_total"] == pytest.approx(5.0)
    assert out.loc[0, "pos_abs_share_mmf"] == pytest.approx(0.4)


def test_build_holder_panel_from_contract_sources() -> None:
    contracts = load_source_contracts(Path("config/source_contracts.yml"))

    panel = build_holder_panel(contracts, _mix_config())

    assert not panel.empty
    assert "tdc_tier1" in panel.columns
    assert "sector_tx_unmapped" in panel.columns
    assert "pos_abs_share_mmf" in panel.columns
    assert panel["sector_tx_fed"].notna().any()
    assert panel["sector_tx_unmapped"].notna().any()
    assert panel["sector_tx_unmapped"].notna().sum() > 100
    assert panel["gross_marketable_issuance"].notna().any()


def test_build_exact_fu_primary_holder_panel_requires_ready_gate() -> None:
    contracts = load_source_contracts(Path("config/source_contracts.yml"))
    mix = _mix_config()
    exact = _exact_sidecar_frame()

    out = build_exact_fu_primary_holder_panel(
        exact,
        pd.DataFrame({"status": ["ready"]}),
        contracts,
        mix,
    )

    assert out["holder_source_layer"].eq("z1_exact_fu_primary_migrated").all()
    assert "sector_tx_mmf" in out.columns
    assert "pos_abs_share_mmf" in out.columns
    assert "tdc_tier1" in out.columns

    with pytest.raises(ValueError, match="migration gate is not ready"):
        build_exact_fu_primary_holder_panel(
            exact,
            pd.DataFrame({"status": ["blocked"]}),
            contracts,
            mix,
        )


def test_exact_fu_primary_migration_comparison_matches_sidecar() -> None:
    contracts = load_source_contracts(Path("config/source_contracts.yml"))
    mix = _mix_config()
    exact = _exact_sidecar_frame()
    primary = build_exact_fu_primary_holder_panel(
        exact,
        pd.DataFrame({"status": ["ready"], "primary_panel_migration_allowed": [True]}),
        contracts,
        mix,
    )

    comparison = build_exact_fu_primary_migration_comparison(exact, exact, primary, mix)
    report = build_exact_fu_primary_migration_report(
        comparison,
        pd.DataFrame(
            {
                "status": ["ready"],
                "primary_panel_migration_allowed": [True],
                "main_empirical_allowed": [False],
                "residual_sensitive_headline_allowed": [False],
            }
        ),
    )

    sidecar = comparison.loc[comparison["comparison"].eq("exact_primary_vs_exact_sidecar")]
    assert not sidecar.empty
    assert pd.to_numeric(sidecar["max_abs_gap"], errors="coerce").max() == 0
    assert "Exact-FU Primary Panel Migration" in report


def test_write_exact_fu_primary_panel_outputs(tmp_path: Path) -> None:
    contracts = load_source_contracts(Path("config/source_contracts.yml"))
    mix = _mix_config()
    legacy = _exact_sidecar_frame().assign(holder_source_layer="legacy")
    exact = _exact_sidecar_frame()
    gate = pd.DataFrame({"status": ["ready"], "primary_panel_migration_allowed": [True]})
    legacy_path = tmp_path / "legacy.csv"
    exact_path = tmp_path / "exact.csv"
    gate_path = tmp_path / "gate.csv"
    legacy.to_csv(legacy_path, index=False)
    exact.to_csv(exact_path, index=False)
    gate.to_csv(gate_path, index=False)

    paths = write_exact_fu_primary_panel_outputs(
        contracts_config=contracts,
        mix_config=mix,
        legacy_panel_path=legacy_path,
        exact_sidecar_panel_path=exact_path,
        migration_gate_path=gate_path,
        output_dir=tmp_path,
    )

    assert {path.name for path in paths} == {
        "exact_primary_migration_comparison.csv",
        "exact_primary_migration_report.md",
        "holder_absorption_panel_exact_primary.csv",
    }
    written = pd.read_csv(tmp_path / "holder_absorption_panel_exact_primary.csv")
    assert written["holder_source_layer"].iloc[0] == "z1_exact_fu_primary_migrated"


def _exact_sidecar_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "quarter": ["2020-03-31", "2020-06-30"],
            "holder_source_layer": ["z1_exact_fu_transactions", "z1_exact_fu_transactions"],
            "residual_interpretation": ["exact", "exact"],
            "sector_tx_fed": [1.0, 2.0],
            "sector_tx_banks": [1.0, 1.0],
            "sector_tx_row": [0.0, 1.0],
            "sector_tx_mmf": [2.0, 3.0],
            "sector_tx_domestic_nonbank": [0.0, 1.0],
            "sector_tx_dealer_bridge": [0.0, 0.0],
            "sector_tx_other_financial": [0.0, 0.0],
            "sector_tx_unmapped": [0.0, 0.0],
            "tdc_tier1": [10.0, 11.0],
            "gross_marketable_issuance": [100.0, 101.0],
            "bill_share": [0.2, 0.3],
            "issue_wam": [50.0, 51.0],
            "wam_gap_trailing": [1.0, 1.1],
            "toc_change": [0.1, 0.2],
            "fed_remit_positive": [1, 1],
            "deposits": [5.0, 6.0],
            "bank_credit": [7.0, 8.0],
            "reserves": [9.0, 10.0],
            "onrrp": [11.0, 12.0],
            "mmf_treasury": [13.0, 14.0],
            "positive_absorption_total": [4.0, 8.0],
            "pos_abs_share_fed": [0.25, 0.25],
            "pos_abs_share_banks": [0.25, 0.125],
            "pos_abs_share_row": [0.0, 0.125],
            "pos_abs_share_mmf": [0.5, 0.375],
            "pos_abs_share_domestic_nonbank": [0.0, 0.125],
            "pos_abs_share_dealer_bridge": [0.0, 0.0],
            "pos_abs_share_other_financial": [0.0, 0.0],
            "pos_abs_share_unmapped": [0.0, 0.0],
        }
    )
