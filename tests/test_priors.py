from pathlib import Path

from tdcmix.contracts import load_source_contracts
from tdcmix.panel import build_holder_panel
from tdcmix.priors import (
    build_downstream_prior_contract,
    build_downstream_prior_contract_report,
    build_holder_absorption_priors,
    build_holder_absorption_priors_report,
    write_holder_absorption_priors,
)
from tdcmix.schema import load_sector_map
from tests.test_panel import _mix_config


def _panel_and_map():
    panel = build_holder_panel(load_source_contracts(Path("config/source_contracts.yml")), _mix_config())
    sector_map = load_sector_map(Path("config/holder_sector_map.yml"))
    return panel, sector_map


def test_build_holder_absorption_priors_covers_all_buckets() -> None:
    panel, sector_map = _panel_and_map()

    priors = build_holder_absorption_priors(panel, sector_map)

    assert sorted(priors["bucket"]) == sorted(sector_map["buckets"])
    assert {"share_median", "calibration_status", "prior_use_note"}.issubset(priors.columns)
    assert priors.loc[priors["bucket"].eq("unmapped"), "calibration_status"].iloc[0] == (
        "blocked_residual_sensitivity_only"
    )
    assert priors["share_observation_count"].max() > 0


def test_holder_absorption_priors_report_keeps_boundary() -> None:
    panel, sector_map = _panel_and_map()
    priors = build_holder_absorption_priors(panel, sector_map)

    report = build_holder_absorption_priors_report(priors)

    assert "descriptive inputs" in report
    assert "Residual and other-financial priors" in report


def test_downstream_prior_contract_splits_central_sensitivity_and_blocked() -> None:
    panel, sector_map = _panel_and_map()
    priors = build_holder_absorption_priors(panel, sector_map)

    contract = build_downstream_prior_contract(priors)
    report = build_downstream_prior_contract_report(contract)

    assert {"downstream_use", "central_calibration_allowed", "sensitivity_allowed"}.issubset(
        contract.columns
    )
    assert contract["schema_version"].eq("0.1.0").all()
    assert contract["contract_name"].eq("tdcmix_holder_absorption_prior_contract").all()
    assert contract.loc[contract["bucket"].eq("banks"), "central_calibration_allowed"].iloc[0]
    assert not contract.loc[contract["bucket"].eq("other_financial"), "central_calibration_allowed"].iloc[0]
    assert contract.loc[contract["bucket"].eq("other_financial"), "sensitivity_allowed"].iloc[0]
    assert not contract.loc[contract["bucket"].eq("unmapped"), "sensitivity_allowed"].iloc[0]
    assert "central calibration candidates" in report
    assert "Schema version" in report


def test_write_holder_absorption_priors(tmp_path: Path) -> None:
    panel, sector_map = _panel_and_map()

    paths = write_holder_absorption_priors(panel, sector_map, tmp_path)

    assert {path.name for path in paths} == {
        "holder_absorption_prior_contract.csv",
        "holder_absorption_prior_contract.md",
        "holder_absorption_priors.csv",
        "holder_absorption_priors_report.md",
    }
    assert (tmp_path / "holder_absorption_priors.csv").stat().st_size > 0
