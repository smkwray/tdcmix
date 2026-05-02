from pathlib import Path

from tdcmix.contracts import load_source_contracts
from tdcmix.design import build_design_panel, build_horizon_design_matrices, build_stop_rule_status
from tdcmix.estimates import (
    build_lp_diagnostic_estimates,
    build_lp_diagnostic_shortlist,
    build_lp_diagnostic_summary,
    build_lp_profile_plot_data,
)
from tdcmix.figures import write_lp_diagnostic_profile_figure, write_stacked_holder_absorption_figure
from tdcmix.panel import build_holder_panel
from tdcmix.validation import build_exact_residual_overlap_diagnostics, build_panel_validation_summary
from tests.test_panel import _mix_config


def test_write_stacked_holder_absorption_figure(tmp_path: Path) -> None:
    panel = build_holder_panel(load_source_contracts(Path("config/source_contracts.yml")), _mix_config())
    output = write_stacked_holder_absorption_figure(panel, tmp_path / "figure.png")

    assert output.exists()
    assert output.stat().st_size > 0


def test_write_lp_diagnostic_profile_figure(tmp_path: Path) -> None:
    panel = build_holder_panel(load_source_contracts(Path("config/source_contracts.yml")), _mix_config())
    design = build_design_panel(panel)
    matrices = build_horizon_design_matrices(design)
    validation = build_panel_validation_summary(panel)
    readiness = __import__("tdcmix.design", fromlist=["build_design_readiness"]).build_design_readiness(
        panel,
        validation,
        build_exact_residual_overlap_diagnostics(panel),
    )
    first_stage = __import__("tdcmix.design", fromlist=["build_first_stage_summary"]).build_first_stage_summary(design)
    stop = build_stop_rule_status(validation, readiness, first_stage)
    estimates = build_lp_diagnostic_estimates(matrices, stop)
    shortlist = build_lp_diagnostic_shortlist(build_lp_diagnostic_summary(estimates))
    plot_data = build_lp_profile_plot_data(estimates, shortlist)

    output = write_lp_diagnostic_profile_figure(plot_data, tmp_path / "profiles.png")

    assert output.exists()
    assert output.stat().st_size > 0
