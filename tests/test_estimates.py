from pathlib import Path

import pandas as pd

from tdcmix.contracts import load_source_contracts
from tdcmix.design import (
    build_design_panel,
    build_horizon_design_matrices,
    build_pretrend_placebo_matrices,
    build_stop_rule_status,
)
from tdcmix.estimates import (
    add_stop_rule_metadata,
    build_lp_diagnostic_report,
    build_lp_diagnostic_estimates,
    build_lp_diagnostic_shortlist,
    build_lp_diagnostic_summary,
    build_lp_profile_plot_data,
    build_lp_shortlist_economics_memo,
    build_lp_shortlist_economics_review,
    build_lp_shortlist_fullspec_estimates,
    build_lp_shortlist_fullspec_report,
    build_lp_shortlist_stability_summary,
    build_pretrend_placebo_estimates,
    build_pretrend_placebo_report,
    build_pretrend_placebo_summary,
    write_estimate_outputs,
)
from tdcmix.panel import build_holder_panel
from tdcmix.validation import (
    build_exact_residual_overlap_diagnostics,
    build_panel_validation_summary,
)
from tests.test_panel import _mix_config


def _inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
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
    return matrices, stop


def _inputs_with_placebo() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    panel = build_holder_panel(load_source_contracts(Path("config/source_contracts.yml")), _mix_config())
    design = build_design_panel(panel)
    matrices = build_horizon_design_matrices(design)
    placebo = build_pretrend_placebo_matrices(design)
    validation = build_panel_validation_summary(panel)
    readiness = __import__("tdcmix.design", fromlist=["build_design_readiness"]).build_design_readiness(
        panel,
        validation,
        build_exact_residual_overlap_diagnostics(panel),
    )
    first_stage = __import__("tdcmix.design", fromlist=["build_first_stage_summary"]).build_first_stage_summary(design)
    stop = build_stop_rule_status(validation, readiness, first_stage)
    return matrices, placebo, stop


def test_build_lp_diagnostic_estimates_runs_ok_cells() -> None:
    matrices, stop = _inputs()

    estimates = build_lp_diagnostic_estimates(matrices, stop)

    assert {"outcome", "horizon", "contrast", "sample", "beta", "status"}.issubset(estimates.columns)
    assert estimates["status"].eq("ok").any()
    assert estimates["sample"].isin(["baseline", "exclude_major_disruption", "with_regime_flags"]).all()


def test_build_lp_diagnostic_summary() -> None:
    matrices, stop = _inputs()
    estimates = build_lp_diagnostic_estimates(matrices, stop)

    summary = build_lp_diagnostic_summary(estimates)

    assert not summary.empty
    assert "diagnostic_status" in summary.columns


def test_shortlist_and_profile_plot_data() -> None:
    matrices, stop = _inputs()
    estimates = build_lp_diagnostic_estimates(matrices, stop)
    summary = build_lp_diagnostic_summary(estimates)

    shortlist = build_lp_diagnostic_shortlist(summary)
    plot_data = build_lp_profile_plot_data(estimates, shortlist)

    assert not shortlist.empty
    assert {"lower_approx_95", "upper_approx_95"}.issubset(plot_data.columns)


def test_shortlist_fullspec_and_stability_summary() -> None:
    matrices, stop = _inputs()
    estimates = build_lp_diagnostic_estimates(matrices, stop)
    summary = build_lp_diagnostic_summary(estimates)
    shortlist = build_lp_diagnostic_shortlist(summary)

    fullspec = build_lp_shortlist_fullspec_estimates(matrices, shortlist, stop)
    stability = build_lp_shortlist_stability_summary(fullspec)

    assert not fullspec.empty
    assert fullspec["status"].eq("ok").any()
    assert {
        "shortlist_baseline",
        "shortlist_exclude_major_disruption",
        "shortlist_regime_flags",
        "shortlist_qe_qt_controls",
        "shortlist_stress_controls",
    }.issubset(set(fullspec["fullspec_variant"]))
    assert not stability.empty
    assert {"stable_horizon_signs", "stability_status"}.issubset(stability.columns)


def test_stop_rule_metadata_and_economics_review() -> None:
    matrices, stop = _inputs()
    estimates = build_lp_diagnostic_estimates(matrices, stop)
    summary = build_lp_diagnostic_summary(estimates)
    shortlist = build_lp_diagnostic_shortlist(summary)
    fullspec = build_lp_shortlist_fullspec_estimates(matrices, shortlist, stop)
    stability = build_lp_shortlist_stability_summary(fullspec)

    annotated = add_stop_rule_metadata(fullspec, stop)
    review = build_lp_shortlist_economics_review(annotated, add_stop_rule_metadata(stability, stop), stop)
    memo = build_lp_shortlist_economics_memo(review)

    assert annotated["estimate_gate_status"].eq("diagnostic_estimation_allowed_non_residual_only").all()
    assert annotated["residual_sensitive_headline_status"].eq("blocked").all()
    assert not review.empty
    assert {"coefficient_reading", "manual_review_note"}.issubset(review.columns)
    assert "not evidence of a causal channel" in memo


def test_pretrend_placebo_screen_runs_on_shortlist() -> None:
    matrices, placebo, stop = _inputs_with_placebo()
    estimates = build_lp_diagnostic_estimates(matrices, stop)
    shortlist = build_lp_diagnostic_shortlist(build_lp_diagnostic_summary(estimates))

    placebo_estimates = build_pretrend_placebo_estimates(placebo, shortlist, stop)
    placebo_summary = build_pretrend_placebo_summary(placebo_estimates)
    report = build_pretrend_placebo_report(placebo_summary)

    assert not placebo_estimates.empty
    assert set(placebo_estimates["horizon"].dropna().unique()).issubset({-4, -3, -2, -1})
    assert {"concerning_placebo_cells", "pretrend_screen_status"}.issubset(placebo_summary.columns)
    assert "Pretrend Placebo Timing Screen" in report


def test_write_estimate_outputs(tmp_path: Path) -> None:
    matrices, placebo, stop = _inputs_with_placebo()
    matrices_path = tmp_path / "horizon_design_matrices.csv"
    placebo_path = tmp_path / "pretrend_placebo_matrices.csv"
    stop_path = tmp_path / "stop_rule_status.csv"
    matrices.to_csv(matrices_path, index=False)
    placebo.to_csv(placebo_path, index=False)
    stop.to_csv(stop_path, index=False)

    outputs = write_estimate_outputs(
        matrices_path=matrices_path,
        stop_rule_status_path=stop_path,
        pretrend_matrices_path=placebo_path,
        output_dir=tmp_path,
    )

    assert {path.name for path in outputs} == {
        "lp_diagnostic_estimates.csv",
        "lp_diagnostic_profile_plot_data.csv",
        "lp_diagnostic_report.md",
        "lp_diagnostic_shortlist.csv",
        "lp_diagnostic_summary.csv",
        "lp_shortlist_fullspec_estimates.csv",
        "lp_shortlist_fullspec_report.md",
        "lp_shortlist_economics_memo.md",
        "lp_shortlist_economics_review.csv",
        "lp_shortlist_stability_summary.csv",
        "pretrend_placebo_estimates.csv",
        "pretrend_placebo_report.md",
        "pretrend_placebo_summary.csv",
    }
    written_estimates = pd.read_csv(tmp_path / "lp_diagnostic_estimates.csv")
    assert "estimate_gate_status" in written_estimates.columns


def test_build_lp_diagnostic_report_marks_boundary() -> None:
    matrices, stop = _inputs()
    estimates = build_lp_diagnostic_estimates(matrices, stop)
    summary = build_lp_diagnostic_summary(estimates)

    report = build_lp_diagnostic_report(estimates, summary, stop)

    assert "diagnostic associations only" in report
    assert "Residual-sensitive headline claims: `blocked`" in report


def test_build_lp_shortlist_fullspec_report_marks_boundary() -> None:
    matrices, stop = _inputs()
    estimates = build_lp_diagnostic_estimates(matrices, stop)
    summary = build_lp_diagnostic_summary(estimates)
    shortlist = build_lp_diagnostic_shortlist(summary)
    fullspec = build_lp_shortlist_fullspec_estimates(matrices, shortlist, stop)
    stability = build_lp_shortlist_stability_summary(fullspec)

    report = build_lp_shortlist_fullspec_report(fullspec, stability, shortlist)

    assert "six-profile diagnostic shortlist" in report
    assert "not causal evidence" in report
