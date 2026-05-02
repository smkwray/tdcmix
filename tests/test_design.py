from pathlib import Path

import pandas as pd

from tdcmix.contracts import load_source_contracts
from tdcmix.design import build_design_panel, build_design_readiness, write_design_outputs
from tdcmix.design import (
    build_first_stage_summary,
    build_horizon_design_matrices,
    build_horizon_matrix_summary,
    build_pretrend_placebo_matrices,
    build_stop_rule_status,
)
from tdcmix.panel import build_holder_panel
from tdcmix.validation import (
    build_exact_residual_overlap_diagnostics,
    build_panel_validation_summary,
)
from tests.test_panel import _mix_config


def _panel() -> pd.DataFrame:
    return build_holder_panel(load_source_contracts(Path("config/source_contracts.yml")), _mix_config())


def test_build_design_panel_has_core_contrasts_and_horizons() -> None:
    design = build_design_panel(_panel())

    assert "contrast_ru_minus_domestic" in design.columns
    assert "contrast_mmf_minus_banks" in design.columns
    assert "deposits_h4" in design.columns
    assert "deposits_h-1" in design.columns
    assert "flag_major_disruption" in design.columns
    assert design["contrast_ru_minus_domestic"].notna().sum() > 40


def test_build_design_readiness_gates_residual_sensitive_path() -> None:
    panel = _panel()
    readiness = build_design_readiness(
        panel,
        build_panel_validation_summary(panel),
        build_exact_residual_overlap_diagnostics(panel),
    )

    readiness_by_design = dict(zip(readiness["design"], readiness["ready"], strict=True))
    assert readiness_by_design["baseline_non_residual_contrasts"]
    assert not readiness_by_design["residual_sensitive_contrasts"]


def test_write_design_outputs(tmp_path: Path) -> None:
    panel = _panel()
    validation = build_panel_validation_summary(panel)
    residual = build_exact_residual_overlap_diagnostics(panel)
    validation_path = tmp_path / "validation_summary.csv"
    residual_path = tmp_path / "exact_residual_overlap_diagnostics.csv"
    validation.to_csv(validation_path, index=False)
    residual.to_csv(residual_path, index=False)

    outputs = write_design_outputs(
        panel,
        validation_summary_path=validation_path,
        exact_residual_overlap_path=residual_path,
        output_dir=tmp_path,
    )

    assert {path.name for path in outputs} == {
        "design_panel.csv",
        "design_readiness.csv",
        "first_stage_summary.csv",
        "horizon_design_matrices.csv",
        "horizon_design_matrix_summary.csv",
        "pretrend_placebo_matrices.csv",
        "pretrend_placebo_matrix_summary.csv",
        "stop_rule_status.csv",
    }


def test_build_first_stage_summary_has_ok_rows() -> None:
    summary = build_first_stage_summary(build_design_panel(_panel()))

    assert {"outcome", "treatment", "sample", "n", "beta", "status"}.issubset(summary.columns)
    assert summary["status"].eq("ok").any()


def test_build_horizon_design_matrices_and_summary() -> None:
    matrices = build_horizon_design_matrices(build_design_panel(_panel()))
    summary = build_horizon_matrix_summary(matrices)

    assert {"outcome", "horizon", "contrast", "y", "tdc_x_holder_contrast"}.issubset(matrices.columns)
    assert matrices["matrix_type"].eq("main_h0_h4").all()
    assert not summary.empty
    assert summary["usable_rows"].max() > 40


def test_build_pretrend_placebo_matrices() -> None:
    matrices = build_pretrend_placebo_matrices(build_design_panel(_panel()))

    assert set(matrices["horizon"].unique()) == {-4, -3, -2, -1}
    assert matrices["matrix_type"].eq("pretrend_placebo").all()
    assert matrices[["y", "holder_contrast"]].dropna().shape[0] > 40


def test_build_stop_rule_status_marks_non_residual_path_ready() -> None:
    panel = _panel()
    validation = build_panel_validation_summary(panel)
    residual = build_exact_residual_overlap_diagnostics(panel)
    readiness = build_design_readiness(panel, validation, residual)
    first_stage = build_first_stage_summary(build_design_panel(panel))

    status = build_stop_rule_status(validation, readiness, first_stage)

    assert status["overall_non_residual_path_status"].eq("ready_for_diagnostic_estimation").all()
    residual_rule = status.loc[status["stop_rule"].eq("residual_not_allowed_as_headline")].iloc[0]
    assert bool(residual_rule["passes"])
