from pathlib import Path

import pandas as pd

from tdcmix.exact_primary_analysis import (
    build_exact_primary_analysis_report,
    build_exact_primary_analysis_status,
    build_exact_primary_artifact_comparison,
    write_exact_primary_analysis_outputs,
)
from tests.test_panel import _exact_sidecar_frame


def test_exact_primary_artifact_comparison_matches_identical_artifacts(tmp_path: Path) -> None:
    primary = tmp_path / "primary"
    sidecar = tmp_path / "sidecar"
    primary.mkdir()
    sidecar.mkdir()
    design = pd.DataFrame(
        {
            "quarter": ["2020-03-31"],
            "tdc_tier1": [1.0],
            "tdc_intensity": [0.1],
            "gross_marketable_issuance": [10.0],
            "bill_share": [0.2],
            "issue_wam": [50.0],
            "wam_gap_trailing": [1.0],
            "share_ru_like": [0.5],
            "share_private_ru_like": [0.25],
            "share_mmf_plumbing": [0.2],
            "share_domestic_final": [0.1],
            "share_bridge_financial": [0.05],
            "share_named_headline": [0.9],
            "share_residual": [0.1],
            "contrast_ru_minus_domestic": [0.4],
            "contrast_mmf_minus_banks": [0.1],
            "contrast_private_ru_minus_mmf": [0.05],
        }
    )
    matrices = pd.DataFrame(
        {
            "matrix_type": ["main_h0_h4"],
            "outcome": ["deposits"],
            "horizon": [0],
            "contrast": ["contrast_mmf_minus_banks"],
            "quarter": ["2020-03-31"],
            "y": [1.0],
            "holder_contrast": [0.1],
            "tdc_intensity": [0.1],
            "tdc_x_holder_contrast": [0.01],
        }
    )
    summary = pd.DataFrame(
        {
            "outcome": ["deposits"],
            "contrast": ["contrast_mmf_minus_banks"],
            "ok_cells": [1],
            "baseline_positive": [1],
            "baseline_negative": [0],
            "max_abs_t_stat": [2.0],
            "min_n": [40],
            "max_n": [40],
        }
    )
    stability = pd.DataFrame(
        {
            "outcome": ["deposits"],
            "contrast": ["contrast_mmf_minus_banks"],
            "stable_horizon_signs": [1],
            "unstable_horizon_signs": [0],
            "positive_cells": [5],
            "negative_cells": [0],
            "max_abs_t_stat": [2.5],
        }
    )
    placebo = pd.DataFrame(
        {
            "outcome": ["deposits"],
            "contrast": ["contrast_mmf_minus_banks"],
            "placebo_cells": [2],
            "concerning_placebo_cells": [0],
            "max_abs_placebo_t_stat": [1.0],
            "min_n": [40],
            "max_n": [40],
        }
    )
    for directory in [primary, sidecar]:
        design.to_csv(directory / "design_panel.csv", index=False)
        matrices.to_csv(directory / "horizon_design_matrices.csv", index=False)
        summary.to_csv(directory / "lp_diagnostic_summary.csv", index=False)
        stability.to_csv(directory / "lp_shortlist_stability_summary.csv", index=False)
        placebo.to_csv(directory / "pretrend_placebo_summary.csv", index=False)

    comparison = build_exact_primary_artifact_comparison(primary, sidecar)
    status = build_exact_primary_analysis_status(comparison, pd.DataFrame({"status": ["ready"]}))
    report = build_exact_primary_analysis_report(status, comparison)

    assert comparison["status"].eq("matches").all()
    assert status["status"].iloc[0] == "ready"
    assert "Exact-Primary Analysis Namespace" in report


def test_write_exact_primary_analysis_outputs(tmp_path: Path) -> None:
    panel = _exact_sidecar_frame().assign(holder_source_layer="z1_exact_fu_primary_migrated")
    panel_path = tmp_path / "holder_absorption_panel_exact_primary.csv"
    gate_path = tmp_path / "z1_exact_primary_migration_gate.csv"
    panel.to_csv(panel_path, index=False)
    pd.DataFrame({"status": ["ready"]}).to_csv(gate_path, index=False)

    paths = write_exact_primary_analysis_outputs(
        panel_path=panel_path,
        migration_gate_path=gate_path,
        exact_sidecar_analysis_dir=tmp_path / "missing_sidecar",
        output_dir=tmp_path / "exact_primary_analysis",
    )

    names = {path.name for path in paths}
    assert "exact_primary_artifact_comparison.csv" in names
    assert "exact_primary_analysis_status.csv" in names
    assert "design_panel.csv" in names
