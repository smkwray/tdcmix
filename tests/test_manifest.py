from pathlib import Path

import pandas as pd

from tdcmix.manifest import (
    build_artifact_manifest,
    build_artifact_manifest_report,
    build_build_status_summary,
    write_artifact_manifest_outputs,
)


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def test_build_artifact_manifest_reads_statuses(tmp_path: Path) -> None:
    _write_csv(tmp_path / "gate.csv", pd.DataFrame({"status": ["blocked"]}))
    _write_csv(tmp_path / "decision.csv", pd.DataFrame({"recommendation": ["mechanism_appendix_track"]}))
    _write_csv(
        tmp_path / "contract.csv",
        pd.DataFrame(
            {
                "central_calibration_allowed": [True, False],
                "sensitivity_allowed": [True, False],
            }
        ),
    )

    manifest = build_artifact_manifest(
        project_root=tmp_path,
        generated_at="2026-04-28T00:00:00+00:00",
        artifacts=[
            ("residual_gate_review", "gate.csv", "validation"),
            ("project_decision_status", "decision.csv", "decision"),
            ("holder_absorption_prior_contract", "contract.csv", "downstream_contract"),
        ],
    )
    summary = build_build_status_summary(manifest)
    report = build_artifact_manifest_report(manifest, summary)

    assert manifest["exists"].all()
    assert set(manifest["status"]) == {
        "blocked",
        "mechanism_appendix_track",
        "central_candidates_1_blocked_1",
    }
    assert summary["build_status"].iloc[0] == "complete_with_blocked_headline_paths"
    assert "z1_exact_accounting_status" in summary.columns
    assert "Build Artifact Manifest" in report


def test_write_artifact_manifest_outputs(tmp_path: Path) -> None:
    data_dir = tmp_path / "data" / "processed"
    for name in [
        "holder_absorption_panel.csv",
        "holder_absorption_panel_exact_primary.csv",
        "exact_primary_migration_comparison.csv",
        "exact_primary_analysis/exact_primary_analysis_status.csv",
        "exact_primary_analysis/exact_primary_artifact_comparison.csv",
        "validation_summary.csv",
        "z1_exact_residual_reconciliation.csv",
        "z1_exact_holder_absorption_panel.csv",
        "z1_exact_holder_share_diagnostics.csv",
        "z1_exact_bucket_definition_contract.csv",
        "z1_exact_bucket_migration_policy.csv",
        "z1_exact_primary_migration_gate.csv",
        "z1_exact_analysis/design_readiness.csv",
        "z1_exact_analysis/source_layer_contrast_comparison.csv",
        "z1_exact_analysis/source_layer_estimate_comparison.csv",
        "z1_exact_analysis/source_layer_stability_comparison.csv",
        "z1_exact_analysis/source_layer_promotion_status.csv",
        "z1_exact_analysis/exact_appendix_candidate_triage.csv",
        "z1_exact_analysis/exact_mechanism_appendix_candidate_table.csv",
        "z1_exact_analysis/exact_candidate_sample_concentration.csv",
        "z1_exact_analysis/exact_candidate_regime_summary.csv",
        "z1_exact_analysis/exact_candidate_sensitivity_summary.csv",
        "z1_exact_analysis/exact_appendix_source_contract.csv",
        "z1_exact_analysis/exact_final_status_packet.csv",
        "downstream_default_source_decision.csv",
        "project_closeout_status.csv",
        "downstream_contract_bundle.csv",
        "downstream_contract_bundle_summary.csv",
        "source_upgrade_recommendations.csv",
        "design_readiness.csv",
        "stop_rule_status.csv",
        "lp_diagnostic_summary.csv",
        "lp_shortlist_stability_summary.csv",
        "pretrend_placebo_summary.csv",
        "z1_exact_analysis/pretrend_placebo_summary.csv",
    ]:
        _write_csv(data_dir / name, pd.DataFrame({"x": [1]}))
    _write_csv(
        data_dir / "pretrend_placebo_summary.csv",
        pd.DataFrame({"pretrend_screen_status": ["timing_concern", "passes_timing_screen"]}),
    )
    _write_csv(
        data_dir / "holder_absorption_panel_exact_primary.csv",
        pd.DataFrame({"holder_source_layer": ["z1_exact_fu_primary_migrated"]}),
    )
    _write_csv(
        data_dir / "exact_primary_migration_comparison.csv",
        pd.DataFrame({"comparison": ["exact_primary_vs_exact_sidecar"], "max_abs_gap": [0.0]}),
    )
    _write_csv(data_dir / "exact_primary_analysis/exact_primary_analysis_status.csv", pd.DataFrame({"status": ["ready"]}))
    _write_csv(
        data_dir / "exact_primary_analysis/exact_primary_artifact_comparison.csv",
        pd.DataFrame({"status": ["matches", "matches"]}),
    )
    _write_csv(
        data_dir / "z1_exact_analysis/pretrend_placebo_summary.csv",
        pd.DataFrame({"pretrend_screen_status": ["passes_timing_screen"]}),
    )
    _write_csv(
        data_dir / "z1_exact_analysis/source_layer_promotion_status.csv",
        pd.DataFrame({"recommendation": ["promote_exact_fu_for_mechanism_appendix_source_layer"]}),
    )
    _write_csv(
        data_dir / "z1_exact_analysis/exact_appendix_candidate_triage.csv",
        pd.DataFrame({"appendix_candidate_status": ["appendix_candidate", "hold_timing_concern"]}),
    )
    _write_csv(
        data_dir / "z1_exact_analysis/exact_mechanism_appendix_candidate_table.csv",
        pd.DataFrame({"outcome": ["deposits"], "contrast": ["contrast_mmf_minus_banks"], "horizon": [0]}),
    )
    _write_csv(
        data_dir / "z1_exact_analysis/exact_candidate_sample_concentration.csv",
        pd.DataFrame({"top5_abs_interaction_share": [0.2, 0.6]}),
    )
    _write_csv(
        data_dir / "z1_exact_analysis/exact_candidate_regime_summary.csv",
        pd.DataFrame({"regime": ["full_sample"]}),
    )
    _write_csv(
        data_dir / "z1_exact_analysis/exact_candidate_sensitivity_summary.csv",
        pd.DataFrame({"sign_robustness_status": ["sign_robust_positive", "sign_mixed_under_exclusions"]}),
    )
    _write_csv(
        data_dir / "z1_exact_analysis/exact_appendix_source_contract.csv",
        pd.DataFrame(
            {
                "appendix_source_promoted": [True],
                "candidate_outcome": ["deposits"],
                "candidate_contrast": ["contrast_mmf_minus_banks"],
            }
        ),
    )
    _write_csv(
        data_dir / "z1_exact_analysis/exact_final_status_packet.csv",
        pd.DataFrame({"main_empirical_slot_status": ["blocked_by_timing_and_diagnostic_instability"]}),
    )
    _write_csv(
        data_dir / "downstream_default_source_decision.csv",
        pd.DataFrame({"default_source_decision_status": ["default_exact_primary_ready"]}),
    )
    _write_csv(
        data_dir / "project_closeout_status.csv",
        pd.DataFrame({"closeout_status": ["backend_complete_for_mechanism_appendix_consumers"]}),
    )
    _write_csv(
        data_dir / "downstream_contract_bundle.csv",
        pd.DataFrame(
            {
                "central_calibration_allowed": [True, False],
                "mechanism_appendix_allowed": [False, True],
            }
        ),
    )
    _write_csv(
        data_dir / "downstream_contract_bundle_summary.csv",
        pd.DataFrame({"bundle_status": ["ready_for_appendix_consumers_with_blocked_headline_paths"]}),
    )
    _write_csv(data_dir / "residual_gate_review.csv", pd.DataFrame({"status": ["blocked"]}))
    _write_csv(data_dir / "z1_exact_gate_review.csv", pd.DataFrame({"status": ["ready"]}))
    _write_csv(
        data_dir / "z1_exact_bucket_definition_contract.csv",
        pd.DataFrame(
            {
                "manual_review_required": [True, False],
                "primary_panel_migration_status": ["requires_perimeter_freeze", "ready_after_identity_gate"],
            }
        ),
    )
    _write_csv(
        data_dir / "z1_exact_bucket_migration_policy.csv",
        pd.DataFrame(
            {
                "decision_status": ["resolved_by_broad_depository_perimeter"],
                "primary_panel_migration_allowed": [True],
            }
        ),
    )
    _write_csv(data_dir / "z1_exact_primary_migration_gate.csv", pd.DataFrame({"status": ["ready"]}))
    _write_csv(data_dir / "project_decision_status.csv", pd.DataFrame({"recommendation": ["mechanism_appendix_track"]}))
    _write_csv(
        data_dir / "holder_absorption_prior_contract.csv",
        pd.DataFrame(
            {
                "central_calibration_allowed": [True],
                "sensitivity_allowed": [True],
            }
        ),
    )

    paths = write_artifact_manifest_outputs(
        project_root=tmp_path,
        output_dir=data_dir,
        generated_at="2026-04-28T00:00:00+00:00",
    )

    assert {path.name for path in paths} == {
        "artifact_manifest.csv",
        "artifact_manifest_report.md",
        "build_status_summary.csv",
    }
    summary = pd.read_csv(data_dir / "build_status_summary.csv")
    assert summary["pretrend_placebo_status"].iloc[0] == "timing_concerns_1"
    assert summary["exact_primary_panel_status"].iloc[0] == "z1_exact_fu_primary_migrated_rows_1"
    assert summary["exact_primary_migration_comparison_status"].iloc[0] == "sidecar_match_rows_1"
    assert summary["exact_primary_analysis_status"].iloc[0] == "ready"
    assert summary["exact_primary_artifact_comparison_status"].iloc[0] == "artifact_comparisons_2_matches_2_differs_0"
    assert summary["z1_exact_bucket_definition_status"].iloc[0] == "bucket_definitions_2_review_1_ready_1"
    assert summary["z1_exact_bucket_migration_policy_status"].iloc[0] == "migration_policies_1_resolved_1_allowed_1"
    assert summary["z1_exact_primary_migration_status"].iloc[0] == "ready"
    assert summary["z1_exact_pretrend_placebo_status"].iloc[0] == "timing_screen_passes"
    assert (
        summary["z1_exact_source_layer_promotion"].iloc[0]
        == "promote_exact_fu_for_mechanism_appendix_source_layer"
    )
    assert summary["z1_exact_appendix_candidate_status"].iloc[0] == "appendix_candidates_1_held_1"
    assert summary["z1_exact_mechanism_appendix_status"].iloc[0] == "mechanism_profiles_1_horizons_1"
    assert summary["z1_exact_candidate_concentration_status"].iloc[0] == "concentration_rows_2_high_top5_1"
    assert summary["z1_exact_candidate_sensitivity_status"].iloc[0] == "sensitivity_rows_2_mixed_1"
    assert (
        summary["z1_exact_appendix_source_contract_status"].iloc[0]
        == "appendix_source_promoted_true_candidates_1"
    )
    assert summary["z1_exact_final_backend_status"].iloc[0] == "blocked_by_timing_and_diagnostic_instability"
    assert summary["downstream_default_source_status"].iloc[0] == "default_exact_primary_ready"
    assert summary["project_closeout_status"].iloc[0] == "backend_complete_for_mechanism_appendix_consumers"
    assert (
        summary["downstream_contract_bundle_status"].iloc[0]
        == "ready_for_appendix_consumers_with_blocked_headline_paths"
    )
