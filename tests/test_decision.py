from pathlib import Path

import pandas as pd

from tdcmix.decision import (
    build_downstream_default_source_decision,
    build_downstream_default_source_decision_report,
    build_project_decision,
    build_project_closeout_report,
    build_project_closeout_status,
    build_project_decision_report,
    write_project_closeout_outputs,
    write_project_decision_outputs,
)


def _inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    validation = pd.DataFrame(
        {
            "check": ["exact_residual_overlap_rows"],
            "value": [16],
            "passes": [True],
        }
    )
    readiness = pd.DataFrame(
        {
            "design": [
                "baseline_non_residual_contrasts",
                "h0_h4_outcome_scaffold",
                "residual_sensitive_contrasts",
            ],
            "ready": [True, True, False],
        }
    )
    stop = pd.DataFrame(
        {
            "stop_rule": ["holder_panel_reconciles_enough_for_non_residual_work"],
            "passes": [True],
            "overall_non_residual_path_status": ["ready_for_diagnostic_estimation"],
        }
    )
    stability = pd.DataFrame(
        {
            "stability_status": [
                "mixed_but_reviewable_diagnostic",
                "mixed_but_reviewable_diagnostic",
                "mixed_but_reviewable_diagnostic",
                "unstable_diagnostic",
            ]
        }
    )
    review = pd.DataFrame({"outcome": ["deposits", "reserves", "deposits"]})
    return validation, readiness, stop, stability, review


def test_build_project_decision_routes_to_mechanism_appendix() -> None:
    decision = build_project_decision(*_inputs())

    row = decision.iloc[0]
    assert row["recommendation"] == "mechanism_appendix_track"
    assert row["main_slot_status"] == "not_main_empirical_slot_yet"
    assert not row["residual_sensitive_ready"]


def test_project_decision_report_marks_boundary() -> None:
    decision = build_project_decision(*_inputs())

    report = build_project_decision_report(decision)

    assert "Project Decision Report" in report
    assert "project-routing decision" in report


def test_write_project_decision_outputs(tmp_path: Path) -> None:
    validation, readiness, stop, stability, review = _inputs()
    paths = {
        "validation": tmp_path / "validation_summary.csv",
        "readiness": tmp_path / "design_readiness.csv",
        "stop": tmp_path / "stop_rule_status.csv",
        "stability": tmp_path / "lp_shortlist_stability_summary.csv",
        "review": tmp_path / "lp_shortlist_economics_review.csv",
    }
    validation.to_csv(paths["validation"], index=False)
    readiness.to_csv(paths["readiness"], index=False)
    stop.to_csv(paths["stop"], index=False)
    stability.to_csv(paths["stability"], index=False)
    review.to_csv(paths["review"], index=False)

    outputs = write_project_decision_outputs(
        validation_summary_path=paths["validation"],
        design_readiness_path=paths["readiness"],
        stop_rule_status_path=paths["stop"],
        stability_summary_path=paths["stability"],
        economics_review_path=paths["review"],
        output_dir=tmp_path,
    )

    assert {path.name for path in outputs} == {
        "project_decision_report.md",
        "project_decision_status.csv",
    }


def test_build_downstream_default_source_decision_freezes_exact_primary() -> None:
    decision = build_downstream_default_source_decision(
        pd.DataFrame({"recommendation": ["mechanism_appendix_track"]}),
        pd.DataFrame({"status": ["ready"]}),
        pd.DataFrame({"status": ["matches", "matches"]}),
        pd.DataFrame({"comparison": ["exact_primary_vs_exact_sidecar"], "max_abs_gap": [0.0]}),
        pd.DataFrame(
            {
                "status": ["ready"],
                "primary_panel_migration_allowed": [True],
                "main_empirical_allowed": [False],
                "residual_sensitive_headline_allowed": [False],
            }
        ),
    )
    report = build_downstream_default_source_decision_report(decision)

    row = decision.iloc[0]
    assert row["default_source_decision_status"] == "default_exact_primary_ready"
    assert row["default_holder_panel"] == "data/processed/holder_absorption_panel_exact_primary.csv"
    assert row["legacy_holder_panel_role"] == "audit_comparison_only"
    assert row["primary_panel_migration_allowed"]
    assert not row["main_empirical_allowed"]
    assert "Downstream Default Source Decision" in report


def test_build_project_closeout_status_marks_backend_complete() -> None:
    default = pd.DataFrame({"default_source_decision_status": ["default_exact_primary_ready"]})
    project = pd.DataFrame({"recommendation": ["mechanism_appendix_track"]})
    exact_primary = pd.DataFrame({"status": ["ready"]})
    final_status = pd.DataFrame(
        {
            "main_empirical_slot_status": ["blocked_by_timing_and_diagnostic_instability"],
            "residual_sensitive_headline_status": ["blocked"],
        }
    )
    appendix = pd.DataFrame({"candidate_outcome": ["deposits", "reserves"]})

    closeout = build_project_closeout_status(default, project, exact_primary, final_status, appendix)
    report = build_project_closeout_report(closeout)

    assert closeout["closeout_status"].iloc[0] == "backend_complete_for_mechanism_appendix_consumers"
    assert closeout["remaining_backend_work"].iloc[0] == "none_for_mechanism_appendix_path"
    assert closeout["next_external_work"].iloc[0] == "consumer_integration_smoke_test"
    assert "Project Closeout Status" in report


def test_write_project_closeout_outputs(tmp_path: Path) -> None:
    paths = {
        "project": tmp_path / "project_decision_status.csv",
        "status": tmp_path / "exact_primary_analysis_status.csv",
        "artifacts": tmp_path / "exact_primary_artifact_comparison.csv",
        "migration": tmp_path / "exact_primary_migration_comparison.csv",
        "gate": tmp_path / "z1_exact_primary_migration_gate.csv",
        "appendix": tmp_path / "exact_appendix_source_contract.csv",
        "final": tmp_path / "exact_final_status_packet.csv",
    }
    pd.DataFrame({"recommendation": ["mechanism_appendix_track"]}).to_csv(paths["project"], index=False)
    pd.DataFrame({"status": ["ready"]}).to_csv(paths["status"], index=False)
    pd.DataFrame({"status": ["matches"]}).to_csv(paths["artifacts"], index=False)
    pd.DataFrame({"comparison": ["exact_primary_vs_exact_sidecar"], "max_abs_gap": [0.0]}).to_csv(
        paths["migration"], index=False
    )
    pd.DataFrame(
        {
            "status": ["ready"],
            "primary_panel_migration_allowed": [True],
            "main_empirical_allowed": [False],
            "residual_sensitive_headline_allowed": [False],
        }
    ).to_csv(paths["gate"], index=False)
    pd.DataFrame({"candidate_outcome": ["deposits"]}).to_csv(paths["appendix"], index=False)
    pd.DataFrame(
        {
            "main_empirical_slot_status": ["blocked_by_timing_and_diagnostic_instability"],
            "residual_sensitive_headline_status": ["blocked"],
        }
    ).to_csv(paths["final"], index=False)

    outputs = write_project_closeout_outputs(
        project_decision_path=paths["project"],
        exact_primary_analysis_status_path=paths["status"],
        exact_primary_artifact_comparison_path=paths["artifacts"],
        exact_primary_migration_comparison_path=paths["migration"],
        primary_migration_gate_path=paths["gate"],
        appendix_source_contract_path=paths["appendix"],
        final_status_packet_path=paths["final"],
        output_dir=tmp_path,
    )

    assert {path.name for path in outputs} == {
        "downstream_default_source_decision.csv",
        "downstream_default_source_decision.md",
        "project_closeout_status.csv",
        "project_closeout_report.md",
    }
