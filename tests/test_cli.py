from pathlib import Path

import pandas as pd
import pytest

from tdcmix.cli import main


def test_cli_validate_contracts() -> None:
    assert main(["validate-contracts"]) == 0


def test_documented_exact_primary_pipeline() -> None:
    for command in (
        "write-panel", "write-validation", "write-design", "write-estimates",
        "write-z1-exact", "write-z1-exact-analysis", "write-exact-primary-panel",
        "write-exact-primary-analysis", "write-audits",
    ):
        assert main([command]) == 0

    directory = Path("data/processed/exact_primary_analysis")
    comparisons = pd.read_csv(directory / "exact_primary_artifact_comparison.csv")
    status = pd.read_csv(directory / "exact_primary_analysis_status.csv")
    assert comparisons["status"].eq("matches").all()
    assert status["status"].eq("ready").all()


def test_cli_writes_audits_and_panel_skeleton(tmp_path: Path) -> None:
    contracts = tmp_path / "source_contracts.yml"
    contracts.write_text(
        Path("config/source_contracts.yml")
        .read_text(encoding="utf-8")
        .replace(
            "data/processed/holder_absorption_panel.csv",
            str(tmp_path / "holder_absorption_panel.csv"),
        )
        .replace(
            "data/processed/source_overlap_audit.csv",
            str(tmp_path / "source_overlap_audit.csv"),
        )
        .replace(
            "data/processed/sector_map_audit.csv",
            str(tmp_path / "sector_map_audit.csv"),
        ),
        encoding="utf-8",
    )

    assert main(
        [
            "write-audits",
            "--contracts",
            str(contracts),
            "--z1-candidate-output",
            str(tmp_path / "z1_candidate_audit.csv"),
            "--source-upgrade-output",
            str(tmp_path / "source_upgrade_recommendations.csv"),
            "--source-upgrade-report",
            str(tmp_path / "source_upgrade_recommendations.md"),
        ]
    ) == 0
    assert main(
        [
            "write-panel-skeleton",
            "--contracts",
            str(contracts),
            "--quarter",
            "2020-03-31",
        ]
    ) == 0

    assert (tmp_path / "sector_map_audit.csv").exists()
    assert (tmp_path / "source_overlap_audit.csv").exists()
    assert (tmp_path / "z1_candidate_audit.csv").exists()
    assert (tmp_path / "source_upgrade_recommendations.csv").exists()
    assert (tmp_path / "source_upgrade_recommendations.md").exists()
    assert (tmp_path / "holder_absorption_panel.csv").exists()


def test_cli_writes_panel_from_contract_sources(tmp_path: Path) -> None:
    contracts = tmp_path / "source_contracts.yml"
    contracts.write_text(
        Path("config/source_contracts.yml")
        .read_text(encoding="utf-8")
        .replace(
            "data/processed/holder_absorption_panel.csv",
            str(tmp_path / "holder_absorption_panel.csv"),
        ),
        encoding="utf-8",
    )

    assert main(["write-panel", "--contracts", str(contracts)]) == 0
    assert (tmp_path / "holder_absorption_panel.csv").stat().st_size > 0


def test_cli_writes_exact_primary_panel(tmp_path: Path) -> None:
    contracts = tmp_path / "source_contracts.yml"
    contracts.write_text(
        Path("config/source_contracts.yml")
        .read_text(encoding="utf-8")
        .replace(
            "data/processed/holder_absorption_panel.csv",
            str(tmp_path / "holder_absorption_panel.csv"),
        ),
        encoding="utf-8",
    )
    main(["write-panel", "--contracts", str(contracts)])
    main(
        [
            "write-z1-exact",
            "--holder-panel",
            str(tmp_path / "holder_absorption_panel.csv"),
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert main(
        [
            "write-exact-primary-panel",
            "--contracts",
            str(contracts),
            "--legacy-panel",
            str(tmp_path / "holder_absorption_panel.csv"),
            "--exact-sidecar-panel",
            str(tmp_path / "z1_exact_holder_absorption_panel.csv"),
            "--migration-gate",
            str(tmp_path / "z1_exact_primary_migration_gate.csv"),
            "--output-dir",
            str(tmp_path),
        ]
    ) == 0
    assert (tmp_path / "holder_absorption_panel_exact_primary.csv").exists()
    assert (tmp_path / "exact_primary_migration_comparison.csv").exists()


def test_cli_writes_exact_primary_analysis(tmp_path: Path) -> None:
    contracts = tmp_path / "source_contracts.yml"
    contracts.write_text(
        Path("config/source_contracts.yml")
        .read_text(encoding="utf-8")
        .replace(
            "data/processed/holder_absorption_panel.csv",
            str(tmp_path / "holder_absorption_panel.csv"),
        ),
        encoding="utf-8",
    )
    main(["write-panel", "--contracts", str(contracts)])
    main(
        [
            "write-z1-exact",
            "--holder-panel",
            str(tmp_path / "holder_absorption_panel.csv"),
            "--output-dir",
            str(tmp_path),
        ]
    )
    main(
        [
            "write-exact-primary-panel",
            "--contracts",
            str(contracts),
            "--legacy-panel",
            str(tmp_path / "holder_absorption_panel.csv"),
            "--exact-sidecar-panel",
            str(tmp_path / "z1_exact_holder_absorption_panel.csv"),
            "--migration-gate",
            str(tmp_path / "z1_exact_primary_migration_gate.csv"),
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert main(
        [
            "write-exact-primary-analysis",
            "--panel",
            str(tmp_path / "holder_absorption_panel_exact_primary.csv"),
            "--migration-gate",
            str(tmp_path / "z1_exact_primary_migration_gate.csv"),
            "--exact-sidecar-analysis-dir",
            str(tmp_path / "missing_sidecar_analysis"),
            "--output-dir",
            str(tmp_path / "exact_primary_analysis"),
        ]
    ) == 0
    assert (tmp_path / "exact_primary_analysis" / "exact_primary_analysis_status.csv").exists()


def test_cli_writes_validation_outputs(tmp_path: Path) -> None:
    panel = tmp_path / "holder_absorption_panel.csv"
    contracts = tmp_path / "source_contracts.yml"
    contracts.write_text(
        Path("config/source_contracts.yml")
        .read_text(encoding="utf-8")
        .replace("data/processed/holder_absorption_panel.csv", str(panel)),
        encoding="utf-8",
    )
    main(["write-panel", "--contracts", str(contracts)])

    assert main(["write-validation", "--panel", str(panel), "--output-dir", str(tmp_path)]) == 0
    assert (tmp_path / "validation_summary.csv").exists()
    assert (tmp_path / "residual_diagnostics.csv").exists()


def test_cli_writes_design_outputs(tmp_path: Path) -> None:
    panel = tmp_path / "holder_absorption_panel.csv"
    contracts = tmp_path / "source_contracts.yml"
    contracts.write_text(
        Path("config/source_contracts.yml")
        .read_text(encoding="utf-8")
        .replace("data/processed/holder_absorption_panel.csv", str(panel)),
        encoding="utf-8",
    )
    main(["write-panel", "--contracts", str(contracts)])
    main(["write-validation", "--panel", str(panel), "--output-dir", str(tmp_path)])

    assert main(["write-design", "--panel", str(panel), "--validation-summary", str(tmp_path / "validation_summary.csv"), "--exact-residual-overlap", str(tmp_path / "exact_residual_overlap_diagnostics.csv"), "--output-dir", str(tmp_path)]) == 0
    assert (tmp_path / "design_panel.csv").exists()
    assert (tmp_path / "design_readiness.csv").exists()


def test_cli_writes_estimate_outputs(tmp_path: Path) -> None:
    panel = tmp_path / "holder_absorption_panel.csv"
    contracts = tmp_path / "source_contracts.yml"
    contracts.write_text(
        Path("config/source_contracts.yml")
        .read_text(encoding="utf-8")
        .replace("data/processed/holder_absorption_panel.csv", str(panel)),
        encoding="utf-8",
    )
    main(["write-panel", "--contracts", str(contracts)])
    main(["write-validation", "--panel", str(panel), "--output-dir", str(tmp_path)])
    main(["write-design", "--panel", str(panel), "--validation-summary", str(tmp_path / "validation_summary.csv"), "--exact-residual-overlap", str(tmp_path / "exact_residual_overlap_diagnostics.csv"), "--output-dir", str(tmp_path)])

    assert main(["write-estimates", "--matrices", str(tmp_path / "horizon_design_matrices.csv"), "--stop-rule-status", str(tmp_path / "stop_rule_status.csv"), "--output-dir", str(tmp_path)]) == 0
    assert (tmp_path / "lp_diagnostic_estimates.csv").exists()
    assert (tmp_path / "lp_diagnostic_summary.csv").exists()


def test_cli_writes_estimate_figures(tmp_path: Path) -> None:
    panel = tmp_path / "holder_absorption_panel.csv"
    contracts = tmp_path / "source_contracts.yml"
    contracts.write_text(
        Path("config/source_contracts.yml")
        .read_text(encoding="utf-8")
        .replace("data/processed/holder_absorption_panel.csv", str(panel)),
        encoding="utf-8",
    )
    main(["write-panel", "--contracts", str(contracts)])
    main(["write-validation", "--panel", str(panel), "--output-dir", str(tmp_path)])
    main(["write-design", "--panel", str(panel), "--validation-summary", str(tmp_path / "validation_summary.csv"), "--exact-residual-overlap", str(tmp_path / "exact_residual_overlap_diagnostics.csv"), "--output-dir", str(tmp_path)])
    main(["write-estimates", "--matrices", str(tmp_path / "horizon_design_matrices.csv"), "--stop-rule-status", str(tmp_path / "stop_rule_status.csv"), "--output-dir", str(tmp_path)])

    assert main(["write-estimate-figures", "--profile-plot-data", str(tmp_path / "lp_diagnostic_profile_plot_data.csv"), "--output", str(tmp_path / "profiles.png")]) == 0
    assert (tmp_path / "profiles.png").exists()


def test_cli_writes_priors(tmp_path: Path) -> None:
    panel = tmp_path / "holder_absorption_panel.csv"
    contracts = tmp_path / "source_contracts.yml"
    contracts.write_text(
        Path("config/source_contracts.yml")
        .read_text(encoding="utf-8")
        .replace("data/processed/holder_absorption_panel.csv", str(panel)),
        encoding="utf-8",
    )
    main(["write-panel", "--contracts", str(contracts)])

    assert main(["write-priors", "--panel", str(panel), "--output-dir", str(tmp_path)]) == 0
    assert (tmp_path / "holder_absorption_priors.csv").exists()
    assert (tmp_path / "holder_absorption_priors_report.md").exists()


def test_cli_writes_decision_outputs(tmp_path: Path) -> None:
    panel = tmp_path / "holder_absorption_panel.csv"
    contracts = tmp_path / "source_contracts.yml"
    contracts.write_text(
        Path("config/source_contracts.yml")
        .read_text(encoding="utf-8")
        .replace("data/processed/holder_absorption_panel.csv", str(panel)),
        encoding="utf-8",
    )
    main(["write-panel", "--contracts", str(contracts)])
    main(["write-validation", "--panel", str(panel), "--output-dir", str(tmp_path)])
    main(
        [
            "write-design",
            "--panel",
            str(panel),
            "--validation-summary",
            str(tmp_path / "validation_summary.csv"),
            "--exact-residual-overlap",
            str(tmp_path / "exact_residual_overlap_diagnostics.csv"),
            "--output-dir",
            str(tmp_path),
        ]
    )
    main(
        [
            "write-estimates",
            "--matrices",
            str(tmp_path / "horizon_design_matrices.csv"),
            "--stop-rule-status",
            str(tmp_path / "stop_rule_status.csv"),
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert main(
        [
            "write-decision",
            "--validation-summary",
            str(tmp_path / "validation_summary.csv"),
            "--design-readiness",
            str(tmp_path / "design_readiness.csv"),
            "--stop-rule-status",
            str(tmp_path / "stop_rule_status.csv"),
            "--stability-summary",
            str(tmp_path / "lp_shortlist_stability_summary.csv"),
            "--economics-review",
            str(tmp_path / "lp_shortlist_economics_review.csv"),
            "--output-dir",
            str(tmp_path),
        ]
    ) == 0
    assert (tmp_path / "project_decision_status.csv").exists()
    assert (tmp_path / "project_decision_report.md").exists()


def test_cli_writes_closeout_outputs(tmp_path: Path) -> None:
    (tmp_path / "project_decision_status.csv").write_text("recommendation\nmechanism_appendix_track\n", encoding="utf-8")
    exact_dir = tmp_path / "exact_primary_analysis"
    exact_dir.mkdir()
    (exact_dir / "exact_primary_analysis_status.csv").write_text("status\nready\n", encoding="utf-8")
    (exact_dir / "exact_primary_artifact_comparison.csv").write_text("status\nmatches\n", encoding="utf-8")
    (tmp_path / "exact_primary_migration_comparison.csv").write_text(
        "comparison,max_abs_gap\nexact_primary_vs_exact_sidecar,0\n",
        encoding="utf-8",
    )
    (tmp_path / "z1_exact_primary_migration_gate.csv").write_text(
        "status,primary_panel_migration_allowed,main_empirical_allowed,residual_sensitive_headline_allowed\n"
        "ready,True,False,False\n",
        encoding="utf-8",
    )
    (tmp_path / "exact_appendix_source_contract.csv").write_text("candidate_outcome\ndeposits\n", encoding="utf-8")
    (tmp_path / "exact_final_status_packet.csv").write_text(
        "main_empirical_slot_status,residual_sensitive_headline_status\n"
        "blocked_by_timing_and_diagnostic_instability,blocked\n",
        encoding="utf-8",
    )

    assert main(
        [
            "write-closeout",
            "--project-decision",
            str(tmp_path / "project_decision_status.csv"),
            "--exact-primary-analysis-status",
            str(exact_dir / "exact_primary_analysis_status.csv"),
            "--exact-primary-artifact-comparison",
            str(exact_dir / "exact_primary_artifact_comparison.csv"),
            "--exact-primary-migration-comparison",
            str(tmp_path / "exact_primary_migration_comparison.csv"),
            "--primary-migration-gate",
            str(tmp_path / "z1_exact_primary_migration_gate.csv"),
            "--appendix-source-contract",
            str(tmp_path / "exact_appendix_source_contract.csv"),
            "--final-status-packet",
            str(tmp_path / "exact_final_status_packet.csv"),
            "--output-dir",
            str(tmp_path),
        ]
    ) == 0
    assert (tmp_path / "downstream_default_source_decision.csv").exists()
    assert (tmp_path / "project_closeout_status.csv").exists()


def test_cli_writes_manifest_outputs(tmp_path: Path) -> None:
    data_dir = tmp_path / "data" / "processed"
    for name in [
        "holder_absorption_panel.csv",
        "holder_absorption_panel_exact_primary.csv",
        "exact_primary_migration_comparison.csv",
        "exact_primary_analysis/exact_primary_analysis_status.csv",
        "exact_primary_analysis/exact_primary_artifact_comparison.csv",
        "validation_summary.csv",
        "source_upgrade_recommendations.csv",
        "design_readiness.csv",
        "stop_rule_status.csv",
        "lp_diagnostic_summary.csv",
        "lp_shortlist_stability_summary.csv",
    ]:
        (data_dir / name).parent.mkdir(parents=True, exist_ok=True)
        (data_dir / name).write_text("x\n1\n", encoding="utf-8")
    (data_dir / "residual_gate_review.csv").write_text("status\nblocked\n", encoding="utf-8")
    (data_dir / "project_decision_status.csv").write_text(
        "recommendation\nmechanism_appendix_track\n",
        encoding="utf-8",
    )
    (data_dir / "holder_absorption_prior_contract.csv").write_text(
        "central_calibration_allowed,sensitivity_allowed\nTrue,True\nFalse,False\n",
        encoding="utf-8",
    )
    (data_dir / "holder_absorption_panel_exact_primary.csv").write_text(
        "holder_source_layer\nz1_exact_fu_primary_migrated\n",
        encoding="utf-8",
    )
    (data_dir / "exact_primary_migration_comparison.csv").write_text(
        "comparison,max_abs_gap\nexact_primary_vs_exact_sidecar,0\n",
        encoding="utf-8",
    )
    (data_dir / "exact_primary_analysis" / "exact_primary_analysis_status.csv").parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    (data_dir / "exact_primary_analysis" / "exact_primary_analysis_status.csv").write_text(
        "status\nready\n",
        encoding="utf-8",
    )
    (data_dir / "exact_primary_analysis" / "exact_primary_artifact_comparison.csv").write_text(
        "status\nmatches\nmatches\n",
        encoding="utf-8",
    )
    (data_dir / "downstream_default_source_decision.csv").write_text(
        "default_source_decision_status\ndefault_exact_primary_ready\n",
        encoding="utf-8",
    )
    (data_dir / "project_closeout_status.csv").write_text(
        "closeout_status\nbackend_complete_for_mechanism_appendix_consumers\n",
        encoding="utf-8",
    )

    assert main(["write-manifest", "--project-root", str(tmp_path), "--output-dir", str(data_dir)]) == 0
    assert (data_dir / "artifact_manifest.csv").exists()
    assert (data_dir / "build_status_summary.csv").exists()


def test_cli_writes_downstream_contract_bundle(tmp_path: Path) -> None:
    prior = tmp_path / "holder_absorption_prior_contract.csv"
    appendix = tmp_path / "exact_appendix_source_contract.csv"
    status = tmp_path / "exact_final_status_packet.csv"
    bucket = tmp_path / "z1_exact_bucket_definition_contract.csv"
    prior.write_text(
        "schema_version,contract_name,bucket,downstream_use,central_calibration_allowed,sensitivity_allowed,contract_note\n"
        "0.1.0,tdcmix_holder_absorption_prior_contract,banks,central_candidate_after_source_review,True,True,Central candidate.\n",
        encoding="utf-8",
    )
    appendix.write_text(
        "schema_version,contract_name,source_layer,appendix_source_promoted,main_empirical_allowed,residual_sensitive_headline_allowed,candidate_outcome,candidate_contrast,recommended_use,required_caveat\n"
        "0.1.0,tdcmix_exact_fu_appendix_source_contract,z1_exact_fu_transactions,True,False,False,deposits,contrast_mmf_minus_banks,mechanism_appendix_diagnostic_only,Diagnostic only.\n",
        encoding="utf-8",
    )
    status.write_text(
        "main_empirical_slot_status,residual_sensitive_headline_status\n"
        "blocked_by_timing_and_diagnostic_instability,blocked\n",
        encoding="utf-8",
    )
    bucket.write_text(
        "schema_version,contract_name,bucket,source_layer,primary_panel_migration_status,manual_review_required,definition_note\n"
        "0.1.0,tdcmix_z1_exact_bucket_definition_contract,banks,z1_exact_fu_transactions,requires_perimeter_freeze,True,Broad depository perimeter.\n",
        encoding="utf-8",
    )

    assert main(
        [
            "write-downstream-contracts",
            "--prior-contract",
            str(prior),
            "--appendix-source-contract",
            str(appendix),
            "--final-status-packet",
            str(status),
            "--bucket-definition-contract",
            str(bucket),
            "--output-dir",
            str(tmp_path),
        ]
    ) == 0
    assert (tmp_path / "downstream_contract_bundle.csv").exists()
    assert (tmp_path / "downstream_contract_bundle_summary.csv").exists()


def test_cli_has_z1_exact_command() -> None:
    with pytest.raises(SystemExit) as exc:
        main(["write-z1-exact", "--help"])
    assert exc.value.code == 0


def test_cli_has_z1_exact_analysis_command() -> None:
    with pytest.raises(SystemExit) as exc:
        main(["write-z1-exact-analysis", "--help"])
    assert exc.value.code == 0


def test_cli_has_downstream_contracts_command() -> None:
    with pytest.raises(SystemExit) as exc:
        main(["write-downstream-contracts", "--help"])
    assert exc.value.code == 0


def test_cli_has_closeout_command() -> None:
    with pytest.raises(SystemExit) as exc:
        main(["write-closeout", "--help"])
    assert exc.value.code == 0


def test_cli_has_exact_primary_panel_command() -> None:
    with pytest.raises(SystemExit) as exc:
        main(["write-exact-primary-panel", "--help"])
    assert exc.value.code == 0


def test_cli_has_exact_primary_analysis_command() -> None:
    with pytest.raises(SystemExit) as exc:
        main(["write-exact-primary-analysis", "--help"])
    assert exc.value.code == 0
