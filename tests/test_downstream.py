from pathlib import Path

import pandas as pd

from tdcmix.downstream import (
    build_downstream_contract_bundle,
    build_downstream_contract_bundle_report,
    build_downstream_contract_bundle_summary,
    write_downstream_contract_bundle_outputs,
)


def test_downstream_contract_bundle_combines_priors_and_exact_appendix_contract() -> None:
    bundle = build_downstream_contract_bundle(
        _prior_contract(),
        _appendix_contract(),
        _final_status(),
        _bucket_contract(),
        _migration_policy(),
        _primary_migration_gate(),
        _default_source_decision(),
    )
    summary = build_downstream_contract_bundle_summary(bundle)
    report = build_downstream_contract_bundle_report(bundle, summary)

    assert len(bundle) == 9
    assert summary["bundle_status"].iloc[0] == "ready_for_appendix_consumers_with_blocked_headline_paths"
    assert summary["central_calibration_rows"].iloc[0] == 1
    assert summary["mechanism_appendix_rows"].iloc[0] == 2
    assert summary["main_empirical_rows"].iloc[0] == 0
    assert summary["residual_sensitive_headline_rows"].iloc[0] == 0
    assert summary["primary_panel_migration_rows"].iloc[0] == 4
    appendix = bundle.loc[bundle["contract_row_type"].eq("exact_fu_mechanism_appendix")].iloc[0]
    buckets = bundle.loc[bundle["contract_row_type"].eq("exact_fu_bucket_definition")]
    assert appendix["object_key"] == "deposits/contrast_mmf_minus_banks"
    assert appendix["mechanism_appendix_allowed"]
    assert not appendix["main_empirical_allowed"]
    assert "blocked_by_timing_and_diagnostic_instability" in appendix["blocked_reason"]
    assert set(buckets["object_key"]) == {"banks", "mmf"}
    assert buckets.loc[buckets["object_key"].eq("banks"), "blocked_reason"].iloc[0]
    assert bundle.loc[
        bundle["contract_row_type"].eq("exact_fu_primary_migration_gate"),
        "primary_panel_migration_allowed",
    ].iloc[0]
    default = bundle.loc[bundle["contract_row_type"].eq("downstream_default_source_decision")].iloc[0]
    assert default["source_layer"] == "z1_exact_fu_primary_migrated"
    assert default["default_holder_panel"] == "data/processed/holder_absorption_panel_exact_primary.csv"
    assert "Downstream Contract Bundle" in report
    assert "preserves exact-FU appendix use" in report


def test_write_downstream_contract_bundle_outputs(tmp_path: Path) -> None:
    prior_path = tmp_path / "holder_absorption_prior_contract.csv"
    appendix_path = tmp_path / "exact_appendix_source_contract.csv"
    status_path = tmp_path / "exact_final_status_packet.csv"
    _prior_contract().to_csv(prior_path, index=False)
    _appendix_contract().to_csv(appendix_path, index=False)
    _final_status().to_csv(status_path, index=False)
    bucket_path = tmp_path / "z1_exact_bucket_definition_contract.csv"
    _bucket_contract().to_csv(bucket_path, index=False)
    policy_path = tmp_path / "z1_exact_bucket_migration_policy.csv"
    gate_path = tmp_path / "z1_exact_primary_migration_gate.csv"
    _migration_policy().to_csv(policy_path, index=False)
    _primary_migration_gate().to_csv(gate_path, index=False)
    default_path = tmp_path / "downstream_default_source_decision.csv"
    _default_source_decision().to_csv(default_path, index=False)

    paths = write_downstream_contract_bundle_outputs(
        prior_contract_path=prior_path,
        appendix_source_contract_path=appendix_path,
        final_status_packet_path=status_path,
        bucket_definition_contract_path=bucket_path,
        bucket_migration_policy_path=policy_path,
        primary_migration_gate_path=gate_path,
        default_source_decision_path=default_path,
        output_dir=tmp_path,
    )

    assert {path.name for path in paths} == {
        "downstream_contract_bundle.csv",
        "downstream_contract_bundle.md",
        "downstream_contract_bundle_summary.csv",
    }
    summary = pd.read_csv(tmp_path / "downstream_contract_bundle_summary.csv")
    assert summary["mechanism_appendix_rows"].iloc[0] == 2
    assert summary["primary_panel_migration_rows"].iloc[0] == 4


def _default_source_decision() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "default_source_decision_status": ["default_exact_primary_ready"],
            "default_holder_panel": ["data/processed/holder_absorption_panel_exact_primary.csv"],
            "default_analysis_namespace": ["data/processed/exact_primary_analysis"],
            "default_source_layer": ["z1_exact_fu_primary_migrated"],
            "primary_panel_migration_allowed": [True],
            "mechanism_appendix_accounting_allowed": [True],
            "main_empirical_allowed": [False],
            "residual_sensitive_headline_allowed": [False],
            "rationale": ["Ready for mechanism appendix consumers."],
        }
    )


def _bucket_contract() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "schema_version": ["0.1.0", "0.1.0"],
            "contract_name": ["tdcmix_z1_exact_bucket_definition_contract"] * 2,
            "bucket": ["banks", "mmf"],
            "source_layer": ["z1_exact_fu_transactions", "z1_exact_fu_transactions"],
            "primary_panel_migration_status": ["requires_perimeter_freeze", "ready_after_identity_gate"],
            "manual_review_required": [True, False],
            "definition_note": ["Broad depository perimeter.", "MMF total Treasuries."],
        }
    )


def _migration_policy() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "schema_version": ["0.1.0", "0.1.0"],
            "policy_name": ["tdcmix_z1_exact_bucket_migration_policy"] * 2,
            "bucket": ["banks", "mmf"],
            "source_layer": ["z1_exact_fu_transactions", "z1_exact_fu_transactions"],
            "decision_status": ["resolved_by_broad_depository_perimeter", "resolved_passthrough_ready_bucket"],
            "primary_panel_migration_allowed": [True, True],
            "required_caveat": ["Broad depository bucket.", "Ready bucket."],
        }
    )


def _primary_migration_gate() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "schema_version": ["0.1.0"],
            "gate": ["z1_exact_primary_panel_migration_gate"],
            "status": ["ready"],
            "primary_panel_migration_allowed": [True],
            "main_empirical_allowed": [False],
            "residual_sensitive_headline_allowed": [False],
            "required_caveat": ["Source-layer readiness only."],
        }
    )


def _prior_contract() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "schema_version": ["0.1.0", "0.1.0"],
            "contract_name": ["tdcmix_holder_absorption_prior_contract"] * 2,
            "bucket": ["banks", "unmapped"],
            "downstream_use": ["central_candidate_after_source_review", "blocked_from_central_calibration"],
            "central_calibration_allowed": [True, False],
            "sensitivity_allowed": [True, False],
            "contract_note": ["Central candidate.", "Blocked residual."],
        }
    )


def _appendix_contract() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "schema_version": ["0.1.0"],
            "contract_name": ["tdcmix_exact_fu_appendix_source_contract"],
            "source_layer": ["z1_exact_fu_transactions"],
            "appendix_source_promoted": [True],
            "main_empirical_allowed": [False],
            "residual_sensitive_headline_allowed": [False],
            "candidate_outcome": ["deposits"],
            "candidate_contrast": ["contrast_mmf_minus_banks"],
            "recommended_use": ["mechanism_appendix_diagnostic_only"],
            "required_caveat": ["Diagnostic only."],
        }
    )


def _final_status() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "main_empirical_slot_status": ["blocked_by_timing_and_diagnostic_instability"],
            "residual_sensitive_headline_status": ["blocked"],
        }
    )
