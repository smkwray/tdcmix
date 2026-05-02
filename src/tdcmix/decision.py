from __future__ import annotations

from pathlib import Path

import pandas as pd


def build_project_decision(
    validation_summary: pd.DataFrame,
    design_readiness: pd.DataFrame,
    stop_rule_status: pd.DataFrame,
    stability_summary: pd.DataFrame,
    economics_review: pd.DataFrame,
) -> pd.DataFrame:
    residual_ready = _readiness_passes(design_readiness, "residual_sensitive_contrasts")
    non_residual_ready = _readiness_passes(design_readiness, "baseline_non_residual_contrasts")
    h_design_ready = _readiness_passes(design_readiness, "h0_h4_outcome_scaffold")
    all_stop_rules_pass = (
        bool(stop_rule_status["passes"].astype(bool).all()) if "passes" in stop_rule_status else False
    )
    reviewable_profiles = int(
        stability_summary["stability_status"]
        .isin(["directionally_stable_diagnostic", "mixed_but_reviewable_diagnostic"])
        .sum()
    )
    economics_review_rows = int(len(economics_review))
    residual_validation = _validation_value(validation_summary, "exact_residual_overlap_rows")

    if (
        non_residual_ready
        and h_design_ready
        and all_stop_rules_pass
        and reviewable_profiles >= 3
        and not residual_ready
    ):
        recommendation = "mechanism_appendix_track"
        main_slot_status = "not_main_empirical_slot_yet"
        rationale = (
            "Non-residual diagnostics are usable for mechanism evidence, but residual-sensitive "
            "headline work remains blocked and shortlist profiles are diagnostic only."
        )
    elif non_residual_ready and h_design_ready and all_stop_rules_pass and residual_ready:
        recommendation = "candidate_main_empirical_track_after_identification_review"
        main_slot_status = "candidate_main_slot"
        rationale = (
            "Core design gates pass including residual readiness; still requires separate identification review."
        )
    else:
        recommendation = "hold_for_source_and_design_repairs"
        main_slot_status = "not_ready"
        rationale = "One or more core non-residual design or stop-rule gates failed."

    return pd.DataFrame(
        [
            {
                "decision_scope": "tdcmix_holder_absorption_path",
                "recommendation": recommendation,
                "main_slot_status": main_slot_status,
                "non_residual_ready": non_residual_ready,
                "h0_h4_design_ready": h_design_ready,
                "all_stop_rules_pass": all_stop_rules_pass,
                "residual_sensitive_ready": residual_ready,
                "reviewable_shortlist_profiles": reviewable_profiles,
                "economics_review_rows": economics_review_rows,
                "exact_residual_overlap_rows": residual_validation,
                "rationale": rationale,
                "next_action": (
                    "Keep as mechanism/appendix evidence; improve source contracts and residual overlap "
                    "before any main empirical claim."
                ),
            }
        ]
    )


def build_project_decision_report(decision: pd.DataFrame) -> str:
    row = decision.iloc[0].to_dict()
    lines = [
        "# Project Decision Report",
        "",
        f"- Recommendation: `{row['recommendation']}`",
        f"- Main slot status: `{row['main_slot_status']}`",
        f"- Non-residual ready: `{row['non_residual_ready']}`",
        f"- h=0..4 design ready: `{row['h0_h4_design_ready']}`",
        f"- Residual-sensitive ready: `{row['residual_sensitive_ready']}`",
        f"- Reviewable shortlist profiles: `{int(row['reviewable_shortlist_profiles'])}`",
        "",
        "## Rationale",
        "",
        str(row["rationale"]),
        "",
        "## Next Action",
        "",
        str(row["next_action"]),
        "",
        "## Boundary",
        "",
        "This is a project-routing decision, not an empirical finding.",
    ]
    return "\n".join(lines) + "\n"


def write_project_decision_outputs(
    *,
    validation_summary_path: str | Path = "data/processed/validation_summary.csv",
    design_readiness_path: str | Path = "data/processed/design_readiness.csv",
    stop_rule_status_path: str | Path = "data/processed/stop_rule_status.csv",
    stability_summary_path: str | Path = "data/processed/lp_shortlist_stability_summary.csv",
    economics_review_path: str | Path = "data/processed/lp_shortlist_economics_review.csv",
    output_dir: str | Path = "data/processed",
) -> list[Path]:
    decision = build_project_decision(
        pd.read_csv(validation_summary_path),
        pd.read_csv(design_readiness_path),
        pd.read_csv(stop_rule_status_path),
        pd.read_csv(stability_summary_path),
        pd.read_csv(economics_review_path),
    )
    report = build_project_decision_report(decision)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    decision_path = out_dir / "project_decision_status.csv"
    report_path = out_dir / "project_decision_report.md"
    decision.to_csv(decision_path, index=False)
    report_path.write_text(report, encoding="utf-8")
    return [decision_path, report_path]


def build_downstream_default_source_decision(
    project_decision: pd.DataFrame,
    exact_primary_analysis_status: pd.DataFrame,
    exact_primary_artifact_comparison: pd.DataFrame,
    exact_primary_migration_comparison: pd.DataFrame,
    primary_migration_gate: pd.DataFrame,
) -> pd.DataFrame:
    project_recommendation = _first_value(project_decision, "recommendation")
    analysis_status = _first_value(exact_primary_analysis_status, "status")
    artifact_differs = _artifact_differs(exact_primary_artifact_comparison)
    sidecar_match = _exact_primary_sidecar_match(exact_primary_migration_comparison)
    migration_gate_status = _first_value(primary_migration_gate, "status")
    migration_allowed = _first_bool(primary_migration_gate, "primary_panel_migration_allowed")
    main_allowed = _first_bool(primary_migration_gate, "main_empirical_allowed")
    residual_allowed = _first_bool(primary_migration_gate, "residual_sensitive_headline_allowed")
    ready = (
        project_recommendation == "mechanism_appendix_track"
        and analysis_status == "ready"
        and artifact_differs == 0
        and sidecar_match
        and migration_gate_status == "ready"
        and migration_allowed
    )
    status = "default_exact_primary_ready" if ready else "hold_default_legacy_until_gate_review"
    default_holder_panel = (
        "data/processed/holder_absorption_panel_exact_primary.csv"
        if ready
        else "data/processed/holder_absorption_panel.csv"
    )
    default_analysis_namespace = (
        "data/processed/exact_primary_analysis"
        if ready
        else "data/processed"
    )
    return pd.DataFrame(
        [
            {
                "decision_scope": "tdcmix_downstream_default_source",
                "default_source_decision_status": status,
                "default_holder_panel": default_holder_panel,
                "default_analysis_namespace": default_analysis_namespace,
                "default_source_layer": (
                    "z1_exact_fu_primary_migrated" if ready else "legacy_holder_absorption_panel"
                ),
                "legacy_holder_panel_role": "audit_comparison_only" if ready else "default_until_exact_primary_ready",
                "primary_panel_migration_allowed": ready,
                "mechanism_appendix_accounting_allowed": ready,
                "main_empirical_allowed": bool(main_allowed),
                "residual_sensitive_headline_allowed": bool(residual_allowed),
                "project_recommendation": project_recommendation,
                "exact_primary_analysis_status": analysis_status,
                "artifact_comparison_differs": artifact_differs,
                "exact_primary_sidecar_match": sidecar_match,
                "primary_migration_gate_status": migration_gate_status,
                "rationale": (
                    "Exact-FU primary-compatible panel and exact-primary analysis namespace match the exact-FU "
                    "sidecar artifacts, so downstream mechanism/appendix consumers should default to the "
                    "exact-primary source layer while keeping main empirical and residual-sensitive headline use blocked."
                    if ready
                    else "One or more exact-primary readiness gates failed; downstream consumers should keep the legacy panel as default."
                ),
                "next_action": (
                    "Run consumer integration smoke tests against the downstream contract bundle."
                    if ready
                    else "Repair exact-primary readiness gates before changing downstream defaults."
                ),
            }
        ]
    )


def build_downstream_default_source_decision_report(decision: pd.DataFrame) -> str:
    row = decision.iloc[0].to_dict()
    lines = [
        "# Downstream Default Source Decision",
        "",
        f"- Status: `{row['default_source_decision_status']}`",
        f"- Default holder panel: `{row['default_holder_panel']}`",
        f"- Default analysis namespace: `{row['default_analysis_namespace']}`",
        f"- Default source layer: `{row['default_source_layer']}`",
        f"- Legacy holder panel role: `{row['legacy_holder_panel_role']}`",
        f"- Primary panel migration allowed: `{row['primary_panel_migration_allowed']}`",
        f"- Mechanism appendix accounting allowed: `{row['mechanism_appendix_accounting_allowed']}`",
        f"- Main empirical allowed: `{row['main_empirical_allowed']}`",
        f"- Residual-sensitive headline allowed: `{row['residual_sensitive_headline_allowed']}`",
        "",
        "## Rationale",
        "",
        str(row["rationale"]),
        "",
        "## Next Action",
        "",
        str(row["next_action"]),
        "",
        "## Boundary",
        "",
        "This decision changes downstream defaults for backend consumers. It does not promote diagnostic estimates into causal, main-text, or residual-sensitive headline evidence.",
    ]
    return "\n".join(lines) + "\n"


def build_project_closeout_status(
    downstream_default_decision: pd.DataFrame,
    project_decision: pd.DataFrame,
    exact_primary_analysis_status: pd.DataFrame,
    final_status_packet: pd.DataFrame,
    appendix_source_contract: pd.DataFrame,
) -> pd.DataFrame:
    default_status = _first_value(downstream_default_decision, "default_source_decision_status")
    project_recommendation = _first_value(project_decision, "recommendation")
    exact_primary_status = _first_value(exact_primary_analysis_status, "status")
    main_status = _first_value(final_status_packet, "main_empirical_slot_status", "unknown")
    residual_status = _first_value(final_status_packet, "residual_sensitive_headline_status", "blocked")
    appendix_candidates = _appendix_candidate_count(appendix_source_contract)
    complete = (
        default_status == "default_exact_primary_ready"
        and project_recommendation == "mechanism_appendix_track"
        and exact_primary_status == "ready"
        and str(main_status).startswith("blocked")
        and str(residual_status).startswith("blocked")
    )
    return pd.DataFrame(
        [
            {
                "closeout_scope": "tdcmix_backend_packaging_closeout",
                "closeout_status": (
                    "backend_complete_for_mechanism_appendix_consumers"
                    if complete
                    else "backend_closeout_requires_review"
                ),
                "default_source_decision_status": default_status,
                "project_recommendation": project_recommendation,
                "exact_primary_analysis_status": exact_primary_status,
                "appendix_source_candidate_rows": appendix_candidates,
                "main_empirical_slot_status": main_status,
                "residual_sensitive_headline_status": residual_status,
                "remaining_backend_work": (
                    "none_for_mechanism_appendix_path" if complete else "review_failed_closeout_gates"
                ),
                "next_external_work": "consumer_integration_smoke_test",
                "rationale": (
                    "The source layer, exact-primary namespace, downstream default decision, and project routing are frozen for mechanism/appendix use; headline paths remain explicitly blocked."
                    if complete
                    else "One or more closeout gates failed; inspect the default-source decision, exact-primary status, and final status packet."
                ),
            }
        ]
    )


def build_project_closeout_report(closeout: pd.DataFrame) -> str:
    row = closeout.iloc[0].to_dict()
    lines = [
        "# Project Closeout Status",
        "",
        f"- Closeout status: `{row['closeout_status']}`",
        f"- Default source decision: `{row['default_source_decision_status']}`",
        f"- Project recommendation: `{row['project_recommendation']}`",
        f"- Exact-primary analysis: `{row['exact_primary_analysis_status']}`",
        f"- Appendix source candidate rows: `{int(row['appendix_source_candidate_rows'])}`",
        f"- Main empirical slot: `{row['main_empirical_slot_status']}`",
        f"- Residual-sensitive headline status: `{row['residual_sensitive_headline_status']}`",
        f"- Remaining backend work: `{row['remaining_backend_work']}`",
        f"- Next external work: `{row['next_external_work']}`",
        "",
        "## Rationale",
        "",
        str(row["rationale"]),
        "",
        "## Boundary",
        "",
        "This closeout marks the backend mechanism/appendix path as packaged for consumers. Consumer integration tests still need to run outside this repository before treating the interface as adopted.",
    ]
    return "\n".join(lines) + "\n"


def write_project_closeout_outputs(
    *,
    project_decision_path: str | Path = "data/processed/project_decision_status.csv",
    exact_primary_analysis_status_path: str | Path = "data/processed/exact_primary_analysis/exact_primary_analysis_status.csv",
    exact_primary_artifact_comparison_path: str | Path = "data/processed/exact_primary_analysis/exact_primary_artifact_comparison.csv",
    exact_primary_migration_comparison_path: str | Path = "data/processed/exact_primary_migration_comparison.csv",
    primary_migration_gate_path: str | Path = "data/processed/z1_exact_primary_migration_gate.csv",
    appendix_source_contract_path: str | Path = "data/processed/z1_exact_analysis/exact_appendix_source_contract.csv",
    final_status_packet_path: str | Path = "data/processed/z1_exact_analysis/exact_final_status_packet.csv",
    output_dir: str | Path = "data/processed",
) -> list[Path]:
    project_decision = pd.read_csv(project_decision_path)
    exact_primary_status = pd.read_csv(exact_primary_analysis_status_path)
    artifact_comparison = pd.read_csv(exact_primary_artifact_comparison_path)
    migration_comparison = pd.read_csv(exact_primary_migration_comparison_path)
    primary_gate = pd.read_csv(primary_migration_gate_path)
    appendix_contract = pd.read_csv(appendix_source_contract_path)
    final_status = pd.read_csv(final_status_packet_path)
    default_decision = build_downstream_default_source_decision(
        project_decision,
        exact_primary_status,
        artifact_comparison,
        migration_comparison,
        primary_gate,
    )
    default_report = build_downstream_default_source_decision_report(default_decision)
    closeout = build_project_closeout_status(
        default_decision,
        project_decision,
        exact_primary_status,
        final_status,
        appendix_contract,
    )
    closeout_report = build_project_closeout_report(closeout)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    default_path = out_dir / "downstream_default_source_decision.csv"
    default_report_path = out_dir / "downstream_default_source_decision.md"
    closeout_path = out_dir / "project_closeout_status.csv"
    closeout_report_path = out_dir / "project_closeout_report.md"
    default_decision.to_csv(default_path, index=False)
    default_report_path.write_text(default_report, encoding="utf-8")
    closeout.to_csv(closeout_path, index=False)
    closeout_report_path.write_text(closeout_report, encoding="utf-8")
    return [default_path, default_report_path, closeout_path, closeout_report_path]


def _readiness_passes(readiness: pd.DataFrame, path_name: str) -> bool:
    name_column = "path" if "path" in readiness else "design" if "design" in readiness else ""
    if not name_column or "ready" not in readiness:
        return False
    rows = readiness.loc[readiness[name_column].eq(path_name), "ready"]
    if rows.empty:
        return False
    return bool(rows.astype(bool).iloc[0])


def _validation_value(validation: pd.DataFrame, check_name: str) -> int:
    if "check" not in validation or "value" not in validation:
        return 0
    rows = validation.loc[validation["check"].eq(check_name), "value"]
    if rows.empty:
        return 0
    return int(pd.to_numeric(rows, errors="coerce").fillna(0).iloc[0])


def _first_value(frame: pd.DataFrame, column: str, fallback: str = "unknown") -> str:
    if frame.empty or column not in frame:
        return fallback
    return str(frame[column].iloc[0])


def _first_bool(frame: pd.DataFrame, column: str) -> bool:
    if frame.empty or column not in frame:
        return False
    return _as_bool(frame[column].iloc[0])


def _artifact_differs(comparison: pd.DataFrame) -> int:
    if comparison.empty or "status" not in comparison:
        return 1
    return int(comparison["status"].astype(str).eq("differs").sum())


def _exact_primary_sidecar_match(comparison: pd.DataFrame) -> bool:
    if comparison.empty or "comparison" not in comparison or "max_abs_gap" not in comparison:
        return False
    sidecar = comparison.loc[comparison["comparison"].eq("exact_primary_vs_exact_sidecar")]
    if sidecar.empty:
        return False
    max_gap = pd.to_numeric(sidecar["max_abs_gap"], errors="coerce").max()
    return bool(pd.notna(max_gap) and max_gap <= 1e-9)


def _appendix_candidate_count(contract: pd.DataFrame) -> int:
    if contract.empty or "candidate_outcome" not in contract:
        return 0
    return int(contract["candidate_outcome"].fillna("").astype(str).ne("").sum())


def _as_bool(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return bool(value)
