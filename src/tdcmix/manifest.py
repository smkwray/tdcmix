from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd


MANIFEST_SCHEMA_VERSION = "0.1.0"


DEFAULT_ARTIFACTS = [
    ("holder_absorption_panel", "data/processed/holder_absorption_panel.csv", "panel"),
    ("holder_absorption_panel_exact_primary", "data/processed/holder_absorption_panel_exact_primary.csv", "panel"),
    ("exact_primary_migration_comparison", "data/processed/exact_primary_migration_comparison.csv", "validation"),
    ("exact_primary_analysis_status", "data/processed/exact_primary_analysis/exact_primary_analysis_status.csv", "validation"),
    ("exact_primary_artifact_comparison", "data/processed/exact_primary_analysis/exact_primary_artifact_comparison.csv", "validation"),
    ("validation_summary", "data/processed/validation_summary.csv", "validation"),
    ("residual_gate_review", "data/processed/residual_gate_review.csv", "validation"),
    ("z1_exact_gate_review", "data/processed/z1_exact_gate_review.csv", "validation"),
    ("z1_exact_residual_reconciliation", "data/processed/z1_exact_residual_reconciliation.csv", "validation"),
    ("z1_exact_holder_absorption_panel", "data/processed/z1_exact_holder_absorption_panel.csv", "panel"),
    ("z1_exact_holder_share_diagnostics", "data/processed/z1_exact_holder_share_diagnostics.csv", "validation"),
    ("z1_exact_bucket_definition_contract", "data/processed/z1_exact_bucket_definition_contract.csv", "downstream_contract"),
    ("z1_exact_bucket_migration_policy", "data/processed/z1_exact_bucket_migration_policy.csv", "downstream_contract"),
    ("z1_exact_primary_migration_gate", "data/processed/z1_exact_primary_migration_gate.csv", "validation"),
    ("z1_exact_analysis_design_readiness", "data/processed/z1_exact_analysis/design_readiness.csv", "design"),
    ("z1_exact_source_layer_contrast_comparison", "data/processed/z1_exact_analysis/source_layer_contrast_comparison.csv", "validation"),
    ("z1_exact_source_layer_estimate_comparison", "data/processed/z1_exact_analysis/source_layer_estimate_comparison.csv", "estimates"),
    ("z1_exact_source_layer_stability_comparison", "data/processed/z1_exact_analysis/source_layer_stability_comparison.csv", "estimates"),
    ("z1_exact_source_layer_promotion_status", "data/processed/z1_exact_analysis/source_layer_promotion_status.csv", "decision"),
    ("z1_exact_appendix_candidate_triage", "data/processed/z1_exact_analysis/exact_appendix_candidate_triage.csv", "decision"),
    ("z1_exact_mechanism_appendix_candidate_table", "data/processed/z1_exact_analysis/exact_mechanism_appendix_candidate_table.csv", "estimates"),
    ("z1_exact_candidate_sample_concentration", "data/processed/z1_exact_analysis/exact_candidate_sample_concentration.csv", "validation"),
    ("z1_exact_candidate_regime_summary", "data/processed/z1_exact_analysis/exact_candidate_regime_summary.csv", "validation"),
    ("z1_exact_candidate_sensitivity_summary", "data/processed/z1_exact_analysis/exact_candidate_sensitivity_summary.csv", "validation"),
    ("z1_exact_appendix_source_contract", "data/processed/z1_exact_analysis/exact_appendix_source_contract.csv", "downstream_contract"),
    ("z1_exact_final_status_packet", "data/processed/z1_exact_analysis/exact_final_status_packet.csv", "decision"),
    ("downstream_default_source_decision", "data/processed/downstream_default_source_decision.csv", "decision"),
    ("project_closeout_status", "data/processed/project_closeout_status.csv", "decision"),
    ("downstream_contract_bundle", "data/processed/downstream_contract_bundle.csv", "downstream_contract"),
    ("downstream_contract_bundle_summary", "data/processed/downstream_contract_bundle_summary.csv", "downstream_contract"),
    ("source_upgrade_recommendations", "data/processed/source_upgrade_recommendations.csv", "audit"),
    ("design_readiness", "data/processed/design_readiness.csv", "design"),
    ("stop_rule_status", "data/processed/stop_rule_status.csv", "design"),
    ("lp_diagnostic_summary", "data/processed/lp_diagnostic_summary.csv", "estimates"),
    ("lp_shortlist_stability_summary", "data/processed/lp_shortlist_stability_summary.csv", "estimates"),
    ("pretrend_placebo_summary", "data/processed/pretrend_placebo_summary.csv", "estimates"),
    ("z1_exact_pretrend_placebo_summary", "data/processed/z1_exact_analysis/pretrend_placebo_summary.csv", "estimates"),
    ("project_decision_status", "data/processed/project_decision_status.csv", "decision"),
    ("holder_absorption_prior_contract", "data/processed/holder_absorption_prior_contract.csv", "downstream_contract"),
]


def build_artifact_manifest(
    *,
    project_root: str | Path = ".",
    generated_at: str | None = None,
    artifacts: list[tuple[str, str, str]] | None = None,
) -> pd.DataFrame:
    root = Path(project_root)
    timestamp = generated_at or datetime.now(UTC).replace(microsecond=0).isoformat()
    rows = []
    for name, relative_path, artifact_type in artifacts or DEFAULT_ARTIFACTS:
        path = root / relative_path
        rows.append(
            {
                "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
                "generated_at": timestamp,
                "artifact_name": name,
                "artifact_type": artifact_type,
                "path": relative_path,
                "exists": path.exists(),
                "row_count": _row_count(path),
                "status": _artifact_status(name, path),
                "consumer_note": _consumer_note(name),
            }
        )
    return pd.DataFrame(rows)


def build_build_status_summary(manifest: pd.DataFrame) -> pd.DataFrame:
    missing = int((~manifest["exists"].astype(bool)).sum())
    statuses = set(manifest["status"].dropna().astype(str))
    if missing:
        build_status = "incomplete_missing_artifacts"
    elif "blocked" in statuses or "mechanism_appendix_track" in statuses:
        build_status = "complete_with_blocked_headline_paths"
    else:
        build_status = "complete"
    return pd.DataFrame(
        [
            {
                "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
                "artifact_count": int(len(manifest)),
                "missing_artifact_count": missing,
                "build_status": build_status,
                "residual_sensitive_headline_status": _status_for(manifest, "residual_gate_review", "unknown"),
                "project_recommendation": _status_for(manifest, "project_decision_status", "unknown"),
                "exact_primary_panel_status": _status_for(manifest, "holder_absorption_panel_exact_primary", "unknown"),
                "exact_primary_migration_comparison_status": _status_for(manifest, "exact_primary_migration_comparison", "unknown"),
                "exact_primary_analysis_status": _status_for(manifest, "exact_primary_analysis_status", "unknown"),
                "exact_primary_artifact_comparison_status": _status_for(manifest, "exact_primary_artifact_comparison", "unknown"),
                "z1_exact_accounting_status": _status_for(manifest, "z1_exact_gate_review", "unknown"),
                "z1_exact_bucket_definition_status": _status_for(manifest, "z1_exact_bucket_definition_contract", "unknown"),
                "z1_exact_bucket_migration_policy_status": _status_for(manifest, "z1_exact_bucket_migration_policy", "unknown"),
                "z1_exact_primary_migration_status": _status_for(manifest, "z1_exact_primary_migration_gate", "unknown"),
                "prior_contract_status": _status_for(manifest, "holder_absorption_prior_contract", "unknown"),
                "pretrend_placebo_status": _status_for(manifest, "pretrend_placebo_summary", "unknown"),
                "z1_exact_pretrend_placebo_status": _status_for(manifest, "z1_exact_pretrend_placebo_summary", "unknown"),
                "z1_exact_source_layer_promotion": _status_for(manifest, "z1_exact_source_layer_promotion_status", "unknown"),
                "z1_exact_appendix_candidate_status": _status_for(manifest, "z1_exact_appendix_candidate_triage", "unknown"),
                "z1_exact_mechanism_appendix_status": _status_for(manifest, "z1_exact_mechanism_appendix_candidate_table", "unknown"),
                "z1_exact_candidate_concentration_status": _status_for(manifest, "z1_exact_candidate_sample_concentration", "unknown"),
                "z1_exact_candidate_sensitivity_status": _status_for(manifest, "z1_exact_candidate_sensitivity_summary", "unknown"),
                "z1_exact_appendix_source_contract_status": _status_for(manifest, "z1_exact_appendix_source_contract", "unknown"),
                "z1_exact_final_backend_status": _status_for(manifest, "z1_exact_final_status_packet", "unknown"),
                "downstream_default_source_status": _status_for(manifest, "downstream_default_source_decision", "unknown"),
                "project_closeout_status": _status_for(manifest, "project_closeout_status", "unknown"),
                "downstream_contract_bundle_status": _status_for(manifest, "downstream_contract_bundle_summary", "unknown"),
            }
        ]
    )


def build_artifact_manifest_report(manifest: pd.DataFrame, summary: pd.DataFrame) -> str:
    row = summary.iloc[0].to_dict()
    lines = [
        "# Build Artifact Manifest",
        "",
        f"- Build status: `{row['build_status']}`",
        f"- Artifact count: `{int(row['artifact_count'])}`",
        f"- Missing artifacts: `{int(row['missing_artifact_count'])}`",
        f"- Residual-sensitive headline status: `{row['residual_sensitive_headline_status']}`",
        f"- Project recommendation: `{row['project_recommendation']}`",
        f"- Exact-FU primary panel: `{row['exact_primary_panel_status']}`",
        f"- Exact-FU primary migration comparison: `{row['exact_primary_migration_comparison_status']}`",
        f"- Exact-FU primary analysis: `{row['exact_primary_analysis_status']}`",
        f"- Exact-FU primary artifact comparison: `{row['exact_primary_artifact_comparison_status']}`",
        f"- Z.1 exact accounting status: `{row['z1_exact_accounting_status']}`",
        f"- Z.1 exact bucket definitions: `{row['z1_exact_bucket_definition_status']}`",
        f"- Z.1 exact bucket migration policy: `{row['z1_exact_bucket_migration_policy_status']}`",
        f"- Z.1 exact primary migration gate: `{row['z1_exact_primary_migration_status']}`",
        f"- Prior contract status: `{row['prior_contract_status']}`",
        f"- Pretrend placebo status: `{row['pretrend_placebo_status']}`",
        f"- Z.1 exact pretrend placebo status: `{row['z1_exact_pretrend_placebo_status']}`",
        f"- Z.1 exact source-layer promotion: `{row['z1_exact_source_layer_promotion']}`",
        f"- Z.1 exact appendix candidates: `{row['z1_exact_appendix_candidate_status']}`",
        f"- Z.1 exact mechanism appendix rows: `{row['z1_exact_mechanism_appendix_status']}`",
        f"- Z.1 exact candidate concentration: `{row['z1_exact_candidate_concentration_status']}`",
        f"- Z.1 exact candidate sensitivity: `{row['z1_exact_candidate_sensitivity_status']}`",
        f"- Z.1 exact appendix source contract: `{row['z1_exact_appendix_source_contract_status']}`",
        f"- Z.1 exact final backend status: `{row['z1_exact_final_backend_status']}`",
        f"- Downstream default source: `{row['downstream_default_source_status']}`",
        f"- Project closeout status: `{row['project_closeout_status']}`",
        f"- Downstream contract bundle: `{row['downstream_contract_bundle_status']}`",
        "",
        "## Artifacts",
        "",
    ]
    for artifact in manifest.to_dict("records"):
        lines.append(
            "- "
            f"`{artifact['artifact_name']}`: `{artifact['status']}`, "
            f"rows `{int(artifact['row_count'])}`, path `{artifact['path']}`"
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "This manifest describes generated artifacts and routing status. It does not promote diagnostic estimates into evidence.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_artifact_manifest_outputs(
    *,
    project_root: str | Path = ".",
    output_dir: str | Path = "data/processed",
    generated_at: str | None = None,
) -> list[Path]:
    manifest = build_artifact_manifest(project_root=project_root, generated_at=generated_at)
    summary = build_build_status_summary(manifest)
    report = build_artifact_manifest_report(manifest, summary)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "artifact_manifest.csv"
    summary_path = out_dir / "build_status_summary.csv"
    report_path = out_dir / "artifact_manifest_report.md"
    manifest.to_csv(manifest_path, index=False)
    summary.to_csv(summary_path, index=False)
    report_path.write_text(report, encoding="utf-8")
    return [manifest_path, summary_path, report_path]


def _row_count(path: Path) -> int:
    if not path.exists() or path.suffix.lower() != ".csv":
        return 0
    return int(pd.read_csv(path, nrows=0).shape[0]) if path.stat().st_size == 0 else int(len(pd.read_csv(path)))


def _artifact_status(name: str, path: Path) -> str:
    if not path.exists():
        return "missing"
    if name == "residual_gate_review":
        frame = pd.read_csv(path)
        return str(frame["status"].iloc[0]) if "status" in frame and not frame.empty else "unknown"
    if name == "z1_exact_gate_review":
        frame = pd.read_csv(path)
        return str(frame["status"].iloc[0]) if "status" in frame and not frame.empty else "unknown"
    if name == "holder_absorption_panel_exact_primary":
        frame = pd.read_csv(path)
        if frame.empty or "holder_source_layer" not in frame:
            return "unknown"
        layer = str(frame["holder_source_layer"].dropna().iloc[0]) if frame["holder_source_layer"].notna().any() else "unknown"
        return f"{layer}_rows_{len(frame)}"
    if name == "exact_primary_migration_comparison":
        frame = pd.read_csv(path)
        if frame.empty or "comparison" not in frame:
            return "unknown"
        sidecar = frame.loc[frame["comparison"].eq("exact_primary_vs_exact_sidecar")]
        max_gap = pd.to_numeric(sidecar["max_abs_gap"], errors="coerce").max() if not sidecar.empty else float("nan")
        status = "sidecar_match" if pd.notna(max_gap) and max_gap <= 1e-9 else "sidecar_differs"
        return f"{status}_rows_{len(frame)}"
    if name == "exact_primary_analysis_status":
        frame = pd.read_csv(path)
        return str(frame["status"].iloc[0]) if "status" in frame and not frame.empty else "unknown"
    if name == "exact_primary_artifact_comparison":
        frame = pd.read_csv(path)
        if frame.empty or "status" not in frame:
            return "unknown"
        matches = int(frame["status"].eq("matches").sum())
        differs = int(frame["status"].eq("differs").sum())
        return f"artifact_comparisons_{len(frame)}_matches_{matches}_differs_{differs}"
    if name == "z1_exact_bucket_definition_contract":
        frame = pd.read_csv(path)
        if frame.empty or "manual_review_required" not in frame:
            return "unknown"
        review = int(frame["manual_review_required"].astype(bool).sum())
        ready = int(frame["primary_panel_migration_status"].astype(str).str.startswith("ready").sum()) if "primary_panel_migration_status" in frame else 0
        return f"bucket_definitions_{len(frame)}_review_{review}_ready_{ready}"
    if name == "z1_exact_bucket_migration_policy":
        frame = pd.read_csv(path)
        if frame.empty or "decision_status" not in frame:
            return "unknown"
        resolved = int(frame["decision_status"].astype(str).str.startswith("resolved").sum())
        migration = int(frame["primary_panel_migration_allowed"].astype(bool).sum()) if "primary_panel_migration_allowed" in frame else 0
        return f"migration_policies_{len(frame)}_resolved_{resolved}_allowed_{migration}"
    if name == "z1_exact_primary_migration_gate":
        frame = pd.read_csv(path)
        return str(frame["status"].iloc[0]) if "status" in frame and not frame.empty else "unknown"
    if name == "project_decision_status":
        frame = pd.read_csv(path)
        return str(frame["recommendation"].iloc[0]) if "recommendation" in frame and not frame.empty else "unknown"
    if name == "holder_absorption_prior_contract":
        frame = pd.read_csv(path)
        central = int(frame["central_calibration_allowed"].astype(bool).sum()) if "central_calibration_allowed" in frame else 0
        blocked = int((~frame["sensitivity_allowed"].astype(bool)).sum()) if "sensitivity_allowed" in frame else 0
        return f"central_candidates_{central}_blocked_{blocked}"
    if name in {"pretrend_placebo_summary", "z1_exact_pretrend_placebo_summary"}:
        frame = pd.read_csv(path)
        if "pretrend_screen_status" not in frame or frame.empty:
            return "unknown"
        concerns = int(frame["pretrend_screen_status"].eq("timing_concern").sum())
        return "timing_screen_passes" if concerns == 0 else f"timing_concerns_{concerns}"
    if name == "z1_exact_source_layer_promotion_status":
        frame = pd.read_csv(path)
        return str(frame["recommendation"].iloc[0]) if "recommendation" in frame and not frame.empty else "unknown"
    if name == "z1_exact_appendix_candidate_triage":
        frame = pd.read_csv(path)
        if "appendix_candidate_status" not in frame:
            return "unknown"
        candidates = int(frame["appendix_candidate_status"].eq("appendix_candidate").sum())
        held = int(frame["appendix_candidate_status"].ne("appendix_candidate").sum())
        return f"appendix_candidates_{candidates}_held_{held}"
    if name == "z1_exact_mechanism_appendix_candidate_table":
        frame = pd.read_csv(path)
        if frame.empty:
            return "no_mechanism_rows"
        profiles = int(frame[["outcome", "contrast"]].drop_duplicates().shape[0])
        horizons = int(frame["horizon"].nunique()) if "horizon" in frame else 0
        return f"mechanism_profiles_{profiles}_horizons_{horizons}"
    if name == "z1_exact_candidate_sample_concentration":
        frame = pd.read_csv(path)
        if frame.empty:
            return "no_concentration_rows"
        high = int(pd.to_numeric(frame.get("top5_abs_interaction_share"), errors="coerce").gt(0.5).sum())
        return f"concentration_rows_{len(frame)}_high_top5_{high}"
    if name == "z1_exact_candidate_sensitivity_summary":
        frame = pd.read_csv(path)
        if frame.empty or "sign_robustness_status" not in frame:
            return "no_sensitivity_rows"
        mixed = int(frame["sign_robustness_status"].eq("sign_mixed_under_exclusions").sum())
        return f"sensitivity_rows_{len(frame)}_mixed_{mixed}"
    if name == "z1_exact_appendix_source_contract":
        frame = pd.read_csv(path)
        if frame.empty or "appendix_source_promoted" not in frame:
            return "unknown"
        promoted = bool(frame["appendix_source_promoted"].astype(bool).all())
        candidates = int(frame["candidate_outcome"].fillna("").ne("").sum()) if "candidate_outcome" in frame else 0
        return f"appendix_source_promoted_{str(promoted).lower()}_candidates_{candidates}"
    if name == "z1_exact_final_status_packet":
        frame = pd.read_csv(path)
        return str(frame["main_empirical_slot_status"].iloc[0]) if "main_empirical_slot_status" in frame and not frame.empty else "unknown"
    if name == "downstream_default_source_decision":
        frame = pd.read_csv(path)
        return str(frame["default_source_decision_status"].iloc[0]) if "default_source_decision_status" in frame and not frame.empty else "unknown"
    if name == "project_closeout_status":
        frame = pd.read_csv(path)
        return str(frame["closeout_status"].iloc[0]) if "closeout_status" in frame and not frame.empty else "unknown"
    if name == "downstream_contract_bundle_summary":
        frame = pd.read_csv(path)
        return str(frame["bundle_status"].iloc[0]) if "bundle_status" in frame and not frame.empty else "unknown"
    if name == "downstream_contract_bundle":
        frame = pd.read_csv(path)
        if frame.empty:
            return "empty"
        appendix = int(frame["mechanism_appendix_allowed"].astype(bool).sum()) if "mechanism_appendix_allowed" in frame else 0
        central = int(frame["central_calibration_allowed"].astype(bool).sum()) if "central_calibration_allowed" in frame else 0
        return f"contract_rows_{len(frame)}_appendix_{appendix}_central_{central}"
    return "available"


def _consumer_note(name: str) -> str:
    notes = {
        "residual_gate_review": "Residual-sensitive headline paths remain blocked unless this status becomes ready.",
        "holder_absorption_panel_exact_primary": "Primary-compatible holder panel using exact-FU source-layer transaction buckets.",
        "exact_primary_migration_comparison": "Compares exact-FU primary panel to the exact-FU sidecar and legacy holder panel.",
        "exact_primary_analysis_status": "Status packet for exact-primary validation/design/estimate namespace.",
        "exact_primary_artifact_comparison": "Compares exact-primary analysis artifacts against exact-FU sidecar analysis artifacts.",
        "z1_exact_gate_review": "Official Z.1 transaction accounting gate; ready status means exact FU component identity closes.",
        "z1_exact_residual_reconciliation": "Compares official exact Z.1 components to the current holder-panel sources and legacy proxy residual.",
        "z1_exact_bucket_definition_contract": "Bucket-level exact-FU definition contract for primary-panel migration review.",
        "z1_exact_bucket_migration_policy": "Policy rows resolving exact-FU bucket-definition review flags.",
        "z1_exact_primary_migration_gate": "Gate for exact-FU primary source-layer migration readiness.",
        "z1_exact_source_layer_contrast_comparison": "Compares legacy holder-share contrasts to the exact FU holder-source layer.",
        "z1_exact_source_layer_estimate_comparison": "Compares legacy diagnostic estimate summaries to exact FU source-layer summaries.",
        "z1_exact_source_layer_stability_comparison": "Compares legacy shortlist stability summaries to exact FU source-layer stability summaries.",
        "z1_exact_source_layer_promotion_status": "Decision artifact for whether exact FU should be the accounting-preferred source layer.",
        "z1_exact_appendix_candidate_triage": "Exact-FU reviewable profile triage after source-layer and pretrend timing screens.",
        "z1_exact_mechanism_appendix_candidate_table": "Horizon-level appendix detail for exact-FU mechanism candidate profiles.",
        "z1_exact_candidate_sample_concentration": "Sample concentration diagnostics for exact-FU appendix candidate profiles.",
        "z1_exact_candidate_regime_summary": "Regime coverage diagnostics for exact-FU appendix candidate profiles.",
        "z1_exact_candidate_sensitivity_summary": "Coefficient sensitivity diagnostics under top-quarter and regime exclusions.",
        "z1_exact_appendix_source_contract": "Machine-readable contract promoting exact-FU only for mechanism appendix accounting.",
        "z1_exact_final_status_packet": "Backend status packet summarizing exact-FU source promotion, candidates, diagnostics, and blockers.",
        "downstream_default_source_decision": "Decision artifact freezing exact-primary outputs as the default backend source for mechanism appendix consumers.",
        "project_closeout_status": "Closeout packet stating remaining backend work and next external consumer-integration work.",
        "downstream_contract_bundle": "Combined downstream interface for holder priors and exact-FU appendix diagnostics.",
        "downstream_contract_bundle_summary": "Summary status for the combined downstream contract bundle.",
        "project_decision_status": "Use to route tdcmix as main, appendix, or hold work.",
        "holder_absorption_prior_contract": "Use this as the downstream interface for calibration candidates and sensitivity-only buckets.",
        "lp_diagnostic_summary": "Diagnostic estimates only; do not cite as causal evidence.",
        "pretrend_placebo_summary": "Legacy-source h=-4..-1 timing screen for shortlisted diagnostics.",
        "z1_exact_pretrend_placebo_summary": "Exact-FU h=-4..-1 timing screen for shortlisted diagnostics.",
    }
    return notes.get(name, "Generated tdcmix artifact.")


def _status_for(manifest: pd.DataFrame, artifact_name: str, fallback: str) -> str:
    rows = manifest.loc[manifest["artifact_name"].eq(artifact_name), "status"]
    if rows.empty:
        return fallback
    return str(rows.iloc[0])
