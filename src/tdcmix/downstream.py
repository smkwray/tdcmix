from __future__ import annotations

from pathlib import Path

import pandas as pd


DOWNSTREAM_BUNDLE_SCHEMA_VERSION = "0.1.0"
DOWNSTREAM_BUNDLE_NAME = "tdcmix_downstream_contract_bundle"


def build_downstream_contract_bundle(
    prior_contract: pd.DataFrame,
    appendix_source_contract: pd.DataFrame,
    final_status_packet: pd.DataFrame,
    bucket_definition_contract: pd.DataFrame | None = None,
    bucket_migration_policy: pd.DataFrame | None = None,
    primary_migration_gate: pd.DataFrame | None = None,
    default_source_decision: pd.DataFrame | None = None,
) -> pd.DataFrame:
    rows = []
    final_status = final_status_packet.iloc[0].to_dict() if not final_status_packet.empty else {}
    for row in prior_contract.to_dict("records"):
        rows.append(
            {
                "schema_version": DOWNSTREAM_BUNDLE_SCHEMA_VERSION,
                "bundle_name": DOWNSTREAM_BUNDLE_NAME,
                "source_contract_name": row.get("contract_name", "tdcmix_holder_absorption_prior_contract"),
                "contract_row_type": "holder_absorption_prior",
                "object_key": str(row.get("bucket", "")),
                "source_layer": "legacy_holder_absorption_panel",
                "downstream_use": row.get("downstream_use", "unknown"),
                "central_calibration_allowed": bool(row.get("central_calibration_allowed", False)),
                "mechanism_appendix_allowed": False,
                "main_empirical_allowed": False,
                "residual_sensitive_headline_allowed": False,
                "sensitivity_allowed": bool(row.get("sensitivity_allowed", False)),
                "blocked_reason": _prior_blocked_reason(row),
                "compatibility_key": _compatibility_key(row),
                "required_caveat": row.get("contract_note", ""),
            }
        )
    for row in appendix_source_contract.to_dict("records"):
        object_key = _appendix_object_key(row)
        appendix_promoted = bool(row.get("appendix_source_promoted", False))
        main_allowed = bool(row.get("main_empirical_allowed", False))
        residual_allowed = bool(row.get("residual_sensitive_headline_allowed", False))
        rows.append(
            {
                "schema_version": DOWNSTREAM_BUNDLE_SCHEMA_VERSION,
                "bundle_name": DOWNSTREAM_BUNDLE_NAME,
                "source_contract_name": row.get("contract_name", "tdcmix_exact_fu_appendix_source_contract"),
                "contract_row_type": "exact_fu_mechanism_appendix",
                "object_key": object_key,
                "source_layer": row.get("source_layer", "z1_exact_fu_transactions"),
                "downstream_use": row.get("recommended_use", "mechanism_appendix_diagnostic_only"),
                "central_calibration_allowed": False,
                "mechanism_appendix_allowed": appendix_promoted and bool(object_key),
                "main_empirical_allowed": main_allowed,
                "residual_sensitive_headline_allowed": residual_allowed,
                "sensitivity_allowed": False,
                "blocked_reason": _appendix_blocked_reason(row, final_status),
                "compatibility_key": _compatibility_key(row),
                "required_caveat": row.get("required_caveat", ""),
            }
        )
    if bucket_definition_contract is not None:
        for row in bucket_definition_contract.to_dict("records"):
            review_required = bool(row.get("manual_review_required", True))
            rows.append(
                {
                    "schema_version": DOWNSTREAM_BUNDLE_SCHEMA_VERSION,
                    "bundle_name": DOWNSTREAM_BUNDLE_NAME,
                    "source_contract_name": row.get("contract_name", "tdcmix_z1_exact_bucket_definition_contract"),
                    "contract_row_type": "exact_fu_bucket_definition",
                    "object_key": str(row.get("bucket", "")),
                    "source_layer": row.get("source_layer", "z1_exact_fu_transactions"),
                    "downstream_use": row.get("primary_panel_migration_status", "requires_review"),
                    "central_calibration_allowed": False,
                    "mechanism_appendix_allowed": False,
                    "main_empirical_allowed": False,
                    "residual_sensitive_headline_allowed": False,
                    "sensitivity_allowed": False,
                    "blocked_reason": (
                        "Manual bucket-definition review required before primary-panel migration."
                        if review_required
                        else ""
                    ),
                    "compatibility_key": _compatibility_key(row),
                    "required_caveat": row.get("definition_note", ""),
                }
            )
    if bucket_migration_policy is not None:
        for row in bucket_migration_policy.to_dict("records"):
            migration_allowed = bool(row.get("primary_panel_migration_allowed", False))
            rows.append(
                {
                    "schema_version": DOWNSTREAM_BUNDLE_SCHEMA_VERSION,
                    "bundle_name": DOWNSTREAM_BUNDLE_NAME,
                    "source_contract_name": row.get("policy_name", "tdcmix_z1_exact_bucket_migration_policy"),
                    "contract_row_type": "exact_fu_bucket_migration_policy",
                    "object_key": str(row.get("bucket", "")),
                    "source_layer": row.get("source_layer", "z1_exact_fu_transactions"),
                    "downstream_use": row.get("decision_status", "unresolved"),
                    "central_calibration_allowed": False,
                    "mechanism_appendix_allowed": False,
                    "main_empirical_allowed": False,
                    "residual_sensitive_headline_allowed": False,
                    "sensitivity_allowed": False,
                    "primary_panel_migration_allowed": migration_allowed,
                    "blocked_reason": "" if migration_allowed else "Bucket migration policy is unresolved.",
                    "compatibility_key": _policy_compatibility_key(row),
                    "required_caveat": row.get("required_caveat", ""),
                }
            )
    if primary_migration_gate is not None and not primary_migration_gate.empty:
        row = primary_migration_gate.iloc[0].to_dict()
        migration_allowed = bool(row.get("primary_panel_migration_allowed", False))
        rows.append(
            {
                "schema_version": DOWNSTREAM_BUNDLE_SCHEMA_VERSION,
                "bundle_name": DOWNSTREAM_BUNDLE_NAME,
                "source_contract_name": "tdcmix_z1_exact_primary_migration_gate",
                "contract_row_type": "exact_fu_primary_migration_gate",
                "object_key": str(row.get("gate", "z1_exact_primary_panel_migration_gate")),
                "source_layer": "z1_exact_fu_transactions",
                "downstream_use": str(row.get("status", "unknown")),
                "central_calibration_allowed": False,
                "mechanism_appendix_allowed": False,
                "main_empirical_allowed": bool(row.get("main_empirical_allowed", False)),
                "residual_sensitive_headline_allowed": bool(row.get("residual_sensitive_headline_allowed", False)),
                "sensitivity_allowed": False,
                "primary_panel_migration_allowed": migration_allowed,
                "blocked_reason": "" if migration_allowed else str(row.get("required_next_step", "")),
                "compatibility_key": f"{row.get('schema_version', '')}:tdcmix_z1_exact_primary_migration_gate",
                "required_caveat": row.get("required_caveat", ""),
            }
        )
    if default_source_decision is not None and not default_source_decision.empty:
        row = default_source_decision.iloc[0].to_dict()
        default_ready = row.get("default_source_decision_status") == "default_exact_primary_ready"
        rows.append(
            {
                "schema_version": DOWNSTREAM_BUNDLE_SCHEMA_VERSION,
                "bundle_name": DOWNSTREAM_BUNDLE_NAME,
                "source_contract_name": "tdcmix_downstream_default_source_decision",
                "contract_row_type": "downstream_default_source_decision",
                "object_key": "holder_absorption_panel",
                "source_layer": str(row.get("default_source_layer", "unknown")),
                "downstream_use": str(row.get("default_source_decision_status", "unknown")),
                "central_calibration_allowed": False,
                "mechanism_appendix_allowed": bool(row.get("mechanism_appendix_accounting_allowed", False)),
                "main_empirical_allowed": bool(row.get("main_empirical_allowed", False)),
                "residual_sensitive_headline_allowed": bool(row.get("residual_sensitive_headline_allowed", False)),
                "sensitivity_allowed": False,
                "primary_panel_migration_allowed": bool(row.get("primary_panel_migration_allowed", False)),
                "blocked_reason": "" if default_ready else "Default source decision is not ready.",
                "compatibility_key": f"{DOWNSTREAM_BUNDLE_SCHEMA_VERSION}:tdcmix_downstream_default_source_decision",
                "required_caveat": str(row.get("rationale", "")),
                "default_holder_panel": str(row.get("default_holder_panel", "")),
                "default_analysis_namespace": str(row.get("default_analysis_namespace", "")),
            }
        )
    return pd.DataFrame(rows)


def build_downstream_contract_bundle_summary(bundle: pd.DataFrame) -> pd.DataFrame:
    if bundle.empty:
        return pd.DataFrame(
            [
                {
                    "schema_version": DOWNSTREAM_BUNDLE_SCHEMA_VERSION,
                    "bundle_name": DOWNSTREAM_BUNDLE_NAME,
                    "contract_rows": 0,
                    "central_calibration_rows": 0,
                    "mechanism_appendix_rows": 0,
                    "main_empirical_rows": 0,
                    "residual_sensitive_headline_rows": 0,
                    "primary_panel_migration_rows": 0,
                    "blocked_rows": 0,
                    "bundle_status": "empty",
                }
            ]
        )
    central = int(bundle["central_calibration_allowed"].astype(bool).sum())
    appendix = int(bundle["mechanism_appendix_allowed"].astype(bool).sum())
    main = int(bundle["main_empirical_allowed"].astype(bool).sum())
    residual = int(bundle["residual_sensitive_headline_allowed"].astype(bool).sum())
    primary_migration = (
        int(bundle["primary_panel_migration_allowed"].fillna(False).astype(bool).sum())
        if "primary_panel_migration_allowed" in bundle
        else 0
    )
    blocked = int(bundle["blocked_reason"].fillna("").ne("").sum()) if "blocked_reason" in bundle else 0
    status = (
        "ready_for_appendix_consumers_with_blocked_headline_paths"
        if appendix and not main and not residual
        else "requires_review"
    )
    return pd.DataFrame(
        [
            {
                "schema_version": DOWNSTREAM_BUNDLE_SCHEMA_VERSION,
                "bundle_name": DOWNSTREAM_BUNDLE_NAME,
                "contract_rows": int(len(bundle)),
                "central_calibration_rows": central,
                "mechanism_appendix_rows": appendix,
                "main_empirical_rows": main,
                "residual_sensitive_headline_rows": residual,
                "primary_panel_migration_rows": primary_migration,
                "blocked_rows": blocked,
                "bundle_status": status,
            }
        ]
    )


def build_downstream_contract_bundle_report(bundle: pd.DataFrame, summary: pd.DataFrame) -> str:
    row = summary.iloc[0].to_dict()
    lines = [
        "# Downstream Contract Bundle",
        "",
        f"- Bundle name: `{DOWNSTREAM_BUNDLE_NAME}`",
        f"- Schema version: `{DOWNSTREAM_BUNDLE_SCHEMA_VERSION}`",
        f"- Bundle status: `{row['bundle_status']}`",
        f"- Contract rows: `{int(row['contract_rows'])}`",
        f"- Central calibration rows: `{int(row['central_calibration_rows'])}`",
        f"- Mechanism appendix rows: `{int(row['mechanism_appendix_rows'])}`",
        f"- Main empirical rows: `{int(row['main_empirical_rows'])}`",
        f"- Residual-sensitive headline rows: `{int(row['residual_sensitive_headline_rows'])}`",
        f"- Primary panel migration rows: `{int(row.get('primary_panel_migration_rows', 0))}`",
        "",
        "## Allowed Rows",
        "",
    ]
    allowed = bundle.loc[
        bundle["central_calibration_allowed"].astype(bool)
        | bundle["mechanism_appendix_allowed"].astype(bool)
        | bundle["sensitivity_allowed"].astype(bool)
        | bundle.get("primary_panel_migration_allowed", pd.Series(False, index=bundle.index)).fillna(False).astype(bool)
    ]
    if allowed.empty:
        lines.append("No downstream rows are currently allowed.")
    else:
        for item in allowed.to_dict("records"):
            lines.append(
                "- "
                f"`{item['contract_row_type']}` / `{item['object_key']}`: "
                f"use `{item['downstream_use']}`, source `{item['source_layer']}`"
            )
    blocked = bundle.loc[bundle["blocked_reason"].fillna("").ne("")]
    lines.extend(["", "## Blocked Boundaries", ""])
    if blocked.empty:
        lines.append("No blocked rows are flagged.")
    else:
        for item in blocked.to_dict("records"):
            lines.append(
                "- "
                f"`{item['contract_row_type']}` / `{item['object_key']}`: "
                f"{item['blocked_reason']}"
            )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "This bundle is a downstream interface. It preserves exact-FU appendix use while keeping main empirical, causal, and residual-sensitive headline paths blocked.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_downstream_contract_bundle_outputs(
    *,
    prior_contract_path: str | Path = "data/processed/holder_absorption_prior_contract.csv",
    appendix_source_contract_path: str | Path = "data/processed/z1_exact_analysis/exact_appendix_source_contract.csv",
    final_status_packet_path: str | Path = "data/processed/z1_exact_analysis/exact_final_status_packet.csv",
    bucket_definition_contract_path: str | Path = "data/processed/z1_exact_bucket_definition_contract.csv",
    bucket_migration_policy_path: str | Path = "data/processed/z1_exact_bucket_migration_policy.csv",
    primary_migration_gate_path: str | Path = "data/processed/z1_exact_primary_migration_gate.csv",
    default_source_decision_path: str | Path = "data/processed/downstream_default_source_decision.csv",
    output_dir: str | Path = "data/processed",
) -> list[Path]:
    prior_contract = pd.read_csv(prior_contract_path)
    appendix_source_contract = pd.read_csv(appendix_source_contract_path)
    final_status_packet = pd.read_csv(final_status_packet_path)
    bucket_definition_contract = (
        pd.read_csv(bucket_definition_contract_path)
        if Path(bucket_definition_contract_path).exists()
        else None
    )
    bucket_migration_policy = (
        pd.read_csv(bucket_migration_policy_path)
        if Path(bucket_migration_policy_path).exists()
        else None
    )
    primary_migration_gate = (
        pd.read_csv(primary_migration_gate_path)
        if Path(primary_migration_gate_path).exists()
        else None
    )
    default_source_decision = (
        pd.read_csv(default_source_decision_path)
        if Path(default_source_decision_path).exists()
        else None
    )
    bundle = build_downstream_contract_bundle(
        prior_contract,
        appendix_source_contract,
        final_status_packet,
        bucket_definition_contract,
        bucket_migration_policy,
        primary_migration_gate,
        default_source_decision,
    )
    summary = build_downstream_contract_bundle_summary(bundle)
    report = build_downstream_contract_bundle_report(bundle, summary)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = out_dir / "downstream_contract_bundle.csv"
    summary_path = out_dir / "downstream_contract_bundle_summary.csv"
    report_path = out_dir / "downstream_contract_bundle.md"
    bundle.to_csv(bundle_path, index=False)
    summary.to_csv(summary_path, index=False)
    report_path.write_text(report, encoding="utf-8")
    return [bundle_path, summary_path, report_path]


def _prior_blocked_reason(row: dict) -> str:
    if bool(row.get("central_calibration_allowed", False)) or bool(row.get("sensitivity_allowed", False)):
        return ""
    return "Blocked by holder-prior contract."


def _appendix_blocked_reason(row: dict, final_status: dict) -> str:
    reasons = []
    if not bool(row.get("main_empirical_allowed", False)):
        reasons.append(str(final_status.get("main_empirical_slot_status", "main empirical use blocked")))
    if not bool(row.get("residual_sensitive_headline_allowed", False)):
        reasons.append("residual-sensitive headline use blocked")
    return "; ".join(reason for reason in reasons if reason)


def _appendix_object_key(row: dict) -> str:
    outcome = str(row.get("candidate_outcome", "") or "")
    contrast = str(row.get("candidate_contrast", "") or "")
    return f"{outcome}/{contrast}" if outcome and contrast else ""


def _compatibility_key(row: dict) -> str:
    return f"{row.get('schema_version', '')}:{row.get('contract_name', '')}"


def _policy_compatibility_key(row: dict) -> str:
    return f"{row.get('schema_version', '')}:{row.get('policy_name', '')}"
