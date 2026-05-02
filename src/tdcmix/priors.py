from __future__ import annotations

from pathlib import Path

import pandas as pd


PRIOR_CONTRACT_SCHEMA_VERSION = "0.1.0"
PRIOR_CONTRACT_NAME = "tdcmix_holder_absorption_prior_contract"
BUCKET_QUALITY_NOTES = {
    "fed": "Long-history holdings-diff proxy; compare against exact Z.1 monetary-authority transaction diagnostics before structural calibration.",
    "banks": "Grouped transaction-flow candidate; bank perimeter differs from tdcest accounting legs.",
    "row": "Grouped foreign-holder transaction-flow candidate; official/private split remains a source-review item.",
    "mmf": "Grouped money-fund flow candidate; validate against direct MMF Z.1 candidate rows before headline use.",
    "domestic_nonbank": "Grouped final domestic nonbank candidate; household residual and nonfinancial-sector perimeters need manual review.",
    "dealer_bridge": "Bridge/warehousing bucket; suitable as a separate transient absorber, not a final-holder prior.",
    "other_financial": "Non-headline financial bucket; use for reconciliation or sensitivity only.",
    "unmapped": "Residual proxy bucket; blocked from headline calibration until exact residual overlap improves.",
}


def build_holder_absorption_priors(
    panel: pd.DataFrame,
    sector_map: dict,
    *,
    recent_quarters: int = 8,
) -> pd.DataFrame:
    rows = []
    quarter = pd.to_datetime(panel["quarter"], errors="coerce", format="mixed")
    for bucket, metadata in sector_map["buckets"].items():
        share_col = f"pos_abs_share_{bucket}"
        tx_col = f"sector_tx_{bucket}"
        share = pd.to_numeric(panel.get(share_col), errors="coerce")
        tx = pd.to_numeric(panel.get(tx_col), errors="coerce")
        valid = share.dropna()
        recent = share.loc[quarter.rank(method="first", ascending=False) <= recent_quarters].dropna()
        rows.append(
            {
                "bucket": bucket,
                "label": metadata["label"],
                "role": metadata["role"],
                "include_in_headline": bool(metadata["include_in_headline"]),
                "calibration_status": _calibration_status(bucket, metadata),
                "share_mean": _stat(valid, "mean"),
                "share_median": _stat(valid, "median"),
                "share_p10": _quantile(valid, 0.10),
                "share_p90": _quantile(valid, 0.90),
                "recent_8q_share_mean": _stat(recent, "mean"),
                "share_observation_count": int(valid.shape[0]),
                "positive_tx_count": int((tx > 0).sum()),
                "negative_tx_count": int((tx < 0).sum()),
                "zero_tx_count": int((tx == 0).sum()),
                "missing_tx_count": int(tx.isna().sum()),
                "start": _date_text(quarter.loc[share.notna()].min()),
                "end": _date_text(quarter.loc[share.notna()].max()),
                "prior_use_note": _prior_use_note(bucket, metadata),
                "source_quality_note": BUCKET_QUALITY_NOTES.get(bucket, "Manual source review required."),
            }
        )
    return pd.DataFrame(rows)


def build_holder_absorption_priors_report(priors: pd.DataFrame) -> str:
    headline = priors.loc[priors["include_in_headline"].astype(bool)].copy()
    residual = priors.loc[~priors["include_in_headline"].astype(bool)].copy()
    lines = [
        "# Holder Absorption Priors",
        "",
        "These priors summarize positive-absorption shares for downstream model calibration. They are descriptive inputs, not empirical estimates of structural behavior.",
        "",
        "## Headline Buckets",
        "",
    ]
    for row in headline.sort_values("share_median", ascending=False).to_dict("records"):
        lines.append(
            "- "
            f"`{row['bucket']}`: median `{float(row['share_median']):.3f}`, "
            f"p10-p90 `{float(row['share_p10']):.3f}`-`{float(row['share_p90']):.3f}`, "
            f"status `{row['calibration_status']}`"
        )
    lines.extend(["", "## Residual / Sensitivity Buckets", ""])
    for row in residual.sort_values("bucket").to_dict("records"):
        lines.append(
            "- "
            f"`{row['bucket']}`: median `{float(row['share_median']):.3f}`, "
            f"status `{row['calibration_status']}`; {row['prior_use_note']}"
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "Residual and other-financial priors should remain sensitivity inputs until source-contract and residual-overlap gates improve.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_downstream_prior_contract(priors: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in priors.to_dict("records"):
        calibration_status = row["calibration_status"]
        rows.append(
            {
                "schema_version": PRIOR_CONTRACT_SCHEMA_VERSION,
                "contract_name": PRIOR_CONTRACT_NAME,
                "bucket": row["bucket"],
                "label": row["label"],
                "role": row["role"],
                "downstream_use": _downstream_use(calibration_status),
                "central_calibration_allowed": calibration_status == "descriptive_prior_ready_for_review",
                "sensitivity_allowed": calibration_status in {
                    "descriptive_prior_ready_for_review",
                    "headline_bridge_not_final_holder",
                    "sensitivity_only",
                },
                "structural_state_hint": _structural_state_hint(row["bucket"], calibration_status),
                "default_prior_value": row["share_median"],
                "lower_prior_value": row["share_p10"],
                "upper_prior_value": row["share_p90"],
                "source_review_required": True,
                "residual_gate_required": row["bucket"] == "unmapped",
                "contract_note": _contract_note(row["bucket"], calibration_status),
            }
        )
    return pd.DataFrame(rows)


def build_downstream_prior_contract_report(contract: pd.DataFrame) -> str:
    central = contract.loc[contract["central_calibration_allowed"].astype(bool)]
    sensitivity = contract.loc[
        contract["sensitivity_allowed"].astype(bool) & ~contract["central_calibration_allowed"].astype(bool)
    ]
    blocked = contract.loc[~contract["sensitivity_allowed"].astype(bool)]
    lines = [
        "# Downstream Prior Contract",
        "",
        f"- Contract name: `{PRIOR_CONTRACT_NAME}`",
        f"- Schema version: `{PRIOR_CONTRACT_SCHEMA_VERSION}`",
        "",
        "This contract separates central calibration candidates from transient, sensitivity-only, and blocked holder-share inputs.",
        "",
        "## Central Candidates",
        "",
    ]
    for row in central.to_dict("records"):
        lines.append(
            "- "
            f"`{row['bucket']}` default `{float(row['default_prior_value']):.3f}` "
            f"range `{float(row['lower_prior_value']):.3f}`-`{float(row['upper_prior_value']):.3f}`; "
            f"{row['contract_note']}"
        )
    lines.extend(["", "## Sensitivity / Transient Inputs", ""])
    for row in sensitivity.to_dict("records"):
        lines.append(f"- `{row['bucket']}`: `{row['downstream_use']}`; {row['contract_note']}")
    lines.extend(["", "## Blocked Inputs", ""])
    if blocked.empty:
        lines.append("No buckets are fully blocked.")
    else:
        for row in blocked.to_dict("records"):
            lines.append(f"- `{row['bucket']}`: {row['contract_note']}")
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "A downstream model should not treat this contract as structural evidence. It is an interface for scenario priors pending source review.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_holder_absorption_priors(
    panel: pd.DataFrame,
    sector_map: dict,
    output_dir: str | Path = "data/processed",
) -> list[Path]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    priors = build_holder_absorption_priors(panel, sector_map)
    report = build_holder_absorption_priors_report(priors)
    contract = build_downstream_prior_contract(priors)
    contract_report = build_downstream_prior_contract_report(contract)
    priors_path = out_dir / "holder_absorption_priors.csv"
    report_path = out_dir / "holder_absorption_priors_report.md"
    contract_path = out_dir / "holder_absorption_prior_contract.csv"
    contract_report_path = out_dir / "holder_absorption_prior_contract.md"
    priors.to_csv(priors_path, index=False)
    report_path.write_text(report, encoding="utf-8")
    contract.to_csv(contract_path, index=False)
    contract_report_path.write_text(contract_report, encoding="utf-8")
    return [priors_path, report_path, contract_path, contract_report_path]


def _calibration_status(bucket: str, metadata: dict) -> str:
    if bucket == "unmapped":
        return "blocked_residual_sensitivity_only"
    if not metadata["include_in_headline"]:
        return "sensitivity_only"
    if bucket == "dealer_bridge":
        return "headline_bridge_not_final_holder"
    return "descriptive_prior_ready_for_review"


def _prior_use_note(bucket: str, metadata: dict) -> str:
    if bucket == "unmapped":
        return "Do not use as a central model prior; residual overlap validation is not strong enough."
    if not metadata["include_in_headline"]:
        return "Use as a reconciliation or sensitivity prior, not a central headline calibration."
    if bucket == "dealer_bridge":
        return "Use as a transient warehousing prior if the downstream model has a bridge-holder state."
    return "Usable as a descriptive calibration candidate after source-contract review."


def _downstream_use(calibration_status: str) -> str:
    uses = {
        "descriptive_prior_ready_for_review": "central_candidate_after_source_review",
        "headline_bridge_not_final_holder": "transient_bridge_state_only",
        "sensitivity_only": "sensitivity_only",
        "blocked_residual_sensitivity_only": "blocked_from_central_calibration",
    }
    return uses[calibration_status]


def _structural_state_hint(bucket: str, calibration_status: str) -> str:
    if calibration_status == "headline_bridge_not_final_holder":
        return "transient_intermediary_state"
    if calibration_status == "blocked_residual_sensitivity_only":
        return "residual_reconciliation_state"
    if calibration_status == "sensitivity_only":
        return "domestic_financial_sensitivity_state"
    hints = {
        "fed": "policy_absorber_state",
        "banks": "bank_absorber_state",
        "row": "foreign_absorber_state",
        "mmf": "money_fund_plumbing_state",
        "domestic_nonbank": "domestic_final_holder_state",
    }
    return hints.get(bucket, "holder_absorber_state")


def _contract_note(bucket: str, calibration_status: str) -> str:
    if bucket == "unmapped":
        return "Blocked from central calibration until exact residual overlap improves."
    if calibration_status == "sensitivity_only":
        return "May be used only as a sensitivity/reconciliation input."
    if calibration_status == "headline_bridge_not_final_holder":
        return "Use only where the downstream model has a separate bridge-holder state."
    return "Central-candidate prior, but still requires source-contract review before headline use."


def _stat(values: pd.Series, name: str) -> float:
    if values.empty:
        return float("nan")
    return float(getattr(values, name)())


def _quantile(values: pd.Series, q: float) -> float:
    if values.empty:
        return float("nan")
    return float(values.quantile(q))


def _date_text(value: pd.Timestamp) -> str:
    if pd.isna(value):
        return ""
    return value.date().isoformat()
