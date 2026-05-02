from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd
import yaml

from tdcmix.shares import add_positive_absorption_shares


FRED_GRAPH_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
EXACT_SECTOR_COLUMNS = {
    "fed": "sector_tx_fed",
    "banks": "sector_tx_banks",
    "row": "sector_tx_row",
    "mmf": "sector_tx_mmf",
    "domestic_nonbank": "sector_tx_domestic_nonbank",
    "dealer_bridge": "sector_tx_dealer_bridge",
    "other_financial": "sector_tx_other_financial",
    "unmapped": "sector_tx_unmapped",
}
BUCKET_DEFINITION_CONTRACT_SCHEMA_VERSION = "0.1.0"
BUCKET_DEFINITION_CONTRACT_NAME = "tdcmix_z1_exact_bucket_definition_contract"
BUCKET_MIGRATION_POLICY_SCHEMA_VERSION = "0.1.0"
BUCKET_MIGRATION_POLICY_NAME = "tdcmix_z1_exact_bucket_migration_policy"
PRIMARY_MIGRATION_GATE_SCHEMA_VERSION = "0.1.0"
BUCKET_INTERPRETATION_POLICY = {
    "fed": {
        "bucket_label": "Federal Reserve / monetary authority",
        "bucket_role": "policy_absorber",
        "primary_panel_migration_status": "ready_after_identity_gate",
        "behavioral_interpretation_allowed": True,
        "manual_review_required": False,
        "definition_note": "Monetary authority Treasury transaction bucket.",
    },
    "banks": {
        "bucket_label": "Depository institutions broad perimeter",
        "bucket_role": "broad_depository_absorber",
        "primary_panel_migration_status": "requires_perimeter_freeze",
        "behavioral_interpretation_allowed": False,
        "manual_review_required": True,
        "definition_note": (
            "Broad depository bucket: U.S.-chartered depositories, credit unions, "
            "banks in U.S.-affiliated areas, and foreign banking offices in the U.S."
        ),
    },
    "row": {
        "bucket_label": "Rest of world total",
        "bucket_role": "foreign_absorber",
        "primary_panel_migration_status": "ready_after_identity_gate",
        "behavioral_interpretation_allowed": True,
        "manual_review_required": False,
        "definition_note": "Total foreign holder Treasury transaction bucket; official/private split remains a sidecar candidate.",
    },
    "mmf": {
        "bucket_label": "Money market funds total Treasuries",
        "bucket_role": "money_fund_plumbing_absorber",
        "primary_panel_migration_status": "ready_after_identity_gate",
        "behavioral_interpretation_allowed": True,
        "manual_review_required": False,
        "definition_note": "Total MMF Treasury transaction bucket; bill/non-bill split is a sidecar candidate.",
    },
    "domestic_nonbank": {
        "bucket_label": "Domestic nonbank including household residual holder",
        "bucket_role": "domestic_final_holder_with_residual_component",
        "primary_panel_migration_status": "requires_residual_holder_label",
        "behavioral_interpretation_allowed": False,
        "manual_review_required": True,
        "definition_note": (
            "Domestic nonbank bucket includes nonfinancial corporate, noncorporate, state/local, "
            "and households/nonprofits residual-holder Treasury transactions."
        ),
    },
    "dealer_bridge": {
        "bucket_label": "Security brokers and dealers",
        "bucket_role": "transient_intermediary_bridge",
        "primary_panel_migration_status": "ready_as_bridge_not_final_holder",
        "behavioral_interpretation_allowed": True,
        "manual_review_required": False,
        "definition_note": "Dealer net Treasury transactions; bridge/warehousing bucket, not final absorption.",
    },
    "other_financial": {
        "bucket_label": "Other financial institutions",
        "bucket_role": "heterogeneous_financial_absorber",
        "primary_panel_migration_status": "requires_subbucket_review",
        "behavioral_interpretation_allowed": False,
        "manual_review_required": True,
        "definition_note": (
            "Heterogeneous bucket combining GSEs, insurance/pensions, funds, ABS issuers, "
            "holding companies, and central clearing counterparties."
        ),
    },
}
BUCKET_REVIEW_RESOLUTION_POLICY = {
    "banks": {
        "decision_status": "resolved_by_broad_depository_perimeter",
        "primary_panel_migration_allowed": True,
        "behavioral_interpretation_allowed": False,
        "policy_label": "broad_depository_institutions_not_domestic_commercial_banks",
        "required_output_label": "banks_broad_depository_exact_fu",
        "required_caveat": (
            "Use as a broad depository absorption bucket. Do not describe it as domestic commercial banks only."
        ),
    },
    "domestic_nonbank": {
        "decision_status": "resolved_by_residual_holder_label",
        "primary_panel_migration_allowed": True,
        "behavioral_interpretation_allowed": False,
        "policy_label": "domestic_nonbank_including_household_np_residual_holder",
        "required_output_label": "domestic_nonbank_with_household_residual_holder_exact_fu",
        "required_caveat": (
            "Use for accounting source-layer migration only. Household/nonprofit component is a Z.1 residual-holder construction."
        ),
    },
    "other_financial": {
        "decision_status": "resolved_as_aggregate_reconciliation_bucket",
        "primary_panel_migration_allowed": True,
        "behavioral_interpretation_allowed": False,
        "policy_label": "heterogeneous_other_financial_reconciliation_bucket",
        "required_output_label": "other_financial_aggregate_exact_fu",
        "required_caveat": (
            "Use as a heterogeneous financial accounting bucket. Keep subcomponents available for mechanism review."
        ),
    },
}


def load_z1_exact_config(path: str | Path = "config/z1_exact_transaction_components.yml") -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Z.1 exact component config must load as a mapping.")
    validate_z1_exact_config(config)
    return config


def validate_z1_exact_config(config: dict) -> None:
    for field in ("schema_version", "defaults", "anchors", "components"):
        if field not in config:
            raise ValueError(f"Z.1 exact config missing {field!r}.")
    anchors = config["anchors"]
    for anchor in ("all_sectors", "issuer_marketable"):
        if anchor not in anchors:
            raise ValueError(f"Z.1 exact config missing anchor {anchor!r}.")
        _require_series_fields(anchors[anchor], anchor)
    keys = [component["key"] for component in config["components"]]
    if len(keys) != len(set(keys)):
        raise ValueError("Z.1 exact component keys must be unique.")
    for component in config["components"]:
        _require_series_fields(component, component.get("key", "component"))
        if "bucket" not in component:
            raise ValueError(f"{component['key']}: component requires bucket.")


def build_z1_exact_transaction_panel(
    config: dict,
    *,
    fetcher: Callable[[str], pd.DataFrame] | None = None,
) -> pd.DataFrame:
    fetch = fetcher or fetch_fred_series
    series_specs = _series_specs(config)
    panel: pd.DataFrame | None = None
    for spec in series_specs:
        frame = _series_frame(fetch(str(spec["fred_id"])), spec)
        panel = frame if panel is None else panel.merge(frame, on="quarter", how="outer")
    if panel is None:
        return pd.DataFrame()
    panel = panel.sort_values("quarter").reset_index(drop=True)

    component_columns = [f"z1_component_{component['key']}_bn" for component in config["components"]]
    panel["z1_component_sum_bn"] = panel[component_columns].sum(axis=1, min_count=1)
    for bucket in sorted({component["bucket"] for component in config["components"]}):
        bucket_columns = [
            f"z1_component_{component['key']}_bn"
            for component in config["components"]
            if component["bucket"] == bucket
        ]
        panel[f"z1_exact_bucket_{bucket}_bn"] = panel[bucket_columns].sum(axis=1, min_count=1)
    panel["z1_exact_official_unmapped_bn"] = (
        panel["exact_all_holder_transactions_bn"] - panel["z1_component_sum_bn"]
    )
    panel["z1_holder_vs_issuer_gap_bn"] = (
        panel["exact_all_holder_transactions_bn"]
        - panel["issuer_marketable_treasury_liability_transactions_bn"]
    )
    lead = [
        "quarter",
        "exact_all_holder_transactions_bn",
        "issuer_marketable_treasury_liability_transactions_bn",
        "z1_component_sum_bn",
        "z1_exact_official_unmapped_bn",
        "z1_holder_vs_issuer_gap_bn",
    ]
    rest = [column for column in panel.columns if column not in lead]
    return panel[lead + rest]


def build_z1_exact_reconciliation(
    exact_panel: pd.DataFrame,
    holder_panel: pd.DataFrame,
) -> pd.DataFrame:
    bucket_pairs = {
        "fed": "sector_tx_fed",
        "banks": "sector_tx_banks",
        "row": "sector_tx_row",
        "mmf": "sector_tx_mmf",
        "domestic_nonbank": "sector_tx_domestic_nonbank",
        "dealer_bridge": "sector_tx_dealer_bridge",
        "other_financial": "sector_tx_other_financial",
    }
    merged = exact_panel.merge(holder_panel, on="quarter", how="left")
    rows = []
    rows.append(_gap_summary(merged, "all_holder_vs_component_sum", "z1_exact_official_unmapped_bn"))
    rows.append(_gap_summary(merged, "all_holder_vs_issuer_liability", "z1_holder_vs_issuer_gap_bn"))
    if "sector_tx_unmapped" in merged:
        merged["legacy_proxy_minus_exact_unmapped_bn"] = (
            pd.to_numeric(merged["sector_tx_unmapped"], errors="coerce")
            - pd.to_numeric(merged["z1_exact_official_unmapped_bn"], errors="coerce")
        )
        rows.append(_gap_summary(merged, "legacy_proxy_minus_exact_unmapped", "legacy_proxy_minus_exact_unmapped_bn"))
    for bucket, panel_column in bucket_pairs.items():
        exact_column = f"z1_exact_bucket_{bucket}_bn"
        if exact_column in merged and panel_column in merged:
            merged[f"{bucket}_current_minus_exact_bn"] = (
                pd.to_numeric(merged[panel_column], errors="coerce")
                - pd.to_numeric(merged[exact_column], errors="coerce")
            )
            rows.append(_gap_summary(merged, f"{bucket}_current_minus_exact", f"{bucket}_current_minus_exact_bn"))
    return pd.DataFrame(rows)


def build_z1_exact_holder_panel(
    exact_panel: pd.DataFrame,
    holder_panel: pd.DataFrame,
) -> pd.DataFrame:
    base_columns = [
        "quarter",
        "tdc_tier1",
        "gross_marketable_issuance",
        "bill_share",
        "issue_wam",
        "wam_gap_trailing",
        "toc_change",
        "fed_remit_positive",
        "deposits",
        "bank_credit",
        "reserves",
        "onrrp",
        "mmf_treasury",
    ]
    available_base = [column for column in base_columns if column in holder_panel.columns]
    out = holder_panel[available_base].merge(
        exact_panel,
        on="quarter",
        how="outer",
    )
    for bucket, column in EXACT_SECTOR_COLUMNS.items():
        source = (
            "z1_exact_official_unmapped_bn"
            if bucket == "unmapped"
            else f"z1_exact_bucket_{bucket}_bn"
        )
        out[column] = pd.to_numeric(out[source], errors="coerce") if source in out else pd.NA
    out = add_positive_absorption_shares(out, EXACT_SECTOR_COLUMNS)
    out["holder_source_layer"] = "z1_exact_fu_transactions"
    out["residual_interpretation"] = (
        "official FU all-sector transactions minus mapped official FU components; "
        "not the legacy holdings-difference proxy"
    )
    lead = [
        "quarter",
        "holder_source_layer",
        "residual_interpretation",
        *EXACT_SECTOR_COLUMNS.values(),
        "positive_absorption_total",
        *[f"pos_abs_share_{bucket}" for bucket in EXACT_SECTOR_COLUMNS],
    ]
    rest = [column for column in out.columns if column not in lead]
    return out[lead + rest].sort_values("quarter").reset_index(drop=True)


def build_z1_exact_holder_share_diagnostics(exact_holder_panel: pd.DataFrame) -> pd.DataFrame:
    share_columns = [column for column in exact_holder_panel.columns if column.startswith("pos_abs_share_")]
    share_sum = exact_holder_panel[share_columns].sum(axis=1, skipna=True)
    share_sum.loc[exact_holder_panel[share_columns].isna().all(axis=1)] = pd.NA
    out = exact_holder_panel[["quarter", "exact_all_holder_transactions_bn", "z1_component_sum_bn"]].copy()
    out["share_sum"] = share_sum
    out["share_abs_error"] = (pd.to_numeric(out["share_sum"], errors="coerce") - 1.0).abs()
    out["share_sum_passes"] = out["share_sum"].isna() | out["share_abs_error"].le(1e-9)
    out["official_unmapped_abs_bn"] = pd.to_numeric(
        exact_holder_panel["sector_tx_unmapped"],
        errors="coerce",
    ).abs()
    out["official_unmapped_is_rounding"] = out["official_unmapped_abs_bn"].le(1e-6) | out[
        "official_unmapped_abs_bn"
    ].isna()
    return out


def build_z1_exact_gate_review(
    exact_panel: pd.DataFrame,
    reconciliation: pd.DataFrame,
    *,
    min_exact_rows: int = 40,
    max_identity_gap_bn: float = 1e-6,
) -> pd.DataFrame:
    exact_rows = int(pd.to_numeric(exact_panel["exact_all_holder_transactions_bn"], errors="coerce").notna().sum())
    identity = reconciliation.loc[reconciliation["check"].eq("all_holder_vs_component_sum")]
    holder_issuer = reconciliation.loc[reconciliation["check"].eq("all_holder_vs_issuer_liability")]
    identity_gap = float(identity["median_abs_gap_bn"].iloc[0]) if not identity.empty else float("nan")
    issuer_gap = float(holder_issuer["median_abs_gap_bn"].iloc[0]) if not holder_issuer.empty else float("nan")
    rows_pass = exact_rows >= min_exact_rows
    identity_pass = pd.notna(identity_gap) and identity_gap <= max_identity_gap_bn
    return pd.DataFrame(
        [
            {
                "gate": "z1_exact_transaction_accounting_gate",
                "passes": bool(rows_pass and identity_pass),
                "status": "ready" if rows_pass and identity_pass else "blocked",
                "exact_rows": exact_rows,
                "min_exact_rows": min_exact_rows,
                "exact_rows_pass": rows_pass,
                "median_abs_identity_gap_bn": identity_gap,
                "max_identity_gap_bn": max_identity_gap_bn,
                "identity_gap_pass": bool(identity_pass),
                "median_abs_holder_issuer_gap_bn": issuer_gap,
                "required_next_step": (
                    "Exact Z.1 transaction accounting layer is ready for residual-gate integration."
                    if rows_pass and identity_pass
                    else "Resolve missing exact rows or component identity gaps before residual-sensitive use."
                ),
            }
        ]
    )


def build_z1_exact_bucket_definition_contract(config: dict) -> pd.DataFrame:
    rows = []
    components = config.get("components", [])
    for bucket in sorted({component["bucket"] for component in components}):
        policy = BUCKET_INTERPRETATION_POLICY.get(bucket, {})
        bucket_components = [component for component in components if component["bucket"] == bucket]
        caveats = [str(component.get("caveat", "")) for component in bucket_components if component.get("caveat")]
        rows.append(
            {
                "schema_version": BUCKET_DEFINITION_CONTRACT_SCHEMA_VERSION,
                "contract_name": BUCKET_DEFINITION_CONTRACT_NAME,
                "bucket": bucket,
                "bucket_label": policy.get("bucket_label", bucket),
                "bucket_role": policy.get("bucket_role", "holder_absorber"),
                "source_layer": "z1_exact_fu_transactions",
                "component_count": int(len(bucket_components)),
                "component_keys": ";".join(component["key"] for component in bucket_components),
                "fof_codes": ";".join(component["fof_code"] for component in bucket_components),
                "fred_ids": ";".join(component["fred_id"] for component in bucket_components),
                "component_caveats": "; ".join(caveats),
                "primary_panel_migration_status": policy.get(
                    "primary_panel_migration_status",
                    "requires_review",
                ),
                "behavioral_interpretation_allowed": bool(
                    policy.get("behavioral_interpretation_allowed", False)
                ),
                "manual_review_required": bool(policy.get("manual_review_required", True)),
                "definition_note": policy.get("definition_note", "Manual bucket review required."),
            }
        )
    return pd.DataFrame(rows).sort_values("bucket").reset_index(drop=True)


def build_z1_exact_bucket_definition_report(contract: pd.DataFrame) -> str:
    ready = int(contract["primary_panel_migration_status"].str.startswith("ready").sum()) if not contract.empty else 0
    review = int(contract["manual_review_required"].astype(bool).sum()) if not contract.empty else 0
    lines = [
        "# Z.1 Exact Bucket Definition Contract",
        "",
        f"- Contract name: `{BUCKET_DEFINITION_CONTRACT_NAME}`",
        f"- Schema version: `{BUCKET_DEFINITION_CONTRACT_SCHEMA_VERSION}`",
        f"- Buckets: `{len(contract)}`",
        f"- Ready/bridge-ready buckets: `{ready}`",
        f"- Buckets requiring manual review: `{review}`",
        "",
        "## Bucket Definitions",
        "",
    ]
    for row in contract.to_dict("records"):
        lines.append(
            "- "
            f"`{row['bucket']}`: `{row['primary_panel_migration_status']}`, "
            f"components `{int(row['component_count'])}`, "
            f"behavioral interpretation `{row['behavioral_interpretation_allowed']}`. "
            f"{row['definition_note']}"
        )
        if row.get("component_caveats"):
            lines.append(f"  Caveat: {row['component_caveats']}")
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "This contract freezes the current exact-FU aggregate definitions for backend consumers. Buckets marked for review should not be migrated into a primary panel without an explicit follow-up decision.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_z1_exact_bucket_migration_policy(bucket_contract: pd.DataFrame) -> pd.DataFrame:
    review_rows = bucket_contract.loc[bucket_contract["manual_review_required"].astype(bool)].copy()
    rows = []
    for row in review_rows.to_dict("records"):
        policy = BUCKET_REVIEW_RESOLUTION_POLICY.get(row["bucket"], {})
        rows.append(
            {
                "schema_version": BUCKET_MIGRATION_POLICY_SCHEMA_VERSION,
                "policy_name": BUCKET_MIGRATION_POLICY_NAME,
                "bucket": row["bucket"],
                "source_layer": row["source_layer"],
                "review_trigger": row["primary_panel_migration_status"],
                "decision_status": policy.get("decision_status", "unresolved"),
                "primary_panel_migration_allowed": bool(
                    policy.get("primary_panel_migration_allowed", False)
                ),
                "behavioral_interpretation_allowed": bool(
                    policy.get("behavioral_interpretation_allowed", False)
                ),
                "policy_label": policy.get("policy_label", ""),
                "required_output_label": policy.get("required_output_label", ""),
                "component_keys": row.get("component_keys", ""),
                "component_caveats": row.get("component_caveats", ""),
                "required_caveat": policy.get("required_caveat", "Manual policy decision required."),
            }
        )
    return pd.DataFrame(rows).sort_values("bucket").reset_index(drop=True)


def build_z1_exact_primary_migration_gate(
    exact_gate: pd.DataFrame,
    bucket_contract: pd.DataFrame,
    migration_policy: pd.DataFrame,
) -> pd.DataFrame:
    exact_ready = (
        not exact_gate.empty
        and "status" in exact_gate
        and str(exact_gate["status"].iloc[0]) == "ready"
    )
    review_buckets = set(
        bucket_contract.loc[
            bucket_contract["manual_review_required"].astype(bool),
            "bucket",
        ].astype(str)
    )
    resolved_buckets = set(
        migration_policy.loc[
            migration_policy["primary_panel_migration_allowed"].astype(bool)
            & migration_policy["decision_status"].astype(str).str.startswith("resolved"),
            "bucket",
        ].astype(str)
    ) if not migration_policy.empty else set()
    unresolved = sorted(review_buckets.difference(resolved_buckets))
    ready = bool(exact_ready and not unresolved)
    return pd.DataFrame(
        [
            {
                "schema_version": PRIMARY_MIGRATION_GATE_SCHEMA_VERSION,
                "gate": "z1_exact_primary_panel_migration_gate",
                "status": "ready" if ready else "blocked",
                "primary_panel_migration_allowed": ready,
                "main_empirical_allowed": False,
                "residual_sensitive_headline_allowed": False,
                "exact_accounting_gate_ready": exact_ready,
                "review_bucket_count": int(len(review_buckets)),
                "resolved_review_bucket_count": int(len(resolved_buckets.intersection(review_buckets))),
                "unresolved_review_buckets": ";".join(unresolved),
                "required_next_step": (
                    "Implement exact-FU primary source-layer mode with legacy comparison output."
                    if ready
                    else "Resolve bucket migration policy rows before exact-FU primary source-layer migration."
                ),
                "required_caveat": (
                    "Gate readiness is source-layer readiness only. It does not authorize causal, main empirical, or residual-sensitive headline claims."
                ),
            }
        ]
    )


def build_z1_exact_bucket_migration_policy_report(
    migration_policy: pd.DataFrame,
    migration_gate: pd.DataFrame,
) -> str:
    gate = migration_gate.iloc[0].to_dict() if not migration_gate.empty else {}
    lines = [
        "# Z.1 Exact Bucket Migration Policy",
        "",
        f"- Policy name: `{BUCKET_MIGRATION_POLICY_NAME}`",
        f"- Schema version: `{BUCKET_MIGRATION_POLICY_SCHEMA_VERSION}`",
        f"- Migration gate status: `{gate.get('status', 'unknown')}`",
        f"- Primary panel migration allowed: `{gate.get('primary_panel_migration_allowed', False)}`",
        f"- Main empirical allowed: `{gate.get('main_empirical_allowed', False)}`",
        f"- Residual-sensitive headline allowed: `{gate.get('residual_sensitive_headline_allowed', False)}`",
        "",
        "## Review Resolutions",
        "",
    ]
    if migration_policy.empty:
        lines.append("No review-resolution rows available.")
    else:
        for row in migration_policy.to_dict("records"):
            lines.append(
                "- "
                f"`{row['bucket']}`: `{row['decision_status']}`, "
                f"migration `{row['primary_panel_migration_allowed']}`, "
                f"behavioral interpretation `{row['behavioral_interpretation_allowed']}`. "
                f"{row['required_caveat']}"
            )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            str(gate.get("required_caveat", "Source-layer policy only; no empirical promotion.")),
        ]
    )
    return "\n".join(lines) + "\n"


def build_z1_exact_report(gate: pd.DataFrame, reconciliation: pd.DataFrame) -> str:
    row = gate.iloc[0].to_dict()
    lines = [
        "# Z.1 Exact Transaction Reconciliation",
        "",
        f"- Gate status: `{row['status']}`",
        f"- Exact all-holder rows: `{int(row['exact_rows'])}` required `{int(row['min_exact_rows'])}`",
        f"- Median absolute identity gap: `{float(row['median_abs_identity_gap_bn']):.6f}` bn",
        f"- Median absolute holder-vs-issuer gap: `{float(row['median_abs_holder_issuer_gap_bn']):.6f}` bn",
        "",
        "## Gap Summary",
        "",
    ]
    for item in reconciliation.to_dict("records"):
        lines.append(
            "- "
            f"`{item['check']}`: overlap `{int(item['overlap_rows'])}`, "
            f"median abs gap `{float(item['median_abs_gap_bn']):.6f}` bn, "
            f"max abs gap `{float(item['max_abs_gap_bn']):.6f}` bn"
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "This layer uses official Z.1 transaction series. It replaces holdings-difference residuals only after source-contract review.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_z1_exact_outputs(
    *,
    config_path: str | Path = "config/z1_exact_transaction_components.yml",
    holder_panel_path: str | Path = "data/processed/holder_absorption_panel.csv",
    output_dir: str | Path = "data/processed",
) -> list[Path]:
    config = load_z1_exact_config(config_path)
    exact_panel = build_z1_exact_transaction_panel(config)
    holder_panel = pd.read_csv(holder_panel_path)
    reconciliation = build_z1_exact_reconciliation(exact_panel, holder_panel)
    exact_holder_panel = build_z1_exact_holder_panel(exact_panel, holder_panel)
    share_diagnostics = build_z1_exact_holder_share_diagnostics(exact_holder_panel)
    gate = build_z1_exact_gate_review(exact_panel, reconciliation)
    report = build_z1_exact_report(gate, reconciliation)
    bucket_contract = build_z1_exact_bucket_definition_contract(config)
    bucket_report = build_z1_exact_bucket_definition_report(bucket_contract)
    migration_policy = build_z1_exact_bucket_migration_policy(bucket_contract)
    migration_gate = build_z1_exact_primary_migration_gate(gate, bucket_contract, migration_policy)
    migration_report = build_z1_exact_bucket_migration_policy_report(
        migration_policy,
        migration_gate,
    )
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "z1_exact_transaction_panel.csv": exact_panel,
        "z1_exact_holder_absorption_panel.csv": exact_holder_panel,
        "z1_exact_holder_share_diagnostics.csv": share_diagnostics,
        "z1_exact_residual_reconciliation.csv": reconciliation,
        "z1_exact_gate_review.csv": gate,
        "z1_exact_bucket_definition_contract.csv": bucket_contract,
        "z1_exact_bucket_migration_policy.csv": migration_policy,
        "z1_exact_primary_migration_gate.csv": migration_gate,
    }
    paths = []
    for name, frame in outputs.items():
        path = out_dir / name
        frame.to_csv(path, index=False)
        paths.append(path)
    report_path = out_dir / "z1_exact_reconciliation_report.md"
    report_path.write_text(report, encoding="utf-8")
    paths.append(report_path)
    bucket_report_path = out_dir / "z1_exact_bucket_definition_contract.md"
    bucket_report_path.write_text(bucket_report, encoding="utf-8")
    paths.append(bucket_report_path)
    migration_report_path = out_dir / "z1_exact_bucket_migration_policy.md"
    migration_report_path.write_text(migration_report, encoding="utf-8")
    paths.append(migration_report_path)
    return paths


def fetch_fred_series(series_id: str) -> pd.DataFrame:
    return pd.read_csv(FRED_GRAPH_URL.format(series_id=series_id))


def _series_specs(config: dict) -> list[dict]:
    specs = [
        {**config["anchors"]["all_sectors"], "output_column": "exact_all_holder_transactions_bn"},
        {
            **config["anchors"]["issuer_marketable"],
            "output_column": "issuer_marketable_treasury_liability_transactions_bn",
        },
    ]
    for component in config["components"]:
        specs.append({**component, "output_column": f"z1_component_{component['key']}_bn"})
    return specs


def _series_frame(frame: pd.DataFrame, spec: dict) -> pd.DataFrame:
    series_id = str(spec["fred_id"])
    if "observation_date" not in frame or series_id not in frame:
        raise ValueError(f"FRED frame for {series_id} missing required columns.")
    out = frame[["observation_date", series_id]].copy()
    out["quarter"] = pd.to_datetime(out["observation_date"], errors="coerce").dt.to_period("Q").dt.end_time.dt.date.astype("string")
    out[spec["output_column"]] = pd.to_numeric(out[series_id], errors="coerce") / 1000.0
    return out[["quarter", spec["output_column"]]]


def _gap_summary(frame: pd.DataFrame, check: str, column: str) -> dict[str, object]:
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    abs_values = values.abs()
    return {
        "check": check,
        "overlap_rows": int(values.shape[0]),
        "median_gap_bn": float(values.median()) if not values.empty else float("nan"),
        "median_abs_gap_bn": float(abs_values.median()) if not values.empty else float("nan"),
        "max_abs_gap_bn": float(abs_values.max()) if not values.empty else float("nan"),
        "positive_gap_rows": int((values > 0).sum()),
        "negative_gap_rows": int((values < 0).sum()),
    }


def _require_series_fields(record: dict, name: str) -> None:
    for field in ("fred_id", "fof_code", "description"):
        if field not in record:
            raise ValueError(f"{name}: missing required field {field!r}.")
