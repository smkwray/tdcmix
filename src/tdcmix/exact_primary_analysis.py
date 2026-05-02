from __future__ import annotations

from pathlib import Path

import pandas as pd

from tdcmix.design import write_design_outputs
from tdcmix.estimates import write_estimate_outputs
from tdcmix.validation import write_validation_outputs


EXACT_PRIMARY_ANALYSIS_SCHEMA_VERSION = "0.1.0"


def build_exact_primary_artifact_comparison(
    exact_primary_dir: str | Path,
    exact_sidecar_analysis_dir: str | Path = "data/processed/z1_exact_analysis",
) -> pd.DataFrame:
    primary_dir = Path(exact_primary_dir)
    sidecar_dir = Path(exact_sidecar_analysis_dir)
    specs = [
        ("design_panel", "design_panel.csv", ["quarter"], _design_numeric_columns()),
        (
            "horizon_design_matrices",
            "horizon_design_matrices.csv",
            ["matrix_type", "outcome", "horizon", "contrast", "quarter"],
            ["y", "holder_contrast", "tdc_intensity", "tdc_x_holder_contrast"],
        ),
        (
            "lp_diagnostic_summary",
            "lp_diagnostic_summary.csv",
            ["outcome", "contrast"],
            ["ok_cells", "baseline_positive", "baseline_negative", "max_abs_t_stat", "min_n", "max_n"],
        ),
        (
            "lp_shortlist_stability_summary",
            "lp_shortlist_stability_summary.csv",
            ["outcome", "contrast"],
            ["stable_horizon_signs", "unstable_horizon_signs", "positive_cells", "negative_cells", "max_abs_t_stat"],
        ),
        (
            "pretrend_placebo_summary",
            "pretrend_placebo_summary.csv",
            ["outcome", "contrast"],
            ["placebo_cells", "concerning_placebo_cells", "max_abs_placebo_t_stat", "min_n", "max_n"],
        ),
    ]
    rows = []
    for artifact, filename, keys, numeric_columns in specs:
        primary_path = primary_dir / filename
        sidecar_path = sidecar_dir / filename
        if not primary_path.exists() or not sidecar_path.exists():
            rows.append(_missing_comparison_row(artifact, primary_path, sidecar_path))
            continue
        primary = pd.read_csv(primary_path)
        sidecar = pd.read_csv(sidecar_path)
        rows.append(
            _artifact_frame_comparison(
                artifact,
                primary,
                sidecar,
                keys=keys,
                numeric_columns=numeric_columns,
            )
        )
    return pd.DataFrame(rows)


def build_exact_primary_analysis_status(
    comparison: pd.DataFrame,
    migration_gate: pd.DataFrame,
) -> pd.DataFrame:
    gate_ready = (
        not migration_gate.empty
        and "status" in migration_gate
        and str(migration_gate["status"].iloc[0]) == "ready"
    )
    comparisons_match = (
        not comparison.empty
        and comparison["status"].isin(["matches", "available_no_numeric_columns"]).all()
    )
    return pd.DataFrame(
        [
            {
                "schema_version": EXACT_PRIMARY_ANALYSIS_SCHEMA_VERSION,
                "analysis_scope": "exact_primary_design_estimate_namespace",
                "status": "ready" if gate_ready and comparisons_match else "requires_review",
                "migration_gate_ready": bool(gate_ready),
                "artifact_comparisons": int(len(comparison)),
                "matching_artifacts": int(comparison["status"].isin(["matches", "available_no_numeric_columns"]).sum())
                if not comparison.empty
                else 0,
                "max_artifact_abs_gap": float(
                    pd.to_numeric(comparison["max_abs_gap"], errors="coerce").max()
                )
                if not comparison.empty
                else float("nan"),
                "main_empirical_allowed": False,
                "residual_sensitive_headline_allowed": False,
                "required_next_step": (
                    "Use exact-primary namespaced diagnostics for review; do not change default downstream interpretation."
                    if gate_ready and comparisons_match
                    else "Inspect exact-primary artifact comparison differences before downstream use."
                ),
            }
        ]
    )


def build_exact_primary_analysis_report(
    status: pd.DataFrame,
    comparison: pd.DataFrame,
) -> str:
    row = status.iloc[0].to_dict()
    lines = [
        "# Exact-Primary Analysis Namespace",
        "",
        f"- Status: `{row['status']}`",
        f"- Migration gate ready: `{row['migration_gate_ready']}`",
        f"- Artifact comparisons: `{int(row['artifact_comparisons'])}`",
        f"- Matching artifacts: `{int(row['matching_artifacts'])}`",
        f"- Max artifact absolute gap: `{float(row['max_artifact_abs_gap']):.6f}`",
        f"- Main empirical allowed: `{row['main_empirical_allowed']}`",
        f"- Residual-sensitive headline allowed: `{row['residual_sensitive_headline_allowed']}`",
        "",
        "## Artifact Comparisons",
        "",
    ]
    for item in comparison.to_dict("records"):
        lines.append(
            "- "
            f"`{item['artifact']}`: `{item['status']}`, overlap `{int(item['overlap_rows'])}`, "
            f"max abs gap `{float(item['max_abs_gap']):.6f}`"
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            str(row["required_next_step"]),
        ]
    )
    return "\n".join(lines) + "\n"


def write_exact_primary_analysis_outputs(
    *,
    panel_path: str | Path = "data/processed/holder_absorption_panel_exact_primary.csv",
    migration_gate_path: str | Path = "data/processed/z1_exact_primary_migration_gate.csv",
    exact_sidecar_analysis_dir: str | Path = "data/processed/z1_exact_analysis",
    output_dir: str | Path = "data/processed/exact_primary_analysis",
) -> list[Path]:
    panel = pd.read_csv(panel_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    validation_paths = write_validation_outputs(panel, out_dir)
    design_paths = write_design_outputs(
        panel,
        validation_summary_path=out_dir / "validation_summary.csv",
        exact_residual_overlap_path=out_dir / "exact_residual_overlap_diagnostics.csv",
        output_dir=out_dir,
    )
    estimate_paths = write_estimate_outputs(
        matrices_path=out_dir / "horizon_design_matrices.csv",
        stop_rule_status_path=out_dir / "stop_rule_status.csv",
        pretrend_matrices_path=out_dir / "pretrend_placebo_matrices.csv",
        output_dir=out_dir,
    )
    comparison = build_exact_primary_artifact_comparison(out_dir, exact_sidecar_analysis_dir)
    migration_gate = pd.read_csv(migration_gate_path)
    status = build_exact_primary_analysis_status(comparison, migration_gate)
    report = build_exact_primary_analysis_report(status, comparison)
    comparison_path = out_dir / "exact_primary_artifact_comparison.csv"
    status_path = out_dir / "exact_primary_analysis_status.csv"
    report_path = out_dir / "exact_primary_analysis_report.md"
    comparison.to_csv(comparison_path, index=False)
    status.to_csv(status_path, index=False)
    report_path.write_text(report, encoding="utf-8")
    return [*validation_paths, *design_paths, *estimate_paths, comparison_path, status_path, report_path]


def _artifact_frame_comparison(
    artifact: str,
    primary: pd.DataFrame,
    sidecar: pd.DataFrame,
    *,
    keys: list[str],
    numeric_columns: list[str],
) -> dict[str, object]:
    comparable_columns = [
        column for column in numeric_columns if column in primary.columns and column in sidecar.columns
    ]
    merged = primary[keys + comparable_columns].merge(
        sidecar[keys + comparable_columns],
        on=keys,
        how="inner",
        suffixes=("_primary", "_sidecar"),
    )
    if not comparable_columns:
        return {
            "artifact": artifact,
            "overlap_rows": int(len(merged)),
            "compared_numeric_columns": 0,
            "max_abs_gap": 0.0,
            "median_abs_gap": 0.0,
            "status": "available_no_numeric_columns",
        }
    gaps = []
    for column in comparable_columns:
        left = pd.to_numeric(merged[f"{column}_primary"], errors="coerce")
        right = pd.to_numeric(merged[f"{column}_sidecar"], errors="coerce")
        gaps.append((left - right).abs())
    all_gaps = pd.concat(gaps, ignore_index=True).dropna()
    max_gap = float(all_gaps.max()) if not all_gaps.empty else 0.0
    median_gap = float(all_gaps.median()) if not all_gaps.empty else 0.0
    return {
        "artifact": artifact,
        "overlap_rows": int(len(merged)),
        "compared_numeric_columns": int(len(comparable_columns)),
        "max_abs_gap": max_gap,
        "median_abs_gap": median_gap,
        "status": "matches" if max_gap <= 1e-9 else "differs",
    }


def _missing_comparison_row(artifact: str, primary_path: Path, sidecar_path: Path) -> dict[str, object]:
    return {
        "artifact": artifact,
        "overlap_rows": 0,
        "compared_numeric_columns": 0,
        "max_abs_gap": float("nan"),
        "median_abs_gap": float("nan"),
        "status": f"missing:{not primary_path.exists()}:{not sidecar_path.exists()}",
    }


def _design_numeric_columns() -> list[str]:
    return [
        "tdc_tier1",
        "tdc_intensity",
        "gross_marketable_issuance",
        "bill_share",
        "issue_wam",
        "wam_gap_trailing",
        "share_ru_like",
        "share_private_ru_like",
        "share_mmf_plumbing",
        "share_domestic_final",
        "share_bridge_financial",
        "share_named_headline",
        "share_residual",
        "contrast_ru_minus_domestic",
        "contrast_mmf_minus_banks",
        "contrast_private_ru_minus_mmf",
    ]
