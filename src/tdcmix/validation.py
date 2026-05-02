from __future__ import annotations

from pathlib import Path

import pandas as pd


def build_share_sum_diagnostics(panel: pd.DataFrame) -> pd.DataFrame:
    share_columns = [column for column in panel.columns if column.startswith("pos_abs_share_")]
    out = panel[["quarter"]].copy()
    out["share_sum"] = panel[share_columns].sum(axis=1, skipna=True)
    out.loc[panel[share_columns].isna().all(axis=1), "share_sum"] = pd.NA
    out["abs_error"] = (pd.to_numeric(out["share_sum"], errors="coerce") - 1.0).abs()
    out["passes"] = out["share_sum"].isna() | (out["abs_error"] <= 1e-9)
    return out


def build_residual_diagnostics(panel: pd.DataFrame) -> pd.DataFrame:
    required = {"quarter", "sector_tx_unmapped", "positive_absorption_total"}
    missing = required.difference(panel.columns)
    if missing:
        raise ValueError(f"Panel missing residual diagnostic columns: {sorted(missing)}")

    out = panel[["quarter", "sector_tx_unmapped", "positive_absorption_total"]].copy()
    residual = pd.to_numeric(out["sector_tx_unmapped"], errors="coerce")
    positive_total = pd.to_numeric(out["positive_absorption_total"], errors="coerce")
    out["positive_residual"] = residual.clip(lower=0)
    out["positive_residual_share"] = out["positive_residual"].div(
        positive_total.where(positive_total != 0)
    )
    out["abs_residual"] = residual.abs()
    out["residual_available"] = residual.notna()
    return out


def build_panel_validation_summary(panel: pd.DataFrame) -> pd.DataFrame:
    share_diag = build_share_sum_diagnostics(panel)
    residual_diag = build_residual_diagnostics(panel)
    wam_diag = build_wam_no_lookahead_diagnostics(panel)
    share_rows = share_diag["share_sum"].notna()
    residual_rows = residual_diag["residual_available"]
    high_residual = pd.to_numeric(
        residual_diag["positive_residual_share"],
        errors="coerce",
    ).gt(0.25)

    exact_residual_diag = build_exact_residual_overlap_diagnostics(panel)
    exact_overlap = exact_residual_diag["exact_residual"].notna()

    rows = [
        _summary_row("rows", "count", len(panel), True, "Total panel rows."),
        _summary_row(
            "share_sum_rows",
            "count",
            int(share_rows.sum()),
            True,
            "Rows with non-null positive absorption shares.",
        ),
        _summary_row(
            "share_sums_pass",
            "boolean",
            bool(share_diag.loc[share_rows, "passes"].all()),
            bool(share_diag.loc[share_rows, "passes"].all()),
            "Positive absorption shares sum to one when defined.",
        ),
        _summary_row(
            "residual_available_rows",
            "count",
            int(residual_rows.sum()),
            int(residual_rows.sum()) > 0,
            "Rows with computed residual/unmapped holder transactions.",
        ),
        _summary_row(
            "high_positive_residual_share_rows",
            "count",
            int(high_residual.sum()),
            True,
            "Rows where positive residual exceeds 25 percent of positive absorption.",
        ),
        _summary_row(
            "latest_quarter",
            "quarter",
            str(panel["quarter"].max()),
            True,
            "Latest quarter in the merged panel.",
        ),
        _summary_row(
            "wam_no_lookahead_pass",
            "boolean",
            bool(wam_diag["passes"].all()),
            bool(wam_diag["passes"].all()),
            "Panel trailing WAM gap equals the prior source-quarter WAM gap where comparable.",
        ),
        _summary_row(
            "exact_residual_overlap_rows",
            "count",
            int(exact_overlap.sum()),
            int(exact_overlap.sum()) > 0,
            "Rows where the long-history residual proxy can be compared with exact wamest all-holder transaction residual.",
        ),
    ]
    return pd.DataFrame(rows)


def build_scale_reconciliation(
    panel: pd.DataFrame,
    *,
    tdc_components_path: str | Path = "../tdcest/data/processed/tdc_components.csv",
) -> pd.DataFrame:
    components = pd.read_csv(tdc_components_path)
    components["quarter"] = (
        pd.to_datetime(components["date"], errors="coerce", format="mixed")
        .dt.to_period("Q")
        .dt.end_time
        .dt.date.astype("string")
    )
    pairs = {
        "fed": ("sector_tx_fed", "fed_tsy_tx"),
        "banks": ("sector_tx_banks", "bank_depository_tsy_tx"),
        "row": ("sector_tx_row", "row_tsy_tx"),
    }
    rows = []
    for bucket, (panel_column, tdc_column) in pairs.items():
        merged = panel[["quarter", panel_column]].merge(
            components[["quarter", tdc_column]],
            on="quarter",
            how="inner",
        )
        panel_bn = pd.to_numeric(merged[panel_column], errors="coerce")
        tdc_raw = pd.to_numeric(merged[tdc_column], errors="coerce")
        valid = panel_bn.notna() & tdc_raw.notna()
        if not valid.any():
            rows.append(_empty_reconciliation_row(bucket, panel_column, tdc_column))
            continue
        ratio = tdc_raw[valid].div(panel_bn[valid].where(panel_bn[valid] != 0))
        rows.append(
            {
                "bucket": bucket,
                "panel_column": panel_column,
                "tdcest_column": tdc_column,
                "overlap_rows": int(valid.sum()),
                "panel_abs_median": float(panel_bn[valid].abs().median()),
                "tdcest_abs_median": float(tdc_raw[valid].abs().median()),
                "tdcest_to_panel_ratio_median": float(ratio.replace([float("inf"), -float("inf")], pd.NA).dropna().median()),
                "interpretation": (
                    "Scale/perimeter diagnostic only; holder-share inputs use wamest/tsyparty "
                    "Z.1 groups in billions, while tdcest remains the TDC accounting anchor "
                    "and may be reported in a different source scale/perimeter."
                ),
            }
        )
    return pd.DataFrame(rows)


def build_wam_no_lookahead_diagnostics(
    panel: pd.DataFrame,
    *,
    issuance_path: str | Path = "../tsyparty/outputs/holder_response/issuance_maturity_response/quarterly_issuance_maturity.csv",
) -> pd.DataFrame:
    source = pd.read_csv(issuance_path, usecols=["date", "issuance_wam_gap_years"])
    source["quarter"] = (
        pd.to_datetime(source["date"], errors="coerce", format="mixed")
        .dt.to_period("Q")
        .dt.end_time
        .dt.date.astype("string")
    )
    source = source.sort_values("quarter")
    source["expected_wam_gap_trailing"] = pd.to_numeric(
        source["issuance_wam_gap_years"],
        errors="coerce",
    ).shift(1)
    merged = panel[["quarter", "wam_gap_trailing"]].merge(
        source[["quarter", "expected_wam_gap_trailing"]],
        on="quarter",
        how="inner",
    )
    observed = pd.to_numeric(merged["wam_gap_trailing"], errors="coerce")
    expected = pd.to_numeric(merged["expected_wam_gap_trailing"], errors="coerce")
    comparable = observed.notna() & expected.notna()
    merged["difference"] = observed - expected
    merged["passes"] = (~comparable) | (merged["difference"].abs() <= 1e-12)
    return merged


def build_exact_residual_overlap_diagnostics(
    panel: pd.DataFrame,
    *,
    wamest_sector_panel_path: str | Path = "../wamest/data/interim/z1_sector_panel_full.csv",
) -> pd.DataFrame:
    exact = pd.read_csv(
        wamest_sector_panel_path,
        usecols=["date", "sector_key", "transactions"],
    )
    exact["quarter"] = (
        pd.to_datetime(exact["date"], errors="coerce", format="mixed")
        .dt.to_period("Q")
        .dt.end_time
        .dt.date.astype("string")
    )
    total = exact.loc[exact["sector_key"] == "all_holders_total", ["quarter", "transactions"]]
    total = total.rename(columns={"transactions": "exact_all_holder_transactions"})
    named_columns = [
        "sector_tx_fed",
        "sector_tx_banks",
        "sector_tx_row",
        "sector_tx_mmf",
        "sector_tx_domestic_nonbank",
        "sector_tx_dealer_bridge",
        "sector_tx_other_financial",
    ]
    merged = panel[["quarter", "sector_tx_unmapped", *named_columns]].merge(
        total,
        on="quarter",
        how="left",
    )
    named_total = sum(
        pd.to_numeric(merged[column], errors="coerce").fillna(0)
        for column in named_columns
    )
    merged["exact_residual"] = pd.to_numeric(
        merged["exact_all_holder_transactions"],
        errors="coerce",
    ) - named_total
    merged["residual_proxy_error"] = (
        pd.to_numeric(merged["sector_tx_unmapped"], errors="coerce")
        - merged["exact_residual"]
    )
    return merged[
        [
            "quarter",
            "sector_tx_unmapped",
            "exact_residual",
            "residual_proxy_error",
            "exact_all_holder_transactions",
        ]
    ]


def build_residual_gate_review(
    residual_diagnostics: pd.DataFrame,
    exact_residual_overlap: pd.DataFrame,
    *,
    min_exact_overlap_rows: int = 40,
    max_median_abs_error_ratio: float = 0.25,
    max_high_residual_share_rows: int = 20,
) -> pd.DataFrame:
    exact = exact_residual_overlap.copy()
    exact_residual = pd.to_numeric(exact["exact_residual"], errors="coerce")
    proxy_error = pd.to_numeric(exact["residual_proxy_error"], errors="coerce")
    comparable = exact_residual.notna() & proxy_error.notna() & exact_residual.ne(0)
    error_ratio = proxy_error.loc[comparable].abs().div(exact_residual.loc[comparable].abs())

    residual_share = pd.to_numeric(
        residual_diagnostics["positive_residual_share"],
        errors="coerce",
    )
    high_residual_rows = int(residual_share.gt(0.25).sum())
    exact_overlap_rows = int(exact_residual.notna().sum())
    median_error_ratio = float(error_ratio.median()) if not error_ratio.empty else float("nan")

    overlap_pass = exact_overlap_rows >= min_exact_overlap_rows
    error_pass = bool(pd.notna(median_error_ratio) and median_error_ratio <= max_median_abs_error_ratio)
    residual_size_pass = high_residual_rows <= max_high_residual_share_rows
    gate_pass = overlap_pass and error_pass and residual_size_pass
    return pd.DataFrame(
        [
            {
                "gate": "residual_sensitive_headline_gate",
                "passes": gate_pass,
                "status": "ready" if gate_pass else "blocked",
                "exact_overlap_rows": exact_overlap_rows,
                "min_exact_overlap_rows": min_exact_overlap_rows,
                "exact_overlap_pass": overlap_pass,
                "median_abs_error_ratio": median_error_ratio,
                "max_median_abs_error_ratio": max_median_abs_error_ratio,
                "error_ratio_pass": error_pass,
                "high_positive_residual_share_rows": high_residual_rows,
                "max_high_positive_residual_share_rows": max_high_residual_share_rows,
                "residual_size_pass": residual_size_pass,
                "headline_use": "blocked" if not gate_pass else "allowed_after_review",
                "required_next_step": _residual_gate_next_step(
                    overlap_pass,
                    error_pass,
                    residual_size_pass,
                ),
            }
        ]
    )


def build_residual_gate_report(gate_review: pd.DataFrame) -> str:
    row = gate_review.iloc[0].to_dict()
    lines = [
        "# Residual Gate Review",
        "",
        f"- Status: `{row['status']}`",
        f"- Headline residual-sensitive use: `{row['headline_use']}`",
        f"- Exact residual overlap rows: `{int(row['exact_overlap_rows'])}` required `{int(row['min_exact_overlap_rows'])}`",
        f"- Median absolute proxy-error ratio: `{float(row['median_abs_error_ratio']):.3f}` maximum `{float(row['max_median_abs_error_ratio']):.3f}`",
        f"- High positive residual-share rows: `{int(row['high_positive_residual_share_rows'])}` maximum `{int(row['max_high_positive_residual_share_rows'])}`",
        "",
        "## Required Next Step",
        "",
        str(row["required_next_step"]),
        "",
        "## Boundary",
        "",
        "Residual-sensitive contrasts must remain blocked while this gate is blocked.",
    ]
    return "\n".join(lines) + "\n"


def write_validation_outputs(
    panel: pd.DataFrame,
    output_dir: str | Path = "data/processed",
) -> list[Path]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    residual_diagnostics = build_residual_diagnostics(panel)
    exact_residual_overlap = build_exact_residual_overlap_diagnostics(panel)
    residual_gate_review = build_residual_gate_review(
        residual_diagnostics,
        exact_residual_overlap,
    )
    outputs = {
        "validation_summary.csv": build_panel_validation_summary(panel),
        "share_sum_diagnostics.csv": build_share_sum_diagnostics(panel),
        "residual_diagnostics.csv": residual_diagnostics,
        "scale_reconciliation.csv": build_scale_reconciliation(panel),
        "wam_no_lookahead_diagnostics.csv": build_wam_no_lookahead_diagnostics(panel),
        "exact_residual_overlap_diagnostics.csv": exact_residual_overlap,
        "residual_gate_review.csv": residual_gate_review,
    }
    paths = []
    for name, frame in outputs.items():
        path = out_dir / name
        frame.to_csv(path, index=False)
        paths.append(path)
    report_path = out_dir / "residual_gate_report.md"
    report_path.write_text(build_residual_gate_report(residual_gate_review), encoding="utf-8")
    paths.append(report_path)
    return paths


def _summary_row(
    check: str,
    value_type: str,
    value: object,
    passes: bool,
    notes: str,
) -> dict[str, object]:
    return {
        "check": check,
        "value_type": value_type,
        "value": value,
        "passes": passes,
        "notes": notes,
    }


def _residual_gate_next_step(
    overlap_pass: bool,
    error_pass: bool,
    residual_size_pass: bool,
) -> str:
    if overlap_pass and error_pass and residual_size_pass:
        return "Residual-sensitive contrasts may proceed to manual source and design review."
    blockers = []
    if not overlap_pass:
        blockers.append("extend exact all-holder residual overlap")
    if not error_pass:
        blockers.append("reduce proxy error by improving all-sector/named-bucket reconciliation")
    if not residual_size_pass:
        blockers.append("explain or reduce high positive residual-share episodes")
    return "; ".join(blockers) + "."


def _empty_reconciliation_row(bucket: str, panel_column: str, tdc_column: str) -> dict[str, object]:
    return {
        "bucket": bucket,
        "panel_column": panel_column,
        "tdcest_column": tdc_column,
        "overlap_rows": 0,
        "panel_abs_median": pd.NA,
        "tdcest_abs_median": pd.NA,
        "tdcest_to_panel_ratio_median": pd.NA,
        "interpretation": "No overlapping non-null rows.",
    }
