from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


TARGET_COEFFICIENT = "tdc_x_holder_contrast"
BASE_PREDICTORS = [
    "tdc_intensity",
    "holder_contrast",
    "tdc_x_holder_contrast",
    "bill_share",
    "wam_gap_trailing",
    "gross_marketable_issuance",
]
REGIME_FLAGS = [
    "flag_gfc",
    "flag_covid",
    "flag_2023_debt_ceiling_bank_stress",
    "flag_qe",
    "flag_qt",
]
STOP_RULE_METADATA_COLUMNS = [
    "non_residual_path_status",
    "residual_sensitive_headline_status",
    "stop_rule_passes_all",
    "failed_stop_rules",
    "estimate_gate_status",
]
CONTRAST_DESCRIPTIONS = {
    "contrast_ru_minus_domestic": "Fed, banks, and ROW absorption share minus domestic final nonbank absorption share",
    "contrast_mmf_minus_banks": "money-market-fund absorption share minus bank absorption share",
    "contrast_private_ru_minus_mmf": "banks and ROW absorption share minus money-market-fund absorption share",
}
OUTCOME_REVIEW_NOTES = {
    "deposits": "Bank-deposit response; check TGA, fiscal payment timing, bank balance-sheet controls, and sample-period concentration.",
    "reserves": "Reserve-balance response; check Fed balance-sheet regimes, ON RRP substitution, and QE/QT flag sensitivity.",
    "onrrp": "ON RRP response; check post-2021 sample leverage, money-fund portfolio composition, and administered-rate regimes.",
    "mmf_treasury": "Money-fund Treasury holdings response; check direct MMF Z.1 candidates and bill-supply controls.",
    "bank_credit": "Bank-credit response; check securities versus loans composition before reading as balance-sheet expansion.",
}
FULLSPEC_VARIANTS = {
    "shortlist_baseline": {
        "predictors": BASE_PREDICTORS,
        "exclude_major_disruption": False,
    },
    "shortlist_exclude_major_disruption": {
        "predictors": BASE_PREDICTORS,
        "exclude_major_disruption": True,
    },
    "shortlist_regime_flags": {
        "predictors": [*BASE_PREDICTORS, *REGIME_FLAGS],
        "exclude_major_disruption": False,
    },
    "shortlist_qe_qt_controls": {
        "predictors": [*BASE_PREDICTORS, "flag_qe", "flag_qt"],
        "exclude_major_disruption": False,
    },
    "shortlist_stress_controls": {
        "predictors": [
            *BASE_PREDICTORS,
            "flag_gfc",
            "flag_covid",
            "flag_2023_debt_ceiling_bank_stress",
        ],
        "exclude_major_disruption": False,
    },
}


def build_lp_diagnostic_estimates(
    matrices: pd.DataFrame,
    stop_rule_status: pd.DataFrame,
    *,
    min_rows: int = 30,
) -> pd.DataFrame:
    if _non_residual_path_blocked(stop_rule_status):
        return _blocked_estimates(matrices)

    rows = []
    grouped = matrices.groupby(["outcome", "horizon", "contrast"], sort=True)
    for (outcome, horizon, contrast), group in grouped:
        rows.append(
            _estimate_one(
                group,
                outcome=outcome,
                horizon=int(horizon),
                contrast=contrast,
                sample="baseline",
                predictors=BASE_PREDICTORS,
                min_rows=min_rows,
            )
        )
        rows.append(
            _estimate_one(
                group.loc[group["flag_major_disruption"].eq(0)],
                outcome=outcome,
                horizon=int(horizon),
                contrast=contrast,
                sample="exclude_major_disruption",
                predictors=BASE_PREDICTORS,
                min_rows=min_rows,
            )
        )
        rows.append(
            _estimate_one(
                group,
                outcome=outcome,
                horizon=int(horizon),
                contrast=contrast,
                sample="with_regime_flags",
                predictors=[*BASE_PREDICTORS, *REGIME_FLAGS],
                min_rows=min_rows,
            )
        )
    out = pd.DataFrame(rows)
    out["interpretation_boundary"] = (
        "diagnostic association only; no causal language or residual-sensitive headline claim"
    )
    return out.sort_values(["outcome", "contrast", "horizon", "sample"]).reset_index(drop=True)


def build_lp_diagnostic_summary(estimates: pd.DataFrame) -> pd.DataFrame:
    ok = estimates.loc[estimates["status"].eq("ok")].copy()
    if ok.empty:
        return pd.DataFrame(
            columns=[
                "outcome",
                "contrast",
                "ok_cells",
                "baseline_cells",
                "baseline_positive",
                "baseline_negative",
                "sample_sign_stable_cells",
                "max_abs_t_stat",
                "min_n",
                "max_n",
                "diagnostic_status",
            ]
        )
    ok["sign"] = np.sign(pd.to_numeric(ok["beta"], errors="coerce"))
    rows = []
    for (outcome, contrast), group in ok.groupby(["outcome", "contrast"], sort=True):
        baseline = group.loc[group["sample"].eq("baseline")]
        stability = _sample_sign_stability(group)
        rows.append(
            {
                "outcome": outcome,
                "contrast": contrast,
                "ok_cells": int(len(group)),
                "baseline_cells": int(len(baseline)),
                "baseline_positive": int((baseline["sign"] > 0).sum()),
                "baseline_negative": int((baseline["sign"] < 0).sum()),
                "sample_sign_stable_cells": int(stability),
                "max_abs_t_stat": float(pd.to_numeric(group["t_stat"], errors="coerce").abs().max()),
                "min_n": int(pd.to_numeric(group["n"], errors="coerce").min()),
                "max_n": int(pd.to_numeric(group["n"], errors="coerce").max()),
                "diagnostic_status": "diagnostic_only",
            }
        )
    return pd.DataFrame(rows)


def build_lp_diagnostic_shortlist(
    summary: pd.DataFrame,
    *,
    max_rows: int = 6,
    min_abs_t_stat: float = 2.0,
) -> pd.DataFrame:
    if summary.empty:
        return summary.copy()
    out = summary.loc[
        pd.to_numeric(summary["max_abs_t_stat"], errors="coerce") >= min_abs_t_stat
    ].copy()
    out = out.sort_values(
        ["max_abs_t_stat", "sample_sign_stable_cells", "baseline_cells"],
        ascending=[False, False, False],
    ).head(max_rows)
    out["shortlist_reason"] = (
        "large diagnostic |t| and enough fitted cells; diagnostic-only triage, not evidence"
    )
    return out.reset_index(drop=True)


def build_lp_profile_plot_data(
    estimates: pd.DataFrame,
    shortlist: pd.DataFrame,
) -> pd.DataFrame:
    if shortlist.empty:
        return pd.DataFrame(columns=list(estimates.columns))
    keys = shortlist[["outcome", "contrast"]].drop_duplicates()
    out = estimates.merge(keys, on=["outcome", "contrast"], how="inner")
    out = out.loc[out["status"].eq("ok")].copy()
    out["lower_approx_95"] = pd.to_numeric(out["beta"], errors="coerce") - 1.96 * pd.to_numeric(
        out["se_hc1"],
        errors="coerce",
    )
    out["upper_approx_95"] = pd.to_numeric(out["beta"], errors="coerce") + 1.96 * pd.to_numeric(
        out["se_hc1"],
        errors="coerce",
    )
    return out.sort_values(["outcome", "contrast", "sample", "horizon"]).reset_index(drop=True)


def build_lp_shortlist_fullspec_estimates(
    matrices: pd.DataFrame,
    shortlist: pd.DataFrame,
    stop_rule_status: pd.DataFrame,
    *,
    min_rows: int = 30,
) -> pd.DataFrame:
    if _non_residual_path_blocked(stop_rule_status):
        blocked = _blocked_estimates(_shortlisted_matrices(matrices, shortlist))
        if blocked.empty:
            return pd.DataFrame(columns=[*_estimate_columns(), "fullspec_variant"])
        blocked["fullspec_variant"] = "blocked"
        return blocked[["fullspec_variant", *_estimate_columns()]]

    shortlisted = _shortlisted_matrices(matrices, shortlist)
    if shortlisted.empty:
        return pd.DataFrame(columns=["fullspec_variant", *_estimate_columns()])

    rows = []
    grouped = shortlisted.groupby(["outcome", "horizon", "contrast"], sort=True)
    for (outcome, horizon, contrast), group in grouped:
        for variant, spec in FULLSPEC_VARIANTS.items():
            work = group
            if spec["exclude_major_disruption"]:
                work = work.loc[work["flag_major_disruption"].eq(0)]
            rows.append(
                {
                    "fullspec_variant": variant,
                    **_estimate_one(
                        work,
                        outcome=outcome,
                        horizon=int(horizon),
                        contrast=contrast,
                        sample=variant,
                        predictors=list(spec["predictors"]),
                        min_rows=min_rows,
                    ),
                }
            )
    out = pd.DataFrame(rows)
    out["interpretation_boundary"] = (
        "shortlisted fuller diagnostic only; compare stability, do not read as causal evidence"
    )
    return out.sort_values(["outcome", "contrast", "horizon", "fullspec_variant"]).reset_index(drop=True)


def build_lp_shortlist_stability_summary(fullspec_estimates: pd.DataFrame) -> pd.DataFrame:
    ok = fullspec_estimates.loc[fullspec_estimates["status"].eq("ok")].copy()
    if ok.empty:
        return pd.DataFrame(
            columns=[
                "outcome",
                "contrast",
                "horizons_fitted",
                "variant_cells",
                "stable_horizon_signs",
                "unstable_horizon_signs",
                "positive_cells",
                "negative_cells",
                "max_abs_t_stat",
                "min_n",
                "max_n",
                "stability_status",
            ]
        )

    ok["sign"] = np.sign(pd.to_numeric(ok["beta"], errors="coerce"))
    rows = []
    for (outcome, contrast), group in ok.groupby(["outcome", "contrast"], sort=True):
        stable_horizons = 0
        unstable_horizons = 0
        for _, horizon_group in group.groupby("horizon", sort=True):
            signs = set(horizon_group["sign"].dropna().astype(float))
            signs.discard(0.0)
            if len(signs) == 1 and len(horizon_group) >= 2:
                stable_horizons += 1
            elif len(signs) > 1:
                unstable_horizons += 1
        rows.append(
            {
                "outcome": outcome,
                "contrast": contrast,
                "horizons_fitted": int(group["horizon"].nunique()),
                "variant_cells": int(len(group)),
                "stable_horizon_signs": int(stable_horizons),
                "unstable_horizon_signs": int(unstable_horizons),
                "positive_cells": int((group["sign"] > 0).sum()),
                "negative_cells": int((group["sign"] < 0).sum()),
                "max_abs_t_stat": float(pd.to_numeric(group["t_stat"], errors="coerce").abs().max()),
                "min_n": int(pd.to_numeric(group["n"], errors="coerce").min()),
                "max_n": int(pd.to_numeric(group["n"], errors="coerce").max()),
                "stability_status": _stability_status(stable_horizons, unstable_horizons),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["max_abs_t_stat", "stable_horizon_signs"],
        ascending=[False, False],
    ).reset_index(drop=True)


def build_pretrend_placebo_estimates(
    placebo_matrices: pd.DataFrame,
    shortlist: pd.DataFrame,
    stop_rule_status: pd.DataFrame,
    *,
    min_rows: int = 30,
) -> pd.DataFrame:
    if _non_residual_path_blocked(stop_rule_status):
        blocked = _blocked_estimates(_shortlisted_matrices(placebo_matrices, shortlist))
        if blocked.empty:
            return pd.DataFrame(columns=_estimate_columns())
        blocked["sample"] = "pretrend_blocked"
        return blocked

    shortlisted = _shortlisted_matrices(placebo_matrices, shortlist)
    if shortlisted.empty:
        return pd.DataFrame(columns=_estimate_columns())

    rows = []
    grouped = shortlisted.groupby(["outcome", "horizon", "contrast"], sort=True)
    for (outcome, horizon, contrast), group in grouped:
        rows.append(
            _estimate_one(
                group,
                outcome=outcome,
                horizon=int(horizon),
                contrast=contrast,
                sample="pretrend_placebo",
                predictors=BASE_PREDICTORS,
                min_rows=min_rows,
            )
        )
        rows.append(
            _estimate_one(
                group.loc[group["flag_major_disruption"].eq(0)],
                outcome=outcome,
                horizon=int(horizon),
                contrast=contrast,
                sample="pretrend_exclude_major_disruption",
                predictors=BASE_PREDICTORS,
                min_rows=min_rows,
            )
        )
    out = pd.DataFrame(rows)
    out["interpretation_boundary"] = (
        "pretrend/placebo timing screen; significant rows are a concern, not evidence"
    )
    return out.sort_values(["outcome", "contrast", "horizon", "sample"]).reset_index(drop=True)


def build_pretrend_placebo_summary(
    placebo_estimates: pd.DataFrame,
    *,
    concern_t_stat: float = 2.0,
) -> pd.DataFrame:
    ok = placebo_estimates.loc[placebo_estimates["status"].eq("ok")].copy()
    if ok.empty:
        return pd.DataFrame(
            columns=[
                "outcome",
                "contrast",
                "placebo_cells",
                "concerning_placebo_cells",
                "max_abs_placebo_t_stat",
                "min_n",
                "max_n",
                "pretrend_screen_status",
            ]
        )
    ok["abs_t_stat"] = pd.to_numeric(ok["t_stat"], errors="coerce").abs()
    rows = []
    for (outcome, contrast), group in ok.groupby(["outcome", "contrast"], sort=True):
        concerning = int(group["abs_t_stat"].ge(concern_t_stat).sum())
        rows.append(
            {
                "outcome": outcome,
                "contrast": contrast,
                "placebo_cells": int(len(group)),
                "concerning_placebo_cells": concerning,
                "max_abs_placebo_t_stat": float(group["abs_t_stat"].max()),
                "min_n": int(pd.to_numeric(group["n"], errors="coerce").min()),
                "max_n": int(pd.to_numeric(group["n"], errors="coerce").max()),
                "pretrend_screen_status": (
                    "passes_timing_screen" if concerning == 0 else "timing_concern"
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["concerning_placebo_cells", "max_abs_placebo_t_stat"],
        ascending=[False, False],
    ).reset_index(drop=True)


def build_pretrend_placebo_report(
    placebo_summary: pd.DataFrame,
    *,
    source_layer_label: str = "holder_source_layer",
) -> str:
    concerns = (
        int(placebo_summary["concerning_placebo_cells"].gt(0).sum())
        if "concerning_placebo_cells" in placebo_summary
        else 0
    )
    lines = [
        "# Pretrend Placebo Timing Screen",
        "",
        f"- Source layer: `{source_layer_label}`",
        f"- Shortlisted profiles screened: `{len(placebo_summary)}`",
        f"- Profiles with timing concerns: `{concerns}`",
        "- Residual-sensitive headline claims: `blocked`",
        "",
        "## Screen Summary",
        "",
    ]
    if placebo_summary.empty:
        lines.append("No pretrend placebo rows available.")
    else:
        for row in placebo_summary.to_dict("records"):
            lines.append(
                "- "
                f"`{row['outcome']}` / `{row['contrast']}`: "
                f"status `{row['pretrend_screen_status']}`, "
                f"concerning cells `{int(row['concerning_placebo_cells'])}` of `{int(row['placebo_cells'])}`, "
                f"max |t| `{float(row['max_abs_placebo_t_stat']):.3f}`"
            )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "A clean placebo screen is necessary but not sufficient for causal interpretation. A timing concern blocks promotion beyond diagnostic mechanism evidence.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_lp_shortlist_fullspec_report(
    fullspec_estimates: pd.DataFrame,
    stability: pd.DataFrame,
    shortlist: pd.DataFrame,
) -> str:
    ok_count = int(fullspec_estimates["status"].eq("ok").sum()) if "status" in fullspec_estimates else 0
    total_count = len(fullspec_estimates)
    lines = [
        "# Shortlist Fuller-Spec Diagnostic Report",
        "",
        "This report extends the six-profile diagnostic shortlist. It is not causal evidence.",
        "",
        f"- Shortlisted profiles: `{len(shortlist)}`",
        f"- Fitted fuller-spec cells: `{ok_count}` of `{total_count}`",
        "- Variants: `baseline`, `exclude major disruptions`, `regime flags`, `QE/QT controls`, `stress controls`",
        "- Residual-sensitive headline claims: `blocked`",
        "",
        "## Stability Summary",
        "",
    ]
    if stability.empty:
        lines.append("No fuller-spec stability rows available.")
    else:
        for row in stability.to_dict("records"):
            lines.append(
                "- "
                f"`{row['outcome']}` / `{row['contrast']}`: "
                f"stable horizons `{int(row['stable_horizon_signs'])}` of `{int(row['horizons_fitted'])}`, "
                f"unstable horizons `{int(row['unstable_horizon_signs'])}`, "
                f"max |t| `{float(row['max_abs_t_stat']):.3f}`, "
                f"status `{row['stability_status']}`"
            )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "Use this layer to identify specifications that need manual economic review, not to make headline claims.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_lp_shortlist_economics_review(
    fullspec_estimates: pd.DataFrame,
    stability: pd.DataFrame,
    stop_rule_status: pd.DataFrame,
) -> pd.DataFrame:
    reviewable = stability.loc[
        stability["stability_status"].isin(
            ["directionally_stable_diagnostic", "mixed_but_reviewable_diagnostic"]
        )
    ].copy()
    if reviewable.empty:
        return pd.DataFrame(
            columns=[
                "outcome",
                "contrast",
                "review_priority",
                "stability_status",
                "dominant_coefficient_sign",
                "positive_cells",
                "negative_cells",
                "stable_horizon_signs",
                "unstable_horizon_signs",
                "max_abs_t_stat",
                "contrast_definition",
                "coefficient_reading",
                "manual_review_note",
                *STOP_RULE_METADATA_COLUMNS,
            ]
        )

    metadata = _stop_rule_metadata(stop_rule_status)
    rows = []
    for row in reviewable.to_dict("records"):
        group = fullspec_estimates.loc[
            fullspec_estimates["outcome"].eq(row["outcome"])
            & fullspec_estimates["contrast"].eq(row["contrast"])
            & fullspec_estimates["status"].eq("ok")
        ].copy()
        sign = _dominant_sign(group)
        rows.append(
            {
                "outcome": row["outcome"],
                "contrast": row["contrast"],
                "review_priority": _review_priority(row),
                "stability_status": row["stability_status"],
                "dominant_coefficient_sign": sign,
                "positive_cells": int(row["positive_cells"]),
                "negative_cells": int(row["negative_cells"]),
                "stable_horizon_signs": int(row["stable_horizon_signs"]),
                "unstable_horizon_signs": int(row["unstable_horizon_signs"]),
                "max_abs_t_stat": float(row["max_abs_t_stat"]),
                "contrast_definition": CONTRAST_DESCRIPTIONS.get(row["contrast"], ""),
                "coefficient_reading": _coefficient_reading(sign, row["outcome"], row["contrast"]),
                "manual_review_note": OUTCOME_REVIEW_NOTES.get(row["outcome"], "Manual review required before interpretation."),
                **metadata,
            }
        )
    return pd.DataFrame(rows).sort_values(["review_priority", "max_abs_t_stat"], ascending=[True, False])


def build_lp_shortlist_economics_memo(review: pd.DataFrame) -> str:
    lines = [
        "# Shortlist Economics Review Memo",
        "",
        "This memo is a manual-review queue for diagnostic profiles. It is not evidence of a causal channel.",
        "",
    ]
    if review.empty:
        lines.append("No reviewable shortlist profiles passed the stability screen.")
    else:
        lines.extend(["## Review Queue", ""])
        for row in review.to_dict("records"):
            lines.extend(
                [
                    f"### {row['outcome']} / {row['contrast']}",
                    "",
                    f"- Priority: `{row['review_priority']}`",
                    f"- Stability: `{row['stability_status']}` with `{int(row['stable_horizon_signs'])}` stable and `{int(row['unstable_horizon_signs'])}` unstable horizons",
                    f"- Dominant coefficient sign: `{row['dominant_coefficient_sign']}`",
                    f"- Contrast definition: {row['contrast_definition']}",
                    f"- Diagnostic reading: {row['coefficient_reading']}",
                    f"- Manual checks: {row['manual_review_note']}",
                    f"- Estimate gate: `{row['estimate_gate_status']}`; residual-sensitive headline status: `{row['residual_sensitive_headline_status']}`",
                    "",
                ]
            )
    lines.extend(
        [
            "## Required Boundary",
            "",
            "Do not promote any row from this memo into headline evidence without separate identification, source-contract review, and residual-gate review.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_estimate_outputs(
    *,
    matrices_path: str | Path = "data/processed/horizon_design_matrices.csv",
    stop_rule_status_path: str | Path = "data/processed/stop_rule_status.csv",
    pretrend_matrices_path: str | Path | None = None,
    output_dir: str | Path = "data/processed",
) -> list[Path]:
    matrices = pd.read_csv(matrices_path)
    stop_rule_status = pd.read_csv(stop_rule_status_path)
    estimates = build_lp_diagnostic_estimates(matrices, stop_rule_status)
    summary = build_lp_diagnostic_summary(estimates)
    shortlist = build_lp_diagnostic_shortlist(summary)
    profile_plot_data = build_lp_profile_plot_data(estimates, shortlist)
    fullspec = build_lp_shortlist_fullspec_estimates(matrices, shortlist, stop_rule_status)
    stability = build_lp_shortlist_stability_summary(fullspec)
    placebo_matrices = _read_pretrend_matrices(pretrend_matrices_path, matrices_path)
    placebo_estimates = build_pretrend_placebo_estimates(placebo_matrices, shortlist, stop_rule_status)
    placebo_summary = build_pretrend_placebo_summary(placebo_estimates)
    estimates = add_stop_rule_metadata(estimates, stop_rule_status)
    summary = add_stop_rule_metadata(summary, stop_rule_status)
    shortlist = add_stop_rule_metadata(shortlist, stop_rule_status)
    profile_plot_data = add_stop_rule_metadata(profile_plot_data, stop_rule_status)
    fullspec = add_stop_rule_metadata(fullspec, stop_rule_status)
    stability = add_stop_rule_metadata(stability, stop_rule_status)
    placebo_estimates = add_stop_rule_metadata(placebo_estimates, stop_rule_status)
    placebo_summary = add_stop_rule_metadata(placebo_summary, stop_rule_status)
    economics_review = build_lp_shortlist_economics_review(fullspec, stability, stop_rule_status)
    report = build_lp_diagnostic_report(estimates, summary, stop_rule_status)
    fullspec_report = build_lp_shortlist_fullspec_report(fullspec, stability, shortlist)
    economics_memo = build_lp_shortlist_economics_memo(economics_review)
    placebo_report = build_pretrend_placebo_report(placebo_summary)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "lp_diagnostic_estimates.csv": estimates,
        "lp_diagnostic_summary.csv": summary,
        "lp_diagnostic_shortlist.csv": shortlist,
        "lp_diagnostic_profile_plot_data.csv": profile_plot_data,
        "lp_shortlist_fullspec_estimates.csv": fullspec,
        "lp_shortlist_stability_summary.csv": stability,
        "lp_shortlist_economics_review.csv": economics_review,
        "pretrend_placebo_estimates.csv": placebo_estimates,
        "pretrend_placebo_summary.csv": placebo_summary,
    }
    paths = []
    for name, frame in outputs.items():
        path = out_dir / name
        frame.to_csv(path, index=False)
        paths.append(path)
    report_path = out_dir / "lp_diagnostic_report.md"
    report_path.write_text(report, encoding="utf-8")
    paths.append(report_path)
    fullspec_report_path = out_dir / "lp_shortlist_fullspec_report.md"
    fullspec_report_path.write_text(fullspec_report, encoding="utf-8")
    paths.append(fullspec_report_path)
    economics_memo_path = out_dir / "lp_shortlist_economics_memo.md"
    economics_memo_path.write_text(economics_memo, encoding="utf-8")
    paths.append(economics_memo_path)
    placebo_report_path = out_dir / "pretrend_placebo_report.md"
    placebo_report_path.write_text(placebo_report, encoding="utf-8")
    paths.append(placebo_report_path)
    return paths


def add_stop_rule_metadata(frame: pd.DataFrame, stop_rule_status: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    metadata = _stop_rule_metadata(stop_rule_status)
    for column, value in metadata.items():
        out[column] = value
    return out


def build_lp_diagnostic_report(
    estimates: pd.DataFrame,
    summary: pd.DataFrame,
    stop_rule_status: pd.DataFrame,
) -> str:
    ok_count = int(estimates["status"].eq("ok").sum()) if "status" in estimates else 0
    total_count = len(estimates)
    path_status = (
        stop_rule_status["overall_non_residual_path_status"].dropna().astype(str).iloc[0]
        if "overall_non_residual_path_status" in stop_rule_status
        and stop_rule_status["overall_non_residual_path_status"].notna().any()
        else "unknown"
    )
    lines = [
        "# LP Diagnostic Report",
        "",
        "This report summarizes diagnostic associations only. Do not use causal language.",
        "",
        f"- Non-residual path status: `{path_status}`",
        f"- Fitted diagnostic cells: `{ok_count}` of `{total_count}`",
        "- Residual-sensitive headline claims: `blocked`",
        "",
        "## Strongest Absolute t-stat Diagnostics",
        "",
    ]
    if summary.empty:
        lines.append("No diagnostic summary rows available.")
    else:
        top = summary.sort_values("max_abs_t_stat", ascending=False).head(8)
        for row in top.to_dict("records"):
            lines.append(
                "- "
                f"`{row['outcome']}` / `{row['contrast']}`: "
                f"max |t| `{float(row['max_abs_t_stat']):.3f}`, "
                f"baseline signs +`{int(row['baseline_positive'])}`/-`{int(row['baseline_negative'])}`, "
                f"n range `{int(row['min_n'])}`-`{int(row['max_n'])}`"
            )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "These estimates are a triage layer for deciding what deserves a fuller specification, not headline evidence.",
        ]
    )
    return "\n".join(lines) + "\n"


def _estimate_one(
    frame: pd.DataFrame,
    *,
    outcome: str,
    horizon: int,
    contrast: str,
    sample: str,
    predictors: list[str],
    min_rows: int,
) -> dict[str, object]:
    columns = ["y", *predictors]
    work = frame[columns].apply(pd.to_numeric, errors="coerce").dropna()
    base = {
        "outcome": outcome,
        "horizon": horizon,
        "contrast": contrast,
        "sample": sample,
        "target_coefficient": TARGET_COEFFICIENT,
        "predictors": ";".join(predictors),
        "n": int(len(work)),
        "beta": np.nan,
        "se_hc1": np.nan,
        "t_stat": np.nan,
        "r2": np.nan,
        "coefficient_sign": "",
        "status": "insufficient_rows",
    }
    if len(work) < max(min_rows, len(predictors) + 8):
        return base

    y = work["y"].to_numpy(dtype=float)
    x_raw = work[predictors].to_numpy(dtype=float)
    x = np.column_stack([np.ones(len(work)), x_raw])
    try:
        beta, *_ = np.linalg.lstsq(x, y, rcond=None)
        fitted = x @ beta
        residual = y - fitted
        xtx_inv = np.linalg.pinv(x.T @ x)
        leverage_middle = x * residual[:, None]
        meat = leverage_middle.T @ leverage_middle
        dof = len(y) - x.shape[1]
        hc1_scale = len(y) / dof if dof > 0 else np.nan
        cov = hc1_scale * xtx_inv @ meat @ xtx_inv
        target_idx = predictors.index(TARGET_COEFFICIENT) + 1
        se = float(np.sqrt(max(cov[target_idx, target_idx], 0.0)))
        y_centered = y - y.mean()
        tss = float(y_centered @ y_centered)
        r2 = np.nan if tss == 0 else 1.0 - float(residual @ residual) / tss
        target_beta = float(beta[target_idx])
        base.update(
            {
                "beta": target_beta,
                "se_hc1": se,
                "t_stat": float(target_beta / se) if se else np.nan,
                "r2": float(r2),
                "coefficient_sign": "positive" if target_beta > 0 else "negative" if target_beta < 0 else "zero",
                "status": "ok",
            }
        )
        return base
    except (np.linalg.LinAlgError, ValueError):
        base["status"] = "linear_algebra_failure"
        return base


def _non_residual_path_blocked(stop_rule_status: pd.DataFrame) -> bool:
    if "overall_non_residual_path_status" not in stop_rule_status.columns:
        return True
    statuses = set(stop_rule_status["overall_non_residual_path_status"].dropna().astype(str))
    return "ready_for_diagnostic_estimation" not in statuses


def _read_pretrend_matrices(
    pretrend_matrices_path: str | Path | None,
    matrices_path: str | Path,
) -> pd.DataFrame:
    path = Path(pretrend_matrices_path) if pretrend_matrices_path is not None else Path(matrices_path).with_name(
        "pretrend_placebo_matrices.csv"
    )
    if not path.exists():
        return pd.DataFrame(columns=["outcome", "horizon", "contrast"])
    return pd.read_csv(path)


def _blocked_estimates(matrices: pd.DataFrame) -> pd.DataFrame:
    combos = matrices[["outcome", "horizon", "contrast"]].drop_duplicates()
    rows = []
    for row in combos.to_dict("records"):
        rows.append(
            {
                **row,
                "sample": "blocked",
                "target_coefficient": TARGET_COEFFICIENT,
                "predictors": ";".join(BASE_PREDICTORS),
                "n": 0,
                "beta": np.nan,
                "se_hc1": np.nan,
                "t_stat": np.nan,
                "r2": np.nan,
                "coefficient_sign": "",
                "status": "blocked_by_stop_rules",
                "interpretation_boundary": (
                    "diagnostic association only; no causal language or residual-sensitive headline claim"
                ),
            }
        )
    return pd.DataFrame(rows)


def _sample_sign_stability(group: pd.DataFrame) -> int:
    stable = 0
    for _, horizon_group in group.groupby(["horizon"], sort=True):
        signs = set(horizon_group["sign"].dropna().astype(float))
        signs.discard(0.0)
        if len(signs) == 1 and len(horizon_group) >= 2:
            stable += 1
    return stable


def _shortlisted_matrices(matrices: pd.DataFrame, shortlist: pd.DataFrame) -> pd.DataFrame:
    if shortlist.empty:
        return matrices.iloc[0:0].copy()
    keys = shortlist[["outcome", "contrast"]].drop_duplicates()
    return matrices.merge(keys, on=["outcome", "contrast"], how="inner")


def _estimate_columns() -> list[str]:
    return [
        "outcome",
        "horizon",
        "contrast",
        "sample",
        "target_coefficient",
        "predictors",
        "n",
        "beta",
        "se_hc1",
        "t_stat",
        "r2",
        "coefficient_sign",
        "status",
        "interpretation_boundary",
    ]


def _stability_status(stable_horizons: int, unstable_horizons: int) -> str:
    if unstable_horizons == 0 and stable_horizons >= 3:
        return "directionally_stable_diagnostic"
    if unstable_horizons <= 1 and stable_horizons >= 2:
        return "mixed_but_reviewable_diagnostic"
    return "unstable_diagnostic"


def _stop_rule_metadata(stop_rule_status: pd.DataFrame) -> dict[str, object]:
    if stop_rule_status.empty:
        return {
            "non_residual_path_status": "unknown",
            "residual_sensitive_headline_status": "blocked",
            "stop_rule_passes_all": False,
            "failed_stop_rules": "missing_stop_rule_status",
            "estimate_gate_status": "blocked_missing_stop_rules",
        }
    passes = (
        stop_rule_status["passes"].astype(bool)
        if "passes" in stop_rule_status.columns
        else pd.Series([False] * len(stop_rule_status))
    )
    failed = (
        stop_rule_status.loc[~passes, "stop_rule"].dropna().astype(str).tolist()
        if "stop_rule" in stop_rule_status.columns
        else ["missing_stop_rule_names"]
    )
    non_residual_status = (
        stop_rule_status["overall_non_residual_path_status"].dropna().astype(str).iloc[0]
        if "overall_non_residual_path_status" in stop_rule_status.columns
        and stop_rule_status["overall_non_residual_path_status"].notna().any()
        else "unknown"
    )
    all_pass = bool(passes.all())
    gate_status = (
        "diagnostic_estimation_allowed_non_residual_only"
        if all_pass and non_residual_status == "ready_for_diagnostic_estimation"
        else "blocked_by_stop_rules"
    )
    return {
        "non_residual_path_status": non_residual_status,
        "residual_sensitive_headline_status": "blocked",
        "stop_rule_passes_all": all_pass,
        "failed_stop_rules": ";".join(failed),
        "estimate_gate_status": gate_status,
    }


def _dominant_sign(group: pd.DataFrame) -> str:
    beta = pd.to_numeric(group["beta"], errors="coerce")
    positive = int((beta > 0).sum())
    negative = int((beta < 0).sum())
    if positive > negative:
        return "positive"
    if negative > positive:
        return "negative"
    return "mixed"


def _review_priority(row: dict[str, object]) -> str:
    if row["stability_status"] == "directionally_stable_diagnostic":
        return "1_review_first"
    if int(row["stable_horizon_signs"]) >= 4 and int(row["unstable_horizon_signs"]) <= 1:
        return "2_review_after_source_checks"
    return "3_review_only_if_needed"


def _coefficient_reading(sign: str, outcome: str, contrast: str) -> str:
    direction = {
        "positive": "higher",
        "negative": "lower",
        "mixed": "mixed-direction",
    }[sign]
    contrast_text = CONTRAST_DESCRIPTIONS.get(contrast, contrast)
    if sign == "mixed":
        return (
            f"The fitted interaction has mixed signs, so the {outcome} outcome has no stable directional "
            f"association with TDC intensity when {contrast_text} is high."
        )
    return (
        f"The fitted interaction is mostly {sign}: the {outcome} outcome is diagnostically associated with "
        f"{direction} values when TDC intensity is high and {contrast_text} is high."
    )
