from __future__ import annotations

from pathlib import Path

import pandas as pd

from tdcmix.design import (
    build_design_panel,
    build_design_readiness,
    build_first_stage_summary,
    build_horizon_design_matrices,
    build_horizon_matrix_summary,
    build_pretrend_placebo_matrices,
    build_stop_rule_status,
)
from tdcmix.estimates import (
    BASE_PREDICTORS,
    build_lp_diagnostic_estimates,
    build_lp_diagnostic_shortlist,
    build_lp_diagnostic_summary,
    build_lp_profile_plot_data,
    build_lp_shortlist_economics_memo,
    build_lp_shortlist_economics_review,
    build_lp_shortlist_fullspec_estimates,
    build_lp_shortlist_fullspec_report,
    build_lp_shortlist_stability_summary,
    build_lp_diagnostic_report,
    build_pretrend_placebo_estimates,
    build_pretrend_placebo_report,
    build_pretrend_placebo_summary,
    add_stop_rule_metadata,
    _estimate_one,
)
from tdcmix.validation import (
    build_exact_residual_overlap_diagnostics,
    build_panel_validation_summary,
    build_residual_diagnostics,
    build_residual_gate_review,
    build_residual_gate_report,
    build_share_sum_diagnostics,
)


APPENDIX_SOURCE_CONTRACT_SCHEMA_VERSION = "0.1.0"
APPENDIX_SOURCE_CONTRACT_NAME = "tdcmix_exact_fu_appendix_source_contract"


def build_source_layer_contrast_comparison(
    legacy_design: pd.DataFrame,
    exact_design: pd.DataFrame,
) -> pd.DataFrame:
    contrasts = [
        "share_ru_like",
        "share_private_ru_like",
        "share_mmf_plumbing",
        "share_domestic_final",
        "share_bridge_financial",
        "share_residual",
        "contrast_ru_minus_domestic",
        "contrast_mmf_minus_banks",
        "contrast_private_ru_minus_mmf",
    ]
    merged = legacy_design[["quarter", *contrasts]].merge(
        exact_design[["quarter", *contrasts]],
        on="quarter",
        how="inner",
        suffixes=("_legacy", "_exact_z1"),
    )
    rows = []
    for contrast in contrasts:
        legacy = pd.to_numeric(merged[f"{contrast}_legacy"], errors="coerce")
        exact = pd.to_numeric(merged[f"{contrast}_exact_z1"], errors="coerce")
        comparable = legacy.notna() & exact.notna()
        diff = exact.loc[comparable] - legacy.loc[comparable]
        corr = _safe_corr(legacy.loc[comparable], exact.loc[comparable])
        rows.append(
            {
                "metric": contrast,
                "overlap_rows": int(comparable.sum()),
                "legacy_nonnull_rows": int(legacy.notna().sum()),
                "exact_z1_nonnull_rows": int(exact.notna().sum()),
                "median_exact_minus_legacy": float(diff.median()) if not diff.empty else float("nan"),
                "median_abs_exact_minus_legacy": float(diff.abs().median()) if not diff.empty else float("nan"),
                "max_abs_exact_minus_legacy": float(diff.abs().max()) if not diff.empty else float("nan"),
                "legacy_exact_correlation": float(corr) if pd.notna(corr) else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def build_source_layer_estimate_comparison(
    legacy_summary: pd.DataFrame,
    exact_summary: pd.DataFrame,
) -> pd.DataFrame:
    keys = ["outcome", "contrast"]
    metrics = [
        "ok_cells",
        "baseline_cells",
        "baseline_positive",
        "baseline_negative",
        "sample_sign_stable_cells",
        "max_abs_t_stat",
        "min_n",
        "max_n",
    ]
    merged = legacy_summary[keys + metrics].merge(
        exact_summary[keys + metrics],
        on=keys,
        how="outer",
        suffixes=("_legacy", "_exact_z1"),
    )
    for metric in metrics:
        left = pd.to_numeric(merged[f"{metric}_legacy"], errors="coerce")
        right = pd.to_numeric(merged[f"{metric}_exact_z1"], errors="coerce")
        merged[f"{metric}_exact_minus_legacy"] = right - left
    merged["comparison_status"] = merged.apply(_estimate_comparison_status, axis=1)
    return merged.sort_values(["comparison_status", "outcome", "contrast"]).reset_index(drop=True)


def build_source_layer_stability_comparison(
    legacy_stability: pd.DataFrame,
    exact_stability: pd.DataFrame,
) -> pd.DataFrame:
    keys = ["outcome", "contrast"]
    keep = [
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
    merged = legacy_stability[keys + keep].merge(
        exact_stability[keys + keep],
        on=keys,
        how="outer",
        suffixes=("_legacy", "_exact_z1"),
    )
    merged["stability_status_changed"] = (
        merged["stability_status_legacy"].fillna("missing")
        != merged["stability_status_exact_z1"].fillna("missing")
    )
    for metric in [
        "stable_horizon_signs",
        "unstable_horizon_signs",
        "positive_cells",
        "negative_cells",
        "max_abs_t_stat",
    ]:
        left = pd.to_numeric(merged[f"{metric}_legacy"], errors="coerce")
        right = pd.to_numeric(merged[f"{metric}_exact_z1"], errors="coerce")
        merged[f"{metric}_exact_minus_legacy"] = right - left
    return merged.sort_values(["stability_status_changed", "outcome", "contrast"], ascending=[False, True, True]).reset_index(drop=True)


def build_source_layer_comparison_report(
    contrast_comparison: pd.DataFrame,
    estimate_comparison: pd.DataFrame,
    stability_comparison: pd.DataFrame,
    exact_gate: pd.DataFrame | None = None,
) -> str:
    exact_status = "unknown"
    exact_rows = "unknown"
    if exact_gate is not None and not exact_gate.empty:
        exact_status = str(exact_gate["status"].iloc[0]) if "status" in exact_gate else "unknown"
        exact_rows = str(int(exact_gate["exact_rows"].iloc[0])) if "exact_rows" in exact_gate else "unknown"
    changed_estimates = int(estimate_comparison["comparison_status"].ne("same_sign_mix").sum()) if not estimate_comparison.empty else 0
    changed_stability = int(stability_comparison["stability_status_changed"].sum()) if not stability_comparison.empty else 0
    lines = [
        "# Exact Z.1 Source-Layer Comparison",
        "",
        f"- Exact accounting gate: `{exact_status}`",
        f"- Exact all-holder rows: `{exact_rows}`",
        f"- Contrast metrics compared: `{len(contrast_comparison)}`",
        f"- Estimate summary rows with changed sign mix/status: `{changed_estimates}`",
        f"- Stability rows with changed status: `{changed_stability}`",
        "- Interpretation: diagnostic comparison only; exact FU accounting readiness does not by itself license causal language.",
        "",
        "## Largest Contrast Differences",
        "",
    ]
    if contrast_comparison.empty:
        lines.append("No contrast comparison rows available.")
    else:
        top = contrast_comparison.sort_values("median_abs_exact_minus_legacy", ascending=False).head(8)
        for row in top.to_dict("records"):
            lines.append(
                "- "
                f"`{row['metric']}`: overlap `{int(row['overlap_rows'])}`, "
                f"median abs difference `{float(row['median_abs_exact_minus_legacy']):.6f}`, "
                f"correlation `{float(row['legacy_exact_correlation']):.3f}`"
            )
    lines.extend(["", "## Estimate Status Changes", ""])
    changed = estimate_comparison.loc[estimate_comparison["comparison_status"].ne("same_sign_mix")]
    if changed.empty:
        lines.append("No estimate-summary sign-mix changes across source layers.")
    else:
        for row in changed.head(12).to_dict("records"):
            lines.append(
                "- "
                f"`{row['outcome']}` / `{row['contrast']}`: `{row['comparison_status']}`, "
                f"legacy +/- `{_as_int(row.get('baseline_positive_legacy'))}`/`{_as_int(row.get('baseline_negative_legacy'))}`, "
                f"exact +/- `{_as_int(row.get('baseline_positive_exact_z1'))}`/`{_as_int(row.get('baseline_negative_exact_z1'))}`"
            )
    lines.extend(["", "## Boundary", ""])
    lines.append(
        "Use the exact FU layer as the accounting-preferred mechanism appendix source. Keep residual-sensitive and causal claims blocked until source promotion, pretrend, regime, and identification checks are complete."
    )
    return "\n".join(lines) + "\n"


def build_source_layer_promotion_status(
    contrast_comparison: pd.DataFrame,
    estimate_comparison: pd.DataFrame,
    placebo_summary: pd.DataFrame,
    exact_gate: pd.DataFrame,
) -> pd.DataFrame:
    exact_ready = (
        not exact_gate.empty
        and "status" in exact_gate
        and str(exact_gate["status"].iloc[0]) == "ready"
    )
    core = contrast_comparison.loc[
        contrast_comparison["metric"].isin(
            [
                "contrast_ru_minus_domestic",
                "contrast_mmf_minus_banks",
                "contrast_private_ru_minus_mmf",
            ]
        )
    ].copy()
    min_core_corr = float(core["legacy_exact_correlation"].min()) if not core.empty else float("nan")
    max_core_median_abs_diff = (
        float(core["median_abs_exact_minus_legacy"].max()) if not core.empty else float("nan")
    )
    changed_rows = (
        int(estimate_comparison["comparison_status"].ne("same_sign_mix").sum())
        if "comparison_status" in estimate_comparison
        else 0
    )
    timing_concerns = (
        int(placebo_summary["pretrend_screen_status"].eq("timing_concern").sum())
        if "pretrend_screen_status" in placebo_summary
        else 0
    )
    recommendation = (
        "promote_exact_fu_for_mechanism_appendix_source_layer"
        if exact_ready
        else "hold_exact_fu_promotion_until_accounting_gate_passes"
    )
    main_slot_status = (
        "blocked_by_timing_and_diagnostic_instability"
        if timing_concerns or changed_rows
        else "still_requires_identification_review"
    )
    return pd.DataFrame(
        [
            {
                "decision_scope": "z1_exact_source_layer_promotion",
                "recommendation": recommendation,
                "main_empirical_slot_status": main_slot_status,
                "exact_accounting_gate_ready": bool(exact_ready),
                "min_core_contrast_legacy_exact_correlation": min_core_corr,
                "max_core_contrast_median_abs_difference": max_core_median_abs_diff,
                "estimate_summary_changed_rows": changed_rows,
                "exact_pretrend_timing_concern_profiles": timing_concerns,
                "source_layer_to_use_for_appendix": (
                    "z1_exact_fu_transactions" if exact_ready else "legacy_holder_panel"
                ),
                "residual_sensitive_headline_status": "blocked",
                "rationale": (
                    "Exact FU accounting closes and core non-residual contrasts are highly correlated with the legacy layer, "
                    "so exact FU should be the accounting-preferred mechanism appendix source. Main empirical promotion remains blocked "
                    "because diagnostic rows move under source-layer replacement and exact-layer placebo timing concerns remain."
                ),
            }
        ]
    )


def build_source_layer_promotion_report(status: pd.DataFrame) -> str:
    row = status.iloc[0].to_dict()
    lines = [
        "# Source-Layer Promotion Status",
        "",
        f"- Recommendation: `{row['recommendation']}`",
        f"- Main empirical slot status: `{row['main_empirical_slot_status']}`",
        f"- Appendix source layer: `{row['source_layer_to_use_for_appendix']}`",
        f"- Exact accounting gate ready: `{row['exact_accounting_gate_ready']}`",
        f"- Minimum core contrast source-layer correlation: `{float(row['min_core_contrast_legacy_exact_correlation']):.3f}`",
        f"- Maximum core contrast median absolute difference: `{float(row['max_core_contrast_median_abs_difference']):.6f}`",
        f"- Estimate summary rows changed by source layer: `{int(row['estimate_summary_changed_rows'])}`",
        f"- Exact-layer pretrend timing concern profiles: `{int(row['exact_pretrend_timing_concern_profiles'])}`",
        f"- Residual-sensitive headline status: `{row['residual_sensitive_headline_status']}`",
        "",
        "## Rationale",
        "",
        str(row["rationale"]),
        "",
        "## Boundary",
        "",
        "This promotes the exact FU source layer for mechanism appendix accounting. It does not promote the project to a main empirical slot.",
    ]
    return "\n".join(lines) + "\n"


def build_exact_appendix_candidate_triage(
    economics_review: pd.DataFrame,
    placebo_summary: pd.DataFrame,
    stability_comparison: pd.DataFrame,
) -> pd.DataFrame:
    if economics_review.empty:
        return pd.DataFrame(
            columns=[
                "outcome",
                "contrast",
                "appendix_candidate_status",
                "recommended_use",
                "review_priority",
                "stability_status",
                "pretrend_screen_status",
                "concerning_placebo_cells",
                "max_abs_placebo_t_stat",
                "source_layer_stability_status_changed",
                "dominant_coefficient_sign",
                "max_abs_t_stat",
                "triage_note",
            ]
        )
    placebo_cols = [
        "outcome",
        "contrast",
        "pretrend_screen_status",
        "concerning_placebo_cells",
        "max_abs_placebo_t_stat",
    ]
    stability_cols = [
        "outcome",
        "contrast",
        "stability_status_changed",
        "stability_status_legacy",
        "stability_status_exact_z1",
    ]
    out = economics_review.merge(
        placebo_summary[placebo_cols],
        on=["outcome", "contrast"],
        how="left",
    ).merge(
        stability_comparison[stability_cols],
        on=["outcome", "contrast"],
        how="left",
    )
    out["source_layer_stability_status_changed"] = out["stability_status_changed"].fillna(False).astype(bool)
    out["appendix_candidate_status"] = out.apply(_appendix_candidate_status, axis=1)
    out["recommended_use"] = out["appendix_candidate_status"].map(
        {
            "appendix_candidate": "mechanism_appendix_candidate",
            "hold_timing_concern": "hold_until_timing_concern_resolved",
            "hold_source_layer_instability": "hold_until_source_layer_stability_review",
        }
    )
    out["triage_note"] = out.apply(_appendix_triage_note, axis=1)
    keep = [
        "outcome",
        "contrast",
        "appendix_candidate_status",
        "recommended_use",
        "review_priority",
        "stability_status",
        "pretrend_screen_status",
        "concerning_placebo_cells",
        "max_abs_placebo_t_stat",
        "source_layer_stability_status_changed",
        "stability_status_legacy",
        "stability_status_exact_z1",
        "dominant_coefficient_sign",
        "max_abs_t_stat",
        "coefficient_reading",
        "manual_review_note",
        "triage_note",
    ]
    return out[keep].sort_values(
        ["appendix_candidate_status", "review_priority", "max_abs_t_stat"],
        ascending=[True, True, False],
    ).reset_index(drop=True)


def build_exact_appendix_candidate_report(triage: pd.DataFrame) -> str:
    candidates = int(triage["appendix_candidate_status"].eq("appendix_candidate").sum()) if not triage.empty else 0
    timing_holds = int(triage["appendix_candidate_status"].eq("hold_timing_concern").sum()) if not triage.empty else 0
    source_holds = int(triage["appendix_candidate_status"].eq("hold_source_layer_instability").sum()) if not triage.empty else 0
    lines = [
        "# Exact-FU Appendix Candidate Triage",
        "",
        f"- Mechanism appendix candidates: `{candidates}`",
        f"- Held for timing concerns: `{timing_holds}`",
        f"- Held for source-layer stability review: `{source_holds}`",
        "- Residual-sensitive headline claims: `blocked`",
        "",
        "## Candidate Rows",
        "",
    ]
    candidate_rows = triage.loc[triage["appendix_candidate_status"].eq("appendix_candidate")]
    if candidate_rows.empty:
        lines.append("No exact-FU reviewable profiles pass the current timing and source-layer screens.")
    else:
        for row in candidate_rows.to_dict("records"):
            lines.append(
                "- "
                f"`{row['outcome']}` / `{row['contrast']}`: "
                f"sign `{row['dominant_coefficient_sign']}`, "
                f"max |t| `{float(row['max_abs_t_stat']):.3f}`, "
                f"placebo max |t| `{float(row['max_abs_placebo_t_stat']):.3f}`"
            )
    held = triage.loc[triage["appendix_candidate_status"].ne("appendix_candidate")]
    lines.extend(["", "## Held Rows", ""])
    if held.empty:
        lines.append("No reviewable profiles are held by this triage.")
    else:
        for row in held.to_dict("records"):
            lines.append(
                "- "
                f"`{row['outcome']}` / `{row['contrast']}`: "
                f"`{row['appendix_candidate_status']}`; {row['triage_note']}"
            )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "Candidate rows are mechanism-appendix diagnostics only. They still require written economic interpretation and are not causal evidence.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_exact_mechanism_appendix_candidate_table(
    triage: pd.DataFrame,
    fullspec_estimates: pd.DataFrame,
    placebo_estimates: pd.DataFrame,
) -> pd.DataFrame:
    candidates = triage.loc[triage["appendix_candidate_status"].eq("appendix_candidate")].copy()
    if candidates.empty:
        return pd.DataFrame(
            columns=[
                "outcome",
                "contrast",
                "horizon",
                "fullspec_variants",
                "positive_variant_cells",
                "negative_variant_cells",
                "max_abs_t_stat",
                "median_beta",
                "baseline_beta",
                "baseline_t_stat",
                "max_abs_placebo_t_stat",
                "appendix_sentence",
            ]
        )
    rows = []
    for candidate in candidates.to_dict("records"):
        key = (
            fullspec_estimates["outcome"].eq(candidate["outcome"])
            & fullspec_estimates["contrast"].eq(candidate["contrast"])
            & fullspec_estimates["status"].eq("ok")
        )
        placebo_key = (
            placebo_estimates["outcome"].eq(candidate["outcome"])
            & placebo_estimates["contrast"].eq(candidate["contrast"])
            & placebo_estimates["status"].eq("ok")
        )
        placebo_abs_t = pd.to_numeric(placebo_estimates.loc[placebo_key, "t_stat"], errors="coerce").abs()
        max_placebo = float(placebo_abs_t.max()) if not placebo_abs_t.empty else float("nan")
        for horizon, group in fullspec_estimates.loc[key].groupby("horizon", sort=True):
            beta = pd.to_numeric(group["beta"], errors="coerce")
            t_stat = pd.to_numeric(group["t_stat"], errors="coerce")
            baseline = group.loc[group["fullspec_variant"].eq("shortlist_baseline")]
            baseline_beta = pd.to_numeric(baseline["beta"], errors="coerce")
            baseline_t = pd.to_numeric(baseline["t_stat"], errors="coerce")
            sign_word = "positive" if int((beta > 0).sum()) >= int((beta < 0).sum()) else "negative"
            rows.append(
                {
                    "outcome": candidate["outcome"],
                    "contrast": candidate["contrast"],
                    "horizon": int(horizon),
                    "fullspec_variants": int(len(group)),
                    "positive_variant_cells": int((beta > 0).sum()),
                    "negative_variant_cells": int((beta < 0).sum()),
                    "max_abs_t_stat": float(t_stat.abs().max()),
                    "median_beta": float(beta.median()),
                    "baseline_beta": float(baseline_beta.iloc[0]) if not baseline_beta.empty else float("nan"),
                    "baseline_t_stat": float(baseline_t.iloc[0]) if not baseline_t.empty else float("nan"),
                    "max_abs_placebo_t_stat": max_placebo,
                    "appendix_sentence": (
                        f"For {candidate['outcome']} at h={int(horizon)}, the exact-FU diagnostic "
                        f"interaction for {candidate['contrast']} is {sign_word} in "
                        f"{int((beta > 0).sum())} of {int(len(group))} fuller-spec variants; "
                        f"this is a mechanism diagnostic only."
                    ),
                }
            )
    return pd.DataFrame(rows).sort_values(["outcome", "contrast", "horizon"]).reset_index(drop=True)


def build_exact_mechanism_appendix_memo(
    triage: pd.DataFrame,
    candidate_table: pd.DataFrame,
    promotion_status: pd.DataFrame,
    concentration_summary: pd.DataFrame | None = None,
    regime_summary: pd.DataFrame | None = None,
    sensitivity_summary: pd.DataFrame | None = None,
) -> str:
    candidate_rows = triage.loc[triage["appendix_candidate_status"].eq("appendix_candidate")]
    held_rows = triage.loc[triage["appendix_candidate_status"].ne("appendix_candidate")]
    promotion = promotion_status.iloc[0].to_dict() if not promotion_status.empty else {}
    lines = [
        "# Exact-FU Mechanism Appendix Memo",
        "",
        "This memo is draft appendix language for the exact Z.1/FRED FU source layer. It is not a main empirical claim.",
        "",
        f"- Appendix source layer: `{promotion.get('source_layer_to_use_for_appendix', 'unknown')}`",
        f"- Source-layer recommendation: `{promotion.get('recommendation', 'unknown')}`",
        f"- Main empirical slot status: `{promotion.get('main_empirical_slot_status', 'unknown')}`",
        f"- Residual-sensitive headline status: `{promotion.get('residual_sensitive_headline_status', 'blocked')}`",
        f"- Mechanism appendix candidates: `{len(candidate_rows)}`",
        f"- Held reviewable profiles: `{len(held_rows)}`",
        "",
        "## Draft Claim",
        "",
    ]
    if candidate_rows.empty:
        lines.append("No exact-FU profile currently passes the appendix candidate triage.")
    else:
        for row in candidate_rows.to_dict("records"):
            lines.extend(
                [
                    (
                        f"A timing-clean exact-FU diagnostic profile links `{row['outcome']}` to "
                        f"`{row['contrast']}`. The sign is `{row['dominant_coefficient_sign']}` "
                        f"across the fuller-spec review, the exact-FU pretrend screen passes, and "
                        "the profile is suitable only as mechanism-appendix evidence."
                    ),
                    "",
                    f"Interpretation note: {row['coefficient_reading']}",
                    "",
                ]
            )
    lines.extend(["## Horizon Detail", ""])
    if candidate_table.empty:
        lines.append("No candidate horizon rows available.")
    else:
        for row in candidate_table.to_dict("records"):
            lines.append(
                "- "
                f"h=`{int(row['horizon'])}`: positive variants `{int(row['positive_variant_cells'])}`/"
                f"`{int(row['fullspec_variants'])}`, median beta `{float(row['median_beta']):.3f}`, "
                f"max |t| `{float(row['max_abs_t_stat']):.3f}`, "
                f"max placebo |t| `{float(row['max_abs_placebo_t_stat']):.3f}`"
            )
    lines.extend(["", "## Sample Concentration", ""])
    if concentration_summary is None or concentration_summary.empty:
        lines.append("No sample-concentration rows available.")
    else:
        for row in concentration_summary.to_dict("records"):
            lines.append(
                "- "
                f"h=`{int(row['horizon'])}`: n `{int(row['n'])}`, "
                f"top-5 interaction share `{float(row['top5_abs_interaction_share']):.3f}`, "
                f"top-10 share `{float(row['top10_abs_interaction_share']):.3f}`, "
                f"major-disruption rows `{int(row['major_disruption_rows'])}`, "
                f"QT rows `{int(row['qt_rows'])}`"
            )
    lines.extend(["", "## Regime Coverage", ""])
    if regime_summary is None or regime_summary.empty:
        lines.append("No regime-summary rows available.")
    else:
        for row in regime_summary.to_dict("records"):
            if str(row["regime"]) not in {"full_sample", "exclude_major_disruption", "qt", "post_2021"}:
                continue
            lines.append(
                "- "
                f"h=`{int(row['horizon'])}` / `{row['regime']}`: rows `{int(row['n'])}`, "
                f"mean interaction `{float(row['mean_interaction']):.6f}`, "
                f"median abs interaction `{float(row['median_abs_interaction']):.6f}`"
            )
    lines.extend(["", "## Coefficient Sensitivity", ""])
    if sensitivity_summary is None or sensitivity_summary.empty:
        lines.append("No coefficient-sensitivity rows available.")
    else:
        for row in sensitivity_summary.to_dict("records"):
            lines.append(
                "- "
                f"h=`{int(row['horizon'])}`: `{row['sign_robustness_status']}`, "
                f"positive variants `{int(row['positive_variants'])}`/`{int(row['ok_variants'])}`, "
                f"median beta `{float(row['median_beta']):.3f}`, "
                f"max |t| `{float(row['max_abs_t_stat']):.3f}`"
            )
    lines.extend(["", "## Held Profiles", ""])
    if held_rows.empty:
        lines.append("No reviewable profiles are held by the current triage.")
    else:
        for row in held_rows.to_dict("records"):
            lines.append(
                "- "
                f"`{row['outcome']}` / `{row['contrast']}`: held as `{row['appendix_candidate_status']}`. "
                f"{row['triage_note']}"
            )
    lines.extend(
        [
            "",
            "## Required Caveats",
            "",
            "- The exact-FU layer is the accounting-preferred mechanism appendix source, not a causal design.",
            "- Residual-sensitive headline claims remain blocked.",
            "- The surviving candidate is descriptive and should be written as a mechanism diagnostic.",
            "- Held profiles should not be promoted unless their timing concerns are resolved by a stricter design.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_exact_candidate_sample_concentration(
    triage: pd.DataFrame,
    matrices: pd.DataFrame,
    *,
    top_n: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidates = triage.loc[triage["appendix_candidate_status"].eq("appendix_candidate")]
    summary_rows = []
    episode_rows = []
    for candidate in candidates.to_dict("records"):
        base = matrices.loc[
            matrices["outcome"].eq(candidate["outcome"])
            & matrices["contrast"].eq(candidate["contrast"])
        ].copy()
        for horizon, group in base.groupby("horizon", sort=True):
            sample = _candidate_matrix_sample(group)
            if sample.empty:
                continue
            sample["abs_interaction"] = sample["tdc_x_holder_contrast"].abs()
            total_abs = float(sample["abs_interaction"].sum())
            ordered = sample.sort_values("abs_interaction", ascending=False).reset_index(drop=True)
            top5 = ordered.head(5)
            top10 = ordered.head(10)
            summary_rows.append(
                {
                    "outcome": candidate["outcome"],
                    "contrast": candidate["contrast"],
                    "horizon": int(horizon),
                    "n": int(len(sample)),
                    "start": str(sample["quarter"].min()),
                    "end": str(sample["quarter"].max()),
                    "total_abs_interaction": total_abs,
                    "top5_abs_interaction_share": _share(top5["abs_interaction"].sum(), total_abs),
                    "top10_abs_interaction_share": _share(top10["abs_interaction"].sum(), total_abs),
                    "major_disruption_rows": int(sample["flag_major_disruption"].eq(1).sum()),
                    "qe_rows": int(sample["flag_qe"].eq(1).sum()),
                    "qt_rows": int(sample["flag_qt"].eq(1).sum()),
                    "post_2021_rows": int(pd.to_datetime(sample["quarter"], errors="coerce").ge(pd.Timestamp("2021-01-01")).sum()),
                    "max_abs_interaction_quarter": str(ordered["quarter"].iloc[0]),
                    "max_abs_interaction": float(ordered["abs_interaction"].iloc[0]),
                }
            )
            for rank, row in enumerate(ordered.head(top_n).to_dict("records"), start=1):
                episode_rows.append(
                    {
                        "outcome": candidate["outcome"],
                        "contrast": candidate["contrast"],
                        "horizon": int(horizon),
                        "rank": rank,
                        "quarter": row["quarter"],
                        "y": float(row["y"]),
                        "holder_contrast": float(row["holder_contrast"]),
                        "tdc_intensity": float(row["tdc_intensity"]),
                        "tdc_x_holder_contrast": float(row["tdc_x_holder_contrast"]),
                        "abs_interaction": float(row["abs_interaction"]),
                        "flag_major_disruption": int(row["flag_major_disruption"]),
                        "flag_qe": int(row["flag_qe"]),
                        "flag_qt": int(row["flag_qt"]),
                    }
                )
    return pd.DataFrame(summary_rows), pd.DataFrame(episode_rows)


def build_exact_candidate_regime_summary(
    triage: pd.DataFrame,
    matrices: pd.DataFrame,
) -> pd.DataFrame:
    candidates = triage.loc[triage["appendix_candidate_status"].eq("appendix_candidate")]
    rows = []
    for candidate in candidates.to_dict("records"):
        base = matrices.loc[
            matrices["outcome"].eq(candidate["outcome"])
            & matrices["contrast"].eq(candidate["contrast"])
        ].copy()
        for horizon, group in base.groupby("horizon", sort=True):
            sample = _candidate_matrix_sample(group)
            if sample.empty:
                continue
            quarter_dates = pd.to_datetime(sample["quarter"], errors="coerce")
            regimes = {
                "full_sample": sample.index == sample.index,
                "exclude_major_disruption": sample["flag_major_disruption"].eq(0),
                "major_disruption": sample["flag_major_disruption"].eq(1),
                "qe": sample["flag_qe"].eq(1),
                "qt": sample["flag_qt"].eq(1),
                "post_2021": quarter_dates.ge(pd.Timestamp("2021-01-01")),
            }
            for regime, mask in regimes.items():
                regime_sample = sample.loc[mask].copy()
                if regime_sample.empty:
                    continue
                rows.append(
                    {
                        "outcome": candidate["outcome"],
                        "contrast": candidate["contrast"],
                        "horizon": int(horizon),
                        "regime": regime,
                        "n": int(len(regime_sample)),
                        "start": str(regime_sample["quarter"].min()),
                        "end": str(regime_sample["quarter"].max()),
                        "mean_y": float(regime_sample["y"].mean()),
                        "mean_holder_contrast": float(regime_sample["holder_contrast"].mean()),
                        "mean_tdc_intensity": float(regime_sample["tdc_intensity"].mean()),
                        "mean_interaction": float(regime_sample["tdc_x_holder_contrast"].mean()),
                        "median_abs_interaction": float(regime_sample["tdc_x_holder_contrast"].abs().median()),
                    }
                )
    return pd.DataFrame(rows)


def build_exact_candidate_concentration_report(
    concentration_summary: pd.DataFrame,
    top_episodes: pd.DataFrame,
    regime_summary: pd.DataFrame,
) -> str:
    lines = [
        "# Exact-FU Candidate Concentration Diagnostics",
        "",
        "These diagnostics describe sample concentration for the timing-clean exact-FU mechanism appendix candidate.",
        "",
    ]
    if concentration_summary.empty:
        lines.append("No concentration summary rows available.")
    else:
        lines.extend(["## Horizon Concentration", ""])
        for row in concentration_summary.to_dict("records"):
            lines.append(
                "- "
                f"`{row['outcome']}` / `{row['contrast']}` h=`{int(row['horizon'])}`: "
                f"top-5 share `{float(row['top5_abs_interaction_share']):.3f}`, "
                f"top-10 share `{float(row['top10_abs_interaction_share']):.3f}`, "
                f"max quarter `{row['max_abs_interaction_quarter']}`"
            )
    if not top_episodes.empty:
        lines.extend(["", "## Top Interaction Quarters", ""])
        for row in top_episodes.loc[top_episodes["rank"].le(3)].to_dict("records"):
            lines.append(
                "- "
                f"h=`{int(row['horizon'])}` rank `{int(row['rank'])}` quarter `{row['quarter']}`: "
                f"interaction `{float(row['tdc_x_holder_contrast']):.6f}`, "
                f"major disruption `{int(row['flag_major_disruption'])}`, QE `{int(row['flag_qe'])}`, QT `{int(row['flag_qt'])}`"
            )
    if not regime_summary.empty:
        lines.extend(["", "## Regime Summary", ""])
        for row in regime_summary.loc[regime_summary["regime"].isin(["full_sample", "exclude_major_disruption", "qt", "post_2021"])].to_dict("records"):
            lines.append(
                "- "
                f"h=`{int(row['horizon'])}` / `{row['regime']}`: "
                f"n `{int(row['n'])}`, mean interaction `{float(row['mean_interaction']):.6f}`"
            )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "These rows are influence and coverage diagnostics. They do not change the non-causal status of the appendix candidate.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_exact_candidate_sensitivity_estimates(
    triage: pd.DataFrame,
    matrices: pd.DataFrame,
    top_episodes: pd.DataFrame,
    *,
    min_rows: int = 30,
) -> pd.DataFrame:
    candidates = triage.loc[triage["appendix_candidate_status"].eq("appendix_candidate")]
    rows = []
    for candidate in candidates.to_dict("records"):
        base = matrices.loc[
            matrices["outcome"].eq(candidate["outcome"])
            & matrices["contrast"].eq(candidate["contrast"])
        ].copy()
        episode_key = (
            top_episodes["outcome"].eq(candidate["outcome"])
            & top_episodes["contrast"].eq(candidate["contrast"])
        ) if not top_episodes.empty else pd.Series(dtype=bool)
        candidate_episodes = top_episodes.loc[episode_key].copy() if not top_episodes.empty else pd.DataFrame()
        for horizon, group in base.groupby("horizon", sort=True):
            horizon_episodes = candidate_episodes.loc[candidate_episodes["horizon"].eq(horizon)]
            top1 = _top_episode_quarters(horizon_episodes, 1)
            top3 = _top_episode_quarters(horizon_episodes, 3)
            top5 = _top_episode_quarters(horizon_episodes, 5)
            quarter_dates = pd.to_datetime(group["quarter"], errors="coerce")
            variants = {
                "candidate_baseline": group,
                "exclude_top1_interaction_quarter": group.loc[~group["quarter"].isin(top1)],
                "exclude_top3_interaction_quarters": group.loc[~group["quarter"].isin(top3)],
                "exclude_top5_interaction_quarters": group.loc[~group["quarter"].isin(top5)],
                "exclude_major_disruption": group.loc[group["flag_major_disruption"].eq(0)],
                "exclude_post_2021": group.loc[quarter_dates.lt(pd.Timestamp("2021-01-01"))],
                "exclude_top3_and_major_disruption": group.loc[
                    ~group["quarter"].isin(top3) & group["flag_major_disruption"].eq(0)
                ],
            }
            for variant, frame in variants.items():
                estimate = _estimate_one(
                    frame,
                    outcome=candidate["outcome"],
                    horizon=int(horizon),
                    contrast=candidate["contrast"],
                    sample=variant,
                    predictors=BASE_PREDICTORS,
                    min_rows=min_rows,
                )
                estimate["sensitivity_variant"] = variant
                estimate["excluded_top_quarters"] = ";".join(
                    {
                        "exclude_top1_interaction_quarter": top1,
                        "exclude_top3_interaction_quarters": top3,
                        "exclude_top5_interaction_quarters": top5,
                        "exclude_top3_and_major_disruption": top3,
                    }.get(variant, [])
                )
                rows.append(estimate)
    if not rows:
        return pd.DataFrame(columns=["sensitivity_variant"])
    out = pd.DataFrame(rows)
    lead = ["outcome", "contrast", "horizon", "sensitivity_variant", "sample"]
    return out[[*lead, *[column for column in out.columns if column not in lead]]].sort_values(
        ["outcome", "contrast", "horizon", "sensitivity_variant"]
    ).reset_index(drop=True)


def build_exact_candidate_sensitivity_summary(sensitivity_estimates: pd.DataFrame) -> pd.DataFrame:
    ok = sensitivity_estimates.loc[sensitivity_estimates["status"].eq("ok")].copy()
    if ok.empty:
        return pd.DataFrame(
            columns=[
                "outcome",
                "contrast",
                "horizon",
                "ok_variants",
                "positive_variants",
                "negative_variants",
                "max_abs_t_stat",
                "min_abs_t_stat",
                "median_beta",
                "sign_robustness_status",
            ]
        )
    beta = pd.to_numeric(ok["beta"], errors="coerce")
    ok["sign"] = beta.apply(lambda value: "positive" if value > 0 else "negative" if value < 0 else "zero")
    ok["abs_t_stat"] = pd.to_numeric(ok["t_stat"], errors="coerce").abs()
    rows = []
    for (outcome, contrast, horizon), group in ok.groupby(["outcome", "contrast", "horizon"], sort=True):
        positive = int(group["sign"].eq("positive").sum())
        negative = int(group["sign"].eq("negative").sum())
        rows.append(
            {
                "outcome": outcome,
                "contrast": contrast,
                "horizon": int(horizon),
                "ok_variants": int(len(group)),
                "positive_variants": positive,
                "negative_variants": negative,
                "max_abs_t_stat": float(group["abs_t_stat"].max()),
                "min_abs_t_stat": float(group["abs_t_stat"].min()),
                "median_beta": float(pd.to_numeric(group["beta"], errors="coerce").median()),
                "sign_robustness_status": (
                    "sign_robust_positive"
                    if positive == len(group)
                    else "sign_robust_negative"
                    if negative == len(group)
                    else "sign_mixed_under_exclusions"
                ),
            }
        )
    return pd.DataFrame(rows)


def build_exact_candidate_sensitivity_report(
    sensitivity_summary: pd.DataFrame,
    sensitivity_estimates: pd.DataFrame,
) -> str:
    mixed = (
        int(sensitivity_summary["sign_robustness_status"].eq("sign_mixed_under_exclusions").sum())
        if "sign_robustness_status" in sensitivity_summary
        else 0
    )
    lines = [
        "# Exact-FU Candidate Sensitivity Checks",
        "",
        f"- Horizon rows checked: `{len(sensitivity_summary)}`",
        f"- Horizons with mixed signs under exclusions: `{mixed}`",
        "- Residual-sensitive headline claims: `blocked`",
        "",
        "## Horizon Summary",
        "",
    ]
    if sensitivity_summary.empty:
        lines.append("No sensitivity rows available.")
    else:
        for row in sensitivity_summary.to_dict("records"):
            lines.append(
                "- "
                f"h=`{int(row['horizon'])}`: `{row['sign_robustness_status']}`, "
                f"positive variants `{int(row['positive_variants'])}`/`{int(row['ok_variants'])}`, "
                f"median beta `{float(row['median_beta']):.3f}`, "
                f"max |t| `{float(row['max_abs_t_stat']):.3f}`"
            )
    lines.extend(["", "## Exclusion Variants", ""])
    ok = sensitivity_estimates.loc[sensitivity_estimates["status"].eq("ok")]
    for row in ok.head(40).to_dict("records"):
        lines.append(
            "- "
            f"h=`{int(row['horizon'])}` / `{row['sensitivity_variant']}`: "
            f"beta `{float(row['beta']):.3f}`, t `{float(row['t_stat']):.3f}`, n `{int(row['n'])}`"
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "These checks are robustness diagnostics for a mechanism appendix candidate. They are not a causal design.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_exact_appendix_source_contract(
    promotion_status: pd.DataFrame,
    appendix_triage: pd.DataFrame,
    mechanism_candidate_table: pd.DataFrame,
    sensitivity_summary: pd.DataFrame,
    concentration_summary: pd.DataFrame,
) -> pd.DataFrame:
    promotion = promotion_status.iloc[0].to_dict() if not promotion_status.empty else {}
    candidate_profiles = (
        appendix_triage.loc[
            appendix_triage["appendix_candidate_status"].eq("appendix_candidate"),
            ["outcome", "contrast"],
        ]
        .drop_duplicates()
        .to_dict("records")
        if not appendix_triage.empty
        else []
    )
    robust_horizons = (
        sensitivity_summary.loc[
            sensitivity_summary["sign_robustness_status"].isin(["sign_robust_positive", "sign_robust_negative"]),
            "horizon",
        ]
        .dropna()
        .astype(int)
        .tolist()
        if not sensitivity_summary.empty and "sign_robustness_status" in sensitivity_summary
        else []
    )
    mixed_horizons = (
        sensitivity_summary.loc[
            sensitivity_summary["sign_robustness_status"].eq("sign_mixed_under_exclusions"),
            "horizon",
        ]
        .dropna()
        .astype(int)
        .tolist()
        if not sensitivity_summary.empty and "sign_robustness_status" in sensitivity_summary
        else []
    )
    max_top5 = (
        float(pd.to_numeric(concentration_summary["top5_abs_interaction_share"], errors="coerce").max())
        if not concentration_summary.empty and "top5_abs_interaction_share" in concentration_summary
        else float("nan")
    )
    rows = []
    if not candidate_profiles:
        rows.append(
            {
                "schema_version": APPENDIX_SOURCE_CONTRACT_SCHEMA_VERSION,
                "contract_name": APPENDIX_SOURCE_CONTRACT_NAME,
                "source_layer": promotion.get("source_layer_to_use_for_appendix", "z1_exact_fu_transactions"),
                "source_layer_role": "mechanism_appendix_accounting_source",
                "appendix_source_promoted": False,
                "main_empirical_allowed": False,
                "residual_sensitive_headline_allowed": False,
                "candidate_outcome": "",
                "candidate_contrast": "",
                "recommended_use": "no_candidate",
                "robust_horizons": "",
                "mixed_horizons": "",
                "max_top5_abs_interaction_share": max_top5,
                "required_caveat": "No exact-FU candidate currently passes appendix triage.",
            }
        )
    for candidate in candidate_profiles:
        rows.append(
            {
                "schema_version": APPENDIX_SOURCE_CONTRACT_SCHEMA_VERSION,
                "contract_name": APPENDIX_SOURCE_CONTRACT_NAME,
                "source_layer": promotion.get("source_layer_to_use_for_appendix", "z1_exact_fu_transactions"),
                "source_layer_role": "mechanism_appendix_accounting_source",
                "appendix_source_promoted": promotion.get("recommendation")
                == "promote_exact_fu_for_mechanism_appendix_source_layer",
                "main_empirical_allowed": False,
                "residual_sensitive_headline_allowed": False,
                "candidate_outcome": candidate["outcome"],
                "candidate_contrast": candidate["contrast"],
                "recommended_use": "mechanism_appendix_diagnostic_only",
                "candidate_horizons": ";".join(
                    str(int(horizon))
                    for horizon in sorted(
                        mechanism_candidate_table.loc[
                            mechanism_candidate_table["outcome"].eq(candidate["outcome"])
                            & mechanism_candidate_table["contrast"].eq(candidate["contrast"]),
                            "horizon",
                        ]
                        .dropna()
                        .astype(int)
                        .unique()
                    )
                ),
                "robust_horizons": ";".join(str(horizon) for horizon in sorted(robust_horizons)),
                "mixed_horizons": ";".join(str(horizon) for horizon in sorted(mixed_horizons)),
                "max_top5_abs_interaction_share": max_top5,
                "required_caveat": (
                    "Exact-FU is promoted only for mechanism appendix accounting. "
                    "Use diagnostic language; do not claim causal identification or residual-sensitive headline evidence."
                ),
            }
        )
    return pd.DataFrame(rows)


def build_exact_appendix_source_contract_report(contract: pd.DataFrame) -> str:
    lines = [
        "# Exact-FU Appendix Source Contract",
        "",
        f"- Contract name: `{APPENDIX_SOURCE_CONTRACT_NAME}`",
        f"- Schema version: `{APPENDIX_SOURCE_CONTRACT_SCHEMA_VERSION}`",
        "",
    ]
    if contract.empty:
        lines.append("No contract rows available.")
    else:
        row = contract.iloc[0].to_dict()
        lines.extend(
            [
                f"- Source layer: `{row['source_layer']}`",
                f"- Appendix source promoted: `{row['appendix_source_promoted']}`",
                f"- Main empirical allowed: `{row['main_empirical_allowed']}`",
                f"- Residual-sensitive headline allowed: `{row['residual_sensitive_headline_allowed']}`",
                "",
                "## Candidate Use",
                "",
            ]
        )
        for item in contract.to_dict("records"):
            if not item.get("candidate_outcome"):
                lines.append("- No current appendix candidate.")
                continue
            lines.append(
                "- "
                f"`{item['candidate_outcome']}` / `{item['candidate_contrast']}`: "
                f"`{item['recommended_use']}`, horizons `{item.get('candidate_horizons', '')}`, "
                f"robust horizons `{item['robust_horizons']}`, mixed horizons `{item['mixed_horizons']}`"
            )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "This is a source-layer contract for appendix accounting diagnostics. It does not authorize main empirical or causal use.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_exact_final_status_packet(
    promotion_status: pd.DataFrame,
    appendix_triage: pd.DataFrame,
    mechanism_candidate_table: pd.DataFrame,
    concentration_summary: pd.DataFrame,
    sensitivity_summary: pd.DataFrame,
) -> pd.DataFrame:
    promotion = promotion_status.iloc[0].to_dict() if not promotion_status.empty else {}
    candidates = int(appendix_triage["appendix_candidate_status"].eq("appendix_candidate").sum()) if not appendix_triage.empty else 0
    held = int(appendix_triage["appendix_candidate_status"].ne("appendix_candidate").sum()) if not appendix_triage.empty else 0
    robust_horizons = (
        int(sensitivity_summary["sign_robustness_status"].isin(["sign_robust_positive", "sign_robust_negative"]).sum())
        if not sensitivity_summary.empty and "sign_robustness_status" in sensitivity_summary
        else 0
    )
    mixed_horizons = (
        int(sensitivity_summary["sign_robustness_status"].eq("sign_mixed_under_exclusions").sum())
        if not sensitivity_summary.empty and "sign_robustness_status" in sensitivity_summary
        else 0
    )
    max_top5 = (
        float(pd.to_numeric(concentration_summary["top5_abs_interaction_share"], errors="coerce").max())
        if not concentration_summary.empty and "top5_abs_interaction_share" in concentration_summary
        else float("nan")
    )
    return pd.DataFrame(
        [
            {
                "status_scope": "tdcmix_exact_fu_backend_status_packet",
                "source_layer_recommendation": promotion.get("recommendation", "unknown"),
                "appendix_source_layer": promotion.get("source_layer_to_use_for_appendix", "unknown"),
                "main_empirical_slot_status": promotion.get("main_empirical_slot_status", "unknown"),
                "residual_sensitive_headline_status": promotion.get("residual_sensitive_headline_status", "blocked"),
                "appendix_candidate_profiles": candidates,
                "held_reviewable_profiles": held,
                "mechanism_horizon_rows": int(len(mechanism_candidate_table)),
                "sign_robust_horizons": robust_horizons,
                "mixed_sensitivity_horizons": mixed_horizons,
                "max_top5_abs_interaction_share": max_top5,
                "next_backend_action": (
                    "Wire the exact-FU appendix source contract into downstream consumers while preserving legacy comparison outputs."
                ),
                "status_note": (
                    "Exact-FU is promoted for mechanism appendix accounting only; all causal, main empirical, and residual-sensitive headline uses remain blocked."
                ),
            }
        ]
    )


def build_exact_final_status_report(status: pd.DataFrame) -> str:
    row = status.iloc[0].to_dict()
    lines = [
        "# Exact-FU Backend Final Status Packet",
        "",
        f"- Source-layer recommendation: `{row['source_layer_recommendation']}`",
        f"- Appendix source layer: `{row['appendix_source_layer']}`",
        f"- Main empirical slot status: `{row['main_empirical_slot_status']}`",
        f"- Residual-sensitive headline status: `{row['residual_sensitive_headline_status']}`",
        f"- Appendix candidate profiles: `{int(row['appendix_candidate_profiles'])}`",
        f"- Held reviewable profiles: `{int(row['held_reviewable_profiles'])}`",
        f"- Mechanism horizon rows: `{int(row['mechanism_horizon_rows'])}`",
        f"- Sign-robust horizons: `{int(row['sign_robust_horizons'])}`",
        f"- Mixed sensitivity horizons: `{int(row['mixed_sensitivity_horizons'])}`",
        f"- Max top-5 interaction share: `{float(row['max_top5_abs_interaction_share']):.3f}`",
        "",
        "## Next Backend Action",
        "",
        str(row["next_backend_action"]),
        "",
        "## Boundary",
        "",
        str(row["status_note"]),
    ]
    return "\n".join(lines) + "\n"



def write_z1_exact_analysis_outputs(
    *,
    exact_holder_panel_path: str | Path = "data/processed/z1_exact_holder_absorption_panel.csv",
    legacy_design_path: str | Path = "data/processed/design_panel.csv",
    legacy_summary_path: str | Path = "data/processed/lp_diagnostic_summary.csv",
    legacy_stability_path: str | Path = "data/processed/lp_shortlist_stability_summary.csv",
    exact_gate_path: str | Path = "data/processed/z1_exact_gate_review.csv",
    output_dir: str | Path = "data/processed/z1_exact_analysis",
) -> list[Path]:
    exact_panel = pd.read_csv(exact_holder_panel_path)
    validation_summary = build_panel_validation_summary(exact_panel)
    share_diag = build_share_sum_diagnostics(exact_panel)
    residual_diag = build_residual_diagnostics(exact_panel)
    residual_overlap = build_exact_residual_overlap_diagnostics(exact_panel)
    residual_gate = build_residual_gate_review(residual_diag, residual_overlap)
    design = build_design_panel(exact_panel)
    readiness = build_design_readiness(exact_panel, validation_summary, residual_overlap)
    first_stage = build_first_stage_summary(design)
    matrices = build_horizon_design_matrices(design)
    placebo_matrices = build_pretrend_placebo_matrices(design)
    matrix_summary = build_horizon_matrix_summary(matrices)
    placebo_matrix_summary = build_horizon_matrix_summary(placebo_matrices)
    stop_rules = build_stop_rule_status(validation_summary, readiness, first_stage)

    estimates = build_lp_diagnostic_estimates(matrices, stop_rules)
    summary = build_lp_diagnostic_summary(estimates)
    shortlist = build_lp_diagnostic_shortlist(summary)
    profile_plot_data = build_lp_profile_plot_data(estimates, shortlist)
    fullspec = build_lp_shortlist_fullspec_estimates(matrices, shortlist, stop_rules)
    stability = build_lp_shortlist_stability_summary(fullspec)
    placebo_estimates = build_pretrend_placebo_estimates(placebo_matrices, shortlist, stop_rules)
    placebo_summary = build_pretrend_placebo_summary(placebo_estimates)
    estimates = add_stop_rule_metadata(estimates, stop_rules)
    summary = add_stop_rule_metadata(summary, stop_rules)
    shortlist = add_stop_rule_metadata(shortlist, stop_rules)
    profile_plot_data = add_stop_rule_metadata(profile_plot_data, stop_rules)
    fullspec = add_stop_rule_metadata(fullspec, stop_rules)
    stability = add_stop_rule_metadata(stability, stop_rules)
    placebo_estimates = add_stop_rule_metadata(placebo_estimates, stop_rules)
    placebo_summary = add_stop_rule_metadata(placebo_summary, stop_rules)
    economics_review = build_lp_shortlist_economics_review(fullspec, stability, stop_rules)

    legacy_design = pd.read_csv(legacy_design_path)
    legacy_summary = pd.read_csv(legacy_summary_path)
    legacy_stability = pd.read_csv(legacy_stability_path)
    contrast_comparison = build_source_layer_contrast_comparison(legacy_design, design)
    estimate_comparison = build_source_layer_estimate_comparison(legacy_summary, summary)
    stability_comparison = build_source_layer_stability_comparison(legacy_stability, stability)
    exact_gate = pd.read_csv(exact_gate_path) if Path(exact_gate_path).exists() else pd.DataFrame()
    promotion_status = build_source_layer_promotion_status(
        contrast_comparison,
        estimate_comparison,
        placebo_summary,
        exact_gate,
    )
    appendix_triage = build_exact_appendix_candidate_triage(
        economics_review,
        placebo_summary,
        stability_comparison,
    )
    mechanism_candidate_table = build_exact_mechanism_appendix_candidate_table(
        appendix_triage,
        fullspec,
        placebo_estimates,
    )
    concentration_summary, top_episodes = build_exact_candidate_sample_concentration(
        appendix_triage,
        matrices,
    )
    regime_summary = build_exact_candidate_regime_summary(appendix_triage, matrices)
    sensitivity_estimates = build_exact_candidate_sensitivity_estimates(
        appendix_triage,
        matrices,
        top_episodes,
    )
    sensitivity_summary = build_exact_candidate_sensitivity_summary(sensitivity_estimates)
    appendix_source_contract = build_exact_appendix_source_contract(
        promotion_status,
        appendix_triage,
        mechanism_candidate_table,
        sensitivity_summary,
        concentration_summary,
    )
    final_status = build_exact_final_status_packet(
        promotion_status,
        appendix_triage,
        mechanism_candidate_table,
        concentration_summary,
        sensitivity_summary,
    )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "validation_summary.csv": validation_summary,
        "share_sum_diagnostics.csv": share_diag,
        "residual_diagnostics.csv": residual_diag,
        "exact_residual_overlap_diagnostics.csv": residual_overlap,
        "residual_gate_review.csv": residual_gate,
        "design_panel.csv": design,
        "design_readiness.csv": readiness,
        "first_stage_summary.csv": first_stage,
        "horizon_design_matrices.csv": matrices,
        "horizon_design_matrix_summary.csv": matrix_summary,
        "pretrend_placebo_matrices.csv": placebo_matrices,
        "pretrend_placebo_matrix_summary.csv": placebo_matrix_summary,
        "stop_rule_status.csv": stop_rules,
        "lp_diagnostic_estimates.csv": estimates,
        "lp_diagnostic_summary.csv": summary,
        "lp_diagnostic_shortlist.csv": shortlist,
        "lp_diagnostic_profile_plot_data.csv": profile_plot_data,
        "lp_shortlist_fullspec_estimates.csv": fullspec,
        "lp_shortlist_stability_summary.csv": stability,
        "lp_shortlist_economics_review.csv": economics_review,
        "pretrend_placebo_estimates.csv": placebo_estimates,
        "pretrend_placebo_summary.csv": placebo_summary,
        "source_layer_contrast_comparison.csv": contrast_comparison,
        "source_layer_estimate_comparison.csv": estimate_comparison,
        "source_layer_stability_comparison.csv": stability_comparison,
        "source_layer_promotion_status.csv": promotion_status,
        "exact_appendix_candidate_triage.csv": appendix_triage,
        "exact_mechanism_appendix_candidate_table.csv": mechanism_candidate_table,
        "exact_candidate_sample_concentration.csv": concentration_summary,
        "exact_candidate_top_interaction_quarters.csv": top_episodes,
        "exact_candidate_regime_summary.csv": regime_summary,
        "exact_candidate_sensitivity_estimates.csv": sensitivity_estimates,
        "exact_candidate_sensitivity_summary.csv": sensitivity_summary,
        "exact_appendix_source_contract.csv": appendix_source_contract,
        "exact_final_status_packet.csv": final_status,
    }
    paths = []
    for name, frame in outputs.items():
        path = out_dir / name
        frame.to_csv(path, index=False)
        paths.append(path)

    text_outputs = {
        "residual_gate_report.md": build_residual_gate_report(residual_gate),
        "lp_diagnostic_report.md": build_lp_diagnostic_report(estimates, summary, stop_rules),
        "lp_shortlist_fullspec_report.md": build_lp_shortlist_fullspec_report(fullspec, stability, shortlist),
        "lp_shortlist_economics_memo.md": build_lp_shortlist_economics_memo(economics_review),
        "pretrend_placebo_report.md": build_pretrend_placebo_report(
            placebo_summary,
            source_layer_label="z1_exact_fu_transactions",
        ),
        "source_layer_comparison_report.md": build_source_layer_comparison_report(
            contrast_comparison,
            estimate_comparison,
            stability_comparison,
            exact_gate,
        ),
        "source_layer_promotion_report.md": build_source_layer_promotion_report(promotion_status),
        "exact_appendix_candidate_report.md": build_exact_appendix_candidate_report(appendix_triage),
        "exact_mechanism_appendix_memo.md": build_exact_mechanism_appendix_memo(
            appendix_triage,
            mechanism_candidate_table,
            promotion_status,
            concentration_summary,
            regime_summary,
            sensitivity_summary,
        ),
        "exact_candidate_concentration_report.md": build_exact_candidate_concentration_report(
            concentration_summary,
            top_episodes,
            regime_summary,
        ),
        "exact_candidate_sensitivity_report.md": build_exact_candidate_sensitivity_report(
            sensitivity_summary,
            sensitivity_estimates,
        ),
        "exact_appendix_source_contract.md": build_exact_appendix_source_contract_report(
            appendix_source_contract,
        ),
        "exact_final_status_packet.md": build_exact_final_status_report(final_status),
    }
    for name, text in text_outputs.items():
        path = out_dir / name
        path.write_text(text, encoding="utf-8")
        paths.append(path)
    return paths


def _estimate_comparison_status(row: pd.Series) -> str:
    legacy_pos = _as_int(row.get("baseline_positive_legacy"))
    legacy_neg = _as_int(row.get("baseline_negative_legacy"))
    exact_pos = _as_int(row.get("baseline_positive_exact_z1"))
    exact_neg = _as_int(row.get("baseline_negative_exact_z1"))
    if legacy_pos == exact_pos and legacy_neg == exact_neg:
        return "same_sign_mix"
    if legacy_pos + legacy_neg == 0 or exact_pos + exact_neg == 0:
        return "coverage_changed"
    if (legacy_pos > legacy_neg) != (exact_pos > exact_neg):
        return "dominant_sign_changed"
    return "sign_mix_changed"


def _appendix_candidate_status(row: pd.Series) -> str:
    if str(row.get("pretrend_screen_status")) == "timing_concern":
        return "hold_timing_concern"
    if bool(row.get("source_layer_stability_status_changed", False)) and str(row.get("stability_status_exact_z1")) != "directionally_stable_diagnostic":
        return "hold_source_layer_instability"
    return "appendix_candidate"


def _appendix_triage_note(row: pd.Series) -> str:
    status = str(row.get("appendix_candidate_status"))
    if status == "appendix_candidate":
        return "Passes the current exact-FU timing screen and remains reviewable under exact-source diagnostics."
    if status == "hold_timing_concern":
        cells = _as_int(row.get("concerning_placebo_cells"))
        max_t = pd.to_numeric(pd.Series([row.get("max_abs_placebo_t_stat")]), errors="coerce").iloc[0]
        return f"Hold because {cells} pretrend/placebo cells exceed the timing-screen threshold; max placebo |t| is {float(max_t):.3f}."
    return "Hold because the profile's source-layer stability classification changed and needs manual review."


def _candidate_matrix_sample(group: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "quarter",
        "y",
        "holder_contrast",
        "tdc_intensity",
        "tdc_x_holder_contrast",
        "flag_major_disruption",
        "flag_qe",
        "flag_qt",
    ]
    available = [column for column in columns if column in group.columns]
    sample = group[available].copy()
    numeric = [
        "y",
        "holder_contrast",
        "tdc_intensity",
        "tdc_x_holder_contrast",
        "flag_major_disruption",
        "flag_qe",
        "flag_qt",
    ]
    for column in numeric:
        if column in sample:
            sample[column] = pd.to_numeric(sample[column], errors="coerce")
    return sample.dropna(subset=["y", "holder_contrast", "tdc_intensity", "tdc_x_holder_contrast"])


def _share(part: float, total: float) -> float:
    if total == 0:
        return float("nan")
    return float(part / total)


def _top_episode_quarters(episodes: pd.DataFrame, n: int) -> list[str]:
    if episodes.empty:
        return []
    return episodes.sort_values("rank").head(n)["quarter"].astype(str).tolist()


def _as_int(value: object) -> int:
    try:
        if pd.isna(value):
            return 0
        return int(value)
    except (TypeError, ValueError):
        return 0


def _safe_corr(left: pd.Series, right: pd.Series) -> float:
    if len(left) < 3:
        return float("nan")
    if left.nunique(dropna=True) <= 1 or right.nunique(dropna=True) <= 1:
        return float("nan")
    corr = left.corr(right)
    return float(corr) if pd.notna(corr) else float("nan")
