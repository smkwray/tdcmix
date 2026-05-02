import pandas as pd

from tdcmix.z1_exact_analysis import (
    build_exact_appendix_candidate_report,
    build_exact_appendix_candidate_triage,
    build_exact_mechanism_appendix_candidate_table,
    build_exact_mechanism_appendix_memo,
    build_exact_candidate_concentration_report,
    build_exact_candidate_regime_summary,
    build_exact_candidate_sample_concentration,
    build_exact_candidate_sensitivity_estimates,
    build_exact_candidate_sensitivity_report,
    build_exact_candidate_sensitivity_summary,
    build_exact_appendix_source_contract,
    build_exact_appendix_source_contract_report,
    build_exact_final_status_packet,
    build_exact_final_status_report,
    build_source_layer_comparison_report,
    build_source_layer_contrast_comparison,
    build_source_layer_estimate_comparison,
    build_source_layer_promotion_report,
    build_source_layer_promotion_status,
    build_source_layer_stability_comparison,
)


def test_source_layer_contrast_comparison_summarizes_differences() -> None:
    legacy = _design_frame(0.2, 0.1, "legacy")
    exact = _design_frame(0.3, 0.0, "z1_exact_fu_transactions")

    comparison = build_source_layer_contrast_comparison(legacy, exact)

    by_metric = comparison.set_index("metric")
    assert by_metric.loc["contrast_mmf_minus_banks", "overlap_rows"] == 3
    assert by_metric.loc["contrast_mmf_minus_banks", "median_abs_exact_minus_legacy"] > 0
    assert by_metric.loc["share_ru_like", "legacy_exact_correlation"] == 1.0


def test_source_layer_estimate_and_stability_comparison_flags_changes() -> None:
    legacy_summary = pd.DataFrame(
        {
            "outcome": ["deposits"],
            "contrast": ["contrast_mmf_minus_banks"],
            "ok_cells": [5],
            "baseline_cells": [5],
            "baseline_positive": [4],
            "baseline_negative": [1],
            "sample_sign_stable_cells": [3],
            "max_abs_t_stat": [2.0],
            "min_n": [50],
            "max_n": [55],
        }
    )
    exact_summary = legacy_summary.assign(baseline_positive=[1], baseline_negative=[4], max_abs_t_stat=[3.0])

    estimate_comparison = build_source_layer_estimate_comparison(legacy_summary, exact_summary)

    assert estimate_comparison["comparison_status"].iloc[0] == "dominant_sign_changed"
    assert estimate_comparison["max_abs_t_stat_exact_minus_legacy"].iloc[0] == 1.0

    legacy_stability = pd.DataFrame(
        {
            "outcome": ["deposits"],
            "contrast": ["contrast_mmf_minus_banks"],
            "horizons_fitted": [5],
            "variant_cells": [25],
            "stable_horizon_signs": [3],
            "unstable_horizon_signs": [1],
            "positive_cells": [20],
            "negative_cells": [5],
            "max_abs_t_stat": [2.5],
            "min_n": [40],
            "max_n": [50],
            "stability_status": ["mixed_but_reviewable_diagnostic"],
        }
    )
    exact_stability = legacy_stability.assign(stability_status=["weak_or_unstable_diagnostic"])

    stability_comparison = build_source_layer_stability_comparison(legacy_stability, exact_stability)

    assert bool(stability_comparison["stability_status_changed"].iloc[0])


def test_source_layer_comparison_report_marks_boundary() -> None:
    contrast = build_source_layer_contrast_comparison(
        _design_frame(0.2, 0.1, "legacy"),
        _design_frame(0.3, 0.0, "z1_exact_fu_transactions"),
    )
    estimate = pd.DataFrame(
        {
            "outcome": ["deposits"],
            "contrast": ["contrast_mmf_minus_banks"],
            "comparison_status": ["same_sign_mix"],
            "baseline_positive_legacy": [2],
            "baseline_negative_legacy": [1],
            "baseline_positive_exact_z1": [2],
            "baseline_negative_exact_z1": [1],
        }
    )
    stability = pd.DataFrame({"stability_status_changed": [False]})
    gate = pd.DataFrame({"status": ["ready"], "exact_rows": [302]})

    report = build_source_layer_comparison_report(contrast, estimate, stability, gate)

    assert "Exact accounting gate: `ready`" in report
    assert "diagnostic comparison only" in report


def test_source_layer_promotion_status_promotes_appendix_only() -> None:
    contrast = build_source_layer_contrast_comparison(
        _design_frame(0.2, 0.1, "legacy"),
        _design_frame(0.3, 0.0, "z1_exact_fu_transactions"),
    )
    estimate = pd.DataFrame(
        {
            "comparison_status": ["same_sign_mix", "dominant_sign_changed"],
        }
    )
    placebo = pd.DataFrame({"pretrend_screen_status": ["timing_concern", "passes_timing_screen"]})
    gate = pd.DataFrame({"status": ["ready"], "exact_rows": [302]})

    status = build_source_layer_promotion_status(contrast, estimate, placebo, gate)
    report = build_source_layer_promotion_report(status)

    assert status["recommendation"].iloc[0] == "promote_exact_fu_for_mechanism_appendix_source_layer"
    assert status["main_empirical_slot_status"].iloc[0] == "blocked_by_timing_and_diagnostic_instability"
    assert "does not promote the project to a main empirical slot" in report


def test_exact_appendix_candidate_triage_holds_timing_concerns() -> None:
    review = pd.DataFrame(
        {
            "outcome": ["deposits", "reserves"],
            "contrast": ["contrast_mmf_minus_banks", "contrast_private_ru_minus_mmf"],
            "review_priority": ["1_review_first", "2_review_after_source_checks"],
            "stability_status": ["directionally_stable_diagnostic", "mixed_but_reviewable_diagnostic"],
            "dominant_coefficient_sign": ["positive", "negative"],
            "max_abs_t_stat": [3.0, 2.5],
            "coefficient_reading": ["reading", "reading"],
            "manual_review_note": ["note", "note"],
        }
    )
    placebo = pd.DataFrame(
        {
            "outcome": ["deposits", "reserves"],
            "contrast": ["contrast_mmf_minus_banks", "contrast_private_ru_minus_mmf"],
            "pretrend_screen_status": ["passes_timing_screen", "timing_concern"],
            "concerning_placebo_cells": [0, 2],
            "max_abs_placebo_t_stat": [1.2, 2.8],
        }
    )
    stability = pd.DataFrame(
        {
            "outcome": ["deposits", "reserves"],
            "contrast": ["contrast_mmf_minus_banks", "contrast_private_ru_minus_mmf"],
            "stability_status_changed": [False, False],
            "stability_status_legacy": ["mixed_but_reviewable_diagnostic", "mixed_but_reviewable_diagnostic"],
            "stability_status_exact_z1": ["directionally_stable_diagnostic", "mixed_but_reviewable_diagnostic"],
        }
    )

    triage = build_exact_appendix_candidate_triage(review, placebo, stability)
    report = build_exact_appendix_candidate_report(triage)

    assert set(triage["appendix_candidate_status"]) == {"appendix_candidate", "hold_timing_concern"}
    assert triage["appendix_candidate_status"].eq("appendix_candidate").sum() == 1
    assert "Mechanism appendix candidates: `1`" in report


def test_exact_mechanism_appendix_memo_summarizes_candidate() -> None:
    triage = pd.DataFrame(
        {
            "outcome": ["deposits"],
            "contrast": ["contrast_mmf_minus_banks"],
            "appendix_candidate_status": ["appendix_candidate"],
            "dominant_coefficient_sign": ["positive"],
            "coefficient_reading": ["diagnostic reading"],
            "triage_note": ["passes timing"],
        }
    )
    fullspec = pd.DataFrame(
        {
            "outcome": ["deposits", "deposits"],
            "contrast": ["contrast_mmf_minus_banks", "contrast_mmf_minus_banks"],
            "horizon": [0, 0],
            "fullspec_variant": ["shortlist_baseline", "shortlist_regime_flags"],
            "beta": [1.0, 2.0],
            "t_stat": [2.1, 3.0],
            "status": ["ok", "ok"],
        }
    )
    placebo = pd.DataFrame(
        {
            "outcome": ["deposits"],
            "contrast": ["contrast_mmf_minus_banks"],
            "t_stat": [1.0],
            "status": ["ok"],
        }
    )
    promotion = pd.DataFrame(
        {
            "source_layer_to_use_for_appendix": ["z1_exact_fu_transactions"],
            "recommendation": ["promote_exact_fu_for_mechanism_appendix_source_layer"],
            "main_empirical_slot_status": ["blocked"],
            "residual_sensitive_headline_status": ["blocked"],
        }
    )

    table = build_exact_mechanism_appendix_candidate_table(triage, fullspec, placebo)
    memo = build_exact_mechanism_appendix_memo(triage, table, promotion)

    assert table["positive_variant_cells"].iloc[0] == 2
    assert "Exact-FU Mechanism Appendix Memo" in memo
    assert "mechanism diagnostic" in memo


def test_exact_candidate_concentration_and_regime_summaries() -> None:
    triage = pd.DataFrame(
        {
            "outcome": ["deposits"],
            "contrast": ["contrast_mmf_minus_banks"],
            "appendix_candidate_status": ["appendix_candidate"],
        }
    )
    matrices = pd.DataFrame(
        {
            "outcome": ["deposits"] * 6,
            "contrast": ["contrast_mmf_minus_banks"] * 6,
            "horizon": [0, 0, 0, 1, 1, 1],
            "quarter": ["2020-03-31", "2020-06-30", "2021-03-31"] * 2,
            "y": [1.0, 2.0, 3.0, 1.5, 2.5, 3.5],
            "holder_contrast": [0.1, 0.2, 0.3, 0.1, 0.2, 0.3],
            "tdc_intensity": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
            "tdc_x_holder_contrast": [0.1, 0.4, 0.9, 0.1, 0.4, 0.9],
            "flag_major_disruption": [1, 1, 0, 1, 1, 0],
            "flag_qe": [1, 1, 0, 1, 1, 0],
            "flag_qt": [0, 0, 1, 0, 0, 1],
        }
    )

    concentration, episodes = build_exact_candidate_sample_concentration(triage, matrices, top_n=2)
    regime = build_exact_candidate_regime_summary(triage, matrices)
    report = build_exact_candidate_concentration_report(concentration, episodes, regime)

    assert set(concentration["horizon"]) == {0, 1}
    assert len(episodes) == 4
    assert {"full_sample", "post_2021", "qt"}.issubset(set(regime["regime"]))
    assert "Exact-FU Candidate Concentration Diagnostics" in report


def test_exact_candidate_sensitivity_estimates() -> None:
    triage = pd.DataFrame(
        {
            "outcome": ["deposits"],
            "contrast": ["contrast_mmf_minus_banks"],
            "appendix_candidate_status": ["appendix_candidate"],
        }
    )
    rows = []
    for idx in range(48):
        rows.append(
            {
                "outcome": "deposits",
                "contrast": "contrast_mmf_minus_banks",
                "horizon": 0,
                "quarter": f"20{idx // 4 + 10:02d}-{((idx % 4) + 1) * 3:02d}-30",
                "y": float(idx + 1),
                "holder_contrast": 0.1 + idx * 0.001,
                "tdc_intensity": 1.0 + idx * 0.01,
                "tdc_x_holder_contrast": (0.1 + idx * 0.001) * (1.0 + idx * 0.01),
                "bill_share": 0.2,
                "wam_gap_trailing": 0.3,
                "gross_marketable_issuance": 100.0 + idx,
                "flag_major_disruption": 1 if idx < 5 else 0,
                "flag_qe": 1 if idx < 8 else 0,
                "flag_qt": 1 if idx > 35 else 0,
            }
        )
    matrices = pd.DataFrame(rows)
    top_episodes = pd.DataFrame(
        {
            "outcome": ["deposits"] * 5,
            "contrast": ["contrast_mmf_minus_banks"] * 5,
            "horizon": [0] * 5,
            "rank": [1, 2, 3, 4, 5],
            "quarter": matrices["quarter"].tail(5).tolist()[::-1],
        }
    )

    estimates = build_exact_candidate_sensitivity_estimates(triage, matrices, top_episodes, min_rows=20)
    summary = build_exact_candidate_sensitivity_summary(estimates)
    report = build_exact_candidate_sensitivity_report(summary, estimates)

    assert {"exclude_top1_interaction_quarter", "exclude_major_disruption"}.issubset(
        set(estimates["sensitivity_variant"])
    )
    assert not summary.empty
    assert "Exact-FU Candidate Sensitivity Checks" in report


def test_exact_appendix_contract_and_final_status_packet() -> None:
    promotion = pd.DataFrame(
        {
            "recommendation": ["promote_exact_fu_for_mechanism_appendix_source_layer"],
            "source_layer_to_use_for_appendix": ["z1_exact_fu_transactions"],
            "main_empirical_slot_status": ["blocked_by_timing_and_diagnostic_instability"],
            "residual_sensitive_headline_status": ["blocked"],
        }
    )
    triage = pd.DataFrame(
        {
            "outcome": ["deposits", "reserves"],
            "contrast": ["contrast_mmf_minus_banks", "contrast_private_ru_minus_mmf"],
            "appendix_candidate_status": ["appendix_candidate", "hold_timing_concern"],
        }
    )
    mechanism = pd.DataFrame(
        {
            "outcome": ["deposits", "deposits"],
            "contrast": ["contrast_mmf_minus_banks", "contrast_mmf_minus_banks"],
            "horizon": [0, 1],
        }
    )
    sensitivity = pd.DataFrame(
        {
            "horizon": [0, 1],
            "sign_robustness_status": ["sign_robust_positive", "sign_mixed_under_exclusions"],
        }
    )
    concentration = pd.DataFrame({"top5_abs_interaction_share": [0.25, 0.3]})

    contract = build_exact_appendix_source_contract(
        promotion,
        triage,
        mechanism,
        sensitivity,
        concentration,
    )
    contract_report = build_exact_appendix_source_contract_report(contract)
    status = build_exact_final_status_packet(
        promotion,
        triage,
        mechanism,
        concentration,
        sensitivity,
    )
    status_report = build_exact_final_status_report(status)

    assert contract["appendix_source_promoted"].iloc[0]
    assert not contract["main_empirical_allowed"].iloc[0]
    assert contract["robust_horizons"].iloc[0] == "0"
    assert status["appendix_candidate_profiles"].iloc[0] == 1
    assert status["mixed_sensitivity_horizons"].iloc[0] == 1
    assert "Exact-FU Appendix Source Contract" in contract_report
    assert "Exact-FU Backend Final Status Packet" in status_report


def _design_frame(mmf_share: float, bank_share: float, source_layer: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "quarter": ["2020-03-31", "2020-06-30", "2020-09-30"],
            "holder_source_layer": [source_layer] * 3,
            "share_ru_like": [0.5, 0.6, 0.7],
            "share_private_ru_like": [0.3, 0.4, 0.5],
            "share_mmf_plumbing": [mmf_share, mmf_share + 0.1, mmf_share + 0.2],
            "share_domestic_final": [0.1, 0.1, 0.1],
            "share_bridge_financial": [0.1, 0.1, 0.1],
            "share_residual": [0.0, 0.0, 0.0],
            "contrast_ru_minus_domestic": [0.4, 0.5, 0.6],
            "contrast_mmf_minus_banks": [
                mmf_share - bank_share,
                mmf_share + 0.1 - bank_share,
                mmf_share + 0.2 - bank_share,
            ],
            "contrast_private_ru_minus_mmf": [0.1, 0.1, 0.1],
        }
    )
