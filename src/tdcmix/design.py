from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


CORE_OUTCOMES = ("deposits", "bank_credit", "reserves", "onrrp", "mmf_treasury")
HORIZONS = (0, 1, 2, 3, 4)
PRETREND_HORIZONS = (-4, -3, -2, -1)
NON_RESIDUAL_CONTRASTS = (
    "contrast_ru_minus_domestic",
    "contrast_mmf_minus_banks",
    "contrast_private_ru_minus_mmf",
)
ISSUANCE_TREATMENTS = ("bill_share", "wam_gap_trailing")
REGIME_WINDOWS = {
    "flag_gfc": ("2007-07-01", "2009-06-30"),
    "flag_covid": ("2020-01-01", "2021-12-31"),
    "flag_2023_debt_ceiling_bank_stress": ("2023-01-01", "2023-09-30"),
    "flag_qe1": ("2008-10-01", "2010-03-31"),
    "flag_qe2": ("2010-10-01", "2011-06-30"),
    "flag_qe3": ("2012-09-01", "2014-10-31"),
    "flag_qt1": ("2017-10-01", "2019-09-30"),
    "flag_covid_qe": ("2020-03-01", "2022-03-31"),
    "flag_qt2": ("2022-06-01", "2026-12-31"),
}


def build_design_panel(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    out["quarter_date"] = pd.to_datetime(out["quarter"], errors="coerce", format="mixed")
    out = out.sort_values("quarter_date").reset_index(drop=True)

    out["share_ru_like"] = _sum_columns(
        out,
        ["pos_abs_share_fed", "pos_abs_share_banks", "pos_abs_share_row"],
    )
    out["share_private_ru_like"] = _sum_columns(
        out,
        ["pos_abs_share_banks", "pos_abs_share_row"],
    )
    out["share_mmf_plumbing"] = pd.to_numeric(out["pos_abs_share_mmf"], errors="coerce")
    out["share_domestic_final"] = pd.to_numeric(
        out["pos_abs_share_domestic_nonbank"],
        errors="coerce",
    )
    out["share_bridge_financial"] = _sum_columns(
        out,
        ["pos_abs_share_dealer_bridge", "pos_abs_share_other_financial"],
    )
    out["share_named_headline"] = _sum_columns(
        out,
        [
            "pos_abs_share_fed",
            "pos_abs_share_banks",
            "pos_abs_share_row",
            "pos_abs_share_mmf",
            "pos_abs_share_domestic_nonbank",
            "pos_abs_share_dealer_bridge",
        ],
    )
    out["share_residual"] = pd.to_numeric(out["pos_abs_share_unmapped"], errors="coerce")
    out["contrast_ru_minus_domestic"] = out["share_ru_like"] - out["share_domestic_final"]
    out["contrast_mmf_minus_banks"] = out["share_mmf_plumbing"] - pd.to_numeric(
        out["pos_abs_share_banks"],
        errors="coerce",
    )
    out["contrast_private_ru_minus_mmf"] = out["share_private_ru_like"] - out["share_mmf_plumbing"]
    out["tdc_intensity"] = pd.to_numeric(out["tdc_tier1"], errors="coerce").div(
        pd.to_numeric(out["gross_marketable_issuance"], errors="coerce").where(
            pd.to_numeric(out["gross_marketable_issuance"], errors="coerce") != 0
        )
    )
    out = add_regime_flags(out)

    for outcome in CORE_OUTCOMES:
        values = pd.to_numeric(out[outcome], errors="coerce")
        for horizon in HORIZONS:
            out[f"{outcome}_h{horizon}"] = values.shift(-horizon)
        for horizon in PRETREND_HORIZONS:
            out[f"{outcome}_h{horizon}"] = values.shift(abs(horizon))

    keep = [
        "quarter",
        "quarter_date",
        *[column for column in ("holder_source_layer", "residual_interpretation") if column in out.columns],
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
        "flag_gfc",
        "flag_covid",
        "flag_2023_debt_ceiling_bank_stress",
        "flag_qe",
        "flag_qt",
        "flag_major_disruption",
        *CORE_OUTCOMES,
        *[f"{outcome}_h{horizon}" for outcome in CORE_OUTCOMES for horizon in [*PRETREND_HORIZONS, *HORIZONS]],
    ]
    return out[keep]


def add_regime_flags(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    dates = pd.to_datetime(out["quarter_date"], errors="coerce", format="mixed")
    for flag, (start, end) in REGIME_WINDOWS.items():
        out[flag] = dates.between(pd.Timestamp(start), pd.Timestamp(end)).astype(int)
    out["flag_qe"] = out[["flag_qe1", "flag_qe2", "flag_qe3", "flag_covid_qe"]].max(axis=1)
    out["flag_qt"] = out[["flag_qt1", "flag_qt2"]].max(axis=1)
    out["flag_major_disruption"] = out[
        ["flag_gfc", "flag_covid", "flag_2023_debt_ceiling_bank_stress"]
    ].max(axis=1)
    return out


def build_first_stage_summary(design: pd.DataFrame) -> pd.DataFrame:
    rows = []
    outcomes = (
        "share_ru_like",
        "share_private_ru_like",
        "share_mmf_plumbing",
        "share_domestic_final",
        "contrast_ru_minus_domestic",
        "contrast_mmf_minus_banks",
        "contrast_private_ru_minus_mmf",
    )
    for outcome in outcomes:
        for treatment in ISSUANCE_TREATMENTS:
            rows.append(
                _ols_summary(
                    design,
                    outcome=outcome,
                    treatment=treatment,
                    controls=["gross_marketable_issuance"],
                    sample_label="baseline",
                )
            )
            rows.append(
                _ols_summary(
                    design.loc[design["flag_major_disruption"].eq(0)],
                    outcome=outcome,
                    treatment=treatment,
                    controls=["gross_marketable_issuance"],
                    sample_label="exclude_major_disruption",
                )
            )
    return pd.DataFrame(rows)


def build_horizon_design_matrices(design: pd.DataFrame) -> pd.DataFrame:
    return _build_horizon_matrices(design, HORIZONS, matrix_type="main_h0_h4")


def build_pretrend_placebo_matrices(design: pd.DataFrame) -> pd.DataFrame:
    return _build_horizon_matrices(design, PRETREND_HORIZONS, matrix_type="pretrend_placebo")


def _build_horizon_matrices(
    design: pd.DataFrame,
    horizons: tuple[int, ...],
    *,
    matrix_type: str,
) -> pd.DataFrame:
    rows = []
    metadata_columns = [column for column in ("holder_source_layer",) if column in design.columns]
    controls = [
        "tdc_intensity",
        "bill_share",
        "wam_gap_trailing",
        "gross_marketable_issuance",
    ]
    flags = [
        "flag_gfc",
        "flag_covid",
        "flag_2023_debt_ceiling_bank_stress",
        "flag_qe",
        "flag_qt",
        "flag_major_disruption",
    ]
    for outcome in CORE_OUTCOMES:
        for horizon in horizons:
            y_col = f"{outcome}_h{horizon}"
            for contrast in NON_RESIDUAL_CONTRASTS:
                matrix = design[
                    [
                        "quarter",
                        *metadata_columns,
                        y_col,
                        contrast,
                        *controls,
                        *flags,
                    ]
                ].copy()
                matrix = matrix.rename(
                    columns={
                        y_col: "y",
                        contrast: "holder_contrast",
                    }
                )
                matrix["outcome"] = outcome
                matrix["horizon"] = horizon
                matrix["matrix_type"] = matrix_type
                matrix["contrast"] = contrast
                matrix["tdc_x_holder_contrast"] = pd.to_numeric(
                    matrix["tdc_intensity"],
                    errors="coerce",
                ) * pd.to_numeric(matrix["holder_contrast"], errors="coerce")
                rows.append(matrix)
    out = pd.concat(rows, ignore_index=True)
    lead = [
        "matrix_type",
        "outcome",
        "horizon",
        "contrast",
        "quarter",
        *metadata_columns,
        "y",
        "holder_contrast",
    ]
    rest = [column for column in out.columns if column not in lead]
    return out[lead + rest]


def build_horizon_matrix_summary(matrices: pd.DataFrame) -> pd.DataFrame:
    required = ["y", "holder_contrast", "tdc_intensity", "tdc_x_holder_contrast"]
    sample = matrices.dropna(subset=required).copy()
    if sample.empty:
        return pd.DataFrame(
            columns=[
                "outcome",
                "horizon",
                "contrast",
                "usable_rows",
                "usable_rows_excluding_major_disruption",
                "start",
                "end",
            ]
        )
    grouped = sample.groupby(["outcome", "horizon", "contrast"], as_index=False)
    out = grouped.agg(
        usable_rows=("quarter", "size"),
        usable_rows_excluding_major_disruption=("flag_major_disruption", lambda s: int((s == 0).sum())),
        start=("quarter", "min"),
        end=("quarter", "max"),
    )
    return out.sort_values(["outcome", "contrast", "horizon"]).reset_index(drop=True)


def build_stop_rule_status(
    validation_summary: pd.DataFrame,
    design_readiness: pd.DataFrame,
    first_stage_summary: pd.DataFrame,
) -> pd.DataFrame:
    validation = dict(zip(validation_summary["check"], validation_summary["value"], strict=False))
    readiness = dict(zip(design_readiness["design"], design_readiness["ready"], strict=False))
    first_ok = first_stage_summary.loc[first_stage_summary["status"].eq("ok")].copy()
    stable_first_stage = False
    if not first_ok.empty:
        stable_first_stage = bool(
            first_ok.groupby("outcome")["t_stat"]
            .apply(lambda values: values.abs().max(skipna=True))
            .ge(1.5)
            .any()
        )
    rows = [
        _stop_rule_row(
            "holder_panel_reconciles_enough_for_non_residual_work",
            str(validation.get("share_sums_pass")).lower() == "true",
            "Share sums pass where defined; residual-sensitive work remains separately gated.",
        ),
        _stop_rule_row(
            "residual_not_allowed_as_headline",
            not bool(readiness.get("residual_sensitive_contrasts", False)),
            "Exact residual overlap error is too large; residual contrasts must stay diagnostic.",
        ),
        _stop_rule_row(
            "coarse_non_residual_contrast_available",
            bool(readiness.get("baseline_non_residual_contrasts", False)),
            "Non-residual holder-mix contrasts have enough rows for first empirical pass.",
        ),
        _stop_rule_row(
            "issuance_composition_moves_holder_mix_somewhere",
            stable_first_stage,
            "At least one first-stage t-stat exceeds the loose diagnostic threshold of 1.5.",
        ),
        _stop_rule_row(
            "h0_h4_design_available",
            bool(readiness.get("h0_h4_outcome_scaffold", False)),
            "Outcome design matrices are available for h=0..4 before final controls and exclusions.",
        ),
    ]
    out = pd.DataFrame(rows)
    out["overall_non_residual_path_status"] = (
        "ready_for_diagnostic_estimation"
        if out.loc[
            out["stop_rule"].isin(
                [
                    "holder_panel_reconciles_enough_for_non_residual_work",
                    "coarse_non_residual_contrast_available",
                    "issuance_composition_moves_holder_mix_somewhere",
                    "h0_h4_design_available",
                ]
            ),
            "passes",
        ].all()
        else "blocked"
    )
    return out


def build_design_readiness(
    panel: pd.DataFrame,
    validation_summary: pd.DataFrame,
    exact_residual_overlap: pd.DataFrame,
) -> pd.DataFrame:
    design = build_design_panel(panel)
    checks = dict(zip(validation_summary["check"], validation_summary["value"], strict=False))
    share_sums_pass = str(checks.get("share_sums_pass")).lower() == "true"
    wam_pass = str(checks.get("wam_no_lookahead_pass")).lower() == "true"
    exact_overlap = exact_residual_overlap["exact_residual"].notna()
    exact_error = pd.to_numeric(
        exact_residual_overlap.loc[exact_overlap, "residual_proxy_error"],
        errors="coerce",
    ).abs()
    exact_residual_median = pd.to_numeric(
        exact_residual_overlap.loc[exact_overlap, "exact_residual"],
        errors="coerce",
    ).abs().median()
    error_ratio = exact_error.median() / exact_residual_median if exact_residual_median else pd.NA

    rows = [
        _readiness_row(
            "baseline_non_residual_contrasts",
            share_sums_pass and wam_pass and _has_min_rows(design, ["contrast_ru_minus_domestic"], 40),
            int(design["contrast_ru_minus_domestic"].notna().sum()),
            "RU-like, MMF, bank, and domestic-final contrasts that do not require residual interpretation.",
        ),
        _readiness_row(
            "issuance_first_stage_scaffold",
            share_sums_pass
            and wam_pass
            and _has_min_rows(design, ["bill_share", "wam_gap_trailing", "share_ru_like"], 40),
            int(design[["bill_share", "wam_gap_trailing", "share_ru_like"]].dropna().shape[0]),
            "Rows available for holder-share first-stage scaffolding on bill share and lagged WAM gap.",
        ),
        _readiness_row(
            "h0_h4_outcome_scaffold",
            share_sums_pass and _has_min_rows(design, ["contrast_ru_minus_domestic", "deposits_h4"], 40),
            int(design[["contrast_ru_minus_domestic", "deposits_h4"]].dropna().shape[0]),
            "Rows available for h=0..4 outcome design matrices before controls/exclusions.",
        ),
        _readiness_row(
            "residual_sensitive_contrasts",
            bool(exact_overlap.sum() >= 12 and pd.notna(error_ratio) and error_ratio <= 0.25),
            int(exact_overlap.sum()),
            "Requires exact residual overlap and small proxy error; currently conservative-gated.",
        ),
    ]
    out = pd.DataFrame(rows)
    out["exact_residual_proxy_median_abs_error_ratio"] = error_ratio
    return out


def write_design_outputs(
    panel: pd.DataFrame,
    *,
    validation_summary_path: str | Path = "data/processed/validation_summary.csv",
    exact_residual_overlap_path: str | Path = "data/processed/exact_residual_overlap_diagnostics.csv",
    output_dir: str | Path = "data/processed",
) -> list[Path]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    validation_summary = pd.read_csv(validation_summary_path)
    exact_residual_overlap = pd.read_csv(exact_residual_overlap_path)
    outputs = {
        "design_panel.csv": build_design_panel(panel),
    }
    outputs["design_readiness.csv"] = build_design_readiness(
        panel,
        validation_summary,
        exact_residual_overlap,
    )
    outputs["first_stage_summary.csv"] = build_first_stage_summary(outputs["design_panel.csv"])
    outputs["horizon_design_matrices.csv"] = build_horizon_design_matrices(outputs["design_panel.csv"])
    outputs["pretrend_placebo_matrices.csv"] = build_pretrend_placebo_matrices(outputs["design_panel.csv"])
    outputs["horizon_design_matrix_summary.csv"] = build_horizon_matrix_summary(
        outputs["horizon_design_matrices.csv"]
    )
    outputs["pretrend_placebo_matrix_summary.csv"] = build_horizon_matrix_summary(
        outputs["pretrend_placebo_matrices.csv"]
    )
    outputs["stop_rule_status.csv"] = build_stop_rule_status(
        validation_summary,
        outputs["design_readiness.csv"],
        outputs["first_stage_summary.csv"],
    )
    paths = []
    for name, frame in outputs.items():
        path = out_dir / name
        frame.to_csv(path, index=False)
        paths.append(path)
    return paths


def _sum_columns(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    values = [pd.to_numeric(frame[column], errors="coerce") for column in columns]
    observed = pd.concat(values, axis=1).notna().any(axis=1)
    return sum(series.fillna(0) for series in values).where(observed)


def _has_min_rows(frame: pd.DataFrame, columns: list[str], minimum: int) -> bool:
    return frame[columns].dropna().shape[0] >= minimum


def _readiness_row(name: str, ready: bool, rows: int, notes: str) -> dict[str, object]:
    return {
        "design": name,
        "ready": bool(ready),
        "usable_rows": rows,
        "notes": notes,
    }


def _stop_rule_row(stop_rule: str, passes: bool, notes: str) -> dict[str, object]:
    return {
        "stop_rule": stop_rule,
        "passes": bool(passes),
        "notes": notes,
    }


def _ols_summary(
    frame: pd.DataFrame,
    *,
    outcome: str,
    treatment: str,
    controls: list[str],
    sample_label: str,
) -> dict[str, object]:
    columns = [outcome, treatment, *controls]
    sample = frame[columns].apply(pd.to_numeric, errors="coerce").dropna()
    base = {
        "outcome": outcome,
        "treatment": treatment,
        "sample": sample_label,
        "controls": ";".join(controls),
        "n": int(len(sample)),
        "beta": np.nan,
        "se": np.nan,
        "t_stat": np.nan,
        "r2": np.nan,
        "status": "insufficient_rows",
    }
    min_rows = len(controls) + 8
    if len(sample) < min_rows:
        return base
    y = sample[outcome].to_numpy(dtype=float)
    x = np.column_stack(
        [
            np.ones(len(sample)),
            sample[treatment].to_numpy(dtype=float),
            *[sample[control].to_numpy(dtype=float) for control in controls],
        ]
    )
    try:
        beta, *_ = np.linalg.lstsq(x, y, rcond=None)
        residual = y - x @ beta
        dof = len(y) - x.shape[1]
        if dof <= 0:
            return base
        sigma2 = float((residual @ residual) / dof)
        cov = sigma2 * np.linalg.pinv(x.T @ x)
        se = float(np.sqrt(max(cov[1, 1], 0.0)))
        y_centered = y - y.mean()
        tss = float(y_centered @ y_centered)
        r2 = np.nan if tss == 0 else 1.0 - float(residual @ residual) / tss
        base.update(
            {
                "beta": float(beta[1]),
                "se": se,
                "t_stat": float(beta[1] / se) if se else np.nan,
                "r2": float(r2),
                "status": "ok",
            }
        )
        return base
    except np.linalg.LinAlgError:
        base["status"] = "linear_algebra_failure"
        return base
