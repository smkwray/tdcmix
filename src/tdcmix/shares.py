from __future__ import annotations

import pandas as pd


def add_positive_absorption_shares(
    frame: pd.DataFrame,
    sector_columns: dict[str, str],
    *,
    prefix: str = "pos_abs_share",
    denominator_column: str = "positive_absorption_total",
) -> pd.DataFrame:
    missing = [column for column in sector_columns.values() if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing sector transaction columns: {missing}")

    out = frame.copy()
    numeric_components = {
        sector: pd.to_numeric(out[column], errors="coerce")
        for sector, column in sector_columns.items()
    }
    observed_any_sector = pd.concat(numeric_components.values(), axis=1).notna().any(axis=1)
    positive_components = {
        sector: series.fillna(0).clip(lower=0)
        for sector, series in numeric_components.items()
    }
    out[denominator_column] = sum(positive_components.values()).where(observed_any_sector)
    denominator = pd.to_numeric(out[denominator_column], errors="coerce")
    nonzero_denominator = denominator.where(denominator != 0)

    for sector, series in positive_components.items():
        share_column = f"{prefix}_{sector}"
        out[share_column] = series.div(nonzero_denominator)

    return out


def validate_share_sums(
    frame: pd.DataFrame,
    share_columns: list[str],
    *,
    tolerance: float = 1e-9,
) -> pd.Series:
    if not share_columns:
        raise ValueError("At least one share column is required.")
    missing = [column for column in share_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing share columns: {missing}")

    sums = frame[share_columns].sum(axis=1, skipna=True)
    bad = sums.notna() & (sums.sub(1.0).abs() > tolerance)
    if bad.any():
        bad_rows = bad[bad].index.tolist()
        raise ValueError(f"Share sums fail tolerance on rows: {bad_rows}")
    return sums
