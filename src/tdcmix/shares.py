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
    positive_components = {
        sector: out[column].clip(lower=0) for sector, column in sector_columns.items()
    }
    out[denominator_column] = sum(positive_components.values())

    for sector, series in positive_components.items():
        share_column = f"{prefix}_{sector}"
        out[share_column] = series.div(out[denominator_column]).where(
            out[denominator_column] != 0
        )

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
