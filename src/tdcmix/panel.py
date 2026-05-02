from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from tdcmix.contracts import SourceContract, panel_columns, source_records
from tdcmix.shares import add_positive_absorption_shares


def holder_panel_columns(contracts_config: dict, mix_config: dict) -> list[str]:
    columns = panel_columns(contracts_config)
    sector_columns = mix_config["sector_transaction_columns"]
    share_columns = [f"pos_abs_share_{sector}" for sector in sector_columns]
    return columns + ["positive_absorption_total", *share_columns]


def build_holder_panel_skeleton(
    contracts_config: dict,
    mix_config: dict,
    quarters: Iterable[str] = (),
) -> pd.DataFrame:
    columns = holder_panel_columns(contracts_config, mix_config)
    frame = pd.DataFrame({"quarter": list(quarters)})
    for column in columns:
        if column not in frame.columns:
            frame[column] = pd.NA
    return frame[columns]


def add_holder_absorption_shares(frame: pd.DataFrame, mix_config: dict) -> pd.DataFrame:
    return add_positive_absorption_shares(frame, mix_config["sector_transaction_columns"])


def write_holder_panel_skeleton(
    contracts_config: dict,
    mix_config: dict,
    output_path: str | Path,
    quarters: Iterable[str] = (),
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    build_holder_panel_skeleton(contracts_config, mix_config, quarters).to_csv(
        path,
        index=False,
    )
    return path


def build_holder_panel(
    contracts_config: dict,
    mix_config: dict,
    *,
    project_root: str | Path = ".",
) -> pd.DataFrame:
    frames = [
        _load_contract_series(record, Path(project_root))
        for record in source_records(contracts_config)
        if record.status != "placeholder"
    ]
    if not frames:
        return build_holder_panel_skeleton(contracts_config, mix_config)

    panel = frames[0]
    for frame in frames[1:]:
        panel = panel.merge(frame, on="quarter", how="outer")
    panel = panel.sort_values("quarter").reset_index(drop=True)

    for column in panel_columns(contracts_config):
        if column not in panel.columns:
            panel[column] = pd.NA
    panel = panel[panel_columns(contracts_config)]
    panel = _compute_residual_holder_bucket(panel, mix_config)
    return add_holder_absorption_shares(panel, mix_config)


def write_holder_panel(
    contracts_config: dict,
    mix_config: dict,
    output_path: str | Path,
    *,
    project_root: str | Path = ".",
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    build_holder_panel(
        contracts_config,
        mix_config,
        project_root=project_root,
    ).to_csv(path, index=False)
    return path


def build_exact_fu_primary_holder_panel(
    exact_holder_panel: pd.DataFrame,
    migration_gate: pd.DataFrame,
    contracts_config: dict,
    mix_config: dict,
) -> pd.DataFrame:
    if _gate_status(migration_gate) != "ready":
        raise ValueError("Exact-FU primary panel migration gate is not ready.")
    required = holder_panel_columns(contracts_config, mix_config)
    metadata = ["holder_source_layer", "residual_interpretation"]
    out = exact_holder_panel.copy()
    for column in required:
        if column not in out.columns:
            out[column] = pd.NA
    out["holder_source_layer"] = "z1_exact_fu_primary_migrated"
    out["residual_interpretation"] = (
        "exact FU primary source-layer migration; official FU all-sector transactions "
        "minus mapped official FU components; legacy holder panel retained separately"
    )
    return out[[*required, *metadata]].sort_values("quarter").reset_index(drop=True)


def build_exact_fu_primary_migration_comparison(
    legacy_panel: pd.DataFrame,
    exact_sidecar_panel: pd.DataFrame,
    exact_primary_panel: pd.DataFrame,
    mix_config: dict,
) -> pd.DataFrame:
    rows = []
    sector_columns = mix_config["sector_transaction_columns"]
    compare_columns = [
        *sector_columns.values(),
        "positive_absorption_total",
        *[f"pos_abs_share_{bucket}" for bucket in sector_columns],
    ]
    rows.append(
        {
            "comparison": "primary_panel_shape",
            "metric": "row_count",
            "overlap_rows": int(len(exact_primary_panel)),
            "median_gap": 0.0,
            "median_abs_gap": 0.0,
            "max_abs_gap": 0.0,
            "status": "available",
        }
    )
    for column in compare_columns:
        if column in exact_sidecar_panel and column in exact_primary_panel:
            rows.append(
                _panel_gap_summary(
                    exact_sidecar_panel,
                    exact_primary_panel,
                    column,
                    comparison="exact_primary_vs_exact_sidecar",
                )
            )
        if column in legacy_panel and column in exact_primary_panel:
            rows.append(
                _panel_gap_summary(
                    legacy_panel,
                    exact_primary_panel,
                    column,
                    comparison="legacy_vs_exact_primary",
                )
            )
    return pd.DataFrame(rows)


def build_exact_fu_primary_migration_report(
    comparison: pd.DataFrame,
    migration_gate: pd.DataFrame,
) -> str:
    gate = migration_gate.iloc[0].to_dict() if not migration_gate.empty else {}
    sidecar = comparison.loc[comparison["comparison"].eq("exact_primary_vs_exact_sidecar")]
    sidecar_max = pd.to_numeric(sidecar["max_abs_gap"], errors="coerce").max() if not sidecar.empty else float("nan")
    lines = [
        "# Exact-FU Primary Panel Migration",
        "",
        f"- Migration gate status: `{gate.get('status', 'unknown')}`",
        f"- Primary panel migration allowed: `{gate.get('primary_panel_migration_allowed', False)}`",
        f"- Main empirical allowed: `{gate.get('main_empirical_allowed', False)}`",
        f"- Residual-sensitive headline allowed: `{gate.get('residual_sensitive_headline_allowed', False)}`",
        f"- Max exact-primary vs sidecar absolute gap: `{float(sidecar_max):.6f}`",
        "",
        "## Sidecar Equivalence",
        "",
    ]
    if sidecar.empty:
        lines.append("No exact-primary versus sidecar comparisons are available.")
    else:
        for row in sidecar.to_dict("records"):
            lines.append(
                "- "
                f"`{row['metric']}`: overlap `{int(row['overlap_rows'])}`, "
                f"median abs gap `{float(row['median_abs_gap']):.6f}`, "
                f"max abs gap `{float(row['max_abs_gap']):.6f}`"
            )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "This migration creates a primary-compatible exact-FU source-layer panel. It does not authorize causal, main empirical, or residual-sensitive headline claims.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_exact_fu_primary_panel_outputs(
    *,
    contracts_config: dict,
    mix_config: dict,
    legacy_panel_path: str | Path = "data/processed/holder_absorption_panel.csv",
    exact_sidecar_panel_path: str | Path = "data/processed/z1_exact_holder_absorption_panel.csv",
    migration_gate_path: str | Path = "data/processed/z1_exact_primary_migration_gate.csv",
    output_dir: str | Path = "data/processed",
) -> list[Path]:
    legacy_panel = pd.read_csv(legacy_panel_path)
    exact_sidecar = pd.read_csv(exact_sidecar_panel_path)
    migration_gate = pd.read_csv(migration_gate_path)
    exact_primary = build_exact_fu_primary_holder_panel(
        exact_sidecar,
        migration_gate,
        contracts_config,
        mix_config,
    )
    comparison = build_exact_fu_primary_migration_comparison(
        legacy_panel,
        exact_sidecar,
        exact_primary,
        mix_config,
    )
    report = build_exact_fu_primary_migration_report(comparison, migration_gate)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    panel_path = out_dir / "holder_absorption_panel_exact_primary.csv"
    comparison_path = out_dir / "exact_primary_migration_comparison.csv"
    report_path = out_dir / "exact_primary_migration_report.md"
    exact_primary.to_csv(panel_path, index=False)
    comparison.to_csv(comparison_path, index=False)
    report_path.write_text(report, encoding="utf-8")
    return [panel_path, comparison_path, report_path]


def _load_contract_series(record: SourceContract, root: Path) -> pd.DataFrame:
    path = _resolve_path(record, root)
    if path is None:
        raise ValueError(f"{record.key}: source path is required.")
    usecols = [record.date_column, *record.source_columns]
    frame = pd.read_csv(path, usecols=usecols)
    frame["quarter"] = _quarter_end(frame[record.date_column])

    if record.transform == "rename":
        return _single_column(frame, record.source_columns[0], record.output_column)
    if record.transform == "sign_normalize_to_toc_change":
        out = _single_column(frame, record.source_columns[0], record.output_column)
        out[record.output_column] = -pd.to_numeric(out[record.output_column], errors="coerce")
        return out
    if record.transform == "lag_one_quarter":
        out = _single_column(frame, record.source_columns[0], record.output_column)
        out = out.sort_values("quarter")
        out[record.output_column] = out[record.output_column].shift(1)
        return out
    if record.transform == "diff_holdings_fed_bn":
        return _holdings_wide_diff_sum(frame, record.output_column, ["fed"])
    if record.transform == "filter_sector_eq_banks":
        return _sector_sum(frame, record.output_column, ["Banks"])
    if record.transform == "filter_sector_eq_foreign_holders":
        return _sector_sum(frame, record.output_column, ["Foreign holders"])
    if record.transform == "filter_sector_eq_money_funds":
        return _sector_sum(frame, record.output_column, ["Money funds"])
    if record.transform == "filter_sector_eq_dealers":
        return _sector_sum(frame, record.output_column, ["Dealers"])
    if record.transform == "sum_sectors_households_nonprofits_nonfinancial_businesses_state_local":
        return _sector_sum(
            frame,
            record.output_column,
            [
                "Households & nonprofits",
                "Nonfinancial businesses",
                "State/local governments",
            ],
        )
    if record.transform == "sum_sectors_mutual_funds_etfs_pensions_insurers_other_domestic_financials":
        return _sector_sum(
            frame,
            record.output_column,
            [
                "Mutual funds & ETFs",
                "Pensions & insurers",
                "Other domestic financials",
            ],
        )
    if record.transform == "filter_sector_key_fed_transactions_bn":
        return _sector_key_sum(frame, record.output_column, ["fed"])
    if record.transform == "sum_sector_keys_bank_transactions_bn":
        return _sector_key_sum(
            frame,
            record.output_column,
            [
                "bank_us_chartered",
                "bank_foreign_banking_offices_us",
                "bank_us_affiliated_areas",
            ],
        )
    if record.transform == "filter_sector_key_foreigners_total_transactions_bn":
        return _sector_key_sum(frame, record.output_column, ["foreigners_total"])
    if record.transform == "all_sector_transactions_minus_named_buckets":
        return _sector_key_sum(frame, record.output_column, ["all_holders_total"])
    if record.transform == "diff_total_holdings_minus_named_buckets":
        return _holdings_long_total_diff(frame, record.output_column)
    raise ValueError(f"{record.key}: unsupported transform {record.transform!r}.")


def _single_column(frame: pd.DataFrame, source_column: str, output_column: str) -> pd.DataFrame:
    out = frame[["quarter", source_column]].copy()
    out[output_column] = pd.to_numeric(out[source_column], errors="coerce")
    return out[["quarter", output_column]]


def _sector_sum(frame: pd.DataFrame, output_column: str, sectors: list[str]) -> pd.DataFrame:
    subset = frame.loc[frame["sector"].isin(sectors), ["quarter", "transactions_bn"]].copy()
    subset["transactions_bn"] = pd.to_numeric(subset["transactions_bn"], errors="coerce")
    grouped = subset.groupby("quarter", as_index=False)["transactions_bn"].sum()
    return grouped.rename(columns={"transactions_bn": output_column})


def _sector_key_sum(frame: pd.DataFrame, output_column: str, sector_keys: list[str]) -> pd.DataFrame:
    subset = frame.loc[frame["sector_key"].isin(sector_keys), ["quarter", "transactions"]].copy()
    subset["transactions"] = pd.to_numeric(subset["transactions"], errors="coerce")
    grouped = subset.groupby("quarter", as_index=False)["transactions"].sum(min_count=1)
    return grouped.rename(columns={"transactions": output_column})


def _holdings_wide_diff_sum(
    frame: pd.DataFrame,
    output_column: str,
    columns: list[str],
) -> pd.DataFrame:
    work = frame[["quarter", *columns]].copy().sort_values("quarter")
    values = sum(pd.to_numeric(work[column], errors="coerce") for column in columns)
    work[output_column] = values.diff() / 1000.0
    return work[["quarter", output_column]]


def _holdings_long_total_diff(frame: pd.DataFrame, output_column: str) -> pd.DataFrame:
    subset = frame.loc[
        (frame["sector"] == "_total") & (frame["instrument"] == "all_treasuries"),
        ["quarter", "holdings"],
    ].copy()
    subset = subset.sort_values("quarter")
    subset[output_column] = pd.to_numeric(subset["holdings"], errors="coerce").diff() / 1000.0
    return subset[["quarter", output_column]]


def _compute_residual_holder_bucket(frame: pd.DataFrame, mix_config: dict) -> pd.DataFrame:
    out = frame.copy()
    sector_columns = mix_config["sector_transaction_columns"]
    residual_column = sector_columns.get("unmapped")
    if residual_column not in out.columns:
        return out
    named_columns = [
        column
        for sector, column in sector_columns.items()
        if sector != "unmapped" and column in out.columns
    ]
    if not named_columns:
        return out
    all_sector_total = pd.to_numeric(out[residual_column], errors="coerce")
    named_total = sum(
        pd.to_numeric(out[column], errors="coerce").fillna(0)
        for column in named_columns
    )
    out[residual_column] = (all_sector_total - named_total).where(all_sector_total.notna())
    return out


def _quarter_end(series: pd.Series) -> pd.Series:
    dates = pd.to_datetime(series, errors="coerce", format="mixed")
    return dates.dt.to_period("Q").dt.end_time.dt.date.astype("string")


def _resolve_path(record: SourceContract, root: Path) -> Path | None:
    if record.path is None:
        return None
    path = Path(record.path)
    return path if path.is_absolute() else root / path


def _gate_status(migration_gate: pd.DataFrame) -> str:
    if migration_gate.empty or "status" not in migration_gate:
        return "unknown"
    return str(migration_gate["status"].iloc[0])


def _panel_gap_summary(
    left: pd.DataFrame,
    right: pd.DataFrame,
    column: str,
    *,
    comparison: str,
) -> dict[str, object]:
    merged = left[["quarter", column]].merge(
        right[["quarter", column]],
        on="quarter",
        how="inner",
        suffixes=("_left", "_right"),
    )
    left_values = pd.to_numeric(merged[f"{column}_left"], errors="coerce")
    right_values = pd.to_numeric(merged[f"{column}_right"], errors="coerce")
    valid = left_values.notna() & right_values.notna()
    gap = right_values.loc[valid] - left_values.loc[valid]
    abs_gap = gap.abs()
    return {
        "comparison": comparison,
        "metric": column,
        "overlap_rows": int(valid.sum()),
        "median_gap": float(gap.median()) if not gap.empty else float("nan"),
        "median_abs_gap": float(abs_gap.median()) if not abs_gap.empty else float("nan"),
        "max_abs_gap": float(abs_gap.max()) if not abs_gap.empty else float("nan"),
        "status": "matches" if not abs_gap.empty and float(abs_gap.max()) <= 1e-9 else "differs",
    }
