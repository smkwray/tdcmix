from __future__ import annotations

from pathlib import Path

import pandas as pd

from tdcmix.contracts import SourceContract, source_records


Z1_CANDIDATE_SOURCES = [
    {
        "source_project": "tsyparty",
        "source_path": "../tsyparty/data/interim/z1_holdings_long.csv",
        "source_kind": "long_holdings",
        "date_column": "date",
    },
    {
        "source_project": "tsyparty",
        "source_path": "../tsyparty/outputs/holder_response/issuance_maturity_response/sector_transaction_groups.csv",
        "source_kind": "grouped_transactions",
        "date_column": "date",
    },
    {
        "source_project": "wamest",
        "source_path": "../wamest/data/interim/z1_series_panel_full.csv",
        "source_kind": "series_panel",
        "date_column": "date",
    },
]


def build_sector_map_audit(sector_map: dict, contracts_config: dict) -> pd.DataFrame:
    records = source_records(contracts_config)
    by_bucket = {
        record.holder_bucket: record
        for record in records
        if record.role == "holder_transaction" and record.holder_bucket is not None
    }

    rows = []
    for bucket, metadata in sector_map["buckets"].items():
        contract = by_bucket.get(bucket)
        rows.append(
            {
                "bucket": bucket,
                "label": metadata["label"],
                "role": metadata["role"],
                "include_in_headline": metadata["include_in_headline"],
                "source_project": contract.source_project if contract else "",
                "source_path": contract.path if contract and contract.path else "",
                "source_columns": ";".join(contract.source_columns) if contract else "",
                "transform": contract.transform if contract else "",
                "status": contract.status if contract else "missing_contract",
                "open_reconciliation_question": _reconciliation_question(bucket, contract),
            }
        )
    return pd.DataFrame(rows)


def write_sector_map_audit(
    sector_map: dict,
    contracts_config: dict,
    output_path: str | Path,
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    build_sector_map_audit(sector_map, contracts_config).to_csv(path, index=False)
    return path


def build_source_overlap_audit(
    contracts_config: dict,
    *,
    project_root: str | Path = ".",
) -> pd.DataFrame:
    root = Path(project_root)
    rows = []
    for record in source_records(contracts_config):
        rows.append(_source_overlap_row(record, root))
    return pd.DataFrame(rows)


def write_source_overlap_audit(
    contracts_config: dict,
    output_path: str | Path,
    *,
    project_root: str | Path = ".",
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    build_source_overlap_audit(contracts_config, project_root=project_root).to_csv(
        path,
        index=False,
    )
    return path


def build_z1_candidate_audit(
    *,
    project_root: str | Path = ".",
    sources: list[dict[str, str]] | None = None,
) -> pd.DataFrame:
    root = Path(project_root)
    rows = []
    for source in sources or Z1_CANDIDATE_SOURCES:
        rows.extend(_z1_candidate_rows(source, root))
    return pd.DataFrame(rows).sort_values(
        ["holder_bucket_candidate", "review_grade", "source_project", "series_or_sector"],
    ).reset_index(drop=True)


def write_z1_candidate_audit(
    output_path: str | Path,
    *,
    project_root: str | Path = ".",
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    build_z1_candidate_audit(project_root=project_root).to_csv(path, index=False)
    return path


def build_source_upgrade_recommendations(
    z1_candidates: pd.DataFrame,
    contracts_config: dict,
) -> pd.DataFrame:
    current = {
        record.holder_bucket: record
        for record in source_records(contracts_config)
        if record.role == "holder_transaction" and record.holder_bucket is not None
    }
    buckets = sorted(current)
    rows = []
    for bucket in buckets:
        candidates = z1_candidates.loc[z1_candidates["holder_bucket_candidate"].eq(bucket)].copy()
        contract = current[bucket]
        best = _best_candidate(candidates)
        rows.append(
            {
                "holder_bucket": bucket,
                "current_source_project": contract.source_project,
                "current_source_path": contract.path or "",
                "current_transform": contract.transform,
                "recommended_action": _recommended_action(bucket, best, contract),
                "recommended_source_project": best.get("source_project", ""),
                "recommended_source_path": best.get("source_path", ""),
                "recommended_series_or_sector": best.get("series_or_sector", ""),
                "recommended_source_kind": best.get("source_kind", ""),
                "recommended_review_grade": best.get("review_grade", ""),
                "recommended_non_null_observations": int(best.get("non_null_observation_count", 0) or 0),
                "candidate_count": int(len(candidates)),
                "upgrade_note": _upgrade_note(bucket, best, contract),
            }
        )
    return pd.DataFrame(rows)


def build_source_upgrade_recommendations_report(recommendations: pd.DataFrame) -> str:
    lines = [
        "# Source Upgrade Recommendations",
        "",
        "This report reviews whether holder-transaction source contracts should change based on local Z.1 candidates.",
        "",
    ]
    for row in recommendations.to_dict("records"):
        lines.append(
            "- "
            f"`{row['holder_bucket']}`: `{row['recommended_action']}`; "
            f"candidate `{row['recommended_series_or_sector']}` "
            f"({row['recommended_source_kind']}, grade `{row['recommended_review_grade']}`, "
            f"non-null `{int(row['recommended_non_null_observations'])}`). "
            f"{row['upgrade_note']}"
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "Recommendations do not edit source contracts automatically. Promote a source only after checking units, perimeter, and no-revaluation leakage.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_source_upgrade_recommendations(
    z1_candidates_output_path: str | Path,
    recommendations_output_path: str | Path,
    report_output_path: str | Path,
    contracts_config: dict,
    *,
    project_root: str | Path = ".",
) -> list[Path]:
    candidates_path = write_z1_candidate_audit(z1_candidates_output_path, project_root=project_root)
    candidates = pd.read_csv(candidates_path)
    recommendations = build_source_upgrade_recommendations(candidates, contracts_config)
    report = build_source_upgrade_recommendations_report(recommendations)
    recommendations_path = Path(recommendations_output_path)
    report_path = Path(report_output_path)
    recommendations_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    recommendations.to_csv(recommendations_path, index=False)
    report_path.write_text(report, encoding="utf-8")
    return [candidates_path, recommendations_path, report_path]


def _source_overlap_row(record: SourceContract, root: Path) -> dict[str, object]:
    resolved = _resolve_source_path(record, root)
    row: dict[str, object] = {
        "key": record.key,
        "output_column": record.output_column,
        "role": record.role,
        "source_project": record.source_project,
        "source_path": record.path or "",
        "exists": bool(resolved and resolved.exists()),
        "date_column": record.date_column,
        "source_columns": ";".join(record.source_columns),
        "status": record.status,
        "row_count": 0,
        "start": "",
        "end": "",
        "missing_columns": "",
    }
    if not resolved or not resolved.exists() or resolved.suffix.lower() != ".csv":
        return row

    header = pd.read_csv(resolved, nrows=0)
    columns = set(header.columns)
    expected = set(record.source_columns) | {record.date_column}
    missing = sorted(expected.difference(columns))
    row["missing_columns"] = ";".join(missing)

    if missing:
        return row

    frame = pd.read_csv(resolved, usecols=[record.date_column])
    dates = pd.to_datetime(
        frame[record.date_column],
        errors="coerce",
        format="mixed",
    ).dropna()
    row["row_count"] = len(frame)
    if not dates.empty:
        row["start"] = dates.min().date().isoformat()
        row["end"] = dates.max().date().isoformat()
    return row


def _resolve_source_path(record: SourceContract, root: Path) -> Path | None:
    if not record.path:
        return None
    path = Path(record.path)
    return path if path.is_absolute() else root / path


def _reconciliation_question(bucket: str, contract: SourceContract | None) -> str:
    if contract is None:
        return "Add a holder transaction source contract."
    questions = {
        "fed": "Confirm tdcest Fed transaction leg matches SOMA and Z.1 monetary authority transactions.",
        "banks": "Reconcile tdcest depository perimeter to wamest bank sub-buckets.",
        "row": "Check ROW total against official/private TIC splits where available.",
        "mmf": "Validate grouped money-fund transactions against direct Z.1 MMF Treasury holdings.",
        "domestic_nonbank": "Confirm household, nonprofit, nonfinancial business, and state/local coverage.",
        "dealer_bridge": "Keep bridge warehousing separate from final holders in headline contrasts.",
        "other_financial": "Confirm remaining financial groups are not absorbing residual headline signal.",
        "unmapped": "Compute after all named buckets and monitor residual size.",
    }
    return questions[bucket]


def _z1_candidate_rows(source: dict[str, str], root: Path) -> list[dict[str, object]]:
    path = Path(source["source_path"])
    resolved = path if path.is_absolute() else root / path
    base = {
        "source_project": source["source_project"],
        "source_path": source["source_path"],
        "source_kind": source["source_kind"],
        "exists": resolved.exists(),
        "series_or_sector": "",
        "description": "",
        "instrument": "",
        "holder_bucket_candidate": "unknown",
        "concept": "",
        "row_count": 0,
        "non_null_observation_count": 0,
        "start": "",
        "end": "",
        "review_grade": "missing",
        "review_note": "Source file is not available in the local workspace.",
    }
    if not resolved.exists() or resolved.suffix.lower() != ".csv":
        return [base]

    header = pd.read_csv(resolved, nrows=0)
    columns = set(header.columns)
    source_kind = source["source_kind"]
    if source_kind == "long_holdings" and {"date", "sector", "instrument", "holdings"}.issubset(columns):
        frame = pd.read_csv(resolved, usecols=["date", "sector", "instrument", "holdings"])
        return [_long_holdings_candidate_row(base, group) for _, group in frame.groupby(["sector", "instrument"])]
    if source_kind == "grouped_transactions" and {"date", "sector", "transactions_bn"}.issubset(columns):
        frame = pd.read_csv(resolved, usecols=["date", "sector", "transactions_bn"])
        return [_grouped_transaction_candidate_row(base, group) for _, group in frame.groupby("sector")]
    if source_kind == "series_panel" and {"date", "series_key", "description"}.issubset(columns):
        value_columns = [column for column in ["level", "transactions", "revaluation"] if column in columns]
        frame = pd.read_csv(resolved, usecols=["date", "series_key", "description", *value_columns])
        return [_series_panel_candidate_row(base, group, value_columns) for _, group in frame.groupby("series_key")]

    row = base.copy()
    row["review_note"] = "Source file exists but does not match a supported Z.1 audit schema."
    return [row]


def _long_holdings_candidate_row(base: dict[str, object], group: pd.DataFrame) -> dict[str, object]:
    sector = str(group["sector"].iloc[0])
    instrument = str(group["instrument"].iloc[0])
    row = _coverage_row(base, group, "holdings")
    row.update(
        {
            "series_or_sector": sector,
            "instrument": instrument,
            "holder_bucket_candidate": _bucket_from_text(sector),
            "concept": "level_holdings;transaction_proxy_from_first_difference",
        }
    )
    row.update(_z1_candidate_review(row))
    return row


def _grouped_transaction_candidate_row(base: dict[str, object], group: pd.DataFrame) -> dict[str, object]:
    sector = str(group["sector"].iloc[0])
    row = _coverage_row(base, group, "transactions_bn")
    row.update(
        {
            "series_or_sector": sector,
            "holder_bucket_candidate": _bucket_from_text(sector),
            "concept": "grouped_transaction_flow",
        }
    )
    row.update(_z1_candidate_review(row))
    return row


def _series_panel_candidate_row(
    base: dict[str, object],
    group: pd.DataFrame,
    value_columns: list[str],
) -> dict[str, object]:
    series_key = str(group["series_key"].iloc[0])
    description = str(group["description"].dropna().iloc[0]) if group["description"].notna().any() else ""
    preferred_value = "transactions" if "transactions" in value_columns else value_columns[0]
    row = _coverage_row(base, group, preferred_value)
    row.update(
        {
            "series_or_sector": series_key,
            "description": description,
            "holder_bucket_candidate": _bucket_from_text(f"{series_key} {description}"),
            "concept": ";".join(value_columns),
        }
    )
    row.update(_z1_candidate_review(row))
    return row


def _coverage_row(base: dict[str, object], frame: pd.DataFrame, value_column: str) -> dict[str, object]:
    row = base.copy()
    dates = pd.to_datetime(frame["date"], errors="coerce", format="mixed").dropna()
    values = pd.to_numeric(frame[value_column], errors="coerce")
    row["row_count"] = int(len(frame))
    row["non_null_observation_count"] = int(values.notna().sum())
    if not dates.empty:
        row["start"] = dates.min().date().isoformat()
        row["end"] = dates.max().date().isoformat()
    return row


def _bucket_from_text(text: str) -> str:
    normalized = text.lower().replace("_", " ").replace("/", " ").replace("&", " and ")
    if "all sector" in normalized or normalized == " total":
        return "all_sectors"
    if "monetary authority" in normalized or "fed" in normalized:
        return "fed"
    if "money market" in normalized or "money funds" in normalized:
        return "mmf"
    if "rest of world" in normalized or "foreign" in normalized:
        return "row"
    if "dealer" in normalized or "brokers" in normalized:
        return "dealer_bridge"
    if "bank" in normalized or "deposit" in normalized or "credit union" in normalized:
        return "banks"
    if (
        "household" in normalized
        or "nonprofit" in normalized
        or "nonfinancial" in normalized
        or "state local" in normalized
    ):
        return "domestic_nonbank"
    if (
        "financial" in normalized
        or "fund" in normalized
        or "insurer" in normalized
        or "pension" in normalized
        or "exchange traded" in normalized
        or "gse" in normalized
        or "government sponsored" in normalized
        or "asset backed" in normalized
        or "issuer" in normalized
        or "holding compan" in normalized
    ):
        return "other_financial"
    if "discrepancy" in normalized:
        return "unmapped"
    return "unknown"


def _z1_candidate_review(row: dict[str, object]) -> dict[str, str]:
    bucket = str(row["holder_bucket_candidate"])
    concept = str(row["concept"])
    source_kind = str(row["source_kind"])
    if bucket == "unknown":
        return {
            "review_grade": "D",
            "review_note": "Candidate cannot be mapped to a holder bucket without manual review.",
        }
    if "grouped_transaction_flow" in concept:
        return {
            "review_grade": "A" if bucket not in {"other_financial", "domestic_nonbank"} else "B",
            "review_note": "Direct grouped transaction-flow candidate; verify sector perimeter before headline use.",
        }
    if source_kind == "series_panel" and "transactions" in concept:
        return {
            "review_grade": "B",
            "review_note": "Series-level Z.1 transaction candidate; useful for validating grouped buckets and sub-bucket coverage.",
        }
    if "transaction_proxy_from_first_difference" in concept:
        return {
            "review_grade": "C",
            "review_note": "Long-history holdings-diff proxy; useful for coverage and residual checks but includes valuation noise.",
        }
    return {
        "review_grade": "C",
        "review_note": "Candidate is useful context but needs a transaction-flow transform before empirical use.",
    }


def _best_candidate(candidates: pd.DataFrame) -> dict[str, object]:
    if candidates.empty:
        return {}
    out = candidates.copy()
    grade_rank = {"A": 0, "B": 1, "C": 2, "D": 3, "missing": 4}
    out["grade_rank"] = out["review_grade"].map(grade_rank).fillna(9).astype(int)
    out["non_null_rank"] = -pd.to_numeric(out["non_null_observation_count"], errors="coerce").fillna(0)
    out["kind_rank"] = out["source_kind"].map(
        {"grouped_transactions": 0, "series_panel": 1, "long_holdings": 2}
    ).fillna(9)
    out = out.sort_values(["grade_rank", "kind_rank", "non_null_rank", "series_or_sector"])
    return out.iloc[0].drop(labels=["grade_rank", "non_null_rank", "kind_rank"]).to_dict()


def _recommended_action(bucket: str, best: dict[str, object], contract: SourceContract) -> str:
    if not best:
        return "no_local_candidate_found"
    best_kind = str(best.get("source_kind", ""))
    best_observations = int(best.get("non_null_observation_count", 0) or 0)
    current_path = contract.path or ""
    if bucket in {"mmf", "other_financial"} and best_kind == "grouped_transactions" and best_observations >= 250:
        return "keep_grouped_flow_primary_validate_with_series_panel"
    if best_kind == "series_panel" and best_observations < 40:
        return "keep_current_source_series_panel_too_short"
    if best_kind == "grouped_transactions" and best.get("source_path") == current_path:
        return "keep_current_source"
    if best_kind == "grouped_transactions":
        return "consider_grouped_flow_primary"
    if best_kind == "long_holdings":
        return "use_only_as_long_history_proxy"
    return "manual_review_required"


def _upgrade_note(bucket: str, best: dict[str, object], contract: SourceContract) -> str:
    if not best:
        return "No local Z.1 candidate mapped to this holder bucket."
    if bucket == "mmf":
        return (
            "Do not move to wamest MMF series yet: the direct series-panel transaction candidates "
            "are short and currently lack non-null transaction observations; keep grouped flow primary."
        )
    if bucket == "other_financial":
        return (
            "Keep grouped flow primary because it has long coverage. Use series-panel rows to audit "
            "sub-bucket perimeter, not to replace the current contract yet."
        )
    if bucket == "unmapped":
        return "Residual remains computed from all-sector coverage less named buckets; do not promote discrepancy rows."
    if str(best.get("source_kind", "")) == "series_panel":
        return "Series-panel candidate is useful validation context but short relative to the current long-history scaffold."
    if str(best.get("source_kind", "")) == "long_holdings":
        return "Long-history holdings proxy is useful for coverage but contains valuation noise."
    return "Current recommendation is to retain the source unless manual perimeter review identifies a cleaner transform."
