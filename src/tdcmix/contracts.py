from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from tdcmix.schema import REQUIRED_BUCKETS


ALLOWED_STATUSES = frozenset({"source_candidate", "placeholder", "blocked"})
ALLOWED_ROLES = frozenset(
    {
        "holder_transaction",
        "tdc_component",
        "issuance_composition",
        "outcome",
    }
)
TRANSACTION_LEAK_TERMS = ("revaluation", "_fr", "reval")


@dataclass(frozen=True)
class SourceContract:
    key: str
    output_column: str
    role: str
    concept: str
    source_project: str
    path: str | None
    source_columns: tuple[str, ...]
    transform: str
    status: str
    frequency: str
    units: str
    date_column: str
    holder_bucket: str | None = None
    no_lookahead: bool = False
    notes: str = ""


def load_source_contracts(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError("Source contracts must load as a dictionary.")
    validate_source_contracts(loaded)
    return loaded


def source_records(config: dict[str, Any]) -> list[SourceContract]:
    defaults = config.get("defaults", {})
    records = []
    for record in config.get("sources", []):
        records.append(
            SourceContract(
                key=str(record["key"]),
                output_column=str(record["output_column"]),
                role=str(record["role"]),
                concept=str(record["concept"]),
                source_project=str(record["source_project"]),
                path=record.get("path"),
                source_columns=tuple(record.get("source_columns", [])),
                transform=str(record["transform"]),
                status=str(record["status"]),
                frequency=str(record.get("frequency", defaults.get("frequency", ""))),
                units=str(record.get("units", defaults.get("units", ""))),
                date_column=str(record.get("date_column", defaults.get("date_column", ""))),
                holder_bucket=record.get("holder_bucket"),
                no_lookahead=bool(record.get("no_lookahead", False)),
                notes=str(record.get("notes", "")),
            )
        )
    return records


def validate_source_contracts(config: dict[str, Any]) -> None:
    for field in ("schema_version", "defaults", "panel", "sources"):
        if field not in config:
            raise ValueError(f"Source contracts missing top-level {field!r}.")
    if not isinstance(config["sources"], list) or not config["sources"]:
        raise ValueError("Source contracts require a non-empty 'sources' list.")

    panel = config["panel"]
    if not isinstance(panel, dict):
        raise ValueError("'panel' must be a mapping.")
    required_output_columns = panel.get("required_output_columns")
    if not isinstance(required_output_columns, list) or "quarter" not in required_output_columns:
        raise ValueError("'panel.required_output_columns' must include 'quarter'.")

    records = source_records(config)
    keys = [record.key for record in records]
    if len(keys) != len(set(keys)):
        raise ValueError("Source contract keys must be unique.")

    output_columns = [record.output_column for record in records]
    duplicate_outputs = sorted(
        column for column in set(output_columns) if output_columns.count(column) > 1
    )
    if duplicate_outputs:
        raise ValueError(f"Output columns mapped more than once: {duplicate_outputs}")

    contracted_outputs = set(output_columns) | {"quarter"}
    missing_outputs = set(required_output_columns).difference(contracted_outputs)
    if missing_outputs:
        raise ValueError(
            "Panel required columns lack source contracts: "
            f"{sorted(missing_outputs)}"
        )

    holder_buckets = {
        record.holder_bucket
        for record in records
        if record.role == "holder_transaction" and record.holder_bucket is not None
    }
    missing_buckets = REQUIRED_BUCKETS.difference(holder_buckets)
    if missing_buckets:
        raise ValueError(
            "Holder transaction contracts missing buckets: "
            f"{sorted(missing_buckets)}"
        )

    for record in records:
        _validate_record(record)


def _validate_record(record: SourceContract) -> None:
    if record.role not in ALLOWED_ROLES:
        raise ValueError(f"{record.key}: unknown role {record.role!r}.")
    if record.status not in ALLOWED_STATUSES:
        raise ValueError(f"{record.key}: unknown status {record.status!r}.")
    if not record.output_column:
        raise ValueError(f"{record.key}: output_column is required.")
    if not record.frequency:
        raise ValueError(f"{record.key}: frequency is required.")
    if not record.units:
        raise ValueError(f"{record.key}: units is required.")
    if record.status != "placeholder" and not record.path:
        raise ValueError(f"{record.key}: non-placeholder contracts require a path.")
    if record.status != "placeholder" and not record.source_columns:
        raise ValueError(f"{record.key}: non-placeholder contracts require source columns.")
    if record.role == "holder_transaction":
        if record.holder_bucket not in REQUIRED_BUCKETS:
            raise ValueError(f"{record.key}: invalid holder bucket {record.holder_bucket!r}.")
        if record.concept != "transaction":
            raise ValueError(f"{record.key}: holder buckets must use transaction concept.")
    if record.concept == "transaction":
        lowered = " ".join(record.source_columns).lower()
        if any(term in lowered for term in TRANSACTION_LEAK_TERMS):
            raise ValueError(f"{record.key}: transaction contract may leak revaluation fields.")
    if record.no_lookahead and "lag" not in record.transform:
        raise ValueError(f"{record.key}: no_lookahead contracts must use a lag transform.")


def panel_columns(config: dict[str, Any]) -> list[str]:
    return list(config["panel"]["required_output_columns"])
