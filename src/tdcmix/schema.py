from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml


class HolderBucket(StrEnum):
    FED = "fed"
    BANKS = "banks"
    ROW = "row"
    MMF = "mmf"
    DOMESTIC_NONBANK = "domestic_nonbank"
    DEALER_BRIDGE = "dealer_bridge"
    OTHER_FINANCIAL = "other_financial"
    UNMAPPED = "unmapped"


REQUIRED_BUCKETS = frozenset(bucket.value for bucket in HolderBucket)


@dataclass(frozen=True)
class SectorBucket:
    name: str
    label: str
    role: str
    include_in_headline: bool
    description: str


def load_sector_map(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError("Sector map must load as a dictionary.")
    validate_sector_map(loaded)
    return loaded


def validate_sector_map(sector_map: dict[str, Any]) -> None:
    buckets = sector_map.get("buckets")
    if not isinstance(buckets, dict):
        raise ValueError("Sector map requires a 'buckets' mapping.")

    missing = REQUIRED_BUCKETS.difference(buckets)
    extra = set(buckets).difference(REQUIRED_BUCKETS)
    if missing:
        raise ValueError(f"Sector map is missing buckets: {sorted(missing)}")
    if extra:
        raise ValueError(f"Sector map has unknown buckets: {sorted(extra)}")

    for name, record in buckets.items():
        if not isinstance(record, dict):
            raise ValueError(f"Bucket {name!r} must be a mapping.")
        for field in ("label", "role", "include_in_headline", "description"):
            if field not in record:
                raise ValueError(f"Bucket {name!r} is missing {field!r}.")

    blocks = sector_map.get("functional_blocks", {})
    if not isinstance(blocks, dict):
        raise ValueError("'functional_blocks' must be a mapping when present.")
    for block, members in blocks.items():
        if not isinstance(members, list):
            raise ValueError(f"Functional block {block!r} must be a list.")
        unknown_members = set(members).difference(REQUIRED_BUCKETS)
        if unknown_members:
            raise ValueError(
                f"Functional block {block!r} has unknown members: {sorted(unknown_members)}"
            )
