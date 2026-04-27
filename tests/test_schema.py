from pathlib import Path

import pytest

from tdcmix.schema import REQUIRED_BUCKETS, load_sector_map, validate_sector_map


def test_sector_map_has_required_buckets() -> None:
    sector_map = load_sector_map(Path("config/holder_sector_map.yml"))

    assert set(sector_map["buckets"]) == REQUIRED_BUCKETS


def test_sector_map_rejects_missing_bucket() -> None:
    sector_map = load_sector_map(Path("config/holder_sector_map.yml"))
    sector_map["buckets"].pop("fed")

    with pytest.raises(ValueError, match="missing buckets"):
        validate_sector_map(sector_map)
