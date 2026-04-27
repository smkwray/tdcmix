import pandas as pd
import pytest

from tdcmix.shares import add_positive_absorption_shares, validate_share_sums


def test_add_positive_absorption_shares() -> None:
    frame = pd.DataFrame(
        {
            "sector_tx_banks": [100.0, -10.0],
            "sector_tx_mmf": [50.0, 20.0],
            "sector_tx_domestic_nonbank": [-25.0, 20.0],
        }
    )
    sector_columns = {
        "banks": "sector_tx_banks",
        "mmf": "sector_tx_mmf",
        "domestic_nonbank": "sector_tx_domestic_nonbank",
    }

    out = add_positive_absorption_shares(frame, sector_columns)

    assert out.loc[0, "positive_absorption_total"] == 150.0
    assert out.loc[0, "pos_abs_share_banks"] == pytest.approx(2 / 3)
    assert out.loc[0, "pos_abs_share_domestic_nonbank"] == 0.0
    assert out.loc[1, "pos_abs_share_mmf"] == pytest.approx(0.5)
    assert out.loc[1, "pos_abs_share_domestic_nonbank"] == pytest.approx(0.5)


def test_validate_share_sums() -> None:
    frame = pd.DataFrame({"a": [0.25, 0.4], "b": [0.75, 0.6]})

    sums = validate_share_sums(frame, ["a", "b"])

    assert sums.tolist() == [1.0, 1.0]


def test_validate_share_sums_rejects_bad_rows() -> None:
    frame = pd.DataFrame({"a": [0.25], "b": [0.70]})

    with pytest.raises(ValueError, match="fail tolerance"):
        validate_share_sums(frame, ["a", "b"])
