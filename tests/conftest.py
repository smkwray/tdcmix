"""Offline sources at the real contract paths; empirical inputs stay untouched."""

from pathlib import Path
import shutil
import urllib.request

import numpy as np
import pandas as pd
import pytest
import yaml

from tdcmix.audits import Z1_CANDIDATE_SOURCES
from tdcmix.contracts import load_source_contracts, source_records
from tdcmix import z1_exact


REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_csv(root: Path, path: str, columns: dict) -> None:
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns).to_csv(target, index=False)


@pytest.fixture(scope="session")
def synthetic_family(tmp_path_factory):
    """260 varied quarters for regressions; only eight exact-overlap quarters.

    Source units, names and transforms match the unmodified project contracts.
    The short exact overlap deliberately keeps residual-sensitive gates blocked.
    No series is an empirical estimate or copied from a retained dataset.
    """
    root = tmp_path_factory.mktemp("synthetic-source-family")
    rng = np.random.default_rng(271828)
    quarters = pd.period_range("1961Q1", periods=260, freq="Q")
    dates = quarters.end_time.strftime("%Y-%m-%d")
    n = len(dates)
    sector_names = [
        "Banks", "Foreign holders", "Money funds", "Dealers",
        "Households & nonprofits", "Nonfinancial businesses", "State/local governments",
        "Mutual funds & ETFs", "Pensions & insurers", "Other domestic financials",
    ]
    flows = rng.uniform(1, 12, (n, len(sector_names)))
    fed = rng.uniform(2, 15, n)
    residual = np.full(n, 2.0)
    total = fed + flows.sum(axis=1) + residual
    tdc = rng.normal(100, 30, n)
    minus_toc = rng.normal(0, 10, n)
    gap = rng.normal(0, 0.5, n)
    # Fixed first observations make sign, units, lag and residual assertions exact.
    fed[1] = 4.0
    total[1] = fed[1] + flows[1].sum() + residual[1]
    minus_toc[1] = -3.0
    gap[0] = 0.75
    _write_csv(root, "tdcest/data/processed/tdc_estimates.csv", {
        "date": dates, "tdc_tier1_fed_corrected_bank_only_ru_flow": tdc,
    })
    _write_csv(root, "tdcest/data/processed/tdc_components.csv", {
        "date": dates, "minus_treasury_operating_cash_tx": minus_toc,
        "fed_remit_positive": rng.uniform(0, 2, n), "fed_tsy_tx": 1000 * fed,
        "bank_depository_tsy_tx": 1000 * flows[:, 0], "row_tsy_tx": 1000 * flows[:, 1],
    })
    _write_csv(root, "tsyparty/data/interim/z1_holdings_wide.csv", {
        "date": dates, "fed": 1000 * np.cumsum(fed),
    })
    _write_csv(root, "tsyparty/data/interim/z1_holdings_long.csv", {
        "date": dates, "sector": "_total", "instrument": "all_treasuries",
        "holdings": 1000 * np.cumsum(total),
    })
    issuance_dir = "tsyparty/outputs/holder_response/issuance_maturity_response"
    _write_csv(root, f"{issuance_dir}/sector_transaction_groups.csv", {
        "date": np.repeat(dates, len(sector_names)),
        "sector": np.tile(sector_names, n), "transactions_bn": flows.ravel(),
    })
    _write_csv(root, f"{issuance_dir}/quarterly_issuance_maturity.csv", {
        "date": dates, "issuance_volume_bn": rng.uniform(500, 1000, n),
        "bill_share": rng.uniform(0.2, 0.7, n), "issuance_wam_years": rng.uniform(3, 8, n),
        "issuance_wam_gap_years": gap,
    })
    outcome_columns = [
        "total_deposits_bank_qoq", "bank_credit_private_qoq", "reserves_qoq",
        "on_rrp_reallocation_qoq", "mmf_treasury_bills_reallocation_qoq",
    ]
    _write_csv(root, "tdcpass/data/derived/quarterly_panel.csv", {
        "quarter": dates,
        **{column: 0.3 * tdc + rng.normal(0, 10, n) for column in outcome_columns},
    })
    _write_csv(root, "wamest/data/interim/z1_sector_panel_full.csv", {
        "date": dates[-8:], "sector_key": "all_holders_total", "transactions": total[-8:],
    })
    _write_csv(root, "wamest/data/interim/z1_series_panel_full.csv", {
        "date": np.tile(dates, 2),
        "series_key": np.repeat([
            "money_market_funds_treasuries", "asset_backed_securities_issuers_treasuries",
        ], n),
        "description": "Synthetic Treasury series", "level": np.tile(np.cumsum(fed), 2),
        "transactions": np.concatenate([flows[:, 2], flows[:, -1]]), "revaluation": 0.0,
    })
    config = yaml.safe_load((REPO_ROOT / "config/z1_exact_transaction_components.yml").read_text())
    fred_values = {
        component["fred_id"]: rng.uniform(1000, 12000, n)
        for component in config["components"]
    }
    component_total = sum(fred_values.values())
    for anchor in config["anchors"].values():
        fred_values[anchor["fred_id"]] = component_total

    def fetch_fred(series_id: str) -> pd.DataFrame:
        return pd.DataFrame({
            "observation_date": quarters.start_time.strftime("%Y-%m-%d"),
            series_id: fred_values[series_id],
        })

    return root, fetch_fred


@pytest.fixture(autouse=True)
def isolated_source_environment(request, tmp_path, monkeypatch):
    """Integration reads retained files; all other tests use isolated sources."""
    def reject_network(*args, **kwargs):
        raise AssertionError("Tests must not fetch source data over the network")

    monkeypatch.setattr(urllib.request, "urlopen", reject_network)
    if request.node.get_closest_marker("integration"):
        monkeypatch.chdir(REPO_ROOT)
        contracts = load_source_contracts("config/source_contracts.yml")
        required = {record.path for record in source_records(contracts) if record.path}
        required.update(source["source_path"] for source in Z1_CANDIDATE_SOURCES)
        missing = sorted(path for path in required if not Path(path).is_file())
        assert not missing, f"Integration requires configured sibling inputs: {missing}"
        return

    family, fetch_fred = request.getfixturevalue("synthetic_family")
    project = family / tmp_path.name
    project.mkdir()
    shutil.copytree(REPO_ROOT / "config", project / "config")
    monkeypatch.chdir(project)
    monkeypatch.setattr(z1_exact, "fetch_fred_series", fetch_fred)
