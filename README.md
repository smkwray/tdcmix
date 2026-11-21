# TDCMix

TDCMix builds quarterly US Treasury holder-absorption panels from Financial Accounts of the United States (Z.1) transaction series and Treasury issuance data. It reconciles sector buyer groups against total Treasury transactions and writes quarterly panels, descriptive local projections, sensitivity tables, and charts relating holder composition to the Treasury Deposit Component (TDC), deposits, reserves, and money-market balances.

## Data

- Public Z.1 transaction series (`config/z1_exact_transaction_components.yml`): `write-z1-exact` downloads 20 quarterly unadjusted Z.1 Treasury transaction series (`FU` codes) and two Z.1 total-Treasury anchor series from FRED graph CSV endpoints (`fred.stlouisfed.org/graph/fredgraph.csv`), scaling USD millions to USD billions. No FRED API key is required; network access is needed for `write-z1-exact`.
- Local sibling CSV inputs (`config/source_contracts.yml`): quarterly measured TDC and components (`../tdcest/data/processed/`), issuance maturity metrics and grouped sector flows (`../tsyparty/outputs/` and `../tsyparty/data/interim/`), and quarterly deposit, bank-credit, reserve, overnight reverse repo (ON RRP), and money-market-fund (MMF) series (`../tdcpass/data/derived/quarterly_panel.csv`). Validation and audit commands also read WAMEST's Z.1 sector panels under `../wamest/data/interim/` when present.

Treasury buyers are grouped into eight holder sectors in `config/holder_sector_map.yml`: the Federal Reserve, depository institutions, rest of the world, money market funds, domestic nonbanks, broker-dealers, other financial institutions, and unmapped holders. Local raw data and generated files are gitignored.

## Install

Requires Python 3.11 or newer. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

No `.env` file or API key is needed.

## Usage

Check that the YAML input definitions load cleanly:

```bash
python -m tdcmix validate-contracts
```

With the local inputs declared in `config/source_contracts.yml` in place, run the build pipeline (`write-z1-exact` requires network access; all other commands use local files):

```bash
python -m tdcmix write-panel
python -m tdcmix write-validation
python -m tdcmix write-design
python -m tdcmix write-estimates
python -m tdcmix write-z1-exact
python -m tdcmix write-z1-exact-analysis
python -m tdcmix write-exact-primary-panel
python -m tdcmix write-exact-primary-analysis
python -m tdcmix write-audits
```

Additional subcommands build charts (`write-figures`, `write-estimate-figures`), holder-absorption priors (`write-priors`), summary status tables, manifests, and empty schema templates. Run `python -m tdcmix --help` or `python -m tdcmix <subcommand> --help` to inspect all subcommands and path flags. Summary subcommands expect their upstream outputs to exist already and do not fetch missing data.

## Output

Main files written by the pipeline:

| File | Description |
|---|---|
| `data/processed/holder_absorption_panel_exact_primary.csv` | Quarterly Treasury holder-absorption panel from Z.1 transaction flows, in USD billions, with positive-absorption shares by sector |
| `data/processed/z1_exact_residual_reconciliation.csv` | Quarterly reconciliation of Z.1 sector flows against total Treasury transactions, in USD billions |
| `data/processed/exact_primary_analysis/exact_primary_analysis_report.md` | Validation and diagnostic summary report for the quarterly transaction panel |
| `output/holder_absorption_shares.png` | Chart of quarterly holder-absorption shares |
| `output/lp_diagnostic_profiles.png` | Chart of descriptive local-projection profiles |

Companion source-comparison panels, detailed Z.1 transaction tables, and diagnostic CSVs are written alongside them under `data/processed/`.

## Limits

- The main panel uses Z.1 unadjusted transaction flows (`FU` series), which differ from changes in holdings levels because levels include valuation shifts.
- Sector scopes follow Z.1 accounting definitions: depository institutions span US-chartered depositories, credit unions, foreign banking offices in the US, and banks in US-affiliated areas; domestic nonbanks include the Z.1 household and nonprofit residual; broker-dealers capture short-run warehousing; and other financial institutions group heterogeneous nonbank lenders.
- Outcome projections and local-projection profiles are descriptive associations. Pretrend, placebo, sample-concentration, and specification checks keep failed or unsupported results visible in the output tables.

## Tests

The default suite creates temporary synthetic inputs and runs offline:

```bash
python -B -m pytest -q
ruff check .
```

When the configured local sibling datasets are present, run the source-inventory integration tests explicitly (they are excluded by default and fail if inputs are missing):

```bash
python -B -m pytest -q -m integration
```
