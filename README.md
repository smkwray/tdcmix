# TDCMix

TDCMix builds quarterly panels of who absorbs US Treasury debt, from Financial Accounts of the United States (Z.1) transaction flows and Treasury issuance data. It groups buyers into eight holder sectors, reconciles them against total Treasury transactions, and compares holder mix with bank deposits, reserves, money-market balances and the Treasury Deposit Component (TDC), the part of the change in deposits at US banks and credit unions that comes from Treasury payments and trades in Treasury securities ([tdcest](https://github.com/smkwray/tdcest)). Results are quarterly panels, descriptive local projections, sensitivity tables and charts.

Apart from the Z.1 download, the inputs are exports from companion repositories, which must be built first.

## Data

- Z.1 transaction series (`config/z1_exact_transaction_components.yml`): `write-z1-exact` downloads 20 quarterly unadjusted transaction series (Z.1 codes beginning `FU`) and two total-Treasury anchor series from Federal Reserve Economic Data (FRED) graph CSV endpoints, without an API key, and converts USD millions to USD billions.
- Companion exports read from sibling checkouts (`config/source_contracts.yml`): TDC and its components from tdcest; Z.1 holdings, sector transactions and issuance maturity from [tsyparty](https://github.com/smkwray/tsyparty); and deposits, bank credit, reserves, overnight reverse repurchase (ON RRP) balances and money market fund (MMF) series from [tdcpass](https://github.com/smkwray/tdcpass). Validation and audits also read [WAMEST](https://github.com/smkwray/wamest) sector panels when present.

`config/holder_sector_map.yml` defines the eight holder sectors: the Federal Reserve, banks and other depositories, the rest of the world, MMFs, domestic nonbanks, dealers and other short-run intermediaries, other financial institutions, and a residual for unmapped holders.

## Install

Python 3.11 or later. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

No `.env` file or API key is needed.

## Usage

Check that the configuration loads:

```bash
python -m tdcmix validate-contracts
```

With the companion exports in place, run the pipeline in order. Each step reads the previous step's files, and only `write-z1-exact` needs network access:

```bash
# 1. Holder panel from companion exports, with validation, design matrices and estimates
python -m tdcmix write-panel
python -m tdcmix write-validation
python -m tdcmix write-design
python -m tdcmix write-estimates

# 2. Z.1 transaction flows, reconciled and compared with the step 1 panel
python -m tdcmix write-z1-exact
python -m tdcmix write-z1-exact-analysis

# 3. The main panel on Z.1 transaction flows, its analysis, and source audits
python -m tdcmix write-exact-primary-panel
python -m tdcmix write-exact-primary-analysis
python -m tdcmix write-audits
```

`write-exact-primary-panel` stops unless `write-z1-exact` has marked the Z.1 panel ready. `write-figures` and `write-estimate-figures` draw the charts, and `python -m tdcmix --help` lists the remaining commands.

## Output

| File | Contents |
|---|---|
| `data/processed/holder_absorption_panel_exact_primary.csv` | Quarterly holder-absorption panel from Z.1 transaction flows, in USD billions, with each sector's share of positive absorption |
| `data/processed/z1_exact_residual_reconciliation.csv` | Quarterly reconciliation of Z.1 sector flows against total Treasury transactions, in USD billions |
| `data/processed/exact_primary_analysis/exact_primary_analysis_report.md` | Validation and diagnostics for the main panel |
| `output/holder_absorption_shares.png` | Quarterly holder-absorption shares |
| `output/lp_diagnostic_profiles.png` | Descriptive local-projection profiles |

Comparison panels, detailed Z.1 tables and diagnostic CSVs are written alongside them in `data/processed/`. `data/` and `output/` contents are ignored by Git.

## Limits

- The main panel uses unadjusted Z.1 transaction flows, which differ from changes in holdings because holdings also move with valuation.
- Sectors follow Z.1 definitions: depositories include US-chartered depositories, credit unions, foreign banking offices in the US and banks in US-affiliated areas; domestic nonbanks include the Z.1 household and nonprofit residual; dealers capture short-run warehousing; and other financial institutions group unlike nonbank lenders.
- Projections and local-projection profiles are descriptive associations. Pretrend, placebo, sample-concentration and specification checks report failed or unsupported results in the output tables.

## Tests

The default suite builds temporary synthetic inputs and runs offline:

```bash
python -B -m pytest -q
ruff check .
```

Integration tests check the companion datasets and fail if they are missing. They are excluded by default:

```bash
python -B -m pytest -q -m integration
```

## License

MIT. See [LICENSE](LICENSE).
