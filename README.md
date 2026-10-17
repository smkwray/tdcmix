# TDCMix

TDCMix builds quarterly Treasury holder-absorption panels from public Financial
Accounts transaction series and issuance data. It relates holder composition to
Treasury financing, deposits, reserves, and money-market balances as descriptive
accounting evidence for the Treasury Deposit Component (TDC).

The primary panel uses Federal Reserve Z.1 unadjusted transaction series (FU
series) for holder absorption. Stock changes are not interchangeable with these
transactions. The code reconciles sector buckets, reports residual coverage, and
keeps source comparisons separate from the primary panel.

## Install and validate

Python 3.11 or later is required. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -B -m pytest -q
ruff check .
python -m tdcmix validate-contracts
```

No private environment file is required for installation or synthetic tests.
The default suite creates temporary synthetic source files and runs offline.
With the configured sibling datasets present, run `python -B -m pytest -q -m integration`
for the retained source-inventory checks. Those checks fail if required inputs
are missing; they are excluded from the default suite.

## Inputs

`config/source_contracts.yml` defines local input paths, frequencies, units, and
transformations. `config/holder_sector_map.yml` defines the holder buckets, and
`config/z1_exact_transaction_components.yml` defines transaction reconciliation.
Supply the declared inputs before building empirical outputs.

Related projects provide distinct inputs and comparisons:

- [TDCest](https://github.com/smkwray/tdcest): measured quarterly TDC and components.
- [TsyParty](https://github.com/smkwray/tsyparty): auction and holder-sector context.
- [WAMEST](https://github.com/smkwray/wamest): holder-sector maturity estimates.
- [TDCpass](https://github.com/smkwray/tdcpass): arithmetic and measurement-boundary comparisons.
- [TDC-HF](https://github.com/smkwray/tdc-hf): timing diagnostics.

## Build and inspect

With the configured source files present:

`write-z1-exact` retrieves the configured public FRED transaction series and
requires network access. The remaining commands below use local inputs.

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

`data/processed/holder_absorption_panel_exact_primary.csv` is the primary
transaction-based panel. Its analysis outputs have their own
`data/processed/exact_primary_analysis/` namespace.
`holder_absorption_panel.csv` and `z1_exact_analysis/` support source and
reconciliation comparisons; they should not silently replace the primary panel.

Additional commands produce validation tables, local projections, sensitivity
checks, holder-absorption assumptions, and downstream contracts. Use
`python -m tdcmix --help` and each subcommand's `--help` for required inputs and
output locations. Commands that summarize a build require their component
artifacts; they do not acquire missing source data.

## Interpretation

The usable role is holder-composition accounting and bounded mechanism context.
Outcome projections are diagnostic associations. Source reconciliation does not
establish exogenous variation, causal effects, stable central calibration, or
reliable residual-sensitive estimates. Pretrend, placebo, sample-concentration,
and specification checks retain failed or unsupported results explicitly.

A clean software build is distinct from a supported empirical claim. Read the
validation and downstream-use contracts attached to each generated result before
using it in another analysis or scenario.

## Repository layout

- `src/tdcmix/`: panel construction, reconciliation, diagnostics, and CLI.
- `config/`: source contracts and holder-sector definitions.
- `tests/`: offline synthetic tests and explicitly selected source-inventory checks.
- `data/raw/` and `data/processed/`: local inputs and generated panels; ignored.
- `output/`: generated reports and figures; ignored.

The repository does not ship source datasets or empirical output bundles.
