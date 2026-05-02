# TDCMix

Holder-mix and issuance-composition tooling for the Treasury Deposit Channel
(TDC).

TDCMix builds a quarterly holder-absorption panel that connects Treasury
issuance, sectoral Treasury absorption, and balance-sheet outcomes. The project
is designed to study whether the deposit and plumbing footprint of Treasury
financing differs when absorption tilts toward banks, the Federal Reserve,
foreign holders, money-market funds, domestic nonbanks, or intermediary sectors.

## Related Projects

- [`tdcest`](https://github.com/smkwray/tdcest): quarterly TDC estimates and accounting definitions.
- [`tdcpass`](https://github.com/smkwray/tdcpass): lower-frequency pass-through analysis.
- [`tdc-hf`](https://github.com/smkwray/tdc-hf): high-frequency TDC proxy and diagnostic branch.
- [`tsyparty`](https://github.com/smkwray/tsyparty): Treasury auction and holder-sector context.
- [`wamest`](https://github.com/smkwray/wamest): holder maturity and sector context.
- [`tdcsfc`](https://github.com/smkwray/tdcsfc): stock-flow-consistent model scaffold.

## Initial Scope

- define a TDC-relevant holder-sector taxonomy;
- build a quarterly holder-absorption panel from public sector and issuance data;
- validate sector mappings against existing TDC accounting components;
- estimate issuance-composition links to holder mix;
- estimate reduced-form holder-mix links to deposits, reserves, ON RRP, money
  funds, and bank credit.

## Setup

```bash
set -a
source .env
set +a
uv pip install -e '.[dev]'
```

Run tests:

```bash
"$UV_PROJECT_ENVIRONMENT/bin/python" -B -m pytest
```

Build MVP artifacts:

```bash
"$UV_PROJECT_ENVIRONMENT/bin/python" -B -m tdcmix write-panel
"$UV_PROJECT_ENVIRONMENT/bin/python" -B -m tdcmix write-z1-exact
"$UV_PROJECT_ENVIRONMENT/bin/python" -B -m tdcmix write-exact-primary-panel
"$UV_PROJECT_ENVIRONMENT/bin/python" -B -m tdcmix write-exact-primary-analysis
"$UV_PROJECT_ENVIRONMENT/bin/python" -B -m tdcmix write-audits
"$UV_PROJECT_ENVIRONMENT/bin/python" -B -m tdcmix write-validation
"$UV_PROJECT_ENVIRONMENT/bin/python" -B -m tdcmix write-design
"$UV_PROJECT_ENVIRONMENT/bin/python" -B -m tdcmix write-estimates
"$UV_PROJECT_ENVIRONMENT/bin/python" -B -m tdcmix write-z1-exact-analysis
"$UV_PROJECT_ENVIRONMENT/bin/python" -B -m tdcmix write-priors
"$UV_PROJECT_ENVIRONMENT/bin/python" -B -m tdcmix write-decision
"$UV_PROJECT_ENVIRONMENT/bin/python" -B -m tdcmix write-closeout
"$UV_PROJECT_ENVIRONMENT/bin/python" -B -m tdcmix write-downstream-contracts
"$UV_PROJECT_ENVIRONMENT/bin/python" -B -m tdcmix write-manifest
"$UV_PROJECT_ENVIRONMENT/bin/python" -B -m tdcmix write-figures
"$UV_PROJECT_ENVIRONMENT/bin/python" -B -m tdcmix write-estimate-figures
```

The audit step writes source-overlap, sector-map, raw Z.1 candidate, and
source-upgrade recommendation audits.
The validation step writes share, residual, scale, WAM no-lookahead, and
residual-gate diagnostics. The exact Z.1 step writes official FU transaction
component reconciliation outputs for the all-sector Treasury transaction
identity, an exact FU holder-panel sidecar, and a bucket-definition contract for
primary-panel migration review. It also writes an exact-FU bucket migration
policy and primary migration gate, allowing source-layer migration while keeping
causal, main empirical, and residual-sensitive headline use blocked. The
exact-primary panel step writes `holder_absorption_panel_exact_primary.csv` as
the primary-compatible exact-FU source-layer panel and leaves
`holder_absorption_panel.csv` as the legacy audit/comparison panel. The
exact-primary analysis step writes a separate validation/design/estimate
namespace under `data/processed/exact_primary_analysis/` and compares it against
the exact-FU sidecar analysis. The design step writes first-stage
diagnostics, h=0..4 design matrices, and stop-rule status tables under
`data/processed/`, including h=-4..-1 pretrend/placebo matrices. The estimate step writes
diagnostic-only LP summaries, a six-profile shortlist, fuller-specification
stability checks for that shortlist, stop-rule metadata on estimate tables, and
pretrend/placebo timing screens that keep causal and residual-sensitive claims explicitly
out of scope. The exact Z.1 analysis step reruns validation/design/estimate
artifacts from the exact FU holder sidecar under `data/processed/z1_exact_analysis/`
and writes source-layer comparison tables plus a source-layer promotion status
against the legacy source layer. It also writes an exact-FU appendix-candidate
triage table that keeps timing-clean reviewable profiles separate from profiles
held by placebo concerns, plus a mechanism appendix memo for the surviving
candidate with sample-concentration, regime-coverage, and coefficient-sensitivity
diagnostics. The exact Z.1 analysis step also writes a machine-readable
appendix source contract and final backend status packet that promote exact-FU
only for mechanism-appendix accounting while keeping causal, main empirical, and
residual-sensitive headline use blocked. The
priors step writes a conservative holder-absorption prior
table, downstream prior contract, and report for model calibration review. The
decision step writes a project-routing status report for main-slot versus
mechanism/appendix treatment. The closeout step freezes the exact-primary panel
and exact-primary analysis namespace as the default backend source for
mechanism/appendix consumers, while recording that main empirical and
residual-sensitive headline paths remain blocked. The downstream-contract step
combines the holder prior contract, exact-FU appendix source contract, and
downstream default-source decision into one consumer-facing bundle with
central-calibration, mechanism-appendix, main-empirical, residual-headline,
exact-FU bucket-definition, primary-migration, and default-source rows. The
manifest step writes a versioned artifact manifest and build-status summary for
downstream consumers.

## Repository Layout

```text
config/          holder-sector maps and variable contracts
data/raw/        local raw inputs, ignored
data/processed/  generated panels, ignored
output/          generated tables and figures, ignored
src/tdcmix/      Python package
tests/           regression tests
```
