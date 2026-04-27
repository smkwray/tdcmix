# TDCMix

Holder-mix and issuance-composition tooling for Treasury Deposit Component
(TDC) research.

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

## Repository Layout

```text
config/          holder-sector maps and variable contracts
data/raw/        local raw inputs, ignored
data/processed/  generated panels, ignored
output/          generated tables and figures, ignored
src/tdcmix/      Python package
tests/           regression tests
```
