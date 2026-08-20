# ds-tabular-example

Generic non-BIDS tabular example dataset.

Use this folder for table workflows that are not BIDS-style.

Canonical input:

- `toy_observations.csv` contains 24 deterministic, balanced toy records used by
  research-io documentation and CLI examples.

Every identifier and value is an algorithmic invention. No participant,
patient, health, demographic, or other human data were used, and no external
dataset was used. Regenerate the canonical input and its validated example
derivatives from the repository root with:

```bash
python3 ops/scripts/generate_toy_tabular_fixtures.py
python3 ops/scripts/generate_toy_tabular_fixtures.py --check
```
