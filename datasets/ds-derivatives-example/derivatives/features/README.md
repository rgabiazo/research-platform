# features derivatives

Checked-in canonical examples live here when a slice needs reusable tabular features.

- `project-pilot-tabular/toy_features.tsv` is the validated merged feature table.
- `project-pilot-tabular/sources/toy_core.tsv` and
  `project-pilot-tabular/sources/toy_measurements.tsv` preserve keyed-merge
  coverage.

All three files are deterministically generated from the same in-code
specification as the canonical
`datasets/ds-tabular-example/toy_observations.csv` input. Every identifier and
value is an algorithmic invention; no participant, patient, health,
demographic, or other human data and no external dataset were used.

```bash
python3 ops/scripts/generate_toy_tabular_fixtures.py
python3 ops/scripts/generate_toy_tabular_fixtures.py --check
```
