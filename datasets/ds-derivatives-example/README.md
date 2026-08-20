
# ds-derivatives-example

Example dataset directory showing canonical reusable derivatives separate from pipeline run artifacts.

The tabular files under
`derivatives/features/project-pilot-tabular/` and the canonical
`datasets/ds-tabular-example/toy_observations.csv` input are generated alongside
one another from the same deterministic in-code specification.
Every identifier and value is an algorithmic invention. No participant,
patient, health, demographic, or other human data were used, and no external
dataset was used.

```bash
python3 ops/scripts/generate_toy_tabular_fixtures.py
python3 ops/scripts/generate_toy_tabular_fixtures.py --check
```
