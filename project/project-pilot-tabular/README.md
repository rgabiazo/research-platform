# project-pilot-tabular

> **Alpha status — Runnable locally.** This is the primary public-alpha
> walkthrough, using only deterministic synthetic tabular inputs. The verified
> contract is intentionally limited to the checked-in preprocessing, logistic
> classification, and evaluation path.

This thin synthetic overlay includes:

- one `toy_binary_logreg` batch consuming the canonical `toy_features.tsv`
  derivative;
- one preprocessing path: numeric standardization;
- one model path: logistic regression; and
- the documented local `rp` transaction.

The checked-in SLURM profile supports planning; it is not evidence of cluster
execution or remote support. Follow the
[source-checkout quickstart](../../docs/onboarding/quickstart.md) for the exact
local sequence and consult the [capability matrix](../../docs/capabilities.md)
for broader boundaries.

Canonical reusable tabular features live under `datasets/.../derivatives/features/`.
Run outputs stay under `artifacts/runs/`.

## Predictor contract

The `toy_binary_logreg` batch owns the canonical `feature_table` and selected
`target_column`. Its ordered predictor contract lives only in
`config/models.yaml` under `models.default.feature_columns`; public `rp`
preprocessing and training do not infer predictors. The checked-in list excludes
`record_id`, the selected `binary_target`, and the alternate outcome
`continuous_target`. Identifiers, targets, group variables, and other
leakage-prone columns must remain excluded. Predictor order is part of the
scientific contract, and invalid configuration is rejected before run output is
created.

The input and derivative fixtures are deterministic algorithmic inventions;
they contain no participant, patient, health, demographic, or other human data
and use no external dataset. Regenerate or verify them from the repository root:

```bash
python3 ops/scripts/generate_toy_tabular_fixtures.py
python3 ops/scripts/generate_toy_tabular_fixtures.py --check
```

## Privacy boundary

This is one of four checked-in public overlays only. Real-study configuration
must live in a separate private repository or another explicit private
boundary, outside the public `project/` tree. Do not weaken the root project
allowlist.
