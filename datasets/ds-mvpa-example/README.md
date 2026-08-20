# Deterministic toy materialized MVPA patterns

This dataset is a tiny public input for verifying generic local
materialized-pattern crossnobis mechanics. The table contains ROI-final
prepared vectors; it does not contain images awaiting masking. It is independent
of the generic-NIfTI inputs in `datasets/ds-roi-example`.

Every identifier, feature, variance, and metadata value is an algorithmic
invention. No participant, patient, clinical, demographic, imaging, or other
human data were used, and no external dataset was used. Runtime products are
ephemeral outputs and belong under the ignored artifacts root, not in this
canonical synthetic input dataset.

## Design and formulas

The table uses schema
`research_platform.neuro.mvpa.materialized_pattern_table.v1`. It crosses two
invented subjects with two actual runs, two conditions, and two ROIs. There is
no session dimension. Each of the resulting 16 rows represents one exact unit
times one condition times one ROI and contains five features.

See the normative
[Materialized Pattern Table v1 producer contract](../../docs/materialized-pattern-table-v1.md)
for all required and optional columns, exact-unit joining, noise semantics,
and portability rules. The checked-in evidence also includes the
[exact-unit batch](../../project/project-example/manifests/batches/toy_mvpa_units.tsv),
[analysis bundle](../../project/project-example/config/analysis/bundles/toy-crossnobis.yaml),
and [MVPA configuration](../../project/project-example/config/analysis/mvpa/toy-crossnobis.yaml).

For one-based subject index `s`, run index `r`, ROI index `q`, and feature index
`f`, and condition indicator `c` (`0` for `condition_a`, `1` for
`condition_b`), the prepared value is:

```text
10*s + 3*r + 2*q + f/4 + c*(((s + q)*f + r)/2)
```

The diagonal noise variance is:

```text
1 + s/4 + q/2 + r/4 + f/8
```

Noise variances are positive and nonuniform. They are identical across the two
conditions for the same exact unit, ROI, and run-based cross-validation
partition. Feature order is `feature_index_ascending`. Each ROI has an
independently generated SHA-256 feature-index identity and explicit feature
space and ROI-definition identities.

### SeedA

`SeedA` uses feature-space identity `toy-feature-space:SeedA:v1` and ROI
definition identity `toy-roi-definition:SeedA:v1`.

### SeedB

`SeedB` uses feature-space identity `toy-feature-space:SeedB:v1` and ROI
definition identity `toy-roi-definition:SeedB:v1`.

## Regeneration and verification

From the repository root, regenerate the canonical files with:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 ops/scripts/generate_toy_mvpa_fixtures.py
```

Verify them byte-for-byte without writing with:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 ops/scripts/generate_toy_mvpa_fixtures.py --check
```
