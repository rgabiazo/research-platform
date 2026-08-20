# research-analysis

`research-analysis` provides leakage-safe tabular preparation, generic
statistics and association contracts, publication-table handoffs, and
protocol-neutral MVPA mathematics.

## Current status

This is an unreleased `0.1.0a1` source-checkout alpha for Python 3.11 and 3.12.
Its package CLI and Python modules are advanced interfaces beneath `rp`, not a
stable unified SDK.

The capability matrix currently identifies bounded synthetic tabular workflows
and one checked-in local materialized-pattern crossnobis workflow as runnable.
Other analysis, publication, imaging, and external-runtime claims remain limited
to their exact classifications in the
[capability matrix](../../docs/capabilities.md).

## Responsibilities

`research-analysis` owns:

- deterministic split manifests and random, binary-stratified, and
  binned-continuous split strategies;
- training-only fit and later application of numeric standardization;
- package-level classification and regression orchestration through
  `research-ml`;
- generic summary, correlation, linear-model, one-way ANOVA, and
  mixed-effects-ready summary helpers;
- bounded generic association calculation, QC, multiplicity, and supplied-result
  contracts;
- no-write publication-table handoffs;
- protocol-neutral pattern, cross-validation, distance, crossnobis, summary,
  and RDM-shaped data contracts.

It does not own:

- tabular storage backends or merge behaviour, which belong in `research-io`;
- estimators and metrics, which belong in `research-ml`;
- image loading, pattern extraction, or neuroimaging semantics;
- BIDS path and derivative naming;
- figure and report rendering;
- scheduler, transfer, or remote execution behaviour;
- project-specific scientific choices.

## Source-checkout use

Use the repository [quickstart](../../docs/onboarding/quickstart.md) to install
the coordinated workspace. `research-analysis` has no standalone console-script
entry point; its package CLI is invoked as:

```bash
python -m research_platform.analysis.cli --help
```

The base package depends on `research-ml`. The optional `rsatoolbox` extra adds
NumPy and `rsatoolbox>=0.3,<0.4` for the guarded crossnobis adapter. The native
reference distance implementation does not require that extra.

## Tested tabular example

The checked-in 24-row synthetic feature table supports a deterministic
binary-stratified split:

```bash
python -m research_platform.analysis.cli split create \
  --table datasets/ds-derivatives-example/derivatives/features/project-pilot-tabular/toy_features.tsv \
  --target-column binary_target \
  --output artifacts/package-doc-example/analysis/split.json
```

The package tests continue this exact table through training-only
standardization, application, logistic classification, and evaluation with an
explicit ordered predictor list. A separate test follows the continuous target
through binned stratification, standardization, ElasticNet training, and
evaluation. The primary integrated execution path remains the `rp` tabular
workflow described in the [quickstart](../../docs/onboarding/quickstart.md).

## Inputs and outputs

Tabular inputs are explicit rows or delimited tables, target columns, ordered
predictor columns, split manifests, and analysis settings. Outputs include
split and preprocessing manifests, transformed tables, model/evaluation
documents delegated through `research-ml`, statistics rows, QC and provenance
rows, and publication-table-compatible handoffs.

MVPA inputs are already prepared numeric pattern vectors with explicit
condition, grouping, and cross-validation labels and optional noise values.
Outputs are package-owned distance estimates, long-form distance rows,
summaries, QC, and provenance. The package does not extract vectors from images.

The checked-in runnable MVPA example is limited to synthetic, materialized
prepared vectors and local crossnobis. It is not evidence for real-data
analysis, image extraction, HPC execution, publication rendering, or a
supported RDM-export workflow.

## Evidence

Tabular and statistics behaviour is exercised by:

- [`test_splits.py`](tests/unit/test_splits.py);
- [`test_tabular_cli.py`](tests/unit/test_tabular_cli.py);
- [`test_tabular_input_integrity.py`](tests/unit/test_tabular_input_integrity.py);
- [`test_publication_tables.py`](tests/unit/test_publication_tables.py);
- the focused `test_tabular_association_*.py` modules under `tests/unit/`.

MVPA contracts are exercised by:

- [`test_mvpa_contracts.py`](tests/unit/test_mvpa_contracts.py);
- [`test_mvpa_native_reference.py`](tests/unit/test_mvpa_native_reference.py);
- [`test_mvpa_row_preparation.py`](tests/unit/test_mvpa_row_preparation.py);
- [`test_mvpa_prepared_distances.py`](tests/unit/test_mvpa_prepared_distances.py);
- [`test_mvpa_prepared_summary.py`](tests/unit/test_mvpa_prepared_summary.py);
- the bounded integrated toy workflow in
  [`test_toy_mvpa_project_cli.py`](../research-core/tests/integration/test_toy_mvpa_project_cli.py).

## Limitations

- Only the documented synthetic tabular and materialized-vector examples have
  their stated local support.
- Explicit predictor selection remains a scientific contract; identifiers,
  targets, alternate outcomes, and leakage-prone columns must not be inferred
  into a model.
- Publication handoffs do not render or publish reports.
- Repeated-measures helpers plan and inspect designs or normalize supplied
  results; they do not fit mixed models.
- Prepared-vector MVPA does not imply image-backed, real-data, HPC, or
  project-level RDM-export support.
- Package-level Python interfaces remain alpha and do not form a stable unified
  SDK.

See the [repository overview](../../README.md),
[architecture](../../ARCHITECTURE.md),
[capability matrix](../../docs/capabilities.md),
[tabular guide](../../docs/tabular-slice.md), and
[MVPA guide](../../docs/mvpa-crossnobis.md).
