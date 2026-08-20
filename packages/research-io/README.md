# research-io

`research-io` provides backend-neutral tabular reading, writing, preview,
inspection, merge, concatenation, and generic cleaning operations.

## Current status

This is an unreleased `0.1.0a1` source-checkout alpha for Python 3.11 and 3.12.
The `research-io` command and package-level Python interfaces are advanced
interfaces beneath the primary `rp` workflow. They are not a stable unified
SDK.

The checked-in synthetic preview and keyed-merge examples are
**Runnable locally** as bounded by the
[capability matrix](../../docs/capabilities.md).

## Responsibilities

`research-io` owns:

- table-oriented path and format handling;
- Polars and pandas backend adapters;
- table preview and inspection;
- concatenation and explicit keyed merge;
- generic missing-value and row-cleaning helpers;
- conversion of dataframe-like values to copied record rows;
- tabular output writing to caller-selected paths.

It does not own:

- BIDS layout discovery or neuroimaging behaviour;
- project-specific raw-data normalization;
- train/test splits or leakage-safe preprocessing;
- statistical analysis, model fitting, or scientific interpretation;
- canonical dataset promotion or run-transaction policy.

## Backends and formats

Polars `1.37.1` is the default and required backend. Pandas support is optional
and uses the package's `pandas` extra, which also installs PyArrow.

Supported table formats are:

- CSV;
- TSV and delimiter-compatible text;
- Parquet;
- Feather.

Callers choose the backend explicitly when behaviour depends on it. The
package's [contribution rules](AGENTS.md) confine backend coupling to
`dataframe/polars_ops.py` and `dataframe/pandas_ops.py`. The current preview
formatter also lazy-imports Polars; that known implementation discrepancy is
not a public compatibility contract.

Two-input merges accept a shared key or explicit left and right keys. Merges of
three or more tables require one shared key and join mode. Non-key name
collisions are resolved deterministically with source-position suffixes.
Pandas-index-based joins are not supported.

## Source-checkout use

Create the coordinated environment with the repository
[quickstart](../../docs/onboarding/quickstart.md). The package is not currently
published on PyPI.

The installed package command is:

```bash
research-io --help
```

## Tested synthetic example

Preview the checked-in backend-neutral table:

```bash
research-io preview \
  datasets/ds-tabular-example/toy_observations.csv \
  --format csv \
  --backend polars \
  --head 4
```

Materialize the two checked-in source tables into a noncanonical artifact:

```bash
research-io merge \
  datasets/ds-derivatives-example/derivatives/features/project-pilot-tabular/sources/toy_core.tsv \
  datasets/ds-derivatives-example/derivatives/features/project-pilot-tabular/sources/toy_measurements.tsv \
  --on record_id \
  --format tsv \
  --backend polars \
  --output artifacts/package-doc-example/io/toy_features.tsv
```

For these synthetic inputs, the merge produces 24 rows with deterministic
column order and exact bytes matching the checked-in `toy_features.tsv`
reference. That evidence does not establish arbitrary-schema normalization or
scientific validity.

## Inputs and outputs

Inputs are caller-selected table paths or supported in-memory dataframe values,
together with explicit format, backend, merge, inspection, or cleaning options.

Outputs are backend dataframe values, copied record rows, terminal previews or
inspection documents, or a table written to the caller-selected path. A write
does not make an output canonical; generated work belongs under `artifacts/`
unless a separate publication policy says otherwise.

The generic record adapters perform no writes. Real pandas and Polars objects
must use their explicit adapters rather than relying on hidden auto-detection.

## Evidence

The bounded public examples and backend rules are exercised by:

- [`test_cli_merge_output.py`](tests/unit/test_cli_merge_output.py);
- [`test_string_columns.py`](tests/unit/test_string_columns.py);
- [`test_dataframe_records.py`](tests/unit/test_dataframe_records.py);
- [`test_pandas_records.py`](tests/unit/test_pandas_records.py);
- [`test_polars_records.py`](tests/unit/test_polars_records.py);
- [`test_tabular_association_source_rows.py`](tests/unit/test_tabular_association_source_rows.py).

## Limitations

- `research-io` is table-oriented and does not discover raw or BIDS directory
  structures.
- Merge keys, row grain, schema compatibility, and scientific meaning remain
  caller responsibilities.
- IO helpers do not create train/test isolation or prevent leakage.
- Writes do not provide the local transaction guarantees owned by `rp`.
- Package-level Python compatibility is alpha and no unified SDK is claimed.

See the [repository overview](../../README.md),
[architecture](../../ARCHITECTURE.md),
[capability matrix](../../docs/capabilities.md), and
[tabular guide](../../docs/tabular-slice.md).
