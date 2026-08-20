# research-io docs

`research-io` provides generic tabular reader and table operations with backend-aware behavior.

## Design

- `src/research_platform/io/readers/tabular.py` handles path and format orchestration.
- `src/research_platform/io/dataframe/ops.py` implements backend-agnostic operations.
- backend-specific behavior remains isolated in:
  - `src/research_platform/io/dataframe/polars_ops.py`
  - `src/research_platform/io/dataframe/pandas_ops.py`

## Command-line interface

Use the CLI for quick terminal previews, inspection, and minimal write-enabled cleaning:

```bash
research-io preview tests/data/tabular/people_a.csv --backend polars --format csv --head 5
research-io concat tests/data/tabular/people_a.csv tests/data/tabular/people_b.csv --backend polars --format csv --head 5
research-io merge tests/data/tabular/people_a.csv tests/data/tabular/people_meta.csv --on id --backend polars --format csv --head 5
research-io merge tests/data/tabular/people_a.csv tests/data/tabular/people_b.csv --how cross --backend polars --format csv --head 4
research-io merge tests/data/tabular/people_a.csv tests/data/tabular/people_meta.csv tests/data/tabular/people_job.csv --on id --backend polars --how left --format csv --head 10

research-io preview datasets/ds-bids-example/participants.tsv --format tsv --backend polars --head 50 --cols 9
research-io merge datasets/ds-bids-example/participants.tsv datasets/ds-bids-example/phenotype/toy_phenotype.tsv --backend polars --format tsv
research-io merge datasets/ds-bids-example/participants.tsv datasets/ds-bids-example/phenotype/toy_phenotype.tsv \
  --left-format tsv --right-format tsv --on participant_id --backend polars

research-io inspect dtypes tests/data/tabular/people_a.csv --format csv --backend polars
research-io inspect describe tests/data/tabular/people_cleaning.csv --format csv --backend polars
research-io inspect nulls tests/data/tabular/people_cleaning.csv --format csv --backend polars

research-io clean replace-values tests/data/tabular/people_cleaning.csv \
  --format csv \
  --backend polars \
  --columns score \
  --invalid-values 999999 \
  --output /tmp/people_cleaned.csv

research-io clean fill-missing tests/data/tabular/people_cleaning.csv \
  --format csv \
  --backend polars \
  --strategy median \
  --columns score \
  --output /tmp/people_cleaned.csv

research-io clean drop-rows tests/data/tabular/people_cleaning.csv \
  --format csv \
  --backend polars \
  --row-numbers 0 2 \
  --output /tmp/people_cleaned.csv

research-io clean fill-missing tests/data/tabular/people_cleaning.csv \
  --format csv \
  --backend polars \
  --strategy mode \
  --columns group \
  --output /tmp/people_cleaned.tsv
```

Common options:

- `--backend {polars,pandas}` (default: `polars`)
- `--head N`
- `--cols N`
- `--display-all`
- `--lazy`
- `--format {csv,tsv,txt,parquet,feather}`
- `--left-format {csv,tsv,txt,parquet,feather}` (override for left input)
- `--right-format {csv,tsv,txt,parquet,feather}` (override for right input)
- `--on`
- For 3+ inputs, use one shared `--on` and one shared `--how`.
- `--left-on`
- `--right-on`
- `--left-format` and `--right-format` apply only to two-input merges.
- `--how`
  - `cross` requires neither `--on` nor `--left-on`/`--right-on`

Cleaning options:

- `clean replace-values --columns <columns> --invalid-values <values>`
- `clean fill-missing --strategy {median,mode} --columns [columns]`
- `clean drop-rows --row-numbers [0-based positions]`
- `clean drop-rows --id-column <column> --id-values <values>`
- `clean` commands accept `--output <path>` to write transformed tables (`csv`, `tsv`, `parquet`, `feather`).
- `inspect dtypes|describe|nulls` prints per-input summaries.

For N-way merges, non-key column collisions are resolved by suffixing the imported column name with `_srcN` where `N`
is the source index in the merge input order.

Notes:

- `research-io` does not discover raw directory/BIDS layouts. Path and format resolution is intentionally table-oriented.
- For 3+ inputs, `--on` must be shared across all inputs.

Preview behavior notes:

- `--head N` and `--cols N` show exactly the first requested window, clipped to available rows/columns.
- If either window exceeds the table size, preview output includes a note and prints the full available window.
- `--display-all` forces full-row/column output and takes precedence over `--head`/`--cols`.
