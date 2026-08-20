# research-io examples

```bash
# Tabular inspection
research-io inspect dtypes tests/data/tabular/people_a.csv --format csv --backend polars
research-io inspect describe tests/data/tabular/people_cleaning.csv --format csv --backend polars
research-io inspect nulls tests/data/tabular/people_cleaning.csv --format csv --backend polars

# Cleaning examples with output
research-io clean replace-values tests/data/tabular/people_cleaning.csv \
  --format csv --backend polars \
  --columns score --invalid-values 999999 \
  --output /tmp/people_cleaned.csv

research-io clean fill-missing tests/data/tabular/people_cleaning.csv \
  --format csv --backend polars \
  --strategy median --columns score \
  --output /tmp/people_cleaned.csv

research-io clean fill-missing tests/data/tabular/people_cleaning.csv \
  --format csv --backend polars \
  --strategy mode --columns group \
  --output /tmp/people_cleaned.tsv

research-io clean drop-rows tests/data/tabular/people_cleaning.csv \
  --format csv --backend polars \
  --row-numbers 0 2 --output /tmp/people_cleaned.csv

research-io clean drop-rows tests/data/tabular/people_cleaning.csv \
  --format csv --backend polars \
  --id-column id --id-values 2 5 --output /tmp/people_cleaned.csv

# Preview one or more rows from BIDS-style subjects
research-io preview datasets/ds-bids-example/participants.tsv --format tsv --backend polars --head 50
research-io preview datasets/ds-bids-example/participants.tsv --format tsv --backend polars --head 50 --cols 9

# Merge synthetic subject-level participant IDs with one toy phenotype table using inferred first-column key
research-io merge datasets/ds-bids-example/participants.tsv datasets/ds-bids-example/phenotype/toy_phenotype.tsv \
  --how left --format tsv --backend polars --head 20

# Merge with explicit key names when join columns differ
research-io merge tests/data/tabular/people_a.csv tests/data/tabular/people_right.tsv \
  --left-on id --right-on key --how inner --left-format csv --right-format tsv --backend polars

# Merge three curated subject-level tables using one shared key
research-io merge tests/data/tabular/people_a.csv \
  tests/data/tabular/people_meta.csv \
  tests/data/tabular/people_job.csv \
  --on id --how left --format csv --backend polars --head 20

# Cross-join two tables intentionally (every row of left matched to every row of right)
research-io merge tests/data/tabular/people_a.csv tests/data/tabular/people_meta.csv \
  --how cross --format csv --backend polars --head 8

# Merge with an explicit join key into a full-window toy phenotype table
research-io merge datasets/ds-bids-example/participants.tsv datasets/ds-bids-example/phenotype/toy_phenotype.tsv \
  --on participant_id --how left --format tsv --backend polars --head 20

# Preview full non-BIDS table output
research-io preview datasets/ds-tabular-example/toy_observations.csv --format csv --backend polars --display-all

# Show all columns and rows in a merged preview
research-io merge datasets/ds-bids-example/participants.tsv \
  datasets/ds-bids-example/phenotype/toy_phenotype.tsv \
  --on participant_id \
  --how left \
  --format tsv \
  --backend polars \
  --display-all

# Preview a generic non-BIDS tabular input
research-io preview datasets/ds-tabular-example/toy_observations.csv --format csv --backend polars --head 20
```

```python
from research_platform.io import (
    merge_tabular,
    merge_tabulars,
    read_tabular,
    inspect_dtypes,
    inspect_describe,
    inspect_nulls,
    replace_invalid_values,
    fill_missing_with_median,
    fill_missing_with_mode,
    drop_rows_by_id_values,
    write_tabular,
)

participants = read_tabular("datasets/ds-bids-example/participants.tsv", format="tsv")
phenotype = read_tabular("datasets/ds-bids-example/phenotype/toy_phenotype.tsv", format="tsv")
joined = merge_tabular(participants, phenotype, on="participant_id", how="left")

left = read_tabular("tests/data/tabular/people_a.csv", format="csv")
middle = read_tabular("tests/data/tabular/people_meta.csv", format="csv")
right = read_tabular("tests/data/tabular/people_job.csv", format="csv")
merged = merge_tabulars([left, middle, right], on="id", how="left")

dtypes = inspect_dtypes(read_tabular("tests/data/tabular/people_cleaning.csv", format="csv"))
print(dtypes)

description = inspect_describe(read_tabular("tests/data/tabular/people_cleaning.csv", format="csv"))
print(description["score"])

nulls = inspect_nulls(read_tabular("tests/data/tabular/people_cleaning.csv", format="csv"))
print(nulls)

cleaned = replace_invalid_values(
    read_tabular("tests/data/tabular/people_cleaning.csv", format="csv"),
    columns=["score"],
    invalid_values=[999999],
)
cleaned = fill_missing_with_median(cleaned, columns=["score"])
cleaned = fill_missing_with_mode(cleaned, columns=["group"])
cleaned = drop_rows_by_id_values(cleaned, id_column="id", id_values=[2, 5])
write_tabular(cleaned, "/tmp/people_cleaned.csv")
```

Notes:

- Raw sourcedata is intentionally not used as the first merge demo.
- Raw files (for example, `sourcedata/phenotype_raw/study_export.csv` and files under `sourcedata/behavioural_task/<participant-id>/<session-id>/...csv`) often need normalization:
  - consistent `participant_id` values
  - consistent row grain (raw rows are often long-form; merge-ready phenotype tables are subject-level).
- `research-io` intentionally does not include raw directory or BIDS discovery logic; do that in `research-bids` or pipeline tooling.

Preview controls:

- `--head N` and `--cols N` are clipped to available rows/columns and render full requested windows without adding internal ellipses.
- If a requested window is larger than the table, the command prints a note and returns the full table section.
- `--display-all` overrides `--head`/`--cols` and prints all loaded rows and columns.
