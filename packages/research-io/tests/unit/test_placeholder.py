from __future__ import annotations

import csv
from pathlib import Path
import os
import subprocess
import sys

import pytest

from research_platform.io import (
    collect_tabular,
    concat_tabular,
    drop_rows_by_id_values,
    drop_rows_by_row_numbers,
    fill_missing_with_median,
    fill_missing_with_mode,
    merge_tabular,
    merge_tabulars,
    inspect_columns_with_nulls,
    inspect_describe,
    inspect_dtypes,
    inspect_nulls,
    preview_tabular,
    read_tabular,
    read_tabulars,
    replace_invalid_values,
    write_tabular,
)
from research_platform.io.dataframe.adapters import supports_backend
from research_platform.io.dataframe._backend_protocol import LazyBackendUnsupportedError, UnsupportedBackendError
from research_platform.io.dataframe._paths import infer_format, ensure_same_format

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PACKAGE_ROOT / "tests" / "data" / "tabular"


def _table_row_count(table) -> int:
    if hasattr(table, "height"):
        return table.height
    return table.shape[0]


def _write_preview_csv(path: Path, row_count: int, col_count: int) -> None:
    header = [f"col_{i}" for i in range(col_count)]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        for row in range(row_count):
            values = [f"row_{row}"] + [str(row * 100 + col) for col in range(1, col_count)]
            writer.writerow(values)


def _build_preview_table(backend: str, row_count: int, col_count: int):
    if backend == "polars":
        import polars as pl  # pragma: no cover
        data = {"col_0": [f"row_{row}" for row in range(row_count)]}
        for col in range(1, col_count):
            data[f"col_{col}"] = [row * 100 + col for row in range(row_count)]
        return pl.DataFrame(data)

    import pandas as pd  # pragma: no cover

    data = {"col_0": [f"row_{row}" for row in range(row_count)]}
    for col in range(1, col_count):
        data[f"col_{col}"] = [row * 100 + col for row in range(row_count)]
    return pd.DataFrame(data)


def _run_cli(args: list[str]) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PACKAGE_ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, "-m", "research_platform.io.cli", *args],
        cwd=PACKAGE_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )


def test_infer_format_from_extension():
    assert infer_format("people.csv") == "csv"


def test_infer_format_supported_aliases():
    assert infer_format("one.tsv") == "tsv"
    assert infer_format("one.txt") == "txt"
    assert infer_format("one.parquet") == "parquet"
    assert infer_format("one.feather") == "feather"


def test_infer_format_all_supported_extensions():
    assert infer_format(Path("/tmp/any.csv")) == "csv"
    assert infer_format(Path("/tmp/any.tsv")) == "tsv"
    assert infer_format(Path("/tmp/any.txt")) == "txt"
    assert infer_format(Path("/tmp/any.parquet")) == "parquet"
    assert infer_format(Path("/tmp/any.feather")) == "feather"


def test_infer_format_mixed_without_explicit_format_raises():
    paths = [DATA_DIR / "people_a.csv", DATA_DIR / "people.tsv"]
    with pytest.raises(ValueError, match="Multiple file formats"):
        ensure_same_format(paths, None)


def test_read_single_csv_file_with_polars_backend_default():
    if not supports_backend("polars"):
        pytest.skip("Polars backend is not installed")

    table = read_tabular(DATA_DIR / "people_a.csv", backend="polars", format="csv")
    import polars as pl  # pragma: no cover

    assert isinstance(table, pl.DataFrame)
    assert table.height == 2
    assert list(table.columns) == ["id", "name"]


def test_glob_expansion_and_multifile_concat():
    if not supports_backend("polars"):
        pytest.skip("Polars backend is not installed")

    table = read_tabular(str(DATA_DIR / "people_[ab].csv"), backend="polars", format="csv")
    import polars as pl  # pragma: no cover

    assert isinstance(table, pl.DataFrame)
    assert table.height == 4
    assert table.select("name").to_series().to_list() == ["Ada", "Grace", "Linus", "Ken"]


def test_read_tabulars_multiple_explicit_paths():
    if not supports_backend("polars"):
        pytest.skip("Polars backend is not installed")

    tables = read_tabulars([DATA_DIR / "people_a.csv", DATA_DIR / "people_b.csv"], backend="polars", format="csv")
    import polars as pl  # pragma: no cover

    assert isinstance(tables, list)
    assert len(tables) == 2
    assert all(isinstance(table, pl.DataFrame) for table in tables)
    assert [table.height for table in tables] == [2, 2]


def test_default_backend_is_polars():
    if not supports_backend("polars"):
        pytest.skip("Polars backend is not installed")

    table = read_tabular(DATA_DIR / "people_a.csv", format="csv")
    import polars as pl  # pragma: no cover

    assert isinstance(table, pl.DataFrame)


def test_lazy_polars_read_and_collect():
    if not supports_backend("polars"):
        pytest.skip("Polars backend is not installed")

    table = read_tabular(DATA_DIR / "people_a.csv", format="csv", lazy=True)
    import polars as pl  # pragma: no branch

    assert isinstance(table, pl.LazyFrame)
    assert isinstance(collect_tabular(table), pl.DataFrame)


def test_concat_tabular_merges_tables():
    if not supports_backend("polars"):
        pytest.skip("Polars backend is not installed")

    left = read_tabular(DATA_DIR / "people_a.csv", backend="polars", format="csv")
    right = read_tabular(DATA_DIR / "people_b.csv", backend="polars", format="csv")
    merged = concat_tabular([left, right], backend="polars")

    assert _table_row_count(merged) == 4
    assert "name" in merged.columns  # type: ignore[attr-defined]


def test_merge_tabular_on_key():
    if not supports_backend("polars"):
        pytest.skip("Polars backend is not installed")

    left = read_tabular(DATA_DIR / "people_a.csv", backend="polars", format="csv")
    right = read_tabular(DATA_DIR / "people_meta.csv", backend="polars", format="csv")
    merged = merge_tabular(left, right, on="id", how="left", backend="polars")

    assert _table_row_count(merged) == 2
    assert "group" in merged.columns  # type: ignore[attr-defined]


def test_merge_tabular_left_right_on_tsv_key():
    if not supports_backend("polars"):
        pytest.skip("Polars backend is not installed")

    left = read_tabular(DATA_DIR / "people_a.csv", backend="polars", format="csv")
    right = read_tabular(DATA_DIR / "people_right.tsv", backend="polars", format="tsv")
    merged = merge_tabular(
        left,
        right,
        left_on="id",
        right_on="key",
        how="inner",
        backend="polars",
    )

    assert _table_row_count(merged) == 2
    assert "role" in merged.columns  # type: ignore[attr-defined]


def test_merge_tabular_rejects_invalid_key_arguments():
    if not supports_backend("polars"):
        pytest.skip("Polars backend is not installed")

    left = read_tabular(DATA_DIR / "people_a.csv", backend="polars", format="csv")
    right = read_tabular(DATA_DIR / "people_meta.csv", backend="polars", format="csv")

    with pytest.raises(ValueError, match="Cannot provide --on and --left-on/--right-on together."):
        merge_tabular(left, right, on="id", left_on="id", right_on="id", backend="polars")

    with pytest.raises(ValueError, match="Merge requires either --on or both --left-on and --right-on."):
        merge_tabular(left, right, backend="polars")


@pytest.mark.parametrize("backend", ["polars", "pandas"])
def test_merge_tabulars_three_way_on_key(backend: str):
    if not supports_backend(backend):
        pytest.skip(f"{backend} backend is not installed")

    tables = [
        read_tabular(DATA_DIR / "people_a.csv", backend=backend, format="csv"),
        read_tabular(DATA_DIR / "people_meta.csv", backend=backend, format="csv"),
        read_tabular(DATA_DIR / "people_job.csv", backend=backend, format="csv"),
    ]
    merged = merge_tabulars(tables, on="id", how="inner", backend=backend)

    assert _table_row_count(merged) == 2
    assert "id" in merged.columns  # type: ignore[attr-defined]
    assert "name" in merged.columns  # type: ignore[attr-defined]
    assert "group" in merged.columns  # type: ignore[attr-defined]
    assert "job" in merged.columns  # type: ignore[attr-defined]


@pytest.mark.parametrize("backend", ["polars", "pandas"])
def test_merge_tabulars_four_way_on_key(backend: str):
    if not supports_backend(backend):
        pytest.skip(f"{backend} backend is not installed")

    tables = [
        read_tabular(DATA_DIR / "people_a.csv", backend=backend, format="csv"),
        read_tabular(DATA_DIR / "people_meta.csv", backend=backend, format="csv"),
        read_tabular(DATA_DIR / "people_job.csv", backend=backend, format="csv"),
        read_tabular(DATA_DIR / "people_department.csv", backend=backend, format="csv"),
    ]
    merged = merge_tabulars(tables, on="id", how="inner", backend=backend)

    assert _table_row_count(merged) == 2
    assert "id" in merged.columns  # type: ignore[attr-defined]
    assert "department" in merged.columns  # type: ignore[attr-defined]


def test_merge_tabulars_suffixes_non_key_collisions():
    if not supports_backend("polars"):
        pytest.skip("Polars backend is not installed")

    tables = [
        read_tabular(DATA_DIR / "people_a.csv", backend="polars", format="csv"),
        read_tabular(DATA_DIR / "people_meta.csv", backend="polars", format="csv"),
        read_tabular(DATA_DIR / "people_name_collision.csv", backend="polars", format="csv"),
    ]
    merged = merge_tabulars(tables, on="id", how="inner", backend="polars")

    assert "name" in merged.columns  # type: ignore[attr-defined]


def _table_column_values(table, column: str):
    if hasattr(table, "get_column"):
        return table.get_column(column).to_list()
    return list(table[column].tolist())


@pytest.mark.parametrize("backend", ["polars", "pandas"])
def test_inspect_dtypes_for_cleaning_fixture(backend: str):
    if not supports_backend(backend):
        pytest.skip(f"{backend} backend is not installed")
    table = read_tabular(DATA_DIR / "people_cleaning.csv", backend=backend, format="csv")
    dtypes = inspect_dtypes(table, backend=backend)
    assert set(dtypes.keys()) == {"id", "age", "score", "group"}
    assert all(isinstance(dtype, str) and dtype for dtype in dtypes.values())


@pytest.mark.parametrize("backend", ["polars", "pandas"])
def test_inspect_describe_for_cleaning_fixture(backend: str):
    if not supports_backend(backend):
        pytest.skip(f"{backend} backend is not installed")
    table = read_tabular(DATA_DIR / "people_cleaning.csv", backend=backend, format="csv")
    description = inspect_describe(table, backend=backend)

    assert description.keys() >= {"id", "age", "score", "group"}
    assert isinstance(description["id"], dict)
    assert description["score"]


@pytest.mark.parametrize("backend", ["polars", "pandas"])
def test_inspect_nulls_for_cleaning_fixture(backend: str):
    if not supports_backend(backend):
        pytest.skip(f"{backend} backend is not installed")
    table = read_tabular(DATA_DIR / "people_cleaning.csv", backend=backend, format="csv")
    null_summary = inspect_nulls(table, backend=backend)
    null_columns = inspect_columns_with_nulls(table, backend=backend)

    assert null_summary["age"] == 1
    assert null_summary["score"] == 1
    assert null_summary["group"] == 1
    assert set(null_columns) == {"age", "score", "group"}


@pytest.mark.parametrize("backend", ["polars", "pandas"])
def test_replace_invalid_values_marks_as_null(backend: str):
    if not supports_backend(backend):
        pytest.skip(f"{backend} backend is not installed")
    table = read_tabular(DATA_DIR / "people_cleaning.csv", backend=backend, format="csv")
    cleaned = replace_invalid_values(
        table,
        columns=["score"],
        invalid_values=[999999],
        backend=backend,
    )
    null_summary = inspect_nulls(cleaned, backend=backend)
    assert null_summary["score"] == 2


@pytest.mark.parametrize("backend", ["polars", "pandas"])
def test_fill_missing_with_median_fills_numeric(backend: str):
    if not supports_backend(backend):
        pytest.skip(f"{backend} backend is not installed")
    table = read_tabular(DATA_DIR / "people_cleaning.csv", backend=backend, format="csv")
    replaced = replace_invalid_values(table, columns=["score"], invalid_values=[999999], backend=backend)
    filled = fill_missing_with_median(
        replaced,
        columns=["score"],
        backend=backend,
    )
    null_summary = inspect_nulls(filled, backend=backend)

    assert null_summary["score"] == 0
    assert _table_column_values(filled, "score")[0] == 30
    assert _table_column_values(filled, "score")[2] == 30


@pytest.mark.parametrize("backend", ["polars", "pandas"])
def test_fill_missing_with_mode_fills_categorical(backend: str):
    if not supports_backend(backend):
        pytest.skip(f"{backend} backend is not installed")
    table = read_tabular(DATA_DIR / "people_cleaning.csv", backend=backend, format="csv")
    filled = fill_missing_with_mode(table, columns=["group"], backend=backend)
    null_summary = inspect_nulls(filled, backend=backend)
    assert null_summary["group"] == 0
    assert "blue" in _table_column_values(filled, "group")


@pytest.mark.parametrize("backend", ["polars", "pandas"])
def test_drop_rows_by_row_numbers(backend: str):
    if not supports_backend(backend):
        pytest.skip(f"{backend} backend is not installed")
    table = read_tabular(DATA_DIR / "people_cleaning.csv", backend=backend, format="csv")
    reduced = drop_rows_by_row_numbers(table, row_numbers=[0, 2], backend=backend)
    assert _table_row_count(reduced) == 3
    assert _table_column_values(reduced, "id") == [2, 4, 5]


@pytest.mark.parametrize("backend", ["polars", "pandas"])
def test_drop_rows_by_id_values(backend: str):
    if not supports_backend(backend):
        pytest.skip(f"{backend} backend is not installed")
    table = read_tabular(DATA_DIR / "people_cleaning.csv", backend=backend, format="csv")
    reduced = drop_rows_by_id_values(table, id_column="id", id_values=[2, 5], backend=backend)
    assert _table_row_count(reduced) == 3
    assert _table_column_values(reduced, "id") == [1, 3, 4]


@pytest.mark.parametrize("backend", ["polars", "pandas"])
@pytest.mark.parametrize("format_", ["csv", "tsv"])
def test_write_transformed_output_to_temp_file(backend: str, format_: str, tmp_path):
    if not supports_backend(backend):
        pytest.skip(f"{backend} backend is not installed")
    table = read_tabular(DATA_DIR / "people_cleaning.csv", backend=backend, format="csv")
    transformed = fill_missing_with_median(
        replace_invalid_values(table, columns=["score"], invalid_values=[999999], backend=backend),
        columns=["score"],
        backend=backend,
    )
    output = tmp_path / f"cleaned.{format_}"
    write_tabular(transformed, output, backend=backend)
    reloaded = read_tabular(output, backend=backend, format=format_)
    assert _table_row_count(reloaded) == 5


def test_merge_tabulars_more_than_two_with_left_right_keys_is_rejected():
    if not supports_backend("polars"):
        pytest.skip("Polars backend is not installed")

    args = [
        "merge",
        "tests/data/tabular/people_a.csv",
        "tests/data/tabular/people_meta.csv",
        "tests/data/tabular/people_job.csv",
        "--left-on",
        "id",
        "--right-on",
        "id",
        "--on",
        "id",
        "--format",
        "csv",
        "--backend",
        "polars",
    ]
    result = _run_cli(args)

    assert result.returncode == 1
    assert "--left-on/--right-on are only supported when merging exactly two inputs." in result.stderr


def test_preview_output_is_terminal_friendly():
    if not supports_backend("polars"):
        pytest.skip("Polars backend is not installed")

    table = read_tabular(DATA_DIR / "people_a.csv", format="csv")
    preview = preview_tabular(table, n=2, backend="polars")

    assert "id" in preview
    assert "name" in preview
    assert "Ada" in preview
    assert "Grace" in preview
    assert "shape" in preview or "rows" in preview


def test_preview_tabular_head_window_is_exact_without_vertical_ellipsis():
    if not supports_backend("polars"):
        pytest.skip("Polars backend is not installed")

    table = _build_preview_table("polars", row_count=60, col_count=9)
    preview = preview_tabular(table, n=50, cols=9, backend="polars")

    assert "row_0" in preview
    assert "row_49" in preview
    assert "row_50" not in preview
    assert "..." not in preview


@pytest.mark.parametrize("backend", ["polars", "pandas"])
def test_preview_tabular_cols_window_is_exact_without_horizontal_ellipsis(backend: str):
    if not supports_backend(backend):
        pytest.skip(f"{backend} backend is not installed")

    table = _build_preview_table(backend, row_count=4, col_count=12)
    preview = preview_tabular(table, n=4, cols=9, backend=backend)

    assert "col_0" in preview
    assert "col_8" in preview
    assert "col_9" not in preview
    assert "col_10" not in preview
    assert "col_11" not in preview
    assert "…" not in preview


@pytest.mark.parametrize("backend", ["polars", "pandas"])
def test_preview_cli_head_rows_and_cols_notes(backend: str, tmp_path: Path):
    if not supports_backend(backend):
        pytest.skip(f"{backend} backend is not installed")

    path = tmp_path / f"preview_head_cols_{backend}.csv"
    _write_preview_csv(path, row_count=4, col_count=12)

    too_few_rows = _run_cli([
        "preview",
        str(path),
        "--format",
        "csv",
        "--backend",
        backend,
        "--head",
        "10",
    ])
    assert too_few_rows.returncode == 0
    assert "note: --head=10 exceeds available rows (4); showing all 4 rows." in too_few_rows.stdout
    assert "row_0" in too_few_rows.stdout
    assert "row_3" in too_few_rows.stdout
    assert "row_4" not in too_few_rows.stdout

    too_many_cols = _run_cli([
        "preview",
        str(path),
        "--format",
        "csv",
        "--backend",
        backend,
        "--head",
        "4",
        "--cols",
        "20",
    ])
    assert too_many_cols.returncode == 0
    assert "note: --cols=20 exceeds available columns (12); showing all 12 columns." in too_many_cols.stdout
    assert "col_11" in too_many_cols.stdout


@pytest.mark.parametrize("backend", ["polars", "pandas"])
def test_preview_cli_display_all_overrides_head_and_cols(backend: str, tmp_path: Path):
    if not supports_backend(backend):
        pytest.skip(f"{backend} backend is not installed")

    path = tmp_path / f"preview_display_all_{backend}.csv"
    _write_preview_csv(path, row_count=4, col_count=12)

    result = _run_cli([
        "preview",
        str(path),
        "--format",
        "csv",
        "--backend",
        backend,
        "--head",
        "2",
        "--cols",
        "3",
        "--display-all",
    ])
    assert result.returncode == 0
    assert "note: --display-all is set; showing all 4 rows and 12 columns." in result.stdout
    assert "note: --display-all overrides --head." in result.stdout
    assert "note: --display-all overrides --cols." in result.stdout
    assert "row_0" in result.stdout
    assert "row_3" in result.stdout
    assert "col_11" in result.stdout
    assert "row_4" not in result.stdout
    assert "col_12" not in result.stdout


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("--head", "0"),
        ("--head", "-1"),
        ("--cols", "0"),
        ("--cols", "-1"),
    ],
)
def test_preview_cli_rejects_non_positive_head_or_cols(flag: str, value: str, tmp_path: Path):
    path = tmp_path / "preview_invalid_windows.csv"
    _write_preview_csv(path, row_count=2, col_count=3)

    result = _run_cli(["preview", str(path), "--format", "csv", "--backend", "polars", flag, value])
    assert result.returncode != 0
    assert "Must be > 0." in result.stderr


def test_preview_cli():
    if not supports_backend("polars"):
        pytest.skip("Polars backend is not installed")

    result = _run_cli(["preview", "tests/data/tabular/people_a.csv", "--format", "csv", "--backend", "polars", "--head", "2"])
    assert result.returncode == 0
    assert "preview (polars)" in result.stdout
    assert "Ada" in result.stdout


def test_concat_cli():
    if not supports_backend("polars"):
        pytest.skip("Polars backend is not installed")

    result = _run_cli([
        "concat",
        "tests/data/tabular/people_a.csv",
        "tests/data/tabular/people_b.csv",
        "--format",
        "csv",
        "--backend",
        "polars",
        "--head",
        "4",
    ])
    assert result.returncode == 0
    assert "concat (polars)" in result.stdout
    assert "Ken" in result.stdout


def test_merge_cli():
    if not supports_backend("polars"):
        pytest.skip("Polars backend is not installed")

    result = _run_cli([
        "merge",
        "tests/data/tabular/people_a.csv",
        "tests/data/tabular/people_meta.csv",
        "--on",
        "id",
        "--format",
        "csv",
        "--backend",
        "polars",
    ])
    assert result.returncode == 0
    assert "merge (polars)" in result.stdout
    assert "team-a" in result.stdout


@pytest.mark.parametrize("backend", ["polars", "pandas"])
def test_merge_cli_three_way_with_common_on(backend: str):
    if not supports_backend(backend):
        pytest.skip(f"{backend} backend is not installed")

    result = _run_cli([
        "merge",
        "tests/data/tabular/people_a.csv",
        "tests/data/tabular/people_meta.csv",
        "tests/data/tabular/people_job.csv",
        "--on",
        "id",
        "--how",
        "inner",
        "--format",
        "csv",
        "--backend",
        backend,
    ])
    assert result.returncode == 0
    assert f"merge ({backend})" in result.stdout
    assert "team-b" in result.stdout
    assert "Engineer" in result.stdout


def test_merge_cli_with_glob_input_for_nway_merge():
    if not supports_backend("polars"):
        pytest.skip("Polars backend is not installed")

    result = _run_cli([
        "merge",
        "tests/data/tabular/people_a.csv",
        "tests/data/tabular/people_[ab].csv",
        "tests/data/tabular/people_meta.csv",
        "--on",
        "id",
        "--format",
        "csv",
        "--backend",
        "polars",
        "--how",
        "left",
    ])
    assert result.returncode == 0
    assert "merge (polars)" in result.stdout
    assert "name_src2" in result.stdout


def test_merge_cli_with_side_specific_formats():
    if not supports_backend("polars"):
        pytest.skip("Polars backend is not installed")

    result = _run_cli([
        "merge",
        "tests/data/tabular/people_a.csv",
        "tests/data/tabular/people.tsv",
        "--on",
        "id",
        "--format",
        "tsv",
        "--left-format",
        "csv",
        "--right-format",
        "tsv",
        "--backend",
        "polars",
    ])
    assert result.returncode == 0
    assert "merge (polars)" in result.stdout
    assert "Ada" in result.stdout
    assert "Grace" in result.stdout


def test_merge_cli_with_side_specific_keys():
    if not supports_backend("polars"):
        pytest.skip("Polars backend is not installed")

    result = _run_cli([
        "merge",
        "tests/data/tabular/people_a.csv",
        "tests/data/tabular/people_right.tsv",
        "--left-on",
        "id",
        "--right-on",
        "key",
        "--left-format",
        "csv",
        "--right-format",
        "tsv",
        "--backend",
        "polars",
    ])
    assert result.returncode == 0
    assert "merge (polars)" in result.stdout
    assert "lead" in result.stdout
    assert "support" in result.stdout


def test_merge_cli_cross_join_without_keys():
    if not supports_backend("polars"):
        pytest.skip("Polars backend is not installed")

    result = _run_cli([
        "merge",
        "tests/data/tabular/people_a.csv",
        "tests/data/tabular/people_meta.csv",
        "--how",
        "cross",
        "--format",
        "csv",
        "--backend",
        "polars",
        "--head",
        "10",
    ])
    assert result.returncode == 0
    assert "merge (polars)" in result.stdout
    assert "shape: (6, 4)" in result.stdout
    assert "Ada" in result.stdout
    assert "Grace" in result.stdout


def test_merge_cli_rejects_invalid_key_combos():
    if not supports_backend("polars"):
        pytest.skip("Polars backend is not installed")

    result = _run_cli([
        "merge",
        "tests/data/tabular/people_a.csv",
        "tests/data/tabular/people_meta.csv",
        "--on",
        "id",
        "--left-on",
        "id",
        "--right-on",
        "id",
        "--format",
        "csv",
        "--backend",
        "polars",
    ])
    assert result.returncode == 1
    assert "Cannot provide --on and --left-on/--right-on together." in result.stderr

    result = _run_cli([
        "merge",
        "tests/data/tabular/people_a.csv",
        "tests/data/tabular/people_meta.csv",
        "--how",
        "cross",
        "--on",
        "id",
        "--format",
        "csv",
        "--backend",
        "polars",
    ])
    assert result.returncode == 1
    assert "Cross joins do not use join keys. Omit --on and --left-on/--right-on." in result.stderr

    result = _run_cli([
        "merge",
        "tests/data/tabular/people_a.csv",
        "tests/data/tabular/people_meta.csv",
        "--left-on",
        "id",
        "--format",
        "csv",
        "--backend",
        "polars",
    ])
    assert result.returncode == 1
    assert "Merge requires either --on or both --left-on and --right-on." in result.stderr


def test_invalid_backend_input_raises():
    with pytest.raises(UnsupportedBackendError, match="Unknown backend"):
        read_tabular(DATA_DIR / "people_a.csv", backend="apache", format="csv")


def test_invalid_format_input_raises():
    with pytest.raises(ValueError, match="Unsupported format"):
        read_tabular(DATA_DIR / "people_a.csv", backend="polars", format="xml")


def test_cli_invalid_backend_flag_errors():
    result = _run_cli(["preview", "tests/data/tabular/people_a.csv", "--backend", "apache", "--format", "csv"])
    assert result.returncode == 2


def test_cli_invalid_format_flag_errors():
    result = _run_cli(["preview", "tests/data/tabular/people_a.csv", "--format", "json", "--backend", "polars"])
    assert result.returncode == 2


def test_pandas_lazy_error_is_clear():
    if not supports_backend("pandas"):
        pytest.skip("pandas backend is not installed")

    with pytest.raises(LazyBackendUnsupportedError, match="pandas backend does not support lazy=True"):
        read_tabular(DATA_DIR / "people_a.csv", backend="pandas", format="csv", lazy=True)
