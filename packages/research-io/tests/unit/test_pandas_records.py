from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import sys

import pytest

from research_platform.io.dataframe.pandas_ops import (
    DEFAULT_PANDAS_INDEX_FIELD,
    PANDAS_RECORDS_ADAPTER_VERSION,
    PandasRecordsAdapter,
    detect_pandas_dataframe_protocol,
    inspect_pandas_dataframe_source,
    iter_pandas_dataframe_records,
    pandas_dataframe_to_records,
    plan_pandas_records_adapter,
)

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = PACKAGE_ROOT.parents[1]
MODULE_PATH = PACKAGE_ROOT / "src" / "research_platform" / "io" / "dataframe" / "pandas_ops.py"


def _pandas():
    return pytest.importorskip("pandas")


def _run_import_check(module_name: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PACKAGE_ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    script = (
        "import json, sys\n"
        f"__import__({module_name!r}, fromlist=['*'])\n"
        "blocked = ['pandas', 'polars', 'numpy', 'research_platform.analysis']\n"
        "print(json.dumps([name for name in blocked if name in sys.modules]))\n"
    )
    return subprocess.run(
        [sys.executable, "-c", script],
        cwd=PACKAGE_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )


def test_importing_io_modules_does_not_import_heavy_or_analysis_modules() -> None:
    for module_name in (
        "research_platform.io",
        "research_platform.io.dataframe",
        "research_platform.io.dataframe.pandas_ops",
    ):
        completed = _run_import_check(module_name)
        assert completed.returncode == 0, completed.stderr
        assert json.loads(completed.stdout) == []


def test_public_exports_are_available_without_importing_pandas() -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PACKAGE_ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    script = (
        "import json, sys\n"
        "from research_platform.io import PANDAS_RECORDS_ADAPTER_VERSION, pandas_dataframe_to_records\n"
        "from research_platform.io.dataframe import DEFAULT_PANDAS_INDEX_FIELD, PandasRecordsAdapter\n"
        "assert PANDAS_RECORDS_ADAPTER_VERSION == "
        "'research_platform.io.dataframe.pandas_records.v1'\n"
        "assert DEFAULT_PANDAS_INDEX_FIELD == 'pandas_index'\n"
        "assert callable(pandas_dataframe_to_records)\n"
        "assert PandasRecordsAdapter.__name__ == 'PandasRecordsAdapter'\n"
        "blocked = ['pandas', 'polars', 'numpy', 'research_platform.analysis']\n"
        "print(json.dumps([name for name in blocked if name in sys.modules]))\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=PACKAGE_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == []


def test_pandas_dataframe_conversion_preserves_order_columns_and_ignores_index_by_default() -> None:
    pd = _pandas()
    df = pd.DataFrame(
        {
            "column-alpha": ["row-alpha", "row-beta", "row-gamma"],
            "column-beta": [1, 2, 3],
        },
        index=["index-alpha", "index-beta", "index-gamma"],
    )

    result = pandas_dataframe_to_records(df)

    assert result.valid is True
    assert result.status == "ok"
    assert result.adapter_spec.runtime_backend == "records"
    assert result.adapter_spec.requested_backend == "pandas"
    assert result.row_source_kind == "pandas_dataframe"
    assert result.observed_columns == ("column-alpha", "column-beta")
    assert [record["column-alpha"] for record in result.records] == ["row-alpha", "row-beta", "row-gamma"]
    assert all(DEFAULT_PANDAS_INDEX_FIELD not in record for record in result.records)


def test_include_index_adds_default_or_supplied_index_field() -> None:
    pd = _pandas()
    df = pd.DataFrame(
        {"column-alpha": ["row-alpha", "row-beta"]},
        index=["index-alpha", "index-beta"],
    )

    default_result = pandas_dataframe_to_records(df, include_index=True)
    named_result = pandas_dataframe_to_records(df, include_index=True, index_field="column-beta")

    assert default_result.valid is True
    assert list(default_result.records[0]) == ["column-alpha", DEFAULT_PANDAS_INDEX_FIELD]
    assert default_result.records[0][DEFAULT_PANDAS_INDEX_FIELD] == "index-alpha"
    assert default_result.observed_columns == ("column-alpha", DEFAULT_PANDAS_INDEX_FIELD)
    assert named_result.records[1]["column-beta"] == "index-beta"
    assert named_result.observed_columns == ("column-alpha", "column-beta")


def test_index_field_collision_returns_safe_error() -> None:
    pd = _pandas()
    df = pd.DataFrame({DEFAULT_PANDAS_INDEX_FIELD: ["value-alpha"]}, index=["index-alpha"])

    result = pandas_dataframe_to_records(df, include_index=True)

    assert result.valid is False
    assert result.status == "error"
    assert result.records == ()
    assert result.qc_rows[0].status == "error"
    assert result.qc_rows[0].code == "pandas_index_field_collision"
    assert result.output_paths_written == ()
    assert result.no_output_written is True


def test_include_input_row_index_adds_deterministic_positional_indexes() -> None:
    pd = _pandas()
    df = pd.DataFrame({"column-alpha": ["row-alpha", "row-beta"]}, index=["index-alpha", "index-beta"])

    result = pandas_dataframe_to_records(df, include_input_row_index=True)

    assert result.valid is True
    assert result.records == (
        {"column-alpha": "row-alpha", "input_row_index": 0},
        {"column-alpha": "row-beta", "input_row_index": 1},
    )
    assert result.observed_columns == ("column-alpha", "input_row_index")
    assert result.provenance_rows[0].include_input_row_index is True


def test_input_row_index_collision_returns_safe_error() -> None:
    pd = _pandas()
    df = pd.DataFrame({"input_row_index": ["value-alpha"]})

    result = pandas_dataframe_to_records(df, include_input_row_index=True)

    assert result.valid is False
    assert result.status == "error"
    assert result.records == ()
    assert result.qc_rows[0].code == "input_row_index_collision"
    assert result.output_paths_written == ()
    assert result.no_output_written is True


def test_pandas_missing_and_non_finite_values_are_json_safe() -> None:
    pd = _pandas()
    df = pd.DataFrame(
        {
            "column-alpha": [pd.NA, pd.NaT, None, float("nan")],
            "column-beta": [float("inf"), float("-inf"), "value-alpha", 1],
        }
    )

    result = pandas_dataframe_to_records(df)
    payload = result.to_dict()

    assert result.valid is True
    assert [record["column-alpha"] for record in result.records] == [None, None, None, None]
    assert payload["records"][0]["column-beta"] == "inf"
    assert payload["records"][1]["column-beta"] == "-inf"
    json.dumps(payload, allow_nan=False)


def test_dataframe_is_not_mutated_and_returned_payloads_are_copied() -> None:
    pd = _pandas()
    df = pd.DataFrame({"column-alpha": ["row-alpha"], "column-beta": [1]}, index=["index-alpha"])
    before = df.copy(deep=True)

    result = pandas_dataframe_to_records(df, include_index=True)
    payload = result.to_dict()
    payload["records"][0]["column-alpha"] = "changed-value"
    result.records[0]["column-beta"] = 99

    pd.testing.assert_frame_equal(df, before)
    assert payload["records"][0]["column-alpha"] == "changed-value"
    assert result.to_dict()["records"][0]["column-alpha"] == "row-alpha"
    assert df.iloc[0]["column-beta"] == 1


def test_pandas_series_is_rejected_safely() -> None:
    pd = _pandas()
    series = pd.Series(["value-alpha"], name="column-alpha")

    result = pandas_dataframe_to_records(series)

    assert result.valid is False
    assert result.status == "error"
    assert result.records == ()
    assert result.row_source_kind == "pandas_series"
    assert result.qc_rows[0].code == "pandas_series_source"
    assert result.provenance_rows[0].row_count == 0


def test_qc_and_provenance_rows_are_tsv_safe_and_no_write() -> None:
    pd = _pandas()
    df = pd.DataFrame({"column-alpha": ["value\talpha\nline"]})

    result = pandas_dataframe_to_records(df, metadata={"source": "source-alpha"})

    assert result.adapter_spec.will_write is False
    assert result.adapter_spec.output_written is False
    assert result.adapter_spec.output_paths_written == ()
    assert result.adapter_spec.no_output_written is True
    assert result.output_paths_written == ()
    assert result.no_output_written is True
    for row in (*result.qc_rows, *result.provenance_rows):
        tsv_row = row.to_tsv_row()
        assert tsv_row
        assert all(isinstance(value, str) for value in tsv_row.values())
        assert all("\t" not in value and "\n" not in value and "\r" not in value for value in tsv_row.values())


def test_planning_detection_inspection_adapter_and_iterator_helpers() -> None:
    pd = _pandas()
    df = pd.DataFrame({"column-alpha": ["row-alpha", "row-beta"]})
    spec = plan_pandas_records_adapter(df, include_index=True, include_input_row_index=True)
    adapter = PandasRecordsAdapter(spec=spec, include_index=True)

    inspected = inspect_pandas_dataframe_source(df)
    iterated = tuple(iter_pandas_dataframe_records(df))
    adapted = adapter.to_records(df)

    assert PANDAS_RECORDS_ADAPTER_VERSION == "research_platform.io.dataframe.pandas_records.v1"
    assert spec.runtime_backend == "records"
    assert spec.requested_backend == "pandas"
    assert spec.row_source_kind == "pandas_dataframe"
    assert detect_pandas_dataframe_protocol(df) == "pandas_dataframe"
    assert inspected.valid is True
    assert iterated == ({"column-alpha": "row-alpha"}, {"column-alpha": "row-beta"})
    assert adapted.records[0][DEFAULT_PANDAS_INDEX_FIELD] == 0
    assert adapted.records[0]["input_row_index"] == 0
    assert adapter.to_dict()["adapter_kind"] == "pandas_records_adapter"


def test_no_hard_coded_study_specific_constants_or_subject_style_ids() -> None:
    production_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (
            MODULE_PATH,
            PACKAGE_ROOT / "README.md",
            REPO_ROOT / "docs" / "decisions" / "ADR-0014-generic-tabular-association-workflow-roadmap.md",
        )
    )
    forbidden = [
        "confidential-study-marker",
        "private-task-marker",
        "private-cohort-marker",
        "participant-alpha",
        "participant-beta",
        "private-feature-marker",
        "private-contrast-marker",
    ]

    assert all(token not in production_text for token in forbidden)
    assert re.search(r"\bsub-\d{3}\b", production_text) is None
