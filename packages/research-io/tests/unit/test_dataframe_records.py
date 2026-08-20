from __future__ import annotations

import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys

from research_platform.io import dataframe_to_records as exported_dataframe_to_records
from research_platform.io.dataframe import dataframe_to_records as dataframe_package_to_records
from research_platform.io.dataframe.records import (
    DATAFRAME_RECORDS_ADAPTER_VERSION,
    DataframeRecordsAdapter,
    dataframe_to_records,
    detect_dataframe_like_protocol,
    inspect_dataframe_like_source,
    iter_dataframe_records,
    plan_dataframe_records_adapter,
)

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = PACKAGE_ROOT / "src" / "research_platform" / "io" / "dataframe" / "records.py"


def _rows() -> list[dict[str, object]]:
    return [
        {"column-alpha": "row-alpha", "column-beta": 1},
        {"column-alpha": "row-beta", "column-beta": 2},
    ]


class FakeToDicts:
    columns = ("column-alpha", "column-beta")

    def __init__(self, rows: list[dict[str, object]]) -> None:
        self._rows = rows

    def to_dicts(self) -> list[dict[str, object]]:
        return self._rows


class FakeToRecords:
    columns = ("column-alpha", "column-beta")

    def __init__(self, rows: list[dict[str, object]]) -> None:
        self._rows = rows

    def to_records(self) -> list[dict[str, object]]:
        return self._rows


class FakeIterRowsNamed:
    columns = ("column-alpha", "column-beta")

    def __init__(self, rows: list[dict[str, object]]) -> None:
        self._rows = rows

    def iter_rows(self, *, named: bool = False) -> list[dict[str, object]]:
        if not named:
            return [{"column-alpha": "value-alpha"}]
        return self._rows


class FakeIterRowsZero:
    columns = ("column-alpha", "column-beta")

    def __init__(self, rows: list[dict[str, object]]) -> None:
        self._rows = rows

    def iter_rows(self, *args: object, **kwargs: object) -> list[dict[str, object]]:
        if args or kwargs:
            raise TypeError("zero-argument iterator only")
        return self._rows


class FakeRowsAttribute:
    columns = ("column-alpha", "column-beta")

    def __init__(self, rows: list[dict[str, object]]) -> None:
        self.rows = rows


class FakeRowsMethod:
    columns = ("column-alpha", "column-beta")

    def __init__(self, rows: list[dict[str, object]]) -> None:
        self._rows = rows

    def rows(self) -> list[dict[str, object]]:
        return self._rows


class FakeRecordsAttribute:
    columns = ("column-alpha", "column-beta")

    def __init__(self, rows: list[dict[str, object]]) -> None:
        self.records = rows


class FakeRecordsMethod:
    columns = ("column-alpha", "column-beta")

    def __init__(self, rows: list[dict[str, object]]) -> None:
        self._records = rows

    def records(self) -> list[dict[str, object]]:
        return self._records


class FakeColumnsPlusRows:
    columns = ("column-alpha", "column-beta")

    def to_dicts(self) -> list[dict[str, object]]:
        return [{"column-beta": 3, "column-gamma": "value-alpha"}]


class FakeCallableColumns:
    def columns(self) -> tuple[str, ...]:
        return ("ignored-column",)

    def to_dicts(self) -> list[dict[str, object]]:
        return [{"column-alpha": "value-alpha"}]


class FakeBadToRecords:
    def to_records(self) -> list[tuple[str, int]]:
        return [("column-alpha", 1)]


def _assert_successful_records(result, *, row_source_kind: str) -> None:
    assert result.valid is True
    assert result.status == "ok"
    assert result.row_source_kind == row_source_kind
    assert result.adapter_spec.runtime_backend == "records"
    assert result.adapter_spec.requested_backend is None
    assert result.records == (
        {"column-alpha": "row-alpha", "column-beta": 1},
        {"column-alpha": "row-beta", "column-beta": 2},
    )
    assert result.observed_columns == ("column-alpha", "column-beta")
    assert result.qc_rows[0].status == "ok"
    assert result.provenance_rows[0].runtime_backend == "records"


def test_dataframe_to_records_from_fake_to_dicts() -> None:
    result = dataframe_to_records(FakeToDicts(_rows()))

    _assert_successful_records(result, row_source_kind="to_dicts")


def test_dataframe_to_records_from_fake_to_records() -> None:
    result = dataframe_to_records(FakeToRecords(_rows()))

    _assert_successful_records(result, row_source_kind="to_records")


def test_dataframe_to_records_from_fake_iter_rows_named() -> None:
    result = dataframe_to_records(FakeIterRowsNamed(_rows()))

    _assert_successful_records(result, row_source_kind="iter_rows_named")


def test_dataframe_to_records_from_fake_iter_rows_zero_argument() -> None:
    result = dataframe_to_records(FakeIterRowsZero(_rows()))

    _assert_successful_records(result, row_source_kind="iter_rows")


def test_dataframe_to_records_from_fake_rows_attribute() -> None:
    result = dataframe_to_records(FakeRowsAttribute(_rows()))

    _assert_successful_records(result, row_source_kind="rows_attribute")


def test_dataframe_to_records_from_fake_rows_method() -> None:
    result = dataframe_to_records(FakeRowsMethod(_rows()))

    _assert_successful_records(result, row_source_kind="rows_method")


def test_dataframe_to_records_from_fake_records_attribute() -> None:
    result = dataframe_to_records(FakeRecordsAttribute(_rows()))

    _assert_successful_records(result, row_source_kind="records_attribute")


def test_dataframe_to_records_from_fake_records_method() -> None:
    result = dataframe_to_records(FakeRecordsMethod(_rows()))

    _assert_successful_records(result, row_source_kind="records_method")


def test_dataframe_to_records_from_mapping_row_sequences() -> None:
    result = dataframe_to_records(_rows())

    _assert_successful_records(result, row_source_kind="mapping_sequence")


def test_public_exports_are_available() -> None:
    assert exported_dataframe_to_records is dataframe_to_records
    assert dataframe_package_to_records is dataframe_to_records
    assert DATAFRAME_RECORDS_ADAPTER_VERSION == "research_platform.io.dataframe.records.v1"


def test_observed_columns_use_columns_attribute_plus_row_keys() -> None:
    result = dataframe_to_records(FakeColumnsPlusRows())

    assert result.valid is True
    assert result.observed_columns == ("column-alpha", "column-beta", "column-gamma")
    assert result.observed_column_count == 3


def test_callable_columns_are_ignored() -> None:
    result = dataframe_to_records(FakeCallableColumns())

    assert result.valid is True
    assert result.observed_columns == ("column-alpha",)


def test_stable_row_order_is_preserved() -> None:
    result = dataframe_to_records(
        [
            {"column-alpha": "row-alpha"},
            {"column-alpha": "row-beta"},
            {"column-alpha": "row-gamma"},
        ]
    )

    assert [record["column-alpha"] for record in result.records] == ["row-alpha", "row-beta", "row-gamma"]


def test_copied_records_do_not_mutate_original_rows() -> None:
    source_rows = [{"column-alpha": "value-alpha"}]
    result = dataframe_to_records(source_rows)

    result.records[0]["column-alpha"] = "changed-value"
    source_rows[0]["column-alpha"] = "source-changed"

    assert source_rows == [{"column-alpha": "source-changed"}]
    assert result.records[0]["column-alpha"] == "changed-value"


def test_original_fake_dataframe_object_is_not_mutated() -> None:
    rows = _rows()
    source = FakeToDicts(rows)
    before_rows = [dict(row) for row in source._rows]
    before_columns = source.columns

    result = dataframe_to_records(source)

    assert result.valid is True
    assert source._rows == before_rows
    assert source.columns == before_columns


def test_include_input_row_index_adds_deterministic_indexes() -> None:
    result = dataframe_to_records(_rows(), include_input_row_index=True)

    assert result.valid is True
    assert result.records == (
        {"column-alpha": "row-alpha", "column-beta": 1, "input_row_index": 0},
        {"column-alpha": "row-beta", "column-beta": 2, "input_row_index": 1},
    )
    assert result.observed_columns == ("column-alpha", "column-beta", "input_row_index")
    assert result.provenance_rows[0].include_input_row_index is True


def test_input_row_index_collision_returns_safe_error() -> None:
    result = dataframe_to_records([{"input_row_index": "row-alpha"}], include_input_row_index=True)

    assert result.valid is False
    assert result.status == "error"
    assert result.records == ()
    assert result.qc_rows[0].code == "input_row_index_collision"
    assert result.output_paths_written == ()
    assert result.no_output_written is True


def test_unsupported_object_produces_safe_error_qc_and_provenance() -> None:
    result = dataframe_to_records(object(), requested_backend="backend-alpha")

    assert result.valid is False
    assert result.status == "error"
    assert result.records == ()
    assert result.errors
    assert result.adapter_spec.requested_backend == "backend-alpha"
    assert result.adapter_spec.runtime_backend == "records"
    assert result.qc_rows[0].status == "error"
    assert result.provenance_rows[0].row_count == 0


def test_strings_and_bytes_are_rejected() -> None:
    string_result = dataframe_to_records("value-alpha")
    bytes_result = dataframe_to_records(b"value-alpha")

    assert string_result.valid is False
    assert string_result.qc_rows[0].code == "string_source"
    assert bytes_result.valid is False
    assert bytes_result.qc_rows[0].code == "bytes_source"


def test_single_mapping_as_whole_row_source_is_rejected() -> None:
    result = dataframe_to_records({"column-alpha": "value-alpha"})

    assert result.valid is False
    assert result.qc_rows[0].code == "single_mapping_source"
    assert result.records == ()


def test_positional_tuple_and_list_rows_are_rejected() -> None:
    tuple_result = dataframe_to_records([("value-alpha", 1)])
    list_result = dataframe_to_records([["value-alpha", 1]])

    assert tuple_result.valid is False
    assert tuple_result.qc_rows[0].code == "non_mapping_row"
    assert list_result.valid is False
    assert list_result.qc_rows[0].code == "non_mapping_row"


def test_row_producing_methods_returning_non_mapping_rows_are_rejected() -> None:
    result = dataframe_to_records(FakeBadToRecords())

    assert result.valid is False
    assert result.qc_rows[0].code == "non_mapping_row"
    assert result.records == ()


def test_json_dumps_allow_nan_false_succeeds_on_result_payloads() -> None:
    result = dataframe_to_records(
        [
            {
                "column-alpha": math.nan,
                "column-beta": math.inf,
                "column-gamma": -math.inf,
            }
        ]
    )
    payload = result.to_dict()

    assert payload["records"][0]["column-alpha"] == "nan"
    assert payload["records"][0]["column-beta"] == "inf"
    assert payload["records"][0]["column-gamma"] == "-inf"
    json.dumps(payload, allow_nan=False)


def test_to_tsv_row_works_on_qc_and_provenance_rows() -> None:
    result = dataframe_to_records([{"column-alpha": "value\talpha\nline"}])

    for row in (*result.qc_rows, *result.provenance_rows):
        tsv_row = row.to_tsv_row()
        assert tsv_row
        assert all(isinstance(value, str) for value in tsv_row.values())
        assert all("\t" not in value and "\n" not in value and "\r" not in value for value in tsv_row.values())


def test_no_write_flags_and_output_paths_are_empty() -> None:
    result = dataframe_to_records(_rows())

    assert result.adapter_spec.will_write is False
    assert result.adapter_spec.output_written is False
    assert result.adapter_spec.output_paths_written == ()
    assert result.adapter_spec.no_output_written is True
    assert result.output_paths_written == ()
    assert result.no_output_written is True
    assert result.to_dict()["output_paths_written"] == []
    assert result.to_dict()["no_output_written"] is True


def test_planning_inspection_adapter_and_iterator_helpers_are_no_write_records_helpers() -> None:
    spec = plan_dataframe_records_adapter(
        FakeToDicts(_rows()),
        requested_backend="dataframe_like",
        include_input_row_index=True,
        metadata={"source": "source-alpha"},
    )
    adapter = DataframeRecordsAdapter(spec)
    inspected = inspect_dataframe_like_source(FakeToDicts(_rows()), requested_backend="fake-test-alpha")
    iterated = tuple(iter_dataframe_records([{"column-alpha": "value-alpha"}]))
    adapted = adapter.to_records(FakeToDicts(_rows()))

    assert spec.runtime_backend == "records"
    assert spec.requested_backend == "dataframe_like"
    assert spec.will_write is False
    assert adapter.to_dict()["adapter_kind"] == "dataframe_records_adapter"
    assert inspected.valid is True
    assert inspected.adapter_spec.requested_backend == "fake-test-alpha"
    assert iterated == ({"column-alpha": "value-alpha"},)
    assert adapted.records[0]["input_row_index"] == 0


def test_detect_dataframe_like_protocol_labels_sources_without_backend_resolution() -> None:
    assert detect_dataframe_like_protocol(FakeToDicts(_rows())) == "to_dicts"
    assert detect_dataframe_like_protocol(FakeToRecords(_rows())) == "to_records"
    assert detect_dataframe_like_protocol(FakeRowsAttribute(_rows())) == "rows_attribute"
    assert detect_dataframe_like_protocol([{"column-alpha": "value-alpha"}]) == "mapping_sequence"
    assert detect_dataframe_like_protocol({"column-alpha": "value-alpha"}) == "single_mapping"
    assert detect_dataframe_like_protocol(None) == "none"


def test_importing_io_modules_does_not_import_heavy_or_analysis_modules() -> None:
    for module_name in (
        "research_platform.io",
        "research_platform.io.dataframe",
        "research_platform.io.dataframe.records",
    ):
        completed = _run_import_check(module_name)
        assert completed.returncode == 0, completed.stderr
        assert json.loads(completed.stdout) == []


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


def test_no_hard_coded_study_specific_constants_or_subject_style_ids() -> None:
    production_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (
            MODULE_PATH,
            PACKAGE_ROOT / "README.md",
            PACKAGE_ROOT.parents[1] / "docs" / "decisions" / "ADR-0014-generic-tabular-association-workflow-roadmap.md",
        )
    )
    forbidden = [
        "confidential-study-marker",
        "private-task-marker",
        "private-cohort-marker",
        "private-site-marker",
    ]

    assert all(token not in production_text for token in forbidden)
    assert re.search(r"\bsub-\d{3}\b", production_text) is None
