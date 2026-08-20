from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

import pytest

import research_platform.analysis.tabular_associations as tabular_associations
from research_platform.analysis.tabular_associations import (
    TabularMissingnessRow,
    plan_tabular_association_qc,
    run_tabular_association_qc,
)


def _source_alpha(*, path: str | None = None, source_format: str | None = "tsv", backend: str = "records") -> dict[str, object]:
    source: dict[str, object] = {
        "source_id": "source-alpha",
        "format": source_format,
        "path": path,
        "backend": backend,
        "schema": {
            "subject_id_column": "participant-id",
            "session_column": "session-label",
            "timepoint_column": "timepoint-label",
            "columns": [
                {"column_name": "participant-id", "value_type": "categorical", "role": "subject_identifier"},
                {"column_name": "session-label", "value_type": "categorical", "role": "session_identifier"},
                {"column_name": "timepoint-label", "value_type": "categorical", "role": "timepoint_identifier"},
                {"column_name": "outcome-alpha", "value_type": "numeric", "role": "outcome"},
                {"column_name": "predictor-alpha", "value_type": "numeric", "role": "predictor"},
                {"column_name": "covariate-alpha", "value_type": "numeric", "role": "covariate", "required": False},
                {"column_name": "group-alpha", "value_type": "categorical", "role": "grouping"},
            ],
            "categorical_validation": {
                "policy": "strict",
                "allowed_values": {"group-alpha": ["group-alpha"]},
            },
            "numeric_validation": {
                "policy": "strict",
                "min_value": 0,
                "max_value": 100,
                "integer_only": True,
            },
        },
    }
    if path is None:
        source.pop("path")
    if source_format is None:
        source.pop("format")
    return source


def _workflow_doc(*, path: str | None = None, source_format: str | None = "tsv", backend: str = "records") -> dict[str, object]:
    return {
        "workflow_id": "workflow-alpha",
        "sources": [_source_alpha(path=path, source_format=source_format, backend=backend)],
        "outcomes": [
            {"variable_id": "outcome-alpha", "source_id": "source-alpha", "column_name": "outcome-alpha"}
        ],
        "predictors": [
            {"variable_id": "predictor-alpha", "source_id": "source-alpha", "column_name": "predictor-alpha"}
        ],
        "covariates": [
            {"variable_id": "covariate-alpha", "source_id": "source-alpha", "column_name": "covariate-alpha"}
        ],
        "groupings": [
            {"variable_id": "group-alpha", "source_id": "source-alpha", "column_name": "group-alpha"}
        ],
        "repeated_measures": {
            "source_id": "source-alpha",
            "subject_id_column": "participant-id",
            "session_column": "session-label",
            "timepoint_column": "timepoint-label",
            "unit_columns": ["participant-id", "session-label", "timepoint-label"],
        },
        "missing_data_policy": {"strategy": "error"},
        "duplicate_subject_policy": {"strategy": "error", "key_columns": ["participant-id"]},
        "nonfinite_policy": {"strategy": "error"},
        "methods": [
            {
                "method_id": "pearson-alpha",
                "method": "pearson",
                "outcome_ids": ["outcome-alpha"],
                "predictor_ids": ["predictor-alpha"],
            }
        ],
    }


def _clean_rows() -> list[dict[str, object]]:
    return [
        {
            "participant-id": "participant-a",
            "session-label": "session-alpha",
            "timepoint-label": "timepoint-alpha",
            "outcome-alpha": "10",
            "predictor-alpha": "2",
            "covariate-alpha": "3",
            "group-alpha": "group-alpha",
        },
        {
            "participant-id": "participant-b",
            "session-label": "session-beta",
            "timepoint-label": "timepoint-beta",
            "outcome-alpha": "12",
            "predictor-alpha": "4",
            "covariate-alpha": "5",
            "group-alpha": "group-alpha",
        },
    ]


def _problem_rows() -> list[dict[str, object]]:
    return [
        {
            "participant-id": "participant-a",
            "session-label": "session-alpha",
            "timepoint-label": "timepoint-alpha",
            "outcome-alpha": "10",
            "predictor-alpha": "2",
            "covariate-alpha": "3",
            "group-alpha": "group-alpha",
            "extra-alpha": "observed",
        },
        {
            "participant-id": "participant-a",
            "session-label": "session-alpha",
            "timepoint-label": "timepoint-alpha",
            "outcome-alpha": "101",
            "predictor-alpha": True,
            "covariate-alpha": "",
            "group-alpha": "group-beta",
        },
        {
            "participant-id": "participant-b",
            "session-label": "session-alpha",
            "timepoint-label": "timepoint-alpha",
            "outcome-alpha": float("nan"),
            "predictor-alpha": "not-numeric",
            "covariate-alpha": "-inf",
            "group-alpha": "",
        },
        {
            "participant-id": "participant-c",
            "session-label": "session-alpha",
            "timepoint-label": "timepoint-alpha",
            "outcome-alpha": "4.5",
            "predictor-alpha": "inf",
            "covariate-alpha": "5",
            "group-alpha": "group-alpha",
        },
    ]


def _rows_by_column(payload: dict[str, object], row_group: str, column_key: str = "column_name") -> dict[str, dict[str, object]]:
    return {str(row[column_key]): row for row in payload[row_group]}  # type: ignore[index]


def _codes(payload: dict[str, object], row_group: str) -> set[str]:
    return {str(row["code"]) for row in payload[row_group]}  # type: ignore[index]


def test_plan_qc_accepts_top_level_document_and_sets_no_write_flags() -> None:
    payload = plan_tabular_association_qc({"tabular_association_workflow": _workflow_doc()}).to_dict()

    assert payload["workflow_id"] == "workflow-alpha"
    assert payload["executed"] is False
    assert payload["plan_only"] is True
    assert payload["will_write"] is False
    assert payload["output_written"] is False
    assert payload["source_inventory_rows"][0]["load_status"] == "planned"
    json.dumps(payload, sort_keys=True, allow_nan=False)


def test_in_memory_source_inventory_and_qc_rows_are_json_safe() -> None:
    before_modules = set(sys.modules)
    payload = run_tabular_association_qc(
        _workflow_doc(backend="pandas"),
        source_rows_by_id={"source-alpha": _problem_rows()},
    ).to_dict()
    imported_during_qc = set(sys.modules) - before_modules

    source_row = payload["source_inventory_rows"][0]
    assert payload["executed"] is False
    assert payload["plan_only"] is False
    assert payload["will_write"] is False
    assert payload["output_written"] is False
    assert payload["status"] == "error"
    assert source_row["source_kind"] == "in_memory"
    assert source_row["requested_backend"] == "pandas"
    assert source_row["runtime_backend"] == "records"
    assert source_row["row_count"] == 4
    assert "extra-alpha" in source_row["observed_only_columns"]
    assert "pandas" not in imported_during_qc

    duplicate_rows = payload["duplicate_rows"]
    assert any(row["key_type"] == "subject" and row["duplicate_count"] == 2 for row in duplicate_rows)
    assert any(row["key_type"] == "subject_session_timepoint" for row in duplicate_rows)
    assert any(row["key_type"] == "repeated_unit" for row in duplicate_rows)

    nonfinite_by_column = _rows_by_column(payload, "nonfinite_rows")
    assert nonfinite_by_column["outcome-alpha"]["tokens"] == ["nan"]
    assert nonfinite_by_column["predictor-alpha"]["tokens"] == ["inf"]
    assert nonfinite_by_column["covariate-alpha"]["tokens"] == ["-inf"]

    numeric_by_column = _rows_by_column(payload, "numeric_qc_rows")
    assert numeric_by_column["predictor-alpha"]["bool_count"] == 1
    assert numeric_by_column["predictor-alpha"]["invalid_numeric_count"] == 3
    assert numeric_by_column["outcome-alpha"]["above_max_count"] == 1
    assert numeric_by_column["outcome-alpha"]["noninteger_count"] == 1

    categorical_by_column = _rows_by_column(payload, "categorical_qc_rows")
    assert categorical_by_column["group-alpha"]["unknown_levels"] == ["group-beta"]
    assert categorical_by_column["group-alpha"]["status"] == "error"

    missing_by_column = _rows_by_column(payload, "missingness_rows")
    assert missing_by_column["covariate-alpha"]["status"] == "ok"
    assert missing_by_column["group-alpha"]["missing_count"] == 1

    forbidden_result_keys = {"p_value", "q_value", "estimate", "effect_size", "diagnostics"}
    for value in payload.values():
        if isinstance(value, list):
            for row in value:
                if isinstance(row, dict):
                    assert forbidden_result_keys.isdisjoint(row)
    json.dumps(payload, sort_keys=True, allow_nan=False)


def test_multiple_in_memory_sources_are_inventoried() -> None:
    doc = _workflow_doc()
    source_beta = {
        "source_id": "source-beta",
        "format": "csv",
        "schema": {
            "subject_id_column": "participant-id",
            "columns": [
                {"column_name": "participant-id", "value_type": "categorical", "role": "subject_identifier"},
                {"column_name": "outcome-beta", "value_type": "numeric", "role": "outcome"},
                {"column_name": "predictor-beta", "value_type": "numeric", "role": "predictor"},
            ],
        },
    }
    doc["sources"] = [doc["sources"][0], source_beta]  # type: ignore[index]
    doc["outcomes"] = [
        doc["outcomes"][0],  # type: ignore[index]
        {"variable_id": "outcome-beta", "source_id": "source-beta", "column_name": "outcome-beta"},
    ]
    doc["predictors"] = [
        doc["predictors"][0],  # type: ignore[index]
        {"variable_id": "predictor-beta", "source_id": "source-beta", "column_name": "predictor-beta"},
    ]

    payload = run_tabular_association_qc(
        doc,
        source_rows_by_id={
            "source-alpha": _clean_rows(),
            "source-beta": [
                {"participant-id": "participant-a", "outcome-beta": "1", "predictor-beta": "2"},
            ],
        },
    ).to_dict()

    assert {row["source_id"] for row in payload["source_inventory_rows"]} == {"source-alpha", "source-beta"}
    assert all(row["source_kind"] == "in_memory" for row in payload["source_inventory_rows"])


@pytest.mark.parametrize(
    ("source_format", "suffix", "contents", "inventory_spec"),
    [
        (
            "tsv",
            ".tsv",
            "participant-id\tsession-label\ttimepoint-label\toutcome-alpha\tpredictor-alpha\tcovariate-alpha\tgroup-alpha\n"
            "participant-a\tsession-alpha\ttimepoint-alpha\t1\t2\t3\tgroup-alpha\n",
            {},
        ),
        (
            "csv",
            ".csv",
            "participant-id,session-label,timepoint-label,outcome-alpha,predictor-alpha,covariate-alpha,group-alpha\n"
            "participant-a,session-alpha,timepoint-alpha,1,2,3,group-alpha\n",
            {},
        ),
        (
            "json",
            ".json",
            json.dumps(_clean_rows()),
            {},
        ),
        (
            "json",
            ".json",
            json.dumps({"rows": _clean_rows()}),
            {"row_key": "rows"},
        ),
    ],
)
def test_stdlib_source_loaders_support_tsv_csv_json_list_and_json_row_key(
    tmp_path: Path,
    source_format: str,
    suffix: str,
    contents: str,
    inventory_spec: dict[str, str],
) -> None:
    source_path = tmp_path / f"source-alpha{suffix}"
    source_path.write_text(contents, encoding="utf-8")
    spec = {"source_id": "source-alpha", "path": str(source_path), "source_format": source_format, **inventory_spec}

    payload = run_tabular_association_qc(
        _workflow_doc(path=str(source_path), source_format=source_format),
        source_inventory_specs=[spec],
    ).to_dict()

    assert payload["source_load_rows"][0]["load_status"] == "loaded"
    assert payload["source_inventory_rows"][0]["row_count"] >= 1
    assert payload["source_inventory_rows"][0]["source_kind"] == source_format
    assert payload["source_inventory_rows"][0]["provenance"]["sha256"]
    json.dumps(payload, sort_keys=True, allow_nan=False)


def test_missing_required_and_optional_sources_report_errors_or_warnings(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing-source-alpha.tsv"
    required_payload = run_tabular_association_qc(_workflow_doc(path=str(missing_path))).to_dict()
    optional_payload = run_tabular_association_qc(
        _workflow_doc(path=None),
        source_inventory_specs=[{"source_id": "source-alpha", "required": False}],
    ).to_dict()

    assert required_payload["source_load_rows"][0]["load_status"] == "missing"
    assert required_payload["source_load_rows"][0]["errors"]
    assert required_payload["valid"] is False
    assert not missing_path.exists()
    assert optional_payload["source_load_rows"][0]["load_status"] == "missing"
    assert optional_payload["source_load_rows"][0]["warnings"]
    assert optional_payload["source_load_rows"][0]["errors"] == []


def test_unsupported_source_format_respects_required_flag(tmp_path: Path) -> None:
    source_path = tmp_path / "source-alpha.unsupported"
    source_path.write_text("neutral", encoding="utf-8")
    required_payload = run_tabular_association_qc(
        _workflow_doc(path=str(source_path), source_format="unsupported")
    ).to_dict()
    optional_payload = run_tabular_association_qc(
        _workflow_doc(path=str(source_path), source_format="unsupported"),
        source_inventory_specs=[{"source_id": "source-alpha", "required": False}],
    ).to_dict()

    assert required_payload["source_load_rows"][0]["load_status"] == "unsupported"
    assert required_payload["source_load_rows"][0]["errors"]
    assert optional_payload["source_load_rows"][0]["warnings"]
    assert optional_payload["source_load_rows"][0]["errors"] == []


def test_required_column_and_variable_role_checks_detect_missing_observed_columns() -> None:
    rows = [
        {
            "participant-id": "participant-a",
            "session-label": "session-alpha",
            "timepoint-label": "timepoint-alpha",
            "outcome-alpha": "1",
            "group-alpha": "group-alpha",
            "extra-alpha": "observed",
        }
    ]

    payload = run_tabular_association_qc(_workflow_doc(), source_rows_by_id={"source-alpha": rows}).to_dict()

    assert "predictor_column_missing" in _codes(payload, "schema_validation_rows")
    assert "required_declared_column_missing" in _codes(payload, "schema_validation_rows")
    assert "variable_column_not_observed" in _codes(payload, "variable_qc_rows")
    assert "extra-alpha" in payload["source_inventory_rows"][0]["observed_only_columns"]


def test_zero_row_source_and_tsv_safe_row_conversion() -> None:
    payload = run_tabular_association_qc(_workflow_doc(), source_rows_by_id={"source-alpha": []}).to_dict()
    tsv_row = TabularMissingnessRow(
        workflow_id="workflow-alpha",
        source_id="source-alpha",
        column_name="outcome-alpha",
        role="outcome",
        required=True,
        missing_count=0,
        nonmissing_count=0,
        total_count=0,
        policy_strategy="error",
        status="ok",
        code="missing_values_absent",
        message="No missing values observed.",
    ).to_tsv_row()

    assert payload["source_load_rows"][0]["load_status"] == "empty"
    assert "zero_row_source" in _codes(payload, "schema_validation_rows")
    assert all(isinstance(value, str) for value in tsv_row.values())
    assert tsv_row["required"] == "true"
    json.dumps(payload, sort_keys=True, allow_nan=False)


def test_malformed_json_reports_qc_error_without_crashing(tmp_path: Path) -> None:
    source_path = tmp_path / "source-alpha.json"
    source_path.write_text("{not valid json", encoding="utf-8")

    payload = run_tabular_association_qc(_workflow_doc(path=str(source_path), source_format="json")).to_dict()

    assert payload["source_load_rows"][0]["load_status"] == "error"
    assert payload["source_load_rows"][0]["errors"]
    json.dumps(payload, sort_keys=True, allow_nan=False)


def test_tabular_association_qc_has_no_forbidden_imports_or_study_specific_constants() -> None:
    forbidden_modules = (
        "numpy",
        "pandas",
        "polars",
        "scipy",
        "sklearn",
        "statsmodels",
        "research_platform.neuro",
        "research_platform.core",
        "research_platform.viz",
        "research_platform.io",
        "pipelines",
        "ops",
    )
    forbidden_production_text = (
        "confidential-study-marker",
        "private-task-marker",
        "private-cohort-marker",
        "participant-alpha",
        "participant-beta",
    )
    imported_modules: list[str] = []
    combined_text = ""
    production_text = Path(tabular_associations.__file__).read_text(encoding="utf-8")
    for path in (Path(tabular_associations.__file__), Path(__file__)):
        source_text = path.read_text(encoding="utf-8")
        combined_text += source_text
        tree = ast.parse(source_text)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                imported_modules.append(node.module)

    for imported_module in imported_modules:
        assert not any(
            imported_module == forbidden or imported_module.startswith(f"{forbidden}.")
            for forbidden in forbidden_modules
        )
    for text in forbidden_production_text:
        assert text not in production_text
    assert re.search(r"sub-\d{3}", combined_text) is None
