from __future__ import annotations

import ast
import json
import math
import re
import sys
from pathlib import Path

import pytest

import research_platform.analysis.tabular_associations as tabular_associations
from research_platform.analysis.tabular_associations import (
    AssociationMultiplicityResultRow,
    plan_tabular_association_multiplicity,
    run_tabular_association_multiplicity,
)


def _source_alpha(backend: str = "records") -> dict[str, object]:
    return {
        "source_id": "source-alpha",
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
                {"column_name": "outcome-beta", "value_type": "numeric", "role": "outcome"},
                {"column_name": "predictor-alpha", "value_type": "numeric", "role": "predictor"},
                {"column_name": "predictor-beta", "value_type": "numeric", "role": "predictor"},
                {"column_name": "covariate-alpha", "value_type": "numeric", "role": "covariate"},
                {"column_name": "covariate-beta", "value_type": "numeric", "role": "covariate"},
                {"column_name": "group-alpha", "value_type": "categorical", "role": "grouping"},
            ],
        },
    }


def _methods() -> list[dict[str, object]]:
    return [
        {
            "method_id": "method-alpha",
            "method": "pearson",
            "outcome_ids": ["outcome-alpha"],
            "predictor_ids": ["predictor-alpha"],
            "family_id": "family-alpha",
        },
        {
            "method_id": "method-beta",
            "method": "regression",
            "outcome_ids": ["outcome-beta"],
            "predictor_ids": ["predictor-beta"],
            "covariate_ids": ["covariate-beta"],
            "family_id": "family-beta",
        },
    ]


def _workflow_doc(
    *,
    backend: str = "records",
    families: list[dict[str, object]] | None = None,
    multiple_testing: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    return {
        "workflow_id": "workflow-alpha",
        "backend": backend,
        "sources": [_source_alpha(backend=backend)],
        "outcomes": [
            {"variable_id": "outcome-alpha", "source_id": "source-alpha", "column_name": "outcome-alpha"},
            {"variable_id": "outcome-beta", "source_id": "source-alpha", "column_name": "outcome-beta"},
        ],
        "predictors": [
            {"variable_id": "predictor-alpha", "source_id": "source-alpha", "column_name": "predictor-alpha"},
            {"variable_id": "predictor-beta", "source_id": "source-alpha", "column_name": "predictor-beta"},
        ],
        "covariates": [
            {"variable_id": "covariate-alpha", "source_id": "source-alpha", "column_name": "covariate-alpha"},
            {"variable_id": "covariate-beta", "source_id": "source-alpha", "column_name": "covariate-beta"},
        ],
        "groupings": [
            {"variable_id": "group-alpha", "source_id": "source-alpha", "column_name": "group-alpha"}
        ],
        "methods": _methods(),
        "families": families
        if families is not None
        else [
            {"family_id": "family-alpha", "method_ids": ["method-alpha"]},
            {"family_id": "family-beta", "method_ids": ["method-beta"]},
        ],
        "multiple_testing": multiple_testing
        if multiple_testing is not None
        else [
            {"family_id": "family-alpha", "method": "benjamini_hochberg"},
            {"family_id": "family-beta", "method": "fdr_bh"},
        ],
    }


def _provenance(payload: dict[str, object]) -> dict[str, object]:
    return {str(row["key"]): row["value"] for row in payload["provenance_rows"]}  # type: ignore[index]


def _q_values(payload: dict[str, object]) -> list[object]:
    return [row["q_value"] for row in payload["result_rows"]]  # type: ignore[index]


def test_benjamini_hochberg_q_values_and_metadata_propagation_without_mutating_rows() -> None:
    rows = [
        {
            "workflow_id": "workflow-alpha",
            "result_row_id": "result-alpha",
            "multiple_testing_family_id": "family-alpha",
            "method_id": "method-alpha",
            "method_kind": "correlation",
            "method_name": "pearson",
            "source_id": "source-alpha",
            "outcome_id": "outcome-alpha",
            "predictor_id": "predictor-alpha",
            "covariate_ids": ["covariate-alpha"],
            "statistic_name": "r",
            "statistic_value": 0.10,
            "p_value": 0.01,
        },
        {
            "workflow_id": "workflow-alpha",
            "result_row_id": "result-beta",
            "comparison_family_id": "family-alpha",
            "method_id": "method-alpha",
            "source_id": "source-alpha",
            "outcome_id": "outcome-alpha",
            "predictor_id": "predictor-beta",
            "covariate_ids": ["covariate-alpha"],
            "statistic_name": "r",
            "statistic_value": 0.20,
            "p_value": 0.04,
        },
        {
            "workflow_id": "workflow-alpha",
            "result_row_id": "result-gamma",
            "family_id": "family-alpha",
            "method_id": "method-alpha",
            "source_id": "source-alpha",
            "outcome_id": "outcome-beta",
            "predictor_id": "predictor-alpha",
            "covariate_ids": ["covariate-beta"],
            "statistic_name": "r",
            "statistic_value": 0.30,
            "p_value": "0.03",
        },
        {
            "workflow_id": "workflow-alpha",
            "result_row_id": "result-delta",
            "method_id": "method-alpha",
            "source_id": "source-alpha",
            "outcome_id": "outcome-beta",
            "predictor_id": "predictor-beta",
            "covariate_ids": ["covariate-beta"],
            "statistic_name": "r",
            "statistic_value": 0.40,
            "p_value": 0.20,
        },
    ]
    original_rows = [dict(row) for row in rows]
    before_modules = set(sys.modules)

    payload = run_tabular_association_multiplicity(
        _workflow_doc(backend="pandas"),
        result_rows=rows,
    ).to_dict()
    imported_during_run = set(sys.modules) - before_modules
    first_row = payload["result_rows"][0]
    provenance = _provenance(payload)

    assert payload["executed"] is True
    assert payload["plan_only"] is False
    assert payload["will_write"] is False
    assert payload["output_written"] is False
    assert payload["status"] == "ok"
    assert _q_values(payload) == pytest.approx([0.04, 0.05333333333333334, 0.05333333333333334, 0.20])
    assert first_row["workflow_id"] == "workflow-alpha"
    assert first_row["family_id"] == "family-alpha"
    assert first_row["multiple_testing_method"] == "benjamini_hochberg"
    assert first_row["correction_method"] == "benjamini_hochberg"
    assert first_row["result_row_id"] == "result-alpha"
    assert first_row["input_row_index"] == 0
    assert first_row["method_id"] == "method-alpha"
    assert first_row["method_kind"] == "correlation"
    assert first_row["method_name"] == "pearson"
    assert first_row["source_id"] == "source-alpha"
    assert first_row["outcome_id"] == "outcome-alpha"
    assert first_row["predictor_id"] == "predictor-alpha"
    assert first_row["covariate_ids"] == ["covariate-alpha"]
    assert first_row["statistic_name"] == "r"
    assert first_row["statistic_value"] == pytest.approx(0.10)
    assert first_row["p_value"] == pytest.approx(0.01)
    assert first_row["n_family_total"] == 4
    assert first_row["n_valid_p"] == 4
    assert first_row["n_adjusted"] == 4
    assert rows == original_rows
    assert provenance["requested_backend"] == "pandas"
    assert provenance["runtime_backend"] == "records"
    assert provenance["adjusted_row_count"] == 4
    assert provenance["output_paths_written"] == []
    assert provenance["no_output_paths_written"] is True
    assert "pandas" not in imported_during_run
    json.dumps(payload, sort_keys=True, allow_nan=False)


def test_multiple_families_are_adjusted_independently_and_tied_p_values_keep_input_order() -> None:
    rows = [
        {"result_row_id": "result-alpha", "family_id": "family-alpha", "p_value": 0.02},
        {"result_row_id": "result-beta", "family_id": "family-alpha", "p_value": 0.01},
        {"result_row_id": "result-gamma", "family_id": "family-beta", "p_value": 0.04},
        {"result_row_id": "result-delta", "family_id": "family-beta", "p_value": 0.04},
    ]

    payload = run_tabular_association_multiplicity(_workflow_doc(), result_rows=rows).to_dict()
    q_by_id = {row["result_row_id"]: row["q_value"] for row in payload["result_rows"]}

    assert [row["result_row_id"] for row in payload["result_rows"]] == [
        "result-alpha",
        "result-beta",
        "result-gamma",
        "result-delta",
    ]
    assert q_by_id == pytest.approx(
        {
            "result-alpha": 0.02,
            "result-beta": 0.02,
            "result-gamma": 0.04,
            "result-delta": 0.04,
        }
    )
    assert {row["family_id"] for row in payload["method_summary_rows"]} == {"family-alpha", "family-beta"}
    assert all(row["n_adjusted"] == 2 for row in payload["result_rows"])


def test_missing_p_value_rows_are_reported_without_inventing_p_or_q_values() -> None:
    rows = [
        {
            "result_row_id": "result-alpha",
            "family_id": "family-alpha",
            "method_id": "method-alpha",
            "source_id": "source-alpha",
            "outcome_id": "outcome-alpha",
            "predictor_id": "predictor-alpha",
            "statistic_name": "r",
            "statistic_value": 1.0,
        },
        {
            "result_row_id": "result-beta",
            "family_id": "family-alpha",
            "method_id": "method-alpha",
            "source_id": "source-alpha",
            "outcome_id": "outcome-alpha",
            "predictor_id": "predictor-beta",
            "statistic_name": "r",
            "statistic_value": 0.5,
            "p_value": "",
        },
    ]

    payload = run_tabular_association_multiplicity(_workflow_doc(), result_rows=rows).to_dict()

    assert payload["valid"] is True
    assert payload["status"] == "warning"
    assert {row["code"] for row in payload["result_rows"]} == {"missing_p_value"}
    assert all(row["p_value"] is None for row in payload["result_rows"])
    assert all(row["q_value"] is None for row in payload["result_rows"])
    assert all(row["status"] == "warning" for row in payload["result_rows"])
    assert {row["code"] for row in payload["qc_rows"]} >= {"missing_p_value", "no_valid_p_values"}
    json.dumps(payload, sort_keys=True, allow_nan=False)


def test_invalid_p_values_are_excluded_from_bh_and_reported_json_safely() -> None:
    rows = [
        {"result_row_id": "result-valid", "family_id": "family-alpha", "p_value": 0.05},
        {"result_row_id": "result-negative", "family_id": "family-alpha", "p_value": -0.01},
        {"result_row_id": "result-large", "family_id": "family-alpha", "p_value": 1.01},
        {"result_row_id": "result-bool", "family_id": "family-alpha", "p_value": True},
        {"result_row_id": "result-text", "family_id": "family-alpha", "p_value": "not-numeric"},
        {"result_row_id": "result-blank", "family_id": "family-alpha", "p_value": " "},
        {"result_row_id": "result-nan", "family_id": "family-alpha", "p_value": math.nan},
        {"result_row_id": "result-inf", "family_id": "family-alpha", "p_value": math.inf},
        {"result_row_id": "result-neg-inf", "family_id": "family-alpha", "p_value": -math.inf},
    ]

    payload = run_tabular_association_multiplicity(_workflow_doc(), result_rows=rows).to_dict()
    result_by_id = {row["result_row_id"]: row for row in payload["result_rows"]}

    assert result_by_id["result-valid"]["q_value"] == pytest.approx(0.05)
    assert result_by_id["result-blank"]["code"] == "missing_p_value"
    for result_id in (
        "result-negative",
        "result-large",
        "result-bool",
        "result-text",
        "result-nan",
        "result-inf",
        "result-neg-inf",
    ):
        assert result_by_id[result_id]["code"] == "invalid_p_value"
        assert result_by_id[result_id]["p_value"] is None
        assert result_by_id[result_id]["q_value"] is None
    assert result_by_id["result-valid"]["n_valid_p"] == 1
    assert result_by_id["result-valid"]["n_missing_p"] == 1
    assert result_by_id["result-valid"]["n_invalid_p"] == 7
    json.dumps(payload, sort_keys=True, allow_nan=False)


def test_p_value_policy_error_marks_missing_and_invalid_p_values_as_errors() -> None:
    payload = run_tabular_association_multiplicity(
        _workflow_doc(),
        result_rows=[
            {"result_row_id": "result-alpha", "family_id": "family-alpha"},
            {"result_row_id": "result-beta", "family_id": "family-alpha", "p_value": "bad-value"},
        ],
        p_value_policy="error",
    ).to_dict()

    assert payload["valid"] is False
    assert payload["status"] == "error"
    assert {row["status"] for row in payload["result_rows"]} == {"error"}
    assert {row["status"] for row in payload["qc_rows"] if row["code"] in {"missing_p_value", "invalid_p_value"}} == {
        "error"
    }


def test_missing_undeclared_and_missing_multiple_testing_family_cases_are_deferred() -> None:
    payload = run_tabular_association_multiplicity(
        _workflow_doc(multiple_testing=[{"family_id": "family-alpha", "method": "benjamini_hochberg"}]),
        result_rows=[
            {"result_row_id": "result-missing-family", "p_value": 0.01},
            {"result_row_id": "result-unknown-family", "family_id": "family-gamma", "p_value": 0.02},
            {"result_row_id": "result-missing-spec", "family_id": "family-beta", "p_value": 0.03},
        ],
    ).to_dict()
    result_by_id = {row["result_row_id"]: row for row in payload["result_rows"]}
    codes = {row["code"] for row in payload["qc_rows"]}

    assert result_by_id["result-missing-family"]["code"] == "missing_family_id"
    assert result_by_id["result-unknown-family"]["code"] == "undeclared_family_id"
    assert result_by_id["result-missing-spec"]["code"] == "missing_multiple_testing_spec"
    assert all(row["q_value"] is None for row in payload["result_rows"])
    assert {"missing_family_id", "undeclared_family_id", "missing_multiple_testing_spec"}.issubset(codes)


def test_declared_non_bh_multiple_testing_methods_are_deferred_without_q_values() -> None:
    plan_payload = plan_tabular_association_multiplicity(
        {
            "tabular_association_workflow": _workflow_doc(
                multiple_testing=[
                    {"family_id": "family-alpha", "method": "bonferroni"},
                    {"family_id": "family-beta", "method": "holm"},
                ]
            )
        }
    ).to_dict()
    run_payload = run_tabular_association_multiplicity(
        _workflow_doc(
            multiple_testing=[
                {"family_id": "family-alpha", "method": "none"},
                {"family_id": "family-beta", "method": "benjamini_yekutieli"},
            ]
        ),
        result_rows=[
            {"result_row_id": "result-alpha", "family_id": "family-alpha", "p_value": 0.01},
            {"result_row_id": "result-beta", "family_id": "family-beta", "p_value": 0.02},
        ],
    ).to_dict()

    assert plan_payload["executed"] is False
    assert plan_payload["plan_only"] is True
    assert {row["code"] for row in plan_payload["qc_rows"]} == {"multiple_testing_method_deferred"}
    assert {row["code"] for row in run_payload["result_rows"]} == {"multiple_testing_method_deferred"}
    assert all(row["q_value"] is None for row in run_payload["result_rows"])


def test_result_rows_are_tsv_safe_and_omit_generation_model_interval_and_publication_fields() -> None:
    result = run_tabular_association_multiplicity(
        _workflow_doc(),
        result_rows=[{"result_row_id": "result-alpha", "family_id": "family-alpha", "p_value": 0.01}],
    )
    payload = result.to_dict()
    row = payload["result_rows"][0]
    forbidden_fields = {
        "computed_p_value",
        "p_value_generated",
        "confidence_interval",
        "ci_low",
        "ci_high",
        "effect_size",
        "bootstrap",
        "permutation",
        "model_parameter_count",
        "standard_error",
        "publication_table",
        "figure_path",
    }
    tsv_row = AssociationMultiplicityResultRow(
        workflow_id="workflow-alpha",
        family_id="family-alpha",
        multiple_testing_method="benjamini_hochberg",
        correction_method="benjamini_hochberg",
        result_row_id="result-alpha",
        input_row_index=0,
        method_id="method-alpha",
        method_kind="correlation",
        method_name="pearson",
        source_id="source-alpha",
        outcome_id="outcome-alpha",
        predictor_id="predictor-alpha",
        covariate_ids=("covariate-alpha",),
        statistic_name="r",
        statistic_value=0.5,
        p_value=0.01,
        q_value=0.01,
        n_family_total=1,
        n_valid_p=1,
        n_missing_p=0,
        n_invalid_p=0,
        n_adjusted=1,
        status="ok",
        code="benjamini_hochberg_adjusted",
    ).to_tsv_row()

    assert forbidden_fields.isdisjoint(row)
    assert all(forbidden_fields.isdisjoint(result_row) for result_row in payload["result_rows"])
    assert all(isinstance(value, str) for value in tsv_row.values())
    assert tsv_row["executed"] == "true"
    assert tsv_row["output_written"] == "false"
    assert payload["will_write"] is False
    assert payload["output_written"] is False
    json.dumps(payload, sort_keys=True, allow_nan=False)


def test_plan_rejects_unknown_p_value_policy_without_writing() -> None:
    payload = plan_tabular_association_multiplicity(_workflow_doc(), p_value_policy="unsupported").to_dict()

    assert payload["valid"] is False
    assert payload["status"] == "error"
    assert payload["will_write"] is False
    assert payload["output_written"] is False
    assert payload["workflow_validation_rows"][0]["code"] == "workflow_parse_error"


def test_tabular_association_multiplicity_has_no_forbidden_imports_or_study_specific_constants() -> None:
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
    assert re.search("sub-" + r"\d{3}", combined_text) is None
