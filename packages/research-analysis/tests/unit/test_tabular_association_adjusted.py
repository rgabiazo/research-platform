from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

import pytest

import research_platform.analysis.tabular_associations as tabular_associations
from research_platform.analysis.tabular_associations import (
    RegressionAssociationResultRow,
    plan_tabular_association_adjusted,
    run_tabular_association_adjusted,
)


def _source_alpha(
    *,
    backend: str = "records",
    path: str | None = None,
    source_format: str | None = None,
    numeric_policy: str = "declare",
) -> dict[str, object]:
    source: dict[str, object] = {
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
            "numeric_validation": {"policy": numeric_policy},
        },
    }
    if path is not None:
        source["path"] = path
    if source_format is not None:
        source["format"] = source_format
    return source


def _source_beta() -> dict[str, object]:
    return {
        "source_id": "source-beta",
        "schema": {
            "subject_id_column": "participant-id",
            "columns": [
                {"column_name": "participant-id", "value_type": "categorical", "role": "subject_identifier"},
                {"column_name": "outcome-beta", "value_type": "numeric", "role": "outcome"},
                {"column_name": "covariate-beta", "value_type": "numeric", "role": "covariate"},
            ],
        },
    }


def _methods() -> list[dict[str, object]]:
    return [
        {
            "method_id": "partial-alpha",
            "method": "partial_correlation",
            "outcome_ids": ["outcome-alpha"],
            "predictor_ids": ["predictor-alpha"],
            "covariate_ids": ["covariate-alpha"],
            "family_id": "family-alpha",
        },
        {
            "method_id": "regression-alpha",
            "method": "regression",
            "outcome_ids": ["outcome-alpha"],
            "predictor_ids": ["predictor-alpha"],
            "covariate_ids": ["covariate-alpha"],
            "family_id": "family-alpha",
        },
    ]


def _workflow_doc(
    *,
    backend: str = "records",
    methods: list[dict[str, object]] | None = None,
    missing_strategy: str = "listwise",
    nonfinite_strategy: str = "drop_rows",
    numeric_policy: str = "declare",
    source_path: str | None = None,
    source_format: str | None = None,
) -> dict[str, object]:
    method_rows = methods if methods is not None else _methods()
    return {
        "workflow_id": "workflow-alpha",
        "backend": backend,
        "sources": [
            _source_alpha(
                backend=backend,
                path=source_path,
                source_format=source_format,
                numeric_policy=numeric_policy,
            )
        ],
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
        "missing_data_policy": {"strategy": missing_strategy},
        "nonfinite_policy": {"strategy": nonfinite_strategy},
        "methods": method_rows,
        "families": [{"family_id": "family-alpha", "method_ids": [str(method["method_id"]) for method in method_rows]}],
    }


def _partial_rows() -> list[dict[str, object]]:
    return [
        {"participant-id": "participant-a", "outcome-alpha": "12", "predictor-alpha": "-4", "covariate-alpha": "0"},
        {"participant-id": "participant-b", "outcome-alpha": "12", "predictor-alpha": "-5", "covariate-alpha": "1"},
        {"participant-id": "participant-c", "outcome-alpha": "14", "predictor-alpha": "1", "covariate-alpha": "2"},
        {"participant-id": "participant-d", "outcome-alpha": "18", "predictor-alpha": "-1", "covariate-alpha": "3"},
        {"participant-id": "participant-e", "outcome-alpha": "24", "predictor-alpha": "4", "covariate-alpha": "4"},
    ]


def _regression_rows() -> list[dict[str, object]]:
    return [
        {"participant-id": "participant-a", "outcome-alpha": "3", "predictor-alpha": "1", "covariate-alpha": "0"},
        {"participant-id": "participant-b", "outcome-alpha": "4", "predictor-alpha": "0", "covariate-alpha": "1"},
        {"participant-id": "participant-c", "outcome-alpha": "11", "predictor-alpha": "2", "covariate-alpha": "2"},
        {"participant-id": "participant-d", "outcome-alpha": "12", "predictor-alpha": "1", "covariate-alpha": "3"},
        {"participant-id": "participant-e", "outcome-alpha": "19", "predictor-alpha": "3", "covariate-alpha": "4"},
    ]


def _result_row(payload: dict[str, object], *, method_id: str | None = None) -> dict[str, object]:
    rows = payload["result_rows"]  # type: ignore[index]
    if method_id is None:
        assert len(rows) == 1
        return rows[0]  # type: ignore[index]
    return next(row for row in rows if row["method_id"] == method_id)  # type: ignore[index]


def _provenance(payload: dict[str, object]) -> dict[str, object]:
    return {str(row["key"]): row["value"] for row in payload["provenance_rows"]}  # type: ignore[index]


def test_valid_partial_association_from_in_memory_rows_has_known_value_and_propagation() -> None:
    methods = [_methods()[0]]
    before_modules = set(sys.modules)

    result = run_tabular_association_adjusted(
        _workflow_doc(backend="pandas", methods=methods),
        source_rows_by_id={"source-alpha": _partial_rows()},
    )
    payload = result.to_dict()
    imported_during_run = set(sys.modules) - before_modules
    row = _result_row(payload)
    provenance = _provenance(payload)

    assert payload["executed"] is True
    assert payload["plan_only"] is False
    assert payload["will_write"] is False
    assert payload["output_written"] is False
    assert row["method_kind"] == "partial_correlation"
    assert row["statistic_name"] == "partial_r"
    assert row["statistic_value"] == pytest.approx(2.0 / 7.0)
    assert row["n_total"] == 5
    assert row["n_used"] == 5
    assert row["family_id"] == "family-alpha"
    assert row["source_id"] == "source-alpha"
    assert row["outcome_id"] == "outcome-alpha"
    assert row["outcome_column"] == "outcome-alpha"
    assert row["predictor_id"] == "predictor-alpha"
    assert row["predictor_column"] == "predictor-alpha"
    assert row["covariate_ids"] == ["covariate-alpha"]
    assert row["covariate_columns"] == ["covariate-alpha"]
    assert row["covariate_count"] == 1
    assert provenance["requested_backend"] == "pandas"
    assert provenance["runtime_backend"] == "records"
    assert provenance["adjusted_regression_method_count"] == 1
    assert "pandas" not in imported_during_run
    json.dumps(payload, sort_keys=True, allow_nan=False)


def test_valid_regression_association_from_in_memory_rows_has_known_primary_coefficient() -> None:
    methods = [_methods()[1]]

    payload = run_tabular_association_adjusted(
        _workflow_doc(methods=methods),
        source_rows_by_id={"source-alpha": _regression_rows()},
    ).to_dict()
    row = _result_row(payload)

    assert row["method_kind"] == "regression"
    assert row["statistic_name"] == "regression_coefficient"
    assert row["statistic_value"] == pytest.approx(2.0)
    assert row["model_parameter_count"] == 3
    assert row["residual_degrees_of_freedom"] == 2
    assert row["covariate_ids"] == ["covariate-alpha"]
    assert row["covariate_columns"] == ["covariate-alpha"]


def test_multiple_declarations_expand_plans_and_preserve_family_source_variable_and_covariate_columns() -> None:
    methods = [
        {"method_id": "partial-alpha", "method": "partial_correlation", "family_id": "family-alpha"},
        {"method_id": "regression-alpha", "method": "regression", "family_id": "family-alpha"},
    ]

    plan_payload = plan_tabular_association_adjusted(_workflow_doc(methods=methods)).to_dict()
    partial_rows = [row for row in plan_payload["pair_plan_rows"] if row["method_id"] == "partial-alpha"]
    regression_rows = [row for row in plan_payload["pair_plan_rows"] if row["method_id"] == "regression-alpha"]

    assert plan_payload["executed"] is False
    assert plan_payload["plan_only"] is True
    assert len(partial_rows) == 4
    assert len(regression_rows) == 4
    assert all(row["covariate_ids"] == ["covariate-alpha", "covariate-beta"] for row in partial_rows)
    assert all(row["covariate_columns"] == ["covariate-alpha", "covariate-beta"] for row in partial_rows)
    assert all(row["covariate_count"] == 0 for row in regression_rows)
    assert all(row["family_id"] == "family-alpha" for row in plan_payload["pair_plan_rows"])
    assert all(row["source_id"] == "source-alpha" for row in plan_payload["pair_plan_rows"])
    assert {
        (row["outcome_id"], row["outcome_column"], row["predictor_id"], row["predictor_column"])
        for row in plan_payload["pair_plan_rows"]
    } >= {
        ("outcome-alpha", "outcome-alpha", "predictor-alpha", "predictor-alpha"),
        ("outcome-beta", "outcome-beta", "predictor-beta", "predictor-beta"),
    }


@pytest.mark.parametrize(
    ("missing_strategy", "expected_status", "expected_fragment"),
    [
        ("error", "error", "missing_data_policy_error"),
        ("listwise", "warning", "missing_data_policy_deferred"),
        ("allow", "warning", "missing_values_allowed"),
    ],
)
def test_listwise_missing_value_counts_and_policy_status(
    missing_strategy: str,
    expected_status: str,
    expected_fragment: str,
) -> None:
    methods = [_methods()[0]]
    rows = [
        {"participant-id": "participant-a", "outcome-alpha": "12", "predictor-alpha": "-4", "covariate-alpha": "0"},
        {"participant-id": "participant-b", "outcome-alpha": "", "predictor-alpha": "-5", "covariate-alpha": "1"},
        {"participant-id": "participant-c", "outcome-alpha": "14", "predictor-alpha": None, "covariate-alpha": "2"},
        {"participant-id": "participant-d", "outcome-alpha": "18", "predictor-alpha": "-1", "covariate-alpha": ""},
        {"participant-id": "participant-e", "outcome-alpha": "24", "predictor-alpha": "4", "covariate-alpha": "4"},
        {"participant-id": "participant-f", "outcome-alpha": "29", "predictor-alpha": "5", "covariate-alpha": "5"},
    ]

    payload = run_tabular_association_adjusted(
        _workflow_doc(methods=methods, missing_strategy=missing_strategy),
        source_rows_by_id={"source-alpha": rows},
    ).to_dict()
    row = _result_row(payload)
    messages = " ".join(row["warnings"] + row["errors"])  # type: ignore[operator]

    assert row["n_total"] == 6
    assert row["n_used"] == 3
    assert row["n_missing_outcome"] == 1
    assert row["n_missing_predictor"] == 1
    assert row["n_missing_covariates"] == 1
    assert row["n_missing_listwise"] == 3
    assert row["status"] == expected_status
    assert expected_fragment in messages


def test_nonfinite_invalid_and_bool_numeric_counts_are_reported_without_raw_nonfinite_output() -> None:
    methods = [_methods()[1]]
    rows = [
        {"participant-id": "participant-a", "outcome-alpha": "3", "predictor-alpha": "1", "covariate-alpha": "0"},
        {"participant-id": "participant-b", "outcome-alpha": "4", "predictor-alpha": "0", "covariate-alpha": "1"},
        {"participant-id": "participant-h", "outcome-alpha": "11", "predictor-alpha": "2", "covariate-alpha": "2"},
        {"participant-id": "participant-c", "outcome-alpha": float("nan"), "predictor-alpha": "2", "covariate-alpha": "2"},
        {"participant-id": "participant-d", "outcome-alpha": "12", "predictor-alpha": "inf", "covariate-alpha": "3"},
        {"participant-id": "participant-e", "outcome-alpha": "nan", "predictor-alpha": "-infinity", "covariate-alpha": "4"},
        {"participant-id": "participant-f", "outcome-alpha": "13", "predictor-alpha": "bad-value", "covariate-alpha": "5"},
        {"participant-id": "participant-g", "outcome-alpha": "14", "predictor-alpha": True, "covariate-alpha": "6"},
    ]

    payload = run_tabular_association_adjusted(
        _workflow_doc(methods=methods),
        source_rows_by_id={"source-alpha": rows},
    ).to_dict()
    row = _result_row(payload)

    assert row["n_total"] == 8
    assert row["n_used"] == 3
    assert row["n_nonfinite"] == 4
    assert row["n_invalid_numeric"] == 6
    assert row["n_bool_numeric"] == 1
    assert row["status"] == "warning"
    json.dumps(payload, sort_keys=True, allow_nan=False)


def test_partial_failure_modes_cover_too_few_singular_and_zero_residual_variance() -> None:
    methods = [_methods()[0]]
    too_few_payload = run_tabular_association_adjusted(
        _workflow_doc(methods=methods),
        source_rows_by_id={
            "source-alpha": [
                {"participant-id": "participant-a", "outcome-alpha": "1", "predictor-alpha": "2", "covariate-alpha": "3"}
            ]
        },
    ).to_dict()
    singular_payload = run_tabular_association_adjusted(
        _workflow_doc(methods=methods),
        source_rows_by_id={
            "source-alpha": [
                {"participant-id": "participant-a", "outcome-alpha": "1", "predictor-alpha": "2", "covariate-alpha": "1"},
                {"participant-id": "participant-b", "outcome-alpha": "2", "predictor-alpha": "3", "covariate-alpha": "1"},
                {"participant-id": "participant-c", "outcome-alpha": "3", "predictor-alpha": "5", "covariate-alpha": "1"},
            ]
        },
    ).to_dict()
    zero_outcome_payload = run_tabular_association_adjusted(
        _workflow_doc(methods=methods),
        source_rows_by_id={
            "source-alpha": [
                {"participant-id": "participant-a", "outcome-alpha": "0", "predictor-alpha": "0", "covariate-alpha": "0"},
                {"participant-id": "participant-b", "outcome-alpha": "1", "predictor-alpha": "2", "covariate-alpha": "1"},
                {"participant-id": "participant-c", "outcome-alpha": "2", "predictor-alpha": "1", "covariate-alpha": "2"},
                {"participant-id": "participant-d", "outcome-alpha": "3", "predictor-alpha": "3", "covariate-alpha": "3"},
            ]
        },
    ).to_dict()
    zero_predictor_payload = run_tabular_association_adjusted(
        _workflow_doc(methods=methods),
        source_rows_by_id={
            "source-alpha": [
                {"participant-id": "participant-a", "outcome-alpha": "0", "predictor-alpha": "0", "covariate-alpha": "0"},
                {"participant-id": "participant-b", "outcome-alpha": "2", "predictor-alpha": "1", "covariate-alpha": "1"},
                {"participant-id": "participant-c", "outcome-alpha": "1", "predictor-alpha": "2", "covariate-alpha": "2"},
                {"participant-id": "participant-d", "outcome-alpha": "3", "predictor-alpha": "3", "covariate-alpha": "3"},
            ]
        },
    ).to_dict()

    assert _result_row(too_few_payload)["statistic_value"] is None
    assert too_few_payload["computation_qc_rows"][0]["code"] == "too_few_valid_rows"
    assert _result_row(singular_payload)["statistic_value"] is None
    assert singular_payload["computation_qc_rows"][0]["code"] == "singular_covariate_design"
    assert _result_row(zero_outcome_payload)["statistic_value"] is None
    assert zero_outcome_payload["computation_qc_rows"][0]["code"] == "zero_residual_variance_outcome"
    assert _result_row(zero_predictor_payload)["statistic_value"] is None
    assert zero_predictor_payload["computation_qc_rows"][0]["code"] == "zero_residual_variance_predictor"


def test_regression_failure_modes_cover_too_few_singular_and_zero_variance() -> None:
    methods = [_methods()[1]]
    too_few_payload = run_tabular_association_adjusted(
        _workflow_doc(methods=methods),
        source_rows_by_id={
            "source-alpha": [
                {"participant-id": "participant-a", "outcome-alpha": "1", "predictor-alpha": "2", "covariate-alpha": "3"}
            ]
        },
    ).to_dict()
    singular_payload = run_tabular_association_adjusted(
        _workflow_doc(methods=methods),
        source_rows_by_id={
            "source-alpha": [
                {"participant-id": "participant-a", "outcome-alpha": "1", "predictor-alpha": "1", "covariate-alpha": "1"},
                {"participant-id": "participant-b", "outcome-alpha": "2", "predictor-alpha": "2", "covariate-alpha": "2"},
                {"participant-id": "participant-c", "outcome-alpha": "4", "predictor-alpha": "3", "covariate-alpha": "3"},
            ]
        },
    ).to_dict()
    zero_outcome_payload = run_tabular_association_adjusted(
        _workflow_doc(methods=methods),
        source_rows_by_id={
            "source-alpha": [
                {"participant-id": "participant-a", "outcome-alpha": "1", "predictor-alpha": "0", "covariate-alpha": "0"},
                {"participant-id": "participant-b", "outcome-alpha": "1", "predictor-alpha": "1", "covariate-alpha": "1"},
                {"participant-id": "participant-c", "outcome-alpha": "1", "predictor-alpha": "3", "covariate-alpha": "2"},
            ]
        },
    ).to_dict()
    zero_predictor_payload = run_tabular_association_adjusted(
        _workflow_doc(methods=methods),
        source_rows_by_id={
            "source-alpha": [
                {"participant-id": "participant-a", "outcome-alpha": "1", "predictor-alpha": "2", "covariate-alpha": "0"},
                {"participant-id": "participant-b", "outcome-alpha": "2", "predictor-alpha": "2", "covariate-alpha": "1"},
                {"participant-id": "participant-c", "outcome-alpha": "3", "predictor-alpha": "2", "covariate-alpha": "2"},
            ]
        },
    ).to_dict()

    assert _result_row(too_few_payload)["statistic_value"] is None
    assert too_few_payload["computation_qc_rows"][0]["code"] == "too_few_valid_rows"
    assert _result_row(singular_payload)["statistic_value"] is None
    assert singular_payload["computation_qc_rows"][0]["code"] == "singular_design_matrix"
    assert _result_row(zero_outcome_payload)["statistic_value"] is None
    assert zero_outcome_payload["computation_qc_rows"][0]["code"] == "zero_variance_outcome"
    assert _result_row(zero_predictor_payload)["statistic_value"] is None
    assert zero_predictor_payload["computation_qc_rows"][0]["code"] == "zero_variance_predictor"


def test_tsv_source_loading_path_is_supported_without_writing_outputs(tmp_path: Path) -> None:
    source_path = tmp_path / "source-alpha.tsv"
    source_path.write_text(
        "participant-id\toutcome-alpha\tpredictor-alpha\tcovariate-alpha\n"
        "participant-a\t3\t1\t0\n"
        "participant-b\t4\t0\t1\n"
        "participant-c\t11\t2\t2\n"
        "participant-d\t12\t1\t3\n"
        "participant-e\t19\t3\t4\n",
        encoding="utf-8",
    )
    methods = [_methods()[1]]

    payload = run_tabular_association_adjusted(
        _workflow_doc(methods=methods, source_path=str(source_path), source_format="tsv"),
        source_inventory_specs=[{"source_id": "source-alpha", "path": str(source_path), "source_format": "tsv"}],
    ).to_dict()

    assert _result_row(payload)["statistic_value"] == pytest.approx(2.0)
    assert payload["source_load_rows"][0]["load_status"] == "loaded"
    assert payload["will_write"] is False
    assert payload["output_written"] is False
    assert source_path.exists()


def test_deferred_cross_source_and_unsupported_adjusted_regression_behavior() -> None:
    methods = [
        {
            "method_id": "partial-alpha",
            "method": "partial_correlation",
            "outcome_ids": ["outcome-beta"],
            "predictor_ids": ["predictor-alpha"],
            "covariate_ids": ["covariate-alpha"],
        },
        {
            "method_id": "partial-rank-alpha",
            "method": "partial_correlation",
            "outcome_ids": ["outcome-alpha"],
            "predictor_ids": ["predictor-alpha"],
            "covariate_ids": ["covariate-alpha"],
            "metadata": {"partial_spearman": True},
        },
        {
            "method_id": "regression-group-alpha",
            "method": "regression",
            "outcome_ids": ["outcome-alpha"],
            "predictor_ids": ["predictor-alpha"],
            "grouping_ids": ["group-alpha"],
            "metadata": {"interaction": True, "model_type": "generalized-alpha"},
        },
    ]
    doc = _workflow_doc(methods=methods)
    doc["sources"] = [doc["sources"][0], _source_beta()]  # type: ignore[index]
    doc["outcomes"] = [
        {"variable_id": "outcome-alpha", "source_id": "source-alpha", "column_name": "outcome-alpha"},
        {"variable_id": "outcome-beta", "source_id": "source-beta", "column_name": "outcome-beta"},
    ]

    plan_payload = plan_tabular_association_adjusted(doc).to_dict()
    run_payload = run_tabular_association_adjusted(
        doc,
        source_rows_by_id={
            "source-alpha": _partial_rows(),
            "source-beta": [{"participant-id": "participant-a", "outcome-beta": "1", "covariate-beta": "2"}],
        },
    ).to_dict()
    warning_text = " ".join(" ".join(row["warnings"]) for row in plan_payload["pair_plan_rows"])

    assert "cross_source_adjusted_association_deferred" in {row["code"] for row in plan_payload["pair_plan_rows"]}
    assert "rank_adjusted_association_deferred" in warning_text
    assert "grouped_adjusted_association_deferred" in warning_text
    assert "interaction_model_deferred" in warning_text
    assert "unsupported_regression_model_deferred" in warning_text
    assert all(row["status"] == "deferred" for row in run_payload["result_rows"])
    assert all(row["statistic_value"] is None for row in run_payload["result_rows"])


def test_result_rows_are_json_safe_tsv_safe_omit_future_statistics_and_write_nothing(tmp_path: Path) -> None:
    declared_path = tmp_path / "declared-source-alpha.tsv"
    payload = run_tabular_association_adjusted(
        _workflow_doc(methods=[_methods()[1]], source_path=str(declared_path), source_format="tsv"),
        source_rows_by_id={"source-alpha": _regression_rows()},
        qc_result={"workflow_id": "workflow-alpha", "status": "ok"},
    ).to_dict()
    row = _result_row(payload)
    provenance = _provenance(payload)
    forbidden_fields = {
        "p_value",
        "q_value",
        "ci_low",
        "ci_high",
        "confidence_interval",
        "fdr",
        "standard_error",
        "r_squared",
        "bootstrap",
        "permutation",
        "mixed_model",
        "diagnostic",
        "regression_slope",
        "regression_intercept",
        "model_fit",
    }
    tsv_row = RegressionAssociationResultRow(
        workflow_id="workflow-alpha",
        pair_id="regression-alpha::outcome-alpha::predictor-alpha",
        method_id="regression-alpha",
        method_kind="regression",
        method_name="regression",
        family_id="family-alpha",
        source_id="source-alpha",
        outcome_id="outcome-alpha",
        outcome_source_id="source-alpha",
        outcome_column="outcome-alpha",
        predictor_id="predictor-alpha",
        predictor_source_id="source-alpha",
        predictor_column="predictor-alpha",
        covariate_ids=("covariate-alpha",),
        covariate_columns=("covariate-alpha",),
        covariate_count=1,
        model_parameter_count=3,
        residual_degrees_of_freedom=2,
        n_total=5,
        n_used=5,
        n_missing_outcome=0,
        n_missing_predictor=0,
        n_missing_covariates=0,
        n_missing_listwise=0,
        n_nonfinite=0,
        n_invalid_numeric=0,
        n_bool_numeric=0,
        statistic_name="regression_coefficient",
        statistic_value=2.0,
    ).to_tsv_row()

    json.dumps(payload, sort_keys=True, allow_nan=False)
    assert forbidden_fields.isdisjoint(row)
    assert all(forbidden_fields.isdisjoint(result_row) for result_row in payload["result_rows"])
    assert all(isinstance(value, str) for value in tsv_row.values())
    assert tsv_row["executed"] == "true"
    assert tsv_row["output_written"] == "false"
    assert payload["will_write"] is False
    assert payload["output_written"] is False
    assert provenance["qc_mode"] == "supplied"
    assert provenance["output_paths_written"] == []
    assert provenance["no_output_paths_written"] is True
    assert not declared_path.exists()


def test_tabular_association_adjusted_has_no_forbidden_imports_or_study_specific_constants() -> None:
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
