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
    CorrelationAssociationResultRow,
    plan_tabular_association_correlations,
    run_tabular_association_correlations,
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
            ],
        },
    }


def _workflow_doc(
    *,
    backend: str = "records",
    methods: list[dict[str, object]] | None = None,
    missing_strategy: str = "pairwise",
    nonfinite_strategy: str = "drop_rows",
    numeric_policy: str = "declare",
    source_path: str | None = None,
    source_format: str | None = None,
) -> dict[str, object]:
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
            {"variable_id": "covariate-alpha", "source_id": "source-alpha", "column_name": "covariate-alpha"}
        ],
        "groupings": [
            {"variable_id": "group-alpha", "source_id": "source-alpha", "column_name": "group-alpha"}
        ],
        "missing_data_policy": {"strategy": missing_strategy},
        "nonfinite_policy": {"strategy": nonfinite_strategy},
        "methods": methods
        if methods is not None
        else [
            {
                "method_id": "pearson-alpha",
                "method": "pearson",
                "outcome_ids": ["outcome-alpha"],
                "predictor_ids": ["predictor-alpha"],
                "family_id": "family-alpha",
            }
        ],
        "families": [{"family_id": "family-alpha", "method_ids": ["pearson-alpha", "spearman-alpha"]}],
    }


def _clean_rows() -> list[dict[str, object]]:
    return [
        {
            "participant-id": "participant-a",
            "session-label": "session-alpha",
            "timepoint-label": "timepoint-alpha",
            "outcome-alpha": "1",
            "outcome-beta": "10",
            "predictor-alpha": "2",
            "predictor-beta": "4",
            "covariate-alpha": "1",
            "group-alpha": "group-alpha",
        },
        {
            "participant-id": "participant-b",
            "session-label": "session-alpha",
            "timepoint-label": "timepoint-beta",
            "outcome-alpha": "2",
            "outcome-beta": "8",
            "predictor-alpha": "4",
            "predictor-beta": "3",
            "covariate-alpha": "2",
            "group-alpha": "group-alpha",
        },
        {
            "participant-id": "participant-c",
            "session-label": "session-beta",
            "timepoint-label": "timepoint-alpha",
            "outcome-alpha": "3",
            "outcome-beta": "6",
            "predictor-alpha": "6",
            "predictor-beta": "2",
            "covariate-alpha": "3",
            "group-alpha": "group-alpha",
        },
        {
            "participant-id": "participant-d",
            "session-label": "session-beta",
            "timepoint-label": "timepoint-beta",
            "outcome-alpha": "4",
            "outcome-beta": "4",
            "predictor-alpha": "8",
            "predictor-beta": "1",
            "covariate-alpha": "4",
            "group-alpha": "group-alpha",
        },
    ]


def _result_row(payload: dict[str, object], *, pair_id: str | None = None) -> dict[str, object]:
    rows = payload["result_rows"]  # type: ignore[index]
    if pair_id is None:
        assert len(rows) == 1
        return rows[0]  # type: ignore[index]
    return next(row for row in rows if row["pair_id"] == pair_id)  # type: ignore[index]


def _provenance(payload: dict[str, object]) -> dict[str, object]:
    return {str(row["key"]): row["value"] for row in payload["provenance_rows"]}  # type: ignore[index]


def test_valid_pearson_correlation_from_in_memory_one_source_rows() -> None:
    before_modules = set(sys.modules)
    result = run_tabular_association_correlations(
        _workflow_doc(backend="pandas"),
        source_rows_by_id={"source-alpha": _clean_rows()},
    )
    payload = result.to_dict()
    imported_during_run = set(sys.modules) - before_modules
    row = _result_row(payload)
    provenance = _provenance(payload)

    assert payload["executed"] is True
    assert payload["plan_only"] is False
    assert payload["will_write"] is False
    assert payload["output_written"] is False
    assert row["statistic_name"] == "r"
    assert row["statistic_value"] == pytest.approx(1.0)
    assert row["n_total"] == 4
    assert row["n_used"] == 4
    assert row["family_id"] == "family-alpha"
    assert row["source_id"] == "source-alpha"
    assert row["outcome_id"] == "outcome-alpha"
    assert row["outcome_column"] == "outcome-alpha"
    assert row["predictor_id"] == "predictor-alpha"
    assert row["predictor_column"] == "predictor-alpha"
    assert provenance["requested_backend"] == "pandas"
    assert provenance["runtime_backend"] == "records"
    assert "pandas" not in imported_during_run
    json.dumps(payload, sort_keys=True, allow_nan=False)


def test_valid_spearman_correlation_from_in_memory_one_source_rows() -> None:
    methods = [
        {
            "method_id": "spearman-alpha",
            "method": "spearman",
            "outcome_ids": ["outcome-alpha"],
            "predictor_ids": ["predictor-alpha"],
            "family_id": "family-alpha",
        }
    ]

    payload = run_tabular_association_correlations(
        _workflow_doc(methods=methods),
        source_rows_by_id={"source-alpha": _clean_rows()},
    ).to_dict()
    row = _result_row(payload)

    assert row["statistic_name"] == "rho"
    assert row["statistic_value"] == pytest.approx(1.0)
    assert row["tie_count_outcome"] == 0
    assert row["tie_count_predictor"] == 0


def test_spearman_ties_use_average_ranks() -> None:
    methods = [
        {
            "method_id": "spearman-alpha",
            "method": "spearman",
            "outcome_ids": ["outcome-alpha"],
            "predictor_ids": ["predictor-alpha"],
        }
    ]
    rows = [
        {"participant-id": "participant-a", "outcome-alpha": "1", "predictor-alpha": "1"},
        {"participant-id": "participant-b", "outcome-alpha": "1", "predictor-alpha": "2"},
        {"participant-id": "participant-c", "outcome-alpha": "2", "predictor-alpha": "2"},
        {"participant-id": "participant-d", "outcome-alpha": "3", "predictor-alpha": "3"},
    ]

    payload = run_tabular_association_correlations(
        _workflow_doc(methods=methods),
        source_rows_by_id={"source-alpha": rows},
    ).to_dict()
    row = _result_row(payload)

    assert row["statistic_value"] == pytest.approx(0.8333333333333334)
    assert row["tie_count_outcome"] == 2
    assert row["tie_count_predictor"] == 2
    assert row["tie_group_count_outcome"] == 1
    assert row["tie_group_count_predictor"] == 1


def test_multiple_pairs_methods_family_and_column_propagation() -> None:
    methods = [
        {"method_id": "pearson-alpha", "method": "pearson", "family_id": "family-alpha"},
        {"method_id": "spearman-alpha", "method": "spearman", "family_id": "family-alpha"},
    ]

    plan_payload = plan_tabular_association_correlations(_workflow_doc(methods=methods)).to_dict()
    run_payload = run_tabular_association_correlations(
        _workflow_doc(methods=methods),
        source_rows_by_id={"source-alpha": _clean_rows()},
    ).to_dict()

    assert plan_payload["executed"] is False
    assert plan_payload["plan_only"] is True
    assert len(plan_payload["pair_plan_rows"]) == 8
    assert len(run_payload["result_rows"]) == 8
    assert {row["method_id"] for row in run_payload["method_summary_rows"]} == {"pearson-alpha", "spearman-alpha"}
    assert all(row["family_id"] == "family-alpha" for row in run_payload["result_rows"])
    assert all(row["source_id"] == "source-alpha" for row in run_payload["result_rows"])
    assert {
        (row["outcome_id"], row["outcome_column"], row["predictor_id"], row["predictor_column"])
        for row in run_payload["result_rows"]
    } >= {
        ("outcome-alpha", "outcome-alpha", "predictor-alpha", "predictor-alpha"),
        ("outcome-beta", "outcome-beta", "predictor-beta", "predictor-beta"),
    }


@pytest.mark.parametrize(
    ("missing_strategy", "expected_status", "expected_fragment"),
    [
        ("error", "error", "missing_data_policy_error"),
        ("pairwise", "warning", "pairwise_missing_values_dropped"),
        ("listwise", "warning", "missing_data_policy_deferred"),
    ],
)
def test_pairwise_missing_value_counts_and_policy_status(
    missing_strategy: str,
    expected_status: str,
    expected_fragment: str,
) -> None:
    rows = [
        {"participant-id": "participant-a", "outcome-alpha": "1", "predictor-alpha": "2"},
        {"participant-id": "participant-b", "outcome-alpha": "", "predictor-alpha": "4"},
        {"participant-id": "participant-c", "outcome-alpha": "3", "predictor-alpha": None},
        {"participant-id": "participant-d", "outcome-alpha": "4", "predictor-alpha": "8"},
    ]

    payload = run_tabular_association_correlations(
        _workflow_doc(missing_strategy=missing_strategy),
        source_rows_by_id={"source-alpha": rows},
    ).to_dict()
    row = _result_row(payload)
    messages = " ".join(row["warnings"] + row["errors"])  # type: ignore[operator]

    assert row["n_total"] == 4
    assert row["n_used"] == 2
    assert row["n_missing_outcome"] == 1
    assert row["n_missing_predictor"] == 1
    assert row["n_missing_pairwise"] == 2
    assert row["status"] == expected_status
    assert expected_fragment in messages


def test_nonfinite_invalid_and_bool_numeric_counts_are_reported() -> None:
    rows = [
        {"participant-id": "participant-a", "outcome-alpha": "1", "predictor-alpha": "2"},
        {"participant-id": "participant-b", "outcome-alpha": "2", "predictor-alpha": "4"},
        {"participant-id": "participant-c", "outcome-alpha": float("nan"), "predictor-alpha": "4"},
        {"participant-id": "participant-d", "outcome-alpha": "5", "predictor-alpha": "inf"},
        {"participant-id": "participant-e", "outcome-alpha": "nan", "predictor-alpha": "-infinity"},
        {"participant-id": "participant-f", "outcome-alpha": "7", "predictor-alpha": "not-numeric"},
        {"participant-id": "participant-g", "outcome-alpha": "8", "predictor-alpha": True},
    ]

    payload = run_tabular_association_correlations(
        _workflow_doc(),
        source_rows_by_id={"source-alpha": rows},
    ).to_dict()
    row = _result_row(payload)

    assert row["n_total"] == 7
    assert row["n_used"] == 2
    assert row["n_nonfinite"] == 4
    assert row["n_invalid_numeric"] == 6
    assert row["n_bool_numeric"] == 1
    assert row["statistic_value"] == pytest.approx(1.0)
    assert row["status"] == "warning"
    json.dumps(payload, sort_keys=True, allow_nan=False)


def test_too_few_valid_pairs_and_zero_variance_cases_return_no_statistic() -> None:
    too_few_payload = run_tabular_association_correlations(
        _workflow_doc(),
        source_rows_by_id={"source-alpha": [{"participant-id": "participant-a", "outcome-alpha": "1", "predictor-alpha": "2"}]},
    ).to_dict()
    zero_outcome_payload = run_tabular_association_correlations(
        _workflow_doc(),
        source_rows_by_id={
            "source-alpha": [
                {"participant-id": "participant-a", "outcome-alpha": "1", "predictor-alpha": "2"},
                {"participant-id": "participant-b", "outcome-alpha": "1", "predictor-alpha": "3"},
                {"participant-id": "participant-c", "outcome-alpha": "1", "predictor-alpha": "4"},
            ]
        },
    ).to_dict()
    zero_predictor_payload = run_tabular_association_correlations(
        _workflow_doc(),
        source_rows_by_id={
            "source-alpha": [
                {"participant-id": "participant-a", "outcome-alpha": "1", "predictor-alpha": "2"},
                {"participant-id": "participant-b", "outcome-alpha": "2", "predictor-alpha": "2"},
                {"participant-id": "participant-c", "outcome-alpha": "3", "predictor-alpha": "2"},
            ]
        },
    ).to_dict()

    assert _result_row(too_few_payload)["statistic_value"] is None
    assert _result_row(too_few_payload)["status"] == "error"
    assert too_few_payload["computation_qc_rows"][0]["code"] == "too_few_valid_pairs"
    assert _result_row(zero_outcome_payload)["statistic_value"] is None
    assert zero_outcome_payload["computation_qc_rows"][0]["code"] == "zero_variance_outcome"
    assert _result_row(zero_predictor_payload)["statistic_value"] is None
    assert zero_predictor_payload["computation_qc_rows"][0]["code"] == "zero_variance_predictor"


def test_spearman_zero_variance_rank_cases_return_no_statistic() -> None:
    methods = [
        {
            "method_id": "spearman-alpha",
            "method": "spearman",
            "outcome_ids": ["outcome-alpha"],
            "predictor_ids": ["predictor-alpha"],
        }
    ]
    payload = run_tabular_association_correlations(
        _workflow_doc(methods=methods),
        source_rows_by_id={
            "source-alpha": [
                {"participant-id": "participant-a", "outcome-alpha": "1", "predictor-alpha": "2"},
                {"participant-id": "participant-b", "outcome-alpha": "2", "predictor-alpha": "2"},
                {"participant-id": "participant-c", "outcome-alpha": "3", "predictor-alpha": "2"},
            ]
        },
    ).to_dict()

    assert _result_row(payload)["statistic_value"] is None
    assert payload["computation_qc_rows"][0]["code"] == "zero_variance_predictor_ranks"


def test_deferred_cross_source_and_covariate_adjusted_correlation_pairs() -> None:
    doc = _workflow_doc(
        methods=[
            {
                "method_id": "pearson-alpha",
                "method": "pearson",
                "outcome_ids": ["outcome-beta"],
                "predictor_ids": ["predictor-alpha"],
            },
            {
                "method_id": "pearson-adjusted-alpha",
                "method": "pearson",
                "outcome_ids": ["outcome-alpha"],
                "predictor_ids": ["predictor-alpha"],
                "covariate_ids": ["covariate-alpha"],
            },
        ]
    )
    doc["sources"] = [doc["sources"][0], _source_beta()]  # type: ignore[index]
    doc["outcomes"] = [
        {"variable_id": "outcome-alpha", "source_id": "source-alpha", "column_name": "outcome-alpha"},
        {"variable_id": "outcome-beta", "source_id": "source-beta", "column_name": "outcome-beta"},
    ]

    plan_payload = plan_tabular_association_correlations(doc).to_dict()
    run_payload = run_tabular_association_correlations(
        doc,
        source_rows_by_id={
            "source-alpha": _clean_rows(),
            "source-beta": [{"participant-id": "participant-a", "outcome-beta": "1"}],
        },
    ).to_dict()
    plan_codes = {row["code"] for row in plan_payload["pair_plan_rows"]}
    result_status_by_pair = {row["pair_id"]: row["status"] for row in run_payload["result_rows"]}

    assert "cross_source_correlation_deferred" in plan_codes
    assert "covariate_adjusted_correlation_deferred" in plan_codes
    assert all(status == "deferred" for status in result_status_by_pair.values())
    assert all(row["statistic_value"] is None for row in run_payload["result_rows"])


def test_stratified_repeated_and_mixed_model_correlation_metadata_are_deferred() -> None:
    methods = [
        {
            "method_id": "pearson-alpha",
            "method": "pearson",
            "outcome_ids": ["outcome-alpha"],
            "predictor_ids": ["predictor-alpha"],
            "metadata": {"stratification": ["group-alpha"], "repeated_measures": True, "mixed_model": True},
        }
    ]

    payload = plan_tabular_association_correlations(_workflow_doc(methods=methods)).to_dict()
    row = payload["pair_plan_rows"][0]
    warning_text = " ".join(row["warnings"])

    assert row["status"] == "deferred"
    assert "stratified_correlation_deferred" in warning_text
    assert "repeated_measures_correlation_deferred" in warning_text
    assert "mixed_model_correlation_deferred" in warning_text


def test_tsv_source_loading_path_is_supported_without_writing_outputs(tmp_path: Path) -> None:
    source_path = tmp_path / "source-alpha.tsv"
    source_path.write_text(
        "participant-id\toutcome-alpha\tpredictor-alpha\n"
        "participant-a\t1\t2\n"
        "participant-b\t2\t4\n"
        "participant-c\t3\t6\n",
        encoding="utf-8",
    )

    payload = run_tabular_association_correlations(
        _workflow_doc(source_path=str(source_path), source_format="tsv"),
        source_inventory_specs=[{"source_id": "source-alpha", "path": str(source_path), "source_format": "tsv"}],
    ).to_dict()

    assert _result_row(payload)["statistic_value"] == pytest.approx(1.0)
    assert payload["source_load_rows"][0]["load_status"] == "loaded"
    assert payload["will_write"] is False
    assert payload["output_written"] is False
    assert source_path.exists()


def test_supplied_qc_result_marks_provenance_mode_without_dataframe_backend() -> None:
    payload = run_tabular_association_correlations(
        _workflow_doc(),
        source_rows_by_id={"source-alpha": _clean_rows()},
        qc_result={"workflow_id": "workflow-alpha", "status": "ok"},
    ).to_dict()
    provenance = _provenance(payload)

    assert _result_row(payload)["statistic_value"] == pytest.approx(1.0)
    assert provenance["qc_mode"] == "supplied"
    assert provenance["runtime_backend"] == "records"


def test_result_rows_are_json_safe_tsv_safe_and_omit_future_statistics_fields() -> None:
    result = run_tabular_association_correlations(
        _workflow_doc(),
        source_rows_by_id={"source-alpha": _clean_rows()},
    )
    payload = result.to_dict()
    row = _result_row(payload)
    forbidden_fields = {
        "p_value",
        "q_value",
        "ci_low",
        "ci_high",
        "confidence_interval",
        "fdr",
        "regression_slope",
        "regression_intercept",
        "model_estimate",
        "diagnostic",
        "mixed_model",
    }
    tsv_row = CorrelationAssociationResultRow(
        workflow_id="workflow-alpha",
        pair_id="pearson-alpha::outcome-alpha::predictor-alpha",
        method_id="pearson-alpha",
        method_kind="correlation",
        method_name="pearson",
        correlation_method="pearson",
        family_id="family-alpha",
        source_id="source-alpha",
        outcome_id="outcome-alpha",
        outcome_source_id="source-alpha",
        outcome_column="outcome-alpha",
        predictor_id="predictor-alpha",
        predictor_source_id="source-alpha",
        predictor_column="predictor-alpha",
        n_total=4,
        n_used=4,
        n_missing_outcome=0,
        n_missing_predictor=0,
        n_missing_pairwise=0,
        n_nonfinite=0,
        n_invalid_numeric=0,
        n_bool_numeric=0,
        statistic_name="r",
        statistic_value=1.0,
    ).to_tsv_row()

    json.dumps(payload, sort_keys=True, allow_nan=False)
    assert forbidden_fields.isdisjoint(row)
    assert all(forbidden_fields.isdisjoint(result_row) for result_row in payload["result_rows"])
    assert all(isinstance(value, str) for value in tsv_row.values())
    assert tsv_row["executed"] == "true"
    assert tsv_row["output_written"] == "false"


def test_correlation_mode_writes_nothing(tmp_path: Path) -> None:
    declared_path = tmp_path / "declared-source-alpha.tsv"
    payload = run_tabular_association_correlations(
        _workflow_doc(source_path=str(declared_path), source_format="tsv"),
        source_rows_by_id={"source-alpha": _clean_rows()},
    ).to_dict()
    provenance = _provenance(payload)

    assert payload["will_write"] is False
    assert payload["output_written"] is False
    assert provenance["output_paths_written"] == []
    assert provenance["no_output_paths_written"] is True
    assert not declared_path.exists()


def test_tabular_association_correlations_have_no_forbidden_imports_or_study_specific_constants() -> None:
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
    assert math.isfinite(1.0)
