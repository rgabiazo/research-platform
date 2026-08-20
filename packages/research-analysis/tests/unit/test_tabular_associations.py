from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

import pytest

import research_platform.analysis.tabular_associations as tabular_associations
from research_platform.analysis.tabular_associations import (
    CorrelationSpec,
    CovariateSpec,
    OutcomeSpec,
    PredictorSpec,
    RegressionAssociationSpec,
    TabularAssociationWorkflowSpec,
    parse_tabular_association_workflow_document,
    plan_tabular_association_workflow,
    validate_tabular_association_workflow_document,
)


def _source_alpha(backend: str = "records") -> dict[str, object]:
    return {
        "source_id": "source-alpha",
        "format": "tsv",
        "path": "declared/source-alpha.tsv",
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
                {"column_name": "covariate-alpha", "value_type": "numeric", "role": "covariate"},
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
            },
        },
    }


def _workflow_doc(backend: str = "records") -> dict[str, object]:
    return {
        "workflow_id": "workflow-alpha",
        "name": "workflow-alpha",
        "description": "Neutral synthetic schema-only workflow.",
        "backend": backend,
        "sources": [_source_alpha(backend=backend)],
        "outcomes": [
            {
                "variable_id": "outcome-alpha",
                "source_id": "source-alpha",
                "column_name": "outcome-alpha",
            }
        ],
        "predictors": [
            {
                "variable_id": "predictor-alpha",
                "source_id": "source-alpha",
                "column_name": "predictor-alpha",
            }
        ],
        "covariates": [
            {
                "variable_id": "covariate-alpha",
                "source_id": "source-alpha",
                "column_name": "covariate-alpha",
            }
        ],
        "groupings": [
            {
                "variable_id": "group-alpha",
                "source_id": "source-alpha",
                "column_name": "group-alpha",
            }
        ],
        "repeated_measures": {
            "source_id": "source-alpha",
            "subject_id_column": "participant-id",
            "session_column": "session-label",
            "timepoint_column": "timepoint-label",
            "unit_columns": ["participant-id", "session-label", "timepoint-label"],
        },
        "missing_data_policy": {"strategy": "listwise"},
        "duplicate_subject_policy": {"strategy": "error", "key_columns": ["participant-id"]},
        "nonfinite_policy": {"strategy": "error"},
        "standardization_policy": {"method": "z_score", "variable_ids": ["predictor-alpha"]},
        "transformation_policy": {"method": "rank", "variable_ids": ["outcome-alpha"]},
        "methods": [
            {
                "method_id": "pearson-alpha",
                "method": "pearson",
                "outcome_ids": ["outcome-alpha"],
                "predictor_ids": ["predictor-alpha"],
                "family_id": "family-alpha",
                "output_id": "association-output-alpha",
            }
        ],
        "families": [
            {
                "family_id": "family-alpha",
                "method_ids": ["pearson-alpha"],
            }
        ],
        "multiple_testing": [
            {
                "family_id": "family-alpha",
                "method": "benjamini_hochberg",
            }
        ],
        "outputs": [
            {
                "output_id": "association-output-alpha",
                "output_type": "association_results",
                "planned_fields": [
                    "estimate",
                    "p_value",
                    "q_value",
                    "confidence_interval",
                    "effect_size",
                    "qc_status",
                    "missingness_status",
                    "provenance",
                ],
                "source_method_ids": ["pearson-alpha"],
                "family_ids": ["family-alpha"],
            }
        ],
        "handoffs": [
            {
                "handoff_id": "publication-alpha",
                "handoff_type": "publication",
                "output_ids": ["association-output-alpha"],
                "planned_fields": ["display_columns", "machine_columns"],
            },
            {
                "handoff_id": "visualization-alpha",
                "handoff_type": "visualization",
                "output_ids": ["association-output-alpha"],
                "planned_fields": ["plot_inputs", "report_inputs"],
            },
        ],
    }


def _method_ids(payload: dict[str, object]) -> set[str]:
    return {str(row["method_id"]) for row in payload["method_rows"]}  # type: ignore[index]


def _error_codes(payload: dict[str, object]) -> set[str]:
    return {
        str(row["code"])
        for row in payload["validation_rows"]  # type: ignore[index]
        if row["status"] == "error"
    }


def test_minimal_valid_workflow_spec_returns_plan_only_preview() -> None:
    payload = plan_tabular_association_workflow(_workflow_doc()).to_dict()

    assert payload["valid"] is True
    assert payload["executed"] is False
    assert payload["plan_only"] is True
    assert payload["will_write"] is False
    assert payload["output_written"] is False
    assert payload["status"] == "ok"
    assert payload["source_rows"][0]["source_id"] == "source-alpha"
    assert payload["variable_rows"][0]["variable_role"] == "outcome"
    assert payload["method_rows"][0]["method_name"] == "pearson"
    assert payload["method_rows"][0]["executable"] is False
    assert payload["output_rows"][0]["planned_fields"] == [
        "estimate",
        "p_value",
        "q_value",
        "confidence_interval",
        "effect_size",
        "qc_status",
        "missingness_status",
        "provenance",
    ]
    assert payload["publication_handoff_rows"][0]["handoff_id"] == "publication-alpha"
    assert payload["visualization_handoff_rows"][0]["handoff_id"] == "visualization-alpha"
    json.dumps(payload, sort_keys=True, allow_nan=False)


def test_parse_and_validate_helpers_accept_mapping_documents() -> None:
    workflow = parse_tabular_association_workflow_document(_workflow_doc())
    preview = validate_tabular_association_workflow_document(workflow)

    assert workflow.workflow_id == "workflow-alpha"
    assert preview.valid is True
    assert preview.to_dict()["provenance_rows"][0]["key"] == "schema_version"


def test_dataclass_style_specs_are_supported_without_computation() -> None:
    workflow = TabularAssociationWorkflowSpec(
        workflow_id="workflow-alpha",
        sources=[_source_alpha()],
        outcomes=[OutcomeSpec(variable_id="outcome-alpha", source_id="source-alpha", column_name="outcome-alpha")],
        predictors=[
            PredictorSpec(variable_id="predictor-alpha", source_id="source-alpha", column_name="predictor-alpha")
        ],
        covariates=[
            CovariateSpec(variable_id="covariate-alpha", source_id="source-alpha", column_name="covariate-alpha")
        ],
        methods=[
            CorrelationSpec(
                method_id="spearman-alpha",
                method_name="spearman",
                outcome_ids=["outcome-alpha"],
                predictor_ids=["predictor-alpha"],
            ),
            RegressionAssociationSpec(
                method_id="regression-alpha",
                outcome_ids=["outcome-alpha"],
                predictor_ids=["predictor-alpha"],
                covariate_ids=["covariate-alpha"],
            ),
        ],
    )

    payload = plan_tabular_association_workflow(workflow).to_dict()

    assert payload["valid"] is True
    assert _method_ids(payload) == {"spearman-alpha", "regression-alpha"}
    assert all(row["executable"] is False for row in payload["method_rows"])


def test_multiple_sources_and_variable_roles_are_declared() -> None:
    doc = _workflow_doc()
    source_beta = {
        "source_id": "source-beta",
        "format": "csv",
        "root_ref": "root-beta",
        "schema": {
            "subject_id_column": "participant-id",
            "columns": [
                {"column_name": "participant-id", "value_type": "categorical"},
                {"column_name": "outcome-beta", "value_type": "numeric"},
                {"column_name": "predictor-beta", "value_type": "numeric"},
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

    payload = plan_tabular_association_workflow(doc).to_dict()

    assert payload["valid"] is True
    assert {row["source_id"] for row in payload["source_rows"]} == {"source-alpha", "source-beta"}
    assert {row["variable_role"] for row in payload["variable_rows"]} == {
        "outcome",
        "predictor",
        "covariate",
        "grouping",
    }


def test_session_timepoint_and_repeated_measures_declarations_are_plan_only() -> None:
    payload = plan_tabular_association_workflow(_workflow_doc()).to_dict()

    source_row = payload["source_rows"][0]
    assert source_row["subject_id_column"] == "participant-id"
    assert source_row["session_column"] == "session-label"
    assert source_row["timepoint_column"] == "timepoint-label"
    assert any(row["code"] == "repeated_measures_declared" for row in payload["validation_rows"])


def test_method_declarations_cover_correlation_partial_regression_and_deferred_repeated_measures() -> None:
    doc = _workflow_doc()
    doc["methods"] = [
        {
            "method_id": "pearson-alpha",
            "method": "pearson",
            "outcome_ids": ["outcome-alpha"],
            "predictor_ids": ["predictor-alpha"],
            "family_id": "family-alpha",
        },
        {
            "method_id": "spearman-alpha",
            "method": "spearman",
            "outcome_ids": ["outcome-alpha"],
            "predictor_ids": ["predictor-alpha"],
            "family_id": "family-alpha",
        },
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
            "grouping_ids": ["group-alpha"],
            "family_id": "family-alpha",
        },
        {
            "method_id": "repeated-alpha",
            "method": "mixed_model",
            "outcome_ids": ["outcome-alpha"],
            "predictor_ids": ["predictor-alpha"],
            "covariate_ids": ["covariate-alpha"],
            "grouping_ids": ["group-alpha"],
            "family_id": "family-alpha",
        },
    ]
    doc["families"] = [{"family_id": "family-alpha", "method_ids": [method["method_id"] for method in doc["methods"]]}]  # type: ignore[index]

    payload = plan_tabular_association_workflow(doc).to_dict()

    assert payload["valid"] is True
    method_rows = {row["method_id"]: row for row in payload["method_rows"]}
    assert method_rows["pearson-alpha"]["method_name"] == "pearson"
    assert method_rows["spearman-alpha"]["method_name"] == "spearman"
    assert method_rows["partial-alpha"]["method_name"] == "partial_correlation"
    assert method_rows["regression-alpha"]["method_name"] == "regression"
    assert method_rows["repeated-alpha"]["method_name"] == "mixed_model"
    assert method_rows["repeated-alpha"]["deferred"] is True
    assert method_rows["repeated-alpha"]["planned_only"] is True
    assert any(row["code"] == "deferred_repeated_measures_method" for row in payload["validation_rows"])


@pytest.mark.parametrize(
    ("policy_key", "bad_policy", "message_fragment"),
    [
        ("missing_data_policy", {"strategy": "unsupported"}, "missing-data policy"),
        ("duplicate_subject_policy", {"strategy": "unsupported"}, "duplicate-subject policy"),
        ("nonfinite_policy", {"strategy": "unsupported"}, "non-finite policy"),
    ],
)
def test_core_policy_validation_rejects_unknown_values(
    policy_key: str,
    bad_policy: dict[str, str],
    message_fragment: str,
) -> None:
    doc = _workflow_doc()
    doc[policy_key] = bad_policy

    payload = plan_tabular_association_workflow(doc).to_dict()

    assert payload["valid"] is False
    assert message_fragment in payload["errors"][0]


def test_categorical_numeric_standardization_and_transformation_policies_are_declared() -> None:
    payload = plan_tabular_association_workflow(_workflow_doc()).to_dict()

    source_row = payload["source_rows"][0]
    assert source_row["categorical_validation"]["policy"] == "strict"
    assert source_row["categorical_validation"]["allowed_values"] == {"group-alpha": ["group-alpha"]}
    assert source_row["numeric_validation"]["policy"] == "strict"
    assert any(row["code"] == "standardization_policy" for row in payload["validation_rows"])
    assert any(row["code"] == "transformation_policy" for row in payload["validation_rows"])


def test_multiple_testing_publication_and_visualization_handoffs_are_metadata_only() -> None:
    payload = plan_tabular_association_workflow(_workflow_doc()).to_dict()

    family_rows = payload["family_rows"]
    assert any(row["row_type"] == "association_family" for row in family_rows)
    assert any(row["row_type"] == "multiple_testing" and row["method"] == "benjamini_hochberg" for row in family_rows)
    assert payload["publication_handoff_rows"][0]["will_write"] is False
    assert payload["visualization_handoff_rows"][0]["output_written"] is False


def test_backend_values_are_schema_only_and_do_not_import_dataframe_libraries() -> None:
    for backend in ("records", "polars", "pandas"):
        before = set(sys.modules)
        payload = plan_tabular_association_workflow(_workflow_doc(backend=backend)).to_dict()
        imported_during_plan = set(sys.modules) - before

        assert payload["valid"] is True
        assert payload["source_rows"][0]["backend"] == backend
        assert "polars" not in imported_during_plan
        assert "pandas" not in imported_during_plan


def test_invalid_backend_and_invalid_method_are_rejected() -> None:
    bad_backend_doc = _workflow_doc(backend="unsupported")
    bad_method_doc = _workflow_doc()
    bad_method_doc["methods"] = [{"method_id": "method-alpha", "method": "unsupported"}]

    backend_payload = plan_tabular_association_workflow(bad_backend_doc).to_dict()
    method_payload = plan_tabular_association_workflow(bad_method_doc).to_dict()

    assert backend_payload["valid"] is False
    assert "Unsupported backend" in backend_payload["errors"][0]
    assert method_payload["valid"] is False
    assert "Unsupported association method" in method_payload["errors"][0]


def test_missing_required_subject_outcome_and_predictor_are_rejected() -> None:
    missing_subject_doc = _workflow_doc()
    missing_subject_doc["sources"][0]["schema"]["subject_id_column"] = "missing-participant-id"  # type: ignore[index]
    missing_outcome_doc = _workflow_doc()
    missing_outcome_doc["outcomes"] = []
    missing_predictor_doc = _workflow_doc()
    missing_predictor_doc["predictors"] = []

    assert "missing_subject_column" in _error_codes(plan_tabular_association_workflow(missing_subject_doc).to_dict())
    assert "missing_outcome" in _error_codes(plan_tabular_association_workflow(missing_outcome_doc).to_dict())
    assert "missing_predictor" in _error_codes(plan_tabular_association_workflow(missing_predictor_doc).to_dict())


def test_duplicate_source_ids_and_duplicate_output_ids_are_rejected() -> None:
    duplicate_sources_doc = _workflow_doc()
    duplicate_sources_doc["sources"] = [_source_alpha(), _source_alpha()]
    duplicate_outputs_doc = _workflow_doc()
    duplicate_outputs_doc["outputs"] = [
        duplicate_outputs_doc["outputs"][0],  # type: ignore[index]
        duplicate_outputs_doc["outputs"][0],  # type: ignore[index]
    ]

    assert "duplicate_source_id" in _error_codes(plan_tabular_association_workflow(duplicate_sources_doc).to_dict())
    assert "duplicate_output_id" in _error_codes(plan_tabular_association_workflow(duplicate_outputs_doc).to_dict())


def test_unknown_covariate_grouping_and_repeated_measure_columns_are_rejected() -> None:
    doc = _workflow_doc()
    doc["covariates"] = [
        {"variable_id": "covariate-alpha", "source_id": "source-alpha", "column_name": "missing-covariate"}
    ]
    doc["groupings"] = [
        {"variable_id": "group-alpha", "source_id": "source-alpha", "column_name": "missing-group"}
    ]
    doc["repeated_measures"]["timepoint_column"] = "missing-timepoint"  # type: ignore[index]

    error_codes = _error_codes(plan_tabular_association_workflow(doc).to_dict())

    assert "unknown_variable_column" in error_codes
    assert "unknown_repeated_measures_column" in error_codes


def test_json_safety_and_absence_of_computed_result_keys() -> None:
    payload = plan_tabular_association_workflow(_workflow_doc()).to_dict()
    forbidden_computed_keys = {
        "r",
        "p_value",
        "q_value",
        "ci_low",
        "ci_high",
        "effect_size",
        "diagnostics",
        "model_estimate",
    }

    json.dumps(payload, sort_keys=True, allow_nan=False)
    row_groups = (
        "validation_rows",
        "source_rows",
        "column_rows",
        "variable_rows",
        "method_rows",
        "family_rows",
        "output_rows",
        "publication_handoff_rows",
        "visualization_handoff_rows",
        "provenance_rows",
    )
    for group in row_groups:
        for row in payload[group]:
            assert forbidden_computed_keys.isdisjoint(row.keys())


def test_plan_mode_writes_nothing(tmp_path: Path) -> None:
    source_path = tmp_path / "declared-source.tsv"
    handoff_target = tmp_path / "declared-handoff"
    doc = _workflow_doc()
    doc["sources"][0]["path"] = str(source_path)  # type: ignore[index]
    doc["handoffs"][0]["target"] = str(handoff_target)  # type: ignore[index]

    payload = plan_tabular_association_workflow(doc).to_dict()

    assert payload["valid"] is True
    assert payload["will_write"] is False
    assert payload["output_written"] is False
    assert not source_path.exists()
    assert not handoff_target.exists()


def test_new_module_has_no_forbidden_imports_or_study_specific_constants() -> None:
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
