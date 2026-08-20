from __future__ import annotations

import ast
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

import research_platform.analysis.tabular_associations as tabular_associations
from research_platform.analysis.tabular_associations import (
    ContrastMetadataSpec,
    FixedEffectTermSpec,
    ModelContrastResultRow,
    ModelDesignMetadataSpec,
    ModelFitSummaryRow,
    ModelFixedEffectResultRow,
    ModelPlannedComparisonResultRow,
    ModelRandomEffectResultRow,
    ModelResultQcRow,
    ModelVarianceComponentResultRow,
    PlannedComparisonSpec,
    RandomEffectTermSpec,
    normalize_tabular_association_model_result_rows,
    plan_tabular_association_model_results,
    run_tabular_association_multiplicity,
    validate_tabular_association_model_result_rows,
)


def _model_result_row(**updates: object) -> dict[str, object]:
    row: dict[str, object] = {
        "workflow_id": "workflow-alpha",
        "result_row_id": "row-alpha",
        "result_id": "result-alpha",
        "result_kind": "fixed_effect",
        "model_id": "model-alpha",
        "model_plan_id": "model-plan-alpha",
        "method_id": "method-alpha",
        "method_name": "mixed_model",
        "method_kind": "mixed_model",
        "family_id": "family-alpha",
        "source_id": "source-alpha",
        "outcome_id": "outcome-alpha",
        "outcome_column": "outcome-alpha",
        "predictor_id": "predictor-alpha",
        "predictor_column": "predictor-alpha",
        "covariate_ids": ["covariate-alpha"],
        "covariate_columns": ["covariate-alpha"],
        "term_id": "term-alpha",
        "term_label": "term-alpha",
        "statistic_name": "coefficient",
        "statistic_value": 0.25,
        "coefficient": 0.25,
        "standard_error": 0.10,
        "p_value": 0.02,
        "q_value": 0.03,
        "ci_low": 0.05,
        "ci_high": 0.45,
        "confidence_level": 0.95,
        "effect_size": 0.50,
        "effect_size_name": "effect-alpha",
        "degrees_of_freedom": 4,
        "observation_count": 8,
        "participant_count": 4,
        "cluster_count": 2,
        "status": "supplied",
        "warnings": ["warning-alpha"],
        "errors": [],
        "metadata": {"metadata-alpha": "preserved"},
    }
    row.update(updates)
    return row


def _workflow_doc() -> dict[str, object]:
    return {
        "workflow_id": "workflow-alpha",
        "sources": [
            {
                "source_id": "source-alpha",
                "schema": {
                    "subject_id_column": "participant-alpha",
                    "columns": [
                        {"column_name": "participant-alpha", "value_type": "categorical"},
                        {"column_name": "outcome-alpha", "value_type": "numeric"},
                        {"column_name": "predictor-alpha", "value_type": "numeric"},
                    ],
                },
            }
        ],
        "outcomes": [{"variable_id": "outcome-alpha", "source_id": "source-alpha", "column_name": "outcome-alpha"}],
        "predictors": [
            {"variable_id": "predictor-alpha", "source_id": "source-alpha", "column_name": "predictor-alpha"}
        ],
        "methods": [
            {
                "method_id": "method-alpha",
                "method": "pearson",
                "outcome_ids": ["outcome-alpha"],
                "predictor_ids": ["predictor-alpha"],
                "family_id": "family-alpha",
            }
        ],
        "families": [{"family_id": "family-alpha", "method_ids": ["method-alpha"]}],
        "multiple_testing": [{"family_id": "family-alpha", "method": "benjamini_hochberg"}],
    }


def _codes(payload: dict[str, object]) -> set[str]:
    return {str(row["code"]) for row in payload["qc_rows"]}  # type: ignore[index]


@pytest.mark.parametrize(
    ("row_class", "result_kind", "updates"),
    [
        (ModelFitSummaryRow, "model_fit_summary", {"model_fit_metric_name": "aic", "model_fit_metric_value": 10.5}),
        (ModelFixedEffectResultRow, "fixed_effect", {"term_id": "term-alpha"}),
        (ModelRandomEffectResultRow, "random_effect", {"term_id": "term-alpha", "grouping_id": "group-alpha"}),
        (ModelVarianceComponentResultRow, "variance_component", {"term_id": "term-alpha", "cluster_id": "cluster-alpha"}),
        (ModelPlannedComparisonResultRow, "planned_comparison", {"comparison_id": "comparison-alpha"}),
        (ModelContrastResultRow, "contrast", {"contrast_id": "contrast-alpha"}),
    ],
)
def test_model_result_kind_rows_serialize_as_supplied_only_and_no_write(
    row_class: type[ModelFitSummaryRow],
    result_kind: str,
    updates: dict[str, object],
) -> None:
    payload = normalize_tabular_association_model_result_rows([_model_result_row(result_kind=result_kind, **updates)]).to_dict()
    row = payload["model_result_rows"][0]
    direct_row = row_class(
        workflow_id="workflow-alpha",
        result_row_id="row-alpha",
        result_id="result-alpha",
        model_id="model-alpha",
        model_plan_id="model-plan-alpha",
        method_id="method-alpha",
        family_id="family-alpha",
        p_value=0.02,
        **updates,
    )
    tsv_row = direct_row.to_tsv_row()

    assert row["result_kind"] == result_kind
    assert row["supplied_only"] is True
    assert row["computed_by_research_analysis"] is False
    assert row["model_fitting_performed"] is False
    assert row["runtime_backend"] == "records"
    assert row["will_write"] is False
    assert row["output_written"] is False
    assert row["no_output_written"] is True
    assert row["output_paths_written"] == []
    assert payload["will_write"] is False
    assert payload["output_written"] is False
    assert payload["no_output_written"] is True
    assert payload["output_paths_written"] == []
    assert all(isinstance(value, str) for value in tsv_row.values())
    assert tsv_row["supplied_only"] == "true"
    assert tsv_row["computed_by_research_analysis"] == "false"
    json.dumps(payload, sort_keys=True, allow_nan=False)


def test_supplied_statistics_are_preserved_without_computation() -> None:
    payload = normalize_tabular_association_model_result_rows([_model_result_row()]).to_dict()
    row = payload["model_result_rows"][0]
    provenance = {str(item["key"]): item["value"] for item in payload["provenance_rows"]}  # type: ignore[index]

    assert row["p_value"] == pytest.approx(0.02)
    assert row["q_value"] == pytest.approx(0.03)
    assert row["ci_low"] == pytest.approx(0.05)
    assert row["ci_high"] == pytest.approx(0.45)
    assert row["confidence_level"] == pytest.approx(0.95)
    assert row["coefficient"] == pytest.approx(0.25)
    assert row["standard_error"] == pytest.approx(0.10)
    assert row["effect_size"] == pytest.approx(0.50)
    assert row["metadata"] == {"metadata-alpha": "preserved"}
    assert provenance["model_results_contract_version"] == (
        tabular_associations.TABULAR_ASSOCIATION_MODEL_RESULTS_CONTRACT_VERSION
    )
    assert provenance["supplied_only"] is True
    assert provenance["computed_by_research_analysis"] is False
    assert provenance["model_fitting_performed"] is False
    assert provenance["will_write"] is False
    assert provenance["output_written"] is False
    assert provenance["no_output_written"] is True
    assert provenance["output_paths_written"] == []
    assert "supplied_only_model_result" in _codes(payload)
    assert "model_fitting_not_performed" in _codes(payload)
    json.dumps(payload, sort_keys=True, allow_nan=False)


def test_malformed_numeric_p_q_and_interval_fields_emit_qc_errors() -> None:
    payload = validate_tabular_association_model_result_rows(
        [
            _model_result_row(result_row_id="row-alpha", statistic_value="not-numeric"),
            _model_result_row(result_row_id="row-beta", p_value=1.01),
            _model_result_row(result_row_id="row-gamma", q_value=-0.01),
            _model_result_row(result_row_id="row-delta", ci_low=2.0, ci_high=1.0),
            _model_result_row(result_row_id="row-epsilon", confidence_level=2.0),
            _model_result_row(result_row_id="row-zeta", standard_error=math.inf),
        ]
    ).to_dict()

    assert payload["valid"] is False
    assert payload["status"] == "error"
    assert payload["valid_row_count"] == 0
    assert payload["invalid_row_count"] == 6
    assert {
        "malformed_supplied_numeric_field",
        "invalid_supplied_p_value",
        "invalid_supplied_q_value",
        "invalid_supplied_confidence_interval",
    }.issubset(_codes(payload))
    json.dumps(payload, sort_keys=True, allow_nan=False)


def test_missing_identifiers_unsupported_kinds_and_malformed_fields_emit_qc_errors() -> None:
    payload = validate_tabular_association_model_result_rows(
        [
            {"result_kind": "fixed_effect", "model_id": "model-alpha"},
            _model_result_row(result_row_id="row-beta", result_kind="unsupported-kind"),
            _model_result_row(result_row_id="row-gamma", status=[]),
            _model_result_row(result_row_id="row-delta", warnings={"bad": True}),
            _model_result_row(result_row_id="row-epsilon", metadata=["bad"]),
        ]
    ).to_dict()
    single_mapping_payload = validate_tabular_association_model_result_rows(_model_result_row()).to_dict()  # type: ignore[arg-type]
    string_payload = validate_tabular_association_model_result_rows("bad-input").to_dict()  # type: ignore[arg-type]

    assert payload["valid"] is False
    assert "missing_required_identifier" in _codes(payload)
    assert "unsupported_model_result_kind" in _codes(payload)
    assert "malformed_model_result_row" in _codes(payload)
    assert single_mapping_payload["invalid_row_count"] == 1
    assert string_payload["invalid_row_count"] == 1
    assert "malformed_model_result_row" in _codes(single_mapping_payload)
    assert "malformed_model_result_row" in _codes(string_payload)
    json.dumps(payload, sort_keys=True, allow_nan=False)


def test_model_plan_and_metadata_reference_validation_emit_qc_rows() -> None:
    design = ModelDesignMetadataSpec(
        fixed_effect_terms=(FixedEffectTermSpec(term_id="term-alpha"),),
        random_effect_terms=(RandomEffectTermSpec(term_id="term-beta"),),
        planned_comparisons=(PlannedComparisonSpec(comparison_id="comparison-alpha"),),
        contrast_metadata=(ContrastMetadataSpec(contrast_id="contrast-alpha"),),
    )
    payload = validate_tabular_association_model_result_rows(
        [
            _model_result_row(result_row_id="row-alpha", model_plan_id="model-plan-missing"),
            _model_result_row(result_row_id="row-beta", term_id="term-missing"),
            _model_result_row(result_row_id="row-gamma", result_kind="planned_comparison", comparison_id="comparison-missing"),
            _model_result_row(result_row_id="row-delta", result_kind="contrast", contrast_id="contrast-missing"),
        ],
        model_plan_rows=[{"model_plan_id": "model-plan-alpha"}],
        model_design_metadata=design,
    ).to_dict()

    assert payload["valid"] is False
    assert {
        "missing_model_plan_reference",
        "unknown_model_term_reference",
        "unknown_model_comparison_reference",
        "unknown_model_contrast_reference",
    }.issubset(_codes(payload))
    json.dumps(payload, sort_keys=True, allow_nan=False)


def test_normalized_rows_are_multiplicity_compatible_without_contract_q_values() -> None:
    contract = normalize_tabular_association_model_result_rows(
        [
            _model_result_row(result_row_id="row-alpha", result_id="result-alpha", p_value=0.01, q_value=None),
            _model_result_row(result_row_id="row-beta", result_id="result-beta", p_value=0.04, q_value=None),
        ]
    ).to_dict()
    rows = contract["model_result_rows"]
    multiplicity = run_tabular_association_multiplicity(_workflow_doc(), result_rows=rows).to_dict()  # type: ignore[arg-type]

    assert [row["q_value"] for row in rows] == [None, None]  # type: ignore[index]
    assert [row["p_value"] for row in rows] == [0.01, 0.04]  # type: ignore[index]
    assert [row["q_value"] for row in multiplicity["result_rows"]] == pytest.approx([0.02, 0.04])  # type: ignore[index]
    assert contract["computed_by_research_analysis"] is False
    assert contract["model_fitting_performed"] is False
    json.dumps(contract, sort_keys=True, allow_nan=False)
    json.dumps(multiplicity, sort_keys=True, allow_nan=False)


def test_model_result_contracts_omit_model_objects_residuals_diagnostics_and_write_nothing(tmp_path: Path) -> None:
    payload = normalize_tabular_association_model_result_rows(
        [
            _model_result_row(
                fitted_model={"bad": "object"},
                residuals=[1, 2],
                diagnostics={"bad": True},
                computed_by_research_analysis=True,
                model_fitting_performed=True,
            )
        ]
    ).to_dict()
    plan = plan_tabular_association_model_results([_model_result_row()]).to_dict()
    row = payload["model_result_rows"][0]
    forbidden_fields = {
        "fitted_model",
        "residuals",
        "diagnostics",
        "model_diagnostics",
        "computed_residuals",
    }

    assert forbidden_fields.isdisjoint(row)
    assert row["computed_by_research_analysis"] is False
    assert row["model_fitting_performed"] is False
    assert payload["computed_by_research_analysis"] is False
    assert payload["model_fitting_performed"] is False
    assert payload["output_paths_written"] == []
    assert payload["no_output_written"] is True
    assert plan["executed"] is False
    assert plan["plan_only"] is True
    assert plan["output_paths_written"] == []
    assert not any(tmp_path.iterdir())
    json.dumps(payload, sort_keys=True, allow_nan=False)


def test_model_result_qc_rows_are_tsv_safe() -> None:
    row = ModelResultQcRow(
        workflow_id="workflow-alpha",
        input_row_index=0,
        result_row_id="row-alpha",
        result_kind="fixed_effect",
        model_id="model-alpha",
        status="warning",
        code="missing_multiplicity_family_id",
        message="Family id is missing.",
        metadata={"fields": ["family_id"]},
    )
    tsv_row = row.to_tsv_row()

    assert all(isinstance(value, str) for value in tsv_row.values())
    assert tsv_row["supplied_only"] == "true"
    assert tsv_row["computed_by_research_analysis"] == "false"
    assert tsv_row["metadata"] == '{"fields":["family_id"]}'


def test_model_results_source_and_import_have_no_forbidden_dependencies_or_specific_constants() -> None:
    forbidden_modules = (
        "numpy",
        "pandas",
        "polars",
        "scipy",
        "sklearn",
        "statsmodels",
        "research_platform.io",
        "research_platform.viz",
        "research_platform.neuro",
        "research_platform.core",
        "research_platform.bids",
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

    env = dict(os.environ)
    env["PYTHONPATH"] = "packages/research-analysis/src"
    script = (
        "import sys\n"
        "before=set(sys.modules)\n"
        "import research_platform.analysis.tabular_associations\n"
        f"forbidden={forbidden_modules!r}\n"
        "imported=set(sys.modules)-before\n"
        "bad=[name for name in imported for forbidden_name in forbidden "
        "if name == forbidden_name or name.startswith(forbidden_name + '.')]\n"
        "raise SystemExit(1 if bad else 0)\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(tabular_associations.__file__).parents[5],
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
