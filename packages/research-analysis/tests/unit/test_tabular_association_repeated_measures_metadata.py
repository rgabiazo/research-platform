from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import research_platform.analysis.tabular_associations as tabular_associations
from research_platform.analysis.tabular_associations import (
    BetweenSubjectFactorSpec,
    CategoricalCodingSpec,
    ClusterTermSpec,
    ContrastMetadataSpec,
    FixedEffectTermSpec,
    GroupingFactorSpec,
    ModelDesignMetadataSpec,
    ModelFormulaMetadataSpec,
    PlannedComparisonSpec,
    RandomEffectTermSpec,
    RandomInterceptSpec,
    RandomSlopeSpec,
    RepeatedFactorSpec,
    RepeatedMeasuresDesignQcRow,
    TimepointRoleSpec,
    WithinSubjectFactorSpec,
    plan_tabular_association_repeated_measures,
    run_tabular_association_repeated_measures_design_qc,
)


def _source_alpha() -> dict[str, object]:
    return {
        "source_id": "source-alpha",
        "format": "tsv",
        "schema": {
            "subject_id_column": "participant-alpha",
            "timepoint_column": "timepoint-alpha",
            "columns": [
                {"column_name": "participant-alpha", "value_type": "categorical", "role": "participant"},
                {"column_name": "timepoint-alpha", "value_type": "categorical", "role": "timepoint"},
                {"column_name": "outcome-alpha", "value_type": "numeric", "role": "outcome"},
                {"column_name": "predictor-alpha", "value_type": "numeric", "role": "predictor"},
                {"column_name": "covariate-alpha", "value_type": "numeric", "role": "covariate"},
                {"column_name": "group-alpha", "value_type": "categorical", "role": "grouping"},
                {"column_name": "cluster-alpha", "value_type": "categorical", "role": "cluster"},
                {"column_name": "column-alpha", "value_type": "numeric", "role": "metadata"},
            ],
        },
    }


def _model_design_metadata(*, missing_metadata_column: bool = False) -> dict[str, object]:
    fixed_column = "column-alpha" if missing_metadata_column else "predictor-alpha"
    return {
        "model_design_id": "model-alpha",
        "fixed_effect_terms": [
            {
                "term_id": "term-alpha",
                "variable_ids": ["predictor-alpha"],
                "column_names": [fixed_column],
                "factor_ids": ["factor-alpha"],
                "coding_ids": ["coding-alpha"],
                "metadata": {"fixed-note": "preserved"},
            }
        ],
        "random_effect_terms": [
            {
                "term_id": "term-beta",
                "random_intercept_ids": ["intercept-alpha"],
                "random_slope_ids": ["slope-alpha"],
                "variable_ids": ["predictor-alpha"],
                "factor_ids": ["factor-alpha"],
                "grouping_ids": ["grouping-alpha"],
                "cluster_ids": ["cluster-alpha"],
                "metadata": {"random-term-note": "preserved"},
            }
        ],
        "random_intercepts": [
            {
                "intercept_id": "intercept-alpha",
                "grouping_ids": ["grouping-alpha"],
                "grouping_columns": ["participant-alpha"],
                "metadata": {"intercept-note": "preserved"},
            }
        ],
        "random_slopes": [
            {
                "slope_id": "slope-alpha",
                "variable_ids": ["predictor-alpha"],
                "column_names": ["predictor-alpha"],
                "factor_ids": ["factor-alpha"],
                "grouping_ids": ["grouping-alpha"],
                "grouping_columns": ["participant-alpha"],
                "metadata": {"slope-note": "preserved"},
            }
        ],
        "within_subject_factors": [
            {
                "factor_id": "factor-beta",
                "column_name": "timepoint-alpha",
                "repeated_factor_id": "factor-alpha",
                "metadata": {"within-note": "preserved"},
            }
        ],
        "between_subject_factors": [
            {
                "factor_id": "factor-gamma",
                "column_name": "group-alpha",
                "variable_id": "group-alpha",
                "metadata": {"between-note": "preserved"},
            }
        ],
        "grouping_factors": [
            {
                "grouping_id": "grouping-alpha",
                "variable_id": "group-alpha",
                "column_name": "group-alpha",
                "metadata": {"grouping-note": "preserved"},
            }
        ],
        "cluster_terms": [
            {
                "cluster_id": "cluster-alpha",
                "column_name": "cluster-alpha",
                "grouping_id": "grouping-alpha",
                "metadata": {"cluster-note": "preserved"},
            }
        ],
        "timepoint_roles": [
            {
                "role_id": "timepoint-role-alpha",
                "column_name": "timepoint-alpha",
                "factor_id": "factor-alpha",
                "role": "timepoint-role-alpha",
                "metadata": {"timepoint-note": "preserved"},
            }
        ],
        "categorical_coding": [
            {
                "coding_id": "coding-alpha",
                "target_id": "factor-alpha",
                "factor_id": "factor-alpha",
                "column_name": "timepoint-alpha",
                "scheme": "treatment",
                "reference_level": "timepoint-alpha",
                "levels": ["timepoint-alpha", "timepoint-beta"],
                "metadata": {"coding-note": "preserved"},
            }
        ],
        "formula_metadata": {
            "formula_id": "formula-alpha",
            "formula_like": "outcome-alpha ~ predictor-alpha + covariate-alpha",
            "design_intent": "metadata-only",
            "variable_ids": ["outcome-alpha", "predictor-alpha"],
            "factor_ids": ["factor-alpha"],
            "metadata": {"formula-note": "preserved"},
        },
        "formula_like": "outcome-alpha ~ predictor-alpha + covariate-alpha",
        "planned_comparisons": [
            {
                "comparison_id": "comparison-alpha",
                "factor_ids": ["factor-alpha"],
                "variable_ids": ["outcome-alpha"],
                "grouping_ids": ["grouping-alpha"],
                "cluster_ids": ["cluster-alpha"],
                "coding_ids": ["coding-alpha"],
                "contrast_metadata_ids": ["contrast-alpha"],
                "metadata": {"comparison-note": "preserved"},
            }
        ],
        "contrast_metadata": [
            {
                "contrast_id": "contrast-alpha",
                "comparison_ids": ["comparison-alpha"],
                "factor_ids": ["factor-alpha"],
                "variable_ids": ["outcome-alpha"],
                "coding_ids": ["coding-alpha"],
                "label": "contrast-alpha",
                "metadata": {"contrast-note": "preserved"},
            }
        ],
        "model_family": "family-alpha",
        "link_function": "identity",
        "metadata": {"design-note": "preserved"},
    }


def _workflow_doc(*, missing_metadata_column: bool = False) -> dict[str, object]:
    return {
        "workflow_id": "workflow-alpha",
        "sources": [_source_alpha()],
        "outcomes": [{"variable_id": "outcome-alpha", "source_id": "source-alpha", "column_name": "outcome-alpha"}],
        "predictors": [
            {"variable_id": "predictor-alpha", "source_id": "source-alpha", "column_name": "predictor-alpha"}
        ],
        "covariates": [
            {"variable_id": "covariate-alpha", "source_id": "source-alpha", "column_name": "covariate-alpha"}
        ],
        "groupings": [{"variable_id": "group-alpha", "source_id": "source-alpha", "column_name": "group-alpha"}],
        "repeated_measures": {
            "source_id": "source-alpha",
            "subject_id_column": "participant-alpha",
            "timepoint_column": "timepoint-alpha",
            "unit_columns": ["participant-alpha", "timepoint-alpha"],
            "metadata": {
                "model_design": {
                    "repeated_factors": [
                        {
                            "factor_id": "factor-alpha",
                            "column_name": "timepoint-alpha",
                            "levels": ["timepoint-alpha", "timepoint-beta"],
                            "metadata": {"repeated-note": "preserved"},
                        }
                    ]
                }
            },
        },
        "methods": [
            {
                "method_id": "method-alpha",
                "method": "mixed_model",
                "outcome_ids": ["outcome-alpha"],
                "predictor_ids": ["predictor-alpha"],
                "covariate_ids": ["covariate-alpha"],
                "grouping_ids": ["group-alpha"],
                "metadata": {
                    "legacy-freeform": {"kept": True},
                    "model_design": _model_design_metadata(missing_metadata_column=missing_metadata_column),
                },
            }
        ],
    }


def _rows(*, include_column_alpha: bool = True) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for participant, cluster in (("participant-a", "cluster-alpha"), ("participant-b", "cluster-beta")):
        for timepoint, outcome in (("timepoint-alpha", "1"), ("timepoint-beta", "2")):
            row: dict[str, object] = {
                "participant-alpha": participant,
                "timepoint-alpha": timepoint,
                "outcome-alpha": outcome,
                "predictor-alpha": "2",
                "covariate-alpha": "3",
                "group-alpha": "group-alpha",
                "cluster-alpha": cluster,
            }
            if include_column_alpha:
                row["column-alpha"] = "4"
            rows.append(row)
    return rows


def _codes(payload: dict[str, object]) -> set[str]:
    return {str(row["code"]) for row in payload["qc_rows"]}  # type: ignore[index]


def test_public_metadata_specs_validate_and_serialize_as_metadata_only() -> None:
    design = ModelDesignMetadataSpec(
        model_design_id="model-alpha",
        fixed_effect_terms=[
            FixedEffectTermSpec(term_id="term-alpha", variable_ids=("predictor-alpha",), metadata={"kept": True})
        ],
        random_effect_terms=[RandomEffectTermSpec(term_id="term-beta", random_intercept_ids=("intercept-alpha",))],
        random_intercepts=[RandomInterceptSpec(intercept_id="intercept-alpha", metadata={"kept": "intercept"})],
        random_slopes=[RandomSlopeSpec(slope_id="slope-alpha", metadata={"kept": "slope"})],
        repeated_factors=[RepeatedFactorSpec(factor_id="factor-alpha", column_name="timepoint-alpha")],
        within_subject_factors=[WithinSubjectFactorSpec(factor_id="factor-beta", column_name="timepoint-alpha")],
        between_subject_factors=[BetweenSubjectFactorSpec(factor_id="factor-gamma", column_name="group-alpha")],
        grouping_factors=[GroupingFactorSpec(grouping_id="grouping-alpha", column_name="group-alpha")],
        cluster_terms=[ClusterTermSpec(cluster_id="cluster-alpha", column_name="cluster-alpha")],
        timepoint_roles=[TimepointRoleSpec(role_id="timepoint-role-alpha", column_name="timepoint-alpha")],
        categorical_coding=[CategoricalCodingSpec(coding_id="coding-alpha", target_id="factor-alpha")],
        formula_metadata=ModelFormulaMetadataSpec(
            formula_id="formula-alpha",
            formula_like="outcome-alpha ~ predictor-alpha",
            metadata={"kept": "formula"},
        ),
        planned_comparisons=[PlannedComparisonSpec(comparison_id="comparison-alpha", metadata={"kept": "comparison"})],
        contrast_metadata=[ContrastMetadataSpec(contrast_id="contrast-alpha", metadata={"kept": "contrast"})],
        model_family="family-alpha",
        link_function="identity",
        metadata={"design": "kept"},
    )

    payload = design.to_dict()
    assert payload["metadata_only"] is True
    assert payload["model_fitting_deferred"] is True
    assert payload["fixed_effect_terms"][0]["metadata"]["kept"] is True
    assert payload["random_intercepts"][0]["metadata"]["kept"] == "intercept"
    assert payload["random_slopes"][0]["metadata"]["kept"] == "slope"
    assert payload["repeated_factors"][0]["column_name"] == "timepoint-alpha"
    assert payload["within_subject_factors"][0]["column_name"] == "timepoint-alpha"
    assert payload["between_subject_factors"][0]["column_name"] == "group-alpha"
    assert payload["grouping_factors"][0]["column_name"] == "group-alpha"
    assert payload["cluster_terms"][0]["column_name"] == "cluster-alpha"
    assert payload["timepoint_roles"][0]["column_name"] == "timepoint-alpha"
    assert payload["categorical_coding"][0]["target_id"] == "factor-alpha"
    assert payload["formula_metadata"]["metadata"]["kept"] == "formula"
    assert payload["planned_comparisons"][0]["metadata"]["kept"] == "comparison"
    assert payload["contrast_metadata"][0]["metadata"]["kept"] == "contrast"
    json.dumps(payload, sort_keys=True, allow_nan=False)


def test_model_plan_rows_include_normalized_metadata_and_remain_deferred() -> None:
    payload = plan_tabular_association_repeated_measures(_workflow_doc()).to_dict()
    row = payload["model_plan_rows"][0]

    assert row["model_design_id"] == "model-alpha"
    assert row["fixed_effect_term_ids"] == ["term-alpha"]
    assert row["random_effect_term_ids"] == ["term-beta"]
    assert row["random_intercept_ids"] == ["intercept-alpha"]
    assert row["random_slope_ids"] == ["slope-alpha"]
    assert row["repeated_factor_ids"] == ["factor-alpha"]
    assert row["repeated_factor_columns"] == ["timepoint-alpha"]
    assert row["within_subject_factor_ids"] == ["factor-beta"]
    assert row["within_subject_factor_columns"] == ["timepoint-alpha"]
    assert row["between_subject_factor_ids"] == ["factor-gamma"]
    assert row["between_subject_factor_columns"] == ["group-alpha"]
    assert row["grouping_factor_ids"] == ["grouping-alpha"]
    assert row["grouping_factor_columns"] == ["group-alpha"]
    assert row["cluster_term_ids"] == ["cluster-alpha"]
    assert row["cluster_columns"] == ["cluster-alpha"]
    assert row["timepoint_role_ids"] == ["timepoint-role-alpha"]
    assert row["timepoint_columns"] == ["timepoint-alpha"]
    assert row["categorical_coding_ids"] == ["coding-alpha"]
    assert row["formula_like"] == "outcome-alpha ~ predictor-alpha + covariate-alpha"
    assert row["formula_metadata"]["design_intent"] == "metadata-only"
    assert row["planned_comparison_ids"] == ["comparison-alpha"]
    assert row["planned_comparison_metadata"]["planned_comparisons"][0]["metadata"]["comparison-note"] == "preserved"
    assert row["contrast_metadata_ids"] == ["contrast-alpha"]
    assert row["contrast_metadata"]["contrast_metadata"][0]["metadata"]["contrast-note"] == "preserved"
    assert row["model_family"] == "family-alpha"
    assert row["link_function"] == "identity"
    assert row["method_metadata"]["legacy-freeform"] == {"kept": True}
    assert row["repeated_measures_metadata"]["model_design"]["repeated_factors"][0]["metadata"]["repeated-note"] == "preserved"
    assert row["metadata_only"] is True
    assert row["model_fitting_deferred"] is True
    assert row["runtime_backend"] == "records"
    assert row["executable"] is False
    assert row["deferred"] is True
    assert row["code"] == "model_fitting_deferred"
    forbidden_fields = {
        "p_value",
        "q_value",
        "confidence_interval",
        "effect_size",
        "residuals",
        "diagnostics",
        "coefficients",
        "fitted_model",
        "fitted_model_fields",
    }
    assert forbidden_fields.isdisjoint(row)
    assert payload["output_paths_written"] == []
    assert payload["no_output_written"] is True
    json.dumps(payload, sort_keys=True, allow_nan=False)


def test_design_qc_preserves_metadata_reports_deferred_status_and_writes_nothing() -> None:
    payload = run_tabular_association_repeated_measures_design_qc(
        _workflow_doc(),
        source_rows_by_id={"source-alpha": _rows()},
    ).to_dict()

    codes = _codes(payload)
    assert "model_design_metadata_only" in codes
    assert "model_fitting_deferred" in codes
    assert payload["status"] == "warning"
    assert payload["output_paths_written"] == []
    assert payload["no_output_written"] is True
    assert payload["will_write"] is False
    assert payload["output_written"] is False
    provenance = {str(row["key"]): row["value"] for row in payload["provenance_rows"]}
    assert provenance["repeated_measures_metadata_version"] == (
        tabular_associations.TABULAR_ASSOCIATION_REPEATED_MEASURES_METADATA_VERSION
    )
    assert provenance["model_design_metadata_only"] is True
    assert provenance["model_fitting_deferred"] is True
    assert provenance["fixed_effect_term_count"] == 1
    assert provenance["random_effect_term_count"] == 1
    assert provenance["repeated_factor_count"] == 1
    assert provenance["planned_comparison_count"] == 1
    assert provenance["contrast_metadata_count"] == 1
    json.dumps(payload, sort_keys=True, allow_nan=False)


def test_legacy_freeform_method_metadata_is_preserved_without_design_inference() -> None:
    doc = _workflow_doc()
    doc["repeated_measures"]["metadata"] = {}  # type: ignore[index]
    doc["methods"][0]["metadata"] = {"legacy-freeform": {"kept": True}}  # type: ignore[index]

    plan_payload = plan_tabular_association_repeated_measures(doc).to_dict()
    row = plan_payload["model_plan_rows"][0]
    assert row["method_metadata"] == {"legacy-freeform": {"kept": True}}
    assert row["fixed_effect_term_ids"] == []
    assert row["random_effect_term_ids"] == []
    assert row["planned_comparison_ids"] == []

    qc_payload = run_tabular_association_repeated_measures_design_qc(
        doc,
        source_rows_by_id={"source-alpha": _rows()},
    ).to_dict()
    assert "model_design_metadata_only" not in _codes(qc_payload)
    json.dumps(qc_payload, sort_keys=True, allow_nan=False)


def test_duplicate_unknown_missing_and_malformed_metadata_emit_qc_errors() -> None:
    duplicate_doc = _workflow_doc()
    duplicate_doc["methods"][0]["metadata"]["model_design"]["fixed_effect_terms"].append(  # type: ignore[index]
        {"term_id": "term-alpha", "variable_ids": ["predictor-alpha"]}
    )
    duplicate_payload = run_tabular_association_repeated_measures_design_qc(
        duplicate_doc,
        source_rows_by_id={"source-alpha": _rows()},
    ).to_dict()
    assert duplicate_payload["status"] == "error"
    assert "duplicate_metadata_id" in _codes(duplicate_payload)

    unknown_doc = _workflow_doc()
    unknown_design = unknown_doc["methods"][0]["metadata"]["model_design"]  # type: ignore[index]
    unknown_design["fixed_effect_terms"][0]["variable_ids"] = ["variable-unknown"]  # type: ignore[index]
    unknown_design["planned_comparisons"][0]["factor_ids"] = ["factor-unknown"]  # type: ignore[index]
    unknown_design["contrast_metadata"][0]["comparison_ids"] = ["comparison-unknown"]  # type: ignore[index]
    unknown_design["categorical_coding"][0]["target_id"] = "coding-target-unknown"  # type: ignore[index]
    unknown_payload = run_tabular_association_repeated_measures_design_qc(
        unknown_doc,
        source_rows_by_id={"source-alpha": _rows()},
    ).to_dict()
    assert unknown_payload["status"] == "error"
    assert "unknown_metadata_reference" in _codes(unknown_payload)

    missing_column_payload = run_tabular_association_repeated_measures_design_qc(
        _workflow_doc(missing_metadata_column=True),
        source_rows_by_id={"source-alpha": _rows(include_column_alpha=False)},
    ).to_dict()
    assert missing_column_payload["status"] == "error"
    assert "missing_metadata_column" in _codes(missing_column_payload)

    malformed_doc = _workflow_doc()
    malformed_doc["methods"][0]["metadata"]["model_design"]["fixed_effect_terms"] = 1  # type: ignore[index]
    malformed_payload = run_tabular_association_repeated_measures_design_qc(
        malformed_doc,
        source_rows_by_id={"source-alpha": _rows()},
    ).to_dict()
    assert malformed_payload["status"] == "error"
    assert "invalid_model_design_metadata" in _codes(malformed_payload)
    json.dumps(malformed_payload, sort_keys=True, allow_nan=False)


def test_metadata_qc_rows_are_tsv_safe() -> None:
    row = RepeatedMeasuresDesignQcRow(
        workflow_id="workflow-alpha",
        source_id="source-alpha",
        method_id="method-alpha",
        method_name="mixed_model",
        model_plan_id="model-alpha",
        runtime_backend="records",
        status="ok",
        code="model_design_metadata_only",
        message="Metadata only.",
        metadata={"fixed_effect_term_ids": ["term-alpha"], "metadata_only": True},
    )

    tsv_row = row.to_tsv_row()
    assert all(isinstance(value, str) for value in tsv_row.values())
    assert tsv_row["metadata"] == '{"fixed_effect_term_ids":["term-alpha"],"metadata_only":true}'
    assert tsv_row["model_fitting_deferred"] == "true"


def test_metadata_source_has_no_forbidden_imports_or_specific_identifiers() -> None:
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
    )
    source_text = Path(tabular_associations.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source_text)
    imported_modules: list[str] = []
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
    forbidden_text = (
        "confidential-study-marker",
        "private-task-marker",
        "private-cohort-marker",
    )
    assert all(text not in source_text for text in forbidden_text)
    assert re.search(r"sub-\d{3}", source_text) is None

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
