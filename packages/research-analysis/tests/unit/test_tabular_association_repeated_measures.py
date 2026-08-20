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
    RepeatedMeasuresDesignQcRow,
    plan_tabular_association_repeated_measures,
    run_tabular_association_repeated_measures_design_qc,
)


def _source_alpha(path: str | None = None) -> dict[str, object]:
    source: dict[str, object] = {
        "source_id": "source-alpha",
        "format": "tsv",
        "schema": {
            "subject_id_column": "participant-id",
            "timepoint_column": "timepoint-alpha",
            "columns": [
                {"column_name": "participant-id", "value_type": "categorical", "role": "subject_identifier"},
                {"column_name": "timepoint-alpha", "value_type": "categorical", "role": "timepoint_identifier"},
                {"column_name": "factor-alpha", "value_type": "categorical", "role": "repeated_factor"},
                {"column_name": "outcome-alpha", "value_type": "numeric", "role": "outcome"},
                {"column_name": "predictor-alpha", "value_type": "numeric", "role": "predictor"},
                {"column_name": "covariate-alpha", "value_type": "numeric", "role": "covariate"},
                {"column_name": "group-alpha", "value_type": "categorical", "role": "grouping"},
                {"column_name": "cluster-alpha", "value_type": "categorical", "role": "cluster"},
            ],
        },
    }
    if path is not None:
        source["path"] = path
    return source


def _workflow_doc(*, method_name: str = "repeated_measures", path: str | None = None) -> dict[str, object]:
    return {
        "workflow_id": "workflow-alpha",
        "sources": [_source_alpha(path=path)],
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
            "timepoint_column": "timepoint-alpha",
            "unit_columns": ["participant-id", "timepoint-alpha"],
            "metadata": {
                "repeated_factor_columns": ["factor-alpha"],
                "cluster_columns": ["cluster-alpha"],
            },
        },
        "methods": [
            {
                "method_id": "method-alpha",
                "method": method_name,
                "outcome_ids": ["outcome-alpha"],
                "predictor_ids": ["predictor-alpha"],
                "covariate_ids": ["covariate-alpha"],
                "grouping_ids": ["group-alpha"],
                "metadata": {
                    "formula_like": "outcome-alpha ~ predictor-alpha + covariate-alpha",
                    "fixed_effect_term_ids": ["predictor-alpha", "covariate-alpha"],
                    "random_effects": {"participant-id": "random-intercept"},
                },
            }
        ],
    }


def _balanced_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for participant_id, cluster_id in (("participant-a", "cluster-alpha"), ("participant-b", "cluster-beta")):
        rows.append(
            {
                "participant-id": participant_id,
                "timepoint-alpha": "timepoint-alpha",
                "factor-alpha": "timepoint-alpha",
                "outcome-alpha": "1",
                "predictor-alpha": "2",
                "covariate-alpha": "3",
                "group-alpha": "group-alpha",
                "cluster-alpha": cluster_id,
            }
        )
        rows.append(
            {
                "participant-id": participant_id,
                "timepoint-alpha": "timepoint-beta",
                "factor-alpha": "timepoint-beta",
                "outcome-alpha": "2",
                "predictor-alpha": "3",
                "covariate-alpha": "4",
                "group-alpha": "group-alpha",
                "cluster-alpha": cluster_id,
            }
        )
    return rows


def _problem_rows() -> list[dict[str, object]]:
    base = {
        "factor-alpha": "timepoint-alpha",
        "outcome-alpha": "1",
        "covariate-alpha": "3",
        "group-alpha": "group-alpha",
    }
    return [
        {"participant-id": "participant-a", "timepoint-alpha": "timepoint-alpha", **base},
        {"participant-id": "participant-a", "timepoint-alpha": "timepoint-alpha", **base},
        {"participant-id": "participant-b", "timepoint-alpha": "timepoint-alpha", **base},
        {"participant-id": "", "timepoint-alpha": "timepoint-beta", "factor-alpha": "timepoint-beta", **base},
        {"participant-id": "participant-c", "timepoint-alpha": "", **base},
    ]


def _codes(payload: dict[str, object]) -> set[str]:
    return {str(row["code"]) for row in payload["qc_rows"]}  # type: ignore[index]


def test_repeated_measures_method_declarations_produce_plan_rows_without_fitting() -> None:
    payload = plan_tabular_association_repeated_measures(_workflow_doc()).to_dict()

    plan_row = payload["model_plan_rows"][0]
    assert plan_row["method_name"] == "repeated_measures"
    assert plan_row["method_kind"] == "repeated_measures"
    assert plan_row["executable"] is False
    assert plan_row["deferred"] is True
    assert plan_row["code"] == "model_fitting_deferred"
    assert payload["will_write"] is False
    assert payload["output_written"] is False
    assert payload["no_output_written"] is True
    assert payload["output_paths_written"] == []
    assert {"p_value", "q_value", "confidence_interval", "effect_size", "residuals"}.isdisjoint(plan_row)
    json.dumps(payload, sort_keys=True, allow_nan=False)


def test_mixed_model_plan_preserves_formula_fixed_and_random_metadata_only() -> None:
    payload = plan_tabular_association_repeated_measures(_workflow_doc(method_name="mixed_model")).to_dict()

    plan_row = payload["model_plan_rows"][0]
    assert plan_row["method_name"] == "mixed_model"
    assert plan_row["formula_metadata"] == {"formula_like": "outcome-alpha ~ predictor-alpha + covariate-alpha"}
    assert plan_row["fixed_effect_term_ids"] == ["predictor-alpha", "covariate-alpha"]
    assert plan_row["random_effect_metadata"] == {"random_effects": {"participant-id": "random-intercept"}}
    assert plan_row["runtime_backend"] == "records"
    assert "result_rows" not in payload
    json.dumps(payload, sort_keys=True, allow_nan=False)


def test_supplied_long_format_records_produce_design_and_factor_summaries(tmp_path: Path) -> None:
    source_path = tmp_path / "source-alpha.tsv"
    payload = run_tabular_association_repeated_measures_design_qc(
        _workflow_doc(method_name="mixed_model", path=str(source_path)),
        source_rows_by_id={"source-alpha": _balanced_rows()},
    ).to_dict()

    summary = payload["design_summary_rows"][0]
    assert summary["row_count"] == 4
    assert summary["observation_count"] == 4
    assert summary["participant_count"] == 2
    assert summary["cluster_count"] == 2
    assert summary["min_observations_per_participant"] == 2
    assert summary["max_observations_per_participant"] == 2
    assert summary["balanced_design"] is True
    factors = {row["factor_column"]: row for row in payload["factor_summary_rows"]}
    assert factors["timepoint-alpha"]["levels"] == ["timepoint-alpha", "timepoint-beta"]
    assert factors["factor-alpha"]["participants_by_level"] == {"timepoint-alpha": 2, "timepoint-beta": 2}
    assert "model_fitting_deferred" in _codes(payload)
    assert payload["no_output_written"] is True
    assert payload["output_paths_written"] == []
    assert not source_path.exists()
    assert any(row["model_fitting_deferred"] is True for row in payload["provenance_rows"])
    json.dumps(payload, sort_keys=True, allow_nan=False)


def test_design_qc_reports_duplicate_missing_singleton_imbalanced_and_missing_columns() -> None:
    payload = run_tabular_association_repeated_measures_design_qc(
        _workflow_doc(method_name="mixed_model"),
        source_rows_by_id={"source-alpha": _problem_rows()},
    ).to_dict()

    codes = _codes(payload)
    assert payload["status"] == "error"
    assert "duplicate_repeated_unit_rows" in codes
    assert "missing_participant_ids" in codes
    assert "missing_repeated_keys" in codes
    assert "singleton_participants" in codes
    assert "insufficient_repeated_observations" in codes
    assert "imbalanced_repeated_design" in codes
    assert "missing_required_predictor_column" in codes
    assert "missing_required_cluster_column" in codes
    summary = payload["design_summary_rows"][0]
    assert summary["duplicate_repeated_unit_count"] == 1
    assert summary["missing_subject_id_count"] == 1
    assert summary["missing_repeated_key_count"] == 2
    assert summary["singleton_participant_count"] == 2
    assert summary["insufficient_repeat_participant_count"] == 2
    assert summary["balanced_design"] is False
    json.dumps(payload, sort_keys=True, allow_nan=False)


def test_zero_row_sources_emit_safe_qc_rows_and_write_nothing() -> None:
    payload = run_tabular_association_repeated_measures_design_qc(
        _workflow_doc(),
        source_rows_by_id={"source-alpha": []},
    ).to_dict()

    assert payload["source_load_rows"][0]["load_status"] == "empty"
    assert payload["design_summary_rows"][0]["row_count"] == 0
    assert "empty_source_rows" in _codes(payload)
    assert payload["will_write"] is False
    assert payload["output_written"] is False
    assert payload["no_output_written"] is True
    assert payload["output_paths_written"] == []
    json.dumps(payload, sort_keys=True, allow_nan=False)


def test_no_repeated_or_mixed_model_methods_declared_emits_qc_warning() -> None:
    doc = _workflow_doc()
    doc["methods"] = [
        {
            "method_id": "method-beta",
            "method": "pearson",
            "outcome_ids": ["outcome-alpha"],
            "predictor_ids": ["predictor-alpha"],
        }
    ]

    payload = run_tabular_association_repeated_measures_design_qc(
        doc,
        source_rows_by_id={"source-alpha": _balanced_rows()},
    ).to_dict()

    assert payload["model_plan_rows"] == []
    assert "no_repeated_mixed_model_methods_declared" in _codes(payload)
    assert payload["source_load_rows"] == []


def test_repeated_measures_rows_are_json_and_tsv_safe() -> None:
    payload = run_tabular_association_repeated_measures_design_qc(
        _workflow_doc(),
        source_rows_by_id={"source-alpha": _balanced_rows()},
    ).to_dict()
    tsv_row = RepeatedMeasuresDesignQcRow(
        workflow_id="workflow-alpha",
        source_id="source-alpha",
        method_id="method-alpha",
        method_name="repeated_measures",
        model_plan_id="model-alpha",
        runtime_backend="records",
        status="deferred",
        code="model_fitting_deferred",
        message="Model fitting is deferred.",
        metadata={"levels": ["timepoint-alpha", "timepoint-beta"]},
    ).to_tsv_row()

    json.dumps(payload, sort_keys=True, allow_nan=False)
    assert all(isinstance(value, str) for value in tsv_row.values())
    assert tsv_row["model_fitting_deferred"] == "true"
    assert tsv_row["no_output_written"] == "true"
    assert tsv_row["output_paths_written"] == "[]"


def test_import_and_production_source_have_no_forbidden_dependencies_or_specific_constants() -> None:
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
