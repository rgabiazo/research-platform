from __future__ import annotations

import ast
from dataclasses import dataclass
import json
import math
import re
from pathlib import Path

import pytest

import research_platform.analysis.tabular_associations as tabular_associations
from research_platform.analysis.tabular_associations import (
    AssociationPublicationTableRow,
    build_tabular_association_publication_tables,
    plan_tabular_association_publication_tables,
)


@dataclass(frozen=True)
class SyntheticAssociationRow:
    workflow_id: str
    result_row_id: str
    pair_id: str
    method_id: str
    method_kind: str
    method_name: str
    family_id: str
    source_id: str
    outcome_id: str
    predictor_id: str
    covariate_ids: tuple[str, ...]
    statistic_name: str
    statistic_value: float
    p_value: float
    n_used: int
    status: str = "ok"


def _association_row(**updates: object) -> dict[str, object]:
    row: dict[str, object] = {
        "workflow_id": "workflow-alpha",
        "result_row_id": "result-alpha",
        "pair_id": "pair-alpha",
        "method_id": "method-alpha",
        "method_kind": "correlation",
        "method_name": "pearson",
        "family_id": "family-alpha",
        "source_id": "source-alpha",
        "outcome_id": "outcome-alpha",
        "predictor_id": "predictor-alpha",
        "covariate_ids": ["covariate-alpha"],
        "statistic_name": "r",
        "statistic_value": 0.123456,
        "p_value": 0.012345,
        "n_used": 7,
        "status": "ok",
        "warnings": [],
        "errors": [],
    }
    row.update(updates)
    return row


def test_publication_handoff_from_correlation_rows_and_manifest_is_json_tsv_safe(tmp_path: Path) -> None:
    association_rows = [
        SyntheticAssociationRow(
            workflow_id="workflow-alpha",
            result_row_id="result-alpha",
            pair_id="pair-alpha",
            method_id="method-alpha",
            method_kind="correlation",
            method_name="pearson",
            family_id="family-alpha",
            source_id="source-alpha",
            outcome_id="outcome-alpha",
            predictor_id="predictor-alpha",
            covariate_ids=("covariate-alpha",),
            statistic_name="r",
            statistic_value=0.123456,
            p_value=0.012345,
            n_used=7,
        )
    ]
    original_row = association_rows[0]
    multiplicity_rows = [
        {
            "workflow_id": "workflow-alpha",
            "result_row_id": "result-alpha",
            "family_id": "family-alpha",
            "p_value": 0.012345,
            "q_value": 0.045678,
            "status": "ok",
            "code": "adjusted",
        }
    ]
    qc_rows = [
        {
            "workflow_id": "workflow-alpha",
            "status": "ok",
            "code": "qc-alpha",
            "message": "Synthetic QC row.",
        }
    ]
    missingness_rows = [
        {
            "workflow_id": "workflow-alpha",
            "source_id": "source-alpha",
            "column_name": "outcome-alpha",
            "role": "outcome",
            "missing_count": 0,
            "nonmissing_count": 7,
            "total_count": 7,
            "status": "ok",
            "code": "missingness-alpha",
            "message": "Synthetic missingness row.",
        }
    ]
    provenance_rows = [{"workflow_id": "workflow-alpha", "key": "source_rowset", "value": "table-alpha"}]

    plan = plan_tabular_association_publication_tables(
        association_rows,
        multiplicity_rows=multiplicity_rows,
        qc_rows=qc_rows,
        missingness_rows=missingness_rows,
        provenance_rows=provenance_rows,
        source_rowset_names={"association_rows": "table-alpha"},
    ).to_dict()
    payload = build_tabular_association_publication_tables(
        association_rows,
        multiplicity_rows=multiplicity_rows,
        qc_rows=qc_rows,
        missingness_rows=missingness_rows,
        provenance_rows=provenance_rows,
        source_rowset_names={"association_rows": "table-alpha"},
    ).to_dict()

    assert plan["executed"] is False
    assert plan["plan_only"] is True
    assert payload["executed"] is True
    assert payload["plan_only"] is False
    assert payload["will_write"] is False
    assert payload["output_written"] is False
    assert payload["no_output_written"] is True
    assert payload["output_paths_written"] == []
    assert payload["association_table_rows"][0]["q_value"] == pytest.approx(0.045678)
    assert payload["association_table_rows"][0]["p_value"] == pytest.approx(0.012345)
    assert payload["association_machine_rows"][0]["statistic_value"] == pytest.approx(0.123456)
    assert payload["association_display_rows"][0]["statistic_value"] == "0.123"
    assert payload["association_machine_rows"][0]["q_value"] == pytest.approx(0.045678)
    assert payload["association_display_rows"][0]["q_value"] == "0.046"
    assert payload["input_summary_rows"][0]["source_rowset_name"] == "table-alpha"
    assert payload["qc_table_rows"][0]["code"] == "qc-alpha"
    assert payload["missingness_table_rows"][0]["missing_count"] == 0
    assert any(row["key"] == "source_rowset" for row in payload["provenance_table_rows"])
    assert payload["manifest_rows"][0]["association_result_row_count"] == 1
    assert payload["manifest_rows"][0]["output_paths_written"] == []
    assert original_row.p_value == pytest.approx(0.012345)
    assert not any(tmp_path.iterdir())
    json.dumps(payload, sort_keys=True, allow_nan=False)

    tsv_row = AssociationPublicationTableRow(
        workflow_id="workflow-alpha",
        method_id="method-alpha",
        method_kind="correlation",
        method_name="pearson",
        family_id="family-alpha",
        source_id="source-alpha",
        outcome_id="outcome-alpha",
        predictor_id="predictor-alpha",
        covariate_ids=("covariate-alpha",),
        statistic_name="r",
        statistic_value=0.5,
        p_value=0.01,
        q_value=0.02,
        n=7,
        n_used=7,
        n_total=7,
        status="ok",
        warnings=(),
        errors=(),
        result_row_id="result-alpha",
        pair_id="pair-alpha",
        input_row_index=0,
    ).to_tsv_row()
    assert all(isinstance(value, str) for value in tsv_row.values())
    assert tsv_row["output_written"] == "false"


def test_adjusted_regression_rows_match_multiplicity_by_pair_input_index_position_and_missing() -> None:
    association_rows = [
        _association_row(
            result_row_id="result-alpha",
            pair_id="pair-alpha",
            method_kind="adjusted",
            method_name="partial_correlation",
            statistic_name="partial_r",
        ),
        _association_row(
            result_row_id="result-beta",
            pair_id=None,
            input_row_index=8,
            method_kind="regression",
            method_name="regression",
            statistic_name="regression_coefficient",
            statistic_value=1.234567,
            p_value=None,
        ),
        _association_row(result_row_id=None, pair_id=None, statistic_value=0.333333, p_value=None),
        _association_row(result_row_id="result-missing", pair_id="pair-missing", statistic_value=0.444444),
    ]
    multiplicity_rows = [
        {"workflow_id": "workflow-alpha", "pair_id": "pair-alpha", "q_value": 0.02, "p_value": 0.01},
        {"workflow_id": "workflow-alpha", "input_row_index": 8, "q_value": 0.05, "p_value": 0.04},
        {"workflow_id": "workflow-alpha", "q_value": 0.07, "p_value": 0.06},
    ]

    payload = build_tabular_association_publication_tables(
        association_rows,
        multiplicity_rows=multiplicity_rows,
    ).to_dict()
    rows = payload["association_table_rows"]

    assert rows[0]["q_value"] == pytest.approx(0.02)
    assert rows[0]["multiplicity_match_field"] == "pair_id"
    assert rows[1]["q_value"] == pytest.approx(0.05)
    assert rows[1]["p_value"] == pytest.approx(0.04)
    assert rows[1]["multiplicity_match_field"] == "input_row_index"
    assert rows[2]["q_value"] == pytest.approx(0.07)
    assert rows[2]["multiplicity_match_field"] == "position"
    assert rows[3]["q_value"] is None
    assert payload["association_machine_rows"][1]["statistic_value"] == pytest.approx(1.234567)
    assert payload["association_display_rows"][1]["statistic_value"] == "1.235"
    json.dumps(payload, sort_keys=True, allow_nan=False)


def test_ambiguous_multiplicity_rows_warn_and_do_not_apply_q_value() -> None:
    payload = build_tabular_association_publication_tables(
        [_association_row(result_row_id="result-alpha")],
        multiplicity_rows=[
            {"workflow_id": "workflow-alpha", "result_row_id": "result-alpha", "q_value": 0.01},
            {"workflow_id": "workflow-alpha", "result_row_id": "result-alpha", "q_value": 0.02},
        ],
    ).to_dict()

    assert payload["status"] == "warning"
    assert payload["association_table_rows"][0]["q_value"] is None
    assert any("ambiguous" in warning for warning in payload["warnings"])
    json.dumps(payload, sort_keys=True, allow_nan=False)


def test_missing_multiplicity_rows_preserve_supplied_p_and_leave_q_missing() -> None:
    payload = build_tabular_association_publication_tables([_association_row(p_value=0.025)]).to_dict()

    assert payload["association_table_rows"][0]["p_value"] == pytest.approx(0.025)
    assert payload["association_table_rows"][0]["q_value"] is None
    assert payload["multiplicity_table_rows"] == []
    json.dumps(payload, sort_keys=True, allow_nan=False)


def test_multiple_workflow_ids_require_explicit_override() -> None:
    error_payload = build_tabular_association_publication_tables(
        [_association_row(workflow_id="workflow-alpha")],
        qc_rows=[{"workflow_id": "workflow-beta", "status": "ok"}],
    ).to_dict()
    override_payload = build_tabular_association_publication_tables(
        [_association_row(workflow_id="workflow-alpha")],
        qc_rows=[{"workflow_id": "workflow-beta", "status": "ok"}],
        workflow_id="workflow-alpha",
    ).to_dict()

    assert error_payload["valid"] is False
    assert error_payload["status"] == "error"
    assert "Multiple workflow_id values" in error_payload["errors"][0]
    assert override_payload["valid"] is True
    assert override_payload["workflow_id"] == "workflow-alpha"
    assert any("overrides" in warning for warning in override_payload["warnings"])


def test_nonfinite_values_are_normalized_to_json_safe_missing_values() -> None:
    payload = build_tabular_association_publication_tables(
        [
            _association_row(
                statistic_value=math.nan,
                p_value=math.inf,
            )
        ],
        multiplicity_rows=[
            {"workflow_id": "workflow-alpha", "result_row_id": "result-alpha", "q_value": math.inf}
        ],
    ).to_dict()

    assert payload["association_table_rows"][0]["statistic_value"] is None
    assert payload["association_table_rows"][0]["p_value"] is None
    assert payload["association_table_rows"][0]["q_value"] is None
    assert payload["association_display_rows"][0]["statistic_value"] == ""
    assert payload["association_machine_rows"][0]["statistic_value"] is None
    json.dumps(payload, sort_keys=True, allow_nan=False)


def test_publication_handoff_has_no_forbidden_imports_or_study_specific_constants() -> None:
    forbidden_modules = (
        "numpy",
        "pandas",
        "polars",
        "scipy",
        "sklearn",
        "statsmodels",
        "research_platform.core",
        "research_platform.neuro",
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
