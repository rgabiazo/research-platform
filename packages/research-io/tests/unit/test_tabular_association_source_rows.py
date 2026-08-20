from __future__ import annotations

import ast
import importlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys

import pytest

from research_platform.io import prepare_tabular_association_source_rows as exported_prepare_source_rows
from research_platform.io.dataframe import (
    TABULAR_ASSOCIATION_SOURCE_ROWS_VERSION,
    prepare_tabular_association_source_rows as dataframe_prepare_source_rows,
)
from research_platform.io.dataframe.association_records import (
    TabularAssociationSourceRowsResult,
    prepare_tabular_association_source_rows,
)

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = PACKAGE_ROOT.parents[1]
MODULE_PATH = PACKAGE_ROOT / "src" / "research_platform" / "io" / "dataframe" / "association_records.py"

BLOCKED_IMPORTS = [
    "pandas",
    "polars",
    "numpy",
    "scipy",
    "sklearn",
    "statsmodels",
    "research_platform.analysis",
    "research_platform.viz",
    "research_platform.core",
    "research_platform.neuro",
    "research_platform.bids",
]


def _rows() -> list[dict[str, object]]:
    return [
        {"column-alpha": "row-alpha", "column-beta": 1},
        {"column-alpha": "row-beta", "column-beta": 2},
    ]


def _association_rows() -> list[dict[str, object]]:
    return [
        {
            "participant-id": "participant-a",
            "session-label": "session-alpha",
            "timepoint-label": "timepoint-alpha",
            "outcome-alpha": "1",
            "predictor-alpha": "2",
        },
        {
            "participant-id": "participant-b",
            "session-label": "session-alpha",
            "timepoint-label": "timepoint-beta",
            "outcome-alpha": "2",
            "predictor-alpha": "4",
        },
        {
            "participant-id": "participant-c",
            "session-label": "session-beta",
            "timepoint-label": "timepoint-alpha",
            "outcome-alpha": "3",
            "predictor-alpha": "6",
        },
        {
            "participant-id": "participant-d",
            "session-label": "session-beta",
            "timepoint-label": "timepoint-beta",
            "outcome-alpha": "4",
            "predictor-alpha": "8",
        },
    ]


class FakeToDicts:
    columns = ("column-alpha", "column-beta")

    def __init__(self, rows: list[dict[str, object]]) -> None:
        self._rows = rows

    def to_dicts(self) -> list[dict[str, object]]:
        return self._rows


def _workflow_doc() -> dict[str, object]:
    return {
        "workflow_id": "workflow-alpha",
        "sources": [
            {
                "source_id": "source-alpha",
                "backend": "records",
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
                    ],
                    "numeric_validation": {"policy": "declare"},
                },
            }
        ],
        "outcomes": [{"variable_id": "outcome-alpha", "source_id": "source-alpha", "column_name": "outcome-alpha"}],
        "predictors": [
            {"variable_id": "predictor-alpha", "source_id": "source-alpha", "column_name": "predictor-alpha"}
        ],
        "missing_data_policy": {"strategy": "pairwise"},
        "nonfinite_policy": {"strategy": "drop_rows"},
        "methods": [
            {
                "method_id": "pearson-alpha",
                "method": "pearson",
                "outcome_ids": ["outcome-alpha"],
                "predictor_ids": ["predictor-alpha"],
                "family_id": "family-alpha",
            }
        ],
        "families": [{"family_id": "family-alpha", "method_ids": ["pearson-alpha"]}],
    }


def _analysis_tabular_associations():
    analysis_src = REPO_ROOT / "packages" / "research-analysis" / "src"
    if analysis_src.exists() and str(analysis_src) not in sys.path:
        sys.path.insert(0, str(analysis_src))
    return pytest.importorskip("research_platform.analysis.tabular_associations")


def _run_import_check(module_name: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PACKAGE_ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    script = (
        "import json, sys\n"
        f"__import__({module_name!r}, fromlist=['*'])\n"
        f"blocked = {BLOCKED_IMPORTS!r}\n"
        "print(json.dumps([name for name in blocked if name in sys.modules]))\n"
    )
    return subprocess.run(
        [sys.executable, "-c", script],
        cwd=PACKAGE_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )


def test_public_exports_are_available() -> None:
    assert exported_prepare_source_rows is prepare_tabular_association_source_rows
    assert dataframe_prepare_source_rows is prepare_tabular_association_source_rows
    assert TABULAR_ASSOCIATION_SOURCE_ROWS_VERSION == "research_platform.io.dataframe.association_records.v1"


def test_mapping_rows_convert_to_source_rows_by_id() -> None:
    result = prepare_tabular_association_source_rows({"source-alpha": _rows()})

    assert isinstance(result, TabularAssociationSourceRowsResult)
    assert result.valid is True
    assert result.status == "ok"
    assert result.source_rows_by_id == {
        "source-alpha": (
            {"column-alpha": "row-alpha", "column-beta": 1},
            {"column-alpha": "row-beta", "column-beta": 2},
        )
    }
    assert result.backend_by_source_id == {"source-alpha": "records"}
    assert result.row_count_by_source_id == {"source-alpha": 2}
    assert result.conversion_results_by_id["source-alpha"].adapter_spec.requested_backend == "records"


def test_fake_dataframe_like_objects_convert_to_source_rows_by_id() -> None:
    result = prepare_tabular_association_source_rows(
        {"source-alpha": FakeToDicts(_rows())},
        default_backend="dataframe_like",
    )

    assert result.valid is True
    assert result.source_rows_by_id["source-alpha"][0]["column-alpha"] == "row-alpha"
    assert result.conversion_results_by_id["source-alpha"].row_source_kind == "to_dicts"
    assert result.conversion_results_by_id["source-alpha"].adapter_spec.requested_backend == "dataframe_like"


def test_multiple_source_ids_convert_independently_and_preserve_order() -> None:
    result = prepare_tabular_association_source_rows(
        {
            "source-alpha": [{"column-alpha": "row-alpha"}, {"column-alpha": "row-beta"}],
            "source-beta": [{"column-alpha": "row-gamma"}],
        }
    )

    assert result.valid is True
    assert result.source_count == 2
    assert result.converted_source_count == 2
    assert result.failed_source_count == 0
    assert [row["column-alpha"] for row in result.source_rows_by_id["source-alpha"]] == ["row-alpha", "row-beta"]
    assert [row["column-alpha"] for row in result.source_rows_by_id["source-beta"]] == ["row-gamma"]


def test_backend_by_source_id_overrides_default_backend() -> None:
    result = prepare_tabular_association_source_rows(
        {
            "source-alpha": _rows(),
            "source-beta": FakeToDicts(_rows()),
        },
        default_backend="dataframe_like",
        backend_by_source_id={"source-alpha": "records"},
    )

    assert result.valid is True
    assert result.backend_by_source_id == {
        "source-alpha": "records",
        "source-beta": "dataframe_like",
    }
    assert result.conversion_results_by_id["source-alpha"].adapter_spec.requested_backend == "records"
    assert result.conversion_results_by_id["source-beta"].adapter_spec.requested_backend == "dataframe_like"


def test_unknown_backend_produces_safe_aggregate_error() -> None:
    result = prepare_tabular_association_source_rows(
        {"source-alpha": _rows()},
        backend_by_source_id={"source-alpha": "backend-alpha"},
    )

    assert result.valid is False
    assert result.status == "error"
    assert result.source_rows_by_id == {}
    assert result.failed_source_count == 1
    assert result.row_count_by_source_id == {"source-alpha": 0}
    assert result.qc_rows[0].status == "error"
    assert result.qc_rows[0].code == "unknown_tabular_association_source_rows_backend"
    assert "source-alpha:" in result.errors[0]


def test_invalid_source_conversion_omits_invalid_source_but_preserves_successful_sources() -> None:
    result = prepare_tabular_association_source_rows(
        {
            "source-alpha": _rows(),
            "source-beta": object(),
        }
    )

    assert result.valid is False
    assert result.status == "error"
    assert tuple(result.source_rows_by_id) == ("source-alpha",)
    assert result.source_rows_by_id["source-alpha"][1]["column-alpha"] == "row-beta"
    assert result.failed_source_count == 1
    assert result.converted_source_count == 1
    assert result.row_count_by_source_id == {"source-alpha": 2, "source-beta": 0}
    assert result.conversion_results_by_id["source-beta"].valid is False


def test_valid_zero_row_sources_remain_present_with_empty_tuples() -> None:
    result = prepare_tabular_association_source_rows({"source-alpha": []})

    assert result.valid is True
    assert result.source_rows_by_id == {"source-alpha": ()}
    assert result.row_count_by_source_id == {"source-alpha": 0}
    assert result.converted_source_count == 1
    assert result.failed_source_count == 0


def test_no_mutation_of_source_rows_or_fake_dataframe_internals() -> None:
    source_rows = _rows()
    fake = FakeToDicts(source_rows)
    before_rows = [dict(row) for row in fake._rows]

    result = prepare_tabular_association_source_rows({"source-alpha": fake})
    result.source_rows_by_id["source-alpha"][0]["column-alpha"] = "changed-value"
    payload = result.to_dict()
    payload["source_rows_by_id"]["source-alpha"][1]["column-alpha"] = "payload-changed"

    assert fake._rows == before_rows
    assert source_rows == before_rows
    assert result.source_rows_by_id["source-alpha"][0]["column-alpha"] == "changed-value"
    assert result.source_rows_by_id["source-alpha"][1]["column-alpha"] == "row-beta"


def test_include_input_row_index_is_preserved_through_conversion() -> None:
    result = prepare_tabular_association_source_rows(
        {"source-alpha": _rows()},
        include_input_row_index=True,
        input_row_index_field="input_row_index",
    )

    assert result.valid is True
    assert result.source_rows_by_id["source-alpha"] == (
        {"column-alpha": "row-alpha", "column-beta": 1, "input_row_index": 0},
        {"column-alpha": "row-beta", "column-beta": 2, "input_row_index": 1},
    )
    assert result.qc_rows[0].include_input_row_index is True
    assert result.provenance_rows[0].input_row_index_field == "input_row_index"


def test_json_safety_with_allow_nan_false() -> None:
    result = prepare_tabular_association_source_rows(
        {
            "source-alpha": [
                {"column-alpha": math.nan, "column-beta": math.inf, "column-gamma": -math.inf},
            ]
        }
    )
    payload = result.to_dict()

    assert payload["source_rows_by_id"]["source-alpha"][0]["column-alpha"] == "nan"
    assert payload["source_rows_by_id"]["source-alpha"][0]["column-beta"] == "inf"
    assert payload["source_rows_by_id"]["source-alpha"][0]["column-gamma"] == "-inf"
    json.dumps(payload, allow_nan=False)


def test_qc_and_provenance_rows_are_tsv_safe() -> None:
    result = prepare_tabular_association_source_rows({"source-alpha": [{"column-alpha": "value\talpha\nline"}]})

    for row in (*result.qc_rows, *result.provenance_rows):
        tsv_row = row.to_tsv_row()
        assert tsv_row
        assert all(isinstance(value, str) for value in tsv_row.values())
        assert all("\t" not in value and "\n" not in value and "\r" not in value for value in tsv_row.values())


def test_no_write_flags_and_output_paths_are_empty() -> None:
    result = prepare_tabular_association_source_rows({"source-alpha": _rows()})
    payload = result.to_dict()

    assert result.runtime_backend == "records"
    assert result.will_write is False
    assert result.output_written is False
    assert result.no_output_written is True
    assert result.output_paths_written == ()
    assert payload["output_paths_written"] == []
    assert payload["no_output_written"] is True
    assert result.conversion_results_by_id["source-alpha"].no_output_written is True


def test_source_rows_can_feed_existing_tabular_association_qc() -> None:
    tabular_associations = _analysis_tabular_associations()
    prepared = prepare_tabular_association_source_rows({"source-alpha": _association_rows()})

    payload = tabular_associations.run_tabular_association_qc(
        _workflow_doc(),
        source_rows_by_id=prepared.source_rows_by_id,
    ).to_dict()

    assert payload["status"] == "ok"
    assert payload["source_inventory_rows"][0]["source_id"] == "source-alpha"
    assert payload["source_inventory_rows"][0]["row_count"] == 4
    json.dumps(payload, allow_nan=False)


def test_source_rows_can_feed_existing_tabular_association_correlations() -> None:
    tabular_associations = _analysis_tabular_associations()
    prepared = prepare_tabular_association_source_rows({"source-alpha": _association_rows()})

    payload = tabular_associations.run_tabular_association_correlations(
        _workflow_doc(),
        source_rows_by_id=prepared.source_rows_by_id,
    ).to_dict()

    assert payload["status"] == "ok"
    assert payload["result_rows"][0]["source_id"] == "source-alpha"
    assert payload["result_rows"][0]["statistic_name"] == "r"
    assert payload["result_rows"][0]["statistic_value"] == pytest.approx(1.0)
    json.dumps(payload, allow_nan=False)


def test_importing_io_modules_does_not_import_heavy_analysis_or_domain_modules() -> None:
    for module_name in (
        "research_platform.io",
        "research_platform.io.dataframe",
        "research_platform.io.dataframe.association_records",
    ):
        completed = _run_import_check(module_name)
        assert completed.returncode == 0, completed.stderr
        assert json.loads(completed.stdout) == []


def test_production_module_has_no_analysis_imports_or_duplicate_analysis_wrappers() -> None:
    tree = ast.parse(MODULE_PATH.read_text())

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all(not alias.name.startswith("research_platform.analysis") for alias in node.names)
        if isinstance(node, ast.ImportFrom):
            assert node.module is None or not node.module.startswith("research_platform.analysis")
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            assert not node.name.startswith("run_tabular_association_")

    source_text = MODULE_PATH.read_text()
    forbidden_tokens = [
        "run_tabular_association_qc",
        "run_tabular_association_correlations",
        "run_tabular_association_adjusted",
        "pearson",
        "spearman",
        "partial_correlation",
        "regression",
        "p_value",
        "q_value",
        "confidence_interval",
        "effect_size",
        "fdr",
    ]
    for token in forbidden_tokens:
        assert token not in source_text


def test_production_module_has_no_study_specific_constants_or_subject_style_ids() -> None:
    source_text = MODULE_PATH.read_text()

    assert not re.search(r"\bsub-\d{3}\b", source_text)
    assert not re.search(r"\btask-[a-z0-9_-]+\b", source_text)
    assert not re.search(r"\bfull\d+\b", source_text)
    assert "confidential-study-marker" not in source_text


def test_optional_pandas_path_uses_explicit_backend() -> None:
    pd = pytest.importorskip("pandas")
    dataframe = pd.DataFrame({"column-alpha": ["row-alpha", "row-beta"]}, index=["index-alpha", "index-beta"])

    result = prepare_tabular_association_source_rows(
        {"source-alpha": dataframe},
        backend_by_source_id={"source-alpha": "pandas"},
        pandas_include_index=True,
        pandas_index_field="pandas_index",
    )

    assert result.valid is True
    assert result.backend_by_source_id == {"source-alpha": "pandas"}
    assert result.source_rows_by_id["source-alpha"] == (
        {"column-alpha": "row-alpha", "pandas_index": "index-alpha"},
        {"column-alpha": "row-beta", "pandas_index": "index-beta"},
    )


def test_optional_pandas_dataframe_is_not_auto_detected_without_explicit_backend() -> None:
    pd = pytest.importorskip("pandas")
    dataframe = pd.DataFrame({"column-alpha": ["row-alpha"]})

    result = prepare_tabular_association_source_rows({"source-alpha": dataframe})

    assert result.valid is False
    assert result.source_rows_by_id == {}
    assert result.conversion_results_by_id["source-alpha"].adapter_spec.requested_backend == "records"


def test_optional_polars_path_uses_explicit_backend() -> None:
    pl = pytest.importorskip("polars")
    dataframe = pl.DataFrame({"column-alpha": ["row-alpha", "row-beta"]})

    result = prepare_tabular_association_source_rows(
        {"source-alpha": dataframe},
        backend_by_source_id={"source-alpha": "polars"},
    )

    assert result.valid is True
    assert result.backend_by_source_id == {"source-alpha": "polars"}
    assert result.source_rows_by_id["source-alpha"] == (
        {"column-alpha": "row-alpha"},
        {"column-alpha": "row-beta"},
    )


def test_optional_polars_lazyframe_remains_rejected_and_uncollected() -> None:
    pl = pytest.importorskip("polars")
    lazy_frame = pl.DataFrame({"column-alpha": ["row-alpha"]}).lazy()

    result = prepare_tabular_association_source_rows(
        {"source-alpha": lazy_frame},
        backend_by_source_id={"source-alpha": "polars"},
    )

    assert result.valid is False
    assert result.source_rows_by_id == {}
    assert result.conversion_results_by_id["source-alpha"].row_source_kind == "polars_lazyframe"
    assert result.qc_rows[0].code == "polars_lazyframe_source"


def test_importing_association_records_does_not_load_optional_or_analysis_modules_after_reload() -> None:
    completed = _run_import_check("research_platform.io.dataframe.association_records")

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == []

    module = importlib.import_module("research_platform.io.dataframe.association_records")
    assert module.prepare_tabular_association_source_rows is prepare_tabular_association_source_rows
