from __future__ import annotations

from dataclasses import dataclass
import importlib
import json
import math
import sys
from typing import Any

import pytest

from research_platform.analysis.tabular_associations import (
    BACKEND_PANDAS,
    BACKEND_POLARS,
    BACKEND_RECORDS,
    TabularAssociationRecordsAdapter,
    TabularAssociationRowSourceProvenanceRow,
    TabularAssociationRowSourceQcRow,
    coerce_tabular_association_records,
    inspect_tabular_association_row_source,
    iter_tabular_association_records,
    plan_tabular_association_row_source_adapter,
    run_tabular_association_correlations,
    run_tabular_association_qc,
)


def _workflow_doc(*, backend: str = "records") -> dict[str, Any]:
    return {
        "workflow_id": "workflow-alpha",
        "backend": backend,
        "sources": [
            {
                "source_id": "source-alpha",
                "backend": backend,
                "schema": {
                    "subject_id_column": "participant-id",
                    "columns": [
                        {"column_name": "participant-id", "value_type": "categorical", "role": "subject_identifier"},
                        {"column_name": "outcome-alpha", "value_type": "numeric", "role": "outcome"},
                        {"column_name": "predictor-alpha", "value_type": "numeric", "role": "predictor"},
                    ],
                    "numeric_validation": {"policy": "declare"},
                },
            }
        ],
        "outcomes": [
            {"variable_id": "outcome-alpha", "source_id": "source-alpha", "column_name": "outcome-alpha"}
        ],
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


def _clean_rows() -> list[dict[str, object]]:
    return [
        {"participant-id": "participant-a", "outcome-alpha": "1", "predictor-alpha": "2"},
        {"participant-id": "participant-b", "outcome-alpha": "2", "predictor-alpha": "4"},
        {"participant-id": "participant-c", "outcome-alpha": "3", "predictor-alpha": "6"},
    ]


@dataclass(frozen=True)
class DataclassRow:
    row_alpha: str
    outcome_alpha: str
    predictor_alpha: str


class RowWithToDict:
    def __init__(self, row_alpha: str, outcome_alpha: str, predictor_alpha: str) -> None:
        self.row_alpha = row_alpha
        self.outcome_alpha = outcome_alpha
        self.predictor_alpha = predictor_alpha

    def to_dict(self) -> dict[str, str]:
        return {
            "row-alpha": self.row_alpha,
            "outcome-alpha": self.outcome_alpha,
            "predictor-alpha": self.predictor_alpha,
        }


class FakeToDicts:
    columns = ("participant-id", "outcome-alpha", "predictor-alpha")

    def __init__(self, rows: list[dict[str, object]]) -> None:
        self._rows = [dict(row) for row in rows]

    def to_dicts(self) -> list[dict[str, object]]:
        return [dict(row) for row in self._rows]


class FakeToRecords:
    columns = ("participant-id", "outcome-alpha", "predictor-alpha")

    def __init__(self, rows: list[dict[str, object]]) -> None:
        self._rows = [dict(row) for row in rows]

    def to_records(self) -> tuple[dict[str, object], ...]:
        return tuple(dict(row) for row in self._rows)


class FakeIterRowsNamed:
    columns = ("participant-id", "outcome-alpha", "predictor-alpha")

    def __init__(self, rows: list[dict[str, object]]) -> None:
        self._rows = [dict(row) for row in rows]

    def iter_rows(self, named: bool = False) -> Any:
        if named:
            return (dict(row) for row in self._rows)
        return ((row["participant-id"], row["outcome-alpha"], row["predictor-alpha"]) for row in self._rows)


class FakeRowsAttribute:
    columns = ("participant-id", "outcome-alpha", "predictor-alpha")

    def __init__(self, rows: list[dict[str, object]]) -> None:
        self.rows = [dict(row) for row in rows]


class FakeRecordsMethod:
    columns = ("participant-id", "outcome-alpha", "predictor-alpha")

    def __init__(self, rows: list[dict[str, object]]) -> None:
        self._rows = [dict(row) for row in rows]

    def records(self) -> tuple[dict[str, object], ...]:
        return tuple(dict(row) for row in self._rows)


def _forbidden_imports(imported_modules: set[str]) -> set[str]:
    forbidden_prefixes = (
        "pandas",
        "polars",
        "numpy",
        "scipy",
        "sklearn",
        "statsmodels",
        "research_platform.io",
        "research_platform.viz",
        "research_platform.core",
        "research_platform.neuro",
        "research_platform.bids",
    )
    return {
        module_name
        for module_name in imported_modules
        if any(module_name == prefix or module_name.startswith(f"{prefix}.") for prefix in forbidden_prefixes)
    }


def test_importing_tabular_associations_does_not_import_deferred_backends() -> None:
    parent = importlib.import_module("research_platform.analysis")
    sys.modules.pop("research_platform.analysis.tabular_associations", None)
    if hasattr(parent, "tabular_associations"):
        delattr(parent, "tabular_associations")

    before_modules = set(sys.modules)
    importlib.import_module("research_platform.analysis.tabular_associations")
    imported_during_import = set(sys.modules) - before_modules

    assert _forbidden_imports(imported_during_import) == set()


def test_adapter_plan_defaults_to_records_runtime_and_no_write_flags() -> None:
    payload = plan_tabular_association_row_source_adapter().to_dict()

    assert payload["requested_backend"] == "records"
    assert payload["runtime_backend"] == "records"
    assert payload["executed"] is False
    assert payload["plan_only"] is True
    assert payload["will_write"] is False
    assert payload["output_written"] is False
    assert payload["no_output_written"] is True
    assert payload["output_paths_written"] == []
    json.dumps(payload, sort_keys=True, allow_nan=False)


@pytest.mark.parametrize("requested_backend", [BACKEND_RECORDS, BACKEND_PANDAS, BACKEND_POLARS])
def test_requested_backend_values_validate_without_backend_imports(requested_backend: str) -> None:
    before_modules = set(sys.modules)

    payload = plan_tabular_association_row_source_adapter(requested_backend=requested_backend).to_dict()
    imported_during_plan = set(sys.modules) - before_modules

    assert payload["requested_backend"] == requested_backend
    assert payload["runtime_backend"] == "records"
    assert _forbidden_imports(imported_during_plan) == set()


def test_mapping_row_sequences_coerce_to_copied_record_tuples_with_stable_columns() -> None:
    rows = [
        {"participant-id": "participant-a", "outcome-alpha": "1"},
        {"participant-id": "participant-b", "predictor-alpha": "4", "outcome-alpha": "2"},
    ]

    result = coerce_tabular_association_records(rows)
    rows[0]["outcome-alpha"] = "changed"

    assert result.valid is True
    assert isinstance(result.records, tuple)
    assert result.records[0] is not rows[0]
    assert result.records[0]["outcome-alpha"] == "1"
    assert result.observed_columns == ("participant-id", "outcome-alpha", "predictor-alpha")
    assert tuple(iter_tabular_association_records(rows))[0]["outcome-alpha"] == "changed"
    json.dumps(result.to_dict(), sort_keys=True, allow_nan=False)


def test_dataclass_rows_and_row_to_dict_objects_are_accepted() -> None:
    dataclass_result = coerce_tabular_association_records(
        [DataclassRow(row_alpha="row-alpha", outcome_alpha="1", predictor_alpha="2")]
    )
    object_result = coerce_tabular_association_records(
        [RowWithToDict(row_alpha="row-alpha", outcome_alpha="3", predictor_alpha="4")]
    )

    assert dataclass_result.valid is True
    assert dataclass_result.records[0]["row_alpha"] == "row-alpha"
    assert object_result.valid is True
    assert object_result.records[0]["row-alpha"] == "row-alpha"


@pytest.mark.parametrize(
    ("factory", "row_source_kind"),
    [
        (FakeToDicts, "to_dicts"),
        (FakeToRecords, "to_records"),
        (FakeIterRowsNamed, "iter_rows_named"),
        (FakeRowsAttribute, "rows"),
        (FakeRecordsMethod, "records"),
    ],
)
def test_fake_dataframe_like_protocols_convert_to_records_without_backend_imports(
    factory: type[Any],
    row_source_kind: str,
) -> None:
    row_source = factory(_clean_rows())
    before_modules = set(sys.modules)

    result = coerce_tabular_association_records(row_source, requested_backend="polars")
    imported_during_coercion = set(sys.modules) - before_modules

    assert result.valid is True
    assert result.spec.requested_backend == "polars"
    assert result.spec.runtime_backend == "records"
    assert result.spec.row_source_kind == row_source_kind
    assert result.records == tuple(_clean_rows())
    assert result.observed_columns == ("participant-id", "outcome-alpha", "predictor-alpha")
    assert _forbidden_imports(imported_during_coercion) == set()


def test_fake_row_source_can_feed_qc_via_source_rows_by_id() -> None:
    result = run_tabular_association_qc(
        _workflow_doc(backend="pandas"),
        source_rows_by_id={"source-alpha": FakeToDicts(_clean_rows())},
    )
    payload = result.to_dict()
    source_load_row = payload["source_load_rows"][0]
    source_inventory_row = payload["source_inventory_rows"][0]

    assert payload["will_write"] is False
    assert payload["output_written"] is False
    assert source_load_row["load_status"] == "loaded"
    assert source_inventory_row["requested_backend"] == "pandas"
    assert source_inventory_row["runtime_backend"] == "records"
    assert source_inventory_row["provenance"]["row_source_kind"] == "to_dicts"
    assert source_inventory_row["provenance"]["row_source_no_output_written"] is True
    assert source_inventory_row["provenance"]["row_source_output_paths_written"] == []
    json.dumps(payload, sort_keys=True, allow_nan=False)


def test_fake_row_source_can_feed_correlation_execution_without_new_statistical_paths() -> None:
    payload = run_tabular_association_correlations(
        _workflow_doc(),
        source_rows_by_id={"source-alpha": FakeIterRowsNamed(_clean_rows())},
    ).to_dict()
    result_row = payload["result_rows"][0]

    assert result_row["statistic_name"] == "r"
    assert result_row["statistic_value"] == pytest.approx(1.0)
    assert result_row["n_total"] == 3
    assert payload["will_write"] is False
    assert payload["output_written"] is False


def test_unsupported_sources_and_positional_rows_report_adapter_errors() -> None:
    unsupported = inspect_tabular_association_row_source(object()).to_dict()
    positional = coerce_tabular_association_records([("participant-a", "1", "2")]).to_dict()

    assert unsupported["valid"] is False
    assert unsupported["status"] == "error"
    assert unsupported["records"] == []
    assert unsupported["qc_rows"][0]["errors"]
    assert positional["valid"] is False
    assert positional["records"] == []
    assert "unsupported row" in positional["errors"][0]
    json.dumps(unsupported, sort_keys=True, allow_nan=False)
    json.dumps(positional, sort_keys=True, allow_nan=False)


def test_input_row_index_is_optional_deterministic_and_collision_safe() -> None:
    indexed = coerce_tabular_association_records(
        [{"row-alpha": "row-alpha"}, {"row-alpha": "row-beta"}],
        include_input_row_index=True,
    )
    collision = coerce_tabular_association_records(
        [{"row-alpha": "row-alpha", "input_row_index": "existing"}],
        include_input_row_index=True,
    )

    assert indexed.valid is True
    assert [row["input_row_index"] for row in indexed.records] == [0, 1]
    assert indexed.observed_columns == ("row-alpha", "input_row_index")
    assert collision.valid is False
    assert collision.records == ()
    assert "already exists" in collision.errors[0]
    assert collision.qc_rows[0]["code"] == "row_source_input_row_index_collision"


def test_supplied_rows_and_fake_dataframe_internals_are_not_mutated() -> None:
    rows = [{"row-alpha": "row-alpha", "outcome-alpha": "1"}]
    fake = FakeToDicts(_clean_rows())
    original_fake_rows = [dict(row) for row in fake._rows]

    row_result = coerce_tabular_association_records(rows, include_input_row_index=True)
    fake_result = coerce_tabular_association_records(fake, include_input_row_index=True)

    assert "input_row_index" not in rows[0]
    assert fake._rows == original_fake_rows
    assert "input_row_index" not in fake._rows[0]
    assert row_result.records[0]["input_row_index"] == 0
    assert fake_result.records[0]["input_row_index"] == 0


def test_result_payloads_are_json_safe_without_raw_nonfinite_values() -> None:
    result = coerce_tabular_association_records(
        [{"row-alpha": "row-alpha", "outcome-alpha": math.nan, "predictor-alpha": math.inf}]
    )
    payload = result.to_dict()

    assert payload["records"][0]["outcome-alpha"] == "nan"
    assert payload["records"][0]["predictor-alpha"] == "inf"
    json.dumps(payload, sort_keys=True, allow_nan=False)


def test_row_source_qc_and_provenance_rows_have_tsv_safe_forms() -> None:
    qc_row = TabularAssociationRowSourceQcRow(
        adapter_id="adapter-alpha",
        requested_backend="records",
        runtime_backend="records",
        row_source_kind="mapping_sequence",
        status="ok",
        code="row_source_records_coerced",
        message="Coerced records.",
        row_count=1,
        observed_column_count=2,
    )
    provenance_row = TabularAssociationRowSourceProvenanceRow(
        adapter_id="adapter-alpha",
        key="output_paths_written",
        value=(),
    )

    assert qc_row.to_tsv_row()["include_input_row_index"] == "false"
    assert provenance_row.to_tsv_row()["value"] == "[]"


def test_records_adapter_object_exposes_plan_coerce_and_iter_helpers() -> None:
    adapter = TabularAssociationRecordsAdapter(
        plan_tabular_association_row_source_adapter(requested_backend="pandas")
    )

    assert adapter.to_dict()["spec"]["requested_backend"] == "pandas"
    assert adapter.plan().runtime_backend == "records"
    assert adapter.coerce(_clean_rows()).records == tuple(_clean_rows())
    assert tuple(adapter.iter_records(_clean_rows())) == tuple(_clean_rows())
