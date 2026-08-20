from __future__ import annotations

from pathlib import Path
import csv
import copy
import json

import pytest

from research_platform.neuro.mvpa.config import validate_mvpa_set_document
from research_platform.neuro.mvpa.materialized_pattern_table import (
    OPTIONAL_COLUMNS,
    REQUIRED_COLUMNS,
    SCHEMA_VERSION,
)
from research_platform.neuro.mvpa.pattern_source_adapters import (
    default_pattern_source_adapter_registry,
)
from research_platform.neuro.mvpa.pattern_sources import (
    PATTERN_BACKEND_MATERIALIZED_PATTERN_TABLE,
)
from research_platform.neuro.mvpa.plan import plan_mvpa_discovery


def _document() -> dict[str, object]:
    return {
        "mvpa_set": {
            "name": "toy_prepared_features",
            "unit_selection": {
                "mode": "exact_units",
                "key_columns": ["subject_id", "run_id"],
            },
            "conditions": [{"id": "condition-a"}],
            "pattern_sources": [
                {
                    "name": "prepared-patterns",
                    "backend": "materialized_pattern_table",
                    "root_ref": "pattern_root",
                    "path": "tables/patterns.tsv",
                    "schema_version": SCHEMA_VERSION,
                }
            ],
            "roi_sources": [
                {
                    "name": "toy-rois",
                    "source": "explicit_masks",
                    "root_ref": "pattern_root",
                    "roi_labels": ["SeedA"],
                    "mask_template": "rois/{roi_label}.nii",
                }
            ],
            "distance": {
                "metrics": ["crossnobis"],
                "engine": "native_reference",
                "cross_validation": {"unit": "run"},
                "noise_normalization": {"method": "identity"},
            },
            "outputs": {
                "runtime_root": {
                    "root_ref": "artifact_root",
                    "path": ".research-platform/mvpa/{mvpa_set}",
                }
            },
        }
    }


def _unit() -> dict[str, str]:
    return {
        "subject_id": "sub-toy01",
        "session_id": "ses-01",
        "task_id": "exampletask",
        "run_id": "run-01",
        "visit_index": "1",
    }


def _write_table(root: Path, *, usable: bool = True) -> Path:
    path = root / "tables" / "patterns.tsv"
    path.parent.mkdir(parents=True)
    columns = tuple(dict.fromkeys((*REQUIRED_COLUMNS, *OPTIONAL_COLUMNS)))
    row = {
        "schema_version": SCHEMA_VERSION,
        "pattern_id": "pattern-a",
        "subject_id": "sub-toy01",
        "session_id": "ses-01",
        "task_id": "exampletask",
        "run_id": "run-01",
        "condition_id": "condition-a",
        "pattern_source_name": "prepared-patterns",
        "roi_source_name": "toy-rois",
        "roi_label": "SeedA",
        "feature_count": "3",
        "voxel_order": "c-order",
        "voxel_index_hash": "index-hash-a",
        "feature_space_id": "space-a",
        "roi_definition_id": "toy-rois:SeedA",
        "feature_values": "[1.0,2.0,3.0]",
        "usable": "true" if usable else "false",
        "status": "ok" if usable else "excluded",
        "mean_centering_applied": "false",
        "mean_centering_scope": "none",
        "noise_status": "unused",
        "noise_usable": "false",
        "cross_validation_label": "run-01",
        "event_count": "4",
        "qc_status": "pass" if usable else "excluded",
        "qc_reason": "" if usable else "Synthetic deterministic exclusion",
        "exclusion_id": "" if usable else "rule-toy",
        "exclusion_reason": "" if usable else "Synthetic deterministic exclusion",
        "roi_reference": "root_ref:pattern_root/rois/SeedA.nii",
        "grouping_values": "{}",
        "warnings": "[]",
        "errors": "[]",
    }
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=columns,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerow(row)
    return path


def test_registry_exposes_materialization_and_local_execution() -> None:
    adapter = default_pattern_source_adapter_registry().require(
        PATTERN_BACKEND_MATERIALIZED_PATTERN_TABLE
    )

    assert adapter.capabilities.schema_supported
    assert adapter.capabilities.planning_supported
    assert adapter.capabilities.materialization_supported
    assert adapter.capabilities.execution_ready
    assert adapter.capabilities.representation_kinds == ("prepared_features",)


def test_materialized_source_requires_exact_units_and_rejects_mixed_representations() -> None:
    legacy = _document()
    selection = legacy["mvpa_set"].pop("unit_selection")  # type: ignore[index]
    legacy["mvpa_set"].update(  # type: ignore[index]
        {
            "subjects": ["sub-toy01"],
            "runs": ["run-01"],
        }
    )
    errors = validate_mvpa_set_document(legacy)
    assert any("unit_selection.mode=exact_units" in error for error in errors)

    mixed = _document()
    mixed["mvpa_set"]["pattern_sources"].append(  # type: ignore[index]
        {
            "name": "image-patterns",
            "backend": "fsl_feat_pe",
            "root_ref": "feat_root",
            "feat_dir_template": "{subject_dir}/model.feat",
            "design_file": "design.fsf",
            "pe_image_template": "stats/pe{pe_number}.nii.gz",
        }
    )
    mixed_errors = validate_mvpa_set_document(mixed)
    assert any("must not mix image and prepared_features" in error for error in mixed_errors)
    assert selection == {
        "mode": "exact_units",
        "key_columns": ["subject_id", "run_id"],
    }


def test_integrated_materialized_plan_is_vector_free_portable_and_skips_roi_masks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    table = _write_table(tmp_path)
    before = table.read_bytes()

    def forbidden_roi_plan(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("prepared-feature planning invoked ROI-mask discovery")

    monkeypatch.setattr(
        "research_platform.neuro.mvpa.plan.plan_mvpa_roi_sources",
        forbidden_roi_plan,
    )
    plan = plan_mvpa_discovery(
        _document(),
        roots={"pattern_root": tmp_path},
        exact_units=(_unit(),),
        unit_key_columns=("subject_id", "run_id"),
    )

    assert table.read_bytes() == before
    assert plan.schema_valid
    assert plan.status == "valid"
    assert plan.ready_for_materialization
    assert plan.ready_for_execution
    assert not plan.executed
    assert len(plan.pattern_rows) == 1
    assert plan.pattern_rows[0]["representation_kind"] == "prepared_features"
    assert plan.pattern_rows[0]["pattern_reference"] == (
        "root_ref:pattern_root/tables/patterns.tsv#row=1"
    )
    assert plan.pattern_rows[0]["unit_metadata"]["visit_index"] == "1"
    assert plan.pattern_source_rows == ()
    assert plan.condition_pe_rows == ()
    assert plan.roi_source_rows == ()
    assert plan.roi_sources[0].status == "valid"
    assert plan.roi_sources[0].reason == (
        "roi_coverage_validated_by_materialized_pattern_table"
    )

    availability = plan.adapter_availability[0]
    assert availability["materialization_supported"] is True
    assert availability["ready_for_materialization"] is True
    assert availability["ready_for_execution"] is True
    assert plan.backend_summary["ready_for_materialization"] is True
    assert plan.backend_summary["ready_for_execution"] is True

    assert len(plan.pattern_source_metadata_rows) == 1
    source_row = plan.pattern_source_metadata_rows[0]
    assert source_row["pattern_id"] == "pattern-a"
    assert source_row["source_reference"].endswith("#row=1")
    assert "feature_values" not in source_row
    assert "noise_values" not in source_row
    summary = plan.pattern_source_summaries[0]
    assert summary["counts"] == {
        "total_rows": 1,
        "selected_rows": 1,
        "usable_selected_rows": 1,
        "usable_units": 1,
        "usable_conditions": 1,
        "usable_rois": 1,
        "unselected_rows": 0,
    }
    assert summary["usable_coverage_complete"] is True
    assert len(summary["source_sha256"]) == 64
    provenance = plan.pattern_source_provenance[0]
    assert provenance["source_reference"] == (
        "root_ref:pattern_root/tables/patterns.tsv"
    )
    assert provenance["source_sha256"] == summary["source_sha256"]

    public_payload = json.dumps(plan.to_dict(), sort_keys=True)
    assert str(tmp_path) not in public_payload
    assert "[1.0,2.0,3.0]" not in public_payload
    assert "materialization_handle" not in public_payload


def test_all_excluded_materialized_source_is_materializable_but_not_execution_ready(
    tmp_path: Path,
) -> None:
    _write_table(tmp_path, usable=False)

    plan = plan_mvpa_discovery(
        _document(),
        roots={"pattern_root": tmp_path},
        exact_units=(_unit(),),
        unit_key_columns=("subject_id", "run_id"),
    )

    summary = plan.pattern_source_summaries[0]
    assert plan.schema_valid
    assert plan.status == "warning"
    assert plan.ready_for_materialization
    assert not plan.ready_for_execution
    assert summary["counts"]["usable_selected_rows"] == 0
    assert summary["usable_coverage_complete"] is False
    assert plan.adapter_availability[0]["ready_for_execution"] is False


def test_materialized_plan_retains_a_private_execution_handle(
    tmp_path: Path,
) -> None:
    _write_table(tmp_path)
    first = plan_mvpa_discovery(
        _document(),
        roots={"pattern_root": tmp_path},
        exact_units=(_unit(),),
        unit_key_columns=("subject_id", "run_id"),
    )
    second = plan_mvpa_discovery(
        copy.deepcopy(_document()),
        roots={"pattern_root": tmp_path},
        exact_units=(copy.deepcopy(_unit()),),
        unit_key_columns=("subject_id", "run_id"),
    )

    assert first.to_dict() == second.to_dict()
    assert first == second
    assert first.ready_for_materialization
    assert first.ready_for_execution
    assert len(first._execution_handles) == 1
    assert first._execution_handles[0].source_name == "prepared-patterns"
    assert "_execution_handles" not in first.to_dict()
    assert first.pattern_source_rows == ()
    assert first.condition_pe_rows == ()
    assert first.roi_source_rows == ()


def test_failed_materialized_source_does_not_claim_roi_coverage_validation(
    tmp_path: Path,
) -> None:
    plan = plan_mvpa_discovery(
        _document(),
        roots={"pattern_root": tmp_path},
        exact_units=(_unit(),),
        unit_key_columns=("subject_id", "run_id"),
    )

    assert plan.errors
    assert not plan.ready_for_materialization
    assert plan.roi_sources[0].status == "error"
    assert plan.roi_sources[0].reason == (
        "roi_coverage_not_validated_due_to_materialized_source_errors"
    )


def test_public_plan_rejects_unsafe_exact_unit_metadata_without_echoing_it(
    tmp_path: Path,
) -> None:
    _write_table(tmp_path)
    unsafe_unit = _unit()
    unsafe_unit["input_note"] = "--input=/home/alice/private.tsv"

    plan = plan_mvpa_discovery(
        _document(),
        roots={"pattern_root": tmp_path},
        exact_units=(unsafe_unit,),
        unit_key_columns=("subject_id", "run_id"),
    )
    payload = json.dumps(plan.to_dict(), sort_keys=True)

    assert plan.errors
    assert any("non-portable local path reference" in error for error in plan.errors)
    assert "/home/alice/private.tsv" not in payload
    assert plan.analysis_units == ()
