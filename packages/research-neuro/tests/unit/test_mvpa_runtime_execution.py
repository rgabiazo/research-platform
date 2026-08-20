from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import csv
import json

import pytest

from research_platform.neuro.mvpa.materialized_pattern_table import (
    OPTIONAL_COLUMNS,
    REQUIRED_COLUMNS,
    SCHEMA_VERSION,
)
from research_platform.neuro.mvpa.plan import plan_mvpa_discovery
from research_platform.neuro.mvpa.runtime_execution import (
    MvpaRuntimeRepresentationError,
    materialize_mvpa_patterns_from_plan,
)


def _document() -> dict[str, object]:
    return {
        "mvpa_set": {
            "name": "prepared_runtime",
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
                    "path": "patterns.tsv",
                    "schema_version": SCHEMA_VERSION,
                }
            ],
            "roi_sources": [
                {
                    "name": "prepared-rois",
                    "source": "explicit_masks",
                    "root_ref": "pattern_root",
                    "roi_labels": ["SeedA"],
                    "mask_template": "unused/{roi_label}.nii",
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
                    "path": "mvpa/{mvpa_set}",
                }
            },
        }
    }


def _write_table(root: Path) -> Path:
    path = root / "patterns.tsv"
    columns = tuple(dict.fromkeys((*REQUIRED_COLUMNS, *OPTIONAL_COLUMNS)))
    row = {
        "schema_version": SCHEMA_VERSION,
        "pattern_id": "pattern-a",
        "subject_id": "sub-toy01",
        "run_id": "run-01",
        "condition_id": "condition-a",
        "pattern_source_name": "prepared-patterns",
        "roi_source_name": "prepared-rois",
        "roi_label": "SeedA",
        "feature_count": "3",
        "voxel_order": "c_flat_index",
        "voxel_index_hash": "index-a",
        "feature_space_id": "space-a",
        "roi_definition_id": "prepared-rois:SeedA",
        "feature_values": "[1.0,2.0,3.0]",
        "usable": "true",
        "status": "ok",
        "mean_centering_applied": "false",
        "mean_centering_scope": "none",
        "noise_status": "unused",
        "noise_usable": "false",
        "cross_validation_label": "run-01",
        "grouping_values": "{}",
        "warnings": "[]",
        "errors": "[]",
    }
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerow(row)
    return path


def _plan(root: Path):
    _write_table(root)
    return plan_mvpa_discovery(
        _document(),
        roots={"pattern_root": root},
        exact_units=({"subject_id": "sub-toy01", "run_id": "run-01"},),
        unit_key_columns=("subject_id", "run_id"),
    )


def test_materialized_dispatch_uses_private_exact_plan_handle(tmp_path: Path) -> None:
    plan = _plan(tmp_path)

    assert plan.ready_for_execution
    assert len(plan._execution_handles) == 1
    assert plan == replace(plan, _execution_handles=())
    public = json.dumps(plan.to_dict(), sort_keys=True)
    assert "_execution_handles" not in public
    assert str(tmp_path) not in public

    result = materialize_mvpa_patterns_from_plan(plan)

    assert result.valid
    assert result.executed
    assert result.representation_kind == "prepared_features"
    assert result.source_audit_kind == "pattern_materialization"
    assert len(result.pattern_rows) == 1
    assert result.pattern_rows[0]["feature_values"] == [1.0, 2.0, 3.0]
    assert result.pattern_rows[0]["cross_validation_label"] == "run-01"
    assert "pe_image" not in result.pattern_rows[0]
    assert "mask_path" not in result.pattern_rows[0]
    assert result.provenance["sources"][0]["source_reference"] == (
        "root_ref:pattern_root/patterns.tsv"
    )


def test_materialized_dispatch_detects_source_mutation_without_replanning(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    table = tmp_path / "patterns.tsv"
    table.write_bytes(table.read_bytes() + b"\n")

    result = materialize_mvpa_patterns_from_plan(plan)

    assert not result.valid
    assert not result.executed
    assert result.pattern_rows == ()
    assert any("changed after planning" in error for error in result.errors)


def test_materialized_dispatch_rejects_public_mapping_without_private_handle(tmp_path: Path) -> None:
    plan = _plan(tmp_path)

    with pytest.raises(MvpaRuntimeRepresentationError, match="private handles"):
        materialize_mvpa_patterns_from_plan(plan.to_dict())


def test_dispatch_rejects_mixed_and_unknown_representations(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    mixed = replace(
        plan,
        pattern_rows=(
            *plan.pattern_rows,
            {
                **plan.pattern_rows[0],
                "representation_kind": "image",
                "pattern_reference": "root_ref:image_root/pattern.nii",
            },
        ),
    )
    unknown = replace(
        plan,
        pattern_rows=({**plan.pattern_rows[0], "representation_kind": "deferred_kind"},),
    )

    with pytest.raises(MvpaRuntimeRepresentationError, match="mixed"):
        materialize_mvpa_patterns_from_plan(mixed)
    with pytest.raises(MvpaRuntimeRepresentationError, match="no implemented"):
        materialize_mvpa_patterns_from_plan(unknown)


def test_image_dispatch_uses_existing_extractor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[bool] = []

    def fake_extract(plan: object, *, load_noise: bool = False) -> dict[str, object]:
        calls.append(load_noise)
        return {
            "pattern_rows": [{"pattern_id": "image-pattern", "usable": True}],
            "qc_rows": [],
            "provenance": {"source": "image-test"},
            "warnings": [],
            "errors": [],
            "executed": True,
        }

    monkeypatch.setattr(
        "research_platform.neuro.mvpa.runtime_execution.extract_mvpa_patterns_from_discovery_plan",
        fake_extract,
    )
    plan = {
        "pattern_rows": [
            {
                "source_name": "images",
                "representation_kind": "image",
                "pattern_reference": "root_ref:image_root/pattern.nii",
            }
        ]
    }

    result = materialize_mvpa_patterns_from_plan(plan, load_noise=True)

    assert calls == [True]
    assert result.valid
    assert result.representation_kind == "image"
    assert result.source_audit_kind == "image_extraction"
    assert result.pattern_rows[0]["pattern_id"] == "image-pattern"
