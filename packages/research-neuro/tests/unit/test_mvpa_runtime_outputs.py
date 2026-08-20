from __future__ import annotations

import ast
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path

import pytest

from research_platform.neuro import _version as neuro_version
import research_platform.neuro.mvpa.runtime_outputs as runtime_outputs
from research_platform.neuro.mvpa.runtime_outputs import (
    plan_mvpa_pattern_extraction_outputs,
    plan_mvpa_pattern_materialization_outputs,
    write_mvpa_pattern_extraction_outputs,
    write_mvpa_pattern_materialization_outputs,
    write_mvpa_runtime_json,
    write_mvpa_runtime_tsv,
)


_PATTERN_HEADER = [
    "pattern_id",
    "condition_id",
    "cv_unit",
    "subject_id",
    "session_id",
    "run_id",
    "task_id",
    "direction",
    "model",
    "pattern_source_name",
    "roi_source_name",
    "roi_label",
    "pe_image",
    "mask_path",
    "noise_image",
    "voxel_count",
    "valid_voxel_count",
    "feature_count",
    "voxel_order",
    "voxel_index_hash",
    "usable",
    "feature_values",
    "event_count",
    "mean_centering_applied",
    "mean_centering_scope",
    "grouping_values",
    "noise_loaded",
    "noise_status",
    "noise_usable",
    "noise_feature_count",
    "noise_voxel_order",
    "noise_voxel_index_hash",
    "noise_min",
    "noise_max",
    "noise_mean",
    "noise_nonfinite_count",
    "noise_nonpositive_count",
    "noise_values",
]
_QC_HEADER = [
    "subject_id",
    "session_id",
    "run_id",
    "condition_id",
    "roi_label",
    "pattern_source_name",
    "roi_source_name",
    "pe_image",
    "mask_path",
    "noise_image",
    "status",
    "usable",
    "reason",
    "excluded",
    "exclusion_reason",
    "exclusion_id",
    "exclusion_source_field",
    "skipped_stage",
    "pe_exists",
    "mask_exists",
    "noise_exists",
    "geometry_status",
    "mask_status",
    "voxel_count",
    "valid_voxel_count",
    "warnings",
    "errors",
    "event_threshold_status",
    "grouping_values",
    "noise_loaded",
    "noise_status",
    "noise_usable",
    "noise_feature_count",
    "noise_voxel_order",
    "noise_voxel_index_hash",
    "noise_min",
    "noise_max",
    "noise_mean",
    "noise_nonfinite_count",
    "noise_nonpositive_count",
]


@dataclass(frozen=True)
class _PatternRow:
    pattern_id: str
    condition_id: str
    cv_unit: str
    subject_id: str
    session_id: str | None
    run_id: str
    task_id: str
    direction: str
    model: str
    pattern_source_name: str
    roi_source_name: str
    roi_label: str
    pe_image: str
    mask_path: str
    noise_image: str
    voxel_count: int
    valid_voxel_count: int
    feature_count: int
    voxel_order: str
    voxel_index_hash: str
    usable: bool
    feature_values: tuple[float, ...]
    event_count: int | None
    noise_loaded: bool
    noise_status: str
    noise_usable: bool
    noise_feature_count: int
    noise_voxel_order: str
    noise_voxel_index_hash: str
    noise_min: float
    noise_max: float
    noise_mean: float
    noise_nonfinite_count: int
    noise_nonpositive_count: int
    noise_values: tuple[float, ...]


class _ResultLike:
    def to_dict(self) -> dict[str, object]:
        return {
            "pattern_rows": (_pattern_row(),),
            "qc_rows": (_qc_row(),),
            "provenance": {
                "source": "synthetic-test-fixture",
                "phase": "3C.2",
                "load_noise": True,
                "output_written": False,
            },
            "warnings": ("Synthetic warning.",),
            "errors": (),
            "executed": True,
        }


@dataclass(frozen=True)
class _DataclassResult:
    pattern_rows: tuple[_PatternRow, ...]
    qc_rows: tuple[dict[str, object], ...]
    provenance: dict[str, object]
    warnings: tuple[str, ...]
    errors: tuple[str, ...]
    executed: bool = True


def _pattern_row() -> _PatternRow:
    return _PatternRow(
        pattern_id="pat-sub01-face-v1",
        condition_id="face",
        cv_unit="run",
        subject_id="sub-01",
        session_id=None,
        run_id="run-01",
        task_id="localizer",
        direction="AP",
        model="model-01",
        pattern_source_name="fsl-pe",
        roi_source_name="manual-roi",
        roi_label="ffa",
        pe_image="derivatives/feat/sub-01/pe1.nii.gz",
        mask_path="derivatives/rois/sub-01/ffa.nii.gz",
        noise_image="derivatives/feat/sub-01/sigmasquareds.nii.gz",
        voxel_count=2,
        valid_voxel_count=2,
        feature_count=2,
        voxel_order="c_flat_index",
        voxel_index_hash="hash-a",
        usable=True,
        feature_values=(1.0, 2.5),
        event_count=2,
        noise_loaded=True,
        noise_status="ok",
        noise_usable=True,
        noise_feature_count=2,
        noise_voxel_order="c_flat_index",
        noise_voxel_index_hash="hash-a",
        noise_min=0.25,
        noise_max=0.5,
        noise_mean=0.375,
        noise_nonfinite_count=0,
        noise_nonpositive_count=0,
        noise_values=(0.25, 0.5),
    )


def _qc_row() -> dict[str, object]:
    return {
        "subject_id": "sub-01",
        "session_id": None,
        "run_id": "run-01",
        "condition_id": "face",
        "roi_label": "ffa",
        "pattern_source_name": "fsl-pe",
        "roi_source_name": "manual-roi",
        "pe_image": "derivatives/feat/sub-01/pe1.nii.gz",
        "mask_path": "derivatives/rois/sub-01/ffa.nii.gz",
        "noise_image": "derivatives/feat/sub-01/sigmasquareds.nii.gz",
        "status": "ok",
        "usable": True,
        "reason": None,
        "excluded": False,
        "exclusion_reason": None,
        "pe_exists": True,
        "mask_exists": True,
        "noise_exists": True,
        "geometry_status": "ok",
        "mask_status": "ok",
        "voxel_count": 2,
        "valid_voxel_count": 2,
        "warnings": ("Synthetic warning.",),
        "errors": (),
        "event_threshold_status": "ok",
        "noise_loaded": True,
        "noise_status": "ok",
        "noise_usable": True,
        "noise_feature_count": 2,
        "noise_voxel_order": "c_flat_index",
        "noise_voxel_index_hash": "hash-a",
        "noise_min": 0.25,
        "noise_max": 0.5,
        "noise_mean": 0.375,
        "noise_nonfinite_count": 0,
        "noise_nonpositive_count": 0,
    }


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _header(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()[0].split("\t")


def test_pattern_extraction_outputs_write_rows_qc_provenance_and_vector_metadata(tmp_path: Path, monkeypatch) -> None:
    output_root = tmp_path / "cache"
    monkeypatch.setattr(neuro_version.metadata, "version", lambda name: "0.1.0a1")

    record = write_mvpa_pattern_extraction_outputs(
        _ResultLike(),
        output_root=output_root,
        patterns_path="mvpa/patterns.tsv",
        qc_path="mvpa/qc.tsv",
        provenance_path="mvpa/provenance.json",
        vector_metadata_path="mvpa/vector_metadata.json",
    )

    assert record["will_write"] is True
    assert [artifact["name"] for artifact in record["artifacts"]] == [
        "patterns",
        "qc",
        "provenance",
        "vector_metadata",
    ]
    assert _header(output_root / "mvpa/patterns.tsv") == _PATTERN_HEADER
    assert _header(output_root / "mvpa/qc.tsv") == _QC_HEADER

    rows = _read_tsv(output_root / "mvpa/patterns.tsv")
    assert rows[0]["pattern_id"] == "pat-sub01-face-v1"
    assert rows[0]["session_id"] == ""
    assert rows[0]["usable"] == "true"
    assert rows[0]["feature_values"] == "[1.0,2.5]"
    assert rows[0]["event_count"] == "2"
    assert rows[0]["noise_values"] == "[0.25,0.5]"
    assert rows[0]["noise_mean"] == "0.375"

    qc_rows = _read_tsv(output_root / "mvpa/qc.tsv")
    assert qc_rows[0]["warnings"] == '["Synthetic warning."]'
    assert qc_rows[0]["errors"] == "[]"
    assert qc_rows[0]["excluded"] == "false"

    provenance_text = (output_root / "mvpa/provenance.json").read_text(encoding="utf-8")
    provenance = json.loads(provenance_text)
    assert str(tmp_path) not in provenance_text
    assert provenance["schema_version"] == runtime_outputs.SCHEMA_VERSION
    assert provenance["artifact_kind"] == "mvpa_pattern_extraction_outputs"
    assert provenance["writer_module"] == "research_platform.neuro.mvpa.runtime_outputs"
    assert provenance["writer_package"] == "research-neuro"
    assert provenance["writer_version"] == "0.1.0a1"
    assert provenance["output_written"] is True
    assert provenance["output_paths"] == {
        "patterns": "mvpa/patterns.tsv",
        "qc": "mvpa/qc.tsv",
        "provenance": "mvpa/provenance.json",
        "vector_metadata": "mvpa/vector_metadata.json",
    }
    assert provenance["row_counts"] == {"patterns": 1, "qc": 1, "vector_metadata": 1}
    assert provenance["columns"]["patterns"] == _PATTERN_HEADER
    assert provenance["columns"]["qc"] == _QC_HEADER
    assert provenance["input_provenance"]["phase"] == "3C.2"
    assert provenance["warnings"] == ["Synthetic warning."]
    assert provenance["errors"] == []
    assert provenance["executed"] is True

    vector_text = (output_root / "mvpa/vector_metadata.json").read_text(encoding="utf-8")
    assert vector_text.count("\n") == 1
    assert "feature_values" not in vector_text
    assert "noise_values" not in vector_text
    vector_metadata = json.loads(vector_text)
    assert vector_metadata["row_count"] == 1
    assert vector_metadata["vectors"][0]["pattern_id"] == "pat-sub01-face-v1"
    assert vector_metadata["vectors"][0]["feature_count"] == 2


def test_dry_run_preview_creates_no_files_or_directories(tmp_path: Path) -> None:
    output_root = tmp_path / "dry-cache"

    plan = plan_mvpa_pattern_extraction_outputs(
        _ResultLike(),
        output_root=output_root,
        patterns_path="mvpa/patterns.tsv",
        qc_path="mvpa/qc.tsv",
        provenance_path="mvpa/provenance.json",
        vector_metadata_path="mvpa/vector_metadata.json",
    )

    assert json.loads(json.dumps(plan, allow_nan=False)) == plan
    assert plan["will_write"] is False
    assert plan["output_written"] is False
    assert not output_root.exists()
    assert [artifact["will_write"] for artifact in plan["artifacts"]] == [False, False, False, False]
    assert [artifact["exists"] for artifact in plan["artifacts"]] == [False, False, False, False]
    assert plan["artifacts"][0]["columns"] == _PATTERN_HEADER
    assert plan["artifacts"][0]["row_count"] == 1


def test_dataclass_result_and_mapping_payloads_are_accepted(tmp_path: Path) -> None:
    dataclass_plan = plan_mvpa_pattern_extraction_outputs(
        _DataclassResult(
            pattern_rows=(_pattern_row(),),
            qc_rows=(_qc_row(),),
            provenance={"shape": "dataclass"},
            warnings=(),
            errors=(),
        ),
        output_root=tmp_path / "dataclass-cache",
        patterns_path="mvpa/patterns.tsv",
        qc_path="mvpa/qc.tsv",
        provenance_path="mvpa/provenance.json",
    )
    mapping_plan = plan_mvpa_pattern_extraction_outputs(
        {
            "pattern_rows": [_pattern_row()],
            "qc_rows": [_qc_row()],
            "provenance": {"shape": "mapping"},
            "warnings": [],
            "errors": [],
            "executed": True,
        },
        output_root=tmp_path / "mapping-cache",
        patterns_path="mvpa/patterns.tsv",
        qc_path="mvpa/qc.tsv",
        provenance_path="mvpa/provenance.json",
    )

    assert dataclass_plan["artifacts"][0]["row_count"] == 1
    assert mapping_plan["artifacts"][1]["row_count"] == 1


def test_overwrite_policy_rejects_then_replaces_existing_outputs(tmp_path: Path) -> None:
    output_root = tmp_path / "cache"
    kwargs = {
        "output_root": output_root,
        "patterns_path": "mvpa/patterns.tsv",
        "qc_path": "mvpa/qc.tsv",
        "provenance_path": "mvpa/provenance.json",
    }

    write_mvpa_pattern_extraction_outputs(_ResultLike(), **kwargs)

    with pytest.raises(FileExistsError):
        write_mvpa_pattern_extraction_outputs(_ResultLike(), **kwargs)

    (output_root / "mvpa/patterns.tsv").write_text("old rows\n", encoding="utf-8")
    write_mvpa_pattern_extraction_outputs(_ResultLike(), **kwargs, overwrite=True)

    assert "old rows" not in (output_root / "mvpa/patterns.tsv").read_text(encoding="utf-8")
    assert _read_tsv(output_root / "mvpa/patterns.tsv")[0]["pattern_id"] == "pat-sub01-face-v1"
    provenance = json.loads((output_root / "mvpa/provenance.json").read_text(encoding="utf-8"))
    assert provenance["overwrite_policy"] == "replace_existing"


def test_rejects_unsafe_paths_duplicate_targets_directories_and_append(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="parent traversal"):
        plan_mvpa_pattern_extraction_outputs(
            _ResultLike(),
            output_root=tmp_path / "cache",
            patterns_path="../patterns.tsv",
            qc_path="mvpa/qc.tsv",
            provenance_path="mvpa/provenance.json",
        )

    with pytest.raises(ValueError, match="outside output_root"):
        plan_mvpa_pattern_extraction_outputs(
            _ResultLike(),
            output_root=tmp_path / "cache",
            patterns_path=tmp_path / "outside.tsv",
            qc_path="mvpa/qc.tsv",
            provenance_path="mvpa/provenance.json",
        )

    with pytest.raises(ValueError, match="must be distinct"):
        plan_mvpa_pattern_extraction_outputs(
            _ResultLike(),
            output_root=tmp_path / "cache",
            patterns_path="mvpa/output.tsv",
            qc_path="mvpa/output.tsv",
            provenance_path="mvpa/provenance.json",
        )

    directory_target = tmp_path / "cache/mvpa/patterns.tsv"
    directory_target.mkdir(parents=True)
    with pytest.raises(IsADirectoryError):
        write_mvpa_pattern_extraction_outputs(
            _ResultLike(),
            output_root=tmp_path / "cache",
            patterns_path="mvpa/patterns.tsv",
            qc_path="mvpa/qc.tsv",
            provenance_path="mvpa/provenance.json",
            overwrite=True,
        )

    with pytest.raises(ValueError, match="Append mode"):
        write_mvpa_pattern_extraction_outputs(
            _ResultLike(),
            output_root=tmp_path / "append-cache",
            patterns_path="mvpa/patterns.tsv",
            qc_path="mvpa/qc.tsv",
            provenance_path="mvpa/provenance.json",
            append=True,
        )


def test_nonfinite_floats_are_rejected_before_any_write(tmp_path: Path) -> None:
    output_root = tmp_path / "cache"
    result = {
        "pattern_rows": [
            {
                "pattern_id": "bad",
                "feature_values": [math.nan],
                "feature_count": 1,
                "voxel_count": 1,
                "valid_voxel_count": 1,
                "voxel_order": "c_flat_index",
                "voxel_index_hash": "hash-bad",
                "usable": True,
            }
        ],
        "qc_rows": [],
        "provenance": {},
        "warnings": [],
        "errors": [],
        "executed": True,
    }

    with pytest.raises(ValueError, match="non-finite"):
        write_mvpa_pattern_extraction_outputs(
            result,
            output_root=output_root,
            patterns_path="mvpa/patterns.tsv",
            qc_path="mvpa/qc.tsv",
            provenance_path="mvpa/provenance.json",
        )

    assert not output_root.exists()


def test_generic_runtime_tsv_and_json_use_same_serialization_and_path_rules(tmp_path: Path) -> None:
    output_root = tmp_path / "cache"

    tsv_record = write_mvpa_runtime_tsv(
        [{"name": "row-1", "flag": True, "missing": None, "context": {"b": 2, "a": [1.0, 2.0]}}],
        columns=("name", "flag", "missing", "context"),
        output_root=output_root,
        output_path="generic/table.tsv",
    )
    json_record = write_mvpa_runtime_json(
        {"kind": "sidecar", "values": [1.0, 2.0]},
        output_root=output_root,
        output_path="generic/sidecar.json",
    )

    rows = _read_tsv(output_root / "generic/table.tsv")
    assert rows[0] == {
        "name": "row-1",
        "flag": "true",
        "missing": "",
        "context": '{"a":[1.0,2.0],"b":2}',
    }
    assert json.loads((output_root / "generic/sidecar.json").read_text(encoding="utf-8"))["values"] == [1.0, 2.0]
    assert tsv_record["artifacts"][0]["relative_path"] == "generic/table.tsv"
    assert json_record["artifacts"][0]["relative_path"] == "generic/sidecar.json"

    with pytest.raises(ValueError, match="Append mode"):
        write_mvpa_runtime_tsv(
            [],
            columns=("name",),
            output_root=output_root,
            output_path="generic/append.tsv",
            append=True,
        )


def test_materialization_outputs_use_prepared_feature_audit_contract(tmp_path: Path) -> None:
    output_root = tmp_path / "runtime"
    result = {
        "pattern_rows": [
            {
                "pattern_id": "pattern-a",
                "unit_id": "unit-a",
                "condition_id": "condition-a",
                "cross_validation_label": "fold-a",
                "subject_id": "sub-toy01",
                "pattern_source_name": "prepared-patterns",
                "roi_source_name": "prepared-rois",
                "roi_label": "SeedA",
                "feature_count": 2,
                "voxel_order": "c_flat_index",
                "voxel_index_hash": "index-a",
                "feature_space_id": "space-a",
                "roi_definition_id": "prepared-rois:SeedA",
                "feature_values": [1.0, 2.0],
                "usable": True,
                "status": "ok",
                "noise_status": "unused",
                "noise_usable": False,
                "noise_values": [],
            }
        ],
        "qc_rows": [
            {
                "pattern_id": "pattern-a",
                "data_row": 1,
                "status": "pass",
                "code": "materialized_row_validated",
                "message": "Prepared feature row validated.",
            }
        ],
        "provenance": {
            "source_reference": "root_ref:mvpa_inputs/patterns.tsv",
            "source_sha256": "a" * 64,
        },
        "warnings": [],
        "errors": [],
        "executed": True,
    }
    kwargs = {
        "output_root": output_root,
        "patterns_path": "neuro/pattern-materialization/patterns.tsv",
        "qc_path": "neuro/pattern-materialization/qc.tsv",
        "provenance_path": "neuro/pattern-materialization/provenance.json",
        "vector_metadata_path": "neuro/pattern-materialization/vector_metadata.json",
    }

    preview = plan_mvpa_pattern_materialization_outputs(result, **kwargs)
    assert not output_root.exists()
    record = write_mvpa_pattern_materialization_outputs(result, **kwargs)

    assert preview["artifact_kind"] == "mvpa_pattern_materialization_outputs"
    assert record["artifact_kind"] == "mvpa_pattern_materialization_outputs"
    header = (output_root / kwargs["patterns_path"]).read_text(encoding="utf-8").splitlines()[0]
    assert "cross_validation_label" in header.split("\t")
    assert "feature_space_id" in header.split("\t")
    assert "pe_image" not in header.split("\t")
    assert "mask_path" not in header.split("\t")
    provenance = json.loads((output_root / kwargs["provenance_path"]).read_text(encoding="utf-8"))
    assert provenance["artifact_kind"] == "mvpa_pattern_materialization_outputs"
    assert provenance["input_provenance"]["source_reference"] == (
        "root_ref:mvpa_inputs/patterns.tsv"
    )


def test_temp_file_is_removed_when_atomic_replace_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output_root = tmp_path / "cache"

    def _fail_replace(source: object, target: object) -> None:
        raise RuntimeError(f"replace failed for {source!s} -> {target!s}")

    monkeypatch.setattr(runtime_outputs.os, "replace", _fail_replace)

    with pytest.raises(RuntimeError, match="replace failed"):
        write_mvpa_pattern_extraction_outputs(
            _ResultLike(),
            output_root=output_root,
            patterns_path="mvpa/patterns.tsv",
            qc_path="mvpa/qc.tsv",
            provenance_path="mvpa/provenance.json",
        )

    assert not (output_root / "mvpa/patterns.tsv").exists()
    assert list(output_root.rglob("*.tmp")) == []


def test_forbidden_import_guard_for_runtime_outputs_module_and_tests() -> None:
    forbidden_modules = (
        "research_platform.neuro.mvpa.extraction",
        "research_platform.analysis",
        "research_platform.bids",
        "research_platform.core",
        "research_platform.viz",
        "research_platform.io",
        "research_platform.ml",
        "pandas",
        "polars",
        "scipy",
        "nilearn",
        "rsatoolbox",
        "sklearn",
        "nibabel",
        "pipelines",
        "ops",
    )

    imported_modules: list[str] = []
    for path in (Path(runtime_outputs.__file__), Path(__file__)):
        tree = ast.parse(path.read_text(encoding="utf-8"))
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
