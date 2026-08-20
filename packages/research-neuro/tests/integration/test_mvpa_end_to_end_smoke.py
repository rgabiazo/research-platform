from __future__ import annotations

import ast
import csv
import json
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
nib = pytest.importorskip("nibabel")

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PACKAGE_ROOT / "src"))
sys.path.insert(0, str(WORKSPACE_ROOT / "packages/research-analysis/src"))

from research_platform.analysis.mvpa import (
    NOISE_NORMALIZATION_DIAGONAL,
    NOISE_NORMALIZATION_IDENTITY,
    compute_mvpa_distances_from_prepared_groups,
    plan_prepared_mvpa_distance_outputs,
    plan_prepared_mvpa_pattern_outputs,
    prepare_mvpa_pattern_row_groups,
    write_prepared_mvpa_distance_outputs,
    write_prepared_mvpa_pattern_outputs,
)
from research_platform.neuro.mvpa import plan_mvpa_discovery
from research_platform.neuro.mvpa.extraction import extract_mvpa_patterns_from_discovery_plan
from research_platform.neuro.mvpa.runtime_outputs import (
    plan_mvpa_pattern_extraction_outputs,
    write_mvpa_pattern_extraction_outputs,
)


SUBJECT_ID = "SYN01"
SESSION_ID = "T1"
RUN_IDS = ("A", "B")
TASK_ID = "toyMemory"
MODEL_ID = "toyModel"
DIRECTION_ID = "LR"
ROI_LABEL = "ToyROI"
CONDITIONS = (("toyAlpha", "ToyAlpha"), ("toyBeta", "ToyBeta"))
MASK_FLAT_INDICES = (1, 6)
FEATURE_VALUES = {
    "A": {"toyAlpha": (2.0, 4.0), "toyBeta": (0.0, 0.0)},
    "B": {"toyAlpha": (6.0, 8.0), "toyBeta": (0.0, 0.0)},
}
NOISE_VALUES = (2.0, 4.0)


def test_tiny_synthetic_mvpa_end_to_end_smoke(tmp_path: Path) -> None:
    feat_root = tmp_path / "feat"
    roi_root = tmp_path / "roi"
    runtime_root = tmp_path / "runtime"
    for run_id in RUN_IDS:
        _write_feat_like_tree(feat_root, run_id)
        _write_roi_mask_and_sidecar(roi_root, run_id)

    inputs_only = _relative_files(tmp_path)
    plan = plan_mvpa_discovery(
        _mvpa_config(),
        roots={"feat_root": feat_root, "roi_root": roi_root, "artifact_root": runtime_root},
    )

    assert plan.status in {"valid", "warning"}
    assert plan.errors == ()
    assert len(plan.condition_pe_rows) == len(RUN_IDS) * len(CONDITIONS)
    assert len(plan.roi_source_rows) == len(RUN_IDS)
    assert {row["status"] for row in plan.condition_pe_rows} == {"ok"}
    assert {row["status"] for row in plan.roi_source_rows} == {"ok"}
    assert {(row["condition_id"], row["matched_ev_title"]) for row in plan.condition_pe_rows} == set(CONDITIONS)
    json.dumps(plan.to_dict(), sort_keys=True, allow_nan=False)
    assert _relative_files(tmp_path) == inputs_only

    extraction = extract_mvpa_patterns_from_discovery_plan(plan, load_noise=True)

    assert len(extraction.pattern_rows) == 4
    assert all(row.usable is True for row in extraction.pattern_rows)
    assert all(qc.usable is True and qc.status == "ok" for qc in extraction.qc_rows)
    assert all(qc.noise_usable is True and qc.noise_status == "ok" for qc in extraction.qc_rows)
    extracted_by_key = {(row.run_id, row.condition_id): row for row in extraction.pattern_rows}
    for run_id in RUN_IDS:
        for condition_id, _ev_title in CONDITIONS:
            row = extracted_by_key[(run_id, condition_id)]
            assert row.voxel_order == "c_flat_index"
            assert row.feature_values == FEATURE_VALUES[run_id][condition_id]
            assert row.noise_voxel_order == "c_flat_index"
            assert row.noise_values == NOISE_VALUES

    neuro_kwargs = {
        "output_root": runtime_root / "neuro",
        "patterns_path": "pattern-extraction/patterns.tsv",
        "qc_path": "pattern-extraction/qc.tsv",
        "provenance_path": "pattern-extraction/provenance.json",
        "vector_metadata_path": "pattern-extraction/vector_metadata.json",
    }
    before_dry_run = _relative_files(tmp_path)
    neuro_dry_run = plan_mvpa_pattern_extraction_outputs(extraction, **neuro_kwargs)
    assert neuro_dry_run["will_write"] is False
    assert neuro_dry_run["output_written"] is False
    assert _relative_files(tmp_path) == before_dry_run
    json.dumps(neuro_dry_run, sort_keys=True, allow_nan=False)

    before_write = _relative_files(tmp_path)
    neuro_record = write_mvpa_pattern_extraction_outputs(extraction, **neuro_kwargs)
    _assert_new_runtime_files(
        tmp_path,
        before_write,
        {
            "runtime/neuro/pattern-extraction/patterns.tsv",
            "runtime/neuro/pattern-extraction/qc.tsv",
            "runtime/neuro/pattern-extraction/provenance.json",
            "runtime/neuro/pattern-extraction/vector_metadata.json",
        },
    )
    _assert_artifacts_under(tmp_path, neuro_record)
    _assert_strict_json_file(tmp_path / "runtime/neuro/pattern-extraction/provenance.json")
    _assert_strict_json_file(tmp_path / "runtime/neuro/pattern-extraction/vector_metadata.json")

    preparation = prepare_mvpa_pattern_row_groups(
        [row.to_dict() for row in extraction.pattern_rows],
        cv_unit="run",
    )

    assert len(preparation.groups) == 1
    group = preparation.groups[0]
    assert len(group.rows) == 4
    assert group.cv_labels == RUN_IDS
    assert group.condition_ids == tuple(condition_id for condition_id, _ev_title in CONDITIONS)
    assert preparation.errors == ()

    identity_distances = compute_mvpa_distances_from_prepared_groups(
        preparation.groups,
        noise_normalization_method=NOISE_NORMALIZATION_IDENTITY,
    )
    diagonal_distances = compute_mvpa_distances_from_prepared_groups(
        preparation.groups,
        noise_normalization_method=NOISE_NORMALIZATION_DIAGONAL,
    )

    assert identity_distances.errors == ()
    assert diagonal_distances.errors == ()
    assert len(identity_distances.distances) == 1
    assert len(diagonal_distances.distances) == 1
    identity = identity_distances.distances[0]
    diagonal = diagonal_distances.distances[0]
    assert (identity.condition_id_a, identity.condition_id_b) == ("toyAlpha", "toyBeta")
    assert identity.engine_name == "native_reference"
    assert identity.normalization_method == NOISE_NORMALIZATION_IDENTITY
    assert identity.distance == pytest.approx(44.0)
    assert diagonal.engine_name == "native_reference"
    assert diagonal.normalization_method == NOISE_NORMALIZATION_DIAGONAL
    assert diagonal.distance == pytest.approx(14.0)

    analysis_pattern_kwargs = {
        "output_root": runtime_root / "analysis",
        "rows_path": "prepared-patterns/rows.tsv",
        "qc_path": "prepared-patterns/qc.tsv",
        "provenance_path": "prepared-patterns/provenance.json",
    }
    before_dry_run = _relative_files(tmp_path)
    pattern_dry_run = plan_prepared_mvpa_pattern_outputs(preparation, **analysis_pattern_kwargs)
    assert pattern_dry_run["will_write"] is False
    assert pattern_dry_run["output_written"] is False
    assert _relative_files(tmp_path) == before_dry_run
    json.dumps(pattern_dry_run, sort_keys=True, allow_nan=False)

    before_write = _relative_files(tmp_path)
    pattern_record = write_prepared_mvpa_pattern_outputs(preparation, **analysis_pattern_kwargs)
    _assert_new_runtime_files(
        tmp_path,
        before_write,
        {
            "runtime/analysis/prepared-patterns/rows.tsv",
            "runtime/analysis/prepared-patterns/qc.tsv",
            "runtime/analysis/prepared-patterns/provenance.json",
        },
    )
    _assert_artifacts_under(tmp_path, pattern_record)
    _assert_strict_json_file(tmp_path / "runtime/analysis/prepared-patterns/provenance.json")

    analysis_distance_kwargs = {
        "output_root": runtime_root / "analysis",
        "distances_path": "prepared-distances/distances.tsv",
        "qc_path": "prepared-distances/qc.tsv",
        "provenance_path": "prepared-distances/provenance.json",
    }
    before_dry_run = _relative_files(tmp_path)
    distance_dry_run = plan_prepared_mvpa_distance_outputs(diagonal_distances, **analysis_distance_kwargs)
    assert distance_dry_run["will_write"] is False
    assert distance_dry_run["output_written"] is False
    assert _relative_files(tmp_path) == before_dry_run
    json.dumps(distance_dry_run, sort_keys=True, allow_nan=False)

    before_write = _relative_files(tmp_path)
    distance_record = write_prepared_mvpa_distance_outputs(diagonal_distances, **analysis_distance_kwargs)
    _assert_new_runtime_files(
        tmp_path,
        before_write,
        {
            "runtime/analysis/prepared-distances/distances.tsv",
            "runtime/analysis/prepared-distances/qc.tsv",
            "runtime/analysis/prepared-distances/provenance.json",
        },
    )
    _assert_artifacts_under(tmp_path, distance_record)
    _assert_strict_json_file(tmp_path / "runtime/analysis/prepared-distances/provenance.json")

    assert _read_tsv(tmp_path / "runtime/analysis/prepared-distances/distances.tsv")[0]["distance"] == "14.0"


def test_mvpa_end_to_end_smoke_import_boundary() -> None:
    forbidden_modules = (
        "research_platform.core",
        "research_platform.bids",
        "research_platform.viz",
        "research_platform.io",
        "pipelines",
        "ops",
        "project",
        "pandas",
        "polars",
        "scipy",
        "nilearn",
        "sklearn",
        "mvpa2",
        "rsatoolbox",
    )
    imported_modules: list[str] = []
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
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


def _mvpa_config() -> dict[str, object]:
    return {
        "mvpa_set": {
            "name": "synthetic_mvpa_smoke",
            "subjects": [SUBJECT_ID],
            "sessions": [SESSION_ID],
            "runs": list(RUN_IDS),
            "entities": {"task": TASK_ID, "direction": DIRECTION_ID, "model": MODEL_ID},
            "conditions": [
                {"id": condition_id, "fsl_ev_title": ev_title} for condition_id, ev_title in CONDITIONS
            ],
            "pattern_sources": [
                {
                    "name": "first_level_pe",
                    "backend": "fsl_feat_pe",
                    "root_ref": "feat_root",
                    "feat_dir_template": (
                        "{subject_dir}/{session_dir}/func/"
                        "{subject}_{session_dir}_task-{task_id}_run-{run_id}_model-{model}.feat"
                    ),
                    "design_file": "design.fsf",
                    "pe_image_template": "stats/pe{pe_number}.nii.gz",
                    "noise_image_template": "stats/sigmasquareds.nii.gz",
                }
            ],
            "roi_sources": [
                {
                    "name": "explicit_rois",
                    "source": "explicit_masks",
                    "root_ref": "roi_root",
                    "roi_labels": [ROI_LABEL],
                    "mask_template": (
                        "manual_masks/{subject_dir}/{session_dir}/"
                        "{subject}_{session_dir}_task-{task_id}_run-{run_id}_label-{roi_label}_mask.nii.gz"
                    ),
                }
            ],
            "distance": {
                "metrics": ["crossnobis"],
                "engine": "native_reference",
                "cross_validation": {"unit": "run"},
                "noise_normalization": {"method": "diagonal", "variance_source": "sigmasquareds"},
            },
            "outputs": {
                "runtime_root": {"root_ref": "artifact_root", "path": ".research-platform/mvpa/{mvpa_set}"}
            },
            "missing_input_policy": "warn",
        }
    }


def _write_feat_like_tree(feat_root: Path, run_id: str) -> None:
    feat_dir = _feat_dir(feat_root, run_id)
    stats_dir = feat_dir / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)
    (feat_dir / "design.fsf").write_text(
        "".join(
            f'set fmri(evtitle{ev_index}) "{ev_title}"\n'
            for ev_index, (_condition_id, ev_title) in enumerate(CONDITIONS, start=1)
        ),
        encoding="utf-8",
    )
    for ev_index, (condition_id, _ev_title) in enumerate(CONDITIONS, start=1):
        _write_selected_voxel_image(
            stats_dir / f"pe{ev_index}.nii.gz",
            FEATURE_VALUES[run_id][condition_id],
        )
    _write_selected_voxel_image(stats_dir / "sigmasquareds.nii.gz", NOISE_VALUES)


def _write_roi_mask_and_sidecar(roi_root: Path, run_id: str) -> None:
    mask_path = _roi_mask_path(roi_root, run_id)
    mask_data = np.zeros((2, 2, 2), dtype=np.uint8)
    flat = mask_data.ravel(order="C")
    for index in MASK_FLAT_INDICES:
        flat[index] = 1
    _write_image(mask_path, mask_data)
    sidecar_path = _sidecar_path(mask_path)
    sidecar_path.write_text(
        json.dumps(
            {
                "roi_label": ROI_LABEL,
                "source": "synthetic",
                "subject_id": SUBJECT_ID,
                "session_id": SESSION_ID,
                "run_id": run_id,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_selected_voxel_image(path: Path, selected_values: tuple[float, ...]) -> Path:
    data = np.zeros((2, 2, 2), dtype=float)
    flat = data.ravel(order="C")
    for index, value in zip(MASK_FLAT_INDICES, selected_values, strict=True):
        flat[index] = value
    return _write_image(path, data)


def _write_image(path: Path, data: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = nib.Nifti1Image(np.asarray(data), np.eye(4))
    nib.save(image, path)
    return path


def _feat_dir(feat_root: Path, run_id: str) -> Path:
    return (
        feat_root
        / f"sub-{SUBJECT_ID}"
        / f"ses-{SESSION_ID}"
        / "func"
        / f"sub-{SUBJECT_ID}_ses-{SESSION_ID}_task-{TASK_ID}_run-{run_id}_model-{MODEL_ID}.feat"
    )


def _roi_mask_path(roi_root: Path, run_id: str) -> Path:
    return (
        roi_root
        / "manual_masks"
        / f"sub-{SUBJECT_ID}"
        / f"ses-{SESSION_ID}"
        / f"sub-{SUBJECT_ID}_ses-{SESSION_ID}_task-{TASK_ID}_run-{run_id}_label-{ROI_LABEL}_mask.nii.gz"
    )


def _sidecar_path(mask_path: Path) -> Path:
    if mask_path.name.endswith(".nii.gz"):
        return mask_path.with_name(f"{mask_path.name[:-7]}.json")
    return mask_path.with_suffix(".json")


def _relative_files(root: Path) -> set[str]:
    return {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}


def _assert_new_runtime_files(tmp_path: Path, before: set[str], expected: set[str]) -> None:
    new_files = _relative_files(tmp_path) - before
    assert new_files == expected
    assert all(Path(relative).suffix in {".json", ".tsv"} for relative in new_files)
    assert all((tmp_path / relative).resolve().is_relative_to(tmp_path.resolve()) for relative in new_files)


def _assert_artifacts_under(tmp_path: Path, record: dict[str, object]) -> None:
    artifacts = record.get("artifacts")
    assert isinstance(artifacts, list)
    for artifact in artifacts:
        assert isinstance(artifact, dict)
        artifact_path = Path(str(artifact["path"])).resolve()
        assert artifact_path.is_relative_to(tmp_path.resolve())


def _assert_strict_json_file(path: Path) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    json.dumps(payload, sort_keys=True, allow_nan=False)


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))
