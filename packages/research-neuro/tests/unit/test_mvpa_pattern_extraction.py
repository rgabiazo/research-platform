from __future__ import annotations

from pathlib import Path
import json
import subprocess
import sys
import textwrap

import pytest

np = pytest.importorskip("numpy")
nib = pytest.importorskip("nibabel")

from research_platform.neuro.mvpa.extraction import extract_mvpa_patterns_from_discovery_plan


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = Path(__file__).resolve().parents[4]


def test_valid_pe_and_roi_mask_extracts_deterministic_pattern(tmp_path: Path) -> None:
    pe_path = _write_image(tmp_path / "pe.nii.gz", np.arange(27, dtype=float).reshape((3, 3, 3)))
    mask_data = np.zeros((3, 3, 3), dtype=np.uint8)
    mask_data.ravel(order="C")[[0, 4, 13]] = 1
    mask_path = _write_image(tmp_path / "mask.nii.gz", mask_data)
    noise_path = _write_image(tmp_path / "noise.nii.gz", np.ones((3, 3, 3), dtype=float))

    result = extract_mvpa_patterns_from_discovery_plan(_plan(pe_path, mask_path, noise_path=noise_path))
    repeated = extract_mvpa_patterns_from_discovery_plan(_plan(pe_path, mask_path, noise_path=noise_path))

    assert len(result.pattern_rows) == 1
    row = result.pattern_rows[0]
    assert row.usable is True
    assert row.voxel_count == 3
    assert row.valid_voxel_count == 3
    assert row.feature_count == 3
    assert row.voxel_order == "c_flat_index"
    assert row.feature_values == (0.0, 4.0, 13.0)
    assert row.voxel_index_hash == repeated.pattern_rows[0].voxel_index_hash
    assert row.noise_loaded is False
    assert row.noise_status == "not_requested"
    assert row.noise_usable is False
    assert row.noise_feature_count == 0
    assert row.noise_values == ()
    assert result.qc_rows[0].status == "ok"
    assert result.qc_rows[0].noise_status == "not_requested"
    assert result.provenance["load_noise"] is False
    assert result.provenance["noise_status_counts"] == {"not_requested": 1}


def test_load_noise_true_extracts_noise_in_same_c_flat_voxel_order(tmp_path: Path) -> None:
    pe_path = _write_image(tmp_path / "pe.nii.gz", np.arange(27, dtype=float).reshape((3, 3, 3)))
    mask_data = np.zeros((3, 3, 3), dtype=np.uint8)
    mask_data.ravel(order="C")[[1, 3, 14]] = 1
    mask_path = _write_image(tmp_path / "mask.nii.gz", mask_data)
    noise_data = (np.arange(27, dtype=float).reshape((3, 3, 3)) + 100.0) / 10.0
    noise_path = _write_image(tmp_path / "sigmasquareds.nii.gz", noise_data)

    result = extract_mvpa_patterns_from_discovery_plan(
        _plan(pe_path, mask_path, noise_path=noise_path),
        load_noise=True,
    )

    assert len(result.pattern_rows) == 1
    row = result.pattern_rows[0]
    assert row.usable is True
    assert row.feature_values == (1.0, 3.0, 14.0)
    assert row.noise_loaded is True
    assert row.noise_status == "ok"
    assert row.noise_usable is True
    assert row.noise_values == (10.1, 10.3, 11.4)
    assert row.noise_feature_count == row.feature_count
    assert row.noise_voxel_order == "c_flat_index"
    assert row.noise_voxel_index_hash == row.voxel_index_hash
    assert row.noise_min == 10.1
    assert row.noise_max == 11.4
    assert row.noise_mean == pytest.approx((10.1 + 10.3 + 11.4) / 3.0)
    assert row.noise_nonfinite_count == 0
    assert row.noise_nonpositive_count == 0

    qc = result.qc_rows[0]
    assert qc.status == "ok"
    assert qc.usable is True
    assert qc.noise_status == "ok"
    assert qc.noise_usable is True
    assert qc.noise_feature_count == row.feature_count
    assert qc.noise_voxel_index_hash == row.voxel_index_hash
    assert result.provenance["noise_requested_count"] == 1
    assert result.provenance["noise_loaded_count"] == 1
    assert result.provenance["noise_usable_count"] == 1
    assert result.provenance["noise_status_counts"] == {"ok": 1}


def test_empty_roi_mask_emits_qc_without_pattern(tmp_path: Path) -> None:
    pe_path = _write_image(tmp_path / "pe.nii.gz", np.arange(27, dtype=float).reshape((3, 3, 3)))
    mask_path = _write_image(tmp_path / "empty_mask.nii.gz", np.zeros((3, 3, 3), dtype=np.uint8))

    result = extract_mvpa_patterns_from_discovery_plan(_plan(pe_path, mask_path))

    assert result.pattern_rows == ()
    qc = result.qc_rows[0]
    assert qc.usable is False
    assert qc.status == "empty_roi_mask"
    assert qc.mask_status == "empty"
    assert qc.voxel_count == 0


def test_geometry_mismatch_emits_qc_without_pattern(tmp_path: Path) -> None:
    pe_path = _write_image(tmp_path / "pe.nii.gz", np.ones((3, 3, 3), dtype=float))
    mask_data = np.ones((2, 2, 2), dtype=np.uint8)
    mask_path = _write_image(tmp_path / "mask.nii.gz", mask_data)

    result = extract_mvpa_patterns_from_discovery_plan(_plan(pe_path, mask_path))

    assert result.pattern_rows == ()
    qc = result.qc_rows[0]
    assert qc.usable is False
    assert qc.status == "geometry_mismatch"
    assert qc.geometry_status == "mismatch"


def test_missing_pe_image_emits_qc_without_pattern(tmp_path: Path) -> None:
    pe_path = tmp_path / "missing_pe.nii.gz"
    mask_path = _write_image(tmp_path / "mask.nii.gz", np.ones((3, 3, 3), dtype=np.uint8))

    result = extract_mvpa_patterns_from_discovery_plan(_plan(pe_path, mask_path))

    assert result.pattern_rows == ()
    qc = result.qc_rows[0]
    assert qc.usable is False
    assert qc.status == "missing_pe_image"
    assert qc.pe_exists is False


def test_missing_roi_mask_emits_qc_without_pattern(tmp_path: Path) -> None:
    pe_path = _write_image(tmp_path / "pe.nii.gz", np.ones((3, 3, 3), dtype=float))
    mask_path = tmp_path / "missing_mask.nii.gz"

    result = extract_mvpa_patterns_from_discovery_plan(_plan(pe_path, mask_path))

    assert result.pattern_rows == ()
    qc = result.qc_rows[0]
    assert qc.usable is False
    assert qc.status == "missing_roi_mask"
    assert qc.mask_exists is False


def test_missing_noise_with_load_noise_false_keeps_pattern_usable(tmp_path: Path) -> None:
    pe_path = _write_image(tmp_path / "pe.nii.gz", np.arange(8, dtype=float).reshape((2, 2, 2)))
    mask_path = _write_image(tmp_path / "mask.nii.gz", np.ones((2, 2, 2), dtype=np.uint8))
    noise_path = tmp_path / "missing_noise.nii.gz"

    result = extract_mvpa_patterns_from_discovery_plan(_plan(pe_path, mask_path, noise_path=noise_path), load_noise=False)

    assert len(result.pattern_rows) == 1
    qc = result.qc_rows[0]
    assert qc.usable is True
    assert qc.status == "ok"
    assert qc.noise_image == noise_path.as_posix()
    assert qc.noise_exists is False
    assert qc.noise_status == "not_requested"
    assert result.pattern_rows[0].noise_status == "not_requested"


def test_missing_noise_with_load_noise_true_keeps_pe_pattern_usable(tmp_path: Path) -> None:
    pe_path = _write_image(tmp_path / "pe.nii.gz", np.arange(8, dtype=float).reshape((2, 2, 2)))
    mask_path = _write_image(tmp_path / "mask.nii.gz", np.ones((2, 2, 2), dtype=np.uint8))
    noise_path = tmp_path / "missing_noise.nii.gz"

    result = extract_mvpa_patterns_from_discovery_plan(_plan(pe_path, mask_path, noise_path=noise_path), load_noise=True)

    assert len(result.pattern_rows) == 1
    row = result.pattern_rows[0]
    assert row.usable is True
    assert row.feature_count == 8
    assert row.noise_loaded is False
    assert row.noise_status == "missing_noise_image"
    assert row.noise_usable is False
    assert row.noise_feature_count == 0
    assert row.noise_values == ()
    qc = result.qc_rows[0]
    assert qc.status == "ok"
    assert qc.usable is True
    assert qc.noise_exists is False
    assert qc.noise_status == "missing_noise_image"
    assert qc.noise_usable is False
    assert result.provenance["noise_status_counts"] == {"missing_noise_image": 1}


def test_noise_geometry_mismatch_keeps_pe_pattern_usable(tmp_path: Path) -> None:
    pe_path = _write_image(tmp_path / "pe.nii.gz", np.arange(27, dtype=float).reshape((3, 3, 3)))
    mask_path = _write_image(tmp_path / "mask.nii.gz", np.ones((3, 3, 3), dtype=np.uint8))
    noise_path = _write_image(tmp_path / "noise_wrong_shape.nii.gz", np.ones((2, 2, 2), dtype=float))

    result = extract_mvpa_patterns_from_discovery_plan(_plan(pe_path, mask_path, noise_path=noise_path), load_noise=True)

    assert len(result.pattern_rows) == 1
    row = result.pattern_rows[0]
    assert row.usable is True
    assert row.noise_loaded is True
    assert row.noise_status == "noise_geometry_mismatch"
    assert row.noise_usable is False
    assert row.noise_feature_count == 0
    assert row.noise_voxel_index_hash is None
    assert row.noise_values == ()
    qc = result.qc_rows[0]
    assert qc.status == "ok"
    assert qc.noise_status == "noise_geometry_mismatch"
    assert qc.noise_loaded is True


def test_nonfinite_noise_values_are_flagged_without_exposing_values(tmp_path: Path) -> None:
    pe_path = _write_image(tmp_path / "pe.nii.gz", np.arange(8, dtype=float).reshape((2, 2, 2)))
    mask_data = np.zeros((2, 2, 2), dtype=np.uint8)
    mask_data.ravel(order="C")[[0, 1, 7]] = 1
    mask_path = _write_image(tmp_path / "mask.nii.gz", mask_data)
    noise_data = np.ones((2, 2, 2), dtype=float)
    noise_data.ravel(order="C")[1] = np.nan
    noise_path = _write_image(tmp_path / "noise_nan.nii.gz", noise_data)

    result = extract_mvpa_patterns_from_discovery_plan(_plan(pe_path, mask_path, noise_path=noise_path), load_noise=True)

    row = result.pattern_rows[0]
    assert row.usable is True
    assert row.noise_status == "nonfinite_noise_values"
    assert row.noise_usable is False
    assert row.noise_feature_count == row.feature_count
    assert row.noise_voxel_index_hash == row.voxel_index_hash
    assert row.noise_nonfinite_count == 1
    assert row.noise_nonpositive_count == 0
    assert row.noise_values == ()
    assert row.noise_min == 1.0
    assert row.noise_max == 1.0
    qc = result.qc_rows[0]
    assert qc.noise_status == "nonfinite_noise_values"
    assert qc.noise_nonfinite_count == 1


def test_zero_or_negative_noise_values_are_flagged_without_exposing_values(tmp_path: Path) -> None:
    pe_path = _write_image(tmp_path / "pe.nii.gz", np.arange(8, dtype=float).reshape((2, 2, 2)))
    mask_data = np.zeros((2, 2, 2), dtype=np.uint8)
    mask_data.ravel(order="C")[[0, 1, 7]] = 1
    mask_path = _write_image(tmp_path / "mask.nii.gz", mask_data)
    noise_data = np.ones((2, 2, 2), dtype=float)
    noise_data.ravel(order="C")[0] = 0.0
    noise_data.ravel(order="C")[7] = -2.0
    noise_path = _write_image(tmp_path / "noise_nonpositive.nii.gz", noise_data)

    result = extract_mvpa_patterns_from_discovery_plan(_plan(pe_path, mask_path, noise_path=noise_path), load_noise=True)

    row = result.pattern_rows[0]
    assert row.usable is True
    assert row.noise_status == "nonpositive_noise_values"
    assert row.noise_usable is False
    assert row.noise_feature_count == row.feature_count
    assert row.noise_voxel_index_hash == row.voxel_index_hash
    assert row.noise_nonfinite_count == 0
    assert row.noise_nonpositive_count == 2
    assert row.noise_values == ()
    assert row.noise_min == -2.0
    assert row.noise_max == 1.0
    qc = result.qc_rows[0]
    assert qc.noise_status == "nonpositive_noise_values"
    assert qc.noise_nonpositive_count == 2


def test_drop_features_noise_policy_preserves_nonpositive_noise_values(tmp_path: Path) -> None:
    pe_path = _write_image(tmp_path / "pe.nii.gz", np.arange(8, dtype=float).reshape((2, 2, 2)))
    mask_data = np.zeros((2, 2, 2), dtype=np.uint8)
    mask_data.ravel(order="C")[[0, 1, 7]] = 1
    mask_path = _write_image(tmp_path / "mask.nii.gz", mask_data)
    noise_data = np.ones((2, 2, 2), dtype=float)
    noise_data.ravel(order="C")[0] = 0.0
    noise_data.ravel(order="C")[7] = -2.0
    noise_path = _write_image(tmp_path / "noise_nonpositive.nii.gz", noise_data)

    result = extract_mvpa_patterns_from_discovery_plan(
        _plan(pe_path, mask_path, noise_path=noise_path, noise_nonpositive_policy="drop_features"),
        load_noise=True,
    )

    row = result.pattern_rows[0]
    assert row.noise_status == "nonpositive_noise_values"
    assert row.noise_usable is False
    assert row.noise_nonpositive_count == 2
    assert row.noise_values == (0.0, 1.0, -2.0)
    assert result.provenance["noise_nonpositive_policy"] == "drop_features"


def test_excluded_unit_emits_qc_without_loading_images(tmp_path: Path) -> None:
    pe_path = tmp_path / "would_fail_if_loaded_pe.nii.gz"
    mask_path = tmp_path / "would_fail_if_loaded_mask.nii.gz"
    noise_path = tmp_path / "would_fail_if_loaded_noise.nii.gz"
    plan = _plan(pe_path, mask_path, noise_path=noise_path)
    plan["condition_pe_rows"][0]["excluded"] = True
    plan["condition_pe_rows"][0]["exclusion_reason"] = "Configured motion exclusion"

    result = extract_mvpa_patterns_from_discovery_plan(plan, load_noise=True)

    assert result.pattern_rows == ()
    qc = result.qc_rows[0]
    assert qc.status == "excluded"
    assert qc.usable is False
    assert qc.excluded is True
    assert qc.exclusion_reason == "Configured motion exclusion"
    assert qc.pe_exists is None
    assert qc.mask_exists is None
    assert qc.noise_exists is None
    assert qc.noise_loaded is False
    assert qc.noise_status == "not_checked"


def test_configured_run_exclusion_records_policy_and_skips_before_extraction(tmp_path: Path) -> None:
    plan = _plan(
        tmp_path / "would_fail_if_loaded_pe.nii.gz",
        tmp_path / "would_fail_if_loaded_mask.nii.gz",
        noise_path=tmp_path / "would_fail_if_loaded_noise.nii.gz",
        subject_id="participant-a",
        session_id="session-a",
        run_id="run-a",
        condition_id="condition-a",
        task_id="task-alpha",
        roi_label="roi-alpha",
    )
    plan["exclusions"] = [
        {
            "id": "exclude-run-a",
            "subject_id": "participant-a",
            "session_id": "session-a",
            "run_id": "run-a",
            "reason": "Synthetic configured exclusion",
            "source_config_field": "mvpa_set.run_exclusions",
        }
    ]

    result = extract_mvpa_patterns_from_discovery_plan(plan, load_noise=True)

    assert result.pattern_rows == ()
    qc = result.qc_rows[0]
    assert qc.status == "excluded"
    assert qc.exclusion_id == "exclude-run-a"
    assert qc.exclusion_source_field == "mvpa_set.run_exclusions"
    assert qc.skipped_stage == "before_extraction"
    assert qc.pe_exists is None
    assert qc.mask_exists is None
    assert qc.noise_exists is None
    assert result.provenance["run_exclusion_policy_rows"] == (
        {
            "id": "exclude-run-a",
            "subject_id": "participant-a",
            "session_id": "session-a",
            "run_id": "run-a",
            "reason": "Synthetic configured exclusion",
            "source_config_field": "mvpa_set.run_exclusions",
            "status": "configured",
        },
    )
    assert result.provenance["excluded_run_rows"] == (
        {
            "subject_id": "participant-a",
            "session_id": "session-a",
            "run_id": "run-a",
            "reason": "Synthetic configured exclusion",
            "exclusion_id": "exclude-run-a",
            "source_config_field": "mvpa_set.run_exclusions",
            "skipped_stage": "before_extraction",
            "status": "excluded",
        },
    )


def test_within_roi_mean_centering_preserves_noise_and_grouping_values(tmp_path: Path) -> None:
    pe_path = _write_image(tmp_path / "pe.nii.gz", np.arange(8, dtype=float).reshape((2, 2, 2)))
    mask_data = np.zeros((2, 2, 2), dtype=np.uint8)
    mask_data.ravel(order="C")[[1, 3, 7]] = 1
    mask_path = _write_image(tmp_path / "mask.nii.gz", mask_data)
    noise_path = _write_image(tmp_path / "noise.nii.gz", np.arange(8, dtype=float).reshape((2, 2, 2)) + 10.0)
    plan = _plan(
        pe_path,
        mask_path,
        noise_path=noise_path,
        subject_id="participant-a",
        session_id="session-a",
        run_id="run-a",
        condition_id="condition-a",
        task_id="task-alpha",
        roi_label="roi-alpha",
    )
    plan["mean_centering"] = {"enabled": True, "scope": "roi"}
    plan["grouping_columns"] = ["cohort_label", "task_id"]
    plan["condition_pe_rows"][0]["cohort_label"] = "cohort-alpha"  # type: ignore[index]
    plan["roi_source_rows"][0]["cohort_label"] = "cohort-alpha"  # type: ignore[index]

    result = extract_mvpa_patterns_from_discovery_plan(plan, load_noise=True)

    row = result.pattern_rows[0]
    assert row.mean_centering_applied is True
    assert row.mean_centering_scope == "roi"
    assert row.feature_values == pytest.approx((-8.0 / 3.0, -2.0 / 3.0, 10.0 / 3.0))
    assert row.noise_values == (11.0, 13.0, 17.0)
    assert row.grouping_values == {"cohort_label": "cohort-alpha", "task_id": "task-alpha"}
    assert row.to_dict()["cohort_label"] == "cohort-alpha"
    assert row.to_dict()["task_id"] == "task-alpha"


def test_event_threshold_metadata_remains_not_evaluated(tmp_path: Path) -> None:
    pe_path = _write_image(tmp_path / "pe.nii.gz", np.ones((2, 2, 2), dtype=float))
    mask_path = _write_image(tmp_path / "mask.nii.gz", np.ones((2, 2, 2), dtype=np.uint8))

    result = extract_mvpa_patterns_from_discovery_plan(_plan(pe_path, mask_path))

    assert result.qc_rows[0].event_threshold_status == "not_evaluated"
    assert result.provenance["event_threshold_rows"][0]["status"] == "not_evaluated"


def test_condition_event_count_is_preserved_on_pattern_rows(tmp_path: Path) -> None:
    pe_path = _write_image(tmp_path / "pe.nii.gz", np.ones((2, 2, 2), dtype=float))
    mask_path = _write_image(tmp_path / "mask.nii.gz", np.ones((2, 2, 2), dtype=np.uint8))

    result = extract_mvpa_patterns_from_discovery_plan(_plan(pe_path, mask_path, event_count=3))

    assert result.pattern_rows[0].event_count == 3
    assert result.pattern_rows[0].to_dict()["event_count"] == 3


def test_output_is_json_safe(tmp_path: Path) -> None:
    pe_path = _write_image(tmp_path / "pe.nii.gz", np.ones((2, 2, 2), dtype=float))
    mask_path = _write_image(tmp_path / "mask.nii.gz", np.ones((2, 2, 2), dtype=np.uint8))

    result = extract_mvpa_patterns_from_discovery_plan(_plan(pe_path, mask_path))

    encoded = json.dumps(result.to_dict(), sort_keys=True, allow_nan=False)
    assert '"executed": true' in encoded
    assert '"pattern_rows"' in encoded
    assert '"qc_rows"' in encoded


def test_output_with_invalid_noise_is_json_safe_without_nan(tmp_path: Path) -> None:
    pe_path = _write_image(tmp_path / "pe.nii.gz", np.ones((2, 2, 2), dtype=float))
    mask_path = _write_image(tmp_path / "mask.nii.gz", np.ones((2, 2, 2), dtype=np.uint8))
    noise_data = np.ones((2, 2, 2), dtype=float)
    noise_data[0, 0, 0] = np.nan
    noise_path = _write_image(tmp_path / "noise_nan.nii.gz", noise_data)

    result = extract_mvpa_patterns_from_discovery_plan(_plan(pe_path, mask_path, noise_path=noise_path), load_noise=True)

    encoded = json.dumps(result.to_dict(), sort_keys=True, allow_nan=False)
    assert '"noise_status": "nonfinite_noise_values"' in encoded
    assert '"noise_values": []' in encoded


def test_extraction_does_not_write_outputs(tmp_path: Path) -> None:
    pe_path = _write_image(tmp_path / "pe.nii.gz", np.arange(8, dtype=float).reshape((2, 2, 2)))
    mask_path = _write_image(tmp_path / "mask.nii.gz", np.ones((2, 2, 2), dtype=np.uint8))
    noise_path = _write_image(tmp_path / "noise.nii.gz", np.ones((2, 2, 2), dtype=float))
    before = sorted(path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*") if path.is_file())

    result = extract_mvpa_patterns_from_discovery_plan(_plan(pe_path, mask_path, noise_path=noise_path), load_noise=True)

    after = sorted(path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*") if path.is_file())
    assert len(result.pattern_rows) == 1
    assert after == before


def test_extraction_imports_use_no_forbidden_dependencies() -> None:
    script = textwrap.dedent(
        """
        import builtins
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path("packages/research-neuro/src").resolve()))
        forbidden = {
            "research_platform.analysis",
            "research_platform.bids",
            "research_platform.core",
            "research_platform.viz",
            "research_platform.ml",
            "rsatoolbox",
            "nilearn",
            "pandas",
            "polars",
            "scipy",
            "sklearn",
            "pipelines",
            "ops",
        }
        real_import = builtins.__import__

        def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name in forbidden or any(name.startswith(prefix + ".") for prefix in forbidden):
                raise RuntimeError(f"forbidden import: {name}")
            return real_import(name, globals, locals, fromlist, level)

        builtins.__import__ = guarded_import
        import research_platform.neuro.mvpa.extraction  # noqa: F401
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=WORKSPACE_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def _write_image(path: Path, data: object, *, affine: object | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = nib.Nifti1Image(np.asarray(data), np.eye(4) if affine is None else affine)
    nib.save(image, path)
    return path


def _plan(
    pe_path: Path,
    mask_path: Path,
    *,
    noise_path: Path | None = None,
    subject_id: str = "001",
    session_id: str = "01",
    run_id: str = "01",
    condition_id: str = "faces",
    task_id: str = "memory",
    roi_label: str = "SeedA",
    event_count: int | None = None,
    noise_nonpositive_policy: str | None = None,
) -> dict[str, object]:
    distance_row = {"metric": "crossnobis", "engine": "native_reference", "cv_unit": "run"}
    if noise_nonpositive_policy is not None:
        distance_row["noise_nonpositive_policy"] = noise_nonpositive_policy
    return {
        "mvpa_set": "memory_mvpa",
        "status": "valid",
        "distances": [distance_row],
        "condition_pe_rows": [
            {
                "subject_id": subject_id,
                "session_id": session_id,
                "run_id": run_id,
                "condition_id": condition_id,
                "pattern_source_name": "first_level_pe",
                "source_name": "first_level_pe",
                "pe_image": pe_path.as_posix(),
                "noise_image": noise_path.as_posix() if noise_path is not None else None,
                "event_count": event_count,
                "status": "ok",
            }
        ],
        "roi_source_rows": [
            {
                "subject_id": subject_id,
                "session_id": session_id,
                "run_id": run_id,
                "task_id": task_id,
                "direction": "AP",
                "model": "modelA",
                "roi_source_name": "explicit_rois",
                "source_name": "explicit_rois",
                "roi_label": roi_label,
                "mask_path": mask_path.as_posix(),
                "status": "ok",
            }
        ],
        "input_checks": [],
        "provenance_rows": [],
        "event_threshold_rows": [
            {
                "threshold": "min_events_per_condition_per_run",
                "value": 1,
                "status": "not_evaluated",
                "reason": "event_counts_not_read_phase_2f",
            }
        ],
    }
