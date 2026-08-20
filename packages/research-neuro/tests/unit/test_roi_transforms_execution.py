from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import ast
import json
import math
import sys
import tempfile
import unittest
from unittest import mock

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PACKAGE_ROOT / "src"))

from research_platform.neuro.roi_transforms import (
    execute_mni_to_t1w_roi_transform_plan,
    plan_mni_to_t1w_roi_transforms,
    select_executable_mni_to_t1w_roi_transform_jobs,
    validate_mni_to_t1w_roi_transform_execution_plan,
)


PARTICIPANT_ID = "participant-a"
SESSION_ID = "session-a"
TASK_ID = "task-alpha"
RUN_ID = "run-a"
MODEL_ID = "model-alpha"
CONTRAST_ID = "contrast-alpha"
ROI_LABEL = "roi-alpha"


def _write(path: Path, text: str = "placeholder\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _chmod_executable(path: Path) -> None:
    path.chmod(path.stat().st_mode | 0o111)


def _file_snapshot(root: Path) -> tuple[str, ...]:
    return tuple(sorted(path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()))


def _identity_affine() -> tuple[tuple[float, ...], ...]:
    return (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )


def _zeros(shape: tuple[int, int, int]) -> list[list[list[int]]]:
    return [[[0 for _ in range(shape[2])] for _ in range(shape[1])] for _ in range(shape[0])]


def _ones(shape: tuple[int, int, int]) -> list[list[list[int]]]:
    return [[[1 for _ in range(shape[2])] for _ in range(shape[1])] for _ in range(shape[0])]


def _shape(value: object) -> tuple[int, ...]:
    if isinstance(value, list):
        if not value:
            return (0,)
        return (len(value), *_shape(value[0]))
    return ()


def _flatten(value: object) -> tuple[object, ...]:
    if isinstance(value, list):
        flattened: list[object] = []
        for child in value:
            flattened.extend(_flatten(child))
        return tuple(flattened)
    return (value,)


def _map_nested(value: object, function) -> object:
    if isinstance(value, list):
        return [_map_nested(child, function) for child in value]
    return function(value)


def _count_nonzero(value: object) -> int:
    return sum(1 for item in _flatten(value) if bool(item))


def _is_finite_number(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _matrices_close(left: object, right: object, *, atol: float = 1e-5) -> bool:
    left_rows = tuple(tuple(float(cell) for cell in row) for row in left)
    right_rows = tuple(tuple(float(cell) for cell in row) for row in right)
    return all(abs(a - b) <= atol for left_row, right_row in zip(left_rows, right_rows) for a, b in zip(left_row, right_row))


@dataclass(frozen=True)
class _FakeVoxelCountQc:
    voxel_count: int
    passed: bool
    warnings: tuple[str, ...]
    qc_flags: tuple[str, ...]


@dataclass(frozen=True)
class _FakeCoverageApplication:
    mask: object
    applied_masks: tuple[str, ...]


class _FakeImage:
    def __init__(self, data: object, affine: object | None = None) -> None:
        self._data = data
        self.shape = _shape(data)
        self.affine = affine if affine is not None else _identity_affine()

    def get_fdata(self) -> object:
        return self._data


class _FakeQcRuntime:
    def __init__(self, images: dict[str, _FakeImage]) -> None:
        self.images = images

    def load_nifti_image(self, path: str | Path) -> _FakeImage:
        return self.images[str(path)]

    def validate_compatible_geometry(self, reference_image: _FakeImage, other_image: _FakeImage) -> None:
        if tuple(reference_image.shape[:3]) != tuple(other_image.shape[:3]):
            raise ValueError("shape mismatch")
        if not _matrices_close(reference_image.affine, other_image.affine):
            raise ValueError("affine mismatch")

    def validate_binary_mask(self, mask: object, *, allow_empty: bool = False, label: str = "mask") -> object:
        if len(_shape(mask)) != 3:
            raise ValueError(f"{label} must be 3D")
        flat = _flatten(mask)
        if not all(_is_finite_number(value) for value in flat):
            raise ValueError(f"{label} must be finite")
        if not all(float(value) in {0.0, 1.0} for value in flat):
            raise ValueError(f"{label} must be binary")
        boolean = _map_nested(mask, bool)
        if not allow_empty and _count_nonzero(boolean) == 0:
            raise ValueError(f"{label} must not be empty")
        return boolean

    def check_min_voxel_count(
        self,
        mask: object,
        *,
        min_voxels_warn: int | None = None,
        min_voxels_fail: int | None = None,
    ) -> _FakeVoxelCountQc:
        voxel_count = _count_nonzero(mask)
        warnings: list[str] = []
        flags: list[str] = []
        passed = True
        if min_voxels_fail is not None and voxel_count < int(min_voxels_fail):
            passed = False
            flags.append("fail_min_voxels")
            warnings.append(f"voxel_count {voxel_count} is below min_voxels_fail {int(min_voxels_fail)}")
        if min_voxels_warn is not None and voxel_count < int(min_voxels_warn):
            flags.append("warn_min_voxels")
            warnings.append(f"voxel_count {voxel_count} is below min_voxels_warn {int(min_voxels_warn)}")
        if passed and not flags:
            flags.append("pass")
        return _FakeVoxelCountQc(
            voxel_count=voxel_count,
            passed=passed,
            warnings=tuple(warnings),
            qc_flags=tuple(flags),
        )

    def apply_coverage_masks(self, mask: object, *, coverage_masks: dict[str, object] | None = None, **_: object) -> _FakeCoverageApplication:
        output = mask
        applied: list[str] = []
        for name, coverage in (coverage_masks or {}).items():
            left = _flatten(output)
            right = _flatten(coverage)
            intersected = [bool(a) and bool(b) for a, b in zip(left, right)]
            iterator = iter(intersected)

            def next_value(_old: object) -> bool:
                return next(iterator)

            output = _map_nested(output, next_value)
            applied.append(str(name))
        return _FakeCoverageApplication(mask=output, applied_masks=tuple(applied))

    def write_roi_nifti_mask(self, mask: object, reference_image: _FakeImage, output_path: str | Path) -> Path:
        Path(output_path).write_text(json.dumps({"voxel_count": _count_nonzero(mask)}), encoding="utf-8")
        self.images[str(output_path)] = _FakeImage(mask, affine=reference_image.affine)
        return Path(output_path)


class _RecordingRunner:
    def __init__(self, *, returncode: int = 0, write_output: bool = True) -> None:
        self.returncode = returncode
        self.write_output = write_output
        self.calls: list[tuple[str, ...]] = []

    def __call__(self, argv: tuple[str, ...]) -> SimpleNamespace:
        self.calls.append(tuple(argv))
        if self.returncode == 0 and self.write_output:
            output_path = Path(argv[argv.index("-o") + 1])
            output_path.write_text("transformed\n", encoding="utf-8")
        return SimpleNamespace(returncode=self.returncode, stdout="runner stdout", stderr="runner stderr")


class RoiTransformExecutionTests(unittest.TestCase):
    def _roots(self, root: Path) -> dict[str, Path]:
        return {
            "roi-root": root / "roi root with spaces",
            "target-root": root / "target root",
            "xfm-root": root / "xfm root with spaces",
            "out-root": root / "planned outputs",
            "tool-root": root / "tool root",
        }

    def _config(
        self,
        executable: Path | None,
        *,
        qc: dict[str, object] | None = None,
        include_coverage: bool = True,
    ) -> dict[str, object]:
        source: dict[str, object] = {
            "subject_id": PARTICIPANT_ID,
            "session_id": SESSION_ID,
            "task_id": TASK_ID,
            "run_id": RUN_ID,
            "model": MODEL_ID,
            "contrast_id": CONTRAST_ID,
            "roi_label": ROI_LABEL,
            "source_space": "MNI",
            "source_mask_path": {"root_ref": "roi-root", "path": "mni masks/roi alpha mask.nii.gz"},
            "target_reference_path": {"root_ref": "target-root", "path": "refs/t1 ref.nii.gz"},
            "planned_output_mask_path": {
                "root_ref": "out-root",
                "path": "transformed/{subject_id}/{session_id}/{run_id}/{roi_label}_mask.nii.gz",
            },
            "planned_qc_path": {
                "root_ref": "out-root",
                "path": "transformed/{subject_id}/{session_id}/{run_id}/{roi_label}_qc.json",
            },
            "planned_provenance_path": {
                "root_ref": "out-root",
                "path": "transformed/{subject_id}/{session_id}/{run_id}/{roi_label}_provenance.json",
            },
            "transforms": [
                {"path": {"root_ref": "xfm-root", "path": "warp one.nii.gz"}},
                {"path": {"root_ref": "xfm-root", "path": "affine two.mat"}, "invert": True},
            ],
        }
        if include_coverage:
            source["coverage_mask_path"] = {"root_ref": "target-root", "path": "refs/brain mask.nii.gz"}
        config: dict[str, object] = {
            "missing_policy": "fail",
            "tool_missing_policy": "fail",
            "qc": qc if qc is not None else {"min_voxels_warn": 1, "min_voxels_fail": 1},
            "source_masks": [source],
        }
        if executable is not None:
            config["tool"] = {"executable": str(executable)}
        return config

    def _materialize_inputs(self, root: Path, executable: Path | None) -> None:
        roots = self._roots(root)
        for path in roots.values():
            path.mkdir(parents=True, exist_ok=True)
        _write(roots["roi-root"] / "mni masks" / "roi alpha mask.nii.gz")
        _write(roots["target-root"] / "refs" / "t1 ref.nii.gz")
        _write(roots["target-root"] / "refs" / "brain mask.nii.gz")
        _write(roots["xfm-root"] / "warp one.nii.gz")
        _write(roots["xfm-root"] / "affine two.mat")
        if executable is not None:
            _write(executable, "#!/bin/sh\nexit 0\n")
            _chmod_executable(executable)

    def _plan(self, root: Path, *, qc: dict[str, object] | None = None, include_coverage: bool = True):
        executable = self._roots(root)["tool-root"] / "ants Apply Transforms"
        self._materialize_inputs(root, executable)
        return plan_mni_to_t1w_roi_transforms(
            self._config(executable, qc=qc, include_coverage=include_coverage),
            roots=self._roots(root),
        )

    def _images(
        self,
        plan,
        *,
        output_data: object | None = None,
        target_data: object | None = None,
        coverage_data: object | None = None,
        output_affine: object | None = None,
        target_affine: object | None = None,
    ) -> dict[str, _FakeImage]:
        target = target_data if target_data is not None else _zeros((2, 2, 2))
        output = output_data if output_data is not None else [[[1, 0], [0, 0]], [[0, 1], [0, 0]]]
        coverage = coverage_data if coverage_data is not None else _ones((2, 2, 2))
        return {
            plan.target_references[0].target_reference_path: _FakeImage(target, affine=target_affine),
            plan.planned_outputs[0].output_mask_path: _FakeImage(output, affine=output_affine),
            plan.qc_preview[[row.check_kind for row in plan.qc_preview].index("coverage_mask_exists")].path: _FakeImage(coverage),
        }

    def _execute_with_fake_qc(self, plan, runner: _RecordingRunner, images: dict[str, _FakeImage], **kwargs):
        with mock.patch(
            "research_platform.neuro.roi_transforms._load_roi_transform_qc_runtime",
            return_value=_FakeQcRuntime(images),
        ):
            return execute_mni_to_t1w_roi_transform_plan(plan, runner=runner, **kwargs)

    def test_step_6a_plan_mode_with_execution_artifact_paths_still_writes_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            plan = self._plan(root)
            before = _file_snapshot(root)
            plan_again = plan_mni_to_t1w_roi_transforms(
                self._config(Path(plan.command_plans[0].argv[0])),
                roots=self._roots(root),
            )
            after = _file_snapshot(root)

        self.assertEqual(before, after)
        self.assertFalse(plan_again.executed)
        self.assertTrue(plan_again.plan_only)
        self.assertIsNotNone(plan_again.planned_outputs[0].qc_path)
        self.assertIsNotNone(plan_again.planned_outputs[0].provenance_path)

    def test_successful_fake_runner_execution_records_exact_argv_and_writes_planned_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            plan = self._plan(root)
            runner = _RecordingRunner()
            result = self._execute_with_fake_qc(plan, runner, self._images(plan))

            self.assertEqual(result.status, "ok")
            self.assertEqual(runner.calls, [plan.command_plans[0].argv])
            self.assertIsInstance(runner.calls[0], tuple)
            self.assertTrue(Path(plan.planned_outputs[0].output_mask_path).is_file())
            self.assertTrue(Path(plan.planned_outputs[0].qc_path).is_file())
            self.assertTrue(Path(plan.planned_outputs[0].provenance_path).is_file())
            self.assertEqual(result.geometry_qc[0].status, "ok")
            self.assertEqual(result.binary_mask_qc[0].status, "ok")
            self.assertEqual(result.voxel_count_qc[0].status, "ok")
            self.assertEqual(json.loads(Path(plan.planned_outputs[0].qc_path).read_text())["artifact_kind"], "mni_to_t1w_roi_transform_qc")

    def test_paths_with_spaces_remain_single_argv_elements_during_execution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            plan = self._plan(root)
            runner = _RecordingRunner()
            self._execute_with_fake_qc(plan, runner, self._images(plan))

        argv = runner.calls[0]
        self.assertIn("roi alpha mask.nii.gz", argv[argv.index("-i") + 1])
        self.assertIn("t1 ref.nii.gz", argv[argv.index("-r") + 1])
        self.assertTrue(any(arg.endswith("warp one.nii.gz") for arg in argv))

    def test_missing_path_command_fails_before_runner_call(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            self._materialize_inputs(root, executable=None)
            config = self._config(None)
            config["tool_missing_policy"] = "warn"
            with mock.patch("research_platform.neuro.roi_transforms.shutil.which", return_value=None):
                plan = plan_mni_to_t1w_roi_transforms(config, roots=self._roots(root))
                runner = _RecordingRunner()
                result = execute_mni_to_t1w_roi_transform_plan(plan, runner=runner)

        self.assertEqual(result.status, "error")
        self.assertEqual(runner.calls, [])
        self.assertTrue(any("not found on PATH" in error for error in result.errors))

    def test_missing_configured_executable_fails_before_runner_call(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            missing_executable = self._roots(root)["tool-root"] / "missing-ants"
            self._materialize_inputs(root, executable=None)
            plan = plan_mni_to_t1w_roi_transforms(self._config(missing_executable), roots=self._roots(root))
            runner = _RecordingRunner()
            result = execute_mni_to_t1w_roi_transform_plan(plan, runner=runner)

        self.assertEqual(result.status, "error")
        self.assertEqual(runner.calls, [])

    def test_missing_source_reference_or_transform_refuses_execution(self) -> None:
        cases = (
            ("source", ("roi-root", "mni masks/roi alpha mask.nii.gz")),
            ("reference", ("target-root", "refs/t1 ref.nii.gz")),
            ("transform", ("xfm-root", "warp one.nii.gz")),
        )
        for _, (root_ref, relative) in cases:
            with self.subTest(root_ref=root_ref):
                with tempfile.TemporaryDirectory() as tmp_dir:
                    root = Path(tmp_dir)
                    executable = self._roots(root)["tool-root"] / "ants Apply Transforms"
                    self._materialize_inputs(root, executable)
                    (self._roots(root)[root_ref] / relative).unlink()
                    plan = plan_mni_to_t1w_roi_transforms(self._config(executable), roots=self._roots(root))
                    runner = _RecordingRunner()
                    result = execute_mni_to_t1w_roi_transform_plan(plan, runner=runner)

                self.assertEqual(result.status, "error")
                self.assertEqual(runner.calls, [])

    def test_existing_transformed_output_refuses_overwrite_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            plan = self._plan(root)
            _write(Path(plan.planned_outputs[0].output_mask_path), "existing\n")
            runner = _RecordingRunner()
            result = execute_mni_to_t1w_roi_transform_plan(plan, runner=runner)

        self.assertEqual(result.status, "error")
        self.assertEqual(runner.calls, [])
        self.assertTrue(any("overwrite is false" in error for error in result.errors))

    def test_duplicate_planned_destinations_fail_before_runner_or_writes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            executable = self._roots(root)["tool-root"] / "ants Apply Transforms"
            self._materialize_inputs(root, executable)
            config = self._config(executable)
            sources = config["source_masks"]
            assert isinstance(sources, list)
            assert isinstance(sources[0], dict)
            sources.append(dict(sources[0]))
            plan = plan_mni_to_t1w_roi_transforms(config, roots=self._roots(root))
            before = _file_snapshot(root)
            runner = _RecordingRunner()

            result = execute_mni_to_t1w_roi_transform_plan(plan, runner=runner)

            after = _file_snapshot(root)

        self.assertEqual(result.status, "error")
        self.assertEqual(runner.calls, [])
        self.assertEqual(after, before)
        self.assertTrue(
            any("duplicate planned ROI transform destination" in error for error in result.errors)
        )

    def test_symbolic_link_output_parent_fails_before_runner_or_writes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            plan = self._plan(root)
            real_output = root / "outside-output"
            real_output.mkdir()
            linked_parent = self._roots(root)["out-root"] / "transformed"
            linked_parent.symlink_to(real_output, target_is_directory=True)
            before = _file_snapshot(root)
            runner = _RecordingRunner()

            result = execute_mni_to_t1w_roi_transform_plan(plan, runner=runner)

            after = _file_snapshot(root)
            outside_entries = tuple(real_output.iterdir())

        self.assertEqual(result.status, "error")
        self.assertEqual(runner.calls, [])
        self.assertEqual(after, before)
        self.assertTrue(any("symbolic-link" in error for error in result.errors))
        self.assertEqual(outside_entries, ())

    def test_overwrite_replaces_only_planned_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            plan = self._plan(root)
            output = Path(plan.planned_outputs[0].output_mask_path)
            qc = Path(plan.planned_outputs[0].qc_path)
            provenance = Path(plan.planned_outputs[0].provenance_path)
            unplanned = self._roots(root)["out-root"] / "unplanned.json"
            _write(output, "old output\n")
            _write(qc, '{"old": true}\n')
            _write(provenance, '{"old": true}\n')
            _write(unplanned, "do not touch\n")
            runner = _RecordingRunner()
            result = self._execute_with_fake_qc(plan, runner, self._images(plan), overwrite=True)

            self.assertEqual(result.status, "ok")
            self.assertEqual(output.read_text(encoding="utf-8"), "transformed\n")
            self.assertEqual(unplanned.read_text(encoding="utf-8"), "do not touch\n")
            self.assertEqual(json.loads(qc.read_text(encoding="utf-8"))["artifact_kind"], "mni_to_t1w_roi_transform_qc")
            self.assertEqual(json.loads(provenance.read_text(encoding="utf-8"))["artifact_kind"], "mni_to_t1w_roi_transform_provenance")

    def test_failed_command_records_failure_and_writes_no_qc_or_provenance_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            plan = self._plan(root)
            runner = _RecordingRunner(returncode=2, write_output=False)
            result = execute_mni_to_t1w_roi_transform_plan(plan, runner=runner)

            self.assertEqual(result.status, "error")
            self.assertEqual(result.command_records[0].returncode, 2)
            self.assertFalse(Path(plan.planned_outputs[0].qc_path).exists())
            self.assertFalse(Path(plan.planned_outputs[0].provenance_path).exists())

    def test_command_success_with_missing_output_marks_verification_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            plan = self._plan(root)
            runner = _RecordingRunner(returncode=0, write_output=False)
            result = execute_mni_to_t1w_roi_transform_plan(plan, runner=runner)

            self.assertEqual(result.status, "error")
            self.assertEqual(result.output_verifications[0].status, "error")
            self.assertFalse(Path(plan.planned_outputs[0].qc_path).exists())
            self.assertFalse(Path(plan.planned_outputs[0].provenance_path).exists())

    def test_geometry_mismatch_fails_actual_qc(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            plan = self._plan(root)
            images = self._images(plan, output_data=_ones((3, 2, 2)))
            result = self._execute_with_fake_qc(plan, _RecordingRunner(), images)

        self.assertEqual(result.status, "error")
        self.assertEqual(result.geometry_qc[0].status, "error")

    def test_nonbinary_mask_fails_actual_qc(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            plan = self._plan(root)
            images = self._images(plan, output_data=[[[2, 0], [0, 0]], [[0, 1], [0, 0]]])
            result = self._execute_with_fake_qc(plan, _RecordingRunner(), images)

        self.assertEqual(result.status, "error")
        self.assertEqual(result.binary_mask_qc[0].status, "error")

    def test_empty_transformed_mask_fails_actual_qc(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            plan = self._plan(root)
            images = self._images(plan, output_data=_zeros((2, 2, 2)))
            result = self._execute_with_fake_qc(plan, _RecordingRunner(), images)

        self.assertEqual(result.status, "error")
        self.assertEqual(result.voxel_count_qc[0].status, "error")
        self.assertTrue(result.voxel_count_qc[0].empty)

    def test_small_mask_warning_and_failure_thresholds_are_applied(self) -> None:
        with tempfile.TemporaryDirectory() as warn_dir:
            root = Path(warn_dir)
            plan = self._plan(root, qc={"min_voxels_warn": 2, "min_voxels_fail": 1})
            images = self._images(plan, output_data=[[[1, 0], [0, 0]], [[0, 0], [0, 0]]])
            warn_result = self._execute_with_fake_qc(plan, _RecordingRunner(), images)
        with tempfile.TemporaryDirectory() as fail_dir:
            root = Path(fail_dir)
            plan = self._plan(root, qc={"min_voxels_warn": 2, "min_voxels_fail": 2})
            images = self._images(plan, output_data=[[[1, 0], [0, 0]], [[0, 0], [0, 0]]])
            fail_result = self._execute_with_fake_qc(plan, _RecordingRunner(), images)

        self.assertEqual(warn_result.status, "warning")
        self.assertEqual(warn_result.voxel_count_qc[0].status, "warning")
        self.assertEqual(fail_result.status, "error")
        self.assertEqual(fail_result.voxel_count_qc[0].status, "error")

    def test_coverage_overlap_ratio_is_computed_and_thresholded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            plan = self._plan(root, qc={"coverage_min_overlap_fraction_warn": 0.75})
            output = [[[1, 0], [0, 0]], [[0, 1], [0, 0]]]
            coverage = [[[1, 0], [0, 0]], [[0, 0], [0, 0]]]
            images = self._images(plan, output_data=output, coverage_data=coverage)
            result = self._execute_with_fake_qc(plan, _RecordingRunner(), images)

        self.assertEqual(result.status, "warning")
        self.assertEqual(result.coverage_overlap_qc[0].status, "warning")
        self.assertEqual(result.coverage_overlap_qc[0].overlap_ratio, 0.5)

    def test_configured_coverage_intersection_is_applied_before_voxel_qc(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            plan = self._plan(root, qc={"min_voxels_warn": 1, "min_voxels_fail": 1, "coverage_intersection_policy": "apply"})
            output = [[[1, 0], [0, 0]], [[0, 1], [0, 0]]]
            coverage = [[[1, 0], [0, 0]], [[0, 0], [0, 0]]]
            images = self._images(plan, output_data=output, coverage_data=coverage)
            result = self._execute_with_fake_qc(plan, _RecordingRunner(), images)

        self.assertEqual(result.status, "ok")
        self.assertEqual(result.coverage_intersection_qc[0].status, "ok")
        self.assertEqual(result.coverage_intersection_qc[0].original_voxel_count, 2)
        self.assertEqual(result.coverage_intersection_qc[0].retained_voxel_count, 1)
        self.assertEqual(result.coverage_intersection_qc[0].dropped_voxel_count, 1)
        self.assertEqual(result.voxel_count_qc[0].voxel_count, 1)
        self.assertEqual(result.coverage_overlap_qc[0].overlap_ratio, 1.0)

    def test_configured_coverage_intersection_can_make_small_roi_fail(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            plan = self._plan(root, qc={"min_voxels_warn": 2, "min_voxels_fail": 2, "coverage_intersection_policy": "apply"})
            output = [[[1, 0], [0, 0]], [[0, 1], [0, 0]]]
            coverage = [[[1, 0], [0, 0]], [[0, 0], [0, 0]]]
            images = self._images(plan, output_data=output, coverage_data=coverage)
            result = self._execute_with_fake_qc(plan, _RecordingRunner(), images)

        self.assertEqual(result.status, "error")
        self.assertEqual(result.coverage_intersection_qc[0].retained_voxel_count, 1)
        self.assertEqual(result.voxel_count_qc[0].status, "error")
        self.assertEqual(result.voxel_count_qc[0].min_voxels_fail, 2)

    def test_plan_dict_payload_is_executable_and_result_is_json_safe(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            plan = self._plan(root)
            runner = _RecordingRunner()
            result = self._execute_with_fake_qc(plan.to_dict(), runner, self._images(plan))
            encoded = json.dumps(result.to_dict(), allow_nan=False, sort_keys=True)

        self.assertIn('"executed": true', encoded)
        self.assertEqual(result.status, "ok")

    def test_execution_writes_only_planned_output_qc_and_provenance_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            plan = self._plan(root)
            before = set(_file_snapshot(root))
            result = self._execute_with_fake_qc(plan, _RecordingRunner(), self._images(plan))
            after = set(_file_snapshot(root))
            new_files = after - before
            expected = {
                Path(plan.planned_outputs[0].output_mask_path).relative_to(root).as_posix(),
                Path(plan.planned_outputs[0].qc_path).relative_to(root).as_posix(),
                Path(plan.planned_outputs[0].provenance_path).relative_to(root).as_posix(),
            }

        self.assertEqual(result.status, "ok")
        self.assertEqual(new_files, expected)

    def test_validation_and_selection_are_side_effect_free(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            plan = self._plan(root)
            before = _file_snapshot(root)
            validation = validate_mni_to_t1w_roi_transform_execution_plan(plan)
            jobs = select_executable_mni_to_t1w_roi_transform_jobs(plan)
            after = _file_snapshot(root)

        self.assertTrue(validation.valid)
        self.assertEqual(len(jobs), 1)
        self.assertEqual(before, after)

    def test_no_mvpa_calls_or_forbidden_imports_in_roi_transform_execution_code(self) -> None:
        module_path = PACKAGE_ROOT / "src" / "research_platform" / "neuro" / "roi_transforms.py"
        source = module_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        forbidden_prefixes = (
            "research_platform." + "core",
            "research_platform." + "analysis",
            "research_platform." + "bids",
            "research_platform." + "viz",
            "research_platform." + "hpc",
            "pipe" + "lines",
            "op" + "s",
            "sk" + "learn",
            "nile" + "arn",
            "rsa" + "toolbox",
        )
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    self.assertFalse(alias.name.startswith(forbidden_prefixes))
            elif isinstance(node, ast.ImportFrom) and node.module:
                self.assertFalse(node.module.startswith(forbidden_prefixes))
        forbidden_calls = (
            "extract_mvpa",
            "compute_distance",
            "compute_distances",
            "cross" + "nobis",
            "rsa" + "toolbox",
            "research_platform.neuro." + "mvpa." + "extraction",
        )
        for text in forbidden_calls:
            self.assertNotIn(text, source)


if __name__ == "__main__":
    unittest.main()
