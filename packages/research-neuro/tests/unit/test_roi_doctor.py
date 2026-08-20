from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PACKAGE_ROOT / "src"))
sys.path.insert(0, str(PACKAGE_ROOT.parent / "research-bids" / "src"))

from research_platform.neuro.roi_doctor import doctor_extraction_set, doctor_roi_set
from research_platform.neuro.roi_execution import RoiExecutionContext
from research_platform.neuro.roi_scaffold import (
    build_extraction_set_document,
    build_roi_set_document,
)


class _Preflight:
    def __init__(self, *, ready: bool, checks: list[dict[str, object]]) -> None:
        self._payload = {
            "ready_for_execution": ready,
            "existing_output": "fail",
            "output_paths": [],
            "checks": checks,
        }

    def to_dict(self) -> dict[str, object]:
        return dict(self._payload)


class RoiDoctorTests(unittest.TestCase):
    def test_real_coordinate_scaffold_can_be_execution_ready(self) -> None:
        try:
            import nibabel as nib
            import numpy as np
        except ImportError:
            self.skipTest("nibabel and numpy are required for the execution-ready doctor regression")

        document = build_roi_set_document("example_rois", "coordinate_sphere")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            context = _context(root)
            reference_path = context.project_root / "inputs" / "roi" / "example_reference.nii.gz"
            reference_path.parent.mkdir(parents=True)
            nib.save(
                nib.Nifti1Image(np.zeros((9, 9, 9), dtype=np.float32), np.eye(4)),
                reference_path,
            )
            before = _relative_paths(root)

            report = doctor_roi_set(document, context=context)

            self.assertEqual(_relative_paths(root), before)

        self.assertTrue(report["schema_valid"])
        self.assertTrue(report["ready_for_execution"])
        self.assertFalse(report["errors"])
        self.assertTrue(all(check["status"] == "ok" for check in report["checks"]))

    def test_real_coordinate_scaffold_preflight_is_non_mutating(self) -> None:
        document = build_roi_set_document("example_rois", "coordinate_sphere")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            context = _context(root)
            before = _relative_paths(root)

            report = doctor_roi_set(document, context=context)

            self.assertEqual(_relative_paths(root), before)

        self.assertTrue(report["schema_valid"])
        self.assertFalse(report["ready_for_execution"])
        self.assertEqual(_check_by_id(report, "input_exists")["status"], "error")
        self.assertTrue(
            all(check.get("action") for check in report["checks"] if check["status"] != "ok")
        )

    def test_real_extraction_scaffold_preflight_is_non_mutating(self) -> None:
        roi_document = build_roi_set_document("example_rois", "coordinate_sphere")
        extraction_document = build_extraction_set_document(
            "example_values",
            roi_set="example_rois",
            template="generic_nifti",
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            context = _context(root)
            before = _relative_paths(root)

            report = doctor_extraction_set(
                extraction_document,
                roi_set_document=roi_document,
                context=context,
            )

            self.assertEqual(_relative_paths(root), before)

        self.assertTrue(report["schema_valid"])
        self.assertFalse(report["ready_for_execution"])
        self.assertEqual(_check_by_id(report, "input_exists")["status"], "error")
        self.assertTrue(
            all(check.get("action") for check in report["checks"] if check["status"] != "ok")
        )

    def test_roi_doctor_reports_unknown_named_root_as_actionable(self) -> None:
        document = build_roi_set_document("example_rois", "coordinate_sphere")
        document["roi_set"]["outputs"] = {"root_ref": "missing_root", "path": "roi-runtime/example_rois"}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            context = _context(root)
            before = _relative_paths(root)
            report = doctor_roi_set(document, context=context)
            self.assertEqual(_relative_paths(root), before)

        root_check = _check_by_id(report, "configured_root_available")
        self.assertTrue(report["schema_valid"])
        self.assertFalse(report["ready_for_execution"])
        self.assertEqual(root_check["status"], "error")
        self.assertIn("Configure or create", root_check["action"])
        self.assertFalse(any("--execute" in step for step in report["next_steps"]))

    def test_extraction_doctor_reports_unknown_named_root_as_actionable(self) -> None:
        roi_document = build_roi_set_document("example_rois", "coordinate_sphere")
        extraction_document = build_extraction_set_document(
            "example_values",
            roi_set="example_rois",
            template="generic_nifti",
        )
        extraction_document["extraction_set"]["outputs"] = {
            "root_ref": "missing_root",
            "path": "roi-runtime/example_values",
            "format": "tsv",
        }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            context = _context(root)
            before = _relative_paths(root)
            report = doctor_extraction_set(
                extraction_document,
                roi_set_document=roi_document,
                context=context,
            )
            self.assertEqual(_relative_paths(root), before)

        root_check = _check_by_id(report, "configured_root_available")
        self.assertTrue(report["schema_valid"])
        self.assertFalse(report["ready_for_execution"])
        self.assertEqual(root_check["status"], "error")
        self.assertIn("Configure or create", root_check["action"])
        self.assertFalse(any("--execute" in step for step in report["next_steps"]))

    def test_schema_valid_scaffold_can_be_not_ready_for_execution(self) -> None:
        document = build_roi_set_document("example_rois", "coordinate_sphere")
        checks = [
            _check("configuration_valid", "ok", "Configuration is structurally valid."),
            _check("configured_root_available", "ok", "The configured output root is available."),
            _check(
                "input_exists",
                "error",
                "A required reference image is missing.",
                path="inputs/roi/example_reference.nii.gz",
                category="reference_image",
            ),
            _check("python_dependency_available", "ok", "The NIfTI dependency is available."),
        ]

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            context = _context(root)
            before = _relative_paths(root)
            with patch(
                "research_platform.neuro.roi_doctor._execution_preflight_functions",
                return_value=(lambda *args, **kwargs: _Preflight(ready=False, checks=checks), lambda *args, **kwargs: None),
            ):
                report = doctor_roi_set(document, context=context)

            self.assertEqual(_relative_paths(root), before)

        self.assertTrue(report["valid"])
        self.assertTrue(report["schema_valid"])
        self.assertFalse(report["ready_for_execution"])
        failed = _check_by_id(report, "input_exists")
        self.assertEqual(failed["status"], "error")
        self.assertIn("Populate or correct", failed["action"])
        self.assertTrue(any(item["kind"] == "reference_image" for item in report["missing_inputs"]))

    def test_invalid_configuration_does_not_run_execution_preflight(self) -> None:
        document = build_roi_set_document("example_rois", "coordinate_sphere")
        document["roi_set"]["rois"] = []

        with tempfile.TemporaryDirectory() as directory:
            context = _context(Path(directory))
            with patch(
                "research_platform.neuro.roi_doctor._execution_preflight_functions"
            ) as preflight_functions:
                report = doctor_roi_set(document, context=context)

        preflight_functions.assert_not_called()
        self.assertFalse(report["valid"])
        self.assertFalse(report["schema_valid"])
        self.assertFalse(report["ready_for_execution"])
        configuration = _check_by_id(report, "configuration_valid")
        self.assertEqual(configuration["status"], "error")
        self.assertIn("Fix the reported configuration", configuration["action"])

    def test_every_execution_readiness_failure_has_stable_id_and_action(self) -> None:
        document = build_roi_set_document("example_rois", "coordinate_sphere")
        readiness_checks = [
            _check(check_id, "error", f"Synthetic {check_id} finding.", category="test")
            for check_id in (
                "roi_family_supported",
                "configured_root_available",
                "input_exists",
                "python_dependency_available",
                "external_tool_available",
                "image_readable",
                "image_geometry_compatible",
                "output_collision",
            )
        ]

        with tempfile.TemporaryDirectory() as directory:
            with patch(
                "research_platform.neuro.roi_doctor._execution_preflight_functions",
                return_value=(
                    lambda *args, **kwargs: _Preflight(ready=False, checks=readiness_checks),
                    lambda *args, **kwargs: None,
                ),
            ):
                report = doctor_roi_set(document, context=_context(Path(directory)))

        observed = {check["check_id"]: check for check in report["checks"]}
        for check_id in (
            "roi_family_supported",
            "configured_root_available",
            "input_exists",
            "python_dependency_available",
            "external_tool_available",
            "image_readable",
            "image_geometry_compatible",
            "output_collision",
        ):
            self.assertEqual(observed[check_id]["status"], "error")
            self.assertTrue(observed[check_id]["action"])

    def test_extraction_doctor_uses_shared_readiness_contract(self) -> None:
        roi_document = build_roi_set_document("example_rois", "coordinate_sphere")
        extraction_document = build_extraction_set_document(
            "example_values",
            roi_set="example_rois",
            template="generic_nifti",
        )
        checks = [
            _check("configuration_valid", "ok", "Configuration is structurally valid."),
            _check(
                "image_geometry_compatible",
                "error",
                "A value image and ROI mask use incompatible geometry.",
                category="geometry",
            ),
        ]

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            context = _context(root)
            before = _relative_paths(root)
            with (
                patch(
                    "research_platform.neuro.roi_doctor.plan_roi_extraction",
                    return_value=SimpleNamespace(actions=()),
                ),
                patch(
                    "research_platform.neuro.roi_doctor._execution_preflight_functions",
                    return_value=(
                        lambda *args, **kwargs: None,
                        lambda *args, **kwargs: _Preflight(ready=False, checks=checks),
                    ),
                ),
            ):
                report = doctor_extraction_set(
                    extraction_document,
                    roi_set_document=roi_document,
                    context=context,
                )

            self.assertEqual(_relative_paths(root), before)

        self.assertTrue(report["valid"])
        self.assertTrue(report["schema_valid"])
        self.assertFalse(report["ready_for_execution"])
        geometry = _check_by_id(report, "image_geometry_compatible")
        self.assertIn("compatible image geometry", geometry["action"])


def _context(root: Path) -> RoiExecutionContext:
    root = root.resolve()
    project_root = root / "project"
    artifacts_root = root / "artifacts"
    project_root.mkdir()
    artifacts_root.mkdir()
    return RoiExecutionContext(
        workspace_root=root,
        project_root=project_root,
        artifacts_root=artifacts_root,
        project_name="project-demo",
    )


def _relative_paths(root: Path) -> tuple[str, ...]:
    return tuple(sorted(str(path.relative_to(root)) for path in root.rglob("*")))


def _check(
    check_id: str,
    status: str,
    message: str,
    *,
    path: str | None = None,
    category: str | None = None,
) -> dict[str, object]:
    return {
        "check_id": check_id,
        "status": status,
        "message": message,
        **({"path": path} if path is not None else {}),
        **({"category": category} if category is not None else {}),
    }


def _check_by_id(report: dict[str, object], check_id: str) -> dict[str, object]:
    checks = report["checks"]
    assert isinstance(checks, list)
    return next(check for check in checks if check["check_id"] == check_id)


if __name__ == "__main__":
    unittest.main()
