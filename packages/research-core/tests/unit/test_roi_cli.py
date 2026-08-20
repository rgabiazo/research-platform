from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

try:
    import numpy as np
    import nibabel as nib
except ImportError:  # pragma: no cover - local minimal environments may skip.
    np = None  # type: ignore[assignment]
    nib = None  # type: ignore[assignment]

CORE_PACKAGE_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
BIDS_PACKAGE_ROOT = WORKSPACE_ROOT / "packages" / "research-bids"
HPC_PACKAGE_ROOT = WORKSPACE_ROOT / "packages" / "research-hpc"
NEURO_PACKAGE_ROOT = WORKSPACE_ROOT / "packages" / "research-neuro"
sys.path.insert(0, str(CORE_PACKAGE_ROOT / "src"))
sys.path.insert(0, str(BIDS_PACKAGE_ROOT / "src"))
sys.path.insert(0, str(HPC_PACKAGE_ROOT / "src"))
sys.path.insert(0, str(NEURO_PACKAGE_ROOT / "src"))

from research_platform.core.cli import _build_parser, _looks_like_absolute_path_literal, main
from research_platform.core.config import load_yaml, parse_yaml, write_yaml


class RoiCliTests(unittest.TestCase):
    def test_roi_transform_literal_path_policy_is_platform_neutral(self) -> None:
        for value in (
            "/srv/research/roi-mask.nii.gz",
            "root=/mnt/research/roi-mask.nii.gz",
            "~/research/roi-mask.nii.gz",
            "~alice/research/roi-mask.nii.gz",
            "C:\\research\\roi-mask.nii.gz",
            "\\\\cluster.example\\share\\roi-mask.nii.gz",
        ):
            with self.subTest(value=value):
                self.assertTrue(_looks_like_absolute_path_literal(value))

        for value in (
            "relative/roi-mask.nii.gz",
            "${ROI_ROOT:-/srv/research}/roi-mask.nii.gz",
            "https://cluster.example/roi-mask.nii.gz",
        ):
            with self.subTest(value=value):
                self.assertFalse(_looks_like_absolute_path_literal(value))

    def _write_workspace(self, workspace_root: Path) -> Path:
        write_yaml(
            workspace_root / "WORKSPACE.yaml",
            {
                "paths": {
                    "artifacts_root": "./artifacts",
                    "datasets_root": "./datasets",
                    "ops_root": "./ops",
                },
                "repos": {
                    "project_root": "./project",
                    "pipelines_root": "./pipelines",
                },
                "projects": {"default": "project-default"},
            },
        )
        project_root = workspace_root / "project" / "project-default"
        write_yaml(project_root / "project.yaml", {"name": "project-default", "version": "0.1.0"})
        return project_root

    def _write_roi_configs(self, project_root: Path) -> None:
        write_yaml(
            project_root / "config" / "analysis" / "roi_sets" / "modelA.yaml",
            {
                "roi_set": {
                    "name": "modelA",
                    "rois": [
                        {
                            "label": "SeedSphere",
                            "family": "coordinate_sphere",
                            "coordinate": [0, -52, 26],
                            "radius_mm": 6,
                        }
                    ],
                }
            },
        )
        write_yaml(
            project_root / "config" / "analysis" / "roi_transforms" / "modelA_t1w.yaml",
            {
                "roi_transform_plan": {
                    "missing_policy": "warn",
                    "tool_missing_policy": "warn",
                    "selectors": {"subjects": ["sub-001"], "sessions": ["ses-01"], "runs": ["run-01"]},
                    "source_masks": [
                        {
                            "roi_label": "SeedSphere",
                            "source_mask_path": {
                                "root_ref": "project_roi_root",
                                "path": "roi_sets/{subject_dir}_{session_dir}_label-{roi_label}_mask.nii.gz",
                            },
                        }
                    ],
                    "target_references": [
                        {
                            "target_reference_path": {
                                "root_ref": "artifacts_root",
                                "path": "feat/{subject_dir}/{session_dir}/{run_entity}/stats/pe1.nii.gz",
                            }
                        }
                    ],
                    "transform_chains": [
                        {
                            "path": {
                                "root_ref": "artifacts_root",
                                "path": "xfm/{subject_dir}/{session_dir}/from-MNI_to-T1w.h5",
                            }
                        }
                    ],
                    "outputs": {
                        "root_ref": "artifacts_root",
                        "path": "roi-transforms/{subject_dir}/{session_dir}/{run_entity}/{roi_label}_mask.nii.gz",
                    },
                }
            },
        )
        write_yaml(
            project_root / "config" / "analysis" / "extraction_sets" / "modelA_timeseries.yaml",
            {
                "extraction_set": {
                    "name": "modelA_timeseries",
                    "roi_set": "modelA",
                    "targets": [
                        {
                            "name": "niftiTimeseries",
                            "backend": "generic_nifti",
                            "desc": "ModelATimeseries",
                        }
                    ],
                }
            },
        )

    def _run_cli(
        self,
        args: list[str],
        *,
        workspace_root: Path,
        extra_env: dict[str, str] | None = None,
    ) -> tuple[int, str]:
        buffer = io.StringIO()
        env = {"RESEARCH_PLATFORM_ROOT": str(workspace_root)}
        if extra_env:
            env.update(extra_env)
        with mock.patch.dict(os.environ, env, clear=False):
            with redirect_stdout(buffer):
                exit_code = main(args)
        return exit_code, buffer.getvalue()

    def _run_cli_exit(self, args: list[str], *, workspace_root: Path) -> str:
        with mock.patch.dict(os.environ, {"RESEARCH_PLATFORM_ROOT": str(workspace_root)}, clear=False):
            with self.assertRaises(SystemExit) as error:
                main(args)
        return str(error.exception)

    def _run_cli_parse_error(self, args: list[str], *, workspace_root: Path) -> tuple[int, str]:
        buffer = io.StringIO()
        with mock.patch.dict(os.environ, {"RESEARCH_PLATFORM_ROOT": str(workspace_root)}, clear=False):
            with redirect_stderr(buffer):
                with self.assertRaises(SystemExit) as error:
                    main(args)
        return int(error.exception.code), buffer.getvalue()

    def _workspace_file_snapshot(self, workspace_root: Path) -> list[str]:
        return sorted(path.relative_to(workspace_root).as_posix() for path in workspace_root.rglob("*") if path.is_file())

    def _workspace_tree_snapshot(self, workspace_root: Path) -> list[tuple[str, str, bytes | None]]:
        snapshot: list[tuple[str, str, bytes | None]] = []
        for path in sorted(workspace_root.rglob("*")):
            relative = path.relative_to(workspace_root).as_posix()
            if path.is_symlink():
                snapshot.append((relative, "symlink", os.readlink(path).encode("utf-8")))
            elif path.is_file():
                snapshot.append((relative, "file", path.read_bytes()))
            elif path.is_dir():
                snapshot.append((relative, "directory", None))
        return snapshot

    def _synthetic_personal_env_roots(self) -> tuple[Path, Path]:
        root = Path("/home/alice/rp-roi-env-placeholder-test")
        return root / "feat", root / "derivatives"

    def _subparser(self, parser: object, name: str) -> object:
        for action in getattr(parser, "_actions", []):
            choices = getattr(action, "choices", None)
            if isinstance(choices, dict) and name in choices:
                return choices[name]
        raise AssertionError(f"Subparser {name!r} not found.")

    def _subparser_choices(self, parser: object) -> dict[str, object]:
        for action in getattr(parser, "_actions", []):
            choices = getattr(action, "choices", None)
            if isinstance(choices, dict) and choices:
                return choices
        raise AssertionError("Subparser choices not found.")

    def _option_action(self, parser: object, option: str) -> object:
        for action in getattr(parser, "_actions", []):
            if option in getattr(action, "option_strings", []):
                return action
        raise AssertionError(f"Option {option!r} not found.")

    def _existing_runtime_path_mocks(self, *roots: Path) -> tuple[object, object]:
        real_exists = Path.exists
        real_stat = Path.stat
        root_texts = tuple(
            dict.fromkeys(
                text
                for root in roots
                for text in (
                    root.as_posix().rstrip("/"),
                    root.resolve(strict=False).as_posix().rstrip("/"),
                )
            )
        )

        def is_under_runtime_root(path: Path) -> bool:
            text = path.as_posix()
            return any(text == root or text.startswith(f"{root}/") for root in root_texts)

        def exists(path: Path) -> bool:
            if path.suffix == ".json" and is_under_runtime_root(path):
                return False
            return True if is_under_runtime_root(path) else real_exists(path)

        def stat(path: Path, *args: object, **kwargs: object) -> os.stat_result:
            if is_under_runtime_root(path):
                return os.stat_result((0, 0, 0, 0, 0, 0, 1, 0, 0, 0))
            return real_stat(path, *args, **kwargs)

        return mock.patch.object(Path, "exists", exists), mock.patch.object(Path, "stat", stat)

    def test_roi_lifecycle_commands_list_show_and_validate_configs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            project_root = self._write_workspace(workspace_root)
            self._write_roi_configs(project_root)

            list_code, list_output = self._run_cli(
                ["analysis", "roi", "list", "--project", "project-default"],
                workspace_root=workspace_root,
            )
            show_code, show_output = self._run_cli(
                ["analysis", "roi", "show", "modelA", "--project", "project-default"],
                workspace_root=workspace_root,
            )
            validate_code, validate_output = self._run_cli(
                ["analysis", "roi", "validate", "modelA", "--project", "project-default"],
                workspace_root=workspace_root,
            )
            extraction_code, extraction_output = self._run_cli(
                [
                    "analysis",
                    "roi",
                    "extraction",
                    "validate",
                    "modelA_timeseries",
                    "--project",
                    "project-default",
                ],
                workspace_root=workspace_root,
            )

        list_payload = json.loads(list_output)
        validate_payload = json.loads(validate_output)
        extraction_payload = json.loads(extraction_output)
        self.assertEqual(list_code, 0)
        self.assertEqual(list_payload["roi_sets"], ["modelA"])
        self.assertEqual(list_payload["transform_sets"], ["modelA_t1w"])
        self.assertEqual(list_payload["extraction_sets"], ["modelA_timeseries"])
        self.assertEqual(show_code, 0)
        self.assertIn("ROI set: modelA", show_output)
        self.assertEqual(validate_code, 0)
        self.assertTrue(validate_payload["valid"])
        self.assertEqual(extraction_code, 0)
        self.assertTrue(extraction_payload["valid"])

    def test_roi_transform_plan_command_is_no_write_and_uses_configured_roots(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            project_root = self._write_workspace(workspace_root)
            self._write_roi_configs(project_root)
            before = self._workspace_file_snapshot(workspace_root)

            code, output = self._run_cli(
                ["analysis", "roi", "transform", "plan", "modelA_t1w", "--project", "project-default"],
                workspace_root=workspace_root,
            )

            after = self._workspace_file_snapshot(workspace_root)

        payload = json.loads(output)
        self.assertEqual(code, 0)
        self.assertTrue(payload["valid"])
        self.assertEqual(payload["mode"], "plan")
        self.assertEqual(payload["transform_set"], "modelA_t1w")
        self.assertEqual(len(payload["command_plans"]), 1)
        self.assertIn("project_roi_root", payload["root_refs"])
        self.assertEqual(after, before)

    def test_roi_transform_validate_is_structural_and_doctor_reports_readiness(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            project_root = self._write_workspace(workspace_root)
            self._write_roi_configs(project_root)
            before = self._workspace_file_snapshot(workspace_root)

            validate_code, validate_output = self._run_cli(
                ["analysis", "roi", "transform", "validate", "modelA_t1w", "--project", "project-default"],
                workspace_root=workspace_root,
            )
            doctor_code, doctor_output = self._run_cli(
                ["analysis", "roi", "transform", "doctor", "modelA_t1w", "--project", "project-default"],
                workspace_root=workspace_root,
            )
            after = self._workspace_file_snapshot(workspace_root)

        validate_payload = json.loads(validate_output)
        doctor_payload = json.loads(doctor_output)
        self.assertEqual(validate_code, 0)
        self.assertTrue(validate_payload["valid"])
        self.assertTrue(validate_payload["schema_valid"])
        self.assertNotIn("ready_for_execution", validate_payload)
        self.assertEqual(doctor_code, 1)
        self.assertTrue(doctor_payload["schema_valid"])
        self.assertFalse(doctor_payload["ready_for_execution"])
        check_ids = {check["check_id"] for check in doctor_payload["checks"]}
        self.assertEqual(
            check_ids,
            {
                "configuration_valid",
                "configured_root_available",
                "input_exists",
                "external_tool_available",
                "output_collision",
                "execution_readiness",
            },
        )
        self.assertTrue(
            all(check.get("action") for check in doctor_payload["checks"] if check["status"] != "ok")
        )
        self.assertEqual(after, before)

    def test_roi_transform_validate_rejects_malformed_structure_without_runtime_preflight(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            project_root = self._write_workspace(workspace_root)
            write_yaml(
                project_root / "config" / "analysis" / "roi_transforms" / "malformed.yaml",
                {"roi_transform_plan": {"source_masks": "not-a-row-container"}},
            )
            before = self._workspace_file_snapshot(workspace_root)

            code, output = self._run_cli(
                ["analysis", "roi", "transform", "validate", "malformed", "--project", "project-default"],
                workspace_root=workspace_root,
            )

            after = self._workspace_file_snapshot(workspace_root)

        payload = json.loads(output)
        self.assertEqual(code, 1)
        self.assertFalse(payload["valid"])
        self.assertFalse(payload["schema_valid"])
        self.assertTrue(any("source_masks" in error for error in payload["errors"]))
        self.assertNotIn("Traceback", output)
        self.assertEqual(after, before)

    def test_roi_transform_plan_and_run_reject_malformed_structure_before_planning(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            project_root = self._write_workspace(workspace_root)
            write_yaml(
                project_root / "config" / "analysis" / "roi_transforms" / "malformed.yaml",
                {"roi_transform_plan": {"source_masks": "not-a-row-container"}},
            )
            before = self._workspace_file_snapshot(workspace_root)

            for command in (("plan",), ("run",), ("run", "--execute")):
                with self.subTest(command=command), mock.patch(
                    "research_platform.core.cli._roi_transform_functions"
                ) as transform_functions:
                    code, output = self._run_cli(
                        [
                            "analysis",
                            "roi",
                            "transform",
                            *command,
                            "malformed",
                            "--project",
                            "project-default",
                        ],
                        workspace_root=workspace_root,
                    )

                    payload = json.loads(output)
                    self.assertEqual(code, 1)
                    self.assertFalse(payload["valid"])
                    self.assertTrue(any("source_masks" in error for error in payload["errors"]))
                    self.assertNotIn("Traceback", output)
                    transform_functions.assert_not_called()

            after = self._workspace_file_snapshot(workspace_root)

        self.assertEqual(after, before)

    def test_roi_transform_run_defaults_to_plan_without_runner_or_writer_calls(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            project_root = self._write_workspace(workspace_root)
            self._write_roi_configs(project_root)
            before = sorted(path.relative_to(workspace_root).as_posix() for path in workspace_root.rglob("*"))

            with (
                mock.patch("research_platform.neuro.roi_transforms._default_transform_runner") as runner,
                mock.patch("research_platform.neuro.roi_transforms._create_planned_parent_directories") as mkdirs,
                mock.patch("research_platform.neuro.roi_transforms._write_planned_execution_json") as writer,
            ):
                code, output = self._run_cli(
                    ["analysis", "roi", "transform", "run", "modelA_t1w", "--project", "project-default"],
                    workspace_root=workspace_root,
                )

            after = sorted(path.relative_to(workspace_root).as_posix() for path in workspace_root.rglob("*"))

        payload = json.loads(output)
        self.assertEqual(code, 0)
        self.assertEqual(payload["mode"], "plan")
        self.assertTrue(payload["plan_only"])
        runner.assert_not_called()
        mkdirs.assert_not_called()
        writer.assert_not_called()
        self.assertEqual(after, before)

    def test_roi_cli_surface_does_not_add_publish_or_runtime_flags(self) -> None:
        parser = _build_parser()
        analysis = self._subparser(parser, "analysis")
        roi = self._subparser(analysis, "roi")
        extraction = self._subparser(roi, "extraction")

        roi_subcommands = self._subparser_choices(roi)
        extraction_subcommands = self._subparser_choices(extraction)
        self.assertNotIn("publish", roi_subcommands)
        self.assertNotIn("publish", extraction_subcommands)

        forbidden_flags = {
            "--" + "runtime-" + "root",
            "--" + "publication-" + "root",
            "--" + "publish",
            "--" + "cleanup-" + "runtime",
        }
        for command in (self._subparser(roi, "init"), self._subparser(extraction, "init"), self._subparser(extraction, "run")):
            option_strings = {option for action in getattr(command, "_actions", []) for option in getattr(action, "option_strings", [])}
            self.assertFalse(forbidden_flags & option_strings)

    def test_roi_scaffold_templates_and_path_profiles_are_discoverable(self) -> None:
        from research_platform.neuro import roi_scaffold

        parser = _build_parser()
        analysis = self._subparser(parser, "analysis")
        roi = self._subparser(analysis, "roi")
        roi_init = self._subparser(roi, "init")
        extraction = self._subparser(roi, "extraction")
        extraction_init = self._subparser(extraction, "init")

        self.assertEqual(
            tuple(self._option_action(roi_init, "--template").choices),
            tuple(sorted(roi_scaffold.ROI_SET_TEMPLATES)),
        )
        self.assertEqual(
            tuple(self._option_action(extraction_init, "--template").choices),
            tuple(sorted(roi_scaffold.EXTRACTION_SET_TEMPLATES)),
        )
        expected_profiles = roi_scaffold.supported_path_profiles()
        self.assertEqual(tuple(self._option_action(roi_init, "--path-profile").choices), expected_profiles)
        self.assertEqual(tuple(self._option_action(extraction_init, "--path-profile").choices), expected_profiles)

        roi_help = roi_init.format_help()
        extraction_help = extraction_init.format_help()
        normalized_roi_help = " ".join(roi_help.split())
        normalized_extraction_help = " ".join(extraction_help.split())
        for template in sorted(roi_scaffold.ROI_SET_TEMPLATES):
            self.assertIn(template, roi_help)
        for template in sorted(roi_scaffold.EXTRACTION_SET_TEMPLATES):
            self.assertIn(template, extraction_help)
        self.assertIn("advanced scaffold overrides", roi_help)
        self.assertIn("advanced scaffold overrides", extraction_help)
        self.assertIn("requires FSL", normalized_roi_help)
        self.assertIn("deferred scaffolds", normalized_roi_help)
        self.assertIn("requires FSL", normalized_extraction_help)
        self.assertIn("Single-entity ROI templates require exactly one subject", normalized_roi_help)
        self.assertIn("loso_group_map accepts", normalized_roi_help)
        self.assertIn("inclusive ranges", normalized_roi_help)
        self.assertIn("generic_nifti requires exactly one subject", normalized_extraction_help)
        self.assertIn("fsl_featquery accepts", normalized_extraction_help)
        self.assertIn("inclusive ranges", normalized_extraction_help)
        self.assertIn("Advanced:", roi.format_help())

    def test_building_main_parser_does_not_import_roi_runtime(self) -> None:
        with mock.patch(
            "research_platform.core.cli._roi_scaffold",
            side_effect=AssertionError("parser construction must keep research-neuro lazy"),
        ):
            parser = _build_parser()

        self.assertIsNotNone(self._subparser(self._subparser(parser, "analysis"), "roi"))

    def test_invalid_roi_scaffold_templates_list_valid_choices_without_traceback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            self._write_workspace(workspace_root)
            cases = (
                (
                    [
                        "analysis",
                        "roi",
                        "init",
                        "example_rois",
                        "--project",
                        "project-default",
                        "--template",
                        "unknown_template",
                    ],
                    ("coordinate_sphere", "loso_group_map"),
                ),
                (
                    [
                        "analysis",
                        "roi",
                        "extraction",
                        "init",
                        "example_values",
                        "--project",
                        "project-default",
                        "--roi-set",
                        "example_rois",
                        "--template",
                        "unknown_template",
                    ],
                    ("generic_nifti", "fsl_featquery"),
                ),
            )
            for args, expected_choices in cases:
                with self.subTest(args=args):
                    code, message = self._run_cli_parse_error(args, workspace_root=workspace_root)
                    self.assertEqual(code, 2)
                    self.assertIn("invalid choice", message)
                    for choice in expected_choices:
                        self.assertIn(choice, message)
                    self.assertNotIn("Traceback", message)

    def test_every_advertised_scaffold_dry_run_is_valid_and_neutral(self) -> None:
        from research_platform.neuro import roi_scaffold

        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            self._write_workspace(workspace_root)
            before = self._workspace_file_snapshot(workspace_root)

            for template in roi_scaffold.supported_roi_set_templates():
                with self.subTest(kind="roi_set", template=template):
                    code, output = self._run_cli(
                        [
                            "analysis",
                            "roi",
                            "init",
                            f"example_{template}",
                            "--project",
                            "project-default",
                            "--template",
                            template,
                            "--dry-run",
                        ],
                        workspace_root=workspace_root,
                    )
                    payload = json.loads(output)
                    self.assertEqual(code, 0)
                    self.assertTrue(payload["valid"])
                    self.assertFalse(payload["written"])
                    self.assertNotIn("MemoryEncoding", payload["yaml"])
                    self.assertNotIn("MemoryRecognition", payload["yaml"])
                    self.assertNotIn("task: memory", payload["yaml"])

            for template in roi_scaffold.supported_extraction_set_templates():
                with self.subTest(kind="extraction_set", template=template):
                    code, output = self._run_cli(
                        [
                            "analysis",
                            "roi",
                            "extraction",
                            "init",
                            f"example_{template}",
                            "--project",
                            "project-default",
                            "--roi-set",
                            "example_rois",
                            "--template",
                            template,
                            "--dry-run",
                        ],
                        workspace_root=workspace_root,
                    )
                    payload = json.loads(output)
                    self.assertEqual(code, 0)
                    self.assertTrue(payload["valid"])
                    self.assertFalse(payload["written"])
                    self.assertNotIn("MemoryEncoding", payload["yaml"])
                    self.assertNotIn("MemoryRecognition", payload["yaml"])
                    self.assertNotIn("task: memory", payload["yaml"])

            self.assertEqual(self._workspace_file_snapshot(workspace_root), before)

    def test_roi_init_coordinate_sphere_writes_valid_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            project_root = self._write_workspace(workspace_root)

            code, output = self._run_cli(
                [
                    "analysis",
                    "roi",
                    "init",
                    "memory_rois",
                    "--project",
                    "project-default",
                    "--template",
                    "coordinate_sphere",
                ],
                workspace_root=workspace_root,
            )

            path = project_root / "config" / "analysis" / "roi_sets" / "memory_rois.yaml"
            payload = json.loads(output)
            document = load_yaml(path, resolve_env=False)
            self.assertEqual(code, 0)
            self.assertTrue(path.exists())
            self.assertTrue(payload["valid"])
            self.assertTrue(payload["written"])
            self.assertEqual(payload["template"], "coordinate_sphere")
            self.assertEqual(document["roi_set"]["name"], "memory_rois")
            self.assertEqual(
                payload["next_steps"][1:],
                [
                    "Validate with: rp analysis roi validate memory_rois --project project-default",
                    "Check execution readiness: rp analysis roi doctor memory_rois --project project-default",
                    "Review the build plan: rp analysis roi build memory_rois --project project-default",
                    (
                        "Execute after reviewing the plan: rp analysis roi build memory_rois "
                        "--project project-default --execute"
                    ),
                ],
            )

    def test_roi_init_loso_group_map_writes_valid_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            project_root = self._write_workspace(workspace_root)

            code, output = self._run_cli(
                [
                    "analysis",
                    "roi",
                    "init",
                    "loso_modelA",
                    "--project",
                    "project-default",
                    "--template",
                    "loso_group_map",
                ],
                workspace_root=workspace_root,
            )

            path = project_root / "config" / "analysis" / "roi_sets" / "loso_modelA.yaml"
            payload = json.loads(output)
            content = path.read_text(encoding="utf-8")
            document = load_yaml(path, resolve_env=False)
            self.assertEqual(code, 0)
            self.assertTrue(payload["valid"])
            self.assertNotIn("${ROI_", content)
            self.assertEqual(
                document["roi_set"]["outputs"],
                {
                    "root_ref": "dataset_derivatives_root",
                    "path": ".research-platform/roi-loso-flame1-runtime/loso_modelA",
                },
            )
            self.assertEqual(document["roi_set"]["runtime"]["existing_output"], "fail")
            self.assertEqual(
                document["roi_set"]["runtime"]["cleanup"],
                {"after_roi_build": "roi_runtime", "after_extraction": "none"},
            )
            self.assertEqual(document["roi_set"]["task"], "exampletask")
            self.assertEqual(document["roi_set"]["fixed_effects_inputs"]["root_ref"], "project_root")
            self.assertEqual(document["roi_set"]["group_mask"]["root_ref"], "project_root")
            self.assertEqual([roi["label"] for roi in document["roi_set"]["rois"]], ["SeedA", "SeedB"])
            self.assertEqual(document["roi_set"]["held_out_subjects"], ["sub-001", "sub-002"])
            self.assertTrue(document["roi_set"]["publication"]["enabled"])
            self.assertEqual(document["roi_set"]["publication"]["layout"], "loso_flame1_bidslike")
            self.assertEqual(
                document["roi_set"]["publication"]["root"],
                {"root_ref": "dataset_derivatives_root", "path": "roi-loso-flame1"},
            )
            self.assertNotIn("contrast_aliases", document["roi_set"]["publication"])

    def test_roi_init_manual_mask_writes_valid_yaml_with_named_root_input(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            project_root = self._write_workspace(workspace_root)

            code, output = self._run_cli(
                [
                    "analysis",
                    "roi",
                    "init",
                    "manual_rois",
                    "--project",
                    "project-default",
                    "--template",
                    "manual_mask",
                ],
                workspace_root=workspace_root,
            )

            path = project_root / "config" / "analysis" / "roi_sets" / "manual_rois.yaml"
            payload = json.loads(output)
            document = load_yaml(path, resolve_env=False)
            self.assertEqual(code, 0)
            self.assertTrue(payload["valid"])
            self.assertEqual(
                document["roi_set"]["rois"][0]["source"],
                {"root_ref": "project_root", "path": "inputs/roi/example_mask.nii.gz"},
            )
            self.assertEqual(document["roi_set"]["runtime"]["existing_output"], "fail")

    def test_extraction_init_generic_nifti_writes_valid_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            project_root = self._write_workspace(workspace_root)

            code, output = self._run_cli(
                [
                    "analysis",
                    "roi",
                    "extraction",
                    "init",
                    "modelA_values",
                    "--project",
                    "project-default",
                    "--roi-set",
                    "loso_modelA",
                    "--template",
                    "generic_nifti",
                ],
                workspace_root=workspace_root,
            )

            path = project_root / "config" / "analysis" / "extraction_sets" / "modelA_values.yaml"
            payload = json.loads(output)
            document = load_yaml(path, resolve_env=False)
            self.assertEqual(code, 0)
            self.assertTrue(payload["valid"])
            self.assertTrue(payload["written"])
            self.assertEqual(payload["roi_set"], "loso_modelA")
            self.assertEqual(document["extraction_set"]["targets"][0]["metrics"], ["mean", "median", "voxel_count"])
            self.assertEqual(
                payload["next_steps"][1:],
                [
                    "Validate with: rp analysis roi extraction validate modelA_values --project project-default",
                    (
                        "Check execution readiness: rp analysis roi extraction doctor modelA_values "
                        "--project project-default"
                    ),
                    (
                        "Review the extraction plan: rp analysis roi extraction run modelA_values "
                        "--project project-default"
                    ),
                    (
                        "Execute after reviewing the plan: rp analysis roi extraction run modelA_values "
                        "--project project-default --execute"
                    ),
                ],
            )

    def test_extraction_init_fsl_featquery_writes_valid_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            project_root = self._write_workspace(workspace_root)

            code, output = self._run_cli(
                [
                    "analysis",
                    "roi",
                    "extraction",
                    "init",
                    "modelA_featquery",
                    "--project",
                    "project-default",
                    "--roi-set",
                    "loso_modelA",
                    "--template",
                    "fsl_featquery",
                ],
                workspace_root=workspace_root,
            )

            path = project_root / "config" / "analysis" / "extraction_sets" / "modelA_featquery.yaml"
            payload = json.loads(output)
            content = path.read_text(encoding="utf-8")
            document = load_yaml(path, resolve_env=False)
            self.assertEqual(code, 0)
            self.assertTrue(payload["valid"])
            self.assertNotIn("percent_signal_change", content)
            self.assertNotIn("${ROI_DERIV_ROOT:-}", content)
            self.assertEqual(
                document["extraction_set"]["outputs"],
                {
                    "root_ref": "dataset_derivatives_root",
                    "path": ".research-platform/roi-loso-flame1-runtime/modelA_featquery",
                    "format": "tsv",
                },
            )
            self.assertEqual(document["extraction_set"]["runtime"]["existing_output"], "fail")
            self.assertEqual(document["extraction_set"]["runtime"]["cleanup"], {"after_extraction": "extraction_runtime"})
            self.assertEqual(document["extraction_set"]["roi_mask_source"], {"source": "roi_set_publication"})
            self.assertEqual(document["extraction_set"]["targets"][0]["metrics"], ["mean_cope", "roi_voxel_count"])
            self.assertNotIn("publication", document["extraction_set"])

    def test_extraction_init_fsl_featquery_inherits_loso_publication_when_roi_set_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            project_root = self._write_workspace(workspace_root)

            roi_code, _roi_output = self._run_cli(
                [
                    "analysis",
                    "roi",
                    "init",
                    "loso_modelA",
                    "--project",
                    "project-default",
                    "--template",
                    "loso_group_map",
                ],
                workspace_root=workspace_root,
            )
            code, output = self._run_cli(
                [
                    "analysis",
                    "roi",
                    "extraction",
                    "init",
                    "modelA_featquery",
                    "--project",
                    "project-default",
                    "--roi-set",
                    "loso_modelA",
                    "--template",
                    "fsl_featquery",
                ],
                workspace_root=workspace_root,
            )

            path = project_root / "config" / "analysis" / "extraction_sets" / "modelA_featquery.yaml"
            payload = json.loads(output)
            extraction_set = load_yaml(path, resolve_env=False)["extraction_set"]
            publication = extraction_set["publication"]
            self.assertEqual(roi_code, 0)
            self.assertEqual(code, 0)
            self.assertTrue(payload["valid"])
            self.assertEqual(
                extraction_set["outputs"],
                {
                    "root_ref": "dataset_derivatives_root",
                    "path": ".research-platform/roi-loso-flame1-runtime/modelA_featquery",
                    "format": "tsv",
                },
            )
            self.assertEqual(extraction_set["runtime"]["existing_output"], "fail")
            self.assertEqual(extraction_set["runtime"]["cleanup"], {"after_extraction": "extraction_runtime"})
            self.assertEqual(extraction_set["roi_mask_source"], {"source": "roi_set_publication"})
            self.assertTrue(publication["enabled"])
            self.assertEqual(publication["layout"], "loso_flame1_bidslike")
            self.assertEqual(publication["root"], {"root_ref": "dataset_derivatives_root", "path": "roi-loso-flame1"})

    def test_roi_init_generic_path_profile_matches_default_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            self._write_workspace(workspace_root)

            default_code, default_output = self._run_cli(
                [
                    "analysis",
                    "roi",
                    "init",
                    "loso_modelA",
                    "--project",
                    "project-default",
                    "--template",
                    "loso_group_map",
                    "--dry-run",
                ],
                workspace_root=workspace_root,
            )
            generic_code, generic_output = self._run_cli(
                [
                    "analysis",
                    "roi",
                    "init",
                    "loso_modelA",
                    "--project",
                    "project-default",
                    "--template",
                    "loso_group_map",
                    "--path-profile",
                    "generic",
                    "--dry-run",
                ],
                workspace_root=workspace_root,
            )

            self.assertEqual(default_code, 0)
            self.assertEqual(generic_code, 0)
            self.assertEqual(json.loads(generic_output)["yaml"], json.loads(default_output)["yaml"])

    def test_extraction_init_generic_path_profile_matches_default_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            self._write_workspace(workspace_root)

            default_code, default_output = self._run_cli(
                [
                    "analysis",
                    "roi",
                    "extraction",
                    "init",
                    "modelA_featquery",
                    "--project",
                    "project-default",
                    "--roi-set",
                    "loso_modelA",
                    "--template",
                    "fsl_featquery",
                    "--dry-run",
                ],
                workspace_root=workspace_root,
            )
            generic_code, generic_output = self._run_cli(
                [
                    "analysis",
                    "roi",
                    "extraction",
                    "init",
                    "modelA_featquery",
                    "--project",
                    "project-default",
                    "--roi-set",
                    "loso_modelA",
                    "--template",
                    "fsl_featquery",
                    "--path-profile",
                    "generic",
                    "--dry-run",
                ],
                workspace_root=workspace_root,
            )

            self.assertEqual(default_code, 0)
            self.assertEqual(generic_code, 0)
            self.assertEqual(json.loads(generic_output)["yaml"], json.loads(default_output)["yaml"])

    def test_roi_init_research_platform_fsl_ffx_path_profile_writes_valid_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            project_root = self._write_workspace(workspace_root)

            code, output = self._run_cli(
                [
                    "analysis",
                    "roi",
                    "init",
                    "loso_modelA",
                    "--project",
                    "project-default",
                    "--template",
                    "loso_group_map",
                    "--path-profile",
                    "research_platform_fsl_ffx",
                ],
                workspace_root=workspace_root,
            )

            path = project_root / "config" / "analysis" / "roi_sets" / "loso_modelA.yaml"
            payload = json.loads(output)
            content = path.read_text(encoding="utf-8")
            document = load_yaml(path, resolve_env=False)
            self.assertEqual(code, 0)
            self.assertTrue(payload["valid"])
            self.assertEqual(document["roi_set"]["missing_input_policy"], "warn")
            self.assertEqual(
                document["roi_set"]["outputs"],
                {
                    "root_ref": "dataset_derivatives_root",
                    "path": ".research-platform/roi-loso-flame1-runtime/loso_modelA",
                },
            )
            self.assertEqual(document["roi_set"]["runtime"]["existing_output"], "fail")
            self.assertEqual(
                document["roi_set"]["runtime"]["cleanup"],
                {"after_roi_build": "roi_runtime", "after_extraction": "none"},
            )
            self.assertTrue(document["roi_set"]["publication"]["enabled"])
            self.assertEqual(document["roi_set"]["publication"]["map_desc"], "{model}LOSOFlame1")
            self.assertEqual(document["roi_set"]["publication"]["mask_desc"], "{model}LOSOFlame1Sphere{sphere_radius_mm}mm")
            self.assertNotIn("${ROI_DERIV_ROOT:-}", content)
            self.assertNotIn("EncPairGtItem", content)
            self.assertIn("FFX.gfeat/cope{cope_number}.feat", content)
            self.assertIn("cope{cope_number}.gfeat/mask.nii.gz", content)

    def test_extraction_init_research_platform_fsl_ffx_path_profile_writes_valid_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            project_root = self._write_workspace(workspace_root)

            code, output = self._run_cli(
                [
                    "analysis",
                    "roi",
                    "extraction",
                    "init",
                    "modelA_featquery",
                    "--project",
                    "project-default",
                    "--roi-set",
                    "loso_modelA",
                    "--template",
                    "fsl_featquery",
                    "--path-profile",
                    "research_platform_fsl_ffx",
                ],
                workspace_root=workspace_root,
            )

            path = project_root / "config" / "analysis" / "extraction_sets" / "modelA_featquery.yaml"
            payload = json.loads(output)
            content = path.read_text(encoding="utf-8")
            target = load_yaml(path, resolve_env=False)["extraction_set"]["targets"][0]
            self.assertEqual(code, 0)
            self.assertTrue(payload["valid"])
            self.assertEqual(
                load_yaml(path, resolve_env=False)["extraction_set"]["outputs"],
                {
                    "root_ref": "dataset_derivatives_root",
                    "path": ".research-platform/roi-loso-flame1-runtime/modelA_featquery",
                    "format": "tsv",
                },
            )
            self.assertEqual(load_yaml(path, resolve_env=False)["extraction_set"]["runtime"]["existing_output"], "fail")
            self.assertEqual(
                load_yaml(path, resolve_env=False)["extraction_set"]["runtime"]["cleanup"],
                {"after_extraction": "extraction_runtime"},
            )
            self.assertEqual(
                load_yaml(path, resolve_env=False)["extraction_set"]["roi_mask_source"],
                {"source": "roi_set_publication"},
            )
            self.assertEqual(target["metrics"], ["mean_cope", "roi_voxel_count"])
            self.assertNotIn("percent_signal_change", content)
            self.assertNotIn("${ROI_DERIV_ROOT:-}", content)
            self.assertIn("FFX.gfeat/cope{cope}.feat", content)
            self.assertIn("value_image: stats/cope1.nii.gz", content)
            self.assertIn("layout: loso_flame1_bidslike", content)
            self.assertNotIn("EncPairGtItem", content)
            self.assertIn("featquery_output_name: fq_loso_{roi_label}_{source_contrast}_cope{cope}", content)

    def test_roi_init_invalid_path_profile_reports_parser_error_with_supported_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            self._write_workspace(workspace_root)

            code, message = self._run_cli_parse_error(
                [
                    "analysis",
                    "roi",
                    "init",
                    "loso_modelA",
                    "--project",
                    "project-default",
                    "--template",
                    "loso_group_map",
                    "--path-profile",
                    "site_local",
                ],
                workspace_root=workspace_root,
            )

            self.assertEqual(code, 2)
            self.assertIn("invalid choice", message)
            self.assertIn("generic", message)
            self.assertIn("research_platform_fsl_ffx", message)
            self.assertNotIn("Traceback", message)

    def test_roi_init_loso_group_map_override_flags_write_valid_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            project_root = self._write_workspace(workspace_root)

            code, output = self._run_cli(
                [
                    "analysis",
                    "roi",
                    "init",
                    "loso_modelA",
                    "--project",
                    "project-default",
                    "--template",
                    "loso_group_map",
                    "--subjects",
                    "sub-002:sub-004",
                    "--held-out-subjects",
                    "same-as-subjects",
                    "--session",
                    "ses-01",
                    "--task",
                    "memory",
                    "--direction",
                    "AP",
                    "--model",
                    "ModelA",
                    "--space",
                    "MNI152NLin6Asym",
                    "--resolution",
                    "2",
                    "--search-radius-mm",
                    "20",
                    "--sphere-radius-mm",
                    "6",
                    "--z-threshold",
                    "3.1",
                    "--min-voxels-warn",
                    "10",
                    "--min-voxels-fail",
                    "5",
                    "--contrast",
                    "pair_enc_hit_gt_item_enc_hit:1:PairEncHitGtItemEncHit",
                    "--roi",
                    "EncodingPrecuneus:pair_enc_hit_gt_item_enc_hit:-2,-58,64:PairEncHitGtItemEncHit",
                ],
                workspace_root=workspace_root,
            )

            path = project_root / "config" / "analysis" / "roi_sets" / "loso_modelA.yaml"
            payload = json.loads(output)
            document = load_yaml(path, resolve_env=False)
            roi_set = document["roi_set"]
            roi = roi_set["rois"][0]
            self.assertEqual(code, 0)
            self.assertTrue(payload["valid"])
            self.assertEqual(roi_set["subjects"], ["sub-002", "sub-003", "sub-004"])
            self.assertEqual(roi_set["held_out_subjects"], ["sub-002", "sub-003", "sub-004"])
            self.assertEqual(roi_set["min_group_n"], 2)
            self.assertEqual(roi_set["contrasts"], [{"id": "pair_enc_hit_gt_item_enc_hit", "cope_number": 1, "desc": "PairEncHitGtItemEncHit"}])
            self.assertEqual(roi["label"], "EncodingPrecuneus")
            self.assertEqual(roi["seed_coordinate"], [-2, -58, 64])
            self.assertEqual(roi["search_radius_mm"], 20)
            self.assertEqual(roi["min_voxels_fail"], 5)

    def test_roi_init_loso_group_map_subjects_without_held_out_infers_min_group_n(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            project_root = self._write_workspace(workspace_root)

            code, output = self._run_cli(
                [
                    "analysis",
                    "roi",
                    "init",
                    "loso_modelA",
                    "--project",
                    "project-default",
                    "--template",
                    "loso_group_map",
                    "--subjects",
                    "sub-002:sub-004",
                ],
                workspace_root=workspace_root,
            )

            path = project_root / "config" / "analysis" / "roi_sets" / "loso_modelA.yaml"
            payload = json.loads(output)
            roi_set = load_yaml(path, resolve_env=False)["roi_set"]
            self.assertEqual(code, 0)
            self.assertTrue(payload["valid"])
            self.assertEqual(roi_set["subjects"], ["sub-002", "sub-003", "sub-004"])
            self.assertEqual(roi_set["held_out_subjects"], ["sub-002", "sub-003", "sub-004"])
            self.assertEqual(roi_set["min_group_n"], 2)

    def test_roi_init_loso_group_map_twenty_eight_same_as_subjects_infers_min_group_n(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            project_root = self._write_workspace(workspace_root)

            code, output = self._run_cli(
                [
                    "analysis",
                    "roi",
                    "init",
                    "loso_modelA",
                    "--project",
                    "project-default",
                    "--template",
                    "loso_group_map",
                    "--subjects",
                    "sub-101:sub-103",
                    "--held-out-subjects",
                    "same-as-subjects",
                ],
                workspace_root=workspace_root,
            )

            path = project_root / "config" / "analysis" / "roi_sets" / "loso_modelA.yaml"
            payload = json.loads(output)
            roi_set = load_yaml(path, resolve_env=False)["roi_set"]
            self.assertEqual(code, 0)
            self.assertTrue(payload["valid"])
            self.assertEqual(len(roi_set["subjects"]), 3)
            self.assertEqual(roi_set["held_out_subjects"], roi_set["subjects"])
            self.assertEqual(roi_set["min_group_n"], 2)

    def test_roi_init_invalid_subject_override_reports_json_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            self._write_workspace(workspace_root)

            message = self._run_cli_exit(
                [
                    "analysis",
                    "roi",
                    "init",
                    "loso_modelA",
                    "--project",
                    "project-default",
                    "--template",
                    "loso_group_map",
                    "--subjects",
                    "sub-004:sub-002",
                ],
                workspace_root=workspace_root,
            )

            self.assertEqual(
                json.loads(message)["error"],
                "--subjects must be a comma-separated list or inclusive range like sub-101:sub-103.",
            )

    def test_extraction_init_fsl_featquery_override_flags_write_valid_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            project_root = self._write_workspace(workspace_root)

            code, output = self._run_cli(
                [
                    "analysis",
                    "roi",
                    "extraction",
                    "init",
                    "modelA_featquery",
                    "--project",
                    "project-default",
                    "--roi-set",
                    "loso_modelA",
                    "--template",
                    "fsl_featquery",
                    "--subjects",
                    "sub-002,sub-004:sub-005",
                    "--session",
                    "ses-01",
                    "--task",
                    "memory",
                    "--direction",
                    "AP",
                    "--model",
                    "ModelA",
                    "--space",
                    "MNI152NLin6Asym",
                    "--resolution",
                    "2",
                    "--metric",
                    "mean_cope",
                    "--metric",
                    "roi_voxel_count",
                    "--contrast",
                    "pair_enc_hit_gt_item_enc_hit:1:PairEncHitGtItemEncHit",
                    "--roi-label",
                    "EncodingPrecuneus",
                    "--roi-label",
                    "EncodingAngular",
                ],
                workspace_root=workspace_root,
            )

            path = project_root / "config" / "analysis" / "extraction_sets" / "modelA_featquery.yaml"
            payload = json.loads(output)
            document = load_yaml(path, resolve_env=False)
            extraction_set = document["extraction_set"]
            target = extraction_set["targets"][0]
            self.assertEqual(code, 0)
            self.assertTrue(payload["valid"])
            self.assertEqual(
                extraction_set["outputs"],
                {
                    "root_ref": "dataset_derivatives_root",
                    "path": ".research-platform/roi-loso-flame1-runtime/modelA_featquery",
                    "format": "tsv",
                },
            )
            self.assertEqual(extraction_set["runtime"]["existing_output"], "fail")
            self.assertEqual(extraction_set["subjects"], ["sub-002", "sub-004", "sub-005"])
            self.assertEqual(target["metrics"], ["mean_cope", "roi_voxel_count"])
            self.assertEqual(target["contrasts"], [{"id": "pair_enc_hit_gt_item_enc_hit", "cope": 1, "desc": "PairEncHitGtItemEncHit"}])
            self.assertEqual(target["roi_labels"], ["EncodingPrecuneus", "EncodingAngular"])

    def test_extraction_init_percent_signal_change_override_writes_psc_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            project_root = self._write_workspace(workspace_root)

            code, output = self._run_cli(
                [
                    "analysis",
                    "roi",
                    "extraction",
                    "init",
                    "modelA_featquery",
                    "--project",
                    "project-default",
                    "--roi-set",
                    "loso_modelA",
                    "--template",
                    "fsl_featquery",
                    "--metric",
                    "percent_signal_change",
                ],
                workspace_root=workspace_root,
            )

            path = project_root / "config" / "analysis" / "extraction_sets" / "modelA_featquery.yaml"
            payload = json.loads(output)
            content = path.read_text(encoding="utf-8")
            target = load_yaml(path, resolve_env=False)["extraction_set"]["targets"][0]
            self.assertEqual(code, 0)
            self.assertTrue(payload["valid"])
            self.assertEqual(target["metrics"], ["percent_signal_change"])
            self.assertEqual(target["featquery"], {"include_percent_signal_change": True})
            self.assertIn("- percent_signal_change", content)
            self.assertIn("include_percent_signal_change: true", content)

    def test_roi_init_dry_run_does_not_write_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            project_root = self._write_workspace(workspace_root)
            before = self._workspace_file_snapshot(workspace_root)

            code, output = self._run_cli(
                [
                    "analysis",
                    "roi",
                    "init",
                    "memory_rois",
                    "--project",
                    "project-default",
                    "--template",
                    "coordinate_sphere",
                    "--dry-run",
                ],
                workspace_root=workspace_root,
            )

            path = project_root / "config" / "analysis" / "roi_sets" / "memory_rois.yaml"
            payload = json.loads(output)
            after = self._workspace_file_snapshot(workspace_root)
            self.assertEqual(code, 0)
            self.assertFalse(path.exists())
            self.assertFalse(payload["written"])
            self.assertIn("roi_set:", payload["yaml"])
            self.assertIn("project/project-default/config/analysis/roi_sets/memory_rois.yaml", payload["path"])
            self.assertEqual(after, before)

    def test_extraction_init_dry_run_does_not_write_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            project_root = self._write_workspace(workspace_root)
            before = self._workspace_file_snapshot(workspace_root)

            code, output = self._run_cli(
                [
                    "analysis",
                    "roi",
                    "extraction",
                    "init",
                    "example_values",
                    "--project",
                    "project-default",
                    "--roi-set",
                    "example_rois",
                    "--template",
                    "generic_nifti",
                    "--dry-run",
                ],
                workspace_root=workspace_root,
            )

            path = project_root / "config" / "analysis" / "extraction_sets" / "example_values.yaml"
            payload = json.loads(output)
            after = self._workspace_file_snapshot(workspace_root)
            self.assertEqual(code, 0)
            self.assertFalse(path.exists())
            self.assertFalse(payload["written"])
            self.assertIn("extraction_set:", payload["yaml"])
            self.assertIn("project/project-default/config/analysis/extraction_sets/example_values.yaml", payload["path"])
            self.assertEqual(after, before)

    def test_single_entity_subject_overrides_dry_run_with_singular_identity_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            project_root = self._write_workspace(workspace_root)
            before = self._workspace_tree_snapshot(workspace_root)
            cases = (
                (
                    [
                        "analysis",
                        "roi",
                        "init",
                        "example_rois",
                        "--project",
                        "project-default",
                        "--template",
                        "coordinate_sphere",
                        "--subjects",
                        "sub-101",
                        "--dry-run",
                    ],
                    "roi_set",
                    project_root / "config" / "analysis" / "roi_sets" / "example_rois.yaml",
                ),
                (
                    [
                        "analysis",
                        "roi",
                        "extraction",
                        "init",
                        "example_values",
                        "--project",
                        "project-default",
                        "--roi-set",
                        "example_rois",
                        "--template",
                        "generic_nifti",
                        "--subjects",
                        "sub-101",
                        "--dry-run",
                    ],
                    "extraction_set",
                    project_root / "config" / "analysis" / "extraction_sets" / "example_values.yaml",
                ),
            )

            for args, payload_key, path in cases:
                with self.subTest(payload_key=payload_key):
                    code, output = self._run_cli(args, workspace_root=workspace_root)
                    result = json.loads(output)
                    document = parse_yaml(result["yaml"], resolve_env=False)
                    scaffold = document[payload_key]
                    self.assertEqual(code, 0)
                    self.assertFalse(result["written"])
                    self.assertEqual(scaffold["subject"], "sub-101")
                    self.assertNotIn("subjects", scaffold)
                    self.assertNotIn("sub-001", result["yaml"])
                    self.assertFalse(path.exists())

            self.assertEqual(self._workspace_tree_snapshot(workspace_root), before)

    def test_single_entity_subject_overrides_reject_multiple_values_before_writing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            project_root = self._write_workspace(workspace_root)
            cases = (
                (
                    [
                        "analysis",
                        "roi",
                        "init",
                        "example_rois_list",
                        "--project",
                        "project-default",
                        "--template",
                        "coordinate_sphere",
                        "--subjects",
                        "sub-101,sub-102",
                    ],
                    project_root / "config" / "analysis" / "roi_sets" / "example_rois_list.yaml",
                ),
                (
                    [
                        "analysis",
                        "roi",
                        "init",
                        "example_rois_range",
                        "--project",
                        "project-default",
                        "--template",
                        "coordinate_sphere",
                        "--subjects",
                        "sub-101:sub-102",
                        "--dry-run",
                    ],
                    project_root / "config" / "analysis" / "roi_sets" / "example_rois_range.yaml",
                ),
                (
                    [
                        "analysis",
                        "roi",
                        "extraction",
                        "init",
                        "example_values_list",
                        "--project",
                        "project-default",
                        "--roi-set",
                        "example_rois",
                        "--template",
                        "generic_nifti",
                        "--subjects",
                        "sub-101,sub-102",
                    ],
                    project_root / "config" / "analysis" / "extraction_sets" / "example_values_list.yaml",
                ),
                (
                    [
                        "analysis",
                        "roi",
                        "extraction",
                        "init",
                        "example_values_range",
                        "--project",
                        "project-default",
                        "--roi-set",
                        "example_rois",
                        "--template",
                        "generic_nifti",
                        "--subjects",
                        "sub-101:sub-102",
                        "--dry-run",
                    ],
                    project_root / "config" / "analysis" / "extraction_sets" / "example_values_range.yaml",
                ),
            )

            for args, path in cases:
                with self.subTest(name=args[3:6]):
                    before = self._workspace_tree_snapshot(workspace_root)
                    message = self._run_cli_exit(args, workspace_root=workspace_root)
                    error = json.loads(message)["error"]
                    self.assertIn("represents one configured subject", error)
                    self.assertIn("does not perform multi-subject or Cartesian expansion", error)
                    self.assertIn("supported only by loso_group_map ROI and fsl_featquery", error)
                    self.assertNotIn("Traceback", message)
                    self.assertFalse(path.exists())
                    self.assertEqual(self._workspace_tree_snapshot(workspace_root), before)

    def test_roi_init_refuses_to_overwrite_existing_file_without_force(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            project_root = self._write_workspace(workspace_root)
            path = project_root / "config" / "analysis" / "roi_sets" / "memory_rois.yaml"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("sentinel\n", encoding="utf-8")

            message = self._run_cli_exit(
                [
                    "analysis",
                    "roi",
                    "init",
                    "memory_rois",
                    "--project",
                    "project-default",
                    "--template",
                    "coordinate_sphere",
                ],
                workspace_root=workspace_root,
            )

            self.assertIn("already exists", message)
            self.assertEqual(path.read_text(encoding="utf-8"), "sentinel\n")

    def test_roi_init_force_overwrites_existing_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            project_root = self._write_workspace(workspace_root)
            path = project_root / "config" / "analysis" / "roi_sets" / "memory_rois.yaml"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("sentinel\n", encoding="utf-8")

            code, output = self._run_cli(
                [
                    "analysis",
                    "roi",
                    "init",
                    "memory_rois",
                    "--project",
                    "project-default",
                    "--template",
                    "coordinate_sphere",
                    "--force",
                ],
                workspace_root=workspace_root,
            )

            payload = json.loads(output)
            self.assertEqual(code, 0)
            self.assertTrue(payload["written"])
            self.assertNotEqual(path.read_text(encoding="utf-8"), "sentinel\n")
            self.assertEqual(load_yaml(path, resolve_env=False)["roi_set"]["name"], "memory_rois")

    def test_plan_only_roi_build_command_reports_generic_plan(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            project_root = self._write_workspace(workspace_root)
            self._write_phase3_images(project_root)
            self._write_coordinate_roi_set(project_root)
            before = self._workspace_tree_snapshot(workspace_root)

            with (
                mock.patch("research_platform.neuro.roi_execution.RoiRuntimeOutputTransaction") as transaction,
                mock.patch("research_platform.neuro.roi_execution._execute_build_action") as builder,
            ):
                code, output = self._run_cli(
                    ["analysis", "roi", "build", "coordinate", "--project", "project-default"],
                    workspace_root=workspace_root,
                )

            after = self._workspace_tree_snapshot(workspace_root)

            payload = json.loads(output)
            action = payload["actions"][0]
            self.assertEqual(code, 0)
            self.assertEqual(payload["mode"], "plan")
            self.assertFalse(payload["executed"])
            self.assertEqual(action["roi_label"], "SeedSphere")
            self.assertIn("label-SeedSphere_desc-CoordinateSphere_mask.nii.gz", action["mask_path"])
            self.assertFalse(Path(action["mask_path"]).exists())
            transaction.assert_not_called()
            builder.assert_not_called()
            self.assertEqual(after, before)

    @unittest.skipIf(nib is None or np is None, "numpy and nibabel are required to create the synthetic build input")
    def test_roi_build_execute_rejects_symlinked_workspace_output_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            project_root = self._write_workspace(workspace_root)
            self._write_phase3_images(project_root)
            self._write_coordinate_roi_set(project_root)
            outside = workspace_root / "outside-artifacts"
            outside.mkdir()
            linked_artifacts = workspace_root / "linked-artifacts"
            linked_artifacts.symlink_to(outside, target_is_directory=True)
            workspace = load_yaml(workspace_root / "WORKSPACE.yaml", resolve_env=False)
            workspace["paths"]["artifacts_root"] = "./linked-artifacts"
            write_yaml(workspace_root / "WORKSPACE.yaml", workspace)
            before = self._workspace_tree_snapshot(workspace_root)

            with mock.patch("research_platform.neuro.roi_execution._execute_build_action") as builder:
                message = self._run_cli_exit(
                    [
                        "analysis",
                        "roi",
                        "build",
                        "coordinate",
                        "--project",
                        "project-default",
                        "--execute",
                    ],
                    workspace_root=workspace_root,
                )

            self.assertIn("symbolic link", message)
            builder.assert_not_called()
            self.assertEqual(self._workspace_tree_snapshot(workspace_root), before)
            self.assertFalse(list(workspace_root.rglob(".roi-runtime-*")))

    def test_plan_only_loso_roi_build_reports_jobs_without_running_fsl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            project_root = self._write_workspace(workspace_root)
            self._write_loso_inputs(workspace_root, nifti_masks=False)
            self._write_loso_roi_set(project_root, two_rois=True)
            before = self._workspace_tree_snapshot(workspace_root)

            with (
                mock.patch("research_platform.neuro.roi_execution.RoiRuntimeOutputTransaction") as transaction,
                mock.patch("research_platform.neuro.fsl.flame.execute_flame1_command_plan") as runner,
            ):
                code, output = self._run_cli(
                    ["analysis", "roi", "build", "loso", "--project", "project-default"],
                    workspace_root=workspace_root,
                )

            after = self._workspace_tree_snapshot(workspace_root)

            payload = json.loads(output)
            actions = payload["actions"]
            self.assertEqual(code, 0)
            self.assertEqual(payload["mode"], "plan")
            self.assertFalse(payload["executed"])
            self.assertEqual(len(actions), 2)
            self.assertEqual(
                actions[0]["metadata"]["loso_group_job"]["cache_key"],
                actions[1]["metadata"]["loso_group_job"]["cache_key"],
            )
            self.assertIn("loso_groupmaps/loso/group/ses-01/func", actions[0]["metadata"]["loso_group_job"]["zstat_path"])
            self.assertFalse(Path(actions[0]["mask_path"]).exists())
            transaction.assert_not_called()
            runner.assert_not_called()
            self.assertEqual(after, before)

    def test_roi_doctor_reports_loso_expected_actions_and_missing_inputs_without_fsl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            project_root = self._write_workspace(workspace_root)
            self._write_loso_roi_set(project_root, two_rois=False)

            with mock.patch("research_platform.neuro.roi_doctor.shutil.which", return_value=None):
                code, output = self._run_cli(
                    ["analysis", "roi", "doctor", "loso", "--project", "project-default"],
                    workspace_root=workspace_root,
                )

            payload = json.loads(output)
            missing_kinds = {item["kind"] for item in payload["missing_inputs"]}
            self.assertEqual(code, 1)
            self.assertTrue(payload["valid"])
            self.assertTrue(payload["schema_valid"])
            self.assertFalse(payload["ready_for_execution"])
            self.assertEqual(payload["expected_actions"], 1)
            self.assertIn("fixed_effects_cope", missing_kinds)
            self.assertIn("fixed_effects_varcope", missing_kinds)
            self.assertIn("fixed_effects_mask", missing_kinds)
            self.assertFalse(payload["fsl_tools"]["tools"]["flameo"]["available"])
            self.assertFalse(payload["fsl_tools"]["tools"]["fslmerge"]["available"])

    def test_roi_doctor_cli_exit_tracks_readiness_not_schema_validity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            project_root = self._write_workspace(workspace_root)
            self._write_roi_configs(project_root)
            doctor = mock.Mock()
            doctor.doctor_roi_set.return_value = {
                "valid": True,
                "schema_valid": True,
                "ready_for_execution": False,
                "checks": [
                    {
                        "id": "input.reference_image",
                        "status": "fail",
                        "message": "Configure a readable reference image before execution.",
                    }
                ],
                "errors": [],
                "warnings": ["A required reference image is not ready."],
            }

            with mock.patch("research_platform.core.cli._roi_doctor", return_value=doctor):
                code, output = self._run_cli(
                    ["analysis", "roi", "doctor", "modelA", "--project", "project-default"],
                    workspace_root=workspace_root,
                )

            payload = json.loads(output)
            self.assertEqual(code, 1)
            self.assertTrue(payload["valid"])
            self.assertTrue(payload["schema_valid"])
            self.assertFalse(payload["ready_for_execution"])
            self.assertEqual(payload["checks"][0]["id"], "input.reference_image")

    def test_roi_doctor_checks_scaffolded_group_mask_paths_with_gfeat(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            project_root = self._write_workspace(workspace_root)
            self._run_cli(
                [
                    "analysis",
                    "roi",
                    "init",
                    "loso_scaffold",
                    "--project",
                    "project-default",
                    "--template",
                    "loso_group_map",
                ],
                workspace_root=workspace_root,
            )
            path = project_root / "config" / "analysis" / "roi_sets" / "loso_scaffold.yaml"
            content = path.read_text(encoding="utf-8")

            with mock.patch("research_platform.neuro.roi_doctor.shutil.which", return_value=None):
                code, output = self._run_cli(
                    ["analysis", "roi", "doctor", "loso_scaffold", "--project", "project-default"],
                    workspace_root=workspace_root,
                )

            payload = json.loads(output)
            group_mask_paths = [item["path"] for item in payload["checked_inputs"] if item["kind"] == "group_mask"]
            self.assertEqual(code, 1)
            self.assertTrue(payload["valid"])
            self.assertTrue(payload["schema_valid"])
            self.assertFalse(payload["ready_for_execution"])
            self.assertEqual(payload["errors"], [])
            self.assertNotIn("${ROI_", content)
            self.assertEqual(
                load_yaml(path, resolve_env=False)["roi_set"]["fixed_effects_inputs"]["root_ref"],
                "project_root",
            )
            self.assertTrue(group_mask_paths)
            self.assertTrue(all(".gfeat/mask.nii.gz" in path for path in group_mask_paths))

    def test_roi_validate_uses_raw_scaffold_paths_for_personal_path_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            project_root = self._write_workspace(workspace_root)
            self._run_cli(
                [
                    "analysis",
                    "roi",
                    "init",
                    "loso_scaffold",
                    "--project",
                    "project-default",
                    "--template",
                    "loso_group_map",
                    "--path-profile",
                    "research_platform_fsl_ffx",
                    "--contrast",
                    "MemoryEncoding:1:MemoryEncoding",
                    "--roi",
                    "MemoryEncodingSeed:MemoryEncoding:0,-52,26:MemoryEncoding",
                ],
                workspace_root=workspace_root,
            )
            path = project_root / "config" / "analysis" / "roi_sets" / "loso_scaffold.yaml"
            content = path.read_text(encoding="utf-8")
            feat_root, _deriv_root = self._synthetic_personal_env_roots()

            with mock.patch.dict(
                os.environ,
                {
                    "ROI_FEAT_ROOT": str(feat_root),
                },
                clear=False,
            ):
                code, output = self._run_cli(
                    ["analysis", "roi", "validate", "loso_scaffold", "--project", "project-default"],
                    workspace_root=workspace_root,
                )

            payload = json.loads(output)
            self.assertEqual(code, 0)
            self.assertTrue(payload["valid"])
            self.assertEqual(payload["errors"], [])
            self.assertIn("${ROI_FEAT_ROOT:-}", content)
            self.assertNotIn("${ROI_DERIV_ROOT:-}", content)

    def test_plan_only_loso_roi_build_uses_hidden_runtime_root_without_fsl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            project_root = self._write_workspace(workspace_root)
            self._run_cli(
                [
                    "analysis",
                    "roi",
                    "init",
                    "loso_scaffold",
                    "--project",
                    "project-default",
                    "--template",
                    "loso_group_map",
                    "--path-profile",
                    "research_platform_fsl_ffx",
                    "--subjects",
                    "sub-001:sub-002",
                    "--held-out-subjects",
                    "same-as-subjects",
                    "--contrast",
                    "MemoryEncoding:1:MemoryEncoding",
                    "--roi",
                    "MemoryEncodingSeed:MemoryEncoding:0,-52,26:MemoryEncoding",
                ],
                workspace_root=workspace_root,
            )
            feat_root, _deriv_root = self._synthetic_personal_env_roots()
            runtime_root = project_root / "derivatives" / ".research-platform" / "roi-loso-flame1-runtime"
            published_root = project_root / "derivatives" / "roi-loso-flame1"
            exists_patch, stat_patch = self._existing_runtime_path_mocks(
                feat_root,
                project_root / "sub-001",
                project_root / "sub-002",
            )

            with mock.patch.dict(
                os.environ,
                {
                    "ROI_FEAT_ROOT": str(feat_root),
                },
                clear=False,
            ):
                with exists_patch, stat_patch:
                    code, output = self._run_cli(
                        ["analysis", "roi", "build", "loso_scaffold", "--project", "project-default"],
                        workspace_root=workspace_root,
                    )

            payload = json.loads(output)
            self.assertEqual(code, 0)
            self.assertEqual(payload["mode"], "plan")
            self.assertFalse(payload["executed"])
            self.assertEqual(len(payload["actions"]), 2)
            self.assertIn(str(feat_root), payload["actions"][0]["metadata"]["loso_group_job"]["group_mask_path"])
            self.assertIn(str(runtime_root), payload["actions"][0]["mask_path"])
            self.assertIn(str(runtime_root), payload["actions"][0]["metadata"]["loso_group_job"]["work_dir"])
            self.assertNotIn(str(published_root), payload["actions"][0]["mask_path"])

    def test_plan_only_new_featquery_extraction_uses_published_masks_and_runtime_tables(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            project_root = self._write_workspace(workspace_root)
            self._run_cli(
                [
                    "analysis",
                    "roi",
                    "init",
                    "loso_scaffold",
                    "--project",
                    "project-default",
                    "--template",
                    "loso_group_map",
                    "--path-profile",
                    "research_platform_fsl_ffx",
                    "--subjects",
                    "sub-001:sub-002",
                    "--held-out-subjects",
                    "same-as-subjects",
                    "--contrast",
                    "MemoryEncoding:1:MemoryEncoding",
                    "--roi",
                    "MemoryEncodingSeed:MemoryEncoding:0,-52,26:MemoryEncoding",
                ],
                workspace_root=workspace_root,
            )
            self._run_cli(
                [
                    "analysis",
                    "roi",
                    "extraction",
                    "init",
                    "loso_featquery",
                    "--project",
                    "project-default",
                    "--roi-set",
                    "loso_scaffold",
                    "--template",
                    "fsl_featquery",
                    "--path-profile",
                    "research_platform_fsl_ffx",
                    "--subjects",
                    "sub-001",
                    "--contrast",
                    "MemoryEncoding:1:MemoryEncoding",
                    "--roi-label",
                    "MemoryEncodingSeed",
                ],
                workspace_root=workspace_root,
            )
            extraction_path = project_root / "config" / "analysis" / "extraction_sets" / "loso_featquery.yaml"
            extraction_set = load_yaml(extraction_path, resolve_env=False)["extraction_set"]
            feat_root, _deriv_root = self._synthetic_personal_env_roots()
            runtime_root = project_root / "derivatives" / ".research-platform" / "roi-loso-flame1-runtime"
            published_root = project_root / "derivatives" / "roi-loso-flame1"
            exists_patch, stat_patch = self._existing_runtime_path_mocks(
                feat_root,
                project_root / "sub-001",
                project_root / "sub-002",
            )

            with mock.patch.dict(os.environ, {"ROI_FEAT_ROOT": str(feat_root)}, clear=False):
                with exists_patch, stat_patch:
                    code, output = self._run_cli(
                        ["analysis", "roi", "extraction", "run", "loso_featquery", "--project", "project-default"],
                        workspace_root=workspace_root,
                    )

            payload = json.loads(output)
            action = payload["actions"][0]
            self.assertEqual(code, 0)
            self.assertEqual(payload["mode"], "plan")
            self.assertFalse(payload["executed"])
            self.assertEqual(
                extraction_set["outputs"],
                {
                    "root_ref": "dataset_derivatives_root",
                    "path": ".research-platform/roi-loso-flame1-runtime/loso_featquery",
                    "format": "tsv",
                },
            )
            self.assertEqual(extraction_set["publication"]["root"], {"root_ref": "dataset_derivatives_root", "path": "roi-loso-flame1"})
            self.assertEqual(extraction_set["roi_mask_source"], {"source": "roi_set_publication"})
            self.assertIn(str(feat_root), action["metadata"]["feat_dir"])
            self.assertIn(str(published_root / "masks"), action["mask_path"])
            self.assertIn(str(runtime_root), action["table_path"])
            self.assertNotIn(str(runtime_root / "rois"), action["mask_path"])
            self.assertNotIn(str(published_root), action["table_path"])

    def test_roi_validate_rejects_literal_personal_absolute_path_in_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            project_root = self._write_workspace(workspace_root)
            personal_root = self._synthetic_personal_env_roots()[1]
            write_yaml(
                project_root / "config" / "analysis" / "roi_sets" / "literal_personal.yaml",
                {
                    "roi_set": {
                        "name": "literal_personal",
                        "outputs": {"root": str(personal_root)},
                        "rois": [
                            {
                                "label": "SeedSphere",
                                "family": "coordinate_sphere",
                                "coordinate": [0, -52, 26],
                                "radius_mm": 6,
                            }
                        ],
                    }
                },
            )

            code, output = self._run_cli(
                ["analysis", "roi", "validate", "literal_personal", "--project", "project-default"],
                workspace_root=workspace_root,
            )

            payload = json.loads(output)
            self.assertEqual(code, 1)
            self.assertFalse(payload["valid"])
            self.assertTrue(any("roi_set.outputs.root contains a personal absolute path" in error for error in payload["errors"]))

    @unittest.skipIf(nib is None or np is None, "numpy and nibabel are required for Phase 3 ROI CLI execution tests")
    def test_execute_coordinate_sphere_build_writes_bids_like_mask_and_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            project_root = self._write_workspace(workspace_root)
            self._write_phase3_images(project_root)
            self._write_coordinate_roi_set(project_root)

            code, output = self._run_cli(
                ["analysis", "roi", "build", "coordinate", "--project", "project-default", "--execute"],
                workspace_root=workspace_root,
            )

            payload = json.loads(output)
            action = payload["actions"][0]
            mask_path = Path(action["mask_path"])
            sidecar_path = Path(action["sidecar_path"])
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
            self.assertEqual(code, 0)
            self.assertEqual(payload["mode"], "execute")
            self.assertTrue(mask_path.exists())
            self.assertTrue(sidecar_path.exists())
            self.assertIn("roi/sub-001/ses-01/func", str(mask_path))
            self.assertEqual(sidecar["roi_family"], "coordinate_sphere")
            self.assertEqual(sidecar["roi_label"], "SeedSphere")
            self.assertEqual(action["result"]["voxel_count"], 7)

    @unittest.skipIf(nib is None or np is None, "numpy and nibabel are required for Phase 3 ROI CLI execution tests")
    def test_execute_manual_mask_build_writes_copied_mask(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            project_root = self._write_workspace(workspace_root)
            self._write_phase3_images(project_root)
            self._write_manual_roi_set(project_root)

            code, output = self._run_cli(
                ["analysis", "roi", "build", "manual", "--project", "project-default", "--execute"],
                workspace_root=workspace_root,
            )

            payload = json.loads(output)
            action = payload["actions"][0]
            mask_data = nib.load(action["mask_path"]).get_fdata()
            self.assertEqual(code, 0)
            self.assertEqual(int(np.count_nonzero(mask_data)), 1)
            self.assertIn("label-ManualMask_desc-CuratedMask_mask.nii.gz", action["mask_path"])

    @unittest.skipIf(nib is None or np is None, "numpy and nibabel are required for Phase 3 ROI CLI execution tests")
    def test_execute_functional_threshold_map_build_thresholded_peak(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            project_root = self._write_workspace(workspace_root)
            self._write_phase3_images(project_root)
            self._write_functional_roi_set(project_root, name="functional", threshold=5.0, fallback=False)

            code, output = self._run_cli(
                ["analysis", "roi", "build", "functional", "--project", "project-default", "--execute"],
                workspace_root=workspace_root,
            )

            payload = json.loads(output)
            action = payload["actions"][0]
            sidecar = json.loads(Path(action["sidecar_path"]).read_text(encoding="utf-8"))
            self.assertEqual(code, 0)
            self.assertEqual(action["result"]["peak"]["fallback_status"], "thresholded")
            self.assertEqual(sidecar["selected_peak_stat"], 5.5)

    @unittest.skipIf(nib is None or np is None, "numpy and nibabel are required for Phase 3 ROI CLI execution tests")
    def test_execute_functional_threshold_map_build_below_threshold_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            project_root = self._write_workspace(workspace_root)
            self._write_phase3_images(project_root)
            self._write_functional_roi_set(project_root, name="functional_fallback", threshold=10.0, fallback=True)

            code, output = self._run_cli(
                [
                    "analysis",
                    "roi",
                    "build",
                    "functional_fallback",
                    "--project",
                    "project-default",
                    "--execute",
                ],
                workspace_root=workspace_root,
            )

            payload = json.loads(output)
            action = payload["actions"][0]
            sidecar = json.loads(Path(action["sidecar_path"]).read_text(encoding="utf-8"))
            self.assertEqual(code, 0)
            self.assertEqual(action["result"]["peak"]["fallback_status"], "below_threshold_fallback")
            self.assertEqual(sidecar["thresholded_voxel_count"], 0)

    @unittest.skipIf(nib is None or np is None, "numpy and nibabel are required for Phase 4 LOSO ROI CLI execution tests")
    def test_execute_loso_roi_build_uses_mocked_fsl_once_for_shared_contrast(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            project_root = self._write_workspace(workspace_root)
            self._write_loso_inputs(workspace_root, nifti_masks=True)
            self._write_loso_roi_set(project_root, two_rois=True)

            def fake_execute(plan: object) -> Path:
                data = np.zeros((5, 5, 5), dtype=float)
                data[2, 2, 2] = 5.5
                nib.save(nib.Nifti1Image(data, np.eye(4)), plan.output_zstat_path)
                return plan.output_zstat_path

            def fake_find_fsl_tool(tool: str, **_: object) -> str | None:
                if tool in {"fslmerge", "flameo"}:
                    return f"/mock/fsl/bin/{tool}"
                return None

            with (
                mock.patch(
                    "research_platform.neuro.roi_execution.shutil.which",
                    side_effect=fake_find_fsl_tool,
                ) as find_fsl_tool,
                mock.patch(
                    "research_platform.neuro.fsl.flame.execute_flame1_command_plan",
                    side_effect=fake_execute,
                ) as execute,
            ):
                code, output = self._run_cli(
                    ["analysis", "roi", "build", "loso", "--project", "project-default", "--execute"],
                    workspace_root=workspace_root,
                )

            payload = json.loads(output)
            actions = payload["actions"]
            sidecar = json.loads(Path(actions[0]["sidecar_path"]).read_text(encoding="utf-8"))
            group_sidecar = json.loads(Path(actions[0]["metadata"]["loso_group_job"]["sidecar_path"]).read_text(encoding="utf-8"))
            command_plan = group_sidecar["command_plan"]
            self.assertEqual(code, 0)
            self.assertEqual(
                find_fsl_tool.call_args_list,
                [mock.call("fslmerge"), mock.call("flameo")],
            )
            self.assertEqual(execute.call_count, 1)
            self.assertEqual(len(actions), 2)
            self.assertTrue(Path(actions[0]["mask_path"]).exists())
            self.assertEqual(sidecar["roi_family"], "loso_group_map")
            self.assertEqual(sidecar["group_map_cache_status"], "computed")
            self.assertTrue(sidecar["group_map_path"].startswith("${ROI_DERIV_ROOT:-}/loso_groupmaps/"))
            self.assertTrue(group_sidecar["zstat_path"].startswith("${ROI_DERIV_ROOT:-}/loso_groupmaps/"))
            self.assertTrue(group_sidecar["group_mask_path"].startswith("${ROI_FEAT_ROOT:-}/group/"))
            self.assertTrue(command_plan["commands"][0][2].startswith("${ROI_DERIV_ROOT:-}/.cache/loso_groupmaps/"))
            self.assertTrue(command_plan["commands"][0][3].startswith("${ROI_FEAT_ROOT:-}/sub-002/"))
            self.assertTrue(command_plan["commands"][1][2].startswith("${ROI_DERIV_ROOT:-}/.cache/loso_groupmaps/"))
            self.assertTrue(command_plan["commands"][1][3].startswith("${ROI_FEAT_ROOT:-}/sub-002/"))
            self.assertTrue(any(part.startswith("--mask=${ROI_FEAT_ROOT:-}/group/") for part in command_plan["commands"][2]))
            self.assertTrue(any(part.startswith("--ld=${ROI_DERIV_ROOT:-}/.cache/") for part in command_plan["commands"][2]))
            self.assertNotIn(str(workspace_root.resolve()), json.dumps(group_sidecar))

    def test_plan_only_generic_extraction_command_reports_table_plan(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            project_root = self._write_workspace(workspace_root)
            self._write_phase3_images(project_root)
            self._write_coordinate_roi_set(project_root)
            self._write_extraction_set(project_root)
            before = self._workspace_tree_snapshot(workspace_root)

            with (
                mock.patch("research_platform.neuro.roi_execution.RoiRuntimeOutputTransaction") as transaction,
                mock.patch("research_platform.neuro.roi_execution._write_extraction_summary_tables") as writer,
                mock.patch("research_platform.neuro.roi_execution._load_nifti_image") as loader,
            ):
                code, output = self._run_cli(
                    ["analysis", "roi", "extraction", "run", "generic_values", "--project", "project-default"],
                    workspace_root=workspace_root,
                )

            after = self._workspace_tree_snapshot(workspace_root)

            payload = json.loads(output)
            action = payload["actions"][0]
            self.assertEqual(code, 0)
            self.assertEqual(payload["mode"], "plan")
            self.assertFalse(payload["executed"])
            self.assertIn("desc-GenericValues_roiextract.tsv", action["table_path"])
            self.assertFalse(Path(action["table_path"]).exists())
            transaction.assert_not_called()
            writer.assert_not_called()
            loader.assert_not_called()
            self.assertEqual(after, before)

    def test_plan_only_fsl_featquery_extraction_reports_command_without_running_fsl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            project_root = self._write_workspace(workspace_root)
            self._write_featquery_inputs(workspace_root, project_root)
            self._write_featquery_extraction_set(project_root)
            before = self._workspace_tree_snapshot(workspace_root)

            with (
                mock.patch("research_platform.neuro.roi_execution.RoiRuntimeOutputTransaction") as transaction,
                mock.patch("research_platform.neuro.fsl.featquery.execute_featquery_command_plan") as runner,
                mock.patch("research_platform.neuro.roi_execution._write_extraction_summary_tables") as writer,
            ):
                code, output = self._run_cli(
                    ["analysis", "roi", "extraction", "run", "featquery_values", "--project", "project-default"],
                    workspace_root=workspace_root,
                )

            after = self._workspace_tree_snapshot(workspace_root)

            payload = json.loads(output)
            action = payload["actions"][0]
            self.assertEqual(code, 0)
            self.assertEqual(payload["mode"], "plan")
            self.assertEqual(action["backend"], "fsl_featquery")
            self.assertEqual(action["metadata"]["command"][0], "featquery")
            self.assertNotIn("-p", action["metadata"]["command"])
            self.assertEqual(action["metadata"]["command"][-1], action["mask_path"])
            self.assertEqual(action["metadata"]["command"][-2], "fq_SeedA_CondA_cope1")
            self.assertIn("roi_extract/featquery_values/group/ses-01", action["table_path"])
            self.assertFalse(Path(action["table_path"]).exists())
            transaction.assert_not_called()
            runner.assert_not_called()
            writer.assert_not_called()
            self.assertEqual(after, before)

    def test_extraction_doctor_reports_actions_and_safe_default_featquery_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            project_root = self._write_workspace(workspace_root)
            self._write_featquery_inputs(workspace_root, project_root)
            self._write_featquery_extraction_set(project_root)

            code, output = self._run_cli(
                ["analysis", "roi", "extraction", "doctor", "featquery_values", "--project", "project-default"],
                workspace_root=workspace_root,
            )

            payload = json.loads(output)
            command_check = payload["command_checks"][0]
            self.assertEqual(code, 1)
            self.assertTrue(payload["valid"])
            self.assertTrue(payload["schema_valid"])
            self.assertFalse(payload["ready_for_execution"])
            self.assertEqual(payload["expected_actions"], 1)
            self.assertFalse(command_check["has_extra_one_before_roi_mask"])
            self.assertFalse(command_check["includes_percent_signal_change_flag"])
            self.assertNotIn("-p", command_check["command"])
            self.assertEqual(command_check["command"][-1], command_check["roi_mask_path"])

    def test_extraction_validate_uses_raw_env_placeholder_paths_for_personal_path_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            project_root = self._write_workspace(workspace_root)
            self._write_env_placeholder_featquery_extraction_set(project_root)
            feat_root, deriv_root = self._synthetic_personal_env_roots()

            with mock.patch.dict(
                os.environ,
                {
                    "ROI_FEAT_ROOT": str(feat_root),
                    "ROI_DERIV_ROOT": str(deriv_root),
                },
                clear=False,
            ):
                code, output = self._run_cli(
                    ["analysis", "roi", "extraction", "validate", "env_featquery_values", "--project", "project-default"],
                    workspace_root=workspace_root,
                )

            payload = json.loads(output)
            self.assertEqual(code, 0)
            self.assertTrue(payload["valid"])
            self.assertEqual(payload["errors"], [])

    def test_extraction_doctor_accepts_env_placeholder_paths_but_is_not_ready_without_fsl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            project_root = self._write_workspace(workspace_root)
            self._write_env_placeholder_featquery_extraction_set(project_root)
            feat_root, deriv_root = self._synthetic_personal_env_roots()
            exists_patch, stat_patch = self._existing_runtime_path_mocks(feat_root, deriv_root)

            with mock.patch.dict(
                os.environ,
                {
                    "ROI_FEAT_ROOT": str(feat_root),
                    "ROI_DERIV_ROOT": str(deriv_root),
                },
                clear=False,
            ):
                with exists_patch, stat_patch:
                    code, output = self._run_cli(
                        ["analysis", "roi", "extraction", "doctor", "env_featquery_values", "--project", "project-default"],
                        workspace_root=workspace_root,
                    )

            payload = json.loads(output)
            feat_dir_checks = [item for item in payload["checked_inputs"] if item["kind"] == "feat_dir"]
            self.assertEqual(code, 1)
            self.assertTrue(payload["valid"])
            self.assertTrue(payload["schema_valid"])
            self.assertFalse(payload["ready_for_execution"])
            failed_checks = [check for check in payload["checks"] if check["status"] != "ok"]
            self.assertTrue(failed_checks)
            self.assertTrue(all(check.get("action") for check in failed_checks))
            self.assertTrue(feat_dir_checks)
            self.assertTrue(all(str(feat_root) in item["path"] for item in feat_dir_checks))

    def test_plan_only_extraction_allows_env_placeholder_personal_runtime_paths_without_fsl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            project_root = self._write_workspace(workspace_root)
            self._write_env_placeholder_featquery_extraction_set(project_root)
            feat_root, deriv_root = self._synthetic_personal_env_roots()
            exists_patch, stat_patch = self._existing_runtime_path_mocks(feat_root, deriv_root)

            with mock.patch.dict(
                os.environ,
                {
                    "ROI_FEAT_ROOT": str(feat_root),
                    "ROI_DERIV_ROOT": str(deriv_root),
                },
                clear=False,
            ):
                with exists_patch, stat_patch:
                    code, output = self._run_cli(
                        ["analysis", "roi", "extraction", "run", "env_featquery_values", "--project", "project-default"],
                        workspace_root=workspace_root,
                    )

            payload = json.loads(output)
            action = payload["actions"][0]
            self.assertEqual(code, 0)
            self.assertEqual(payload["mode"], "plan")
            self.assertFalse(payload["executed"])
            self.assertIn(str(feat_root), action["metadata"]["feat_dir"])
            self.assertIn(str(deriv_root), action["table_path"])

    def test_extraction_doctor_accepts_referenced_roi_env_paths_but_is_not_ready_without_fsl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            project_root = self._write_workspace(workspace_root)
            self._write_env_placeholder_loso_roi_set(project_root)
            self._write_env_roi_set_ref_featquery_extraction_set(project_root)
            feat_root, deriv_root = self._synthetic_personal_env_roots()
            exists_patch, stat_patch = self._existing_runtime_path_mocks(feat_root, deriv_root)

            with mock.patch.dict(
                os.environ,
                {
                    "ROI_FEAT_ROOT": str(feat_root),
                    "ROI_DERIV_ROOT": str(deriv_root),
                },
                clear=False,
            ):
                with exists_patch, stat_patch:
                    code, output = self._run_cli(
                        ["analysis", "roi", "extraction", "doctor", "env_loso_featquery", "--project", "project-default"],
                        workspace_root=workspace_root,
                    )

            payload = json.loads(output)
            roi_mask_checks = [item for item in payload["checked_inputs"] if item["kind"] == "roi_mask"]
            self.assertEqual(code, 1)
            self.assertTrue(payload["valid"])
            self.assertTrue(payload["schema_valid"])
            self.assertFalse(payload["ready_for_execution"])
            failed_checks = [check for check in payload["checks"] if check["status"] != "ok"]
            self.assertTrue(failed_checks)
            self.assertTrue(all(check.get("action") for check in failed_checks))
            self.assertEqual(payload["roi_set"], "env_loso")
            self.assertEqual(payload["expected_actions"], 1)
            self.assertTrue(roi_mask_checks)
            self.assertTrue(all(str(deriv_root) in item["path"] for item in roi_mask_checks))

    def test_plan_only_extraction_allows_referenced_roi_set_env_placeholder_personal_runtime_paths_without_fsl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            project_root = self._write_workspace(workspace_root)
            self._write_env_placeholder_loso_roi_set(project_root)
            self._write_env_roi_set_ref_featquery_extraction_set(project_root)
            feat_root, deriv_root = self._synthetic_personal_env_roots()
            exists_patch, stat_patch = self._existing_runtime_path_mocks(feat_root, deriv_root)

            with mock.patch.dict(
                os.environ,
                {
                    "ROI_FEAT_ROOT": str(feat_root),
                    "ROI_DERIV_ROOT": str(deriv_root),
                },
                clear=False,
            ):
                with exists_patch, stat_patch:
                    code, output = self._run_cli(
                        ["analysis", "roi", "extraction", "run", "env_loso_featquery", "--project", "project-default"],
                        workspace_root=workspace_root,
                    )

            payload = json.loads(output)
            action = payload["actions"][0]
            self.assertEqual(code, 0)
            self.assertEqual(payload["mode"], "plan")
            self.assertFalse(payload["executed"])
            self.assertEqual(payload["roi_set"], "env_loso")
            self.assertIn(str(feat_root), action["metadata"]["feat_dir"])
            self.assertIn(str(deriv_root), action["mask_path"])
            self.assertIn(str(deriv_root), action["table_path"])

    def test_extraction_doctor_rejects_literal_personal_absolute_path_in_referenced_roi_set(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            project_root = self._write_workspace(workspace_root)
            self._write_env_placeholder_loso_roi_set(project_root, literal_output_root=True)
            self._write_env_roi_set_ref_featquery_extraction_set(project_root)
            feat_root, deriv_root = self._synthetic_personal_env_roots()

            with mock.patch.dict(
                os.environ,
                {
                    "ROI_FEAT_ROOT": str(feat_root),
                    "ROI_DERIV_ROOT": str(deriv_root),
                },
                clear=False,
            ):
                code, output = self._run_cli(
                    ["analysis", "roi", "extraction", "doctor", "env_loso_featquery", "--project", "project-default"],
                    workspace_root=workspace_root,
                )

            payload = json.loads(output)
            self.assertEqual(code, 1)
            self.assertFalse(payload["valid"])
            self.assertTrue(
                any("referenced ROI set: roi_set.outputs.root contains a personal absolute path" in error for error in payload["errors"])
            )

    def test_extraction_validate_rejects_literal_personal_absolute_path_in_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            project_root = self._write_workspace(workspace_root)
            self._write_env_placeholder_featquery_extraction_set(project_root)
            path = project_root / "config" / "analysis" / "extraction_sets" / "env_featquery_values.yaml"
            document = load_yaml(path, resolve_env=False)
            extraction_set = document["extraction_set"]
            extraction_set["outputs"] = {"root": str(self._synthetic_personal_env_roots()[1])}
            write_yaml(path, document)

            code, output = self._run_cli(
                ["analysis", "roi", "extraction", "validate", "env_featquery_values", "--project", "project-default"],
                workspace_root=workspace_root,
            )

            payload = json.loads(output)
            self.assertEqual(code, 1)
            self.assertFalse(payload["valid"])
            self.assertTrue(
                any("extraction_set.outputs.root contains a personal absolute path" in error for error in payload["errors"])
            )

    @unittest.skipIf(nib is None or np is None, "numpy and nibabel are required for Phase 3 ROI CLI execution tests")
    def test_execute_generic_nifti_extraction_writes_summary_table(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            project_root = self._write_workspace(workspace_root)
            self._write_phase3_images(project_root)
            self._write_coordinate_roi_set(project_root)
            self._write_extraction_set(project_root)

            self._run_cli(
                ["analysis", "roi", "build", "coordinate", "--project", "project-default", "--execute"],
                workspace_root=workspace_root,
            )
            code, output = self._run_cli(
                [
                    "analysis",
                    "roi",
                    "extraction",
                    "run",
                    "generic_values",
                    "--project",
                    "project-default",
                    "--execute",
                ],
                workspace_root=workspace_root,
            )

            payload = json.loads(output)
            table_path = Path(payload["tables"][0])
            lines = table_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(code, 0)
            self.assertTrue(table_path.exists())
            self.assertIn("mean", lines[0])
            self.assertIn("valid_voxel_count", lines[0])
            self.assertIn("7", lines[1])

    def test_missing_roi_lifecycle_configs_are_clear_and_actionable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            self._write_workspace(workspace_root)

            cases = (
                (
                    ["analysis", "roi", "build", "missing", "--project", "project-default"],
                    "ROI set",
                    "project/project-default/config/analysis/roi_sets/missing.yaml",
                    "rp analysis roi init missing --project project-default --template coordinate_sphere",
                ),
                (
                    ["analysis", "roi", "transform", "doctor", "missing", "--project", "project-default"],
                    "ROI transform set",
                    "project/project-default/config/analysis/roi_transforms/missing.yaml",
                    None,
                ),
                (
                    [
                        "analysis",
                        "roi",
                        "extraction",
                        "run",
                        "missing",
                        "--project",
                        "project-default",
                    ],
                    "ROI extraction set",
                    "project/project-default/config/analysis/extraction_sets/missing.yaml",
                    (
                        "rp analysis roi extraction init missing --project project-default "
                        "--roi-set <roi-set> --template generic_nifti"
                    ),
                ),
            )
            for args, label, expected_path, next_step in cases:
                with self.subTest(args=args):
                    message = self._run_cli_exit(args, workspace_root=workspace_root)
                    payload = json.loads(message)
                    self.assertIn(label, payload["error"])
                    self.assertIn("missing", payload["error"])
                    self.assertEqual(payload["name"], "missing")
                    self.assertEqual(payload["expected_path"], expected_path)
                    self.assertIn(expected_path, payload["error"])
                    self.assertNotIn("Traceback", message)
                    if next_step is None:
                        self.assertNotIn("next_step", payload)
                    else:
                        self.assertEqual(payload["next_step"], next_step)
                        self.assertIn(next_step, payload["error"])

    def test_unsupported_deferred_roi_families_error_clearly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            project_root = self._write_workspace(workspace_root)
            for name, family in (("atlas", "atlas_label"),):
                self._write_deferred_roi_set(project_root, name=name, family=family)
                message = self._run_cli_exit(
                    ["analysis", "roi", "build", name, "--project", "project-default"],
                    workspace_root=workspace_root,
                )
                payload = json.loads(message)
                self.assertIn("schema-only/deferred", payload["error"])

    def test_roi_doctor_reports_deferred_family_as_actionable_without_execution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            project_root = self._write_workspace(workspace_root)
            self._write_deferred_roi_set(project_root, name="atlas_example", family="atlas_label")
            before = self._workspace_tree_snapshot(workspace_root)

            with mock.patch("research_platform.neuro.roi_execution._execute_build_action") as builder:
                code, output = self._run_cli(
                    ["analysis", "roi", "doctor", "atlas_example", "--project", "project-default"],
                    workspace_root=workspace_root,
                )

            payload = json.loads(output)
            family_checks = [
                check
                for check in payload["checks"]
                if check["check_id"] == "roi_family_supported" and check["status"] == "error"
            ]
            self.assertEqual(code, 1)
            self.assertTrue(payload["schema_valid"])
            self.assertFalse(payload["ready_for_execution"])
            self.assertEqual(len(family_checks), 1)
            self.assertIn("supported ROI family", family_checks[0]["action"])
            builder.assert_not_called()
            self.assertEqual(self._workspace_tree_snapshot(workspace_root), before)

    @unittest.skipIf(nib is None or np is None, "numpy and nibabel are required to create the synthetic doctor input")
    def test_roi_doctor_reports_missing_nifti_dependency_as_actionable_without_execution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            project_root = self._write_workspace(workspace_root)
            self._write_phase3_images(project_root)
            self._write_coordinate_roi_set(project_root)
            before = self._workspace_tree_snapshot(workspace_root)

            with (
                mock.patch("research_platform.neuro.nifti.nib", None),
                mock.patch("research_platform.neuro.roi_execution._execute_build_action") as builder,
            ):
                code, output = self._run_cli(
                    ["analysis", "roi", "doctor", "coordinate", "--project", "project-default"],
                    workspace_root=workspace_root,
                )

            payload = json.loads(output)
            dependency_checks = [
                check
                for check in payload["checks"]
                if check["check_id"] == "python_dependency_available" and check["status"] == "error"
            ]
            self.assertEqual(code, 1)
            self.assertTrue(payload["schema_valid"])
            self.assertFalse(payload["ready_for_execution"])
            self.assertEqual(len(dependency_checks), 1)
            self.assertIn("installation profile", dependency_checks[0]["action"])
            builder.assert_not_called()
            self.assertEqual(self._workspace_tree_snapshot(workspace_root), before)

    def test_loso_roi_build_rejects_non_flame_backend(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            project_root = self._write_workspace(workspace_root)
            self._write_deferred_roi_set(project_root, name="loso_bad_backend", family="loso_group_map")

            message = self._run_cli_exit(
                ["analysis", "roi", "build", "loso_bad_backend", "--project", "project-default"],
                workspace_root=workspace_root,
            )

            payload = json.loads(message)
            self.assertIn("supports only backend fsl_flame1", payload["error"])

    def _phase3_entities(self) -> dict[str, object]:
        return {
            "subject": "sub-001",
            "session": "ses-01",
            "task": "memory",
            "space": "MNI152NLin2009cAsym",
        }

    def _phase3_outputs(self) -> dict[str, object]:
        return {"root_ref": "artifacts_root", "path": "roi-derivatives"}

    def _write_phase3_images(self, project_root: Path) -> None:
        if nib is None or np is None:
            return
        fixtures = project_root / "config" / "analysis" / "fixtures"
        fixtures.mkdir(parents=True, exist_ok=True)
        reference = np.zeros((5, 5, 5), dtype=float)
        manual = np.zeros((5, 5, 5), dtype=np.uint8)
        manual[1, 1, 1] = 1
        stat = np.zeros((5, 5, 5), dtype=float)
        stat[2, 2, 2] = 5.5
        stat[1, 1, 1] = 4.0
        values = np.arange(125, dtype=float).reshape((5, 5, 5))
        for filename, data in (
            ("reference.nii.gz", reference),
            ("manual_mask.nii.gz", manual),
            ("stat_map.nii.gz", stat),
            ("value_map.nii.gz", values),
        ):
            nib.save(nib.Nifti1Image(data, np.eye(4)), fixtures / filename)

    def _write_coordinate_roi_set(self, project_root: Path) -> None:
        write_yaml(
            project_root / "config" / "analysis" / "roi_sets" / "coordinate.yaml",
            {
                "roi_set": {
                    "name": "coordinate",
                    **self._phase3_entities(),
                    "outputs": self._phase3_outputs(),
                    "rois": [
                        {
                            "label": "SeedSphere",
                            "family": "coordinate_sphere",
                            "backend": "generic_nifti",
                            "desc": "CoordinateSphere",
                            "reference_image": {
                                "root_ref": "project_roi_root",
                                "pattern": "fixtures/reference.nii.gz",
                            },
                            "coordinate": [2, 2, 2],
                            "radius_mm": 1.01,
                        }
                    ],
                }
            },
        )

    def _write_manual_roi_set(self, project_root: Path) -> None:
        write_yaml(
            project_root / "config" / "analysis" / "roi_sets" / "manual.yaml",
            {
                "roi_set": {
                    "name": "manual",
                    **self._phase3_entities(),
                    "outputs": self._phase3_outputs(),
                    "rois": [
                        {
                            "label": "ManualMask",
                            "family": "manual_mask",
                            "backend": "manual",
                            "desc": "CuratedMask",
                            "source": {
                                "root_ref": "project_roi_root",
                                "pattern": "fixtures/manual_mask.nii.gz",
                            },
                            "reference_image": {
                                "root_ref": "project_roi_root",
                                "pattern": "fixtures/reference.nii.gz",
                            },
                        }
                    ],
                }
            },
        )

    def _write_functional_roi_set(self, project_root: Path, *, name: str, threshold: float, fallback: bool) -> None:
        write_yaml(
            project_root / "config" / "analysis" / "roi_sets" / f"{name}.yaml",
            {
                "roi_set": {
                    "name": name,
                    **self._phase3_entities(),
                    "outputs": self._phase3_outputs(),
                    "rois": [
                        {
                            "label": "PeakSphere",
                            "family": "functional_threshold_map",
                            "backend": "generic_nifti",
                            "desc": "ExistingMap",
                            "stat_map": {
                                "root_ref": "project_roi_root",
                                "pattern": "fixtures/stat_map.nii.gz",
                            },
                            "seed_coordinate": [2, 2, 2],
                            "search_radius_mm": 3,
                            "sphere_radius_mm": 1.01,
                            "z_threshold": threshold,
                            "allow_below_threshold_fallback": fallback,
                        }
                    ],
                }
            },
        )

    def _write_extraction_set(self, project_root: Path) -> None:
        write_yaml(
            project_root / "config" / "analysis" / "extraction_sets" / "generic_values.yaml",
            {
                "extraction_set": {
                    "name": "generic_values",
                    "roi_set": "coordinate",
                    **self._phase3_entities(),
                    "outputs": {**self._phase3_outputs(), "format": "tsv"},
                    "targets": [
                        {
                            "name": "GenericValues",
                            "backend": "generic_nifti",
                            "desc": "GenericValues",
                            "metrics": ["mean", "voxel_count", "valid_voxel_count", "max"],
                            "inputs": {
                                "root_ref": "project_roi_root",
                                "pattern": "fixtures/value_map.nii.gz",
                                "desc": "ValueMap",
                            },
                            "roi_labels": ["SeedSphere"],
                        }
                    ],
                }
            },
        )

    def _write_loso_inputs(self, workspace_root: Path, *, nifti_masks: bool) -> None:
        derivative = workspace_root / "datasets" / "demo-derivatives"
        group_mask = derivative / "group" / "ses-01" / "func" / "group_ses-01_task-memory_space-MNI152NLin2009cAsym_mask.nii.gz"
        group_mask.parent.mkdir(parents=True, exist_ok=True)
        if nifti_masks and nib is not None and np is not None:
            nib.save(nib.Nifti1Image(np.ones((5, 5, 5), dtype=np.uint8), np.eye(4)), group_mask)
        else:
            group_mask.write_text("group mask", encoding="utf-8")
        for subject in ("001", "002", "003"):
            cope_dir = (
                derivative
                / f"sub-{subject}"
                / "ses-01"
                / "func"
                / "task-memory_model-ModelA_contrast-CondA"
            )
            cope_dir.mkdir(parents=True, exist_ok=True)
            for kind, filename in (
                ("cope", "cope1.nii.gz"),
                ("varcope", "varcope1.nii.gz"),
                ("mask", "mask.nii.gz"),
            ):
                path = cope_dir / filename
                if nifti_masks and nib is not None and np is not None:
                    if kind == "mask":
                        data = np.ones((5, 5, 5), dtype=np.uint8)
                        if subject == "001":
                            data[1, 2, 2] = 0
                    else:
                        data = np.full((5, 5, 5), float(int(subject)), dtype=float)
                    nib.save(nib.Nifti1Image(data, np.eye(4)), path)
                else:
                    path.write_text(kind, encoding="utf-8")

    def _write_featquery_inputs(self, workspace_root: Path, project_root: Path) -> None:
        feat_dir = (
            workspace_root
            / "datasets"
            / "first-level"
            / "sub-001"
            / "ses-01"
            / "func"
            / "sub-001_ses-01_task-memory_model-ModelA.feat"
        )
        (feat_dir / "stats").mkdir(parents=True, exist_ok=True)
        (feat_dir / "stats" / "cope1").write_text("cope", encoding="utf-8")
        mask = (
            project_root
            / "config"
            / "analysis"
            / "masks"
            / "sub-001"
            / "ses-01"
            / "func"
            / "sub-001_ses-01_task-memory_label-SeedA_mask.nii.gz"
        )
        mask.parent.mkdir(parents=True, exist_ok=True)
        mask.write_text("mask", encoding="utf-8")
        write_yaml(
            project_root / "config" / "analysis.yaml",
            {"analysis": {"external_input_roots": {"first_level": {"local_root": "datasets/first-level"}}}},
        )

    def _write_featquery_extraction_set(self, project_root: Path) -> None:
        write_yaml(
            project_root / "config" / "analysis" / "extraction_sets" / "featquery_values.yaml",
            {
                "extraction_set": {
                    "name": "featquery_values",
                    "subjects": ["sub-001"],
                    "session": "ses-01",
                    "task": "memory",
                    "model": "ModelA",
                    "outputs": {"root_ref": "artifacts_root", "path": "roi-derivatives", "format": "tsv"},
                    "targets": [
                        {
                            "name": "FeatqueryValues",
                            "backend": "fsl_featquery",
                            "desc": "ModelAFeatquery",
                            "metrics": ["mean_cope", "roi_voxel_count"],
                            "inputs": {
                                "feat_root_ref": "first_level",
                                "feat_dir": "{subject_dir}/{session_dir}/func/sub-{subject_id}_{session_dir}_task-{task_id}_model-{model}.feat",
                                "value_image": "stats/cope{cope}",
                            },
                            "contrasts": [{"id": "CondA", "cope": 1, "desc": "CondA"}],
                            "featquery_output_name": "fq_{roi_label}_{source_contrast}_cope{cope}",
                            "roi_masks": [
                                {
                                    "label": "SeedA",
                                    "family": "manual_mask",
                                    "root_ref": "project_roi_root",
                                    "pattern": "masks/{subject_dir}/{session_dir}/func/sub-{subject_id}_{session_dir}_task-{task_id}_label-SeedA_mask.nii.gz",
                                }
                            ],
                        }
                    ],
                }
            },
        )

    def _write_env_placeholder_featquery_extraction_set(self, project_root: Path) -> None:
        mask = (
            project_root
            / "config"
            / "analysis"
            / "masks"
            / "sub-001"
            / "ses-01"
            / "func"
            / "sub-001_ses-01_task-memory_label-SeedA_mask.nii.gz"
        )
        mask.parent.mkdir(parents=True, exist_ok=True)
        mask.write_text("mask", encoding="utf-8")
        write_yaml(
            project_root / "config" / "analysis" / "extraction_sets" / "env_featquery_values.yaml",
            {
                "extraction_set": {
                    "name": "env_featquery_values",
                    "subjects": ["sub-001"],
                    "session": "ses-01",
                    "task": "memory",
                    "model": "ModelA",
                    "outputs": {"root": "${ROI_DERIV_ROOT:-}", "format": "tsv"},
                    "targets": [
                        {
                            "name": "EnvFeatqueryValues",
                            "backend": "fsl_featquery",
                            "desc": "EnvFeatquery",
                            "metrics": ["mean_cope", "roi_voxel_count"],
                            "inputs": {
                                "feat_dir": "${ROI_FEAT_ROOT:-}/{subject_dir}/{session_dir}/func/{subject_dir}_{session_dir}_task-{task_id}_model-{model}.feat",
                                "value_image": "stats/cope{cope}",
                            },
                            "contrasts": [{"id": "CondA", "cope": 1, "desc": "CondA"}],
                            "featquery_output_name": "fq_{roi_label}_{source_contrast}_cope{cope}",
                            "roi_masks": [
                                {
                                    "label": "SeedA",
                                    "path": "config/analysis/masks/sub-001/ses-01/func/sub-001_ses-01_task-memory_label-SeedA_mask.nii.gz",
                                }
                            ],
                        }
                    ],
                }
            },
        )

    def _write_env_placeholder_loso_roi_set(self, project_root: Path, *, literal_output_root: bool = False) -> None:
        outputs_root = str(self._synthetic_personal_env_roots()[1]) if literal_output_root else "${ROI_DERIV_ROOT:-}"
        write_yaml(
            project_root / "config" / "analysis" / "roi_sets" / "env_loso.yaml",
            {
                "roi_set": {
                    "name": "env_loso",
                    "backend": "fsl_flame1",
                    "subjects": ["sub-001", "sub-002", "sub-003"],
                    "held_out_subjects": ["sub-001"],
                    "session": "ses-01",
                    "task": "memory",
                    "model": "ModelA",
                    "space": "MNI152NLin2009cAsym",
                    "min_group_n": 2,
                    "outputs": {"root": outputs_root},
                    "fixed_effects_inputs": {
                        "root": "${ROI_FEAT_ROOT:-}",
                        "cope_dir": "{subject_dir}/{session_dir}/func/task-{task_id}_model-{model}_contrast-{contrast_id}",
                        "cope_image": "cope{cope_number}.nii.gz",
                        "varcope_image": "varcope{cope_number}.nii.gz",
                        "mask_image": "mask.nii.gz",
                    },
                    "group_mask": {
                        "path": "${ROI_FEAT_ROOT:-}/group/{session_dir}/func/group_{session_dir}_task-{task_id}_space-{space}_mask.nii.gz",
                    },
                    "contrasts": [{"id": "CondA", "cope_number": 1, "desc": "CondA"}],
                    "rois": [
                        {
                            "label": "SeedA",
                            "family": "loso_group_map",
                            "backend": "fsl_flame1",
                            "desc": "CondA",
                            "contrast": "CondA",
                            "seed_coordinate": [2, 2, 2],
                            "search_radius_mm": 2,
                            "sphere_radius_mm": 1.01,
                            "z_threshold": 3.1,
                            "allow_below_threshold_fallback": True,
                        }
                    ],
                }
            },
        )

    def _write_env_roi_set_ref_featquery_extraction_set(self, project_root: Path) -> None:
        write_yaml(
            project_root / "config" / "analysis" / "extraction_sets" / "env_loso_featquery.yaml",
            {
                "extraction_set": {
                    "name": "env_loso_featquery",
                    "roi_set": "env_loso",
                    "subjects": ["sub-001"],
                    "session": "ses-01",
                    "task": "memory",
                    "model": "ModelA",
                    "outputs": {"root": "${ROI_DERIV_ROOT:-}", "format": "tsv"},
                    "targets": [
                        {
                            "name": "EnvLosoFeatquery",
                            "backend": "fsl_featquery",
                            "desc": "EnvLosoFeatquery",
                            "metrics": ["mean_cope", "roi_voxel_count"],
                            "inputs": {
                                "feat_dir": "${ROI_FEAT_ROOT:-}/{subject_dir}/{session_dir}/func/{subject_dir}_{session_dir}_task-{task_id}_model-{model}.feat",
                                "value_image": "stats/cope{cope}",
                            },
                            "contrasts": [{"id": "CondA", "cope": 1, "desc": "CondA"}],
                            "featquery_output_name": "fq_{roi_label}_{source_contrast}_cope{cope}",
                            "roi_labels": ["SeedA"],
                        }
                    ],
                }
            },
        )

    def _write_loso_roi_set(self, project_root: Path, *, two_rois: bool) -> None:
        rois = [
            {
                "label": "SeedA",
                "family": "loso_group_map",
                "backend": "fsl_flame1",
                "desc": "CondA",
                "contrast": "CondA",
                "seed_coordinate": [2, 2, 2],
                "search_radius_mm": 2,
                "sphere_radius_mm": 1.01,
                "z_threshold": 3.1,
                "allow_below_threshold_fallback": True,
            }
        ]
        if two_rois:
            rois.append({**rois[0], "label": "SeedB"})
        write_yaml(
            project_root / "config" / "analysis.yaml",
            {
                "analysis": {
                    "external_input_roots": {
                        "loso_inputs": {"local_root": "datasets/demo-derivatives"},
                    }
                }
            },
        )
        write_yaml(
            project_root / "config" / "analysis" / "roi_sets" / "loso.yaml",
            {
                "roi_set": {
                    "name": "loso",
                    "backend": "fsl_flame1",
                    "subjects": ["sub-001", "sub-002", "sub-003"],
                    "held_out_subjects": ["sub-001"],
                    "session": "ses-01",
                    "task": "memory",
                    "model": "ModelA",
                    "space": "MNI152NLin2009cAsym",
                    "min_group_n": 2,
                    "outputs": {"root_ref": "artifacts_root", "path": "roi-derivatives"},
                    "fixed_effects_inputs": {
                        "root_ref": "loso_inputs",
                        "cope_dir": "{subject_dir}/{session_dir}/func/task-{task_id}_model-{model}_contrast-{contrast_id}",
                        "cope_image": "cope{cope_number}.nii.gz",
                        "varcope_image": "varcope{cope_number}.nii.gz",
                        "mask_image": "mask.nii.gz",
                    },
                    "group_mask": {
                        "root_ref": "loso_inputs",
                        "pattern": "group/{session_dir}/func/group_{session_dir}_task-{task_id}_space-{space}_mask.nii.gz",
                    },
                    "contrasts": [{"id": "CondA", "cope_number": 1, "desc": "CondA"}],
                    "rois": rois,
                }
            },
        )

    def _write_deferred_roi_set(self, project_root: Path, *, name: str, family: str) -> None:
        write_yaml(
            project_root / "config" / "analysis" / "roi_sets" / f"{name}.yaml",
            {
                "roi_set": {
                    "name": name,
                    **self._phase3_entities(),
                    "outputs": self._phase3_outputs(),
                    "rois": [
                        {
                            "label": "DeferredMask",
                            "family": family,
                            "backend": "generic_nifti",
                            "desc": "DeferredMask",
                        }
                    ],
                }
            },
        )


if __name__ == "__main__":
    unittest.main()
