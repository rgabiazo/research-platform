from __future__ import annotations

from contextlib import redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


CORE_PACKAGE_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
HPC_PACKAGE_ROOT = WORKSPACE_ROOT / "packages" / "research-hpc"
sys.path.insert(0, str(CORE_PACKAGE_ROOT / "src"))
sys.path.insert(0, str(HPC_PACKAGE_ROOT / "src"))

from research_platform.core.cli import _build_parser, _read_tsv, _write_batch_manifest, main
from research_platform.core.config import load_yaml, write_yaml


class AnalysisBundleCliTests(unittest.TestCase):
    def _write_workspace(self, workspace_root: Path, *, project_name: str = "project-demo") -> Path:
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
                "projects": {"default": project_name},
            },
        )
        project_root = workspace_root / "project" / project_name
        write_yaml(project_root / "project.yaml", {"name": project_name, "version": "0.1.0"})
        return project_root

    def _write_ready_bundle(self, project_root: Path, *, name: str = "demo-bundle") -> None:
        write_yaml(
            project_root / "config" / "cohorts.yaml",
            {
                "cohorts": {
                    "included-units": {
                        "batch": "exact_units",
                        "include": {"eligible": ["true"]},
                        "exclude": [
                            {
                                "id": "omit-review",
                                "filters": {"qc_status": ["review"]},
                                "reason": "held for review",
                            }
                        ],
                    }
                }
            },
        )
        batch_path = project_root / "manifests" / "batches" / "exact_units.tsv"
        batch_path.parent.mkdir(parents=True, exist_ok=True)
        batch_path.write_text(
            "subject_id\tsession_id\ttask_id\trun_id\teligible\tqc_status\tadapter_note\n"
            "sub-alpha\tses-visit2\texampletask\trun-01\ttrue\tpass\tfirst\n"
            "sub-alpha\tses-visit2\texampletask\trun-03\ttrue\treview\tsecond\n",
            encoding="utf-8",
            newline="",
        )
        write_yaml(
            project_root / "config" / "analysis" / "bundles" / f"{name}.yaml",
            {
                "analysis_bundle": {
                    "name": name,
                    "selection": {"cohort": "included-units"},
                    "units": {
                        "key_columns": ["subject_id", "session_id", "run_id"],
                        "subject_column": "subject_id",
                        "occasion_column": "session_id",
                        "incomplete": "allow",
                    },
                    "components": {"roi_set": "demo-rois", "extraction_set": "demo-values"},
                    "stages": ["roi_build", "roi_extraction"],
                }
            },
        )
        write_yaml(
            project_root / "config" / "analysis" / "roi_sets" / "demo-rois.yaml",
            {"roi_set": {"name": "demo-rois"}},
        )
        write_yaml(
            project_root / "config" / "analysis" / "extraction_sets" / "demo-values.yaml",
            {"extraction_set": {"name": "demo-values"}},
        )

    def _run_cli(self, args: list[str], *, workspace_root: Path) -> tuple[int, dict[str, object]]:
        output = io.StringIO()
        with mock.patch.dict(os.environ, {"RESEARCH_PLATFORM_ROOT": str(workspace_root)}, clear=False):
            with redirect_stdout(output):
                exit_code = main(args)
        return exit_code, json.loads(output.getvalue())

    def _run_cli_error(self, args: list[str], *, workspace_root: Path) -> dict[str, object]:
        with mock.patch.dict(os.environ, {"RESEARCH_PLATFORM_ROOT": str(workspace_root)}, clear=False):
            with self.assertRaises(SystemExit) as exc_info:
                main(args)
        message = str(exc_info.exception)
        self.assertNotIn("Traceback", message)
        return json.loads(message)

    def _tree_digest(self, root: Path) -> str:
        digest = hashlib.sha256()
        if not root.exists():
            return digest.hexdigest()
        for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
            relative = path.relative_to(root).as_posix()
            digest.update(relative.encode("utf-8"))
            digest.update(b"\0d\0" if path.is_dir() else b"\0f\0")
            if path.is_file():
                digest.update(path.read_bytes())
                digest.update(b"\0")
        return digest.hexdigest()

    def test_bundle_parser_exposes_only_configuration_owned_selection(self) -> None:
        parser = _build_parser()

        parsed = parser.parse_args(
            ["analysis", "bundle", "plan", "demo-bundle", "--project", "project-demo"]
        )
        self.assertEqual(parsed.analysis_command, "bundle")
        self.assertEqual(parsed.analysis_bundle_command, "plan")
        self.assertFalse(hasattr(parsed, "subject"))
        self.assertFalse(hasattr(parsed, "session"))
        self.assertFalse(hasattr(parsed, "run_id"))

    def test_bundle_parser_exposes_exactly_the_plan_only_command_family(self) -> None:
        parser = _build_parser()
        root_subparsers = next(action for action in parser._actions if action.dest == "command")
        analysis_parser = root_subparsers.choices["analysis"]
        analysis_subparsers = next(
            action for action in analysis_parser._actions if action.dest == "analysis_command"
        )
        bundle_parser = analysis_subparsers.choices["bundle"]
        bundle_subparsers = next(
            action
            for action in bundle_parser._actions
            if action.dest == "analysis_bundle_command"
        )

        self.assertEqual(
            tuple(bundle_subparsers.choices),
            ("init", "list", "show", "validate", "doctor", "plan"),
        )

    def test_init_dry_run_is_nonmutating_then_init_list_show_and_force_are_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            project_root = self._write_workspace(workspace_root)
            before = self._tree_digest(workspace_root)

            dry_exit, dry_payload = self._run_cli(
                [
                    "analysis",
                    "bundle",
                    "init",
                    "starter",
                    "--project",
                    "project-demo",
                    "--dry-run",
                ],
                workspace_root=workspace_root,
            )
            after_dry_run = self._tree_digest(workspace_root)
            bundle_path = project_root / "config" / "analysis" / "bundles" / "starter.yaml"
            dry_run_created_bundle = bundle_path.exists()

            init_exit, init_payload = self._run_cli(
                ["analysis", "bundle", "init", "starter", "--project", "project-demo"],
                workspace_root=workspace_root,
            )
            first_content = bundle_path.read_bytes()
            scaffold_before_checks = self._tree_digest(workspace_root)
            validate_exit, validate_payload = self._run_cli(
                ["analysis", "bundle", "validate", "starter", "--project", "project-demo"],
                workspace_root=workspace_root,
            )
            doctor_exit, doctor_payload = self._run_cli(
                ["analysis", "bundle", "doctor", "starter", "--project", "project-demo"],
                workspace_root=workspace_root,
            )
            plan_exit, plan_payload = self._run_cli(
                ["analysis", "bundle", "plan", "starter", "--project", "project-demo"],
                workspace_root=workspace_root,
            )
            scaffold_after_checks = self._tree_digest(workspace_root)
            list_exit, list_payload = self._run_cli(
                ["analysis", "bundle", "list", "--project", "project-demo"],
                workspace_root=workspace_root,
            )
            show_exit, show_payload = self._run_cli(
                ["analysis", "bundle", "show", "starter", "--project", "project-demo"],
                workspace_root=workspace_root,
            )
            collision = self._run_cli_error(
                ["analysis", "bundle", "init", "starter", "--project", "project-demo"],
                workspace_root=workspace_root,
            )
            force_exit, force_payload = self._run_cli(
                [
                    "analysis",
                    "bundle",
                    "init",
                    "starter",
                    "--project",
                    "project-demo",
                    "--force",
                ],
                workspace_root=workspace_root,
            )
            second_content = bundle_path.read_bytes()

        self.assertEqual(dry_exit, 0)
        self.assertFalse(dry_payload["written"])
        self.assertIn("analysis_bundle:", dry_payload["yaml"])
        self.assertEqual(before, after_dry_run)
        self.assertFalse(dry_run_created_bundle)
        self.assertEqual(init_exit, 0)
        self.assertTrue(init_payload["written"])
        self.assertEqual(
            init_payload["path"],
            "project/project-demo/config/analysis/bundles/starter.yaml",
        )
        self.assertEqual(validate_exit, 0)
        self.assertTrue(validate_payload["valid"])
        self.assertEqual(doctor_exit, 1)
        self.assertFalse(doctor_payload["ready_for_planning"])
        self.assertEqual(plan_exit, 1)
        self.assertFalse(plan_payload["executed"])
        self.assertEqual(scaffold_before_checks, scaffold_after_checks)
        self.assertEqual(list_exit, 0)
        self.assertEqual(list_payload["bundles"], ["starter"])
        self.assertEqual(
            list_payload["config_dir"],
            "project/project-demo/config/analysis/bundles",
        )
        self.assertEqual(show_exit, 0)
        self.assertEqual(show_payload["path"], init_payload["path"])
        self.assertEqual(show_payload["document"]["analysis_bundle"]["selection"], {"batch": "default"})
        self.assertEqual(show_payload["document"]["analysis_bundle"]["components"], {})
        self.assertEqual(show_payload["document"]["analysis_bundle"]["stages"], [])
        self.assertIn("Use --force", collision["error"])
        self.assertEqual(force_exit, 0)
        self.assertTrue(force_payload["written"])
        self.assertEqual(first_content, second_content)

    def test_init_rejects_invalid_leading_characters_before_writing_and_preserves_valid_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            project_root = self._write_workspace(workspace_root)
            before = self._tree_digest(workspace_root)

            for invalid_name in (".hidden", "_private", "-option"):
                with self.subTest(invalid_name=invalid_name):
                    init_args = ["analysis", "bundle", "init"]
                    if invalid_name.startswith("-"):
                        init_args.extend(["--project", "project-demo", "--", invalid_name])
                    else:
                        init_args.extend([invalid_name, "--project", "project-demo"])
                    payload = self._run_cli_error(
                        init_args,
                        workspace_root=workspace_root,
                    )
                    self.assertIn("Start with a letter or number", payload["error"])

            after_invalid = self._tree_digest(workspace_root)
            valid_name = "Bundle_1.2-test"
            valid_exit, valid_payload = self._run_cli(
                ["analysis", "bundle", "init", valid_name, "--project", "project-demo"],
                workspace_root=workspace_root,
            )
            document = load_yaml(
                project_root / "config" / "analysis" / "bundles" / f"{valid_name}.yaml",
                resolve_env=False,
            )

        self.assertEqual(before, after_invalid)
        self.assertEqual(valid_exit, 0)
        self.assertEqual(valid_payload["bundle"], valid_name)
        self.assertEqual(document["analysis_bundle"]["name"], valid_name)

    def test_bids_batch_writer_preserves_deterministic_metadata_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "batches" / "discovered.tsv"

            _write_batch_manifest(
                path,
                [
                    {
                        "subject_id": "sub-alpha",
                        "session_id": "ses-01",
                        "task_id": "exampletask",
                        "run_id": "run-02",
                        "direction": "dir-ap",
                        "acquisition": "acq-fast",
                        "adapter_note": "retained",
                    }
                ],
            )
            content = path.read_text(encoding="utf-8")

        self.assertEqual(
            content,
            "subject_id\tsession_id\ttask_id\trun_id\tdirection\tacquisition\tadapter_note\n"
            "sub-alpha\tses-01\texampletask\trun-02\tdir-ap\tacq-fast\tretained\n",
        )

    def test_legacy_batch_reader_still_expands_environment_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "tabular.tsv"
            path.write_text(
                "feature_table\ttarget_column\n${DATA_ROOT}/toy.tsv\tbinary_target\n",
                encoding="utf-8",
                newline="",
            )
            with mock.patch.dict(os.environ, {"DATA_ROOT": "datasets/example"}, clear=False):
                rows = _read_tsv(path)

        self.assertEqual(
            rows,
            [{"feature_table": "datasets/example/toy.tsv", "target_column": "binary_target"}],
        )

    def test_missing_bundle_error_has_project_relative_path_and_exact_init_hint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            self._write_workspace(workspace_root)
            for command in ("show", "validate", "doctor", "plan"):
                with self.subTest(command=command):
                    payload = self._run_cli_error(
                        ["analysis", "bundle", command, "missing", "--project", "project-demo"],
                        workspace_root=workspace_root,
                    )

                    self.assertEqual(payload["name"], "missing")
                    self.assertEqual(
                        payload["expected_path"],
                        "project/project-demo/config/analysis/bundles/missing.yaml",
                    )
                    self.assertEqual(
                        payload["next_step"],
                        "rp analysis bundle init missing --project project-demo",
                    )

    def test_bundle_list_and_lookup_ignore_yaml_named_directories(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            project_root = self._write_workspace(workspace_root)
            bundles_dir = project_root / "config" / "analysis" / "bundles"
            (bundles_dir / "not-a-bundle.yaml").mkdir(parents=True)

            list_exit, list_payload = self._run_cli(
                ["analysis", "bundle", "list", "--project", "project-demo"],
                workspace_root=workspace_root,
            )
            missing_payload = self._run_cli_error(
                [
                    "analysis",
                    "bundle",
                    "show",
                    "not-a-bundle",
                    "--project",
                    "project-demo",
                ],
                workspace_root=workspace_root,
            )

        self.assertEqual(list_exit, 0)
        self.assertEqual(list_payload["bundles"], [])
        self.assertEqual(missing_payload["name"], "not-a-bundle")
        self.assertIn("rp analysis bundle init not-a-bundle", missing_payload["next_step"])

    def test_validate_is_schema_only_even_when_selection_inputs_are_absent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            project_root = self._write_workspace(workspace_root)
            bundle_path = project_root / "config" / "analysis" / "bundles" / "schema-only.yaml"
            write_yaml(bundle_path, {
                "analysis_bundle": {
                    "name": "schema-only",
                    "selection": {"batch": "not-yet-created"},
                    "units": {
                        "key_columns": ["subject_id"],
                        "subject_column": "subject_id",
                        "incomplete": "allow",
                    },
                    "components": {},
                    "stages": [],
                }
            })
            before = self._tree_digest(workspace_root)

            exit_code, payload = self._run_cli(
                ["analysis", "bundle", "validate", "schema-only", "--project", "project-demo"],
                workspace_root=workspace_root,
            )
            after = self._tree_digest(workspace_root)

        self.assertEqual(exit_code, 0)
        self.assertTrue(payload["valid"])
        self.assertEqual(payload["errors"], [])
        self.assertEqual(before, after)

    def test_validate_rejects_bundle_name_that_does_not_match_filename(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            project_root = self._write_workspace(workspace_root)
            write_yaml(
                project_root / "config" / "analysis" / "bundles" / "expected-name.yaml",
                {
                    "analysis_bundle": {
                        "name": "different-name",
                        "selection": {"batch": "default"},
                        "units": {
                            "key_columns": ["subject_id"],
                            "subject_column": "subject_id",
                            "incomplete": "allow",
                        },
                        "components": {},
                        "stages": [],
                    }
                },
            )

            exit_code, payload = self._run_cli(
                [
                    "analysis",
                    "bundle",
                    "validate",
                    "expected-name",
                    "--project",
                    "project-demo",
                ],
                workspace_root=workspace_root,
            )

        self.assertEqual(exit_code, 1)
        self.assertFalse(payload["valid"])
        self.assertIn("must match its configuration filename", payload["errors"][0])

    def test_empty_bundle_payload_is_a_concise_nonzero_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            project_root = self._write_workspace(workspace_root)
            write_yaml(
                project_root / "config" / "analysis" / "bundles" / "empty.yaml",
                {"analysis_bundle": {}},
            )

            for command in ("validate", "doctor", "plan"):
                with self.subTest(command=command):
                    exit_code, payload = self._run_cli(
                        ["analysis", "bundle", command, "empty", "--project", "project-demo"],
                        workspace_root=workspace_root,
                    )

                    self.assertNotEqual(exit_code, 0)
                    self.assertFalse(payload["valid"])
                    self.assertTrue(payload["errors"])
                    self.assertNotIn("Traceback", json.dumps(payload))

    def test_validate_rejects_wrapped_top_level_selector_and_command_siblings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            project_root = self._write_workspace(workspace_root)
            base_payload = {
                "analysis_bundle": {
                    "name": "invalid-sibling",
                    "selection": {"batch": "default"},
                    "units": {
                        "key_columns": ["subject_id"],
                        "subject_column": "subject_id",
                        "incomplete": "allow",
                    },
                    "components": {},
                    "stages": [],
                }
            }

            for sibling, value, expected_error in (
                ("subjects", ["sub-alpha"], "sibling inline subject"),
                ("command", "run-example", "sibling execution declarations"),
            ):
                with self.subTest(sibling=sibling):
                    document = dict(base_payload)
                    document[sibling] = value
                    write_yaml(
                        project_root
                        / "config"
                        / "analysis"
                        / "bundles"
                        / "invalid-sibling.yaml",
                        document,
                    )
                    exit_code, payload = self._run_cli(
                        [
                            "analysis",
                            "bundle",
                            "validate",
                            "invalid-sibling",
                            "--project",
                            "project-demo",
                        ],
                        workspace_root=workspace_root,
                    )

                    self.assertEqual(exit_code, 1)
                    self.assertFalse(payload["valid"])
                    self.assertTrue(any(expected_error in error for error in payload["errors"]))
                    self.assertNotIn("Traceback", json.dumps(payload))

    def test_selected_malformed_batch_error_is_concise_and_project_relative(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            project_root = self._write_workspace(workspace_root)
            self._write_ready_bundle(project_root)
            (project_root / "manifests" / "batches" / "exact_units.tsv").write_text(
                "duplicate\tduplicate\nvalue-a\tvalue-b\n",
                encoding="utf-8",
                newline="",
            )

            for command in ("doctor", "plan"):
                with self.subTest(command=command):
                    payload = self._run_cli_error(
                        ["analysis", "bundle", command, "demo-bundle", "--project", "project-demo"],
                        workspace_root=workspace_root,
                    )

                    self.assertEqual(payload["batch"], "exact_units")
                    self.assertEqual(
                        payload["expected_path"],
                        "project/project-demo/manifests/batches/exact_units.tsv",
                    )
                    self.assertIn("not a valid TSV manifest", payload["error"])
                    self.assertNotIn(str(workspace_root), json.dumps(payload))

    def test_selected_batch_os_error_is_concise_and_project_relative(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            project_root = self._write_workspace(workspace_root)
            self._write_ready_bundle(project_root)

            for command in ("doctor", "plan"):
                with self.subTest(command=command):
                    with mock.patch(
                        "research_platform.core.manifests.read_manifest_table",
                        side_effect=OSError("unavailable manifest"),
                    ):
                        payload = self._run_cli_error(
                            [
                                "analysis",
                                "bundle",
                                command,
                                "demo-bundle",
                                "--project",
                                "project-demo",
                            ],
                            workspace_root=workspace_root,
                        )

                    self.assertEqual(payload["batch"], "exact_units")
                    self.assertEqual(
                        payload["expected_path"],
                        "project/project-demo/manifests/batches/exact_units.tsv",
                    )
                    self.assertNotIn(str(workspace_root), json.dumps(payload))

    def test_component_inventory_ignores_yaml_named_directories(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            project_root = self._write_workspace(workspace_root)
            self._write_ready_bundle(project_root)
            for directory, name in (
                ("roi_sets", "demo-rois"),
                ("extraction_sets", "demo-values"),
            ):
                path = project_root / "config" / "analysis" / directory / f"{name}.yaml"
                path.unlink()
                path.mkdir()

            doctor_exit, doctor = self._run_cli(
                ["analysis", "bundle", "doctor", "demo-bundle", "--project", "project-demo"],
                workspace_root=workspace_root,
            )

        self.assertEqual(doctor_exit, 1)
        self.assertFalse(doctor["ready_for_planning"])
        self.assertTrue(any("was not found" in error for error in doctor["errors"]))

    def test_doctor_reports_allowed_incomplete_longitudinal_units_as_warning(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            project_root = self._write_workspace(workspace_root)
            batch_path = project_root / "manifests" / "batches" / "visits.tsv"
            batch_path.parent.mkdir(parents=True, exist_ok=True)
            batch_path.write_text(
                "subject_id\tsession_id\n"
                "sub-alpha\tses-01\n"
                "sub-alpha\tses-02\n"
                "sub-beta\tses-01\n",
                encoding="utf-8",
                newline="",
            )
            write_yaml(
                project_root / "config" / "analysis" / "roi_sets" / "demo-rois.yaml",
                {"roi_set": {"name": "demo-rois"}},
            )
            write_yaml(
                project_root / "config" / "analysis" / "bundles" / "visits.yaml",
                {
                    "analysis_bundle": {
                        "name": "visits",
                        "selection": {"batch": "visits"},
                        "units": {
                            "key_columns": ["subject_id", "session_id"],
                            "subject_column": "subject_id",
                            "occasion_column": "session_id",
                            "required_occasions": ["ses-01", "ses-02"],
                            "incomplete": "allow",
                        },
                        "components": {"roi_set": "demo-rois"},
                        "stages": ["roi_build"],
                    }
                },
            )

            doctor_exit, doctor = self._run_cli(
                ["analysis", "bundle", "doctor", "visits", "--project", "project-demo"],
                workspace_root=workspace_root,
            )

        longitudinal_check = next(
            check for check in doctor["checks"] if check["id"] == "longitudinal_completeness"
        )
        self.assertEqual(doctor_exit, 0)
        self.assertTrue(doctor["ready_for_planning"])
        self.assertEqual(longitudinal_check["status"], "warning")
        self.assertTrue(longitudinal_check["ok"])
        self.assertTrue(doctor["warnings"])
        self.assertTrue(
            any("policy 'allow'" in warning and "incomplete subject" in warning for warning in doctor["warnings"])
        )

    def test_doctor_and_plan_share_exact_resolution_and_are_nonmutating(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            project_root = self._write_workspace(workspace_root)
            self._write_ready_bundle(project_root)
            (project_root / "manifests" / "batches" / "unrelated.tsv").write_text(
                "duplicate\tduplicate\nvalue-a\tvalue-b\n",
                encoding="utf-8",
                newline="",
            )
            before = self._tree_digest(workspace_root)

            with mock.patch(
                "research_platform.core.cli._roi_schema",
                side_effect=AssertionError("bundle doctor must not load ROI validators"),
            ), mock.patch(
                "research_platform.core.cli._mvpa_lifecycle_functions",
                side_effect=AssertionError("bundle doctor must not load MVPA validators"),
            ):
                doctor_exit, doctor = self._run_cli(
                    ["analysis", "bundle", "doctor", "demo-bundle", "--project", "project-demo"],
                    workspace_root=workspace_root,
                )
                after_doctor = self._tree_digest(workspace_root)
                plan_exit, plan = self._run_cli(
                    ["analysis", "bundle", "plan", "demo-bundle", "--project", "project-demo"],
                    workspace_root=workspace_root,
                )
            after_plan = self._tree_digest(workspace_root)

        self.assertEqual(doctor_exit, 0)
        self.assertTrue(doctor["valid"])
        self.assertTrue(doctor["ready_for_planning"])
        self.assertEqual(doctor["errors"], [])
        self.assertEqual(plan_exit, 0)
        self.assertTrue(plan["valid"])
        self.assertFalse(plan["executed"])
        self.assertEqual(
            [row["values"]["adapter_note"] for row in plan["units"]["included"]],
            ["first"],
        )
        self.assertEqual(
            [row["values"]["adapter_note"] for row in plan["units"]["excluded"]],
            ["second"],
        )
        self.assertEqual(plan["units"]["excluded"][0]["exclusion_ids"], ["omit-review"])
        self.assertEqual(before, after_doctor)
        self.assertEqual(before, after_plan)

    def test_bundle_context_does_not_apply_project_hpc_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            self._write_workspace(workspace_root)

            with mock.patch(
                "research_platform.core.cli.apply_hpc_target_defaults"
            ) as apply_defaults:
                exit_code, payload = self._run_cli(
                    ["analysis", "bundle", "list", "--project", "project-demo"],
                    workspace_root=workspace_root,
                )

        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["bundles"], [])
        self.assertEqual(apply_defaults.call_count, 1)
        self.assertIsNone(apply_defaults.call_args.kwargs["project_name"])

    def test_checked_in_toy_roi_bundle_lifecycle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            artifact_root = Path(tmp_dir) / "artifacts"
            env = {
                "RESEARCH_PLATFORM_ROOT": str(WORKSPACE_ROOT),
                "ARTIFACTS_ROOT": str(artifact_root),
            }
            payloads: dict[str, dict[str, object]] = {}
            with mock.patch.dict(os.environ, env, clear=False):
                for command in ("validate", "doctor", "plan"):
                    output = io.StringIO()
                    with redirect_stdout(output):
                        exit_code = main(
                            ["analysis", "bundle", command, "toy-roi", "--project", "project-example"]
                        )
                    self.assertEqual(exit_code, 0, output.getvalue())
                    payloads[command] = json.loads(output.getvalue())

        self.assertTrue(payloads["validate"]["valid"])
        self.assertTrue(payloads["doctor"]["ready_for_planning"])
        self.assertFalse(payloads["plan"]["executed"])
        self.assertEqual(len(payloads["plan"]["units"]["included"]), 1)
        self.assertEqual(
            payloads["plan"]["units"]["included"][0]["values"]["subject_id"],
            "sub-toy01",
        )


if __name__ == "__main__":
    unittest.main()
