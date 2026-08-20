from __future__ import annotations

from contextlib import redirect_stdout
import csv
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
NEURO_PACKAGE_ROOT = WORKSPACE_ROOT / "packages" / "research-neuro"
ANALYSIS_PACKAGE_ROOT = WORKSPACE_ROOT / "packages" / "research-analysis"
sys.path.insert(0, str(CORE_PACKAGE_ROOT / "src"))
sys.path.insert(0, str(HPC_PACKAGE_ROOT / "src"))
sys.path.insert(0, str(NEURO_PACKAGE_ROOT / "src"))
sys.path.insert(0, str(ANALYSIS_PACKAGE_ROOT / "src"))

import research_platform.core.cli as core_cli
from research_platform.core.cli import _build_parser, main
from research_platform.core.config import load_yaml, write_yaml
from research_platform.neuro._roi_path_safety import published_text_contains_local_path_reference
from research_platform.neuro.mvpa.materialized_pattern_table import (
    OPTIONAL_COLUMNS,
    REQUIRED_COLUMNS,
    SCHEMA_VERSION as MATERIALIZED_SCHEMA_VERSION,
)
from research_platform.neuro.mvpa.runtime_transaction import (
    MANIFEST_RELATIVE_PATH,
    runtime_output_specs,
)


PROJECT_NAME = "project-demo-mvpa"
MVPA_SET = "toy-crossnobis"
BUNDLE_NAME = "toy-units"
COHORT_NAME = "included-units"
PATTERN_SOURCE = "prepared-patterns"
ROI_SOURCE = "prepared-rois"
ROI_LABEL = "SeedA"
FEATURE_SPACE = "example-feature-space"
ROI_DEFINITION = "example-roi-definition"


class MvpaBundleLifecycleIntegrationTests(unittest.TestCase):
    def _write_workspace(self, workspace_root: Path, *, runtime_path: str | None = None) -> Path:
        write_yaml(
            workspace_root / "WORKSPACE.yaml",
            {
                "paths": {
                    "artifacts_root": "./artifacts",
                    "datasets_root": "./datasets",
                    "ops_root": "./ops",
                    "mvpa_inputs": "./inputs/mvpa",
                },
                "repos": {
                    "project_root": "./project",
                    "pipelines_root": "./pipelines",
                },
                "projects": {"default": PROJECT_NAME},
            },
        )
        project_root = workspace_root / "project" / PROJECT_NAME
        write_yaml(
            project_root / "project.yaml",
            {"name": PROJECT_NAME, "version": "0.1.0"},
        )
        (project_root / "config" / "analysis").mkdir(parents=True, exist_ok=True)
        (workspace_root / "artifacts").mkdir(parents=True, exist_ok=True)
        (workspace_root / "inputs" / "mvpa").mkdir(parents=True, exist_ok=True)
        if runtime_path is not None:
            self._write_ready_mvpa_config(project_root, runtime_path=runtime_path)
        return project_root

    def _write_ready_mvpa_config(
        self,
        project_root: Path,
        *,
        runtime_path: str = ".research-platform/mvpa/{mvpa_set}",
    ) -> Path:
        path = project_root / "config" / "analysis" / "mvpa" / f"{MVPA_SET}.yaml"
        write_yaml(
            path,
            {
                "mvpa_set": {
                    "name": MVPA_SET,
                    "description": "Synthetic prepared-vector lifecycle verification.",
                    "unit_selection": {
                        "mode": "exact_units",
                        "key_columns": ["subject_id", "session_id", "run_id"],
                    },
                    "conditions": [
                        {"id": "condition_a", "description": "First invented condition."},
                        {"id": "condition_b", "description": "Second invented condition."},
                    ],
                    "condition_pairs": [
                        {
                            "id": "condition_a_minus_condition_b",
                            "condition_a": "condition_a",
                            "condition_b": "condition_b",
                        }
                    ],
                    "pattern_sources": [
                        {
                            "name": PATTERN_SOURCE,
                            "backend": "materialized_pattern_table",
                            "root_ref": "mvpa_inputs",
                            "path": "patterns.tsv",
                            "schema_version": MATERIALIZED_SCHEMA_VERSION,
                        }
                    ],
                    "roi_sources": [
                        {
                            "name": ROI_SOURCE,
                            "source": "materialized_features",
                            "roi_labels": [ROI_LABEL],
                            "feature_space_id": FEATURE_SPACE,
                            "roi_definition_id": ROI_DEFINITION,
                        }
                    ],
                    "event_thresholds": {
                        "min_events_per_condition_per_run": 1,
                        "min_runs_per_condition": 2,
                    },
                    "mean_centering": {"enabled": False, "scope": "none"},
                    "distance": {
                        "metrics": ["crossnobis"],
                        "engine": "native_reference",
                        "cross_validation": {"unit": "run"},
                        "noise_normalization": {
                            "method": "identity",
                            "nonpositive_policy": "strict",
                            "min_retained_features": 1,
                        },
                    },
                    "outputs": {
                        "runtime_root": {
                            "root_ref": "artifact_root",
                            "path": runtime_path,
                        }
                    },
                    "runtime": {"existing_output": "fail"},
                    "missing_input_policy": "fail",
                }
            },
        )
        return path

    def _write_bundle(self, project_root: Path, *, mvpa_set: str = MVPA_SET) -> None:
        write_yaml(
            project_root / "config" / "cohorts.yaml",
            {
                "cohorts": {
                    COHORT_NAME: {
                        "batch": "exact_units",
                        "include": {"eligible": ["true"]},
                        "exclude": [
                            {
                                "id": "exclude-review",
                                "filters": {"qc_status": ["review"]},
                                "reason": "Held out by the synthetic QC example.",
                            }
                        ],
                    }
                }
            },
        )
        batch_path = project_root / "manifests" / "batches" / "exact_units.tsv"
        batch_path.parent.mkdir(parents=True, exist_ok=True)
        batch_path.write_text(
            "subject_id\tsession_id\ttask_id\trun_id\teligible\tqc_status\tvisit_index\tadapter_note\n"
            "sub-beta\tses-01\texampletask\trun-02\ttrue\tpass\t1\tfirst\n"
            "sub-alpha\tses-01\texampletask\trun-01\ttrue\tpass\t1\tsecond\n"
            "sub-beta\tses-01\texampletask\trun-01\ttrue\tpass\t1\tthird\n"
            "sub-alpha\tses-01\texampletask\trun-03\ttrue\tpass\t1\tfourth\n"
            "sub-gamma\tses-02\texampletask\trun-09\ttrue\treview\t2\texcluded\n",
            encoding="utf-8",
            newline="",
        )
        write_yaml(
            project_root / "config" / "analysis" / "bundles" / f"{BUNDLE_NAME}.yaml",
            {
                "analysis_bundle": {
                    "name": BUNDLE_NAME,
                    "selection": {"cohort": COHORT_NAME},
                    "units": {
                        "key_columns": ["subject_id", "session_id", "run_id"],
                        "subject_column": "subject_id",
                        "occasion_column": "session_id",
                        "occasion_order_column": "visit_index",
                        "incomplete": "allow",
                    },
                    "components": {"mvpa_set": mvpa_set},
                    "stages": ["mvpa"],
                }
            },
        )

    def _write_pattern_table(self, workspace_root: Path) -> Path:
        table = workspace_root / "inputs" / "mvpa" / "patterns.tsv"
        columns = tuple(dict.fromkeys((*REQUIRED_COLUMNS, *OPTIONAL_COLUMNS, "adapter_note")))
        vectors = {
            ("sub-beta", "run-02", "condition_a"): [4.0, 2.0, 1.0],
            ("sub-beta", "run-02", "condition_b"): [1.0, 0.0, 0.0],
            ("sub-alpha", "run-01", "condition_a"): [2.0, 1.0, 1.0],
            ("sub-alpha", "run-01", "condition_b"): [0.0, 0.0, 0.0],
            ("sub-beta", "run-01", "condition_a"): [3.0, 1.0, 2.0],
            ("sub-beta", "run-01", "condition_b"): [0.0, 0.0, 1.0],
            ("sub-alpha", "run-03", "condition_a"): [4.0, 2.0, 2.0],
            ("sub-alpha", "run-03", "condition_b"): [1.0, 0.0, 0.0],
        }
        notes = {
            ("sub-beta", "run-02"): "first",
            ("sub-alpha", "run-01"): "second",
            ("sub-beta", "run-01"): "third",
            ("sub-alpha", "run-03"): "fourth",
        }
        rows: list[dict[str, str]] = []
        # Deliberately reverse table order. Planning must follow exact bundle-unit order.
        for (subject_id, run_id, condition_id), values in reversed(tuple(vectors.items())):
            rows.append(
                self._pattern_row(
                    pattern_id=f"{subject_id}-{run_id}-{condition_id}",
                    subject_id=subject_id,
                    session_id="ses-01",
                    run_id=run_id,
                    condition_id=condition_id,
                    feature_values=values,
                    adapter_note=notes[(subject_id, run_id)],
                )
            )
        rows.append(
            self._pattern_row(
                pattern_id="sub-gamma-run-09-condition_a",
                subject_id="sub-gamma",
                session_id="ses-02",
                run_id="run-09",
                condition_id="condition_a",
                feature_values=[9.0, 9.0, 9.0],
                adapter_note="excluded",
            )
        )
        with table.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=columns,
                delimiter="\t",
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(rows)
        return table

    def _pattern_row(
        self,
        *,
        pattern_id: str,
        subject_id: str,
        session_id: str,
        run_id: str,
        condition_id: str,
        feature_values: list[float],
        adapter_note: str,
    ) -> dict[str, str]:
        return {
            "schema_version": MATERIALIZED_SCHEMA_VERSION,
            "pattern_id": pattern_id,
            "subject_id": subject_id,
            "session_id": session_id,
            "task_id": "exampletask",
            "run_id": run_id,
            "condition_id": condition_id,
            "pattern_source_name": PATTERN_SOURCE,
            "roi_source_name": ROI_SOURCE,
            "roi_label": ROI_LABEL,
            "feature_count": "3",
            "voxel_order": "c-order",
            "voxel_index_hash": "toy-index-v1",
            "feature_space_id": FEATURE_SPACE,
            "roi_definition_id": ROI_DEFINITION,
            "feature_values": json.dumps(feature_values, separators=(",", ":")),
            "usable": "true",
            "status": "ok",
            "mean_centering_applied": "false",
            "mean_centering_scope": "none",
            "noise_status": "unused",
            "noise_usable": "false",
            "cross_validation_label": run_id,
            "event_count": "4",
            "qc_status": "pass",
            "qc_reason": "",
            "exclusion_id": "",
            "exclusion_reason": "",
            "grouping_values": "{}",
            "warnings": "[]",
            "errors": "[]",
            "roi_reference": "root_ref:mvpa_inputs/rois/SeedA.nii",
            "generator_version": "synthetic-formula-v1",
            "software_version": "research-platform-alpha",
            "derivation_id": "toy-derivation-v1",
            "holdout_id": "none",
            "noise_values": "",
            "noise_feature_count": "",
            "noise_voxel_order": "",
            "noise_voxel_index_hash": "",
            "noise_feature_space_id": "",
            "noise_roi_definition_id": "",
            "noise_value_kind": "",
            "noise_estimation_scope": "",
            "noise_source": "",
            "adapter_note": adapter_note,
        }

    def _write_ready_workspace(self, workspace_root: Path) -> tuple[Path, Path]:
        project_root = self._write_workspace(workspace_root)
        self._write_ready_mvpa_config(project_root)
        self._write_bundle(project_root)
        table = self._write_pattern_table(workspace_root)
        return project_root, table

    def _run_cli(
        self,
        args: list[str],
        *,
        workspace_root: Path,
    ) -> tuple[int, dict[str, object]]:
        output = io.StringIO()
        with mock.patch.dict(
            os.environ,
            {"RESEARCH_PLATFORM_ROOT": str(workspace_root)},
            clear=False,
        ):
            with redirect_stdout(output):
                exit_code = main(args)
        return exit_code, json.loads(output.getvalue())

    def _run_cli_error(self, args: list[str], *, workspace_root: Path) -> dict[str, object]:
        with mock.patch.dict(
            os.environ,
            {"RESEARCH_PLATFORM_ROOT": str(workspace_root)},
            clear=False,
        ):
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

    def _tree_bytes(self, root: Path) -> dict[str, bytes]:
        return {
            path.relative_to(root).as_posix(): path.read_bytes()
            for path in sorted(root.rglob("*"))
            if path.is_file()
        }

    def _runtime_root(self, workspace_root: Path) -> Path:
        return workspace_root / "artifacts" / ".research-platform" / "mvpa" / MVPA_SET

    def _canonical_args(self, command: str, *, execute: bool = False) -> list[str]:
        args = [
            "analysis",
            "mvpa",
            command,
            MVPA_SET,
            "--project",
            PROJECT_NAME,
        ]
        if command in {"doctor", "plan", "run"}:
            args.extend(["--bundle", BUNDLE_NAME])
        if execute:
            args.append("--execute")
        return args

    def test_materialized_scaffold_is_one_yaml_nonmutating_and_replace_is_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            project_root = self._write_workspace(workspace_root)
            before = self._tree_digest(workspace_root)

            dry_code, dry_payload = self._run_cli(
                [
                    "analysis",
                    "mvpa",
                    "init",
                    "starter",
                    "--project",
                    PROJECT_NAME,
                    "--template",
                    "materialized-crossnobis",
                    "--dry-run",
                ],
                workspace_root=workspace_root,
            )
            after_dry = self._tree_digest(workspace_root)
            config_path = project_root / "config" / "analysis" / "mvpa" / "starter.yaml"

            init_code, init_payload = self._run_cli(
                ["analysis", "mvpa", "init", "starter", "--project", PROJECT_NAME],
                workspace_root=workspace_root,
            )
            generated_document = load_yaml(config_path)
            validate_code, validate_payload = self._run_cli(
                ["analysis", "mvpa", "validate", "starter", "--project", PROJECT_NAME],
                workspace_root=workspace_root,
            )
            first_bytes = config_path.read_bytes()
            collision_code, collision_payload = self._run_cli(
                ["analysis", "mvpa", "init", "starter", "--project", PROJECT_NAME],
                workspace_root=workspace_root,
            )
            force_code, force_payload = self._run_cli(
                [
                    "analysis",
                    "mvpa",
                    "init",
                    "starter",
                    "--project",
                    PROJECT_NAME,
                    "--force",
                ],
                workspace_root=workspace_root,
            )
            second_bytes = config_path.read_bytes()

        # The temporary workspace is gone here; inspect the deterministic bytes directly.
        config_text = first_bytes.decode("utf-8")
        self.assertEqual(dry_code, 0)
        self.assertFalse(dry_payload["executed"])
        self.assertEqual(before, after_dry)
        self.assertEqual(init_code, 0)
        self.assertTrue(init_payload["executed"])
        self.assertEqual(len(init_payload["planned_output_files"]), 1)
        self.assertEqual(validate_code, 0)
        self.assertTrue(validate_payload["schema_valid"])
        self.assertEqual(collision_code, 1)
        self.assertFalse(collision_payload["valid"])
        self.assertTrue(collision_payload["existing_file_collisions"])
        self.assertEqual(force_code, 0)
        self.assertTrue(force_payload["executed"])
        self.assertEqual(first_bytes, second_bytes)
        self.assertIn("mode: exact_units", config_text)
        self.assertIn("backend: materialized_pattern_table", config_text)
        self.assertIn("existing_output: fail", config_text)
        self.assertNotIn("subjects:", config_text)
        self.assertNotIn("sessions:", config_text)
        self.assertNotIn("runs:", config_text)
        self.assertNotIn("pe_image", config_text)
        self.assertNotIn("mask_template", config_text)
        scaffold = generated_document["mvpa_set"]
        self.assertEqual(scaffold["unit_selection"]["mode"], "exact_units")
        self.assertEqual(scaffold["runtime"], {"existing_output": "fail"})
        self.assertEqual(len(scaffold["conditions"]), 2)
        self.assertEqual(len(scaffold["condition_pairs"]), 1)

    def test_parser_advertises_templates_and_bundle_as_the_only_selection_flag(self) -> None:
        parser = _build_parser()
        parsed = parser.parse_args(
            [
                "analysis",
                "mvpa",
                "run",
                MVPA_SET,
                "--project",
                PROJECT_NAME,
                "--bundle",
                BUNDLE_NAME,
            ]
        )
        self.assertEqual(parsed.bundle, BUNDLE_NAME)
        for name in ("subject", "session", "run_id", "condition", "roi", "feature", "noise"):
            self.assertFalse(hasattr(parsed, name))

        analysis_action = next(action for action in parser._actions if action.dest == "command")
        analysis_parser = analysis_action.choices["analysis"]
        mvpa_action = next(
            action for action in analysis_parser._actions if action.dest == "analysis_command"
        )
        mvpa_parser = mvpa_action.choices["mvpa"]
        command_action = next(
            action for action in mvpa_parser._actions if action.dest == "analysis_mvpa_command"
        )
        init_parser = command_action.choices["init"]
        template_action = next(
            action for action in init_parser._actions if "--template" in action.option_strings
        )
        self.assertEqual(template_action.default, "materialized-crossnobis")
        self.assertEqual(
            tuple(template_action.choices),
            ("materialized-crossnobis", "fsl-feat-crossnobis", "distance-rdm"),
        )

    def test_bundle_lifecycle_is_nonmutating_and_preserves_exact_units(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            self._write_ready_workspace(workspace_root)
            before = self._tree_digest(workspace_root)

            validate_code, validate_payload = self._run_cli(
                self._canonical_args("validate"),
                workspace_root=workspace_root,
            )
            doctor_code, doctor_payload = self._run_cli(
                self._canonical_args("doctor"),
                workspace_root=workspace_root,
            )
            plan_code, plan_payload = self._run_cli(
                self._canonical_args("plan"),
                workspace_root=workspace_root,
            )
            run_code, run_payload = self._run_cli(
                self._canonical_args("run"),
                workspace_root=workspace_root,
            )
            after = self._tree_digest(workspace_root)

        self.assertEqual(validate_code, 0)
        self.assertTrue(validate_payload["schema_valid"])
        self.assertEqual(doctor_code, 0)
        self.assertTrue(doctor_payload["schema_valid"])
        self.assertTrue(doctor_payload["bundle_valid"])
        self.assertTrue(doctor_payload["plan_valid"])
        self.assertTrue(doctor_payload["ready_for_materialization"])
        self.assertTrue(doctor_payload["ready_for_execution"])
        self.assertFalse(doctor_payload["executed"])
        self.assertEqual(plan_code, 0)
        self.assertTrue(plan_payload["plan_valid"])
        self.assertEqual(run_code, 0)
        self.assertFalse(run_payload["executed"])
        self.assertEqual(
            {
                record["relative_path"]
                for record in run_payload["planned_outputs"].values()
            },
            {
                item.relative_path
                for item in runtime_output_specs("prepared_features")
            },
        )
        self.assertEqual(before, after)

        expected_keys = [
            ["sub-beta", "ses-01", "run-02"],
            ["sub-alpha", "ses-01", "run-01"],
            ["sub-beta", "ses-01", "run-01"],
            ["sub-alpha", "ses-01", "run-03"],
        ]
        for payload in (doctor_payload, plan_payload, run_payload):
            bundle = payload["bundle"]
            units = bundle["included_units"]
            self.assertEqual(
                [
                    [row["values"][column] for column in bundle["unit_key_columns"]]
                    for row in units
                ],
                expected_keys,
            )
            self.assertEqual(
                [row["values"]["adapter_note"] for row in units],
                ["first", "second", "third", "fourth"],
            )
            self.assertEqual(len(bundle["excluded_units"]), 1)
            self.assertEqual(
                bundle["excluded_units"][0]["values"]["subject_id"],
                "sub-gamma",
            )
        plan_rows = plan_payload["pattern_rows"]
        self.assertEqual(len(plan_rows), 8)
        self.assertEqual(
            [(row["subject_id"], row["run_id"], row["condition_id"]) for row in plan_rows],
            [
                (subject, run, condition)
                for subject, _session, run in (tuple(values) for values in expected_keys)
                for condition in ("condition_a", "condition_b")
            ],
        )
        self.assertEqual(
            [row["cross_validation_label"] for row in plan_rows],
            [row["run_id"] for row in plan_rows],
        )

    def test_real_execute_is_atomic_deterministic_and_refuses_same_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            parent = Path(tmp_dir)
            roots = [parent / "first", parent / "second"]
            runtime_trees: list[dict[str, bytes]] = []
            execute_payloads: list[dict[str, object]] = []
            for workspace_root in roots:
                self._write_ready_workspace(workspace_root)
                execute_code, execute_payload = self._run_cli(
                    self._canonical_args("run", execute=True),
                    workspace_root=workspace_root,
                )
                runtime_root = self._runtime_root(workspace_root)
                self.assertEqual(execute_code, 0)
                self.assertTrue(execute_payload["executed"])
                self.assertTrue(runtime_root.is_dir())
                expected_paths = {
                    item.relative_path
                    for item in runtime_output_specs("prepared_features")
                }
                tree = self._tree_bytes(runtime_root)
                self.assertEqual(set(tree), expected_paths)
                for relative_path, content in tree.items():
                    self.assertFalse(
                        published_text_contains_local_path_reference(content.decode("utf-8")),
                        relative_path,
                    )
                manifest = json.loads(tree[MANIFEST_RELATIVE_PATH])
                self.assertEqual(manifest["status"], "succeeded")
                self.assertEqual(manifest["mvpa_set"], MVPA_SET)
                self.assertEqual(manifest["project"], PROJECT_NAME)
                self.assertEqual(manifest["representation_kind"], "prepared_features")
                self.assertEqual(manifest["selection"]["selected_unit_count"], 4)
                self.assertEqual(manifest["selection"]["excluded_unit_count"], 1)
                self.assertEqual(manifest["errors"], [])
                self.assertEqual(
                    manifest["cross_validation"]["label_column"],
                    "cross_validation_label",
                )
                self.assertEqual(
                    manifest["source_tables"][0]["planned_sha256"],
                    manifest["source_tables"][0]["loaded_sha256"],
                )
                for record in manifest["outputs"]:
                    relative_path = record["relative_path"]
                    self.assertEqual(
                        record["sha256"],
                        hashlib.sha256(tree[relative_path]).hexdigest(),
                    )
                distance_rows = tuple(
                    csv.DictReader(
                        io.StringIO(
                            tree["analysis/prepared-distances/distances.tsv"].decode(
                                "utf-8"
                            )
                        ),
                        delimiter="\t",
                    )
                )
                self.assertEqual(
                    {
                        row["subject_id"]: float(row["distance"])
                        for row in distance_rows
                    },
                    {"sub-alpha": 10.0, "sub-beta": 12.0},
                )
                runtime_trees.append(tree)
                execute_payloads.append(execute_payload)

            self.assertEqual(runtime_trees[0], runtime_trees[1])

            first_root = self._runtime_root(roots[0])
            before_rerun = self._tree_digest(first_root)
            rerun_code, rerun_payload = self._run_cli(
                self._canonical_args("run", execute=True),
                workspace_root=roots[0],
            )
            after_rerun = self._tree_digest(first_root)

        self.assertEqual(rerun_code, 1)
        self.assertFalse(rerun_payload["executed"])
        self.assertTrue(any("existing_output=fail" in error for error in rerun_payload["errors"]))
        self.assertEqual(before_rerun, after_rerun)
        self.assertEqual(execute_payloads[0]["row_counts"], execute_payloads[1]["row_counts"])

    def test_source_mutation_after_exact_planning_fails_before_runtime_writes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            _project_root, table = self._write_ready_workspace(workspace_root)
            original_materialize, original_image_writer = (
                core_cli._mvpa_pattern_extraction_runtime_functions()
            )

            def mutate_then_materialize(plan: object, *, load_noise: bool = False) -> object:
                table.write_bytes(table.read_bytes() + b"\n")
                return original_materialize(plan, load_noise=load_noise)

            with mock.patch.object(
                core_cli,
                "_mvpa_pattern_extraction_runtime_functions",
                return_value=(mutate_then_materialize, original_image_writer),
            ):
                execute_code, execute_payload = self._run_cli(
                    self._canonical_args("run", execute=True),
                    workspace_root=workspace_root,
                )

            runtime_root = self._runtime_root(workspace_root)
            transaction_remnants = tuple(
                (workspace_root / "artifacts" / ".research-platform" / "mvpa").glob(
                    f".{MVPA_SET}.*"
                )
            )

        self.assertEqual(execute_code, 1)
        self.assertFalse(execute_payload["executed"])
        self.assertTrue(
            any(
                "sha-256 mismatch" in error.casefold()
                or "digest mismatch" in error.casefold()
                for error in execute_payload["errors"]
            )
        )
        self.assertFalse(runtime_root.exists())
        self.assertEqual(transaction_remnants, ())

    def test_success_manifest_uses_the_exact_planned_configuration_digest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            project_root, _table = self._write_ready_workspace(workspace_root)
            config_path = (
                project_root
                / "config"
                / "analysis"
                / "mvpa"
                / f"{MVPA_SET}.yaml"
            )
            planned_digest = hashlib.sha256(config_path.read_bytes()).hexdigest()
            runtime_functions = core_cli._mvpa_analysis_runtime_functions()
            summarize = runtime_functions[2]

            def mutate_config_after_computation(*args: object, **kwargs: object) -> object:
                result = summarize(*args, **kwargs)
                config_path.write_bytes(config_path.read_bytes() + b"\n")
                return result

            with mock.patch.object(
                core_cli,
                "_mvpa_analysis_runtime_functions",
                return_value=(
                    runtime_functions[0],
                    runtime_functions[1],
                    mutate_config_after_computation,
                    *runtime_functions[3:],
                ),
            ):
                execute_code, execute_payload = self._run_cli(
                    self._canonical_args("run", execute=True),
                    workspace_root=workspace_root,
                )

        self.assertEqual(execute_code, 0)
        self.assertEqual(
            execute_payload["manifest"]["mvpa_config_sha256"],
            planned_digest,
        )

    def test_bundle_digest_uses_the_exact_bytes_resolved_for_the_plan(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            project_root, _table = self._write_ready_workspace(workspace_root)
            bundle_path = (
                project_root
                / "config"
                / "analysis"
                / "bundles"
                / f"{BUNDLE_NAME}.yaml"
            )
            resolved_bytes = bundle_path.read_bytes()
            resolved_digest = hashlib.sha256(resolved_bytes).hexdigest()
            resolve_bundle = core_cli._resolve_analysis_bundle_for_cli

            def resolve_then_mutate(*args: object, **kwargs: object) -> object:
                resolution = resolve_bundle(*args, **kwargs)
                bundle_path.write_bytes(resolved_bytes + b"\n# synthetic post-resolution drift\n")
                return resolution

            with mock.patch.object(
                core_cli,
                "_resolve_analysis_bundle_for_cli",
                side_effect=resolve_then_mutate,
            ):
                plan_code, plan_payload = self._run_cli(
                    self._canonical_args("plan"),
                    workspace_root=workspace_root,
                )

            changed_digest = hashlib.sha256(bundle_path.read_bytes()).hexdigest()

        self.assertEqual(plan_code, 0)
        self.assertNotEqual(changed_digest, resolved_digest)
        self.assertEqual(
            plan_payload["bundle"]["digests"]["bundle_config_sha256"],
            resolved_digest,
        )

    def test_missing_and_mismatched_configuration_errors_are_concise(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            project_root = self._write_workspace(workspace_root)

            missing_mvpa = self._run_cli_error(
                ["analysis", "mvpa", "validate", "missing", "--project", PROJECT_NAME],
                workspace_root=workspace_root,
            )
            self._write_ready_mvpa_config(project_root)
            missing_bundle = self._run_cli_error(
                self._canonical_args("doctor"),
                workspace_root=workspace_root,
            )
            self._write_bundle(project_root, mvpa_set="different-mvpa")
            mismatch = self._run_cli_error(
                self._canonical_args("plan"),
                workspace_root=workspace_root,
            )

        self.assertEqual(missing_mvpa["name"], "missing")
        self.assertIn("project/project-demo-mvpa/config/analysis/mvpa/missing.yaml", missing_mvpa["expected_path"])
        self.assertEqual(
            missing_mvpa["next_step"],
            "rp analysis mvpa init missing --project project-demo-mvpa --template materialized-crossnobis",
        )
        self.assertEqual(missing_bundle["name"], BUNDLE_NAME)
        self.assertEqual(
            missing_bundle["next_step"],
            f"rp analysis bundle init {BUNDLE_NAME} --project {PROJECT_NAME}",
        )
        self.assertEqual(mismatch["expected_mvpa_set"], MVPA_SET)
        self.assertEqual(mismatch["observed_mvpa_set"], "different-mvpa")
        self.assertEqual(
            mismatch["next_step"],
            f"rp analysis bundle show {BUNDLE_NAME} --project {PROJECT_NAME}",
        )


if __name__ == "__main__":
    unittest.main()
