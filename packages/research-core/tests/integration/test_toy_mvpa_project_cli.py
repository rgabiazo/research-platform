from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import csv
from fractions import Fraction
from hashlib import sha256
import io
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


CORE_PACKAGE_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
ANALYSIS_PACKAGE_ROOT = WORKSPACE_ROOT / "packages" / "research-analysis"
BIDS_PACKAGE_ROOT = WORKSPACE_ROOT / "packages" / "research-bids"
HPC_PACKAGE_ROOT = WORKSPACE_ROOT / "packages" / "research-hpc"
NEURO_PACKAGE_ROOT = WORKSPACE_ROOT / "packages" / "research-neuro"
sys.path.insert(0, str(CORE_PACKAGE_ROOT / "src"))
sys.path.insert(0, str(ANALYSIS_PACKAGE_ROOT / "src"))
sys.path.insert(0, str(BIDS_PACKAGE_ROOT / "src"))
sys.path.insert(0, str(HPC_PACKAGE_ROOT / "src"))
sys.path.insert(0, str(NEURO_PACKAGE_ROOT / "src"))

from research_platform.core.cli import main
from research_platform.neuro._roi_path_safety import (
    published_text_contains_local_path_reference,
)
from research_platform.neuro.mvpa.pattern_sources import REPRESENTATION_PREPARED_FEATURES
from research_platform.neuro.mvpa.runtime_transaction import runtime_output_specs


PATTERN_TABLE = (
    WORKSPACE_ROOT
    / "datasets"
    / "ds-mvpa-example"
    / "patterns"
    / "toy_crossnobis_patterns.tsv"
)
RUNTIME_RELATIVE_ROOT = Path(".research-platform/mvpa/toy-crossnobis")
EXACT_UNITS = (
    ("sub-toy01", "exampletask", "run-01"),
    ("sub-toy01", "exampletask", "run-02"),
    ("sub-toy02", "exampletask", "run-01"),
    ("sub-toy02", "exampletask", "run-02"),
)
EXPECTED_DISTANCE_FRACTIONS = {
    ("sub-toy01", "SeedA"): Fraction(61172, 1995),
    ("sub-toy01", "SeedB"): Fraction(25327351, 493350),
    ("sub-toy02", "SeedA"): Fraction(1284, 23),
    ("sub-toy02", "SeedB"): Fraction(2381, 30),
}


class ToyMvpaProjectCliIntegrationTests(unittest.TestCase):
    def test_real_rp_handlers_run_the_complete_toy_crossnobis_lifecycle(self) -> None:
        with tempfile.TemporaryDirectory(prefix="rp-toy-mvpa-integration-") as temp_dir:
            temp_root = Path(temp_dir)
            first_artifacts = temp_root / "first"
            second_artifacts = temp_root / "second"
            first_artifacts.mkdir()
            second_artifacts.mkdir()

            first = self._run_lifecycle(first_artifacts)
            second = self._run_lifecycle(second_artifacts)
            first_runtime = first_artifacts / RUNTIME_RELATIVE_ROOT
            second_runtime = second_artifacts / RUNTIME_RELATIVE_ROOT

            self._assert_plans(first)
            self._assert_runtime(first_runtime)
            self._assert_runtime(second_runtime)
            self.assertEqual(self._tree_snapshot(first_runtime), self._tree_snapshot(second_runtime))
            self.assertEqual(
                self._relative_file_bytes(first_runtime),
                self._relative_file_bytes(second_runtime),
            )

            original = self._tree_snapshot(first_artifacts)
            code, output = self._invoke(
                self._mvpa_command("run", execute=True),
                artifacts_root=first_artifacts,
            )
            self.assertNotEqual(code, 0)
            self.assertIn("already exists", output)
            self.assertEqual(self._tree_snapshot(first_artifacts), original)
            self.assertFalse(self._transaction_remnants(temp_root))

    def test_reference_distances_are_derived_without_production_distance_code(self) -> None:
        independently_calculated = {
            (f"sub-toy{subject:02d}", f"Seed{chr(64 + roi)}"): self._reference_distance(
                subject_index=subject,
                roi_index=roi,
            )
            for subject in (1, 2)
            for roi in (1, 2)
        }
        self.assertEqual(independently_calculated, EXPECTED_DISTANCE_FRACTIONS)
        self.assertEqual(len(set(independently_calculated.values())), 4)
        self.assertTrue(all(value > 0 for value in independently_calculated.values()))

    def _run_lifecycle(self, artifacts_root: Path) -> dict[str, object]:
        commands = (
            (
                "bundle_validate",
                [
                    "analysis",
                    "bundle",
                    "validate",
                    "toy-crossnobis",
                    "--project",
                    "project-example",
                ],
            ),
            (
                "bundle_doctor",
                [
                    "analysis",
                    "bundle",
                    "doctor",
                    "toy-crossnobis",
                    "--project",
                    "project-example",
                ],
            ),
            (
                "bundle_plan",
                [
                    "analysis",
                    "bundle",
                    "plan",
                    "toy-crossnobis",
                    "--project",
                    "project-example",
                ],
            ),
            ("mvpa_validate", self._mvpa_command("validate")),
            ("mvpa_doctor", self._mvpa_command("doctor")),
            ("mvpa_plan", self._mvpa_command("plan")),
            ("mvpa_run_plan", self._mvpa_command("run")),
            ("mvpa_execute", self._mvpa_command("run", execute=True)),
        )
        payloads: dict[str, object] = {}
        for name, command in commands:
            before = self._tree_snapshot(artifacts_root)
            code, output = self._invoke(command, artifacts_root=artifacts_root)
            self.assertEqual(code, 0, f"{name}: {output}")
            payload = json.loads(output)
            payloads[name] = payload
            if name != "mvpa_execute":
                self.assertEqual(self._tree_snapshot(artifacts_root), before, name)
                self.assertFalse(payload.get("executed", False), name)
            self.assertFalse(self._transaction_remnants(artifacts_root.parent), name)

        self.assertTrue(payloads["bundle_validate"]["valid"])
        self.assertTrue(payloads["bundle_doctor"]["ready_for_planning"])
        self.assertTrue(payloads["mvpa_validate"]["schema_valid"])
        self.assertTrue(payloads["mvpa_doctor"]["ready_for_execution"])
        self.assertTrue(payloads["mvpa_plan"]["plan_valid"])
        self.assertEqual(payloads["mvpa_run_plan"]["mode"], "plan")
        self.assertEqual(payloads["mvpa_execute"]["mode"], "execute")
        self.assertTrue(payloads["mvpa_execute"]["executed"])
        return payloads

    def _assert_plans(self, payloads: dict[str, object]) -> None:
        bundle_plan = payloads["bundle_plan"]
        included = bundle_plan["units"]["included"]
        self.assertEqual(bundle_plan["counts"]["unit_count"], 4)
        self.assertEqual(bundle_plan["counts"]["excluded_units"], 0)
        self.assertEqual(
            [
                (
                    row["values"]["subject_id"],
                    row["values"]["task_id"],
                    row["values"]["run_id"],
                )
                for row in included
            ],
            list(EXACT_UNITS),
        )
        self.assertNotIn("session_id", bundle_plan["source_batch"]["columns"])

        mvpa_plan = payloads["mvpa_plan"]
        self.assertEqual(mvpa_plan["counts"]["included_units"], 4)
        self.assertEqual(mvpa_plan["counts"]["pattern_rows"], 16)
        self.assertEqual(mvpa_plan["counts"]["conditions"], 2)
        self.assertEqual(mvpa_plan["counts"]["roi_identities"], 2)
        rows = mvpa_plan["pattern_rows"]
        self.assertEqual(len(rows), 16)
        self.assertEqual(
            {
                (
                    row["subject_id"],
                    row["task_id"],
                    row["run_id"],
                    row["condition_id"],
                    row["backend_metadata"]["roi_label"],
                )
                for row in rows
            },
            {
                (*unit, condition, roi)
                for unit in EXACT_UNITS
                for condition in ("condition_a", "condition_b")
                for roi in ("SeedA", "SeedB")
            },
        )
        self.assertEqual({row["session_id"] for row in rows}, {None})
        self.assertTrue(all(row["cross_validation_label"] == row["run_id"] for row in rows))
        self.assertEqual({row["representation_kind"] for row in rows}, {"prepared_features"})
        self.assertEqual(
            mvpa_plan["pattern_source_summaries"][0]["source_sha256"],
            sha256(PATTERN_TABLE.read_bytes()).hexdigest(),
        )

    def _assert_runtime(self, runtime_root: Path) -> None:
        expected_paths = {
            spec.relative_path for spec in runtime_output_specs(REPRESENTATION_PREPARED_FEATURES)
        }
        observed_paths = set(self._relative_file_bytes(runtime_root))
        self.assertEqual(observed_paths, expected_paths)
        self.assertEqual(len(observed_paths), 14)

        materialized_rows = self._read_tsv(
            runtime_root / "neuro/pattern-materialization/patterns.tsv"
        )
        self.assertEqual(len(materialized_rows), 16)
        self.assertEqual({row["condition_id"] for row in materialized_rows}, {"condition_a", "condition_b"})
        self.assertEqual({row["roi_label"] for row in materialized_rows}, {"SeedA", "SeedB"})
        self.assertEqual({row["feature_count"] for row in materialized_rows}, {"5"})
        self.assertEqual({row["session_id"] for row in materialized_rows}, {""})
        self.assertEqual({row["usable"] for row in materialized_rows}, {"true"})
        self.assertEqual({row["status"] for row in materialized_rows}, {"ok"})
        self.assertEqual({row["qc_status"] for row in materialized_rows}, {"pass"})
        self.assertEqual({row["mean_centering_applied"] for row in materialized_rows}, {"false"})
        self.assertEqual({row["mean_centering_scope"] for row in materialized_rows}, {"none"})
        self.assertEqual({row["noise_status"] for row in materialized_rows}, {"ok"})
        self.assertEqual({row["noise_usable"] for row in materialized_rows}, {"true"})
        self.assertEqual({row["noise_value_kind"] for row in materialized_rows}, {"variance"})
        self.assertEqual(
            {row["noise_estimation_scope"] for row in materialized_rows},
            {"exact_unit_roi"},
        )
        self.assertTrue(
            all(row["cross_validation_label"] == row["run_id"] for row in materialized_rows)
        )

        distance_rows = self._read_tsv(
            runtime_root / "analysis/prepared-distances/distances.tsv"
        )
        self.assertEqual(len(distance_rows), 4)
        observed_distances: dict[tuple[str, str], float] = {}
        for row in distance_rows:
            key = (row["subject_id"], row["roi_label"])
            value = float(row["distance"])
            observed_distances[key] = value
            self.assertTrue(math.isfinite(value))
            self.assertGreater(value, 0.0)
            self.assertEqual(row["condition_id_a"], "condition_a")
            self.assertEqual(row["condition_id_b"], "condition_b")
            self.assertEqual(row["metric"], "crossnobis")
            self.assertEqual(row["engine_name"], "native_reference")
            self.assertEqual(row["normalization_method"], "diagonal")
            self.assertEqual(row["cv_unit_count"], "2")
            self.assertEqual(row["feature_count"], "5")
            self.assertEqual(row["observation_count"], "4")
        self.assertEqual(set(observed_distances), set(EXPECTED_DISTANCE_FRACTIONS))
        for key, expected in EXPECTED_DISTANCE_FRACTIONS.items():
            self.assertAlmostEqual(observed_distances[key], float(expected), places=12)
        self.assertEqual(len(set(observed_distances.values())), 4)

        summary_rows = self._read_tsv(
            runtime_root / "analysis/prepared-summaries/summaries.tsv"
        )
        self.assertEqual(len(summary_rows), 4)
        for row in summary_rows:
            expected = float(EXPECTED_DISTANCE_FRACTIONS[(row["subject_id"], row["roi_label"])])
            self.assertEqual(row["n"], "1")
            self.assertAlmostEqual(float(row["mean_distance"]), expected, places=12)
            self.assertAlmostEqual(float(row["min_distance"]), expected, places=12)
            self.assertAlmostEqual(float(row["max_distance"]), expected, places=12)
            self.assertEqual(float(row["std_distance"]), 0.0)
            self.assertEqual(float(row["sem_distance"]), 0.0)

        manifest_path = runtime_root / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["status"], "succeeded")
        self.assertEqual(manifest["project"], "project-example")
        self.assertEqual(manifest["mvpa_set"], "toy-crossnobis")
        self.assertEqual(manifest["adapter"]["names"], ["materialized_pattern_table"])
        self.assertEqual(manifest["representation_kind"], "prepared_features")
        self.assertEqual(manifest["selected_unit_count"], 4)
        self.assertEqual(manifest["excluded_unit_count"], 0)
        self.assertEqual(manifest["cross_validation"]["unit"], "run")
        self.assertEqual(
            manifest["cross_validation"]["label_column"],
            "cross_validation_label",
        )
        self.assertEqual(manifest["centering"], {"enabled": False, "scope": "none"})
        self.assertEqual(manifest["noise"]["methods"], ["diagonal"])
        self.assertEqual(manifest["noise"]["value_semantics"], "variance")
        self.assertEqual(manifest["noise"]["policies"], ["strict"])
        self.assertEqual(manifest["row_counts"]["source_pattern_rows"], 16)
        self.assertEqual(manifest["row_counts"]["distance_rows"], 4)
        self.assertEqual(manifest["row_counts"]["summary_rows"], 4)
        source = manifest["source_tables"][0]
        source_digest = sha256(PATTERN_TABLE.read_bytes()).hexdigest()
        self.assertEqual(source["planned_sha256"], source_digest)
        self.assertEqual(source["loaded_sha256"], source_digest)
        self.assertEqual(
            source["portable_reference"],
            "root_ref:mvpa_example/patterns/toy_crossnobis_patterns.tsv",
        )
        materialization_provenance = json.loads(
            (
                runtime_root
                / "neuro/pattern-materialization/provenance.json"
            ).read_text(encoding="utf-8")
        )
        provenance_source = materialization_provenance["input_provenance"]["sources"][0]
        self.assertEqual(provenance_source["source_sha256"], source_digest)
        self.assertEqual(
            provenance_source["source_reference"],
            "root_ref:mvpa_example/patterns/toy_crossnobis_patterns.tsv",
        )
        self.assertEqual(
            manifest["thresholds"]["event_thresholds"],
            {
                "min_events_per_condition_per_run": 1,
                "min_runs_per_condition": 2,
            },
        )
        self.assertEqual(
            {
                (
                    identity["roi_source_name"],
                    identity["roi_label"],
                    identity["feature_count"],
                    identity["feature_space_id"],
                    identity["roi_definition_id"],
                )
                for identity in manifest["feature_identities"]
            },
            {
                (
                    "toy-rois",
                    "SeedA",
                    5,
                    "toy-feature-space:SeedA:v1",
                    "toy-roi-definition:SeedA:v1",
                ),
                (
                    "toy-rois",
                    "SeedB",
                    5,
                    "toy-feature-space:SeedB:v1",
                    "toy-roi-definition:SeedB:v1",
                ),
            },
        )

        manifest_outputs = {row["relative_path"]: row for row in manifest["outputs"]}
        self.assertEqual(set(manifest_outputs), expected_paths - {"manifest.json"})
        for relative, record in manifest_outputs.items():
            self.assertEqual(
                record["sha256"],
                sha256((runtime_root / relative).read_bytes()).hexdigest(),
            )

        for relative, contents in self._relative_file_bytes(runtime_root).items():
            self.assertFalse(
                published_text_contains_local_path_reference(relative),
                relative,
            )
            self.assertFalse(
                published_text_contains_local_path_reference(contents.decode("utf-8")),
                relative,
            )
        self.assertFalse(self._transaction_remnants(runtime_root.parent.parent.parent))

    @staticmethod
    def _reference_distance(*, subject_index: int, roi_index: int) -> Fraction:
        total = Fraction(0)
        for feature_index in range(1, 6):
            delta_run_1 = -Fraction(
                (subject_index + roi_index) * feature_index + 1,
                2,
            )
            delta_run_2 = -Fraction(
                (subject_index + roi_index) * feature_index + 2,
                2,
            )
            variance_run_1 = (
                Fraction(1)
                + Fraction(subject_index, 4)
                + Fraction(roi_index, 2)
                + Fraction(1, 4)
                + Fraction(feature_index, 8)
            )
            variance_run_2 = (
                Fraction(1)
                + Fraction(subject_index, 4)
                + Fraction(roi_index, 2)
                + Fraction(2, 4)
                + Fraction(feature_index, 8)
            )
            mean_variance = (variance_run_1 + variance_run_2) / 2
            total += delta_run_1 * delta_run_2 / mean_variance
        return total

    @staticmethod
    def _mvpa_command(action: str, *, execute: bool = False) -> list[str]:
        command = [
            "analysis",
            "mvpa",
            action,
            "toy-crossnobis",
            "--project",
            "project-example",
        ]
        if action != "validate":
            command.extend(("--bundle", "toy-crossnobis"))
        if execute:
            command.append("--execute")
        return command

    def _invoke(self, args: list[str], *, artifacts_root: Path) -> tuple[int, str]:
        stdout = io.StringIO()
        stderr = io.StringIO()
        env = {
            "RESEARCH_PLATFORM_ROOT": str(WORKSPACE_ROOT),
            "ARTIFACTS_ROOT": str(artifacts_root),
        }
        with mock.patch.dict(os.environ, env, clear=False):
            with redirect_stdout(stdout), redirect_stderr(stderr):
                try:
                    code = main(args)
                except SystemExit as exc:
                    code = int(exc.code) if isinstance(exc.code, int) else 1
                    if exc.code not in (None, 0):
                        stderr.write(str(exc.code))
        return int(code or 0), stdout.getvalue() + stderr.getvalue()

    @staticmethod
    def _read_tsv(path: Path) -> list[dict[str, str]]:
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle, delimiter="\t"))

    @staticmethod
    def _relative_file_bytes(root: Path) -> dict[str, bytes]:
        return {
            path.relative_to(root).as_posix(): path.read_bytes()
            for path in sorted(root.rglob("*"), key=str)
            if path.is_file()
        }

    @staticmethod
    def _tree_snapshot(root: Path) -> tuple[tuple[str, str], ...]:
        rows: list[tuple[str, str]] = []
        for path in sorted(root.rglob("*"), key=str):
            relative = path.relative_to(root).as_posix()
            if path.is_symlink():
                rows.append((relative, f"symlink:{os.readlink(path)}"))
            elif path.is_file():
                rows.append((relative, sha256(path.read_bytes()).hexdigest()))
            else:
                rows.append((relative, "directory"))
        return tuple(rows)

    @staticmethod
    def _transaction_remnants(root: Path) -> list[Path]:
        return [
            path
            for path in root.rglob("*")
            if (
                path.name == ".toy-crossnobis.claim"
                or (path.name.startswith(".toy-crossnobis.") and path.name.endswith(".tmp"))
                or (path.name.startswith(".manifest.json.") and path.name.endswith(".tmp"))
            )
        ]


if __name__ == "__main__":
    unittest.main()
