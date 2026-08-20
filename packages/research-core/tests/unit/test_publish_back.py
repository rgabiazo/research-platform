from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PACKAGE_ROOT / "src"))

from research_platform.core.publish_back import (
    build_publish_back_plan,
    build_publish_back_scaffold,
    resolve_publish_back_policy,
)


class PublishBackTests(unittest.TestCase):
    def test_build_publish_back_scaffold_defaults_to_never(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            run_root = workspace_root / "artifacts" / "runs" / "unit-run"

            scaffold = build_publish_back_scaffold(workspace_root=workspace_root, run_root=run_root)

        self.assertEqual(
            scaffold,
            {
                "default_policy": "never",
                "plan_path": "artifacts/runs/unit-run/publish-back-plan.yaml",
            },
        )

    def test_resolve_publish_back_policy_rejects_unknown_policy(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported publish-back policy"):
            resolve_publish_back_policy("sometimes")

    def test_build_publish_back_plan_returns_empty_plan_for_empty_items(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            run_root = workspace_root / "artifacts" / "runs" / "unit-run"
            run_root.mkdir(parents=True, exist_ok=True)

            plan = build_publish_back_plan(
                workspace_root=workspace_root,
                run_root=run_root,
                manifest={"run_id": "unit-run", "publish_back": {"default_policy": "never", "items": []}},
            )

        self.assertEqual(plan["run_id"], "unit-run")
        self.assertEqual(plan["default_policy"], "never")
        self.assertNotIn("items", plan)
        self.assertEqual(plan["summary"], {"total_items": 0, "actionable_items": 0})

    def test_build_publish_back_plan_uses_default_policy_when_item_policy_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            run_root = workspace_root / "artifacts" / "runs" / "unit-run"
            source_path = run_root / "outputs" / "report.txt"
            destination_path = workspace_root / "canonical" / "report.txt"
            source_path.parent.mkdir(parents=True, exist_ok=True)
            source_path.write_text("artifact", encoding="utf-8")

            plan = build_publish_back_plan(
                workspace_root=workspace_root,
                run_root=run_root,
                manifest={
                    "run_id": "unit-run",
                    "publish_back": {
                        "default_policy": "if_absent",
                        "items": [{"source": "artifacts/runs/unit-run/outputs/report.txt", "destination": "canonical/report.txt"}],
                    },
                },
            )

        self.assertEqual(plan["items"][0]["policy"], "if_absent")
        self.assertEqual(plan["items"][0]["action"], "copy")
        self.assertEqual(plan["items"][0]["destination"], "canonical/report.txt")
        self.assertFalse(destination_path.exists())

    def test_build_publish_back_plan_marks_never_as_skip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            run_root = workspace_root / "artifacts" / "runs" / "unit-run"
            source_path = run_root / "outputs" / "report.txt"
            source_path.parent.mkdir(parents=True, exist_ok=True)
            source_path.write_text("artifact", encoding="utf-8")

            plan = build_publish_back_plan(
                workspace_root=workspace_root,
                run_root=run_root,
                manifest={
                    "run_id": "unit-run",
                    "publish_back": {
                        "default_policy": "never",
                        "items": [{"source": "artifacts/runs/unit-run/outputs/report.txt", "destination": "canonical/report.txt"}],
                    },
                },
            )

        self.assertEqual(plan["items"][0]["action"], "skip")
        self.assertEqual(plan["items"][0]["reason"], "policy=never")

    def test_build_publish_back_plan_marks_if_absent_as_copy_when_destination_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            run_root = workspace_root / "artifacts" / "runs" / "unit-run"
            source_path = run_root / "outputs" / "report.txt"
            source_path.parent.mkdir(parents=True, exist_ok=True)
            source_path.write_text("artifact", encoding="utf-8")

            plan = build_publish_back_plan(
                workspace_root=workspace_root,
                run_root=run_root,
                manifest={
                    "run_id": "unit-run",
                    "publish_back": {
                        "default_policy": "never",
                        "items": [
                            {
                                "source": "artifacts/runs/unit-run/outputs/report.txt",
                                "destination": "canonical/report.txt",
                                "policy": "if_absent",
                            }
                        ],
                    },
                },
            )

        self.assertEqual(plan["items"][0]["action"], "copy")
        self.assertFalse(plan["items"][0]["destination_exists"])

    def test_build_publish_back_plan_marks_if_absent_as_skip_when_destination_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            run_root = workspace_root / "artifacts" / "runs" / "unit-run"
            source_path = run_root / "outputs" / "report.txt"
            destination_path = workspace_root / "canonical" / "report.txt"
            source_path.parent.mkdir(parents=True, exist_ok=True)
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            source_path.write_text("artifact", encoding="utf-8")
            destination_path.write_text("canonical", encoding="utf-8")

            plan = build_publish_back_plan(
                workspace_root=workspace_root,
                run_root=run_root,
                manifest={
                    "run_id": "unit-run",
                    "publish_back": {
                        "default_policy": "never",
                        "items": [
                            {
                                "source": "artifacts/runs/unit-run/outputs/report.txt",
                                "destination": "canonical/report.txt",
                                "policy": "if_absent",
                            }
                        ],
                    },
                },
            )

        self.assertEqual(plan["items"][0]["action"], "skip")
        self.assertTrue(plan["items"][0]["destination_exists"])

    def test_build_publish_back_plan_marks_overwrite_as_overwrite_when_destination_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            run_root = workspace_root / "artifacts" / "runs" / "unit-run"
            source_path = run_root / "outputs" / "report.txt"
            destination_path = workspace_root / "canonical" / "report.txt"
            source_path.parent.mkdir(parents=True, exist_ok=True)
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            source_path.write_text("artifact", encoding="utf-8")
            destination_path.write_text("canonical", encoding="utf-8")

            plan = build_publish_back_plan(
                workspace_root=workspace_root,
                run_root=run_root,
                manifest={
                    "run_id": "unit-run",
                    "publish_back": {
                        "default_policy": "never",
                        "items": [
                            {
                                "source": "artifacts/runs/unit-run/outputs/report.txt",
                                "destination": "canonical/report.txt",
                                "policy": "overwrite",
                            }
                        ],
                    },
                },
            )

        self.assertEqual(plan["items"][0]["action"], "overwrite")
        self.assertTrue(plan["items"][0]["destination_exists"])

    def test_build_publish_back_plan_rejects_source_outside_run_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            run_root = workspace_root / "artifacts" / "runs" / "unit-run"
            source_path = workspace_root / "artifacts" / "shared" / "report.txt"
            source_path.parent.mkdir(parents=True, exist_ok=True)
            source_path.write_text("artifact", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "Publish-back source must stay under the run root"):
                build_publish_back_plan(
                    workspace_root=workspace_root,
                    run_root=run_root,
                    manifest={
                        "run_id": "unit-run",
                        "publish_back": {
                            "default_policy": "never",
                            "items": [{"source": "artifacts/shared/report.txt", "destination": "canonical/report.txt"}],
                        },
                    },
                )

    def test_build_publish_back_plan_rejects_destination_inside_run_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            run_root = workspace_root / "artifacts" / "runs" / "unit-run"
            source_path = run_root / "outputs" / "report.txt"
            source_path.parent.mkdir(parents=True, exist_ok=True)
            source_path.write_text("artifact", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "Publish-back destination must not point back into the run root"):
                build_publish_back_plan(
                    workspace_root=workspace_root,
                    run_root=run_root,
                    manifest={
                        "run_id": "unit-run",
                        "publish_back": {
                            "default_policy": "never",
                            "items": [
                                {
                                    "source": "artifacts/runs/unit-run/outputs/report.txt",
                                    "destination": "artifacts/runs/unit-run/published/report.txt",
                                }
                            ],
                        },
                    },
                )

    def test_build_publish_back_plan_allows_absolute_destination_outside_run_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir, tempfile.TemporaryDirectory() as canonical_dir:
            workspace_root = Path(tmp_dir)
            run_root = workspace_root / "artifacts" / "runs" / "unit-run"
            source_path = run_root / "outputs" / "report.txt"
            destination_path = Path(canonical_dir) / "report.txt"
            source_path.parent.mkdir(parents=True, exist_ok=True)
            source_path.write_text("artifact", encoding="utf-8")

            plan = build_publish_back_plan(
                workspace_root=workspace_root,
                run_root=run_root,
                manifest={
                    "run_id": "unit-run",
                    "publish_back": {
                        "default_policy": "if_absent",
                        "items": [
                            {
                                "source": "artifacts/runs/unit-run/outputs/report.txt",
                                "destination": str(destination_path),
                            }
                        ],
                    },
                },
            )

        self.assertEqual(plan["items"][0]["destination"], str(destination_path.resolve()))
        self.assertEqual(plan["items"][0]["action"], "copy")


if __name__ == "__main__":
    unittest.main()
