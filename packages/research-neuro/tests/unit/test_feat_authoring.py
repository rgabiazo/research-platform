from __future__ import annotations

from contextlib import redirect_stdout
import io
from pathlib import Path
import sys
import unittest
from unittest import mock

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = PACKAGE_ROOT.parents[1]
CORE_PACKAGE_ROOT = WORKSPACE_ROOT / "packages" / "research-core"
HPC_PACKAGE_ROOT = WORKSPACE_ROOT / "packages" / "research-hpc"
sys.path.insert(0, str(PACKAGE_ROOT / "src"))
sys.path.insert(0, str(CORE_PACKAGE_ROOT / "src"))
sys.path.insert(0, str(HPC_PACKAGE_ROOT / "src"))

from research_platform.neuro.fsl.feat.authoring import (
    init_model_document,
    interactive_init_model_document,
    rename_model_document,
    summarize_model_document,
    validate_model_document,
)


class FeatAuthoringTests(unittest.TestCase):
    def test_validate_rejects_duplicate_evs(self) -> None:
        errors = validate_model_document(
            model_name="task_glm",
            document={
                "model": {
                    "name": "task_glm",
                    "ev_order": ["condition_a", "condition_a"],
                    "derivative_on": [],
                    "nonconvolved": [],
                    "contrasts": [{"name": "a", "weights": [1, 0]}],
                }
            },
        )

        self.assertTrue(any("duplicate EV names" in error for error in errors))

    def test_validate_rejects_unknown_derivative_and_nonconvolved_evs(self) -> None:
        errors = validate_model_document(
            model_name="task_glm",
            document={
                "model": {
                    "name": "task_glm",
                    "ev_order": ["condition_a", "condition_b"],
                    "derivative_on": ["missing_ev"],
                    "nonconvolved": ["missing_ev_2"],
                    "contrasts": [{"name": "a", "weights": [1, 0]}],
                }
            },
        )

        self.assertTrue(any("model.derivative_on contains EV names not present" in error for error in errors))
        self.assertTrue(any("model.nonconvolved contains EV names not present" in error for error in errors))

    def test_validate_rejects_overlap_duplicate_contrasts_and_length_mismatch(self) -> None:
        errors = validate_model_document(
            model_name="task_glm",
            document={
                "model": {
                    "name": "task_glm",
                    "ev_order": ["condition_a", "condition_b"],
                    "derivative_on": ["condition_a"],
                    "nonconvolved": ["condition_a"],
                    "contrasts": [
                        {"name": "dup", "weights": [1]},
                        {"name": "dup", "weights": [0, 1]},
                    ],
                }
            },
        )

        self.assertTrue(any("must not overlap" in error for error in errors))
        self.assertTrue(any("duplicate contrast names" in error for error in errors))
        self.assertTrue(any("must contain exactly 2 values" in error for error in errors))

    def test_validate_rejects_filename_name_mismatch(self) -> None:
        errors = validate_model_document(
            model_name="file_name",
            document={
                "model": {
                    "name": "different_name",
                    "ev_order": ["condition_a"],
                    "derivative_on": [],
                    "nonconvolved": [],
                    "contrasts": [{"name": "a", "weights": [1]}],
                }
            },
        )

        self.assertTrue(any("must match the file name" in error for error in errors))

    def test_init_model_document_accepts_space_and_comma_separated_values(self) -> None:
        document = init_model_document(
            name="task_glm",
            options={
                "ev_order": ["condition_a", "condition_b,button_press"],
                "derivative_on": ["condition_a", "condition_b"],
                "nonconvolved": ["button_press"],
                "contrasts": ["condition_a_gt_baseline:1,0,0"],
            },
        )

        self.assertEqual(document["model"]["ev_order"], ["condition_a", "condition_b", "button_press"])
        self.assertEqual(document["model"]["contrasts"][0]["weights"], [1, 0, 0])

    def test_interactive_init_builds_valid_document(self) -> None:
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            with mock.patch(
                "builtins.input",
                side_effect=[
                    "condition_a condition_b button_press",
                    "condition_a condition_b",
                    "button_press",
                    "condition_a_gt_baseline:1,0,0",
                    "",
                ],
            ):
                document = interactive_init_model_document(name="task_glm")

        errors = validate_model_document(model_name="task_glm", document=document)
        self.assertEqual(errors, [])

    def test_rename_and_summary_are_user_facing(self) -> None:
        document = {
            "model": {
                "name": "task_glm",
                "ev_order": ["condition_a", "condition_b"],
                "derivative_on": ["condition_a"],
                "nonconvolved": [],
                "contrasts": [{"name": "a_gt_b", "weights": [1, -1]}],
            }
        }

        renamed = rename_model_document(new_name="task_glm_copy", document=document)
        summary = summarize_model_document(model_name="task_glm_copy", document=renamed)

        self.assertEqual(renamed["model"]["name"], "task_glm_copy")
        self.assertIn("FEAT first-level model: task_glm_copy", summary)
        self.assertIn("Contrasts (1): a_gt_b", summary)


if __name__ == "__main__":
    unittest.main()
