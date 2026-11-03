from __future__ import annotations

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from research_platform.utils.output_targets import (
    has_parent_traversal,
    reject_duplicate_targets,
    reject_existing_targets,
    resolved_output_root,
)


class ParentTraversalTests(unittest.TestCase):
    def test_detects_parent_traversal(self) -> None:
        self.assertTrue(has_parent_traversal(Path("a/../b")))
        self.assertFalse(has_parent_traversal(Path("a/b")))

    def test_resolved_output_root_rejects_parent_traversal(self) -> None:
        with self.assertRaises(ValueError) as caught:
            resolved_output_root("a/../b")
        self.assertEqual(str(caught.exception), "output_root must not contain parent traversal.")

    def test_resolved_output_root_returns_absolute_path(self) -> None:
        root = resolved_output_root("relative/output")
        self.assertTrue(root.is_absolute())
        self.assertEqual(root, Path("relative/output").resolve(strict=False))


class RejectTargetsTests(unittest.TestCase):
    def test_duplicate_paths_are_rejected(self) -> None:
        shared = Path("same.tsv")
        targets = {"first": SimpleNamespace(path=shared), "second": SimpleNamespace(path=shared)}
        with self.assertRaises(ValueError) as caught:
            reject_duplicate_targets(targets)
        self.assertEqual(
            str(caught.exception),
            "Output paths for 'first' and 'second' must be distinct.",
        )

    def test_distinct_paths_pass(self) -> None:
        targets = {"first": SimpleNamespace(path=Path("a.tsv")), "second": SimpleNamespace(path=Path("b.tsv"))}
        reject_duplicate_targets(targets)

    def test_existing_directory_is_rejected_regardless_of_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            targets = {"table": SimpleNamespace(path=directory)}
            for overwrite in (False, True):
                with self.assertRaises(IsADirectoryError) as caught:
                    reject_existing_targets(targets, overwrite=overwrite)
                self.assertEqual(
                    str(caught.exception),
                    f"Output path for 'table' is an existing directory: {directory}",
                )

    def test_existing_file_requires_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "table.tsv"
            path.write_text("x", encoding="utf-8")
            targets = {"table": SimpleNamespace(path=path)}
            with self.assertRaises(FileExistsError) as caught:
                reject_existing_targets(targets, overwrite=False)
            self.assertEqual(str(caught.exception), f"Output path for 'table' already exists: {path}")
            reject_existing_targets(targets, overwrite=True)

    def test_missing_path_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            targets = {"table": SimpleNamespace(path=Path(tmp) / "missing.tsv")}
            reject_existing_targets(targets, overwrite=False)


if __name__ == "__main__":
    unittest.main()
