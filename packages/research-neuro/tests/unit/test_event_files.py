from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PACKAGE_ROOT / "src"))

from research_platform.neuro.events.files import inspect_numeric_event_file


class EventFileInspectionTests(unittest.TestCase):
    def test_inspect_numeric_event_file_classifies_empty_valid_and_invalid_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            empty_path = root / "empty.txt"
            valid_path = root / "valid.txt"
            invalid_path = root / "invalid.txt"
            empty_path.write_text("\n# comments only\n", encoding="utf-8")
            valid_path.write_text("0 1 1\n2.5 0.5 1\n", encoding="utf-8")
            invalid_path.write_text("0 one 1\n", encoding="utf-8")

            empty = inspect_numeric_event_file(empty_path)
            valid = inspect_numeric_event_file(valid_path)
            invalid = inspect_numeric_event_file(invalid_path)
            missing = inspect_numeric_event_file(root / "missing.txt")

        self.assertEqual(empty.status, "empty")
        self.assertEqual(valid.status, "valid")
        self.assertEqual(valid.row_count, 2)
        self.assertEqual(invalid.status, "invalid")
        self.assertIn("non-numeric", invalid.message or "")
        self.assertEqual(missing.status, "missing")


if __name__ == "__main__":
    unittest.main()
