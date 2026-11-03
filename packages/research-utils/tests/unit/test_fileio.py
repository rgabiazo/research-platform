from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest import mock

from research_platform.utils.fileio import write_text_atomic


def _temp_files(directory: Path, name: str) -> list[Path]:
    return sorted(directory.glob(f".{name}.*.tmp"))


class WriteTextAtomicTests(unittest.TestCase):
    def test_writes_new_file_without_leaving_temp_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            target = directory / "out.txt"
            write_text_atomic(target, "hello\n")
            self.assertEqual(target.read_text(encoding="utf-8"), "hello\n")
            self.assertEqual(sorted(directory.iterdir()), [target])

    def test_replaces_existing_content(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "out.txt"
            target.write_text("old", encoding="utf-8")
            write_text_atomic(target, "new")
            self.assertEqual(target.read_text(encoding="utf-8"), "new")

    def test_creates_missing_parent_directories(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "a" / "b" / "out.txt"
            write_text_atomic(target, "text")
            self.assertEqual(target.read_text(encoding="utf-8"), "text")

    def test_removes_temp_file_when_replace_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            target = directory / "out.txt"
            with mock.patch(
                "research_platform.utils.fileio.os.replace",
                side_effect=OSError("replace failed"),
            ):
                with self.assertRaises(OSError):
                    write_text_atomic(target, "text")
            self.assertEqual(_temp_files(directory, "out.txt"), [])
            self.assertFalse(target.exists())


if __name__ == "__main__":
    unittest.main()
