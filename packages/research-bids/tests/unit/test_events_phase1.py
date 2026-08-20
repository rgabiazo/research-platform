from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys
import importlib.util

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PACKAGE_ROOT / "src"))

from research_platform.bids.events.writers import write_events_tsv


@unittest.skipUnless(importlib.util.find_spec("pandas") is not None, "pandas is not installed")
class PandasWriterContractTests(unittest.TestCase):
    def test_pandas_backend_preserves_onset_strings_and_normalizes_response_time(self) -> None:
        rows = [
            {"onset": "90.48271683300845", "duration": "3.0", "response_time": "1.7962680830387399"},
            {"onset": 509.14106433297275, "duration": "10.0", "response_time": "n/a"},
        ]
        columns = ["onset", "duration", "response_time"]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "events.tsv"
            write_events_tsv(output_path, rows, columns, backend="pandas")
            lines = output_path.read_text(encoding="utf-8").splitlines()

        self.assertEqual(lines[0], "onset\tduration\tresponse_time")
        self.assertEqual(lines[1], "90.48271683300845\t3.0\t1.79626808303874")
        self.assertEqual(lines[2], "509.14106433297275\t10.0\tn/a")


if __name__ == "__main__":
    unittest.main()
