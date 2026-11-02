"""Tests for the research_platform.neuro.events compatibility facade."""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PACKAGE_ROOT / "src"))

import research_platform.bids.events.semantics as semantics  # noqa: E402
import research_platform.neuro.events as neuro_events  # noqa: E402

LOCAL_NAMES = {"NumericEventFileInspection", "inspect_numeric_event_file"}


class EventsFacadeTests(unittest.TestCase):
    def test_moved_names_resolve_to_semantics_objects(self) -> None:
        self.assertEqual(set(semantics.__all__), set(neuro_events.__all__) - LOCAL_NAMES)
        for name in semantics.__all__:
            self.assertIs(getattr(neuro_events, name), getattr(semantics, name), name)

    def test_unknown_attribute_raises(self) -> None:
        with self.assertRaises(AttributeError):
            getattr(neuro_events, "missing_name")

    def test_import_does_not_load_bids(self) -> None:
        script = (
            "import sys\n"
            "import research_platform.neuro.events\n"
            'raise SystemExit(int("research_platform.bids" in sys.modules))'
        )
        result = subprocess.run(
            [sys.executable, "-c", script], check=False, capture_output=True, text=True
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
