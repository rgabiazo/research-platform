from __future__ import annotations

import unittest

from research_platform.utils._version import DISTRIBUTION_NAME, package_version


class VersionTests(unittest.TestCase):
    def test_distribution_name_and_version(self) -> None:
        self.assertEqual(DISTRIBUTION_NAME, "research-utils")
        self.assertEqual(package_version(), "0.1.0a1")


if __name__ == "__main__":
    unittest.main()
