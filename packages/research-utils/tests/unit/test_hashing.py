from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
import unittest

from research_platform.utils.hashing import file_sha256


class FileSha256Tests(unittest.TestCase):
    def test_matches_hashlib_across_chunk_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.bin"
            size = 1024 * 1024 + 1
            path.write_bytes(bytes(index % 251 for index in range(size)))
            self.assertEqual(file_sha256(path), hashlib.sha256(path.read_bytes()).hexdigest())


if __name__ == "__main__":
    unittest.main()
