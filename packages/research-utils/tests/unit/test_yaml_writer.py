from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import research_platform.core.config as core_config
import research_platform.hpc._yaml as hpc_yaml
import research_platform.utils as utils
from research_platform.utils.yaml_writer import dump_yaml, write_yaml


class DumpYamlTests(unittest.TestCase):
    def test_nested_mappings_and_lists_use_two_space_indent(self) -> None:
        document = {"outer": {"inner": 1, "items": ["a", {"key": "b"}]}}
        self.assertEqual(
            dump_yaml(document),
            "outer:\n  inner: 1\n  items:\n    - a\n    -\n      key: b\n",
        )

    def test_path_values_become_strings(self) -> None:
        self.assertEqual(dump_yaml({"path": Path("a/b.txt")}), "path: a/b.txt\n")

    def test_string_quoting(self) -> None:
        self.assertEqual(dump_yaml({"value": "plain_value-1.${X}/y"}), "value: plain_value-1.${X}/y\n")
        self.assertEqual(dump_yaml({"value": "two words"}), 'value: "two words"\n')
        self.assertEqual(dump_yaml({"value": "true"}), 'value: "true"\n')
        self.assertEqual(dump_yaml({"value": "12"}), 'value: "12"\n')
        self.assertEqual(dump_yaml({"value": 'say "hi"'}), 'value: "say \\"hi\\""\n')
        self.assertEqual(dump_yaml({"value": ""}), 'value: ""\n')

    def test_scalars(self) -> None:
        self.assertEqual(
            dump_yaml({"none": None, "yes": True, "no": False, "int": 3, "float": 1.5}),
            "none: null\nyes: true\nno: false\nint: 3\nfloat: 1.5\n",
        )

    def test_empty_containers(self) -> None:
        self.assertEqual(dump_yaml({}), "{}\n")
        self.assertEqual(dump_yaml([]), "[]\n")
        self.assertEqual(dump_yaml({"m": {}, "l": []}), "m: {}\nl: []\n")
        self.assertEqual(dump_yaml([{}, []]), "- {}\n- []\n")


class WriteYamlTests(unittest.TestCase):
    def test_creates_parents_and_returns_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "nested" / "dir" / "doc.yaml"
            result = write_yaml(target, {"key": "value"})
            self.assertEqual(result, target)
            self.assertEqual(target.read_text(encoding="utf-8"), "key: value\n")


class DumpYamlMatchesExistingWritersTests(unittest.TestCase):
    def test_utils_hpc_and_core_dump_yaml_produce_identical_text(self) -> None:
        document = {
            "name": "example",
            "path": Path("runs/output"),
            "missing": None,
            "enabled": True,
            "disabled": False,
            "count": 7,
            "ratio": 0.25,
            "quoted": "needs quoting",
            "bool_text": "false",
            "number_text": "42",
            "empty_map": {},
            "empty_list": [],
            "nested": {
                "items": [1, "two", Path("three/four"), None],
                "records": [{"id": "a", "tags": ["x", "y z"]}, {"id": "b", "tags": []}, {}],
                "lists": [[1, 2], []],
            },
        }
        expected = utils.dump_yaml(document)
        self.assertEqual(expected, hpc_yaml.dump_yaml(document))
        self.assertEqual(expected, core_config.dump_yaml(document))


if __name__ == "__main__":
    unittest.main()
