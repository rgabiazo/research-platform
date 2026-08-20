"""Contracts for lexical managed-payload paths."""

from __future__ import annotations

import unittest

from research_platform.hpc.safety.paths import (
    MAX_PORTABLE_COMPONENT_BYTES,
    MAX_PORTABLE_PATH_BYTES,
    PORTABLE_PATH_SCHEMA,
    PortablePathCollisionError,
    PortablePathError,
    PortableRelativePath,
    portable_path_sort_key,
    require_distinct_file_paths,
)


class PortableRelativePathTests(unittest.TestCase):
    def test_schema_and_valid_spelling_are_preserved(self) -> None:
        self.assertEqual(
            PORTABLE_PATH_SCHEMA,
            "research_platform.hpc.portable_relative_path.v1",
        )
        for value in (
            "file.txt",
            "nested/path-01_data.json",
            ".hidden",
            ".well-known/item",
            "name..",
            "...",
        ):
            with self.subTest(value=value):
                path = PortableRelativePath.parse(value)
                self.assertEqual(path.value, value)
                self.assertEqual(path.parts, tuple(value.split("/")))
                self.assertEqual(
                    portable_path_sort_key(path),
                    value.encode("ascii"),
                )

    def test_component_and_complete_path_caps_are_inclusive(self) -> None:
        component = "a" * MAX_PORTABLE_COMPONENT_BYTES
        self.assertEqual(PortableRelativePath.parse(component).value, component)
        components = [component] * 16
        exact_maximum = "/".join(components)
        self.assertEqual(len(exact_maximum), MAX_PORTABLE_PATH_BYTES)
        self.assertEqual(
            PortableRelativePath.parse(exact_maximum).value,
            exact_maximum,
        )

        with self.assertRaises(PortablePathError):
            PortableRelativePath.parse("a" * (MAX_PORTABLE_COMPONENT_BYTES + 1))
        with self.assertRaises(PortablePathError):
            PortableRelativePath.parse(f"{exact_maximum}/a")

    def test_non_string_and_unsafe_lexical_forms_are_rejected(self) -> None:
        invalid = (
            b"file.txt",
            "",
            "/absolute",
            "C:/drive-like",
            r"\\server\share",
            r"a\b",
            "a//b",
            "a/",
            ".",
            "..",
            "a/./b",
            "a/../b",
            "white space",
            "tab\tname",
            "line\nname",
            "nul\0name",
            "café",
            "a:b",
            "a+b",
            "a@b",
        )
        for value in invalid:
            with self.subTest(value=value):
                with self.assertRaises(PortablePathError):
                    PortableRelativePath.parse(value)

    def test_deterministic_order_is_ascii_byte_order(self) -> None:
        values = ("z/file", "A/file", "a0", ".hidden", "_value", "-value")
        paths = [PortableRelativePath.parse(value) for value in values]
        ordered = require_distinct_file_paths(paths)
        self.assertEqual(
            tuple(path.value for path in ordered),
            tuple(sorted(values, key=lambda value: value.encode("ascii"))),
        )

    def test_exact_duplicates_and_full_path_case_aliases_are_rejected(self) -> None:
        for values in (("a/file", "a/file"), ("a/File", "a/file")):
            with self.subTest(values=values):
                with self.assertRaises(PortablePathCollisionError):
                    require_distinct_file_paths(
                        PortableRelativePath.parse(value) for value in values
                    )

    def test_case_aliases_are_rejected_at_every_directory_prefix(self) -> None:
        for values in (
            ("A/x.txt", "a/y.txt"),
            ("root/A/x.txt", "root/a/y.txt"),
            ("root/a/X.txt", "root/a/x.txt"),
        ):
            with self.subTest(values=values):
                with self.assertRaises(PortablePathCollisionError):
                    require_distinct_file_paths(
                        PortableRelativePath.parse(value) for value in values
                    )

    def test_file_directory_prefix_collisions_are_rejected_in_either_order(
        self,
    ) -> None:
        for values in (
            ("a", "a/b.txt"),
            ("a/b.txt", "a"),
            ("A", "a/b.txt"),
            ("a/b.txt", "A"),
        ):
            with self.subTest(values=values):
                with self.assertRaises(PortablePathCollisionError):
                    require_distinct_file_paths(
                        PortableRelativePath.parse(value) for value in values
                    )

    def test_shared_same_spelling_directory_prefixes_are_allowed(self) -> None:
        ordered = require_distinct_file_paths(
            PortableRelativePath.parse(value)
            for value in ("a/z.txt", "a/b.txt", "a/nested/c.txt")
        )
        self.assertEqual(
            tuple(path.value for path in ordered),
            ("a/b.txt", "a/nested/c.txt", "a/z.txt"),
        )

    def test_collection_rejects_unvalidated_values(self) -> None:
        with self.assertRaises(TypeError):
            require_distinct_file_paths(["a.txt"])  # type: ignore[list-item]
        with self.assertRaises(TypeError):
            portable_path_sort_key("a.txt")  # type: ignore[arg-type]

    def test_hostile_string_subclasses_cannot_enter_the_authority(self) -> None:
        class HostileString(str):
            def startswith(self, *_args: object, **_kwargs: object) -> bool:
                return False

            def split(
                self,
                *_args: object,
                **_kwargs: object,
            ) -> list[str]:
                return ["safe.txt"]

            def __len__(self) -> int:
                return 1

            def __iter__(self):
                return iter("safe.txt")

            def encode(
                self,
                *_args: object,
                **_kwargs: object,
            ) -> bytes:
                return b"safe.txt"

        for unsafe in ("/outside", "../outside"):
            value = HostileString(unsafe)
            with self.subTest(value=unsafe):
                with self.assertRaises(PortablePathError):
                    PortableRelativePath.parse(value)
                with self.assertRaises(PortablePathError):
                    PortableRelativePath(value)  # type: ignore[arg-type]

    def test_portable_path_subclasses_cannot_enter_collections(self) -> None:
        class HostilePath(PortableRelativePath):
            pass

        hostile = HostilePath("safe.txt")
        with self.assertRaises(TypeError):
            portable_path_sort_key(hostile)
        with self.assertRaises(TypeError):
            require_distinct_file_paths((hostile,))


if __name__ == "__main__":
    unittest.main()
