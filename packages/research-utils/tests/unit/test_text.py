from __future__ import annotations

import unittest

from research_platform.utils.text import non_empty_text, optional_text


class OptionalTextTests(unittest.TestCase):
    def test_none_and_blank_values_become_none(self) -> None:
        self.assertIsNone(optional_text(None))
        self.assertIsNone(optional_text(""))
        self.assertIsNone(optional_text("  "))

    def test_text_is_stripped(self) -> None:
        self.assertEqual(optional_text(" a "), "a")

    def test_non_strings_are_stringified(self) -> None:
        self.assertEqual(optional_text(0), "0")


class NonEmptyTextTests(unittest.TestCase):
    def test_text_is_stripped(self) -> None:
        self.assertEqual(non_empty_text(" x ", field_name="name"), "x")

    def test_blank_text_raises_with_field_name(self) -> None:
        with self.assertRaises(ValueError) as caught:
            non_empty_text("  ", field_name="name")
        self.assertEqual(str(caught.exception), "name must be a non-empty string.")


if __name__ == "__main__":
    unittest.main()
