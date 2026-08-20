from __future__ import annotations

from decimal import Decimal
import hashlib
import json
from pathlib import Path
import struct
import sys
import unicodedata
import unittest
from unittest.mock import patch


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PACKAGE_ROOT / "src"))

from research_platform.hpc import safety as safety_api
from research_platform.hpc.safety.canonical import (
    CANONICAL_JSON_VERSION,
    CANONICAL_UNICODE_VERSION,
    DEFAULT_CANONICAL_JSON_LIMITS,
    MAX_CANONICAL_CONTAINER_ITEMS,
    MAX_CANONICAL_DEPTH,
    MAX_CANONICAL_DOCUMENT_BYTES,
    MAX_CANONICAL_STRING_BYTES,
    MAX_SIGNED_64,
    MIN_SIGNED_64,
    RECEIPT_DIGEST_ALGORITHM,
    RECEIPT_DIGEST_DOMAIN,
    TREE_DIGEST_ALGORITHM,
    TREE_DIGEST_DOMAIN,
    CanonicalJsonDecodeError,
    CanonicalJsonLimitError,
    CanonicalJsonLimits,
    CanonicalJsonTypeError,
    Sha256Digest,
    canonical_json_bytes,
    domain_separated_sha256,
    parse_canonical_json_bytes,
)


class CanonicalJsonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        fixture_path = (
            PACKAGE_ROOT
            / "tests"
            / "fixtures"
            / "safety-v1"
            / "canonical-golden.json"
        )
        cls.fixture = json.loads(fixture_path.read_text(encoding="utf-8"))

    def test_versions_domains_and_default_limits_are_frozen(self) -> None:
        self.assertEqual(
            CANONICAL_JSON_VERSION,
            "research_platform.hpc.canonical_json.v1",
        )
        self.assertEqual(CANONICAL_UNICODE_VERSION, "3.2.0")
        self.assertEqual(
            safety_api.CANONICAL_UNICODE_VERSION,
            CANONICAL_UNICODE_VERSION,
        )
        self.assertEqual(
            CANONICAL_UNICODE_VERSION,
            unicodedata.ucd_3_2_0.unidata_version,
        )
        self.assertEqual(
            TREE_DIGEST_ALGORITHM,
            "research_platform.hpc.sha256_tree.v1",
        )
        self.assertEqual(
            RECEIPT_DIGEST_ALGORITHM,
            "research_platform.hpc.sha256_receipt_envelope.v1",
        )
        self.assertEqual(
            TREE_DIGEST_DOMAIN,
            b"research-platform:hpc:regular-file-tree:v1\0",
        )
        self.assertEqual(
            RECEIPT_DIGEST_DOMAIN,
            b"research-platform:hpc:receipt-envelope:v1\0",
        )
        self.assertEqual(
            DEFAULT_CANONICAL_JSON_LIMITS,
            CanonicalJsonLimits(
                maximum_document_bytes=MAX_CANONICAL_DOCUMENT_BYTES,
                maximum_depth=MAX_CANONICAL_DEPTH,
                maximum_container_items=MAX_CANONICAL_CONTAINER_ITEMS,
                maximum_string_bytes=MAX_CANONICAL_STRING_BYTES,
            ),
        )
        self.assertEqual(MAX_CANONICAL_DOCUMENT_BYTES, 16 * 1024 * 1024)
        self.assertEqual(MAX_CANONICAL_DEPTH, 64)
        self.assertEqual(MAX_CANONICAL_CONTAINER_ITEMS, 100_000)
        self.assertEqual(MAX_CANONICAL_STRING_BYTES, 1024 * 1024)
        self.assertEqual(MIN_SIGNED_64, -(2**63))
        self.assertEqual(MAX_SIGNED_64, 2**63 - 1)

    def test_golden_canonical_bytes_are_exact(self) -> None:
        self.assertEqual(
            self.fixture["canonical_json_version"],
            CANONICAL_JSON_VERSION,
        )
        self.assertEqual(
            self.fixture["canonical_unicode_version"],
            CANONICAL_UNICODE_VERSION,
        )
        for case in self.fixture["cases"]:
            with self.subTest(case=case["name"]):
                expected = bytes.fromhex(case["canonical_hex"])
                actual = canonical_json_bytes(case["value"])
                self.assertEqual(actual, expected)
                self.assertEqual(parse_canonical_json_bytes(actual), case["value"])

    def test_utf8_key_order_no_bom_and_no_outer_whitespace(self) -> None:
        encoded = canonical_json_bytes({"é": 1, "a": 2, "Z": 3})

        self.assertEqual(encoded, '{"Z":3,"a":2,"é":1}'.encode("utf-8"))
        self.assertFalse(encoded.startswith(b"\xef\xbb\xbf"))
        self.assertEqual(encoded, encoded.strip())
        self.assertFalse(encoded.endswith(b"\n"))

    def test_serialization_is_deterministic_for_a_fixed_value(self) -> None:
        first = canonical_json_bytes(
            {"items": [None, True, -4], "label": "café"}
        )
        for _ in range(20):
            self.assertEqual(
                canonical_json_bytes(
                    {"label": "café", "items": [None, True, -4]}
                ),
                first,
            )

    def test_parser_rejects_duplicate_object_keys(self) -> None:
        with self.assertRaises(CanonicalJsonDecodeError):
            parse_canonical_json_bytes(b'{"a":1,"a":2}')

    def test_parser_rejects_noncanonical_whitespace_and_escapes(self) -> None:
        rejected = (
            b" null",
            b"null ",
            b"null\n",
            b'{"a": 1}',
            b'{"b":2,"a":1}',
            b'{"text":"caf\\u00e9"}',
            b'{"slash":"\\/"}',
            b"-0",
        )
        for data in rejected:
            with self.subTest(data=data):
                with self.assertRaises(CanonicalJsonDecodeError):
                    parse_canonical_json_bytes(data)

    def test_parser_rejects_bom_invalid_utf8_and_nonbytes(self) -> None:
        for data in (b"\xef\xbb\xbfnull", b'"\xff"'):
            with self.subTest(data=data):
                with self.assertRaises(CanonicalJsonDecodeError):
                    parse_canonical_json_bytes(data)
        with self.assertRaises(CanonicalJsonTypeError):
            parse_canonical_json_bytes(bytearray(b"null"))  # type: ignore[arg-type]

    def test_only_the_frozen_json_value_types_are_accepted(self) -> None:
        class Unsupported:
            pass

        rejected = (
            1.0,
            float("nan"),
            float("inf"),
            Decimal("1"),
            b"text",
            ("array",),
            Unsupported(),
            {1: "non-string key"},
        )
        for value in rejected:
            with self.subTest(value=repr(value)):
                with self.assertRaises(CanonicalJsonTypeError):
                    canonical_json_bytes(value)

    def test_parser_rejects_json_floats_and_nonfinite_constants(self) -> None:
        for data in (b"1.0", b"1e0", b"NaN", b"Infinity", b"-Infinity"):
            with self.subTest(data=data):
                with self.assertRaises(CanonicalJsonTypeError):
                    parse_canonical_json_bytes(data)

    def test_parser_preserves_decode_errors_for_malformed_numbers(self) -> None:
        for data in (b"1.", b"1e", b"1e+", b"-", b".1"):
            with self.subTest(data=data):
                with patch.object(
                    json,
                    "loads",
                    side_effect=AssertionError("json.loads must not run"),
                ) as loads:
                    with self.assertRaises(CanonicalJsonDecodeError):
                        parse_canonical_json_bytes(data)
                    loads.assert_not_called()

    def test_booleans_are_checked_before_integers(self) -> None:
        self.assertEqual(canonical_json_bytes(True), b"true")
        self.assertEqual(canonical_json_bytes(False), b"false")
        self.assertNotEqual(canonical_json_bytes(True), b"1")

    def test_signed_64_boundaries_are_enforced(self) -> None:
        self.assertEqual(
            parse_canonical_json_bytes(str(MIN_SIGNED_64).encode("ascii")),
            MIN_SIGNED_64,
        )
        self.assertEqual(
            parse_canonical_json_bytes(str(MAX_SIGNED_64).encode("ascii")),
            MAX_SIGNED_64,
        )
        for value in (MIN_SIGNED_64 - 1, MAX_SIGNED_64 + 1):
            with self.subTest(value=value):
                with self.assertRaises(CanonicalJsonTypeError):
                    canonical_json_bytes(value)
                with self.assertRaises(CanonicalJsonTypeError):
                    parse_canonical_json_bytes(str(value).encode("ascii"))
        with self.assertRaises(CanonicalJsonTypeError):
            parse_canonical_json_bytes(b"1" * 10_000)

    def test_unicode_scalars_and_nfc_are_required(self) -> None:
        self.assertEqual(canonical_json_bytes("café"), '"café"'.encode("utf-8"))
        for value in ("cafe\u0301", "\ud800", "\udc00"):
            with self.subTest(value=ascii(value)):
                with self.assertRaises(CanonicalJsonTypeError):
                    canonical_json_bytes(value)
        with self.assertRaises(CanonicalJsonTypeError):
            parse_canonical_json_bytes(b'"cafe\\u0301"')
        for data in (b'"\\ud800"', b'"\\udc00"'):
            with self.subTest(data=data):
                with self.assertRaises(CanonicalJsonTypeError):
                    parse_canonical_json_bytes(data)

    def test_unicode_repertoire_is_frozen_to_version_3_2(self) -> None:
        post_3_2_character = "\U0001f9ea"
        self.assertEqual(
            unicodedata.ucd_3_2_0.category(post_3_2_character),
            "Cn",
        )
        with (
            patch.object(
                unicodedata,
                "category",
                side_effect=AssertionError(
                    "interpreter-default Unicode category must not be used"
                ),
            ) as default_category,
            patch.object(
                unicodedata,
                "normalize",
                side_effect=AssertionError(
                    "interpreter-default Unicode normalization must not be used"
                ),
            ) as default_normalize,
        ):
            self.assertEqual(
                canonical_json_bytes("café"),
                '"café"'.encode("utf-8"),
            )
            default_category.assert_not_called()
            default_normalize.assert_not_called()
        for value in (
            post_3_2_character,
            {post_3_2_character: None},
        ):
            with self.subTest(value=value):
                with self.assertRaises(CanonicalJsonTypeError):
                    canonical_json_bytes(value)
        with self.assertRaises(CanonicalJsonTypeError):
            parse_canonical_json_bytes(
                f'"{post_3_2_character}"'.encode("utf-8")
            )

    def test_known_cross_version_unicode_divergence_vectors_fail_closed(
        self,
    ) -> None:
        vectors = (
            "a" + "\U0001e4ec" + "\u0301",
            "x\u0301\U00010efd",
        )
        for value in vectors:
            with self.subTest(value=ascii(value)):
                with self.assertRaises(CanonicalJsonTypeError):
                    canonical_json_bytes(value)
                with self.assertRaises(CanonicalJsonTypeError):
                    parse_canonical_json_bytes(
                        ('"' + value + '"').encode("utf-8")
                    )

    def test_document_byte_limit_is_enforced_for_write_and_parse(self) -> None:
        limits = CanonicalJsonLimits(maximum_document_bytes=3)
        self.assertEqual(canonical_json_bytes("a", limits=limits), b'"a"')
        with self.assertRaises(CanonicalJsonLimitError):
            canonical_json_bytes(True, limits=limits)
        with self.assertRaises(CanonicalJsonLimitError):
            parse_canonical_json_bytes(b"true", limits=limits)

    def test_serialization_stops_as_soon_as_document_limit_is_exceeded(
        self,
    ) -> None:
        def oversized_chunks(*_args: object, **_kwargs: object):
            yield "1234"
            yield "56"
            raise AssertionError("encoder was consumed past the document limit")

        limits = CanonicalJsonLimits(maximum_document_bytes=5)
        with patch.object(json.JSONEncoder, "iterencode", new=oversized_chunks):
            with self.assertRaises(CanonicalJsonLimitError):
                canonical_json_bytes(None, limits=limits)

    def test_depth_limit_counts_nested_containers(self) -> None:
        limits = CanonicalJsonLimits(maximum_depth=2)
        self.assertEqual(canonical_json_bytes([[]], limits=limits), b"[[]]")
        with self.assertRaises(CanonicalJsonLimitError):
            canonical_json_bytes([[[]]], limits=limits)
        with self.assertRaises(CanonicalJsonLimitError):
            parse_canonical_json_bytes(b"[[[]]]", limits=limits)

    def test_total_container_item_limit_is_enforced(self) -> None:
        limits = CanonicalJsonLimits(maximum_container_items=3)
        self.assertEqual(
            canonical_json_bytes({"a": [1, 2]}, limits=limits),
            b'{"a":[1,2]}',
        )
        with self.assertRaises(CanonicalJsonLimitError):
            canonical_json_bytes({"a": [1, 2, 3]}, limits=limits)

    def test_string_byte_limit_applies_to_values_and_keys(self) -> None:
        limits = CanonicalJsonLimits(maximum_string_bytes=2)
        self.assertEqual(canonical_json_bytes("é", limits=limits), b'"\xc3\xa9"')
        for value in ("abc", {"abc": None}):
            with self.subTest(value=value):
                with self.assertRaises(CanonicalJsonLimitError):
                    canonical_json_bytes(value, limits=limits)

    def test_cycles_are_rejected(self) -> None:
        value: list[object] = []
        value.append(value)
        with self.assertRaises(CanonicalJsonTypeError):
            canonical_json_bytes(value)

    def test_invalid_limit_objects_are_rejected(self) -> None:
        with self.assertRaises(TypeError):
            CanonicalJsonLimits(maximum_depth=True)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            CanonicalJsonLimits(maximum_document_bytes=0)
        with self.assertRaises(ValueError):
            CanonicalJsonLimits(maximum_container_items=-1)
        with self.assertRaises(TypeError):
            canonical_json_bytes(None, limits=object())  # type: ignore[arg-type]

    def test_limits_are_exact_type_and_may_only_narrow_protocol_caps(self) -> None:
        narrowed = CanonicalJsonLimits(
            maximum_document_bytes=1024,
            maximum_depth=4,
            maximum_container_items=10,
            maximum_string_bytes=64,
        )
        self.assertEqual(canonical_json_bytes(None, limits=narrowed), b"null")

        widened_values = (
            {
                "maximum_document_bytes": MAX_CANONICAL_DOCUMENT_BYTES + 1,
            },
            {"maximum_depth": MAX_CANONICAL_DEPTH + 1},
            {
                "maximum_container_items": (
                    MAX_CANONICAL_CONTAINER_ITEMS + 1
                )
            },
            {"maximum_string_bytes": MAX_CANONICAL_STRING_BYTES + 1},
        )
        for values in widened_values:
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    CanonicalJsonLimits(**values)

        class HostileLimits(CanonicalJsonLimits):
            def __post_init__(self) -> None:
                pass

            def __getattribute__(self, name: str) -> object:
                if name.startswith("maximum_"):
                    raise AssertionError("hostile limit field was accessed")
                return super().__getattribute__(name)

        with self.assertRaises(TypeError):
            canonical_json_bytes(None, limits=HostileLimits())
        with self.assertRaises(TypeError):
            parse_canonical_json_bytes(b"null", limits=HostileLimits())

    def test_lexical_limits_and_escapes_fail_before_json_loads(self) -> None:
        cases = (
            (
                b"[" * (MAX_CANONICAL_DEPTH + 1)
                + b"]" * (MAX_CANONICAL_DEPTH + 1),
                DEFAULT_CANONICAL_JSON_LIMITS,
                CanonicalJsonLimitError,
            ),
            (
                b"[null,null]",
                CanonicalJsonLimits(maximum_container_items=1),
                CanonicalJsonLimitError,
            ),
            (
                b'{"a":null,"b":null}',
                CanonicalJsonLimits(maximum_container_items=1),
                CanonicalJsonLimitError,
            ),
            (
                b'"\\u00e9"',
                CanonicalJsonLimits(maximum_string_bytes=1),
                CanonicalJsonLimitError,
            ),
            (
                b"9223372036854775808",
                DEFAULT_CANONICAL_JSON_LIMITS,
                CanonicalJsonTypeError,
            ),
            (
                b"1" * 10_000,
                DEFAULT_CANONICAL_JSON_LIMITS,
                CanonicalJsonTypeError,
            ),
            (
                b'"\\ud834\\udd1e"',
                CanonicalJsonLimits(maximum_string_bytes=3),
                CanonicalJsonLimitError,
            ),
            (
                b'"\\x00"',
                DEFAULT_CANONICAL_JSON_LIMITS,
                CanonicalJsonDecodeError,
            ),
            (
                b'"\\ud800"',
                DEFAULT_CANONICAL_JSON_LIMITS,
                CanonicalJsonTypeError,
            ),
        )
        for data, limits, error_type in cases:
            with self.subTest(data=data, limits=limits):
                with patch.object(
                    json,
                    "loads",
                    side_effect=AssertionError("json.loads must not run"),
                ) as loads:
                    with self.assertRaises(error_type):
                        parse_canonical_json_bytes(data, limits=limits)
                    loads.assert_not_called()

    def test_frozen_protocol_maxima_fail_before_json_loads(self) -> None:
        too_many_items = (
            b"["
            + (b"null," * MAX_CANONICAL_CONTAINER_ITEMS)
            + b"null]"
        )
        too_many_nested_items = (
            b'{"items":['
            + (b"null," * (MAX_CANONICAL_CONTAINER_ITEMS - 1))
            + b"null]}"
        )
        oversized_raw_string = (
            b'"' + (b"a" * (MAX_CANONICAL_STRING_BYTES + 1)) + b'"'
        )
        oversized_key = (
            b'{"'
            + (b"a" * (MAX_CANONICAL_STRING_BYTES + 1))
            + b'":null}'
        )
        escaped_character = b"\\u00e9"
        escaped_repetitions = (MAX_CANONICAL_STRING_BYTES // 2) + 1
        oversized_escaped_string = (
            b'"' + (escaped_character * escaped_repetitions) + b'"'
        )
        cases = (
            too_many_items,
            too_many_nested_items,
            oversized_raw_string,
            oversized_key,
            oversized_escaped_string,
        )
        for data in cases:
            with self.subTest(document_bytes=len(data)):
                with patch.object(
                    json,
                    "loads",
                    side_effect=AssertionError("json.loads must not run"),
                ) as loads:
                    with self.assertRaises(CanonicalJsonLimitError):
                        parse_canonical_json_bytes(data)
                    loads.assert_not_called()

    def test_malformed_escapes_fail_before_json_loads(self) -> None:
        malformed = (
            b'"\\u12"',
            b'"\\u12xz"',
            b'"\\q"',
            b'"unterminated' + bytes((92,)),
        )
        for data in malformed:
            with self.subTest(data=data):
                with patch.object(
                    json,
                    "loads",
                    side_effect=AssertionError("json.loads must not run"),
                ) as loads:
                    with self.assertRaises(CanonicalJsonDecodeError):
                        parse_canonical_json_bytes(data)
                    loads.assert_not_called()

    def test_malformed_document_errors_precede_deferred_semantic_errors(
        self,
    ) -> None:
        malformed = (
            b'"\\ud800',
            b'"\\ud800\\q"',
            b'"\\udc00',
            b'"\\udc00\\q"',
            b'["\\ud800",]',
            b'["\\udc00",]',
            b"[9223372036854775808,]",
            b'{"a":9223372036854775808,}',
            b'"aa',
        )
        for data in malformed:
            with self.subTest(data=data):
                limits = (
                    CanonicalJsonLimits(maximum_string_bytes=1)
                    if data == b'"aa'
                    else DEFAULT_CANONICAL_JSON_LIMITS
                )
                with patch.object(
                    json,
                    "loads",
                    side_effect=AssertionError("json.loads must not run"),
                ) as loads:
                    with self.assertRaises(CanonicalJsonDecodeError):
                        parse_canonical_json_bytes(data, limits=limits)
                    loads.assert_not_called()

    def test_domain_separated_digest_matches_frozen_vector(self) -> None:
        vector = self.fixture["domain_digest"]
        domain = bytes.fromhex(vector["domain_hex"])
        body = bytes.fromhex(vector["body_hex"])

        digest = domain_separated_sha256(domain, body)

        self.assertEqual(vector["algorithm"], TREE_DIGEST_ALGORITHM)
        self.assertEqual(digest, Sha256Digest(vector["sha256"]))
        self.assertEqual(
            digest.value,
            hashlib.sha256(
                domain + struct.pack(">Q", len(body)) + body
            ).hexdigest(),
        )

    def test_digest_types_and_arguments_are_strict(self) -> None:
        for value in (
            "A" * 64,
            "g" * 64,
            "a" * 63,
            "a" * 65,
            b"a" * 64,
        ):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    Sha256Digest(value)  # type: ignore[arg-type]

        for domain, body in (
            (b"", b"body"),
            ("domain", b"body"),
            (b"domain", "body"),
        ):
            with self.subTest(domain=domain, body=body):
                with self.assertRaises(TypeError):
                    domain_separated_sha256(  # type: ignore[arg-type]
                        domain,
                        body,
                    )


if __name__ == "__main__":
    unittest.main()
