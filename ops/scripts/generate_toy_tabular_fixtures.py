#!/usr/bin/env python3
"""Generate deterministic, human-data-free toy tabular fixtures.

Every identifier and value is an algorithmic invention.  The 24 records are
formed from 12 ordinal levels crossed with two balanced labels.  Features use
only modular arithmetic and exact binary fractions; ``continuous_target`` is
a fixed linear combination of the generated features.  No external dataset,
environment-derived randomness, participant, patient, health, demographic,
or other human data are used.
"""

from __future__ import annotations

import argparse
import csv
import io
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
CANONICAL_COLUMNS = (
    "record_id",
    "feature_a",
    "feature_b",
    "feature_c",
    "binary_target",
    "measure_x",
    "measure_y",
    "feature_d",
    "continuous_target",
)
CORE_COLUMNS = (
    "record_id",
    "feature_a",
    "feature_b",
    "feature_c",
    "binary_target",
)
MEASUREMENT_COLUMNS = (
    "record_id",
    "measure_x",
    "measure_y",
    "feature_d",
    "continuous_target",
)


def _number(value: int | float) -> str:
    """Render numbers exactly as the tabular example backends materialize them."""

    return str(value)


def _synthetic_records() -> list[dict[str, str]]:
    """Return the complete formula specification for all 24 toy records.

    Levels 1 through 12 are each crossed with labels 0 and 1. ``feature_a``
    changes sign with the label; the remaining inputs use the level, modular
    arithmetic, and quarter-step constants. ``continuous_target`` intentionally
    omits both ``feature_a`` and the label, so it is identical within each pair.
    """

    records: list[dict[str, str]] = []
    for level in range(1, 13):
        for binary_target in (0, 1):
            row_number = (level - 1) * 2 + binary_target + 1
            sign = 1 if binary_target else -1

            feature_a = sign * (level + 2)
            feature_b = (level * 5) % 17 - 8
            feature_c = 1.0 + ((level * 3) % 11) * 0.5
            measure_x = 2.0 + ((level * 7) % 19) * 0.25
            measure_y = (level * 11) % 29 - 14
            feature_d = ((level % 5) - 2) * 1.25
            continuous_target = (
                20.0
                - 0.75 * feature_b
                + 2.0 * feature_c
                + 0.5 * measure_x
                - 0.25 * measure_y
                + 1.25 * feature_d
            )

            records.append(
                {
                    "record_id": f"record-{row_number:03d}",
                    "feature_a": _number(feature_a),
                    "feature_b": _number(feature_b),
                    "feature_c": _number(feature_c),
                    "binary_target": _number(binary_target),
                    "measure_x": _number(measure_x),
                    "measure_y": _number(measure_y),
                    "feature_d": _number(feature_d),
                    "continuous_target": _number(continuous_target),
                }
            )
    return records


def _render_delimited(
    records: list[dict[str, str]],
    columns: tuple[str, ...],
    *,
    delimiter: str,
) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=columns,
        delimiter=delimiter,
        lineterminator="\n",
        extrasaction="ignore",
    )
    writer.writeheader()
    writer.writerows(records)
    return buffer.getvalue().encode("utf-8")


def _generated_files() -> dict[Path, bytes]:
    records = _synthetic_records()
    feature_root = (
        REPO_ROOT
        / "datasets"
        / "ds-derivatives-example"
        / "derivatives"
        / "features"
        / "project-pilot-tabular"
    )
    return {
        REPO_ROOT / "datasets" / "ds-tabular-example" / "toy_observations.csv": _render_delimited(
            records, CANONICAL_COLUMNS, delimiter=","
        ),
        feature_root / "sources" / "toy_core.tsv": _render_delimited(
            records, CORE_COLUMNS, delimiter="\t"
        ),
        feature_root / "sources" / "toy_measurements.tsv": _render_delimited(
            records, MEASUREMENT_COLUMNS, delimiter="\t"
        ),
        feature_root / "toy_features.tsv": _render_delimited(
            records, CANONICAL_COLUMNS, delimiter="\t"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify checked-in fixtures byte-for-byte without writing",
    )
    args = parser.parse_args()

    mismatches: list[Path] = []
    generated_files = _generated_files()
    for path, expected_bytes in generated_files.items():
        if args.check:
            if not path.is_file() or path.read_bytes() != expected_bytes:
                mismatches.append(path)
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(expected_bytes)

    if mismatches:
        for path in mismatches:
            print(f"mismatch: {path.relative_to(REPO_ROOT)}", file=sys.stderr)
        return 1

    action = "verified" if args.check else "wrote"
    print(f"{action} {len(generated_files)} deterministic toy tabular fixtures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
