from __future__ import annotations

from pathlib import Path
import random
from typing import Any

from ._tabular import numeric_value
from ._tabular import read_json, write_json


def create_split_manifest(
    *,
    rows: list[dict[str, str]],
    target_column: str,
    table_path: str | Path,
    test_fraction: float,
    seed: int,
    split_strategy: str = "stratified_binary",
    stratify_bin_count: int = 5,
) -> dict[str, Any]:
    row_count = len(rows)
    if row_count < 2:
        raise ValueError("Split creation requires at least two rows.")
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1.")

    randomizer = random.Random(seed)
    split_groups = _split_groups(
        rows=rows,
        target_column=target_column,
        split_strategy=split_strategy,
        stratify_bin_count=stratify_bin_count,
    )

    train_rows: list[int] = []
    test_rows: list[int] = []
    for indices in split_groups.values():
        randomizer.shuffle(indices)
        group_test_count = max(1, int(round(len(indices) * test_fraction))) if len(indices) > 1 else 0
        if group_test_count >= len(indices):
            group_test_count = len(indices) - 1
        test_rows.extend(indices[:group_test_count])
        train_rows.extend(indices[group_test_count:])

    train_rows.sort()
    test_rows.sort()
    manifest = {
        "kind": "tabular_split",
        "table_path": str(Path(table_path)),
        "target_column": target_column,
        "row_count": row_count,
        "seed": seed,
        "test_fraction": test_fraction,
        "split_strategy": split_strategy,
        "train_rows": train_rows,
        "test_rows": test_rows,
    }
    if split_strategy == "stratified_binned":
        manifest["stratify_bin_count"] = stratify_bin_count
    return manifest


def _split_groups(
    *,
    rows: list[dict[str, str]],
    target_column: str,
    split_strategy: str,
    stratify_bin_count: int,
) -> dict[str, list[int]]:
    if split_strategy == "random":
        return {"all_rows": list(range(len(rows)))}
    if split_strategy == "stratified_binary":
        return _binary_groups(rows=rows, target_column=target_column)
    if split_strategy == "stratified_binned":
        return _binned_groups(rows=rows, target_column=target_column, stratify_bin_count=stratify_bin_count)
    raise ValueError(
        f"Unsupported split strategy {split_strategy!r}. Use random, stratified_binary, or stratified_binned."
    )


def _binary_groups(*, rows: list[dict[str, str]], target_column: str) -> dict[str, list[int]]:
    by_target: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        by_target.setdefault(str(row[target_column]), []).append(index)
    if len(by_target) != 2:
        raise ValueError(
            f"stratified_binary requires exactly two target values in column {target_column!r}; found {len(by_target)}."
        )
    return by_target


def _binned_groups(
    *,
    rows: list[dict[str, str]],
    target_column: str,
    stratify_bin_count: int,
) -> dict[str, list[int]]:
    if stratify_bin_count < 2:
        raise ValueError("stratify_bin_count must be at least 2 for stratified_binned splits.")

    values = [
        numeric_value(row[target_column], column=target_column, row_number=index + 1) for index, row in enumerate(rows)
    ]
    effective_bin_count = min(stratify_bin_count, len(values))
    ordered = sorted(enumerate(values), key=lambda item: (item[1], item[0]))
    binned_groups: dict[str, list[int]] = {}
    for rank, (row_index, _) in enumerate(ordered):
        bin_index = min(effective_bin_count - 1, (rank * effective_bin_count) // len(values))
        binned_groups.setdefault(f"bin_{bin_index}", []).append(row_index)
    return binned_groups


def write_split_manifest(path: str | Path, manifest: dict[str, Any]) -> Path:
    return write_json(path, manifest)


def load_split_manifest(path: str | Path, *, expected_sha256: str | None = None) -> dict[str, Any]:
    manifest = read_json(path, expected_sha256=expected_sha256)
    if manifest.get("kind") != "tabular_split":
        raise ValueError(f"Unsupported split manifest: {path}")
    if "split_strategy" not in manifest:
        manifest["split_strategy"] = "stratified_binary"
    return manifest


def split_membership(split_manifest: dict[str, Any]) -> dict[int, str]:
    membership: dict[int, str] = {}
    for row_number in split_manifest["train_rows"]:
        membership[int(row_number)] = "train"
    for row_number in split_manifest["test_rows"]:
        membership[int(row_number)] = "test"
    return membership
