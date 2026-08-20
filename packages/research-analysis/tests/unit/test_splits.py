from __future__ import annotations

import json
from pathlib import Path

from research_platform.analysis.splits import create_split_manifest, load_split_manifest


def test_create_random_split_manifest() -> None:
    rows = [{"target": str(index % 2)} for index in range(10)]

    manifest = create_split_manifest(
        rows=rows,
        target_column="target",
        table_path="table.tsv",
        test_fraction=0.3,
        seed=7,
        split_strategy="random",
    )

    assert manifest["split_strategy"] == "random"
    assert len(manifest["train_rows"]) == 7
    assert len(manifest["test_rows"]) == 3
    assert sorted(manifest["train_rows"] + manifest["test_rows"]) == list(range(10))


def test_create_stratified_binary_split_manifest() -> None:
    rows = [{"target": "0"} for _ in range(4)] + [{"target": "1"} for _ in range(4)]

    manifest = create_split_manifest(
        rows=rows,
        target_column="target",
        table_path="table.tsv",
        test_fraction=0.25,
        seed=11,
        split_strategy="stratified_binary",
    )

    train_targets = {rows[index]["target"] for index in manifest["train_rows"]}
    test_targets = {rows[index]["target"] for index in manifest["test_rows"]}

    assert manifest["split_strategy"] == "stratified_binary"
    assert train_targets == {"0", "1"}
    assert test_targets == {"0", "1"}


def test_create_stratified_binned_split_manifest() -> None:
    rows = [{"score": str(index)} for index in range(12)]

    manifest = create_split_manifest(
        rows=rows,
        target_column="score",
        table_path="table.tsv",
        test_fraction=0.34,
        seed=5,
        split_strategy="stratified_binned",
        stratify_bin_count=4,
    )

    def quantile_bin(index: int) -> int:
        return index // 3

    test_bins = {quantile_bin(int(rows[index]["score"])) for index in manifest["test_rows"]}

    assert manifest["split_strategy"] == "stratified_binned"
    assert manifest["stratify_bin_count"] == 4
    assert test_bins == {0, 1, 2, 3}


def test_load_split_manifest_backfills_legacy_strategy(tmp_path: Path) -> None:
    manifest_path = tmp_path / "legacy_split.json"
    manifest_path.write_text(
        json.dumps(
            {
                "kind": "tabular_split",
                "table_path": "table.tsv",
                "target_column": "target",
                "row_count": 4,
                "seed": 23,
                "test_fraction": 0.25,
                "train_rows": [0, 1, 2],
                "test_rows": [3],
            }
        ),
        encoding="utf-8",
    )

    manifest = load_split_manifest(manifest_path)

    assert manifest["split_strategy"] == "stratified_binary"
