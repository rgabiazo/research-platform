from __future__ import annotations

import pytest

from research_platform.analysis.mvpa._path_safety import (
    configured_path_is_unsafe,
    published_text_contains_local_path_reference,
    published_value_contains_local_path_reference,
)


@pytest.mark.parametrize(
    "value",
    [
        "/home/alice/example/data.tsv",
        "/mnt/example/data.tsv",
        "/private/var/folders/example/data.tsv",
        r"C:\Data\example.tsv",
        "D:/Data/example.tsv",
        r"\\cluster.example\example-share\data.tsv",
        "//cluster.example/example-share/data.tsv",
        r"\\?\C:\Data\example.tsv",
        r"\\?\UNC\cluster.example\example-share\data.tsv",
        "~/example/data.tsv",
        r"~\example\data.tsv",
        "file:///home/alice/example.tsv",
        "file:///C:/Data/example.tsv",
        "file://cluster.example/example-share/example.tsv",
        "--input=/home/alice/example.tsv",
        "command --input /home/alice/example.tsv",
        "source:/home/alice/example.tsv",
        "relative/one.tsv,/home/alice/example/two.tsv",
        "relative/one.tsv;/mnt/example/two.tsv",
        '--input="/home/alice/example data.tsv"',
        r'{"path":"C:\\Data\\example.tsv"}',
        r'{"path":"\\\\cluster.example\\example-share\\data.tsv"}',
        r'{"path":"\\\\?\\C:\\Data\\example.tsv"}',
        r'{"path":"\\\\?\\UNC\\cluster.example\\example-share\\data.tsv"}',
    ],
)
def test_published_text_detects_local_path_references(value: str) -> None:
    assert published_text_contains_local_path_reference(value) is True


@pytest.mark.parametrize(
    "value",
    [
        "relative/path.tsv",
        "sub-001/ses-01/func/example.nii.gz",
        r"C:relative\path.tsv",
        "${DATA_ROOT}/example.tsv",
        "artifact_root:.research-platform/report.tsv",
        "https://example.org/data.tsv",
        "s3://bucket/key",
        "gs://bucket/key",
        "ssh://cluster.example/path",
        "doi:10.1234/example",
        "input/output",
        "ratio=1/2",
        "rois/primary/sub-*/ses-01/func/*",
        "./relative/path.tsv",
        "../relative/path.tsv",
        "${DATA_ROOT:-/srv/example}/example.tsv",
        r'{"path":"relative\\path.tsv"}',
        r'{"url":"https://example.org/data.tsv"}',
    ],
)
def test_published_text_allows_public_safe_values(value: str) -> None:
    assert published_text_contains_local_path_reference(value) is False


@pytest.mark.parametrize(
    "value",
    [
        "/home/alice/example/data.tsv",
        r"C:\Data\example.tsv",
        "D:/Data/example.tsv",
        r"\\cluster.example\example-share\data.tsv",
        "//cluster.example/example-share/data.tsv",
        r"\\?\C:\Data\example.tsv",
        r"\\?\UNC\cluster.example\example-share\data.tsv",
        "~/example/data.tsv",
        r"~\example\data.tsv",
        "../example/data.tsv",
        r"..\example\data.tsv",
        "example/../data.tsv",
        r"example\..\data.tsv",
    ],
)
def test_configured_path_rejects_absolute_user_and_parent_paths(value: str) -> None:
    assert configured_path_is_unsafe(value) is True


@pytest.mark.parametrize(
    "value",
    [
        "relative/path.tsv",
        "sub-001/ses-01/func/example.nii.gz",
        r"C:relative\path.tsv",
        "${DATA_ROOT}/example.tsv",
        "artifact_root:.research-platform/report.tsv",
        ".research-platform/report.tsv",
        "input/output",
    ],
)
def test_configured_path_allows_relative_paths(value: str) -> None:
    assert configured_path_is_unsafe(value) is False


def test_published_value_scans_mapping_keys_and_nested_table_cells() -> None:
    assert published_value_contains_local_path_reference(
        [{"safe_column": "prefix C:\\Data\\example.tsv suffix"}]
    ) is True
    assert published_value_contains_local_path_reference(
        [{r"\\cluster.example\example-share\column.tsv": "safe value"}]
    ) is True
    assert published_value_contains_local_path_reference(
        [{"safe_column": "sub-001/ses-01/func/example.nii.gz"}]
    ) is False
