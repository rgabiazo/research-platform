from __future__ import annotations

from pathlib import PureWindowsPath

import pytest

from research_platform.neuro._roi_path_safety import (
    UnmappedLocalPathError,
    configured_path_is_unsafe,
    portable_path_reference,
    published_text_contains_local_path_reference,
    published_value_local_path_fields,
)


@pytest.mark.parametrize(
    "value",
    [
        "/home/example/data.tsv",
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
        "file:///home/example/data.tsv",
        "file:///C:/Data/example.tsv",
        "file://cluster.example/example-share/example.tsv",
        "--input=/home/example/data.tsv",
        "command --input /home/example/data.tsv",
        "source:/home/example/data.tsv",
        "command >/home/example/output.tsv",
        "value|/home/example/data.tsv",
        "value)/home/example/data.tsv",
        "value{/home/example/data.tsv",
        "/mnt/example/a.tsv,/mnt/example/b.tsv",
        "/mnt/example/a.tsv;/mnt/example/b.tsv",
        'command --input "/mnt/example/data file.tsv"',
        r'{"path":"C:\\Data\\example.tsv","share":"\\\\cluster.example\\example-share\\data.tsv"}',
    ],
)
def test_published_text_detects_local_path_references(value: str) -> None:
    assert published_text_contains_local_path_reference(value)


@pytest.mark.parametrize(
    "value",
    [
        "relative/path.tsv",
        "sub-001/ses-01/func/example.nii.gz",
        r"C:relative\path.tsv",
        "${DATA_ROOT}/example.tsv",
        "artifact_root:.research-platform/report.tsv",
        "root_ref:dataset_root/example.tsv",
        "https://example.org/data.tsv",
        "s3://bucket/key",
        "gs://bucket/key",
        "ssh://cluster.example/path",
        "doi:10.1234/example",
        "input/output",
        "ratio=1/2",
        "command --input relative/path.tsv",
    ],
)
def test_published_text_preserves_public_safe_values(value: str) -> None:
    assert not published_text_contains_local_path_reference(value)


@pytest.mark.parametrize(
    "value",
    [
        "/mnt/example/data.tsv",
        r"C:\Data\example.tsv",
        r"\\cluster.example\share\data.tsv",
        r"\\?\C:\Data\example.tsv",
        "~/example/data.tsv",
        r"~\example\data.tsv",
        "file:///mnt/example/data.tsv",
        "relative/../outside.tsv",
        r"relative\..\outside.tsv",
    ],
)
def test_configured_path_rejects_absolute_user_and_parent_paths(value: str) -> None:
    assert configured_path_is_unsafe(value)


@pytest.mark.parametrize(
    "value",
    [
        "relative/path.tsv",
        "sub-001/ses-01/func/example.nii.gz",
        r"C:relative\path.tsv",
        "${DATA_ROOT}/example.tsv",
        "artifact_root:.research-platform/report.tsv",
        "https://example.org/data.tsv",
    ],
)
def test_configured_path_accepts_relative_and_named_values(value: str) -> None:
    assert not configured_path_is_unsafe(value)


def test_nested_field_inventory_covers_keys_values_sequences_and_json_without_echoing_paths() -> None:
    payload = {
        "source": "/mnt/example/input.tsv",
        "nested": [
            {"command": r"tool --input C:\Data\example.tsv"},
            {r"\\cluster.example\share\field.tsv": "safe"},
            {"serialized": r'{"path":"\\\\cluster.example\\share\\data.tsv"}'},
        ],
    }

    fields = published_value_local_path_fields(payload, label="payload")

    assert fields == (
        "payload.source",
        "payload.nested[0].command",
        "payload.nested[1].<mapping-key:0>",
        "payload.nested[2].serialized",
    )
    rendered = "\n".join(fields)
    assert "/mnt/" not in rendered
    assert "C:" not in rendered
    assert "cluster.example" not in rendered


def test_portable_reference_prefers_dataset_relative_path() -> None:
    reference = portable_path_reference(
        "/srv/public/roi/maps/group/map.nii.gz",
        dataset_root="/srv/public/roi",
        named_roots={"maps_root": "/srv/public/roi/maps"},
    )

    assert reference == "maps/group/map.nii.gz"


def test_portable_reference_maps_windows_dataset_path_host_independently() -> None:
    reference = portable_path_reference(
        PureWindowsPath(r"C:\Public\ROI\masks\mask.nii.gz"),
        dataset_root=PureWindowsPath(r"C:\Public\ROI"),
    )

    assert reference == "masks/mask.nii.gz"


def test_portable_reference_uses_deepest_named_root_with_deterministic_tie_break() -> None:
    reference = portable_path_reference(
        "/srv/work/dataset/sub-001/file.tsv",
        named_roots={
            "workspace_root": "/srv/work",
            "z_dataset_root": "/srv/work/dataset",
            "a_dataset_root": "/srv/work/dataset",
        },
    )

    assert reference == "root_ref:a_dataset_root/sub-001/file.tsv"


def test_portable_reference_maps_unc_path_to_named_root() -> None:
    reference = portable_path_reference(
        r"\\cluster.example\example-share\data\file.tsv",
        named_roots={"shared_root": r"\\cluster.example\example-share"},
    )

    assert reference == "root_ref:shared_root/data/file.tsv"


def test_portable_reference_maps_complete_local_file_uri() -> None:
    reference = portable_path_reference(
        "file:///srv/public/roi/tables/values.tsv",
        dataset_root="/srv/public/roi",
    )

    assert reference == "tables/values.tsv"


@pytest.mark.parametrize(
    "value",
    [
        "relative/path.tsv",
        "sub-001/ses-01/func/example.nii.gz",
        r"C:relative\path.tsv",
        "${DATA_ROOT}/example.tsv",
        "root_ref:dataset_root/example.tsv",
        "https://example.org/data.tsv",
        "s3://bucket/key",
        "https://example.org/archive/../data.tsv",
        "doi:10.1234/example",
    ],
)
def test_portable_reference_preserves_safe_values(value: str) -> None:
    assert portable_path_reference(value) == value


@pytest.mark.parametrize(
    "value",
    [
        "/outside/example/data.tsv",
        "/srv/public/roi/../secret.tsv",
        "../outside/example.tsv",
        "~/example/data.tsv",
        r"~\example\data.tsv",
        "--input=/srv/public/roi/maps/map.nii.gz",
        'command --input "/srv/public/roi/maps/map with spaces.nii.gz"',
        r'{"path":"C:\\Data\\example.tsv"}',
    ],
)
def test_portable_reference_rejects_unmapped_or_embedded_local_values_without_echo(value: str) -> None:
    with pytest.raises(UnmappedLocalPathError) as error:
        portable_path_reference(
            value,
            dataset_root="/srv/public/roi",
            named_roots={"dataset_root": "/srv/public"},
        )

    message = str(error.value)
    assert value not in message
    assert "/srv/" not in message
    assert "C:" not in message


def test_portable_reference_exact_named_root_uses_root_name_only() -> None:
    assert portable_path_reference("/srv/data", named_roots={"data_root": "/srv/data"}) == "root_ref:data_root"


def test_portable_reference_rejects_windows_parent_traversal_beneath_dataset_root() -> None:
    with pytest.raises(UnmappedLocalPathError):
        portable_path_reference(
            PureWindowsPath(r"C:\Public\ROI\..\secret.tsv"),
            dataset_root=PureWindowsPath(r"C:\Public\ROI"),
        )


def test_portable_reference_does_not_use_a_filesystem_root_as_a_catch_all() -> None:
    with pytest.raises(UnmappedLocalPathError):
        portable_path_reference("/outside/example.tsv", named_roots={"system_root": "/"})
