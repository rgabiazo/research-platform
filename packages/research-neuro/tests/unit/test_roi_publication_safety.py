from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
BIDS_PACKAGE_ROOT = WORKSPACE_ROOT / "packages" / "research-bids"
sys.path.insert(0, str(PACKAGE_ROOT / "src"))
sys.path.insert(0, str(BIDS_PACKAGE_ROOT / "src"))

from research_platform.neuro import roi_publication
from research_platform.neuro._roi_path_safety import published_text_contains_local_path_reference
from research_platform.neuro.roi_execution import RoiExecutionContext
from research_platform.neuro.roi_publication import (
    RoiPublicationError,
    publish_loso_featquery_extraction_result,
    publish_loso_roi_build_result,
)


TEXT_SUFFIXES = frozenset({".csv", ".json", ".md", ".tsv", ".txt"})


def _context(root: Path) -> tuple[RoiExecutionContext, dict[str, Path]]:
    paths = {
        "project_root": root / "project" / "project-demo",
        "artifacts_root": root / "artifacts",
        "dataset_derivatives_root": root / "public-derivatives",
        "runtime_root": root / "runtime",
        "source_root": root / "source-inputs",
    }
    paths["project_root"].mkdir(parents=True, exist_ok=True)
    context = RoiExecutionContext(
        workspace_root=root,
        project_root=paths["project_root"],
        artifacts_root=paths["artifacts_root"],
        project_name="project-demo",
        root_refs={
            "dataset_derivatives_root": paths["dataset_derivatives_root"],
            "runtime_root": paths["runtime_root"],
            "source_root": paths["source_root"],
        },
    )
    return context, paths


def _publication_root(paths: dict[str, Path]) -> Path:
    return paths["dataset_derivatives_root"] / "roi-loso-flame1"


def _roi_document(*, existing_output: str | None = None) -> dict[str, object]:
    publication: dict[str, object] = {
        "enabled": True,
        "layout": "loso_flame1_bidslike",
        "root": {
            "root_ref": "dataset_derivatives_root",
            "path": "roi-loso-flame1",
        },
        "dataset_description": {
            "name": "Synthetic ROI derivative",
            "generated_by_name": "synthetic-roi-workflow",
            "dataset_links": {
                "documentation": "https://example.org/synthetic-roi",
                "archive": "s3://public-example/roi",
            },
        },
        "map_desc": "{model}LOSOFlame1",
        "mask_desc": "{model}LOSOFlame1Sphere{sphere_radius_mm}mm",
    }
    if existing_output is not None:
        publication["existing_output"] = existing_output
    return {
        "roi_set": {
            "name": "toy_roi_set",
            "backend": "fsl_flame1",
            "subjects": ["sub-001", "sub-002"],
            "held_out_subjects": ["sub-001"],
            "session": "ses-01",
            "task": "exampletask",
            "direction": "AP",
            "model": "ToyModel",
            "space": "ToySpace",
            "resolution": "2",
            "publication": publication,
            "contrasts": [
                {
                    "id": "ToyContrast",
                    "cope_number": 1,
                    "desc": "ToyContrast",
                }
            ],
            "rois": [
                {
                    "label": "ToySeed",
                    "family": "loso_group_map",
                    "backend": "fsl_flame1",
                    "desc": "ToyContrast",
                    "contrast": "ToyContrast",
                    "seed_coordinate": [0, 0, 0],
                    "search_radius_mm": 8,
                    "sphere_radius_mm": 4,
                    "z_threshold": 3.1,
                }
            ],
        }
    }


def _extraction_document(*, existing_output: str | None = None) -> dict[str, object]:
    publication: dict[str, object] = {
        "enabled": True,
        "layout": "loso_flame1_bidslike",
        "root": {
            "root_ref": "dataset_derivatives_root",
            "path": "roi-loso-flame1",
        },
        "dataset_description": {
            "name": "Synthetic ROI derivative",
            "generated_by_name": "synthetic-roi-workflow",
        },
        "table_desc": "{model}LOSOFlame1Featquery",
    }
    if existing_output is not None:
        publication["existing_output"] = existing_output
    return {
        "extraction_set": {
            "name": "toy_extraction_set",
            "roi_set": "toy_roi_set",
            "session": "ses-01",
            "task": "exampletask",
            "direction": "AP",
            "model": "ToyModel",
            "publication": publication,
            "targets": [
                {
                    "name": "ToyValues",
                    "backend": "fsl_featquery",
                    "metrics": ["mean_cope", "roi_voxel_count"],
                }
            ],
        }
    }


def _write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


def _replace_text(path: Path, text: str) -> None:
    replacement = path.with_name(f"{path.name}.replacement")
    _write_text(replacement, text)
    replacement.replace(path)


def _build_action(paths: dict[str, Path], *, unmapped_group_mask: str | None = None) -> SimpleNamespace:
    runtime_root = paths["runtime_root"]
    source_root = paths["source_root"]
    zstat_path = _write_text(runtime_root / "maps" / "toy_zstat.nii.gz", "synthetic z statistic\n")
    mask_path = _write_text(runtime_root / "masks" / "toy_mask.nii.gz", "synthetic mask\n")
    sidecar_path = runtime_root / "masks" / "toy_mask.json"

    group_mask = source_root / "group" / "group_mask.nii.gz"
    cope_path = source_root / "sub-002" / "cope1.nii.gz"
    varcope_path = source_root / "sub-002" / "varcope1.nii.gz"
    training_mask = source_root / "sub-002" / "mask.nii.gz"
    heldout_mask = source_root / "sub-001" / "mask.nii.gz"
    coverage_mask = source_root / "coverage" / "coverage_mask.nii.gz"
    for source in (group_mask, cope_path, varcope_path, training_mask, heldout_mask, coverage_mask):
        _write_text(source, "synthetic source\n")

    sidecar_payload = {
        "seed_coordinate": [0, 0, 0],
        "loso_peak_coordinate": [1, 2, 3],
        "selected_peak_z": 4.25,
        "sphere_radius_mm": 4,
        "search_radius_mm": 8,
        "coverage_masks": {
            "nested": [
                str(coverage_mask),
                {"archive": "s3://public-example/coverage-mask"},
            ]
        },
        "voxel_count": 27,
        "fallback_status": "thresholded",
        "qc_flags": ["pass"],
        "warnings": [],
    }
    _write_text(sidecar_path, json.dumps(sidecar_payload, sort_keys=True) + "\n")

    return SimpleNamespace(
        family="loso_group_map",
        backend="fsl_flame1",
        roi_label="ToySeed",
        mask_path=mask_path,
        sidecar_path=sidecar_path,
        metadata={
            "entities": {
                "subject_id": "001",
                "session_id": "01",
                "task_id": "exampletask",
                "direction": "AP",
                "space": "ToySpace",
                "resolution": "2",
                "model": "ToyModel",
                "datatype": "func",
            },
            "roi_parameters": {
                "seed_coordinate": [0, 0, 0],
                "search_radius_mm": 8,
                "sphere_radius_mm": 4,
            },
            "loso_group_job": {
                "contrast": {
                    "contrast_id": "ToyContrast",
                    "cope_number": 1,
                    "desc": "ToyContrast",
                },
                "zstat_path": zstat_path,
                "heldout_subject": "001",
                "session_id": "01",
                "task_id": "exampletask",
                "model": "ToyModel",
                "group_mask_path": unmapped_group_mask or group_mask,
                "training_inputs": [
                    {
                        "subject_id": "002",
                        "cope_path": cope_path,
                        "varcope_path": varcope_path,
                        "mask_path": training_mask,
                    }
                ],
                "heldout_input": {
                    "subject_id": "001",
                    "mask_path": heldout_mask,
                },
                "output_root": runtime_root,
            },
        },
    )


def _write_extraction_tables(paths: dict[str, Path], *, unsafe_note: str | None = None) -> tuple[Path, Path]:
    runtime_root = paths["runtime_root"]
    source_root = paths["source_root"]
    publication_root = _publication_root(paths)
    values_path = runtime_root / "tables" / "group_ses-01_task-exampletask_desc-ToyModel_values.tsv"
    qc_path = values_path.with_name("group_ses-01_task-exampletask_desc-ToyModel_qc.tsv")
    header = ["subject_id", "mean_cope", "roi_voxel_count", "safe_reference"]
    values = ["001", "1.25", "27", "https://example.org/value"]
    if unsafe_note is not None:
        header.append("note")
        values.append(unsafe_note)
    _write_table(values_path, header, [values])
    _write_table(
        qc_path,
        [
            "subject_id",
            "mean_cope",
            "roi_voxel_count",
            "feat_dir",
            "roi_mask_path",
            "featquery_output_dir",
            "report_path",
            "featquery_command",
            "usable",
            "qc_flags",
            "warnings",
        ],
        [
            [
                "001",
                "1.25",
                "27",
                str(source_root / "sub-001" / "toy.feat"),
                str(publication_root / "masks" / "sub-001" / "toy_mask.nii.gz"),
                str(runtime_root / "featquery" / "output"),
                str(runtime_root / "featquery" / "report.txt"),
                json.dumps(["featquery", str(source_root / "sub-001" / "toy.feat")]),
                "true",
                "pass",
                "",
            ]
        ],
    )
    return values_path, qc_path


def _write_table(path: Path, fieldnames: list[str], rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(fieldnames)
        writer.writerows(rows)


def _extraction_actions(tables: tuple[Path, Path]) -> tuple[SimpleNamespace, ...]:
    return tuple(
        SimpleNamespace(
            backend="fsl_featquery",
            roi_label="ToySeed",
            table_path=table,
            metrics=("mean_cope", "roi_voxel_count"),
            metadata={
                "session_id": "01",
                "task_id": "exampletask",
                "direction": "AP",
                "model": "ToyModel",
                "source_contrast": "ToyContrast",
                "cope": "1",
            },
        )
        for table in tables
    )


def _assert_no_publication_temporary_paths(root: Path) -> None:
    parent = root.parent
    assert not list(parent.glob(f".{root.name}.publication-*"))


def _assert_public_tree_is_portable(root: Path) -> None:
    assert root.is_dir()
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        assert not published_text_contains_local_path_reference(relative), relative
        if path.is_file() and path.suffix.casefold() in TEXT_SUFFIXES:
            content = path.read_text(encoding="utf-8")
            assert not published_text_contains_local_path_reference(content), relative


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _publish_build(root: Path, *, existing_output: str | None = None) -> tuple[Path, SimpleNamespace]:
    context, paths = _context(root)
    action = _build_action(paths)
    result = publish_loso_roi_build_result(
        _roi_document(existing_output=existing_output),
        actions=[action],
        context=context,
    )
    assert result.complete
    return _publication_root(paths), action


def test_build_publication_converts_nested_sources_and_has_no_public_path_leaks(tmp_path: Path) -> None:
    context, paths = _context(tmp_path)
    action = _build_action(paths)

    result = publish_loso_roi_build_result(_roi_document(), actions=[action], context=context)

    assert result.complete
    root = _publication_root(paths)
    map_sidecar = json.loads(next((root / "maps").rglob("*.json")).read_text(encoding="utf-8"))
    mask_sidecar = json.loads(next((root / "masks").rglob("*.json")).read_text(encoding="utf-8"))
    assert map_sidecar["PublishedPath"].startswith("maps/group/")
    assert map_sidecar["Sources"][0].startswith("root_ref:runtime_root/")
    assert all(source.startswith("root_ref:source_root/") for source in map_sidecar["Sources"][1:])
    assert mask_sidecar["DefiningMap"] == map_sidecar["PublishedPath"]
    assert mask_sidecar["CoverageMasksApplied"]["nested"][0].startswith("root_ref:source_root/")
    assert mask_sidecar["CoverageMasksApplied"]["nested"][1] == {
        "archive": "s3://public-example/coverage-mask"
    }
    assert "RuntimePath" not in map_sidecar
    assert "FLAMECommand" not in map_sidecar
    assert "RuntimeSidecar" not in map_sidecar
    assert "RuntimeSidecar" not in mask_sidecar
    _assert_public_tree_is_portable(root)
    _assert_no_publication_temporary_paths(root)


def test_table_publication_converts_inputs_and_excludes_private_runtime_columns(tmp_path: Path) -> None:
    context, paths = _context(tmp_path)
    tables = _write_extraction_tables(paths)

    result = publish_loso_featquery_extraction_result(
        _extraction_document(),
        roi_set_document=None,
        actions=_extraction_actions(tables),
        tables=tables,
        context=context,
    )

    assert result.complete
    root = _publication_root(paths)
    qc_path = next((root / "tables").rglob("*QC*_roistats.tsv"))
    with qc_path.open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle, delimiter="\t"))
    assert row["feat_dir"].startswith("root_ref:source_root/")
    assert row["roi_mask_path"].startswith("masks/sub-001/")
    assert "featquery_output_dir" not in row
    assert "report_path" not in row
    assert "featquery_command" not in row
    _assert_public_tree_is_portable(root)
    _assert_no_publication_temporary_paths(root)


def test_unmapped_source_fails_without_echo_or_partial_output(tmp_path: Path) -> None:
    context, paths = _context(tmp_path)
    unsafe_source = "/outside/local-only/group mask.nii.gz"
    action = _build_action(paths, unmapped_group_mask=unsafe_source)

    with pytest.raises(RoiPublicationError) as error:
        publish_loso_roi_build_result(_roi_document(), actions=[action], context=context)

    message = str(error.value)
    assert unsafe_source not in message
    assert "/outside/" not in message
    root = _publication_root(paths)
    assert not root.exists()
    _assert_no_publication_temporary_paths(root)


@pytest.mark.parametrize(
    "unsafe_note",
    [
        "command --input /outside/local-only/value.tsv",
        r"command --input C:\Data\example.tsv",
        r"command --input \\cluster.example\example-share\value.tsv",
        "command --input ~/local-only/value.tsv",
        "command --input file:///home/example/value.tsv",
        r'{"path":"C:\\Data\\example.tsv"}',
        "command >/home/example/output.tsv",
        "value|/home/example/value.tsv",
        "value)/home/example/value.tsv",
        "value{/home/example/value.tsv",
    ],
)
def test_unsafe_table_cell_fails_without_echo_or_partial_output(
    tmp_path: Path,
    unsafe_note: str,
) -> None:
    context, paths = _context(tmp_path)
    tables = _write_extraction_tables(paths, unsafe_note=unsafe_note)
    root = _publication_root(paths)
    sentinel = _write_text(root / "dataset_description.json", "sentinel dataset metadata\n")

    with pytest.raises(RoiPublicationError) as error:
        publish_loso_featquery_extraction_result(
            _extraction_document(existing_output="replace"),
            roi_set_document=None,
            actions=_extraction_actions(tables),
            tables=tables,
            context=context,
        )

    message = str(error.value)
    assert unsafe_note not in message
    assert sentinel.read_text(encoding="utf-8") == "sentinel dataset metadata\n"
    assert [path.relative_to(root).as_posix() for path in root.rglob("*")] == [
        "dataset_description.json"
    ]
    _assert_no_publication_temporary_paths(root)


@pytest.mark.parametrize(
    "unsafe_reference",
    [
        "/outside/local-only/value.tsv",
        r"C:\Data\example.tsv",
        r"\\cluster.example\example-share\value.tsv",
        "~/local-only/value.tsv",
        "file:///home/example/value.tsv",
    ],
)
def test_nested_json_path_fails_without_echo_or_partial_output(
    tmp_path: Path,
    unsafe_reference: str,
) -> None:
    context, paths = _context(tmp_path)
    action = _build_action(paths)
    document = _roi_document()
    publication = document["roi_set"]["publication"]  # type: ignore[index]
    dataset = publication["dataset_description"]  # type: ignore[index]
    dataset["dataset_links"]["unsafe"] = {"nested": [unsafe_reference]}  # type: ignore[index]

    with pytest.raises(RoiPublicationError) as error:
        publish_loso_roi_build_result(document, actions=[action], context=context)

    assert unsafe_reference not in str(error.value)
    root = _publication_root(paths)
    assert not root.exists()
    _assert_no_publication_temporary_paths(root)


def test_default_collision_rejects_before_changing_existing_sentinel(tmp_path: Path) -> None:
    context, paths = _context(tmp_path)
    action = _build_action(paths)
    root = _publication_root(paths)
    sentinel = _write_text(root / "dataset_description.json", "sentinel dataset metadata\n")

    with pytest.raises(RoiPublicationError, match="refused existing output"):
        publish_loso_roi_build_result(_roi_document(), actions=[action], context=context)

    assert sentinel.read_text(encoding="utf-8") == "sentinel dataset metadata\n"
    assert [path.relative_to(root).as_posix() for path in root.rglob("*")] == [
        "dataset_description.json"
    ]
    _assert_no_publication_temporary_paths(root)


def test_explicit_replace_updates_conflicts_and_preserves_unrelated_files(tmp_path: Path) -> None:
    context, paths = _context(tmp_path)
    action = _build_action(paths)
    root = _publication_root(paths)
    sentinel = _write_text(root / "dataset_description.json", "sentinel dataset metadata\n")
    unrelated = _write_text(root / "unrelated.txt", "keep this file\n")

    result = publish_loso_roi_build_result(
        _roi_document(existing_output="replace"),
        actions=[action],
        context=context,
    )

    assert result.complete
    assert json.loads(sentinel.read_text(encoding="utf-8"))["DatasetType"] == "derivative"
    assert unrelated.read_text(encoding="utf-8") == "keep this file\n"
    _assert_public_tree_is_portable(root)
    _assert_no_publication_temporary_paths(root)


def test_publication_rejects_symlinked_parent_without_writing_outside_root(tmp_path: Path) -> None:
    context, paths = _context(tmp_path)
    action = _build_action(paths)
    root = _publication_root(paths)
    outside = tmp_path / "outside-publication-root"
    root.mkdir(parents=True)
    outside.mkdir()
    (root / "maps").symlink_to(outside, target_is_directory=True)

    with pytest.raises(RoiPublicationError, match="destination parent is a symbolic link"):
        publish_loso_roi_build_result(_roi_document(), actions=[action], context=context)

    assert not list(outside.iterdir())
    assert (root / "maps").is_symlink()
    assert not (root / "dataset_description.json").exists()
    _assert_no_publication_temporary_paths(root)


def test_publication_rejects_directory_at_file_destination_without_data_loss(tmp_path: Path) -> None:
    context, paths = _context(tmp_path)
    action = _build_action(paths)
    root = _publication_root(paths)
    collision = root / "dataset_description.json"
    nested_sentinel = _write_text(collision / "unrelated.txt", "keep nested content\n")

    with pytest.raises(RoiPublicationError, match="not a replaceable file"):
        publish_loso_roi_build_result(
            _roi_document(existing_output="replace"),
            actions=[action],
            context=context,
        )

    assert nested_sentinel.read_text(encoding="utf-8") == "keep nested content\n"
    assert not (root / "README.md").exists()
    _assert_no_publication_temporary_paths(root)


def test_published_maps_and_masks_are_independent_copies(tmp_path: Path) -> None:
    root, action = _publish_build(tmp_path)
    public_map = next((root / "maps").rglob("*.nii.gz"))
    public_mask = next((root / "masks").rglob("*.nii.gz"))
    map_bytes = public_map.read_bytes()
    mask_bytes = public_mask.read_bytes()

    Path(action.metadata["loso_group_job"]["zstat_path"]).write_text(
        "mutated runtime map\n",
        encoding="utf-8",
    )
    Path(action.mask_path).write_text("mutated runtime mask\n", encoding="utf-8")

    assert public_map.read_bytes() == map_bytes
    assert public_mask.read_bytes() == mask_bytes


def test_mid_render_failure_leaves_existing_sentinel_and_no_temporary_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context, paths = _context(tmp_path)
    action = _build_action(paths)
    root = _publication_root(paths)
    sentinel = _write_text(root / "dataset_description.json", "sentinel dataset metadata\n")
    real_stage = roi_publication._stage_source_file
    call_count = 0

    def fail_second_stage(source: Path, target: Path) -> None:
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise OSError("injected render failure")
        real_stage(source, target)

    monkeypatch.setattr(roi_publication, "_stage_source_file", fail_second_stage)

    with pytest.raises(RoiPublicationError, match="could not render output"):
        publish_loso_roi_build_result(
            _roi_document(existing_output="replace"),
            actions=[action],
            context=context,
        )

    assert call_count == 2
    assert sentinel.read_text(encoding="utf-8") == "sentinel dataset metadata\n"
    assert [path.relative_to(root).as_posix() for path in root.rglob("*")] == [
        "dataset_description.json"
    ]
    _assert_no_publication_temporary_paths(root)


def test_serialization_failure_leaves_existing_sentinel_unchanged(tmp_path: Path) -> None:
    context, paths = _context(tmp_path)
    action = _build_action(paths)
    root = _publication_root(paths)
    sentinel = _write_text(root / "dataset_description.json", "sentinel dataset metadata\n")
    document = _roi_document(existing_output="replace")
    publication = document["roi_set"]["publication"]  # type: ignore[index]
    dataset = publication["dataset_description"]  # type: ignore[index]
    dataset["dataset_links"]["unsupported"] = object()  # type: ignore[index]

    with pytest.raises(RoiPublicationError, match="could not render output"):
        publish_loso_roi_build_result(document, actions=[action], context=context)

    assert sentinel.read_text(encoding="utf-8") == "sentinel dataset metadata\n"
    assert [path.relative_to(root).as_posix() for path in root.rglob("*")] == [
        "dataset_description.json"
    ]
    _assert_no_publication_temporary_paths(root)


def test_partial_parent_creation_failure_removes_created_directories(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context, paths = _context(tmp_path)
    action = _build_action(paths)
    root = _publication_root(paths)
    real_mkdir = Path.mkdir
    failure_injected = False

    def fail_publication_root_mkdir(path: Path, *args: object, **kwargs: object) -> None:
        nonlocal failure_injected
        if path == root and not failure_injected:
            failure_injected = True
            raise OSError("injected parent creation failure")
        real_mkdir(path, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", fail_publication_root_mkdir)

    with pytest.raises(RoiPublicationError, match="prior destination set was restored"):
        publish_loso_roi_build_result(_roi_document(), actions=[action], context=context)

    assert failure_injected
    assert not paths["dataset_derivatives_root"].exists()
    assert not list(tmp_path.glob(f".{root.name}.publication-*"))


def test_one_time_staging_cleanup_failure_is_retried(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context, paths = _context(tmp_path)
    action = _build_action(paths)
    root = _publication_root(paths)
    real_rmtree = roi_publication.shutil.rmtree
    cleanup_attempts = 0

    def fail_first_cleanup(path: str | Path, *args: object, **kwargs: object) -> None:
        nonlocal cleanup_attempts
        if Path(path).name.startswith(f".{root.name}.publication-"):
            cleanup_attempts += 1
            if cleanup_attempts == 1:
                raise OSError("injected cleanup failure")
        real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(roi_publication.shutil, "rmtree", fail_first_cleanup)

    result = publish_loso_roi_build_result(_roi_document(), actions=[action], context=context)

    assert result.complete
    assert cleanup_attempts == 2
    assert not list(tmp_path.glob(f".{root.name}.publication-*"))


def test_mid_promotion_failure_restores_complete_existing_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, action = _publish_build(tmp_path)
    unrelated = _write_text(root / "unrelated.txt", "keep this file\n")
    original = _tree_bytes(root)
    _replace_text(Path(action.metadata["loso_group_job"]["zstat_path"]), "replacement map\n")
    _replace_text(Path(action.mask_path), "replacement mask\n")
    real_replace = roi_publication.os.replace
    candidate_promotions = 0
    failure_injected = False
    rollback_replace_failures = 0

    def fail_second_candidate(source: str | Path, destination: str | Path) -> None:
        nonlocal candidate_promotions, failure_injected, rollback_replace_failures
        source_path = Path(source)
        if "candidate" in source_path.parts:
            candidate_promotions += 1
            if candidate_promotions == 2 and not failure_injected:
                failure_injected = True
                raise OSError("injected promotion failure")
        if "backup" in source_path.parts:
            rollback_replace_failures += 1
            raise OSError("injected rollback replace failure")
        real_replace(source, destination)

    monkeypatch.setattr(roi_publication.os, "replace", fail_second_candidate)
    context, _paths = _context(tmp_path)

    with pytest.raises(RoiPublicationError, match="prior destination set was restored"):
        publish_loso_roi_build_result(
            _roi_document(existing_output="replace"),
            actions=[action],
            context=context,
        )

    assert candidate_promotions == 2
    assert failure_injected
    assert rollback_replace_failures > 0
    assert unrelated.read_text(encoding="utf-8") == "keep this file\n"
    assert _tree_bytes(root) == original
    _assert_no_publication_temporary_paths(root)


def test_complete_synthetic_derivative_is_byte_deterministic_and_recursively_portable(
    tmp_path: Path,
) -> None:
    published: list[Path] = []
    for run_name in ("run-a", "run-b"):
        run_root = tmp_path / run_name
        context, paths = _context(run_root)
        action = _build_action(paths)
        build_result = publish_loso_roi_build_result(
            _roi_document(),
            actions=[action],
            context=context,
        )
        tables = _write_extraction_tables(paths)
        extraction_result = publish_loso_featquery_extraction_result(
            _extraction_document(existing_output="replace"),
            roi_set_document=None,
            actions=_extraction_actions(tables),
            tables=tables,
            context=context,
        )
        assert build_result.complete
        assert extraction_result.complete
        root = _publication_root(paths)
        _assert_public_tree_is_portable(root)
        _assert_no_publication_temporary_paths(root)
        published.append(root)

    assert _tree_bytes(published[0]) == _tree_bytes(published[1])
