from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import os
from pathlib import Path
import sys
from unittest import mock

import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PACKAGE_ROOT / "src"))

np = pytest.importorskip("numpy")
nib = pytest.importorskip("nibabel")

from research_platform.neuro import _roi_runtime_outputs as runtime_outputs
from research_platform.neuro import roi_execution
from research_platform.neuro._roi_runtime_outputs import (
    RoiRuntimeOutput,
    RoiRuntimeOutputError,
    preflight_runtime_outputs,
)
from research_platform.neuro.roi_execution import (
    RoiExecutionContext,
    plan_roi_build,
    plan_roi_extraction,
    preflight_roi_build,
    preflight_roi_extraction,
    run_roi_build,
    run_roi_extraction,
)
from research_platform.neuro.roi_scaffold import (
    build_extraction_set_document,
    build_roi_set_document,
)


def test_generic_build_preflight_is_read_only_and_reports_stable_checks(tmp_path: Path) -> None:
    context = _context(tmp_path)
    document = _roi_document()
    _write_reference(context)
    before = _tree_snapshot(tmp_path)

    report = preflight_roi_build(document, context=context)

    assert report.ready_for_execution
    assert _tree_snapshot(tmp_path) == before
    assert {
        "configuration_valid",
        "roi_family_supported",
        "configured_root_available",
        "output_collision",
        "python_dependency_available",
        "input_exists",
        "image_readable",
        "external_tool_available",
    }.issubset({check.check_id for check in report.checks})
    assert all(check.message for check in report.checks)
    assert not list(tmp_path.rglob(".roi-runtime-*"))


@pytest.mark.parametrize("reference_state", ["missing", "unreadable"])
def test_build_input_failures_are_reported_before_writes(
    tmp_path: Path,
    reference_state: str,
) -> None:
    context = _context(tmp_path)
    document = _roi_document()
    reference = context.project_root / "inputs/roi/example_reference.nii.gz"
    if reference_state == "unreadable":
        reference.parent.mkdir(parents=True)
        reference.write_text("not a NIfTI image\n", encoding="utf-8")
    before = _tree_snapshot(tmp_path)

    report = preflight_roi_build(document, context=context)

    assert not report.ready_for_execution
    failed_ids = {check.check_id for check in report.checks if check.status == "error"}
    expected = "input_exists" if reference_state == "missing" else "image_readable"
    assert expected in failed_ids
    with mock.patch.object(roi_execution, "_execute_build_action") as executor:
        with pytest.raises(RoiRuntimeOutputError, match="not ready for execution"):
            run_roi_build(document, context=context)
    executor.assert_not_called()
    assert _tree_snapshot(tmp_path) == before


def test_build_collision_fails_before_execution_and_preserves_sentinel(tmp_path: Path) -> None:
    context = _context(tmp_path)
    document = _roi_document()
    _write_reference(context)
    action = plan_roi_build(document, context=context).actions[0]
    action.sidecar_path.parent.mkdir(parents=True)
    action.sidecar_path.write_text("sentinel\n", encoding="utf-8")

    with mock.patch.object(roi_execution, "_execute_build_action") as executor:
        with pytest.raises(RoiRuntimeOutputError, match="already exists"):
            run_roi_build(document, context=context)

    executor.assert_not_called()
    assert action.sidecar_path.read_text(encoding="utf-8") == "sentinel\n"
    assert not action.mask_path.exists()
    assert not list(tmp_path.rglob(".roi-runtime-*"))


def test_build_rejects_symlinked_configured_output_root_before_execution(tmp_path: Path) -> None:
    context = _context(tmp_path)
    document = _roi_document()
    _write_reference(context)
    outside = tmp_path / "outside-build"
    outside.mkdir()
    linked_root = context.project_root / "linked-output"
    linked_root.symlink_to(outside, target_is_directory=True)
    document["roi_set"]["outputs"] = {"root": "linked-output"}
    before = _tree_snapshot(outside)

    with mock.patch.object(roi_execution, "_execute_build_action") as executor:
        with pytest.raises(RoiRuntimeOutputError, match="symbolic link"):
            run_roi_build(document, context=context)

    executor.assert_not_called()
    assert _tree_snapshot(outside) == before
    assert not list(tmp_path.rglob(".roi-runtime-*"))


def test_configured_build_replacement_is_deterministic(tmp_path: Path) -> None:
    context = _context(tmp_path)
    document = _roi_document(existing_output="replace")
    _write_reference(context)

    first = run_roi_build(document, context=context)
    paths = _build_output_paths(first)
    first_hashes = _hashes(paths)
    second = run_roi_build(document, context=context)

    assert second.executed
    assert _hashes(paths) == first_hashes
    assert ".roi-runtime-" not in paths[1].read_text(encoding="utf-8")
    assert second.actions[0].result["mask_path"] == str(second.actions[0].mask_path)
    assert second.actions[0].result["sidecar_path"] == str(second.actions[0].sidecar_path)
    assert not list(tmp_path.rglob(".roi-runtime-*"))


def test_build_execution_failure_leaves_sentinels_and_no_partial_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context(tmp_path)
    document = _roi_document(existing_output="replace", roi_count=2)
    _write_reference(context)
    plan = plan_roi_build(document, context=context)
    sentinels = _seed_sentinels(_build_output_paths(plan))
    original = roi_execution._execute_build_action
    calls = 0

    def fail_second(action: object, roi: object, *, context: RoiExecutionContext) -> object:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected build failure")
        return original(action, roi, context=context)

    monkeypatch.setattr(roi_execution, "_execute_build_action", fail_second)
    with pytest.raises(RuntimeError, match="injected build failure"):
        run_roi_build(document, context=context)

    assert _read_paths(sentinels) == sentinels
    assert not list(tmp_path.rglob(".roi-runtime-*"))


def test_build_promotion_failure_rolls_back_complete_output_set(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context(tmp_path)
    document = _roi_document(existing_output="replace")
    _write_reference(context)
    plan = plan_roi_build(document, context=context)
    sentinels = _seed_sentinels(_build_output_paths(plan))
    _inject_second_candidate_promotion_failure(monkeypatch)

    with pytest.raises(RoiRuntimeOutputError, match="prior destination set was restored"):
        run_roi_build(document, context=context)

    assert _read_paths(sentinels) == sentinels
    assert not list(tmp_path.rglob(".roi-runtime-*"))


def test_extraction_preflight_is_read_only_and_detects_incompatible_geometry(tmp_path: Path) -> None:
    context = _context(tmp_path)
    roi_document, extraction_document = _prepare_extraction(context, value_shape=(7, 7, 7))
    before = _tree_snapshot(tmp_path)

    report = preflight_roi_extraction(
        extraction_document,
        roi_set_document=roi_document,
        context=context,
    )

    assert not report.ready_for_execution
    assert _tree_snapshot(tmp_path) == before
    geometry = [
        check
        for check in report.checks
        if check.check_id == "image_geometry_compatible" and check.status == "error"
    ]
    assert geometry
    assert "incompatible geometry" in geometry[0].message
    plan = plan_roi_extraction(
        extraction_document,
        roi_set_document=roi_document,
        context=context,
    )
    assert not any(path.exists() for path in _extraction_output_paths(plan))
    assert not list(tmp_path.rglob(".roi-runtime-*"))


def test_extraction_collision_fails_before_table_writes(tmp_path: Path) -> None:
    context = _context(tmp_path)
    roi_document, extraction_document = _prepare_extraction(context)
    plan = plan_roi_extraction(
        extraction_document,
        roi_set_document=roi_document,
        context=context,
    )
    values_path = plan.tables[0]
    values_path.parent.mkdir(parents=True)
    values_path.write_text("sentinel\n", encoding="utf-8")

    with mock.patch.object(roi_execution, "_write_extraction_summary_tables") as writer:
        with pytest.raises(RoiRuntimeOutputError, match="already exists"):
            run_roi_extraction(
                extraction_document,
                roi_set_document=roi_document,
                context=context,
            )

    writer.assert_not_called()
    assert values_path.read_text(encoding="utf-8") == "sentinel\n"
    assert not roi_execution._qc_summary_table_path(values_path).exists()
    assert not list(tmp_path.rglob(".roi-runtime-*"))


def test_extraction_rejects_symlinked_configured_output_root_before_writes(tmp_path: Path) -> None:
    context = _context(tmp_path)
    roi_document, extraction_document = _prepare_extraction(context)
    outside = tmp_path / "outside-extraction"
    outside.mkdir()
    linked_root = context.project_root / "linked-output"
    linked_root.symlink_to(outside, target_is_directory=True)
    extraction_document["extraction_set"]["outputs"] = {
        "root": "linked-output",
        "format": "tsv",
    }
    before = _tree_snapshot(outside)

    with mock.patch.object(roi_execution, "_write_extraction_summary_tables") as writer:
        with pytest.raises(RoiRuntimeOutputError, match="symbolic link"):
            run_roi_extraction(
                extraction_document,
                roi_set_document=roi_document,
                context=context,
            )

    writer.assert_not_called()
    assert _tree_snapshot(outside) == before
    assert not list(tmp_path.rglob(".roi-runtime-*"))


def test_configured_extraction_replacement_is_deterministic(tmp_path: Path) -> None:
    context = _context(tmp_path)
    roi_document, extraction_document = _prepare_extraction(context, existing_output="replace")

    first = run_roi_extraction(
        extraction_document,
        roi_set_document=roi_document,
        context=context,
    )
    paths = tuple(first.tables)
    first_hashes = _hashes(paths)
    second = run_roi_extraction(
        extraction_document,
        roi_set_document=roi_document,
        context=context,
    )

    assert second.executed
    assert _hashes(paths) == first_hashes
    assert not list(tmp_path.rglob(".roi-runtime-*"))


def test_extraction_writer_failure_leaves_no_partial_tables(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context(tmp_path)
    roi_document, extraction_document = _prepare_extraction(context)
    plan = plan_roi_extraction(
        extraction_document,
        roi_set_document=roi_document,
        context=context,
    )
    original = roi_execution._write_extraction_summary_tables

    def fail_after_staging(*args: object, **kwargs: object) -> object:
        original(*args, **kwargs)
        raise RuntimeError("injected table failure")

    monkeypatch.setattr(roi_execution, "_write_extraction_summary_tables", fail_after_staging)
    with pytest.raises(RuntimeError, match="injected table failure"):
        run_roi_extraction(
            extraction_document,
            roi_set_document=roi_document,
            context=context,
        )

    assert not any(path.exists() for path in _extraction_output_paths(plan))
    assert not list(tmp_path.rglob(".roi-runtime-*"))


def test_extraction_promotion_failure_restores_sentinels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context(tmp_path)
    roi_document, extraction_document = _prepare_extraction(context, existing_output="replace")
    plan = plan_roi_extraction(
        extraction_document,
        roi_set_document=roi_document,
        context=context,
    )
    sentinels = _seed_sentinels(_extraction_output_paths(plan))
    _inject_second_candidate_promotion_failure(monkeypatch)

    with pytest.raises(RoiRuntimeOutputError, match="prior destination set was restored"):
        run_roi_extraction(
            extraction_document,
            roi_set_document=roi_document,
            context=context,
        )

    assert _read_paths(sentinels) == sentinels
    assert not list(tmp_path.rglob(".roi-runtime-*"))


def test_duplicate_extraction_destinations_are_rejected(tmp_path: Path) -> None:
    context = _context(tmp_path)
    roi_document, extraction_document = _prepare_extraction(context)
    duplicate = deepcopy(extraction_document["extraction_set"]["targets"][0])
    duplicate["name"] = "SecondTarget"
    extraction_document["extraction_set"]["targets"].append(duplicate)

    report = preflight_roi_extraction(
        extraction_document,
        roi_set_document=roi_document,
        context=context,
    )

    assert not report.ready_for_execution
    assert any("same summary-table destination" in message for message in report.errors)
    assert not list(tmp_path.rglob(".roi-runtime-*"))


def test_runtime_output_preflight_rejects_duplicates_and_symlink_parents(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.tsv"
    with pytest.raises(RoiRuntimeOutputError, match="duplicate destinations"):
        preflight_runtime_outputs(
            (
                RoiRuntimeOutput(duplicate, "values"),
                RoiRuntimeOutput(duplicate, "QC"),
            ),
            existing_output="fail",
        )

    real_parent = tmp_path / "real"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(RoiRuntimeOutputError, match="symbolic link"):
        preflight_runtime_outputs(
            (RoiRuntimeOutput(linked_parent / "unsafe.tsv", "values"),),
            existing_output="fail",
        )


def test_runtime_output_preflight_rejects_lexical_alias_duplicates(tmp_path: Path) -> None:
    alias_parent = tmp_path / "alias-parent"
    alias_parent.mkdir()
    canonical = tmp_path / "same.tsv"
    lexical_alias = alias_parent / ".." / canonical.name

    with pytest.raises(RoiRuntimeOutputError, match="duplicate destinations"):
        preflight_runtime_outputs(
            (
                RoiRuntimeOutput(canonical, "values"),
                RoiRuntimeOutput(lexical_alias, "QC"),
            ),
            existing_output="fail",
        )


def test_runtime_output_cleanup_removes_dangling_sibling_candidate(tmp_path: Path) -> None:
    destination = tmp_path / "featquery-output"
    candidate: Path | None = None

    with pytest.raises(RuntimeError, match="injected failure"):
        with runtime_outputs.RoiRuntimeOutputTransaction(
            (RoiRuntimeOutput(destination, "featquery output", kind="directory"),),
            existing_output="fail",
        ) as transaction:
            candidate = transaction.sibling_candidate_path(destination)
            candidate.symlink_to(tmp_path / "missing-target", target_is_directory=True)
            raise RuntimeError("injected failure")

    assert candidate is not None
    assert not candidate.exists()
    assert not candidate.is_symlink()
    assert not list(tmp_path.rglob(".roi-runtime-*"))


def test_missing_optional_candidate_preserves_existing_output(tmp_path: Path) -> None:
    optional_directory = tmp_path / "optional-work"
    optional_directory.mkdir()
    sentinel = optional_directory / "sentinel.txt"
    sentinel.write_text("keep\n", encoding="utf-8")
    required_output = tmp_path / "required.tsv"

    with runtime_outputs.RoiRuntimeOutputTransaction(
        (
            RoiRuntimeOutput(optional_directory, "optional work", kind="directory", required=False),
            RoiRuntimeOutput(required_output, "required table"),
        ),
        existing_output="replace",
    ) as transaction:
        transaction.candidate_path(required_output).write_text("value\n", encoding="utf-8")
        transaction.promote()

    assert sentinel.read_text(encoding="utf-8") == "keep\n"
    assert required_output.read_text(encoding="utf-8") == "value\n"
    assert not list(tmp_path.rglob(".roi-runtime-*"))


def test_omitted_optional_output_does_not_shift_rollback_backup_slots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    optional_directory = tmp_path / "0-optional-work"
    guarded = tmp_path / "1-guarded" / "value.tsv"
    staged = tmp_path / "2-staged" / "value.tsv"
    sentinels = _seed_sentinels((guarded, staged))
    real_replace = os.replace

    def fail_candidate_promotion(source: object, destination: object) -> None:
        if "candidate" in Path(source).parts:
            raise OSError("injected promotion failure")
        real_replace(source, destination)

    monkeypatch.setattr(runtime_outputs.os, "replace", fail_candidate_promotion)
    with pytest.raises(RoiRuntimeOutputError, match="prior destination set was restored"):
        with runtime_outputs.RoiRuntimeOutputTransaction(
            (
                RoiRuntimeOutput(optional_directory, "optional work", kind="directory", required=False),
                RoiRuntimeOutput(guarded, "guarded tool cache"),
                RoiRuntimeOutput(staged, "staged table"),
            ),
            existing_output="replace",
        ) as transaction:
            transaction.prepare_direct_output(guarded, preserve_existing_for_read=True)
            guarded.write_text("changed\n", encoding="utf-8")
            transaction.candidate_path(staged).write_text("replacement\n", encoding="utf-8")
            transaction.promote()

    assert not optional_directory.exists()
    assert _read_paths(sentinels) == sentinels
    assert not list(tmp_path.rglob(".roi-runtime-*"))


def test_guarded_output_and_staged_files_roll_back_together(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    guarded = tmp_path / "cache.nii.gz"
    first = tmp_path / "first.tsv"
    second = tmp_path / "second.tsv"
    sentinels = _seed_sentinels((guarded, first, second))
    _inject_second_candidate_promotion_failure(monkeypatch)

    with pytest.raises(RoiRuntimeOutputError, match="prior destination set was restored"):
        with runtime_outputs.RoiRuntimeOutputTransaction(
            (
                RoiRuntimeOutput(guarded, "tool cache"),
                RoiRuntimeOutput(first, "first table"),
                RoiRuntimeOutput(second, "second table"),
            ),
            existing_output="replace",
        ) as transaction:
            transaction.prepare_direct_output(guarded, preserve_existing_for_read=True)
            guarded.write_text("changed\n", encoding="utf-8")
            transaction.candidate_path(first).write_text("new first\n", encoding="utf-8")
            transaction.candidate_path(second).write_text("new second\n", encoding="utf-8")
            transaction.promote()

    assert _read_paths(sentinels) == sentinels
    assert not list(tmp_path.rglob(".roi-runtime-*"))


def test_generic_extraction_rejects_4d_values_and_unknown_metrics(tmp_path: Path) -> None:
    context = _context(tmp_path)
    roi_document, extraction_document = _prepare_extraction(context, value_shape=(9, 9, 9, 2))

    report = preflight_roi_extraction(
        extraction_document,
        roi_set_document=roi_document,
        context=context,
    )

    assert not report.ready_for_execution
    assert any("must be a 3D image" in message for message in report.errors)
    extraction_document["extraction_set"]["targets"][0]["metrics"] = ["unsupported_metric"]
    invalid_metrics = preflight_roi_extraction(
        extraction_document,
        roi_set_document=roi_document,
        context=context,
    )
    assert not invalid_metrics.ready_for_execution
    assert any("Unsupported generic_nifti extraction metric" in message for message in invalid_metrics.errors)


def test_configured_tool_lookup_uses_declared_path(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, str | None]] = []

    def finder(tool: str, path: str | None = None) -> str | None:
        calls.append((tool, path))
        return "/opt/example/bin/featquery" if path == "/opt/example/bin" else None

    monkeypatch.setattr(roi_execution.shutil, "which", finder)

    assert roi_execution._find_configured_tool(finder, "featquery", "/opt/example/bin") == "/opt/example/bin/featquery"
    assert calls == [("featquery", "/opt/example/bin")]


def test_single_subject_scaffold_override_reaches_build_and_extraction_plans(tmp_path: Path) -> None:
    context = _context(tmp_path)
    roi_document = build_roi_set_document(
        "example_rois",
        "coordinate_sphere",
        overrides={"subjects": "sub-101"},
    )
    extraction_document = build_extraction_set_document(
        "example_values",
        roi_set="example_rois",
        template="generic_nifti",
        overrides={"subjects": "sub-101"},
    )

    build_plan = plan_roi_build(roi_document, context=context)
    extraction_plan = plan_roi_extraction(
        extraction_document,
        roi_set_document=roi_document,
        context=context,
    )

    assert roi_document["roi_set"]["subject"] == "sub-101"
    assert "subjects" not in roi_document["roi_set"]
    assert extraction_document["extraction_set"]["subject"] == "sub-101"
    assert "subjects" not in extraction_document["extraction_set"]
    for action in build_plan.actions:
        assert action.metadata["entities"]["subject_id"] == "101"
        assert action.metadata["entities"]["subject_dir"] == "sub-101"
        assert "/sub-101/" in action.mask_path.as_posix()
        assert "/sub-101/" in action.sidecar_path.as_posix()
    for action in extraction_plan.actions:
        assert action.metadata["subject_id"] == "101"
        assert "/sub-101/" in action.mask_path.as_posix()
        assert "/sub-101/" in action.table_path.as_posix()
    assert "sub-001" not in str(build_plan.to_dict())
    assert "sub-001" not in str(extraction_plan.to_dict())


def _context(root: Path) -> RoiExecutionContext:
    project_root = root / "project-demo"
    artifacts_root = root / "artifacts"
    project_root.mkdir()
    artifacts_root.mkdir()
    return RoiExecutionContext(
        workspace_root=root,
        project_root=project_root,
        artifacts_root=artifacts_root,
        project_name="project-demo",
    )


def _roi_document(*, existing_output: str = "fail", roi_count: int = 1) -> dict[str, object]:
    document = build_roi_set_document("example_rois", "coordinate_sphere")
    roi_set = document["roi_set"]
    roi_set["runtime"] = {"existing_output": existing_output}
    roi_set["rois"] = roi_set["rois"][:roi_count]
    for index, roi in enumerate(roi_set["rois"]):
        roi["coordinate"] = [2 + index * 2, 2, 2]
        roi["radius_mm"] = 1
        roi["sphere_radius_mm"] = 1
    return document


def _extraction_document(*, existing_output: str = "fail") -> dict[str, object]:
    document = build_extraction_set_document(
        "example_values",
        roi_set="example_rois",
        template="generic_nifti",
    )
    extraction_set = document["extraction_set"]
    extraction_set["runtime"] = {"existing_output": existing_output}
    extraction_set["targets"][0]["roi_labels"] = ["SeedA"]
    return document


def _prepare_extraction(
    context: RoiExecutionContext,
    *,
    existing_output: str = "fail",
    value_shape: tuple[int, ...] = (9, 9, 9),
) -> tuple[dict[str, object], dict[str, object]]:
    roi_document = _roi_document()
    extraction_document = _extraction_document(existing_output=existing_output)
    reference = _write_reference(context)
    mask = np.zeros(reference.shape, dtype=np.uint8)
    mask[2:4, 2:4, 2:4] = 1
    for action in plan_roi_build(roi_document, context=context).actions:
        _write_image(action.mask_path, mask)
    values = np.arange(np.prod(value_shape), dtype=float).reshape(value_shape)
    _write_image(context.project_root / "inputs/roi/example_value_map.nii.gz", values)
    return roi_document, extraction_document


def _write_reference(context: RoiExecutionContext) -> object:
    data = np.zeros((9, 9, 9), dtype=float)
    return _write_image(context.project_root / "inputs/roi/example_reference.nii.gz", data)


def _write_image(path: Path, data: object) -> object:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = nib.Nifti1Image(np.asarray(data), np.eye(4))
    nib.save(image, path)
    return image


def _build_output_paths(plan: object) -> tuple[Path, ...]:
    return tuple(
        path
        for action in plan.actions
        for path in (action.mask_path, action.sidecar_path)
    )


def _extraction_output_paths(plan: object) -> tuple[Path, ...]:
    return tuple(
        path
        for table in plan.tables
        for path in (table, roi_execution._qc_summary_table_path(table))
    )


def _hashes(paths: tuple[Path, ...]) -> tuple[str, ...]:
    return tuple(sha256(path.read_bytes()).hexdigest() for path in paths)


def _seed_sentinels(paths: tuple[Path, ...]) -> dict[Path, bytes]:
    expected: dict[Path, bytes] = {}
    for index, path in enumerate(paths):
        path.parent.mkdir(parents=True, exist_ok=True)
        value = f"sentinel-{index}\n".encode()
        path.write_bytes(value)
        expected[path] = value
    return expected


def _read_paths(expected: dict[Path, bytes]) -> dict[Path, bytes]:
    return {path: path.read_bytes() for path in expected}


def _inject_second_candidate_promotion_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    real_replace = os.replace
    promotions = 0

    def fail_second(source: object, destination: object) -> None:
        nonlocal promotions
        if "candidate" in Path(source).parts:
            promotions += 1
            if promotions == 2:
                raise OSError("injected promotion failure")
        real_replace(source, destination)

    monkeypatch.setattr(runtime_outputs.os, "replace", fail_second)


def _tree_snapshot(root: Path) -> tuple[tuple[str, str], ...]:
    rows: list[tuple[str, str]] = []
    for path in sorted(root.rglob("*"), key=lambda item: str(item)):
        relative = str(path.relative_to(root))
        if path.is_symlink():
            rows.append((relative, f"symlink:{os.readlink(path)}"))
        elif path.is_file():
            rows.append((relative, sha256(path.read_bytes()).hexdigest()))
        else:
            rows.append((relative, "directory"))
    return tuple(rows)
