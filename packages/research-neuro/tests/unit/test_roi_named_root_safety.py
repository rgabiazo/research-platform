from __future__ import annotations

from pathlib import Path
import sys
from unittest import mock

import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PACKAGE_ROOT / "src"))

from research_platform.neuro.roi import (
    validate_extraction_set_document,
    validate_roi_set_document,
)
from research_platform.neuro import roi_execution, roi_loso
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
)
from research_platform.neuro.roi_scaffold import build_extraction_set_document, build_roi_set_document


@pytest.mark.parametrize(
    "unsafe_path",
    (
        "/srv/example/output",
        r"C:\Data\output",
        "D:/Data/output",
        r"\\cluster.example\example-share\output",
        r"\\?\C:\Data\output",
        "~/output",
        r"~\output",
        "../outside",
        r"..\outside",
    ),
)
def test_runtime_output_paths_must_be_lexically_relative(unsafe_path: str) -> None:
    roi_document = build_roi_set_document("example_rois", "coordinate_sphere")
    extraction_document = build_extraction_set_document(
        "example_values",
        roi_set="example_rois",
        template="generic_nifti",
    )
    roi_document["roi_set"]["outputs"] = {  # type: ignore[index]
        "root_ref": "artifacts_root",
        "path": unsafe_path,
    }
    extraction_document["extraction_set"]["outputs"] = {  # type: ignore[index]
        "root_ref": "artifacts_root",
        "path": unsafe_path,
    }

    assert (
        "roi_set.outputs.path must be a relative path that remains beneath its configured root."
        in validate_roi_set_document(roi_document)
    )
    assert (
        "extraction_set.outputs.path must be a relative path that remains beneath its configured root."
        in validate_extraction_set_document(extraction_document)
    )


def test_nested_and_phase_specific_runtime_output_paths_are_validated() -> None:
    roi_document = build_roi_set_document("example_rois", "coordinate_sphere")
    extraction_document = build_extraction_set_document(
        "example_values",
        roi_set="example_rois",
        template="generic_nifti",
    )
    roi_set = roi_document["roi_set"]
    extraction_set = extraction_document["extraction_set"]
    roi_set["outputs"] = {  # type: ignore[index]
        "root": {"root_ref": "artifacts_root", "path": "../outside"}
    }
    roi_set["rois"][0]["outputs"] = {"path": "../outside"}  # type: ignore[index]
    extraction_set["targets"][0]["outputs"] = {"path": "../outside"}  # type: ignore[index]

    roi_errors = validate_roi_set_document(roi_document)
    extraction_errors = validate_extraction_set_document(extraction_document)

    assert (
        "roi_set.outputs.root.path must be a relative path that remains beneath its configured root."
        in roi_errors
    )
    assert (
        "roi_set.rois[0].outputs.path must be a relative path that remains beneath its configured root."
        in roi_errors
    )
    assert (
        "extraction_set.targets[0].outputs.path must be a relative path that remains beneath its configured root."
        in extraction_errors
    )


@pytest.mark.parametrize(
    "unsafe_path",
    (
        "/srv/example/input.nii.gz",
        r"C:\Data\input.nii.gz",
        r"\\cluster.example\example-share\input.nii.gz",
        "~/input.nii.gz",
        "../outside/input.nii.gz",
    ),
)
def test_named_root_input_paths_are_rejected_during_schema_validation(
    unsafe_path: str,
) -> None:
    roi_document = build_roi_set_document("example_rois", "coordinate_sphere")
    extraction_document = build_extraction_set_document(
        "example_values",
        roi_set="example_rois",
        template="generic_nifti",
    )
    roi_document["roi_set"]["rois"][0]["reference_image"] = {  # type: ignore[index]
        "root_ref": "project_root",
        "path": unsafe_path,
    }
    extraction_document["extraction_set"]["targets"][0]["inputs"] = {  # type: ignore[index]
        "root_ref": "project_root",
        "path": unsafe_path,
    }

    assert (
        "roi_set.rois[0].reference_image.path must be a relative path that remains beneath its configured root."
        in validate_roi_set_document(roi_document)
    )
    assert (
        "extraction_set.targets[0].inputs.path must be a relative path that remains beneath its configured root."
        in validate_extraction_set_document(extraction_document)
    )


@pytest.mark.parametrize("unsafe_path", ("../outside/input.nii.gz", r"..\outside\input.nii.gz"))
def test_implicit_project_relative_inputs_cannot_escape_project_root(
    unsafe_path: str,
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)
    roi_document = build_roi_set_document("example_rois", "coordinate_sphere")
    extraction_document = build_extraction_set_document(
        "example_values",
        roi_set="example_rois",
        template="generic_nifti",
    )
    roi_document["roi_set"]["rois"][0]["reference_image"] = unsafe_path  # type: ignore[index]
    extraction_document["extraction_set"]["targets"][0]["inputs"] = unsafe_path  # type: ignore[index]

    roi_errors = validate_roi_set_document(roi_document)
    extraction_errors = validate_extraction_set_document(extraction_document)

    assert any("reference_image" in error and "safe project-relative" in error for error in roi_errors)
    assert any("inputs" in error and "safe project-relative" in error for error in extraction_errors)
    with pytest.raises(ValueError, match="safe project-relative"):
        plan_roi_build(roi_document, context=context, validate_personal_paths=False)
    with pytest.raises(ValueError, match="safe project-relative"):
        plan_roi_extraction(
            extraction_document,
            roi_set_document=build_roi_set_document("example_rois", "coordinate_sphere"),
            context=context,
            validate_personal_paths=False,
        )
    with pytest.raises(ValueError, match="implicit project root"):
        roi_execution._resolve_project_relative_path(context, unsafe_path)


@pytest.mark.parametrize("malformed_outputs", ("../outside", [], 7))
def test_declared_runtime_outputs_must_be_mappings(malformed_outputs: object) -> None:
    roi_document = build_roi_set_document("example_rois", "coordinate_sphere")
    extraction_document = build_extraction_set_document(
        "example_values",
        roi_set="example_rois",
        template="generic_nifti",
    )
    roi_document["roi_set"]["outputs"] = malformed_outputs  # type: ignore[index]
    extraction_document["extraction_set"]["outputs"] = malformed_outputs  # type: ignore[index]

    assert (
        "roi_set.outputs must contain a mapping when declared."
        in validate_roi_set_document(roi_document)
    )
    assert (
        "extraction_set.outputs must contain a mapping when declared."
        in validate_extraction_set_document(extraction_document)
    )


@pytest.mark.parametrize("root_key", ("root_ref", "output_root_ref", "feat_root_ref"))
def test_declared_named_root_references_must_be_nonempty_strings(root_key: str) -> None:
    roi_document = build_roi_set_document("example_rois", "coordinate_sphere")
    extraction_document = build_extraction_set_document(
        "example_values",
        roi_set="example_rois",
        template="generic_nifti",
    )
    roi_document["roi_set"]["rois"][0]["reference_image"] = {  # type: ignore[index]
        root_key: "",
        "path": "inputs/roi/example_reference.nii.gz",
    }
    extraction_document["extraction_set"]["targets"][0]["inputs"] = {  # type: ignore[index]
        root_key: 7,
        "path": "inputs/roi/example_value_map.nii.gz",
    }

    assert any(
        f"reference_image.{root_key} must be a non-empty string" in error
        for error in validate_roi_set_document(roi_document)
    )
    assert any(
        f"inputs.{root_key} must be a non-empty string" in error
        for error in validate_extraction_set_document(extraction_document)
    )


def test_optional_fsl_path_profile_removes_shadowing_named_roots() -> None:
    roi_document = build_roi_set_document(
        "example_loso",
        "loso_group_map",
        path_profile="research_platform_fsl_ffx",
    )
    extraction_document = build_extraction_set_document(
        "example_featquery",
        roi_set="example_loso",
        template="fsl_featquery",
        path_profile="research_platform_fsl_ffx",
    )
    roi_set = roi_document["roi_set"]
    fixed_effects = roi_set["fixed_effects_inputs"]  # type: ignore[index]
    group_mask = roi_set["group_mask"]  # type: ignore[index]
    inputs = extraction_document["extraction_set"]["targets"][0]["inputs"]  # type: ignore[index]

    assert fixed_effects["root"] == "${ROI_FEAT_ROOT:-}"
    assert "root_ref" not in fixed_effects
    assert group_mask["path"].startswith("${ROI_FEAT_ROOT:-}/")
    assert "root_ref" not in group_mask
    assert inputs["feat_dir"].startswith("${ROI_FEAT_ROOT:-}/")
    assert "root_ref" not in inputs
    assert validate_roi_set_document(roi_document) == []
    assert validate_extraction_set_document(extraction_document) == []


def test_named_root_fsl_input_patterns_are_validated_structurally() -> None:
    roi_document = build_roi_set_document("example_loso", "loso_group_map")
    extraction_document = build_extraction_set_document(
        "example_featquery",
        roi_set="example_loso",
        template="fsl_featquery",
    )
    roi_document["roi_set"]["fixed_effects_inputs"]["cope_dir"] = "../outside/cope.feat"  # type: ignore[index]
    extraction_document["extraction_set"]["targets"][0]["inputs"]["feat_dir"] = "../outside/cope.feat"  # type: ignore[index]

    assert (
        "roi_set.fixed_effects_inputs.cope_dir must be a relative path that remains beneath its configured root."
        in validate_roi_set_document(roi_document)
    )
    assert (
        "extraction_set.targets[0].inputs.feat_dir must be a relative path that remains beneath its configured root."
        in validate_extraction_set_document(extraction_document)
    )


def test_safe_named_root_path_and_legacy_root_forms_remain_valid() -> None:
    named_root_document = build_roi_set_document("example_rois", "coordinate_sphere")
    named_root_document["roi_set"]["outputs"] = {  # type: ignore[index]
        "root_ref": "artifacts_root",
        "path": "roi-runtime/example_rois",
    }
    env_root_document = build_roi_set_document("example_rois", "coordinate_sphere")
    env_root_document["roi_set"]["outputs"] = {  # type: ignore[index]
        "root": "${ROI_DERIV_ROOT:-artifacts/roi/example-rois}"
    }
    literal_root_document = build_roi_set_document("example_rois", "coordinate_sphere")
    literal_root_document["roi_set"]["outputs"] = {"root": "artifacts/roi/example-rois"}  # type: ignore[index]

    assert validate_roi_set_document(named_root_document) == []
    assert validate_roi_set_document(env_root_document) == []
    assert validate_roi_set_document(literal_root_document) == []


@pytest.mark.parametrize("container_key", ("derivative_root", "output_root"))
@pytest.mark.parametrize("unsafe_path", ("../outside", r"..\outside"))
def test_top_level_legacy_output_roots_cannot_traverse(
    container_key: str,
    unsafe_path: str,
) -> None:
    roi_document = build_roi_set_document("example_rois", "coordinate_sphere")
    extraction_document = build_extraction_set_document(
        "example_values",
        roi_set="example_rois",
        template="generic_nifti",
    )
    roi_set = roi_document["roi_set"]
    extraction_set = extraction_document["extraction_set"]
    roi_set.pop("outputs")  # type: ignore[union-attr]
    extraction_set.pop("outputs")  # type: ignore[union-attr]
    roi_set[container_key] = unsafe_path  # type: ignore[index]
    extraction_set[container_key] = unsafe_path  # type: ignore[index]

    assert any(
        f"roi_set.{container_key}" in error and "safe project-relative" in error
        for error in validate_roi_set_document(roi_document)
    )
    assert any(
        f"extraction_set.{container_key}" in error and "safe project-relative" in error
        for error in validate_extraction_set_document(extraction_document)
    )


@pytest.mark.parametrize("container_key", ("root", "derivative_root", "output_root"))
@pytest.mark.parametrize(
    "unsafe_spec",
    (
        "../outside",
        r"..\outside",
        {"path": "../outside"},
        {"pattern": r"..\outside"},
    ),
)
def test_outputs_mapping_legacy_roots_cannot_traverse(
    container_key: str,
    unsafe_spec: object,
) -> None:
    roi_document = build_roi_set_document("example_rois", "coordinate_sphere")
    extraction_document = build_extraction_set_document(
        "example_values",
        roi_set="example_rois",
        template="generic_nifti",
    )
    roi_document["roi_set"]["outputs"] = {container_key: unsafe_spec}  # type: ignore[index]
    extraction_document["extraction_set"]["outputs"] = {container_key: unsafe_spec}  # type: ignore[index]

    assert any(
        f"roi_set.outputs.{container_key}" in error and "safe project-relative" in error
        for error in validate_roi_set_document(roi_document)
    )
    assert any(
        f"extraction_set.outputs.{container_key}" in error
        and "safe project-relative" in error
        for error in validate_extraction_set_document(extraction_document)
    )


def test_phase_specific_legacy_output_roots_cannot_traverse() -> None:
    roi_document = build_roi_set_document("example_rois", "coordinate_sphere")
    extraction_document = build_extraction_set_document(
        "example_values",
        roi_set="example_rois",
        template="generic_nifti",
    )
    roi_document["roi_set"]["rois"][0]["output_root"] = "../outside"  # type: ignore[index]
    extraction_document["extraction_set"]["targets"][0]["outputs"] = {  # type: ignore[index]
        "derivative_root": {"path": "../outside"}
    }

    assert any(
        "roi_set.rois[0].output_root" in error and "safe project-relative" in error
        for error in validate_roi_set_document(roi_document)
    )
    assert any(
        "extraction_set.targets[0].outputs.derivative_root.path" in error
        and "safe project-relative" in error
        for error in validate_extraction_set_document(extraction_document)
    )


def test_legacy_runtime_root_expansion_uses_raw_portable_document(tmp_path: Path) -> None:
    raw_roi_document = build_roi_set_document("example_rois", "coordinate_sphere")
    runtime_roi_document = build_roi_set_document("example_rois", "coordinate_sphere")
    raw_extraction_document = build_extraction_set_document(
        "example_values",
        roi_set="example_rois",
        template="generic_nifti",
    )
    runtime_extraction_document = build_extraction_set_document(
        "example_values",
        roi_set="example_rois",
        template="generic_nifti",
    )
    runtime_root = tmp_path / "runtime-output"
    raw_roi_document["roi_set"]["outputs"] = {  # type: ignore[index]
        "root": "${ROI_RUNTIME_ROOT:-artifacts/roi/example-rois}"
    }
    runtime_roi_document["roi_set"]["outputs"] = {"root": str(runtime_root)}  # type: ignore[index]
    raw_extraction_document["extraction_set"].pop("outputs")  # type: ignore[union-attr,index]
    runtime_extraction_document["extraction_set"].pop("outputs")  # type: ignore[union-attr,index]
    raw_extraction_document["extraction_set"]["output_root"] = {  # type: ignore[index]
        "path": "${ROI_RUNTIME_ROOT:-artifacts/roi/example-values}"
    }
    runtime_extraction_document["extraction_set"]["output_root"] = {  # type: ignore[index]
        "path": str(runtime_root)
    }

    assert validate_roi_set_document(
        runtime_roi_document,
        personal_path_document=raw_roi_document,
    ) == []
    assert validate_extraction_set_document(
        runtime_extraction_document,
        personal_path_document=raw_extraction_document,
    ) == []
    context = _context(tmp_path)
    roi_plan = plan_roi_build(
        runtime_roi_document,
        context=context,
        validate_personal_paths=False,
    )
    extraction_plan = plan_roi_extraction(
        runtime_extraction_document,
        roi_set_document=runtime_roi_document,
        context=context,
        validate_personal_paths=False,
    )

    assert all(
        action.mask_path.is_relative_to(runtime_root)
        and action.sidecar_path.is_relative_to(runtime_root)
        for action in roi_plan.actions
    )
    assert all(action.table_path.is_relative_to(runtime_root) for action in extraction_plan.actions)


def test_blank_runtime_root_and_subpath_specs_are_rejected(tmp_path: Path) -> None:
    context = _context(tmp_path)
    roi_documents: list[dict[str, object]] = []
    extraction_documents: list[dict[str, object]] = []
    for outputs in (
        {"root": ""},
        {"root": {}},
        {"root": {"path": ""}},
        {"root_ref": "artifacts_root", "path": ""},
    ):
        document = build_roi_set_document("example_rois", "coordinate_sphere")
        document["roi_set"]["outputs"] = outputs  # type: ignore[index]
        roi_documents.append(document)
        extraction_document = build_extraction_set_document(
            "example_values",
            roi_set="example_rois",
            template="generic_nifti",
        )
        extraction_document["extraction_set"]["outputs"] = outputs  # type: ignore[index]
        extraction_documents.append(extraction_document)
    top_level_document = build_roi_set_document("example_rois", "coordinate_sphere")
    top_level_document["roi_set"]["output_root"] = ""  # type: ignore[index]
    roi_documents.append(top_level_document)
    top_level_extraction_document = build_extraction_set_document(
        "example_values",
        roi_set="example_rois",
        template="generic_nifti",
    )
    top_level_extraction_document["extraction_set"]["output_root"] = ""  # type: ignore[index]
    extraction_documents.append(top_level_extraction_document)

    for document in roi_documents:
        assert any(
            "non-empty" in error or "define path" in error
            for error in validate_roi_set_document(document)
        )
        with pytest.raises(ValueError, match="non-empty|define path"):
            plan_roi_build(document, context=context)
    for document in extraction_documents:
        assert any(
            "non-empty" in error or "define path" in error
            for error in validate_extraction_set_document(document)
        )
        with pytest.raises(ValueError, match="non-empty|define path"):
            plan_roi_extraction(
                document,
                roi_set_document=build_roi_set_document("example_rois", "coordinate_sphere"),
                context=context,
            )


def test_parent_traversal_fails_before_runtime_planning(tmp_path: Path) -> None:
    context = _context(tmp_path)
    document = build_roi_set_document("example_rois", "coordinate_sphere")
    document["roi_set"]["outputs"] = {  # type: ignore[index]
        "root_ref": "artifacts_root",
        "path": "../outside/example_rois",
    }
    before = _tree_snapshot(tmp_path)

    with pytest.raises(ValueError, match="remains beneath its configured root"):
        plan_roi_build(document, context=context)
    report = preflight_roi_build(document, context=context)

    assert not report.ready_for_execution
    assert any(
        check.check_id == "configuration_valid" and check.status == "error"
        for check in report.checks
    )
    assert _tree_snapshot(tmp_path) == before


def test_named_root_symlink_ancestor_is_preserved_and_rejected(tmp_path: Path) -> None:
    context = _context(tmp_path)
    document = build_roi_set_document("example_rois", "coordinate_sphere")
    outside = tmp_path / "outside"
    outside.mkdir()
    linked = context.artifacts_root / "linked"
    linked.symlink_to(outside, target_is_directory=True)
    document["roi_set"]["outputs"] = {  # type: ignore[index]
        "root_ref": "artifacts_root",
        "path": "linked/example_rois",
    }
    before = _tree_snapshot(tmp_path)

    plan = plan_roi_build(document, context=context)
    report = preflight_roi_build(document, context=context)

    assert "linked" in plan.actions[0].mask_path.parts
    assert not report.ready_for_execution
    assert any(
        check.check_id == "configured_root_available"
        and check.status == "error"
        and "symbolic link" in check.message
        for check in report.checks
    )
    with mock.patch.object(roi_execution, "_execute_build_action") as executor:
        with pytest.raises(RoiRuntimeOutputError, match="not ready for execution"):
            roi_execution.run_roi_build(document, context=context)
    executor.assert_not_called()
    assert _tree_snapshot(tmp_path) == before
    assert not list(tmp_path.rglob(".roi-runtime-*"))


def test_loso_named_root_preserves_descendant_symlink_for_preflight(tmp_path: Path) -> None:
    project_root = tmp_path / "project-demo"
    artifacts_root = tmp_path / "artifacts"
    derivatives_root = tmp_path / "derivatives"
    outside = tmp_path / "outside"
    for directory in (project_root, artifacts_root, derivatives_root, outside):
        directory.mkdir()
    linked = derivatives_root / "linked"
    linked.symlink_to(outside, target_is_directory=True)
    context = RoiExecutionContext(
        workspace_root=tmp_path,
        project_root=project_root,
        artifacts_root=artifacts_root,
        project_name="project-demo",
        root_refs={"dataset_derivatives_root": derivatives_root},
    )
    document = build_roi_set_document("example_loso", "loso_group_map")
    document["roi_set"]["outputs"] = {  # type: ignore[index]
        "root_ref": "dataset_derivatives_root",
        "path": "linked/example_loso",
    }

    output_root = roi_loso._resolve_base_output_root(document["roi_set"], context=context)  # type: ignore[arg-type,index]

    assert "linked" in output_root.parts
    with pytest.raises(RoiRuntimeOutputError, match="symbolic link"):
        preflight_runtime_outputs(
            (RoiRuntimeOutput(output_root / "example.nii.gz", "LOSO example output"),),
            existing_output="fail",
        )


def test_named_root_non_directory_parent_has_stable_root_check(tmp_path: Path) -> None:
    context = _context(tmp_path)
    document = build_roi_set_document("example_rois", "coordinate_sphere")
    blocked = context.artifacts_root / "blocked"
    blocked.write_text("sentinel\n", encoding="utf-8")
    document["roi_set"]["outputs"] = {  # type: ignore[index]
        "root_ref": "artifacts_root",
        "path": "blocked/example_rois",
    }
    before = _tree_snapshot(tmp_path)

    report = preflight_roi_build(document, context=context)

    assert not report.ready_for_execution
    assert any(
        check.check_id == "configured_root_available"
        and check.status == "error"
        and "not a directory" in check.message
        for check in report.checks
    )
    assert blocked.read_text(encoding="utf-8") == "sentinel\n"
    assert _tree_snapshot(tmp_path) == before


def test_existing_destination_remains_an_output_collision(tmp_path: Path) -> None:
    context = _context(tmp_path)
    document = build_roi_set_document("example_rois", "coordinate_sphere")
    plan = plan_roi_build(document, context=context)
    destination = plan.actions[0].sidecar_path
    destination.parent.mkdir(parents=True)
    destination.write_text("sentinel\n", encoding="utf-8")
    before = _tree_snapshot(tmp_path)

    report = preflight_roi_build(document, context=context)

    assert not report.ready_for_execution
    assert any(
        check.check_id == "output_collision" and check.status == "error"
        for check in report.checks
    )
    assert destination.read_text(encoding="utf-8") == "sentinel\n"
    assert _tree_snapshot(tmp_path) == before


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


def _tree_snapshot(root: Path) -> tuple[tuple[str, str], ...]:
    rows: list[tuple[str, str]] = []
    for path in sorted(root.rglob("*"), key=lambda item: str(item)):
        relative = str(path.relative_to(root))
        if path.is_symlink():
            rows.append((relative, f"symlink:{path.readlink()}"))
        elif path.is_file():
            rows.append((relative, path.read_text(encoding="utf-8")))
        else:
            rows.append((relative, "directory"))
    return tuple(rows)
