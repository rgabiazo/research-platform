"""BIDS-like derivative path helpers for ROI workflow artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable
import re


_BIDS_LABEL_VALUE = re.compile(r"^[A-Za-z0-9]+$")
_PATH_SEGMENT = re.compile(r"^[A-Za-z0-9_.-]+$")


def build_roi_mask_filename(
    *,
    subject_id: str,
    task_id: str,
    space: str,
    roi_label: str,
    method_desc: str,
    session_id: str | None = None,
    direction: str | None = None,
    resolution: str | None = None,
) -> str:
    """Return the default Phase 1 ROI mask filename.

    The default form is:
    sub-<id>[_ses-<id>]_task-<task>[_dir-<dir>]_space-<space>[_res-<res>]
    _label-<roiLabel>_desc-<methodContrast>_mask.nii.gz
    """

    entities = [
        ("sub", _normalize_subject(subject_id)),
        ("ses", _normalize_session(session_id)),
        ("task", _label_value(task_id, "task_id")),
        ("dir", _optional_label_value(direction, "direction")),
        ("space", _label_value(space, "space")),
        ("res", _optional_label_value(resolution, "resolution")),
        ("label", _label_value(roi_label, "roi_label")),
        ("desc", _label_value(method_desc, "method_desc")),
    ]
    return f"{_entity_stem(entities)}_mask.nii.gz"


def build_roi_mask_path(
    derivative_root: str | Path,
    *,
    subject_id: str,
    task_id: str,
    space: str,
    roi_label: str,
    method_desc: str,
    session_id: str | None = None,
    direction: str | None = None,
    resolution: str | None = None,
    pipeline_name: str | None = "roi",
    datatype: str = "func",
) -> Path:
    """Return a BIDS-like derivative path for an ROI mask."""

    subject_dir = _normalize_subject(subject_id)
    session_dir = _normalize_session(session_id)
    filename = build_roi_mask_filename(
        subject_id=subject_id,
        session_id=session_id,
        task_id=task_id,
        direction=direction,
        space=space,
        resolution=resolution,
        roi_label=roi_label,
        method_desc=method_desc,
    )
    return _derivative_base(derivative_root, pipeline_name) / subject_dir / _optional_dir(session_dir) / datatype / filename


def build_roi_sidecar_path(mask_path: str | Path) -> Path:
    """Return the JSON sidecar path for a mask path using the same stem."""

    path = Path(mask_path)
    if path.name.endswith(".nii.gz"):
        return path.with_name(f"{path.name[:-7]}.json")
    return path.with_suffix(".json")


def build_roi_sidecar_path_from_entities(
    derivative_root: str | Path,
    *,
    subject_id: str,
    task_id: str,
    space: str,
    roi_label: str,
    method_desc: str,
    session_id: str | None = None,
    direction: str | None = None,
    resolution: str | None = None,
    pipeline_name: str | None = "roi",
    datatype: str = "func",
) -> Path:
    """Return the JSON sidecar path matching the default ROI mask path."""

    return build_roi_sidecar_path(
        build_roi_mask_path(
            derivative_root,
            subject_id=subject_id,
            session_id=session_id,
            task_id=task_id,
            direction=direction,
            space=space,
            resolution=resolution,
            roi_label=roi_label,
            method_desc=method_desc,
            pipeline_name=pipeline_name,
            datatype=datatype,
        )
    )


def build_loso_group_map_filename(
    *,
    task_id: str,
    space: str,
    method_desc: str,
    roi_label: str | None = None,
    heldout_subject: str | None = None,
    session_id: str | None = None,
    direction: str | None = None,
    resolution: str | None = None,
    statistic: str | None = "z",
    suffix: str = "groupmap",
) -> str:
    """Return a BIDS-like LOSO/group-map filename.

    The optional ``heldout`` entity is a platform convention for LOSO maps and
    should be treated as BIDS-like derivative metadata, not a claim of full
    BIDS validator compliance.
    """

    entities = [
        ("ses", _normalize_session(session_id)),
        ("task", _label_value(task_id, "task_id")),
        ("dir", _optional_label_value(direction, "direction")),
        ("space", _label_value(space, "space")),
        ("res", _optional_label_value(resolution, "resolution")),
        ("label", _optional_label_value(roi_label, "roi_label")),
        ("desc", _label_value(method_desc, "method_desc")),
        ("stat", _optional_label_value(statistic, "statistic")),
        ("heldout", _normalize_heldout_subject(heldout_subject)),
    ]
    return f"{_entity_stem(entities)}_{_label_value(suffix, 'suffix')}.nii.gz"


def build_loso_group_map_path(
    derivative_root: str | Path,
    *,
    task_id: str,
    space: str,
    method_desc: str,
    roi_label: str | None = None,
    heldout_subject: str | None = None,
    session_id: str | None = None,
    direction: str | None = None,
    resolution: str | None = None,
    statistic: str | None = "z",
    suffix: str = "groupmap",
    pipeline_name: str | None = "roi",
    datatype: str = "func",
) -> Path:
    """Return a group-level BIDS-like path for LOSO or group maps."""

    session_dir = _normalize_session(session_id)
    filename = build_loso_group_map_filename(
        session_id=session_id,
        task_id=task_id,
        direction=direction,
        space=space,
        resolution=resolution,
        roi_label=roi_label,
        method_desc=method_desc,
        heldout_subject=heldout_subject,
        statistic=statistic,
        suffix=suffix,
    )
    return _derivative_base(derivative_root, pipeline_name) / "group" / _optional_dir(session_dir) / datatype / filename


def build_roi_extraction_table_filename(
    *,
    subject_id: str,
    task_id: str,
    extraction_desc: str,
    session_id: str | None = None,
    direction: str | None = None,
    space: str | None = None,
    resolution: str | None = None,
    suffix: str = "roiextract",
    extension: str = ".tsv",
) -> str:
    """Return a BIDS-like ROI extraction table filename."""

    normalized_extension = extension if extension.startswith(".") else f".{extension}"
    entities = [
        ("sub", _normalize_subject(subject_id)),
        ("ses", _normalize_session(session_id)),
        ("task", _label_value(task_id, "task_id")),
        ("dir", _optional_label_value(direction, "direction")),
        ("space", _optional_label_value(space, "space")),
        ("res", _optional_label_value(resolution, "resolution")),
        ("desc", _label_value(extraction_desc, "extraction_desc")),
    ]
    return f"{_entity_stem(entities)}_{_label_value(suffix, 'suffix')}{normalized_extension}"


def build_roi_extraction_table_path(
    derivative_root: str | Path,
    *,
    subject_id: str,
    task_id: str,
    extraction_desc: str,
    session_id: str | None = None,
    direction: str | None = None,
    space: str | None = None,
    resolution: str | None = None,
    pipeline_name: str | None = "roi",
    datatype: str = "func",
    extension: str = ".tsv",
) -> Path:
    """Return a BIDS-like derivative path for an ROI extraction table."""

    subject_dir = _normalize_subject(subject_id)
    session_dir = _normalize_session(session_id)
    filename = build_roi_extraction_table_filename(
        subject_id=subject_id,
        session_id=session_id,
        task_id=task_id,
        direction=direction,
        space=space,
        resolution=resolution,
        extraction_desc=extraction_desc,
        extension=extension,
    )
    return _derivative_base(derivative_root, pipeline_name) / subject_dir / _optional_dir(session_dir) / datatype / filename


def build_roi_group_extraction_table_filename(
    *,
    task_id: str,
    extraction_desc: str,
    session_id: str | None = None,
    suffix: str = "values",
    extension: str = ".tsv",
) -> str:
    """Return a BIDS-like group-level ROI extraction summary filename."""

    normalized_extension = extension if extension.startswith(".") else f".{extension}"
    entities = [
        ("ses", _normalize_session(session_id)),
        ("task", _label_value(task_id, "task_id")),
        ("desc", _label_value(extraction_desc, "extraction_desc")),
    ]
    return f"group_{_entity_stem(entities)}_{_label_value(suffix, 'suffix')}{normalized_extension}"


def build_roi_group_extraction_table_path(
    derivative_root: str | Path,
    *,
    task_id: str,
    extraction_desc: str,
    session_id: str | None = None,
    pipeline_name: str | None = "roi_extract",
    datatype: str | None = None,
    extension: str = ".tsv",
) -> Path:
    """Return a BIDS-like group-level ROI extraction summary table path."""

    session_dir = _normalize_session(session_id)
    filename = build_roi_group_extraction_table_filename(
        session_id=session_id,
        task_id=task_id,
        extraction_desc=extraction_desc,
        extension=extension,
    )
    base = _derivative_base(derivative_root, pipeline_name) / "group" / _optional_dir(session_dir)
    return base / datatype / filename if datatype else base / filename


def build_loso_flame1_statmap_filename(
    *,
    task_id: str,
    space: str,
    resolution: str,
    contrast_alias: str,
    heldout_subject: str,
    map_desc: str,
    session_id: str | None = None,
    direction: str | None = None,
    statistic: str = "z",
) -> str:
    """Return the published LOSO FLAME1 statistical-map filename."""

    entities = [
        ("ses", _normalize_session(session_id)),
        ("task", _label_value(task_id, "task_id")),
        ("dir", _optional_label_value(direction, "direction")),
        ("space", _label_value(space, "space")),
        ("res", _label_value(resolution, "resolution")),
        ("contrast", _label_value(contrast_alias, "contrast_alias")),
        ("stat", _label_value(statistic, "statistic")),
        ("heldout", _normalize_heldout_subject(heldout_subject)),
        ("desc", _label_value(map_desc, "map_desc")),
    ]
    return f"{_entity_stem(entities)}_statmap.nii.gz"


def build_loso_flame1_statmap_path(
    publication_root: str | Path,
    *,
    task_id: str,
    space: str,
    resolution: str,
    contrast_alias: str,
    heldout_subject: str,
    map_desc: str,
    session_id: str | None = None,
    direction: str | None = None,
    datatype: str = "func",
) -> Path:
    """Return the published LOSO FLAME1 statistical-map path."""

    session_dir = _normalize_session(session_id)
    filename = build_loso_flame1_statmap_filename(
        session_id=session_id,
        task_id=task_id,
        direction=direction,
        space=space,
        resolution=resolution,
        contrast_alias=contrast_alias,
        heldout_subject=heldout_subject,
        map_desc=map_desc,
    )
    return Path(publication_root) / "maps" / "group" / _optional_dir(session_dir) / datatype / filename


def build_loso_flame1_mask_filename(
    *,
    subject_id: str,
    task_id: str,
    space: str,
    resolution: str,
    roi_label: str,
    contrast_alias: str,
    mask_desc: str,
    session_id: str | None = None,
    direction: str | None = None,
) -> str:
    """Return the published LOSO FLAME1 ROI-mask filename."""

    entities = [
        ("sub", _normalize_subject(subject_id)),
        ("ses", _normalize_session(session_id)),
        ("task", _label_value(task_id, "task_id")),
        ("dir", _optional_label_value(direction, "direction")),
        ("space", _label_value(space, "space")),
        ("res", _label_value(resolution, "resolution")),
        ("label", _label_value(roi_label, "roi_label")),
        ("contrast", _label_value(contrast_alias, "contrast_alias")),
        ("desc", _label_value(mask_desc, "mask_desc")),
    ]
    return f"{_entity_stem(entities)}_mask.nii.gz"


def build_loso_flame1_mask_path(
    publication_root: str | Path,
    *,
    subject_id: str,
    task_id: str,
    space: str,
    resolution: str,
    roi_label: str,
    contrast_alias: str,
    mask_desc: str,
    session_id: str | None = None,
    direction: str | None = None,
    datatype: str = "func",
) -> Path:
    """Return the published LOSO FLAME1 ROI-mask path."""

    subject_dir = _normalize_subject(subject_id)
    session_dir = _normalize_session(session_id)
    filename = build_loso_flame1_mask_filename(
        subject_id=subject_id,
        session_id=session_id,
        task_id=task_id,
        direction=direction,
        space=space,
        resolution=resolution,
        roi_label=roi_label,
        contrast_alias=contrast_alias,
        mask_desc=mask_desc,
    )
    return Path(publication_root) / "masks" / subject_dir / _optional_dir(session_dir) / datatype / filename


def build_loso_flame1_roistats_filename(
    *,
    task_id: str,
    table_desc: str,
    session_id: str | None = None,
    direction: str | None = None,
    extension: str = ".tsv",
) -> str:
    """Return the published LOSO FLAME1 ROI-stats table filename."""

    normalized_extension = extension if extension.startswith(".") else f".{extension}"
    entities = [
        ("ses", _normalize_session(session_id)),
        ("task", _label_value(task_id, "task_id")),
        ("dir", _optional_label_value(direction, "direction")),
        ("desc", _label_value(table_desc, "table_desc")),
    ]
    return f"{_entity_stem(entities)}_roistats{normalized_extension}"


def build_loso_flame1_roistats_path(
    publication_root: str | Path,
    *,
    task_id: str,
    table_desc: str,
    session_id: str | None = None,
    direction: str | None = None,
    datatype: str = "func",
    extension: str = ".tsv",
) -> Path:
    """Return the published LOSO FLAME1 ROI-stats table path."""

    session_dir = _normalize_session(session_id)
    filename = build_loso_flame1_roistats_filename(
        session_id=session_id,
        task_id=task_id,
        direction=direction,
        table_desc=table_desc,
        extension=extension,
    )
    return Path(publication_root) / "tables" / "group" / _optional_dir(session_dir) / datatype / filename


def _derivative_base(derivative_root: str | Path, pipeline_name: str | None) -> Path:
    root = Path(derivative_root)
    pipeline = _optional_path_segment(pipeline_name, "pipeline_name")
    return root / pipeline if pipeline else root


def _entity_stem(entities: Iterable[tuple[str, str | None]]) -> str:
    parts: list[str] = []
    for key, value in entities:
        if value is None:
            continue
        if key in {"sub", "ses"}:
            parts.append(value)
        else:
            parts.append(f"{key}-{value}")
    if not parts:
        raise ValueError("At least one entity is required.")
    return "_".join(parts)


def _normalize_subject(value: str) -> str:
    text = _required_text(value, "subject_id")
    if text.startswith("sub-"):
        subject_value = text[4:]
    else:
        subject_value = text
    _ensure_label(subject_value, "subject_id")
    return f"sub-{subject_value}"


def _normalize_session(value: str | None) -> str | None:
    text = _optional_text(value)
    if text is None:
        return None
    if text.startswith("ses-"):
        session_value = text[4:]
    else:
        session_value = text
    _ensure_label(session_value, "session_id")
    return f"ses-{session_value}"


def _normalize_heldout_subject(value: str | None) -> str | None:
    text = _optional_text(value)
    if text is None:
        return None
    subject = _normalize_subject(text)
    return subject.replace("-", "")


def _optional_dir(value: str | None) -> Path:
    return Path(value) if value else Path()


def _label_value(value: str, label: str) -> str:
    text = _required_text(value, label)
    _ensure_label(text, label)
    return text


def _optional_label_value(value: str | None, label: str) -> str | None:
    text = _optional_text(value)
    if text is None:
        return None
    _ensure_label(text, label)
    return text


def _optional_path_segment(value: str | None, label: str) -> str | None:
    text = _optional_text(value)
    if text is None:
        return None
    if not _PATH_SEGMENT.fullmatch(text) or "/" in text or "\\" in text:
        raise ValueError(f"{label} must be a single safe path segment.")
    return text


def _ensure_label(value: str, label: str) -> None:
    if not _BIDS_LABEL_VALUE.fullmatch(value):
        raise ValueError(f"{label} must contain only letters and numbers.")


def _required_text(value: str | None, label: str) -> str:
    text = _optional_text(value)
    if text is None:
        raise ValueError(f"{label} must be defined.")
    return text


def _optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
