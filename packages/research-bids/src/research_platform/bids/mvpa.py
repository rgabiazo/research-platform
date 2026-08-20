"""BIDS-like derivative path helpers for MVPA workflow artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import re


MVPA_ENTITY_ORDER = ("sub", "ses", "task", "acq", "dir", "run", "space", "res", "model", "label", "desc")

_BIDS_LABEL_VALUE = re.compile(r"^[A-Za-z0-9]+$")
_PATH_SEGMENT = re.compile(r"^[A-Za-z0-9_.-]+$")
_EXTENSION = re.compile(r"^\.[A-Za-z0-9]+$")


def build_mvpa_pattern_table_filename(
    *,
    entities: Mapping[str, str | None] | None = None,
    subject_id: str | None = None,
    session_id: str | None = None,
    task_id: str | None = None,
    acquisition: str | None = None,
    direction: str | None = None,
    run: str | None = None,
    space: str | None = None,
    resolution: str | None = None,
    model: str | None = None,
    label: str | None = None,
    desc: str | None = None,
    extension: str = ".tsv",
) -> str:
    """Return a BIDS-like MVPA pattern-table filename."""

    return _build_mvpa_filename(
        suffix="patterns",
        extension=extension,
        entities=entities,
        subject_id=subject_id,
        session_id=session_id,
        task_id=task_id,
        acquisition=acquisition,
        direction=direction,
        run=run,
        space=space,
        resolution=resolution,
        model=model,
        label=label,
        desc=desc,
    )


def build_mvpa_pattern_table_path(
    *,
    runtime_root: str | Path | None = None,
    publication_root: str | Path | None = None,
    mvpa_set: str,
    derivative_name: str = "mvpa",
    entities: Mapping[str, str | None] | None = None,
    subject_id: str | None = None,
    session_id: str | None = None,
    task_id: str | None = None,
    acquisition: str | None = None,
    direction: str | None = None,
    run: str | None = None,
    space: str | None = None,
    resolution: str | None = None,
    model: str | None = None,
    label: str | None = None,
    desc: str | None = None,
    extension: str = ".tsv",
) -> Path:
    """Return a caller-rooted MVPA pattern-table path."""

    ordered = _ordered_entities(
        entities=entities,
        subject_id=subject_id,
        session_id=session_id,
        task_id=task_id,
        acquisition=acquisition,
        direction=direction,
        run=run,
        space=space,
        resolution=resolution,
        model=model,
        label=label,
        desc=desc,
    )
    filename = _compose_filename(ordered, suffix="patterns", extension=extension)
    return _build_mvpa_path(
        runtime_root=runtime_root,
        publication_root=publication_root,
        mvpa_set=mvpa_set,
        derivative_name=derivative_name,
        section="patterns",
        filename=filename,
        ordered_entities=ordered,
    )


def build_mvpa_distance_table_filename(
    *,
    entities: Mapping[str, str | None] | None = None,
    subject_id: str | None = None,
    session_id: str | None = None,
    task_id: str | None = None,
    acquisition: str | None = None,
    direction: str | None = None,
    run: str | None = None,
    space: str | None = None,
    resolution: str | None = None,
    model: str | None = None,
    label: str | None = None,
    desc: str | None = None,
    extension: str = ".tsv",
) -> str:
    """Return a BIDS-like MVPA distance-table filename."""

    return _build_mvpa_filename(
        suffix="distances",
        extension=extension,
        entities=entities,
        subject_id=subject_id,
        session_id=session_id,
        task_id=task_id,
        acquisition=acquisition,
        direction=direction,
        run=run,
        space=space,
        resolution=resolution,
        model=model,
        label=label,
        desc=desc,
    )


def build_mvpa_distance_table_path(
    *,
    runtime_root: str | Path | None = None,
    publication_root: str | Path | None = None,
    mvpa_set: str,
    derivative_name: str = "mvpa",
    entities: Mapping[str, str | None] | None = None,
    subject_id: str | None = None,
    session_id: str | None = None,
    task_id: str | None = None,
    acquisition: str | None = None,
    direction: str | None = None,
    run: str | None = None,
    space: str | None = None,
    resolution: str | None = None,
    model: str | None = None,
    label: str | None = None,
    desc: str | None = None,
    extension: str = ".tsv",
) -> Path:
    """Return a caller-rooted MVPA distance-table path."""

    ordered = _ordered_entities(
        entities=entities,
        subject_id=subject_id,
        session_id=session_id,
        task_id=task_id,
        acquisition=acquisition,
        direction=direction,
        run=run,
        space=space,
        resolution=resolution,
        model=model,
        label=label,
        desc=desc,
    )
    filename = _compose_filename(ordered, suffix="distances", extension=extension)
    return _build_mvpa_path(
        runtime_root=runtime_root,
        publication_root=publication_root,
        mvpa_set=mvpa_set,
        derivative_name=derivative_name,
        section="distances",
        filename=filename,
        ordered_entities=ordered,
    )


def build_mvpa_rdm_table_filename(
    *,
    entities: Mapping[str, str | None] | None = None,
    subject_id: str | None = None,
    session_id: str | None = None,
    task_id: str | None = None,
    acquisition: str | None = None,
    direction: str | None = None,
    run: str | None = None,
    space: str | None = None,
    resolution: str | None = None,
    model: str | None = None,
    label: str | None = None,
    desc: str | None = None,
    extension: str = ".tsv",
) -> str:
    """Return a BIDS-like MVPA RDM-table filename."""

    return _build_mvpa_filename(
        suffix="rdm",
        extension=extension,
        entities=entities,
        subject_id=subject_id,
        session_id=session_id,
        task_id=task_id,
        acquisition=acquisition,
        direction=direction,
        run=run,
        space=space,
        resolution=resolution,
        model=model,
        label=label,
        desc=desc,
    )


def build_mvpa_rdm_table_path(
    *,
    runtime_root: str | Path | None = None,
    publication_root: str | Path | None = None,
    mvpa_set: str,
    derivative_name: str = "mvpa",
    entities: Mapping[str, str | None] | None = None,
    subject_id: str | None = None,
    session_id: str | None = None,
    task_id: str | None = None,
    acquisition: str | None = None,
    direction: str | None = None,
    run: str | None = None,
    space: str | None = None,
    resolution: str | None = None,
    model: str | None = None,
    label: str | None = None,
    desc: str | None = None,
    extension: str = ".tsv",
) -> Path:
    """Return a caller-rooted MVPA RDM-table path."""

    ordered = _ordered_entities(
        entities=entities,
        subject_id=subject_id,
        session_id=session_id,
        task_id=task_id,
        acquisition=acquisition,
        direction=direction,
        run=run,
        space=space,
        resolution=resolution,
        model=model,
        label=label,
        desc=desc,
    )
    filename = _compose_filename(ordered, suffix="rdm", extension=extension)
    return _build_mvpa_path(
        runtime_root=runtime_root,
        publication_root=publication_root,
        mvpa_set=mvpa_set,
        derivative_name=derivative_name,
        section="rdms",
        filename=filename,
        ordered_entities=ordered,
    )


def build_mvpa_group_summary_table_filename(
    *,
    entities: Mapping[str, str | None] | None = None,
    session_id: str | None = None,
    task_id: str | None = None,
    acquisition: str | None = None,
    direction: str | None = None,
    run: str | None = None,
    space: str | None = None,
    resolution: str | None = None,
    model: str | None = None,
    label: str | None = None,
    desc: str | None = None,
    extension: str = ".tsv",
) -> str:
    """Return a BIDS-like MVPA group-summary table filename."""

    return _build_mvpa_filename(
        prefix="group",
        suffix="mvpasummary",
        extension=extension,
        entities=entities,
        session_id=session_id,
        task_id=task_id,
        acquisition=acquisition,
        direction=direction,
        run=run,
        space=space,
        resolution=resolution,
        model=model,
        label=label,
        desc=desc,
    )


def build_mvpa_group_summary_table_path(
    *,
    runtime_root: str | Path | None = None,
    publication_root: str | Path | None = None,
    mvpa_set: str,
    derivative_name: str = "mvpa",
    entities: Mapping[str, str | None] | None = None,
    session_id: str | None = None,
    task_id: str | None = None,
    acquisition: str | None = None,
    direction: str | None = None,
    run: str | None = None,
    space: str | None = None,
    resolution: str | None = None,
    model: str | None = None,
    label: str | None = None,
    desc: str | None = None,
    extension: str = ".tsv",
) -> Path:
    """Return a caller-rooted MVPA group-summary table path."""

    ordered = _ordered_entities(
        entities=entities,
        session_id=session_id,
        task_id=task_id,
        acquisition=acquisition,
        direction=direction,
        run=run,
        space=space,
        resolution=resolution,
        model=model,
        label=label,
        desc=desc,
    )
    filename = _compose_filename(ordered, prefix="group", suffix="mvpasummary", extension=extension)
    return _build_mvpa_path(
        runtime_root=runtime_root,
        publication_root=publication_root,
        mvpa_set=mvpa_set,
        derivative_name=derivative_name,
        section="tables",
        filename=filename,
        ordered_entities=ordered,
        group_level=True,
    )


def build_mvpa_qc_table_filename(
    *,
    entities: Mapping[str, str | None] | None = None,
    subject_id: str | None = None,
    session_id: str | None = None,
    task_id: str | None = None,
    acquisition: str | None = None,
    direction: str | None = None,
    run: str | None = None,
    space: str | None = None,
    resolution: str | None = None,
    model: str | None = None,
    label: str | None = None,
    desc: str | None = None,
    extension: str = ".tsv",
) -> str:
    """Return a BIDS-like MVPA QC-table filename."""

    return _build_mvpa_filename(
        suffix="mvpaqc",
        extension=extension,
        entities=entities,
        subject_id=subject_id,
        session_id=session_id,
        task_id=task_id,
        acquisition=acquisition,
        direction=direction,
        run=run,
        space=space,
        resolution=resolution,
        model=model,
        label=label,
        desc=desc,
    )


def build_mvpa_qc_table_path(
    *,
    runtime_root: str | Path | None = None,
    publication_root: str | Path | None = None,
    mvpa_set: str,
    derivative_name: str = "mvpa",
    entities: Mapping[str, str | None] | None = None,
    subject_id: str | None = None,
    session_id: str | None = None,
    task_id: str | None = None,
    acquisition: str | None = None,
    direction: str | None = None,
    run: str | None = None,
    space: str | None = None,
    resolution: str | None = None,
    model: str | None = None,
    label: str | None = None,
    desc: str | None = None,
    extension: str = ".tsv",
) -> Path:
    """Return a caller-rooted MVPA QC-table path."""

    ordered = _ordered_entities(
        entities=entities,
        subject_id=subject_id,
        session_id=session_id,
        task_id=task_id,
        acquisition=acquisition,
        direction=direction,
        run=run,
        space=space,
        resolution=resolution,
        model=model,
        label=label,
        desc=desc,
    )
    filename = _compose_filename(ordered, suffix="mvpaqc", extension=extension)
    return _build_mvpa_path(
        runtime_root=runtime_root,
        publication_root=publication_root,
        mvpa_set=mvpa_set,
        derivative_name=derivative_name,
        section="qc",
        filename=filename,
        ordered_entities=ordered,
    )


def build_mvpa_sidecar_path(path: str | Path) -> Path:
    """Return the JSON sidecar path for an MVPA artifact using the same stem."""

    return Path(path).with_suffix(".json")


def build_mvpa_provenance_filename(
    *,
    entities: Mapping[str, str | None] | None = None,
    subject_id: str | None = None,
    session_id: str | None = None,
    task_id: str | None = None,
    acquisition: str | None = None,
    direction: str | None = None,
    run: str | None = None,
    space: str | None = None,
    resolution: str | None = None,
    model: str | None = None,
    label: str | None = None,
    desc: str | None = None,
    extension: str = ".json",
) -> str:
    """Return a BIDS-like MVPA provenance filename."""

    return _build_mvpa_filename(
        suffix="provenance",
        extension=extension,
        entities=entities,
        subject_id=subject_id,
        session_id=session_id,
        task_id=task_id,
        acquisition=acquisition,
        direction=direction,
        run=run,
        space=space,
        resolution=resolution,
        model=model,
        label=label,
        desc=desc,
    )


def build_mvpa_provenance_path(
    *,
    runtime_root: str | Path | None = None,
    publication_root: str | Path | None = None,
    mvpa_set: str,
    derivative_name: str = "mvpa",
    entities: Mapping[str, str | None] | None = None,
    subject_id: str | None = None,
    session_id: str | None = None,
    task_id: str | None = None,
    acquisition: str | None = None,
    direction: str | None = None,
    run: str | None = None,
    space: str | None = None,
    resolution: str | None = None,
    model: str | None = None,
    label: str | None = None,
    desc: str | None = None,
    extension: str = ".json",
) -> Path:
    """Return a caller-rooted MVPA provenance path."""

    ordered = _ordered_entities(
        entities=entities,
        subject_id=subject_id,
        session_id=session_id,
        task_id=task_id,
        acquisition=acquisition,
        direction=direction,
        run=run,
        space=space,
        resolution=resolution,
        model=model,
        label=label,
        desc=desc,
    )
    filename = _compose_filename(ordered, suffix="provenance", extension=extension)
    return _build_mvpa_path(
        runtime_root=runtime_root,
        publication_root=publication_root,
        mvpa_set=mvpa_set,
        derivative_name=derivative_name,
        section="provenance",
        filename=filename,
        ordered_entities=ordered,
    )


def build_mvpa_figure_filename(
    *,
    entities: Mapping[str, str | None] | None = None,
    subject_id: str | None = None,
    session_id: str | None = None,
    task_id: str | None = None,
    acquisition: str | None = None,
    direction: str | None = None,
    run: str | None = None,
    space: str | None = None,
    resolution: str | None = None,
    model: str | None = None,
    label: str | None = None,
    desc: str | None = None,
    extension: str = ".png",
) -> str:
    """Return a BIDS-like MVPA figure filename."""

    return _build_mvpa_filename(
        suffix="figure",
        extension=extension,
        entities=entities,
        subject_id=subject_id,
        session_id=session_id,
        task_id=task_id,
        acquisition=acquisition,
        direction=direction,
        run=run,
        space=space,
        resolution=resolution,
        model=model,
        label=label,
        desc=desc,
    )


def build_mvpa_figure_path(
    *,
    runtime_root: str | Path | None = None,
    publication_root: str | Path | None = None,
    mvpa_set: str,
    derivative_name: str = "mvpa",
    entities: Mapping[str, str | None] | None = None,
    subject_id: str | None = None,
    session_id: str | None = None,
    task_id: str | None = None,
    acquisition: str | None = None,
    direction: str | None = None,
    run: str | None = None,
    space: str | None = None,
    resolution: str | None = None,
    model: str | None = None,
    label: str | None = None,
    desc: str | None = None,
    extension: str = ".png",
) -> Path:
    """Return a caller-rooted MVPA figure path."""

    ordered = _ordered_entities(
        entities=entities,
        subject_id=subject_id,
        session_id=session_id,
        task_id=task_id,
        acquisition=acquisition,
        direction=direction,
        run=run,
        space=space,
        resolution=resolution,
        model=model,
        label=label,
        desc=desc,
    )
    filename = _compose_filename(ordered, suffix="figure", extension=extension)
    return _build_mvpa_path(
        runtime_root=runtime_root,
        publication_root=publication_root,
        mvpa_set=mvpa_set,
        derivative_name=derivative_name,
        section="figures",
        filename=filename,
        ordered_entities=ordered,
    )


def build_mvpa_report_filename(
    *,
    entities: Mapping[str, str | None] | None = None,
    subject_id: str | None = None,
    session_id: str | None = None,
    task_id: str | None = None,
    acquisition: str | None = None,
    direction: str | None = None,
    run: str | None = None,
    space: str | None = None,
    resolution: str | None = None,
    model: str | None = None,
    label: str | None = None,
    desc: str | None = None,
    extension: str = ".html",
) -> str:
    """Return a BIDS-like MVPA report filename."""

    return _build_mvpa_filename(
        suffix="report",
        extension=extension,
        entities=entities,
        subject_id=subject_id,
        session_id=session_id,
        task_id=task_id,
        acquisition=acquisition,
        direction=direction,
        run=run,
        space=space,
        resolution=resolution,
        model=model,
        label=label,
        desc=desc,
    )


def build_mvpa_report_path(
    *,
    runtime_root: str | Path | None = None,
    publication_root: str | Path | None = None,
    mvpa_set: str,
    derivative_name: str = "mvpa",
    entities: Mapping[str, str | None] | None = None,
    subject_id: str | None = None,
    session_id: str | None = None,
    task_id: str | None = None,
    acquisition: str | None = None,
    direction: str | None = None,
    run: str | None = None,
    space: str | None = None,
    resolution: str | None = None,
    model: str | None = None,
    label: str | None = None,
    desc: str | None = None,
    extension: str = ".html",
) -> Path:
    """Return a caller-rooted MVPA report path."""

    ordered = _ordered_entities(
        entities=entities,
        subject_id=subject_id,
        session_id=session_id,
        task_id=task_id,
        acquisition=acquisition,
        direction=direction,
        run=run,
        space=space,
        resolution=resolution,
        model=model,
        label=label,
        desc=desc,
    )
    filename = _compose_filename(ordered, suffix="report", extension=extension)
    return _build_mvpa_path(
        runtime_root=runtime_root,
        publication_root=publication_root,
        mvpa_set=mvpa_set,
        derivative_name=derivative_name,
        section="reports",
        filename=filename,
        ordered_entities=ordered,
    )


def _build_mvpa_filename(
    *,
    suffix: str,
    extension: str,
    prefix: str | None = None,
    entities: Mapping[str, str | None] | None = None,
    subject_id: str | None = None,
    session_id: str | None = None,
    task_id: str | None = None,
    acquisition: str | None = None,
    direction: str | None = None,
    run: str | None = None,
    space: str | None = None,
    resolution: str | None = None,
    model: str | None = None,
    label: str | None = None,
    desc: str | None = None,
) -> str:
    ordered = _ordered_entities(
        entities=entities,
        subject_id=subject_id,
        session_id=session_id,
        task_id=task_id,
        acquisition=acquisition,
        direction=direction,
        run=run,
        space=space,
        resolution=resolution,
        model=model,
        label=label,
        desc=desc,
    )
    return _compose_filename(ordered, prefix=prefix, suffix=suffix, extension=extension)


def _build_mvpa_path(
    *,
    runtime_root: str | Path | None,
    publication_root: str | Path | None,
    mvpa_set: str,
    derivative_name: str,
    section: str,
    filename: str,
    ordered_entities: list[tuple[str, str]],
    group_level: bool = False,
) -> Path:
    base = _mvpa_base(
        runtime_root=runtime_root,
        publication_root=publication_root,
        mvpa_set=mvpa_set,
        derivative_name=derivative_name,
    )
    return base / _path_segment(section, "section") / _scope_path(ordered_entities, group_level=group_level) / filename


def _mvpa_base(
    *,
    runtime_root: str | Path | None,
    publication_root: str | Path | None,
    mvpa_set: str,
    derivative_name: str,
) -> Path:
    if (runtime_root is None) == (publication_root is None):
        raise ValueError("Exactly one of runtime_root or publication_root must be defined.")

    mvpa_set_segment = _path_segment(mvpa_set, "mvpa_set")
    if runtime_root is not None:
        return _root_path(runtime_root, "runtime_root") / ".research-platform" / "mvpa" / mvpa_set_segment

    return _root_path(publication_root, "publication_root") / _path_segment(derivative_name, "derivative_name") / mvpa_set_segment


def _ordered_entities(
    *,
    entities: Mapping[str, str | None] | None = None,
    subject_id: str | None = None,
    session_id: str | None = None,
    task_id: str | None = None,
    acquisition: str | None = None,
    direction: str | None = None,
    run: str | None = None,
    space: str | None = None,
    resolution: str | None = None,
    model: str | None = None,
    label: str | None = None,
    desc: str | None = None,
) -> list[tuple[str, str]]:
    raw: dict[str, str | None] = {}
    if entities is not None:
        for key, value in entities.items():
            if key not in MVPA_ENTITY_ORDER:
                raise ValueError(f"Unsupported MVPA entity {key!r}.")
            raw[key] = value

    explicit_entities = {
        "sub": subject_id,
        "ses": session_id,
        "task": task_id,
        "acq": acquisition,
        "dir": direction,
        "run": run,
        "space": space,
        "res": resolution,
        "model": model,
        "label": label,
        "desc": desc,
    }
    for key, value in explicit_entities.items():
        if value is not None:
            raw[key] = value

    ordered: list[tuple[str, str]] = []
    for key in MVPA_ENTITY_ORDER:
        if key not in raw or raw[key] is None:
            continue
        ordered.append((key, _entity_value(key, raw[key])))
    return ordered


def _compose_filename(
    ordered_entities: list[tuple[str, str]],
    *,
    suffix: str,
    extension: str,
    prefix: str | None = None,
) -> str:
    parts: list[str] = []
    if prefix is not None:
        parts.append(_label_value(prefix, "prefix"))

    stem = _entity_stem(ordered_entities)
    if stem:
        parts.append(stem)

    parts.append(_label_value(suffix, "suffix"))
    return f"{'_'.join(parts)}{_extension(extension)}"


def _entity_stem(entities: list[tuple[str, str]]) -> str | None:
    parts: list[str] = []
    for key, value in entities:
        if key in {"sub", "ses"}:
            parts.append(value)
        else:
            parts.append(f"{key}-{value}")
    return "_".join(parts) if parts else None


def _scope_path(ordered_entities: list[tuple[str, str]], *, group_level: bool) -> Path:
    entity_map = dict(ordered_entities)
    parts: list[str] = []
    if group_level:
        parts.append("group")
    elif entity_map.get("sub"):
        parts.append(entity_map["sub"])

    if entity_map.get("ses"):
        parts.append(entity_map["ses"])

    path = Path()
    for part in parts:
        path /= _path_segment(part, "scope")
    return path


def _entity_value(key: str, value: str | None) -> str:
    if key == "sub":
        return _normalize_subject(value)
    if key == "ses":
        return _normalize_session(value)
    return _label_value(value, key)


def _normalize_subject(value: str | None) -> str:
    text = _required_text(value, "subject_id")
    subject_value = text[4:] if text.startswith("sub-") else text
    _ensure_label(subject_value, "subject_id")
    return f"sub-{subject_value}"


def _normalize_session(value: str | None) -> str:
    text = _required_text(value, "session_id")
    session_value = text[4:] if text.startswith("ses-") else text
    _ensure_label(session_value, "session_id")
    return f"ses-{session_value}"


def _label_value(value: str | None, label: str) -> str:
    text = _required_text(value, label)
    _ensure_label(text, label)
    return text


def _path_segment(value: str | None, label: str) -> str:
    text = _required_text(value, label)
    if text in {".", ".."} or not _PATH_SEGMENT.fullmatch(text):
        raise ValueError(f"{label} must be a single safe path segment.")
    return text


def _root_path(value: str | Path | None, label: str) -> Path:
    if value is None:
        raise ValueError(f"{label} must be defined.")
    text = str(value).strip()
    if not text:
        raise ValueError(f"{label} must not be empty.")
    return Path(text)


def _extension(value: str) -> str:
    text = _required_text(value, "extension")
    normalized = text if text.startswith(".") else f".{text}"
    if not _EXTENSION.fullmatch(normalized):
        raise ValueError("extension must be a single safe file extension.")
    return normalized


def _ensure_label(value: str, label: str) -> None:
    if not _BIDS_LABEL_VALUE.fullmatch(value):
        raise ValueError(f"{label} must contain only letters and numbers.")


def _required_text(value: str | None, label: str) -> str:
    if value is None:
        raise ValueError(f"{label} must be defined.")
    text = str(value).strip()
    if not text:
        raise ValueError(f"{label} must not be empty.")
    if "/" in text or "\\" in text or ":" in text or text.startswith("~") or Path(text).is_absolute():
        raise ValueError(f"{label} must not look like a path.")
    return text


__all__ = [
    "MVPA_ENTITY_ORDER",
    "build_mvpa_distance_table_filename",
    "build_mvpa_distance_table_path",
    "build_mvpa_figure_filename",
    "build_mvpa_figure_path",
    "build_mvpa_group_summary_table_filename",
    "build_mvpa_group_summary_table_path",
    "build_mvpa_pattern_table_filename",
    "build_mvpa_pattern_table_path",
    "build_mvpa_provenance_filename",
    "build_mvpa_provenance_path",
    "build_mvpa_qc_table_filename",
    "build_mvpa_qc_table_path",
    "build_mvpa_rdm_table_filename",
    "build_mvpa_rdm_table_path",
    "build_mvpa_report_filename",
    "build_mvpa_report_path",
    "build_mvpa_sidecar_path",
]
