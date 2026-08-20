"""Internal publication helpers for polished ROI derivative layouts."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from string import Formatter
from typing import Any
import csv
import json
import os
import re
import shutil
import tempfile

from research_platform.bids.roi import (
    build_loso_flame1_mask_path,
    build_loso_flame1_roistats_path,
    build_loso_flame1_statmap_path,
    build_roi_sidecar_path,
)
from research_platform.neuro.roi import (
    ExtractionSet,
    RoiSet,
    parse_extraction_set_document,
    parse_roi_set_document,
)
from research_platform.neuro._roi_path_safety import (
    UnmappedLocalPathError,
    portable_path_reference,
    published_text_contains_local_path_reference,
    published_value_local_path_fields,
)


LOSO_FLAME1_BIDSLIKE_LAYOUT = "loso_flame1_bidslike"
DEFAULT_PUBLICATION_DIRNAME = "roi-loso-flame1"
DEFAULT_BIDS_VERSION = "1.11.1"
_PRIVATE_RUNTIME_TABLE_COLUMNS = frozenset({"featquery_command", "featquery_output_dir", "report_path"})
_PORTABLE_INPUT_TABLE_COLUMNS = frozenset({"feat_dir", "roi_mask_path"})


class RoiPublicationError(ValueError):
    """Raised when a canonical ROI derivative cannot be published safely."""


class _RoiPublicationRecoveryError(RoiPublicationError):
    """Raised when a failed promotion could not be rolled back completely."""


@dataclass(frozen=True)
class _PublicationFile:
    destination: Path
    category: str
    source: Path | None = None
    json_payload: Mapping[str, Any] | None = None
    text: str | None = None
    table_source: Path | None = None


@dataclass(frozen=True)
class RoiPublicationResult:
    """Summary used to gate destructive runtime cleanup."""

    enabled: bool
    root: Path | None = None
    paths: tuple[Path, ...] = ()
    missing_sources: tuple[Path, ...] = ()
    missing_destinations: tuple[Path, ...] = ()
    duplicate_destinations: tuple[Path, ...] = ()
    expected_sources: int = 0
    details: Mapping[str, Any] = field(default_factory=dict)

    @property
    def complete(self) -> bool:
        return (
            self.enabled
            and self.expected_sources > 0
            and bool(self.paths)
            and not self.missing_sources
            and not self.missing_destinations
            and not self.duplicate_destinations
        )

    def to_dict(self) -> dict[str, Any]:
        return _json_ready(
            {
                "enabled": self.enabled,
                "complete": self.complete,
                "root": self.root,
                "paths": list(self.paths),
                "missing_sources": list(self.missing_sources),
                "missing_destinations": list(self.missing_destinations),
                "duplicate_destinations": list(self.duplicate_destinations),
                "expected_sources": self.expected_sources,
                "details": dict(self.details),
            }
        )


def publish_loso_roi_build(
    document: Mapping[str, Any],
    *,
    actions: Sequence[Any],
    context: Any,
) -> tuple[Path, ...]:
    """Publish executed LOSO FLAME1 maps and masks when config enables it."""

    return publish_loso_roi_build_result(document, actions=actions, context=context).paths


def publish_loso_roi_build_result(
    document: Mapping[str, Any],
    *,
    actions: Sequence[Any],
    context: Any,
) -> RoiPublicationResult:
    """Publish executed LOSO FLAME1 maps and masks and report completeness."""

    roi_set = parse_roi_set_document(document, validate_personal_paths=False)
    settings = _publication_settings(roi_set.fields)
    if settings is None or not _has_loso_publication_actions(actions):
        return RoiPublicationResult(enabled=False)

    root = _publication_root(settings, context=context, default_root=_default_build_publication_base(actions))
    contrast_aliases = _contrast_aliases_from_roi_set(roi_set, settings=settings)
    files = list(_dataset_publication_files(root, settings=settings, contrast_aliases=contrast_aliases))
    expected_destinations: list[Path] = [entry.destination for entry in files]
    destination_owners: dict[Path, str] = {}
    duplicate_destinations: list[Path] = []
    missing_sources: list[Path] = []
    portability_errors: list[RoiPublicationError] = []
    expected_sources = 0
    published_maps: dict[tuple[str, str], Path] = {}
    for action_index, action in enumerate(actions):
        if action.family != "loso_group_map":
            continue
        job = _action_loso_job(action)
        if job is None:
            continue
        contrast = _mapping(job.get("contrast"))
        contrast_id = _text(contrast.get("contrast_id") or contrast.get("id") or contrast.get("name"))
        if contrast_id is None:
            continue
        contrast_desc = _text(contrast.get("desc") or contrast.get("contrast_desc")) or contrast_id
        contrast_alias = contrast_alias_for(contrast_id, contrast_desc=contrast_desc, contrast_fields=contrast, settings=settings)
        entities = _action_entities(action, job=job)
        desc_values = _publication_template_values(action, job=job, entities=entities)
        map_desc = _bids_label(_format_desc(settings.get("map_desc") or "{model}LOSOFlame1", desc_values))
        mask_desc = _bids_label(_format_desc(settings.get("mask_desc") or "{model}LOSOFlame1Sphere{sphere_radius_mm}mm", desc_values))

        map_key = (str(job.get("zstat_path")), contrast_id)
        published_map = published_maps.get(map_key)
        if published_map is None:
            source_map = Path(str(job["zstat_path"]))
            published_map = build_loso_flame1_statmap_path(
                root,
                session_id=entities.get("session_id"),
                task_id=str(entities["task_id"]),
                direction=entities.get("direction"),
                space=str(entities["space"]),
                resolution=str(entities["resolution"]),
                contrast_alias=contrast_alias,
                heldout_subject=str(job.get("heldout_subject") or entities["subject_id"]),
                map_desc=map_desc,
                datatype=str(entities.get("datatype", "func")),
            )
            map_sidecar = build_roi_sidecar_path(published_map)
            expected_destinations.extend([published_map, map_sidecar])
            map_owner = f"map:{action_index}:{contrast_id}"
            map_destination_unique = _record_destinations(
                (published_map, map_sidecar),
                owner=map_owner,
                destination_owners=destination_owners,
                duplicate_destinations=duplicate_destinations,
            )
            expected_sources += 1
            if not source_map.is_file():
                missing_sources.append(source_map)
            if map_destination_unique:
                files.append(_PublicationFile(destination=published_map, category="map", source=source_map))
                try:
                    map_payload = _statmap_sidecar(
                        action,
                        job=job,
                        entities=entities,
                        contrast_alias=contrast_alias,
                        contrast_id=contrast_id,
                        contrast_desc=contrast_desc,
                        published_map=published_map,
                        publication_root=root,
                        context=context,
                        output_name=_publication_relative_path(map_sidecar, root).as_posix(),
                    )
                except RoiPublicationError as exc:
                    portability_errors.append(exc)
                else:
                    files.append(
                        _PublicationFile(
                            destination=map_sidecar,
                            category="map JSON sidecar",
                            json_payload=map_payload,
                        )
                    )
            published_maps[map_key] = published_map

        source_mask = Path(action.mask_path)
        mask_path = build_loso_flame1_mask_path(
            root,
            subject_id=str(entities["subject_id"]),
            session_id=entities.get("session_id"),
            task_id=str(entities["task_id"]),
            direction=entities.get("direction"),
            space=str(entities["space"]),
            resolution=str(entities["resolution"]),
            roi_label=action.roi_label,
            contrast_alias=contrast_alias,
            mask_desc=mask_desc,
            datatype=str(entities.get("datatype", "func")),
        )
        mask_sidecar = build_roi_sidecar_path(mask_path)
        expected_destinations.extend([mask_path, mask_sidecar])
        mask_owner = f"mask:{action_index}:{action.roi_label}:{contrast_id}"
        mask_destination_unique = _record_destinations(
            (mask_path, mask_sidecar),
            owner=mask_owner,
            destination_owners=destination_owners,
            duplicate_destinations=duplicate_destinations,
        )
        expected_sources += 1
        if not source_mask.is_file():
            missing_sources.append(source_mask)
        if mask_destination_unique:
            files.append(_PublicationFile(destination=mask_path, category="mask", source=source_mask))
            try:
                mask_payload = _mask_sidecar(
                    action,
                    job=job,
                    entities=entities,
                    contrast_alias=contrast_alias,
                    contrast_id=contrast_id,
                    contrast_desc=contrast_desc,
                    published_map=published_map,
                    publication_root=root,
                    context=context,
                    output_name=_publication_relative_path(mask_sidecar, root).as_posix(),
                )
            except RoiPublicationError as exc:
                portability_errors.append(exc)
            else:
                files.append(
                    _PublicationFile(
                        destination=mask_sidecar,
                        category="mask JSON sidecar",
                        json_payload=mask_payload,
                    )
                )
    published_paths = tuple(_unique_paths(expected_destinations))
    if not missing_sources and not duplicate_destinations and expected_sources > 0:
        if portability_errors:
            raise portability_errors[0]
        _render_validate_and_promote_publication(files, root=root, settings=settings, context=context)
    missing_destinations = tuple(path for path in published_paths if not path.exists())
    return RoiPublicationResult(
        enabled=True,
        root=root,
        paths=published_paths,
        missing_sources=tuple(_unique_paths(missing_sources)),
        missing_destinations=missing_destinations,
        duplicate_destinations=tuple(_unique_paths(duplicate_destinations)),
        expected_sources=expected_sources,
        details={"kind": "loso_roi_build"},
    )


def publish_loso_featquery_extraction(
    extraction_document: Mapping[str, Any],
    *,
    roi_set_document: Mapping[str, Any] | None,
    actions: Sequence[Any],
    tables: Sequence[Path],
    context: Any,
) -> tuple[Path, ...]:
    """Publish executed FSL featquery summary tables when config enables it."""

    return publish_loso_featquery_extraction_result(
        extraction_document,
        roi_set_document=roi_set_document,
        actions=actions,
        tables=tables,
        context=context,
    ).paths


def publish_loso_featquery_extraction_result(
    extraction_document: Mapping[str, Any],
    *,
    roi_set_document: Mapping[str, Any] | None,
    actions: Sequence[Any],
    tables: Sequence[Path],
    context: Any,
) -> RoiPublicationResult:
    """Publish executed FSL featquery tables and report completeness."""

    extraction_set = parse_extraction_set_document(extraction_document, validate_personal_paths=False)
    roi_set = parse_roi_set_document(roi_set_document, validate_personal_paths=False) if roi_set_document is not None else None
    settings = _publication_settings(extraction_set.fields, fallback=roi_set.fields if roi_set is not None else None)
    if settings is None or not any(action.backend == "fsl_featquery" for action in actions):
        return RoiPublicationResult(enabled=False)

    root = _publication_root(settings, context=context, default_root=_default_extraction_publication_base(actions, tables))
    contrast_aliases = _contrast_aliases_from_extraction_set(extraction_set, settings=settings, roi_set=roi_set)
    files = list(_dataset_publication_files(root, settings=settings, contrast_aliases=contrast_aliases))
    expected_destinations: list[Path] = [entry.destination for entry in files]
    destination_owners: dict[Path, str] = {}
    duplicate_destinations: list[Path] = []
    missing_sources: list[Path] = []
    expected_sources = 0
    table_paths = tuple(_unique_paths(Path(path) for path in tables))
    for table_index, table_path in enumerate(table_paths):
        expected_sources += 1
        if not table_path.is_file():
            missing_sources.append(table_path)
        table_actions = [action for action in actions if Path(action.table_path) == table_path]
        entities = _table_entities(table_actions, extraction_set=extraction_set)
        desc_values = _table_template_values(table_actions, extraction_set=extraction_set, roi_set=roi_set, entities=entities)
        base_desc = _bids_label(_format_desc(settings.get("table_desc") or "{model}LOSOFlame1Featquery", desc_values))
        table_desc = f"{base_desc}QC" if _is_qc_table(table_path) else base_desc
        published_table = build_loso_flame1_roistats_path(
            root,
            session_id=entities.get("session_id"),
            task_id=str(entities["task_id"]),
            direction=entities.get("direction"),
            table_desc=table_desc,
            datatype=str(entities.get("datatype", "func")),
            extension=table_path.suffix or ".tsv",
        )
        sidecar = _table_sidecar_path(published_table)
        expected_destinations.extend([published_table, sidecar])
        table_destination_unique = _record_destinations(
            (published_table, sidecar),
            owner=f"table:{table_index}",
            destination_owners=destination_owners,
            duplicate_destinations=duplicate_destinations,
        )
        if table_destination_unique:
            public_columns = _public_table_columns(table_path)
            files.extend(
                (
                    _PublicationFile(
                        destination=published_table,
                        category="ROI extraction table",
                        table_source=table_path,
                    ),
                    _PublicationFile(
                        destination=sidecar,
                        category="ROI extraction table JSON sidecar",
                        json_payload=_roistats_dictionary(
                            public_columns,
                            extraction_set=extraction_set,
                            roi_set=roi_set,
                            actions=table_actions or actions,
                            settings=settings,
                            is_qc=_is_qc_table(table_path),
                        ),
                    ),
                )
            )
    published_paths = tuple(_unique_paths(expected_destinations))
    if not missing_sources and not duplicate_destinations and expected_sources > 0:
        _render_validate_and_promote_publication(files, root=root, settings=settings, context=context)
    missing_destinations = tuple(path for path in published_paths if not path.exists())
    return RoiPublicationResult(
        enabled=True,
        root=root,
        paths=published_paths,
        missing_sources=tuple(_unique_paths(missing_sources)),
        missing_destinations=missing_destinations,
        duplicate_destinations=tuple(_unique_paths(duplicate_destinations)),
        expected_sources=expected_sources,
        details={"kind": "loso_featquery_extraction"},
    )


def contrast_alias_for(
    contrast_id: str,
    *,
    contrast_desc: str | None = None,
    contrast_fields: Mapping[str, Any] | None = None,
    settings: Mapping[str, Any] | None = None,
) -> str:
    """Return a compact BIDS-safe contrast alias without study-specific defaults."""

    fields = _mapping(contrast_fields)
    aliases = _mapping(_mapping(settings).get("contrast_aliases"))
    for key in (contrast_id, contrast_desc):
        if key is not None and aliases.get(key) is not None:
            return _bids_label(str(aliases[key]))
    for key in ("alias", "compact_alias", "filename_alias", "bids_alias"):
        if fields.get(key) is not None:
            return _bids_label(str(fields[key]))
    return _safe_camel_alias(contrast_desc or contrast_id)


def expected_loso_roi_publication_mask_specs(
    document: Mapping[str, Any],
    *,
    actions: Sequence[Any],
    context: Any,
) -> tuple[dict[str, Any], ...]:
    """Return expected published LOSO mask paths for already planned build actions."""

    roi_set = parse_roi_set_document(document, validate_personal_paths=False)
    settings = _publication_settings(roi_set.fields)
    if settings is None:
        raise ValueError("roi_mask_source.source=roi_set_publication requires the referenced ROI set to enable LOSO publication.")
    root = _publication_root(settings, context=context, default_root=_default_build_publication_base(actions))
    specs: list[dict[str, Any]] = []
    destinations: list[Path] = []
    for action in actions:
        if getattr(action, "family", None) != "loso_group_map":
            continue
        spec = _expected_loso_mask_spec(action, settings=settings, root=root)
        if spec is None:
            continue
        specs.append(spec)
        destinations.append(Path(spec["mask_path"]))
    duplicates = _duplicate_paths(destinations)
    if duplicates:
        duplicate_text = ", ".join(str(path) for path in duplicates)
        raise ValueError(f"Published LOSO ROI mask path generation produced duplicate destination path(s): {duplicate_text}.")
    if not specs:
        raise ValueError("No LOSO ROI publication masks could be derived from the referenced ROI set.")
    return tuple(specs)


def _publication_settings(
    primary: Mapping[str, Any],
    *,
    fallback: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    fallback_publication = _mapping(_mapping(fallback).get("publication"))
    primary_publication = _mapping(_mapping(primary).get("publication"))
    if not fallback_publication and not primary_publication:
        return None
    settings = {**fallback_publication, **primary_publication}
    if not _bool_value(settings.get("enabled"), default=False):
        return None
    if _text(settings.get("layout")) not in {None, LOSO_FLAME1_BIDSLIKE_LAYOUT}:
        return None
    settings["layout"] = LOSO_FLAME1_BIDSLIKE_LAYOUT
    return settings


def _expected_loso_mask_spec(action: Any, *, settings: Mapping[str, Any], root: Path) -> dict[str, Any] | None:
    job = _action_loso_job(action)
    if job is None:
        return None
    contrast = _mapping(job.get("contrast"))
    contrast_id = _text(contrast.get("contrast_id") or contrast.get("id") or contrast.get("name"))
    if contrast_id is None:
        return None
    contrast_desc = _text(contrast.get("desc") or contrast.get("contrast_desc")) or contrast_id
    contrast_alias = contrast_alias_for(contrast_id, contrast_desc=contrast_desc, contrast_fields=contrast, settings=settings)
    entities = _action_entities(action, job=job)
    desc_values = _publication_template_values(action, job=job, entities=entities)
    mask_desc = _bids_label(_format_desc(settings.get("mask_desc") or "{model}LOSOFlame1Sphere{sphere_radius_mm}mm", desc_values))
    mask_path = build_loso_flame1_mask_path(
        root,
        subject_id=str(entities["subject_id"]),
        session_id=entities.get("session_id"),
        task_id=str(entities["task_id"]),
        direction=entities.get("direction"),
        space=str(entities["space"]),
        resolution=str(entities["resolution"]),
        roi_label=action.roi_label,
        contrast_alias=contrast_alias,
        mask_desc=mask_desc,
        datatype=str(entities.get("datatype", "func")),
    )
    return {
        "action": action,
        "roi_label": action.roi_label,
        "mask_path": mask_path,
        "roi_desc": _text(_mapping(getattr(action, "metadata", {})).get("desc")),
        "roi_family": getattr(action, "family", None),
        "entities": dict(entities),
        "contrast_id": contrast_id,
        "contrast_alias": contrast_alias,
        "mask_desc": mask_desc,
        "publication_root": root,
    }


def _publication_root(settings: Mapping[str, Any], *, context: Any, default_root: Path | None) -> Path:
    raw = settings.get("root") or settings.get("output_root")
    if isinstance(raw, Mapping):
        root_ref = _text(raw.get("root_ref"))
        base = Path(context.resolve_root_ref(root_ref)).resolve() if root_ref else Path(context.project_root).resolve()
        subpath = _text(raw.get("path") or raw.get("subpath"))
        return (base / subpath).resolve() if subpath else base
    if raw is not None:
        path = Path(str(raw)).expanduser()
        if path.is_absolute():
            resolved_path = path.resolve()
            if (
                default_root is not None
                and resolved_path == (Path(default_root) / DEFAULT_PUBLICATION_DIRNAME).resolve()
                and "dataset_derivatives_root" in getattr(context, "root_refs", {})
            ):
                return (Path(context.resolve_root_ref("dataset_derivatives_root")) / DEFAULT_PUBLICATION_DIRNAME).resolve()
            if str(path) == f"/{DEFAULT_PUBLICATION_DIRNAME}" and not os.environ.get("ROI_DERIV_ROOT") and default_root is not None:
                if "dataset_derivatives_root" in getattr(context, "root_refs", {}):
                    return (Path(context.resolve_root_ref("dataset_derivatives_root")) / DEFAULT_PUBLICATION_DIRNAME).resolve()
                return (default_root / DEFAULT_PUBLICATION_DIRNAME).resolve()
            return resolved_path
        return (Path(context.project_root) / path).resolve()
    if default_root is not None:
        return (default_root / DEFAULT_PUBLICATION_DIRNAME).resolve()
    return (Path(context.artifacts_root) / DEFAULT_PUBLICATION_DIRNAME).resolve()


def _default_build_publication_base(actions: Sequence[Any]) -> Path | None:
    for action in actions:
        job = _action_loso_job(action)
        if job is not None and job.get("output_root") is not None:
            return Path(str(job["output_root"]))
    return None


def _default_extraction_publication_base(actions: Sequence[Any], tables: Sequence[Path]) -> Path | None:
    for action in actions:
        metadata = _mapping(action.metadata)
        roi_sidecar = _mapping(metadata.get("roi_sidecar"))
        group_map = _text(roi_sidecar.get("group_map_path"))
        if group_map and "/loso_groupmaps/" in group_map:
            return Path(group_map.split("/loso_groupmaps/", 1)[0])
    for table in tables:
        parts = Path(table).parts
        if "roi_extract" in parts:
            index = parts.index("roi_extract")
            if index:
                return Path(*parts[:index])
    return None


def _dataset_publication_files(
    root: Path,
    *,
    settings: Mapping[str, Any],
    contrast_aliases: Mapping[str, str],
) -> tuple[_PublicationFile, ...]:
    dataset = _mapping(settings.get("dataset_description"))
    generated_by_name = _text(dataset.get("generated_by_name")) or _text(settings.get("generated_by_name")) or DEFAULT_PUBLICATION_DIRNAME
    generated_by: list[dict[str, Any]] = [
        {
            "Name": generated_by_name,
            "Description": _text(dataset.get("description"))
            or "Leave-one-subject-out FLAME1 group maps, subject-specific ROI masks, and featquery ROI extraction tables.",
        }
    ]
    version = _text(dataset.get("generated_by_version") or settings.get("version"))
    if version:
        generated_by[0]["Version"] = version
    generated_by.append({"Name": "FSL", "Description": "FSL FLAME1 and featquery command-line tools."})

    payload: dict[str, Any] = {
        "Name": _text(dataset.get("name")) or "ROI LOSO FLAME1 outputs",
        "BIDSVersion": _text(dataset.get("bids_version")) or DEFAULT_BIDS_VERSION,
        "DatasetType": "derivative",
        "GeneratedBy": generated_by,
    }
    dataset_links = dataset.get("dataset_links") or settings.get("dataset_links")
    if isinstance(dataset_links, Mapping):
        payload["DatasetLinks"] = dict(dataset_links)
    if contrast_aliases:
        payload["ContrastAliases"] = dict(sorted(contrast_aliases.items()))
    return (
        _PublicationFile(
            destination=root / "dataset_description.json",
            category="dataset JSON metadata",
            json_payload=payload,
        ),
        _PublicationFile(
            destination=root / "README.md",
            category="dataset Markdown metadata",
            text=_publication_readme_text(payload),
        ),
    )


def _publication_readme_text(dataset_description: Mapping[str, Any]) -> str:
    name = _text(dataset_description.get("Name")) or "ROI LOSO FLAME1 outputs"
    lines = [
        f"# {name}",
        "",
        "This derivative layout is generated automatically from the LOSO FLAME1 ROI and FSL featquery workflow configuration.",
        "",
        "- `maps/` contains held-out-subject LOSO FLAME1 z-statistic maps.",
        "- `masks/` contains subject-specific ROI masks used for extraction.",
        "- `tables/` contains analysis and QC ROI extraction summaries.",
        "",
        "The runtime/cache layout remains separate from this published layout.",
        "Exact runtime paths and commands remain in private run artifacts and are not copied into this derivative.",
    ]
    return "\n".join(lines) + "\n"


def _statmap_sidecar(
    action: Any,
    *,
    job: Mapping[str, Any],
    entities: Mapping[str, Any],
    contrast_alias: str,
    contrast_id: str,
    contrast_desc: str,
    published_map: Path,
    publication_root: Path,
    context: Any,
    output_name: str,
) -> dict[str, Any]:
    contrast = _mapping(job.get("contrast"))
    payload = {
        "Description": "Leave-one-subject-out FLAME1 group z-statistic map used for ROI definition.",
        "AnalysisLevel": "group",
        "Model": job.get("model"),
        "Estimator": "FLAME1",
        "TaskName": job.get("task_id"),
        "Direction": entities.get("direction"),
        "Space": entities.get("space"),
        "Resolution": entities.get("resolution"),
        "Statistic": "z",
        "ContrastAlias": contrast_alias,
        "ContrastName": contrast_id,
        "ContrastDescription": contrast_desc,
        "FSLCOPE": _maybe_int(contrast.get("cope_number") or contrast.get("cope")),
        "HeldOutSubject": _subject_label(str(job.get("heldout_subject"))),
        "TrainingSubjects": [_subject_label(str(item.get("subject_id"))) for item in _sequence(job.get("training_inputs")) if isinstance(item, Mapping)],
        "Sources": _portable_source_references(
            _unique_text([str(job["zstat_path"]), *_source_paths_for_job(job)]),
            publication_root=publication_root,
            context=context,
            output_name=output_name,
            category="map source reference",
        ),
        "PublishedPath": _portable_source_reference(
            published_map,
            publication_root=publication_root,
            context=context,
            output_name=output_name,
            category="published map reference",
        ),
    }
    return _compact(payload)


def _mask_sidecar(
    action: Any,
    *,
    job: Mapping[str, Any],
    entities: Mapping[str, Any],
    contrast_alias: str,
    contrast_id: str,
    contrast_desc: str,
    published_map: Path,
    publication_root: Path,
    context: Any,
    output_name: str,
) -> dict[str, Any]:
    runtime = _read_json(Path(action.sidecar_path))
    parameters = _mapping(_mapping(action.metadata).get("roi_parameters"))
    payload = {
        "Description": "Subject-specific LOSO FLAME1 ROI mask used for ROI extraction.",
        "AnalysisLevel": "participant",
        "Subject": _subject_label(str(entities.get("subject_id"))),
        "ROI": action.roi_label,
        "ROILabel": action.roi_label,
        "ContrastAlias": contrast_alias,
        "ContrastName": contrast_id,
        "ContrastDescription": contrast_desc,
        "FSLCOPE": _maybe_int(_mapping(job.get("contrast")).get("cope_number")),
        "HeldOutSubject": _subject_label(str(job.get("heldout_subject"))),
        "DefiningMap": _portable_source_reference(
            published_map,
            publication_root=publication_root,
            context=context,
            output_name=output_name,
            category="defining map reference",
        ),
        "SeedCoordinate": runtime.get("seed_coordinate") or parameters.get("seed_coordinate"),
        "PeakCoordinate": runtime.get("loso_peak_coordinate")
        or runtime.get("selected_peak_coordinate")
        or runtime.get("peak_coordinate"),
        "PeakZStatistic": runtime.get("selected_peak_z") or runtime.get("selected_peak_stat") or runtime.get("z_at_peak"),
        "SphereRadiusMM": runtime.get("sphere_radius_mm") or parameters.get("sphere_radius_mm"),
        "SearchRadiusMM": runtime.get("search_radius_mm") or parameters.get("search_radius_mm"),
        "CoverageMasksApplied": _portable_source_value(
            runtime.get("coverage_masks") or _mapping(action.metadata).get("coverage_masks"),
            publication_root=publication_root,
            context=context,
            output_name=output_name,
            category="coverage mask reference",
        ),
        "VoxelCount": runtime.get("voxel_count"),
        "FallbackStatus": runtime.get("fallback_status"),
        "QCFlags": runtime.get("qc_flags"),
        "Warnings": runtime.get("warnings"),
        "Sources": _portable_source_references(
            _unique_text([str(action.mask_path), str(job.get("zstat_path")), *_source_paths_for_job(job)]),
            publication_root=publication_root,
            context=context,
            output_name=output_name,
            category="mask source reference",
        ),
    }
    return _compact(payload)


def _roistats_dictionary(
    columns: Sequence[str],
    *,
    extraction_set: ExtractionSet,
    roi_set: RoiSet | None,
    actions: Sequence[Any],
    settings: Mapping[str, Any],
    is_qc: bool,
) -> dict[str, Any]:
    column_dict = {column: _column_description(column) for column in columns}
    contrasts = _table_contrasts(actions, settings=settings)
    payload = {
        "Description": "QC/audit ROI extraction table." if is_qc else "Analysis-facing ROI extraction values table.",
        "Columns": column_dict,
        "ROIExtraction": {
            "ExtractionSet": extraction_set.name,
            "ROISet": roi_set.name if roi_set is not None else extraction_set.roi_set,
            "Backend": "fsl_featquery",
            "Metrics": sorted({metric for action in actions for metric in getattr(action, "metrics", ())}),
        },
        "Contrasts": contrasts,
        "ROIs": sorted({str(action.roi_label) for action in actions}),
    }
    return _compact(payload)


def _column_description(column: str) -> dict[str, Any]:
    descriptions = {
        "subject_id": "Participant identifier without the `sub-` prefix.",
        "session_id": "Session identifier without the `ses-` prefix.",
        "task_id": "Task identifier.",
        "model": "Model identifier used for the source fixed-effects analysis.",
        "roi_set": "ROI set identifier.",
        "roi_label": "ROI label.",
        "roi_desc": "ROI description.",
        "roi_family": "ROI-generation family.",
        "source_contrast": "Configured source contrast identifier.",
        "cope": "FSL COPE index.",
        "mean_cope": "Raw mean COPE value reported by featquery without percent-signal-change conversion.",
        "mean_psc": "Mean percent signal change reported by featquery with FSL's -p conversion.",
        "roi_voxel_count": "Number of ROI voxels.",
        "thresholded_peak": "Whether the ROI peak met the configured z threshold.",
        "below_threshold_fallback": "Whether the ROI used the below-threshold fallback peak.",
        "peak_x_mm": "Selected peak x coordinate in millimeters.",
        "peak_y_mm": "Selected peak y coordinate in millimeters.",
        "peak_z_mm": "Selected peak z coordinate in millimeters.",
        "z_at_peak": "Z statistic at the selected peak.",
        "included_in_values": "Whether this QC row was included in the analysis-facing values table.",
        "exclude_reason": "Reason a QC row was excluded from the values table.",
        "feat_dir": "Source FEAT directory.",
        "roi_mask_path": "ROI mask path used by featquery.",
        "featquery_output_dir": "Runtime featquery output directory.",
        "report_path": "Runtime featquery report path.",
        "backend": "Extraction backend.",
        "featquery_command": "Serialized featquery command.",
        "usable": "Whether the row passed blocking QC for analysis use.",
        "qc_flags": "Semicolon-separated QC flags.",
        "warnings": "Semicolon-separated warnings.",
    }
    payload: dict[str, Any] = {"Description": descriptions.get(column, f"Column `{column}` from the ROI extraction workflow.")}
    if column == "mean_cope":
        payload["Units"] = "arbitrary"
    if column == "mean_psc":
        payload["Units"] = "percent signal change"
    if column in {"peak_x_mm", "peak_y_mm", "peak_z_mm"}:
        payload["Units"] = "millimeters"
    if column == "z_at_peak":
        payload["Units"] = "z statistic"
    if column in {"cope", "roi_voxel_count"}:
        payload["Format"] = "integer"
    return payload


def _table_contrasts(actions: Sequence[Any], *, settings: Mapping[str, Any]) -> list[dict[str, Any]]:
    contrasts: dict[str, dict[str, Any]] = {}
    for action in actions:
        metadata = _mapping(action.metadata)
        contrast_id = _text(metadata.get("source_contrast"))
        if contrast_id is None:
            continue
        alias = contrast_alias_for(contrast_id, contrast_desc=contrast_id, settings=settings)
        contrasts[contrast_id] = {
            "ContrastAlias": alias,
            "ContrastName": contrast_id,
            "FSLCOPE": _maybe_int(metadata.get("cope")),
        }
    return [contrasts[key] for key in sorted(contrasts)]


def _contrast_aliases_from_roi_set(roi_set: RoiSet, *, settings: Mapping[str, Any]) -> dict[str, str]:
    aliases: dict[str, str] = {}
    raw = roi_set.fields.get("contrasts")
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        for item in raw:
            if not isinstance(item, Mapping):
                continue
            contrast_id = _text(item.get("id") or item.get("name") or item.get("contrast_id") or item.get("contrast"))
            if contrast_id is None:
                continue
            desc = _text(item.get("desc") or item.get("contrast_desc")) or contrast_id
            aliases[contrast_id] = contrast_alias_for(contrast_id, contrast_desc=desc, contrast_fields=item, settings=settings)
    return aliases


def _contrast_aliases_from_extraction_set(
    extraction_set: ExtractionSet,
    *,
    settings: Mapping[str, Any],
    roi_set: RoiSet | None,
) -> dict[str, str]:
    aliases = _contrast_aliases_from_roi_set(roi_set, settings=settings) if roi_set is not None else {}
    for target in extraction_set.targets:
        raw = target.fields.get("contrasts") or target.fields.get("source_contrasts")
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
            for item in raw:
                if not isinstance(item, Mapping):
                    continue
                contrast_id = _text(item.get("id") or item.get("name") or item.get("source_contrast") or item.get("contrast_id"))
                if contrast_id is None:
                    continue
                desc = _text(item.get("desc") or item.get("contrast_desc")) or contrast_id
                aliases[contrast_id] = contrast_alias_for(contrast_id, contrast_desc=desc, contrast_fields=item, settings=settings)
    return aliases


def _action_entities(action: Any, *, job: Mapping[str, Any]) -> dict[str, Any]:
    metadata_entities = _mapping(_mapping(action.metadata).get("entities"))
    output = {
        "subject_id": _strip_prefix(_text(metadata_entities.get("subject_id")) or _text(job.get("heldout_subject")) or "", "sub"),
        "session_id": _strip_prefix(_text(metadata_entities.get("session_id")) or _text(job.get("session_id")), "ses"),
        "task_id": _text(metadata_entities.get("task_id")) or _text(job.get("task_id")) or "task",
        "model": _text(metadata_entities.get("model")) or _text(job.get("model")) or "",
        "direction": _text(metadata_entities.get("direction") or metadata_entities.get("dir")),
        "space": _text(metadata_entities.get("space")) or "space",
        "resolution": _text(metadata_entities.get("resolution") or metadata_entities.get("res")) or "1",
        "datatype": _text(metadata_entities.get("datatype")) or "func",
    }
    return {key: value for key, value in output.items() if value is not None}


def _publication_template_values(action: Any, *, job: Mapping[str, Any], entities: Mapping[str, Any]) -> dict[str, Any]:
    parameters = _mapping(_mapping(action.metadata).get("roi_parameters"))
    return {
        **entities,
        "model": entities.get("model") or job.get("model") or "",
        "sphere_radius_mm": _format_number(parameters.get("sphere_radius_mm") or parameters.get("radius_mm") or ""),
    }


def _table_entities(actions: Sequence[Any], *, extraction_set: ExtractionSet) -> dict[str, Any]:
    metadata = _mapping(actions[0].metadata) if actions else {}
    fields = extraction_set.fields
    session = _strip_prefix(_text(metadata.get("session_id")) or _first_list_text(fields.get("sessions") or fields.get("session")), "ses")
    task = _text(metadata.get("task_id")) or _first_list_text(fields.get("tasks") or fields.get("task")) or "task"
    direction = _text(metadata.get("direction") or metadata.get("dir")) or _first_list_text(fields.get("directions") or fields.get("direction"))
    return {
        "session_id": session,
        "task_id": task,
        "direction": direction,
        "model": _text(metadata.get("model")) or _first_list_text(fields.get("models") or fields.get("model")) or "",
        "datatype": _text(fields.get("datatype")) or "func",
    }


def _table_template_values(
    actions: Sequence[Any],
    *,
    extraction_set: ExtractionSet,
    roi_set: RoiSet | None,
    entities: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        **entities,
        "extraction_set": extraction_set.name,
        "roi_set": roi_set.name if roi_set is not None else extraction_set.roi_set or "",
    }


def _format_desc(template: Any, values: Mapping[str, Any]) -> str:
    text = str(template)
    names = {field_name for _, field_name, _, _ in Formatter().parse(text) if field_name}
    return text.format(**{name: values.get(name, "") for name in names})


def _safe_camel_alias(value: str) -> str:
    _reject_unsafe_filename_component(value)
    words = re.findall(r"[A-Za-z0-9]+", str(value))
    alias = "".join(word[:1].upper() + word[1:] for word in words)
    return _bids_label(alias or "Contrast")


def _bids_label(value: str) -> str:
    _reject_unsafe_filename_component(value)
    label = "".join(character for character in str(value) if character.isalnum())
    if not label:
        raise ValueError(f"Cannot derive a BIDS-like label from {value!r}.")
    return label


def _reject_unsafe_filename_component(value: Any) -> None:
    if published_text_contains_local_path_reference(str(value)):
        raise RoiPublicationError(
            "ROI publication rejected an output filename component: configured label contains a local path reference."
        )


def _render_validate_and_promote_publication(
    files: Sequence[_PublicationFile],
    *,
    root: Path,
    settings: Mapping[str, Any],
    context: Any,
) -> None:
    ordered = tuple(sorted(files, key=lambda entry: _publication_relative_path(entry.destination, root).as_posix()))
    duplicate_destinations = _duplicate_paths(entry.destination for entry in ordered)
    if duplicate_destinations:
        raise RoiPublicationError("ROI publication produced duplicate destination paths before rendering.")
    for entry in ordered:
        _preflight_destination(entry.destination, root=root)

    staging_parent = _nearest_existing_directory(root.parent)
    transaction_root = Path(tempfile.mkdtemp(prefix=f".{root.name}.publication-", dir=staging_parent))
    try:
        candidate_root = transaction_root / "candidate"
        for entry in ordered:
            _render_publication_file(entry, candidate_root=candidate_root, root=root, context=context)
        _validate_staged_publication(ordered, candidate_root=candidate_root, root=root)
        _promote_publication_transaction(
            ordered,
            candidate_root=candidate_root,
            backup_root=transaction_root / "backup",
            root=root,
            replace_existing=_text(settings.get("existing_output")) == "replace",
        )
    except _RoiPublicationRecoveryError:
        raise
    except Exception:
        _cleanup_transaction_root(transaction_root)
        raise
    if not _cleanup_transaction_root(transaction_root):
        raise RoiPublicationError(
            "ROI publication completed, but its temporary staging directory could not be removed."
        )


def _render_publication_file(
    entry: _PublicationFile,
    *,
    candidate_root: Path,
    root: Path,
    context: Any,
) -> None:
    relative = _publication_relative_path(entry.destination, root)
    target = candidate_root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    modes = sum(
        value is not None
        for value in (entry.source, entry.json_payload, entry.text, entry.table_source)
    )
    if modes != 1:
        raise RoiPublicationError(
            f"ROI publication could not render output '{relative.as_posix()}': invalid internal output specification."
        )
    try:
        if entry.source is not None:
            _stage_source_file(entry.source, target)
        elif entry.json_payload is not None:
            target.write_text(
                json.dumps(_json_ready(entry.json_payload), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
                newline="\n",
            )
        elif entry.text is not None:
            target.write_text(entry.text, encoding="utf-8", newline="\n")
        elif entry.table_source is not None:
            _write_portable_table(
                entry.table_source,
                target,
                output_name=relative.as_posix(),
                publication_root=root,
                context=context,
            )
    except RoiPublicationError:
        raise
    except Exception as exc:
        raise RoiPublicationError(
            f"ROI publication could not render output '{relative.as_posix()}' ({entry.category})."
        ) from exc


def _stage_source_file(source: Path, target: Path) -> None:
    shutil.copy2(source, target)


def _validate_staged_publication(
    files: Sequence[_PublicationFile],
    *,
    candidate_root: Path,
    root: Path,
) -> None:
    for entry in files:
        relative = _publication_relative_path(entry.destination, root)
        relative_text = relative.as_posix()
        if published_text_contains_local_path_reference(relative_text):
            raise RoiPublicationError(
                f"ROI publication rejected output '{relative_text}': output filename contains a local path reference."
            )
        staged = candidate_root / relative
        suffix = staged.suffix.casefold()
        if suffix == ".json":
            try:
                payload = json.loads(staged.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise RoiPublicationError(
                    f"ROI publication could not validate output '{relative_text}' as JSON."
                ) from exc
            fields = published_value_local_path_fields(payload, label="JSON")
            if fields:
                category = "JSON mapping key" if "<mapping-key:" in fields[0] else "JSON value"
                raise RoiPublicationError(
                    f"ROI publication rejected output '{relative_text}': {category} contains a local path reference."
                )
        elif suffix in {".tsv", ".csv"}:
            delimiter = "\t" if suffix == ".tsv" else ","
            try:
                with staged.open(newline="", encoding="utf-8") as handle:
                    reader = csv.DictReader(handle, delimiter=delimiter)
                    table_payload = {
                        "columns": list(reader.fieldnames or ()),
                        "rows": [dict(row) for row in reader],
                    }
            except (OSError, UnicodeError, csv.Error) as exc:
                raise RoiPublicationError(
                    f"ROI publication could not validate output '{relative_text}' as a table."
                ) from exc
            fields = published_value_local_path_fields(table_payload, label="table")
            if fields:
                category = "table header" if "columns" in fields[0] or "<mapping-key:" in fields[0] else "table cell"
                raise RoiPublicationError(
                    f"ROI publication rejected output '{relative_text}': {category} contains a local path reference."
                )
        elif suffix in {".md", ".txt"}:
            try:
                text = staged.read_text(encoding="utf-8")
            except (OSError, UnicodeError) as exc:
                raise RoiPublicationError(
                    f"ROI publication could not validate output '{relative_text}' as text."
                ) from exc
            if published_text_contains_local_path_reference(text):
                raise RoiPublicationError(
                    f"ROI publication rejected output '{relative_text}': Markdown content contains a local path reference."
                )


def _promote_publication_transaction(
    files: Sequence[_PublicationFile],
    *,
    candidate_root: Path,
    backup_root: Path,
    root: Path,
    replace_existing: bool,
) -> None:
    existing = [entry for entry in files if entry.destination.exists() or entry.destination.is_symlink()]
    if existing and not replace_existing:
        relative = _publication_relative_path(existing[0].destination, root).as_posix()
        raise RoiPublicationError(
            f"ROI publication refused existing output '{relative}'; set publication.existing_output to replace to authorize replacement."
        )

    backed_up: list[tuple[Path, Path]] = []
    promoted: list[Path] = []
    created_directories: list[Path] = []
    current_relative = "publication output"
    try:
        for entry in existing:
            relative = _publication_relative_path(entry.destination, root)
            current_relative = relative.as_posix()
            backup = backup_root / relative
            backup.parent.mkdir(parents=True, exist_ok=True)
            _stage_existing_backup(entry.destination, backup)
            backed_up.append((entry.destination, backup))

        for entry in files:
            relative = _publication_relative_path(entry.destination, root)
            current_relative = relative.as_posix()
            _ensure_parent_directories(entry.destination.parent, created_directories)
            os.replace(candidate_root / relative, entry.destination)
            promoted.append(entry.destination)
    except Exception as exc:
        try:
            _rollback_publication_transaction(
                promoted=promoted,
                backed_up=backed_up,
                created_directories=created_directories,
            )
        except _RoiPublicationRecoveryError as rollback_exc:
            raise _RoiPublicationRecoveryError(
                f"ROI publication failed while promoting output '{current_relative}'; rollback could not be completed and recovery staging was retained."
            ) from rollback_exc
        raise RoiPublicationError(
            f"ROI publication failed while promoting output '{current_relative}'; the prior destination set was restored."
        ) from exc


def _rollback_publication_transaction(
    *,
    promoted: Sequence[Path],
    backed_up: Sequence[tuple[Path, Path]],
    created_directories: Sequence[Path],
) -> None:
    rollback_failed = False
    backed_up_destinations = {destination for destination, _backup in backed_up}
    promoted_destinations = set(promoted)
    for destination in reversed(promoted):
        if destination in backed_up_destinations:
            continue
        rollback_failed = not _retry_filesystem_operation(
            lambda destination=destination: _remove_promoted_destination(destination)
        ) or rollback_failed
    for destination, backup in reversed(backed_up):
        if destination not in promoted_destinations:
            continue
        if not backup.exists() and not backup.is_symlink():
            continue
        rollback_failed = not _retry_filesystem_operation(
            lambda destination=destination, backup=backup: _restore_publication_backup(
                destination,
                backup,
            )
        ) or rollback_failed
    for directory in sorted(set(created_directories), key=lambda path: len(path.parts), reverse=True):
        if not directory.exists():
            continue
        rollback_failed = not _retry_filesystem_operation(directory.rmdir) or rollback_failed
    if rollback_failed:
        raise _RoiPublicationRecoveryError("ROI publication rollback could not restore the complete destination set.")


def _remove_promoted_destination(destination: Path) -> None:
    if destination.is_file() or destination.is_symlink():
        destination.unlink()


def _stage_existing_backup(source: Path, backup: Path) -> None:
    if source.is_symlink():
        os.symlink(os.readlink(source), backup)
        return
    try:
        os.link(source, backup)
    except OSError:
        shutil.copy2(source, backup)


def _restore_publication_backup(destination: Path, backup: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.replace(backup, destination)
        return
    except OSError:
        pass
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    if backup.is_symlink():
        os.symlink(os.readlink(backup), destination)
        backup.unlink()
        return
    shutil.copy2(backup, destination)
    backup.unlink()


def _retry_filesystem_operation(operation: Any) -> bool:
    for _attempt in range(2):
        try:
            operation()
        except Exception:
            continue
        return True
    return False


def _cleanup_transaction_root(transaction_root: Path) -> bool:
    return _retry_filesystem_operation(lambda: shutil.rmtree(transaction_root))


def _ensure_parent_directories(path: Path, created: list[Path]) -> None:
    missing: list[Path] = []
    current = path
    while not current.exists() and current != current.parent:
        missing.append(current)
        current = current.parent
    for directory in reversed(missing):
        directory.mkdir()
        created.append(directory)


def _nearest_existing_directory(path: Path) -> Path:
    current = path
    while not current.is_dir() and current != current.parent:
        current = current.parent
    if not current.is_dir():
        raise RoiPublicationError("ROI publication could not locate a staging filesystem.")
    return current


def _preflight_destination(destination: Path, *, root: Path) -> None:
    relative = _publication_relative_path(destination, root)
    current = root
    for part in relative.parts[:-1]:
        current /= part
        if current.is_symlink():
            raise RoiPublicationError(
                f"ROI publication rejected output '{relative.as_posix()}': destination parent is a symbolic link."
            )
        if current.exists() and not current.is_dir():
            raise RoiPublicationError(
                f"ROI publication rejected output '{relative.as_posix()}': destination parent is not a directory."
            )
    if destination.exists() and not destination.is_file() and not destination.is_symlink():
        raise RoiPublicationError(
            f"ROI publication rejected output '{relative.as_posix()}': conflicting destination is not a replaceable file."
        )


def _publication_relative_path(destination: Path, root: Path) -> Path:
    try:
        relative = Path(destination).relative_to(root)
    except ValueError as exc:
        raise RoiPublicationError("ROI publication produced a destination outside its configured root.") from exc
    if not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
        raise RoiPublicationError("ROI publication produced an invalid destination beneath its configured root.")
    return relative


def _public_table_columns(path: Path) -> list[str]:
    if not path.is_file():
        return []
    delimiter = "\t" if path.suffix.casefold() == ".tsv" else ","
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter=delimiter)
        return [column for column in next(reader, []) if column not in _PRIVATE_RUNTIME_TABLE_COLUMNS]


def _write_portable_table(
    source: Path,
    target: Path,
    *,
    output_name: str,
    publication_root: Path,
    context: Any,
) -> None:
    delimiter = "\t" if source.suffix.casefold() == ".tsv" else ","
    with source.open(newline="", encoding="utf-8") as input_handle:
        reader = csv.DictReader(input_handle, delimiter=delimiter)
        if reader.fieldnames is None:
            raise RoiPublicationError(f"ROI publication could not render output '{output_name}': table has no header.")
        fieldnames = [column for column in reader.fieldnames if column not in _PRIVATE_RUNTIME_TABLE_COLUMNS]
        with target.open("w", newline="", encoding="utf-8") as output_handle:
            writer = csv.DictWriter(output_handle, fieldnames=fieldnames, delimiter=delimiter, lineterminator="\n")
            writer.writeheader()
            for row in reader:
                if None in row:
                    raise RoiPublicationError(
                        f"ROI publication could not render output '{output_name}': table row has unexpected fields."
                    )
                public_row: dict[str, Any] = {}
                for column in fieldnames:
                    value = row.get(column, "")
                    if column in _PORTABLE_INPUT_TABLE_COLUMNS and value:
                        value = _portable_source_reference(
                            value,
                            publication_root=publication_root,
                            context=context,
                            output_name=output_name,
                            category=f"{column} table cell",
                        )
                    public_row[column] = value
                writer.writerow(public_row)


def _portable_source_references(
    values: Iterable[str | Path],
    *,
    publication_root: Path,
    context: Any,
    output_name: str,
    category: str,
) -> list[str]:
    return [
        _portable_source_reference(
            value,
            publication_root=publication_root,
            context=context,
            output_name=output_name,
            category=category,
        )
        for value in values
    ]


def _portable_source_value(
    value: Any,
    *,
    publication_root: Path,
    context: Any,
    output_name: str,
    category: str,
) -> Any:
    if isinstance(value, Mapping):
        output: dict[Any, Any] = {}
        for key, child in value.items():
            if isinstance(key, (str, Path)) and published_text_contains_local_path_reference(str(key)):
                raise RoiPublicationError(
                    f"ROI publication rejected output '{output_name}': {category} mapping key contains a local path reference."
                )
            output[key] = _portable_source_value(
                child,
                publication_root=publication_root,
                context=context,
                output_name=output_name,
                category=category,
            )
        return output
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [
            _portable_source_value(
                child,
                publication_root=publication_root,
                context=context,
                output_name=output_name,
                category=category,
            )
            for child in value
        ]
    if isinstance(value, (str, Path)) and published_text_contains_local_path_reference(str(value)):
        return _portable_source_reference(
            value,
            publication_root=publication_root,
            context=context,
            output_name=output_name,
            category=category,
        )
    return value


def _portable_source_reference(
    value: str | Path,
    *,
    publication_root: Path,
    context: Any,
    output_name: str,
    category: str,
) -> str:
    try:
        reference = portable_path_reference(
            value,
            dataset_root=publication_root,
            named_roots=getattr(context, "root_refs", {}),
        )
    except UnmappedLocalPathError as exc:
        raise RoiPublicationError(
            f"ROI publication rejected output '{output_name}': {category} cannot be expressed beneath a configured public root."
        ) from exc
    if published_text_contains_local_path_reference(reference):
        raise RoiPublicationError(
            f"ROI publication rejected output '{output_name}': {category} contains a local path reference."
        )
    return reference


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"Warnings": ["Could not parse runtime JSON sidecar."]}
    return dict(payload) if isinstance(payload, Mapping) else {}


def _table_sidecar_path(path: Path) -> Path:
    return path.with_suffix(".json")


def _is_qc_table(path: Path) -> bool:
    stem = path.stem.lower()
    return stem.endswith("_qc") or "_qc_" in stem or stem.endswith("qc")


def _has_loso_publication_actions(actions: Sequence[Any]) -> bool:
    return any(getattr(action, "family", None) == "loso_group_map" for action in actions)


def _action_loso_job(action: Any) -> Mapping[str, Any] | None:
    job = _mapping(_mapping(getattr(action, "metadata", {})).get("loso_group_job"))
    return job or None


def _source_paths_for_job(job: Mapping[str, Any]) -> list[str]:
    sources: list[str] = []
    if job.get("group_mask_path") is not None:
        sources.append(str(job["group_mask_path"]))
    for item in _sequence(job.get("training_inputs")):
        if not isinstance(item, Mapping):
            continue
        for key in ("cope_path", "varcope_path", "mask_path"):
            if item.get(key) is not None:
                sources.append(str(item[key]))
    heldout = job.get("heldout_input")
    if isinstance(heldout, Mapping) and heldout.get("mask_path") is not None:
        sources.append(str(heldout["mask_path"]))
    return _unique_text(sources)


def _sequence(value: Any) -> Sequence[Any]:
    return value if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) else ()


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _bool_value(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _strip_prefix(value: str | None, prefix: str) -> str | None:
    text = _text(value)
    if text is None:
        return None
    marker = f"{prefix}-"
    return text[len(marker) :] if text.startswith(marker) else text


def _subject_label(value: str) -> str:
    return f"sub-{_strip_prefix(value, 'sub')}"


def _maybe_int(value: Any) -> int | str | None:
    text = _text(value)
    if text is None:
        return None
    try:
        return int(text)
    except ValueError:
        return text


def _format_number(value: Any) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    text = _text(value)
    if text is None:
        return ""
    if re.fullmatch(r"-?\d+\.0+", text):
        return text.split(".", 1)[0]
    return text


def _first_list_text(value: Any) -> str | None:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return _text(value[0]) if value else None
    return _text(value)


def _unique_paths(paths: Iterable[Path]) -> list[Path]:
    seen: set[Path] = set()
    output: list[Path] = []
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        output.append(path)
    return output


def _duplicate_paths(paths: Iterable[Path]) -> tuple[Path, ...]:
    seen: set[Path] = set()
    duplicates: list[Path] = []
    for path in paths:
        resolved = Path(path)
        if resolved in seen and resolved not in duplicates:
            duplicates.append(resolved)
        seen.add(resolved)
    return tuple(duplicates)


def _record_destinations(
    paths: Iterable[Path],
    *,
    owner: str,
    destination_owners: dict[Path, str],
    duplicate_destinations: list[Path],
) -> bool:
    unique = True
    for path in paths:
        destination = Path(path)
        if destination in destination_owners:
            if destination not in duplicate_destinations:
                duplicate_destinations.append(destination)
            unique = False
            continue
        destination_owners[destination] = owner
    return unique


def _unique_text(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output


def _compact(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {key: _json_ready(value) for key, value in payload.items() if value not in (None, "", [], {})}


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items() if item is not None}
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    return value
