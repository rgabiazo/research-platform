"""Config-driven cleanup helpers for LOSO FLAME1 ROI runtime outputs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
import shutil

from research_platform.neuro.roi import parse_extraction_set_document, parse_roi_set_document


def cleanup_after_loso_roi_build(
    document: Mapping[str, Any],
    *,
    context: Any,
    publication_complete: bool,
    publication_root: Path | None,
) -> tuple[dict[str, Any], ...]:
    """Apply ROI-set cleanup after successful LOSO map/mask publication."""

    roi_set = parse_roi_set_document(document, validate_personal_paths=False)
    policy = _cleanup_policy(roi_set.fields, "after_roi_build")
    if policy is None:
        return ()
    return (
        _cleanup_summary(
            scope="roi_set",
            phase="after_roi_build",
            owner=roi_set.name,
            policy=policy,
            runtime_root=_resolve_runtime_root(roi_set.fields, context=context),
            publication_complete=publication_complete,
            publication_root=publication_root,
            targets=_roi_targets(roi_set.name, policy),
        ),
    )


def cleanup_after_loso_featquery_extraction(
    extraction_document: Mapping[str, Any],
    *,
    roi_set_document: Mapping[str, Any] | None,
    context: Any,
    publication_complete: bool,
    publication_root: Path | None,
) -> tuple[dict[str, Any], ...]:
    """Apply extraction-set and referenced ROI-set cleanup after table publication."""

    extraction_set = parse_extraction_set_document(extraction_document, validate_personal_paths=False)
    summaries: list[dict[str, Any]] = []
    extraction_policy = _cleanup_policy(extraction_set.fields, "after_extraction")
    if extraction_policy is not None:
        summaries.append(
            _cleanup_summary(
                scope="extraction_set",
                phase="after_extraction",
                owner=extraction_set.name,
                policy=extraction_policy,
                runtime_root=_resolve_runtime_root(extraction_set.fields, context=context),
                publication_complete=publication_complete,
                publication_root=publication_root,
                targets=_extraction_targets(extraction_set.name, extraction_policy),
            )
        )

    if roi_set_document is not None:
        roi_set = parse_roi_set_document(roi_set_document, validate_personal_paths=False)
        roi_policy = _cleanup_policy(roi_set.fields, "after_extraction")
        if roi_policy is not None:
            summaries.append(
                _cleanup_summary(
                    scope="roi_set",
                    phase="after_extraction",
                    owner=roi_set.name,
                    policy=roi_policy,
                    runtime_root=_resolve_runtime_root(roi_set.fields, context=context),
                    publication_complete=publication_complete,
                    publication_root=publication_root,
                    targets=_roi_targets(roi_set.name, roi_policy),
                )
            )
    return tuple(summaries)


def _cleanup_summary(
    *,
    scope: str,
    phase: str,
    owner: str,
    policy: str,
    runtime_root: Path,
    publication_complete: bool,
    publication_root: Path | None,
    targets: Sequence[tuple[str, Path]],
) -> dict[str, Any]:
    runtime_root = Path(runtime_root).resolve()
    summary: dict[str, Any] = {
        "scope": scope,
        "phase": phase,
        "owner": owner,
        "policy": policy,
        "publication_complete": publication_complete,
        "runtime_root": str(runtime_root),
        "targets": [],
    }
    if policy == "none":
        summary["status"] = "skipped"
        summary["reason"] = "cleanup_disabled"
        return summary
    if not publication_complete:
        summary["status"] = "skipped"
        summary["reason"] = "publication_incomplete"
        summary["targets"] = [
            _target_payload(kind, (runtime_root / path).resolve(), status="skipped", reason="publication_incomplete")
            for kind, path in targets
        ]
        return summary

    statuses: list[str] = []
    for kind, relative_path in targets:
        target = (runtime_root / relative_path).resolve()
        if not _is_safe_cleanup_target(target, runtime_root=runtime_root, publication_root=publication_root):
            payload = _target_payload(kind, target, status="skipped", reason="unsafe_target")
        elif not target.exists():
            payload = _target_payload(kind, target, status="missing", reason="target_missing")
        elif not target.is_dir():
            payload = _target_payload(kind, target, status="skipped", reason="target_not_directory")
        else:
            shutil.rmtree(target)
            payload = _target_payload(kind, target, status="removed")
        statuses.append(str(payload["status"]))
        summary["targets"].append(payload)

    summary["status"] = "completed" if any(status == "removed" for status in statuses) else "skipped"
    return summary


def _cleanup_policy(fields: Mapping[str, Any], phase: str) -> str | None:
    runtime = fields.get("runtime")
    if not isinstance(runtime, Mapping):
        return None
    cleanup = runtime.get("cleanup")
    if not isinstance(cleanup, Mapping) or phase not in cleanup:
        return None
    value = str(cleanup.get(phase) or "none").strip()
    return value or "none"


def _roi_targets(roi_set_name: str, policy: str) -> tuple[tuple[str, Path], ...]:
    owner = _workflow_segment(roi_set_name)
    if owner is None:
        return (("unsafe_owner", Path("..")),)
    cache = ("loso_groupmap_cache", Path(".cache") / "loso_groupmaps" / owner)
    if policy == "cache_only":
        return (cache,)
    if policy == "roi_runtime":
        return (
            cache,
            ("loso_groupmaps", Path("loso_groupmaps") / owner),
            ("rois", Path("rois") / owner),
        )
    return ()


def _extraction_targets(extraction_set_name: str, policy: str) -> tuple[tuple[str, Path], ...]:
    owner = _workflow_segment(extraction_set_name)
    if owner is None:
        return (("unsafe_owner", Path("..")),)
    if policy == "extraction_runtime":
        return (("roi_extract", Path("roi_extract") / owner),)
    return ()


def _workflow_segment(value: str) -> str | None:
    text = str(value).strip()
    if text in {"", ".", ".."} or "/" in text or "\\" in text:
        return None
    return text


def _resolve_runtime_root(fields: Mapping[str, Any], *, context: Any) -> Path:
    outputs = fields.get("outputs")
    if isinstance(outputs, Mapping):
        root_value = outputs.get("derivative_root") or outputs.get("root") or outputs.get("output_root")
        if root_value is not None:
            return _resolve_path_spec(root_value, context=context, label="outputs.root")
        if outputs.get("root_ref") is not None:
            root = Path(context.resolve_root_ref(str(outputs["root_ref"]))).resolve()
            subpath = _optional_text(outputs.get("path") or outputs.get("subpath"))
            return (root / subpath).resolve() if subpath else root
        if outputs.get("path") is not None:
            return _resolve_path_spec(outputs["path"], context=context, label="outputs.path")
    for key in ("output_root", "derivative_root"):
        if fields.get(key) is not None:
            return _resolve_path_spec(fields[key], context=context, label=key)
    project = getattr(context, "project_name", None) or "project"
    return Path(context.artifacts_root).resolve() / "roi" / project / "derivatives"


def _resolve_path_spec(spec: Any, *, context: Any, label: str) -> Path:
    if isinstance(spec, (str, Path)):
        path = Path(str(spec)).expanduser()
        return path.resolve() if path.is_absolute() else (Path(context.project_root) / path).resolve()
    if not isinstance(spec, Mapping):
        raise ValueError(f"{label} must be a path string or mapping.")
    root_ref = _optional_text(spec.get("root_ref"))
    base = Path(context.resolve_root_ref(root_ref)).resolve() if root_ref is not None else Path(context.project_root).resolve()
    raw = spec.get("pattern") if spec.get("pattern") is not None else spec.get("path")
    if raw is None:
        return base
    path = Path(str(raw)).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _is_safe_cleanup_target(target: Path, *, runtime_root: Path, publication_root: Path | None) -> bool:
    if target == runtime_root:
        return False
    if not _is_relative_to(target, runtime_root):
        return False
    if publication_root is not None:
        published = Path(publication_root).resolve()
        if target == published or _is_relative_to(target, published) or _is_relative_to(published, target):
            return False
    return True


def _is_relative_to(path: Path, base: Path) -> bool:
    try:
        path.relative_to(base)
    except ValueError:
        return False
    return True


def _target_payload(kind: str, path: Path, *, status: str, reason: str | None = None) -> dict[str, Any]:
    payload = {"kind": kind, "path": str(path), "status": status}
    if reason is not None:
        payload["reason"] = reason
    return payload


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
