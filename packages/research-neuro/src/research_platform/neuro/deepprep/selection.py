"""Raw-BIDS discovery helpers for DeepPrep preprocessing."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Mapping

_BATCH_SELECTOR_KEYS = ("subject_id", "session_id", "task_id", "run_id")
_ENTITY_PREFIXES = {
    "sub": "subject_id",
    "ses": "session_id",
    "task": "task_id",
    "run": "run_id",
}
_SELECTOR_PREFIXES = {
    "subject_id": "sub",
    "session_id": "ses",
    "task_id": "task",
    "run_id": "run",
}
_IGNORED_RAW_ROOT_PARTS = {"derivatives", "sourcedata"}


def normalize_entity_label(value: str | None, prefix: str | None = None) -> str | None:
    if value is None:
        return None
    label = str(value).strip()
    if not label:
        return None
    if prefix and label.startswith(f"{prefix}-"):
        return label.split("-", 1)[1]
    return label


def discover_batch_rows(
    bids_root: str | Path,
    *,
    selectors: Mapping[str, str | None] | None = None,
) -> list[dict[str, str]]:
    root = Path(bids_root)
    if not root.exists():
        return []

    normalized_selectors = _normalize_selectors(selectors or {})
    subjects = _discover_subjects(root)
    discovered: dict[tuple[str, str, str, str], dict[str, str]] = {}
    subjects_with_func_matches: set[str] = set()

    for subject_id in subjects:
        if normalized_selectors["subject_id"] is not None and subject_id != normalized_selectors["subject_id"]:
            continue
        subject_dir = root / f"sub-{subject_id}"
        for bold_path in _iter_raw_bold_files(subject_dir):
            entities = _parse_bids_entities(bold_path.name)
            if entities.get("subject_id") != subject_id:
                continue
            row = {
                "subject_id": _format_entity("sub", subject_id),
                "session_id": _format_entity("ses", entities.get("session_id")),
                "task_id": _format_entity("task", entities.get("task_id")),
                "run_id": _format_entity("run", entities.get("run_id")),
            }
            if not _matches_selectors(row, normalized_selectors):
                continue
            subjects_with_func_matches.add(subject_id)
            key = tuple(row.get(name, "") for name in _BATCH_SELECTOR_KEYS)
            discovered[key] = {name: row.get(name, "") for name in _BATCH_SELECTOR_KEYS if row.get(name, "")}

        if subject_id not in subjects_with_func_matches and _subject_only_row_allowed(normalized_selectors):
            row = {"subject_id": _format_entity("sub", subject_id)}
            if _matches_selectors(row, normalized_selectors):
                discovered[(row["subject_id"], "", "", "")] = row

    return sorted(
        discovered.values(),
        key=lambda row: tuple(row.get(name, "") for name in _BATCH_SELECTOR_KEYS),
    )


def expected_remote_input_files(
    bids_root: str | Path,
    *,
    remote_bids_root: str,
    row: Mapping[str, str],
) -> list[str]:
    subject_id = normalize_entity_label(row.get("subject_id"), "sub")
    if not subject_id or not str(remote_bids_root).strip():
        return []

    local_root = Path(bids_root).resolve()
    remote_root = Path("/", str(remote_bids_root).lstrip("/"))
    subject_dir = local_root / f"sub-{subject_id}"
    expected: list[str] = []

    for root_file in ("dataset_description.json", "participants.tsv"):
        path = local_root / root_file
        if path.exists():
            expected.append(str((remote_root / root_file).as_posix()))

    if subject_dir.exists():
        expected.append(str((remote_root / subject_dir.relative_to(local_root)).as_posix()))

    for bold_path in _matching_raw_bold_files(
        subject_dir,
        subject_id=subject_id,
        session_id=row.get("session_id"),
        task_id=row.get("task_id"),
        run_id=row.get("run_id"),
    ):
        expected.append(_remote_path(local_root=local_root, remote_root=remote_root, local_path=bold_path))
        sidecar = _sidecar_json_path(bold_path)
        if sidecar.exists():
            expected.append(_remote_path(local_root=local_root, remote_root=remote_root, local_path=sidecar))

    for t1w_path in _iter_raw_t1w_files(subject_dir):
        expected.append(_remote_path(local_root=local_root, remote_root=remote_root, local_path=t1w_path))
        sidecar = _sidecar_json_path(t1w_path)
        if sidecar.exists():
            expected.append(_remote_path(local_root=local_root, remote_root=remote_root, local_path=sidecar))

    for fmap_path in _matching_raw_fmap_files(subject_dir, session_id=row.get("session_id")):
        expected.append(_remote_path(local_root=local_root, remote_root=remote_root, local_path=fmap_path))

    return _dedupe(expected)


def _discover_subjects(root: Path) -> list[str]:
    subjects: list[str] = []
    participants_path = root / "participants.tsv"
    if participants_path.exists():
        with participants_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                subject_id = normalize_entity_label(row.get("participant_id"), "sub")
                if subject_id:
                    subjects.append(subject_id)

    for path in root.iterdir():
        if path.is_dir() and path.name.startswith("sub-"):
            subject_id = normalize_entity_label(path.name, "sub")
            if subject_id:
                subjects.append(subject_id)

    return _dedupe(subjects)


def _iter_raw_bold_files(subject_dir: Path) -> list[Path]:
    if not subject_dir.exists():
        return []
    return sorted(
        path
        for path in subject_dir.rglob("*_bold.nii*")
        if path.is_file()
        and (path.name.endswith("_bold.nii") or path.name.endswith("_bold.nii.gz"))
        and not (_IGNORED_RAW_ROOT_PARTS & set(path.parts))
    )


def _iter_raw_t1w_files(subject_dir: Path) -> list[Path]:
    if not subject_dir.exists():
        return []
    return sorted(
        path
        for path in subject_dir.rglob("*_T1w.nii*")
        if path.is_file()
        and (path.name.endswith("_T1w.nii") or path.name.endswith("_T1w.nii.gz"))
        and not (_IGNORED_RAW_ROOT_PARTS & set(path.parts))
    )


def _matching_raw_fmap_files(subject_dir: Path, *, session_id: str | None) -> list[Path]:
    if not subject_dir.exists():
        return []
    fmap_dirs: list[Path] = []
    subject_level_fmap = subject_dir / "fmap"
    if subject_level_fmap.exists():
        fmap_dirs.append(subject_level_fmap)
    session_label = normalize_entity_label(session_id, "ses")
    if session_label:
        session_fmap = subject_dir / f"ses-{session_label}" / "fmap"
        if session_fmap.exists():
            fmap_dirs.append(session_fmap)
    else:
        fmap_dirs.extend(path for path in subject_dir.glob("ses-*/fmap") if path.is_dir())
    return sorted(
        path
        for fmap_dir in fmap_dirs
        for path in fmap_dir.rglob("*")
        if path.is_file() and not (_IGNORED_RAW_ROOT_PARTS & set(path.parts))
    )


def _matching_raw_bold_files(
    subject_dir: Path,
    *,
    subject_id: str,
    session_id: str | None,
    task_id: str | None,
    run_id: str | None,
) -> list[Path]:
    session_label = normalize_entity_label(session_id, "ses")
    task_label = normalize_entity_label(task_id, "task")
    run_label = normalize_entity_label(run_id, "run")
    matches: list[Path] = []
    for path in _iter_raw_bold_files(subject_dir):
        entities = _parse_bids_entities(path.name)
        if entities.get("subject_id") != subject_id:
            continue
        if session_label is not None and entities.get("session_id") != session_label:
            continue
        if task_label is not None and entities.get("task_id") != task_label:
            continue
        if run_label is not None and entities.get("run_id") != run_label:
            continue
        matches.append(path)
    return sorted(matches)


def _normalize_selectors(selectors: Mapping[str, str | None]) -> dict[str, str | None]:
    return {
        key: normalize_entity_label(selectors.get(key), prefix)
        for key, prefix in _SELECTOR_PREFIXES.items()
    }


def _matches_selectors(row: Mapping[str, str], normalized_selectors: Mapping[str, str | None]) -> bool:
    for key, expected in normalized_selectors.items():
        if expected is None:
            continue
        prefix = _SELECTOR_PREFIXES[key]
        actual = normalize_entity_label(row.get(key), prefix)
        if actual != expected:
            return False
    return True


def _subject_only_row_allowed(normalized_selectors: Mapping[str, str | None]) -> bool:
    return all(normalized_selectors.get(key) is None for key in ("session_id", "task_id", "run_id"))


def _parse_bids_entities(name: str) -> dict[str, str]:
    entities: dict[str, str] = {}
    for token in _strip_known_extensions(name).split("_"):
        key, separator, value = token.partition("-")
        if separator != "-" or key not in _ENTITY_PREFIXES or not value:
            continue
        entities[_ENTITY_PREFIXES[key]] = value
    return entities


def _strip_known_extensions(name: str) -> str:
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return Path(name).stem


def _format_entity(prefix: str, value: str | None) -> str:
    label = normalize_entity_label(value, prefix)
    return f"{prefix}-{label}" if label else ""


def _sidecar_json_path(path: Path) -> Path:
    if path.name.endswith(".nii.gz"):
        return path.with_name(path.name[:-7] + ".json")
    return path.with_suffix(".json")


def _remote_path(*, local_root: Path, remote_root: Path, local_path: Path) -> str:
    return str((remote_root / local_path.resolve().relative_to(local_root)).as_posix())


def _dedupe(values: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = str(value).strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped
