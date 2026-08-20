"""Derivative-first run discovery and flat BIDS filter helpers for fMRIPost-AROMA."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

_ENTITY_PREFIXES = {
    "sub": "subject_id",
    "ses": "session_id",
    "task": "task_id",
    "run": "run_id",
    "acq": "acquisition",
    "dir": "direction",
}
_BATCH_SELECTOR_KEYS = ("subject_id", "session_id", "task_id", "run_id")
_STRICT_BOLD_SUFFIX = "_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz"
_FILTER_ENTITY_MAP = (
    ("session_id", "session"),
    ("task_id", "task"),
    ("run_id", "run"),
    ("acquisition", "acquisition"),
    ("direction", "direction"),
)
_SELECTOR_PREFIXES = {
    "subject_id": "sub",
    "session_id": "ses",
    "task_id": "task",
    "run_id": "run",
}


@dataclass(frozen=True)
class DerivativeRun:
    subject_id: str
    session_id: str | None
    task_id: str | None
    run_id: str | None
    acquisition: str | None
    direction: str | None
    source_file: str

    def to_dict(self) -> dict[str, str | None]:
        return {
            "subject_id": self.subject_id,
            "session_id": self.session_id,
            "task_id": self.task_id,
            "run_id": self.run_id,
            "acquisition": self.acquisition,
            "direction": self.direction,
            "source_file": self.source_file,
        }


def normalize_entity_label(value: str | None, prefix: str | None = None) -> str | None:
    if value is None:
        return None
    label = str(value).strip()
    if not label:
        return None
    if prefix and label.startswith(f"{prefix}-"):
        return label.split("-", 1)[1]
    return label


def discover_derivative_runs(
    derivative_root: str | Path,
    *,
    subject_id: str,
    session_id: str | None = None,
    task_id: str | None = None,
    run_id: str | None = None,
) -> list[DerivativeRun]:
    root = Path(derivative_root)
    if not root.exists():
        return []

    subject_label = normalize_entity_label(subject_id, "sub")
    session_label = normalize_entity_label(session_id, "ses")
    task_label = normalize_entity_label(task_id, "task")
    run_label = normalize_entity_label(run_id, "run")
    discovered: dict[tuple[str, str | None, str | None, str | None, str | None, str | None], DerivativeRun] = {}

    for path in root.rglob("*"):
        if not _is_derivative_bold_candidate(path):
            continue
        entities = _parse_bids_entities(path.name)
        if entities.get("subject_id") != subject_label:
            continue
        if session_label is not None and entities.get("session_id") != session_label:
            continue
        if task_label is not None and entities.get("task_id") != task_label:
            continue
        if run_label is not None and entities.get("run_id") != run_label:
            continue
        run = DerivativeRun(
            subject_id=subject_label or "",
            session_id=entities.get("session_id"),
            task_id=entities.get("task_id"),
            run_id=entities.get("run_id"),
            acquisition=entities.get("acquisition"),
            direction=entities.get("direction"),
            source_file=str(path.resolve()),
        )
        key = (run.subject_id, run.session_id, run.task_id, run.run_id, run.acquisition, run.direction)
        discovered[key] = run

    return sorted(
        discovered.values(),
        key=lambda run: (
            run.subject_id,
            run.session_id or "",
            run.task_id or "",
            run.acquisition or "",
            run.direction or "",
            run.run_id or "",
            run.source_file,
        ),
    )


def discover_batch_rows(
    derivative_root: str | Path,
    *,
    selectors: dict[str, str | None] | None = None,
) -> list[dict[str, str]]:
    root = Path(derivative_root)
    if not root.exists():
        return []

    normalized = {
        "subject_id": normalize_entity_label((selectors or {}).get("subject_id"), "sub"),
        "session_id": normalize_entity_label((selectors or {}).get("session_id"), "ses"),
        "task_id": normalize_entity_label((selectors or {}).get("task_id"), "task"),
        "run_id": normalize_entity_label((selectors or {}).get("run_id"), "run"),
    }
    discovered: dict[tuple[str, str, str, str], dict[str, str]] = {}
    for path in root.rglob(f"*{_STRICT_BOLD_SUFFIX}"):
        if not path.is_file():
            continue
        entities = _parse_bids_entities(path.name)
        row = {
            "subject_id": _format_entity("sub", entities.get("subject_id")),
            "session_id": _format_entity("ses", entities.get("session_id")),
            "task_id": _format_entity("task", entities.get("task_id")),
            "run_id": _format_entity("run", entities.get("run_id")),
        }
        if not row["subject_id"]:
            continue
        if not _matches_selectors(row, normalized):
            continue
        key = tuple(row.get(name, "") for name in _BATCH_SELECTOR_KEYS)
        discovered[key] = {name: row.get(name, "") for name in _BATCH_SELECTOR_KEYS if row.get(name, "")}
    return sorted(
        discovered.values(),
        key=lambda row: tuple(row.get(name, "") for name in _BATCH_SELECTOR_KEYS),
    )


def expected_remote_input_files(
    derivative_root: str | Path,
    *,
    remote_derivative_root: str,
    row: Mapping[str, str],
) -> list[str]:
    subject_id = row.get("subject_id", "")
    if not subject_id:
        return []

    local_root = Path(derivative_root).resolve()
    runs = discover_derivative_runs(
        local_root,
        subject_id=subject_id,
        session_id=row.get("session_id"),
        task_id=row.get("task_id"),
        run_id=row.get("run_id"),
    )
    discovered: list[str] = []
    seen: set[str] = set()
    remote_root = Path("/", str(remote_derivative_root).lstrip("/"))
    for run in runs:
        source_path = Path(run.source_file).resolve()
        try:
            relative_path = source_path.relative_to(local_root)
        except ValueError:
            continue
        remote_path = str((remote_root / relative_path).as_posix())
        if remote_path in seen:
            continue
        seen.add(remote_path)
        discovered.append(remote_path)
    return discovered


def build_flat_bids_filter(
    runs: list[DerivativeRun],
    *,
    session_id: str | None = None,
    task_id: str | None = None,
    run_id: str | None = None,
) -> dict[str, list[str]]:
    if not runs and task_id is None and session_id is None and run_id is None:
        raise ValueError("Cannot build a BIDS filter without derivative-backed runs or explicit selectors.")

    filter_payload: dict[str, list[str]] = {}
    session_label = normalize_entity_label(session_id, "ses")
    if session_label is not None:
        filter_payload["session"] = [session_label]

    if task_id is not None:
        task_values = [normalize_entity_label(task_id, "task")]
    else:
        task_values = sorted({run.task_id for run in runs if run.task_id})
    if task_values and task_values[0] is not None:
        filter_payload["task"] = [value for value in task_values if value is not None]

    explicit_run_label = normalize_entity_label(run_id, "run")
    if explicit_run_label is not None:
        filter_payload["run"] = [explicit_run_label]
    else:
        run_values = sorted({run.run_id for run in runs if run.run_id})
        if run_values:
            filter_payload["run"] = run_values

    for entity_name, filter_key in _FILTER_ENTITY_MAP:
        if filter_key in {"session", "task", "run"}:
            continue
        values = sorted({getattr(run, entity_name) for run in runs if getattr(run, entity_name)})
        if values:
            filter_payload[filter_key] = values

    if not filter_payload:
        raise ValueError("No flat BIDS entities were available for the filter payload.")
    return filter_payload


def group_runtime_plan_runs(
    runs: Sequence[DerivativeRun],
    *,
    runtime_grouping: str,
) -> list[list[DerivativeRun]]:
    if runtime_grouping == "compatible":
        return group_flat_filter_compatible_runs(runs)
    if runtime_grouping == "row":
        discovered = {_derivative_run_identity(run): run for run in runs}
        return [[run] for run in sorted(discovered.values(), key=_derivative_run_sort_key)]
    raise ValueError(f"Unsupported runtime grouping: {runtime_grouping}")


def group_flat_filter_compatible_runs(runs: Sequence[DerivativeRun]) -> list[list[DerivativeRun]]:
    grouped: dict[tuple[str, str | None, str | None, str | None, str | None], dict[tuple[str, ...], DerivativeRun]] = {}
    for run in runs:
        grouped.setdefault(_flat_filter_group_key(run), {})[_derivative_run_identity(run)] = run

    compatible_groups: list[list[DerivativeRun]] = []
    for key in sorted(grouped):
        grouped_runs = sorted(grouped[key].values(), key=_derivative_run_sort_key)
        if len(grouped_runs) > 1 and any(run.run_id is None for run in grouped_runs):
            compatible_groups.extend([[run] for run in grouped_runs])
            continue
        compatible_groups.append(grouped_runs)
    return compatible_groups


def _is_derivative_bold_candidate(path: Path) -> bool:
    return path.is_file() and path.name.endswith(_STRICT_BOLD_SUFFIX)


def _flat_filter_group_key(run: DerivativeRun) -> tuple[str, str | None, str | None, str | None, str | None]:
    return (
        run.subject_id,
        run.session_id,
        run.task_id,
        run.acquisition,
        run.direction,
    )


def _derivative_run_identity(run: DerivativeRun) -> tuple[str, ...]:
    return (
        run.subject_id,
        run.session_id or "",
        run.task_id or "",
        run.run_id or "",
        run.acquisition or "",
        run.direction or "",
        run.source_file,
    )


def _derivative_run_sort_key(run: DerivativeRun) -> tuple[str, ...]:
    return (
        run.subject_id,
        run.session_id or "",
        run.task_id or "",
        run.acquisition or "",
        run.direction or "",
        run.run_id or "",
        run.source_file,
    )


def _strip_known_extensions(name: str) -> str:
    if name.endswith(".nii.gz"):
        return name[:-7]
    return Path(name).stem


def _parse_bids_entities(name: str) -> dict[str, str]:
    entities: dict[str, str] = {}
    for token in _strip_known_extensions(name).split("_"):
        key, separator, value = token.partition("-")
        if separator != "-" or key not in _ENTITY_PREFIXES or not value:
            continue
        entities[_ENTITY_PREFIXES[key]] = value
    return entities


def _format_entity(prefix: str, value: str | None) -> str:
    if value is None:
        return ""
    return f"{prefix}-{value}"


def _matches_selectors(row: dict[str, str], selectors: dict[str, str | None]) -> bool:
    for key in _BATCH_SELECTOR_KEYS:
        expected = selectors.get(key)
        if expected is None:
            continue
        actual = normalize_entity_label(row.get(key), _SELECTOR_PREFIXES[key])
        if actual != expected:
            return False
    return True
