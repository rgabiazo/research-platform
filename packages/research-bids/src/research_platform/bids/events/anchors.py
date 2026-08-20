from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from research_platform.neuro.events import BuildSpec
from research_platform.bids.entities import parse_bids_name


@dataclass(frozen=True)
class AnchorMatch:
    path: Path
    subject: str
    session: str | None
    task: str
    run: str | None
    acq: str | None
    dir_label: str | None


@dataclass(frozen=True)
class ResolvedBidsContext:
    subject: str
    session: str | None
    task: str
    run: str
    acq: str | None
    dir_label: str | None
    anchor_path: Path | None


def _normalize_run(value: str | int) -> str:
    return f"{int(value):02d}"


def _normalize_session_entity(value: str | None) -> str | None:
    if value is None:
        return None
    text = value.strip().removeprefix("ses-")
    if not text:
        raise ValueError("Session entity must not be empty.")
    return f"ses-{text}"


def _anchor_json_path(path: Path) -> Path:
    return path.with_suffix("").with_suffix(".json")


def _ped_dir_label(path: Path, spec: BuildSpec) -> str | None:
    payload = json.loads(_anchor_json_path(path).read_text(encoding="utf-8"))
    ped = str(payload.get("PhaseEncodingDirection", "")).strip()
    if not ped:
        raise ValueError(f"Anchor {path} is missing PhaseEncodingDirection for PED fallback.")
    if ped not in spec.ped_dir_map:
        raise ValueError(f"Anchor {path} has unmapped PhaseEncodingDirection {ped!r}.")
    return spec.ped_dir_map[ped]


def _discover_bold_anchors(dataset_root: str | Path) -> list[AnchorMatch]:
    matches: list[AnchorMatch] = []
    for pattern in ("*_bold.nii.gz", "*_bold.nii"):
        for path in sorted(Path(dataset_root).rglob(pattern)):
            entities, suffix = parse_bids_name(path.name)
            if suffix != "bold":
                continue
            matches.append(
                AnchorMatch(
                    path=path,
                    subject=entities["sub"],
                    session=entities.get("ses"),
                    task=entities.get("task", ""),
                    run=entities.get("run"),
                    acq=entities.get("acq"),
                    dir_label=entities.get("dir"),
                )
            )
    return matches


def resolve_bold_anchor(
    *,
    dataset_root: str | Path,
    spec: BuildSpec,
    subject: str,
    session: str | None,
    run: int,
) -> ResolvedBidsContext:
    subject_entity = subject if subject.startswith("sub-") else f"sub-{subject}"
    session_entity = _normalize_session_entity(session if session is None or session.startswith("ses-") else f"ses-{session}")
    run_entity = _normalize_run(run)

    candidates = []
    for anchor in _discover_bold_anchors(dataset_root):
        if anchor.subject != subject_entity:
            continue
        if anchor.task != spec.task:
            continue
        if anchor.run is not None and _normalize_run(anchor.run) != run_entity:
            continue
        if session_entity is not None and _normalize_session_entity(anchor.session) != session_entity:
            continue
        if spec.acq_label and anchor.acq != spec.acq_label:
            continue
        if spec.dir_label and anchor.dir_label != spec.dir_label:
            continue
        candidates.append(anchor)

    if not candidates:
        raise ValueError(
            f"No BOLD anchor matched subject={subject_entity} session={session_entity or '<none>'} "
            f"task={spec.task} run={run_entity}."
        )
    if len(candidates) != 1:
        candidate_list = ", ".join(str(anchor.path) for anchor in candidates)
        raise ValueError(
            f"Ambiguous BOLD anchor for subject={subject_entity} session={session_entity or '<none>'} "
            f"task={spec.task} run={run_entity}: {candidate_list}"
        )

    anchor = candidates[0]
    dir_label = spec.dir_label or anchor.dir_label
    if dir_label is None and spec.ped_fallback_enabled:
        dir_label = _ped_dir_label(anchor.path, spec)

    return ResolvedBidsContext(
        subject=anchor.subject.removeprefix("sub-"),
        session=None if anchor.session is None else anchor.session.removeprefix("ses-"),
        task=spec.task,
        run=run_entity,
        acq=spec.acq_label or anchor.acq,
        dir_label=dir_label,
        anchor_path=anchor.path,
    )
