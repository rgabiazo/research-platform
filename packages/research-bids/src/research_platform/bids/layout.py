
from __future__ import annotations

from pathlib import Path

from .entities import build_bids_name


def build_bids_events_relpath(
    *,
    subject: str,
    session: str | None,
    task: str,
    run: str,
    acq_label: str | None,
    dir_label: str | None,
    datatype: str,
    suffix: str,
) -> Path:
    entities: dict[str, str | None] = {
        "sub": subject,
        "ses": session,
        "task": task,
        "acq": acq_label,
        "dir": dir_label,
        "run": run,
    }
    subject_dir = subject if subject.startswith("sub-") else f"sub-{subject}"
    parts = [subject_dir]
    if session:
        session_dir = session if session.startswith("ses-") else f"ses-{session}"
        parts.append(session_dir)
    parts.append(datatype)
    filename = build_bids_name(entities, suffix)
    return Path(*parts) / f"{filename}.tsv"
