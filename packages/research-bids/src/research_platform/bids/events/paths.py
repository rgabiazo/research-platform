from __future__ import annotations

from pathlib import Path

from research_platform.neuro.events import BuildSpec
from research_platform.bids.layout import build_bids_events_relpath


def build_staged_output_path(
    *,
    artifact_root: str | Path,
    spec: BuildSpec,
    subject: str,
    session: str | None,
    run: int,
    acq_label: str | None = None,
    dir_label: str | None = None,
) -> Path:
    relpath = build_bids_events_relpath(
        subject=subject,
        session=session,
        task=spec.task,
        run=f"{run:02d}",
        acq_label=acq_label if acq_label is not None else spec.acq_label,
        dir_label=dir_label if dir_label is not None else spec.dir_label,
        datatype=spec.datatype,
        suffix=spec.suffix,
    )
    return Path(artifact_root) / "staged" / relpath
