#!/usr/bin/env python3
"""Generate deterministic, participant-free toy-memory event fixtures."""

from __future__ import annotations

import argparse
import csv
from decimal import Decimal
import io
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_ROOTS = (
    REPO_ROOT / "packages" / "research-bids" / "tests" / "fixtures" / "toy-memory",
    REPO_ROOT / "packages" / "research-neuro" / "tests" / "fixtures" / "toy-memory",
)
SOURCE_NAME = "toy01_visit01_toymemory_2099-01-01.csv"
RUNS = (1, 2, 3)
CONDITIONS = (
    ("shape", "Shape", "response_shape_match", ("match", "different")),
    ("color", "Color", "response_color_tone", ("warm", "cool")),
    ("word", "Word", "response_word_kind", ("concrete", "abstract")),
)
OUTPUT_COLUMNS = (
    "onset",
    "duration",
    "trial_type",
    "stim_file",
    "response_time",
    "response",
    "phase",
    "condition",
    "stim_id",
    "acc_label",
    "probe_type",
    "enc_is_tested",
    "enc_later_outcome",
    "is_instruction",
    "is_error",
    "block_n",
    "trial_n",
    "analysis_include",
)
MISSING = "n/a"


def _source_columns() -> list[str]:
    columns = [
        "toy_id",
        "visit",
        "fixture_date",
        "task_name",
        "run",
        "stimulus_file",
        "image_old_new",
        "response_seen_new",
        "response_shape_match",
        "response_color_tone",
        "response_word_kind",
    ]
    for run in RUNS:
        columns.extend(
            [
                f"toy_run{run}_encoding_onset",
                f"toy_run{run}_encoding_response_time",
                f"toy_run{run}_recognition_onset",
                f"toy_run{run}_recognition_response_time",
            ]
        )
    return columns


def _blank_source_row() -> dict[str, str]:
    row = {column: "" for column in _source_columns()}
    row.update(
        {
            "toy_id": "toy01",
            "visit": "visit01",
            "fixture_date": "2099-01-01",
            "task_name": "ToyMemoryFixture",
        }
    )
    return row


def _onset(run: int, condition_index: int, phase_offset: int, trial_index: int) -> str:
    value = (
        Decimal((run - 1) * 1000)
        + Decimal("20.2500")
        + Decimal(condition_index * 120)
        + Decimal(phase_offset)
        + Decimal((trial_index - 1) * 4)
    )
    return f"{value:.4f}"


def _response_time(phase: str, trial_index: int) -> str:
    base = Decimal("1.2500") if phase == "encoding" else Decimal("1.5000")
    return f"{base + Decimal(trial_index - 1) / Decimal(100):.4f}"


def _source_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for run in RUNS:
        for condition_index, (condition, title, response_column, responses) in enumerate(CONDITIONS):
            encoding_prefix = f"toy_run{run}_{title}_encoding"
            recognition_prefix = f"toy_run{run}_{title}_recognition"

            for item_index in range(1, 13):
                row = _blank_source_row()
                row.update(
                    {
                        "run": str(run),
                        "stimulus_file": (
                            f"{encoding_prefix}/{condition}_run{run}_item{item_index:02d}.svg"
                        ),
                        "image_old_new": "studied",
                        response_column: responses[(item_index - 1) % 2],
                        f"toy_run{run}_encoding_onset": _onset(
                            run, condition_index, 0, item_index
                        ),
                        f"toy_run{run}_encoding_response_time": _response_time(
                            "encoding", item_index
                        ),
                    }
                )
                rows.append(row)

            for trial_index in range(1, 13):
                is_target = trial_index <= 6
                item_kind = "item" if is_target else "lure"
                item_index = trial_index if is_target else trial_index - 6
                response = "seen" if item_index % 2 else "new"
                response_time = _response_time("recognition", trial_index)
                if run == 2 and condition == "color" and item_kind == "lure" and item_index == 6:
                    response = "no_response"
                    response_time = ""

                row = _blank_source_row()
                row.update(
                    {
                        "run": str(run),
                        "stimulus_file": (
                            f"{recognition_prefix}/{condition}_run{run}_{item_kind}{item_index:02d}.svg"
                        ),
                        "image_old_new": "studied" if is_target else "novel",
                        "response_seen_new": response,
                        f"toy_run{run}_recognition_onset": _onset(
                            run, condition_index, 60, trial_index
                        ),
                        f"toy_run{run}_recognition_response_time": response_time,
                    }
                )
                rows.append(row)
    return rows


def _output_row(**values: str) -> dict[str, str]:
    row = {column: MISSING for column in OUTPUT_COLUMNS}
    row.update(values)
    return row


def _instruction_row(
    *, onset: str, condition: str, phase: str, block_n: str, probe_type: str
) -> dict[str, str]:
    return _output_row(
        onset=str(float(onset) - 10.0),
        duration="10.0",
        trial_type=f"instruction_{condition}_{phase}",
        phase=phase,
        condition=condition,
        probe_type=probe_type,
        enc_is_tested="0" if phase == "encoding" else MISSING,
        enc_later_outcome="not_tested" if phase == "encoding" else MISSING,
        is_instruction="1",
        is_error="0",
        block_n=block_n,
        trial_n="1",
        analysis_include="0",
    )


def _expected_rows(run: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for condition_index, (condition, _title, _response_column, responses) in enumerate(CONDITIONS):
        first_encoding_onset = _onset(run, condition_index, 0, 1)
        rows.append(
            _instruction_row(
                onset=first_encoding_onset,
                condition=condition,
                phase="encoding",
                block_n="1",
                probe_type=MISSING,
            )
        )
        for item_index in range(1, 13):
            outcome = "hit" if item_index <= 6 and item_index % 2 else "miss"
            if item_index > 6:
                outcome = "not_tested"
            stim_id = f"{condition}_run{run}_item{item_index:02d}.svg"
            rows.append(
                _output_row(
                    onset=_onset(run, condition_index, 0, item_index),
                    duration="3.0",
                    trial_type=f"encoding_{condition}",
                    stim_file=stim_id,
                    response_time=str(float(_response_time("encoding", item_index))),
                    response=responses[(item_index - 1) % 2],
                    phase="encoding",
                    condition=condition,
                    stim_id=stim_id,
                    enc_is_tested="1" if item_index <= 6 else "0",
                    enc_later_outcome=outcome,
                    is_error="1" if outcome == "miss" else "0",
                    block_n="1",
                    trial_n=str(item_index + 1),
                    analysis_include="0",
                )
            )

        first_recognition_onset = _onset(run, condition_index, 60, 1)
        rows.append(
            _instruction_row(
                onset=first_recognition_onset,
                condition=condition,
                phase="recognition",
                block_n="2",
                probe_type="lure",
            )
        )
        for trial_index in range(1, 13):
            is_target = trial_index <= 6
            item_kind = "item" if is_target else "lure"
            item_index = trial_index if is_target else trial_index - 6
            response = "seen" if item_index % 2 else "new"
            response_time = str(float(_response_time("recognition", trial_index)))
            if run == 2 and condition == "color" and item_kind == "lure" and item_index == 6:
                response = "no_response"
                response_time = MISSING

            probe_type = "target" if is_target else "lure"
            labels = {
                ("target", "seen"): "hit",
                ("target", "new"): "miss",
                ("lure", "seen"): "false_alarm",
                ("lure", "new"): "correct_rejection",
            }
            acc_label = labels.get((probe_type, response), MISSING)
            stim_id = f"{condition}_run{run}_{item_kind}{item_index:02d}.svg"
            rows.append(
                _output_row(
                    onset=_onset(run, condition_index, 60, trial_index),
                    duration="3.0",
                    trial_type=f"recognition_{condition}",
                    stim_file=stim_id,
                    response_time=response_time,
                    response=response,
                    phase="recognition",
                    condition=condition,
                    stim_id=stim_id,
                    acc_label=acc_label,
                    probe_type=probe_type,
                    is_error="1" if acc_label in {"miss", "false_alarm"} else "0",
                    block_n="2",
                    trial_n=str(trial_index + 1),
                    analysis_include="1" if acc_label in {"hit", "correct_rejection"} else "0",
                )
            )
    return rows


def _render_delimited(
    rows: list[dict[str, str]], columns: list[str] | tuple[str, ...], *, delimiter: str
) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer, fieldnames=columns, delimiter=delimiter, lineterminator="\n"
    )
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def _generated_files() -> dict[Path, bytes]:
    source_bytes = _render_delimited(
        _source_rows(), _source_columns(), delimiter=","
    ).encode("utf-8-sig")
    files: dict[Path, bytes] = {}
    for fixture_root in FIXTURE_ROOTS:
        files[fixture_root / "raw" / SOURCE_NAME] = source_bytes
        for run in RUNS:
            output_name = (
                f"sub-toy01_ses-01_task-toymemory_dir-AP_run-{run:02d}_events.tsv"
            )
            files[fixture_root / "expected" / output_name] = _render_delimited(
                _expected_rows(run), OUTPUT_COLUMNS, delimiter="\t"
            ).encode("utf-8")
    return files


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="verify checked-in fixtures without writing"
    )
    args = parser.parse_args()

    mismatches: list[Path] = []
    for path, expected_bytes in _generated_files().items():
        if args.check:
            if not path.is_file() or path.read_bytes() != expected_bytes:
                mismatches.append(path)
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(expected_bytes)

    if mismatches:
        for path in mismatches:
            print(path.relative_to(REPO_ROOT))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
