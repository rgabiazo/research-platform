from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


NUMERIC_COLUMNS = ("duration", "response_time")


def _serialize_value(value: Any) -> str:
    if isinstance(value, float):
        return str(value)
    return str(value)


def _write_events_tsv_polars(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows([{column: _serialize_value(row[column]) for column in columns} for row in rows])


def _write_events_tsv_pandas(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "The pandas backend requires pandas to be installed. "
            "Install research-bids with the pandas extra or provide pandas in the active environment."
        ) from exc

    frame = pd.DataFrame(rows, columns=columns)
    # Compatibility shim: established encoding/recognition goldens use a pandas write
    # path that preserves emitted onset strings while normalizing selected numeric
    # columns. Keep that behavior explicit until a more general output-format contract
    # replaces this compatibility mode.
    for column in NUMERIC_COLUMNS:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame.to_csv(path, sep="\t", index=False, na_rep="n/a")


def write_events_tsv(path: str | Path, rows: list[dict[str, Any]], columns: list[str], *, backend: str = "polars") -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if backend == "pandas":
        _write_events_tsv_pandas(output_path, rows, columns)
        return
    _write_events_tsv_polars(output_path, rows, columns)


def write_sidecar_json(path: str | Path, payload: dict[str, Any]) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return output_path
