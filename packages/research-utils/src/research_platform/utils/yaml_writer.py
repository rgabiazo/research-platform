"""YAML writer for plain mappings, lists and scalars."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import re

_SAFE_SCALAR = re.compile(r"^[A-Za-z0-9_./${}-]+$")


def write_yaml(path: str | Path, data: Any) -> Path:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(dump_yaml(data), encoding="utf-8")
    return file_path


def dump_yaml(data: Any) -> str:
    return "\n".join(_dump_yaml_lines(_normalize_data(data), indent=0)) + "\n"


def _normalize_data(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _normalize_data(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize_data(item) for item in value]
    return value


def _dump_yaml_lines(value: Any, *, indent: int) -> list[str]:
    prefix = " " * indent
    if isinstance(value, dict):
        if not value:
            return [f"{prefix}{{}}"]
        lines: list[str] = []
        for key, item in value.items():
            if isinstance(item, dict):
                if not item:
                    lines.append(f"{prefix}{key}: {{}}")
                    continue
                lines.append(f"{prefix}{key}:")
                lines.extend(_dump_yaml_lines(item, indent=indent + 2))
                continue
            if isinstance(item, list):
                if not item:
                    lines.append(f"{prefix}{key}: []")
                    continue
                lines.append(f"{prefix}{key}:")
                lines.extend(_dump_yaml_lines(item, indent=indent + 2))
            else:
                lines.append(f"{prefix}{key}: {_format_scalar(item)}")
        return lines
    if isinstance(value, list):
        if not value:
            return [f"{prefix}[]"]
        lines = []
        for item in value:
            if isinstance(item, dict):
                if not item:
                    lines.append(f"{prefix}- {{}}")
                    continue
                lines.append(f"{prefix}-")
                lines.extend(_dump_yaml_lines(item, indent=indent + 2))
                continue
            if isinstance(item, list):
                if not item:
                    lines.append(f"{prefix}- []")
                    continue
                lines.append(f"{prefix}-")
                lines.extend(_dump_yaml_lines(item, indent=indent + 2))
            else:
                lines.append(f"{prefix}- {_format_scalar(item)}")
        return lines
    return [f"{prefix}{_format_scalar(value)}"]


def _format_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value)
    if not text:
        return '""'
    if text.lower() in {"true", "false", "null", "~"}:
        return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'
    if re.fullmatch(r"-?\d+", text) or re.fullmatch(r"-?\d+\.\d+", text):
        return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'
    if _SAFE_SCALAR.fullmatch(text):
        return text
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'
