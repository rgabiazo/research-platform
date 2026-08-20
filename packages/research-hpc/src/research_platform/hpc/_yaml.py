"""Small YAML reader/writer for run manifest and status files."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any
import os
import re

_SAFE_SCALAR = re.compile(r"^[A-Za-z0-9_./${}-]+$")
_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def load_yaml(path: str | Path) -> Any:
    return parse_yaml(Path(path).read_text(encoding="utf-8"))


def write_yaml(path: str | Path, data: Any) -> Path:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(dump_yaml(data), encoding="utf-8")
    return file_path


def parse_yaml(text: str) -> Any:
    lines = text.splitlines()
    value, index = _parse_block(lines, 0, 0)
    index = _skip_ignored(lines, index)
    if index != len(lines):
        raise ValueError(f"Unexpected trailing YAML content at line {index + 1}.")
    return value if value is not None else {}


def dump_yaml(data: Any) -> str:
    return "\n".join(_dump_yaml_lines(_normalize_data(data), indent=0)) + "\n"


def expand_env_placeholders(data: Any, env: Mapping[str, str] | None = None) -> Any:
    values = dict(os.environ if env is None else env)
    return _expand_env_value(data, values)


def _skip_ignored(lines: list[str], index: int) -> int:
    while index < len(lines):
        stripped = lines[index].strip()
        if not stripped or stripped.startswith("#"):
            index += 1
            continue
        return index
    return index


def _indent_of(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _parse_block(lines: list[str], index: int, indent: int) -> tuple[Any, int]:
    index = _skip_ignored(lines, index)
    if index >= len(lines):
        return None, index
    if _indent_of(lines[index]) < indent:
        return None, index
    if lines[index].strip() == "-" or lines[index].strip().startswith("- "):
        return _parse_list(lines, index, indent)
    return _parse_mapping(lines, index, indent)


def _parse_mapping(lines: list[str], index: int, indent: int) -> tuple[dict[str, Any], int]:
    mapping: dict[str, Any] = {}
    while True:
        index = _skip_ignored(lines, index)
        if index >= len(lines):
            break
        line = lines[index]
        current_indent = _indent_of(line)
        if current_indent < indent:
            break
        if current_indent > indent:
            raise ValueError(f"Unexpected indentation at line {index + 1}.")

        stripped = line.strip()
        if stripped == "-" or stripped.startswith("- "):
            break
        if ":" not in stripped:
            raise ValueError(f"Expected mapping entry at line {index + 1}.")

        key, raw_value = stripped.split(":", 1)
        key = key.strip()
        raw_value = raw_value.strip()
        index += 1
        if raw_value:
            mapping[key] = _parse_scalar(raw_value)
            continue

        next_index = _skip_ignored(lines, index)
        if next_index >= len(lines) or _indent_of(lines[next_index]) <= indent:
            mapping[key] = None
            index = next_index
            continue

        nested, index = _parse_block(lines, index, indent + 2)
        mapping[key] = nested
    return mapping, index


def _parse_list(lines: list[str], index: int, indent: int) -> tuple[list[Any], int]:
    items: list[Any] = []
    while True:
        index = _skip_ignored(lines, index)
        if index >= len(lines):
            break
        line = lines[index]
        current_indent = _indent_of(line)
        if current_indent < indent:
            break
        if current_indent != indent:
            raise ValueError(f"Unexpected list indentation at line {index + 1}.")

        stripped = line.strip()
        if stripped != "-" and not stripped.startswith("- "):
            break

        raw_value = "" if stripped == "-" else stripped[2:].strip()
        index += 1
        if not raw_value:
            nested, index = _parse_block(lines, index, indent + 2)
            items.append(nested)
            continue
        items.append(_parse_scalar(raw_value))
    return items, index


def _parse_scalar(value: str) -> Any:
    if value == "{}":
        return {}
    if value == "[]":
        return []
    if value.startswith('"') and value.endswith('"'):
        return value[1:-1].replace('\\"', '"').replace("\\\\", "\\")
    if value.startswith("'") and value.endswith("'"):
        return value[1:-1]
    lowered = value.lower()
    if lowered in {"null", "~"}:
        return None
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if re.fullmatch(r"-?\d+", value):
        return int(value)
    if re.fullmatch(r"-?\d+\.\d+", value):
        return float(value)
    return value


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


def _expand_env_value(value: Any, env: Mapping[str, str]) -> Any:
    if isinstance(value, str):
        return _expand_env_string(value, env)
    if isinstance(value, list):
        return [_expand_env_value(item, env) for item in value]
    if isinstance(value, dict):
        return {key: _expand_env_value(item, env) for key, item in value.items()}
    return value


def _expand_env_string(value: str, env: Mapping[str, str]) -> str:
    return _expand_env_text(value, env, resolving=())


def _expand_env_text(value: str, env: Mapping[str, str], *, resolving: tuple[str, ...]) -> str:
    """Expand nested ``${NAME}`` and ``${NAME:-default}`` expressions.

    Keeping the variable-resolution stack explicit prevents mutually
    recursive environment values or defaults from alternating forever.
    Malformed expressions are left unchanged so their caller can report the
    unresolved placeholder through the applicable configuration contract.
    """

    rendered: list[str] = []
    cursor = 0
    while cursor < len(value):
        marker = value.find("${", cursor)
        if marker < 0:
            rendered.append(value[cursor:])
            break
        rendered.append(value[cursor:marker])
        parsed = _parse_env_expression(value, marker)
        if parsed is None:
            rendered.append(value[marker:])
            break
        end, name, default = parsed
        if name in resolving:
            chain = " -> ".join((*resolving, name))
            raise ValueError(f"Cyclic environment placeholder expansion detected: {chain}.")
        resolved = env.get(name)
        if resolved not in {None, ""}:
            replacement = _expand_env_text(str(resolved), env, resolving=(*resolving, name))
        elif default is not None:
            replacement = _expand_env_text(default, env, resolving=(*resolving, name))
        else:
            replacement = ""
        rendered.append(replacement)
        cursor = end
    return "".join(rendered)


def _parse_env_expression(value: str, start: int) -> tuple[int, str, str | None] | None:
    """Return the end offset, name, and optional default for one expression."""

    depth = 1
    cursor = start + 2
    while cursor < len(value) and depth:
        if value.startswith("${", cursor):
            depth += 1
            cursor += 2
            continue
        if value[cursor] == "}":
            depth -= 1
            cursor += 1
            if depth == 0:
                break
            continue
        cursor += 1
    if depth:
        return None

    body = value[start + 2 : cursor - 1]
    separator = _find_top_level_default_separator(body)
    if separator is None:
        name = body
        default = None
    else:
        name = body[:separator]
        default = body[separator + 2 :]
    if not _ENV_NAME.fullmatch(name):
        return None
    return cursor, name, default


def _find_top_level_default_separator(body: str) -> int | None:
    depth = 0
    cursor = 0
    while cursor < len(body) - 1:
        if body.startswith("${", cursor):
            depth += 1
            cursor += 2
            continue
        if body[cursor] == "}" and depth:
            depth -= 1
            cursor += 1
            continue
        if depth == 0 and body.startswith(":-", cursor):
            return cursor
        cursor += 1
    return None
