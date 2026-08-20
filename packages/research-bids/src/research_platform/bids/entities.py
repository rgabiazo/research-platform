
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

BIDS_ENTITY_ORDER = ("sub", "ses", "task", "acq", "ce", "dir", "rec", "run")


def _normalized_entity_value(key: str, value: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"Entity {key!r} must not be empty.")
    if key == "sub" and not text.startswith("sub-"):
        return f"sub-{text}"
    if key == "ses" and not text.startswith("ses-"):
        return f"ses-{text}"
    return text


def ordered_bids_entities(entities: Mapping[str, str | None]) -> list[tuple[str, str]]:
    ordered: list[tuple[str, str]] = []
    for key in BIDS_ENTITY_ORDER:
        value = entities.get(key)
        if value in (None, ""):
            continue
        ordered.append((key, _normalized_entity_value(key, str(value))))
    return ordered


def build_bids_name(entities: Mapping[str, str | None], suffix: str) -> str:
    parts = [f"{key}-{value}" if key not in {"sub", "ses"} else value for key, value in ordered_bids_entities(entities)]
    parts.append(suffix)
    return "_".join(parts)


def parse_bids_name(path: str | Path) -> tuple[dict[str, str], str]:
    name = Path(path).name
    if name.endswith(".nii.gz"):
        stem = name[:-7]
    else:
        stem = Path(name).stem
    tokens = stem.split("_")
    if not tokens:
        raise ValueError(f"Unable to parse BIDS entities from {name!r}.")
    suffix = tokens[-1]
    entities: dict[str, str] = {}
    for token in tokens[:-1]:
        key, sep, value = token.partition("-")
        if not sep or not value:
            raise ValueError(f"Unable to parse BIDS entity token {token!r} from {name!r}.")
        entities[key] = value if key not in {"sub", "ses"} else token
    return entities, suffix
