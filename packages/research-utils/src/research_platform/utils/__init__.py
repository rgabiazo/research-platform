"""Stdlib-only helpers shared by the research-platform packages."""

from .fileio import write_text_atomic
from .hashing import file_sha256
from .output_targets import (
    has_parent_traversal,
    reject_duplicate_targets,
    reject_existing_targets,
    resolved_output_root,
)
from .text import non_empty_text, optional_text
from .yaml_writer import dump_yaml, write_yaml

__all__ = [
    "optional_text",
    "non_empty_text",
    "write_text_atomic",
    "file_sha256",
    "resolved_output_root",
    "has_parent_traversal",
    "reject_duplicate_targets",
    "reject_existing_targets",
    "dump_yaml",
    "write_yaml",
]
