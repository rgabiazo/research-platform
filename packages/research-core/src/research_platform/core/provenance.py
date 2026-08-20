
"""Provenance utilities placeholder."""

from dataclasses import dataclass


@dataclass
class ProvenanceRecord:
    code_version: str = "0.0.0"
    config_profile: str = "default"
    execution_backend: str = "local"
