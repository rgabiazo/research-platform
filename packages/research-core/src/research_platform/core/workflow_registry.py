"""Beginner-facing workflow registry for guided onboarding."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WorkflowEntry:
    name: str
    description: str
    subtypes: tuple[str, ...]
    required_packages: tuple[str, ...]
    scaffold_handler: str
    smoke_guidance: str
    normal_guidance: str


WORKFLOWS: tuple[WorkflowEntry, ...] = (
    WorkflowEntry(
        name="preprocess",
        description="Prepare raw or derivative data for downstream analysis.",
        subtypes=("bids",),
        required_packages=("research-core", "research-hpc", "research-neuro"),
        scaffold_handler="project init bids-preprocess",
        smoke_guidance="Discover one subject into a smoke batch, dry-run sync, verify, then submit.",
        normal_guidance="Reuse the same project with named batches and run ids.",
    ),
    WorkflowEntry(
        name="analysis",
        description="Run scientific or statistical analyses.",
        subtypes=("bids", "tabular", "longitudinal", "custom"),
        required_packages=("research-core", "research-hpc", "research-analysis", "research-neuro"),
        scaffold_handler="project init bids-analysis or config/analysis spec",
        smoke_guidance="Start with one BIDS subject or a small tabular batch.",
        normal_guidance="Keep analysis specs in project config and run named analyses.",
    ),
    WorkflowEntry(
        name="tabular",
        description="Prepare feature tables, splits, train, and evaluate models.",
        subtypes=("model",),
        required_packages=("research-core", "research-hpc", "research-analysis", "research-ml"),
        scaffold_handler="project init tabular-model",
        smoke_guidance="Use a small feature table batch before scaling.",
        normal_guidance="Reuse canonical feature roots and run ids for train/evaluate cycles.",
    ),
    WorkflowEntry(
        name="notebook",
        description="Scaffold notebook directories for a user-supplied notebook and runtime.",
        subtypes=("hpc", "local"),
        required_packages=("research-core", "research-hpc"),
        scaffold_handler="minimal notebook overlay",
        smoke_guidance="Add a notebook and runtime before planning local or remote execution.",
        normal_guidance="Use project-aware notebook start or submit helpers.",
    ),
    WorkflowEntry(
        name="custom",
        description="Create a minimal project overlay with declared data roots.",
        subtypes=("generic",),
        required_packages=("research-core",),
        scaffold_handler="minimal custom overlay",
        smoke_guidance="Validate config and run hpc doctor before adding workflow-specific commands.",
        normal_guidance="Add reusable logic to packages and keep project glue thin.",
    ),
)


def workflow_names() -> tuple[str, ...]:
    return tuple(entry.name for entry in WORKFLOWS)


def get_workflow(name: str) -> WorkflowEntry:
    normalized = str(name).strip()
    for entry in WORKFLOWS:
        if entry.name == normalized:
            return entry
    allowed = ", ".join(workflow_names())
    raise ValueError(f"Unknown workflow {name!r}. Available workflows: {allowed}")


def render_workflow_menu() -> list[str]:
    width = max(len(entry.name) for entry in WORKFLOWS)
    lines = ["Available workflows:"]
    lines.extend(f"  {entry.name.ljust(width)}   {entry.description}" for entry in WORKFLOWS)
    return lines
