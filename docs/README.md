# Documentation

Choose the path that matches what you want to do. The
[capability and support matrix](capabilities.md) shows what the current alpha
supports; the guides below explain how to work within those limits.

## Start here

- [Source-checkout quickstart](onboarding/quickstart.md) — new users can install
  the editable workspace and run the primary synthetic tabular example.
- [Capability and support matrix](capabilities.md) — users and reviewers can
  check what runs locally, what only plans or validates, and what requires an
  external runtime.
- [Local development](onboarding/local-dev.md) — contributors can set up
  development profiles and understand the supported environment boundary.

## Bring your own data

- [Bring your own data](byod.md) — experienced users can connect private
  project overlays and external data roots without placing them in the public
  repository.
- [Add a project](onboarding/add-a-project.md) — users can create a thin project
  overlay and supply their own configuration and inputs.
- [Data lifecycle](architecture/data-lifecycle.md) — users and reviewers can
  see how canonical datasets, reusable derivatives, and run-specific artifacts
  are separated.
- [Configuration conventions](conventions/configuration.md) — experienced
  users can review workspace, project, dataset, compute, and analysis
  configuration rules.

## Local examples

- [Tabular workflow](tabular-slice.md) — run the checked-in synthetic
  preprocessing, modeling, and evaluation path and review its transaction
  guarantees.
- [ROI workflows](roi-workflows.md) — build synthetic coordinate-sphere ROIs
  and extract values locally, then review the requirements for advanced paths.
- [MVPA and crossnobis](mvpa-crossnobis.md) — run the checked-in
  materialized-pattern example and understand which image-backed and export
  paths remain external or experimental.
- [BIDS events builder](bids-events-builder.md) — construct deterministic toy
  event tables without implying an imaging or preprocessing workflow.

## BIDS and neuroimaging

- [BIDS events builder](bids-events-builder.md) — users working with task
  events can review discovery, mapping, validation, and publication behavior.
- [BIDS analysis and FEAT](bids-analysis-slice.md) — experienced users can
  inspect first-level analysis configuration and external-runtime requirements.
- [ROI workflows](roi-workflows.md) — users can distinguish the checked-in local
  synthetic examples from FSL-, ANTs-, imaging-, and HPC-backed paths.
- [MVPA and crossnobis](mvpa-crossnobis.md) — users can distinguish prepared
  vectors from image-backed execution, RDM export, and publication workflows.
- [Materialized pattern table v1](materialized-pattern-table-v1.md) —
  implementers can follow the normative producer contract for prepared-vector
  inputs.

## HPC planning and troubleshooting

- [BIDS and HPC planning](bids-hpc-slice.md) — experienced users can configure,
  preview, and review plan-only operations and the requirements and limitations
  of experimental execute paths.
- [HPC troubleshooting](how-to/hpc-troubleshooting.md) — operators can
  distinguish local validation from commands that contact a remote host.
- [DeepPrep on SLURM](how-to/run-deepprep-on-slurm.md) — operators can review
  the external data, software, container, and scheduler requirements.
- [First-level FEAT on SLURM](how-to/run-feat-first-level-on-slurm.md) —
  operators can review FEAT planning and the requirements for external
  execution.

These guides do not claim validation on a live cluster. The complete
transactional HPC/fMRIPost lifecycle remains under active development and
validation.

## Architecture and decisions

- [Architecture overview](../ARCHITECTURE.md) — architecture reviewers can see
  component ownership, dependency direction, and the data/output boundaries.
- [Architecture reference](architecture/README.md) — reviewers can navigate
  deeper notes on directory boundaries and data lifecycle.
- [Conventions](conventions/README.md) — contributors can find naming,
  configuration, data-management, provenance, and notebook conventions.
- [Architecture Decision Records](decisions/README.md) — reviewers and
  maintainers can inspect the decisions behind structural and safety
  boundaries.

## Contributing and testing

- [Contributing guide](../CONTRIBUTING.md) — contributors can review
  repository boundaries, privacy expectations, change requirements, and the
  pull-request checklist.
- [Instructions for contributors and automated coding tools](../AGENTS.md) —
  contributors can review the repository's placement and implementation rules.
- [Coding-tool contribution workflow](onboarding/coding-agent-workflow.md) —
  contributors using automated coding tools can follow the same focused review
  and validation process.
- [Continuous integration](../ops/ci/README.md) — contributors and maintainers
  can inspect package-test separation, public contracts, archive checks, and
  clean-checkout verification.
- [Add a package](onboarding/add-a-package.md), [add a pipeline](onboarding/add-a-pipeline.md),
  and [add a project](onboarding/add-a-project.md) — contributors can place new
  work in the correct layer.

## Capabilities and current limitations

- [Capability and support matrix](capabilities.md) — this lists the current
  classification of **Runnable locally**, **Plan/validation only**,
  **Experimental or external-runtime**, and **Scaffold only** behavior.
- [Bring your own data](byod.md) — this explains where synthetic verification
  ends and user data, scientific choices, or external infrastructure begin.

## Project and release information

- [Roadmap](../ROADMAP.md) — users, contributors, and maintainers can review
  future direction without treating it as current support or a release promise.
- [Changelog](../CHANGELOG.md) — users and maintainers can review completed
  changes and the current unreleased section.
- [Citation metadata](../CITATION.cff) — users can find the preferred software
  citation details.
- [MIT License](../LICENSE) — users and contributors can review reuse terms.
- [Security policy](../SECURITY.md) — users can find the current
  security-reporting guidance.
