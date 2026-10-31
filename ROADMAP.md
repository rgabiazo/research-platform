# Roadmap

This roadmap describes project direction rather than current support, and it
makes no calendar commitment. The [capability matrix](docs/capabilities.md)
records what works today, the [changelog](CHANGELOG.md) records completed
changes, and the [accepted decision records](docs/decisions/README.md) document
reviewed architectural and safety decisions.

## Current status

Research Platform is an unreleased source-checkout alpha at version `0.1.0a1`.
Its packages are not published on PyPI. SLURM is the only scheduler and rsync
over SSH the only transfer tool; neither has live-cluster validation, and there
is no mode for working directly on a cluster. Bounded synthetic examples run
locally; remote execution and real-data neuroimaging workflows stay within the
limitations in the capability matrix.

## Completed foundations

- Eight Python packages share one workspace, with `rp` as the integrated CLI
  and package-level interfaces for programmatic use.
- Deterministic synthetic examples cover the tabular workflow, BIDS event
  construction, the documented local ROI examples, and materialized-pattern
  crossnobis from prepared synthetic vectors. The crossnobis example stops at
  an RDM-ready pairwise-distance table, not an exported RDM.
- The local tabular workflow uses reviewed planning, explicit execution,
  provenance, one-shot run identity, and collision-safe publication.
- Canonical inputs and reusable derivatives stay in `datasets/`; run-scoped
  plans, outputs, and recovery evidence stay in `artifacts/`.
- Public templates and synthetic overlays are separate from private study
  configuration and data.

## Now

- One run command that runs locally or submits on a cluster, chosen by the
  compute profile. Today `rp run local` and `rp run submit` are separate
  commands that read the same project config.
- A mode for working directly on the cluster, with scheduler-submitted jobs
  and no local workstation driving SSH and rsync.
- Remote bootstrap that installs the platform on the cluster and verifies the
  installation with a smoke job, and can be rerun without side effects.
- A one-line rationale next to each statistical and processing default in the
  example configurations.
- Provider-neutral remote safety and lifecycle foundations, completed before
  any remote support claim changes.

## Planned next

Listing an item here does not change a current capability classification.

- Installation from reviewed, checksummed release artifacts.
- One deterministic synthetic remote lifecycle on one documented SLURM
  environment: reviewed planning, staging, duplicate-safe submission, status
  reconciliation, cancellation, receipts, verified retrieval, and no-overwrite
  publication ([ADR-0022](docs/decisions/ADR-0022-headline-hpc-execution-contract.md),
  [ADR-0023](docs/decisions/ADR-0023-hpc-safety-primitives.md)).
- A versioned adapter boundary through which a workflow declares inputs,
  commands, resources, outputs, provenance, and success criteria.
- fMRIPost-AROMA as a later, separately reviewed target on that lifecycle.
  fMRIPost remains Experimental or external-runtime until then.

## After the first public alpha

- Artifact installation and update workflows.
- More checked-in synthetic examples.
- Package-level Python interface documentation and a compatibility policy.
- ROI percent signal change under a separately reviewed acceptance contract,
  including independent numerical validation of the Featquery-backed path.
- Bounded neuroimaging adapters, after the generic remote lifecycle is
  accepted.
- Provenance, reporting, diagnostics, and recovery guidance.

## Longer-term possibilities

Exploration areas include shared application services, a stable high-level
Python API, HTTP services, dashboards or other graphical clients, additional
schedulers and execution backends, cloud storage or execution, additional
scientific adapters, and broader reporting and visualization. These are
possibilities, not committed work or current support. Any such client would
reuse application services and package-owned behaviour rather than
reimplementing scientific, BIDS, or HPC logic, and the CLI would remain a
supported automation interface.

## Explicitly deferred or unsupported scope

The current roadmap does not promise:

- arbitrary pipelines, arbitrary clusters, or arbitrary shell execution;
- multiple supported schedulers, cloud backends, or generic backend
  interchangeability;
- automatic overwrite, force replacement, foreign-state adoption, automatic
  stale cleanup, or unattended MFA;
- DeepPrep execution in the first remote alpha or a complete raw-BIDS
  preprocessing chain;
- FEAT, Featquery percent signal change, advanced ROI, or image-backed MVPA/RSA
  as part of the synthetic remote acceptance target; or
- a stable unified SDK, HTTP API, dashboard, Streamlit application, desktop
  client, or other graphical product in the current alpha.

## How roadmap items become supported capabilities

A roadmap item becomes supported only after the evidence for that capability
has been reviewed: an owning implementation with focused tests; a checked-in
synthetic example where one applies; documented prerequisites and limits; CI,
fake-remote, or separately authorized live acceptance where applicable;
privacy, scientific, and licensing review; and a capability-matrix update.
Source presence, a command name, a rendered plan, mocked execution, one
successful external run, or documentation wording is not sufficient by itself
to establish support.
