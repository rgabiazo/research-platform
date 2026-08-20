# Architecture

Research Platform is a CLI-first, multi-package Python research software
platform for reproducible tabular and BIDS-oriented workflows. This
source-checkout alpha keeps reusable code, orchestration, project configuration,
canonical data, and run-specific outputs in one coordinated workspace with
explicit ownership boundaries.

## Purpose

The architecture separates reusable software from project-specific scientific
choices, local infrastructure, and generated outputs. That separation makes it
possible to reuse methods across projects without embedding one dataset,
machine, or execution environment in shared code.

This document explains the current structural boundaries and intended dependency
direction. It does not expand support beyond the evidence recorded in the
[capability and support matrix](docs/capabilities.md).

## System at a glance

The source checkout is one coordinated workspace. Its components have clear,
separately owned responsibilities, but they are directories in the current
repository rather than separate repositories.

| Path | Role |
| --- | --- |
| `packages/` | Installable Python packages containing reusable behaviour and package-level interfaces |
| `pipelines/` | Workflow orchestration that combines package behaviour with workflow engines, external tools, containers, and execution profiles |
| `project/` | Thin project overlays containing reviewed configuration, manifests, notebooks, reports, and light integration code |
| `datasets/` | Approved canonical inputs and reusable derivatives |
| `artifacts/` | Run-scoped, noncanonical plans, outputs, diagnostics, and recovery evidence |
| `ops/` | Development environments, CI, containers, synchronization, scheduler support, and operational guidance |
| `secrets/` | Ignored local credentials and connection settings that must never be tracked |
| `scratch/` | Disposable local working space that is not part of a reproducible result |
| `apps/` | **Scaffold only** extension points, not supported application runtimes |
| `docs/` | User guidance, capability status, architecture, decisions, and contributor documentation |

Package code provides reusable behaviour. Pipelines orchestrate that behaviour.
Project overlays select data and scientific configuration. Datasets and artifacts
record different stages of the data and output lifecycle.

## CLI and programmatic use

`rp` is the primary integrated user interface. It coordinates project setup,
configuration validation, planning, and the bounded workflows listed in the
capability matrix. Package-specific commands remain available for advanced or
specialized work.

The underlying packages contain importable modules and package-level interfaces.
Those interfaces have package-specific maturity and compatibility expectations;
together they are not one frozen unified SDK, and no stable high-level platform
SDK is currently claimed.

No HTTP API currently exists. The repository does not provide a supported
dashboard, Streamlit application, desktop client, or other graphical product.
Future interfaces would compose shared application services and package
behaviour rather than replacing the CLI or duplicating scientific, BIDS, or HPC
logic.

## Component ownership

| Package | Owns | Does not own or imply |
| --- | --- | --- |
| `research-core` | Workspace and project configuration, generic path and reference resolution, orchestration, reviewed-plan composition, generic run lifecycle concepts, generic provenance and manifests, the top-level `rp` CLI, and thin wrappers around behaviour owned elsewhere | BIDS or neuroimaging semantics, tool-specific scientific validity, or low-level SSH, rsync, SLURM, and remote-safety mechanics |
| `research-io` | Generic tabular reading, writing, inspection, merge, cleaning, format support, and dataframe backend adapters | BIDS layouts, neuroimaging behaviour, or split-aware train/test semantics |
| `research-analysis` | Split manifests and strategies, leakage-safe preprocessing, generic statistics and associations, publication-table handoffs, and protocol-neutral MVPA/RSA mathematics and contracts | Image extraction or project-specific scientific decisions |
| `research-ml` | Estimators, model helpers, metrics, and supported classification and regression behaviour | Protocol-neutral crossnobis and RDM mathematics, which remain analysis concerns |
| `research-bids` | BIDS entities, traversal, discovery, event-table behaviour, portable BIDS paths, and derivative naming semantics | Neuroimaging-tool execution, HPC mechanics, or generic statistical analysis |
| `research-neuro` | Neuroimaging adapters and semantics, including DeepPrep, fMRIPost-AROMA, FEAT, NIfTI utilities, ROI behaviour, image and pattern extraction, and neuroimaging-specific MVPA inputs | Ownership of an adapter does not mean its execution path is currently supported |
| `research-hpc` | The provider-neutral implementation home for target configuration, SSH, rsync, SLURM, remote readiness, safety primitives, claims, receipts, staging, cancellation, retrieval, quarantine, recovery, no-overwrite publication mechanics, and remote lifecycle behaviour where implemented | Ownership does not mean that the complete remote lifecycle is implemented, accepted, or supported |
| `research-viz` | Reusable visualization and reporting specifications, rendering primitives, figures, tables, and report outputs | A checked-in dashboard product or a general project-level reporting workflow |

Package ownership identifies where behaviour belongs. It does not by itself make
that behaviour a supported public workflow.

## Dependency direction

The architecture uses stable ownership rules rather than treating the workspace
as one unrestricted import graph:

- Shared packages must not depend on project overlays or apps.
- Project overlays configure reusable behaviour and remain thin; they do not
  become alternate homes for package logic.
- Pipelines orchestrate packages and external tools. Scientific and operational
  semantics remain in their owning packages.
- Core may compose and orchestrate another package through generic wrappers, but
  it should not absorb scientific or tool semantics owned by that package.
- Neuro may depend on BIDS-owned entities, portable paths, and naming
  contracts. BIDS must not depend on Neuro-specific types or scientific
  semantics.
- HPC remains independent of BIDS and neuroimaging scientific meaning. It owns
  provider-neutral remote mechanics and safety contracts.
- Generic Analysis contracts remain separate from image and pattern extraction
  owned by Neuro.
- Application or product assembly may compose packages without making every
  package depend on every other package.
- Third-party library objects should not become the platform's public
  compatibility boundary unintentionally. Adapters should translate them into
  package-owned contracts.

These rules describe the intended direction. They do not claim that every
current import already follows an ideal dependency graph. Discrepancies are
handled through focused source changes and reviewed decision records, not hidden
in this overview.

## Projects and configuration

Project overlays hold project-specific configuration, manifests, exact analysis
units, cohort views, workflow selections, model choices, reporting definitions,
and other reviewed scientific decisions. They should be configuration-driven
and contain only light integration code. Reusable validation, transformation,
analysis, or execution behaviour belongs in the owning package.

The public repository contains templates and deterministic synthetic examples.
Real-study overlays may remain private while using the public platform. Private
overlays, participant data, credentials, private infrastructure settings, and
machine-local paths must not be added to the public repository.

Configuration expresses scientific and operational choices explicitly. The CLI
and pipelines resolve that configuration; they should not replace reviewed
project decisions with hidden command defaults.

## Data and output lifecycle

`datasets/` contains approved canonical inputs and reusable derivatives.
`artifacts/` contains run-scoped, noncanonical material. Artifacts can include
plans, logs, diagnostics, generated tables and figures, staging material, and
failure, uncertainty, or recovery evidence. Some artifacts can be regenerated;
others must be retained long enough to explain what happened during a run.

An artifact becomes a canonical derivative only through an explicit,
workflow-specific validation and publication policy. The presence of a file or
a successful command exit does not authorize generic promotion into
`datasets/`.

The [data lifecycle guide](docs/architecture/data-lifecycle.md) explains the
directory-level flow. [ADR-0002](docs/decisions/ADR-0002-canonical-vs-ephemeral-data.md)
records the accepted canonical-versus-run-specific boundary.

## Local and external execution boundaries

The capability matrix uses four exact classifications:

- **Runnable locally**
- **Plan/validation only**
- **Experimental or external-runtime**
- **Scaffold only**

Bounded synthetic examples are runnable from the source checkout, including the
primary tabular workflow, deterministic BIDS event construction, the documented
local ROI examples, and local materialized-pattern crossnobis. These examples
establish only their documented software and scientific boundaries.

The checked-in BIDS pilot, generic SLURM planning, and other planning paths are
**Plan/validation only** where classified. Plan rendering, command presence, or
mocked command-boundary evidence does not prove end-to-end execution.

DeepPrep, fMRIPost-AROMA, FEAT, advanced ROI paths, image-backed MVPA, live
scheduler operations, and other remote actions remain **Experimental or
external-runtime** where classified. They require separate user data, external
software, credentials, containers, infrastructure, and acceptance evidence.
Apps and placeholder pipelines remain **Scaffold only**.

The complete transactional remote lifecycle is under development. It is not a
supported or released public workflow, and no accepted end-to-end live-cluster
validation exists for it. The project makes no arbitrary-cluster or
arbitrary-pipeline claim. Detailed
conditional design and safety decisions live in
[ADR-0022](docs/decisions/ADR-0022-headline-hpc-execution-contract.md) and
[ADR-0023](docs/decisions/ADR-0023-hpc-safety-primitives.md); neither decision
changes the current classifications in the
[capability matrix](docs/capabilities.md).

## Provenance and publication

Provenance and publication guarantees are workflow-specific. The documented
local tabular workflow uses a reviewable plan, explicit execution, provenance,
collision-safe publication, and a one-shot run identity. An unchanged reviewed
plan can make its one permitted transition to execution under the planned run
ID; completed or failed runs are preserved rather than overwritten. The detailed
contracts are recorded in
[ADR-0020](docs/decisions/ADR-0020-tabular-run-identity-and-execution-claims.md)
and
[ADR-0021](docs/decisions/ADR-0021-transactional-tabular-outputs.md).

The local ROI and materialized-pattern MVPA examples have narrower contracts.
They do not automatically inherit the tabular transaction, run-identity,
publication, or recovery model.

Scheduler completion, command exit, file existence, or a log marker is not a
universal proof of scientific success. Canonical publication must be explicit, validated, and provenance-aware, and
must reject overwrite where the owning workflow supports that guarantee. Current local guarantees do not establish complete
remote staging, scheduler reconciliation, receipts, verified retrieval, or
publication.

## Extension points

Possible future extensions include additional adapters, shared application
services, a stable high-level Python API, HTTP services, dashboards or graphical
clients, and additional execution backends. These are architectural directions,
not current support claims.

A future API or graphical client should call shared application services and
package-owned behaviour. It should not reimplement scientific methods, BIDS
rules, or remote-safety logic. The CLI should remain a supported automation
interface rather than being replaced by a graphical client.

The current app directories are **Scaffold only**. No cloud execution backend,
arbitrary backend interchangeability, or arbitrary third-party plugin system is
currently supported.

## Current limitations

- Research Platform is an unreleased source-checkout alpha at version
  `0.1.0a1`; its packages are not currently published on PyPI and it is not
  production-ready.
- Package-level Python interfaces exist, but no unified stable SDK is claimed.
- No HTTP API or supported graphical application currently exists.
- Remote and external-runtime workflows remain bounded by the capability
  matrix and by user- or environment-supplied data, tools, configuration, and
  acceptance evidence.
- Scaffold presence is not implementation or support evidence.
- The project does not claim arbitrary pipeline or arbitrary cluster support.

## Decision records

This architecture document is a stable overview, not the complete decision
history. Accepted [Architecture Decision Records](docs/decisions/README.md)
contain detailed technical, scientific-boundary, and safety decisions. The
[capability matrix](docs/capabilities.md) records present workflow status, and
the [README](README.md) remains the public entry point.

Future direction belongs in the [roadmap](ROADMAP.md) rather than in this
architecture overview. This document therefore avoids implementation-progress
ledgers and historical task checklists.
