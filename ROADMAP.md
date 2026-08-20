# Roadmap

This roadmap describes project direction rather than current support, and it
makes no calendar commitment. The [capability matrix](docs/capabilities.md)
records what works today, the [changelog](CHANGELOG.md) records completed
changes, and the [accepted decision records](docs/decisions/README.md) document
reviewed architectural and safety decisions.

## Current status

Research Platform is an unreleased source-checkout alpha at version `0.1.0a1`.
Its packages are not published on PyPI, and it is not production-ready. Bounded
synthetic examples run locally, while remote execution, real-data neuroimaging
workflows, and applications remain within the limitations described by the
capability matrix.

## Completed foundations

- Eight separately owned Python packages share one coordinated workspace, with
  `rp` as the integrated CLI and package-level interfaces for programmatic use.
  Those interfaces do not yet form a unified stable SDK.
- Package ownership, intended dependency direction, project configuration, and
  data and output boundaries are documented explicitly.
- Deterministic synthetic examples cover the primary tabular workflow, BIDS
  event construction, the documented local ROI examples, and
  materialized-pattern crossnobis from prepared synthetic vectors. The
  crossnobis example stops at an RDM-ready pairwise-distance table rather than
  an image workflow or exported RDM.
- The documented local tabular workflow uses reviewed planning, explicit
  execution, provenance, one-shot run identity, and collision-safe publication.
- Canonical inputs and reusable derivatives remain in `datasets/`; run-scoped,
  noncanonical plans, outputs, diagnostics, and recovery evidence remain in
  `artifacts/`.
- CI, synthetic fixtures, capability classifications, and decision records make
  implementation evidence and support boundaries reviewable.
- Public templates and synthetic overlays are kept separate from private
  real-study configuration and data.

## Now

- Polish the source-checkout user experience and public documentation.
- Prepare a curated, sanitized, history-free public repository export with
  synthetic fixtures, CI, and accurate project and package metadata.
- Keep the current local examples deterministic and make their scientific
  limitations explicit.
- Complete provider-neutral remote safety and lifecycle foundations before any
  remote support claim changes.
- Reduce stale, duplicated, or implementation-led documentation while
  preserving accepted decision and capability boundaries.

## Next public-alpha milestones

These are conditional milestones. Listing them here does not change a current
capability classification or promise their inclusion in a particular release.

- Support coordinated installation from reviewed, checksummed release artifacts
  rather than relying only on an editable source checkout.
- Demonstrate one deterministic synthetic remote lifecycle on one documented,
  separately reviewed SLURM environment after fake-remote acceptance and
  separately authorized live-cluster acceptance using the deterministic
  synthetic workload. The acceptance path would
  include reviewed planning, safe staging, duplicate-safe submission, status
  and terminal-state reconciliation, confirmed cancellation, outcome receipts,
  verified retrieval, collision rejection, and explicit no-overwrite
  publication. The detailed lifecycle and safety boundaries are recorded in
  [ADR-0022](docs/decisions/ADR-0022-headline-hpc-execution-contract.md) and
  [ADR-0023](docs/decisions/ADR-0023-hpc-safety-primitives.md).
- Clarify the guided CLI journey so that planning, remote mutation,
  cancellation, retrieval, verification, and publication remain visibly
  distinct operations.
- Design and review a versioned adapter boundary through which a supported
  scientific workflow can declare inputs, commands, resources, expected
  outputs, provenance, and success criteria without duplicating generic HPC
  mechanics.
- Consider a narrow fMRIPost-AROMA integration only as a separate conditional
  neuroimaging target after the synthetic lifecycle and after separately
  reviewed architecture, runtime, input, privacy, and acceptance evidence. A
  separate accepted tracked decision and reproducible evidence would be required
  before any fMRIPost support claim or capability-classification change.
  Separate accepted authority would also be required before the workflow could
  become mandatory or release-blocking for an alpha. This roadmap, product
  direction, source presence, and planning code do not supply that authority.
  fMRIPost remains **Experimental or external-runtime** until the required
  acceptance and final claim-promotion review. The real workflow cannot replace
  fake-remote or live-cluster synthetic acceptance, and no DeepPrep execution,
  complete raw-BIDS preprocessing chain, or arbitrary neuroimaging-pipeline
  support is implied.

## After the first public alpha

Likely follow-up work may include:

- improving artifact installation and update workflows;
- expanding checked-in synthetic examples;
- hardening package-level Python interface documentation and compatibility
  policy;
- scientifically hardening ROI percent signal change under a separately
  reviewed acceptance contract, including independent numerical validation of
  the Featquery-backed path before any promotion; this is not current support or
  part of the first remote alpha;
- adding carefully bounded neuroimaging adapters only after the generic remote
  lifecycle is accepted;
- improving provenance, reporting, diagnostics, and recovery guidance; and
- continuing to refine the private-to-public export workflow.

## Longer-term possibilities

Exploration areas include shared application services, a stable high-level
Python API, HTTP services, dashboards, Streamlit or other graphical clients,
desktop clients, additional schedulers and execution backends, cloud storage or
execution, additional scientific adapters, and broader reporting and
visualization. These are possibilities, not committed work or current support.

Future clients should reuse application services and package-owned behaviour
rather than reimplementing scientific, BIDS, or HPC logic. The CLI would remain
a supported automation interface. This roadmap does not propose a plugin
marketplace or an arbitrary third-party extension contract.

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

The separately evidenced local ROI and materialized-pattern crossnobis examples
remain governed by the capability matrix outside the headline remote target.

## How roadmap items become supported capabilities

A roadmap item becomes supported only after the appropriate evidence for that
capability has been reviewed. Depending on the capability, that evidence may
include:

- an owning implementation, focused tests, and broader package and
  public-contract tests;
- a checked-in public synthetic example, fixture, or configuration where
  appropriate;
- documented prerequisites, limitations, and scientific boundaries;
- hosted CI, fake-remote acceptance, separately authorized live-environment
  acceptance, or real-tool acceptance where applicable;
- privacy, scientific, and licensing review; and
- an explicit capability-matrix update and release review.

For the synthetic remote support claim, the evidence must also bind the exact
accepted source commit, the canonical source/release payload inventory with a
SHA-256 tree digest, any coordinated artifact identities, the accepted site and
profile identity, and input, plan, runner, runtime, image, asset, and resource
identities. Versioned claims and receipts must cover successful and
controlled-failure outcomes, cancellation, collision rejection, invalid-output
rejection, interruption and recovery, verified retrieval, and publication. Full
operational receipts remain private; their sanitized publishable projection
requires privacy and technical review. Fake-remote evidence, separately
authorized live-environment evidence, real-tool evidence
where scientifically required, and explicit capability-matrix and
claim-promotion review under
[ADR-0022](docs/decisions/ADR-0022-headline-hpc-execution-contract.md) are also
required.

Not every item requires every type of evidence. The required combination
depends on the capability and its risks. Source presence, a command name, a
rendered plan, mocked execution, one successful external run, or documentation
wording is not sufficient by itself to establish support.
