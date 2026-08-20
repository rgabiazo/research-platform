# Changelog

All notable changes to Research Platform will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
The coordinated Python distributions use
[PEP 440](https://peps.python.org/pep-0440/) version identifiers.

## [Unreleased]

### v0.1.0a1 development series

This unreleased section records completed changes in the `v0.1.0a1`
development series; it does not describe a published release. Future planned
work belongs in the [roadmap](ROADMAP.md). The future coordinated release title
is **Research Platform 0.1.0 Alpha 1**; it remains unpublished.

- Prepared the source checkout for a future public alpha; the coordinated packages are
  not currently distributed through PyPI.
- Added runnable local examples for synthetic tabular analysis,
  coordinate-sphere ROI construction, generic-NIfTI extraction, and
  materialized-pattern crossnobis analysis.
- Added an RDM-ready pairwise-distance table to the crossnobis example;
  RDM/report export remains a separate deferred capability.
- Kept local HPC validation and rendering plan-only, while remote
  and scheduler execution, and other external-runtime integrations, as
  experimental or externally configured surfaces. No live-cluster or public
  real-data validation is claimed.
- Added deterministic, generated-from-scratch public fixtures with documented
  provenance and non-mutating verification commands.
- Established public/private overlay boundaries, repository sanitation checks,
  portable path safeguards, side-effect-free planning, and failure-safe
  transactional output handling for the documented local tabular workflow.

See the [capability and support matrix](docs/capabilities.md) for the exact
runnable, plan-only, external-runtime, and scaffold boundaries.
