# ADR-0004: BIDS Events Builder Boundaries

## Status

Accepted

## Context

The workspace needs a reusable BIDS events builder that can map project-specific behavioral exports into BIDS `_events.tsv` outputs without pushing study logic into shared package code.

## Decision

- Keep BIDS-specific events plan/build/publish logic in `packages/research-bids`.
- Keep study-specific mappings, sidecar metadata, and stimuli policy in `project/*/config/events/`.
- Stage previews, sidecars, copied stimuli, and manifests in `artifacts/...`.
- Publish canonical `_events.tsv`, optional `_events.json`, and optional `stimuli/` assets into `datasets/...`.
- Make `publish` manifest-driven so it copies only files enumerated by `build`.
- Treat the current row-semantics layer as specialized for encoding/recognition-style tasks rather than as a fully generic BIDS events transform engine.

## Consequences

- Shared code stays reusable across studies.
- Project overlays stay thin and config-driven.
- Canonical vs ephemeral boundaries remain explicit.
- Build and publish behavior is auditable through staged manifests.
- BIDS output mechanics are reusable now, while broader task-family support will require a more generic row-semantics engine later.
