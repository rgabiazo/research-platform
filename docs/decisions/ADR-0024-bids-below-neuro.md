# ADR-0024: BIDS below Neuro

## Status

Accepted.

## Context

`research-bids` owns BIDS entities, portable paths, naming, and the events
facade. `research-neuro` uses those contracts for ROI and derivative paths.
The events build-spec semantics (spec loading, run grouping, and row building)
are part of that facade. Placing them under `research_platform.neuro.events`
makes `research-bids` import `research-neuro`, so the two packages depend on
each other.

## Decision

- `research-bids` never imports `research-neuro`.
- The events semantics modules `spec`, `grouping`, `rows`, and `service` live
  in `research_platform.bids.events.semantics`.
- `research_platform.neuro.events` keeps its public names. The semantics
  names resolve lazily from `research_platform.bids.events.semantics` on first
  access; the event-file inspection helper stays in `research-neuro`.
  Importing `research_platform.neuro.events` does not import `research-bids`,
  because remote tool adapters sync only `packages/research-neuro` and the
  remote bootstrap installs only the declared dependencies of each staged
  package.
- `research-neuro` declares `research-core`. `research-bids` and
  `research-analysis` are the optional extras `bids` and `analysis` until
  remote hosts receive those packages, after which they become declared
  dependencies. `research-bids` declares no internal package dependency.

## Consequences

- The package graph is acyclic: Neuro depends on BIDS, never the reverse.
- New event semantics belong in `research-bids`; Neuro-specific types and
  scientific methods stay out of it.
- Code importing the semantics names from `research_platform.neuro.events` keeps
  working, provided `research-bids` is installed when those names are used.
