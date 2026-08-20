# ADR-0006: Beginner-Friendly BIDS Onboarding and Submit Flow

## Status

Accepted

## Context

The adapter seam from ADR-0005 keeps `research-core` orchestration-only, but first-time users still had too much manual setup:

- they had to hand-write project YAML before a basic BIDS preprocess run
- they had to run batch discovery as a separate step before submit
- `rp hpc sync data` did not help when projects kept BIDS roots outside the workspace

We want a smoother beginner path without collapsing tool-specific logic back into `research-core`.

## Decision

- Add `rp project init bids-preprocess` in `research-core` as a beginner-facing scaffold command.
- Keep tool and adapter choices config-driven by resolving `--tool` through a small registry and asking the adapter for scaffold defaults.
- Extend the BIDS adapter contract with scaffold metadata/defaults.
- Add `--discover` to `rp run submit preprocess bids` so submit can auto-discover and write the batch manifest before the existing generic plan/stage/submit flow.
- Let planned SLURM runs record generic `hpc.connection` metadata for a named SSH profile when available.
- Extend `rp hpc init` so the saved local defaults can include the named SSH profile used by manifest-based stage/pull/submit commands.
- Keep `rp run slurm ...` planning-only and keep publish-back planning-only.
- Extend `rp hpc sync data` so it can sync project-declared external local roots only when the project config also declares explicit remote destinations.
- Do not infer or sync arbitrary external paths.

## Consequences

- Beginners can scaffold a valid BIDS preprocessing project without hand-editing YAML on the happy path.
- Beginners can configure a named SSH profile once and reuse it across sync, stage, pull, and submit flows without a separate `RP_HPC_HOST` export.
- The generic command surface stays intact:
  - `rp project init bids-preprocess`
  - `rp run submit preprocess bids --discover`
  - `rp hpc sync data`
- `research-neuro` continues to own fMRIPost-AROMA-specific defaults, discovery, and runtime behavior.
- `research-hpc` continues to own SSH/rsync/SLURM execution behavior.
- `research-core` remains orchestration-only, but now provides a more usable first-run experience.
