# ADR-0010: Local HPC Targets For Multi-Cluster Defaults

## Status

Accepted

## Context

Users need to move the same project between local, institutional, and future SLURM clusters without
editing tracked project configs or repeating connection and remote-root flags on every command.
Existing commands already know how to stage, sync, verify, render SLURM scripts, submit, pull, and
load project env placeholders. The missing layer is a local, reusable target selection point.

## Decision

- Add local-only target presets under `secrets/hpc/targets.yaml`, with tracked examples under `ops/`.
- Add `rp hpc target list/show/use`.
- Let `rp hpc target use <target>` persist `RESEARCH_HPC_TARGET=<target>` in `secrets/.env`.
- Resolve active target defaults before project bundles are loaded, so project YAML can keep using env placeholders.
- Let target env override values that were loaded from `secrets/.env`, while preserving shell-exported values as explicit user overrides.
- Treat target `slurm` settings as site defaults over existing `compute.slurm`, not as a new runtime engine.
- Render optional `#SBATCH` account, partition, qos, node, constraint, and export directives only when configured.
- Record the resolved target name and SLURM site settings in each planned SLURM run manifest.

## Consequences

- Existing `rp hpc sync`, `rp hpc verify data`, `rp run slurm`, and `rp run submit` commands work unchanged after target selection.
- Project configs remain thin and env-driven across studies and clusters.
- Real usernames, accounts, key paths, and study paths stay local under `secrets/`.
- Future clusters can be added by adding target entries, not by adding command flags.
- If a site rejects the current controller-job submission pattern, that should be handled as an additional submit mode rather than by changing target or project config semantics.
