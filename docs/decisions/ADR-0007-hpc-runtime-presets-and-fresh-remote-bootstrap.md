# ADR-0007: Scaffold-Time HPC Runtime Presets and Fresh-Remote Bootstrap Payload

## Status

Accepted

## Context

The beginner-friendly BIDS slice already let `research-core` scaffold projects from tool adapters and let `research-hpc` own stage, pull, submit, and explicit bootstrap execution. Two gaps remained for first-time HPC use:

- newly scaffolded projects had no reusable way to inherit site-specific `compute.slurm.modules` and `compute.slurm.pre_activate_commands`
- fresh-remote submit could stage a planned run, but repo bootstrap still depended on files and package dependencies that were not guaranteed to be present unless the user had already run a broader workspace sync

We want a smoother first-run path without introducing a second runtime mechanism, without making runtime behavior depend dynamically on `WORKSPACE.yaml`, and without collapsing cluster-specific behavior into tool adapters.

## Decision

- Allow `WORKSPACE.yaml` to define optional scaffold-time HPC runtime presets under:
  - `hpc.runtime_defaults.default`
  - `hpc.runtime_defaults.catalog.<name>.slurm.modules`
  - `hpc.runtime_defaults.catalog.<name>.slurm.pre_activate_commands`
- Apply those presets only during project scaffold, materializing the selected values into project `config/compute.yaml`.
- Keep adapter scaffold defaults as the base, with workspace presets contributing only scaffold-time defaults for the two list fields above.
- Keep `project/*/config/compute.yaml` as the runtime source of truth after scaffold.
- Keep `compute.slurm.modules` and `compute.slurm.pre_activate_commands` as the only runtime environment mechanism in this slice.
- Expand the staged common payload for fresh-remote submit to include `ops/` and `packages/research-hpc`, so repo bootstrap can run without a separate `rp hpc sync workspace`.
- Make explicit bootstrap resolve manifest `hpc.connection` metadata first, falling back to legacy `hpc.ssh_host` only when no profile metadata is available.

## Consequences

- Workspaces can encode site-specific scaffold defaults once without forcing users to hand-edit every new project.
- The public root workspace uses a provider-neutral SLURM preset with remote-root placeholders only. Module stacks,
  scheduler accounts, and site-specific environment setup belong in optional examples or local, untracked targets.
- Runtime behavior remains explicit in project config, avoiding hidden coupling from later `WORKSPACE.yaml` changes.
- Fresh-remote submit now carries the bootstrap assets and minimum package closure needed for repo bootstrap in the staged payload.
- Explicit bootstrap becomes consistent with stage, pull, and submit in how it resolves remote connections.
- The design stays narrow:
  - no parallel runtime mechanism was added
  - no dynamic runtime lookup from `WORKSPACE.yaml` was added
  - no full workspace sync is required just to make bootstrap assets available
