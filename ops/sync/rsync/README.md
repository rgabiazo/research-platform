# rsync helper scripts and filters

Phase 2A adds scope-oriented exclude files for first-time HPC provisioning:

- `exclude.txt`: global excludes that should never be uploaded
- `exclude.workspace.txt`: tracked default excludes for whole-workspace bootstrap syncs
- `exclude.common.txt`: shared workspace/config payload excludes
- `exclude.project-overlay.txt`: project overlay excludes that keep tracked notebooks available remotely
- `exclude.neuro-bids.txt`: excludes applied to raw neuro/BIDS dataset syncs
- `exclude.tabular-ml.txt`: excludes applied to tabular/ML package syncs

`research-core` selects scopes and attachs exclude-file references in the run manifest.
`research-hpc` consumes those references when rendering `rsync` commands.
