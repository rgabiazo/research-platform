# hpc

`requirements-runtime.txt` contains the scheduler-side workflow driver requirements. The
development bootstrap installs this group with `--no-index` under the `hpc` profile, using only the
local directory configured by `RP_BOOTSTRAP_WHEELHOUSE`. Sites may instead provide a prebuilt,
site-managed Python 3.11 or 3.12 runtime that passes the bootstrap's read-only HPC check: working
package metadata, current editable packages from the staged checkout, the SLURM executor
plugin, and the relevant workspace and `snakemake` commands. Any installation or reconciliation
still requires an isolated virtual environment.

Notebook launchers also require a site-provided Jupyter runtime; it is intentionally not part of
this minimal scheduler requirement group.

## Provider-adaptation preflight

The checked-in configuration is scaffolding, not live-cluster validation. Before
authorizing any remote operation, an operator must review the selected site's:

- SSH host/profile, host-key policy, and interactive, MFA, or non-interactive
  authentication requirements;
- account and partition, plus QoS, constraint, and other scheduler allocation
  rules;
- remote workspace, artifact, container, temporary, and scratch roots instead
  of assuming `$SCRATCH`, `/scratch`, or another provider layout;
- module commands, if any, without assuming a particular site module stack;
- supported Python version, currently limited to the project's verified
  Python 3.11 and 3.12 boundary;
- isolated virtual environment bootstrap or offline wheelhouse availability;
- scheduler commands for submission, queue inspection, cancellation, and accounting,
  including whether `sacct` or an equivalent accounting service is available;
- container runtime, supported version, image policy, cache placement, and
  whether compute nodes can access registries;
- outbound-network restrictions on login and compute nodes;
- storage quotas, inode limits, retention policy, and transfer behavior; and
- institutional data-governance, privacy, encryption, residency, and access
  requirements.

Alliance/MFA configuration may be used as an optional provider-specific
example only after site review. Neither that example nor adaptable generic
fields establish Alliance, Nibi, or other provider compatibility. Repository
tests exercise local plans and mocked remote command boundaries; they do not
validate a live scheduler, runtime, transfer, or scientific workload.
