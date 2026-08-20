# Local development

The development environment installs the workspace packages editably from this
source checkout; it does not claim or install them from PyPI. Only third-party
dependencies are obtained from the configured package source. The bootstrap does
not change the base environment or the repository's generic HPC environment
definitions.

## Bootstrap profiles

Preview the default local bootstrap without creating or changing an environment:

```bash
bash ops/envs/dev/bootstrap.sh --print-plan --profile minimal
```

Create and activate the minimal environment:

```bash
bash ops/envs/dev/bootstrap.sh --profile minimal
source .venv/bin/activate
```

If `RP_DEV_VENV` selects a different location, activate it with
`source "$RP_DEV_VENV/bin/activate"` and leave `RP_DEV_VENV` set for the smoke
check.

The available profiles are:

- `minimal` (the local default): editable installs of `research-core`,
  `research-hpc`, `research-io`, `research-bids`, `research-neuro`,
  `research-analysis`, and `research-ml`, plus their declared dependencies. It
  supports the public command surfaces and toy tabular workflow. In a newly
  created environment it does not add `research-viz`, test tools, notebooks,
  workflow-engine extras, or advanced optional model packages.
- `dev`: `minimal` plus `pytest>=8` for focused development tests.
- `full`: `dev` plus `research-viz`; the declared analysis RDM, BIDS/IO pandas,
  and ML XGBoost extras; and the repository's notebook, workflow,
  visualization, and advanced optional dependency stack.
- `hpc`: the staged workspace packages plus the scheduler runtime, installed
  without querying a package index. A real SLURM job selects this profile
  automatically; it requires an existing usable environment or an explicit
  local wheelhouse/package source. Read-only reuse requires working package
  metadata and current editable packages from the staged checkout;
  installation requires an isolated virtual environment.

Reusing an environment is additive and idempotent: the selected profile's
requirements are ensured, but packages installed earlier by a broader profile
are not removed. Use a fresh virtual environment when you need to prove the
minimal dependency boundary itself.

Run `bash ops/envs/dev/bootstrap.sh --help` for the complete option and package
source contract. Set `RP_DEV_VENV` to use a location other than `.venv`.
Outside the index-free HPC profile, the initial bootstrap requires network
access to a Python package index or a configured internal package source or
mirror.

## Installation and entry-point checks

After activation, run the read-only installation smoke check:

```bash
bash ops/envs/dev/smoke-check.sh
```

The Makefile exposes the same focused workflow:

```bash
make bootstrap-plan
make install-smoke
make test-bootstrap
```

`make test-bootstrap` requires the `dev` or `full` profile. It disables Python
bytecode and pytest's cache provider. Set `PYTHON` if the selected development
interpreter is not the active `python3`.

Useful installed entry-point checks include:

```bash
rp --help
research-hpc --help
research-io --help
research-bids --help
python -m research_platform.analysis.cli --help
rp config validate --project project-pilot-tabular
```

The analysis package has no separate console alias; use its module entry point.
Set `RESEARCH_PLATFORM_ROOT` to the workspace root when running workspace-aware
commands from outside the checkout.

## Local-only data and outputs

- Confidential data must stay out of Git.
- The four checked-in overlays are public examples only:
  `project-template`, `project-example`, `project-pilot-bids`, and
  `project-pilot-tabular`.
- Real `project/*` overlays are ignored by default and must live in a separate
  private repository or another explicit private boundary; do not weaken the
  public allowlist.
- Keep private source data under a clearly named
  `datasets/<private-dataset-name>/` directory protected by the ignore policy.
- Organize generated outputs under `artifacts/`, for example
  `artifacts/figures/<project-name>/` and `artifacts/tables/<project-name>/`.

For a notebook-first overlay, local work can use its notebook directly. Remote
notebook start and submit commands only render plans by default. Review every
reported local write and remote command, then repeat the selected command with
`--execute` to authorize remote action. The `hpc` bootstrap profile provides
the staged workspace packages and scheduler runtime, but not Jupyter; remote
notebook launch also requires Jupyter in the site-managed environment.
