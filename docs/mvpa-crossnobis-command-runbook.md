# MVPA Crossnobis Command Runbook

This runbook is for public-safe, study-agnostic MVPA/crossnobis command flows. It intentionally avoids
active study labels, workstation paths, private cluster hostnames, real subject manifests, and exclusion
notes.

## Environment

Use placeholders or local environment variables:

```bash
export BIDS_ROOT=/path/to/bids
export DERIV_ROOT=/path/to/derivatives
export PROJECT_ROOT=${PROJECT_ROOT:-project/project-pilot-bids}
export HPC_PROFILE=${HPC_PROFILE:-target-a}
```

For a possible future HPC run, use the canonical local starter to define an
ignored SSH profile under `secrets/` while keeping public configuration
generic:

```bash
rp hpc setup \
  --target <target-name> \
  --profile "${HPC_PROFILE}" \
  --user <ssh-user> \
  --host <login-host> \
  --remote-workspace-root <remote-workspace-root> \
  --ssh-config secrets/hpc/ssh-profiles.yaml \
  --targets-config secrets/hpc/targets.yaml

rp hpc validate --target <target-name>
```

`rp hpc setup` writes local starter files and makes no network call. It does
not prove credentials, host reachability, scheduler, storage, runtime, data, or
MVPA readiness. The default generic starter is provider-neutral;
Alliance/MFA behavior remains an optional provider integration requiring
explicit `--template alliance` selection and site review.
`rp hpc validate` is subprocess-free, write-free, and
network-free; its promotion-policy result is declared, not remotely verified.
The older `rp hpc init` is retained only as a legacy, backward-compatible
Alliance-oriented local-default helper. Any later `rp hpc doctor` is
SSH-active and immediately contacts its configured host. Remote MVPA/HPC
execution is experimental and has no live-cluster validation.

## Validate Configuration

```bash
rp config validate --project <toy-project>
rp config paths --project <toy-project>
rp analysis mvpa list --project <toy-project>
rp analysis mvpa validate <analysis-name> --project <toy-project>
```

The rendered paths should contain `${BIDS_ROOT}`, `${DERIV_ROOT}`, `${PROJECT_ROOT}`, `/path/to/bids`,
`/path/to/derivatives`, or other generic placeholders. They should not contain local usernames,
mounted volumes, private hostnames, full study manifests, participant totals, or study-specific
exclusion ids.

## Plan Analysis

```bash
rp analysis mvpa plan <analysis-name> --project <toy-project>
```

Review the JSON payload for:

- `valid: true`
- expected root references
- no private absolute paths
- no real subject/run lists
- no generated outputs written during plan mode

## Plan Exports

```bash
rp analysis mvpa export-tables <table-set> --project <toy-project>
rp analysis mvpa export-figures <figure-set> --project <toy-project>
rp analysis mvpa export-rdms <rdm-set> --project <toy-project>
rp analysis mvpa export-publication <publication-set> --project <toy-project>
rp analysis mvpa publish-derivatives <publish-set> --project <toy-project>
```

Use `--execute` only when the inputs are synthetic or public-safe and outputs are meant to be generated
locally under `artifacts/` or `.research-platform/`.

## What To Generalize Later

When private overlays contain useful behavior, port only the reusable parts:

- analysis schema validation into `packages/research-analysis`
- BIDS/derivative discovery into `packages/research-bids` or `packages/research-neuro`
- HPC execution behavior into `packages/research-hpc` or `ops/`
- small synthetic examples into `datasets/` or `project/`

Do not port active study names, cohort manifests, exclusion notes, participant totals, local paths, or
cluster-specific assumptions into the public branch.
