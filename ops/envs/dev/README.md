# dev

`bootstrap.sh` creates or reuses `.venv` and ensures one explicit dependency profile:

- `minimal` (the local default) installs the reusable CLI and toy-tabular package closure.
- `dev` adds the test dependency used by the repository.
- `full` adds `research-viz`; the declared analysis RDM, BIDS/IO pandas, and ML XGBoost
  extras; and the notebook, workflow, visualization, and advanced-analysis requirements in
  `requirements-notebook.txt`.
- `hpc` installs the `research-*` packages present in a staged checkout plus `../hpc/requirements-runtime.txt`, always with the package index disabled.

Profile selection is additive when an existing virtual environment is reused: the bootstrap does
not uninstall packages left by a broader profile. A newly created `minimal` environment is the
smallest supported default.

Inspect a profile without creating an environment or invoking an installer:

```bash
bash ops/envs/dev/bootstrap.sh --print-plan --profile minimal
```

Run `bash ops/envs/dev/bootstrap.sh --help` for profile details. Inside a
SLURM job, the script selects `hpc` automatically and rejects an explicit non-HPC profile. An HPC
installation requires a usable existing site-managed runtime or an explicitly configured local
package source through `RP_BOOTSTRAP_WHEELHOUSE`; it never assumes that a site wheelhouse exists or
modifies a non-isolated Python installation.

After a local-profile installation, `smoke-check.sh` runs the read-only public CLI checks used to
verify the environment. Without a wheelhouse, the HPC profile performs a read-only reuse check for
working package metadata, current editable staged packages, workspace commands, `snakemake`, and the SLURM
executor plugin. With a wheelhouse, installation or reconciliation is allowed only in an isolated
virtual environment. A partial staged checkout is checked against only the workspace packages it
contains.
