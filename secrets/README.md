# secrets

Local-only credentials and connection material.

Keep real SSH keys, tokens, cluster usernames, host aliases, and other private connection details here on your machine only. These files should stay untracked.

Preferred HPC setup:

```bash
rp hpc setup \
  --target <target-name> \
  --host <login-host> \
  --user <ssh-user> \
  --remote-workspace-root /remote/workspace/root

rp hpc validate --target <target-name>
```

The default provider-neutral `generic` setup writes local-only SSH-profile and target
configuration without committing usernames, hosts, key paths, accounts, or
remote paths. It assumes no provider, authentication method, module stack,
scratch convention, container runtime, account, partition, or software
version. The artifacts root defaults to
`<remote-workspace-root>/artifacts`; no container root is generated unless it
was explicitly supplied.

`rp hpc validate` reads those local files without writing or contacting a
host. It must pass before any SSH-active check. Validation confirms that
`promotion.mode: atomic_no_replace` is declared, but it cannot prove that a
remote filesystem supports that policy.

After a project overlay exists and validation passes, deliberately cross the
network boundary with the SSH-active command:

```bash
rp hpc doctor --project <project-name>
```

MFA-backed Alliance systems remain available through the optional
`--template alliance` integration; select it only when those assumptions apply
and after site review. No provider has been live validated.

For advanced/manual setup, you can still create the files directly:

```bash
research-hpc ssh init-config \
  --template generic \
  --profile <profile-name> \
  --host <login-host> \
  --user <ssh-user> \
  --output secrets/hpc/ssh-profiles.yaml
cp ops/sync/ssh/targets.example.yaml secrets/hpc/targets.yaml
```

Replace every placeholder, add only site-reviewed optional settings, and then
run `rp hpc validate`. The generic default creates only the `login` role.
`research-hpc ssh init-config --template alliance ...` remains an explicitly
selected optional provider integration.

The lower-level command accepts a caller-selected `--output`; unlike
high-level `rp hpc setup`, it does not enforce placement beneath `secrets/`.
Its narrowly scoped private writer rejects symlinks, special files, and
hard-linked destinations; on POSIX it creates newly needed directories with
mode `0700` and creates or secures the output file with mode `0600` before
writing. The example deliberately chooses an ignored `secrets/` path. The
caller remains responsible for choosing an untracked private location.

Good candidates for `secrets/`:

- `secrets/hpc/ssh-profiles.yaml`
- `secrets/hpc/targets.yaml`
- private SSH key paths referenced by your local config
- local environment files with usernames, tokens, or cluster-specific overrides

Never commit real usernames, key material, hostnames, or machine-specific paths into tracked docs, examples, or config files.
High-level `rp hpc setup` destinations must remain beneath `secrets/`. On
POSIX it requires permissions equivalent to `0700` for created directories and
`0600` for files, rejects symlink, special-file, and hard-linked destinations,
and fails before content mutation if those private modes cannot be established
and verified. Public scaffold exceptions such as `README.md`, `.gitkeep`, and
`*.example` remain forbidden setup destinations even when they are lexically
beneath `secrets/`.

Offline validation never reads identity or known-hosts contents. It checks
declared references for controls and unresolved placeholders, but does not
prove that a referenced path exists or is a regular file.
