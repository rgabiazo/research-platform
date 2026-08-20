
# Configuration conventions

- Use YAML or TOML for human-edited configs
- Put cross-workspace paths in `WORKSPACE.yaml`
- Use project-level configs to express cohorts, models, features, viz, and compute choices
- Prefer environment variables for machine-specific overrides
- Do not bake cluster names, mount points, or usernames into source files

## Machine-local input roots

For a canonical dataset name not explicitly mapped in `WORKSPACE.yaml`,
tabular paths are derived from `DATASETS_ROOT`, the project's
`dataset.canonical_dataset` and `dataset.canonical_features_root`, and the
selected batch row's `feature_table`. An explicit workspace dataset mapping
takes precedence. Set `DATASETS_ROOT` and
`ARTIFACTS_ROOT` before initialization, validation, planning, and execution
when those roots live outside the checkout.

ROI and MVPA inputs use named roots under
`analysis.external_input_roots` in `config/analysis.yaml`:

```yaml
analysis:
  external_input_roots:
    private_inputs:
      label: private-inputs
      local_root: ${RP_PRIVATE_INPUT_ROOT}
      sync_enabled: false
```

Use `root_ref: private_inputs` plus safe relative paths in component
configuration. Environment variables expand at resolution time. Existing
relative paths follow the established workspace-first, then project-overlay
resolution behavior; if neither candidate exists, the declaration remains
project-relative. Use `rp config paths --project <project>` to inspect the
resolved local path and existence state.

For projects with declarations, `rp config paths` adds an
`analysis_external_input_roots` mapping keyed by configured name. Entries
contain `label`, resolved `local_root`, `exists`, and `sync_enabled`, plus
`remote_root` only when declared. The command is read-only and does not create
the path or contact the remote destination.

`sync_enabled: false` states that the local private input is not a
synchronization source. Omitting `remote_root` creates no remote destination.
Named-root registration neither copies nor publishes data. Do not store
credentials or literal personal host paths in tracked YAML, and remember that
path-inspection output and project configuration may themselves be sensitive.

## Exact analysis units, cohorts, and bundles

Use `project/<name>/manifests/batches/*.tsv` as the only canonical
project-level row store for analysis or input units. Each row describes one
combination that actually exists. A neuro unit requires `subject_id`;
`session_id`, `task_id`, and `run_id` are optional. Do not add placeholder
dimensions or generate a Cartesian product from separate entity lists.

Arbitrary deterministic metadata columns are allowed and preserved, including
cohort, eligibility, QC, exclusion, timepoint, visit-order, acquisition,
direction, and adapter-specific fields. Stored BIDS-like values such as
`sub-toy01`, `ses-01`, and `run-01` remain canonical. A later adapter may derive
an unprefixed alias, but bundle resolution does not rewrite the stored identity.

Define named cohort views in `config/cohorts.yaml`:

```yaml
cohorts:
  example-view:
    batch: example_units
    include:
      eligible:
        - "true"
      qc_status:
        - pass
    exclude:
      - id: configured-exclusion
        filters:
          exclusion_id:
            - configured-exclusion
        reason_field: exclusion_reason
```

Values within one filter column are OR alternatives. Filters across columns
are combined with AND. Exclusion rules run after inclusion and carry stable
identifiers plus fixed reason text or a reason-field reference. Unknown columns
and missing required values are errors. Unmatched exclusion rules are reported,
and excluded rows remain visible in plans. Cohorts do not create copied TSVs
under `manifests/cohorts/`; exclusions do not create another row-list family.

Plan-only analysis bundles live at
`config/analysis/bundles/<name>.yaml`. A bundle selects exactly one cohort or
one batch and references existing named component configuration:

```yaml
analysis_bundle:
  name: example-bundle
  selection:
    cohort: example-view
  units:
    key_columns:
      - subject_id
    subject_column: subject_id
    incomplete: allow
  components:
    roi_set: example-rois
  stages:
    - roi_build
```

Literal subject, session, task, or run lists do not belong in bundle YAML or on
the bundle command line. Unit keys must be unique. For longitudinal work,
configure an occasion column, any required occasion values, and an explicit
occasion-order column such as `visit_index` when chronology matters. Session
labels are not sorted lexically to infer visit order. The incomplete-case
policy is explicit: `fail` rejects incomplete subjects, `drop` removes and
reports all of their units, and `allow` retains and reports them. No policy
silently balances a cohort.

The bundle lifecycle is non-mutating and has no `run` command:

```bash
rp analysis bundle init <name> --project <project>
rp analysis bundle list --project <project>
rp analysis bundle show <name> --project <project>
rp analysis bundle validate <name> --project <project>
rp analysis bundle doctor <name> --project <project>
rp analysis bundle plan <name> --project <project>
```

`init` writes the scaffold YAML by default. Add `--dry-run` to preview the
destination and YAML without writing, and use `--force` only when deliberately
replacing an existing scaffold file. It does not authorize analysis execution.

Plans report exact included and excluded rows, incomplete cases, entity counts,
component references, ordered stages, and deterministic source/configuration/
plan digests with `executed: false`. BIDS `run_id` remains an imaging entity; a
future bundle execution will use a separate execution or analysis-run ID.

## ROI lifecycle configuration

ROI and extraction initialization writes editable YAML under the project
overlay. Schema validation is intentionally separate from execution readiness:
use `rp analysis roi validate` for structure and `rp analysis roi doctor` for
inputs, dependencies, tools, image geometry, and output conflicts. ROI build
and extraction replacement is controlled only by
`runtime.existing_output: fail|replace`; its default is `fail`, and it is
independent of scaffold `--force` and
`publication.existing_output`. See [Reusable ROI Workflows](../roi-workflows.md)
for the canonical plan-before-execute sequence.

## MVPA lifecycle configuration

MVPA sets live at `config/analysis/mvpa/<name>.yaml`. Start a prepared-vector
configuration with:

```bash
rp analysis mvpa init <name> \
  --project <project> \
  --template materialized-crossnobis
```

Normal initialization writes one editable YAML file. `--dry-run` previews that
file without writing, and `--force` replaces only an existing scaffold YAML.
It never authorizes runtime replacement. The generated YAML is the source of
truth for conditions, comparisons, ROI and feature-space identity, CV,
centering, noise, thresholds, source roots, and output policy.

New MVPA configurations use `unit_selection.mode: exact_units`. Select their
rows through a named analysis bundle whose `components.mvpa_set` names the MVPA
set and whose ordered stages include `mvpa`:

```yaml
analysis_bundle:
  name: example-mvpa-bundle
  selection:
    batch: example_units
  units:
    key_columns:
      - subject_id
      - session_id
      - run_id
    subject_column: subject_id
    incomplete: allow
  components:
    mvpa_set: example-crossnobis
  stages:
    - mvpa
```

The MVPA CLI accepts only `--bundle` as a scientific-selection flag. It uses
the bundle resolver's exact included rows, excluded-unit audit, key columns,
and digests. It does not copy rows into the MVPA YAML, infer a bundle, or
construct missing subject/session/run combinations. The older inline selector
shape remains an advanced `legacy_cartesian` compatibility path for existing
FSL configurations only when no bundle is requested.

The dependency-light prepared-vector source is the fixed
`research_platform.neuro.mvpa.materialized_pattern_table.v1` TSV contract. Its
configuration declares a named root and safe relative path; it does not accept
host paths, alternate delimiters, globs, or column mappings. The
`fsl-feat-crossnobis` template is an advanced external-input image path.

Local runtime ownership is explicit:

```yaml
mvpa_set:
  outputs:
    runtime_root:
      root_ref: artifact_root
      path: .research-platform/mvpa/{mvpa_set}
  runtime:
    existing_output: fail
```

`fail` is the default and only v1 runtime collision policy. The complete final
runtime root must be absent before execution. A deliberate rerun uses a
different configured artifacts path; there is no runtime overwrite, replace,
resume, retry, or cleanup flag. The lifecycle computes the complete result in
memory, writes into one same-filesystem sibling staging directory, validates
the full inventory, and promotes the runtime root as one transaction. See
[MVPA Crossnobis Workflows](../mvpa-crossnobis.md) and
[ADR-0018](../decisions/ADR-0018-mvpa-exact-unit-runtime-transactions.md).
