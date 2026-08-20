
# Add a project

Create a private-by-default overlay with the canonical scaffold command:

```bash
PROJECT_NAME=private-project
rp project init "$PROJECT_NAME"
```

Then:

1. Run `rp config validate --project <name>`.
2. Run `rp config paths --project <name>` and review every resolved root.
3. Set `config/dataset.yaml` `dataset.primary` and any other dataset references.
4. Set `config/preprocessing.yaml` `preprocessing.default_batch` and review the matching manifest under `manifests/batches/`.
5. Add only the cohort, feature, model, visualization, and compute configuration the project needs.
6. Keep project-specific code small; move reusable utilities into `packages/`.
7. Keep reports and notebooks as consumers of package APIs.
8. Run `rp config validate --project <name>` and inspect paths again after every configuration change.

Use the specialized `rp project init` forms when you need a workflow-specific BIDS or tabular scaffold.
`project/project-template/` is a checked-in public schema example, not the normal starting mechanism.

## Keep inputs and outputs outside the checkout

For a tabular project, set `DATASETS_ROOT` and `ARTIFACTS_ROOT` before project
initialization or validation. For a private canonical dataset name that is not
explicitly mapped in `WORKSPACE.yaml`, the table path is
`DATASETS_ROOT / canonical_dataset / canonical_features_root / feature_table`.
The dataset and feature-root values come from `config/dataset.yaml`; the final
name comes from the selected one-row batch. An explicit workspace dataset
mapping takes precedence. See the
[bring-your-own-data guide](../byod.md) for a complete copyable scaffold.

For ROI or MVPA input, prefer one environment-backed named root in
`config/analysis.yaml`:

```yaml
analysis:
  external_input_roots:
    private_inputs:
      label: private-inputs
      local_root: ${RP_PRIVATE_INPUT_ROOT}
      sync_enabled: false
```

Analysis configurations then use `root_ref: private_inputs` and a relative
path. This declaration resolves a local name only: it does not copy or publish
data, and without `remote_root` it creates no remote capability. Do not commit a
literal personal host path. Review the resolved declaration with
`rp config paths --project <name>`.

## Add an analysis bundle

For an analysis that spans named ROI, extraction, or MVPA components:

1. Put each exact input or analysis unit in one
   `manifests/batches/<batch>.tsv` row. Do not create a subject/session/run grid.
2. If the same selection will be reused, define a named view over that batch in
   `config/cohorts.yaml`. Keep exclusion identifiers and reasons in the view or
   its source-row metadata rather than copying rows to another manifest.
3. Create a small bundle scaffold:

   ```bash
   PROJECT_NAME=private-project
   BUNDLE_NAME=example-bundle
   rp analysis bundle init "$BUNDLE_NAME" --project "$PROJECT_NAME"
   ```

4. Review its batch-or-cohort selection, unique key columns, longitudinal
   completeness policy, component references, and stage order in YAML.
5. Run `validate`, `doctor`, and `plan` before considering any component's own
   execution lifecycle.

Bundle planning is non-mutating and does not execute ROI, MVPA, external tools,
or HPC actions. Keep scientific selection in reviewed configuration rather than
reconstructing it with a long command-line selector list.

## Public/private boundary

The root allowlist exposes only these public overlays:

- `project-template`
- `project-example`
- `project-pilot-bids`
- `project-pilot-tabular`

These four checked-in overlays are public examples only. All other `project/*` overlays are ignored by
default. Real project overlays must live in a separate private repository or another explicit private
boundary. Do not weaken the allowlist to make a real overlay visible in this repository.

Confirm the actual ignore rules without creating probe files:

```bash
PROJECT_NAME=private-project
git check-ignore --no-index -v "project/$PROJECT_NAME/project.yaml"
git check-ignore --no-index -v artifacts/runs/example/output.json
git check-ignore --no-index -v datasets/private-example/rawdata/input.tsv
```

These commands do not establish that arbitrary payloads under every dataset
directory are ignored. External data outside the checkout is outside this
repository's tracking boundary; an in-checkout private path must match a real
ignore rule.
