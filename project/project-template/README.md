# project-template

> **Alpha status — Plan/validation only.** This is the current-schema public
> template, not a runnable study. Use `rp project init` to create an overlay,
> then supply reviewed project choices and data outside this public template.

This valid thin overlay is backed only by deterministic public toy tabular
fixtures. It demonstrates the current `dataset.primary`,
`preprocessing.default_batch`, compute, model, and batch-manifest contracts.
Its notebook, report, and test directories are scaffold structure rather than
runnable content.

For the runnable public walkthrough, use the
[source-checkout quickstart](../../docs/onboarding/quickstart.md). See
[Add a project](../../docs/onboarding/add-a-project.md) and the
[capability matrix](../../docs/capabilities.md) before adapting this template.

It should answer:
- Which datasets are used?
- Which cohorts and splits are used?
- Which pipelines are used?
- Which model and visualization configs are used?
- Which compute profile is used?

Shared logic should remain in `packages/`.

Create a new overlay with `rp project init <name>` rather than copying this directory. Then validate it
with `rp config validate --project <name>` before adding project-specific configuration.

## Tabular predictor contract

The selected batch row owns `feature_table` and `target_column`. The ordered
predictors live only in `config/models.yaml` under
`models.default.feature_columns`; public `rp` workflows do not infer them.
Review that list for each real table, excluding identifiers, targets, alternate
outcomes, grouping variables, and other leakage-prone columns. Changing the
order changes the scientific model contract, and invalid entries are rejected
before run output is created.

## Analysis-unit convention

Store each real analysis or input unit as one row in
`manifests/batches/<batch>.tsv`. Sessions, tasks, and runs are optional columns;
do not create placeholder values or expand separate subject/session/run lists
into a Cartesian product. Preserve deterministic metadata columns that are
needed by downstream adapters.

Define reusable cohort views in `config/cohorts.yaml`. A cohort references one
batch and applies configured include filters and auditable exclusion rules; it
does not copy rows into another manifest. Analysis bundles belong in
`config/analysis/bundles/` and reference either one batch or one cohort plus
named component configurations. Initialize a bundle scaffold with:

```bash
rp analysis bundle init <name> --project <project>
```

This template intentionally includes neither pretend neuroimaging units nor an
executable bundle. Scientific selection and longitudinal policy belong in the
reviewed YAML, not in a long command-line selector list.

## Privacy boundary

This is one of four checked-in public overlays only. Real-study configuration
must live in a separate private repository or another explicit private
boundary, outside the public `project/` tree. Do not weaken the root project
allowlist.
