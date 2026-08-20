# Analysis bundles

Analysis bundles are plan-only, configuration-owned references to exact unit
rows and named analysis component configurations. Create a starter file with:

```bash
rp analysis bundle init <name> --project <project>
```

Keep each bundle at `config/analysis/bundles/<name>.yaml`. Select either one
batch from `manifests/batches/` or one named cohort view from
`config/cohorts.yaml`; do not copy subject or visit rows into bundle YAML.

This template deliberately contains no pretend analysis units or executable
bundle configuration.
