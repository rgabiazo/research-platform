# datasets

This directory contains approved canonical synthetic inputs and reusable
synthetic derivative examples. It is not a destination for run outputs or a
place to copy private study data.

The checked-in public data serve distinct bounded examples:

- `ds-tabular-example` and `ds-derivatives-example` support the deterministic
  tabular walkthrough;
- `ds-roi-example` supplies non-anatomical generic-NIfTI inputs;
- `ds-mvpa-example` supplies ROI-final prepared vectors; and
- `ds-bids-example` contains metadata and placeholders for plan/validation
  only, not an executable imaging dataset.

Generated run products belong under `artifacts/`. Real or participant-derived
datasets must remain outside the public repository and be connected through a
private project boundary. Canonical publication is workflow-specific and
requires explicit validation; file presence alone is not approval.

See the [data lifecycle](../docs/architecture/data-lifecycle.md) and the
[capability matrix](../docs/capabilities.md).
