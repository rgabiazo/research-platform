# LOSO ROI To Manual Crossnobis Template

This is a public-safe command template for a generic localizer-to-MVPA workflow. It intentionally
does not point at an active study overlay or a committed full-study config tree.

Before adapting it, create a toy or private project overlay outside the public branch and use
placeholder roots:

```bash
export BIDS_ROOT=/path/to/bids
export DERIV_ROOT=/path/to/derivatives
export PROJECT_ROOT=${PROJECT_ROOT:-project/<toy-project>}
```

Safe read-only validation pattern:

```bash
rp analysis roi list --project <toy-project>
rp analysis roi validate <roi-set> --project <toy-project>
rp analysis roi transform validate <roi-transform> --project <toy-project>
rp analysis mvpa validate <mvpa-set> --project <toy-project>
```

Dry-run planning pattern:

```bash
rp analysis roi doctor <roi-set> --project <toy-project>
rp analysis roi build <roi-set> --project <toy-project>
rp analysis roi transform plan <roi-transform> --project <toy-project>
rp analysis mvpa doctor <mvpa-set> --project <toy-project>
rp analysis mvpa plan <mvpa-set> --project <toy-project>
```

Execution commands should be reviewed before adding `--execute`. Real datasets, localizer fixed-effects
inputs, run manifests, exclusion rules, and generated derivatives belong outside the public repository.
