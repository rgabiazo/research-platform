# ds-bids-example

> **Alpha status — Plan/validation only.** This is metadata and minimal-fixture
> material. It does not include a complete BOLD series, preprocessing inputs,
> FEAT EV/confound inputs, or an executable DeepPrep, fMRIPost-AROMA, or FEAT
> dataset.

Tiny synthetic BIDS-like fixture for documentation, config validation, and CI smoke tests.

This directory must not contain real imaging data, raw task exports, participant notes, recruitment
metadata, or copied study manifests. Keep committed files small and synthetic. Put real datasets
outside the repository and point projects at them with `${BIDS_ROOT}`, `${DERIV_ROOT}`, or local
environment variables.

The fixture intentionally includes only minimal metadata and README placeholders for derivative roots
that public-safe project overlays can reference.
