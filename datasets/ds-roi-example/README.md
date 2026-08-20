# Deterministic toy ROI dataset

This tiny generic-NIfTI dataset exists only to verify local coordinate-sphere
ROI construction and generic NIfTI value extraction. It is deliberately
non-anatomical and is not a BIDS imaging dataset.

Every identifier, voxel, affine element, header field, and metadata value is
an algorithmic invention. No participant, patient, health, demographic, or
other human data were used, and no external dataset was used.

Both images have a `9 x 9 x 9` grid, little-endian `float32` values, and a
2 mm diagonal affine with `(-8, -8, -8)` mm translation. The reference image
is zero everywhere. For zero-based voxel indices `x`, `y`, and `z`, the value
image uses:

```text
10 + x*x + 2*y + 3*z
```

The uncompressed `.nii` representation keeps byte-level regeneration stable.
Regenerate and verify the generator-owned images and metadata from the
repository root with:

```bash
python3 ops/scripts/generate_toy_roi_fixtures.py
python3 ops/scripts/generate_toy_roi_fixtures.py --check
```

`images/toy_reference.nii` supplies only a synthetic grid and affine.
`images/toy_values.nii` supplies deterministic spatial variation for extraction.
Generated ROI outputs belong under the ignored `artifacts/` tree, never in this
canonical input dataset.
