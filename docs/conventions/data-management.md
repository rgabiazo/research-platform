
# Data management conventions

- `datasets/` is canonical
- `artifacts/` is ephemeral
- `scratch/` is disposable
- Preserve provenance for anything promoted into canonical derivatives
- Keep metadata near data where possible
- Never assume all datasets are BIDS, but support BIDS cleanly where applicable
