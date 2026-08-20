#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../.." && pwd)"
python_bin="${PYTHON_BIN:-python}"
rp_bin="${RP_BIN:-rp}"

fixture_generators=(
  ops/scripts/generate_toy_memory_fixtures.py
  ops/scripts/generate_toy_tabular_fixtures.py
  ops/scripts/generate_toy_roi_fixtures.py
  ops/scripts/generate_toy_mvpa_fixtures.py
)
public_overlays=(
  project-template
  project-example
  project-pilot-bids
  project-pilot-tabular
)

if [ "${1:-}" = "--print-plan" ]; then
  if [ "$#" -ne 1 ]; then
    printf 'Usage: %s [--print-plan]\n' "${BASH_SOURCE[0]}" >&2
    exit 2
  fi
  printf '%q -m pytest -p no:cacheprovider ops/tests\n' "${python_bin}"
  for generator in "${fixture_generators[@]}"; do
    printf '%q %q --check\n' "${python_bin}" "${generator}"
  done
  for overlay in "${public_overlays[@]}"; do
    printf '%q config validate --project %q\n' "${rp_bin}" "${overlay}"
  done
  exit 0
fi
if [ "$#" -ne 0 ]; then
  printf 'Usage: %s [--print-plan]\n' "${BASH_SOURCE[0]}" >&2
  exit 2
fi

export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_NO_INDEX=1

cd -- "${repo_root}"
"${python_bin}" -m pytest -p no:cacheprovider ops/tests
for generator in "${fixture_generators[@]}"; do
  "${python_bin}" "${generator}" --check
done
for overlay in "${public_overlays[@]}"; do
  "${rp_bin}" config validate --project "${overlay}"
done
