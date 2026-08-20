#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../.." && pwd)"
python_bin="${PYTHON_BIN:-python}"

test_directories=(
  packages/research-analysis/tests
  packages/research-bids/tests
  packages/research-core/tests
  packages/research-hpc/tests
  packages/research-io/tests
  packages/research-ml/tests
  packages/research-neuro/tests
  packages/research-viz/tests
)

if [ "${1:-}" = "--print-plan" ]; then
  if [ "$#" -ne 1 ]; then
    printf 'Usage: %s [--print-plan]\n' "${BASH_SOURCE[0]}" >&2
    exit 2
  fi
  for test_directory in "${test_directories[@]}"; do
    printf '%q -m pytest -p no:cacheprovider %q\n' \
      "${python_bin}" "${test_directory}"
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
for test_directory in "${test_directories[@]}"; do
  printf 'Running %s\n' "${test_directory}"
  "${python_bin}" -m pytest -p no:cacheprovider "${test_directory}"
done
