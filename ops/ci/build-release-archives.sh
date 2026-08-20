#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../.." && pwd)"
python_bin="${PYTHON_BIN:-python}"
temp_root="${CI_TEMP_ROOT:-${RUNNER_TEMP:-${TMPDIR:-/tmp}}}"

packages=(
  research-analysis
  research-bids
  research-core
  research-hpc
  research-io
  research-ml
  research-neuro
  research-viz
)

print_plan=false
if [ "${1:-}" = "--print-plan" ]; then
  print_plan=true
  shift
fi
if [ "$#" -ne 0 ]; then
  printf 'Usage: %s [--print-plan]\n' "${BASH_SOURCE[0]}" >&2
  exit 2
fi

if [ "${print_plan}" = true ]; then
  printf 'Temporary root: %s\n' "${temp_root}"
  if [ -n "${RP_RELEASE_DIST_DIR:-}" ]; then
    printf 'Archive output: %s\n' "${RP_RELEASE_DIST_DIR}"
  else
    printf 'Archive output: <unique-directory-beneath-temporary-root>/dist\n'
  fi
  printf 'Materialize the complete staged Git index: '
  printf 'git checkout-index --all --prefix=<absolute-fresh-index-snapshot>/\n'
  for package in "${packages[@]}"; do
    printf 'Build complete tracked package directory: packages/%s\n' "${package}"
    printf '%q -m build --no-isolation --outdir <archive-output> <index-snapshot-of-%s>\n' \
      "${python_bin}" "${package}"
  done
  exit 0
fi

if [ ! -d "${temp_root}" ]; then
  printf 'CI temporary root does not exist: %s\n' "${temp_root}" >&2
  exit 2
fi
case "${temp_root}" in
  /*) ;;
  *)
    printf 'CI temporary root must be absolute: %s\n' "${temp_root}" >&2
    exit 2
    ;;
esac
temp_root="$(cd -- "${temp_root}" && pwd -P)"
case "${temp_root%/}/" in
  "${repo_root%/}/"*)
    printf 'CI temporary root must be outside the checkout: %s\n' "${temp_root}" >&2
    exit 2
    ;;
esac

export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_NO_INDEX=1
export PIP_NO_CACHE_DIR=1

work_root="$(mktemp -d "${temp_root%/}/research-platform-release.XXXXXX")"
snapshot_root="${work_root}/index"
dist_root="${RP_RELEASE_DIST_DIR:-${work_root}/dist}"

case "${dist_root}" in
  /*) ;;
  *)
    printf 'RP_RELEASE_DIST_DIR must be absolute: %s\n' "${dist_root}" >&2
    exit 2
    ;;
esac
dist_parent="$(dirname -- "${dist_root}")"
if [ ! -d "${dist_parent}" ]; then
  printf 'Archive output parent does not exist: %s\n' "${dist_parent}" >&2
  exit 2
fi
dist_parent="$(cd -- "${dist_parent}" && pwd -P)"
dist_root="${dist_parent%/}/$(basename -- "${dist_root}")"
case "${dist_root%/}/" in
  "${repo_root%/}/"*)
    printf 'Archive output must be outside the checkout: %s\n' "${dist_root}" >&2
    exit 2
    ;;
esac
if [ -e "${dist_root}" ]; then
  if [ -L "${dist_root}" ] || [ ! -d "${dist_root}" ]; then
    printf 'Archive output exists and is not a plain directory: %s\n' "${dist_root}" >&2
    exit 2
  fi
  shopt -s nullglob dotglob
  dist_entries=("${dist_root}"/*)
  shopt -u nullglob dotglob
  if [ "${#dist_entries[@]}" -ne 0 ]; then
    printf 'Archive output must be absent or empty: %s\n' "${dist_root}" >&2
    exit 2
  fi
else
  mkdir -- "${dist_root}"
fi
mkdir -- "${snapshot_root}"
shopt -s nullglob dotglob
snapshot_entries=("${snapshot_root}"/*)
shopt -u nullglob dotglob
if [ "${#snapshot_entries[@]}" -ne 0 ]; then
  printf 'Index snapshot must be fresh and empty: %s\n' "${snapshot_root}" >&2
  exit 2
fi

cd -- "${repo_root}"
git checkout-index --all --prefix="${snapshot_root%/}/"

for package in "${packages[@]}"; do
  package_snapshot="${snapshot_root}/packages/${package}"
  if [ ! -f "${package_snapshot}/pyproject.toml" ]; then
    printf 'Package metadata was not found in the staged index snapshot: %s\n' \
      "${package_snapshot}/pyproject.toml" >&2
    exit 1
  fi
  "${python_bin}" -m build --no-isolation --outdir "${dist_root}" "${package_snapshot}"
done

"${python_bin}" - "${dist_root}" <<'PY'
from pathlib import Path
import sys

root = Path(sys.argv[1])
entries = sorted(root.iterdir(), key=lambda path: path.name)
regular_files = [path for path in entries if path.is_file() and not path.is_symlink()]
wheels = [path for path in regular_files if path.suffix == ".whl"]
sdists = [path for path in regular_files if path.name.endswith(".tar.gz")]
if len(entries) != 16 or len(regular_files) != 16 or len(wheels) != 8 or len(sdists) != 8:
    raise SystemExit(
        "Expected exactly eight wheels and eight sdists as the only sixteen "
        f"regular archive entries; found {len(wheels)} wheels, {len(sdists)} "
        f"sdists, {len(regular_files)} regular files, and {len(entries)} total entries."
    )
PY

printf 'Release build workspace: %s\n' "${work_root}"
printf 'Release archive directory: %s\n' "${dist_root}"
