#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../../.." && pwd)"
venv_dir="${RP_DEV_VENV:-${repo_root}/.venv}"
quiet=false

usage() {
  cat <<'EOF'
Usage: ops/envs/dev/smoke-check.sh [--venv PATH] [--quiet]

Run read-only installation checks against the selected virtual environment.
The checks disable Python bytecode and pip index access and do not create caches or artifacts.
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --venv)
      if [ "$#" -lt 2 ] || [ -z "$2" ] || [[ "$2" = -* ]]; then
        printf '%s\n' '--venv requires a path.' >&2
        exit 2
      fi
      venv_dir="$2"
      shift 2
      ;;
    --quiet)
      quiet=true
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      printf 'Unknown option: %s\n' "$1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

case "${venv_dir}" in
  /*) ;;
  *) venv_dir="$(pwd -P)/${venv_dir}" ;;
esac

bin_dir="${venv_dir}/bin"
python_bin="${bin_dir}/python"
required_commands=(rp research-io research-bids research-hpc)
if [ ! -x "${python_bin}" ]; then
  printf 'Installation smoke check could not find an executable Python: %s\n' "${python_bin}" >&2
  exit 1
fi
for command_name in "${required_commands[@]}"; do
  if [ ! -x "${bin_dir}/${command_name}" ]; then
    printf 'Installation smoke check could not find %s in %s.\n' "${command_name}" "${bin_dir}" >&2
    exit 1
  fi
done

export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_NO_INDEX=1
unset PYTHONHOME PYTHONPATH PYTHONPYCACHEPREFIX

run_check() {
  if [ "${quiet}" = false ]; then
    printf '$ '
    printf '%q ' "$@"
    printf '\n'
  fi
  "$@"
}

cd -- "${repo_root}"
run_check "${bin_dir}/rp" --help
run_check "${bin_dir}/rp" --version
run_check "${bin_dir}/rp" setup
run_check "${bin_dir}/rp" config validate --project project-pilot-tabular
run_check "${bin_dir}/research-io" --help
run_check "${bin_dir}/research-bids" --help
run_check "${bin_dir}/research-hpc" --help
run_check "${python_bin}" -m research_platform.analysis.cli --help

if [ "${quiet}" = false ]; then
  printf 'Installation smoke check passed.\n'
fi
