#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../../.." && pwd)"
venv_dir="${RP_DEV_VENV:-${repo_root}/.venv}"
case "${venv_dir}" in
  /*) ;;
  *) venv_dir="$(pwd -P)/${venv_dir}" ;;
esac
python_bin="${PYTHON_BIN:-python3}"
profile="${RP_BOOTSTRAP_PROFILE:-}"
profile_explicit=false
print_plan=false
wheelhouse="${RP_BOOTSTRAP_WHEELHOUSE:-}"
requirements_notebook="${repo_root}/ops/envs/dev/requirements-notebook.txt"
requirements_hpc="${RP_DEV_REQUIREMENTS:-${repo_root}/ops/envs/hpc/requirements-runtime.txt}"
smoke_script="${repo_root}/ops/envs/dev/smoke-check.sh"
dev_requirements=("pytest>=8")
hpc_import_check_code='import importlib, importlib.metadata, sys; [importlib.import_module(name) for name in sys.argv[1:]]; importlib.metadata.version("snakemake-executor-plugin-slurm")'
hpc_editable_check_code='import importlib.metadata as metadata, json, sys; from pathlib import Path; from urllib.parse import unquote, urlparse; pairs = zip(sys.argv[1::2], sys.argv[2::2]); records = [(json.loads(metadata.distribution(name).read_text("direct_url.json") or "{}"), Path(path).resolve()) for name, path in pairs]; raise SystemExit(any(not (data.get("dir_info") or {}).get("editable") or urlparse(data.get("url", "")).scheme != "file" or Path(unquote(urlparse(data["url"]).path)).resolve() != expected for data, expected in records))'
hpc_dependency_check_code='import importlib.metadata as metadata
import sys
import tomllib
from pathlib import Path
try:
    from packaging.requirements import Requirement
except ImportError:
    from pip._vendor.packaging.requirements import Requirement
requirements = []
for raw_line in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines():
    line = raw_line.strip()
    if line and not line.startswith("#"):
        requirements.append(line)
for pyproject_path in sys.argv[2:]:
    with Path(pyproject_path).open("rb") as handle:
        requirements.extend(tomllib.load(handle).get("project", {}).get("dependencies", []))
for requirement_text in requirements:
    requirement = Requirement(requirement_text)
    if requirement.marker is not None and not requirement.marker.evaluate():
        continue
    installed_version = metadata.version(requirement.name)
    if requirement.specifier and installed_version not in requirement.specifier:
        raise SystemExit(f"{requirement.name} {installed_version} does not satisfy {requirement.specifier}")'
python_version_check_code='import sys; raise SystemExit(sys.version_info < (3, 11))'
environment_isolation_check_code='from pathlib import Path; import sys; expected = Path(sys.argv[1]).resolve(); raise SystemExit(Path(sys.prefix).resolve() != expected or sys.prefix == sys.base_prefix)'

if [ -n "${profile}" ]; then
  profile_explicit=true
fi

usage() {
  cat <<'EOF'
Usage: ops/envs/dev/bootstrap.sh [OPTIONS]

Create or reuse a workspace virtual environment and install one dependency profile.

Options:
  --profile PROFILE       minimal (default), dev, full, or hpc
  --print-plan            Print the complete plan without changing files or running installers
  --help                  Show this help and exit

Environment:
  RP_DEV_VENV                   Virtual environment path (default: <repo>/.venv)
  PYTHON_BIN                    Python used to create a missing environment (default: python3)
  RP_BOOTSTRAP_PROFILE          Profile default when --profile is omitted
  RP_BOOTSTRAP_WHEELHOUSE       Local wheel or package-archive directory
  RP_DEV_REQUIREMENTS           HPC requirements file override

Inside a SLURM job, an omitted profile becomes hpc. An explicit non-hpc profile is rejected.
The hpc profile never uses a package index. Read-only reuse requires current editable workspace
packages, their declared dependencies, the scheduler requirements, and working pip metadata support.
EOF
}

require_option_value() {
  if [ "$#" -lt 2 ] || [ -z "${2}" ] || [[ "${2}" = -* ]]; then
    printf '%s requires a value.\n' "$1" >&2
    exit 2
  fi
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --profile)
      require_option_value "$@"
      profile="$2"
      profile_explicit=true
      shift 2
      ;;
    --print-plan)
      print_plan=true
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

in_slurm=false
if [ -n "${SLURM_JOB_ID:-}" ] || [ -n "${SLURM_CLUSTER_NAME:-}" ]; then
  in_slurm=true
fi
if [ "${in_slurm}" = true ]; then
  if [ "${profile_explicit}" = true ] && [ "${profile}" != "hpc" ]; then
    printf 'A SLURM job requires --profile hpc; refusing explicit profile %q.\n' "${profile}" >&2
    exit 2
  fi
  profile="hpc"
elif [ -z "${profile}" ]; then
  profile="minimal"
fi

case "${profile}" in
  minimal|dev|full|hpc) ;;
  *)
    printf 'Unsupported bootstrap profile: %s (choose minimal, dev, full, or hpc).\n' "${profile}" >&2
    exit 2
    ;;
esac

resolve_local_directory() {
  local label="$1"
  local value="$2"
  if [ ! -d "${value}" ]; then
    printf '%s was not found or is not a directory: %s\n' "${label}" "${value}" >&2
    exit 2
  fi
  (cd -- "${value}" && pwd -P)
}

if [ -n "${wheelhouse}" ]; then
  wheelhouse="$(resolve_local_directory "Wheelhouse" "${wheelhouse}")"
fi

minimal_package_names=(
  research-analysis
  research-bids
  research-core
  research-hpc
  research-io
  research-ml
  research-neuro
)
full_package_names=("${minimal_package_names[@]}" research-viz)
if [ "${profile}" = "hpc" ]; then
  selected_package_names=()
  for package_dir in "${repo_root}"/packages/research-*; do
    if [ -f "${package_dir}/pyproject.toml" ]; then
      selected_package_names+=("$(basename -- "${package_dir}")")
    fi
  done
  if [ "${#selected_package_names[@]}" -eq 0 ]; then
    printf 'No research-* workspace packages were found in the staged checkout.\n' >&2
    exit 1
  fi
elif [ "${profile}" = "full" ]; then
  selected_package_names=("${full_package_names[@]}")
else
  selected_package_names=("${minimal_package_names[@]}")
fi

hpc_import_names=(snakemake_executor_plugin_slurm)
hpc_command_names=(research-hpc snakemake)

editable_args=()
editable_paths=()
hpc_editable_check_args=()
hpc_pyproject_paths=()
for package_name in "${selected_package_names[@]}"; do
  package_dir="${repo_root}/packages/${package_name}"
  if [ ! -f "${package_dir}/pyproject.toml" ]; then
    printf 'Workspace package metadata was not found: packages/%s/pyproject.toml\n' "${package_name}" >&2
    exit 1
  fi
  editable_install_target="${package_dir}"
  if [ "${profile}" = "full" ]; then
    case "${package_name}" in
      research-analysis)
        editable_install_target="${package_dir}[rsatoolbox]"
        ;;
      research-bids|research-io)
        editable_install_target="${package_dir}[pandas]"
        ;;
      research-ml)
        editable_install_target="${package_dir}[xgboost]"
        ;;
    esac
  fi
  editable_args+=("-e" "${editable_install_target}")
  editable_paths+=("packages/${package_name}")
  hpc_editable_check_args+=("${package_name}" "${package_dir}")
  hpc_pyproject_paths+=("${package_dir}/pyproject.toml")
  package_module="${package_name#research-}"
  hpc_import_names+=("research_platform.${package_module//-/_}")
  case "${package_name}" in
    research-bids)
      hpc_command_names+=(research-bids)
      ;;
    research-core)
      hpc_command_names+=(rp)
      ;;
    research-io)
      hpc_command_names+=(research-io)
      ;;
  esac
done

offline_dependency_line_is_unsafe() {
  local line="$1"
  [[ "${line}" =~ :// ]] \
    || [[ "${line}" =~ ^[[:space:]]*(git|hg|svn|bzr)\+ ]] \
    || [[ "${line}" =~ ^[[:space:]]*(-r|--requirement|-c|--constraint|-e|--editable|--index-url|--extra-index-url|--find-links|--trusted-host|--proxy)([=[:space:]]|$) ]] \
    || [[ "${line}" =~ [[:space:]]@[[:space:]] ]]
}

reject_unsafe_offline_dependency() {
  local dependency_path="$1"
  local line_number="$2"
  printf 'HPC dependency inputs must contain only offline-resolvable package requirements: %s:%s\n' \
    "${dependency_path}" "${line_number}" >&2
  printf 'Put local archives in RP_BOOTSTRAP_WHEELHOUSE; remote URLs and nested installer options are not allowed.\n' >&2
  exit 2
}

validate_offline_requirements_file() {
  local dependency_path="$1"
  local line
  local line_number=0
  if [ ! -f "${dependency_path}" ]; then
    printf 'HPC dependency input was not found: %s\n' "${dependency_path}" >&2
    exit 2
  fi
  while IFS= read -r line || [ -n "${line}" ]; do
    line_number=$((line_number + 1))
    if offline_dependency_line_is_unsafe "${line}"; then
      reject_unsafe_offline_dependency "${dependency_path}" "${line_number}"
    fi
  done < "${dependency_path}"
}

validate_offline_pyproject_dependencies() {
  local dependency_path="$1"
  local line
  local line_number=0
  local section=""
  local capture=false
  if [ ! -f "${dependency_path}" ]; then
    printf 'HPC package metadata was not found: %s\n' "${dependency_path}" >&2
    exit 2
  fi
  while IFS= read -r line || [ -n "${line}" ]; do
    line_number=$((line_number + 1))
    if [[ "${line}" =~ ^[[:space:]]*\[([^][]+)\] ]]; then
      section="${BASH_REMATCH[1]}"
      capture=false
    fi
    if { [ "${section}" = "project" ] && [[ "${line}" =~ ^[[:space:]]*dependencies[[:space:]]*= ]]; } \
      || { [ "${section}" = "build-system" ] && [[ "${line}" =~ ^[[:space:]]*requires[[:space:]]*= ]]; }; then
      capture=true
    fi
    if [ "${capture}" = true ]; then
      if offline_dependency_line_is_unsafe "${line}"; then
        reject_unsafe_offline_dependency "${dependency_path}" "${line_number}"
      fi
      if [[ "${line}" =~ \][[:space:]]*(#.*)?$ ]]; then
        capture=false
      fi
    fi
  done < "${dependency_path}"
}

if [ "${profile}" = "hpc" ]; then
  validate_offline_requirements_file "${requirements_hpc}"
  for package_name in "${selected_package_names[@]}"; do
    validate_offline_pyproject_dependencies "${repo_root}/packages/${package_name}/pyproject.toml"
  done
fi

pip_source_args=()
if [ "${profile}" = "hpc" ]; then
  pip_source_args+=("--no-index")
fi
if [ -n "${wheelhouse}" ]; then
  pip_source_args+=("--find-links" "${wheelhouse}")
fi

venv_python="${venv_dir}/bin/python"
python_environment=(
  env
  -u PYTHONHOME
  -u PYTHONPATH
  -u PYTHONPYCACHEPREFIX
  PYTHONDONTWRITEBYTECODE=1
  PYTHONNOUSERSITE=1
)
local_pip_command=(
  env
  -u PYTHONHOME
  -u PYTHONPATH
  -u PYTHONPYCACHEPREFIX
  -u PIP_REQUIREMENT
  -u PIP_CONSTRAINT
  -u PIP_EDITABLE
  -u PIP_TARGET
  -u PIP_PREFIX
  -u PIP_USER
  -u PIP_ROOT
  PYTHONDONTWRITEBYTECODE=1
  PYTHONNOUSERSITE=1
  PIP_REQUIRE_VIRTUALENV=1
  "${venv_python}" -m pip
)
hpc_pip_environment=(
  env
  -u PYTHONHOME
  -u PYTHONPATH
  -u PYTHONPYCACHEPREFIX
  -u PIP_FIND_LINKS
  -u PIP_INDEX_URL
  -u PIP_EXTRA_INDEX_URL
  -u PIP_TRUSTED_HOST
  -u PIP_CONFIG_FILE
  -u PIP_REQUIREMENT
  -u PIP_CONSTRAINT
  -u PIP_EDITABLE
  -u PIP_TARGET
  -u PIP_PREFIX
  -u PIP_USER
  -u PIP_ROOT
  -u PIP_REQUIRE_VIRTUALENV
  PIP_CONFIG_FILE=/dev/null
  PIP_NO_INDEX=1
  PIP_NO_CACHE_DIR=1
  PIP_DISABLE_PIP_VERSION_CHECK=1
  PYTHONDONTWRITEBYTECODE=1
  PYTHONNOUSERSITE=1
)
hpc_pip_command=("${hpc_pip_environment[@]}" PIP_REQUIRE_VIRTUALENV=1 "${venv_python}" -m pip)
hpc_check_environment=("${hpc_pip_environment[@]}")

venv_action="create"
if [ -x "${venv_python}" ]; then
  venv_action="reuse"
fi

print_command() {
  printf '  '
  printf '%q ' "$@"
  printf '\n'
}

pip_install() {
  local action="$1"
  local command_scope="$2"
  shift 2
  local install_command=()
  case "${command_scope}" in
    hpc)
      install_command=("${hpc_pip_command[@]}" install)
      ;;
    local)
      install_command=("${local_pip_command[@]}" install)
      ;;
    *)
      printf 'Unsupported pip command scope: %s\n' "${command_scope}" >&2
      return 2
      ;;
  esac
  if [ "${#pip_source_args[@]}" -gt 0 ]; then
    install_command+=("${pip_source_args[@]}")
  fi
  install_command+=("$@")
  case "${action}" in
    print)
      print_command "${install_command[@]}"
      ;;
    run)
      "${install_command[@]}"
      ;;
    *)
      printf 'Unsupported pip install action: %s\n' "${action}" >&2
      return 2
      ;;
  esac
}

print_hpc_runtime_checks() {
  print_command test -x "${venv_python}"
  print_command "${hpc_check_environment[@]}" "${venv_python}" -c "${python_version_check_code}"
  print_command "${hpc_check_environment[@]}" "${venv_python}" -c "${hpc_dependency_check_code}" "${requirements_hpc}" "${hpc_pyproject_paths[@]}"
  print_command "${hpc_check_environment[@]}" "${venv_python}" -c "${hpc_import_check_code}" "${hpc_import_names[@]}"
  print_command "${hpc_check_environment[@]}" "${venv_python}" -c "${hpc_editable_check_code}" "${hpc_editable_check_args[@]}"
  for command_name in "${hpc_command_names[@]}"; do
    print_command "${hpc_check_environment[@]}" "${venv_dir}/bin/${command_name}" --help
  done
}

print_bootstrap_plan() {
  printf 'Bootstrap plan\n'
  printf '  profile: %s\n' "${profile}"
  if [ "${profile}" = "hpc" ] && [ -z "${wheelhouse}" ]; then
    printf '  runtime prefix: %s (inspect only)\n' "${venv_dir}"
  elif [ "${venv_action}" = "reuse" ]; then
    printf '  virtual environment: %s (reuse candidate)\n' "${venv_dir}"
  else
    printf '  virtual environment: %s (%s)\n' "${venv_dir}" "${venv_action}"
  fi
  if [ "${venv_action}" = "reuse" ] && ! { [ "${profile}" = "hpc" ] && [ -z "${wheelhouse}" ]; }; then
    printf '  reuse policy: additive after preflight (existing packages are not removed)\n'
  fi
  if [ "${profile}" = "hpc" ]; then
    printf '  package index: disabled\n'
    if [ -z "${wheelhouse}" ]; then
      printf '  execution prerequisite: an already usable environment or RP_BOOTSTRAP_WHEELHOUSE\n'
    fi
  else
    printf '  package index: permitted during installation\n'
  fi
  printf '  editable packages:\n'
  printf '    - %s\n' "${editable_paths[@]}"
  printf '  additional requirements:\n'
  case "${profile}" in
    minimal)
      printf '    - none (package metadata supplies the minimal runtime)\n'
      ;;
    dev)
      printf '    - %s\n' "${dev_requirements[@]}"
      ;;
    full)
      printf '    - %s\n' "${dev_requirements[@]}"
      printf '    - research-analysis[rsatoolbox]\n'
      printf '    - research-bids[pandas]\n'
      printf '    - research-io[pandas]\n'
      printf '    - research-ml[xgboost]\n'
      printf '    - %s\n' "${requirements_notebook}"
      ;;
    hpc)
      printf '    - %s\n' "${requirements_hpc}"
      ;;
  esac
  if [ "${profile}" = "hpc" ] && [ -z "${wheelhouse}" ]; then
    printf '  mode: read-only reuse check; no installation will be attempted\n'
    printf '  checks:\n'
    print_hpc_runtime_checks
    printf '  outcome: reuse when every check passes; otherwise stop and request RP_BOOTSTRAP_WHEELHOUSE\n'
    return
  fi
  printf '  commands:\n'
  if [ "${venv_action}" = "reuse" ]; then
    print_command test -f "${venv_dir}/pyvenv.cfg"
    print_command "${python_environment[@]}" "${venv_python}" -c "${environment_isolation_check_code}" "${venv_dir}"
    print_command "${python_environment[@]}" "${venv_python}" -c "${python_version_check_code}"
  else
    print_command "${python_environment[@]}" "${python_bin}" -c "${python_version_check_code}"
    print_command "${python_environment[@]}" "${python_bin}" -m venv "${venv_dir}"
    print_command test -f "${venv_dir}/pyvenv.cfg"
    print_command "${python_environment[@]}" "${venv_python}" -c "${environment_isolation_check_code}" "${venv_dir}"
  fi
  if [ "${profile}" = "hpc" ]; then
    pip_install print hpc --no-cache-dir --upgrade setuptools wheel
  else
    pip_install print local --upgrade pip setuptools wheel
  fi
  if [ "${profile}" = "hpc" ]; then
    pip_install print hpc --no-cache-dir --no-build-isolation "${editable_args[@]}"
  else
    pip_install print local --no-build-isolation "${editable_args[@]}"
  fi
  case "${profile}" in
    dev)
      pip_install print local "${dev_requirements[@]}"
      ;;
    full)
      pip_install print local "${dev_requirements[@]}"
      pip_install print local -r "${requirements_notebook}"
      ;;
    hpc)
      pip_install print hpc --no-cache-dir -r "${requirements_hpc}"
      ;;
  esac
  if [ "${profile}" = "hpc" ]; then
    print_hpc_runtime_checks
  else
    print_command bash "${smoke_script}" --venv "${venv_dir}"
  fi
}

if [ "${print_plan}" = true ]; then
  print_bootstrap_plan
  exit 0
fi

environment_is_isolated() {
  [ -f "${venv_dir}/pyvenv.cfg" ] \
    && "${python_environment[@]}" "${venv_python}" -c \
      "${environment_isolation_check_code}" \
      "${venv_dir}" >/dev/null 2>&1
}

hpc_runtime_is_usable() {
  [ -x "${venv_python}" ] || return 1
  "${hpc_check_environment[@]}" "${venv_python}" -c \
    "${python_version_check_code}" >/dev/null 2>&1 || return 1
  "${hpc_check_environment[@]}" "${venv_python}" -c \
    "${hpc_dependency_check_code}" "${requirements_hpc}" "${hpc_pyproject_paths[@]}" >/dev/null 2>&1 || return 1
  "${hpc_check_environment[@]}" "${venv_python}" -c \
    "${hpc_import_check_code}" "${hpc_import_names[@]}" >/dev/null 2>&1 || return 1
  "${hpc_check_environment[@]}" "${venv_python}" -c \
    "${hpc_editable_check_code}" "${hpc_editable_check_args[@]}" >/dev/null 2>&1 || return 1
  for command_name in "${hpc_command_names[@]}"; do
    [ -x "${venv_dir}/bin/${command_name}" ] || return 1
    "${hpc_check_environment[@]}" "${venv_dir}/bin/${command_name}" --help >/dev/null 2>&1 || return 1
  done
}

if [ "${profile}" = "hpc" ] && [ -z "${wheelhouse}" ] && hpc_runtime_is_usable; then
    printf 'Reusing usable offline HPC environment: %s\n' "${venv_dir}"
    exit 0
fi

if [ "${profile}" = "hpc" ] && [ -z "${wheelhouse}" ]; then
  cat >&2 <<EOF
HPC bootstrap cannot install packages without a local source.
Set RP_BOOTSTRAP_WHEELHOUSE to a local wheel or package-archive directory,
or point RP_DEV_VENV at an existing environment with working pip metadata, current editable
workspace packages, and scheduler commands that pass the HPC runtime check.
No installation was attempted.
EOF
  exit 2
fi

preflight_python="${python_bin}"
if [ "${venv_action}" = "reuse" ]; then
  if ! environment_is_isolated; then
    printf 'Refusing to modify a non-isolated Python environment: %s\n' "${venv_dir}" >&2
    printf 'Set RP_DEV_VENV to a Python virtual environment with its own pyvenv.cfg.\n' >&2
    exit 2
  fi
  preflight_python="${venv_python}"
fi
if ! "${python_environment[@]}" "${preflight_python}" -c "${python_version_check_code}"; then
  printf 'Bootstrap requires Python 3.11 or newer: %s\n' "${preflight_python}" >&2
  exit 1
fi

printf 'Bootstrapping development environment (profile=%s)\n' "${profile}"
if [ "${venv_action}" = "reuse" ]; then
  printf 'Reusing virtual environment: %s\n' "${venv_dir}"
else
  printf 'Creating virtual environment: %s\n' "${venv_dir}"
  "${python_environment[@]}" "${python_bin}" -m venv "${venv_dir}"
fi

if ! environment_is_isolated; then
  printf 'Bootstrap did not produce an isolated Python virtual environment: %s\n' "${venv_dir}" >&2
  exit 1
fi

if [ "${profile}" = "hpc" ]; then
  pip_install run hpc --no-cache-dir --upgrade setuptools wheel
else
  pip_install run local --upgrade pip setuptools wheel
fi

if [ "${profile}" = "hpc" ]; then
  pip_install run hpc --no-cache-dir --no-build-isolation "${editable_args[@]}"
else
  pip_install run local --no-build-isolation "${editable_args[@]}"
fi
case "${profile}" in
  dev)
    pip_install run local "${dev_requirements[@]}"
    ;;
  full)
    pip_install run local "${dev_requirements[@]}"
    pip_install run local -r "${requirements_notebook}"
    ;;
  hpc)
    pip_install run hpc --no-cache-dir -r "${requirements_hpc}"
    ;;
esac

if [ "${profile}" = "hpc" ]; then
  if ! hpc_runtime_is_usable; then
    printf 'HPC runtime check failed; expected the staged package closure and scheduler requirements in %s.\n' "${venv_dir}" >&2
    exit 1
  fi
  printf 'HPC runtime check passed.\n'
else
  bash "${smoke_script}" --venv "${venv_dir}"
fi

cat <<EOF
Bootstrap complete (profile=${profile}).
Activate the environment with:
  source "${venv_dir}/bin/activate"
EOF
