#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../.." && pwd)"

if [ "${1:-}" = "--print-plan" ]; then
  if [ "$#" -ne 1 ]; then
    printf 'Usage: %s [--print-plan]\n' "${BASH_SOURCE[0]}" >&2
    exit 2
  fi
  cat <<'EOF'
git diff --exit-code
git diff --cached --exit-code
git status --porcelain=v1 --untracked-files=all
scan for repository-local environments, caches, package build residue, coverage output, transaction claims, and temporary runtime/publication trees
EOF
  exit 0
fi
if [ "$#" -ne 0 ]; then
  printf 'Usage: %s [--print-plan]\n' "${BASH_SOURCE[0]}" >&2
  exit 2
fi

cd -- "${repo_root}"
git diff --exit-code
git diff --cached --exit-code
status="$(git status --porcelain=v1 --untracked-files=all)"
if [ -n "${status}" ]; then
  printf 'The checkout contains tracked or untracked changes:\n%s\n' "${status}" >&2
  exit 1
fi

generated="$({
  find . -path ./.git -prune -o \( \
    -name .venv -o \
    -name __pycache__ -o \
    -name .pytest_cache -o \
    -name '*.egg-info' -o \
    -name htmlcov \
  \) -print
  for path in packages/research-*/build packages/research-*/dist; do
    if [ -e "${path}" ] || [ -L "${path}" ]; then
      printf '%s\n' "${path}"
    fi
  done
  find . -path ./.git -prune -o \( \
    -name .coverage -o \
    -name '.coverage.*' -o \
    -name coverage.xml -o \
    -name coverage.json -o \
    -name '*.claim' \
  \) -print
  find . -path ./.git -prune -o \( \
    -name '.*.publication-*' -o \
    -name '.*.transaction-*' -o \
    -name '.*.staging-*' \
  \) -print
} | LC_ALL=C sort -u)"
if [ -n "${generated}" ]; then
  printf 'Generated checkout-local files or directories were found:\n%s\n' "${generated}" >&2
  exit 1
fi

printf 'Checkout is clean and contains no prohibited generated residue.\n'
