#!/usr/bin/env sh
set -eu

if [ -n "${RP_EXECUTION_PROFILE:-}" ]; then
  printf '%s\n' "${RP_EXECUTION_PROFILE}"
  exit 0
fi

if [ -n "${CI:-}" ]; then
  printf 'ci\n'
  exit 0
fi

if [ -n "${SLURM_JOB_ID:-}" ] || [ -n "${SLURM_CLUSTER_NAME:-}" ]; then
  printf 'hpc\n'
  exit 0
fi

if [ -n "${CODEX_HOME:-}" ]; then
  printf 'codex\n'
  exit 0
fi

printf 'local\n'
