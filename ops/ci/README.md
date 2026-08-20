# Continuous integration

The public-alpha workflow verifies the supported source-checkout contract. It
does not publish packages, deploy software, contact research infrastructure, or
run external neuroimaging tools. GitHub-hosted jobs use only the reviewed
checkout and Python-setup actions. Network-backed package installation is
limited to each job's explicit bootstrap or build-tool installation phase;
validation then runs with `PIP_NO_INDEX=1`.

All commands below are repository-root commands. Temporary environments and
archives belong outside the checkout. Pick a fresh directory for each run:

```bash
set -euo pipefail
TMP_BASE="${TMPDIR:-/tmp}"
CI_TEMP_ROOT="$(mktemp -d "${TMP_BASE%/}/research-platform-ci.XXXXXX")"
export CI_TEMP_ROOT
```

Hosted bootstrap and test jobs copy the complete, uncurated source checkout to
a unique directory beneath `RUNNER_TEMP` and work from that copy. Editable
installation can create ignored `*.egg-info` or package build metadata, so this
boundary leaves the original `GITHUB_WORKSPACE` read-only for the mandatory
final clean-checkout guard without hiding or deleting generated files. After
installation, `ops/ci/source-tree-integrity.py` records every path in the tested
copy except `.git`, including ignored installer metadata. It records types,
modes, file SHA-256 values, and symbolic-link targets in a baseline outside the
copy. An always-run comparison rejects any validation- or test-phase change.

## Package-by-package tests

The development profile installs the seven packages in the normal development
contract. CI then installs only the local `research-viz` source, without
resolving dependencies, so it can test all eight package trees without using
the heavyweight `full` profile:

```bash
set -euo pipefail
export RP_DEV_VENV="${CI_TEMP_ROOT}/dev-venv"
PYTHON_BIN=python3 \
  bash ops/envs/dev/bootstrap.sh --profile dev
"${RP_DEV_VENV}/bin/python" -m pip install \
  --no-deps \
  --no-build-isolation \
  -e packages/research-viz
export PIP_NO_INDEX=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export PIP_DISABLE_PIP_VERSION_CHECK=1
PYTHON_BIN="${RP_DEV_VENV}/bin/python" \
  bash ops/ci/run-package-tests.sh
```

The runner invokes these paths as eight separate pytest processes, in package
order, each with `-p no:cacheprovider`:

```text
packages/research-analysis/tests
packages/research-bids/tests
packages/research-core/tests
packages/research-hpc/tests
packages/research-io/tests
packages/research-ml/tests
packages/research-neuro/tests
packages/research-viz/tests
```

This separation is required because some package trees contain duplicate test
module basenames. Optional full-profile coverage remains a later
release-candidate gate.

## Public contracts

Using a populated `dev` environment, run the operational contract tests, all
four deterministic fixture checks, and validation of exactly the four public
project overlays:

```bash
set -euo pipefail
export PIP_NO_INDEX=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export PIP_DISABLE_PIP_VERSION_CHECK=1
PYTHON_BIN="${RP_DEV_VENV}/bin/python" \
RP_BIN="${RP_DEV_VENV}/bin/rp" \
  bash ops/ci/run-public-contracts.sh
```

The script's pytest process uses `-p no:cacheprovider`. Its remaining commands
are the four `ops/scripts/generate_toy_*_fixtures.py --check` invocations and:

```text
rp config validate --project project-template
rp config validate --project project-example
rp config validate --project project-pilot-bids
rp config validate --project project-pilot-tabular
```

## Release archives

Archive validation uses an isolated build environment. Installing this
CI-only tool closure is the one network-permitted phase in this local
equivalent; it does not change package runtime dependencies:

```bash
set -euo pipefail
BUILD_VENV="${CI_TEMP_ROOT}/build-venv"
python3 -m venv "${BUILD_VENV}"
"${BUILD_VENV}/bin/python" -m pip install \
  "build==1.5.0" \
  "setuptools>=69" \
  wheel \
  "pytest>=8"
export PIP_NO_INDEX=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export PIP_DISABLE_PIP_VERSION_CHECK=1
export RP_RELEASE_DIST_DIR="${CI_TEMP_ROOT}/release-dist"
PYTHON_BIN="${BUILD_VENV}/bin/python" \
  bash ops/ci/build-release-archives.sh
RP_RELEASE_WHEEL_DIR="${RP_RELEASE_DIST_DIR}" \
RP_RELEASE_SDIST_DIR="${RP_RELEASE_DIST_DIR}" \
  "${BUILD_VENV}/bin/python" -m pytest \
    -p no:cacheprovider \
    ops/tests/test_release_metadata.py
"${BUILD_VENV}/bin/python" ops/ci/inspect-release-archives.py \
  --archive-dir "${RP_RELEASE_DIST_DIR}"
```

The builder uses `git checkout-index --all` to materialize the complete staged
Git index in a fresh temporary directory, then runs `python -m build
--no-isolation` against each of its eight tracked package directories. It never
copies live package directories, so ignored or untracked build residue cannot
enter the source boundary. For a local pre-commit check, stage the intended
snapshot before invoking the builder. The expected result is exactly eight
wheels and eight source distributions as the only sixteen regular output
entries. The metadata tests and inspector verify identity, licenses, paths,
contents, and SHA-256 digests. CI does not upload or retain these archives as
workflow artifacts.

## Workflow and checkout contracts

Run the focused static workflow contract with an already populated development
interpreter:

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONNOUSERSITE=1 \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  python3 -m pytest -p no:cacheprovider ops/tests/test_ci_contract.py
```

The four jobs that execute from temporary source copies capture and verify that
copy with:

```bash
python ops/ci/source-tree-integrity.py capture \
  --root "${CI_WORKSPACE}" \
  --output "${CI_SOURCE_BASELINE}"
python ops/ci/source-tree-integrity.py verify \
  --root "${CI_WORKSPACE}" \
  --baseline "${CI_SOURCE_BASELINE}"
```

Every hosted job also finishes by running the same read-only original-checkout
guard:

```bash
bash ops/ci/check-clean-checkout.sh
```

That guard verifies the index, tracked worktree, and untracked-file state and
fails on repository-local environments, caches, package build residue,
coverage output, or transaction remnants. It reports contamination rather
than deleting it. Run it from an otherwise clean checkout; intentional local
changes will correctly make it fail.
