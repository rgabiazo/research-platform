
PYTHON ?= python3

.PHONY: tree docs clean bootstrap-plan install-smoke test-bootstrap test-ci-contract

tree:
	find . -maxdepth 4 | sort

docs:
	@echo "Read README.md, ARCHITECTURE.md, ROADMAP.md, and docs/"

bootstrap-plan:
	bash ops/envs/dev/bootstrap.sh --print-plan --profile minimal

install-smoke:
	bash ops/envs/dev/smoke-check.sh

test-bootstrap:
	PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $(PYTHON) -m pytest -p no:cacheprovider ops/tests/test_bootstrap.py ops/tests/test_install_smoke.py

test-ci-contract:
	PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $(PYTHON) -m pytest -p no:cacheprovider ops/tests/test_ci_contract.py

clean:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
