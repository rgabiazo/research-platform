# research-ml

`research-ml` provides the estimators and metrics used by the bounded synthetic
classification and regression workflows.

## Current status

This is an unreleased `0.1.0a1` source-checkout alpha for Python 3.11 and 3.12.
It is an internal package-level interface used by `research-analysis` and `rp`;
it has no standalone CLI and is not a stable model service or unified SDK.

The checked-in logistic-classification and ElasticNet-regression paths are
tested on synthetic tabular data. The
[capability matrix](../../docs/capabilities.md) defines their exact public
status and limitations.

## Responsibilities

`research-ml` owns:

- binary logistic-regression fit, probability prediction, and evaluation;
- ElasticNet regression fit, prediction, and evaluation;
- mean squared error and R² metrics;
- deterministic bootstrap intervals for R²;
- an optional XGBoost regression adapter with explicit dependency handling.

It does not own:

- table reading, writing, or merging;
- train/test split construction or leakage-safe preprocessing;
- protocol-neutral crossnobis or RDM mathematics;
- project configuration or run orchestration;
- model registries, hosted inference, deployment, experiment tracking, or
  monitoring.

The placeholder `infer.py` and `registry.py` modules do not provide inference
service or registry functionality.

## Source-checkout use

Install the coordinated workspace through the repository
[quickstart](../../docs/onboarding/quickstart.md). The base package depends on
`scikit-learn>=1.4,<2`. The optional `xgboost` extra adds
`xgboost>=2,<3`.

The optional XGBoost code path does not broaden the checked-in public example:
the unit evidence verifies clear failure when the dependency is absent, while
the primary continuous-target example uses ElasticNet.

## Tested metrics example

```python
from research_platform.ml.metrics import mean_squared_error, r2_score

actuals = [1.0, 2.0, 3.0, 4.0]
predictions = [1.0, 2.0, 3.0, 5.0]

assert mean_squared_error(actuals=actuals, predictions=predictions) == 0.25
assert r2_score(actuals=actuals, predictions=predictions) == 0.8
```

Estimator helpers accept numeric feature-row mappings, explicit ordered feature
columns, targets, model settings, and caller-supplied table metadata. They
return JSON-compatible model and evaluation dictionaries containing model
parameters, metrics, predictions, residuals or confusion counts, and provenance
fields. The caller owns split integrity, persistence, access control, and any
publication decision.

## Evidence

- [`test_regression.py`](tests/unit/test_regression.py) covers metric values,
  deterministic bootstrap behaviour, synthetic ElasticNet fit/evaluation, and
  the missing-XGBoost error.
- [`research-analysis/test_tabular_cli.py`](../research-analysis/tests/unit/test_tabular_cli.py)
  exercises the package through synthetic logistic-classification and
  ElasticNet-regression lifecycles.
- [`research-core/test_tabular_transaction_cli.py`](../research-core/tests/unit/test_tabular_transaction_cli.py)
  exercises the integrated local model workflow.

## Limitations

- The public examples cover small synthetic classification and regression
  problems, not broad model selection or production prediction.
- No registry, inference service, deployment API, experiment tracker, or model
  monitoring system is implemented.
- LDA, SVM, nonlinear decoding, AutoML, and arbitrary estimator compatibility
  are not current support claims.
- Split construction and leakage prevention remain analysis responsibilities.
- Crossnobis and RDM mathematics remain in `research-analysis`.
- Package-level Python interfaces are alpha and no unified stable SDK is
  claimed.

See the [repository overview](../../README.md),
[architecture](../../ARCHITECTURE.md),
[capability matrix](../../docs/capabilities.md), and
[tabular guide](../../docs/tabular-slice.md).
