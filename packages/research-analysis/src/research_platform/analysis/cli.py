from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Callable, TypeVar

from research_platform.ml.evaluate import evaluate_logistic_regression, evaluate_regression_model
from research_platform.ml.train import fit_logistic_regression, fit_regression_model

from ._tabular import infer_numeric_feature_columns, numeric_value, read_json, read_rows, write_json, write_rows
from .prep import apply_standardization_plan, fit_standardization_plan
from .splits import create_split_manifest, load_split_manifest, split_membership, write_split_manifest
from .statistics import anova_report, correlation_report, linear_model_report, mixed_effects_report, summary_table_report


TargetValue = TypeVar("TargetValue", int, float)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Split-aware tabular analysis CLI.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    split = subparsers.add_parser("split", help="Split manifest operations.")
    split_subparsers = split.add_subparsers(dest="split_command", required=True)
    split_create = split_subparsers.add_parser("create", help="Create a train/test split manifest.")
    split_create.add_argument("--table", required=True, help="Canonical feature table path.")
    split_create.add_argument("--expected-table-sha256", help="Optional expected SHA-256 of the exact table bytes.")
    split_create.add_argument("--target-column", required=True, help="Target column used for stratification.")
    split_create.add_argument("--test-fraction", type=float, default=0.25, help="Fraction assigned to test.")
    split_create.add_argument("--seed", type=int, default=23, help="Split seed.")
    split_create.add_argument(
        "--strategy",
        default="stratified_binary",
        choices=["random", "stratified_binary", "stratified_binned"],
        help="Train/test split strategy.",
    )
    split_create.add_argument(
        "--stratify-bin-count",
        type=int,
        default=5,
        help="Quantile bin count for stratified_binned splits.",
    )
    split_create.add_argument("--output", required=True, help="Output split manifest path.")
    split_create.set_defaults(handler=_run_split_create)

    prep = subparsers.add_parser("prep", help="Preprocessing plan operations.")
    prep_subparsers = prep.add_subparsers(dest="prep_command", required=True)
    prep_fit = prep_subparsers.add_parser("fit", help="Fit a preprocessing plan on the training split.")
    prep_fit.add_argument("--table", required=True, help="Canonical feature table path.")
    prep_fit.add_argument("--split", required=True, help="Split manifest path.")
    prep_fit.add_argument("--expected-table-sha256", help="Optional expected SHA-256 of the exact table bytes.")
    prep_fit.add_argument("--expected-split-sha256", help="Optional expected SHA-256 of the exact split bytes.")
    prep_fit.add_argument("--target-column", required=True, help="Target column excluded from feature inference.")
    prep_fit.add_argument("--feature-columns", nargs="+", help="Optional explicit feature columns.")
    prep_fit.add_argument("--output", required=True, help="Output preprocessing plan path.")
    prep_fit.set_defaults(handler=_run_prep_fit)

    prep_apply = prep_subparsers.add_parser("apply", help="Apply a preprocessing plan.")
    prep_apply.add_argument("--table", required=True, help="Canonical feature table path.")
    prep_apply.add_argument("--plan", required=True, help="Preprocessing plan path.")
    prep_apply.add_argument("--split", help="Optional split manifest to annotate output rows.")
    prep_apply.add_argument("--expected-table-sha256", help="Optional expected SHA-256 of the exact table bytes.")
    prep_apply.add_argument("--expected-plan-sha256", help="Optional expected SHA-256 of the exact plan bytes.")
    prep_apply.add_argument("--expected-split-sha256", help="Optional expected SHA-256 of the exact split bytes.")
    prep_apply.add_argument("--output", required=True, help="Output transformed table path.")
    prep_apply.set_defaults(handler=_run_prep_apply)

    model = subparsers.add_parser("model", help="Model operations.")
    model_subparsers = model.add_subparsers(dest="model_command", required=True)
    model_train = model_subparsers.add_parser("train", help="Train a logistic regression model.")
    model_train.add_argument("--table", required=True, help="Preprocessed feature table path.")
    model_train.add_argument("--split", required=True, help="Split manifest path.")
    model_train.add_argument("--expected-table-sha256", help="Optional expected SHA-256 of the exact table bytes.")
    model_train.add_argument("--expected-split-sha256", help="Optional expected SHA-256 of the exact split bytes.")
    model_train.add_argument(
        "--table-reference",
        help="Optional metadata-only table reference; the input is still read from --table.",
    )
    model_train.add_argument("--target-column", required=True, help="Binary target column.")
    model_train.add_argument("--feature-columns", nargs="+", help="Optional explicit feature columns.")
    model_train.add_argument("--kind", default="logistic_regression", help="Only logistic_regression is supported.")
    model_train.add_argument("--learning-rate", type=float, default=0.2)
    model_train.add_argument("--iterations", type=int, default=350)
    model_train.add_argument("--output", required=True, help="Output model manifest path.")
    model_train.set_defaults(handler=_run_model_train)

    model_evaluate = model_subparsers.add_parser("evaluate", help="Evaluate a logistic regression model.")
    model_evaluate.add_argument("--table", required=True, help="Preprocessed feature table path.")
    model_evaluate.add_argument("--split", required=True, help="Split manifest path.")
    model_evaluate.add_argument("--target-column", required=True, help="Binary target column.")
    model_evaluate.add_argument("--model", required=True, help="Trained model manifest path.")
    model_evaluate.add_argument("--expected-table-sha256", help="Optional expected SHA-256 of the exact table bytes.")
    model_evaluate.add_argument("--expected-split-sha256", help="Optional expected SHA-256 of the exact split bytes.")
    model_evaluate.add_argument("--expected-model-sha256", help="Optional expected SHA-256 of the exact model bytes.")
    model_evaluate.add_argument("--output", required=True, help="Output evaluation report path.")
    model_evaluate.set_defaults(handler=_run_model_evaluate)

    regression = subparsers.add_parser("regression", help="Regression model operations.")
    regression_subparsers = regression.add_subparsers(dest="regression_command", required=True)

    regression_train = regression_subparsers.add_parser("train", help="Train a regression model.")
    regression_train.add_argument("--table", required=True, help="Preprocessed feature table path.")
    regression_train.add_argument("--split", required=True, help="Split manifest path.")
    regression_train.add_argument("--expected-table-sha256", help="Optional expected SHA-256 of the exact table bytes.")
    regression_train.add_argument("--expected-split-sha256", help="Optional expected SHA-256 of the exact split bytes.")
    regression_train.add_argument(
        "--table-reference",
        help="Optional metadata-only table reference; the input is still read from --table.",
    )
    regression_train.add_argument("--target-column", required=True, help="Continuous target column.")
    regression_train.add_argument("--feature-columns", nargs="+", help="Optional explicit feature columns.")
    regression_train.add_argument(
        "--kind",
        default="elastic_net_regression",
        choices=["elastic_net_regression", "xgboost_regression"],
        help="Regression model kind.",
    )
    regression_train.add_argument("--alpha", type=float, default=1.0, help="ElasticNet alpha penalty.")
    regression_train.add_argument("--l1-ratio", type=float, default=0.5, help="ElasticNet L1/L2 mixing.")
    regression_train.add_argument("--max-iter", type=int, default=1000, help="ElasticNet max iterations.")
    regression_train.add_argument("--random-state", type=int, default=23, help="Model random seed.")
    regression_train.add_argument("--learning-rate", type=float, default=0.1, help="XGBoost learning rate.")
    regression_train.add_argument("--n-estimators", type=int, default=200, help="XGBoost tree count.")
    regression_train.add_argument("--max-depth", type=int, default=6, help="XGBoost max tree depth.")
    regression_train.add_argument("--subsample", type=float, default=1.0, help="XGBoost row subsample ratio.")
    regression_train.add_argument(
        "--colsample-bytree",
        type=float,
        default=1.0,
        help="XGBoost feature subsample ratio per tree.",
    )
    regression_train.add_argument("--output", required=True, help="Output regression model manifest path.")
    regression_train.set_defaults(handler=_run_regression_train)

    regression_evaluate = regression_subparsers.add_parser("evaluate", help="Evaluate a regression model.")
    regression_evaluate.add_argument("--table", required=True, help="Preprocessed feature table path.")
    regression_evaluate.add_argument("--split", required=True, help="Split manifest path.")
    regression_evaluate.add_argument("--target-column", required=True, help="Continuous target column.")
    regression_evaluate.add_argument("--model", required=True, help="Trained model manifest path.")
    regression_evaluate.add_argument("--expected-table-sha256", help="Optional expected SHA-256 of the exact table bytes.")
    regression_evaluate.add_argument("--expected-split-sha256", help="Optional expected SHA-256 of the exact split bytes.")
    regression_evaluate.add_argument("--expected-model-sha256", help="Optional expected SHA-256 of the exact model bytes.")
    regression_evaluate.add_argument(
        "--bootstrap-iterations",
        type=int,
        default=1000,
        help="Bootstrap iterations for the R2 confidence interval.",
    )
    regression_evaluate.add_argument("--bootstrap-seed", type=int, default=23, help="Bootstrap random seed.")
    regression_evaluate.add_argument("--output", required=True, help="Output evaluation report path.")
    regression_evaluate.set_defaults(handler=_run_regression_evaluate)

    stats = subparsers.add_parser("stats", help="Statistical analysis operations.")
    stats_subparsers = stats.add_subparsers(dest="stats_command", required=True)

    correlation = stats_subparsers.add_parser("correlation", help="Compute a Pearson or Spearman correlation.")
    correlation.add_argument("--table", required=True)
    correlation.add_argument("--expected-table-sha256")
    correlation.add_argument("--x", required=True)
    correlation.add_argument("--y", required=True)
    correlation.add_argument("--method", choices=("pearson", "spearman"), default="pearson")
    correlation.add_argument("--output", required=True)
    correlation.set_defaults(handler=_run_stats_correlation)

    summary = stats_subparsers.add_parser("summary_table", help="Summarize numeric columns.")
    summary.add_argument("--table", required=True)
    summary.add_argument("--expected-table-sha256")
    summary.add_argument("--column", action="append", required=True)
    summary.add_argument("--output", required=True)
    summary.set_defaults(handler=_run_stats_summary_table)

    linear_model = stats_subparsers.add_parser("linear_model", help="Fit a small ordinary least-squares model.")
    linear_model.add_argument("--table", required=True)
    linear_model.add_argument("--expected-table-sha256")
    linear_model.add_argument("--outcome", required=True)
    linear_model.add_argument("--predictor", action="append", required=True)
    linear_model.add_argument("--output", required=True)
    linear_model.set_defaults(handler=_run_stats_linear_model)

    anova = stats_subparsers.add_parser("anova", help="Compute a one-way ANOVA summary.")
    anova.add_argument("--table", required=True)
    anova.add_argument("--expected-table-sha256")
    anova.add_argument("--outcome", required=True)
    anova.add_argument("--group", required=True)
    anova.add_argument("--output", required=True)
    anova.set_defaults(handler=_run_stats_anova)

    mixed = stats_subparsers.add_parser("mixed_effects", help="Render a grouped mixed-effects-ready summary.")
    mixed.add_argument("--table", required=True)
    mixed.add_argument("--expected-table-sha256")
    mixed.add_argument("--outcome", required=True)
    mixed.add_argument("--predictor", action="append", default=[])
    mixed.add_argument("--group")
    mixed.add_argument("--output", required=True)
    mixed.set_defaults(handler=_run_stats_mixed_effects)
    return parser


def _run_split_create(args: argparse.Namespace) -> int:
    try:
        _, rows = read_rows(args.table, expected_sha256=args.expected_table_sha256)
        manifest = create_split_manifest(
            rows=rows,
            target_column=args.target_column,
            table_path=args.table,
            test_fraction=args.test_fraction,
            seed=args.seed,
            split_strategy=args.strategy,
            stratify_bin_count=args.stratify_bin_count,
        )
        write_split_manifest(args.output, manifest)
        print(json.dumps({"written": args.output, "row_count": len(rows)}, indent=2))
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


def _run_prep_fit(args: argparse.Namespace) -> int:
    try:
        fieldnames, rows = read_rows(args.table, expected_sha256=args.expected_table_sha256)
        split_manifest = load_split_manifest(args.split, expected_sha256=args.expected_split_sha256)
        feature_columns = infer_numeric_feature_columns(
            rows=rows,
            fieldnames=fieldnames,
            target_column=args.target_column,
            feature_columns=args.feature_columns,
        )
        plan = fit_standardization_plan(
            rows=rows,
            feature_columns=feature_columns,
            target_column=args.target_column,
            split_manifest=split_manifest,
            table_path=args.table,
        )
        write_json(args.output, plan)
        print(json.dumps({"written": args.output, "feature_count": len(feature_columns)}, indent=2))
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


def _run_prep_apply(args: argparse.Namespace) -> int:
    try:
        fieldnames, rows = read_rows(args.table, expected_sha256=args.expected_table_sha256)
        plan = read_json(args.plan, expected_sha256=args.expected_plan_sha256)
        split_manifest = (
            load_split_manifest(args.split, expected_sha256=args.expected_split_sha256) if args.split else None
        )
        output_fields, output_rows = apply_standardization_plan(
            rows=rows,
            fieldnames=fieldnames,
            plan=plan,
            split_manifest=split_manifest,
        )
        write_rows(args.output, fieldnames=output_fields, rows=output_rows)
        print(json.dumps({"written": args.output, "row_count": len(output_rows)}, indent=2))
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


def _run_model_train(args: argparse.Namespace) -> int:
    try:
        if args.kind != "logistic_regression":
            raise ValueError("Only logistic_regression is supported in this slice.")
        fieldnames, rows = read_rows(args.table, expected_sha256=args.expected_table_sha256)
        split_manifest = load_split_manifest(args.split, expected_sha256=args.expected_split_sha256)
        feature_columns = infer_numeric_feature_columns(
            rows=rows,
            fieldnames=fieldnames,
            target_column=args.target_column,
            feature_columns=args.feature_columns,
        )
        feature_rows, targets = _numeric_split_rows(
            rows=rows,
            split_manifest=split_manifest,
            split_name="train",
            feature_columns=feature_columns,
            target_column=args.target_column,
        )
        model = fit_logistic_regression(
            feature_rows=feature_rows,
            targets=targets,
            feature_columns=feature_columns,
            target_column=args.target_column,
            learning_rate=args.learning_rate,
            iterations=args.iterations,
            table_path=args.table_reference if args.table_reference is not None else args.table,
        )
        write_json(args.output, model)
        print(json.dumps({"written": args.output, "feature_count": len(feature_columns)}, indent=2))
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


def _run_model_evaluate(args: argparse.Namespace) -> int:
    try:
        _, rows = read_rows(args.table, expected_sha256=args.expected_table_sha256)
        split_manifest = load_split_manifest(args.split, expected_sha256=args.expected_split_sha256)
        model = read_json(args.model, expected_sha256=args.expected_model_sha256)
        feature_rows, targets = _numeric_split_rows(
            rows=rows,
            split_manifest=split_manifest,
            split_name="test",
            feature_columns=list(model["feature_columns"]),
            target_column=args.target_column,
        )
        report = evaluate_logistic_regression(
            feature_rows=feature_rows,
            targets=targets,
            model=model,
            target_column=args.target_column,
            table_path=args.table,
        )
        write_json(args.output, report)
        print(json.dumps({"written": args.output, "accuracy": report["metrics"]["accuracy"]}, indent=2))
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


def _run_regression_train(args: argparse.Namespace) -> int:
    try:
        fieldnames, rows = read_rows(args.table, expected_sha256=args.expected_table_sha256)
        split_manifest = load_split_manifest(args.split, expected_sha256=args.expected_split_sha256)
        feature_columns = infer_numeric_feature_columns(
            rows=rows,
            fieldnames=fieldnames,
            target_column=args.target_column,
            feature_columns=args.feature_columns,
        )
        feature_rows, targets = _typed_numeric_split_rows(
            rows=rows,
            split_manifest=split_manifest,
            split_name="train",
            feature_columns=feature_columns,
            target_column=args.target_column,
            target_parser=float,
        )
        model = fit_regression_model(
            kind=args.kind,
            feature_rows=feature_rows,
            targets=targets,
            feature_columns=feature_columns,
            target_column=args.target_column,
            table_path=args.table_reference if args.table_reference is not None else args.table,
            alpha=args.alpha,
            l1_ratio=args.l1_ratio,
            max_iter=args.max_iter,
            random_state=args.random_state,
            learning_rate=args.learning_rate,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            subsample=args.subsample,
            colsample_bytree=args.colsample_bytree,
        )
        write_json(args.output, model)
        print(
            json.dumps(
                {"written": args.output, "feature_count": len(feature_columns), "kind": model["kind"]},
                indent=2,
            )
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


def _run_regression_evaluate(args: argparse.Namespace) -> int:
    try:
        _, rows = read_rows(args.table, expected_sha256=args.expected_table_sha256)
        split_manifest = load_split_manifest(args.split, expected_sha256=args.expected_split_sha256)
        model = read_json(args.model, expected_sha256=args.expected_model_sha256)
        feature_rows, targets = _typed_numeric_split_rows(
            rows=rows,
            split_manifest=split_manifest,
            split_name="test",
            feature_columns=list(model["feature_columns"]),
            target_column=args.target_column,
            target_parser=float,
        )
        report = evaluate_regression_model(
            feature_rows=feature_rows,
            targets=targets,
            model=model,
            target_column=args.target_column,
            table_path=args.table,
            bootstrap_iterations=args.bootstrap_iterations,
            bootstrap_seed=args.bootstrap_seed,
        )
        write_json(args.output, report)
        print(json.dumps({"written": args.output, "r2": report["metrics"]["r2"]}, indent=2))
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


def _run_stats_correlation(args: argparse.Namespace) -> int:
    try:
        _, rows = read_rows(args.table, expected_sha256=args.expected_table_sha256)
        report = correlation_report(rows, x_column=args.x, y_column=args.y, method=args.method)
        report["table"] = args.table
        write_json(args.output, report)
        print(json.dumps({"written": args.output, "r": report["r"]}, indent=2))
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


def _run_stats_summary_table(args: argparse.Namespace) -> int:
    try:
        _, rows = read_rows(args.table, expected_sha256=args.expected_table_sha256)
        report = summary_table_report(rows, columns=list(args.column))
        report["table"] = args.table
        write_json(args.output, report)
        print(json.dumps({"written": args.output, "columns": args.column}, indent=2))
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


def _run_stats_linear_model(args: argparse.Namespace) -> int:
    try:
        _, rows = read_rows(args.table, expected_sha256=args.expected_table_sha256)
        report = linear_model_report(rows, outcome=args.outcome, predictors=list(args.predictor))
        report["table"] = args.table
        write_json(args.output, report)
        print(json.dumps({"written": args.output, "n": report["n"]}, indent=2))
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


def _run_stats_anova(args: argparse.Namespace) -> int:
    try:
        _, rows = read_rows(args.table, expected_sha256=args.expected_table_sha256)
        report = anova_report(rows, outcome=args.outcome, group=args.group)
        report["table"] = args.table
        write_json(args.output, report)
        print(json.dumps({"written": args.output, "f": report["f"]}, indent=2))
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


def _run_stats_mixed_effects(args: argparse.Namespace) -> int:
    try:
        _, rows = read_rows(args.table, expected_sha256=args.expected_table_sha256)
        report = mixed_effects_report(rows, outcome=args.outcome, predictors=list(args.predictor), group=args.group)
        report["table"] = args.table
        write_json(args.output, report)
        print(json.dumps({"written": args.output, "engine": report["engine"]}, indent=2))
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.handler(args)


def _numeric_split_rows(
    *,
    rows: list[dict[str, str]],
    split_manifest: dict[str, Any],
    split_name: str,
    feature_columns: list[str],
    target_column: str,
) -> tuple[list[dict[str, float]], list[int]]:
    return _typed_numeric_split_rows(
        rows=rows,
        split_manifest=split_manifest,
        split_name=split_name,
        feature_columns=feature_columns,
        target_column=target_column,
        target_parser=int,
    )


def _typed_numeric_split_rows(
    *,
    rows: list[dict[str, str]],
    split_manifest: dict[str, Any],
    split_name: str,
    feature_columns: list[str],
    target_column: str,
    target_parser: Callable[[float], TargetValue],
) -> tuple[list[dict[str, float]], list[TargetValue]]:
    membership = split_membership(split_manifest)
    selected_indices = sorted(index for index, split in membership.items() if split == split_name)
    numeric_rows: list[dict[str, float]] = []
    targets: list[TargetValue] = []
    for index in selected_indices:
        row = rows[index]
        numeric_rows.append(
            {column: numeric_value(row[column], column=column, row_number=index + 1) for column in feature_columns}
        )
        target_value = numeric_value(row[target_column], column=target_column, row_number=index + 1)
        targets.append(target_parser(target_value))
    return numeric_rows, targets


if __name__ == "__main__":
    raise SystemExit(main())
