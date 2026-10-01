"""Standard regression metrics for immutable legacy MOF GBT predictions."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any


def regression_metrics(y_true: list[float], y_pred: list[float]) -> dict[str, float]:
    """Return standard residual-based metrics for one evaluation fold."""
    if not y_true or len(y_true) != len(y_pred):
        raise ValueError("Expected non-empty, equally sized true and predicted values")

    residual_sum_squares = sum((actual - predicted) ** 2 for actual, predicted in zip(y_true, y_pred))
    rmse = math.sqrt(residual_sum_squares / len(y_true))
    mean_true = fmean(y_true)
    total_sum_squares = sum((actual - mean_true) ** 2 for actual in y_true)
    r2 = float("nan") if total_sum_squares == 0.0 else 1.0 - residual_sum_squares / total_sum_squares
    return {"rmse": rmse, "r2_standard": r2}


def summarize_predictions(prediction_csv: Path) -> dict[str, Any]:
    """Compute per-fold and macro summaries without altering legacy output."""
    with prediction_csv.open(newline="") as handle:
        reader = csv.DictReader(handle)
        expected = {"repeat", "fold", "true", "pred"}
        if reader.fieldnames is None or not expected.issubset(reader.fieldnames):
            raise ValueError(f"Predictions must contain {sorted(expected)}: {prediction_csv}")
        grouped: dict[tuple[int, int], tuple[list[float], list[float]]] = defaultdict(lambda: ([], []))
        for row in reader:
            key = (int(row["repeat"]), int(row["fold"]))
            actual, predicted = grouped[key]
            actual.append(float(row["true"]))
            predicted.append(float(row["pred"]))

    folds = []
    for (repeat, fold), (actual, predicted) in sorted(grouped.items()):
        folds.append({"repeat": repeat, "fold": fold, "test_size": len(actual), **regression_metrics(actual, predicted)})
    if not folds:
        raise ValueError(f"No prediction rows found: {prediction_csv}")

    finite_r2 = [row["r2_standard"] for row in folds if math.isfinite(row["r2_standard"])]
    return {
        "metric_schema": "mint-agent.mof-standard-regression-metrics.v1",
        "prediction_csv": str(prediction_csv),
        "aggregation": "macro_mean_over_legacy_test_folds",
        "fold_metrics": folds,
        "mean_rmse": fmean(row["rmse"] for row in folds),
        "std_rmse": pstdev(row["rmse"] for row in folds),
        "mean_r2_standard": fmean(finite_r2) if finite_r2 else None,
        "std_r2_standard": pstdev(finite_r2) if finite_r2 else None,
    }


def write_summary(prediction_csv: Path, output: Path | None = None) -> Path:
    output = output or prediction_csv.with_name(prediction_csv.stem.replace("_predictions", "") + "_standard_metrics.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summarize_predictions(prediction_csv), indent=2, sort_keys=True) + "\n")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute standard MOF regression metrics from legacy GBT predictions")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(write_summary(args.predictions, args.output))


if __name__ == "__main__":
    main()
