from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np

from mint_scout.evaluation.complementarity import camc, exclusion_necessity, pair_gains
from mint_scout.evaluation.ensemble import score_subsets


@dataclass(frozen=True)
class MetricEstimate:
    mean: float
    std: float
    ci_low: float
    ci_high: float
    n_bootstrap: int


def summarize(values: np.ndarray, *, ci: float) -> MetricEstimate:
    alpha = (1.0 - ci) / 2.0
    return MetricEstimate(
        mean=float(np.mean(values)),
        std=float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        ci_low=float(np.quantile(values, alpha)),
        ci_high=float(np.quantile(values, 1.0 - alpha)),
        n_bootstrap=int(len(values)),
    )


def bootstrap_prediction_evidence(
    y_true: np.ndarray,
    predictions: Mapping[str, np.ndarray],
    *,
    metric: Callable[[np.ndarray, np.ndarray], float],
    n_bootstrap: int,
    ci: float,
    seed: int,
) -> dict[str, dict[object, MetricEstimate]]:
    y_true = np.asarray(y_true, dtype=float)
    pred_arrays = {name: np.asarray(pred, dtype=float) for name, pred in predictions.items()}
    n = len(y_true)
    rng = np.random.default_rng(seed)

    singleton_values: dict[str, list[float]] = {name: [] for name in pred_arrays}
    subset_values: dict[tuple[str, ...], list[float]] = {}
    camc_values: dict[str, list[float]] = {name: [] for name in pred_arrays}
    pair_values: dict[tuple[str, str], list[float]] = {}
    necessity_values: dict[str, list[float]] = {name: [] for name in pred_arrays}

    active = tuple(pred_arrays)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        y_b = y_true[idx]
        pred_b = {name: pred[idx] for name, pred in pred_arrays.items()}
        subset_scores = score_subsets(y_b, pred_b, metric=metric)
        for subset, value in subset_scores.items():
            subset_values.setdefault(subset, []).append(value)
            if len(subset) == 1:
                singleton_values[subset[0]].append(value)
        for name, value in camc(subset_scores, active).items():
            if value is not None:
                camc_values[name].append(value)
        for pair, value in pair_gains(subset_scores, active).items():
            pair_values.setdefault(pair, []).append(value)
        for name, value in exclusion_necessity(subset_scores, active).items():
            if value is not None:
                necessity_values[name].append(value)

    return {
        "singleton": {k: summarize(np.asarray(v), ci=ci) for k, v in singleton_values.items()},
        "subset": {k: summarize(np.asarray(v), ci=ci) for k, v in subset_values.items()},
        "camc": {k: summarize(np.asarray(v), ci=ci) for k, v in camc_values.items() if v},
        "pair_gain": {k: summarize(np.asarray(v), ci=ci) for k, v in pair_values.items()},
        "necessity": {k: summarize(np.asarray(v), ci=ci) for k, v in necessity_values.items() if v},
    }
