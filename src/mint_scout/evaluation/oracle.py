from __future__ import annotations

from collections.abc import Callable, Mapping

import numpy as np

from mint_scout.evaluation.ensemble import best_subset, score_subsets


def exhaustive_prediction_oracle(
    y_true: np.ndarray,
    predictions: Mapping[str, np.ndarray],
    *,
    metric: Callable[[np.ndarray, np.ndarray], float],
) -> tuple[tuple[str, ...], float]:
    return best_subset(score_subsets(y_true, predictions, metric=metric))
