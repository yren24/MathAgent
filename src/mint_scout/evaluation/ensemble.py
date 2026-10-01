from __future__ import annotations

import itertools
from collections.abc import Callable, Iterable, Mapping
from typing import Tuple

import numpy as np


Subset = Tuple[str, ...]


def enumerate_nonempty_subsets(invariants: Iterable[str]) -> tuple[Subset, ...]:
    ordered = tuple(sorted(invariants))
    subsets: list[Subset] = []
    for size in range(1, len(ordered) + 1):
        subsets.extend(tuple(combo) for combo in itertools.combinations(ordered, size))
    return tuple(subsets)


def mean_aggregate(predictions: Mapping[str, np.ndarray], subset: Iterable[str]) -> np.ndarray:
    arrays = [np.asarray(predictions[name], dtype=float) for name in subset]
    if not arrays:
        raise ValueError("subset must be non-empty")
    return np.mean(np.vstack(arrays), axis=0)


def score_subsets(
    y_true: np.ndarray,
    predictions: Mapping[str, np.ndarray],
    *,
    metric: Callable[[np.ndarray, np.ndarray], float],
) -> dict[Subset, float]:
    scores: dict[Subset, float] = {}
    for subset in enumerate_nonempty_subsets(predictions.keys()):
        scores[subset] = metric(y_true, mean_aggregate(predictions, subset))
    return scores


def best_subset(subset_scores: Mapping[Subset, float]) -> tuple[Subset, float]:
    if not subset_scores:
        raise ValueError("subset_scores must be non-empty")
    return max(subset_scores.items(), key=lambda item: item[1])
