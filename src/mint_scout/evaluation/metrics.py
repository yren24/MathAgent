from __future__ import annotations

import math

import numpy as np


def pcc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred shapes differ")
    if y_true.size < 2:
        return float("nan")
    if np.std(y_true) == 0.0 or np.std(y_pred) == 0.0:
        return float("nan")
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def pcc2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    value = pcc(y_true, y_pred)
    return value * value if not math.isnan(value) else value


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    from sklearn.metrics import r2_score

    return float(r2_score(np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)))


METRICS = {
    "pcc": pcc,
    "pcc2": pcc2,
    "rmse": rmse,
    "mae": mae,
    "r2": r2,
}

METRIC_DIRECTIONS = {
    "pcc": "higher",
    "pcc2": "higher",
    "r2": "higher",
    "rmse": "lower",
    "mae": "lower",
}


def get_metric(name: str):
    key = name.lower()
    try:
        return METRICS[key]
    except KeyError as exc:
        raise KeyError(f"Unknown metric {name!r}; expected one of {sorted(METRICS)}") from exc


def metric_direction(name: str) -> str:
    key = name.lower()
    try:
        return METRIC_DIRECTIONS[key]
    except KeyError as exc:
        raise KeyError(f"Unknown metric {name!r}; expected one of {sorted(METRICS)}") from exc


def higher_is_better(name: str) -> bool:
    return metric_direction(name) == "higher"
