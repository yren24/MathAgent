from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np

from mint_scout.invariants.manifest import stable_hash
from mint_scout.search.cost import CostRecord, measure_cost


@dataclass(frozen=True)
class GBTConfig:
    config_id: str = "plbind_feature_gbt_ensemble_v1"
    n_estimators: int = 4000
    max_depth: int = 7
    min_samples_split: int = 5
    learning_rate: float = 0.01
    subsample: float = 0.5
    max_features: str = "sqrt"
    random_state: int = 42
    n_runs: int = 10
    normalize: str = "StandardScaler"

    def __post_init__(self) -> None:
        if self.n_estimators < 1 or self.n_runs < 1:
            raise ValueError("n_estimators and n_runs must be positive")
        if self.max_depth < 1 or self.min_samples_split < 2:
            raise ValueError("max_depth must be positive and min_samples_split must be at least 2")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if not 0.0 < self.subsample <= 1.0:
            raise ValueError("subsample must be in (0, 1]")

    @property
    def parameter_hash(self) -> str:
        return stable_hash(asdict(self))


PLBIND_FEATURE_GBT_CONFIG = GBTConfig()


@dataclass(frozen=True)
class OOFPredictionArtifact:
    sample_ids: tuple[str, ...]
    y_true: np.ndarray
    y_pred: np.ndarray
    fold_ids: dict[str, int]
    gbt_config: GBTConfig
    cost: CostRecord


@dataclass(frozen=True)
class FittedGBTEnsemble:
    scaler: object | None
    models: tuple[object, ...]
    gbt_config: GBTConfig

    def predict(self, x_pred: np.ndarray) -> np.ndarray:
        x = np.asarray(x_pred, dtype=float)
        if self.scaler is not None:
            x = self.scaler.transform(x)
        predictions = [model.predict(x) for model in self.models]
        return np.mean(np.asarray(predictions), axis=0)


def _make_model(config: GBTConfig, run_id: int):
    from sklearn.ensemble import GradientBoostingRegressor

    return GradientBoostingRegressor(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        min_samples_split=config.min_samples_split,
        learning_rate=config.learning_rate,
        subsample=config.subsample,
        max_features=config.max_features,
        random_state=config.random_state + run_id,
    )


def _make_scaler(config: GBTConfig):
    if config.normalize == "StandardScaler":
        from sklearn.preprocessing import StandardScaler

        return StandardScaler()
    if config.normalize in {"none", None}:
        return None
    raise ValueError(f"Unsupported normalize={config.normalize!r}")


def fit_predict_gbt(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_pred: np.ndarray,
    *,
    config: GBTConfig = PLBIND_FEATURE_GBT_CONFIG,
) -> np.ndarray:
    ensemble = fit_gbt_ensemble(x_train, y_train, config=config)
    return ensemble.predict(x_pred)


def fit_gbt_ensemble(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    config: GBTConfig = PLBIND_FEATURE_GBT_CONFIG,
) -> FittedGBTEnsemble:
    x = np.asarray(x_train, dtype=float)
    y = np.asarray(y_train, dtype=float)
    if x.shape[0] != y.shape[0]:
        raise ValueError("x_train and y_train must have aligned first dimensions")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("x_train and y_train must be finite")
    scaler = _make_scaler(config)
    if scaler is not None:
        x = scaler.fit_transform(x)
    models = []
    for run_id in range(config.n_runs):
        model = _make_model(config, run_id)
        model.fit(x, y)
        models.append(model)
    return FittedGBTEnsemble(scaler=scaler, models=tuple(models), gbt_config=config)


def make_fold_ids(sample_ids: Sequence[str], *, n_folds: int) -> dict[str, int]:
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2")
    sample_tuple = tuple(sample_ids)
    if len(sample_tuple) < n_folds:
        raise ValueError("sample count must be at least n_folds")
    if len(set(sample_tuple)) != len(sample_tuple):
        raise ValueError("sample_ids contains duplicates")
    return {sample_id: index % n_folds for index, sample_id in enumerate(sample_tuple)}


def run_oof_gbt(
    features: np.ndarray,
    targets: np.ndarray,
    sample_ids: Sequence[str],
    fold_ids: dict[str, int],
    *,
    config: GBTConfig = PLBIND_FEATURE_GBT_CONFIG,
) -> OOFPredictionArtifact:
    sample_tuple = tuple(sample_ids)
    if features.shape[0] != len(sample_tuple) or targets.shape[0] != len(sample_tuple):
        raise ValueError("features, targets, and sample_ids must have aligned first dimensions")
    if len(set(sample_tuple)) != len(sample_tuple):
        raise ValueError("sample_ids contains duplicates")
    if not np.all(np.isfinite(features)) or not np.all(np.isfinite(targets)):
        raise ValueError("features and targets must be finite")
    missing = set(sample_tuple) - set(fold_ids)
    if missing:
        raise ValueError(f"Missing fold assignments for {sorted(missing)[:10]}")

    y_pred = np.full(len(sample_tuple), np.nan, dtype=float)
    prediction_counts = np.zeros(len(sample_tuple), dtype=int)
    fold_values = sorted({fold_ids[sid] for sid in sample_tuple})
    with measure_cost() as cost_records:
        for fold in fold_values:
            val_idx = np.asarray([idx for idx, sid in enumerate(sample_tuple) if fold_ids[sid] == fold])
            train_idx = np.asarray([idx for idx, sid in enumerate(sample_tuple) if fold_ids[sid] != fold])
            if len(val_idx) == 0 or len(train_idx) == 0:
                raise ValueError(f"Fold {fold} has empty train or validation set")
            y_pred[val_idx] = fit_predict_gbt(
                features[train_idx],
                targets[train_idx],
                features[val_idx],
                config=config,
            )
            prediction_counts[val_idx] += 1
    if not np.all(prediction_counts == 1) or not np.all(np.isfinite(y_pred)):
        raise AssertionError("Every OOF sample must receive exactly one finite held-out prediction")
    return OOFPredictionArtifact(
        sample_ids=sample_tuple,
        y_true=np.asarray(targets, dtype=float),
        y_pred=y_pred,
        fold_ids={sid: fold_ids[sid] for sid in sample_tuple},
        gbt_config=config,
        cost=cost_records[0],
    )
