from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from mint_scout.models.gbt import GBTConfig, OOFPredictionArtifact, run_oof_gbt


@dataclass
class AlignedOOFStore:
    sample_ids: tuple[str, ...]
    y_true: np.ndarray
    fold_ids: Mapping[str, int]
    _predictions: dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.y_true = np.asarray(self.y_true, dtype=float)
        if len(self.sample_ids) != len(self.y_true):
            raise ValueError("sample_ids and y_true must be aligned")
        if len(set(self.sample_ids)) != len(self.sample_ids):
            raise ValueError("sample_ids contains duplicates")
        if set(self.fold_ids) != set(self.sample_ids):
            raise ValueError("fold_ids must contain exactly the OOF sample ids")
        self.y_true.setflags(write=False)
        self.fold_ids = MappingProxyType(dict(self.fold_ids))

    def add(self, invariant_name: str, artifact: OOFPredictionArtifact) -> None:
        name = invariant_name.upper()
        if name in self._predictions:
            raise ValueError(f"OOF predictions already stored for {name}")
        if set(artifact.sample_ids) != set(self.sample_ids):
            raise ValueError(f"{name} OOF sample ids do not match the shared store")
        index = {sample_id: row for row, sample_id in enumerate(artifact.sample_ids)}
        order = np.asarray([index[sample_id] for sample_id in self.sample_ids], dtype=int)
        aligned_true = np.asarray(artifact.y_true, dtype=float)[order]
        aligned_pred = np.asarray(artifact.y_pred, dtype=float)[order]
        if not np.allclose(aligned_true, self.y_true, rtol=0.0, atol=0.0):
            raise ValueError(f"{name} OOF targets do not match the shared store")
        if any(artifact.fold_ids[sample_id] != self.fold_ids[sample_id] for sample_id in self.sample_ids):
            raise ValueError(f"{name} OOF fold assignments do not match the shared store")
        if not np.all(np.isfinite(aligned_pred)):
            raise ValueError(f"{name} OOF predictions must be finite")
        aligned_pred.setflags(write=False)
        self._predictions[name] = aligned_pred

    @property
    def predictions(self) -> Mapping[str, np.ndarray]:
        return MappingProxyType(dict(self._predictions))


def run_shared_oof_gbt(
    *,
    features_by_invariant: Mapping[str, np.ndarray],
    targets: np.ndarray,
    sample_ids: Sequence[str],
    fold_ids: Mapping[str, int],
    config: GBTConfig,
) -> AlignedOOFStore:
    ids = tuple(sample_ids)
    target_array = np.asarray(targets, dtype=float)
    store = AlignedOOFStore(ids, target_array, dict(fold_ids))
    for invariant_name, feature_array in features_by_invariant.items():
        features = np.asarray(feature_array, dtype=float)
        if features.shape[0] != len(ids):
            raise ValueError(f"{invariant_name} features do not match the shared sample ids")
        flattened = features.reshape(len(ids), -1)
        artifact = run_oof_gbt(
            flattened,
            target_array,
            ids,
            dict(fold_ids),
            config=config,
        )
        store.add(invariant_name, artifact)
    return store
