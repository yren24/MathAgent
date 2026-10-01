from types import SimpleNamespace

import numpy as np

from mint_scout.models.gbt import GBTConfig, run_oof_gbt
from mint_scout.scout.oof import AlignedOOFStore


def test_oof_runner_never_uses_prediction_rows_for_training(monkeypatch):
    sample_ids = tuple(f"s{index}" for index in range(10))
    features = np.arange(10, dtype=float).reshape(-1, 1)
    targets = np.linspace(0.0, 1.0, 10)
    fold_ids = {sample_id: index % 5 for index, sample_id in enumerate(sample_ids)}

    def fake_fit_predict(x_train, y_train, x_pred, *, config):
        assert set(x_train[:, 0]).isdisjoint(set(x_pred[:, 0]))
        return x_pred[:, 0]

    monkeypatch.setattr("mint_scout.models.gbt.fit_predict_gbt", fake_fit_predict)
    artifact = run_oof_gbt(
        features,
        targets,
        sample_ids,
        fold_ids,
        config=GBTConfig(n_estimators=1, n_runs=1),
    )

    assert np.array_equal(artifact.y_pred, features[:, 0])


def test_oof_store_aligns_artifacts_by_sample_id():
    shared_ids = ("a", "b", "c")
    folds = {"a": 0, "b": 1, "c": 2}
    store = AlignedOOFStore(shared_ids, np.asarray([1.0, 2.0, 3.0]), folds)
    artifact = SimpleNamespace(
        sample_ids=("c", "a", "b"),
        y_true=np.asarray([3.0, 1.0, 2.0]),
        y_pred=np.asarray([30.0, 10.0, 20.0]),
        fold_ids=folds,
    )

    store.add("PL", artifact)

    assert np.array_equal(store.predictions["PL"], np.asarray([10.0, 20.0, 30.0]))
