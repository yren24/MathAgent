from __future__ import annotations

import numpy as np
import pytest

from mint_scout.models.ann import ANNConfig, _scale_fold, run_oof_ann
from mint_scout.run_casf_ann_probe import (
    ProbeANNConfig,
    _paired_gain_summary,
    _paired_gain_vs_best_singleton,
    _prediction_summary,
)


def test_ann_config_is_fixed_and_rejects_hpo():
    config = ProbeANNConfig.from_mapping(
        {
            "model": {
                "family": "fixed_probe_ann",
                "hpo": False,
                "params": {"hidden_layers": [8, 4], "random_seeds": [7]},
            },
            "subsets": {"baseline": ["PL"], "candidates": [["PL"], ["PH", "PL"]]},
        }
    )

    assert config.ann.hidden_layers == (8, 4)
    assert config.ann.random_seeds == (7,)
    assert config.subsets == (("PL",), ("PH", "PL"))
    with pytest.raises(ValueError, match="forbids hyperparameter optimization"):
        ProbeANNConfig.from_mapping({"model": {"hpo": True}})


def test_standard_scaler_is_fit_on_training_fold_only():
    train = np.asarray([[0.0], [2.0]])
    held_out = np.asarray([[100.0]])

    scaled_train, scaled_held_out = _scale_fold(train, held_out, normalize="StandardScaler")

    assert np.allclose(scaled_train[:, 0], [-1.0, 1.0])
    assert scaled_held_out[0, 0] == pytest.approx(99.0)


def test_paired_gain_uses_positive_for_candidate_improvement():
    y_true = np.asarray([0.0, 1.0, 2.0, 3.0])
    baseline = np.asarray([0.0, 0.5, 1.0, 1.5])
    candidate = y_true.copy()
    indices = [np.arange(4), np.asarray([0, 1, 2, 3])]

    pcc_gain = _paired_gain_summary(
        y_true,
        baseline,
        candidate,
        metric_name="PCC",
        bootstrap_indices=indices,
        ci=0.95,
    )
    rmse_gain = _paired_gain_summary(
        y_true,
        baseline,
        candidate,
        metric_name="RMSE",
        bootstrap_indices=indices,
        ci=0.95,
    )

    assert pcc_gain["nominal"] >= 0.0
    assert rmse_gain["nominal"] > 0.0


def test_prediction_summary_reports_shared_fold_metrics_and_gain():
    sample_ids = ("a", "b", "c", "d")
    folds = {"a": 0, "b": 0, "c": 1, "d": 1}
    y_true = np.asarray([0.0, 1.0, 2.0, 3.0])
    baseline = np.asarray([0.0, 0.5, 1.0, 1.5])
    candidate = y_true.copy()

    summary = _prediction_summary(
        targets=y_true,
        prediction=candidate,
        baseline_prediction=baseline,
        sample_ids=sample_ids,
        fold_ids=folds,
        metrics=("RMSE",),
        bootstrap_indices=(np.arange(4),),
        bootstrap_ci=0.95,
    )

    assert set(summary["fold_metrics"]) == {"0", "1"}
    assert summary["metrics"]["RMSE"] == 0.0
    assert summary["paired_gain_vs_baseline"]["RMSE"]["nominal"] > 0.0


def test_gain_vs_best_singleton_uses_best_component_for_each_metric():
    y_true = np.asarray([0.0, 1.0, 2.0, 3.0])
    strong = y_true.copy()
    weak = np.asarray([0.0, 0.5, 1.0, 1.5])
    result = _paired_gain_vs_best_singleton(
        targets=y_true,
        candidate_prediction=weak,
        singleton_predictions={"PH": strong, "PL": weak},
        metrics=("PCC", "RMSE"),
        bootstrap_indices=(np.arange(4),),
        bootstrap_ci=0.95,
    )

    assert result["PCC"]["reference_singleton"] == "PH"
    assert result["RMSE"]["reference_singleton"] == "PH"
    assert result["RMSE"]["nominal"] < 0.0


def test_oof_ann_is_deterministic_and_assigns_every_sample_once():
    pytest.importorskip("torch")
    rng = np.random.default_rng(5)
    sample_ids = tuple(f"s{index}" for index in range(20))
    features = rng.normal(size=(20, 4))
    targets = 2.0 * features[:, 0] - features[:, 1]
    folds = {sample_id: index % 2 for index, sample_id in enumerate(sample_ids)}
    config = ANNConfig(
        config_id="test",
        hidden_layers=(8,),
        epochs=2,
        batch_size=8,
        learning_rate=1.0e-3,
        weight_decay=0.0,
        dropout=0.0,
        random_seeds=(3,),
        device="cpu",
    )

    first = run_oof_ann(features, targets, sample_ids, folds, config=config)
    second = run_oof_ann(features, targets, sample_ids, folds, config=config)

    assert first.y_pred.shape == (20,)
    assert np.all(np.isfinite(first.y_pred))
    assert np.array_equal(first.y_pred, second.y_pred)
    assert first.input_dimension == 4
    assert first.parameter_count > 0
    assert first.device_name == "CPU"
    assert first.torch_version

    with pytest.raises(ValueError, match="must be aligned"):
        run_oof_ann(features[:10], targets, sample_ids, folds, config=config)
