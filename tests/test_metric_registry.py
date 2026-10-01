import pytest

from mint_scout.evaluation.metrics import higher_is_better, metric_direction


def test_metric_directions_cover_v1_regression_metrics():
    assert higher_is_better("PCC") is True
    assert higher_is_better("R2") is True
    assert higher_is_better("RMSE") is False
    assert higher_is_better("MAE") is False
    assert metric_direction("rmse") == "lower"


def test_unknown_metric_direction_raises():
    with pytest.raises(KeyError):
        metric_direction("accuracy")
