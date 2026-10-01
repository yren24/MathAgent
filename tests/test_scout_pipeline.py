import json
from pathlib import Path

import numpy as np
import pytest

from mint_scout.config import load_yaml
from mint_scout.models.gbt import GBTConfig
from mint_scout.scout.pipeline import ScoutConfig, run_scout
from mint_scout.scout.sampling import ProbeSamplingConfig
from mint_scout.scout.stability import BootstrapConfig


def test_scout_config_loads_centralized_v1_defaults():
    path = Path(__file__).parents[1] / "configs" / "scout" / "v1.yaml"

    config = ScoutConfig.from_mapping(load_yaml(path))

    assert config.probe.max_samples == 1000
    assert config.cv_folds == 5
    assert config.gbt.n_estimators == 4000
    assert config.gbt.normalize == "StandardScaler"
    assert config.bootstrap.replicates == 1000
    assert config.labels.min_labeled_samples == 300
    assert config.labels.allow_small_data_override is False


def test_scout_config_rejects_hpo_and_weighted_consensus():
    with pytest.raises(ValueError, match="hyperparameter optimization"):
        ScoutConfig.from_mapping({"model": {"family": "fixed_gbt", "hpo": True}})
    with pytest.raises(ValueError, match="consensus weights"):
        ScoutConfig.from_mapping({"consensus": {"weighted": True}})


def test_synthetic_scout_runs_shared_oof_and_all_31_subsets():
    rng = np.random.default_rng(21)
    n_samples = 60
    sample_ids = tuple(f"s{index:03d}" for index in range(n_samples))
    base = rng.normal(size=(n_samples, 3))
    targets = 1.5 * base[:, 0] - 0.8 * base[:, 1] + 0.1 * rng.normal(size=n_samples)
    features = {
        "PH": base + 0.05 * rng.normal(size=base.shape),
        "PL": base[:, :2],
        "CA": np.column_stack((base[:, 0], base[:, 2])),
        "FPRC": base * np.asarray([1.0, 0.5, 2.0]),
        "EIC": np.column_stack((base[:, 1], base[:, 2], base[:, 0] ** 2)),
    }
    config = ScoutConfig(
        probe=ProbeSamplingConfig(
            fraction=0.75,
            min_samples=45,
            max_samples=45,
            target_quantile_bins=5,
            size_quantile_bins=3,
            min_pair_support=0,
            random_seed=9,
        ),
        gbt=GBTConfig(
            config_id="synthetic_test",
            n_estimators=12,
            max_depth=2,
            min_samples_split=2,
            learning_rate=0.05,
            subsample=1.0,
            random_state=3,
            n_runs=1,
        ),
        bootstrap=BootstrapConfig(replicates=20, random_seed=12),
    )

    result = run_scout(
        sample_ids=sample_ids,
        targets=targets,
        structure_sizes=np.linspace(10, 70, n_samples),
        features_by_invariant=features,
        user_target=0.50,
        config=config,
    )
    payload = result.to_dict()

    assert result.selection.final_size == 45
    assert len(result.fold_assignment) == 45
    assert set(result.oof_store.predictions) == {"PH", "PL", "CA", "FPRC", "EIC"}
    assert all(len(values) == 45 for values in result.oof_store.predictions.values())
    assert len(result.ranking.priority_order) == 31
    assert result.target.source == "user"
    assert len(result.probe_hash) == 16
    json.dumps(payload)
