from pathlib import Path

from mint_scout.execute_casf import _load_gbt_config
from mint_scout.models.gbt import PLBIND_FEATURE_GBT_CONFIG


def test_plbind_feature_gbt_config_matches_legacy_active_path():
    cfg = PLBIND_FEATURE_GBT_CONFIG
    assert cfg.normalize == "StandardScaler"
    assert cfg.n_estimators == 4000
    assert cfg.max_depth == 7
    assert cfg.min_samples_split == 5
    assert cfg.learning_rate == 0.01
    assert cfg.subsample == 0.5
    assert cfg.max_features == "sqrt"
    assert cfg.random_state == 42
    assert cfg.n_runs == 10


def test_adaptive_gbt_profile_uses_three_deterministic_runs():
    root = Path(__file__).resolve().parents[1]
    cfg = _load_gbt_config(root / "configs/gbt/plbind_adaptive_gbt.yaml")

    assert cfg.config_id == "plbind_feature_gbt_ensemble_3run_v1"
    assert cfg.random_state == 42
    assert cfg.n_runs == 3
    assert cfg.normalize == "StandardScaler"
    assert cfg.n_estimators == 4000
