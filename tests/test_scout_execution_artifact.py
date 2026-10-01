from __future__ import annotations

import numpy as np
import pytest

from mint_scout.execution import ExecutionConfig, ProgressiveExecutionEngine, ScoutExecutionArtifact
from mint_scout.execution.provider import InMemoryFeatureProvider
from mint_scout.models.gbt import GBTConfig
from mint_scout.schemas import EvaluationMode, EvaluationPlan
from mint_scout.scout.pipeline import ScoutConfig, run_scout
from mint_scout.scout.sampling import ProbeSamplingConfig
from mint_scout.scout.stability import BootstrapConfig


def _gbt() -> GBTConfig:
    return GBTConfig(
        config_id="artifact_test",
        n_estimators=4,
        max_depth=2,
        min_samples_split=2,
        learning_rate=0.1,
        subsample=1.0,
        random_state=9,
        n_runs=1,
    )


def _scout_result():
    sample_ids = tuple(f"s{index:02d}" for index in range(20))
    x = np.linspace(-2.0, 2.0, len(sample_ids))
    targets = 2.0 * x + np.sin(x)
    folds = {sample_id: index % 5 for index, sample_id in enumerate(sample_ids)}
    config = ScoutConfig(
        cv_folds=5,
        probe=ProbeSamplingConfig(
            fraction=1.0,
            min_samples=20,
            max_samples=20,
            min_pair_support=0,
        ),
        gbt=_gbt(),
        bootstrap=BootstrapConfig(replicates=5, random_seed=3),
    )
    result = run_scout(
        sample_ids=sample_ids,
        targets=targets,
        structure_sizes=np.arange(len(sample_ids), dtype=float),
        features_by_invariant={"PL": x.reshape(-1, 1), "PH": np.column_stack((x, x**2))},
        user_target=0.5,
        modeling_fold_assignment=folds,
        config=config,
    )
    return sample_ids, targets, folds, result


def test_scout_artifact_round_trip_preserves_full_folds_and_queue(tmp_path):
    sample_ids, _, folds, result = _scout_result()
    artifact = result.to_execution_artifact(representation_hash="repr-123")
    path = tmp_path / "scout-execution.json"
    artifact.write(path)
    restored = ScoutExecutionArtifact.read(path)

    assert restored.modeling_sample_ids == sample_ids
    assert dict(restored.full_fold_assignment) == folds
    assert restored.frozen_priority_order == result.ranking.priority_order
    assert restored.gbt_parameter_hash == _gbt().parameter_hash
    assert restored.ranking_policy == "hierarchical_empirical_v1"


def test_execution_from_scout_rejects_representation_mismatch():
    sample_ids, targets, _, result = _scout_result()
    artifact = result.to_execution_artifact(representation_hash="frozen-repr")
    provider = InMemoryFeatureProvider(
        {"PL": np.arange(len(sample_ids), dtype=float).reshape(-1, 1)},
        sample_ids,
    )
    engine = ProgressiveExecutionEngine(
        provider,
        config=ExecutionConfig(
            target_value=artifact.target_value,
            representation_hash="different-repr",
            gbt=_gbt(),
        ),
    )

    with pytest.raises(ValueError, match="representation hash"):
        engine.run_from_scout(
            artifact=artifact,
            plan=EvaluationPlan(EvaluationMode.FULL_LABELED_CV, sample_ids),
            targets_by_sample=dict(zip(sample_ids, targets)),
        )


def test_execution_from_scout_consumes_frozen_queue_target_and_folds():
    sample_ids, targets, _, result = _scout_result()
    artifact = result.to_execution_artifact(representation_hash="frozen-repr")
    x = np.linspace(-2.0, 2.0, len(sample_ids))
    provider = InMemoryFeatureProvider(
        {"PL": x.reshape(-1, 1), "PH": np.column_stack((x, x**2))},
        sample_ids,
    )
    engine = ProgressiveExecutionEngine(
        provider,
        config=ExecutionConfig(
            target_value=artifact.target_value,
            representation_hash=artifact.representation_hash,
            require_shared_fold_assignment=True,
            gbt=_gbt(),
        ),
    )

    execution = engine.run_from_scout(
        artifact=artifact,
        plan=EvaluationPlan(EvaluationMode.FULL_LABELED_CV, sample_ids),
        targets_by_sample=dict(zip(sample_ids, targets)),
    )

    first_needed = artifact.frozen_priority_order[0][0]
    assert execution.acquisition_order[0] == first_needed
    assert execution.target_value == artifact.target_value


def test_execution_can_require_shared_folds_for_cv():
    sample_ids = tuple(f"s{index}" for index in range(10))
    provider = InMemoryFeatureProvider({"PL": np.arange(10).reshape(-1, 1)}, sample_ids)
    engine = ProgressiveExecutionEngine(
        provider,
        config=ExecutionConfig(
            require_shared_fold_assignment=True,
            gbt=_gbt(),
        ),
    )

    with pytest.raises(ValueError, match="shared full-fold assignment"):
        engine.run(
            frozen_priority_order=(("PL",),),
            plan=EvaluationPlan(EvaluationMode.FULL_LABELED_CV, sample_ids),
            targets_by_sample={sample_id: float(index) for index, sample_id in enumerate(sample_ids)},
        )
