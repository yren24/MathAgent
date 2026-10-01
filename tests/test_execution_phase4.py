import json
from pathlib import Path

import numpy as np

from mint_scout.execution import ExecutionConfig, ProgressiveExecutionEngine, SelectionStatus
from mint_scout.execution.provider import InMemoryFeatureProvider, PLBindCachedFeatureProvider
from mint_scout.execution.engine import _select_subset
from mint_scout.execute_casf import _result_to_dict
from mint_scout.invariants.contracts import FeatureToolResult
from mint_scout.models.gbt import GBTConfig
from mint_scout.representation import make_legacy_casf_representation_spec
from mint_scout.schemas import EvaluationMode, EvaluationPlan


def _gbt_config() -> GBTConfig:
    return GBTConfig(
        config_id="phase4_test",
        n_estimators=8,
        max_depth=2,
        min_samples_split=2,
        learning_rate=0.1,
        subsample=1.0,
        random_state=7,
        n_runs=1,
    )


def _linear_dataset(n_train: int = 30, n_test: int = 12, n_infer: int = 3):
    total = n_train + n_test + n_infer
    sample_ids = tuple(f"s{index:03d}" for index in range(total))
    x = np.linspace(-2.0, 2.0, total)
    y = 3.0 * x + 0.2 * np.sin(x)
    features = {
        "PL": x.reshape(-1, 1),
        "PH": np.column_stack((x, x**2)),
        "CA": np.cos(3.0 * x).reshape(-1, 1),
        "FPRC": np.column_stack((x, np.sin(x))),
        "EIC": np.sin(5.0 * x).reshape(-1, 1),
    }
    targets = {sample_id: float(y[index]) for index, sample_id in enumerate(sample_ids[: n_train + n_test])}
    return sample_ids, targets, features


def test_external_labeled_test_stops_immediately_when_target_is_reached():
    sample_ids, targets, features = _linear_dataset()
    provider = InMemoryFeatureProvider(features, sample_ids)
    plan = EvaluationPlan(
        mode=EvaluationMode.EXPLICIT_LABELED_TEST,
        modeling_sample_ids=sample_ids[:30],
        evaluation_sample_ids=sample_ids[30:42],
    )
    engine = ProgressiveExecutionEngine(
        provider,
        config=ExecutionConfig(target_value=-1.0, gbt=_gbt_config()),
    )

    result = engine.run(
        frozen_priority_order=(("PL",), ("PH",), ("CA",)),
        plan=plan,
        targets_by_sample=targets,
    )

    assert result.status == SelectionStatus.TARGET_REACHED
    assert result.acquisition_order == ("PL",)
    assert result.selected_subset == ("PL",)
    assert len(result.rounds) == 1
    assert result.n_acceptance_queries == 1
    assert result.evaluation_set_used_for_acceptance is True
    assert result.untouched_final_test is False


def test_progressive_acquisition_reuses_cached_features_for_overlapping_subsets():
    sample_ids, targets, features = _linear_dataset()
    provider = InMemoryFeatureProvider(features, sample_ids, precomputed={"PL": sample_ids[:42]})
    plan = EvaluationPlan(
        mode=EvaluationMode.EXPLICIT_LABELED_TEST,
        modeling_sample_ids=sample_ids[:30],
        evaluation_sample_ids=sample_ids[30:42],
    )
    engine = ProgressiveExecutionEngine(
        provider,
        config=ExecutionConfig(target_value=1.01, gbt=_gbt_config()),
    )

    result = engine.run(
        frozen_priority_order=(("PL", "FPRC"), ("PL", "CA")),
        plan=plan,
        targets_by_sample=targets,
    )

    assert result.status == SelectionStatus.TARGET_NOT_REACHED
    assert result.acquisition_order == ("PL", "FPRC", "CA")
    assert result.acquisition_records[0].cached_count == 42
    assert result.acquisition_records[0].computed_count == 0
    assert result.acquisition_records[1].computed_count == 42
    assert result.acquisition_records[2].computed_count == 42
    assert [len(round_result.subset_scores) for round_result in result.rounds] == [1, 3, 7]


def test_full_labeled_cv_uses_oof_and_reports_exhausted_failure():
    sample_ids, targets, features = _linear_dataset(n_train=45, n_test=0, n_infer=0)
    provider = InMemoryFeatureProvider(features, sample_ids)
    plan = EvaluationPlan(
        mode=EvaluationMode.FULL_LABELED_CV,
        modeling_sample_ids=sample_ids,
    )
    engine = ProgressiveExecutionEngine(
        provider,
        config=ExecutionConfig(target_value=0.99, cv_folds=5, gbt=_gbt_config()),
    )

    result = engine.run(
        frozen_priority_order=(("CA",), ("EIC",)),
        plan=plan,
        targets_by_sample=targets,
    )

    assert result.status == SelectionStatus.TARGET_NOT_REACHED
    assert result.best_achieved_subset == result.selected_subset
    assert np.isfinite(result.best_achieved_score)
    assert result.rounds[-1].acquired_invariants == ("CA", "EIC")
    assert result.evaluation_sample_ids == sample_ids
    assert set(result.evaluation_fold_assignment) == set(sample_ids)
    assert set(result.evaluation_predictions) == {"CA", "EIC"}
    assert all(values.shape == (len(sample_ids),) for values in result.evaluation_predictions.values())

    report = _result_to_dict(result, targets_by_sample=targets)
    assert report["external_eval_predictions"] is None
    assert set(report["full_oof_predictions"]) == {"CA", "EIC"}
    assert set(report["evaluation_metrics"]["selected_subset"]) == {"PCC", "RMSE", "MAE", "R2"}
    assert set(report["evaluation_metrics"]["selected_subset_by_fold"]) == {"0", "1", "2", "3", "4"}


def test_acquisition_limit_pauses_frozen_queue_without_acquiring_next_invariant():
    sample_ids, targets, features = _linear_dataset(n_train=45, n_test=0, n_infer=0)
    provider = InMemoryFeatureProvider(features, sample_ids)
    plan = EvaluationPlan(
        mode=EvaluationMode.FULL_LABELED_CV,
        modeling_sample_ids=sample_ids,
    )
    engine = ProgressiveExecutionEngine(
        provider,
        config=ExecutionConfig(
            target_value=1.01,
            cv_folds=5,
            max_acquisitions=1,
            gbt=_gbt_config(),
        ),
    )

    result = engine.run(
        frozen_priority_order=(("PL",), ("PH",)),
        plan=plan,
        targets_by_sample=targets,
    )

    assert result.status == SelectionStatus.ACQUISITION_LIMIT_REACHED
    assert result.acquisition_order == ("PL",)
    assert len(result.rounds) == 1


def test_acquisition_limit_equal_to_full_queue_reports_target_not_reached():
    sample_ids, targets, features = _linear_dataset(n_train=45, n_test=0, n_infer=0)
    provider = InMemoryFeatureProvider(features, sample_ids)
    plan = EvaluationPlan(
        mode=EvaluationMode.FULL_LABELED_CV,
        modeling_sample_ids=sample_ids,
    )
    engine = ProgressiveExecutionEngine(
        provider,
        config=ExecutionConfig(
            target_value=1.01,
            cv_folds=5,
            max_acquisitions=2,
            gbt=_gbt_config(),
        ),
    )

    result = engine.run(
        frozen_priority_order=(("PL",), ("PH",)),
        plan=plan,
        targets_by_sample=targets,
    )

    assert result.status == SelectionStatus.TARGET_NOT_REACHED
    assert result.acquisition_order == ("PL", "PH")
    assert len(result.rounds) == 2


def test_inference_predictions_have_disagreement_for_multi_invariant_only():
    sample_ids, targets, features = _linear_dataset()
    provider = InMemoryFeatureProvider(features, sample_ids)
    plan = EvaluationPlan(
        mode=EvaluationMode.LABELED_CV_PLUS_UNLABELED_INFERENCE,
        modeling_sample_ids=sample_ids[:30],
        inference_sample_ids=sample_ids[42:],
    )
    engine = ProgressiveExecutionEngine(
        provider,
        config=ExecutionConfig(target_value=0.99, cv_folds=5, gbt=_gbt_config()),
    )

    result = engine.run(
        frozen_priority_order=(("PL", "PH"),),
        plan=plan,
        targets_by_sample=targets,
    )

    assert result.inference is not None
    assert result.inference.sample_ids == sample_ids[42:]
    assert result.inference.predictions.shape == (3,)
    if len(result.selected_subset) > 1:
        assert result.inference.disagreement_std is not None
        assert result.inference.disagreement_range is not None
    else:
        assert result.inference.disagreement_std is None
        assert result.inference.disagreement_range is None


def test_external_evaluation_predictions_are_kept_separate_from_full_oof():
    sample_ids, targets, features = _linear_dataset()
    provider = InMemoryFeatureProvider(features, sample_ids)
    plan = EvaluationPlan(
        mode=EvaluationMode.EXPLICIT_LABELED_TEST,
        modeling_sample_ids=sample_ids[:30],
        evaluation_sample_ids=sample_ids[30:42],
    )
    result = ProgressiveExecutionEngine(
        provider,
        config=ExecutionConfig(target_value=-1.0, gbt=_gbt_config()),
    ).run(
        frozen_priority_order=(("PL",),),
        plan=plan,
        targets_by_sample=targets,
    )

    report = _result_to_dict(result, targets_by_sample=targets)

    assert report["full_oof_predictions"] is None
    assert set(report["external_eval_predictions"]) == {"PL"}
    assert report["evaluation_fold_assignment"] == {}
    assert report["evaluation_metrics"]["selected_subset_by_fold"] == {}


def test_frozen_queue_is_not_reranked_by_later_scores():
    sample_ids, targets, features = _linear_dataset()
    provider = InMemoryFeatureProvider(features, sample_ids)
    plan = EvaluationPlan(
        mode=EvaluationMode.EXPLICIT_LABELED_TEST,
        modeling_sample_ids=sample_ids[:30],
        evaluation_sample_ids=sample_ids[30:42],
    )
    engine = ProgressiveExecutionEngine(
        provider,
        config=ExecutionConfig(target_value=1.01, gbt=_gbt_config()),
    )

    result = engine.run(
        frozen_priority_order=(("CA",), ("PL",)),
        plan=plan,
        targets_by_sample=targets,
    )

    assert result.acquisition_order == ("CA", "PL")


def test_selection_prefers_simplest_subset_after_target_passes():
    selected = _select_subset(
        {
            ("PL",): 0.81,
            ("PH",): 0.80,
            ("PH", "PL"): 0.92,
            ("CA", "PH", "PL"): 0.94,
        },
        is_higher=True,
        target_value=0.80,
        cost_by_invariant={"PL": 3.0, "PH": 1.0, "CA": 1.0},
    )

    assert selected == ("PL",)


def test_selection_uses_cost_after_size_and_score_tie():
    selected = _select_subset(
        {
            ("PL",): 0.81,
            ("PH",): 0.81,
            ("PH", "PL"): 0.95,
        },
        is_higher=True,
        target_value=0.80,
        cost_by_invariant={"PL": 3.0, "PH": 1.0},
    )

    assert selected == ("PH",)


def test_plbind_cached_provider_loads_and_flattens_legacy_outputs(tmp_path):
    class FakeTool:
        def __init__(self, root: Path):
            self.root = root

        def compute_with_spec(
            self,
            sample_id: str,
            invariant_name: str,
            representation_spec,
            *,
            dry_run: bool = False,
        ):
            assert representation_spec.spec_hash == legacy_spec.spec_hash
            path = self.root / invariant_name / f"{sample_id}.npy"
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, np.full((2, 3, 4), fill_value=float(len(sample_id))))
            return FeatureToolResult(
                sample_id=sample_id,
                invariant_name=invariant_name,
                legacy_name=invariant_name.lower(),
                output_path=path,
                cost=None,
                status="cached",
            )

    legacy_spec = make_legacy_casf_representation_spec()
    provider = PLBindCachedFeatureProvider(FakeTool(tmp_path), legacy_spec)

    artifact = provider.acquire("PL", ("aa", "bbbb"), representation_hash=legacy_spec.spec_hash)

    assert artifact.sample_ids == ("aa", "bbbb")
    assert artifact.cached_sample_ids == ("aa", "bbbb")
    assert artifact.computed_sample_ids == ()
    assert artifact.features.shape == (2, 24)
    assert np.all(artifact.features[0] == 2.0)
    assert np.all(artifact.features[1] == 4.0)


def test_plbind_provider_reads_qc_manifest_paths_without_recomputing(tmp_path):
    class ToolMustNotRun:
        def compute_with_spec(self, *args, **kwargs):
            raise AssertionError("precomputed manifest features must be loaded directly")

    feature_a = tmp_path / "content-addressed" / "a.npy"
    feature_b = tmp_path / "content-addressed" / "b.npy"
    feature_a.parent.mkdir()
    np.save(feature_a, np.full((2, 3, 4), 1.0))
    np.save(feature_b, np.full((2, 3, 4), 2.0))
    manifest = tmp_path / "PL.jsonl"
    manifest.write_text(
        "\n".join(
            json.dumps(
                {
                    "sample_id": sample_id,
                    "invariant": "PL",
                    "status": "computed",
                    "output_path": str(path),
                }
            )
            for sample_id, path in (("a", feature_a), ("b", feature_b))
        )
        + "\n",
        encoding="utf-8",
    )
    legacy_spec = make_legacy_casf_representation_spec()
    provider = PLBindCachedFeatureProvider(
        ToolMustNotRun(),
        legacy_spec,
        manifest_paths={"PL": manifest},
    )

    artifact = provider.acquire(
        "PL",
        ("b", "a"),
        representation_hash=legacy_spec.spec_hash,
    )

    assert artifact.sample_ids == ("b", "a")
    assert artifact.cached_sample_ids == ("b", "a")
    assert artifact.computed_sample_ids == ()
    assert artifact.features.shape == (2, 24)
    assert np.all(artifact.features[0] == 2.0)
    assert np.all(artifact.features[1] == 1.0)
