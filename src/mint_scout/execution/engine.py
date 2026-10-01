from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from mint_scout.evaluation.ensemble import Subset, enumerate_nonempty_subsets, mean_aggregate
from mint_scout.evaluation.metrics import get_metric, higher_is_better
from mint_scout.execution.artifacts import ScoutExecutionArtifact
from mint_scout.execution.provider import FeatureArtifact, FeatureProvider
from mint_scout.models.gbt import GBTConfig, fit_gbt_ensemble, fit_predict_gbt, make_fold_ids, run_oof_gbt
from mint_scout.schemas import EvaluationMode, EvaluationPlan


class SelectionStatus(str, Enum):
    TARGET_REACHED = "TARGET_REACHED"
    TARGET_NOT_REACHED = "TARGET_NOT_REACHED"
    ACQUISITION_LIMIT_REACHED = "ACQUISITION_LIMIT_REACHED"


@dataclass(frozen=True)
class ExecutionConfig:
    target_metric: str = "PCC"
    target_value: float = 0.8
    cv_folds: int = 5
    representation_hash: str = "default"
    require_shared_fold_assignment: bool = False
    require_precomputed: bool = False
    max_acquisitions: int | None = None
    gbt: GBTConfig = field(default_factory=GBTConfig)

    def __post_init__(self) -> None:
        if self.max_acquisitions is not None and self.max_acquisitions < 1:
            raise ValueError("max_acquisitions must be positive when provided")


@dataclass(frozen=True)
class AcquisitionRecord:
    invariant: str
    cached_count: int
    computed_count: int
    cost: float


@dataclass(frozen=True)
class EvaluationRound:
    acquired_invariants: tuple[str, ...]
    subset_scores: Mapping[Subset, float]
    selected_subset: Subset
    selected_score: float
    target_reached: bool


@dataclass(frozen=True)
class InferenceResult:
    sample_ids: tuple[str, ...]
    predictions: np.ndarray
    disagreement_std: np.ndarray | None
    disagreement_range: np.ndarray | None


@dataclass(frozen=True)
class ExecutionResult:
    status: SelectionStatus
    selected_subset: Subset
    selected_score: float
    target_metric: str
    target_value: float
    acquisition_order: tuple[str, ...]
    acquisition_records: tuple[AcquisitionRecord, ...]
    rounds: tuple[EvaluationRound, ...]
    best_achieved_subset: Subset
    best_achieved_score: float
    representation_hash: str
    gbt_parameter_hash: str
    evaluation_mode: EvaluationMode
    n_acceptance_queries: int
    evaluation_set_used_for_acceptance: bool
    untouched_final_test: bool | None
    evaluation_sample_ids: tuple[str, ...]
    evaluation_fold_assignment: Mapping[str, int]
    evaluation_predictions: Mapping[str, np.ndarray]
    inference: InferenceResult | None = None


class ProgressiveExecutionEngine:
    def __init__(self, provider: FeatureProvider, *, config: ExecutionConfig = ExecutionConfig()):
        self.provider = provider
        self.config = config

    def run(
        self,
        *,
        frozen_priority_order: Sequence[Sequence[str]],
        plan: EvaluationPlan,
        targets_by_sample: Mapping[str, float],
        inference_sample_ids: Sequence[str] = (),
        full_fold_assignment: Mapping[str, int] | None = None,
    ) -> ExecutionResult:
        if plan.mode not in {
            EvaluationMode.EXPLICIT_LABELED_TEST,
            EvaluationMode.FULL_LABELED_CV,
            EvaluationMode.LABELED_CV_PLUS_UNLABELED_INFERENCE,
        }:
            raise ValueError(f"Unsupported execution mode: {plan.mode.value}")

        queue = _flatten_frozen_queue(frozen_priority_order)
        if not queue:
            raise ValueError("frozen_priority_order must include at least one invariant")
        modeling_ids = tuple(plan.modeling_sample_ids)
        evaluation_ids = tuple(plan.evaluation_sample_ids)
        inference_ids = tuple(inference_sample_ids or plan.inference_sample_ids)
        _assert_targets_present(targets_by_sample, modeling_ids)
        if plan.mode == EvaluationMode.EXPLICIT_LABELED_TEST:
            _assert_targets_present(targets_by_sample, evaluation_ids)
        shared_folds = _validated_full_fold_assignment(
            plan=plan,
            modeling_ids=modeling_ids,
            fold_assignment=full_fold_assignment,
            cv_folds=self.config.cv_folds,
            required=self.config.require_shared_fold_assignment,
        )
        if plan.mode != EvaluationMode.EXPLICIT_LABELED_TEST and shared_folds is None:
            shared_folds = make_fold_ids(
                modeling_ids,
                n_folds=min(self.config.cv_folds, len(modeling_ids)),
            )

        acquired_features: dict[str, FeatureArtifact] = {}
        evaluation_predictions: dict[str, np.ndarray] = {}
        final_models: dict[str, object] = {}
        acquisition_records: list[AcquisitionRecord] = []
        rounds: list[EvaluationRound] = []
        best_subset: Subset | None = None
        best_score = _worst_score(self.config.target_metric)
        is_higher = higher_is_better(self.config.target_metric)

        for required_subset in frozen_priority_order:
            missing = [name.upper() for name in required_subset if name.upper() not in acquired_features]
            for invariant in missing:
                acquisition = self._acquire_and_evaluate(
                    invariant=invariant,
                    plan=plan,
                    modeling_ids=modeling_ids,
                    evaluation_ids=evaluation_ids,
                    targets_by_sample=targets_by_sample,
                    final_models=final_models,
                    full_fold_assignment=shared_folds,
                    needs_final_model=bool(inference_ids),
                )
                acquired_features[invariant] = acquisition.artifact
                evaluation_predictions[invariant] = acquisition.evaluation_prediction
                acquisition_records.append(acquisition.record)

                round_result = _evaluate_available_subsets(
                    predictions=evaluation_predictions,
                    y_true=_evaluation_targets(plan, targets_by_sample),
                    metric_name=self.config.target_metric,
                    target_value=self.config.target_value,
                    cost_by_invariant={record.invariant: record.cost for record in acquisition_records},
                )
                rounds.append(round_result)
                if _is_better(round_result.selected_score, best_score, is_higher):
                    best_subset = round_result.selected_subset
                    best_score = round_result.selected_score
                if round_result.target_reached:
                    return self._result(
                        status=SelectionStatus.TARGET_REACHED,
                        selected_subset=round_result.selected_subset,
                        selected_score=round_result.selected_score,
                        acquisition_records=acquisition_records,
                        rounds=rounds,
                        plan=plan,
                        final_models=final_models,
                        inference_ids=inference_ids,
                        evaluation_predictions=evaluation_predictions,
                        full_fold_assignment=shared_folds,
                    )
                if (
                    self.config.max_acquisitions is not None
                    and len(acquisition_records) >= self.config.max_acquisitions
                    and len(acquisition_records) < len(queue)
                ):
                    return self._result(
                        status=SelectionStatus.ACQUISITION_LIMIT_REACHED,
                        selected_subset=round_result.selected_subset,
                        selected_score=round_result.selected_score,
                        acquisition_records=acquisition_records,
                        rounds=rounds,
                        plan=plan,
                        final_models=final_models,
                        inference_ids=inference_ids,
                        evaluation_predictions=evaluation_predictions,
                        full_fold_assignment=shared_folds,
                    )

        if best_subset is None:
            raise AssertionError("At least one evaluation round must be produced")
        return self._result(
            status=SelectionStatus.TARGET_NOT_REACHED,
            selected_subset=best_subset,
            selected_score=best_score,
            acquisition_records=acquisition_records,
            rounds=rounds,
            plan=plan,
            final_models=final_models,
            inference_ids=inference_ids,
            evaluation_predictions=evaluation_predictions,
            full_fold_assignment=shared_folds,
        )

    def run_from_scout(
        self,
        *,
        artifact: ScoutExecutionArtifact,
        plan: EvaluationPlan,
        targets_by_sample: Mapping[str, float],
        inference_sample_ids: Sequence[str] = (),
    ) -> ExecutionResult:
        if tuple(plan.modeling_sample_ids) != artifact.modeling_sample_ids:
            raise ValueError("Evaluation plan modeling sample ids do not match Scout artifact")
        if self.config.representation_hash != artifact.representation_hash:
            raise ValueError("Execution representation hash does not match Scout artifact")
        if self.config.gbt.parameter_hash != artifact.gbt_parameter_hash:
            raise ValueError("Execution GBT configuration does not match Scout artifact")
        if self.config.target_metric.upper() != artifact.target_metric.upper():
            raise ValueError("Execution target metric does not match Scout artifact")
        if self.config.target_value != artifact.target_value:
            raise ValueError("Execution target value does not match Scout artifact")
        return self.run(
            frozen_priority_order=artifact.frozen_priority_order,
            plan=plan,
            targets_by_sample=targets_by_sample,
            inference_sample_ids=inference_sample_ids,
            full_fold_assignment=artifact.full_fold_assignment,
        )

    def _acquire_and_evaluate(
        self,
        *,
        invariant: str,
        plan: EvaluationPlan,
        modeling_ids: tuple[str, ...],
        evaluation_ids: tuple[str, ...],
        targets_by_sample: Mapping[str, float],
        final_models: dict[str, object],
        full_fold_assignment: Mapping[str, int] | None,
        needs_final_model: bool,
    ) -> "_EvaluatedAcquisition":
        sample_ids = _feature_sample_ids(modeling_ids, evaluation_ids, tuple(plan.inference_sample_ids))
        artifact = self.provider.acquire(
            invariant,
            sample_ids,
            representation_hash=self.config.representation_hash,
        )
        if self.config.require_precomputed and artifact.computed_sample_ids:
            raise RuntimeError(
                f"{invariant} produced {len(artifact.computed_sample_ids)} new feature files after QC; "
                "rerun feature QC before modeling"
            )
        features = _features_by_sample_id(artifact)
        y_model = _targets_for(modeling_ids, targets_by_sample)

        if plan.mode == EvaluationMode.EXPLICIT_LABELED_TEST:
            prediction = fit_predict_gbt(
                _stack_features(modeling_ids, features),
                y_model,
                _stack_features(evaluation_ids, features),
                config=self.config.gbt,
            )
        else:
            if full_fold_assignment is None:
                raise AssertionError("Full labeled CV requires a shared fold assignment")
            fold_ids = dict(full_fold_assignment)
            prediction = run_oof_gbt(
                _stack_features(modeling_ids, features),
                y_model,
                modeling_ids,
                fold_ids,
                config=self.config.gbt,
            ).y_pred

        if needs_final_model:
            final_models[invariant] = fit_gbt_ensemble(
                _stack_features(modeling_ids, features),
                y_model,
                config=self.config.gbt,
            )
        return _EvaluatedAcquisition(
            artifact=artifact,
            evaluation_prediction=prediction,
            record=AcquisitionRecord(
                invariant=invariant,
                cached_count=len(artifact.cached_sample_ids),
                computed_count=len(artifact.computed_sample_ids),
                cost=artifact.cost,
            ),
        )

    def _result(
        self,
        *,
        status: SelectionStatus,
        selected_subset: Subset,
        selected_score: float,
        acquisition_records: list[AcquisitionRecord],
        rounds: list[EvaluationRound],
        plan: EvaluationPlan,
        final_models: Mapping[str, object],
        inference_ids: tuple[str, ...],
        evaluation_predictions: Mapping[str, np.ndarray],
        full_fold_assignment: Mapping[str, int] | None,
    ) -> ExecutionResult:
        inference = None
        if inference_ids:
            inference = self._predict_inference(
                selected_subset=selected_subset,
                final_models=final_models,
                sample_ids=inference_ids,
            )
        evaluation_sample_ids = (
            tuple(plan.evaluation_sample_ids)
            if plan.mode == EvaluationMode.EXPLICIT_LABELED_TEST
            else tuple(plan.modeling_sample_ids)
        )
        frozen_predictions: dict[str, np.ndarray] = {}
        for invariant, values in evaluation_predictions.items():
            prediction = np.asarray(values, dtype=float).copy()
            if prediction.shape != (len(evaluation_sample_ids),):
                raise ValueError(f"{invariant} evaluation predictions do not align with sample IDs")
            prediction.setflags(write=False)
            frozen_predictions[invariant] = prediction
        return ExecutionResult(
            status=status,
            selected_subset=selected_subset,
            selected_score=selected_score,
            target_metric=self.config.target_metric.upper(),
            target_value=self.config.target_value,
            acquisition_order=tuple(record.invariant for record in acquisition_records),
            acquisition_records=tuple(acquisition_records),
            rounds=tuple(rounds),
            best_achieved_subset=selected_subset,
            best_achieved_score=selected_score,
            representation_hash=self.config.representation_hash,
            gbt_parameter_hash=self.config.gbt.parameter_hash,
            evaluation_mode=plan.mode,
            n_acceptance_queries=len(rounds),
            evaluation_set_used_for_acceptance=(
                plan.mode == EvaluationMode.EXPLICIT_LABELED_TEST and bool(rounds)
            ),
            untouched_final_test=(
                False if plan.mode == EvaluationMode.EXPLICIT_LABELED_TEST and bool(rounds) else None
            ),
            evaluation_sample_ids=evaluation_sample_ids,
            evaluation_fold_assignment=MappingProxyType(dict(full_fold_assignment or {})),
            evaluation_predictions=MappingProxyType(frozen_predictions),
            inference=inference,
        )

    def _predict_inference(
        self,
        *,
        selected_subset: Subset,
        final_models: Mapping[str, object],
        sample_ids: tuple[str, ...],
    ) -> InferenceResult:
        predictions = []
        for invariant in selected_subset:
            artifact = self.provider.acquire(
                invariant,
                sample_ids,
                representation_hash=self.config.representation_hash,
            )
            model = final_models[invariant]
            predictions.append(model.predict(artifact.features))
        matrix = np.vstack(predictions)
        return InferenceResult(
            sample_ids=sample_ids,
            predictions=np.mean(matrix, axis=0),
            disagreement_std=np.std(matrix, axis=0) if len(selected_subset) > 1 else None,
            disagreement_range=np.ptp(matrix, axis=0) if len(selected_subset) > 1 else None,
        )


@dataclass(frozen=True)
class _EvaluatedAcquisition:
    artifact: FeatureArtifact
    evaluation_prediction: np.ndarray
    record: AcquisitionRecord


def _flatten_frozen_queue(priority_order: Sequence[Sequence[str]]) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for subset in priority_order:
        for invariant in subset:
            name = invariant.upper()
            if name not in seen:
                seen.add(name)
                ordered.append(name)
    return tuple(ordered)


def _feature_sample_ids(
    modeling_ids: tuple[str, ...],
    evaluation_ids: tuple[str, ...],
    inference_ids: tuple[str, ...],
) -> tuple[str, ...]:
    return tuple(dict.fromkeys(modeling_ids + evaluation_ids + inference_ids))


def _evaluation_targets(plan: EvaluationPlan, targets_by_sample: Mapping[str, float]) -> np.ndarray:
    if plan.mode == EvaluationMode.EXPLICIT_LABELED_TEST:
        return _targets_for(plan.evaluation_sample_ids, targets_by_sample)
    return _targets_for(plan.modeling_sample_ids, targets_by_sample)


def _targets_for(sample_ids: Sequence[str], targets_by_sample: Mapping[str, float]) -> np.ndarray:
    return np.asarray([targets_by_sample[sample_id] for sample_id in sample_ids], dtype=float)


def _assert_targets_present(targets_by_sample: Mapping[str, float], sample_ids: Sequence[str]) -> None:
    missing = [sample_id for sample_id in sample_ids if sample_id not in targets_by_sample]
    if missing:
        raise ValueError(f"Missing target values for samples: {missing[:10]}")


def _features_by_sample_id(artifact: FeatureArtifact) -> dict[str, np.ndarray]:
    if artifact.features.shape[0] != len(artifact.sample_ids):
        raise ValueError("Feature artifact rows do not align with sample ids")
    if not np.all(np.isfinite(artifact.features)):
        raise ValueError(f"{artifact.invariant} features must be finite")
    return {sample_id: artifact.features[index] for index, sample_id in enumerate(artifact.sample_ids)}


def _stack_features(sample_ids: Sequence[str], features: Mapping[str, np.ndarray]) -> np.ndarray:
    return np.asarray([features[sample_id] for sample_id in sample_ids], dtype=float)


def _evaluate_available_subsets(
    *,
    predictions: Mapping[str, np.ndarray],
    y_true: np.ndarray,
    metric_name: str,
    target_value: float,
    cost_by_invariant: Mapping[str, float],
) -> EvaluationRound:
    metric = get_metric(metric_name)
    is_higher = higher_is_better(metric_name)
    subset_scores = {
        subset: float(metric(y_true, mean_aggregate(predictions, subset)))
        for subset in enumerate_nonempty_subsets(predictions.keys())
    }
    selected_subset = _select_subset(
        subset_scores,
        is_higher=is_higher,
        target_value=target_value,
        cost_by_invariant=cost_by_invariant,
    )
    selected_score = subset_scores[selected_subset]
    return EvaluationRound(
        acquired_invariants=tuple(sorted(predictions)),
        subset_scores=subset_scores,
        selected_subset=selected_subset,
        selected_score=selected_score,
        target_reached=_target_reached(selected_score, target_value, is_higher),
    )


def _select_subset(
    subset_scores: Mapping[Subset, float],
    *,
    is_higher: bool,
    target_value: float,
    cost_by_invariant: Mapping[str, float],
) -> Subset:
    finite_subsets = [subset for subset, score in subset_scores.items() if np.isfinite(score)]
    if not finite_subsets:
        raise ValueError("No finite subset scores were produced")
    passing = [subset for subset in finite_subsets if _target_reached(subset_scores[subset], target_value, is_higher)]
    if passing:
        return min(
            passing,
            key=lambda subset: (
                len(subset),
                _performance_key(subset_scores[subset], is_higher),
                _subset_cost(subset, cost_by_invariant),
                subset,
            ),
        )
    best_score = min(_performance_key(subset_scores[subset], is_higher) for subset in finite_subsets)
    passing = [subset for subset in finite_subsets if _performance_key(subset_scores[subset], is_higher) == best_score]
    return min(
        passing,
        key=lambda subset: (
            len(subset),
            _subset_cost(subset, cost_by_invariant),
            subset,
        ),
    )


def _target_reached(score: float, target: float, is_higher: bool) -> bool:
    if not np.isfinite(score):
        return False
    return score >= target if is_higher else score <= target


def _is_better(score: float, incumbent: float, is_higher: bool) -> bool:
    return _performance_key(score, is_higher) < _performance_key(incumbent, is_higher)


def _performance_key(value: float, is_higher: bool) -> float:
    if not np.isfinite(value):
        return float("inf")
    return -float(value) if is_higher else float(value)


def _worst_score(metric_name: str) -> float:
    return -float("inf") if higher_is_better(metric_name) else float("inf")


def _subset_cost(subset: Subset, cost_by_invariant: Mapping[str, float]) -> float:
    return float(sum(cost_by_invariant.get(name, 0.0) for name in subset))


def _validated_full_fold_assignment(
    *,
    plan: EvaluationPlan,
    modeling_ids: tuple[str, ...],
    fold_assignment: Mapping[str, int] | None,
    cv_folds: int,
    required: bool,
) -> Mapping[str, int] | None:
    if plan.mode == EvaluationMode.EXPLICIT_LABELED_TEST:
        return None
    if fold_assignment is None:
        if required:
            raise ValueError("A frozen shared full-fold assignment is required")
        return None
    if set(fold_assignment) != set(modeling_ids):
        raise ValueError("full_fold_assignment must contain exactly the modeling sample ids")
    if len(set(fold_assignment.values())) != min(cv_folds, len(modeling_ids)):
        raise ValueError("full_fold_assignment has the wrong number of distinct folds")
    return dict(fold_assignment)
