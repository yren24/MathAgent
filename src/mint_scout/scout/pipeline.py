from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from mint_scout.data.splits import make_kfold_assignments
from mint_scout.execution.artifacts import ScoutExecutionArtifact
from mint_scout.invariants.manifest import stable_hash
from mint_scout.models.gbt import GBTConfig
from mint_scout.sample_size import LabeledSamplePolicy
from mint_scout.scout.oof import AlignedOOFStore, run_shared_oof_gbt
from mint_scout.scout.sampling import ElementPair, ProbeSamplingConfig, ProbeSelection, select_probe_samples
from mint_scout.scout.stability import (
    BootstrapConfig,
    RankingWeights,
    ScoutRanking,
    rank_consensus_subsets,
)
from mint_scout.scout.target import TargetConfig, TargetSpec, resolve_target


@dataclass(frozen=True)
class ScoutConfig:
    primary_metric: str = "PCC"
    secondary_metrics: tuple[str, ...] = ("RMSE", "MAE", "R2")
    cv_folds: int = 5
    fold_seed: int = 2026
    probe: ProbeSamplingConfig = field(default_factory=ProbeSamplingConfig)
    gbt: GBTConfig = field(default_factory=GBTConfig)
    bootstrap: BootstrapConfig = field(default_factory=BootstrapConfig)
    ranking_weights: RankingWeights = field(default_factory=RankingWeights)
    target: TargetConfig = field(default_factory=TargetConfig)
    labels: LabeledSamplePolicy = field(default_factory=LabeledSamplePolicy)

    def __post_init__(self) -> None:
        if self.cv_folds < 2:
            raise ValueError("cv_folds must be at least 2")

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> "ScoutConfig":
        probe_data = dict(config.get("probe", {}))
        bootstrap_data = dict(config.get("bootstrap", {}))
        target_data = dict(config.get("target", {}))
        ranking_data = dict(config.get("ranking", {}).get("weights", {}))
        label_data = dict(config.get("labels", {}))
        model_data = dict(config.get("model", {}))
        metrics_data = dict(config.get("metrics", {}))
        consensus_data = dict(config.get("consensus", {}))
        if bool(model_data.get("hpo", False)):
            raise ValueError("V1 Scout forbids GBT hyperparameter optimization")
        if model_data.get("family", "fixed_gbt") != "fixed_gbt":
            raise ValueError("V1 Scout supports only model.family=fixed_gbt")
        if bool(consensus_data.get("weighted", False)):
            raise ValueError("V1 Scout forbids learned or configured consensus weights")
        if consensus_data.get("aggregation", "arithmetic_mean") != "arithmetic_mean":
            raise ValueError("V1 Scout requires arithmetic_mean consensus")
        if "enabled" in bootstrap_data and not bool(bootstrap_data.pop("enabled")):
            raise ValueError("V1 Scout requires bootstrap stability analysis")
        if "stratified" in bootstrap_data and not bool(bootstrap_data.pop("stratified")):
            raise ValueError("V1 Scout requires stratified paired bootstrap")
        target_precedence = target_data.pop("user_target_takes_precedence", True)
        if not bool(target_precedence):
            raise ValueError("V1 Scout requires user targets to take precedence")
        gbt_params = dict(model_data.get("params", {}))
        if "seed" in bootstrap_data and "random_seed" not in bootstrap_data:
            bootstrap_data["random_seed"] = bootstrap_data.pop("seed")
        if "n_bootstrap" in bootstrap_data and "replicates" not in bootstrap_data:
            bootstrap_data["replicates"] = bootstrap_data.pop("n_bootstrap")
        if "ci" in bootstrap_data and "ci_level" not in bootstrap_data:
            bootstrap_data["ci_level"] = bootstrap_data.pop("ci")
        return cls(
            primary_metric=str(metrics_data.get("primary", config.get("primary_metric", "PCC"))),
            secondary_metrics=tuple(metrics_data.get("secondary", ("RMSE", "MAE", "R2"))),
            cv_folds=int(probe_data.pop("cv_folds", config.get("cv_folds", 5))),
            fold_seed=int(config.get("fold_seed", probe_data.get("random_seed", 2026))),
            probe=ProbeSamplingConfig(**probe_data),
            gbt=GBTConfig(**gbt_params),
            bootstrap=BootstrapConfig(**bootstrap_data),
            ranking_weights=RankingWeights(**ranking_data),
            target=TargetConfig(**target_data),
            labels=LabeledSamplePolicy.from_mapping(label_data),
        )


@dataclass(frozen=True)
class ProbeResult:
    modeling_sample_ids: tuple[str, ...]
    selection: ProbeSelection
    fold_assignment: Mapping[str, int]
    modeling_fold_assignment: Mapping[str, int]
    oof_store: AlignedOOFStore
    ranking: ScoutRanking
    target: TargetSpec
    probe_hash: str
    gbt_parameter_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "probe_sample_ids": list(self.selection.sample_ids),
            "modeling_sample_ids": list(self.modeling_sample_ids),
            "probe_size": self.selection.final_size,
            "probe_base_size": self.selection.base_size,
            "fold_assignment": dict(self.fold_assignment),
            "modeling_fold_assignment": dict(self.modeling_fold_assignment),
            "invariant_oof_predictions": {
                name: {
                    sample_id: float(values[index])
                    for index, sample_id in enumerate(self.selection.sample_ids)
                }
                for name, values in self.oof_store.predictions.items()
            },
            "consensus_metrics": [
                {
                    **asdict(summary),
                    "invariants": list(summary.invariants),
                    "secondary_metrics": dict(summary.secondary_metrics),
                }
                for summary in self.ranking.summaries
            ],
            "top_tier": [list(subset) for subset in self.ranking.top_tier],
            "frozen_priority_order": [list(subset) for subset in self.ranking.priority_order],
            "target": asdict(self.target),
            "warnings": list(self.selection.warnings + self.target.warnings),
            "probe_hash": self.probe_hash,
            "gbt_parameter_hash": self.gbt_parameter_hash,
        }

    def to_execution_artifact(self, *, representation_hash: str) -> ScoutExecutionArtifact:
        return ScoutExecutionArtifact(
            artifact_version="mint-agent.scout-execution.v1",
            modeling_sample_ids=self.modeling_sample_ids,
            representation_hash=representation_hash,
            frozen_priority_order=self.ranking.priority_order,
            full_fold_assignment=self.modeling_fold_assignment,
            target_metric=self.target.metric.upper(),
            target_direction=self.target.direction,
            target_value=self.target.value,
            target_source=self.target.source,
            gbt_parameter_hash=self.gbt_parameter_hash,
            probe_hash=self.probe_hash,
            ranking_policy="hierarchical_empirical_v1",
        )


def run_scout(
    *,
    sample_ids: Sequence[str],
    targets: Sequence[float],
    structure_sizes: Sequence[float],
    features_by_invariant: Mapping[str, np.ndarray],
    retained_pairs: Sequence[ElementPair] = (),
    pair_presence_by_sample: Mapping[str, Sequence[ElementPair]] | None = None,
    user_target: float | None = None,
    modeling_fold_assignment: Mapping[str, int] | None = None,
    config: ScoutConfig = ScoutConfig(),
) -> ProbeResult:
    ids = tuple(sample_ids)
    target_array = np.asarray(targets, dtype=float)
    selection = select_probe_samples(
        sample_ids=ids,
        targets=target_array,
        structure_sizes=structure_sizes,
        retained_pairs=retained_pairs,
        pair_presence_by_sample=pair_presence_by_sample,
        config=config.probe,
    )
    if selection.final_size < config.cv_folds:
        raise ValueError("probe size is smaller than the configured number of CV folds")
    position = {sample_id: index for index, sample_id in enumerate(ids)}
    probe_indices = np.asarray([position[sample_id] for sample_id in selection.sample_ids], dtype=int)
    probe_targets = target_array[probe_indices]
    probe_features: dict[str, np.ndarray] = {}
    for invariant_name, values in features_by_invariant.items():
        array = np.asarray(values)
        if array.shape[0] != len(ids):
            raise ValueError(f"{invariant_name} features do not align with modeling sample ids")
        probe_features[invariant_name.upper()] = array[probe_indices]
    if not probe_features:
        raise ValueError("features_by_invariant cannot be empty")

    full_fold_assignment = dict(
        modeling_fold_assignment
        or make_kfold_assignments(ids, n_splits=config.cv_folds, seed=config.fold_seed)
    )
    return run_scout_with_frozen_probe(
        modeling_sample_ids=ids,
        selection=selection,
        probe_targets=probe_targets,
        features_by_invariant=probe_features,
        modeling_fold_assignment=full_fold_assignment,
        user_target=user_target,
        config=config,
    )


def run_scout_with_frozen_probe(
    *,
    modeling_sample_ids: Sequence[str],
    selection: ProbeSelection,
    probe_targets: Sequence[float],
    features_by_invariant: Mapping[str, np.ndarray],
    modeling_fold_assignment: Mapping[str, int],
    user_target: float | None = None,
    config: ScoutConfig = ScoutConfig(),
) -> ProbeResult:
    ids = tuple(modeling_sample_ids)
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("modeling_sample_ids must be nonempty and unique")
    if not set(selection.sample_ids).issubset(ids):
        raise ValueError("Frozen probe contains samples outside the modeling pool")
    if selection.final_size != len(selection.sample_ids):
        raise ValueError("Frozen probe final_size does not match sample_ids")
    target_array = np.asarray(probe_targets, dtype=float)
    if target_array.shape != (selection.final_size,) or not np.all(np.isfinite(target_array)):
        raise ValueError("probe_targets must be a finite vector aligned with the frozen probe")
    probe_features: dict[str, np.ndarray] = {}
    for invariant_name, values in features_by_invariant.items():
        array = np.asarray(values)
        if array.shape[0] != selection.final_size:
            raise ValueError(f"{invariant_name} features do not align with the frozen probe")
        probe_features[invariant_name.upper()] = array
    if not probe_features:
        raise ValueError("features_by_invariant cannot be empty")

    full_fold_assignment = dict(modeling_fold_assignment)
    if set(full_fold_assignment) != set(ids):
        raise ValueError("modeling_fold_assignment must contain exactly the modeling sample ids")
    if len(set(full_fold_assignment.values())) != config.cv_folds:
        raise ValueError("modeling_fold_assignment must contain exactly cv_folds distinct folds")
    fold_assignment = {
        sample_id: full_fold_assignment[sample_id] for sample_id in selection.sample_ids
    }
    oof_store = run_shared_oof_gbt(
        features_by_invariant=probe_features,
        targets=target_array,
        sample_ids=selection.sample_ids,
        fold_ids=fold_assignment,
        config=config.gbt,
    )
    ranking = rank_consensus_subsets(
        y_true=target_array,
        predictions=oof_store.predictions,
        primary_metric=config.primary_metric,
        secondary_metrics=config.secondary_metrics,
        strata=[selection.target_bins[sample_id] for sample_id in selection.sample_ids],
        bootstrap=config.bootstrap,
        ranking_weights=config.ranking_weights,
    )
    target = resolve_target(
        metric=config.primary_metric,
        ranking=ranking,
        user_target=user_target,
        config=config.target,
    )
    prediction_hashes = {
        name: hashlib.sha256(np.asarray(values).tobytes()).hexdigest()
        for name, values in oof_store.predictions.items()
    }
    probe_hash = stable_hash(
        {
            "sample_ids": selection.sample_ids,
            "fold_assignment": fold_assignment,
            "gbt_parameter_hash": config.gbt.parameter_hash,
            "primary_metric": config.primary_metric,
            "prediction_hashes": prediction_hashes,
            "priority_order": ranking.priority_order,
        }
    )
    return ProbeResult(
        modeling_sample_ids=ids,
        selection=selection,
        fold_assignment=fold_assignment,
        modeling_fold_assignment=full_fold_assignment,
        oof_store=oof_store,
        ranking=ranking,
        target=target,
        probe_hash=probe_hash,
        gbt_parameter_hash=config.gbt.parameter_hash,
    )
