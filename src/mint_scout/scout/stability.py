from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np
from scipy.stats import rankdata

from mint_scout.evaluation.ensemble import Subset, enumerate_nonempty_subsets, mean_aggregate
from mint_scout.evaluation.metrics import get_metric, higher_is_better
from mint_scout.scout.sampling import stratified_bootstrap_indices


@dataclass(frozen=True)
class BootstrapConfig:
    replicates: int = 1000
    ci_level: float = 0.95
    random_seed: int = 2026

    def __post_init__(self) -> None:
        if self.replicates < 1:
            raise ValueError("bootstrap replicates must be positive")
        if not 0.0 < self.ci_level < 1.0:
            raise ValueError("bootstrap ci_level must be in (0, 1)")


@dataclass(frozen=True)
class RankingWeights:
    probe_performance: float = 0.55
    ranking_stability: float = 0.20
    feature_quality: float = 0.15
    computational_efficiency: float = 0.10
    llm_prior: float = 0.00

    def __post_init__(self) -> None:
        values = tuple(asdict(self).values())
        if any(not np.isfinite(value) or value < 0.0 for value in values):
            raise ValueError("ranking weights must be finite and non-negative")
        if not np.isclose(sum(values), 1.0, rtol=0.0, atol=1.0e-12):
            raise ValueError("ranking weights must sum to one")


@dataclass(frozen=True)
class SubsetSummary:
    subset_id: str
    invariants: Subset
    subset_size: int
    nominal_primary_score: float
    secondary_metrics: Mapping[str, float]
    bootstrap_mean: float
    bootstrap_median: float
    bootstrap_std: float
    bootstrap_ci_low: float
    bootstrap_ci_high: float
    top1_frequency: float
    median_rank: float
    top_tier: bool
    initial_full_acquisition_count: int
    estimated_full_acquisition_wall_seconds: float
    probe_performance_component: float | None = None
    ranking_stability_component: float | None = None
    feature_quality_component: float | None = None
    computational_efficiency_component: float | None = None
    llm_prior_component: float | None = None
    final_score: float | None = None


@dataclass(frozen=True)
class ScoutRanking:
    primary_metric: str
    nominal_best: Subset
    summaries: tuple[SubsetSummary, ...]
    top_tier: tuple[Subset, ...]
    priority_order: tuple[Subset, ...]
    bootstrap_scores: Mapping[Subset, np.ndarray]
    bootstrap_index_hashes: tuple[str, ...]

    def summary_for(self, subset: Sequence[str]) -> SubsetSummary:
        key = tuple(subset)
        for summary in self.summaries:
            if summary.invariants == key:
                return summary
        raise KeyError(f"Unknown invariant subset: {key}")


def rank_consensus_subsets(
    *,
    y_true: Sequence[float],
    predictions: Mapping[str, Sequence[float]],
    primary_metric: str = "PCC",
    secondary_metrics: Sequence[str] = ("RMSE", "MAE", "R2"),
    strata: Sequence[int] | None = None,
    bootstrap: BootstrapConfig = BootstrapConfig(),
    invariant_costs: Mapping[str, float] | None = None,
    invariant_quality_scores: Mapping[str, float] | None = None,
    llm_prior_order: Sequence[Sequence[str]] | None = None,
    ranking_weights: RankingWeights | None = None,
) -> ScoutRanking:
    y = np.asarray(y_true, dtype=float)
    pred_arrays = {name.upper(): np.asarray(values, dtype=float) for name, values in predictions.items()}
    _validate_prediction_arrays(y, pred_arrays)
    subsets = enumerate_nonempty_subsets(pred_arrays)
    if not subsets:
        raise ValueError("At least one invariant prediction vector is required")
    costs = _normalize_invariant_costs(pred_arrays, invariant_costs)
    subset_costs = {
        subset: float(sum(costs[invariant] for invariant in subset))
        for subset in subsets
    }

    metric = get_metric(primary_metric)
    is_higher = higher_is_better(primary_metric)
    consensus_predictions = {
        subset: mean_aggregate(pred_arrays, subset)
        for subset in subsets
    }
    nominal_scores = {
        subset: float(metric(y, consensus_predictions[subset]))
        for subset in subsets
    }
    nominal_best = min(
        subsets,
        key=lambda subset: (_performance_key(nominal_scores[subset], is_higher), subset),
    )
    if not np.isfinite(nominal_scores[nominal_best]):
        raise ValueError(f"No finite {primary_metric} score is available for consensus ranking")

    if strata is None:
        strata_array = np.zeros(len(y), dtype=int)
    else:
        strata_array = np.asarray(strata, dtype=int)
        if len(strata_array) != len(y):
            raise ValueError("bootstrap strata must align with y_true")
    bootstrap_indices = stratified_bootstrap_indices(
        strata_array,
        n_replicates=bootstrap.replicates,
        seed=bootstrap.random_seed,
    )
    score_matrix = np.full((bootstrap.replicates, len(subsets)), np.nan, dtype=float)
    rank_matrix = np.full_like(score_matrix, np.nan)
    top_counts = np.zeros(len(subsets), dtype=float)
    valid_replicates = 0
    for replicate, indices in enumerate(bootstrap_indices):
        y_boot = y[indices]
        replicate_scores = np.asarray(
            [metric(y_boot, consensus_predictions[subset][indices]) for subset in subsets],
            dtype=float,
        )
        score_matrix[replicate] = replicate_scores
        finite_indices = [index for index, value in enumerate(replicate_scores) if np.isfinite(value)]
        if not finite_indices:
            continue
        performance_values = np.asarray(
            [_performance_key(replicate_scores[index], is_higher) for index in finite_indices]
        )
        tied_ranks = rankdata(performance_values, method="average")
        valid_replicates += 1
        best_positions = np.flatnonzero(
            np.isclose(
                performance_values,
                np.min(performance_values),
                rtol=1e-12,
                atol=1e-12,
            )
        )
        top_credit = 1.0 / len(best_positions)
        for position in best_positions:
            top_counts[finite_indices[int(position)]] += top_credit
        for position, index in enumerate(finite_indices):
            rank_matrix[replicate, index] = tied_ranks[position]
    if valid_replicates == 0:
        raise ValueError("All bootstrap metric values are non-finite")

    best_index = subsets.index(nominal_best)
    top_tier_set: set[Subset] = {nominal_best}
    alpha = (1.0 - bootstrap.ci_level) / 2.0
    for index, subset in enumerate(subsets):
        if subset == nominal_best:
            continue
        if is_higher:
            differences = score_matrix[:, best_index] - score_matrix[:, index]
        else:
            differences = score_matrix[:, index] - score_matrix[:, best_index]
        finite = differences[np.isfinite(differences)]
        if len(finite) == 0:
            continue
        ci_low, ci_high = np.quantile(finite, [alpha, 1.0 - alpha])
        if ci_low <= 0.0 <= ci_high:
            top_tier_set.add(subset)

    stats = {
        subset: _summarize_bootstrap(
            score_matrix[:, index],
            rank_matrix[:, index],
            top_counts[index] / valid_replicates,
            alpha,
        )
        for index, subset in enumerate(subsets)
    }
    quality_scores = _normalize_invariant_quality_scores(
        pred_arrays, invariant_quality_scores
    )
    performance_components = _normalize_values(
        nominal_scores, higher_is_better=is_higher
    )
    median_rank_components = _normalize_values(
        {subset: stats[subset][6] for subset in subsets},
        higher_is_better=False,
    )
    bootstrap_std_components = _normalize_values(
        {subset: stats[subset][2] for subset in subsets},
        higher_is_better=False,
    )
    stability_components = {
        subset: float(
            (
                median_rank_components[subset]
                + float(stats[subset][5])
                + bootstrap_std_components[subset]
            )
            / 3.0
        )
        for subset in subsets
    }
    subset_quality_components = {
        subset: float(np.mean([quality_scores[name] for name in subset]))
        for subset in subsets
    }
    efficiency_components = _normalize_values(
        subset_costs, higher_is_better=False
    )
    llm_components = _llm_prior_components(subsets, llm_prior_order)
    final_scores = (
        {
            subset: float(
                ranking_weights.probe_performance
                * performance_components[subset]
                + ranking_weights.ranking_stability
                * stability_components[subset]
                + ranking_weights.feature_quality
                * subset_quality_components[subset]
                + ranking_weights.computational_efficiency
                * efficiency_components[subset]
                + ranking_weights.llm_prior * llm_components[subset]
            )
            for subset in subsets
        }
        if ranking_weights is not None
        else {}
    )
    # Method selection is deliberately hierarchical. The weighted composite is
    # retained only as a backward-compatible diagnostic and never determines
    # the frozen execution order.
    top_tier = tuple(
        sorted(
            top_tier_set,
            key=lambda subset: (
                stats[subset][6],
                -stats[subset][5],
                stats[subset][2],
                subset_costs[subset],
                len(subset),
                _performance_key(stats[subset][1], is_higher),
                subset,
            ),
        )
    )
    remaining = tuple(
        sorted(
            (subset for subset in subsets if subset not in top_tier_set),
            key=lambda subset: (
                _performance_key(nominal_scores[subset], is_higher),
                _performance_key(stats[subset][1], is_higher),
                stats[subset][6],
                -stats[subset][5],
                stats[subset][2],
                subset_costs[subset],
                len(subset),
                subset,
            ),
        )
    )
    priority_order = top_tier + remaining

    secondary = {
        subset: {
            name.upper(): float(get_metric(name)(y, consensus_predictions[subset]))
            for name in secondary_metrics
        }
        for subset in subsets
    }
    summaries_by_subset = {
        subset: SubsetSummary(
            subset_id="+".join(subset),
            invariants=subset,
            subset_size=len(subset),
            nominal_primary_score=nominal_scores[subset],
            secondary_metrics=secondary[subset],
            bootstrap_mean=stats[subset][0],
            bootstrap_median=stats[subset][1],
            bootstrap_std=stats[subset][2],
            bootstrap_ci_low=stats[subset][3],
            bootstrap_ci_high=stats[subset][4],
            top1_frequency=stats[subset][5],
            median_rank=stats[subset][6],
            top_tier=subset in top_tier_set,
            initial_full_acquisition_count=len(subset),
            estimated_full_acquisition_wall_seconds=subset_costs[subset],
            probe_performance_component=performance_components[subset],
            ranking_stability_component=stability_components[subset],
            feature_quality_component=subset_quality_components[subset],
            computational_efficiency_component=efficiency_components[subset],
            llm_prior_component=llm_components[subset],
            final_score=final_scores.get(subset),
        )
        for subset in subsets
    }
    score_map: dict[Subset, np.ndarray] = {}
    for index, subset in enumerate(subsets):
        values = score_matrix[:, index].copy()
        values.setflags(write=False)
        score_map[subset] = values
    return ScoutRanking(
        primary_metric=primary_metric.upper(),
        nominal_best=nominal_best,
        summaries=tuple(summaries_by_subset[subset] for subset in priority_order),
        top_tier=top_tier,
        priority_order=priority_order,
        bootstrap_scores=MappingProxyType(score_map),
        bootstrap_index_hashes=tuple(sha256(indices.tobytes()).hexdigest()[:16] for indices in bootstrap_indices),
    )


def _normalize_invariant_quality_scores(
    predictions: Mapping[str, np.ndarray],
    values: Mapping[str, float] | None,
) -> dict[str, float]:
    if values is None:
        return {name: 1.0 for name in predictions}
    normalized = {str(name).upper(): float(value) for name, value in values.items()}
    if set(normalized) != set(predictions):
        raise ValueError("invariant_quality_scores must match the prediction universe")
    if any(
        not np.isfinite(value) or not 0.0 <= value <= 1.0
        for value in normalized.values()
    ):
        raise ValueError("invariant quality scores must be finite and in [0, 1]")
    return normalized


def _normalize_values(
    values: Mapping[Subset, float], *, higher_is_better: bool
) -> dict[Subset, float]:
    finite = np.asarray([float(value) for value in values.values()], dtype=float)
    if not np.all(np.isfinite(finite)):
        raise ValueError("ranking components must be finite")
    minimum = float(np.min(finite))
    maximum = float(np.max(finite))
    if np.isclose(minimum, maximum, rtol=0.0, atol=1.0e-15):
        return {subset: 1.0 for subset in values}
    return {
        subset: float(
            (float(value) - minimum) / (maximum - minimum)
            if higher_is_better
            else (maximum - float(value)) / (maximum - minimum)
        )
        for subset, value in values.items()
    }


def _llm_prior_components(
    subsets: Sequence[Subset], prior_order: Sequence[Sequence[str]] | None
) -> dict[Subset, float]:
    if not prior_order:
        return {subset: 0.5 for subset in subsets}
    normalized_order = [
        tuple(str(value).upper() for value in subset) for subset in prior_order
    ]
    if len(set(normalized_order)) != len(normalized_order):
        raise ValueError("llm_prior_order contains duplicate subsets")
    if not set(normalized_order).issubset(set(subsets)):
        raise ValueError("llm_prior_order contains an unknown candidate subset")
    denominator = max(len(normalized_order), 1)
    scores = {subset: 0.5 for subset in subsets}
    for rank, subset in enumerate(normalized_order):
        scores[subset] = float(1.0 - 0.4 * (rank / denominator))
    return scores


def _normalize_invariant_costs(
    predictions: Mapping[str, np.ndarray],
    invariant_costs: Mapping[str, float] | None,
) -> dict[str, float]:
    if invariant_costs is None:
        return {name: 1.0 for name in predictions}
    normalized = {str(name).upper(): float(value) for name, value in invariant_costs.items()}
    if set(normalized) != set(predictions):
        raise ValueError("invariant_costs must contain exactly the prediction invariants")
    if any(not np.isfinite(value) or value < 0.0 for value in normalized.values()):
        raise ValueError("invariant_costs must be finite and non-negative")
    return normalized


def _validate_prediction_arrays(y: np.ndarray, predictions: Mapping[str, np.ndarray]) -> None:
    if len(y) < 2 or not np.all(np.isfinite(y)):
        raise ValueError("y_true must contain at least two finite values")
    for name, values in predictions.items():
        if values.shape != y.shape:
            raise ValueError(f"{name} predictions do not align with y_true")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} predictions must be finite")


def _performance_key(value: float, is_higher: bool) -> float:
    if not np.isfinite(value):
        return float("inf")
    return -float(value) if is_higher else float(value)


def _summarize_bootstrap(
    scores: np.ndarray,
    ranks: np.ndarray,
    top1_frequency: float,
    alpha: float,
) -> tuple[float, float, float, float, float, float, float]:
    finite_scores = scores[np.isfinite(scores)]
    finite_ranks = ranks[np.isfinite(ranks)]
    if len(finite_scores) == 0:
        return (float("nan"),) * 5 + (float(top1_frequency), float("nan"))
    ci_low, ci_high = np.quantile(finite_scores, [alpha, 1.0 - alpha])
    return (
        float(np.mean(finite_scores)),
        float(np.median(finite_scores)),
        float(np.std(finite_scores, ddof=1)) if len(finite_scores) > 1 else 0.0,
        float(ci_low),
        float(ci_high),
        float(top1_frequency),
        float(np.median(finite_ranks)) if len(finite_ranks) else float("nan"),
    )
