from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from mint_scout.evaluation.metrics import metric_direction
from mint_scout.scout.stability import ScoutRanking


@dataclass(frozen=True)
class TargetConfig:
    auto_lower_quantile: float = 0.10
    auto_upper_quantile: float = 0.90
    absolute_relaxation: Mapping[str, float] = field(
        default_factory=lambda: {"PCC": 0.02, "R2": 0.02}
    )
    relative_relaxation: Mapping[str, float] = field(
        default_factory=lambda: {"RMSE": 0.05, "MAE": 0.05}
    )
    sanity_floor: float | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.auto_lower_quantile <= 1.0:
            raise ValueError("auto_lower_quantile must be in [0, 1]")
        if not 0.0 <= self.auto_upper_quantile <= 1.0:
            raise ValueError("auto_upper_quantile must be in [0, 1]")
        if any(value < 0.0 for value in self.absolute_relaxation.values()):
            raise ValueError("absolute target relaxations must be non-negative")
        if any(value < 0.0 for value in self.relative_relaxation.values()):
            raise ValueError("relative target relaxations must be non-negative")


@dataclass(frozen=True)
class TargetSpec:
    metric: str
    direction: str
    value: float
    source: str
    derivation: Mapping[str, Any] | None
    warnings: tuple[str, ...] = ()


def resolve_target(
    *,
    metric: str,
    ranking: ScoutRanking,
    user_target: float | None,
    config: TargetConfig = TargetConfig(),
) -> TargetSpec:
    metric_name = metric.upper()
    direction = metric_direction(metric_name)
    if user_target is not None:
        if not math.isfinite(user_target):
            raise ValueError("user target must be finite")
        return TargetSpec(
            metric=metric_name,
            direction=direction,
            value=float(user_target),
            source="user",
            derivation=None,
        )

    lead_subset = ranking.priority_order[0]
    values = np.asarray(ranking.bootstrap_scores[lead_subset], dtype=float)
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        raise ValueError("Cannot derive target from non-finite bootstrap scores")
    lead_summary = ranking.summary_for(lead_subset)
    if direction == "higher":
        quantile = config.auto_lower_quantile
        quantile_value = float(np.quantile(finite, quantile))
        relaxation = float(config.absolute_relaxation.get(metric_name, 0.0))
        target_value = quantile_value - relaxation
        relaxation_kind = "absolute"
    else:
        quantile = config.auto_upper_quantile
        quantile_value = float(np.quantile(finite, quantile))
        relaxation = float(config.relative_relaxation.get(metric_name, 0.0))
        target_value = quantile_value * (1.0 + relaxation)
        relaxation_kind = "relative"

    warnings: list[str] = []
    if direction == "higher" and config.sanity_floor is not None and target_value < config.sanity_floor:
        warnings.append(
            f"LOW_PREDICTIVE_SIGNAL: derived {metric_name} target {target_value:.6g} "
            f"is below configured sanity floor {config.sanity_floor:.6g}."
        )
    derivation = {
        "lead_subset": list(lead_subset),
        "lead_nominal_probe_score": lead_summary.nominal_primary_score,
        "bootstrap_quantile": quantile,
        "bootstrap_quantile_value": quantile_value,
        "relaxation_kind": relaxation_kind,
        "relaxation": relaxation,
        "sanity_floor": config.sanity_floor,
    }
    return TargetSpec(
        metric=metric_name,
        direction=direction,
        value=float(target_value),
        source="probe_derived",
        derivation=derivation,
        warnings=tuple(warnings),
    )
