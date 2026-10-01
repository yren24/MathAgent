from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class PromotionDecision:
    next_active_invariants: tuple[str, ...]
    promoted_invariants: tuple[str, ...]
    eliminated_invariants: tuple[str, ...] = ()
    deferred_invariants: tuple[str, ...] = ()
    reason: str = ""


@dataclass(frozen=True)
class SearchRoundRecord:
    round_id: int
    fidelity_level: float
    active_invariants: tuple[str, ...]
    promoted_invariants: tuple[str, ...]
    singleton_scores: dict[str, float]
    subset_scores: dict[tuple[str, ...], float]
    camc: dict[str, float | None]
    pair_gains: dict[tuple[str, str], float]
    necessity_scores: dict[str, float | None]
    uncertainty_intervals: dict[str, Any]
    measured_cost_this_round: float
    cumulative_cost: float
    remaining_budget: float | None
    current_best_subset: tuple[str, ...]
    current_best_score: float
    decision_reason: str


@dataclass
class SearchState:
    task_id: str
    system_type: str
    pair_schema_id: str
    primary_metric: str
    dev_sample_ids: tuple[str, ...]
    test_sample_ids: tuple[str, ...]
    fold_ids: dict[str, int]
    fidelity_sample_ids: dict[float, tuple[str, ...]]
    current_fidelity: float
    active_invariants: tuple[str, ...]
    target_score: float | None = None
    budget_cpu_hours: float | None = None
    deferred_invariants: tuple[str, ...] = ()
    eliminated_invariants: tuple[str, ...] = ()
    singleton_scores: dict[str, float] = field(default_factory=dict)
    subset_scores: dict[tuple[str, ...], float] = field(default_factory=dict)
    camc: dict[str, float | None] = field(default_factory=dict)
    pair_gains: dict[tuple[str, str], float] = field(default_factory=dict)
    necessity_scores: dict[str, float | None] = field(default_factory=dict)
    uncertainty: dict[str, Any] = field(default_factory=dict)
    measured_costs: dict[str, float] = field(default_factory=dict)
    estimated_next_costs: dict[str, float] = field(default_factory=dict)
    cumulative_cost: float = 0.0
    remaining_budget: float | None = None
    best_subset: tuple[str, ...] = ()
    best_dev_score: float | None = None
    decision_history: list[PromotionDecision] = field(default_factory=list)
    stop: bool = False
    stop_reason: str | None = None

    def visible_evidence(self) -> dict[str, Any]:
        return {
            "current_fidelity": self.current_fidelity,
            "active_invariants": self.active_invariants,
            "singleton_scores": self.singleton_scores,
            "subset_scores": self.subset_scores,
            "camc": self.camc,
            "pair_gains": self.pair_gains,
            "necessity_scores": self.necessity_scores,
            "uncertainty": self.uncertainty,
            "cumulative_cost": self.cumulative_cost,
            "remaining_budget": self.remaining_budget,
            "best_subset": self.best_subset,
            "best_dev_score": self.best_dev_score,
        }
