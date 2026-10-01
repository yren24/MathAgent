from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from mint_scout.search.state import PromotionDecision, SearchState


class PromotionPolicy(Protocol):
    def choose_next_active_set(self, state: SearchState) -> PromotionDecision:
        ...


@dataclass(frozen=True)
class FixedStandaloneRankingPolicy:
    retain_schedule: dict[float, int]

    def choose_next_active_set(self, state: SearchState) -> PromotionDecision:
        retain_count = self.retain_schedule.get(state.current_fidelity, len(state.active_invariants))
        ranked = sorted(
            state.active_invariants,
            key=lambda name: state.singleton_scores.get(name, float("-inf")),
            reverse=True,
        )
        promoted = tuple(ranked[:retain_count])
        eliminated = tuple(name for name in state.active_invariants if name not in promoted)
        return PromotionDecision(
            next_active_invariants=promoted,
            promoted_invariants=promoted,
            eliminated_invariants=eliminated,
            reason=f"Retained top {retain_count} invariants by singleton score at fidelity {state.current_fidelity}.",
        )


@dataclass(frozen=True)
class CheapestFirstPolicy:
    retain_count: int

    def choose_next_active_set(self, state: SearchState) -> PromotionDecision:
        ranked = sorted(
            state.active_invariants,
            key=lambda name: (state.estimated_next_costs.get(name, float("inf")), name),
        )
        promoted = tuple(ranked[: self.retain_count])
        eliminated = tuple(name for name in state.active_invariants if name not in promoted)
        return PromotionDecision(
            next_active_invariants=promoted,
            promoted_invariants=promoted,
            eliminated_invariants=eliminated,
            reason=f"Retained {self.retain_count} invariants with the lowest estimated next-fidelity cost.",
        )


@dataclass(frozen=True)
class ConservativeComplementarityPolicy:
    min_retain: int = 2
    necessity_floor: float = 0.0
    pair_gain_floor: float = 0.0

    def choose_next_active_set(self, state: SearchState) -> PromotionDecision:
        protected = {
            name
            for name, value in state.necessity_scores.items()
            if value is not None and value > self.necessity_floor
        }
        for pair, value in state.pair_gains.items():
            if value > self.pair_gain_floor:
                protected.update(pair)

        ranked = sorted(
            state.active_invariants,
            key=lambda name: (
                name not in protected,
                -state.singleton_scores.get(name, float("-inf")),
                state.estimated_next_costs.get(name, float("inf")),
                name,
            ),
        )
        retain_count = max(self.min_retain, len(protected))
        promoted = tuple(ranked[: min(retain_count, len(ranked))])
        eliminated = tuple(name for name in state.active_invariants if name not in promoted)
        return PromotionDecision(
            next_active_invariants=promoted,
            promoted_invariants=promoted,
            eliminated_invariants=eliminated,
            reason="Protected invariants with positive necessity or pair-complementarity evidence, then ranked by singleton score and cost.",
        )


@dataclass(frozen=True)
class StoppingDecision:
    stop: bool
    reason: str


@dataclass(frozen=True)
class PccStoppingRule:
    target_pcc: float
    patience: int
    min_delta: float
    max_subsets: int

    def should_stop(self, scores_by_round: tuple[float, ...], evaluated_subset_count: int) -> StoppingDecision:
        if not scores_by_round:
            return StoppingDecision(stop=False, reason="No validation PCC has been observed yet.")

        best_score = max(scores_by_round)
        if best_score >= self.target_pcc:
            return StoppingDecision(
                stop=True,
                reason=f"Best validation PCC {best_score:.4f} reached target {self.target_pcc:.4f}.",
            )

        if evaluated_subset_count >= self.max_subsets:
            return StoppingDecision(
                stop=True,
                reason=f"Evaluated subset count {evaluated_subset_count} reached max_subsets {self.max_subsets}.",
            )

        if len(scores_by_round) > self.patience:
            recent = scores_by_round[-self.patience :]
            prior_best = max(scores_by_round[: -self.patience])
            recent_best = max(recent)
            if recent_best - prior_best < self.min_delta:
                return StoppingDecision(
                    stop=True,
                    reason=(
                        f"Recent best validation PCC improved by {recent_best - prior_best:.4f}, "
                        f"below min_delta {self.min_delta:.4f} for patience {self.patience}."
                    ),
                )

        return StoppingDecision(stop=False, reason="Stopping criteria not met.")
