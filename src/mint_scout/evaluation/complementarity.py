from __future__ import annotations

from collections.abc import Mapping

from mint_scout.evaluation.ensemble import Subset


def marginal_gain(subset_scores: Mapping[Subset, float], candidate: str, context: Subset) -> float:
    context_set = tuple(sorted(context))
    if not context_set:
        raise ValueError("context must be non-empty for marginal_gain")
    expanded = tuple(sorted(set(context_set) | {candidate}))
    return subset_scores[expanded] - subset_scores[context_set]


def camc(subset_scores: Mapping[Subset, float], active_invariants: tuple[str, ...]) -> dict[str, float | None]:
    active = tuple(active_invariants)
    values: dict[str, float | None] = {}
    for invariant in active:
        contexts = [
            subset
            for subset in subset_scores
            if invariant not in subset and len(subset) > 0 and set(subset).issubset(set(active) - {invariant})
        ]
        if not contexts:
            values[invariant] = None
            continue
        gains = []
        for context in contexts:
            expanded = tuple(sorted(set(context) | {invariant}))
            gains.append(subset_scores[expanded] - subset_scores[tuple(sorted(context))])
        values[invariant] = sum(gains) / len(gains)
    return values


def pair_gains(subset_scores: Mapping[Subset, float], active_invariants: tuple[str, ...]) -> dict[tuple[str, str], float]:
    gains: dict[tuple[str, str], float] = {}
    active = tuple(active_invariants)
    for i, left in enumerate(active):
        for right in active[i + 1 :]:
            pair = tuple(sorted((left, right)))
            singleton_best = max(subset_scores[(left,)], subset_scores[(right,)])
            gains[pair] = subset_scores[pair] - singleton_best
    return gains


def strongest_pair_gain(
    subset_scores: Mapping[Subset, float], active_invariants: tuple[str, ...]
) -> dict[str, float | None]:
    pair_gain_map = pair_gains(subset_scores, active_invariants)
    strongest: dict[str, float | None] = {}
    for invariant in active_invariants:
        vals = [gain for pair, gain in pair_gain_map.items() if invariant in pair]
        strongest[invariant] = max(vals) if vals else None
    return strongest


def exclusion_necessity(
    subset_scores: Mapping[Subset, float], active_invariants: tuple[str, ...]
) -> dict[str, float | None]:
    if not subset_scores:
        return {}
    best_all = max(subset_scores.values())
    necessity: dict[str, float | None] = {}
    for invariant in active_invariants:
        allowed_scores = [
            score
            for subset, score in subset_scores.items()
            if invariant not in subset and set(subset).issubset(set(active_invariants) - {invariant})
        ]
        necessity[invariant] = best_all - max(allowed_scores) if allowed_scores else None
    return necessity
