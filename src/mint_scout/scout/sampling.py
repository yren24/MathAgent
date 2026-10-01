from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence, Tuple

import numpy as np


ElementPair = Tuple[str, str]


@dataclass(frozen=True)
class ProbeSamplingConfig:
    sampling_strategy: str = "stratified_pair_coverage"
    fraction: float = 0.20
    min_samples: int = 300
    max_samples: int = 1000
    target_quantile_bins: int = 10
    size_quantile_bins: int = 5
    min_pair_support: int = 3
    augmentation_fraction_limit: float = 0.10
    random_seed: int = 2026

    def __post_init__(self) -> None:
        if self.sampling_strategy not in {"stratified_pair_coverage", "uniform_random"}:
            raise ValueError(
                "sampling_strategy must be stratified_pair_coverage or uniform_random"
            )
        if not 0.0 < self.fraction <= 1.0:
            raise ValueError("probe fraction must be in (0, 1]")
        if self.min_samples < 1 or self.max_samples < self.min_samples:
            raise ValueError("probe min/max samples are invalid")
        if self.target_quantile_bins < 1 or self.size_quantile_bins < 1:
            raise ValueError("probe quantile-bin counts must be positive")
        if self.min_pair_support < 0:
            raise ValueError("min_pair_support must be non-negative")
        if not 0.0 <= self.augmentation_fraction_limit <= 1.0:
            raise ValueError("augmentation_fraction_limit must be in [0, 1]")


@dataclass(frozen=True)
class ProbeSelection:
    sample_ids: tuple[str, ...]
    base_size: int
    final_size: int
    target_bins: Mapping[str, int]
    size_bins: Mapping[str, int]
    pair_support: Mapping[ElementPair, int]
    undercovered_pairs: tuple[ElementPair, ...]
    warnings: tuple[str, ...]


def probe_size(n_modeling: int, config: ProbeSamplingConfig = ProbeSamplingConfig()) -> int:
    if n_modeling < 1:
        raise ValueError("n_modeling must be positive")
    requested = max(config.min_samples, round(config.fraction * n_modeling))
    return min(n_modeling, config.max_samples, requested)


def quantile_bin_assignments(
    values: Sequence[float],
    sample_ids: Sequence[str],
    *,
    max_bins: int,
) -> dict[str, int]:
    ids = tuple(sample_ids)
    array = np.asarray(values, dtype=float)
    if len(ids) != len(array):
        raise ValueError("values and sample_ids must be aligned")
    if len(set(ids)) != len(ids):
        raise ValueError("sample_ids contains duplicates")
    if max_bins < 1:
        raise ValueError("max_bins must be positive")
    if not np.all(np.isfinite(array)):
        raise ValueError("quantile-binning values must be finite")
    if len(array) == 0:
        return {}

    n_bins = min(max_bins, len(array))
    edges = np.quantile(array, np.linspace(0.0, 1.0, n_bins + 1)[1:-1])
    unique_edges = np.unique(edges)
    bins = np.searchsorted(unique_edges, array, side="right")
    return {sample_id: int(bin_id) for sample_id, bin_id in zip(ids, bins)}


def select_probe_samples(
    *,
    sample_ids: Sequence[str],
    targets: Sequence[float],
    structure_sizes: Sequence[float],
    retained_pairs: Sequence[ElementPair] = (),
    pair_presence_by_sample: Mapping[str, Iterable[ElementPair]] | None = None,
    config: ProbeSamplingConfig = ProbeSamplingConfig(),
) -> ProbeSelection:
    ids = tuple(sample_ids)
    target_array = np.asarray(targets, dtype=float)
    size_array = np.asarray(structure_sizes, dtype=float)
    _validate_sampling_inputs(ids, target_array, size_array)

    base_size = probe_size(len(ids), config)
    target_bins = quantile_bin_assignments(
        target_array,
        ids,
        max_bins=config.target_quantile_bins,
    )
    size_bins = quantile_bin_assignments(
        size_array,
        ids,
        max_bins=config.size_quantile_bins,
    )
    groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, sample_id in enumerate(ids):
        groups[(target_bins[sample_id], size_bins[sample_id])].append(index)

    quotas = _allocate_group_quotas({group: len(indices) for group, indices in groups.items()}, base_size)
    rng = np.random.default_rng(config.random_seed)
    selected: set[int] = set()
    for group in sorted(groups):
        candidates = np.asarray(sorted(groups[group]), dtype=int)
        shuffled = rng.permutation(candidates)
        selected.update(int(index) for index in shuffled[: quotas[group]])

    retained = tuple(dict.fromkeys(tuple(pair) for pair in retained_pairs))
    presence = _normalize_pair_presence(ids, pair_presence_by_sample or {}, set(retained))
    augmentation_limit = min(
        len(ids) - len(selected),
        int(math.ceil(base_size * config.augmentation_fraction_limit)),
    )
    _augment_pair_coverage(
        ids=ids,
        selected=selected,
        retained_pairs=retained,
        presence=presence,
        min_support=config.min_pair_support,
        augmentation_limit=augmentation_limit,
    )

    ordered_selected = tuple(sample_id for index, sample_id in enumerate(ids) if index in selected)
    support = {
        pair: sum(pair in presence[sample_id] for sample_id in ordered_selected)
        for pair in retained
    }
    undercovered = tuple(
        pair for pair in retained if support[pair] < config.min_pair_support
    )
    warnings: tuple[str, ...] = ()
    if undercovered:
        encoded = ", ".join(f"{left}|{right}" for left, right in undercovered[:10])
        warnings = (f"PROBE_PAIR_UNDERCOVERED: {encoded}",)
    return ProbeSelection(
        sample_ids=ordered_selected,
        base_size=base_size,
        final_size=len(ordered_selected),
        target_bins={sample_id: target_bins[sample_id] for sample_id in ordered_selected},
        size_bins={sample_id: size_bins[sample_id] for sample_id in ordered_selected},
        pair_support=support,
        undercovered_pairs=undercovered,
        warnings=warnings,
    )


def select_uniform_random_probe_samples(
    *,
    sample_ids: Sequence[str],
    targets: Sequence[float],
    structure_sizes: Sequence[float],
    retained_pairs: Sequence[ElementPair] = (),
    pair_presence_by_sample: Mapping[str, Iterable[ElementPair]] | None = None,
    config: ProbeSamplingConfig = ProbeSamplingConfig(
        sampling_strategy="uniform_random"
    ),
) -> ProbeSelection:
    """Select an equal-budget uniform-random Probe for controlled comparisons.

    This baseline deliberately does not stratify or repair element-pair coverage.
    It still records bins and pair support so the same audit can quantify what was
    lost relative to the production selector.
    """
    ids = tuple(sample_ids)
    target_array = np.asarray(targets, dtype=float)
    size_array = np.asarray(structure_sizes, dtype=float)
    _validate_sampling_inputs(ids, target_array, size_array)
    if config.sampling_strategy != "uniform_random":
        raise ValueError("uniform-random selection requires sampling_strategy=uniform_random")

    base_size = probe_size(len(ids), config)
    target_bins = quantile_bin_assignments(
        target_array, ids, max_bins=config.target_quantile_bins
    )
    size_bins = quantile_bin_assignments(
        size_array, ids, max_bins=config.size_quantile_bins
    )
    rng = np.random.default_rng(config.random_seed)
    selected = set(int(index) for index in rng.choice(len(ids), size=base_size, replace=False))
    retained = tuple(dict.fromkeys(tuple(pair) for pair in retained_pairs))
    presence = _normalize_pair_presence(ids, pair_presence_by_sample or {}, set(retained))
    ordered_selected = tuple(
        sample_id for index, sample_id in enumerate(ids) if index in selected
    )
    support = {
        pair: sum(pair in presence[sample_id] for sample_id in ordered_selected)
        for pair in retained
    }
    undercovered = tuple(
        pair for pair in retained if support[pair] < config.min_pair_support
    )
    warnings: tuple[str, ...] = ()
    if undercovered:
        encoded = ", ".join(f"{left}|{right}" for left, right in undercovered[:10])
        warnings = (f"PROBE_PAIR_UNDERCOVERED: {encoded}",)
    return ProbeSelection(
        sample_ids=ordered_selected,
        base_size=base_size,
        final_size=base_size,
        target_bins={sample_id: target_bins[sample_id] for sample_id in ordered_selected},
        size_bins={sample_id: size_bins[sample_id] for sample_id in ordered_selected},
        pair_support=support,
        undercovered_pairs=undercovered,
        warnings=warnings,
    )


def stratified_bootstrap_indices(
    strata: Sequence[int],
    *,
    n_replicates: int,
    seed: int,
) -> tuple[np.ndarray, ...]:
    stratum_array = np.asarray(strata, dtype=int)
    if len(stratum_array) == 0:
        raise ValueError("strata cannot be empty")
    if n_replicates < 1:
        raise ValueError("n_replicates must be positive")
    groups = {
        stratum: np.flatnonzero(stratum_array == stratum)
        for stratum in sorted(set(int(value) for value in stratum_array))
    }
    rng = np.random.default_rng(seed)
    replicates: list[np.ndarray] = []
    for _ in range(n_replicates):
        pieces = [rng.choice(indices, size=len(indices), replace=True) for indices in groups.values()]
        replicates.append(np.concatenate(pieces))
    return tuple(replicates)


def _validate_sampling_inputs(ids: tuple[str, ...], targets: np.ndarray, sizes: np.ndarray) -> None:
    if not ids:
        raise ValueError("sample_ids cannot be empty")
    if len(set(ids)) != len(ids):
        raise ValueError("sample_ids contains duplicates")
    if len(targets) != len(ids) or len(sizes) != len(ids):
        raise ValueError("sample_ids, targets, and structure_sizes must be aligned")
    if not np.all(np.isfinite(targets)) or not np.all(np.isfinite(sizes)):
        raise ValueError("targets and structure_sizes must be finite")


def _allocate_group_quotas(group_sizes: Mapping[tuple[int, int], int], total: int) -> dict[tuple[int, int], int]:
    groups = tuple(sorted(group_sizes))
    if total < 0 or total > sum(group_sizes.values()):
        raise ValueError("invalid stratified allocation total")
    quotas = {group: 0 for group in groups}
    if total == 0:
        return quotas

    if total >= len(groups):
        for group in groups:
            quotas[group] = 1
    remaining = total - sum(quotas.values())
    while remaining:
        capacities = {group: group_sizes[group] - quotas[group] for group in groups}
        available = tuple(group for group in groups if capacities[group] > 0)
        if not available:
            raise AssertionError("stratified allocation exhausted before reaching requested total")
        capacity_total = sum(capacities[group] for group in available)
        raw = {group: remaining * capacities[group] / capacity_total for group in available}
        floor_added = 0
        for group in available:
            addition = min(capacities[group], int(math.floor(raw[group])))
            quotas[group] += addition
            floor_added += addition
        remaining -= floor_added
        if remaining == 0:
            break
        ranked = sorted(
            available,
            key=lambda group: (-(raw[group] - math.floor(raw[group])), group),
        )
        for group in ranked:
            if remaining == 0:
                break
            if quotas[group] < group_sizes[group]:
                quotas[group] += 1
                remaining -= 1
    return quotas


def _normalize_pair_presence(
    ids: tuple[str, ...],
    pair_presence_by_sample: Mapping[str, Iterable[ElementPair]],
    retained: set[ElementPair],
) -> dict[str, set[ElementPair]]:
    unknown_ids = sorted(set(pair_presence_by_sample) - set(ids))
    if unknown_ids:
        raise ValueError(f"pair presence includes unknown sample ids: {unknown_ids[:10]}")
    return {
        sample_id: {tuple(pair) for pair in pair_presence_by_sample.get(sample_id, ()) if tuple(pair) in retained}
        for sample_id in ids
    }


def _augment_pair_coverage(
    *,
    ids: tuple[str, ...],
    selected: set[int],
    retained_pairs: tuple[ElementPair, ...],
    presence: Mapping[str, set[ElementPair]],
    min_support: int,
    augmentation_limit: int,
) -> None:
    for _ in range(augmentation_limit):
        selected_ids = tuple(ids[index] for index in selected)
        support = Counter(
            pair
            for sample_id in selected_ids
            for pair in presence[sample_id]
            if pair in retained_pairs
        )
        undercovered = {pair for pair in retained_pairs if support[pair] < min_support}
        if not undercovered:
            return
        candidates: list[tuple[int, str, int]] = []
        for index, sample_id in enumerate(ids):
            if index in selected:
                continue
            gain = len(presence[sample_id] & undercovered)
            if gain:
                candidates.append((-gain, sample_id, index))
        if not candidates:
            return
        selected.add(min(candidates)[2])
