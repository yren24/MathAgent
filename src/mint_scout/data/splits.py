from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True)
class SplitDefinition:
    dev_sample_ids: tuple[str, ...]
    test_sample_ids: tuple[str, ...]

    def assert_no_overlap(self) -> None:
        overlap = set(self.dev_sample_ids) & set(self.test_sample_ids)
        if overlap:
            raise AssertionError(f"Dev/test split overlap: {sorted(overlap)[:10]}")


def make_holdout_split(
    sample_ids: Sequence[str],
    *,
    test_fraction: float,
    seed: int,
) -> SplitDefinition:
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be in (0, 1)")
    ids = list(sample_ids)
    if len(set(ids)) != len(ids):
        raise ValueError("sample_ids contains duplicates")
    rng = random.Random(seed)
    rng.shuffle(ids)
    n_test = max(1, round(len(ids) * test_fraction))
    split = SplitDefinition(
        dev_sample_ids=tuple(sorted(ids[n_test:])),
        test_sample_ids=tuple(sorted(ids[:n_test])),
    )
    split.assert_no_overlap()
    return split


def make_kfold_assignments(
    dev_sample_ids: Sequence[str],
    *,
    n_splits: int,
    seed: int,
) -> dict[str, int]:
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    ids = list(dev_sample_ids)
    if n_splits > len(ids):
        raise ValueError("n_splits cannot exceed number of dev samples")
    if len(set(ids)) != len(ids):
        raise ValueError("dev_sample_ids contains duplicates")
    rng = random.Random(seed)
    rng.shuffle(ids)
    return {sample_id: idx % n_splits for idx, sample_id in enumerate(ids)}


def make_group_kfold_assignments(
    dev_sample_ids: Sequence[str],
    *,
    group_by_sample: Mapping[str, str],
    n_splits: int,
    seed: int,
) -> dict[str, int]:
    ids = list(dev_sample_ids)
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    if len(set(ids)) != len(ids):
        raise ValueError("dev_sample_ids contains duplicates")
    if set(group_by_sample) != set(ids):
        raise ValueError("group_by_sample must contain exactly the dev sample ids")

    samples_by_group: dict[str, list[str]] = defaultdict(list)
    for sample_id in ids:
        group = str(group_by_sample[sample_id]).strip().casefold()
        if not group:
            raise ValueError(f"Sample {sample_id!r} has an empty CV group")
        samples_by_group[group].append(sample_id)
    if n_splits > len(samples_by_group):
        raise ValueError("n_splits cannot exceed number of distinct CV groups")

    groups = list(samples_by_group)
    rng = random.Random(seed)
    rng.shuffle(groups)
    groups.sort(key=lambda group: -len(samples_by_group[group]))
    fold_sizes = [0] * n_splits
    assignment: dict[str, int] = {}
    for group in groups:
        fold_id = min(range(n_splits), key=lambda fold: (fold_sizes[fold], fold))
        for sample_id in samples_by_group[group]:
            assignment[sample_id] = fold_id
        fold_sizes[fold_id] += len(samples_by_group[group])
    return assignment


def assert_no_test_leakage(
    *,
    test_sample_ids: Sequence[str],
    fold_ids: dict[str, int],
    fidelity_sample_ids: dict[float, Sequence[str]],
    evidence_sample_ids: Sequence[str] = (),
) -> None:
    test = set(test_sample_ids)
    fold_overlap = test & set(fold_ids)
    if fold_overlap:
        raise AssertionError(f"Test samples appear in CV folds: {sorted(fold_overlap)[:10]}")
    for level, sample_ids in fidelity_sample_ids.items():
        overlap = test & set(sample_ids)
        if overlap:
            raise AssertionError(f"Test samples appear in fidelity {level}: {sorted(overlap)[:10]}")
    evidence_overlap = test & set(evidence_sample_ids)
    if evidence_overlap:
        raise AssertionError(f"Test samples appear in evidence: {sorted(evidence_overlap)[:10]}")
