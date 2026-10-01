import pytest

from mint_scout.data.splits import (
    assert_no_test_leakage,
    make_group_kfold_assignments,
    make_holdout_split,
    make_kfold_assignments,
)


def test_no_test_leakage_for_generated_split():
    sample_ids = [f"s{i}" for i in range(30)]
    split = make_holdout_split(sample_ids, test_fraction=0.2, seed=1)
    folds = make_kfold_assignments(split.dev_sample_ids, n_splits=5, seed=2)
    fidelity = {0.5: split.dev_sample_ids[:10], 1.0: split.dev_sample_ids}
    assert_no_test_leakage(
        test_sample_ids=split.test_sample_ids,
        fold_ids=folds,
        fidelity_sample_ids=fidelity,
    )


def test_group_kfold_keeps_groups_together_and_balances_sample_counts():
    sample_ids = tuple(f"s{index}" for index in range(9))
    groups = {
        "s0": "A",
        "s1": "a",
        "s2": "a",
        "s3": "b",
        "s4": "b",
        "s5": "c",
        "s6": "d",
        "s7": "e",
        "s8": "f",
    }

    first = make_group_kfold_assignments(
        sample_ids, group_by_sample=groups, n_splits=3, seed=7
    )
    second = make_group_kfold_assignments(
        sample_ids, group_by_sample=groups, n_splits=3, seed=7
    )

    assert first == second
    assert first["s0"] == first["s1"] == first["s2"]
    assert first["s3"] == first["s4"]
    fold_sizes = [sum(fold == fold_id for fold in first.values()) for fold_id in range(3)]
    assert max(fold_sizes) - min(fold_sizes) <= 1


def test_group_kfold_requires_enough_distinct_groups():
    with pytest.raises(ValueError, match="distinct CV groups"):
        make_group_kfold_assignments(
            ("a", "b", "c"),
            group_by_sample={"a": "x", "b": "x", "c": "y"},
            n_splits=3,
            seed=1,
        )
