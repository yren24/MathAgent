from collections import Counter

import numpy as np

from mint_scout.scout.sampling import (
    ProbeSamplingConfig,
    probe_size,
    quantile_bin_assignments,
    select_probe_samples,
    select_uniform_random_probe_samples,
    stratified_bootstrap_indices,
)


def test_probe_size_uses_configured_fraction_floor_and_cap():
    config = ProbeSamplingConfig()

    assert probe_size(3772, config) == 754
    assert probe_size(5931, config) == 1000
    assert probe_size(1000, config) == 300
    assert probe_size(300, config) == 300
    assert probe_size(100, config) == 100


def test_probe_sampling_uses_target_and_structure_size_strata():
    sample_ids = [f"s{index:03d}" for index in range(200)]
    targets = np.linspace(-3.0, 3.0, len(sample_ids))
    sizes = np.tile(np.arange(20), 10)
    config = ProbeSamplingConfig(
        fraction=0.25,
        min_samples=50,
        max_samples=50,
        target_quantile_bins=5,
        size_quantile_bins=4,
        random_seed=17,
    )

    selection = select_probe_samples(
        sample_ids=sample_ids,
        targets=targets,
        structure_sizes=sizes,
        config=config,
    )
    all_target_bins = set(quantile_bin_assignments(targets, sample_ids, max_bins=5).values())
    all_size_bins = set(quantile_bin_assignments(sizes, sample_ids, max_bins=4).values())

    assert selection.final_size == 50
    assert set(selection.target_bins.values()) == all_target_bins
    assert set(selection.size_bins.values()) == all_size_bins
    assert selection.sample_ids == tuple(sample_id for sample_id in sample_ids if sample_id in selection.sample_ids)


def test_greedy_pair_coverage_augmentation_adds_only_up_to_limit():
    sample_ids = [f"s{index}" for index in range(6)]
    pair = ("C", "N")
    config = ProbeSamplingConfig(
        fraction=0.2,
        min_samples=2,
        max_samples=2,
        target_quantile_bins=1,
        size_quantile_bins=1,
        min_pair_support=3,
        augmentation_fraction_limit=0.5,
        random_seed=3,
    )

    selection = select_probe_samples(
        sample_ids=sample_ids,
        targets=np.arange(6),
        structure_sizes=np.arange(6),
        retained_pairs=[pair],
        pair_presence_by_sample={sample_id: [pair] for sample_id in sample_ids},
        config=config,
    )

    assert selection.base_size == 2
    assert selection.final_size == 3
    assert selection.pair_support[pair] == 3
    assert selection.undercovered_pairs == ()


def test_uniform_random_probe_has_equal_budget_without_pair_repair():
    sample_ids = [f"s{index}" for index in range(10)]
    pair = ("C", "N")
    config = ProbeSamplingConfig(
        sampling_strategy="uniform_random",
        fraction=0.4,
        min_samples=4,
        max_samples=4,
        target_quantile_bins=2,
        size_quantile_bins=2,
        min_pair_support=3,
        random_seed=9,
    )

    selection = select_uniform_random_probe_samples(
        sample_ids=sample_ids,
        targets=np.arange(10),
        structure_sizes=np.arange(10),
        retained_pairs=[pair],
        pair_presence_by_sample={sample_id: [pair] for sample_id in sample_ids[:2]},
        config=config,
    )

    assert selection.base_size == 4
    assert selection.final_size == 4
    assert selection.undercovered_pairs == (pair,)


def test_unreachable_pair_support_is_logged_without_deleting_pair():
    sample_ids = [f"s{index}" for index in range(6)]
    pair = ("C", "N")
    config = ProbeSamplingConfig(
        fraction=0.2,
        min_samples=2,
        max_samples=2,
        target_quantile_bins=1,
        size_quantile_bins=1,
        min_pair_support=4,
        augmentation_fraction_limit=0.5,
    )

    selection = select_probe_samples(
        sample_ids=sample_ids,
        targets=np.arange(6),
        structure_sizes=np.arange(6),
        retained_pairs=[pair],
        pair_presence_by_sample={sample_id: [pair] for sample_id in sample_ids},
        config=config,
    )

    assert selection.undercovered_pairs == (pair,)
    assert "PROBE_PAIR_UNDERCOVERED" in selection.warnings[0]


def test_stratified_bootstrap_preserves_each_stratum_count():
    strata = np.asarray([0, 0, 0, 1, 1, 2])
    expected = Counter(strata)

    replicates = stratified_bootstrap_indices(strata, n_replicates=12, seed=11)

    for indices in replicates:
        assert Counter(strata[indices]) == expected
