from mint_scout.data.fidelity_sampler import assert_nested, make_nested_fidelity_subsets


def test_nested_fidelity_subsets_are_monotone():
    sample_ids = [f"s{i}" for i in range(20)]
    subsets = make_nested_fidelity_subsets(sample_ids, [0.1, 0.25, 0.5, 1.0], seed=7)
    assert_nested(subsets)
    assert set(subsets[1.0]) == set(sample_ids)
    assert len(subsets[0.1]) == 2
