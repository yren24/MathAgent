from __future__ import annotations

import numpy as np

from mint_scout.data.geometry import AtomCloud
from mint_scout.representation import make_legacy_toxicity_representation_spec
from mint_scout.toxicity.probe import ToxicityProbePolicy, select_toxicity_probe


def _cloud(size: int) -> AtomCloud:
    elements = ["H", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I"]
    elements.extend("C" for _ in range(size - len(elements)))
    return AtomCloud(
        elements=tuple(elements),
        coordinates=np.arange(len(elements) * 3, dtype=float).reshape(-1, 3),
    )


def _carbon_oxygen_cloud(size: int) -> AtomCloud:
    elements = ["C", "O"]
    elements.extend("C" for _ in range(size - len(elements)))
    return AtomCloud(
        elements=tuple(elements),
        coordinates=np.arange(len(elements) * 3, dtype=float).reshape(-1, 3),
    )


def test_toxicity_probe_uses_hard_gates_and_selects_smallest_eligible_candidate():
    targets = {f"train-{index:03d}": float(index) for index in range(100)}
    clouds = {
        sample_id: _cloud(12 + index % 10)
        for index, sample_id in enumerate(targets)
    }
    policy = ToxicityProbePolicy(
        candidate_sizes=(40, 60, 80),
        target_quantile_bins=5,
        size_quantile_bins=3,
        min_pair_support=1,
        augmentation_fraction_limit=0.10,
        max_ks_distance=1.0,
        max_abs_standardized_mean_difference=1.0,
        max_joint_cell_share_error=1.0,
    )

    report = select_toxicity_probe(
        training_targets=targets,
        training_clouds=clouds,
        representation_specs=(make_legacy_toxicity_representation_spec(),),
        policy=policy,
    )

    eligible = [candidate for candidate in report["candidates"] if candidate["eligible"]]
    assert report["evidence_scope"] == "train_only"
    assert report["test_labels_used"] is False
    assert report["selection_basis"] == (
        "representativeness_hard_gates_then_minimum_sample_cost"
    )
    assert report["probe_sample_count"] == min(
        candidate["final_size"] for candidate in eligible
    )
    assert set(report["probe_fold_assignment"]) == set(report["probe_sample_ids"])
    assert report["retained_pair_union_count"] == 30


def test_toxicity_probe_relaxes_pair_support_when_it_is_the_only_failed_gate():
    targets = {f"train-{index:03d}": float(index) for index in range(100)}
    clouds = {
        sample_id: _carbon_oxygen_cloud(12 + index % 6)
        for index, sample_id in enumerate(targets)
    }
    policy = ToxicityProbePolicy(
        candidate_sizes=(40, 60),
        target_quantile_bins=5,
        size_quantile_bins=3,
        min_pair_support=3,
        augmentation_fraction_limit=0.10,
        max_ks_distance=1.0,
        max_abs_standardized_mean_difference=1.0,
        max_joint_cell_share_error=1.0,
    )

    report = select_toxicity_probe(
        training_targets=targets,
        training_clouds=clouds,
        representation_specs=(make_legacy_toxicity_representation_spec(),),
        policy=policy,
    )

    assert report["selection_basis"] == (
        "representativeness_hard_gates_with_pair_support_fallback"
    )
    assert report["selected_under_pair_support_fallback"] is True
    assert report["selected_candidate_all_gates_passed"] is False
    assert report["selected_candidate_id"] == "hierarchical-probe-40"
    assert report["decision_trace"][1]["stage"] == "pair_support_fallback"


def test_toxicity_probe_rejects_misaligned_train_inputs():
    try:
        select_toxicity_probe(
            training_targets={"a": 1.0},
            training_clouds={"b": _cloud(8)},
            representation_specs=(make_legacy_toxicity_representation_spec(),),
            policy=ToxicityProbePolicy(candidate_sizes=(2,)),
        )
    except ValueError as exc:
        assert "aligned" in str(exc)
    else:
        raise AssertionError("misaligned toxicity Probe inputs must fail")


def test_toxicity_probe_falls_back_to_full_train_for_small_data():
    targets = {f"train-{index:03d}": float(index) for index in range(12)}
    clouds = {
        sample_id: _cloud(14 + index % 4)
        for index, sample_id in enumerate(targets)
    }
    policy = ToxicityProbePolicy(
        candidate_sizes=(300, 500),
        target_quantile_bins=4,
        size_quantile_bins=2,
        min_pair_support=1,
        max_ks_distance=0.0,
        max_abs_standardized_mean_difference=0.0,
        max_joint_cell_share_error=0.0,
    )

    report = select_toxicity_probe(
        training_targets=targets,
        training_clouds=clouds,
        representation_specs=(make_legacy_toxicity_representation_spec(),),
        policy=policy,
    )

    assert report["selected_candidate_id"] == "hierarchical-probe-12"
    assert report["probe_sample_count"] == 12
    assert report["cv_folds"] == 5
    assert report["candidates"][0]["size_source"] == "small_data_full_train_fallback"
    assert set(report["probe_sample_ids"]) == set(targets)
