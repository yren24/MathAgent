import numpy as np

from mint_scout.scout.stability import (
    BootstrapConfig,
    RankingWeights,
    rank_consensus_subsets,
)
from mint_scout.scout.target import TargetConfig, resolve_target


def test_five_invariants_produce_exactly_31_consensus_subsets():
    y = np.linspace(-2.0, 2.0, 50)
    predictions = {
        name: y + offset * np.sin(np.arange(len(y)))
        for name, offset in zip(
            ("PH", "PL", "CA", "FPRC", "EIC"), (0.1, 0.2, 0.3, 0.4, 0.5)
        )
    }

    ranking = rank_consensus_subsets(
        y_true=y,
        predictions=predictions,
        strata=np.repeat(np.arange(5), 10),
        bootstrap=BootstrapConfig(replicates=20, random_seed=7),
    )

    assert len(ranking.summaries) == 31
    assert len(ranking.priority_order) == 31
    assert len(set(ranking.priority_order)) == 31


def test_top_tier_prefers_fewer_invariants_when_predictions_are_indistinguishable():
    y = np.linspace(-1.0, 1.0, 40)
    ranking = rank_consensus_subsets(
        y_true=y,
        predictions={"A": y, "B": y.copy()},
        strata=np.repeat(np.arange(4), 10),
        bootstrap=BootstrapConfig(replicates=30, random_seed=2),
    )

    assert set(ranking.top_tier) == {("A",), ("B",), ("A", "B")}
    assert ranking.priority_order[:2] == (("A",), ("B",))
    assert ranking.priority_order[2] == ("A", "B")


def test_metric_direction_changes_nominal_best_for_rmse():
    y = np.linspace(0.0, 1.0, 40)
    ranking = rank_consensus_subsets(
        y_true=y,
        predictions={"GOOD": y + 0.01, "BAD": y + 1.0},
        primary_metric="RMSE",
        secondary_metrics=("MAE",),
        bootstrap=BootstrapConfig(replicates=20, random_seed=4),
    )

    assert ranking.nominal_best == ("GOOD",)
    assert ranking.summaries[0].invariants == ("GOOD",)


def test_top_tier_uses_stability_then_measured_cost():
    y = np.linspace(-1.0, 1.0, 40)
    ranking = rank_consensus_subsets(
        y_true=y,
        predictions={"A": y, "B": y.copy()},
        invariant_costs={"A": 10.0, "B": 1.0},
        strata=np.repeat(np.arange(4), 10),
        bootstrap=BootstrapConfig(replicates=30, random_seed=2),
    )

    assert ranking.summary_for(("B",)).estimated_full_acquisition_wall_seconds == 1.0
    assert (
        ranking.summary_for(("A", "B")).estimated_full_acquisition_wall_seconds == 11.0
    )
    assert ranking.priority_order == (("B",), ("A",), ("A", "B"))


def test_composite_score_is_diagnostic_only_and_does_not_choose_method_order():
    y = np.linspace(-1.0, 1.0, 40)
    ranking = rank_consensus_subsets(
        y_true=y,
        predictions={"A": y, "B": y.copy()},
        invariant_costs={"A": 1.0, "B": 10.0},
        invariant_quality_scores={"A": 0.50, "B": 1.0},
        llm_prior_order=(("B",),),
        ranking_weights=RankingWeights(
            probe_performance=0.0,
            ranking_stability=0.0,
            feature_quality=1.0,
            computational_efficiency=0.0,
            llm_prior=0.0,
        ),
        strata=np.repeat(np.arange(4), 10),
        bootstrap=BootstrapConfig(replicates=30, random_seed=2),
    )

    assert len(ranking.priority_order) == 3
    assert ranking.priority_order == (("A",), ("B",), ("A", "B"))
    assert ranking.summary_for(("B",)).final_score > ranking.summary_for(("A",)).final_score
    for summary in ranking.summaries:
        assert summary.final_score is not None
        assert 0.0 <= summary.probe_performance_component <= 1.0
        assert 0.0 <= summary.ranking_stability_component <= 1.0
        assert 0.0 <= summary.feature_quality_component <= 1.0
        assert 0.0 <= summary.computational_efficiency_component <= 1.0
        assert 0.0 <= summary.llm_prior_component <= 1.0
    assert RankingWeights().llm_prior == 0.0


def test_default_method_ranking_is_invariant_to_llm_proposal_order():
    y = np.linspace(-1.0, 1.0, 40)
    common = {
        "y_true": y,
        "predictions": {"A": y + 0.02, "B": y + 0.10},
        "ranking_weights": RankingWeights(),
        "strata": np.repeat(np.arange(4), 10),
        "bootstrap": BootstrapConfig(replicates=30, random_seed=2),
    }

    prefer_a = rank_consensus_subsets(llm_prior_order=(("A",),), **common)
    prefer_b = rank_consensus_subsets(llm_prior_order=(("B",),), **common)

    assert prefer_a.priority_order == prefer_b.priority_order


def test_method_order_is_invariant_to_diagnostic_weight_changes():
    y = np.linspace(-1.0, 1.0, 40)
    common = {
        "y_true": y,
        "predictions": {"A": y, "B": y.copy()},
        "invariant_costs": {"A": 1.0, "B": 10.0},
        "invariant_quality_scores": {"A": 0.5, "B": 1.0},
        "strata": np.repeat(np.arange(4), 10),
        "bootstrap": BootstrapConfig(replicates=30, random_seed=2),
    }
    performance_diagnostic = rank_consensus_subsets(
        ranking_weights=RankingWeights(), **common
    )
    quality_diagnostic = rank_consensus_subsets(
        ranking_weights=RankingWeights(
            probe_performance=0.0,
            ranking_stability=0.0,
            feature_quality=1.0,
            computational_efficiency=0.0,
            llm_prior=0.0,
        ),
        **common,
    )

    assert performance_diagnostic.priority_order == quality_diagnostic.priority_order


def test_invariant_costs_must_match_prediction_universe():
    y = np.linspace(-1.0, 1.0, 10)
    with np.testing.assert_raises_regex(ValueError, "exactly"):
        rank_consensus_subsets(
            y_true=y,
            predictions={"A": y},
            invariant_costs={"B": 1.0},
            bootstrap=BootstrapConfig(replicates=5),
        )


def test_user_target_overrides_probe_derived_target_exactly():
    y = np.linspace(-1.0, 1.0, 30)
    ranking = rank_consensus_subsets(
        y_true=y,
        predictions={"PL": y},
        bootstrap=BootstrapConfig(replicates=15, random_seed=5),
    )

    target = resolve_target(metric="PCC", ranking=ranking, user_target=0.75)

    assert target.value == 0.75
    assert target.source == "user"
    assert target.derivation is None


def test_probe_derived_pcc_target_records_quantile_and_relaxation():
    y = np.linspace(-1.0, 1.0, 30)
    ranking = rank_consensus_subsets(
        y_true=y,
        predictions={"PL": y},
        bootstrap=BootstrapConfig(replicates=15, random_seed=5),
    )
    config = TargetConfig(
        auto_lower_quantile=0.10, absolute_relaxation={"PCC": 0.02}, sanity_floor=0.99,
    )

    target = resolve_target(
        metric="PCC", ranking=ranking, user_target=None, config=config
    )

    assert np.isclose(target.value, 0.98)
    assert target.source == "probe_derived"
    assert target.derivation["bootstrap_quantile"] == 0.10
    assert target.derivation["relaxation"] == 0.02
    assert "LOW_PREDICTIVE_SIGNAL" in target.warnings[0]


def test_probe_derived_rmse_target_uses_upper_quantile_and_relative_relaxation():
    y = np.linspace(0.0, 1.0, 30)
    ranking = rank_consensus_subsets(
        y_true=y,
        predictions={"PL": y + 0.10},
        primary_metric="RMSE",
        secondary_metrics=("MAE",),
        bootstrap=BootstrapConfig(replicates=15, random_seed=6),
    )

    target = resolve_target(
        metric="RMSE",
        ranking=ranking,
        user_target=None,
        config=TargetConfig(relative_relaxation={"RMSE": 0.05}),
    )

    assert np.isclose(target.value, 0.105)
    assert target.direction == "lower"
    assert target.derivation["bootstrap_quantile"] == 0.90
    assert target.derivation["relaxation_kind"] == "relative"
