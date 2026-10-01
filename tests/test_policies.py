from mint_scout.search.policies import (
    CheapestFirstPolicy,
    ConservativeComplementarityPolicy,
    FixedStandaloneRankingPolicy,
    PccStoppingRule,
)
from mint_scout.search.state import SearchState


def _state():
    return SearchState(
        task_id="synthetic",
        system_type="protein_ligand",
        pair_schema_id="casf_protein_ligand_40_v1",
        primary_metric="pcc",
        dev_sample_ids=("a", "b", "c"),
        test_sample_ids=("d",),
        fold_ids={"a": 0, "b": 1, "c": 2},
        fidelity_sample_ids={0.1: ("a",), 1.0: ("a", "b", "c")},
        current_fidelity=0.1,
        active_invariants=("PH", "PL", "CA"),
        singleton_scores={"PH": 0.5, "PL": 0.2, "CA": 0.4},
        pair_gains={("CA", "PL"): 0.03},
        necessity_scores={"PL": 0.01},
        estimated_next_costs={"PH": 3.0, "PL": 1.0, "CA": 2.0},
    )


def test_fixed_standalone_policy_uses_singleton_scores():
    decision = FixedStandaloneRankingPolicy({0.1: 2}).choose_next_active_set(_state())
    assert decision.promoted_invariants == ("PH", "CA")
    assert decision.eliminated_invariants == ("PL",)


def test_cheapest_first_policy_uses_incremental_cost():
    decision = CheapestFirstPolicy(retain_count=2).choose_next_active_set(_state())
    assert decision.promoted_invariants == ("PL", "CA")


def test_conservative_policy_protects_complementary_invariants():
    decision = ConservativeComplementarityPolicy(min_retain=2).choose_next_active_set(_state())
    assert set(decision.promoted_invariants) == {"PL", "CA"}


def test_pcc_stopping_rule_stops_at_target():
    rule = PccStoppingRule(target_pcc=0.85, patience=3, min_delta=0.005, max_subsets=31)
    decision = rule.should_stop((0.78, 0.851), evaluated_subset_count=2)
    assert decision.stop is True
    assert "reached target" in decision.reason


def test_pcc_stopping_rule_stops_after_patience_without_gain():
    rule = PccStoppingRule(target_pcc=0.85, patience=3, min_delta=0.005, max_subsets=31)
    decision = rule.should_stop((0.80, 0.802, 0.803, 0.804), evaluated_subset_count=4)
    assert decision.stop is True
    assert "below min_delta" in decision.reason


def test_pcc_stopping_rule_stops_at_subset_budget():
    rule = PccStoppingRule(target_pcc=0.85, patience=3, min_delta=0.005, max_subsets=31)
    decision = rule.should_stop((0.80,), evaluated_subset_count=31)
    assert decision.stop is True
    assert "max_subsets" in decision.reason
