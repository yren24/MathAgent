from mint_scout.summarize_probe_ablation import build_summary


def _arm(name, strategy, order):
    metrics = []
    for index, subset in enumerate(order):
        metrics.append(
            {
                "subset_id": subset,
                "nominal_primary_score": 0.8 - index * 0.01,
                "bootstrap_ci_low": 0.7,
                "bootstrap_ci_high": 0.9,
                "estimated_full_acquisition_wall_seconds": 10.0 + index,
            }
        )
    return {
        "arm_id": name,
        "selection_path": "selection.json",
        "audit_path": "audit.json",
        "scout_path": "scout.json",
        "selection": {
            "sampling_strategy": strategy,
            "sampling_seed": 1,
            "probe_final_size": 750,
            "selection_hash": name,
            "representation_hash": "repr",
        },
        "audit": {
            "status": "PASS",
            "distribution_comparisons": {
                "target": {"ks_distance": 0.02, "standardized_mean_difference": 0.03}
            },
        },
        "scout": {
            "consensus_metrics": metrics,
            "frozen_priority_order": [[item] for item in order],
            "top_tier": [[order[0]], [order[1]]],
        },
    }


def test_summary_reports_rank_agreement_and_random_aggregate():
    summary = build_summary(
        [
            _arm("stratified", "stratified_pair_coverage", ["PL", "PH", "CA"]),
            _arm("random-1", "uniform_random", ["PL", "CA", "PH"]),
        ],
        baseline="stratified",
    )

    random = summary["arms"][1]
    assert random["agreement_with_baseline"]["top_subset_exact_match"] is True
    assert random["agreement_with_baseline"]["top_tier_jaccard"] == 1 / 3
    assert summary["random_arm_summary"]["arm_count"] == 1
