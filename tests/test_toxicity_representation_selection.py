from __future__ import annotations

from mint_scout.toxicity.representation_selection import (
    ToxicityRepresentationPolicy,
    _hierarchical_select,
)


def _row(candidate_id, coverage, filtration, warnings, dimensions):
    return {
        "candidate_id": candidate_id,
        "pair_coverage": coverage,
        "filtration_adjustment_count": filtration,
        "feature_degeneracy_warning_count": warnings,
        "feature_warning_count": warnings,
        "total_feature_dimensions": dimensions,
    }


def test_representation_selection_is_hierarchical_not_weighted():
    rows = [
        _row("legacy-30-pairs-legacy-grid", 1.0, 2, 0, 100),
        _row("adaptive", 0.99, 0, 0, 200),
        _row("cheap-low-coverage", 0.80, 0, 0, 10),
    ]

    selected, trace = _hierarchical_select(
        rows,
        baseline_id="legacy-30-pairs-legacy-grid",
        policy=ToxicityRepresentationPolicy(pair_coverage_tolerance=0.02),
    )

    assert selected["candidate_id"] == "adaptive"
    coverage = next(
        stage for stage in trace if stage["stage"] == "element_pair_support_coverage"
    )
    assert coverage["survivors"] == [
        "legacy-30-pairs-legacy-grid",
        "adaptive",
    ]
    assert all("score" not in stage for stage in trace)


def test_representation_selection_prefers_baseline_on_exact_tie():
    rows = [
        _row("legacy-30-pairs-legacy-grid", 1.0, 0, 0, 100),
        _row("adaptive", 1.0, 0, 0, 100),
    ]

    selected, _ = _hierarchical_select(
        rows,
        baseline_id="legacy-30-pairs-legacy-grid",
        policy=ToxicityRepresentationPolicy(),
    )

    assert selected["candidate_id"] == "legacy-30-pairs-legacy-grid"
