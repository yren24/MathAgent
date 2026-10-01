import numpy as np

from mint_scout.evaluation.complementarity import camc, exclusion_necessity, pair_gains
from mint_scout.evaluation.ensemble import enumerate_nonempty_subsets, score_subsets
from mint_scout.evaluation.metrics import pcc


def test_subset_counts():
    assert len(enumerate_nonempty_subsets(("PH", "PL", "CA", "FPRC", "EIC"))) == 31
    assert len(enumerate_nonempty_subsets(("PH", "PL", "CA", "FPRC"))) == 15
    assert len(enumerate_nonempty_subsets(("PH", "PL", "CA"))) == 7


def test_complementarity_outputs_cover_active_invariants():
    y = np.array([0.0, 1.0, 2.0, 3.0])
    preds = {
        "PH": np.array([0.0, 1.0, 2.0, 3.0]),
        "PL": np.array([3.0, 2.0, 1.0, 0.0]),
        "CA": np.array([0.0, 1.2, 1.8, 3.0]),
    }
    subset_scores = score_subsets(y, preds, metric=pcc)
    active = tuple(preds)
    assert set(camc(subset_scores, active)) == set(active)
    assert set(exclusion_necessity(subset_scores, active)) == set(active)
    assert set(pair_gains(subset_scores, active)) == {("PH", "PL"), ("CA", "PH"), ("CA", "PL")}
