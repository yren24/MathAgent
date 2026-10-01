from __future__ import annotations

import numpy as np
from mint_scout.data.casf_index import CasfRecord
from mint_scout.evaluate_external_gbt import (
    _casefold_overlap,
    bootstrap_pcc_interval,
)


def test_external_overlap_check_is_case_insensitive():
    train = (CasfRecord("1abc", 6.0, "train"),)
    external = (CasfRecord("1ABC", 7.0, "test"),)

    assert _casefold_overlap(train, external) == ["1abc"]


def test_bootstrap_pcc_interval_is_deterministic():
    y_true = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.asarray([1.1, 1.9, 3.2, 3.8, 5.1])

    first = bootstrap_pcc_interval(y_true, y_pred, n_bootstrap=100, seed=2026)
    second = bootstrap_pcc_interval(y_true, y_pred, n_bootstrap=100, seed=2026)

    assert first == second
    assert first["valid_replicates"] > 0
    assert first["lower"] <= first["upper"]
    assert -1.0 <= first["lower"] <= 1.0
    assert -1.0 <= first["upper"] <= 1.0
