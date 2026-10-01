from pathlib import Path

import numpy as np
import pytest

from mint_scout.data.casf_index import CasfRecord
from mint_scout.evaluate_legacy_pl import LEGACY_PL_SHAPE, load_feature_matrix


def test_load_feature_matrix_validates_and_flattens(tmp_path: Path):
    records = (CasfRecord("a001", 1.0, "train"), CasfRecord("a002", 2.0, "train"))
    for index, record in enumerate(records):
        np.save(tmp_path / f"{record.pdb_id}.npy", np.full(LEGACY_PL_SHAPE, index, dtype=np.float32))
    matrix = load_feature_matrix(records, tmp_path)
    assert matrix.shape == (2, 9600)
    assert matrix.dtype == np.float32


def test_load_feature_matrix_rejects_shape_mismatch(tmp_path: Path):
    record = CasfRecord("a001", 1.0, "train")
    np.save(tmp_path / "a001.npy", np.zeros((2, 2, 2)))
    with pytest.raises(ValueError, match="expected legacy PL shape"):
        load_feature_matrix((record,), tmp_path)
