#!/usr/bin/env python3
"""Evaluate one historical 20/40-pair feature directory with the current GBT protocol."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import numpy as np

from mint_scout.evaluate_external_gbt import _metrics
from mint_scout.evaluate_split_gbt import _load_gbt_config
from mint_scout.models.gbt import fit_predict_gbt


BAD_IDS = frozenset({"3tei", "4as6"})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-root", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--invariant", required=True)
    parser.add_argument("--gbt-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    train_ids, train_y = _legacy_split(args.index_root, "train")
    test_ids, test_y = _legacy_split(args.index_root, "test")
    train_x = _load_feature_matrix(args.feature_dir, train_ids)
    test_x = _load_feature_matrix(args.feature_dir, test_ids)
    config = _load_gbt_config(args.gbt_config)
    prediction = fit_predict_gbt(train_x, train_y, test_x, config=config)
    report = {
        "report_schema": "mint-agent.legacy-feature-direct-gbt.v1",
        "evaluation_mode": "legacy_20_40_pair_feature_direct_train_test",
        "selection_prohibited": True,
        "invariant": args.invariant.upper(),
        "feature_dir": str(args.feature_dir),
        "feature_shape": list(np.load(args.feature_dir / f"{train_ids[0]}.npy", mmap_mode="r").shape),
        "input_feature_dimension": int(train_x.shape[1]),
        "sample_counts": {"train": len(train_ids), "test": len(test_ids)},
        "gbt_config": config.__dict__,
        "gbt_parameter_hash": config.parameter_hash,
        "metrics": _metrics(test_y, prediction),
        "test_predictions": {sample_id: float(prediction[index]) for index, sample_id in enumerate(test_ids)},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(f"invariant={report['invariant']} PCC={report['metrics']['PCC']:.6f}")
    return 0


def _legacy_split(index_root: Path, split: str) -> tuple[tuple[str, ...], np.ndarray]:
    sample_ids = tuple(
        sample_id
        for sample_id in ast.literal_eval((index_root / f"{split}_data_2016.txt").read_text())
        if sample_id not in BAD_IDS
    )
    labels = {}
    for line in (index_root / "2016_INDEX_refined.data").read_text(encoding="utf-8").splitlines()[6:4063]:
        labels[line[:4]] = float(line[18:23])
    return sample_ids, np.asarray([labels[sample_id] for sample_id in sample_ids], dtype=float)


def _load_feature_matrix(feature_dir: Path, sample_ids: tuple[str, ...]) -> np.ndarray:
    arrays = []
    shape = None
    for sample_id in sample_ids:
        path = feature_dir / f"{sample_id}.npy"
        if not path.exists():
            raise FileNotFoundError(path)
        array = np.load(path, allow_pickle=False).reshape(-1)
        if shape is None:
            shape = array.shape
        elif array.shape != shape:
            raise ValueError(f"Inconsistent feature shape for {path}: {array.shape} != {shape}")
        arrays.append(array)
    return np.asarray(arrays, dtype=np.float32)


if __name__ == "__main__":
    raise SystemExit(main())
