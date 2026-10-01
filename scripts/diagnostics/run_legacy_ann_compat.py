#!/usr/bin/env python3
"""Run the historical PLBind MLP on a manifest-backed feature representation.

The network, optimizer, scheduler, scaling, seed list, and final-epoch output
come directly from legacy ``ltest_ann_final.py``.  This adapter only replaces
the legacy feature-folder reader so a frozen manifest can be supplied safely.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.preprocessing import MinMaxScaler


LEGACY_SEEDS = (42, 1, 2, 3, 4, 5, 6, 7, 8, 9)
LEGACY_BAD_IDS = frozenset({"3tei", "4as6"})
LEGACY_LAYERS = (2048, 1024, 1024, 512, 512, 64)
LEGACY_LR = 8e-5
LEGACY_EPOCHS = 200
LEGACY_BATCH_SIZE = 32


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-root", type=Path, required=True)
    parser.add_argument("--index-root", type=Path, required=True)
    parser.add_argument("--invariant", required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--learning-rate", type=float, default=LEGACY_LR)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args(argv)

    train_ids, train_y = _legacy_split(args.index_root, "train")
    test_ids, test_y = _legacy_split(args.index_root, "test")
    train_rows = _manifest_rows(args.train_manifest, args.invariant)
    test_rows = _manifest_rows(args.test_manifest, args.invariant)
    _assert_exact_ids(train_rows, train_ids, args.train_manifest)
    _assert_exact_ids(test_rows, test_ids, args.test_manifest)
    print(
        f"verified invariant={args.invariant} train={len(train_ids)} test={len(test_ids)} "
        f"legacy_seeds={len(LEGACY_SEEDS)}",
        flush=True,
    )
    if args.verify_only:
        return 0

    train_x = _load_features(train_rows, train_ids)
    test_x = _load_features(test_rows, test_ids)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    train_x = scaler.fit_transform(train_x)
    test_x = scaler.transform(test_x)
    _run_exact_legacy_training(
        legacy_root=args.legacy_root,
        train_x=train_x,
        train_y=train_y,
        test_x=test_x,
        test_y=test_y,
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
    )
    return 0


def _legacy_split(index_root: Path, split: str) -> tuple[tuple[str, ...], np.ndarray]:
    split_ids = ast.literal_eval((index_root / f"{split}_data_2016.txt").read_text())
    sample_ids = tuple(str(sample_id) for sample_id in split_ids if sample_id not in LEGACY_BAD_IDS)
    labels: dict[str, float] = {}
    lines = (index_root / "2016_INDEX_refined.data").read_text(encoding="utf-8").splitlines()
    for line in lines[6:4063]:
        labels[line[:4]] = float(line[18:23])
    missing = [sample_id for sample_id in sample_ids if sample_id not in labels]
    if missing:
        raise ValueError(f"Legacy label index misses sample IDs: {missing[:5]}")
    return sample_ids, np.asarray([labels[sample_id] for sample_id in sample_ids], dtype=np.float32)


def _manifest_rows(path: Path, invariant: str) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        sample_id = str(row["sample_id"])
        if sample_id in rows:
            raise ValueError(f"Duplicate sample ID {sample_id} in {path}")
        if str(row.get("invariant", "")).upper() != invariant.upper():
            raise ValueError(f"Invariant mismatch in {path}: {sample_id}")
        if row.get("status") not in {"computed", "cached"}:
            raise ValueError(f"Unusable feature status for {sample_id}: {row.get('status')}")
        rows[sample_id] = row
    return rows


def _assert_exact_ids(rows: dict[str, dict[str, Any]], expected: tuple[str, ...], path: Path) -> None:
    if set(rows) != set(expected):
        missing = sorted(set(expected) - set(rows))
        extra = sorted(set(rows) - set(expected))
        raise ValueError(f"Manifest IDs differ from legacy split: {path}; missing={missing[:5]} extra={extra[:5]}")


def _load_features(rows: dict[str, dict[str, Any]], sample_ids: tuple[str, ...]) -> np.ndarray:
    arrays = [np.load(Path(str(rows[sample_id]["output_path"])), allow_pickle=False).reshape(-1) for sample_id in sample_ids]
    shape = arrays[0].shape
    if any(array.shape != shape for array in arrays[1:]):
        raise ValueError("Feature vectors do not have a common flattened shape")
    return np.asarray(arrays, dtype=np.float32)


def _run_exact_legacy_training(
    *,
    legacy_root: Path,
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    output_dir: Path,
    learning_rate: float,
) -> None:
    sys.path.insert(0, str(legacy_root))
    import ltest_ann_final as legacy  # pylint: disable=import-outside-toplevel

    # This replaces only ltest_ann_final.get_feature_label. Every later training
    # and inference operation is the original legacy implementation.
    legacy.get_feature_label = lambda year, feature_folders: (train_x, train_y, test_x, test_y)
    para = legacy.Para()
    para.lr = learning_rate
    para.epoch = LEGACY_EPOCHS
    para.batch_size = LEGACY_BATCH_SIZE
    para.layers = list(LEGACY_LAYERS)
    output_dir.mkdir(parents=True, exist_ok=True)
    for seed in LEGACY_SEEDS:
        legacy.prediction(para, 2016, seed, str(output_dir), ["manifest-backed"])
    pcc, rmse = legacy.get_metrics(2016, list(LEGACY_SEEDS), str(output_dir))
    (output_dir / "legacy_ann_protocol.json").write_text(
        json.dumps(
            {
                "protocol": "legacy_ltest_ann_final_task3_compatible",
                "seeds": list(LEGACY_SEEDS),
                "layers": list(LEGACY_LAYERS),
                "learning_rate": learning_rate,
                "epochs": LEGACY_EPOCHS,
                "batch_size": LEGACY_BATCH_SIZE,
                "normalization": "MinMaxScaler(feature_range=(-1, 1))",
                "prediction_rule": "legacy final epoch, mean over ten seed predictions",
                "pcc": pcc,
                "legacy_scaled_rmse": rmse,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"legacy_ann_pcc={pcc:.6f} output={output_dir}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
