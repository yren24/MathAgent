#!/usr/bin/env python3
"""Score every nonempty mean-prediction subset from frozen test reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from mint_scout.evaluate_external_gbt import _metrics
from mint_scout.evaluation.ensemble import enumerate_nonempty_subsets, mean_aggregate
from mint_scout.invariants.manifest import stable_hash


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-feature-manifest", type=Path, required=True)
    parser.add_argument("--report", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    sample_ids, y_true = _load_test_labels(args.test_feature_manifest)
    predictions, provenance = _load_predictions(args.report, sample_ids)
    rows = []
    for subset in enumerate_nonempty_subsets(tuple(predictions)):
        metrics = _metrics(y_true, mean_aggregate(predictions, subset))
        rows.append({"subset": list(subset), "metrics": metrics, "score": float(metrics["PCC"])})
    rows.sort(key=lambda row: row["score"], reverse=True)
    output = {
        "report_schema": "mint-agent.frozen-prediction-consensus.v1",
        "evaluation_mode": "direct_train_test_late_fusion_posthoc",
        "selection_prohibited": True,
        "primary_metric": "PCC",
        "sample_count": len(sample_ids),
        "sample_order_hash": stable_hash(sample_ids),
        "invariants": list(predictions),
        "best_subset": rows[0]["subset"],
        "best_score": rows[0]["score"],
        "subset_metrics": rows,
        "by_invariant_metrics": {
            name: _metrics(y_true, values) for name, values in predictions.items()
        },
        "report_provenance": provenance,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("best={} PCC={:.6f}".format("+".join(rows[0]["subset"]), rows[0]["score"]))
    return 0


def _load_test_labels(path: Path) -> tuple[tuple[str, ...], np.ndarray]:
    sample_ids: list[str] = []
    labels: list[float] = []
    seen: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        entry = json.loads(line)
        if entry.get("status") != "computed":
            continue
        sample_id = str(entry["sample_id"])
        if sample_id in seen:
            raise ValueError(f"Duplicate test sample: {sample_id}")
        seen.add(sample_id)
        sample_ids.append(sample_id)
        labels.append(float(entry["label"]))
    y_true = np.asarray(labels, dtype=float)
    if not sample_ids or not np.isfinite(y_true).all():
        raise ValueError("Test feature manifest has no finite computed labels")
    return tuple(sample_ids), y_true


def _load_predictions(
    values: list[str], sample_ids: tuple[str, ...]
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    predictions: dict[str, np.ndarray] = {}
    provenance: dict[str, str] = {}
    for value in values:
        name, separator, raw_path = value.partition("=")
        name = name.upper()
        if not separator or not name or name in predictions:
            raise ValueError(f"Expected unique NAME=PATH; got {value!r}")
        path = Path(raw_path)
        report = json.loads(path.read_text(encoding="utf-8"))
        raw_predictions = report.get("test_predictions", {})
        if name in raw_predictions and isinstance(raw_predictions[name], dict):
            raw_predictions = raw_predictions[name]
        missing = [sample_id for sample_id in sample_ids if sample_id not in raw_predictions]
        if missing:
            raise ValueError(f"{path} lacks {len(missing)} expected test predictions")
        predictions[name] = np.asarray([raw_predictions[sample_id] for sample_id in sample_ids], dtype=float)
        provenance[name] = str(path)
    return predictions, provenance


if __name__ == "__main__":
    raise SystemExit(main())
