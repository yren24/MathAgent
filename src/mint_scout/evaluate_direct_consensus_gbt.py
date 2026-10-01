from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from mint_scout.config import load_yaml
from mint_scout.data.manifest_io import load_configured_protein_ligand_records
from mint_scout.evaluate_external_gbt import _metrics, _ordered_manifest_args
from mint_scout.evaluate_split_gbt import (
    _assert_disjoint_splits,
    _assert_labeled,
    _features_for_split,
    _load_gbt_config,
    _targets,
)
from mint_scout.evaluation.ensemble import enumerate_nonempty_subsets, mean_aggregate
from mint_scout.evaluation.metrics import get_metric, higher_is_better
from mint_scout.invariants.manifest import stable_hash
from mint_scout.models.gbt import fit_predict_gbt
from mint_scout.representation import RepresentationSpec


REPORT_SCHEMA = "mint-agent.direct-test-consensus-gbt.v1"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fit fixed GBT models on train and report all consensus subsets on test."
    )
    parser.add_argument("--task-config", type=Path, required=True)
    parser.add_argument("--representation-spec", type=Path, required=True)
    parser.add_argument("--gbt-config", type=Path, required=True)
    parser.add_argument(
        "--train-feature-manifest", action="append", required=True, metavar="INVARIANT=PATH"
    )
    parser.add_argument(
        "--test-feature-manifest", action="append", required=True, metavar="INVARIANT=PATH"
    )
    parser.add_argument(
        "--prior-report",
        action="append",
        type=Path,
        default=[],
        help="Compatible earlier direct-evaluation report whose test predictions can be reused.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    report = evaluate_direct_test_consensus_gbt(
        task_config_path=args.task_config,
        representation_path=args.representation_spec,
        gbt_config_path=args.gbt_config,
        train_manifest_values=args.train_feature_manifest,
        test_manifest_values=args.test_feature_manifest,
        prior_report_paths=args.prior_report,
    )
    _write_report(args.output, report)
    print(
        f"status={report['status']} best={'+'.join(report['best_subset'])} "
        f"{report['primary_metric']}={report['best_score']:.6f} output={args.output}"
    )
    return 0


def evaluate_direct_test_consensus_gbt(
    *,
    task_config_path: Path,
    representation_path: Path,
    gbt_config_path: Path,
    train_manifest_values: list[str],
    test_manifest_values: list[str],
    prior_report_paths: list[Path] | None = None,
) -> dict[str, Any]:
    """Report every nonempty late-fusion subset on a user-requested test split.

    This is intentionally a post-hoc comparative analysis, not a candidate-selection
    procedure: test metrics are reported but must not be fed back into the workflow.
    """

    task = load_yaml(task_config_path)
    representation = RepresentationSpec.read(representation_path)
    representation.assert_frozen()
    gbt_config = _load_gbt_config(gbt_config_path)
    train_manifests = _ordered_manifest_args(train_manifest_values)
    test_manifests = _ordered_manifest_args(test_manifest_values)
    if not train_manifests or tuple(train_manifests) != tuple(test_manifests):
        raise ValueError("Train and test feature manifests must name the same invariants")

    train_records = load_configured_protein_ligand_records(
        task, config_path=task_config_path, split="train"
    )
    test_records = load_configured_protein_ligand_records(
        task, config_path=task_config_path, split="test"
    )
    _assert_labeled(train_records, "training")
    _assert_labeled(test_records, "test")
    _assert_disjoint_splits(train_records, test_records)
    train_ids = tuple(record.pdb_id for record in train_records)
    test_ids = tuple(record.pdb_id for record in test_records)
    y_train = _targets(train_records)
    y_test = _targets(test_records)
    predictions = _load_prior_predictions(
        prior_report_paths or [],
        allowed_invariants=tuple(train_manifests),
        representation_hash=representation.spec_hash,
        gbt_parameter_hash=gbt_config.parameter_hash,
        test_ids=test_ids,
    )
    missing_manifests = {
        invariant: path
        for invariant, path in train_manifests.items()
        if invariant not in predictions
    }
    missing_test_manifests = {
        invariant: test_manifests[invariant] for invariant in missing_manifests
    }

    started = time.perf_counter()
    if missing_manifests:
        train_features = _features_for_split(train_records, missing_manifests, representation)
        test_features = _features_for_split(test_records, missing_test_manifests, representation)
        predictions.update(
            {
                invariant: fit_predict_gbt(
                    train_features[invariant],
                    y_train,
                    test_features[invariant],
                    config=gbt_config,
                )
                for invariant in missing_manifests
            }
        )
    primary_metric = str(task.get("primary_metric") or "PCC").upper()
    metric = get_metric(primary_metric)
    is_higher = higher_is_better(primary_metric)
    subset_metrics = [
        {
            "subset": list(subset),
            "metrics": _metrics(y_test, mean_aggregate(predictions, subset)),
        }
        for subset in enumerate_nonempty_subsets(tuple(train_manifests))
    ]
    for row in subset_metrics:
        row["score"] = float(row["metrics"][primary_metric])
    best_score = (
        max(row["score"] for row in subset_metrics)
        if is_higher
        else min(row["score"] for row in subset_metrics)
    )
    best_row = next(row for row in subset_metrics if np.isclose(row["score"], best_score))
    report: dict[str, Any] = {
        "report_schema": REPORT_SCHEMA,
        "status": "COMPLETE",
        "dataset_id": str(task.get("task_id") or ""),
        "evidence_scope": "external_test",
        "evaluation_mode": "direct_train_test_consensus_posthoc",
        "selection_prohibited": True,
        "representation_hash": representation.spec_hash,
        "invariants": list(train_manifests),
        "primary_metric": primary_metric,
        "best_subset": best_row["subset"],
        "best_score": best_row["score"],
        "subset_metrics": subset_metrics,
        "by_invariant_metrics": {
            name: _metrics(y_test, values) for name, values in predictions.items()
        },
        "gbt_config": asdict(gbt_config),
        "gbt_parameter_hash": gbt_config.parameter_hash,
        "run_aggregation": "mean_predictions_then_compute_metric",
        "reused_prior_prediction_invariants": sorted(
            set(train_manifests) - set(missing_manifests)
        ),
        "protocol": {
            "full_train_cross_validation": False,
            "fit_split": "train",
            "evaluation_split": "test",
            "test_used_for_candidate_selection": False,
            "all_nonempty_subsets_scored": True,
        },
        "sample_counts": {"train": len(train_records), "test": len(test_records)},
        "sample_order_hashes": {
            "train": stable_hash(train_ids),
            "test": stable_hash(test_ids),
        },
        "feature_manifests": {
            "train": {name: str(path) for name, path in train_manifests.items()},
            "test": {name: str(path) for name, path in test_manifests.items()},
        },
        "test_predictions": {
            name: {
                sample_id: float(values[index])
                for index, sample_id in enumerate(test_ids)
            }
            for name, values in predictions.items()
        },
        "fit_predict_elapsed_seconds": time.perf_counter() - started,
    }
    report["evaluation_hash"] = stable_hash(
        {
            "representation_hash": representation.spec_hash,
            "gbt_parameter_hash": gbt_config.parameter_hash,
            "sample_order_hashes": report["sample_order_hashes"],
            "subset_metrics": subset_metrics,
        }
    )
    return report


def _load_prior_predictions(
    paths: list[Path],
    *,
    allowed_invariants: tuple[str, ...],
    representation_hash: str,
    gbt_parameter_hash: str,
    test_ids: tuple[str, ...],
) -> dict[str, np.ndarray]:
    """Load prior train-to-test predictions only when their provenance matches exactly."""

    predictions: dict[str, np.ndarray] = {}
    for path in paths:
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("report_schema") != REPORT_SCHEMA:
            raise ValueError(f"Unsupported prior report schema: {path}")
        if report.get("representation_hash") != representation_hash:
            raise ValueError(f"Prior report representation hash mismatch: {path}")
        if report.get("gbt_parameter_hash") != gbt_parameter_hash:
            raise ValueError(f"Prior report GBT configuration mismatch: {path}")
        protocol = report.get("protocol", {})
        if protocol.get("fit_split") != "train" or protocol.get("evaluation_split") != "test":
            raise ValueError(f"Prior report does not use direct train-to-test evaluation: {path}")
        if report.get("sample_order_hashes", {}).get("test") != stable_hash(test_ids):
            raise ValueError(f"Prior report test sample order mismatch: {path}")
        for invariant, by_sample_id in report.get("test_predictions", {}).items():
            if invariant not in allowed_invariants:
                raise ValueError(f"Prior report has unavailable invariant {invariant}: {path}")
            if invariant in predictions:
                raise ValueError(f"Duplicate prior predictions for {invariant}: {path}")
            if set(by_sample_id) != set(test_ids):
                raise ValueError(f"Prior report prediction IDs mismatch for {invariant}: {path}")
            values = np.asarray([by_sample_id[sample_id] for sample_id in test_ids], dtype=float)
            if not np.all(np.isfinite(values)):
                raise ValueError(f"Prior report has non-finite predictions for {invariant}: {path}")
            predictions[invariant] = values
    return predictions


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())
