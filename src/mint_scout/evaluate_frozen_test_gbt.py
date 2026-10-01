from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from mint_scout.config import load_yaml
from mint_scout.data.manifest_io import load_configured_protein_ligand_records
from mint_scout.evaluate_external_gbt import (
    _file_sha256,
    _metrics,
    _ordered_manifest_args,
    bootstrap_pcc_interval,
)
from mint_scout.evaluate_split_gbt import (
    _assert_disjoint_splits,
    _assert_labeled,
    _assert_qc,
    _features_for_split,
    _load_gbt_config,
    _targets,
)
from mint_scout.evaluation.ensemble import mean_aggregate
from mint_scout.evaluation.metrics import higher_is_better
from mint_scout.execution.engine import _target_reached
from mint_scout.invariants.manifest import stable_hash
from mint_scout.models.gbt import fit_predict_gbt
from mint_scout.representation import RepresentationSpec


REPORT_SCHEMA = "mint-agent.frozen-test-gbt-evaluation.v1"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Refit a frozen invariant subset and evaluate the test split once."
    )
    parser.add_argument("--task-config", type=Path, required=True)
    parser.add_argument("--representation-spec", type=Path, required=True)
    parser.add_argument("--gbt-config", type=Path, required=True)
    parser.add_argument("--selection-report", type=Path, required=True)
    parser.add_argument(
        "--train-feature-manifest",
        action="append",
        required=True,
        metavar="INVARIANT=PATH",
    )
    parser.add_argument(
        "--validation-feature-manifest",
        action="append",
        default=[],
        metavar="INVARIANT=PATH",
    )
    parser.add_argument(
        "--test-feature-manifest",
        action="append",
        required=True,
        metavar="INVARIANT=PATH",
    )
    parser.add_argument("--train-feature-qc-report", type=Path, required=True)
    parser.add_argument("--validation-feature-qc-report", type=Path, default=None)
    parser.add_argument("--test-feature-qc-report", type=Path, required=True)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    report = evaluate_frozen_test_gbt(
        task_config_path=args.task_config,
        representation_path=args.representation_spec,
        gbt_config_path=args.gbt_config,
        selection_report_path=args.selection_report,
        train_manifest_values=args.train_feature_manifest,
        validation_manifest_values=args.validation_feature_manifest,
        test_manifest_values=args.test_feature_manifest,
        train_qc_report=args.train_feature_qc_report,
        validation_qc_report=args.validation_feature_qc_report,
        test_qc_report=args.test_feature_qc_report,
        n_bootstrap=args.n_bootstrap,
        bootstrap_seed=args.bootstrap_seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"selected={'+'.join(report['selected_subset'])} "
        f"test_pcc={report['final_test']['metrics']['PCC']:.6f} output={args.output}"
    )
    return 0


def evaluate_frozen_test_gbt(
    *,
    task_config_path: Path,
    representation_path: Path,
    gbt_config_path: Path,
    selection_report_path: Path,
    train_manifest_values: list[str],
    validation_manifest_values: list[str],
    test_manifest_values: list[str],
    train_qc_report: Path,
    validation_qc_report: Path | None,
    test_qc_report: Path,
    n_bootstrap: int,
    bootstrap_seed: int,
) -> dict[str, object]:
    if n_bootstrap < 0:
        raise ValueError("n_bootstrap must be non-negative")
    task = load_yaml(task_config_path)
    representation = RepresentationSpec.read(representation_path)
    representation.assert_frozen()
    gbt_config = _load_gbt_config(gbt_config_path)
    selection_report = _load_selection_report(selection_report_path)
    selected_subset = _selected_subset(selection_report)
    if selection_report.get("representation_hash") != representation.spec_hash:
        raise ValueError("Selection and final-test representations do not match")
    if selection_report.get("gbt_parameter_hash") != gbt_config.parameter_hash:
        raise ValueError("Selection and final-test GBT parameters do not match")

    train_manifests = _ordered_manifest_args(train_manifest_values)
    validation_manifests = _ordered_manifest_args(validation_manifest_values)
    test_manifests = _ordered_manifest_args(test_manifest_values)
    for split_name, manifests in (
        ("train", train_manifests),
        ("test", test_manifests),
    ):
        if tuple(manifests) != selected_subset:
            raise ValueError(
                f"{split_name} manifests must contain only the frozen subset"
            )
    if validation_manifests and tuple(validation_manifests) != selected_subset:
        raise ValueError("validation manifests must contain only the frozen subset")
    if bool(validation_manifests) != bool(validation_qc_report):
        raise ValueError("Validation manifests and QC report must be provided together")

    train_records = load_configured_protein_ligand_records(
        task, config_path=task_config_path, split="train"
    )
    test_records = load_configured_protein_ligand_records(
        task, config_path=task_config_path, split="test"
    )
    _assert_labeled(train_records, "training")
    _assert_labeled(test_records, "test")
    validation_records = ()
    if validation_manifests:
        validation_records = load_configured_protein_ligand_records(
            task, config_path=task_config_path, split="validation"
        )
        _assert_labeled(validation_records, "validation")
    _assert_disjoint_splits(train_records, validation_records, test_records)
    train_ids = tuple(record.pdb_id for record in train_records)
    validation_ids = tuple(record.pdb_id for record in validation_records)
    test_ids = tuple(record.pdb_id for record in test_records)
    _assert_qc(
        train_qc_report,
        representation=representation,
        sample_ids=train_ids,
        manifests=train_manifests,
    )
    if validation_manifests and validation_qc_report is not None:
        _assert_qc(
            validation_qc_report,
            representation=representation,
            sample_ids=validation_ids,
            manifests=validation_manifests,
        )
    _assert_qc(
        test_qc_report,
        representation=representation,
        sample_ids=test_ids,
        manifests=test_manifests,
    )
    train_features = _features_for_split(train_records, train_manifests, representation)
    validation_features = (
        _features_for_split(validation_records, validation_manifests, representation)
        if validation_manifests
        else {}
    )
    test_features = _features_for_split(test_records, test_manifests, representation)
    y_train = _targets(train_records)
    y_validation = (
        _targets(validation_records) if validation_records else np.asarray([])
    )
    y_fit = np.concatenate((y_train, y_validation)) if validation_records else y_train
    y_test = _targets(test_records)

    started = time.perf_counter()
    test_predictions: dict[str, np.ndarray] = {}
    for invariant in selected_subset:
        x_fit = train_features[invariant]
        if validation_records:
            x_fit = np.concatenate((x_fit, validation_features[invariant]), axis=0)
        test_predictions[invariant] = fit_predict_gbt(
            x_fit, y_fit, test_features[invariant], config=gbt_config
        )
    final_prediction = mean_aggregate(test_predictions, selected_subset)
    final_metrics = _metrics(y_test, final_prediction)
    target_metric = str(selection_report.get("target_metric") or "").upper()
    target_value = float(selection_report["target_value"])
    if target_metric not in final_metrics:
        raise ValueError(
            f"Final-test metrics do not contain target metric {target_metric!r}"
        )
    target_score = float(final_metrics[target_metric])
    target_reached = _target_reached(
        target_score, target_value, higher_is_better(target_metric),
    )
    bootstrap = bootstrap_pcc_interval(
        y_test, final_prediction, n_bootstrap=n_bootstrap, seed=bootstrap_seed,
    )
    selection_sha256 = _file_sha256(selection_report_path)
    report: dict[str, object] = {
        "report_schema": REPORT_SCHEMA,
        "status": "TARGET_REACHED" if target_reached else "TARGET_NOT_REACHED",
        "dataset_id": str(task.get("task_id") or ""),
        "evidence_scope": "external_test",
        "representation_hash": representation.spec_hash,
        "selection_hash": str(
            selection_report.get("selection_hash") or selection_sha256
        ),
        "selection_report": str(selection_report_path),
        "selection_report_sha256": selection_sha256,
        "selected_subset": list(selected_subset),
        "invariants": list(selected_subset),
        "evaluation_mode": (
            "explicit_validation_and_test"
            if validation_records
            else "explicit_labeled_test"
        ),
        "target_metric": target_metric,
        "target_value": target_value,
        "target_source": selection_report.get("target_source"),
        "selection_objective": selection_report.get("selection_objective")
        or "satisfy_target",
        "target_score": target_score,
        "target_reached": target_reached,
        "gbt_config": asdict(gbt_config),
        "gbt_parameter_hash": gbt_config.parameter_hash,
        "protocol": {
            "selection_evidence_scope": selection_report.get("evidence_scope"),
            "selection_objective": selection_report.get("selection_objective")
            or "satisfy_target",
            "representation_design_split": "train",
            "refit_splits": ["train", "validation"]
            if validation_records
            else ["train"],
            "final_evaluation_split": "test",
            "selection_frozen_before_test_feature_loading": True,
            "test_used_for_selection": False,
            "test_query_count": 1,
        },
        "sample_counts": {
            "train": len(train_records),
            "validation": len(validation_records),
            "test": len(test_records),
        },
        "sample_order_hashes": {
            "train": stable_hash(train_ids),
            "validation": stable_hash(validation_ids) if validation_ids else None,
            "test": stable_hash(test_ids),
        },
        "feature_manifests": {
            "train": {name: str(path) for name, path in train_manifests.items()},
            "validation": {
                name: str(path) for name, path in validation_manifests.items()
            },
            "test": {name: str(path) for name, path in test_manifests.items()},
        },
        "feature_manifest_sha256": {
            "train": {
                name: _file_sha256(path) for name, path in train_manifests.items()
            },
            "validation": {
                name: _file_sha256(path) for name, path in validation_manifests.items()
            },
            "test": {name: _file_sha256(path) for name, path in test_manifests.items()},
        },
        "feature_qc_reports": {
            "train": str(train_qc_report),
            "validation": str(validation_qc_report) if validation_qc_report else None,
            "test": str(test_qc_report),
        },
        "final_test": {
            "metrics": final_metrics,
            "target_reached": target_reached,
            "bootstrap_pcc_95": {
                "n_bootstrap": n_bootstrap,
                "seed": bootstrap_seed,
                **bootstrap,
            },
            "predictions": [
                {
                    "sample_id": record.pdb_id,
                    "target": float(target),
                    "prediction": float(prediction),
                    "absolute_error": abs(float(target) - float(prediction)),
                }
                for record, target, prediction in zip(
                    test_records, y_test, final_prediction
                )
            ],
        },
        "fit_predict_elapsed_seconds": time.perf_counter() - started,
    }
    report["evaluation_hash"] = stable_hash(
        {
            "final_metrics": final_metrics,
            "gbt_parameter_hash": gbt_config.parameter_hash,
            "representation_hash": representation.spec_hash,
            "selection_report_sha256": selection_sha256,
            "test_manifest_hashes": report["feature_manifest_sha256"]["test"],
        }
    )
    return report


def _load_selection_report(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Selection report must contain a JSON object")
    if payload.get("evidence_scope") not in {"probe", "full_train", "validation"}:
        raise ValueError("Selection must be frozen from probe, train CV, or validation")
    if payload.get("status") not in {
        "TARGET_REACHED",
        "TARGET_NOT_REACHED",
        "MAXIMIZATION_COMPLETE",
    }:
        raise ValueError("Selection report is not a frozen terminal selection")
    return payload


def _selected_subset(payload: Mapping[str, Any]) -> tuple[str, ...]:
    raw = payload.get("selected_subset")
    selection = payload.get("selection")
    result = payload.get("result")
    if raw is None and isinstance(selection, Mapping):
        raw = selection.get("selected_subset")
    if raw is None and isinstance(result, Mapping):
        raw = result.get("selected_subset")
    if not isinstance(raw, list) or not raw:
        raise ValueError("Selection report does not contain selected_subset")
    subset = tuple(str(value).upper() for value in raw)
    if len(set(subset)) != len(subset):
        raise ValueError("Selection report contains duplicate invariants")
    return subset


if __name__ == "__main__":
    raise SystemExit(main())
