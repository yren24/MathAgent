from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, fields
from pathlib import Path
from typing import Mapping

import numpy as np

from mint_scout.config import load_yaml
from mint_scout.data.casf_index import CasfRecord
from mint_scout.data.manifest_io import load_configured_protein_ligand_records
from mint_scout.evaluate_external_gbt import (
    _file_sha256,
    _load_feature_manifest,
    _metrics,
    _ordered_manifest_args,
    bootstrap_pcc_interval,
)
from mint_scout.evaluation.ensemble import (
    Subset,
    enumerate_nonempty_subsets,
    mean_aggregate,
)
from mint_scout.evaluation.metrics import get_metric, higher_is_better
from mint_scout.invariants.manifest import stable_hash
from mint_scout.models.gbt import GBTConfig, fit_predict_gbt
from mint_scout.representation import RepresentationSpec


REPORT_SCHEMA = "mint-agent.split-gbt-evaluation.v1"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Select a GBT invariant subset on validation, then evaluate the frozen "
            "selection once on test."
        )
    )
    parser.add_argument("--task-config", type=Path, required=True)
    parser.add_argument("--representation-spec", type=Path, required=True)
    parser.add_argument("--gbt-config", type=Path, required=True)
    for split in ("train", "validation", "test"):
        parser.add_argument(
            f"--{split}-feature-manifest",
            action="append",
            required=True,
            metavar="INVARIANT=PATH",
        )
        parser.add_argument(
            f"--{split}-feature-qc-report", type=Path, required=True
        )
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    report = evaluate_split_gbt(
        task_config_path=args.task_config,
        representation_path=args.representation_spec,
        gbt_config_path=args.gbt_config,
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
        f"dataset={report['dataset_id']} selected="
        f"{'+'.join(report['selection']['selected_subset'])} "
        f"validation_{report['primary_metric'].lower()}="
        f"{report['selection']['selected_score']:.6f} "
        f"test_pcc={report['final_test']['metrics']['PCC']:.6f} "
        f"output={args.output}"
    )
    return 0


def evaluate_split_gbt(
    *,
    task_config_path: Path,
    representation_path: Path,
    gbt_config_path: Path,
    train_manifest_values: list[str],
    validation_manifest_values: list[str],
    test_manifest_values: list[str],
    train_qc_report: Path,
    validation_qc_report: Path,
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
    primary_metric = str(task.get("primary_metric") or "PCC").upper()
    get_metric(primary_metric)

    manifests = {
        "train": _ordered_manifest_args(train_manifest_values),
        "validation": _ordered_manifest_args(validation_manifest_values),
        "test": _ordered_manifest_args(test_manifest_values),
    }
    invariant_order = tuple(manifests["train"])
    if not invariant_order:
        raise ValueError("At least one invariant feature manifest is required")
    for split in ("validation", "test"):
        if tuple(manifests[split]) != invariant_order:
            raise ValueError(
                "Train, validation, and test feature manifests must list the same "
                "invariants in the same order"
            )

    train_records = load_configured_protein_ligand_records(
        task, config_path=task_config_path, split="train"
    )
    validation_records = load_configured_protein_ligand_records(
        task, config_path=task_config_path, split="validation"
    )
    _assert_labeled(train_records, "training")
    _assert_labeled(validation_records, "validation")
    _assert_disjoint_splits(train_records, validation_records)
    train_ids = tuple(record.pdb_id for record in train_records)
    validation_ids = tuple(record.pdb_id for record in validation_records)
    _assert_qc(
        train_qc_report,
        representation=representation,
        sample_ids=train_ids,
        manifests=manifests["train"],
    )
    _assert_qc(
        validation_qc_report,
        representation=representation,
        sample_ids=validation_ids,
        manifests=manifests["validation"],
    )
    train_features = _features_for_split(
        train_records, manifests["train"], representation
    )
    validation_features = _features_for_split(
        validation_records, manifests["validation"], representation
    )
    y_train = _targets(train_records)
    y_validation = _targets(validation_records)

    started = time.perf_counter()
    validation_predictions = {
        invariant: fit_predict_gbt(
            train_features[invariant],
            y_train,
            validation_features[invariant],
            config=gbt_config,
        )
        for invariant in invariant_order
    }
    selected_subset, subset_scores = select_validation_subset(
        y_validation,
        validation_predictions,
        metric_name=primary_metric,
    )
    selection_hash = stable_hash(
        {
            "gbt_parameter_hash": gbt_config.parameter_hash,
            "primary_metric": primary_metric,
            "representation_hash": representation.spec_hash,
            "selected_subset": selected_subset,
            "subset_scores": [
                {"subset": subset, "score": score}
                for subset, score in sorted(subset_scores.items())
            ],
            "train_ids": train_ids,
            "validation_ids": validation_ids,
        }
    )

    # The test split is not scored until the validation choice is frozen above.
    test_records = load_configured_protein_ligand_records(
        task, config_path=task_config_path, split="test"
    )
    _assert_labeled(test_records, "test")
    _assert_disjoint_splits(train_records, validation_records, test_records)
    test_ids = tuple(record.pdb_id for record in test_records)
    _assert_qc(
        test_qc_report,
        representation=representation,
        sample_ids=test_ids,
        manifests=manifests["test"],
    )
    test_features = _features_for_split(
        test_records,
        {name: manifests["test"][name] for name in selected_subset},
        representation,
    )
    y_test = _targets(test_records)
    y_train_validation = np.concatenate((y_train, y_validation))
    test_predictions = {}
    for invariant in selected_subset:
        x_train_validation = np.concatenate(
            (train_features[invariant], validation_features[invariant]), axis=0
        )
        test_predictions[invariant] = fit_predict_gbt(
            x_train_validation,
            y_train_validation,
            test_features[invariant],
            config=gbt_config,
        )
    final_prediction = mean_aggregate(test_predictions, selected_subset)
    elapsed_seconds = time.perf_counter() - started
    final_metrics = _metrics(y_test, final_prediction)
    bootstrap = bootstrap_pcc_interval(
        y_test,
        final_prediction,
        n_bootstrap=n_bootstrap,
        seed=bootstrap_seed,
    )

    report: dict[str, object] = {
        "report_schema": REPORT_SCHEMA,
        "status": "COMPLETE",
        "dataset_id": str(task.get("task_id") or ""),
        "evidence_scope": "external_test",
        "primary_metric": primary_metric,
        "protocol": {
            "selection_split": "validation",
            "final_evaluation_split": "test",
            "representation_design_split": "train",
            "selection_frozen_before_test_scoring": True,
            "test_used_for_selection": False,
            "test_query_count": 1,
        },
        "representation_hash": representation.spec_hash,
        "representation_spec": representation.to_dict(),
        "selection_hash": selection_hash,
        "gbt_config": asdict(gbt_config),
        "gbt_parameter_hash": gbt_config.parameter_hash,
        "invariants": list(invariant_order),
        "sample_counts": {
            "train": len(train_records),
            "validation": len(validation_records),
            "test": len(test_records),
        },
        "sample_order_hashes": {
            "train": stable_hash(train_ids),
            "validation": stable_hash(validation_ids),
            "test": stable_hash(test_ids),
        },
        "feature_dimensions": {
            invariant: int(train_features[invariant].shape[1])
            for invariant in invariant_order
        },
        "feature_manifests": {
            split: {name: str(path) for name, path in split_manifests.items()}
            for split, split_manifests in manifests.items()
        },
        "feature_manifest_sha256": {
            split: {
                name: _file_sha256(path)
                for name, path in split_manifests.items()
            }
            for split, split_manifests in manifests.items()
        },
        "feature_qc_reports": {
            "train": str(train_qc_report),
            "validation": str(validation_qc_report),
            "test": str(test_qc_report),
        },
        "selection": {
            "selection_hash": selection_hash,
            "selected_subset": list(selected_subset),
            "selected_score": subset_scores[selected_subset],
            "subset_scores": [
                {"subset": list(subset), "score": score}
                for subset, score in sorted(
                    subset_scores.items(), key=lambda item: (len(item[0]), item[0])
                )
            ],
            "by_invariant_metrics": {
                invariant: _metrics(y_validation, prediction)
                for invariant, prediction in validation_predictions.items()
            },
        },
        "final_test": {
            "metrics": final_metrics,
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
        "fit_predict_elapsed_seconds": elapsed_seconds,
    }
    report["evaluation_hash"] = stable_hash(
        {
            "final_metrics": final_metrics,
            "gbt_parameter_hash": gbt_config.parameter_hash,
            "representation_hash": representation.spec_hash,
            "selection_hash": selection_hash,
            "test_manifest_hashes": report["feature_manifest_sha256"]["test"],
        }
    )
    return report


def select_validation_subset(
    y_true: np.ndarray,
    predictions: Mapping[str, np.ndarray],
    *,
    metric_name: str,
) -> tuple[Subset, dict[Subset, float]]:
    if not predictions:
        raise ValueError("Validation selection requires at least one invariant")
    metric = get_metric(metric_name)
    is_higher = higher_is_better(metric_name)
    scores = {
        subset: float(metric(y_true, mean_aggregate(predictions, subset)))
        for subset in enumerate_nonempty_subsets(predictions)
    }
    finite = [subset for subset, score in scores.items() if math.isfinite(score)]
    if not finite:
        raise ValueError("Validation selection produced no finite subset scores")
    selected = min(
        finite,
        key=lambda subset: (
            -scores[subset] if is_higher else scores[subset],
            len(subset),
            subset,
        ),
    )
    return selected, scores


def _features_for_split(
    records: tuple[CasfRecord, ...],
    manifests: Mapping[str, Path],
    representation: RepresentationSpec,
) -> dict[str, np.ndarray]:
    sample_ids = tuple(record.pdb_id for record in records)
    return {
        invariant: _load_feature_manifest(
            path,
            invariant=invariant,
            sample_ids=sample_ids,
            representation=representation,
        )
        for invariant, path in manifests.items()
    }


def _targets(records: tuple[CasfRecord, ...]) -> np.ndarray:
    return np.asarray([record.label for record in records], dtype=float)


def _assert_labeled(records: tuple[CasfRecord, ...], name: str) -> None:
    if not records:
        raise ValueError(f"{name} split contains no samples")
    missing = [record.pdb_id for record in records if record.label is None]
    if missing:
        raise ValueError(f"{name} split has missing labels: {missing[:20]}")


def _assert_disjoint_splits(*record_groups: tuple[CasfRecord, ...]) -> None:
    seen: dict[str, int] = {}
    overlaps: set[str] = set()
    for group_index, records in enumerate(record_groups):
        for record in records:
            key = record.pdb_id.casefold()
            previous = seen.setdefault(key, group_index)
            if previous != group_index:
                overlaps.add(record.pdb_id)
    if overlaps:
        raise ValueError(
            "Train, validation, and test sample IDs must be disjoint: "
            f"{sorted(overlaps)[:20]}"
        )


def _assert_qc(
    path: Path,
    *,
    representation: RepresentationSpec,
    sample_ids: tuple[str, ...],
    manifests: Mapping[str, Path],
) -> None:
    from mint_scout.feature_qc import assert_qc_report_compatible

    assert_qc_report_compatible(
        path=path,
        representation=representation,
        sample_ids=sample_ids,
        manifest_paths=dict(manifests),
    )


def _load_gbt_config(path: Path) -> GBTConfig:
    raw = load_yaml(path)
    allowed = {item.name for item in fields(GBTConfig)}
    unknown = sorted(set(raw) - allowed - {"source"})
    if unknown:
        raise ValueError(f"Unknown GBT config fields: {unknown}")
    return GBTConfig(**{key: value for key, value in raw.items() if key in allowed})


if __name__ == "__main__":
    raise SystemExit(main())
