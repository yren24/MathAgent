from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

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
from mint_scout.invariants.manifest import stable_hash
from mint_scout.models.gbt import fit_predict_gbt
from mint_scout.representation import RepresentationSpec


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train one GBT on concatenated fixed features and evaluate the test split."
    )
    parser.add_argument("--task-config", type=Path, required=True)
    parser.add_argument("--representation-spec", type=Path, required=True)
    parser.add_argument("--gbt-config", type=Path, required=True)
    parser.add_argument("--invariants", required=True, help="Colon-separated invariant names.")
    parser.add_argument("--train-feature-manifest", action="append", required=True)
    parser.add_argument("--test-feature-manifest", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = evaluate_direct_concat_gbt(
        task_config_path=args.task_config,
        representation_path=args.representation_spec,
        gbt_config_path=args.gbt_config,
        invariants=tuple(name.upper() for name in args.invariants.split(":") if name),
        train_manifest_values=args.train_feature_manifest,
        test_manifest_values=args.test_feature_manifest,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(
        f"subset={'+'.join(report['invariants'])} {report['primary_metric']}="
        f"{report['metrics'][report['primary_metric']]:.6f}",
        flush=True,
    )
    return 0


def evaluate_direct_concat_gbt(
    *,
    task_config_path: Path,
    representation_path: Path,
    gbt_config_path: Path,
    invariants: tuple[str, ...],
    train_manifest_values: list[str],
    test_manifest_values: list[str],
) -> dict[str, object]:
    if not invariants or len(set(invariants)) != len(invariants):
        raise ValueError("invariants must be a nonempty, duplicate-free sequence")
    task = load_yaml(task_config_path)
    representation = RepresentationSpec.read(representation_path)
    representation.assert_frozen()
    train_manifests = _ordered_manifest_args(train_manifest_values)
    test_manifests = _ordered_manifest_args(test_manifest_values)
    if tuple(train_manifests) != tuple(test_manifests) or set(train_manifests) != set(invariants):
        raise ValueError("Train/test manifests must exactly match requested invariants")
    gbt_config = _load_gbt_config(gbt_config_path)
    train_records = load_configured_protein_ligand_records(task, config_path=task_config_path, split="train")
    test_records = load_configured_protein_ligand_records(task, config_path=task_config_path, split="test")
    _assert_labeled(train_records, "training")
    _assert_labeled(test_records, "test")
    _assert_disjoint_splits(train_records, test_records)
    train_features = _features_for_split(train_records, train_manifests, representation)
    test_features = _features_for_split(test_records, test_manifests, representation)
    x_train = np.concatenate(
        [train_features[name].reshape(len(train_records), -1) for name in invariants], axis=1
    )
    x_test = np.concatenate(
        [test_features[name].reshape(len(test_records), -1) for name in invariants], axis=1
    )
    y_train = _targets(train_records)
    y_test = _targets(test_records)
    prediction = fit_predict_gbt(x_train, y_train, x_test, config=gbt_config)
    primary_metric = str(task.get("primary_metric") or "PCC").upper()
    test_ids = tuple(record.pdb_id for record in test_records)
    return {
        "report_schema": "mint-agent.direct-concat-gbt.v1",
        "evaluation_mode": "direct_train_test_early_fusion_posthoc",
        "selection_prohibited": True,
        "invariants": list(invariants),
        "primary_metric": primary_metric,
        "metrics": _metrics(y_test, prediction),
        "representation_hash": representation.spec_hash,
        "gbt_config": asdict(gbt_config),
        "gbt_parameter_hash": gbt_config.parameter_hash,
        "sample_counts": {"train": len(train_records), "test": len(test_records)},
        "test_sample_order_hash": stable_hash(test_ids),
        "input_feature_dimension": int(x_train.shape[1]),
        "test_predictions": {sample_id: float(prediction[index]) for index, sample_id in enumerate(test_ids)},
    }


if __name__ == "__main__":
    raise SystemExit(main())
