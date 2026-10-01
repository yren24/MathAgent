from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.stats import spearmanr

from mint_scout.compute_feature_batch import _selected_records
from mint_scout.config import load_yaml
from mint_scout.data.casf_index import CasfRecord
from mint_scout.evaluation.metrics import mae, pcc, r2, rmse
from mint_scout.feature_qc import assert_qc_report_compatible
from mint_scout.invariants.manifest import stable_hash
from mint_scout.invariants.plbind_tools import expected_feature_shape
from mint_scout.models.gbt import GBTConfig, fit_predict_gbt
from mint_scout.representation import RepresentationSpec
from mint_scout.run_casf_scout import _parse_manifest_args


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train frozen-representation GBT models and evaluate an external dataset."
    )
    parser.add_argument("--training-task-config", type=Path, required=True)
    parser.add_argument("--external-task-config", type=Path, required=True)
    parser.add_argument("--representation-spec", type=Path, required=True)
    parser.add_argument(
        "--train-feature-manifest", action="append", required=True, metavar="INVARIANT=PATH"
    )
    parser.add_argument(
        "--external-feature-manifest", action="append", required=True, metavar="INVARIANT=PATH"
    )
    parser.add_argument("--train-feature-qc-report", type=Path, default=None)
    parser.add_argument("--external-feature-qc-report", type=Path, default=None)
    parser.add_argument("--n-estimators", type=int, default=4000)
    parser.add_argument("--n-runs", type=int, default=10)
    parser.add_argument("--n-bootstrap", type=int, default=500)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    parser.add_argument(
        "--run-kind",
        choices=("formal_evaluation", "engineering_smoke"),
        default="formal_evaluation",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    report = evaluate_external_gbt(
        training_task_config=args.training_task_config,
        external_task_config=args.external_task_config,
        representation_path=args.representation_spec,
        train_manifest_values=args.train_feature_manifest,
        external_manifest_values=args.external_feature_manifest,
        train_qc_report=args.train_feature_qc_report,
        external_qc_report=args.external_feature_qc_report,
        gbt_config=replace(
            GBTConfig(), n_estimators=args.n_estimators, n_runs=args.n_runs
        ),
        n_bootstrap=args.n_bootstrap,
        bootstrap_seed=args.bootstrap_seed,
        run_kind=args.run_kind,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"dataset={report['external_dataset_id']} samples={report['evaluation_sample_count']} "
        f"features={'+'.join(report['invariants'])} "
        f"pcc={report['metrics']['PCC']:.6f} output={args.output}"
    )
    return 0


def evaluate_external_gbt(
    *,
    training_task_config: Path,
    external_task_config: Path,
    representation_path: Path,
    train_manifest_values: list[str],
    external_manifest_values: list[str],
    train_qc_report: Path | None,
    external_qc_report: Path | None,
    gbt_config: GBTConfig,
    n_bootstrap: int,
    bootstrap_seed: int,
    run_kind: str = "formal_evaluation",
) -> dict[str, object]:
    if n_bootstrap < 0:
        raise ValueError("n_bootstrap must be non-negative")
    if run_kind not in {"formal_evaluation", "engineering_smoke"}:
        raise ValueError("run_kind must be formal_evaluation or engineering_smoke")
    training_task = load_yaml(training_task_config)
    external_task = load_yaml(external_task_config)
    representation = RepresentationSpec.read(representation_path)
    representation.assert_frozen()
    train_manifests = _ordered_manifest_args(train_manifest_values)
    external_manifests = _ordered_manifest_args(external_manifest_values)
    if tuple(train_manifests) != tuple(external_manifests):
        raise ValueError(
            "Training and external feature manifests must list the same invariants in the same order"
        )

    train_records = _selected_records(
        training_task,
        sample_ids=None,
        split="train",
        offset=0,
        limit=None,
        config_path=training_task_config,
    )
    external_records = _selected_records(
        external_task,
        sample_ids=None,
        split="test",
        offset=0,
        limit=None,
        config_path=external_task_config,
    )
    _assert_labeled(train_records, "training")
    _assert_labeled(external_records, "external")
    overlap = _casefold_overlap(train_records, external_records)
    if overlap:
        raise ValueError(
            f"External dataset overlaps training PDB/sample IDs: {', '.join(overlap[:20])}"
        )

    train_ids = tuple(record.pdb_id for record in train_records)
    external_ids = tuple(record.pdb_id for record in external_records)
    if train_qc_report is not None:
        assert_qc_report_compatible(
            path=train_qc_report,
            representation=representation,
            sample_ids=train_ids,
            manifest_paths=train_manifests,
        )
    if external_qc_report is not None:
        assert_qc_report_compatible(
            path=external_qc_report,
            representation=representation,
            sample_ids=external_ids,
            manifest_paths=external_manifests,
        )

    x_train = _load_combined_features(
        train_records, train_manifests, representation=representation
    )
    x_external = _load_combined_features(
        external_records, external_manifests, representation=representation
    )
    y_train = np.asarray([record.label for record in train_records], dtype=float)
    y_external = np.asarray([record.label for record in external_records], dtype=float)
    started = time.perf_counter()
    predictions = fit_predict_gbt(
        x_train, y_train, x_external, config=gbt_config
    )
    elapsed_seconds = time.perf_counter() - started
    metrics = _metrics(y_external, predictions)
    pcc_interval = bootstrap_pcc_interval(
        y_external,
        predictions,
        n_bootstrap=n_bootstrap,
        seed=bootstrap_seed,
    )

    payload: dict[str, object] = {
        "report_schema": "mint-agent.external-gbt-evaluation.v1",
        "status": "COMPLETE",
        "evidence_scope": (
            "external_test" if run_kind == "formal_evaluation" else "smoke"
        ),
        "run_kind": run_kind,
        "training_dataset_id": str(training_task.get("task_id")),
        "external_dataset_id": str(external_task.get("task_id")),
        "representation_hash": representation.spec_hash,
        "representation_spec": representation.to_dict(),
        "invariants": list(train_manifests),
        "train_feature_manifests": {
            name: str(path) for name, path in train_manifests.items()
        },
        "external_feature_manifests": {
            name: str(path) for name, path in external_manifests.items()
        },
        "train_feature_manifest_sha256": {
            name: _file_sha256(path) for name, path in train_manifests.items()
        },
        "external_feature_manifest_sha256": {
            name: _file_sha256(path) for name, path in external_manifests.items()
        },
        "train_feature_qc_report": str(train_qc_report) if train_qc_report else None,
        "external_feature_qc_report": (
            str(external_qc_report) if external_qc_report else None
        ),
        "train_external_id_overlap": [],
        "train_sample_count": len(train_records),
        "evaluation_sample_count": len(external_records),
        "train_sample_order_hash": stable_hash(train_ids),
        "evaluation_sample_order_hash": stable_hash(external_ids),
        "feature_dimension": int(x_train.shape[1]),
        "gbt_config": asdict(gbt_config),
        "gbt_parameter_hash": gbt_config.parameter_hash,
        "fit_predict_elapsed_seconds": elapsed_seconds,
        "metrics": metrics,
        "bootstrap": {
            "metric": "PCC",
            "n_bootstrap": n_bootstrap,
            "seed": bootstrap_seed,
            "confidence": 0.95,
            **pcc_interval,
        },
        "target_summary": {
            "training": _numeric_summary(y_train),
            "external": _numeric_summary(y_external),
        },
        "predictions": [
            {
                "sample_id": record.pdb_id,
                "target": float(target),
                "prediction": float(prediction),
                "absolute_error": abs(float(target) - float(prediction)),
            }
            for record, target, prediction in zip(
                external_records, y_external, predictions
            )
        ],
    }
    payload["evaluation_hash"] = stable_hash(
        {
            "external_dataset_id": payload["external_dataset_id"],
            "external_manifest_hashes": payload["external_feature_manifest_sha256"],
            "gbt_parameter_hash": payload["gbt_parameter_hash"],
            "metrics": metrics,
            "representation_hash": representation.spec_hash,
            "train_manifest_hashes": payload["train_feature_manifest_sha256"],
        }
    )
    return payload


def bootstrap_pcc_interval(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    n_bootstrap: int,
    seed: int,
) -> dict[str, object]:
    if n_bootstrap == 0:
        return {"lower": None, "upper": None, "valid_replicates": 0}
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rng = np.random.default_rng(seed)
    estimates: list[float] = []
    for _ in range(n_bootstrap):
        indices = rng.integers(0, len(y_true), size=len(y_true))
        sampled_true = y_true[indices]
        sampled_pred = y_pred[indices]
        if np.std(sampled_true) == 0.0 or np.std(sampled_pred) == 0.0:
            continue
        value = float(np.corrcoef(sampled_true, sampled_pred)[0, 1])
        if math.isfinite(value):
            estimates.append(value)
    if not estimates:
        raise ValueError("No valid bootstrap PCC replicates")
    lower, upper = np.quantile(np.asarray(estimates), (0.025, 0.975))
    return {
        "lower": float(lower),
        "upper": float(upper),
        "valid_replicates": len(estimates),
    }


def _load_combined_features(
    records: tuple[CasfRecord, ...],
    manifest_paths: dict[str, Path],
    *,
    representation: RepresentationSpec,
) -> np.ndarray:
    sample_ids = tuple(record.pdb_id for record in records)
    matrices = [
        _load_feature_manifest(
            path,
            invariant=invariant,
            sample_ids=sample_ids,
            representation=representation,
        )
        for invariant, path in manifest_paths.items()
    ]
    return np.concatenate(matrices, axis=1)


def _load_feature_manifest(
    path: Path,
    *,
    invariant: str,
    sample_ids: tuple[str, ...],
    representation: RepresentationSpec,
) -> np.ndarray:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        sample_id = str(row["sample_id"])
        if sample_id in by_id:
            raise ValueError(f"Duplicate sample {sample_id} in {path}")
        by_id[sample_id] = row
    if set(by_id) != set(sample_ids):
        missing = sorted(set(sample_ids) - set(by_id))
        extra = sorted(set(by_id) - set(sample_ids))
        raise ValueError(
            f"{invariant} manifest IDs do not match records; missing={missing[:5]} extra={extra[:5]}"
        )
    expected_shape = expected_feature_shape(invariant, representation)
    arrays = []
    for sample_id in sample_ids:
        row = by_id[sample_id]
        if str(row.get("invariant", "")).upper() != invariant:
            raise ValueError(f"Manifest invariant mismatch for {sample_id}")
        if row.get("status") not in {"computed", "cached"}:
            raise ValueError(f"Feature generation failed for {invariant}/{sample_id}")
        output_path = Path(str(row["output_path"]))
        if f"/repr-{representation.spec_hash}/" not in output_path.as_posix():
            raise ValueError(f"Feature path has wrong representation hash: {output_path}")
        array = np.load(output_path, allow_pickle=False)
        if array.shape != expected_shape:
            raise ValueError(
                f"Feature shape mismatch for {invariant}/{sample_id}: "
                f"expected {expected_shape}, got {array.shape}"
            )
        if not np.all(np.isfinite(array)):
            raise ValueError(f"Feature contains NaN or Inf for {invariant}/{sample_id}")
        arrays.append(np.asarray(array, dtype=np.float32).reshape(-1))
    return np.stack(arrays)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    spearman = spearmanr(y_true, y_pred).statistic
    return {
        "PCC": pcc(y_true, y_pred),
        "Spearman": float(spearman),
        "RMSE": rmse(y_true, y_pred),
        "MAE": mae(y_true, y_pred),
        "R2": r2(y_true, y_pred),
    }


def _numeric_summary(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=float)
    return {
        "count": int(values.size),
        "minimum": float(np.min(values)),
        "q25": float(np.quantile(values, 0.25)),
        "median": float(np.median(values)),
        "q75": float(np.quantile(values, 0.75)),
        "maximum": float(np.max(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
    }


def _assert_labeled(records: Iterable[CasfRecord], name: str) -> None:
    records = tuple(records)
    if not records:
        raise ValueError(f"{name} dataset contains no selected samples")
    missing = [record.pdb_id for record in records if record.label is None]
    if missing:
        raise ValueError(f"{name} dataset has missing labels: {missing[:20]}")


def _casefold_overlap(
    train_records: Iterable[CasfRecord], external_records: Iterable[CasfRecord]
) -> list[str]:
    train_ids = {record.pdb_id.casefold() for record in train_records}
    external_ids = {record.pdb_id.casefold() for record in external_records}
    return sorted(train_ids & external_ids)


def _ordered_manifest_args(values: list[str]) -> dict[str, Path]:
    parsed = _parse_manifest_args(values)
    return {name: parsed[name] for name in parsed}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
