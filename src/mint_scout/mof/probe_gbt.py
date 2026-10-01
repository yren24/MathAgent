"""Frozen-probe GBT comparison for cached MOF legacy topology features."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from .agent_workflow import MofAgentRecord, load_agent_manifest
from .legacy import topology_for
from .standard_metrics import regression_metrics


PROBE_GBT_SCHEMA = "mint-agent.mof-probe-gbt.v1"


def run_probe_gbt(
    *,
    manifest_path: str | Path,
    probe_id_path: str | Path,
    feature_dir: str | Path,
    topology: str,
    property_name: str,
    output_dir: str | Path,
    folds: int = 5,
    repeats: int = 3,
    seed: int = 2026,
    n_estimators: int = 10_000,
    max_depth: int = 7,
    learning_rate: float = 0.005,
) -> dict[str, Any]:
    """Evaluate one cached topology on a frozen, train-only MOF probe.

    The estimator settings intentionally mirror the audited legacy GBT script;
    this runner differs only by restricting inputs to the frozen probe.
    """
    if folds < 2 or repeats < 1:
        raise ValueError("folds must be at least two and repeats must be positive")
    records = {record.sample_id: record for record in load_agent_manifest(manifest_path)}
    probe_ids = _load_probe_ids(probe_id_path)
    probe_records = _resolve_probe_records(records, probe_ids)
    if len(probe_records) < folds:
        raise ValueError(f"Probe has {len(probe_records)} records but needs {folds} folds")

    topology_name = topology_for(topology) if topology.upper() in {key.upper() for key in _tool_names()} else topology
    X = _load_features(probe_records, Path(feature_dir) / topology_name / property_name)
    y = np.asarray([record.target for record in probe_records], dtype=np.float64)
    sample_ids = np.asarray([record.sample_id for record in probe_records])
    fold_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []

    for repeat in range(repeats):
        splitter = KFold(n_splits=folds, shuffle=True, random_state=seed + repeat)
        for fold, (train_index, test_index) in enumerate(splitter.split(X)):
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X[train_index])
            X_test = scaler.transform(X[test_index])
            model = GradientBoostingRegressor(
                loss="squared_error",
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=2,
                min_samples_leaf=1,
                subsample=0.5,
                max_features="sqrt",
                learning_rate=learning_rate,
                random_state=seed + repeat * 10 + fold,
            )
            model.fit(X_train, y[train_index])
            prediction = model.predict(X_test)
            actual = y[test_index]
            metrics = regression_metrics(actual.tolist(), prediction.tolist())
            pcc = _pcc(actual, prediction)
            row = {
                "repeat": repeat,
                "fold": fold,
                "test_size": len(test_index),
                "mae": float(mean_absolute_error(actual, prediction)),
                "pcc": pcc,
                **metrics,
            }
            fold_rows.append(row)
            prediction_rows.extend(
                {
                    "repeat": repeat,
                    "fold": fold,
                    "sample_id": sample_id,
                    "true": float(target),
                    "pred": float(predicted),
                }
                for sample_id, target, predicted in zip(sample_ids[test_index], actual, prediction)
            )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    stem = f"{property_name}_{topology_name}_probe_gbt"
    prediction_path = output / f"{stem}_predictions.csv"
    metrics_path = output / f"{stem}_metrics.json"
    _write_predictions(prediction_rows, prediction_path)
    report = {
        "report_schema": PROBE_GBT_SCHEMA,
        "evidence_scope": "train_only_probe",
        "property": property_name,
        "topology": topology_name,
        "probe_id_path": str(probe_id_path),
        "probe_size": len(probe_records),
        "folds": folds,
        "repeats": repeats,
        "seed": seed,
        "model": {
            "estimator": "GradientBoostingRegressor",
            "legacy_matched_settings": True,
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": 0.5,
            "max_features": "sqrt",
        },
        "fold_metrics": fold_rows,
        "repeat_summaries": _repeat_summaries(fold_rows),
        "summary": _summarize_metrics(fold_rows),
        "predictions_path": str(prediction_path),
    }
    metrics_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _tool_names() -> tuple[str, ...]:
    return ("MOF_HOMOLOGY", "MOF_LAPLACIAN", "MOF_FACET", "MOF_FORMAN", "MOF_CURVATURE")


def _load_probe_ids(path: str | Path) -> tuple[str, ...]:
    ids = tuple(line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip())
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("Probe ID file must contain unique, non-empty sample IDs")
    return ids


def _resolve_probe_records(records: dict[str, MofAgentRecord], probe_ids: tuple[str, ...]) -> tuple[MofAgentRecord, ...]:
    missing = [sample_id for sample_id in probe_ids if sample_id not in records]
    if missing:
        raise ValueError(f"Probe IDs are absent from manifest: {missing[:5]}")
    selected = tuple(records[sample_id] for sample_id in probe_ids)
    non_train = [record.sample_id for record in selected if record.split != "train"]
    if non_train:
        raise ValueError(f"Probe contains protected non-train IDs: {non_train[:5]}")
    return selected


def _load_features(records: tuple[MofAgentRecord, ...], directory: Path) -> np.ndarray:
    features = []
    missing = []
    for record in records:
        path = directory / f"{record.sample_id}.npy"
        if not path.exists():
            missing.append(record.sample_id)
            continue
        features.append(np.load(path).reshape(-1))
    if missing:
        raise ValueError(f"Probe feature files are missing from {directory}: {missing[:5]}")
    width = features[0].shape[0]
    if any(row.shape[0] != width for row in features):
        raise ValueError(f"Inconsistent MOF feature widths in {directory}")
    return np.asarray(features, dtype=np.float32)


def _pcc(actual: np.ndarray, predicted: np.ndarray) -> float | None:
    if np.std(actual) == 0.0 or np.std(predicted) == 0.0:
        return None
    return float(np.corrcoef(actual, predicted)[0, 1])


def _summarize_metrics(rows: list[dict[str, Any]]) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    for metric in ("mae", "pcc", "rmse", "r2_standard"):
        values = [float(row[metric]) for row in rows if row[metric] is not None and math.isfinite(float(row[metric]))]
        result[f"mean_{metric}"] = fmean(values) if values else None
        result[f"std_{metric}"] = pstdev(values) if values else None
    return result


def _repeat_summaries(rows: list[dict[str, Any]]) -> list[dict[str, float | int | None]]:
    summaries = []
    for repeat in sorted({int(row["repeat"]) for row in rows}):
        repeat_rows = [row for row in rows if int(row["repeat"]) == repeat]
        summary: dict[str, float | int | None] = {"repeat": repeat}
        for metric in ("mae", "pcc", "rmse", "r2_standard"):
            values = [float(row[metric]) for row in repeat_rows if row[metric] is not None and math.isfinite(float(row[metric]))]
            summary[f"mean_{metric}"] = fmean(values) if values else None
        summaries.append(summary)
    return summaries


def _write_predictions(rows: list[dict[str, Any]], destination: Path) -> None:
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("repeat", "fold", "sample_id", "true", "pred"))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a train-only MOF probe GBT comparison")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--probe-ids", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--topology", required=True)
    parser.add_argument("--property", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--n-estimators", type=int, default=10_000)
    parser.add_argument("--max-depth", type=int, default=7)
    parser.add_argument("--learning-rate", type=float, default=0.005)
    args = parser.parse_args()
    report = run_probe_gbt(
        manifest_path=args.manifest,
        probe_id_path=args.probe_ids,
        feature_dir=args.feature_dir,
        topology=args.topology,
        property_name=args.property,
        output_dir=args.output_dir,
        folds=args.folds,
        repeats=args.repeats,
        seed=args.seed,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
    )
    print(json.dumps(report["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
