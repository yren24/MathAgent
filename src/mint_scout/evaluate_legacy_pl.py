from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Iterable

import numpy as np

from mint_scout.config import load_yaml
from mint_scout.data.casf_index import CasfRecord, load_casf_records
from mint_scout.evaluation.metrics import mae, pcc, r2, rmse
from mint_scout.execution.artifacts import ScoutExecutionArtifact
from mint_scout.invariants.manifest import stable_hash
from mint_scout.models.gbt import GBTConfig, fit_predict_gbt, run_oof_gbt


LEGACY_PL_SHAPE = (30, 40, 8)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate the frozen legacy PL representation.")
    parser.add_argument("--task-config", type=Path, required=True)
    parser.add_argument("--feature-folder", type=Path, required=True)
    parser.add_argument("--mode", choices=("train-cv", "historical-fixed-test"), required=True)
    parser.add_argument("--scout-execution", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-estimators", type=int, default=4000)
    parser.add_argument("--n-runs", type=int, default=10)
    args = parser.parse_args(argv)

    task = load_yaml(args.task_config)
    records = _load_records(task)
    gbt = replace(GBTConfig(), n_estimators=args.n_estimators, n_runs=args.n_runs)
    if args.mode == "train-cv":
        if args.scout_execution is None:
            raise ValueError("--scout-execution is required for train-cv")
        scout = ScoutExecutionArtifact.read(args.scout_execution)
        by_id = {record.pdb_id: record for record in records}
        selected = tuple(by_id[sample_id] for sample_id in scout.modeling_sample_ids)
        features = load_feature_matrix(selected, args.feature_folder)
        targets = np.asarray([record.label for record in selected], dtype=float)
        result = run_oof_gbt(
            features,
            targets,
            scout.modeling_sample_ids,
            dict(scout.full_fold_assignment),
            config=gbt,
        )
        payload = _report(
            mode=args.mode,
            records=selected,
            y_true=targets,
            y_pred=result.y_pred,
            gbt=gbt,
            feature_folder=args.feature_folder,
            fold_assignment=dict(scout.full_fold_assignment),
        )
        payload["cost"] = asdict(result.cost)
        payload["comparison_scope"] = "same frozen train-only folds as the adaptive PL Phase 4 run"
    else:
        train = tuple(record for record in records if record.split == "train")
        test = tuple(record for record in records if record.split == "test")
        x_train = load_feature_matrix(train, args.feature_folder)
        x_test = load_feature_matrix(test, args.feature_folder)
        y_train = np.asarray([record.label for record in train], dtype=float)
        y_test = np.asarray([record.label for record in test], dtype=float)
        prediction = fit_predict_gbt(x_train, y_train, x_test, config=gbt)
        payload = _report(
            mode=args.mode,
            records=test,
            y_true=y_test,
            y_pred=prediction,
            gbt=gbt,
            feature_folder=args.feature_folder,
            fold_assignment=None,
        )
        payload["train_sample_count"] = len(train)
        payload["comparison_scope"] = "historical CASF2016 split parity only; not parameter selection"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"mode={args.mode} samples={payload['evaluation_sample_count']} "
        f"pcc={payload['metrics']['PCC']:.6f} output={args.output}"
    )
    return 0


def load_feature_matrix(
    records: Iterable[CasfRecord],
    feature_folder: Path,
    expected_shape: tuple[int, ...] = LEGACY_PL_SHAPE,
) -> np.ndarray:
    rows = []
    for record in records:
        path = feature_folder / f"{record.pdb_id}.npy"
        if not path.exists():
            raise FileNotFoundError(f"Missing legacy PL feature: {path}")
        array = np.load(path, allow_pickle=False)
        if tuple(array.shape) != expected_shape:
            raise ValueError(f"{record.pdb_id}: expected legacy PL shape {expected_shape}, got {array.shape}")
        if not np.isfinite(array).all():
            raise ValueError(f"{record.pdb_id}: legacy PL feature contains NaN or Inf")
        rows.append(np.asarray(array, dtype=np.float32).reshape(-1))
    if not rows:
        raise ValueError("No records supplied for legacy PL evaluation")
    return np.stack(rows)


def _load_records(task: dict) -> tuple[CasfRecord, ...]:
    casf = task.get("data_audit", {}).get("casf", {})
    return load_casf_records(
        index_root=Path(casf["index_root"]),
        structures_root=Path(casf["structures_root"]),
        year=int(casf.get("year", 2016)),
        protein_template=str(casf.get("protein_template", "{pdb}_pocket.pdb")),
        ligand_template=str(casf.get("ligand_template", "{pdb}_ligand.mol2")),
    )


def _report(*, mode, records, y_true, y_pred, gbt, feature_folder, fold_assignment):
    sample_ids = tuple(record.pdb_id for record in records)
    metrics = {
        "PCC": pcc(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAE": mae(y_true, y_pred),
        "R2": r2(y_true, y_pred),
    }
    payload = {
        "report_schema": "mint-agent.legacy-pl-evaluation.v1",
        "representation": {
            "name": "legacy_pl_fil15_step05_element40",
            "feature_shape": list(LEGACY_PL_SHAPE),
            "feature_dimension": int(np.prod(LEGACY_PL_SHAPE)),
            "element_pair_count": 40,
            "filtration_start_angstrom": 0.0,
            "filtration_stop_angstrom": 14.5,
            "grid_max_exclusive_angstrom": 15.0,
            "filtration_step_angstrom": 0.5,
            "feature_folder": str(feature_folder),
        },
        "mode": mode,
        "evaluation_sample_count": len(sample_ids),
        "evaluation_sample_ids": list(sample_ids),
        "fold_assignment": fold_assignment,
        "gbt": asdict(gbt),
        "gbt_parameter_hash": gbt.parameter_hash,
        "metrics": metrics,
        "legacy_scaled_rmse": metrics["RMSE"] * 1.36,
        "predictions": [
            {"sample_id": sample_id, "target": float(target), "prediction": float(prediction)}
            for sample_id, target, prediction in zip(sample_ids, y_true, y_pred)
        ],
    }
    payload["evaluation_hash"] = stable_hash(
        {
            "representation": payload["representation"],
            "mode": mode,
            "sample_ids": sample_ids,
            "fold_assignment": fold_assignment,
            "gbt_parameter_hash": gbt.parameter_hash,
            "metrics": metrics,
        }
    )
    return payload


if __name__ == "__main__":
    raise SystemExit(main())
