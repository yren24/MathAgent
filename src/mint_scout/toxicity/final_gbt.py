from __future__ import annotations

import argparse
import itertools
import json
import math
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from mint_scout.config import load_yaml
from mint_scout.evaluation.metrics import get_metric, mae, pcc, pcc2, r2, rmse
from mint_scout.invariants.manifest import stable_hash
from mint_scout.models.gbt import GBTConfig, fit_predict_gbt
from mint_scout.search.cost import measure_cost
from mint_scout.toxicity.selection_contract import ToxicitySelectionContract


METHOD_REPORT_SCHEMA = "mint-agent.toxicity-final-method.v1"
FINAL_REPORT_SCHEMA = "mint-agent.toxicity-final-evaluation.v1"


def run_final_method_task(
    *,
    plan_path: str | Path,
    gbt_config_path: str | Path,
    method_index: int,
    output_dir: str | Path,
) -> dict[str, Any]:
    plan = _load_object(plan_path, "toxicity final plan")
    selected = tuple(str(value).upper() for value in plan.get("selected_subset", ()))
    if method_index < 0:
        raise ValueError("method_index must be non-negative")
    if method_index >= len(selected):
        return {
            "status": "SKIPPED",
            "method_index": method_index,
            "reason": "method_index_outside_selected_subset",
            "selected_method_count": len(selected),
        }
    method = selected[method_index]
    split_tasks = _tasks_for_method(plan, method)
    train_ids, y_train, x_train = _load_split_matrix(
        Path(str(split_tasks["train"]["manifest_path"])),
        invariant=method,
        split="train",
    )
    test_ids, y_test, x_test = _load_split_matrix(
        Path(str(split_tasks["test"]["manifest_path"])),
        invariant=method,
        split="test",
    )
    overlap = sorted(set(train_ids) & set(test_ids))
    if overlap:
        raise ValueError(f"toxicity train/test sample overlap: {overlap[:10]}")
    gbt = _load_gbt_config(Path(gbt_config_path))
    with measure_cost() as costs:
        prediction = fit_predict_gbt(x_train, y_train, x_test, config=gbt)
    report: dict[str, Any] = {
        "report_schema": METHOD_REPORT_SCHEMA,
        "status": "COMPLETE",
        "evidence_scope": "full_train_fixed_test",
        "plan_hash": str(plan.get("plan_hash") or ""),
        "probe_scout_hash": str(plan.get("probe_scout_hash") or ""),
        "method_index": method_index,
        "invariant": method,
        "representation_hash": str(split_tasks["train"]["representation_hash"]),
        "feature_signature": str(split_tasks["train"]["feature_signature"]),
        "train_feature_manifest": str(split_tasks["train"]["manifest_path"]),
        "test_feature_manifest": str(split_tasks["test"]["manifest_path"]),
        "train_sample_count": len(train_ids),
        "test_sample_count": len(test_ids),
        "train_sample_order_hash": stable_hash(train_ids),
        "test_sample_order_hash": stable_hash(test_ids),
        "test_labels_used_for_selection": False,
        "gbt_config": asdict(gbt),
        "gbt_parameter_hash": gbt.parameter_hash,
        "feature_dimension": int(x_train.shape[1]),
        "cost": asdict(costs[0]),
        "sample_ids": list(test_ids),
        "targets": [float(value) for value in y_test],
        "predictions": [float(value) for value in prediction],
        "metrics": _metrics(y_test, prediction),
    }
    report["method_report_hash"] = stable_hash(report)
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    output = destination / f"{method}.json"
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {"output": str(output), **report}


def combine_final_reports(
    *,
    plan_path: str | Path,
    method_report_dir: str | Path,
    n_bootstrap: int = 1000,
    bootstrap_seed: int = 2026,
) -> dict[str, Any]:
    if n_bootstrap < 0:
        raise ValueError("n_bootstrap must be non-negative")
    plan = _load_object(plan_path, "toxicity final plan")
    selected = tuple(str(value).upper() for value in plan.get("selected_subset", ()))
    if not selected:
        raise ValueError("toxicity final plan has no selected subset")
    raw_contract = plan.get("selection_contract")
    selection = (
        ToxicitySelectionContract.from_mapping(raw_contract)
        if isinstance(raw_contract, Mapping)
        else ToxicitySelectionContract()
    )
    if set(selected) - set(selection.requested_invariants):
        raise ValueError("toxicity final subset includes an unrequested method")
    if plan.get("primary_metric") is not None and plan["primary_metric"] != selection.primary_metric:
        raise ValueError("toxicity final plan metric differs from selection contract")
    report_root = Path(method_report_dir).expanduser().resolve()
    reports: dict[str, dict[str, Any]] = {}
    sample_ids: tuple[str, ...] | None = None
    targets: np.ndarray | None = None
    predictions: dict[str, np.ndarray] = {}
    for method in selected:
        report = _load_object(report_root / f"{method}.json", f"toxicity {method} report")
        if report.get("status") != "COMPLETE":
            raise ValueError(f"toxicity final {method} report is incomplete")
        if report.get("plan_hash") != plan.get("plan_hash"):
            raise ValueError(f"toxicity final {method} report uses another plan")
        current_ids = tuple(str(value) for value in report["sample_ids"])
        current_targets = np.asarray(report["targets"], dtype=float)
        if sample_ids is None:
            sample_ids = current_ids
            targets = current_targets
        elif current_ids != sample_ids or not np.array_equal(current_targets, targets):
            raise ValueError("toxicity final method reports do not share test order")
        predictions[method] = np.asarray(report["predictions"], dtype=float)
        reports[method] = report
    assert sample_ids is not None and targets is not None

    subset_results = []
    for size in range(1, len(selected) + 1):
        for subset in itertools.combinations(selected, size):
            prediction = np.mean([predictions[name] for name in subset], axis=0)
            subset_results.append(
                {
                    "invariants": list(subset),
                    "aggregation": "arithmetic_mean_prediction",
                    "metrics": _metrics(targets, prediction),
                }
            )
    final_prediction = np.mean([predictions[name] for name in selected], axis=0)
    final_metrics = _metrics(targets, final_prediction)
    final_target_met = (
        bool(
            math.isfinite(final_metrics[selection.primary_metric])
            and final_metrics[selection.primary_metric] >= selection.user_target
        )
        if selection.user_target is not None else None
    )
    bootstrap = _bootstrap_pcc2_interval(
        targets,
        final_prediction,
        n_bootstrap=n_bootstrap,
        seed=bootstrap_seed,
    )
    bootstrap_primary = (
        bootstrap if selection.primary_metric == "PCC2" else _bootstrap_metric_interval(
            targets,
            final_prediction,
            metric_name=selection.primary_metric,
            n_bootstrap=n_bootstrap,
            seed=bootstrap_seed,
        )
    )
    result: dict[str, Any] = {
        "report_schema": FINAL_REPORT_SCHEMA,
        "status": "TARGET_NOT_REACHED" if final_target_met is False else "COMPLETE",
        "evidence_scope": "full_train_fixed_test",
        "protocol": {
            "selection_source": "train_probe_oof",
            "selection_frozen_before_test_scoring": True,
            "test_used_for_selection": False,
            "test_query_count": 1,
            "full_train_cross_validation_used": False,
        },
        "plan_hash": str(plan.get("plan_hash") or ""),
        "probe_scout_hash": str(plan.get("probe_scout_hash") or ""),
        "selected_subset": list(selected),
        "requested_invariants": list(selection.requested_invariants),
        "selection_policy": plan.get("selection_policy"),
        "selection_objective": selection.selection_objective,
        "user_target": selection.user_target,
        "probe_target_met": plan.get("probe_target_met"),
        "final_target_met": final_target_met,
        "aggregation": "arithmetic_mean_prediction",
        "primary_metric": selection.primary_metric,
        "secondary_metrics": [
            metric for metric in ("PCC2", "PCC", "R2", "RMSE", "MAE")
            if metric != selection.primary_metric
        ],
        "train_sample_count": int(reports[selected[0]]["train_sample_count"]),
        "test_sample_count": len(sample_ids),
        "gbt_config": reports[selected[0]]["gbt_config"],
        "gbt_parameter_hash": reports[selected[0]]["gbt_parameter_hash"],
        "method_report_hashes": {
            method: reports[method]["method_report_hash"] for method in selected
        },
        "method_costs": {method: reports[method]["cost"] for method in selected},
        "subset_results": subset_results,
        "final_test": {
            "metrics": final_metrics,
            "bootstrap_primary": {
                "metric": selection.primary_metric,
                **bootstrap_primary,
            },
            "bootstrap_pcc2": bootstrap,
            "predictions": [
                {
                    "sample_id": sample_id,
                    "target": float(target),
                    "prediction": float(prediction),
                    "absolute_error": abs(float(target) - float(prediction)),
                }
                for sample_id, target, prediction in zip(
                    sample_ids, targets, final_prediction
                )
            ],
        },
    }
    result["evaluation_hash"] = stable_hash(result)
    return result


def _tasks_for_method(
    plan: Mapping[str, Any], method: str
) -> dict[str, Mapping[str, Any]]:
    matches = {
        str(task.get("split")): task
        for task in plan.get("tasks", ())
        if isinstance(task, Mapping)
        and str(task.get("invariant") or "").upper() == method
    }
    if set(matches) != {"train", "test"}:
        raise ValueError(f"toxicity final plan lacks train/test tasks for {method}")
    if matches["train"]["feature_signature"] != matches["test"]["feature_signature"]:
        raise ValueError(f"toxicity final train/test signatures differ for {method}")
    return matches


def _load_split_matrix(
    manifest_path: Path,
    *,
    invariant: str,
    split: str,
) -> tuple[tuple[str, ...], np.ndarray, np.ndarray]:
    rows = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"toxicity {invariant} {split} feature manifest is empty")
    ids: list[str] = []
    targets: list[float] = []
    arrays: list[np.ndarray] = []
    for row in rows:
        sample_id = str(row["sample_id"])
        if str(row.get("invariant") or "").upper() != invariant:
            raise ValueError(f"toxicity feature invariant mismatch for {sample_id}")
        if str(row.get("split") or "").lower() != split:
            raise ValueError(f"toxicity feature split mismatch for {sample_id}")
        if row.get("status") not in {"computed", "cached"}:
            raise ValueError(f"toxicity feature generation failed for {sample_id}")
        array = np.load(Path(str(row["output_path"])), allow_pickle=False)
        if not np.isfinite(array).all():
            raise ValueError(f"toxicity feature is nonfinite for {sample_id}")
        ids.append(sample_id)
        targets.append(float(row["label"]))
        arrays.append(np.asarray(array, dtype=np.float32).reshape(-1))
    if len(set(ids)) != len(ids):
        raise ValueError(f"toxicity {invariant} {split} manifest repeats sample IDs")
    try:
        features = np.stack(arrays)
    except ValueError as exc:
        raise ValueError(
            f"toxicity {invariant} {split} features do not share one shape"
        ) from exc
    return tuple(ids), np.asarray(targets, dtype=float), features


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "PCC2": pcc2(y_true, y_pred),
        "PCC": pcc(y_true, y_pred),
        "R2": r2(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAE": mae(y_true, y_pred),
    }


def _bootstrap_pcc2_interval(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    n_bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    return _bootstrap_metric_interval(
        y_true, y_pred, metric_name="PCC2", n_bootstrap=n_bootstrap, seed=seed
    )


def _bootstrap_metric_interval(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    metric_name: str,
    n_bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    if n_bootstrap == 0:
        return {
            "n_bootstrap": 0,
            "seed": seed,
            "confidence": 0.95,
            "lower": None,
            "upper": None,
            "valid_replicates": 0,
        }
    metric = get_metric(metric_name)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_bootstrap):
        indices = rng.integers(0, len(y_true), size=len(y_true))
        value = metric(y_true[indices], y_pred[indices])
        if math.isfinite(value):
            values.append(value)
    if not values:
        raise ValueError(f"toxicity final {metric_name} bootstrap produced no finite replicates")
    return {
        "n_bootstrap": n_bootstrap,
        "seed": seed,
        "confidence": 0.95,
        "lower": float(np.quantile(values, 0.025)),
        "upper": float(np.quantile(values, 0.975)),
        "valid_replicates": len(values),
    }


def _load_gbt_config(path: Path) -> GBTConfig:
    raw = load_yaml(path)
    allowed = {item.name for item in fields(GBTConfig)}
    return GBTConfig(**{key: value for key, value in raw.items() if key in allowed})


def _load_object(path: str | Path, name: str) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain a JSON object")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train and combine frozen toxicity final GBT models."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    method = subparsers.add_parser("method")
    method.add_argument("--plan", type=Path, required=True)
    method.add_argument("--gbt-config", type=Path, required=True)
    method.add_argument("--method-index", type=int, required=True)
    method.add_argument("--output-dir", type=Path, required=True)
    combine = subparsers.add_parser("combine")
    combine.add_argument("--plan", type=Path, required=True)
    combine.add_argument("--method-report-dir", type=Path, required=True)
    combine.add_argument("--n-bootstrap", type=int, default=1000)
    combine.add_argument("--bootstrap-seed", type=int, default=2026)
    combine.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.command == "method":
        report = run_final_method_task(
            plan_path=args.plan,
            gbt_config_path=args.gbt_config,
            method_index=args.method_index,
            output_dir=args.output_dir,
        )
        print(json.dumps(report, sort_keys=True))
        return 0
    report = combine_final_reports(
        plan_path=args.plan,
        method_report_dir=args.method_report_dir,
        n_bootstrap=args.n_bootstrap,
        bootstrap_seed=args.bootstrap_seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    metrics = report["final_test"]["metrics"]
    print(
        f"selected={'+'.join(report['selected_subset'])} "
        f"test_pcc2={metrics['PCC2']:.6f} test_pcc={metrics['PCC']:.6f} "
        f"test_r2={metrics['R2']:.6f} output={args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
