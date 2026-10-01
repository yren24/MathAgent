from __future__ import annotations

import argparse
import json
from dataclasses import asdict, fields, replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from mint_scout.compute_feature import _tool_config_from_task
from mint_scout.compute_feature_batch import _selected_records
from mint_scout.config import load_yaml
from mint_scout.evaluation.ensemble import mean_aggregate
from mint_scout.evaluation.metrics import get_metric
from mint_scout.execution import ExecutionConfig, ProgressiveExecutionEngine, ScoutExecutionArtifact
from mint_scout.execution.provider import PLBindCachedFeatureProvider
from mint_scout.feature_qc import assert_qc_report_compatible
from mint_scout.invariants.plbind_tools import PLBindFeatureTool
from mint_scout.invariants.registry import get_invariant
from mint_scout.models.gbt import GBTConfig
from mint_scout.representation import RepresentationSpec, make_legacy_casf_representation_spec
from mint_scout.run_casf_scout import _parse_manifest_args
from mint_scout.schemas import EvaluationMode, EvaluationPlan


DEFAULT_PRIORITY_ORDER = "PL"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run deterministic Phase 4 GBT execution on labeled protein-ligand records."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--gbt-config", type=Path, required=True)
    parser.add_argument("--split", choices=("all", "train", "test"), default="train")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sample-id", action="append", default=None)
    parser.add_argument(
        "--priority-order",
        default=None,
        help="Semicolon-separated frozen subsets, e.g. 'PL;PH;PL,PH'.",
    )
    parser.add_argument(
        "--scout-artifact",
        type=Path,
        default=None,
        help="Frozen Scout execution artifact; disables manual priority, target, and fold overrides.",
    )
    parser.add_argument("--representation-spec", type=Path, default=None)
    parser.add_argument("--feature-qc-report", type=Path, default=None)
    parser.add_argument(
        "--feature-manifest",
        action="append",
        default=None,
        metavar="INVARIANT=PATH",
    )
    parser.add_argument("--max-acquisitions", type=int, default=None)
    parser.add_argument("--target-value", type=float, default=None)
    parser.add_argument("--cv-folds", type=int, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--legacy-root", type=Path, default=None)
    parser.add_argument("--pdb-folder", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    if args.offset < 0:
        raise ValueError("--offset must be non-negative")
    if args.limit is not None and args.limit < 2:
        raise ValueError("--limit must be at least 2")

    task_config = load_yaml(args.config)
    gbt_config = _load_gbt_config(args.gbt_config)
    scout_artifact = ScoutExecutionArtifact.read(args.scout_artifact) if args.scout_artifact else None
    _assert_no_scout_overrides(args, scout_artifact)
    priority_order = (
        scout_artifact.frozen_priority_order
        if scout_artifact
        else _parse_priority_order(args.priority_order or DEFAULT_PRIORITY_ORDER)
    )
    records = _selected_records(
        task_config,
        sample_ids=args.sample_id,
        split=args.split,
        offset=args.offset,
        limit=args.limit,
        config_path=args.config,
    )
    if len(records) < 2:
        raise ValueError("Phase 4 CV requires at least two labeled samples")

    cv_folds = (
        len(set(scout_artifact.full_fold_assignment.values()))
        if scout_artifact
        else args.cv_folds or int(task_config.get("cv", {}).get("n_splits", 5))
    )
    if cv_folds < 2:
        raise ValueError("cv_folds must be at least 2")
    if len(records) < cv_folds:
        raise ValueError(f"Selected {len(records)} samples, fewer than cv_folds={cv_folds}")
    sample_ids = tuple(record.pdb_id for record in records)
    if scout_artifact and sample_ids != scout_artifact.modeling_sample_ids:
        raise ValueError("Selected sample ids/order do not match Scout artifact modeling samples")
    target_value = scout_artifact.target_value if scout_artifact else _target_value(task_config, args.target_value)
    representation_spec = (
        RepresentationSpec.read(args.representation_spec)
        if args.representation_spec is not None
        else make_legacy_casf_representation_spec()
    )
    representation_spec.assert_frozen()
    representation_hash = representation_spec.spec_hash
    if scout_artifact and scout_artifact.representation_hash != representation_hash:
        raise ValueError(
            "Execution RepresentationSpec does not match the frozen Scout artifact"
        )
    target_metric = (
        scout_artifact.target_metric
        if scout_artifact
        else str(task_config.get("primary_metric", "pcc")).upper()
    )
    if scout_artifact and scout_artifact.gbt_parameter_hash != gbt_config.parameter_hash:
        raise ValueError("GBT config does not match Scout artifact")
    targets = {record.pdb_id: record.label for record in records}
    manifest_paths = _parse_manifest_args(args.feature_manifest or [])
    if args.feature_qc_report is not None:
        if not manifest_paths:
            raise ValueError("--feature-qc-report requires at least one --feature-manifest")
        qc_report = assert_qc_report_compatible(
            path=args.feature_qc_report,
            representation=representation_spec,
            sample_ids=sample_ids,
            manifest_paths=manifest_paths,
        )
    else:
        if manifest_paths:
            raise ValueError("--feature-manifest requires --feature-qc-report")
        qc_report = None
    if args.max_acquisitions is not None and args.max_acquisitions < 1:
        raise ValueError("--max-acquisitions must be positive")
    plan = EvaluationPlan(
        mode=EvaluationMode.FULL_LABELED_CV,
        modeling_sample_ids=sample_ids,
    )
    common_report = {
        "report_schema": "mint-agent.phase4.casf.v1",
        "dataset_id": task_config.get("task_id"),
        "evidence_scope": (
            "smoke"
            if args.limit is not None
            else "full_train" if args.split == "train" else "unspecified"
        ),
        "run_kind": "engineering_smoke" if args.limit is not None else "full_execution",
        "scientific_interpretation": (
            "Engineering smoke metrics are diagnostic only and must not be used as scientific evidence."
            if args.limit is not None
            else "Full deterministic execution; scientific interpretation still requires dataset-level review."
        ),
        "task_id": task_config.get("task_id"),
        "split": args.split,
        "offset": args.offset,
        "sample_count": len(records),
        "sample_ids": list(sample_ids),
        "targets": targets,
        "evaluation_mode": plan.mode.value,
        "cv_folds": cv_folds,
        "priority_order": [list(subset) for subset in priority_order],
        "priority_source": "scout_artifact" if scout_artifact else "manual_engineering_input",
        "scout_artifact": str(args.scout_artifact) if args.scout_artifact else None,
        "probe_hash": scout_artifact.probe_hash if scout_artifact else None,
        "target_metric": target_metric,
        "target_value": target_value,
        "target_source": scout_artifact.target_source if scout_artifact else "task_or_cli",
        "fold_source": "scout_artifact" if scout_artifact else "execution_generated",
        "representation_hash": representation_hash,
        "representation_spec": representation_spec.to_dict(),
        "representation_spec_source": (
            str(args.representation_spec) if args.representation_spec is not None else "legacy_default"
        ),
        "feature_qc_report": str(args.feature_qc_report) if args.feature_qc_report else None,
        "feature_qc_hash": qc_report.get("qc_hash") if qc_report else None,
        "feature_manifests": {name: str(path) for name, path in manifest_paths.items()},
        "max_acquisitions": args.max_acquisitions,
        "gbt_config": asdict(gbt_config),
        "gbt_parameter_hash": gbt_config.parameter_hash,
    }

    if args.dry_run:
        report = {**common_report, "status": "DRY_RUN", "result": None}
    else:
        tool_config = replace(
            _tool_config_from_task(task_config, args=args),
            representation_mode=representation_spec.mode.value,
        )
        tool = PLBindFeatureTool(tool_config)
        provider = PLBindCachedFeatureProvider(
            tool,
            representation_spec,
            manifest_paths=manifest_paths,
        )
        engine = ProgressiveExecutionEngine(
            provider,
            config=ExecutionConfig(
                target_metric=common_report["target_metric"],
                target_value=target_value,
                cv_folds=cv_folds,
                representation_hash=representation_hash,
                require_shared_fold_assignment=scout_artifact is not None,
                require_precomputed=qc_report is not None,
                max_acquisitions=args.max_acquisitions,
                gbt=gbt_config,
            ),
        )
        if scout_artifact:
            result = engine.run_from_scout(
                artifact=scout_artifact,
                plan=plan,
                targets_by_sample=targets,
            )
        else:
            result = engine.run(
                frozen_priority_order=priority_order,
                plan=plan,
                targets_by_sample=targets,
            )
        report = {
            **common_report,
            "status": result.status.value,
            "result": _result_to_dict(result, targets_by_sample=targets),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"status={report['status']} samples={len(records)} target={target_value} "
        f"output={args.output}"
    )
    return 0


def _load_gbt_config(path: Path) -> GBTConfig:
    raw = load_yaml(path)
    allowed = {item.name for item in fields(GBTConfig)}
    params = {key: value for key, value in raw.items() if key in allowed}
    unknown = sorted(set(raw) - allowed - {"source"})
    if unknown:
        raise ValueError(f"Unknown GBT config fields: {unknown}")
    return GBTConfig(**params)


def _parse_priority_order(value: str) -> tuple[tuple[str, ...], ...]:
    subsets: list[tuple[str, ...]] = []
    for raw_subset in value.split(";"):
        names = tuple(name.strip().upper() for name in raw_subset.split(",") if name.strip())
        if not names:
            continue
        if len(set(names)) != len(names):
            raise ValueError(f"Duplicate invariant within priority subset: {raw_subset!r}")
        for name in names:
            get_invariant(name)
        subsets.append(tuple(sorted(names)))
    if not subsets:
        raise ValueError("--priority-order must contain at least one invariant subset")
    if len(set(subsets)) != len(subsets):
        raise ValueError("--priority-order contains duplicate subsets")
    return tuple(subsets)


def _target_value(config: dict[str, Any], override: float | None) -> float:
    if override is not None:
        return float(override)
    stopping = config.get("stopping", {})
    if not isinstance(stopping, dict) or "target_pcc" not in stopping:
        raise ValueError("stopping.target_pcc or --target-value is required")
    return float(stopping["target_pcc"])


def _assert_no_scout_overrides(
    args: argparse.Namespace,
    artifact: ScoutExecutionArtifact | None,
) -> None:
    if artifact is None:
        return
    overrides = []
    if args.priority_order is not None:
        overrides.append("--priority-order")
    if args.target_value is not None:
        overrides.append("--target-value")
    if args.cv_folds is not None:
        overrides.append("--cv-folds")
    if overrides:
        raise ValueError(
            "Scout artifact freezes priority, target, and folds; remove overrides: " + ", ".join(overrides)
        )


def _result_to_dict(
    result: Any,
    *,
    targets_by_sample: Mapping[str, float],
) -> dict[str, Any]:
    evaluation_ids = tuple(result.evaluation_sample_ids)
    y_true = np.asarray([targets_by_sample[sample_id] for sample_id in evaluation_ids], dtype=float)
    predictions = {
        name: np.asarray(values, dtype=float)
        for name, values in result.evaluation_predictions.items()
    }
    selected_prediction = mean_aggregate(predictions, result.selected_subset)
    metric_names = ("PCC", "RMSE", "MAE", "R2")
    by_invariant_metrics = {
        name: _metrics(y_true, values, metric_names)
        for name, values in sorted(predictions.items())
    }
    selected_metrics = _metrics(y_true, selected_prediction, metric_names)
    fold_metrics = _fold_metrics(
        sample_ids=evaluation_ids,
        y_true=y_true,
        y_pred=selected_prediction,
        fold_assignment=result.evaluation_fold_assignment,
        metric_names=metric_names,
    )
    invariant_prediction_rows = {
        name: {
            sample_id: float(values[index])
            for index, sample_id in enumerate(evaluation_ids)
        }
        for name, values in sorted(predictions.items())
    }
    prediction_field = (
        "external_eval_predictions"
        if result.evaluation_mode == EvaluationMode.EXPLICIT_LABELED_TEST
        else "full_oof_predictions"
    )
    payload = {
        "status": result.status.value,
        "selected_subset": list(result.selected_subset),
        "selected_score": result.selected_score,
        "best_achieved_subset": list(result.best_achieved_subset),
        "best_achieved_score": result.best_achieved_score,
        "acquisition_order": list(result.acquisition_order),
        "acquisition_records": [asdict(record) for record in result.acquisition_records],
        "rounds": [
            {
                "acquired_invariants": list(round_result.acquired_invariants),
                "selected_subset": list(round_result.selected_subset),
                "selected_score": round_result.selected_score,
                "target_reached": round_result.target_reached,
                "subset_scores": [
                    {"subset": list(subset), "score": score}
                    for subset, score in sorted(round_result.subset_scores.items())
                ],
            }
            for round_result in result.rounds
        ],
        "evaluation_sample_ids": list(evaluation_ids),
        "evaluation_fold_assignment": dict(result.evaluation_fold_assignment),
        "evaluation_metrics": {
            "selected_subset": selected_metrics,
            "by_invariant": by_invariant_metrics,
            "selected_subset_by_fold": fold_metrics,
        },
        "selected_consensus_predictions": {
            sample_id: float(selected_prediction[index])
            for index, sample_id in enumerate(evaluation_ids)
        },
        "full_oof_predictions": None,
        "external_eval_predictions": None,
        "inference": None,
    }
    payload[prediction_field] = invariant_prediction_rows
    return payload


def _metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    names: tuple[str, ...],
) -> dict[str, float]:
    return {name: float(get_metric(name)(y_true, y_pred)) for name in names}


def _fold_metrics(
    *,
    sample_ids: tuple[str, ...],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    fold_assignment: Mapping[str, int],
    metric_names: tuple[str, ...],
) -> dict[str, dict[str, float]]:
    if not fold_assignment:
        return {}
    if set(fold_assignment) != set(sample_ids):
        raise ValueError("Evaluation fold assignment does not match evaluation sample IDs")
    reports = {}
    for fold in sorted(set(fold_assignment.values())):
        indices = np.asarray(
            [index for index, sample_id in enumerate(sample_ids) if fold_assignment[sample_id] == fold],
            dtype=int,
        )
        reports[str(fold)] = _metrics(y_true[indices], y_pred[indices], metric_names)
    return reports


if __name__ == "__main__":
    raise SystemExit(main())
