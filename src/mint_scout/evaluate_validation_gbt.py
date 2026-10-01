from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from mint_scout.config import load_yaml
from mint_scout.data.manifest_io import load_configured_protein_ligand_records
from mint_scout.evaluate_external_gbt import _file_sha256, _metrics, _ordered_manifest_args
from mint_scout.evaluate_split_gbt import (
    _assert_disjoint_splits,
    _assert_labeled,
    _assert_qc,
    _features_for_split,
    _load_gbt_config,
    _targets,
)
from mint_scout.evaluation.ensemble import enumerate_nonempty_subsets, mean_aggregate
from mint_scout.evaluation.metrics import get_metric, higher_is_better
from mint_scout.execution.artifacts import ScoutExecutionArtifact
from mint_scout.execution.engine import _select_subset, _target_reached
from mint_scout.invariants.manifest import stable_hash
from mint_scout.models.gbt import fit_predict_gbt
from mint_scout.representation import RepresentationSpec


REPORT_SCHEMA = "mint-agent.validation-gbt-evaluation.v1"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate one frozen progressive acquisition stage on validation."
    )
    parser.add_argument("--task-config", type=Path, required=True)
    parser.add_argument("--representation-spec", type=Path, required=True)
    parser.add_argument("--gbt-config", type=Path, required=True)
    parser.add_argument("--scout-artifact", type=Path, required=True)
    parser.add_argument(
        "--train-feature-manifest", action="append", required=True, metavar="INVARIANT=PATH"
    )
    parser.add_argument(
        "--validation-feature-manifest",
        action="append",
        required=True,
        metavar="INVARIANT=PATH",
    )
    parser.add_argument("--train-feature-qc-report", type=Path, required=True)
    parser.add_argument("--validation-feature-qc-report", type=Path, required=True)
    parser.add_argument("--prior-evaluation-report", type=Path, default=None)
    parser.add_argument("--max-acquisitions", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    report = evaluate_validation_gbt(
        task_config_path=args.task_config,
        representation_path=args.representation_spec,
        gbt_config_path=args.gbt_config,
        scout_artifact_path=args.scout_artifact,
        train_manifest_values=args.train_feature_manifest,
        validation_manifest_values=args.validation_feature_manifest,
        train_qc_report=args.train_feature_qc_report,
        validation_qc_report=args.validation_feature_qc_report,
        max_acquisitions=args.max_acquisitions,
        prior_evaluation_report=args.prior_evaluation_report,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"status={report['status']} selected="
        f"{'+'.join(report['selection']['selected_subset'])} "
        f"score={report['selection']['selected_score']:.6f} output={args.output}"
    )
    return 0


def evaluate_validation_gbt(
    *,
    task_config_path: Path,
    representation_path: Path,
    gbt_config_path: Path,
    scout_artifact_path: Path,
    train_manifest_values: list[str],
    validation_manifest_values: list[str],
    train_qc_report: Path,
    validation_qc_report: Path,
    max_acquisitions: int,
    prior_evaluation_report: Path | None = None,
) -> dict[str, object]:
    if max_acquisitions < 1:
        raise ValueError("max_acquisitions must be positive")
    task = load_yaml(task_config_path)
    representation = RepresentationSpec.read(representation_path)
    representation.assert_frozen()
    gbt_config = _load_gbt_config(gbt_config_path)
    scout = ScoutExecutionArtifact.read(scout_artifact_path)
    if scout.representation_hash != representation.spec_hash:
        raise ValueError("Scout and validation evaluation representations do not match")
    if scout.gbt_parameter_hash != gbt_config.parameter_hash:
        raise ValueError("Scout and validation evaluation GBT parameters do not match")

    acquisition_order = _flatten_priority_order(scout.frozen_priority_order)
    if max_acquisitions > len(acquisition_order):
        raise ValueError("max_acquisitions exceeds the frozen Scout acquisition queue")
    acquired = acquisition_order[:max_acquisitions]
    train_manifests = _ordered_manifest_args(train_manifest_values)
    validation_manifests = _ordered_manifest_args(validation_manifest_values)
    if tuple(train_manifests) != acquired or tuple(validation_manifests) != acquired:
        raise ValueError("Feature manifests must match the frozen acquisition prefix in order")

    preferences = task.get("selection_preferences", {})
    preferences = preferences if isinstance(preferences, dict) else {}
    selection_objective = str(
        preferences.get("selection_objective") or "satisfy_target"
    )
    if selection_objective not in {
        "satisfy_target",
        "maximize_rank1",
        "maximize_top_k",
        "maximize_all",
    }:
        raise ValueError("Unsupported selection objective")
    rank1_size = len(tuple(dict.fromkeys(scout.frozen_priority_order[0])))
    candidate_rank_limit = _candidate_rank_limit(
        selection_objective, preferences, scout.frozen_priority_order
    )
    candidate_pool = _candidate_pool_for_objective(
        selection_objective,
        scout.frozen_priority_order,
        candidate_rank_limit=candidate_rank_limit,
    )
    minimum_acquisitions = _minimum_acquisitions_for_objective(
        selection_objective,
        scout.frozen_priority_order,
        acquisition_order=acquisition_order,
        rank1_size=rank1_size,
        candidate_rank_limit=candidate_rank_limit,
    )
    direct_terminal_maximization = (
        selection_objective in {"maximize_rank1", "maximize_top_k"}
        and max_acquisitions == minimum_acquisitions
    )
    if max_acquisitions == 1 and prior_evaluation_report is not None:
        raise ValueError("The first acquisition stage cannot reuse a prior evaluation")
    if (
        max_acquisitions > 1
        and prior_evaluation_report is None
        and not direct_terminal_maximization
    ):
        raise ValueError(
            "Later acquisition stages require the prior evaluation unless a "
            "terminal maximization stage is being evaluated directly"
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
    if train_ids != scout.modeling_sample_ids:
        raise ValueError("Training samples do not match the frozen Scout modeling samples")
    _assert_qc(
        train_qc_report,
        representation=representation,
        sample_ids=train_ids,
        manifests=train_manifests,
    )
    _assert_qc(
        validation_qc_report,
        representation=representation,
        sample_ids=validation_ids,
        manifests=validation_manifests,
    )
    reused_predictions, prior_sha256 = _load_prior_predictions(
        prior_evaluation_report,
        acquired=acquired,
        train_ids=train_ids,
        validation_ids=validation_ids,
        representation=representation,
        gbt_parameter_hash=gbt_config.parameter_hash,
        scout_artifact_path=scout_artifact_path,
        target_metric=scout.target_metric,
        target_value=scout.target_value,
        selection_objective=selection_objective,
        candidate_rank_limit=candidate_rank_limit,
    )
    newly_fitted = tuple(name for name in acquired if name not in reused_predictions)
    train_features = _features_for_split(
        train_records,
        {name: train_manifests[name] for name in newly_fitted},
        representation,
    )
    validation_features = _features_for_split(
        validation_records,
        {name: validation_manifests[name] for name in newly_fitted},
        representation,
    )
    y_train = _targets(train_records)
    y_validation = _targets(validation_records)

    started = time.perf_counter()
    new_predictions = {
        invariant: fit_predict_gbt(
            train_features[invariant],
            y_train,
            validation_features[invariant],
            config=gbt_config,
        )
        for invariant in newly_fitted
    }
    predictions = {
        invariant: (
            reused_predictions[invariant]
            if invariant in reused_predictions
            else new_predictions[invariant]
        )
        for invariant in acquired
    }
    metric = get_metric(scout.target_metric)
    enumerated_scores = {
        subset: float(metric(y_validation, mean_aggregate(predictions, subset)))
        for subset in enumerate_nonempty_subsets(acquired)
    }
    available_priority = tuple(
        subset
        for subset in scout.frozen_priority_order
        if set(subset).issubset(acquired)
    )
    score_by_members = {
        frozenset(subset): score for subset, score in enumerated_scores.items()
    }
    if {frozenset(subset) for subset in available_priority} != set(score_by_members):
        raise ValueError("Frozen Scout ranking does not cover all available subsets")
    scoring_priority = tuple(
        subset for subset in candidate_pool if set(subset).issubset(acquired)
    )
    if not scoring_priority:
        raise ValueError("No frozen candidate is available at this acquisition stage")
    subset_scores = {
        subset: score_by_members[frozenset(subset)] for subset in scoring_priority
    }
    costs = {
        invariant: _manifest_wall_seconds(train_manifests[invariant])
        + _manifest_wall_seconds(validation_manifests[invariant])
        for invariant in acquired
    }
    is_higher = higher_is_better(scout.target_metric)
    if selection_objective == "satisfy_target":
        selected_subset = _select_subset(
            subset_scores,
            is_higher=is_higher,
            target_value=scout.target_value,
            cost_by_invariant=costs,
        )
    else:
        best_score = (
            max(subset_scores.values())
            if is_higher
            else min(subset_scores.values())
        )
        selected_subset = next(
            subset
            for subset in scoring_priority
            if np.isclose(subset_scores[subset], best_score)
        )
    selected_score = subset_scores[selected_subset]
    reached = _target_reached(selected_score, scout.target_value, is_higher)
    acquisition_exhausted = max_acquisitions >= len(acquisition_order)
    objective_complete = (
        reached or acquisition_exhausted
        if selection_objective == "satisfy_target"
        else max_acquisitions >= minimum_acquisitions
    )
    if selection_objective != "satisfy_target" and objective_complete:
        status = "MAXIMIZATION_COMPLETE"
    elif reached:
        status = "TARGET_REACHED"
    elif not acquisition_exhausted:
        status = "ACQUISITION_LIMIT_REACHED"
    else:
        status = "TARGET_NOT_REACHED"
    selection_hash = stable_hash(
        {
            "acquired_invariants": acquired,
            "gbt_parameter_hash": gbt_config.parameter_hash,
            "representation_hash": representation.spec_hash,
            "scout_artifact_sha256": _file_sha256(scout_artifact_path),
            "prior_evaluation_sha256": prior_sha256,
            "selected_subset": selected_subset,
            "selection_objective": selection_objective,
            "candidate_rank_limit": candidate_rank_limit,
            "subset_scores": [
                {"subset": subset, "score": score}
                for subset, score in sorted(subset_scores.items())
            ],
            "train_ids": train_ids,
            "validation_ids": validation_ids,
        }
    )
    report: dict[str, object] = {
        "report_schema": REPORT_SCHEMA,
        "status": status,
        "dataset_id": str(task.get("task_id") or ""),
        "evidence_scope": "validation",
        "evaluation_mode": "explicit_validation_and_test",
        "representation_hash": representation.spec_hash,
        "selection_hash": selection_hash,
        "scout_artifact": str(scout_artifact_path),
        "scout_probe_hash": scout.probe_hash,
        "gbt_config": asdict(gbt_config),
        "gbt_parameter_hash": gbt_config.parameter_hash,
        "target_metric": scout.target_metric.upper(),
        "target_value": scout.target_value,
        "target_source": scout.target_source,
        "invariants": list(acquired),
        "max_acquisitions": max_acquisitions,
        "minimum_acquisitions_for_objective": minimum_acquisitions,
        "selection_objective": selection_objective,
        "candidate_rank_limit": candidate_rank_limit,
        "candidate_pool": [list(subset) for subset in candidate_pool],
        "objective_complete": objective_complete,
        "acquisition_order": list(acquired),
        "sample_ids": list(validation_ids),
        "sample_count": len(validation_ids),
        "sample_order_hash": stable_hash(validation_ids),
        "training_sample_count": len(train_ids),
        "training_sample_order_hash": stable_hash(train_ids),
        "feature_manifests": {
            "train": {name: str(path) for name, path in train_manifests.items()},
            "validation": {
                name: str(path) for name, path in validation_manifests.items()
            },
        },
        "feature_manifest_sha256": {
            "train": {name: _file_sha256(path) for name, path in train_manifests.items()},
            "validation": {
                name: _file_sha256(path) for name, path in validation_manifests.items()
            },
        },
        "feature_qc_reports": {
            "train": str(train_qc_report),
            "validation": str(validation_qc_report),
        },
        "prediction_reuse": {
            "prior_evaluation_report": (
                str(prior_evaluation_report) if prior_evaluation_report else None
            ),
            "prior_evaluation_sha256": prior_sha256,
            "reused_invariants": list(reused_predictions),
            "newly_fitted_invariants": list(newly_fitted),
        },
        "selection": {
            "selected_subset": list(selected_subset),
            "selected_score": selected_score,
            "target_reached": reached,
            "subset_scores": [
                {"subset": list(subset), "score": score}
                for subset, score in sorted(subset_scores.items())
            ],
            "by_invariant_metrics": {
                name: _metrics(y_validation, values)
                for name, values in predictions.items()
            },
        },
        "validation_predictions": {
            name: {
                sample_id: float(values[index])
                for index, sample_id in enumerate(validation_ids)
            }
            for name, values in predictions.items()
        },
        "fit_predict_elapsed_seconds": time.perf_counter() - started,
    }
    report["evaluation_hash"] = stable_hash(
        {
            "selection_hash": selection_hash,
            "status": status,
            "target_metric": scout.target_metric,
            "target_value": scout.target_value,
        }
    )
    return report


def _flatten_priority_order(priority_order: tuple[tuple[str, ...], ...]) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            str(invariant).upper()
            for subset in priority_order
            for invariant in subset
        )
    )


def _candidate_rank_limit(
    selection_objective: str,
    preferences: Mapping[str, Any],
    priority_order: tuple[tuple[str, ...], ...],
) -> int | None:
    if selection_objective != "maximize_top_k":
        return None
    raw_limit = preferences.get("max_candidate_rank", 5)
    try:
        limit = int(raw_limit)
    except (TypeError, ValueError) as exc:
        raise ValueError("max_candidate_rank must be a positive integer") from exc
    if limit < 1:
        raise ValueError("max_candidate_rank must be a positive integer")
    return min(limit, len(priority_order))


def _candidate_pool_for_objective(
    selection_objective: str,
    priority_order: tuple[tuple[str, ...], ...],
    *,
    candidate_rank_limit: int | None,
) -> tuple[tuple[str, ...], ...]:
    if selection_objective == "maximize_top_k":
        if candidate_rank_limit is None:
            raise ValueError("maximize_top_k requires max_candidate_rank")
        return priority_order[:candidate_rank_limit]
    return priority_order


def _minimum_acquisitions_for_objective(
    selection_objective: str,
    priority_order: tuple[tuple[str, ...], ...],
    *,
    acquisition_order: tuple[str, ...],
    rank1_size: int,
    candidate_rank_limit: int | None,
) -> int:
    if selection_objective == "maximize_rank1":
        return rank1_size
    if selection_objective == "maximize_top_k":
        pool = _candidate_pool_for_objective(
            selection_objective,
            priority_order,
            candidate_rank_limit=candidate_rank_limit,
        )
        return len(_flatten_priority_order(pool))
    if selection_objective == "maximize_all":
        return len(acquisition_order)
    return 1


def _manifest_wall_seconds(path: Path) -> float:
    total = 0.0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        value = row.get("wall_seconds")
        if value is not None and np.isfinite(float(value)) and float(value) >= 0.0:
            total += float(value)
    return total


def _load_prior_predictions(
    path: Path | None,
    *,
    acquired: tuple[str, ...],
    train_ids: tuple[str, ...],
    validation_ids: tuple[str, ...],
    representation: RepresentationSpec,
    gbt_parameter_hash: str,
    scout_artifact_path: Path,
    target_metric: str,
    target_value: float,
    selection_objective: str = "satisfy_target",
    candidate_rank_limit: int | None = None,
) -> tuple[dict[str, np.ndarray], str | None]:
    if path is None:
        return {}, None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("report_schema") != REPORT_SCHEMA:
        raise ValueError("Prior evaluation has an unsupported report schema")
    prior_invariants = tuple(str(value).upper() for value in payload.get("invariants", ()))
    if prior_invariants != acquired[:-1]:
        raise ValueError("Prior evaluation must contain the immediately preceding acquisition prefix")
    if payload.get("status") != "ACQUISITION_LIMIT_REACHED":
        raise ValueError("Prior evaluation is not an unfinished acquisition stage")
    if (payload.get("selection_objective") or "satisfy_target") != selection_objective:
        raise ValueError("Prior evaluation selection objective does not match")
    if payload.get("candidate_rank_limit") != candidate_rank_limit:
        raise ValueError("Prior evaluation candidate rank limit does not match")
    if payload.get("representation_hash") != representation.spec_hash:
        raise ValueError("Prior evaluation representation does not match")
    if payload.get("gbt_parameter_hash") != gbt_parameter_hash:
        raise ValueError("Prior evaluation GBT parameters do not match")
    if payload.get("scout_artifact") != str(scout_artifact_path):
        raise ValueError("Prior evaluation Scout artifact does not match")
    if str(payload.get("target_metric") or "").upper() != target_metric.upper():
        raise ValueError("Prior evaluation target metric does not match")
    try:
        same_target = float(payload.get("target_value")) == float(target_value)
    except (TypeError, ValueError):
        same_target = False
    if not same_target:
        raise ValueError("Prior evaluation target value does not match")
    if payload.get("training_sample_order_hash") != stable_hash(train_ids):
        raise ValueError("Prior evaluation training sample order does not match")
    if tuple(str(value) for value in payload.get("sample_ids", ())) != validation_ids:
        raise ValueError("Prior evaluation validation sample order does not match")

    raw_predictions = payload.get("validation_predictions")
    if not isinstance(raw_predictions, dict) or set(raw_predictions) != set(prior_invariants):
        raise ValueError("Prior evaluation predictions do not match its invariant prefix")
    predictions: dict[str, np.ndarray] = {}
    for invariant in prior_invariants:
        values = raw_predictions.get(invariant)
        if not isinstance(values, dict) or set(values) != set(validation_ids):
            raise ValueError(f"Prior {invariant} validation predictions are incomplete")
        prediction = np.asarray([values[sample_id] for sample_id in validation_ids], dtype=float)
        if not np.all(np.isfinite(prediction)):
            raise ValueError(f"Prior {invariant} validation predictions are non-finite")
        predictions[invariant] = prediction
    return predictions, _file_sha256(path)


if __name__ == "__main__":
    raise SystemExit(main())
