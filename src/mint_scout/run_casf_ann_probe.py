from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from mint_scout.compute_feature_batch import _selected_records
from mint_scout.config import load_yaml
from mint_scout.evaluation.metrics import get_metric, higher_is_better, pcc
from mint_scout.feature_qc import assert_qc_report_compatible
from mint_scout.invariants.manifest import stable_hash
from mint_scout.models.ann import ANNConfig, ANNOOFPredictionArtifact, run_oof_ann
from mint_scout.representation import RepresentationSpec
from mint_scout.run_casf_scout import (
    _load_json_object,
    _load_manifest_features,
    _parse_manifest_args,
    _selection_from_payload,
)
from mint_scout.scout.sampling import stratified_bootstrap_indices


@dataclass(frozen=True)
class ProbeANNConfig:
    ann: ANNConfig
    baseline: tuple[str, ...]
    subsets: tuple[tuple[str, ...], ...]
    metrics: tuple[str, ...]
    bootstrap_replicates: int
    bootstrap_ci: float
    bootstrap_seed: int

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "ProbeANNConfig":
        model = dict(raw.get("model", {}))
        if model.get("family", "fixed_probe_ann") != "fixed_probe_ann":
            raise ValueError("ANN probe diagnostic requires model.family=fixed_probe_ann")
        if bool(model.get("hpo", False)):
            raise ValueError("ANN probe diagnostic forbids hyperparameter optimization")
        params = dict(model.get("params", {}))
        if "hidden_layers" in params:
            params["hidden_layers"] = tuple(params["hidden_layers"])
        if "random_seeds" in params:
            params["random_seeds"] = tuple(params["random_seeds"])
        subsets_raw = raw.get("subsets", {})
        baseline = _normalize_subset(subsets_raw.get("baseline", ("PL",)))
        candidates = tuple(_normalize_subset(value) for value in subsets_raw.get("candidates", (baseline,)))
        subsets = tuple(dict.fromkeys((baseline,) + candidates))
        bootstrap = dict(raw.get("bootstrap", {}))
        metrics = tuple(str(value).upper() for value in raw.get("metrics", ("PCC", "RMSE", "MAE", "R2")))
        if not metrics:
            raise ValueError("metrics cannot be empty")
        replicates = int(bootstrap.get("replicates", 1000))
        ci = float(bootstrap.get("ci_level", 0.95))
        if replicates < 1 or not 0.0 < ci < 1.0:
            raise ValueError("bootstrap replicates and ci_level are invalid")
        return cls(
            ann=ANNConfig(**params),
            baseline=baseline,
            subsets=subsets,
            metrics=metrics,
            bootstrap_replicates=replicates,
            bootstrap_ci=ci,
            bootstrap_seed=int(bootstrap.get("random_seed", 2026)),
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a fixed ANN complementarity diagnostic on the frozen CASF probe."
    )
    parser.add_argument("--task-config", type=Path, required=True)
    parser.add_argument("--ann-config", type=Path, required=True)
    parser.add_argument("--representation-spec", type=Path, required=True)
    parser.add_argument("--probe-selection", type=Path, required=True)
    parser.add_argument("--feature-manifest", action="append", required=True, metavar="INVARIANT=PATH")
    parser.add_argument("--feature-qc-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    task = load_yaml(args.task_config)
    config_mapping = load_yaml(args.ann_config)
    config = ProbeANNConfig.from_mapping(config_mapping)
    representation = RepresentationSpec.read(args.representation_spec)
    representation.assert_frozen()
    probe_payload = _load_json_object(args.probe_selection)
    if probe_payload.get("representation_hash") != representation.spec_hash:
        raise ValueError("Probe selection and RepresentationSpec hashes do not match")
    selection = _selection_from_payload(probe_payload)

    records = _selected_records(
        task,
        sample_ids=None,
        split="train",
        offset=0,
        limit=None,
        config_path=args.task_config,
    )
    modeling_ids = tuple(record.pdb_id for record in records)
    if tuple(probe_payload.get("modeling_sample_ids", ())) != modeling_ids:
        raise ValueError("Probe selection modeling IDs do not match the CASF training pool")
    target_by_id = {record.pdb_id: record.label for record in records}
    targets = np.asarray([target_by_id[sample_id] for sample_id in selection.sample_ids], dtype=float)
    fold_assignment = {
        sample_id: int(probe_payload["modeling_fold_assignment"][sample_id])
        for sample_id in selection.sample_ids
    }

    manifest_paths = _parse_manifest_args(args.feature_manifest)
    assert_qc_report_compatible(
        path=args.feature_qc_report,
        representation=representation,
        sample_ids=selection.sample_ids,
        manifest_paths=manifest_paths,
    )
    required_invariants = tuple(sorted({name for subset in config.subsets for name in subset}))
    missing = set(required_invariants) - set(manifest_paths)
    if missing:
        raise ValueError(f"Missing feature manifests required by ANN subsets: {sorted(missing)}")
    features = {
        invariant: _load_manifest_features(
            path=manifest_paths[invariant],
            invariant=invariant,
            sample_ids=selection.sample_ids,
            representation=representation,
        )
        for invariant in required_invariants
    }

    artifacts: dict[tuple[str, ...], ANNOOFPredictionArtifact] = {}
    for subset in config.subsets:
        blocks = [features[name].reshape(len(selection.sample_ids), -1) for name in subset]
        artifacts[subset] = run_oof_ann(
            np.concatenate(blocks, axis=1),
            targets,
            selection.sample_ids,
            fold_assignment,
            config=config.ann,
        )

    report = _build_report(
        task=task,
        raw_config=config_mapping,
        config=config,
        representation=representation,
        probe_payload=probe_payload,
        selection=selection,
        targets=targets,
        artifacts=artifacts,
        manifest_paths=manifest_paths,
        feature_qc_report=args.feature_qc_report,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    scores = ", ".join(
        f"{'+'.join(subset)}={report['subsets']['+'.join(subset)]['metrics']['PCC']:.6f}"
        for subset in config.subsets
    )
    print(f"probe_ann diagnostic_only=true samples={len(selection.sample_ids)} {scores} output={args.output}")
    return 0


def _build_report(
    *,
    task: Mapping[str, Any],
    raw_config: Mapping[str, Any],
    config: ProbeANNConfig,
    representation: RepresentationSpec,
    probe_payload: Mapping[str, Any],
    selection,
    targets: np.ndarray,
    artifacts: Mapping[tuple[str, ...], ANNOOFPredictionArtifact],
    manifest_paths: Mapping[str, Path],
    feature_qc_report: Path,
) -> dict[str, Any]:
    baseline = artifacts[config.baseline]
    strata = np.asarray([selection.target_bins[sample_id] for sample_id in selection.sample_ids], dtype=int)
    bootstrap_indices = stratified_bootstrap_indices(
        strata,
        n_replicates=config.bootstrap_replicates,
        seed=config.bootstrap_seed,
    )
    subset_reports: dict[str, Any] = {}
    for subset, artifact in artifacts.items():
        subset_id = "+".join(subset)
        subset_reports[subset_id] = {
            "invariants": list(subset),
            "fusion": "feature_concatenation",
            "input_dimension": artifact.input_dimension,
            "parameter_count": artifact.parameter_count,
            "device": artifact.device,
            "device_name": artifact.device_name,
            "torch_version": artifact.torch_version,
            "cuda_version": artifact.cuda_version,
            "fit_seconds": artifact.fit_seconds,
            "oof_predictions": {
                sample_id: float(artifact.y_pred[index])
                for index, sample_id in enumerate(selection.sample_ids)
            },
            **_prediction_summary(
                targets=targets,
                prediction=artifact.y_pred,
                baseline_prediction=baseline.y_pred,
                sample_ids=selection.sample_ids,
                fold_ids=artifact.fold_ids,
                metrics=config.metrics,
                bootstrap_indices=bootstrap_indices,
                bootstrap_ci=config.bootstrap_ci,
            ),
        }
    singleton_artifacts = {
        subset[0]: artifact for subset, artifact in artifacts.items() if len(subset) == 1
    }
    consensus_reports: dict[str, Any] = {}
    singleton_names = tuple(sorted(singleton_artifacts))
    for subset_size in range(2, len(singleton_names) + 1):
        for names in combinations(singleton_names, subset_size):
            prediction = np.mean(
                np.asarray([singleton_artifacts[name].y_pred for name in names]),
                axis=0,
            )
            subset_id = "+".join(names)
            consensus_reports[subset_id] = {
                "invariants": list(names),
                "fusion": "arithmetic_mean_of_singleton_ann_predictions",
                "oof_predictions": {
                    sample_id: float(prediction[index])
                    for index, sample_id in enumerate(selection.sample_ids)
                },
                **_prediction_summary(
                    targets=targets,
                    prediction=prediction,
                    baseline_prediction=baseline.y_pred,
                    sample_ids=selection.sample_ids,
                    fold_ids=baseline.fold_ids,
                    metrics=config.metrics,
                    bootstrap_indices=bootstrap_indices,
                    bootstrap_ci=config.bootstrap_ci,
                ),
            }
    for subset, artifact in artifacts.items():
        if len(subset) > 1 and all((name,) in artifacts for name in subset):
            subset_reports["+".join(subset)]["paired_gain_vs_best_singleton"] = (
                _paired_gain_vs_best_singleton(
                    targets=targets,
                    candidate_prediction=artifact.y_pred,
                    singleton_predictions={name: artifacts[(name,)].y_pred for name in subset},
                    metrics=config.metrics,
                    bootstrap_indices=bootstrap_indices,
                    bootstrap_ci=config.bootstrap_ci,
                )
            )
    for subset_id, report in consensus_reports.items():
        names = tuple(report["invariants"])
        report["paired_gain_vs_best_singleton"] = _paired_gain_vs_best_singleton(
            targets=targets,
            candidate_prediction=np.asarray(
                [report["oof_predictions"][sample_id] for sample_id in selection.sample_ids],
                dtype=float,
            ),
            singleton_predictions={name: artifacts[(name,)].y_pred for name in names},
            metrics=config.metrics,
            bootstrap_indices=bootstrap_indices,
            bootstrap_ci=config.bootstrap_ci,
        )
    prediction_hashes = {
        subset_id: stable_hash(report["oof_predictions"])
        for subset_id, report in subset_reports.items()
    }
    return {
        "report_schema": "mint-agent.probe-ann-diagnostic.v1",
        "diagnostic_only": True,
        "scientific_scope": (
            "Fixed ANN comparison on the frozen probe; does not alter the V1 fixed-GBT Scout ranking, "
            "target, representation, or held-out test boundary."
        ),
        "task_id": task.get("task_id"),
        "representation_hash": representation.spec_hash,
        "selection_hash": probe_payload.get("selection_hash"),
        "probe_sample_ids": list(selection.sample_ids),
        "probe_size": len(selection.sample_ids),
        "fold_assignment": dict(baseline.fold_ids),
        "ann_config": asdict(config.ann),
        "ann_parameter_hash": config.ann.parameter_hash,
        "diagnostic_config_hash": stable_hash(raw_config),
        "baseline_subset": list(config.baseline),
        "feature_manifests": {name: str(path) for name, path in manifest_paths.items()},
        "feature_qc_report": str(feature_qc_report),
        "bootstrap": {
            "replicates": config.bootstrap_replicates,
            "ci_level": config.bootstrap_ci,
            "random_seed": config.bootstrap_seed,
        },
        "subsets": subset_reports,
        "prediction_consensus": consensus_reports,
        "prediction_hashes": prediction_hashes,
        "test_set_used": False,
    }


def _prediction_summary(
    *,
    targets: np.ndarray,
    prediction: np.ndarray,
    baseline_prediction: np.ndarray,
    sample_ids: Sequence[str],
    fold_ids: Mapping[str, int],
    metrics: Sequence[str],
    bootstrap_indices: Sequence[np.ndarray],
    bootstrap_ci: float,
) -> dict[str, Any]:
    return {
        "metrics": {
            name: float(get_metric(name)(targets, prediction))
            for name in metrics
        },
        "fold_metrics": {
            str(fold): {
                name: float(get_metric(name)(targets[indices], prediction[indices]))
                for name in metrics
            }
            for fold, indices in _fold_indices(sample_ids, fold_ids).items()
        },
        "prediction_correlation_with_baseline": float(pcc(baseline_prediction, prediction)),
        "residual_correlation_with_baseline": float(
            pcc(targets - baseline_prediction, targets - prediction)
        ),
        "paired_gain_vs_baseline": {
            name: _paired_gain_summary(
                targets,
                baseline_prediction,
                prediction,
                metric_name=name,
                bootstrap_indices=bootstrap_indices,
                ci=bootstrap_ci,
            )
            for name in metrics
        },
    }


def _paired_gain_summary(
    y_true: np.ndarray,
    baseline_prediction: np.ndarray,
    candidate_prediction: np.ndarray,
    *,
    metric_name: str,
    bootstrap_indices: Sequence[np.ndarray],
    ci: float,
) -> dict[str, float | int | str]:
    metric = get_metric(metric_name)
    sign = 1.0 if higher_is_better(metric_name) else -1.0
    nominal = sign * (
        metric(y_true, candidate_prediction) - metric(y_true, baseline_prediction)
    )
    gains = np.asarray(
        [
            sign
            * (
                metric(y_true[index], candidate_prediction[index])
                - metric(y_true[index], baseline_prediction[index])
            )
            for index in bootstrap_indices
        ],
        dtype=float,
    )
    gains = gains[np.isfinite(gains)]
    if gains.size == 0:
        return {
            "positive_means": "candidate_better",
            "nominal": float(nominal),
            "mean": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "probability_positive": float("nan"),
            "valid_replicates": 0,
        }
    alpha = (1.0 - ci) / 2.0
    low, high = np.quantile(gains, [alpha, 1.0 - alpha])
    return {
        "positive_means": "candidate_better",
        "nominal": float(nominal),
        "mean": float(np.mean(gains)),
        "ci_low": float(low),
        "ci_high": float(high),
        "probability_positive": float(np.mean(gains > 0.0)),
        "valid_replicates": int(gains.size),
    }


def _paired_gain_vs_best_singleton(
    *,
    targets: np.ndarray,
    candidate_prediction: np.ndarray,
    singleton_predictions: Mapping[str, np.ndarray],
    metrics: Sequence[str],
    bootstrap_indices: Sequence[np.ndarray],
    bootstrap_ci: float,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for metric_name in metrics:
        metric = get_metric(metric_name)
        direction = higher_is_better(metric_name)
        best_name = min(
            singleton_predictions,
            key=lambda name: (
                -metric(targets, singleton_predictions[name])
                if direction
                else metric(targets, singleton_predictions[name]),
                name,
            ),
        )
        result[metric_name] = {
            "reference_singleton": best_name,
            **_paired_gain_summary(
                targets,
                singleton_predictions[best_name],
                candidate_prediction,
                metric_name=metric_name,
                bootstrap_indices=bootstrap_indices,
                ci=bootstrap_ci,
            ),
        }
    return result


def _fold_indices(
    sample_ids: Sequence[str],
    fold_ids: Mapping[str, int],
) -> dict[int, np.ndarray]:
    return {
        fold: np.asarray(
            [index for index, sample_id in enumerate(sample_ids) if fold_ids[sample_id] == fold],
            dtype=int,
        )
        for fold in sorted(set(fold_ids.values()))
    }


def _normalize_subset(values: Sequence[str]) -> tuple[str, ...]:
    subset = tuple(str(value).upper() for value in values)
    if not subset or len(set(subset)) != len(subset):
        raise ValueError("Each ANN subset must be nonempty and contain no duplicate invariant")
    return subset


if __name__ == "__main__":
    raise SystemExit(main())
