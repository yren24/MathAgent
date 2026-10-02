from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from mint_scout.config import load_yaml
from mint_scout.execution.artifacts import ScoutExecutionArtifact
from mint_scout.feature_qc import assert_qc_report_compatible
from mint_scout.invariants.manifest import stable_hash
from mint_scout.representation import RepresentationSpec
from mint_scout.scout.pipeline import ScoutConfig
from mint_scout.scout.stability import rank_consensus_subsets
from mint_scout.scout.target import resolve_target


RANKING_POLICY = "hierarchical_empirical_v1"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Combine independently computed invariant OOF predictions and rank all subsets."
    )
    parser.add_argument("--scout-config", type=Path, required=True)
    parser.add_argument("--dataset-id", default=None)
    parser.add_argument("--evidence-scope", default="probe")
    parser.add_argument("--representation-spec", type=Path, required=True)
    parser.add_argument("--probe-selection", type=Path, required=True)
    parser.add_argument("--scout-report", type=Path, action="append", required=True)
    parser.add_argument("--feature-qc-report", type=Path, default=None)
    parser.add_argument("--ranking-policy", default=RANKING_POLICY)
    parser.add_argument("--llm-prior-candidate", action="append", default=[])
    parser.add_argument("--user-target", type=float, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execution-artifact", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.ranking_policy != RANKING_POLICY:
        raise ValueError(f"Unsupported Scout ranking policy: {args.ranking_policy!r}")

    config = ScoutConfig.from_mapping(load_yaml(args.scout_config))
    representation = RepresentationSpec.read(args.representation_spec)
    representation.assert_frozen()
    probe = _load_object(args.probe_selection)
    if probe.get("representation_hash") != representation.spec_hash:
        raise ValueError("Probe selection and RepresentationSpec hashes do not match")
    sample_ids = tuple(str(sample_id) for sample_id in probe["probe_sample_ids"])
    rows = {str(row["sample_id"]): row for row in probe["probe_samples"]}
    if set(rows) != set(sample_ids):
        raise ValueError("probe_samples do not match probe_sample_ids")
    y_true = np.asarray([float(rows[sample_id]["label"]) for sample_id in sample_ids])
    strata = [int(rows[sample_id]["target_bin"]) for sample_id in sample_ids]

    predictions: dict[str, np.ndarray] = {}
    sources: dict[str, str] = {}
    feature_manifests: dict[str, Path] = {}
    gbt_hashes: set[str] = set()
    for report_path in args.scout_report:
        report = _load_object(report_path)
        if report.get("representation_hash") != representation.spec_hash:
            raise ValueError(f"Representation hash mismatch in {report_path}")
        if report.get("selection_hash") != probe.get("selection_hash"):
            raise ValueError(f"Selection hash mismatch in {report_path}")
        scout = report["scout"]
        gbt_hash = str(scout.get("gbt_parameter_hash") or "")
        if not gbt_hash:
            raise ValueError(f"Missing GBT parameter hash in {report_path}")
        gbt_hashes.add(gbt_hash)
        if tuple(scout.get("probe_sample_ids", ())) != sample_ids:
            raise ValueError(f"Probe sample order mismatch in {report_path}")
        report_predictions = scout.get("invariant_oof_predictions", {})
        if len(report_predictions) != 1:
            raise ValueError(
                f"Each source report must contain exactly one invariant: {report_path}"
            )
        invariant, by_id = next(iter(report_predictions.items()))
        name = str(invariant).upper()
        if name in predictions:
            raise ValueError(f"Duplicate OOF predictions for {name}")
        if set(by_id) != set(sample_ids):
            raise ValueError(f"OOF sample IDs mismatch for {name}")
        values = np.asarray([float(by_id[sample_id]) for sample_id in sample_ids])
        if not np.all(np.isfinite(values)):
            raise ValueError(f"OOF predictions contain NaN or Inf for {name}")
        predictions[name] = values
        sources[name] = str(report_path)
        manifest = report.get("feature_manifests", {}).get(name)
        if manifest is not None:
            feature_manifests[name] = Path(str(manifest))
    if not predictions:
        raise ValueError("No OOF predictions were loaded")
    if len(gbt_hashes) != 1:
        raise ValueError(
            "Source Scout OOF reports must use one shared GBT configuration; "
            f"found {sorted(gbt_hashes)}"
        )
    gbt_parameter_hash = next(iter(gbt_hashes))
    qc_report = None
    if args.feature_qc_report is not None:
        if set(feature_manifests) != set(predictions):
            raise ValueError(
                "Feature QC checking requires source reports with feature_manifests"
            )
        qc_report = assert_qc_report_compatible(
            path=args.feature_qc_report,
            representation=representation,
            sample_ids=sample_ids,
            manifest_paths=feature_manifests,
        )

    cost_profile = _estimate_full_acquisition_costs(
        feature_manifests=feature_manifests,
        invariant_names=tuple(predictions),
        modeling_sample_count=len(probe["modeling_sample_ids"]),
    )
    quality_profile = _feature_quality_scores(qc_report, tuple(predictions))
    llm_prior_order = _parse_llm_prior_order(
        args.llm_prior_candidate, tuple(predictions)
    )
    # LLM proposals are hypotheses and provenance only. They never receive
    # numerical credit in the empirical method ranking.
    effective_llm_prior_order: tuple[tuple[str, ...], ...] = ()

    ranking = rank_consensus_subsets(
        y_true=y_true,
        predictions=predictions,
        primary_metric=config.primary_metric,
        secondary_metrics=config.secondary_metrics,
        strata=strata,
        bootstrap=config.bootstrap,
        invariant_costs={
            name: float(details["estimated_full_wall_seconds"])
            for name, details in cost_profile.items()
        },
        invariant_quality_scores=quality_profile,
        llm_prior_order=effective_llm_prior_order,
        ranking_weights=config.ranking_weights,
    )
    target = resolve_target(
        metric=config.primary_metric,
        ranking=ranking,
        user_target=args.user_target,
        config=config.target,
    )
    prediction_hashes = {
        name: hashlib.sha256(values.tobytes()).hexdigest()
        for name, values in sorted(predictions.items())
    }
    probe_folds = {
        str(sample_id): int(fold)
        for sample_id, fold in probe["probe_fold_assignment"].items()
    }
    combined_probe_hash = stable_hash(
        {
            "sample_ids": sample_ids,
            "fold_assignment": probe_folds,
            "gbt_parameter_hash": gbt_parameter_hash,
            "primary_metric": config.primary_metric,
            "prediction_hashes": prediction_hashes,
            "priority_order": ranking.priority_order,
            "cost_profile": cost_profile,
            "quality_profile": quality_profile,
            "llm_prior_order": effective_llm_prior_order,
            "ranking_policy": RANKING_POLICY,
        }
    )
    summaries = [
        {
            **asdict(summary),
            "invariants": list(summary.invariants),
            "secondary_metrics": dict(summary.secondary_metrics),
        }
        for summary in ranking.summaries
    ]
    payload = {
        "report_schema": "mint-agent.combined-scout.v1",
        "dataset_id": args.dataset_id,
        "evidence_scope": args.evidence_scope,
        "status": "COMPLETE",
        "representation_hash": representation.spec_hash,
        "selection_hash": probe.get("selection_hash"),
        "source_oof_reports": sources,
        "feature_qc_report": str(args.feature_qc_report)
        if args.feature_qc_report
        else None,
        "sample_count": len(sample_ids),
        "sample_ids": list(sample_ids),
        "invariants": sorted(predictions),
        "gbt_parameter_hash": gbt_parameter_hash,
        "ranking_policy": RANKING_POLICY,
        "combined_probe_hash": combined_probe_hash,
        "consensus_metrics": summaries,
        "cost_profile": cost_profile,
        "quality_profile": quality_profile,
        "llm_prior": {
            "proposal_recorded": bool(llm_prior_order),
            "applied": False,
            "priority_order": [list(subset) for subset in llm_prior_order],
            "configured_diagnostic_weight": config.ranking_weights.llm_prior,
            "decision_weight": 0.0,
        },
        "priority_policy": {
            "candidate_universe": "all_nonempty_invariant_subsets",
            "candidate_count": len(ranking.priority_order),
            "ranking_method": "hierarchical_empirical",
            "decision_order": [
                "feature_qc_hard_gate",
                "paired_bootstrap_top_tier",
                "ranking_stability_within_top_tier",
                "computational_cost_tie_break",
                "deterministic_subset_tie_break",
            ],
            "diagnostic_composite_only": True,
            "diagnostic_weights": asdict(config.ranking_weights),
            "feature_quality_policy": "FAIL=blocked_before_ranking; PASS/WARN=eligible and reported",
            "llm_prior_used": False,
            "run_aggregation": "mean_predictions_then_compute_metric",
        },
        "top_tier": [list(subset) for subset in ranking.top_tier],
        "frozen_priority_order": [list(subset) for subset in ranking.priority_order],
        "target": asdict(target),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    artifact = ScoutExecutionArtifact(
        artifact_version="mint-agent.scout-execution.v1",
        modeling_sample_ids=tuple(
            str(sample_id) for sample_id in probe["modeling_sample_ids"]
        ),
        representation_hash=representation.spec_hash,
        frozen_priority_order=ranking.priority_order,
        full_fold_assignment={
            str(sample_id): int(fold)
            for sample_id, fold in probe["modeling_fold_assignment"].items()
        },
        target_metric=target.metric.upper(),
        target_direction=target.direction,
        target_value=target.value,
        target_source=target.source,
        gbt_parameter_hash=gbt_parameter_hash,
        probe_hash=combined_probe_hash,
        ranking_policy=RANKING_POLICY,
    )
    artifact.write(args.execution_artifact)
    print(
        f"probe_hash={combined_probe_hash} invariants={','.join(sorted(predictions))} "
        f"subsets={len(ranking.priority_order)} target={target.value:.6g} output={args.output}"
    )
    return 0


def _load_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _estimate_full_acquisition_costs(
    *,
    feature_manifests: dict[str, Path],
    invariant_names: tuple[str, ...],
    modeling_sample_count: int,
) -> dict[str, dict[str, object]]:
    if modeling_sample_count < 1:
        raise ValueError("modeling_sample_count must be positive")
    profile: dict[str, dict[str, object]] = {}
    for invariant in invariant_names:
        path = feature_manifests.get(invariant)
        if path is None:
            profile[invariant] = {
                "source": "unit_fallback_missing_manifest",
                "observed_sample_count": 0,
                "per_sample_wall_seconds": 1.0,
                "estimated_full_wall_seconds": float(modeling_sample_count),
            }
            continue
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        computed = [
            float(row["wall_seconds"])
            for row in rows
            if row.get("status") == "computed"
            and row.get("wall_seconds") is not None
            and np.isfinite(float(row["wall_seconds"]))
            and float(row["wall_seconds"]) >= 0.0
        ]
        observed = computed or [
            float(row["wall_seconds"])
            for row in rows
            if row.get("status") in {"computed", "cached"}
            and row.get("wall_seconds") is not None
            and np.isfinite(float(row["wall_seconds"]))
            and float(row["wall_seconds"]) >= 0.0
        ]
        if observed:
            per_sample = float(np.median(observed))
            source = (
                "computed_manifest_median" if computed else "cached_manifest_median"
            )
        else:
            per_sample = 1.0
            source = "unit_fallback_missing_timing"
        profile[invariant] = {
            "source": source,
            "feature_manifest": str(path),
            "observed_sample_count": len(observed),
            "per_sample_wall_seconds": per_sample,
            "estimated_full_wall_seconds": per_sample * modeling_sample_count,
        }
    return profile


def _feature_quality_scores(
    report: dict[str, Any] | None, invariant_names: tuple[str, ...]
) -> dict[str, float]:
    if report is None:
        return {name: 1.0 for name in invariant_names}
    invariant_reports = report.get("invariants")
    if not isinstance(invariant_reports, dict):
        raise ValueError("Feature QC report does not contain invariant summaries")
    scores: dict[str, float] = {}
    for name in invariant_names:
        details = invariant_reports.get(name)
        if not isinstance(details, dict):
            raise ValueError(f"Feature QC report is missing {name}")
        status = str(details.get("status") or "").upper()
        if status == "FAIL":
            raise ValueError(f"Feature QC failed for {name}")
        if status == "PASS":
            scores[name] = 1.0
        elif status == "WARN":
            warning_count = max(int(details.get("warning_issue_count") or 1), 1)
            loaded_count = max(int(details.get("loaded_sample_count") or 1), 1)
            scores[name] = max(0.50, 1.0 - min(warning_count / loaded_count, 0.50))
        else:
            raise ValueError(f"Unsupported Feature QC status for {name}: {status!r}")
    return scores


def _parse_llm_prior_order(
    values: list[str], invariant_names: tuple[str, ...]
) -> tuple[tuple[str, ...], ...]:
    if not values:
        return ()
    allowed = set(invariant_names)
    result: list[tuple[str, ...]] = []
    for value in values:
        subset = tuple(
            sorted({item.strip().upper() for item in value.split(",") if item.strip()})
        )
        if not subset or not set(subset).issubset(allowed):
            raise ValueError(f"Invalid LLM prior candidate: {value!r}")
        result.append(subset)
    if len(set(result)) != len(result):
        raise ValueError("LLM prior candidates must be unique")
    return tuple(result)


if __name__ == "__main__":
    raise SystemExit(main())
