from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from mint_scout.invariants.manifest import stable_hash
from mint_scout.representation import RepresentationSpec


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare two filtration representations under an identical OOF protocol."
    )
    parser.add_argument("--baseline-representation", type=Path, required=True)
    parser.add_argument("--candidate-representation", type=Path, required=True)
    parser.add_argument("--baseline-oof", type=Path, action="append", required=True)
    parser.add_argument("--candidate-oof", type=Path, action="append", required=True)
    parser.add_argument("--min-pcc-delta", type=float, required=True)
    parser.add_argument("--min-labeled-samples", type=int, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args(argv)

    report = build_filtration_comparison(
        baseline_representation=RepresentationSpec.read(args.baseline_representation),
        candidate_representation=RepresentationSpec.read(args.candidate_representation),
        baseline_oof=[_read_object(path) for path in args.baseline_oof],
        candidate_oof=[_read_object(path) for path in args.candidate_oof],
        min_pcc_delta=args.min_pcc_delta,
        min_labeled_samples=args.min_labeled_samples,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.output_md.write_text(render_markdown(report), encoding="utf-8")
    print(
        f"status={report['status']} recommendation={report['decision']['recommendation']} "
        f"output={args.output_json}"
    )
    return 0


def build_filtration_comparison(
    *,
    baseline_representation: RepresentationSpec,
    candidate_representation: RepresentationSpec,
    baseline_oof: Sequence[Mapping[str, Any]],
    candidate_oof: Sequence[Mapping[str, Any]],
    min_pcc_delta: float,
    min_labeled_samples: int,
) -> dict[str, Any]:
    if min_pcc_delta < 0.0:
        raise ValueError("min_pcc_delta must be non-negative")
    if min_labeled_samples < 1:
        raise ValueError("min_labeled_samples must be positive")
    baseline_representation.assert_frozen()
    candidate_representation.assert_frozen()

    baseline = _index_oof(baseline_oof)
    candidate = _index_oof(candidate_oof)
    if set(baseline) != set(candidate):
        raise ValueError("Baseline and candidate invariant sets differ")
    invariant_names = tuple(sorted(baseline))
    if not invariant_names:
        raise ValueError("At least one invariant OOF report is required")

    reference = baseline[invariant_names[0]]
    dataset_id = str(reference["dataset_id"])
    sample_ids = tuple(str(value) for value in reference["sample_ids"])
    folds = _folds(reference)
    gbt_hash = _gbt_hash(reference)
    evidence_scope = str(reference["evidence_scope"])
    comparisons: dict[str, Any] = {}
    material_improvements = 0
    material_regressions = 0

    for invariant in invariant_names:
        before = baseline[invariant]
        after = candidate[invariant]
        _assert_shared_protocol(
            before,
            dataset_id=dataset_id,
            sample_ids=sample_ids,
            folds=folds,
            gbt_hash=gbt_hash,
            evidence_scope=evidence_scope,
        )
        _assert_shared_protocol(
            after,
            dataset_id=dataset_id,
            sample_ids=sample_ids,
            folds=folds,
            gbt_hash=gbt_hash,
            evidence_scope=evidence_scope,
        )
        baseline_metrics = _singleton_metrics(before, invariant)
        candidate_metrics = _singleton_metrics(after, invariant)
        delta = candidate_metrics["pcc"] - baseline_metrics["pcc"]
        if delta >= min_pcc_delta:
            outcome = "CANDIDATE_MATERIALLY_BETTER"
            material_improvements += 1
        elif delta <= -min_pcc_delta:
            outcome = "CANDIDATE_MATERIALLY_WORSE"
            material_regressions += 1
        else:
            outcome = "NO_MATERIAL_DIFFERENCE"
        comparisons[invariant] = {
            "baseline": baseline_metrics,
            "candidate": candidate_metrics,
            "candidate_minus_baseline_pcc": delta,
            "outcome": outcome,
            "baseline_filtration": baseline_representation.filtration_profiles[
                invariant
            ].to_dict(),
            "candidate_filtration": candidate_representation.filtration_profiles[
                invariant
            ].to_dict(),
        }

    engineering_only = len(sample_ids) < min_labeled_samples
    if engineering_only:
        recommendation = "DO_NOT_ADOPT_SMALL_SAMPLE"
    elif material_regressions:
        recommendation = "KEEP_BASELINE"
    elif material_improvements:
        recommendation = "REVIEW_CANDIDATE"
    else:
        recommendation = "KEEP_BASELINE"

    payload: dict[str, Any] = {
        "report_schema": "mint-agent.filtration-comparison.v1",
        "status": "COMPLETE",
        "dataset_id": dataset_id,
        "evidence_scope": evidence_scope,
        "sample_count": len(sample_ids),
        "invariants": list(invariant_names),
        "protocol": {
            "sample_order_hash": stable_hash(sample_ids),
            "fold_assignment_hash": stable_hash(folds),
            "gbt_parameter_hash": gbt_hash,
            "same_samples": True,
            "same_folds": True,
            "same_gbt_parameters": True,
        },
        "representation": {
            "baseline_hash": baseline_representation.spec_hash,
            "candidate_hash": candidate_representation.spec_hash,
            "element_pair_schema_changed": (
                baseline_representation.pair_order
                != candidate_representation.pair_order
            ),
            "baseline_pair_count": len(baseline_representation.pair_order),
            "candidate_pair_count": len(candidate_representation.pair_order),
        },
        "thresholds": {
            "min_material_pcc_delta": min_pcc_delta,
            "min_labeled_samples": min_labeled_samples,
        },
        "comparisons": comparisons,
        "decision": {
            "engineering_evidence_only": engineering_only,
            "automatic_adoption_permitted": False,
            "recommendation": recommendation,
            "material_improvement_count": material_improvements,
            "material_regression_count": material_regressions,
        },
    }
    payload["comparison_hash"] = stable_hash(payload)
    return payload


def render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Filtration Candidate Comparison",
        "",
        f"- Dataset: `{report['dataset_id']}`",
        f"- Samples: `{report['sample_count']}`",
        f"- Baseline representation: `{report['representation']['baseline_hash']}`",
        f"- Candidate representation: `{report['representation']['candidate_hash']}`",
        f"- Same element-pair schema: `{not report['representation']['element_pair_schema_changed']}`",
        f"- Minimum material PCC delta: `{report['thresholds']['min_material_pcc_delta']}`",
        f"- Engineering evidence only: `{report['decision']['engineering_evidence_only']}`",
        f"- Recommendation: `{report['decision']['recommendation']}`",
        "",
        "| Invariant | Baseline PCC | Candidate PCC | Delta | Outcome |",
        "|---|---:|---:|---:|---|",
    ]
    for invariant, comparison in sorted(report["comparisons"].items()):
        lines.append(
            f"| {invariant} | {comparison['baseline']['pcc']:.6f} | "
            f"{comparison['candidate']['pcc']:.6f} | "
            f"{comparison['candidate_minus_baseline_pcc']:.6f} | "
            f"{comparison['outcome']} |"
        )
    lines.extend(
        [
            "",
            "The comparison validates identical sample order, fold assignment, and GBT parameters. "
            "It never changes the accepted representation automatically.",
        ]
    )
    return "\n".join(lines) + "\n"


def _index_oof(reports: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    indexed: dict[str, Mapping[str, Any]] = {}
    for report in reports:
        if report.get("status") != "COMPLETE":
            raise ValueError("OOF report is not complete")
        prediction_map = report.get("scout", {}).get("invariant_oof_predictions", {})
        if not isinstance(prediction_map, Mapping) or len(prediction_map) != 1:
            raise ValueError("Each OOF report must contain exactly one invariant")
        invariant = str(next(iter(prediction_map))).upper()
        if invariant in indexed:
            raise ValueError(f"Duplicate OOF report for {invariant}")
        indexed[invariant] = report
    return indexed


def _assert_shared_protocol(
    report: Mapping[str, Any],
    *,
    dataset_id: str,
    sample_ids: tuple[str, ...],
    folds: Mapping[str, int],
    gbt_hash: str,
    evidence_scope: str,
) -> None:
    if str(report.get("dataset_id")) != dataset_id:
        raise ValueError("Baseline and candidate dataset IDs differ")
    if tuple(str(value) for value in report.get("sample_ids", ())) != sample_ids:
        raise ValueError("Baseline and candidate sample order differs")
    if _folds(report) != folds:
        raise ValueError("Baseline and candidate fold assignments differ")
    if _gbt_hash(report) != gbt_hash:
        raise ValueError("Baseline and candidate GBT parameters differ")
    if str(report.get("evidence_scope")) != evidence_scope:
        raise ValueError("Baseline and candidate evidence scopes differ")


def _folds(report: Mapping[str, Any]) -> dict[str, int]:
    return {
        str(key): int(value)
        for key, value in report["scout"]["fold_assignment"].items()
    }


def _gbt_hash(report: Mapping[str, Any]) -> str:
    return str(report["scout"]["gbt_parameter_hash"])


def _singleton_metrics(report: Mapping[str, Any], invariant: str) -> dict[str, Any]:
    candidates = report["scout"]["consensus_metrics"]
    for candidate in candidates:
        if tuple(candidate.get("invariants", ())) == (invariant,):
            secondary = candidate.get("secondary_metrics", {})
            return {
                "pcc": float(candidate["nominal_primary_score"]),
                "bootstrap_ci_low": float(candidate["bootstrap_ci_low"]),
                "bootstrap_ci_high": float(candidate["bootstrap_ci_high"]),
                "rmse": float(secondary["RMSE"]),
                "mae": float(secondary["MAE"]),
                "r2": float(secondary["R2"]),
            }
    raise ValueError(f"OOF report lacks singleton metrics for {invariant}")


def _read_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


if __name__ == "__main__":
    raise SystemExit(main())
