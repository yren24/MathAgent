from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from mint_scout.invariants.manifest import stable_hash
from mint_scout.representation import RepresentationSpec
from mint_scout.toxicity.selection_contract import methods_from_plan


SELECTION_SCHEMA = "mint-agent.toxicity-representation-selection.v1"


@dataclass(frozen=True)
class ToxicityRepresentationPolicy:
    minimum_pair_coverage: float = 0.0
    pair_coverage_tolerance: float = 0.02
    filtration_issue_tolerance: int = 0
    feature_warning_tolerance: int = 0
    dimension_tolerance: int = 0

    def __post_init__(self) -> None:
        for name in ("minimum_pair_coverage", "pair_coverage_tolerance"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1]")
        for name in (
            "filtration_issue_tolerance",
            "feature_warning_tolerance",
            "dimension_tolerance",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")


def select_toxicity_representation(
    *,
    design_report_path: str | Path,
    representation_dir: str | Path,
    feature_plan_path: str | Path,
    feature_qc_path: str | Path,
    probe_selection_path: str | Path,
    policy: ToxicityRepresentationPolicy = ToxicityRepresentationPolicy(),
    baseline_id: str = "legacy-30-pairs-legacy-grid",
) -> dict[str, Any]:
    design = _load_object(design_report_path, "toxicity design report")
    plan = _load_object(feature_plan_path, "toxicity feature plan")
    qc = _load_object(feature_qc_path, "toxicity feature QC")
    probe = _load_object(probe_selection_path, "toxicity Probe selection")
    if qc.get("plan_hash") != plan.get("plan_hash"):
        raise ValueError("toxicity feature QC does not match the feature plan")
    if qc.get("test_labels_used") is not False or qc.get("test_structures_used") is not False:
        raise ValueError("toxicity representation selection requires train-only QC")
    if probe.get("test_labels_used") is not False:
        raise ValueError("toxicity representation selection requires train-only Probe evidence")

    selected_probe_id = str(probe.get("selected_candidate_id") or "")
    probe_candidates = probe.get("candidates")
    if not isinstance(probe_candidates, list):
        raise ValueError("toxicity Probe report has no candidates")
    selected_probe = next(
        (
            candidate
            for candidate in probe_candidates
            if candidate.get("candidate_id") == selected_probe_id
        ),
        None,
    )
    if not isinstance(selected_probe, Mapping):
        raise ValueError("selected toxicity Probe candidate is unavailable")
    pair_support = {
        _decode_pair(name): int(count)
        for name, count in _required_mapping(selected_probe, "pair_support").items()
    }
    total_pair_support = sum(pair_support.values())
    if total_pair_support <= 0:
        raise ValueError("toxicity Probe pair-support mass must be positive")

    qc_tasks = {
        str(task["feature_signature"]): task
        for task in _required_list(qc, "tasks")
    }
    qc_candidates = {
        str(candidate["candidate_id"]): candidate
        for candidate in _required_list(qc, "candidates")
    }
    signature_map = _required_mapping(plan, "candidate_feature_signatures")
    methods = methods_from_plan(plan)
    candidates = _required_list(design, "candidates")
    rows = []
    spec_root = Path(representation_dir).expanduser().resolve()
    for candidate in candidates:
        candidate_id = str(candidate.get("candidate_id") or "")
        spec_path = spec_root / f"{candidate_id}.json"
        spec = RepresentationSpec.read(spec_path)
        expected_hash = str(candidate.get("representation_hash") or "")
        if spec.spec_hash != expected_hash:
            raise ValueError(f"representation hash mismatch for {candidate_id}")
        method_signatures = _required_mapping(signature_map, candidate_id)
        method_qc = {
            method: qc_tasks[str(method_signatures[method])]
            for method in methods
        }
        candidate_qc = qc_candidates[candidate_id]
        retained_mass = sum(
            support for pair, support in pair_support.items() if pair in set(spec.pair_order)
        )
        pair_coverage = retained_mass / total_pair_support
        filtration_adjustments = sum(
            spec.filtration_profiles[method].scale_kind == "distance"
            and report["filtration"]["recommendation"] != "KEEP"
            for method, report in method_qc.items()
        )
        degeneracy_warnings = sum(
            warning
            in {
                "EXCESSIVE_ALL_ZERO_SAMPLES",
                "ALL_FEATURE_DIMENSIONS_CONSTANT",
            }
            for report in method_qc.values()
            for warning in report["warnings"]
        )
        filtration_warning_codes = {
            "SPARSE_FILTRATION_TAIL",
            "SATURATED_FILTRATION_TAIL",
            "ACTIVE_AT_FILTRATION_BOUNDARY",
        }
        feature_warnings = sum(
            warning not in filtration_warning_codes
            for report in method_qc.values()
            for warning in report["warnings"]
        )
        total_dimensions = sum(
            int(report["feature_dimension"]) for report in method_qc.values()
        )
        gate_failures = []
        if not bool(candidate_qc.get("hard_gate_passed")):
            gate_failures.append("FEATURE_QC_HARD_GATE")
        if pair_coverage < policy.minimum_pair_coverage:
            gate_failures.append("MINIMUM_PAIR_COVERAGE")
        rows.append(
            {
                "candidate_id": candidate_id,
                "source": candidate.get("source"),
                "ablation": candidate.get("ablation"),
                "proposal_id": candidate.get("proposal_id"),
                "representation_path": str(spec_path),
                "representation_hash": spec.spec_hash,
                "pair_count": len(spec.pair_order),
                "retained_pair_support_mass": retained_mass,
                "total_pair_support_mass": total_pair_support,
                "pair_coverage": pair_coverage,
                "filtration_adjustment_count": filtration_adjustments,
                "feature_degeneracy_warning_count": degeneracy_warnings,
                "feature_warning_count": feature_warnings,
                "total_feature_dimensions": total_dimensions,
                "method_feature_signatures": dict(method_signatures),
                "gate_failures": gate_failures,
                "eligible": not gate_failures,
            }
        )
    eligible = [row for row in rows if row["eligible"]]
    if not eligible:
        raise ValueError("all toxicity representation candidates failed hard gates")
    selected, trace = _hierarchical_select(
        eligible, baseline_id=baseline_id, policy=policy
    )
    ranked = sorted(
        rows,
        key=lambda row: (
            0 if row["candidate_id"] == selected["candidate_id"] else 1,
            not row["eligible"],
            -row["pair_coverage"],
            row["filtration_adjustment_count"],
            row["feature_degeneracy_warning_count"],
            row["feature_warning_count"],
            row["total_feature_dimensions"],
            row["candidate_id"],
        ),
    )
    for rank, row in enumerate(ranked, 1):
        row["decision_rank"] = rank
        row["selected"] = row["candidate_id"] == selected["candidate_id"]
    result: dict[str, Any] = {
        "report_schema": SELECTION_SCHEMA,
        "status": "COMPLETE",
        "selection_basis": "hierarchical_train_only_representation_suitability",
        "evidence_scope": "train_probe_only",
        "test_structures_used": False,
        "test_labels_used": False,
        "model_performance_used": False,
        "ranking_stability_used": False,
        "runtime_cost_used": False,
        "llm_prior_used": False,
        "policy": asdict(policy),
        "probe_selection_hash": probe.get("selection_hash"),
        "feature_plan_hash": plan.get("plan_hash"),
        "feature_qc_hash": qc.get("qc_hash"),
        "decision_trace": trace,
        "candidates": ranked,
        "selected_candidate_id": selected["candidate_id"],
        "selected_representation": selected["representation_path"],
        "selected_representation_hash": selected["representation_hash"],
        "selected_method_feature_signatures": selected[
            "method_feature_signatures"
        ],
    }
    result["selection_hash"] = stable_hash(result)
    return result


def _hierarchical_select(
    rows: Sequence[dict[str, Any]],
    *,
    baseline_id: str,
    policy: ToxicityRepresentationPolicy,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    survivors = list(rows)
    trace = [
        {
            "stage": "train_only_feature_quality_hard_gates",
            "survivors": _ids(survivors),
        }
    ]
    survivors = _keep_near_best(
        survivors,
        "pair_coverage",
        higher_is_better=True,
        tolerance=policy.pair_coverage_tolerance,
    )
    trace.append(
        {
            "stage": "element_pair_support_coverage",
            "tolerance": policy.pair_coverage_tolerance,
            "survivors": _ids(survivors),
        }
    )
    for stage, field, tolerance in (
        (
            "filtration_adequacy",
            "filtration_adjustment_count",
            policy.filtration_issue_tolerance,
        ),
        (
            "feature_degeneracy",
            "feature_degeneracy_warning_count",
            policy.feature_warning_tolerance,
        ),
        (
            "feature_health",
            "feature_warning_count",
            policy.feature_warning_tolerance,
        ),
        (
            "minimum_feature_dimension_cost",
            "total_feature_dimensions",
            policy.dimension_tolerance,
        ),
    ):
        survivors = _keep_near_best(
            survivors, field, higher_is_better=False, tolerance=float(tolerance)
        )
        trace.append(
            {"stage": stage, "tolerance": tolerance, "survivors": _ids(survivors)}
        )
    selected = next(
        (row for row in survivors if row["candidate_id"] == baseline_id),
        min(survivors, key=lambda row: row["candidate_id"]),
    )
    trace.append(
        {
            "stage": "deterministic_tie_break",
            "rule": "baseline_then_candidate_id",
            "selected": selected["candidate_id"],
        }
    )
    return selected, trace


def _keep_near_best(
    rows: Sequence[dict[str, Any]],
    field: str,
    *,
    higher_is_better: bool,
    tolerance: float,
) -> list[dict[str, Any]]:
    values = [float(row[field]) for row in rows]
    best = max(values) if higher_is_better else min(values)
    if higher_is_better:
        return [row for row in rows if float(row[field]) >= best - tolerance]
    return [row for row in rows if float(row[field]) <= best + tolerance]


def _ids(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    return [str(row["candidate_id"]) for row in rows]


def _decode_pair(value: str) -> tuple[str, str]:
    try:
        left, right = value.split("-", 1)
    except ValueError as exc:
        raise ValueError(f"invalid toxicity pair name: {value}") from exc
    return left, right


def _load_object(path: str | Path, name: str) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain a JSON object")
    return value


def _required_mapping(value: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    item = value.get(key)
    if not isinstance(item, Mapping):
        raise ValueError(f"{key} must be an object")
    return item


def _required_list(value: Mapping[str, Any], key: str) -> list[Mapping[str, Any]]:
    item = value.get(key)
    if not isinstance(item, list) or not all(isinstance(row, Mapping) for row in item):
        raise ValueError(f"{key} must be a list of objects")
    return item


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Select a toxicity representation using label-free Probe evidence."
    )
    parser.add_argument("--design-report", type=Path, required=True)
    parser.add_argument("--representation-dir", type=Path, required=True)
    parser.add_argument("--feature-plan", type=Path, required=True)
    parser.add_argument("--feature-qc", type=Path, required=True)
    parser.add_argument("--probe-selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pair-coverage-tolerance", type=float, default=0.02)
    args = parser.parse_args(argv)
    result = select_toxicity_representation(
        design_report_path=args.design_report,
        representation_dir=args.representation_dir,
        feature_plan_path=args.feature_plan,
        feature_qc_path=args.feature_qc,
        probe_selection_path=args.probe_selection,
        policy=ToxicityRepresentationPolicy(
            pair_coverage_tolerance=args.pair_coverage_tolerance
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"status={result['status']} selected={result['selected_candidate_id']} "
        f"selection_hash={result['selection_hash']} output={args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
