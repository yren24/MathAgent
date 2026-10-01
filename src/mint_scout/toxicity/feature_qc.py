from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from mint_scout.invariants.manifest import stable_hash
from mint_scout.representation import RepresentationSpec
from mint_scout.toxicity.legacy import TRAILING_FEATURE_DIMS, TOXICITY_INVARIANTS


QC_SCHEMA = "mint-agent.toxicity-feature-qc.v1"


@dataclass(frozen=True)
class ToxicityFeatureQCPolicy:
    expected_sample_count: int | None = None
    max_all_zero_sample_fraction: float = 0.10
    near_constant_std: float = 1e-12
    zero_epsilon: float = 1e-12
    filtration_tail_fraction: float = 0.20
    filtration_min_tail_points: int = 5
    filtration_sparse_zero_fraction: float = 0.995
    filtration_saturation_relative_l2: float = 0.005
    filtration_boundary_relative_l2: float = 0.05
    filtration_boundary_window: int = 3
    active_boundary_sample_fraction_warning: float = 0.50

    def __post_init__(self) -> None:
        if self.expected_sample_count is not None and self.expected_sample_count < 1:
            raise ValueError("expected_sample_count must be positive")
        for name in (
            "max_all_zero_sample_fraction",
            "filtration_tail_fraction",
            "filtration_sparse_zero_fraction",
            "active_boundary_sample_fraction_warning",
        ):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.near_constant_std < 0.0 or self.zero_epsilon < 0.0:
            raise ValueError("QC numerical tolerances must be non-negative")
        if self.filtration_min_tail_points < 2:
            raise ValueError("filtration_min_tail_points must be at least two")
        if self.filtration_boundary_window < 1:
            raise ValueError("filtration_boundary_window must be positive")
        if (
            self.filtration_saturation_relative_l2 < 0.0
            or self.filtration_boundary_relative_l2 < 0.0
        ):
            raise ValueError("filtration change thresholds must be non-negative")


def audit_toxicity_feature_plan(
    *,
    plan_path: str | Path,
    policy: ToxicityFeatureQCPolicy = ToxicityFeatureQCPolicy(),
) -> dict[str, Any]:
    plan = _load_object(plan_path, "toxicity feature plan")
    tasks = plan.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("toxicity feature plan has no tasks")
    expected_count = policy.expected_sample_count
    task_reports = []
    by_signature: dict[str, dict[str, Any]] = {}
    for task in tasks:
        if not isinstance(task, Mapping):
            raise ValueError("toxicity feature plan task must be an object")
        report = audit_toxicity_feature_manifest(
            task=task,
            expected_sample_count=expected_count,
            policy=policy,
        )
        task_reports.append(report)
        by_signature[report["feature_signature"]] = report

    candidate_signatures = plan.get("candidate_feature_signatures")
    if not isinstance(candidate_signatures, Mapping):
        raise ValueError("toxicity feature plan lacks candidate signature mapping")
    candidate_reports = []
    for candidate_id, methods in sorted(candidate_signatures.items()):
        if not isinstance(methods, Mapping):
            raise ValueError("candidate feature signatures must be mappings")
        method_reports = {
            name: by_signature[str(signature)]["status"]
            for name, signature in sorted(methods.items())
        }
        failed = sorted(name for name, status in method_reports.items() if status == "FAIL")
        review = sorted(name for name, status in method_reports.items() if status == "REVIEW")
        candidate_reports.append(
            {
                "candidate_id": str(candidate_id),
                "method_status": method_reports,
                "hard_gate_passed": not failed,
                "failed_methods": failed,
                "review_methods": review,
                "status": "FAIL" if failed else ("REVIEW" if review else "PASS"),
            }
        )
    result: dict[str, Any] = {
        "report_schema": QC_SCHEMA,
        "status": "COMPLETE",
        "evidence_scope": "train_probe_only",
        "test_structures_used": False,
        "test_labels_used": False,
        "plan_hash": plan.get("plan_hash"),
        "policy": asdict(policy),
        "task_count": len(task_reports),
        "task_status_counts": _status_counts(task_reports),
        "tasks": task_reports,
        "candidate_count": len(candidate_reports),
        "candidate_status_counts": _status_counts(candidate_reports),
        "candidates": candidate_reports,
    }
    result["qc_hash"] = stable_hash(result)
    return result


def audit_toxicity_feature_manifest(
    *,
    task: Mapping[str, Any],
    expected_sample_count: int | None,
    policy: ToxicityFeatureQCPolicy,
) -> dict[str, Any]:
    invariant = str(task.get("invariant") or "").upper()
    if invariant not in TOXICITY_INVARIANTS:
        raise ValueError(f"unknown toxicity invariant in feature task: {invariant}")
    spec = RepresentationSpec.read(Path(str(task["representation_spec"])))
    profile = spec.filtration_profiles[invariant]
    expected_shape = (
        profile.num_points,
        len(spec.pair_order),
        TRAILING_FEATURE_DIMS[invariant],
    )
    manifest_path = Path(str(task["manifest_path"]))
    rows = _load_jsonl(manifest_path)
    hard_failures = []
    warnings = []
    if expected_sample_count is not None and len(rows) != expected_sample_count:
        hard_failures.append("SAMPLE_COUNT_MISMATCH")
    sample_ids = [str(row.get("sample_id") or "") for row in rows]
    if not all(sample_ids) or len(set(sample_ids)) != len(sample_ids):
        hard_failures.append("INVALID_OR_DUPLICATE_SAMPLE_ID")
    failed_rows = sum(row.get("status") not in {"computed", "cached"} for row in rows)
    if failed_rows:
        hard_failures.append("FEATURE_TOOL_FAILURE")

    shape_failures = 0
    nonfinite_failures = 0
    missing_files = 0
    all_zero_samples = 0
    active_boundary_samples = 0
    total_values = 0
    zero_values = 0
    sums = np.zeros(int(np.prod(expected_shape)), dtype=np.float64)
    square_sums = np.zeros_like(sums)
    axis_value_count = np.zeros(expected_shape[0], dtype=np.float64)
    axis_nonzero_count = np.zeros(expected_shape[0], dtype=np.float64)
    transition_diff_square = np.zeros(expected_shape[0] - 1, dtype=np.float64)
    transition_base_square = np.zeros(expected_shape[0] - 1, dtype=np.float64)
    valid_samples = 0
    for row in rows:
        output_value = row.get("output_path")
        if not output_value or not Path(str(output_value)).is_file():
            missing_files += 1
            continue
        array = np.load(Path(str(output_value)), allow_pickle=False)
        if tuple(array.shape) != expected_shape:
            shape_failures += 1
            continue
        if not np.isfinite(array).all():
            nonfinite_failures += 1
            continue
        values = np.asarray(array, dtype=np.float64)
        flat = values.reshape(-1)
        filtration_matrix = values.reshape(values.shape[0], -1)
        valid_samples += 1
        total_values += int(flat.size)
        zero_values += int(np.count_nonzero(flat == 0.0))
        all_zero_samples += int(not np.any(flat))
        axis_value_count += filtration_matrix.shape[1]
        axis_nonzero_count += np.count_nonzero(
            np.abs(filtration_matrix) > policy.zero_epsilon, axis=1
        )
        differences = np.diff(filtration_matrix, axis=0)
        sample_diff_square = np.sum(differences * differences, axis=1)
        sample_base_square = np.sum(
            filtration_matrix[:-1] * filtration_matrix[:-1], axis=1
        )
        transition_diff_square += sample_diff_square
        transition_base_square += sample_base_square
        sample_relative_change = np.sqrt(sample_diff_square) / np.maximum(
            np.sqrt(sample_base_square),
            policy.zero_epsilon or np.finfo(float).tiny,
        )
        boundary_window = min(
            policy.filtration_boundary_window, len(sample_relative_change)
        )
        active_boundary_samples += int(
            boundary_window > 0
            and float(np.mean(sample_relative_change[-boundary_window:]))
            >= policy.filtration_boundary_relative_l2
        )
        sums += flat
        square_sums += flat * flat
    if missing_files:
        hard_failures.append("MISSING_FEATURE_FILE")
    if shape_failures:
        hard_failures.append("SHAPE_MISMATCH")
    if nonfinite_failures:
        hard_failures.append("NONFINITE_FEATURE")
    if valid_samples != len(rows):
        hard_failures.append("INCOMPLETE_VALID_FEATURE_SET")

    zero_sample_fraction = all_zero_samples / valid_samples if valid_samples else 1.0
    boundary_fraction = active_boundary_samples / valid_samples if valid_samples else 0.0
    if zero_sample_fraction > policy.max_all_zero_sample_fraction:
        warnings.append("EXCESSIVE_ALL_ZERO_SAMPLES")
    filtration = _filtration_summary(
        axis_value_count=axis_value_count,
        axis_nonzero_count=axis_nonzero_count,
        transition_diff_square=transition_diff_square,
        transition_base_square=transition_base_square,
        policy=policy,
    )
    if (
        boundary_fraction > policy.active_boundary_sample_fraction_warning
        and "ACTIVE_AT_FILTRATION_BOUNDARY" not in filtration["reason_codes"]
    ):
        filtration["reason_codes"].append("ACTIVE_AT_FILTRATION_BOUNDARY")
        if filtration["recommendation"] == "KEEP":
            filtration["recommendation"] = "EXTEND"
    warnings.extend(filtration["reason_codes"])
    if valid_samples:
        means = sums / valid_samples
        variances = np.maximum(square_sums / valid_samples - means * means, 0.0)
        near_constant_count = int(
            np.count_nonzero(np.sqrt(variances) <= policy.near_constant_std)
        )
    else:
        near_constant_count = int(np.prod(expected_shape))
    feature_dimension = int(np.prod(expected_shape))
    if near_constant_count == feature_dimension:
        warnings.append("ALL_FEATURE_DIMENSIONS_CONSTANT")
    status = "FAIL" if hard_failures else ("REVIEW" if warnings else "PASS")
    return {
        "task_index": task.get("task_index"),
        "feature_signature": str(task["feature_signature"]),
        "invariant": invariant,
        "candidate_ids": list(task["candidate_ids"]),
        "manifest_path": str(manifest_path),
        "expected_shape": list(expected_shape),
        "feature_dimension": feature_dimension,
        "sample_count": len(rows),
        "valid_sample_count": valid_samples,
        "failed_row_count": failed_rows,
        "missing_file_count": missing_files,
        "shape_failure_count": shape_failures,
        "nonfinite_failure_count": nonfinite_failures,
        "all_zero_sample_count": all_zero_samples,
        "all_zero_sample_fraction": zero_sample_fraction,
        "zero_value_fraction": zero_values / total_values if total_values else 1.0,
        "near_constant_dimension_count": near_constant_count,
        "near_constant_dimension_fraction": near_constant_count / feature_dimension,
        "active_boundary_sample_count": active_boundary_samples,
        "active_boundary_sample_fraction": boundary_fraction,
        "filtration": filtration,
        "hard_gate_passed": not hard_failures,
        "hard_failures": sorted(set(hard_failures)),
        "warnings": sorted(set(warnings)),
        "status": status,
    }


def _filtration_summary(
    *,
    axis_value_count: np.ndarray,
    axis_nonzero_count: np.ndarray,
    transition_diff_square: np.ndarray,
    transition_base_square: np.ndarray,
    policy: ToxicityFeatureQCPolicy,
) -> dict[str, Any]:
    zero_fraction = 1.0 - np.divide(
        axis_nonzero_count,
        axis_value_count,
        out=np.ones_like(axis_nonzero_count),
        where=axis_value_count > 0,
    )
    relative_change = np.sqrt(transition_diff_square) / np.maximum(
        np.sqrt(transition_base_square),
        policy.zero_epsilon or np.finfo(float).tiny,
    )
    point_count = len(zero_fraction)
    tail_points = min(
        point_count,
        max(
            policy.filtration_min_tail_points,
            int(np.ceil(point_count * policy.filtration_tail_fraction)),
        ),
    )
    tail_start = point_count - tail_points
    tail_changes = relative_change[max(0, tail_start - 1) :]
    boundary_window = min(policy.filtration_boundary_window, len(relative_change))
    boundary_changes = relative_change[-boundary_window:]
    tail_sparse = bool(
        float(np.mean(zero_fraction[tail_start:]))
        >= policy.filtration_sparse_zero_fraction
    )
    tail_saturated = bool(
        len(tail_changes) >= policy.filtration_min_tail_points - 1
        and float(np.max(tail_changes))
        <= policy.filtration_saturation_relative_l2
    )
    boundary_active = bool(
        boundary_window > 0
        and float(np.mean(boundary_changes))
        >= policy.filtration_boundary_relative_l2
    )
    effective_transitions = (
        relative_change > policy.filtration_saturation_relative_l2
    )
    effective_indices = np.flatnonzero(effective_transitions)
    last_effective_point_index = (
        int(effective_indices[-1] + 1) if effective_indices.size else None
    )
    trailing_stable_transition_count = 0
    for changed in effective_transitions[::-1]:
        if changed:
            break
        trailing_stable_transition_count += 1
    reason_codes = []
    if tail_sparse:
        reason_codes.append("SPARSE_FILTRATION_TAIL")
    if tail_saturated:
        reason_codes.append("SATURATED_FILTRATION_TAIL")
    if boundary_active:
        reason_codes.append("ACTIVE_AT_FILTRATION_BOUNDARY")
    recommendation = (
        "SHORTEN"
        if tail_sparse or tail_saturated
        else ("EXTEND" if boundary_active else "KEEP")
    )
    return {
        "recommendation": recommendation,
        "reason_codes": reason_codes,
        "tail_start_index": tail_start,
        "tail_point_count": tail_points,
        "tail_zero_fraction_mean": float(np.mean(zero_fraction[tail_start:])),
        "tail_relative_l2_change_max": (
            float(np.max(tail_changes)) if len(tail_changes) else None
        ),
        "boundary_relative_l2_change_mean": (
            float(np.mean(boundary_changes)) if len(boundary_changes) else None
        ),
        "effective_transition_count": int(np.count_nonzero(effective_transitions)),
        "last_effective_point_index": last_effective_point_index,
        "trailing_stable_transition_count": trailing_stable_transition_count,
    }


def _load_object(path: str | Path, name: str) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain a JSON object")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"toxicity feature manifest does not exist: {path}")
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"manifest line {line_number} must contain an object")
        rows.append(value)
    if not rows:
        raise ValueError(f"toxicity feature manifest is empty: {path}")
    return rows


def _status_counts(reports: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return {
        status: sum(report.get("status") == status for report in reports)
        for status in ("PASS", "REVIEW", "FAIL")
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit deduplicated toxicity Probe feature outputs."
    )
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-sample-count", type=int, default=None)
    parser.add_argument("--max-all-zero-sample-fraction", type=float, default=0.10)
    parser.add_argument(
        "--active-boundary-sample-fraction-warning", type=float, default=0.50
    )
    args = parser.parse_args(argv)
    report = audit_toxicity_feature_plan(
        plan_path=args.plan,
        policy=ToxicityFeatureQCPolicy(
            expected_sample_count=args.expected_sample_count,
            max_all_zero_sample_fraction=args.max_all_zero_sample_fraction,
            active_boundary_sample_fraction_warning=(
                args.active_boundary_sample_fraction_warning
            ),
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"status={report['status']} tasks={report['task_status_counts']} "
        f"candidates={report['candidate_status_counts']} output={args.output}"
    )
    return int(report["task_status_counts"]["FAIL"] > 0)


if __name__ == "__main__":
    raise SystemExit(main())
