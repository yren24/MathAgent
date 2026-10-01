"""Explainable hierarchical selection of MOF topology methods from probe evidence."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any, Iterable, Mapping


SELECTION_SCHEMA = "mint-agent.mof-probe-selection.v1"


def select_probe_method(
    tool_statuses: Iterable[Mapping[str, Any]],
    *,
    r2_tolerance: float = 0.01,
    stability_tolerance: float = 0.005,
) -> dict[str, Any]:
    """Select one method by QC, standard R2, stability, then cost."""
    if r2_tolerance < 0 or stability_tolerance < 0:
        raise ValueError("selection tolerances must be non-negative")
    candidates = [_candidate(status) for status in tool_statuses]
    eligible = [candidate for candidate in candidates if candidate["eligible"]]
    all_candidates_evaluated = all(candidate["eligible"] for candidate in candidates)
    if not eligible:
        return {
            "report_schema": SELECTION_SCHEMA,
            "status": "NO_ELIGIBLE_CANDIDATE",
            "is_final_selection": False,
            "candidates": candidates,
            "selected_tool": None,
        }

    best_r2 = max(float(candidate["mean_r2_standard"]) for candidate in eligible)
    performance_tier = [
        candidate
        for candidate in eligible
        if float(candidate["mean_r2_standard"]) >= best_r2 - r2_tolerance
    ]
    best_stability = min(float(candidate["std_repeat_r2_standard"]) for candidate in performance_tier)
    stability_tier = [
        candidate
        for candidate in performance_tier
        if float(candidate["std_repeat_r2_standard"]) <= best_stability + stability_tolerance
    ]
    final_tier = sorted(
        stability_tier,
        key=lambda candidate: (
            candidate["feature_dimension"] if isinstance(candidate["feature_dimension"], int) else math.inf,
            str(candidate["tool_name"]),
        ),
    )
    selected = final_tier[0]
    return {
        "report_schema": SELECTION_SCHEMA,
        "status": "SELECTED" if all_candidates_evaluated else "PROVISIONAL_SELECTED",
        "is_final_selection": all_candidates_evaluated,
        "rule": {
            "hard_gate": "finite probe metrics and complete frozen probe features",
            "performance": {"metric": "mean_r2_standard", "tolerance": r2_tolerance, "best": best_r2},
            "stability": {
                "metric": "std_repeat_r2_standard",
                "tolerance": stability_tolerance,
                "best": best_stability,
            },
            "cost_tie_break": "smaller feature_dimension",
        },
        "candidates": candidates,
        "performance_tier": [candidate["tool_name"] for candidate in performance_tier],
        "stability_tier": [candidate["tool_name"] for candidate in stability_tier],
        "selected_tool": selected["tool_name"],
        "selection_reason": _selection_reason(performance_tier, stability_tier, selected),
    }


def _candidate(status: Mapping[str, Any]) -> dict[str, Any]:
    metrics = status.get("probe_metrics")
    result = {
        "tool_name": status.get("tool_name"),
        "feature_dimension": status.get("feature_dimension"),
        "eligible": False,
        "reason": None,
        "mean_r2_standard": None,
        "std_repeat_r2_standard": None,
    }
    if not status.get("probe_feature_ready"):
        result["reason"] = "probe_features_incomplete"
        return result
    if not isinstance(metrics, Mapping):
        result["reason"] = "probe_metrics_missing"
        return result
    summary = metrics.get("summary")
    if not isinstance(summary, Mapping) or not _finite(summary.get("mean_r2_standard")):
        result["reason"] = "invalid_probe_r2_standard"
        return result
    repeat_r2 = _repeat_r2_values(metrics)
    result["mean_r2_standard"] = float(summary["mean_r2_standard"])
    result["std_repeat_r2_standard"] = pstdev(repeat_r2) if len(repeat_r2) > 1 else 0.0
    result["eligible"] = True
    result["reason"] = "eligible"
    return result


def _repeat_r2_values(metrics: Mapping[str, Any]) -> list[float]:
    repeat_summaries = metrics.get("repeat_summaries")
    if isinstance(repeat_summaries, list):
        values = [
            float(row["mean_r2_standard"])
            for row in repeat_summaries
            if isinstance(row, Mapping) and _finite(row.get("mean_r2_standard"))
        ]
        if values:
            return values
    fold_metrics = metrics.get("fold_metrics")
    if not isinstance(fold_metrics, list):
        return []
    by_repeat: dict[int, list[float]] = {}
    for row in fold_metrics:
        if not isinstance(row, Mapping) or not _finite(row.get("r2_standard")):
            continue
        by_repeat.setdefault(int(row.get("repeat", 0)), []).append(float(row["r2_standard"]))
    return [fmean(values) for _, values in sorted(by_repeat.items()) if values]


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _selection_reason(performance_tier: list[dict[str, Any]], stability_tier: list[dict[str, Any]], selected: Mapping[str, Any]) -> str:
    if len(performance_tier) == 1:
        return "highest_probe_standard_r2_outside_performance_tolerance"
    if len(stability_tier) == 1:
        return "probe_standard_r2_tie_resolved_by_repeat_stability"
    return "probe_standard_r2_and_stability_tie_resolved_by_feature_dimension"


def main() -> None:
    parser = argparse.ArgumentParser(description="Select a MOF topology method from frozen-probe reports")
    parser.add_argument("--workflow", type=Path, required=True)
    parser.add_argument("--probe-result-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--r2-tolerance", type=float, default=0.01)
    parser.add_argument("--stability-tolerance", type=float, default=0.005)
    args = parser.parse_args()
    workflow = json.loads(args.workflow.read_text(encoding="utf-8"))
    property_name = str(workflow["property"])
    statuses = []
    for tool in workflow["tools"]:
        topology = tool["topology"]
        metric_path = args.probe_result_dir / f"{property_name}_{topology}_probe_gbt_metrics.json"
        status = dict(tool)
        status["probe_metrics"] = json.loads(metric_path.read_text(encoding="utf-8")) if metric_path.exists() else None
        statuses.append(status)
    report = select_probe_method(
        statuses,
        r2_tolerance=args.r2_tolerance,
        stability_tolerance=args.stability_tolerance,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"status={report['status']}")
    print(f"selected_tool={report['selected_tool']}")


if __name__ == "__main__":
    main()
