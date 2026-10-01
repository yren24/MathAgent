from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


REPORT_SCHEMA = "mint-agent.probe-ablation-summary.v1"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Summarize fixed-budget Probe-selection ablation artifacts."
    )
    parser.add_argument(
        "--arm",
        action="append",
        required=True,
        metavar="NAME=SELECTION,AUDIT,SCOUT",
        help="One arm's Probe selection, audit, and combined Scout report.",
    )
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, default=None)
    args = parser.parse_args(argv)

    arms = [_load_arm(value) for value in args.arm]
    names = [str(arm["arm_id"]) for arm in arms]
    if len(set(names)) != len(names):
        raise ValueError("arm names must be unique")
    if args.baseline not in names:
        raise ValueError("baseline must name one supplied arm")

    summary = build_summary(arms, baseline=args.baseline)
    _write_json(args.output, summary)
    if args.markdown_output is not None:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(_render_markdown(summary), encoding="utf-8")
    print(f"arms={len(arms)} baseline={args.baseline} output={args.output}")
    return 0


def build_summary(
    arms: list[Mapping[str, Any]], *, baseline: str,
) -> dict[str, Any]:
    normalized = [_normalize_arm(arm) for arm in arms]
    reference = next(arm for arm in normalized if arm["arm_id"] == baseline)
    rows = []
    for arm in normalized:
        agreement = _ranking_agreement(reference, arm)
        rows.append({**arm, "agreement_with_baseline": agreement})
    return {
        "report_schema": REPORT_SCHEMA,
        "selection_prohibited": True,
        "purpose": "post-hoc ablation reporting only; results cannot modify the production selector",
        "baseline_arm": baseline,
        "arms": rows,
        "random_arm_summary": _random_arm_summary(rows, baseline=baseline),
    }


def _load_arm(value: str) -> dict[str, Any]:
    if "=" not in value:
        raise ValueError("arm must use NAME=SELECTION,AUDIT,SCOUT")
    arm_id, raw_paths = value.split("=", 1)
    paths = [Path(item).expanduser() for item in raw_paths.split(",")]
    if len(paths) != 3 or not arm_id:
        raise ValueError("arm must provide exactly selection, audit, and Scout paths")
    selection, audit, scout = (_read_json(path) for path in paths)
    return {
        "arm_id": arm_id,
        "selection_path": str(paths[0]),
        "audit_path": str(paths[1]),
        "scout_path": str(paths[2]),
        "selection": selection,
        "audit": audit,
        "scout": scout,
    }


def _normalize_arm(arm: Mapping[str, Any]) -> dict[str, Any]:
    selection = _mapping(arm.get("selection"), "selection")
    audit = _mapping(arm.get("audit"), "audit")
    scout = _mapping(arm.get("scout"), "scout")
    ranking = _ranking_map(scout)
    priority = tuple(
        _subset_id(value) for value in scout.get("frozen_priority_order", [])
    )
    if not priority:
        raise ValueError(f"{arm['arm_id']}: Scout report has no priority order")
    lead = ranking[priority[0]]
    distribution = _mapping_or_empty(audit.get("distribution_comparisons"))
    max_ks = _max_distribution_field(distribution, "ks_distance")
    max_smd = _max_distribution_field(distribution, "standardized_mean_difference")
    return {
        "arm_id": str(arm["arm_id"]),
        "selection_path": str(arm["selection_path"]),
        "audit_path": str(arm["audit_path"]),
        "scout_path": str(arm["scout_path"]),
        "sampling_strategy": selection.get("sampling_strategy", "legacy_unspecified"),
        "sampling_seed": selection.get("sampling_seed"),
        "probe_size": int(selection.get("probe_final_size", 0)),
        "selection_hash": selection.get("selection_hash"),
        "representation_hash": selection.get("representation_hash"),
        "audit_status": audit.get("status"),
        "audit_max_ks_distance": max_ks,
        "audit_max_standardized_mean_difference": max_smd,
        "pair_coverage": _mapping_or_empty(audit.get("element_pair_coverage")),
        "top_subset": priority[0],
        "top_subset_probe_pcc": float(lead["nominal_primary_score"]),
        "top_subset_bootstrap_ci": [
            float(lead["bootstrap_ci_low"]),
            float(lead["bootstrap_ci_high"]),
        ],
        "top_subset_estimated_cost_seconds": float(
            lead["estimated_full_acquisition_wall_seconds"]
        ),
        "top_tier": [
            _subset_id(value) for value in scout.get("top_tier", [])
        ],
        "frozen_priority_order": list(priority),
        "ranking_scores": ranking,
    }


def _ranking_agreement(reference: Mapping[str, Any], arm: Mapping[str, Any]) -> dict[str, Any]:
    reference_order = list(reference["frozen_priority_order"])
    arm_order = list(arm["frozen_priority_order"])
    common = [item for item in reference_order if item in set(arm_order)]
    left_ranks = np.asarray([reference_order.index(item) for item in common], dtype=float)
    right_ranks = np.asarray([arm_order.index(item) for item in common], dtype=float)
    correlation = _pearson(left_ranks, right_ranks)
    reference_tier = set(reference["top_tier"])
    arm_tier = set(arm["top_tier"])
    union = reference_tier | arm_tier
    return {
        "top_subset_exact_match": reference["top_subset"] == arm["top_subset"],
        "top_tier_jaccard": float(len(reference_tier & arm_tier) / len(union)) if union else 1.0,
        "priority_rank_correlation": correlation,
        "common_subset_count": len(common),
    }


def _random_arm_summary(rows: list[Mapping[str, Any]], *, baseline: str) -> dict[str, Any]:
    random_rows = [row for row in rows if row["arm_id"] != baseline]
    if not random_rows:
        return {"arm_count": 0}
    return {
        "arm_count": len(random_rows),
        "top_subset_probe_pcc_mean": float(
            np.mean([row["top_subset_probe_pcc"] for row in random_rows])
        ),
        "top_subset_probe_pcc_std": float(
            np.std([row["top_subset_probe_pcc"] for row in random_rows], ddof=0)
        ),
        "top_subset_exact_match_frequency": float(
            np.mean(
                [
                    row["agreement_with_baseline"]["top_subset_exact_match"]
                    for row in random_rows
                ]
            )
        ),
        "top_tier_jaccard_mean": float(
            np.mean(
                [row["agreement_with_baseline"]["top_tier_jaccard"] for row in random_rows]
            )
        ),
        "priority_rank_correlation_mean": float(
            np.mean(
                [
                    row["agreement_with_baseline"]["priority_rank_correlation"]
                    for row in random_rows
                ]
            )
        ),
    }


def _ranking_map(scout: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    rows = scout.get("consensus_metrics")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Scout report has no consensus_metrics")
    result = {}
    for row in rows:
        item = _mapping(row, "consensus metric")
        subset_id = str(item.get("subset_id") or _subset_id(item.get("invariants", [])))
        result[subset_id] = item
    return result


def _max_distribution_field(
    comparisons: Mapping[str, Any], field: str,
) -> float | None:
    values = [
        float(item[field])
        for item in comparisons.values()
        if isinstance(item, Mapping) and item.get(field) is not None
    ]
    return max(values) if values else None


def _subset_id(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "+".join(str(item) for item in value)
    if isinstance(value, tuple):
        return "+".join(str(item) for item in value)
    raise ValueError("subset must be a string or list")


def _pearson(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2 or np.std(left) == 0 or np.std(right) == 0:
        return 1.0
    return float(np.corrcoef(left, right)[0, 1])


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    return _mapping(data, str(path))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return dict(value)


def _mapping_or_empty(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _render_markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Probe Selection Ablation",
        "",
        "This is a post-hoc ablation summary and cannot modify the production selector.",
        "",
        "| Arm | Strategy | Probe size | Audit | Top subset | Probe PCC | Top-tier Jaccard | Rank correlation |",
        "| --- | --- | ---: | --- | --- | ---: | ---: | ---: |",
    ]
    for row in summary["arms"]:
        agreement = row["agreement_with_baseline"]
        lines.append(
            "| {arm} | {strategy} | {size} | {audit} | {subset} | {pcc:.4f} | {jaccard:.3f} | {correlation:.3f} |".format(
                arm=row["arm_id"],
                strategy=row["sampling_strategy"],
                size=row["probe_size"],
                audit=row["audit_status"],
                subset=row["top_subset"],
                pcc=row["top_subset_probe_pcc"],
                jaccard=agreement["top_tier_jaccard"],
                correlation=agreement["priority_rank_correlation"],
            )
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
