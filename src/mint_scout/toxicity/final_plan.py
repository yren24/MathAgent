from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from mint_scout.invariants.manifest import stable_hash
from mint_scout.toxicity.legacy import TOXICITY_INVARIANTS
from mint_scout.toxicity.selection_contract import (
    ToxicitySelectionContract,
    methods_from_plan,
)


FINAL_PLAN_SCHEMA = "mint-agent.toxicity-final-plan.v1"


def build_final_plan(
    *,
    scout_report: str | Path,
    feature_plan: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    scout_path = Path(scout_report).expanduser().resolve()
    feature_plan_path = Path(feature_plan).expanduser().resolve()
    root = Path(output_root).expanduser().resolve()
    scout = _load_object(scout_path, "toxicity Probe Scout report")
    source_plan = _load_object(feature_plan_path, "toxicity feature plan")
    if scout.get("status") != "COMPLETE":
        raise ValueError("toxicity Probe Scout report is not complete")
    if scout.get("test_labels_used") is not False:
        raise ValueError("toxicity final selection must be frozen without test labels")
    if scout.get("feature_plan_hash") != source_plan.get("plan_hash"):
        raise ValueError("toxicity Probe Scout report and feature plan do not match")

    methods = methods_from_plan(source_plan)
    raw_contract = source_plan.get("selection_contract")
    selection = (
        ToxicitySelectionContract.from_mapping(raw_contract)
        if isinstance(raw_contract, Mapping)
        else ToxicitySelectionContract(requested_invariants=methods)
    )
    if selection.requested_invariants != methods:
        raise ValueError("toxicity final plan method contract differs from feature plan")
    if scout.get("primary_metric") is not None and scout["primary_metric"] != selection.primary_metric:
        raise ValueError("toxicity Probe Scout metric differs from requested metric")
    if scout.get("requested_invariants") is not None and set(scout["requested_invariants"]) != set(methods):
        raise ValueError("toxicity Probe Scout methods differ from requested methods")
    priority = scout.get("priority_order")
    if priority is None and not isinstance(raw_contract, Mapping):
        priority = [scout.get("nominal_best", ())]
    if not isinstance(priority, list) or not priority:
        raise ValueError("toxicity Probe Scout report has no priority_order")
    candidates = [tuple(str(value).upper() for value in subset) for subset in priority]
    if any(not subset or len(set(subset)) != len(subset) or set(subset) - set(methods) for subset in candidates):
        raise ValueError("toxicity Probe Scout priority contains invalid method subsets")
    if len(set(candidates)) != len(candidates):
        raise ValueError("toxicity Probe Scout priority contains duplicate subsets")
    summaries = {
        tuple(str(value).upper() for value in row["invariants"]): row
        for row in scout.get("subset_summaries", ())
    }
    selected = candidates[0]
    selected_index = 0
    if selection.selection_objective == "satisfy_target":
        if any(subset not in summaries for subset in candidates):
            raise ValueError("toxicity target selection requires scores for every priority candidate")
        for index, subset in enumerate(candidates):
            score = float(summaries[subset]["nominal_primary_score"])
            if math.isfinite(score) and score >= selection.user_target:
                selected = subset
                selected_index = index
                break
    selected_score = (
        float(summaries[selected]["nominal_primary_score"])
        if selected in summaries else None
    )
    probe_target_met = (
        selected_score is not None and math.isfinite(selected_score)
        and selected_score >= selection.user_target
        if selection.user_target is not None else None
    )
    unknown = sorted(set(selected) - set(TOXICITY_INVARIANTS))
    if unknown:
        raise ValueError(f"toxicity Probe Scout selected unknown methods: {unknown}")
    if len(set(selected)) != len(selected):
        raise ValueError("toxicity Probe Scout nominal_best contains duplicates")

    source_by_method: dict[str, Mapping[str, Any]] = {}
    for task in source_plan.get("tasks", ()):
        if not isinstance(task, Mapping):
            raise ValueError("toxicity feature plan task must be an object")
        method = str(task.get("invariant") or "").upper()
        if method in selected:
            if method in source_by_method:
                raise ValueError(f"toxicity feature plan repeats selected method {method}")
            source_by_method[method] = task
    missing = sorted(set(selected) - set(source_by_method))
    if missing:
        raise ValueError(f"toxicity feature plan is missing selected methods: {missing}")

    tasks: list[dict[str, Any]] = []
    for method in selected:
        source = source_by_method[method]
        for split in ("train", "test"):
            task_index = len(tasks)
            signature = str(source["feature_signature"])
            tasks.append(
                {
                    "task_index": task_index,
                    "invariant": method,
                    "split": split,
                    "feature_signature": signature,
                    "representation_hash": str(source["representation_hash"]),
                    "representation_spec": str(source["representation_spec"]),
                    "manifest_path": str(
                        root / "manifests" / split / f"{signature}.jsonl"
                    ),
                }
            )

    result: dict[str, Any] = {
        "report_schema": FINAL_PLAN_SCHEMA,
        "status": "READY",
        "evidence_scope": "frozen_probe_selection_then_full_train_fixed_test",
        "selection_policy": "probe_priority_order_with_target" if selection.selection_objective == "satisfy_target" else "probe_priority_rank1",
        "selection_contract": selection.to_dict(),
        "primary_metric": selection.primary_metric,
        "selection_objective": selection.selection_objective,
        "user_target": selection.user_target,
        "probe_target_met": probe_target_met,
        "probe_primary_score": selected_score,
        "priority_index": selected_index,
        "target_check_scope": "train_probe_oof",
        "selected_subset": list(selected),
        "probe_scout_report": str(scout_path),
        "probe_scout_hash": str(scout.get("scout_hash") or ""),
        "source_feature_plan": str(feature_plan_path),
        "source_feature_plan_hash": str(source_plan.get("plan_hash") or ""),
        "test_labels_used_for_selection": False,
        "test_query_budget": 1,
        "task_count": len(tasks),
        "tasks": tasks,
    }
    result["plan_hash"] = stable_hash(result)
    return result


def _load_object(path: Path, name: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain a JSON object")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Freeze the Probe winner and plan full toxicity train/test features."
    )
    parser.add_argument("--scout-report", type=Path, required=True)
    parser.add_argument("--feature-plan", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = build_final_plan(
        scout_report=args.scout_report,
        feature_plan=args.feature_plan,
        output_root=args.output_root,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"status={report['status']} selected={'+'.join(report['selected_subset'])} "
        f"tasks={report['task_count']} output={args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
