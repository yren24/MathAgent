from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from mint_scout.filtration import FiltrationProfile
from mint_scout.invariants.manifest import stable_hash
from mint_scout.representation import RepresentationSpec
from mint_scout.toxicity.feature_plan import (
    FEATURE_PLAN_SCHEMA,
    toxicity_feature_signature,
)
from mint_scout.toxicity.selection_contract import methods_from_plan


REPAIR_SCHEMA = "mint-agent.toxicity-filtration-repair.v1"


@dataclass(frozen=True)
class ToxicityFiltrationRepairPolicy:
    extension_factor: float = 1.25
    shorten_buffer_points: int = 3
    minimum_retained_span_fraction: float = 0.50
    max_repair_iterations: int = 1

    def __post_init__(self) -> None:
        if self.extension_factor <= 1.0:
            raise ValueError("extension_factor must exceed one")
        if self.shorten_buffer_points < 1:
            raise ValueError("shorten_buffer_points must be positive")
        if not 0.0 < self.minimum_retained_span_fraction <= 1.0:
            raise ValueError("minimum_retained_span_fraction must be in (0, 1]")
        if self.max_repair_iterations != 1:
            raise ValueError("toxicity V1 permits exactly one repair iteration")


def build_toxicity_filtration_repair(
    *,
    representation_selection_path: str | Path,
    feature_plan_path: str | Path,
    feature_qc_path: str | Path,
    output_dir: str | Path,
    policy: ToxicityFiltrationRepairPolicy = ToxicityFiltrationRepairPolicy(),
) -> dict[str, Any]:
    selection = _load_object(
        representation_selection_path, "toxicity representation selection"
    )
    plan = _load_object(feature_plan_path, "toxicity feature plan")
    qc = _load_object(feature_qc_path, "toxicity feature QC")
    if selection.get("feature_plan_hash") != plan.get("plan_hash"):
        raise ValueError("toxicity representation selection and feature plan differ")
    if selection.get("feature_qc_hash") != qc.get("qc_hash"):
        raise ValueError("toxicity representation selection and feature QC differ")
    if selection.get("test_labels_used") is not False:
        raise ValueError("toxicity filtration repair must use train-only evidence")

    selected_id = str(selection["selected_candidate_id"])
    selected_spec = RepresentationSpec.read(selection["selected_representation"])
    signatures = _required_mapping(
        selection, "selected_method_feature_signatures"
    )
    methods = methods_from_plan(plan)
    if set(signatures) != set(methods):
        raise ValueError("toxicity repair method signatures differ from requested methods")
    qc_tasks = {
        str(task["feature_signature"]): task for task in _required_list(qc, "tasks")
    }
    old_tasks = {
        str(task["feature_signature"]): task
        for task in _required_list(plan, "tasks")
    }
    repaired_profiles = dict(selected_spec.filtration_profiles)
    method_repairs = {}
    for method in methods:
        signature = str(signatures[method])
        report = qc_tasks[signature]
        recommendation = str(report["filtration"]["recommendation"])
        old_profile = selected_spec.filtration_profiles[method]
        applied_recommendation = (
            recommendation if old_profile.scale_kind == "distance" else "KEEP"
        )
        new_profile = repair_filtration_profile(
            old_profile,
            recommendation=applied_recommendation,
            last_effective_point_index=report["filtration"].get(
                "last_effective_point_index"
            ),
            policy=policy,
            method=method,
        )
        repaired_profiles[method] = new_profile
        method_repairs[method] = {
            "observed_recommendation": recommendation,
            "applied_recommendation": applied_recommendation,
            "repair_eligible": old_profile.scale_kind == "distance",
            "changed": new_profile != old_profile,
            "old_profile": old_profile.to_dict(),
            "new_profile": new_profile.to_dict(),
            "source_feature_signature": signature,
        }
    changed_methods = [
        method for method, report in method_repairs.items() if report["changed"]
    ]
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    if not changed_methods:
        return _write_passthrough_repair(
            destination=destination,
            selected_id=selected_id,
            selected_spec=selected_spec,
            selected_representation_path=str(selection["selected_representation"]),
            signatures=signatures,
            methods=methods,
            selection_contract=plan.get("selection_contract"),
            old_tasks=old_tasks,
            method_repairs=method_repairs,
            policy=policy,
            selection=selection,
            qc=qc,
        )

    parameters = dict(selected_spec.parameters)
    parameters["filtration_repair"] = {
        "iteration": 1,
        "source_representation_hash": selected_spec.spec_hash,
        "source_selection_hash": selection.get("selection_hash"),
        "source_qc_hash": qc.get("qc_hash"),
        "policy": asdict(policy),
        "changed_methods": changed_methods,
    }
    provisional = replace(
        selected_spec,
        representation_id="toxicity_repair_pending",
        filtration_profiles=repaired_profiles,
        parameters=parameters,
        frozen=False,
    )
    repaired_spec = replace(
        provisional,
        representation_id=f"dataset_adaptive_{provisional.spec_hash}",
        frozen=True,
    )
    repaired_id = f"{selected_id}-filtration-repair-v1"
    spec_path = destination / f"{repaired_id}.json"
    repaired_spec.write(spec_path)
    manifest_dir = destination / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    tasks = []
    repaired_signatures = {}
    for task_index, method in enumerate(methods):
        old_signature = str(signatures[method])
        new_signature = toxicity_feature_signature(repaired_spec, method)
        repaired_signatures[method] = new_signature
        if new_signature == old_signature:
            task = dict(old_tasks[old_signature])
            task.update(
                {
                    "task_index": task_index,
                    "candidate_ids": [repaired_id],
                    "execute": False,
                    "reuse_reason": "method_representation_unchanged_by_repair",
                }
            )
        else:
            task = {
                "task_index": task_index,
                "feature_signature": new_signature,
                "invariant": method,
                "canonical_candidate_id": repaired_id,
                "representation_hash": repaired_spec.spec_hash,
                "representation_spec": str(spec_path),
                "candidate_ids": [repaired_id],
                "manifest_path": str(manifest_dir / f"{new_signature}.jsonl"),
                "execute": True,
            }
        tasks.append(task)
    repaired_plan: dict[str, Any] = {
        "report_schema": FEATURE_PLAN_SCHEMA,
        "status": "READY",
        "evidence_scope": "train_probe_only",
        "test_structures_used": False,
        "test_labels_used": False,
        "candidate_count": 1,
        "selection_contract": plan.get("selection_contract"),
        "naive_task_count": len(methods),
        "unique_task_count": len(tasks),
        "deduplicated_task_count": sum(task["execute"] is False for task in tasks),
        "candidate_feature_signatures": {repaired_id: repaired_signatures},
        "tasks": tasks,
    }
    repaired_plan["plan_hash"] = stable_hash(repaired_plan)
    plan_path = destination / "toxicity_repair_feature_plan.json"
    plan_path.write_text(
        json.dumps(repaired_plan, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    result: dict[str, Any] = {
        "report_schema": REPAIR_SCHEMA,
        "status": "PLANNED",
        "evidence_scope": "train_probe_only",
        "test_structures_used": False,
        "test_labels_used": False,
        "repair_iteration": 1,
        "max_repair_iterations": policy.max_repair_iterations,
        "policy": asdict(policy),
        "source_candidate_id": selected_id,
        "source_representation": str(selection["selected_representation"]),
        "source_representation_hash": selected_spec.spec_hash,
        "repaired_candidate_id": repaired_id,
        "repaired_representation": str(spec_path),
        "repaired_representation_hash": repaired_spec.spec_hash,
        "changed_methods": changed_methods,
        "reused_methods": [
            method for method in methods if method not in changed_methods
        ],
        "method_repairs": method_repairs,
        "feature_plan": str(plan_path),
        "feature_plan_hash": repaired_plan["plan_hash"],
    }
    result["repair_hash"] = stable_hash(result)
    report_path = destination / "toxicity_filtration_repair.json"
    report_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def _write_passthrough_repair(
    *,
    destination: Path,
    selected_id: str,
    selected_spec: RepresentationSpec,
    selected_representation_path: str,
    signatures: Mapping[str, Any],
    methods: Sequence[str],
    selection_contract: object,
    old_tasks: Mapping[str, Mapping[str, Any]],
    method_repairs: Mapping[str, Any],
    policy: ToxicityFiltrationRepairPolicy,
    selection: Mapping[str, Any],
    qc: Mapping[str, Any],
) -> dict[str, Any]:
    tasks = []
    repaired_signatures = {}
    for task_index, method in enumerate(methods):
        signature = str(signatures[method])
        repaired_signatures[method] = signature
        task = dict(old_tasks[signature])
        task.update(
            {
                "task_index": task_index,
                "candidate_ids": [selected_id],
                "execute": False,
                "reuse_reason": "filtration_repair_not_required",
            }
        )
        tasks.append(task)
    repaired_plan: dict[str, Any] = {
        "report_schema": FEATURE_PLAN_SCHEMA,
        "status": "READY",
        "evidence_scope": "train_probe_only",
        "test_structures_used": False,
        "test_labels_used": False,
        "candidate_count": 1,
        "selection_contract": selection_contract,
        "naive_task_count": len(methods),
        "unique_task_count": len(tasks),
        "deduplicated_task_count": len(tasks),
        "candidate_feature_signatures": {selected_id: repaired_signatures},
        "tasks": tasks,
    }
    repaired_plan["plan_hash"] = stable_hash(repaired_plan)
    plan_path = destination / "toxicity_repair_feature_plan.json"
    plan_path.write_text(
        json.dumps(repaired_plan, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    result: dict[str, Any] = {
        "report_schema": REPAIR_SCHEMA,
        "status": "NOT_REQUIRED",
        "evidence_scope": "train_probe_only",
        "test_structures_used": False,
        "test_labels_used": False,
        "repair_iteration": 0,
        "max_repair_iterations": policy.max_repair_iterations,
        "policy": asdict(policy),
        "source_candidate_id": selected_id,
        "source_representation": selected_representation_path,
        "source_representation_hash": selected_spec.spec_hash,
        "repaired_candidate_id": selected_id,
        "repaired_representation": selected_representation_path,
        "repaired_representation_hash": selected_spec.spec_hash,
        "changed_methods": [],
        "reused_methods": list(methods),
        "method_repairs": method_repairs,
        "feature_plan": str(plan_path),
        "feature_plan_hash": repaired_plan["plan_hash"],
        "selection_hash": selection.get("selection_hash"),
        "source_qc_hash": qc.get("qc_hash"),
    }
    result["repair_hash"] = stable_hash(result)
    report_path = destination / "toxicity_filtration_repair.json"
    report_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def repair_filtration_profile(
    profile: FiltrationProfile,
    *,
    recommendation: str,
    last_effective_point_index: int | None,
    policy: ToxicityFiltrationRepairPolicy,
    method: str,
) -> FiltrationProfile:
    action = recommendation.upper()
    if action == "KEEP":
        return profile
    if profile.scale_kind != "distance":
        raise ValueError("automatic toxicity repair supports distance filtration only")
    span = profile.stop - profile.start
    if action == "EXTEND":
        new_span = span * policy.extension_factor
    elif action == "SHORTEN":
        if last_effective_point_index is None:
            return profile
        target_index = min(
            profile.num_points - 1,
            last_effective_point_index + policy.shorten_buffer_points,
        )
        proposed_span = target_index * profile.step
        new_span = max(
            span * policy.minimum_retained_span_fraction, proposed_span
        )
        if new_span >= span:
            return profile
    else:
        raise ValueError(f"unsupported filtration repair recommendation: {action}")
    new_step = new_span / (profile.num_points - 1)
    return FiltrationProfile(
        scale_kind=profile.scale_kind,
        start=profile.start,
        stop=profile.start + new_span,
        step=new_step,
        num_points=profile.num_points,
        source=f"repair_v1_{method.lower()}_{action.lower()}",
    )


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
        description="Create one method-specific toxicity filtration repair."
    )
    parser.add_argument("--representation-selection", type=Path, required=True)
    parser.add_argument("--feature-plan", type=Path, required=True)
    parser.add_argument("--feature-qc", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    result = build_toxicity_filtration_repair(
        representation_selection_path=args.representation_selection,
        feature_plan_path=args.feature_plan,
        feature_qc_path=args.feature_qc,
        output_dir=args.output_dir,
    )
    print(
        f"status={result['status']} changed={','.join(result['changed_methods'])} "
        f"reused={','.join(result['reused_methods'])} "
        f"repair_hash={result['repair_hash']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
