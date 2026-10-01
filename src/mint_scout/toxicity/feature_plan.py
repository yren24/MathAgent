from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from mint_scout.invariants.manifest import stable_hash
from mint_scout.representation import RepresentationSpec
from mint_scout.toxicity.legacy import TOXICITY_INVARIANTS
from mint_scout.toxicity.selection_contract import ToxicitySelectionContract


FEATURE_PLAN_SCHEMA = "mint-agent.toxicity-feature-plan.v1"


def build_toxicity_feature_plan(
    *,
    design_report: str | Path,
    representation_dir: str | Path,
    manifest_dir: str | Path,
    selection: ToxicitySelectionContract = ToxicitySelectionContract(),
) -> dict[str, Any]:
    report_path = Path(design_report).expanduser().resolve()
    spec_dir = Path(representation_dir).expanduser().resolve()
    output_dir = Path(manifest_dir).expanduser().resolve()
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("toxicity design report must contain a JSON object")
    scientific_controls = payload.get("scientific_controls", {})
    if not isinstance(scientific_controls, Mapping):
        raise ValueError("toxicity design scientific_controls must be an object")
    design_metric = scientific_controls.get("primary_metric")
    if design_metric is not None and str(design_metric).upper() != selection.primary_metric:
        raise ValueError("toxicity design primary_metric differs from requested metric")
    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("toxicity design report has no candidates")

    grouped: dict[str, dict[str, Any]] = {}
    candidate_tasks: dict[str, dict[str, str]] = {}
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            raise ValueError("toxicity design candidate must be an object")
        candidate_id = str(candidate.get("candidate_id") or "")
        if not candidate_id:
            raise ValueError("toxicity design candidate is missing candidate_id")
        spec_path = spec_dir / f"{candidate_id}.json"
        spec = RepresentationSpec.read(spec_path)
        spec.assert_frozen()
        expected_hash = str(candidate.get("representation_hash") or "")
        if spec.spec_hash != expected_hash:
            raise ValueError(
                f"toxicity representation hash mismatch for {candidate_id}"
            )
        candidate_tasks[candidate_id] = {}
        for invariant in selection.requested_invariants:
            signature = toxicity_feature_signature(spec, invariant)
            candidate_tasks[candidate_id][invariant] = signature
            task = grouped.setdefault(
                signature,
                {
                    "feature_signature": signature,
                    "invariant": invariant,
                    "canonical_candidate_id": candidate_id,
                    "representation_hash": spec.spec_hash,
                    "representation_spec": str(spec_path),
                    "candidate_ids": [],
                },
            )
            if task["invariant"] != invariant:
                raise AssertionError("feature signature collision across invariants")
            task["candidate_ids"].append(candidate_id)

    tasks = []
    for task_index, signature in enumerate(sorted(grouped)):
        task = grouped[signature]
        task["task_index"] = task_index
        task["candidate_ids"] = sorted(task["candidate_ids"])
        task["manifest_path"] = str(output_dir / f"{signature}.jsonl")
        tasks.append(task)
    result: dict[str, Any] = {
        "report_schema": FEATURE_PLAN_SCHEMA,
        "status": "READY",
        "evidence_scope": "train_probe_only",
        "test_structures_used": False,
        "test_labels_used": False,
        "design_report": str(report_path),
        "candidate_count": len(candidate_tasks),
        "selection_contract": selection.to_dict(),
        "naive_task_count": len(candidate_tasks) * len(selection.requested_invariants),
        "unique_task_count": len(tasks),
        "deduplicated_task_count": (
            len(candidate_tasks) * len(selection.requested_invariants) - len(tasks)
        ),
        "candidate_feature_signatures": candidate_tasks,
        "tasks": tasks,
    }
    result["plan_hash"] = stable_hash(result)
    return result


def toxicity_feature_signature(
    spec: RepresentationSpec,
    invariant: str,
) -> str:
    name = invariant.upper()
    if name not in TOXICITY_INVARIANTS:
        raise KeyError(f"unknown toxicity invariant: {invariant}")
    spec.assert_frozen()
    profile = spec.filtration_profiles[name]
    return stable_hash(
        {
            "adapter": "toxicity_legacy_adapter_v1",
            "invariant": name,
            "pair_order": [list(pair) for pair in spec.pair_order],
            "filtration_profile": {
                "scale_kind": profile.scale_kind,
                "start": profile.start,
                "stop": profile.stop,
                "step": profile.step,
                "num_points": profile.num_points,
            },
        }
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Plan deduplicated toxicity Probe feature tool calls."
    )
    parser.add_argument("--design-report", type=Path, required=True)
    parser.add_argument("--representation-dir", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--task-config", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    plan = build_toxicity_feature_plan(
        design_report=args.design_report,
        representation_dir=args.representation_dir,
        manifest_dir=args.manifest_dir,
        selection=(
            ToxicitySelectionContract.from_task(args.task_config)
            if args.task_config is not None
            else ToxicitySelectionContract()
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"status={plan['status']} candidates={plan['candidate_count']} "
        f"unique_tasks={plan['unique_task_count']} "
        f"deduplicated={plan['deduplicated_task_count']} output={args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
