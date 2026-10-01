from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

from mint_scout.data.manifest_io import load_dataset_manifest
from mint_scout.representation import RepresentationSpec
from mint_scout.schemas import SampleSplit
from mint_scout.toxicity.legacy import ToxicityFeatureTool, ToxicityToolConfig


def run_feature_plan_task(
    *,
    plan_path: str | Path,
    task_index: int,
    probe_selection_path: str | Path,
    dataset_manifest_path: str | Path,
    legacy_root: str | Path,
    molecule_dirs: Sequence[str | Path],
    feature_root: str | Path,
    dataset_id: str = "LD50",
) -> dict[str, Any]:
    plan = _load_object(plan_path, "toxicity feature plan")
    tasks = plan.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("toxicity feature plan has no tasks")
    if task_index < 0:
        raise ValueError("task_index must be non-negative")
    if task_index >= len(tasks):
        return {
            "status": "SKIPPED",
            "task_index": task_index,
            "reason": "task_index_outside_deduplicated_plan",
            "unique_task_count": len(tasks),
        }
    task = tasks[task_index]
    if not isinstance(task, Mapping):
        raise ValueError(f"toxicity feature task {task_index} must be an object")
    if task.get("execute") is False:
        manifest_path = Path(str(task["manifest_path"]))
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"reused toxicity feature manifest does not exist: {manifest_path}"
            )
        return {
            "status": "REUSED",
            "task_index": task_index,
            "invariant": str(task["invariant"]),
            "feature_signature": str(task["feature_signature"]),
            "manifest_path": str(manifest_path),
        }

    probe = _load_object(probe_selection_path, "toxicity Probe selection")
    if probe.get("test_labels_used") is not False:
        raise ValueError("toxicity Probe selection must explicitly exclude test labels")
    sample_ids = probe.get("probe_sample_ids")
    if not isinstance(sample_ids, list) or not sample_ids:
        raise ValueError("toxicity Probe selection has no probe_sample_ids")
    sample_ids = [str(sample_id) for sample_id in sample_ids]

    dataset = load_dataset_manifest(
        dataset_manifest_path,
        dataset_id=dataset_id,
        system_type="small_molecule",
        columns={
            "identifiers": {
                "smiles": "smiles",
                "source_filename": "source_filename",
            }
        },
        label_name=dataset_id,
    )
    training = {sample.sample_id: sample for sample in dataset.samples_in_split(SampleSplit.TRAIN)}
    missing = sorted(set(sample_ids) - set(training))
    if missing:
        raise ValueError(f"Probe samples are absent from the training split: {missing[:10]}")
    if any(training[sample_id].target is None for sample_id in sample_ids):
        raise ValueError("every toxicity Probe sample must have a training label")

    spec_path = Path(str(task["representation_spec"]))
    spec = RepresentationSpec.read(spec_path)
    expected_hash = str(task["representation_hash"])
    if spec.spec_hash != expected_hash:
        raise ValueError("toxicity feature task representation hash mismatch")
    invariant = str(task["invariant"])
    tool = ToxicityFeatureTool(
        ToxicityToolConfig(
            legacy_root=Path(legacy_root),
            molecule_dirs=tuple(Path(path) for path in molecule_dirs),
            output_root=Path(feature_root),
            dataset_id=dataset_id,
        )
    )
    manifest_path = Path(str(task["manifest_path"]))
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = manifest_path.with_suffix(manifest_path.suffix + ".partial")
    failures = 0
    computed = 0
    cached = 0
    with partial_path.open("w", encoding="utf-8") as handle:
        for batch_index, sample_id in enumerate(sample_ids):
            started = time.perf_counter()
            try:
                result = tool.compute_with_spec(sample_id, invariant, spec)
                elapsed = time.perf_counter() - started
                status = result.status
                computed += int(status == "computed")
                cached += int(status == "cached")
                entry = {
                    "batch_index": batch_index,
                    "sample_id": sample_id,
                    "split": "train",
                    "label": training[sample_id].target,
                    "invariant": invariant,
                    "feature_signature": task["feature_signature"],
                    "candidate_ids": list(task["candidate_ids"]),
                    "canonical_candidate_id": task["canonical_candidate_id"],
                    "representation_hash": expected_hash,
                    "status": status,
                    "output_path": str(result.output_path),
                    "wall_seconds": (
                        result.cost.wall_seconds if result.cost else elapsed
                    ),
                    "cpu_seconds": (
                        result.cost.cpu_seconds if result.cost else None
                    ),
                    "error_type": None,
                    "error": None,
                    "traceback": None,
                }
            except Exception as exc:
                failures += 1
                entry = {
                    "batch_index": batch_index,
                    "sample_id": sample_id,
                    "split": "train",
                    "label": training[sample_id].target,
                    "invariant": invariant,
                    "feature_signature": task["feature_signature"],
                    "candidate_ids": list(task["candidate_ids"]),
                    "canonical_candidate_id": task["canonical_candidate_id"],
                    "representation_hash": expected_hash,
                    "status": "failed",
                    "output_path": None,
                    "wall_seconds": time.perf_counter() - started,
                    "cpu_seconds": None,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            handle.write(json.dumps(entry, sort_keys=True) + "\n")
            handle.flush()
    if failures:
        raise RuntimeError(
            f"toxicity feature task {task_index} failed for {failures} samples; "
            f"partial manifest: {partial_path}"
        )
    partial_path.replace(manifest_path)
    return {
        "status": "COMPLETE",
        "task_index": task_index,
        "invariant": invariant,
        "feature_signature": task["feature_signature"],
        "candidate_ids": list(task["candidate_ids"]),
        "sample_count": len(sample_ids),
        "computed_count": computed,
        "cached_count": cached,
        "manifest_path": str(manifest_path),
    }


def _load_object(path: str | Path, name: str) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain a JSON object")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run one deduplicated toxicity Probe feature task."
    )
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--probe-selection", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--legacy-root", type=Path, required=True)
    parser.add_argument("--molecule-dir", type=Path, action="append", required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--dataset-id", default="LD50")
    args = parser.parse_args(argv)
    report = run_feature_plan_task(
        plan_path=args.plan,
        task_index=args.task_index,
        probe_selection_path=args.probe_selection,
        dataset_manifest_path=args.dataset_manifest,
        legacy_root=args.legacy_root,
        molecule_dirs=args.molecule_dir,
        feature_root=args.feature_root,
        dataset_id=args.dataset_id,
    )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
