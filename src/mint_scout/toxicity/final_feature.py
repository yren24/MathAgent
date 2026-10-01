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


def run_final_feature_task(
    *,
    plan_path: str | Path,
    task_index: int,
    dataset_manifest_path: str | Path,
    legacy_root: str | Path,
    molecule_dirs: Sequence[str | Path],
    feature_root: str | Path,
    dataset_id: str = "LD50",
) -> dict[str, Any]:
    plan = _load_object(plan_path, "toxicity final plan")
    tasks = plan.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("toxicity final plan has no feature tasks")
    if task_index < 0:
        raise ValueError("task_index must be non-negative")
    if task_index >= len(tasks):
        return {
            "status": "SKIPPED",
            "task_index": task_index,
            "reason": "task_index_outside_final_plan",
            "task_count": len(tasks),
        }
    task = tasks[task_index]
    if not isinstance(task, Mapping):
        raise ValueError(f"toxicity final task {task_index} must be an object")
    split = str(task.get("split") or "").lower()
    if split not in {"train", "test"}:
        raise ValueError(f"unsupported toxicity final split: {split!r}")

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
    samples = tuple(dataset.samples_in_split(SampleSplit(split)))
    if not samples:
        raise ValueError(f"toxicity final split {split!r} is empty")
    if any(sample.target is None for sample in samples):
        raise ValueError(f"every toxicity {split} sample must have a label")

    spec = RepresentationSpec.read(Path(str(task["representation_spec"])))
    spec.assert_frozen()
    if spec.spec_hash != str(task["representation_hash"]):
        raise ValueError("toxicity final feature representation hash mismatch")
    invariant = str(task["invariant"]).upper()
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
    failures = computed = cached = 0
    with partial_path.open("w", encoding="utf-8") as handle:
        for batch_index, sample in enumerate(samples):
            started = time.perf_counter()
            try:
                result = tool.compute_with_spec(sample.sample_id, invariant, spec)
                elapsed = time.perf_counter() - started
                status = result.status
                computed += int(status == "computed")
                cached += int(status == "cached")
                entry = {
                    "batch_index": batch_index,
                    "sample_id": sample.sample_id,
                    "split": split,
                    "label": sample.target,
                    "invariant": invariant,
                    "feature_signature": task["feature_signature"],
                    "representation_hash": spec.spec_hash,
                    "status": status,
                    "output_path": str(result.output_path),
                    "wall_seconds": result.cost.wall_seconds if result.cost else elapsed,
                    "cpu_seconds": result.cost.cpu_seconds if result.cost else None,
                    "error_type": None,
                    "error": None,
                    "traceback": None,
                }
            except Exception as exc:
                failures += 1
                entry = {
                    "batch_index": batch_index,
                    "sample_id": sample.sample_id,
                    "split": split,
                    "label": sample.target,
                    "invariant": invariant,
                    "feature_signature": task["feature_signature"],
                    "representation_hash": spec.spec_hash,
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
            f"toxicity final feature task {task_index} failed for {failures} samples; "
            f"partial manifest: {partial_path}"
        )
    partial_path.replace(manifest_path)
    return {
        "status": "COMPLETE",
        "task_index": task_index,
        "split": split,
        "invariant": invariant,
        "sample_count": len(samples),
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
        description="Generate one full toxicity train/test feature manifest."
    )
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--legacy-root", type=Path, required=True)
    parser.add_argument("--molecule-dir", type=Path, action="append", required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--dataset-id", default="LD50")
    args = parser.parse_args(argv)
    result = run_final_feature_task(
        plan_path=args.plan,
        task_index=args.task_index,
        dataset_manifest_path=args.dataset_manifest,
        legacy_root=args.legacy_root,
        molecule_dirs=args.molecule_dir,
        feature_root=args.feature_root,
        dataset_id=args.dataset_id,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
