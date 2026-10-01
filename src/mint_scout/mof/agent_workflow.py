"""Deterministic MOF agent-workflow preparation over legacy feature artifacts.

This module deliberately does not implement MOF topology mathematics.  It
inspects feature files produced by the audited legacy programs, selects a
frozen train-only probe, and writes the state needed by later Scout jobs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable

import numpy as np

from .legacy import LEGACY_MOF_TOOLS, topology_for


WORKFLOW_SCHEMA = "mint-agent.mof-agent-workflow.v1"
DEFAULT_CATEGORY_SCHEMA_ID = "mof_legacy_c0_c7_call_v1"


@dataclass(frozen=True)
class MofAgentRecord:
    sample_id: str
    target: float
    split: str
    cif_path: Path


def load_agent_manifest(path: str | Path) -> tuple[MofAgentRecord, ...]:
    """Load a portable MOF CSV manifest and validate its train-only labels."""
    source = Path(path)
    with source.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"sample_id", "target", "split", "cif_path"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"MOF manifest must contain {sorted(required)}: {source}")
        records = []
        seen = set()
        for row in reader:
            sample_id = str(row["sample_id"]).strip()
            if not sample_id or sample_id in seen:
                raise ValueError(f"MOF manifest has missing or duplicate sample_id: {sample_id!r}")
            target = float(row["target"])
            if not math.isfinite(target):
                raise ValueError(f"MOF manifest target must be finite: {sample_id}")
            split = str(row["split"]).strip().lower()
            if split not in {"train", "validation", "test"}:
                raise ValueError(f"Unsupported MOF split {split!r} for {sample_id}")
            seen.add(sample_id)
            records.append(MofAgentRecord(sample_id, target, split, Path(row["cif_path"])))
    if not records:
        raise ValueError(f"MOF manifest is empty: {source}")
    return tuple(records)


def select_stratified_probe(
    records: Iterable[MofAgentRecord], *, probe_size: int, strata: int, seed: int
) -> tuple[str, ...]:
    """Choose an exact-size, deterministic probe balanced across target ranks."""
    train = sorted((row for row in records if row.split == "train"), key=lambda row: (row.target, row.sample_id))
    if probe_size < 2:
        raise ValueError("probe_size must be at least two")
    if len(train) < probe_size:
        raise ValueError(f"probe_size={probe_size} exceeds train records={len(train)}")
    bucket_count = min(max(1, strata), probe_size, len(train))
    buckets: list[list[MofAgentRecord]] = [[] for _ in range(bucket_count)]
    for index, record in enumerate(train):
        buckets[min(bucket_count - 1, index * bucket_count // len(train))].append(record)

    base, remainder = divmod(probe_size, bucket_count)
    selected: list[MofAgentRecord] = []
    selected_ids: set[str] = set()
    for index, bucket in enumerate(buckets):
        shuffled = list(bucket)
        random.Random(seed + index).shuffle(shuffled)
        for record in shuffled[: base + int(index < remainder)]:
            selected.append(record)
            selected_ids.add(record.sample_id)

    if len(selected) < probe_size:
        remaining = [record for record in train if record.sample_id not in selected_ids]
        random.Random(seed).shuffle(remaining)
        selected.extend(remaining[: probe_size - len(selected)])
    return tuple(sorted(record.sample_id for record in selected))


def prepare_workflow(
    *,
    manifest_path: str | Path,
    feature_dir: str | Path,
    result_dir: str | Path,
    property_name: str,
    probe_size: int,
    probe_strata: int,
    seed: int,
    category_schema_id: str = DEFAULT_CATEGORY_SCHEMA_ID,
) -> dict[str, Any]:
    """Create a MOF workflow snapshot using only training data and artifacts."""
    records = load_agent_manifest(manifest_path)
    train = tuple(record for record in records if record.split == "train")
    if not train:
        raise ValueError("MOF agent workflow requires at least one train record")
    probe_ids = select_stratified_probe(train, probe_size=probe_size, strata=probe_strata, seed=seed)
    expected_ids = {record.sample_id for record in train}
    feature_root = Path(feature_dir)
    results_root = Path(result_dir)
    tools = [
        _tool_status(
            tool_name=tool_name,
            property_name=property_name,
            feature_root=feature_root,
            results_root=results_root,
            expected_ids=expected_ids,
            probe_ids=set(probe_ids),
        )
        for tool_name in LEGACY_MOF_TOOLS
    ]
    ready_for_probe = [tool["tool_name"] for tool in tools if tool["probe_feature_ready"]]
    complete_tools = [tool["tool_name"] for tool in tools if tool["full_feature_ready"]]
    pending_tools = [tool["tool_name"] for tool in tools if not tool["full_feature_ready"]]
    metrics_available = [tool["tool_name"] for tool in tools if tool["standard_metrics"] is not None]
    return {
        "report_schema": WORKFLOW_SCHEMA,
        "system_type": "mof",
        "task_type": "regression",
        "property": property_name,
        "primary_metric": "r2_standard",
        "category_schema": {
            "schema_id": category_schema_id,
            "mode": "frozen_legacy_category_specific",
            "adaptive_category_design": False,
        },
        "dataset_audit": {
            "manifest_path": str(Path(manifest_path)),
            "record_count": len(records),
            "train_count": len(train),
            "cif_files_present": sum(record.cif_path.exists() for record in train),
            "target": _target_summary(train),
        },
        "probe_selection": {
            "evidence_scope": "train_only",
            "strategy": "target_rank_stratified",
            "probe_size": len(probe_ids),
            "requested_probe_size": probe_size,
            "target_strata": min(probe_strata, probe_size, len(train)),
            "seed": seed,
            "sample_ids": list(probe_ids),
        },
        "tools": tools,
        "workflow_status": _workflow_status(ready_for_probe, complete_tools, pending_tools),
        "next_actions": _next_actions(ready_for_probe, complete_tools, pending_tools, metrics_available),
    }


def write_workflow(report: dict[str, Any], output: str | Path) -> tuple[Path, Path]:
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    probe_path = destination.with_name(destination.stem + "_probe_ids.txt")
    probe_path.write_text("\n".join(report["probe_selection"]["sample_ids"]) + "\n", encoding="utf-8")
    return destination, probe_path


def _tool_status(
    *,
    tool_name: str,
    property_name: str,
    feature_root: Path,
    results_root: Path,
    expected_ids: set[str],
    probe_ids: set[str],
) -> dict[str, Any]:
    topology = topology_for(tool_name)
    feature_path = feature_root / topology / property_name
    computed_ids = {path.stem for path in feature_path.glob("*.npy")} if feature_path.is_dir() else set()
    feature_dimension = _feature_dimension(feature_path, computed_ids)
    available_ids = computed_ids & expected_ids
    missing_ids = expected_ids - computed_ids
    stem = f"{property_name}_{topology}_repeat0_gbt"
    standard_metrics_path = results_root / f"{stem}_standard_metrics.json"
    standard_metrics = _read_json(standard_metrics_path) if standard_metrics_path.exists() else None
    return {
        "tool_name": tool_name,
        "topology": topology,
        "feature_directory": str(feature_path),
        "computed_feature_count": len(available_ids),
        "expected_train_feature_count": len(expected_ids),
        "missing_feature_count": len(missing_ids),
        "feature_dimension": feature_dimension,
        "full_feature_ready": not missing_ids,
        "probe_feature_ready": probe_ids.issubset(computed_ids),
        "standard_metrics": standard_metrics,
        "standard_metrics_path": str(standard_metrics_path) if standard_metrics else None,
    }


def _feature_dimension(feature_path: Path, computed_ids: set[str]) -> int | None:
    if not computed_ids:
        return None
    path = feature_path / f"{min(computed_ids)}.npy"
    try:
        return int(np.load(path, mmap_mode="r").size)
    except (OSError, ValueError):
        return None


def _target_summary(records: Iterable[MofAgentRecord]) -> dict[str, float]:
    values = sorted(record.target for record in records)
    return {
        "min": values[0],
        "max": values[-1],
        "mean": fmean(values),
        "median": _quantile(values, 0.5),
        "q10": _quantile(values, 0.1),
        "q90": _quantile(values, 0.9),
    }


def _quantile(values: list[float], fraction: float) -> float:
    position = (len(values) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def _workflow_status(ready: list[str], complete: list[str], pending: list[str]) -> str:
    if not ready:
        return "WAITING_FOR_PROBE_FEATURES"
    if pending:
        return "PROBE_SCOUT_READY_PARTIAL_TOOLSET"
    if len(complete) == len(LEGACY_MOF_TOOLS):
        return "PROBE_SCOUT_READY_ALL_TOOLS"
    return "WAITING_FOR_FULL_FEATURES"


def _next_actions(ready: list[str], complete: list[str], pending: list[str], metrics_available: list[str]) -> list[str]:
    actions = []
    if ready:
        actions.append(f"Run frozen-probe GBT Scout for available tools: {', '.join(ready)}.")
    if pending:
        actions.append(f"Continue cached legacy feature generation for: {', '.join(pending)}.")
    if metrics_available:
        actions.append("Treat completed full-data legacy CV metrics as parity baselines, not final agent selection evidence.")
    if len(complete) == len(LEGACY_MOF_TOOLS):
        actions.append("Compare all methods on the same frozen probe using the hierarchical selection rule.")
    return actions


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a deterministic MOF agent workflow from legacy artifacts")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--property", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--probe-size", type=int, default=750)
    parser.add_argument("--probe-strata", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--category-schema-id", default=DEFAULT_CATEGORY_SCHEMA_ID)
    args = parser.parse_args()
    report = prepare_workflow(
        manifest_path=args.manifest,
        feature_dir=args.feature_dir,
        result_dir=args.result_dir,
        property_name=args.property,
        probe_size=args.probe_size,
        probe_strata=args.probe_strata,
        seed=args.seed,
        category_schema_id=args.category_schema_id,
    )
    report_path, probe_path = write_workflow(report, args.output)
    print(f"workflow={report_path}")
    print(f"probe_ids={probe_path}")
    print(f"status={report['workflow_status']}")


if __name__ == "__main__":
    main()
