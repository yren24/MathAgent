#!/usr/bin/env python3
"""Combine disjoint direct-GBT seed ensembles and score all late-fusion subsets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from mint_scout.config import load_yaml
from mint_scout.data.manifest_io import load_configured_protein_ligand_records
from mint_scout.evaluate_external_gbt import _metrics
from mint_scout.evaluation.ensemble import enumerate_nonempty_subsets, mean_aggregate
from mint_scout.evaluation.metrics import higher_is_better
from mint_scout.invariants.manifest import stable_hash


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-config", type=Path)
    parser.add_argument(
        "--test-feature-manifest",
        type=Path,
        help="Frozen feature manifest containing the test sample IDs and labels.",
    )
    parser.add_argument(
        "--seed-reports",
        action="append",
        required=True,
        metavar="INVARIANT=FIRST_REPORT,SECOND_REPORT",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    reports = _parse_report_args(args.seed_reports)
    sample_ids, y_true = _load_test_labels(args)
    predictions, provenance = _merge_seed_reports(reports, sample_ids)
    primary_metric = "PCC"
    if args.task_config is not None:
        primary_metric = str(load_yaml(args.task_config).get("primary_metric") or primary_metric).upper()
    rows = [
        {
            "subset": list(subset),
            "metrics": _metrics(y_true, mean_aggregate(predictions, subset)),
        }
        for subset in enumerate_nonempty_subsets(tuple(predictions))
    ]
    for row in rows:
        row["score"] = float(row["metrics"][primary_metric])
    best = (
        max(rows, key=lambda row: row["score"])
        if higher_is_better(primary_metric)
        else min(rows, key=lambda row: row["score"])
    )
    output = {
        "report_schema": "mint-agent.direct-gbt-seed-ensemble-consensus.v1",
        "evaluation_mode": "direct_train_test_consensus_posthoc",
        "selection_prohibited": True,
        "invariants": list(predictions),
        "primary_metric": primary_metric,
        "best_subset": best["subset"],
        "best_score": best["score"],
        "subset_metrics": rows,
        "by_invariant_metrics": {
            name: _metrics(y_true, values) for name, values in predictions.items()
        },
        "sample_order_hash": stable_hash(sample_ids),
        "sample_count": len(sample_ids),
        "seed_ensemble_provenance": provenance,
    }
    output["evaluation_hash"] = stable_hash(
        {"sample_order_hash": output["sample_order_hash"], "provenance": provenance, "rows": rows}
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    temporary.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(f"best={'+'.join(best['subset'])} {primary_metric}={best['score']:.6f}")
    return 0


def _load_test_labels(args: argparse.Namespace) -> tuple[tuple[str, ...], np.ndarray]:
    if args.test_feature_manifest is not None:
        seen: set[str] = set()
        sample_ids: list[str] = []
        labels: list[float] = []
        for line in args.test_feature_manifest.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            if entry.get("status") != "computed":
                continue
            sample_id = str(entry["sample_id"])
            if sample_id in seen:
                raise ValueError(f"Duplicate sample in test feature manifest: {sample_id}")
            seen.add(sample_id)
            sample_ids.append(sample_id)
            labels.append(float(entry["label"]))
        if not sample_ids or not np.isfinite(labels).all():
            raise ValueError("Test feature manifest contains no finite computed labels")
        return tuple(sample_ids), np.asarray(labels, dtype=float)
    if args.task_config is None:
        raise ValueError("Provide --test-feature-manifest or --task-config")
    task = load_yaml(args.task_config)
    records = load_configured_protein_ligand_records(task, config_path=args.task_config, split="test")
    return (
        tuple(record.pdb_id for record in records),
        np.asarray([record.label for record in records], dtype=float),
    )


def _parse_report_args(values: list[str]) -> dict[str, tuple[Path, Path]]:
    parsed: dict[str, tuple[Path, Path]] = {}
    for value in values:
        name, separator, paths = value.partition("=")
        first, comma, second = paths.partition(",")
        invariant = name.upper()
        if not separator or not comma or not invariant or invariant in parsed:
            raise ValueError(f"Expected unique INVARIANT=FIRST_REPORT,SECOND_REPORT; got {value!r}")
        parsed[invariant] = (Path(first), Path(second))
    return parsed


def _merge_seed_reports(
    reports: dict[str, tuple[Path, Path]], sample_ids: tuple[str, ...]
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    predictions: dict[str, np.ndarray] = {}
    provenance: dict[str, object] = {}
    test_hash = stable_hash(sample_ids)
    for invariant, (first_path, second_path) in reports.items():
        first = json.loads(first_path.read_text(encoding="utf-8"))
        second = json.loads(second_path.read_text(encoding="utf-8"))
        for report, path in ((first, first_path), (second, second_path)):
            if report.get("report_schema") != "mint-agent.direct-test-consensus-gbt.v1":
                raise ValueError(f"Unexpected report schema: {path}")
            if report.get("sample_order_hashes", {}).get("test") != test_hash:
                raise ValueError(f"Test order mismatch: {path}")
            if invariant not in report.get("test_predictions", {}):
                raise ValueError(f"Missing {invariant} predictions: {path}")
        first_config = dict(first["gbt_config"])
        second_config = dict(second["gbt_config"])
        # Config IDs distinguish report provenance; numerical training settings
        # determine whether two disjoint seed ensembles are compatible.
        first_config.pop("config_id", None)
        second_config.pop("config_id", None)
        first_runs = int(first_config.pop("n_runs"))
        second_runs = int(second_config.pop("n_runs"))
        first_seed = int(first_config.pop("random_state"))
        second_seed = int(second_config.pop("random_state"))
        if first_config != second_config or second_seed != first_seed + first_runs:
            raise ValueError(f"GBT configurations are not consecutive seed ranges for {invariant}")
        first_pred = np.asarray([first["test_predictions"][invariant][sid] for sid in sample_ids], dtype=float)
        second_pred = np.asarray([second["test_predictions"][invariant][sid] for sid in sample_ids], dtype=float)
        predictions[invariant] = (first_runs * first_pred + second_runs * second_pred) / (first_runs + second_runs)
        provenance[invariant] = {
            "first_report": str(first_path),
            "second_report": str(second_path),
            "seed_ranges": [list(range(first_seed, first_seed + first_runs)), list(range(second_seed, second_seed + second_runs))],
            "n_runs": first_runs + second_runs,
        }
    return predictions, provenance


if __name__ == "__main__":
    raise SystemExit(main())
