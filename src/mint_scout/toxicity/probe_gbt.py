from __future__ import annotations

import argparse
import json
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from mint_scout.config import load_yaml
from mint_scout.invariants.manifest import stable_hash
from mint_scout.models.gbt import GBTConfig, run_oof_gbt
from mint_scout.scout.sampling import quantile_bin_assignments
from mint_scout.scout.stability import (
    BootstrapConfig,
    rank_consensus_subsets,
)
from mint_scout.toxicity.selection_contract import (
    ToxicitySelectionContract,
    methods_from_plan,
)


OOF_SCHEMA = "mint-agent.toxicity-probe-oof.v1"
SCOUT_SCHEMA = "mint-agent.toxicity-probe-scout.v1"


def run_toxicity_oof_task(
    *,
    feature_plan_path: str | Path,
    feature_qc_path: str | Path,
    probe_selection_path: str | Path,
    gbt_config_path: str | Path,
    task_index: int,
    output_dir: str | Path,
) -> dict[str, Any]:
    plan = _load_object(feature_plan_path, "toxicity feature plan")
    qc = _load_object(feature_qc_path, "toxicity feature QC")
    probe = _load_object(probe_selection_path, "toxicity Probe selection")
    if qc.get("plan_hash") != plan.get("plan_hash"):
        raise ValueError("toxicity Probe GBT feature plan and QC do not match")
    if qc.get("test_labels_used") is not False or probe.get("test_labels_used") is not False:
        raise ValueError("toxicity Probe GBT must not use test labels")
    tasks = _required_list(plan, "tasks")
    if task_index < 0:
        raise ValueError("toxicity Probe GBT task index must be non-negative")
    if task_index >= len(tasks):
        return {"status": "SKIPPED", "task_index": task_index, "task_count": len(tasks)}
    task = tasks[task_index]
    signature = str(task["feature_signature"])
    qc_by_signature = {
        str(row["feature_signature"]): row for row in _required_list(qc, "tasks")
    }
    task_qc = qc_by_signature[signature]
    if not bool(task_qc.get("hard_gate_passed")):
        raise ValueError(f"feature QC hard gate failed for {signature}")

    sample_ids = tuple(str(value) for value in probe["probe_sample_ids"])
    fold_ids = {
        str(sample_id): int(fold)
        for sample_id, fold in _required_mapping(
            probe, "probe_fold_assignment"
        ).items()
    }
    features, targets = _load_feature_matrix(
        Path(str(task["manifest_path"])),
        sample_ids=sample_ids,
        invariant=str(task["invariant"]),
    )
    gbt = _load_gbt_config(Path(gbt_config_path))
    artifact = run_oof_gbt(
        features.reshape(len(sample_ids), -1),
        targets,
        sample_ids,
        fold_ids,
        config=gbt,
    )
    payload: dict[str, Any] = {
        "report_schema": OOF_SCHEMA,
        "status": "COMPLETE",
        "evidence_scope": "train_probe_oof",
        "test_structures_used": False,
        "test_labels_used": False,
        "feature_plan_hash": plan.get("plan_hash"),
        "feature_qc_hash": qc.get("qc_hash"),
        "probe_selection_hash": probe.get("selection_hash"),
        "task_index": task_index,
        "invariant": str(task["invariant"]),
        "feature_signature": signature,
        "candidate_ids": list(task["candidate_ids"]),
        "feature_manifest": str(task["manifest_path"]),
        "sample_ids": list(sample_ids),
        "fold_ids": artifact.fold_ids,
        "targets": [float(value) for value in artifact.y_true],
        "oof_predictions": [float(value) for value in artifact.y_pred],
        "gbt_config": asdict(gbt),
        "gbt_parameter_hash": gbt.parameter_hash,
        "cost": {
            "wall_seconds": artifact.cost.wall_seconds,
            "cpu_seconds": artifact.cost.cpu_seconds,
            "cpu_core_hours": artifact.cost.cpu_core_hours,
        },
    }
    payload["oof_hash"] = stable_hash(payload)
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    output = destination / f"{signature}.json"
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {"output": str(output), **payload}


def combine_toxicity_probe_gbt(
    *,
    feature_plan_path: str | Path,
    feature_qc_path: str | Path,
    probe_selection_path: str | Path,
    gbt_config_path: str | Path,
    oof_dir: str | Path,
    bootstrap: BootstrapConfig = BootstrapConfig(),
) -> dict[str, Any]:
    plan = _load_object(feature_plan_path, "toxicity feature plan")
    qc = _load_object(feature_qc_path, "toxicity feature QC")
    probe = _load_object(probe_selection_path, "toxicity Probe selection")
    if qc.get("plan_hash") != plan.get("plan_hash"):
        raise ValueError("toxicity Probe Scout feature plan and QC do not match")
    qc_candidates = _required_list(qc, "candidates")
    if len(qc_candidates) != 1 or not bool(qc_candidates[0].get("hard_gate_passed")):
        raise ValueError("toxicity Probe Scout requires one QC-approved representation")
    signature_map = _required_mapping(plan, "candidate_feature_signatures")
    if len(signature_map) != 1:
        raise ValueError("toxicity Probe Scout requires exactly one representation")
    candidate_id, method_signatures = next(iter(signature_map.items()))
    method_signatures = _required_mapping(
        {"method_signatures": method_signatures}, "method_signatures"
    )
    methods = methods_from_plan(plan)
    if set(method_signatures) != set(methods):
        raise ValueError("toxicity Probe Scout method signatures differ from requested methods")
    raw_contract = plan.get("selection_contract")
    selection = (
        ToxicitySelectionContract.from_mapping(raw_contract)
        if isinstance(raw_contract, Mapping)
        else ToxicitySelectionContract(requested_invariants=methods)
    )
    gbt = _load_gbt_config(Path(gbt_config_path))
    sample_ids = tuple(str(value) for value in probe["probe_sample_ids"])
    predictions = {}
    invariant_costs = {}
    targets = None
    fold_ids = None
    oof_reports = {}
    root = Path(oof_dir).expanduser().resolve()
    for method in methods:
        signature = str(method_signatures[method])
        report_path = root / f"{signature}.json"
        report = _load_object(report_path, f"toxicity {method} OOF report")
        _assert_oof_compatible(
            report,
            method=method,
            signature=signature,
            sample_ids=sample_ids,
            plan=plan,
            qc=qc,
            probe=probe,
            gbt=gbt,
        )
        current_targets = np.asarray(report["targets"], dtype=float)
        current_folds = {
            str(key): int(value) for key, value in report["fold_ids"].items()
        }
        if targets is None:
            targets = current_targets
            fold_ids = current_folds
        elif not np.array_equal(targets, current_targets) or fold_ids != current_folds:
            raise ValueError("toxicity OOF reports do not share targets and folds")
        predictions[method] = np.asarray(report["oof_predictions"], dtype=float)
        invariant_costs[method] = float(report["cost"]["wall_seconds"])
        oof_reports[method] = str(report_path)
    assert targets is not None and fold_ids is not None
    target_bins = quantile_bin_assignments(
        targets, sample_ids, max_bins=10
    )
    ranking = rank_consensus_subsets(
        y_true=targets,
        predictions=predictions,
        primary_metric=selection.primary_metric,
        secondary_metrics=tuple(
            metric for metric in ("PCC2", "PCC", "R2", "RMSE", "MAE")
            if metric != selection.primary_metric
        ),
        strata=[target_bins[sample_id] for sample_id in sample_ids],
        bootstrap=bootstrap,
        invariant_costs=invariant_costs,
        ranking_weights=None,
    )
    summaries = []
    for summary in ranking.summaries:
        row = asdict(summary)
        row["invariants"] = list(summary.invariants)
        row["secondary_metrics"] = dict(summary.secondary_metrics)
        summaries.append(row)
    result: dict[str, Any] = {
        "report_schema": SCOUT_SCHEMA,
        "status": "COMPLETE",
        "evidence_scope": "train_probe_oof",
        "test_structures_used": False,
        "test_labels_used": False,
        "representation_candidate_id": str(candidate_id),
        "feature_plan_hash": plan.get("plan_hash"),
        "feature_qc_hash": qc.get("qc_hash"),
        "probe_selection_hash": probe.get("selection_hash"),
        "sample_count": len(sample_ids),
        "cv_folds": len(set(fold_ids.values())),
        "selection_contract": selection.to_dict(),
        "requested_invariants": list(methods),
        "primary_metric": selection.primary_metric,
        "secondary_metrics": [
            metric for metric in ("PCC2", "PCC", "R2", "RMSE", "MAE")
            if metric != selection.primary_metric
        ],
        "gbt_config": asdict(gbt),
        "gbt_parameter_hash": gbt.parameter_hash,
        "bootstrap_config": asdict(bootstrap),
        "ranking_policy": (
            "feature_qc_gate_then_bootstrap_top_tier_then_stability_then_cost"
        ),
        "llm_prior_used": False,
        "weighted_final_score_used": False,
        "invariant_oof_reports": oof_reports,
        "invariant_cost_seconds": invariant_costs,
        "nominal_best": list(ranking.nominal_best),
        "top_tier": [list(subset) for subset in ranking.top_tier],
        "priority_order": [list(subset) for subset in ranking.priority_order],
        "subset_summaries": summaries,
        "bootstrap_index_hashes": list(ranking.bootstrap_index_hashes),
    }
    result["scout_hash"] = stable_hash(result)
    return result


def _load_feature_matrix(
    path: Path,
    *,
    sample_ids: tuple[str, ...],
    invariant: str,
) -> tuple[np.ndarray, np.ndarray]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_id = {str(row["sample_id"]): row for row in rows}
    if len(by_id) != len(rows) or set(by_id) != set(sample_ids):
        raise ValueError(f"toxicity {invariant} manifest does not match the Probe")
    arrays = []
    targets = []
    for sample_id in sample_ids:
        row = by_id[sample_id]
        if str(row.get("invariant") or "").upper() != invariant.upper():
            raise ValueError(f"toxicity feature invariant mismatch for {sample_id}")
        if row.get("status") not in {"computed", "cached"}:
            raise ValueError(f"toxicity feature generation failed for {sample_id}")
        array = np.load(Path(str(row["output_path"])), allow_pickle=False)
        if not np.isfinite(array).all():
            raise ValueError(f"toxicity feature is nonfinite for {sample_id}")
        arrays.append(np.asarray(array, dtype=np.float32))
        targets.append(float(row["label"]))
    return np.stack(arrays), np.asarray(targets, dtype=float)


def _assert_oof_compatible(
    report: Mapping[str, Any],
    *,
    method: str,
    signature: str,
    sample_ids: tuple[str, ...],
    plan: Mapping[str, Any],
    qc: Mapping[str, Any],
    probe: Mapping[str, Any],
    gbt: GBTConfig,
) -> None:
    checks = {
        "schema": report.get("report_schema") == OOF_SCHEMA,
        "status": report.get("status") == "COMPLETE",
        "invariant": report.get("invariant") == method,
        "signature": report.get("feature_signature") == signature,
        "samples": tuple(report.get("sample_ids", ())) == sample_ids,
        "plan": report.get("feature_plan_hash") == plan.get("plan_hash"),
        "qc": report.get("feature_qc_hash") == qc.get("qc_hash"),
        "probe": report.get("probe_selection_hash") == probe.get("selection_hash"),
        "gbt": report.get("gbt_parameter_hash") == gbt.parameter_hash,
        "test_labels": report.get("test_labels_used") is False,
    }
    failures = [name for name, passed in checks.items() if not passed]
    if failures:
        raise ValueError(f"incompatible toxicity {method} OOF report: {failures}")


def _load_gbt_config(path: Path) -> GBTConfig:
    raw = load_yaml(path)
    allowed = {item.name for item in fields(GBTConfig)}
    unknown = sorted(set(raw) - allowed - {"source"})
    if unknown:
        raise ValueError(f"unknown toxicity GBT config fields: {unknown}")
    return GBTConfig(**{key: value for key, value in raw.items() if key in allowed})


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
    parser = argparse.ArgumentParser(description="Run toxicity Probe GBT OOF and ranking.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    oof = subparsers.add_parser("oof")
    oof.add_argument("--feature-plan", type=Path, required=True)
    oof.add_argument("--feature-qc", type=Path, required=True)
    oof.add_argument("--probe-selection", type=Path, required=True)
    oof.add_argument("--gbt-config", type=Path, required=True)
    oof.add_argument("--task-index", type=int, required=True)
    oof.add_argument("--output-dir", type=Path, required=True)
    combine = subparsers.add_parser("combine")
    combine.add_argument("--feature-plan", type=Path, required=True)
    combine.add_argument("--feature-qc", type=Path, required=True)
    combine.add_argument("--probe-selection", type=Path, required=True)
    combine.add_argument("--gbt-config", type=Path, required=True)
    combine.add_argument("--oof-dir", type=Path, required=True)
    combine.add_argument("--output", type=Path, required=True)
    combine.add_argument("--bootstrap-replicates", type=int, default=1000)
    args = parser.parse_args(argv)
    if args.command == "oof":
        result = run_toxicity_oof_task(
            feature_plan_path=args.feature_plan,
            feature_qc_path=args.feature_qc,
            probe_selection_path=args.probe_selection,
            gbt_config_path=args.gbt_config,
            task_index=args.task_index,
            output_dir=args.output_dir,
        )
        if result["status"] == "SKIPPED":
            print(f"status=SKIPPED task_index={args.task_index}")
        else:
            print(
                f"status={result['status']} invariant={result['invariant']} "
                f"wall_seconds={result['cost']['wall_seconds']:.3f} output={result['output']}"
            )
        return 0
    result = combine_toxicity_probe_gbt(
        feature_plan_path=args.feature_plan,
        feature_qc_path=args.feature_qc,
        probe_selection_path=args.probe_selection,
        gbt_config_path=args.gbt_config,
        oof_dir=args.oof_dir,
        bootstrap=BootstrapConfig(replicates=args.bootstrap_replicates),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"status={result['status']} nominal_best={'+'.join(result['nominal_best'])} "
        f"top_tier={len(result['top_tier'])} output={args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
