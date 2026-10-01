from __future__ import annotations

import argparse
import json
from dataclasses import fields, replace
from pathlib import Path
from typing import Any

import numpy as np

from mint_scout.compute_feature_batch import _selected_records
from mint_scout.config import load_yaml
from mint_scout.feature_qc import assert_qc_report_compatible
from mint_scout.filtration_audit import assert_filtration_audit_compatible
from mint_scout.invariants.plbind_tools import expected_feature_shape
from mint_scout.models.gbt import GBTConfig
from mint_scout.representation import RepresentationSpec
from mint_scout.scout.pipeline import ScoutConfig, run_scout_with_frozen_probe
from mint_scout.scout.sampling import ProbeSelection


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run fixed-GBT Scout from a frozen protein-ligand probe and feature manifests."
    )
    parser.add_argument("--task-config", type=Path, required=True)
    parser.add_argument("--scout-config", type=Path, required=True)
    parser.add_argument("--gbt-config", type=Path, default=None)
    parser.add_argument("--representation-spec", type=Path, required=True)
    parser.add_argument("--probe-selection", type=Path, required=True)
    parser.add_argument(
        "--feature-manifest",
        action="append",
        required=True,
        metavar="INVARIANT=PATH",
    )
    parser.add_argument("--user-target", type=float, default=None)
    parser.add_argument("--feature-qc-report", type=Path, default=None)
    parser.add_argument("--filtration-audit-report", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execution-artifact", type=Path, required=True)
    args = parser.parse_args(argv)

    task = load_yaml(args.task_config)
    scout_config = ScoutConfig.from_mapping(load_yaml(args.scout_config))
    if args.gbt_config is not None:
        scout_config = replace(
            scout_config,
            gbt=_load_gbt_config(args.gbt_config),
        )
    representation = RepresentationSpec.read(args.representation_spec)
    representation.assert_frozen()
    probe_payload = _load_json_object(args.probe_selection)
    if probe_payload.get("representation_hash") != representation.spec_hash:
        raise ValueError("Probe selection and RepresentationSpec hashes do not match")

    train_records = _selected_records(
        task,
        sample_ids=None,
        split="train",
        offset=0,
        limit=None,
        config_path=args.task_config,
    )
    modeling_ids = tuple(record.pdb_id for record in train_records)
    if tuple(probe_payload.get("modeling_sample_ids", ())) != modeling_ids:
        raise ValueError("Probe selection modeling IDs do not match the training pool")
    selection = _selection_from_payload(probe_payload)
    target_by_id = {record.pdb_id: record.label for record in train_records}
    probe_targets = np.asarray([target_by_id[sample_id] for sample_id in selection.sample_ids])
    for row in probe_payload["probe_samples"]:
        sample_id = str(row["sample_id"])
        if float(row["label"]) != target_by_id[sample_id]:
            raise ValueError(f"Probe label mismatch for {sample_id}")

    manifest_paths = _parse_manifest_args(args.feature_manifest)
    if args.feature_qc_report is not None:
        assert_qc_report_compatible(
            path=args.feature_qc_report,
            representation=representation,
            sample_ids=selection.sample_ids,
            manifest_paths=manifest_paths,
        )
    filtration_audit = None
    if args.filtration_audit_report is not None:
        filtration_audit = assert_filtration_audit_compatible(
            path=args.filtration_audit_report,
            representation=representation,
            sample_ids=selection.sample_ids,
            manifest_paths=manifest_paths,
        )
    features = {
        invariant: _load_manifest_features(
            path=path,
            invariant=invariant,
            sample_ids=selection.sample_ids,
            representation=representation,
        )
        for invariant, path in manifest_paths.items()
    }
    full_folds = {
        str(sample_id): int(fold)
        for sample_id, fold in probe_payload["modeling_fold_assignment"].items()
    }
    result = run_scout_with_frozen_probe(
        modeling_sample_ids=modeling_ids,
        selection=selection,
        probe_targets=probe_targets,
        features_by_invariant=features,
        modeling_fold_assignment=full_folds,
        user_target=args.user_target,
        config=scout_config,
    )
    payload = {
        "report_schema": "mint-agent.casf-scout.v1",
        "task_id": task.get("task_id"),
        "dataset_id": task.get("task_id"),
        "evidence_scope": "probe",
        "status": "COMPLETE",
        "representation_hash": representation.spec_hash,
        "selection_hash": probe_payload.get("selection_hash"),
        "sample_count": len(selection.sample_ids),
        "sample_ids": list(selection.sample_ids),
        "feature_manifests": {name: str(path) for name, path in manifest_paths.items()},
        "feature_qc_report": (
            str(args.feature_qc_report) if args.feature_qc_report is not None else None
        ),
        "filtration_audit_report": (
            str(args.filtration_audit_report) if args.filtration_audit_report is not None else None
        ),
        "filtration_recommendations": (
            {
                name: report["recommendation"]
                for name, report in filtration_audit["invariants"].items()
            }
            if filtration_audit is not None
            else None
        ),
        "scout": result.to_dict(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result.to_execution_artifact(representation_hash=representation.spec_hash).write(
        args.execution_artifact
    )
    print(
        f"probe_hash={result.probe_hash} invariants={','.join(sorted(features))} "
        f"priority_count={len(result.ranking.priority_order)} target={result.target.value:.6g} "
        f"output={args.output}"
    )
    return 0


def _load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def _load_gbt_config(path: Path) -> GBTConfig:
    payload = load_yaml(path)
    if not isinstance(payload, dict):
        raise ValueError(f"GBT config must be a mapping: {path}")
    allowed = {field.name for field in fields(GBTConfig)}
    return GBTConfig(**{key: value for key, value in payload.items() if key in allowed})


def _selection_from_payload(payload: dict[str, Any]) -> ProbeSelection:
    sample_ids = tuple(str(sample_id) for sample_id in payload["probe_sample_ids"])
    sample_rows = {str(row["sample_id"]): row for row in payload["probe_samples"]}
    if set(sample_rows) != set(sample_ids):
        raise ValueError("probe_samples do not match probe_sample_ids")
    presence = {
        sample_id: {tuple(pair) for pair in sample_rows[sample_id]["present_pairs"]}
        for sample_id in sample_ids
    }
    retained_pairs = tuple(
        _decode_pair_name(name) for name in payload.get("pair_support", {})
    )
    pair_support = {
        pair: sum(pair in presence[sample_id] for sample_id in sample_ids)
        for pair in retained_pairs
    }
    recorded_support = {
        _decode_pair_name(name): int(value)
        for name, value in payload.get("pair_support", {}).items()
    }
    if pair_support != recorded_support:
        raise ValueError("Recorded probe pair support does not match sample presence")
    return ProbeSelection(
        sample_ids=sample_ids,
        base_size=int(payload["probe_base_size"]),
        final_size=int(payload["probe_final_size"]),
        target_bins={sample_id: int(sample_rows[sample_id]["target_bin"]) for sample_id in sample_ids},
        size_bins={sample_id: int(sample_rows[sample_id]["size_bin"]) for sample_id in sample_ids},
        pair_support=pair_support,
        undercovered_pairs=tuple(tuple(pair) for pair in payload.get("undercovered_pairs", ())),
        warnings=tuple(str(warning) for warning in payload.get("warnings", ())),
    )


def _parse_manifest_args(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Feature manifest must use INVARIANT=PATH: {value!r}")
        invariant, raw_path = value.split("=", 1)
        name = invariant.upper()
        if name in result:
            raise ValueError(f"Duplicate feature manifest for {name}")
        result[name] = Path(raw_path)
    return result


def _load_manifest_features(
    *,
    path: Path,
    invariant: str,
    sample_ids: tuple[str, ...],
    representation: RepresentationSpec,
) -> np.ndarray:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        sample_id = str(row["sample_id"])
        if sample_id in by_id:
            raise ValueError(f"Duplicate sample {sample_id} in {path}")
        by_id[sample_id] = row
    if set(by_id) != set(sample_ids):
        raise ValueError(f"{invariant} manifest sample IDs do not match the frozen probe")
    expected = expected_feature_shape(invariant, representation)
    arrays = []
    for sample_id in sample_ids:
        row = by_id[sample_id]
        if str(row.get("invariant", "")).upper() != invariant:
            raise ValueError(f"Manifest invariant mismatch for {sample_id}")
        if row.get("status") not in {"computed", "cached"}:
            raise ValueError(f"Feature generation failed for {invariant}/{sample_id}")
        feature_path = Path(str(row["output_path"]))
        if f"/repr-{representation.spec_hash}/" not in feature_path.as_posix():
            raise ValueError(f"Feature path has wrong representation hash: {feature_path}")
        array = np.load(feature_path, allow_pickle=False)
        if array.shape != expected:
            raise ValueError(
                f"Feature shape mismatch for {invariant}/{sample_id}: expected {expected}, got {array.shape}"
            )
        if not np.all(np.isfinite(array)):
            raise ValueError(f"Feature contains NaN or Inf for {invariant}/{sample_id}")
        arrays.append(array)
    return np.stack(arrays)


def _decode_pair_name(name: str) -> tuple[str, str]:
    try:
        left, right = name.split("|")
        return left.split(":", 1)[1], right.split(":", 1)[1]
    except (IndexError, ValueError) as exc:
        raise ValueError(f"Invalid role-aware pair name: {name!r}") from exc


if __name__ == "__main__":
    raise SystemExit(main())
