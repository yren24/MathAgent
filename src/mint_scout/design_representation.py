from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from mint_scout.compute_feature_batch import _selected_records
from mint_scout.config import load_yaml
from mint_scout.data.element_pairs import SupportThresholds
from mint_scout.filtration import FiltrationConfig
from mint_scout.invariants.manifest import stable_hash
from mint_scout.representation_design import (
    HydrogenPolicyConfig,
    MetalAwarenessConfig,
    RepresentationDesignConfig,
    design_adaptive_representation,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Design and freeze a dataset-adaptive RepresentationSpec from the modeling pool."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "test", "all"), default="train")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--data-audit-report", type=Path, default=None)
    parser.add_argument("--expected-input-hash", default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.offset < 0:
        raise ValueError("--offset must be non-negative")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")

    task_config = load_yaml(args.config)
    if str(task_config.get("system_type")) != "protein_ligand":
        raise ValueError("Representation design currently requires system_type=protein_ligand")
    records = _selected_records(
        task_config,
        sample_ids=None,
        split=args.split,
        offset=args.offset,
        limit=args.limit,
        config_path=args.config,
    )
    data_audit_hash = None
    if args.data_audit_report is not None:
        data_audit_hash = _validate_data_audit(
            args.data_audit_report,
            task_id=str(task_config.get("task_id") or ""),
            sample_ids=tuple(record.pdb_id for record in records),
        )
    role_paths = {}
    for record in records:
        if record.protein_path is None or record.ligand_path is None:
            raise ValueError(f"Sample {record.pdb_id!r} is missing protein/ligand structure paths")
        role_paths[record.pdb_id] = {
            "protein": record.protein_path,
            "ligand": record.ligand_path,
        }
    design_config = _design_config_from_task(task_config)
    computed_design_input_hash = representation_design_input_hash(
        task_config,
        data_audit_content_sha256=data_audit_hash,
        split=args.split,
        offset=args.offset,
        limit=args.limit,
    )
    if args.expected_input_hash is not None:
        compatible_hashes = {
            computed_design_input_hash,
            _representation_design_input_hash_without_policy_extensions(
                task_config,
                data_audit_content_sha256=data_audit_hash,
                split=args.split,
                offset=args.offset,
                limit=args.limit,
            ),
        }
        if args.expected_input_hash not in compatible_hashes:
            raise ValueError("Representation-design input hash does not match the submitted plan")
    result = design_adaptive_representation(
        system_type="protein_ligand",
        role_paths_by_sample=role_paths,
        config=design_config,
    )
    report = {
        "report_schema": "mint-agent.representation-design.v1",
        "run_kind": "engineering_smoke" if args.limit is not None else "full_modeling_pool_design",
        "task_id": task_config.get("task_id"),
        "dataset_id": task_config.get("task_id"),
        "evidence_scope": "design",
        "modeling_scope": args.split,
        "offset": args.offset,
        "limit": args.limit,
        "sample_count": len(records),
        "sample_ids": [record.pdb_id for record in records],
        "config": {
            "support": asdict(design_config.support),
            "filtration": design_config.filtration.to_dict(),
            "eic_tau_start": design_config.eic_tau_start,
            "eic_tau_stop": design_config.eic_tau_stop,
            "eic_tau_step": design_config.eic_tau_step,
            "saturation_audit_enabled": design_config.saturation_audit_enabled,
            "distance_chunk_size": design_config.distance_chunk_size,
            "hydrogen_policy": asdict(design_config.hydrogen_policy),
            "metal_awareness": asdict(design_config.metal_awareness),
        },
        "representation_spec": result.representation_spec.to_dict(),
        "representation_hash": result.representation_spec.spec_hash,
        "geometry_profile": result.geometry_profile.to_dict(),
        "data_audit_report": (
            str(args.data_audit_report.resolve())
            if args.data_audit_report is not None
            else None
        ),
        "data_audit_content_sha256": data_audit_hash,
        "design_input_hash": computed_design_input_hash,
        "status": "COMPLETE",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"representation_hash={result.representation_spec.spec_hash} samples={len(records)} "
        f"pairs={len(result.representation_spec.pair_order)} "
        f"rmax={result.geometry_profile.max_filtration_angstrom:.6g} output={args.output}"
    )
    return 0


def _validate_data_audit(
    path: Path, *, task_id: str, sample_ids: tuple[str, ...]
) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("data audit report must contain a JSON object")
    if payload.get("report_schema") != "mint-agent.dataset-audit.v1":
        raise ValueError("representation design requires a dataset-audit artifact")
    if str(payload.get("task_id") or "") != task_id:
        raise ValueError("data audit task does not match representation design")
    if payload.get("status") != "PASS":
        raise ValueError("data audit requires review before representation design")
    audited_ids = tuple(str(value) for value in payload.get("sample_ids", ()))
    if audited_ids and audited_ids != sample_ids:
        raise ValueError("data audit sample IDs do not match the modeling pool")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def representation_design_input_hash(
    config: dict[str, Any],
    *,
    data_audit_content_sha256: str | None,
    split: str,
    offset: int,
    limit: int | None,
) -> str:
    return stable_hash(
        {
            "hash_schema": "mint-agent.representation-design-input.v1",
            "task_id": config.get("task_id"),
            "system_type": config.get("system_type"),
            "representation_mode": config.get("representation_mode"),
            "pair_schema": config.get("pair_schema"),
            "representation_design": config.get("representation_design"),
            "hydrogen_request": config.get("hydrogen_request"),
            "data_audit_policy": config.get("data_audit"),
            "data_audit_content_sha256": data_audit_content_sha256,
            "split": split,
            "offset": offset,
            "limit": limit,
        }
    )


def _representation_design_input_hash_without_policy_extensions(
    config: dict[str, Any],
    *,
    data_audit_content_sha256: str | None,
    split: str,
    offset: int,
    limit: int | None,
) -> str:
    return stable_hash(
        {
            "hash_schema": "mint-agent.representation-design-input.v1",
            "task_id": config.get("task_id"),
            "system_type": config.get("system_type"),
            "representation_mode": config.get("representation_mode"),
            "pair_schema": config.get("pair_schema"),
            "representation_design": config.get("representation_design"),
            "data_audit_content_sha256": data_audit_content_sha256,
            "split": split,
            "offset": offset,
            "limit": limit,
        }
    )


def _design_config_from_task(config: dict[str, Any]) -> RepresentationDesignConfig:
    raw = config.get("representation_design", {})
    if not isinstance(raw, dict):
        raise ValueError("representation_design config must be a mapping")
    support = raw.get("support", {})
    filtration = raw.get("filtration", {})
    if not isinstance(support, dict) or not isinstance(filtration, dict):
        raise ValueError("representation_design support/filtration must be mappings")
    hydrogen_defaults = raw.get("hydrogen_policy", {})
    hydrogen_request = config.get("hydrogen_request", {})
    metal_awareness = raw.get("metal_awareness", {})
    if not isinstance(hydrogen_defaults, dict):
        raise ValueError("representation_design.hydrogen_policy must be a mapping")
    if not isinstance(hydrogen_request, dict):
        raise ValueError("hydrogen_request must be a mapping")
    if not isinstance(metal_awareness, dict):
        raise ValueError("representation_design.metal_awareness must be a mapping")
    data_audit = config.get("data_audit", {})
    data_audit = data_audit if isinstance(data_audit, dict) else {}
    return RepresentationDesignConfig(
        support=SupportThresholds(**support),
        filtration=FiltrationConfig(**filtration),
        eic_tau_start=float(raw.get("eic_tau_start", 0.2)),
        eic_tau_stop=float(raw.get("eic_tau_stop", 5.0)),
        eic_tau_step=float(raw.get("eic_tau_step", 0.1)),
        saturation_audit_enabled=bool(raw.get("saturation_audit_enabled", False)),
        distance_chunk_size=int(raw.get("distance_chunk_size", 2048)),
        adapter_supported_elements=_adapter_supported_elements(raw.get("adapter_supported_elements")),
        tolerated_unsupported_elements=_elements_by_role(
            data_audit.get("tolerated_out_of_schema"),
            "data_audit.tolerated_out_of_schema",
        ),
        hydrogen_policy=HydrogenPolicyConfig(
            mode=str(hydrogen_request.get("mode") or hydrogen_defaults.get("default_mode", "auto")),
            task_relevance=str(hydrogen_request.get("task_relevance") or "uncertain"),
            min_explicit_sample_fraction=float(
                hydrogen_defaults.get("min_explicit_sample_fraction", 0.95)
            ),
        ),
        metal_awareness=MetalAwarenessConfig(
            enabled=bool(metal_awareness.get("enabled", True)),
            protein_elements=tuple(
                str(value)
                for value in metal_awareness.get(
                    "protein_elements", MetalAwarenessConfig().protein_elements
                )
            ),
            min_sample_fraction=float(
                metal_awareness.get("min_sample_fraction", 0.005)
            ),
            min_samples=int(metal_awareness.get("min_samples", 10)),
            max_ligand_distance_angstrom=float(
                metal_awareness.get("max_ligand_distance_angstrom", 6.0)
            ),
            min_proximal_samples=int(
                metal_awareness.get("min_proximal_samples", 10)
            ),
        ),
    )


def _adapter_supported_elements(raw: object) -> dict[str, tuple[str, ...]] | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError("representation_design.adapter_supported_elements must be a mapping")
    result: dict[str, tuple[str, ...]] = {}
    for role, elements in raw.items():
        if not isinstance(elements, list) or not all(isinstance(element, str) for element in elements):
            raise ValueError(f"adapter_supported_elements.{role} must be a list of symbols")
        result[str(role)] = tuple(elements)
    return result


def _elements_by_role(
    raw: object, name: str
) -> dict[str, tuple[str, ...]] | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError(f"{name} must be a mapping")
    result: dict[str, tuple[str, ...]] = {}
    for role, elements in raw.items():
        if not isinstance(elements, list) or not all(
            isinstance(element, str) for element in elements
        ):
            raise ValueError(f"{name}.{role} must be a list of symbols")
        result[str(role)] = tuple(elements)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
