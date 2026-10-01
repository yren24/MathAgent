from __future__ import annotations

import argparse
import json
import time
import traceback
from dataclasses import replace
from pathlib import Path
from typing import Any

from mint_scout.compute_feature import _tool_config_from_task
from mint_scout.config import load_yaml
from mint_scout.data.casf_index import CasfRecord, load_casf_records
from mint_scout.data.manifest_io import load_configured_protein_ligand_records
from mint_scout.data.structure_staging import stage_protein_ligand_input
from mint_scout.invariants.plbind_tools import PLBindFeatureTool
from mint_scout.representation import RepresentationSpec


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Batch compute PLBind features from a configured dataset source.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--invariant", required=True, choices=("PH", "PL", "CA", "FPRC", "EIC"))
    parser.add_argument(
        "--split", choices=("all", "train", "validation", "test"), default="all"
    )
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sample-id", action="append", default=None)
    parser.add_argument("--sample-id-file", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--legacy-root", type=Path, default=None)
    parser.add_argument("--pdb-folder", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--representation-spec", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-failures", action="store_true")
    args = parser.parse_args(argv)

    if args.offset < 0:
        raise ValueError("--offset must be non-negative")
    if args.limit is not None and args.limit < 0:
        raise ValueError("--limit must be non-negative")
    if args.sample_id and args.sample_id_file:
        raise ValueError("Use either --sample-id or --sample-id-file, not both")

    representation_spec = RepresentationSpec.read(args.representation_spec) if args.representation_spec else None
    config = load_yaml(args.config)
    sample_ids = args.sample_id or _sample_ids_from_file(args.sample_id_file)
    uses_manifest = config.get("dataset_manifest") is not None
    records = _selected_records(
        config,
        sample_ids=sample_ids,
        split=args.split,
        offset=args.offset,
        limit=args.limit,
        config_path=args.config,
    )
    tool_config = _tool_config_from_task(config, args=args)
    if representation_spec is not None:
        tool_config = replace(tool_config, representation_mode=representation_spec.mode.value)
    tool = PLBindFeatureTool(tool_config)

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    failures = 0
    with args.manifest.open("w", encoding="utf-8") as handle:
        for batch_index, record in enumerate(records):
            entry = _compute_record(
                tool,
                record,
                invariant=args.invariant,
                batch_index=batch_index,
                dry_run=args.dry_run,
                representation_spec=representation_spec,
                stage_structure_paths=uses_manifest,
            )
            if entry["status"] == "failed":
                failures += 1
            handle.write(json.dumps(entry, sort_keys=True) + "\n")
            handle.flush()
            print(
                "sample_id={sample_id} split={split} invariant={invariant} status={status} output={output_path}".format(
                    **entry
                )
            )

    print(f"manifest={args.manifest} total={len(records)} failures={failures}")
    if failures and not args.allow_failures:
        return 1
    return 0


def _selected_records(
    config: dict[str, Any],
    *,
    sample_ids: list[str] | None,
    split: str,
    offset: int,
    limit: int | None,
    config_path: Path | None = None,
) -> tuple[CasfRecord, ...]:
    if config.get("dataset_manifest") is not None:
        if config_path is None:
            raise ValueError("config_path is required when dataset_manifest is configured")
        records = load_configured_protein_ligand_records(
            config,
            config_path=config_path,
            split=split,
        )
    else:
        casf = _casf_config(config)
        records = load_casf_records(
            index_root=Path(casf["index_root"]).expanduser(),
            structures_root=Path(casf["structures_root"]).expanduser(),
            year=int(casf.get("year", 2016)),
            protein_template=str(casf.get("protein_template", "{pdb}_pocket.pdb")),
            ligand_template=str(casf.get("ligand_template", "{pdb}_ligand.mol2")),
        )
    if sample_ids:
        requested = {sample_id.casefold() for sample_id in sample_ids}
        records = tuple(record for record in records if record.pdb_id.casefold() in requested)
        missing = sorted(requested - {record.pdb_id.casefold() for record in records})
        if missing:
            raise ValueError(f"Requested sample ids are not in the dataset: {', '.join(missing)}")
    if config.get("dataset_manifest") is None and split != "all":
        records = tuple(record for record in records if record.split == split)
    return records[offset : None if limit is None else offset + limit]


def _casf_config(config: dict[str, Any]) -> dict[str, Any]:
    data_audit = config.get("data_audit", {})
    casf = data_audit.get("casf", {}) if isinstance(data_audit, dict) else {}
    if not isinstance(casf, dict):
        raise ValueError("data_audit.casf config must be a mapping")
    for key in ("index_root", "structures_root"):
        if key not in casf:
            raise ValueError(f"data_audit.casf.{key} is required")
    return casf


def _sample_ids_from_file(path: Path | None) -> list[str] | None:
    if path is None:
        return None
    text = path.read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        sample_ids = [line.strip() for line in text.splitlines() if line.strip()]
    else:
        if isinstance(data, dict):
            data = data.get("probe_sample_ids", data.get("sample_ids"))
        if not isinstance(data, list):
            raise ValueError("sample-id JSON must contain probe_sample_ids or sample_ids list")
        sample_ids = [str(sample_id) for sample_id in data]
    if not sample_ids:
        raise ValueError("sample-id file contains no sample IDs")
    if len(set(sample_id.lower() for sample_id in sample_ids)) != len(sample_ids):
        raise ValueError("sample-id file contains duplicate sample IDs")
    return sample_ids


def _compute_record(
    tool: PLBindFeatureTool,
    record: CasfRecord,
    *,
    invariant: str,
    batch_index: int,
    dry_run: bool,
    representation_spec: RepresentationSpec | None = None,
    stage_structure_paths: bool = False,
) -> dict[str, Any]:
    started = time.perf_counter()
    base: dict[str, Any] = {
        "batch_index": batch_index,
        "sample_id": record.pdb_id,
        "split": record.split,
        "label": record.label,
        "invariant": invariant,
        "protein_path": str(record.protein_path) if record.protein_path else None,
        "ligand_path": str(record.ligand_path) if record.ligand_path else None,
    }
    staged_input = None
    try:
        _assert_structure_paths(record)
        input_hash = None
        if stage_structure_paths:
            if tool.config.pdb_folder is None:
                raise ValueError("structure staging requires a configured pdb_folder")
            staged_input = stage_protein_ligand_input(
                record,
                staging_root=tool.config.pdb_folder,
                create_links=not dry_run,
            )
            input_hash = staged_input.combined_sha256
        if representation_spec is None:
            result = tool.compute_one(
                record.pdb_id,
                invariant,
                dry_run=dry_run,
                input_hash=input_hash,
            )
        else:
            result = tool.compute_with_spec(
                record.pdb_id,
                invariant,
                representation_spec,
                dry_run=dry_run,
                input_hash=input_hash,
            )
        cost = result.cost
        return {
            **base,
            "legacy": result.legacy_name,
            "status": result.status,
            "output_path": str(result.output_path),
            "wall_seconds": cost.wall_seconds if cost else time.perf_counter() - started,
            "cpu_seconds": cost.cpu_seconds if cost else None,
            "cpu_core_hours": cost.cpu_core_hours if cost else None,
            "structure_inputs": staged_input.to_manifest_dict() if staged_input else None,
            "error_type": None,
            "error": None,
            "traceback": None,
        }
    except Exception as exc:
        return {
            **base,
            "legacy": None,
            "status": "failed",
            "output_path": None,
            "wall_seconds": time.perf_counter() - started,
            "cpu_seconds": None,
            "cpu_core_hours": None,
            "structure_inputs": staged_input.to_manifest_dict() if staged_input else None,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }


def _assert_structure_paths(record: CasfRecord) -> None:
    if record.protein_path is None or record.ligand_path is None:
        raise ValueError(
            f"Sample {record.pdb_id!r} requires explicit protein and ligand paths"
        )
    missing = [path for path in (record.protein_path, record.ligand_path) if not path.exists()]
    if missing:
        raise FileNotFoundError("; ".join(str(path) for path in missing))


if __name__ == "__main__":
    raise SystemExit(main())
