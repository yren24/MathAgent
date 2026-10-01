from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from mint_scout.config import load_yaml
from mint_scout.invariants.plbind_tools import PLBindFeatureTool, PLBindToolConfig
from mint_scout.representation import RepresentationSpec


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compute one or more PLBind legacy features as isolated tool calls.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--invariant", required=True, choices=("PH", "PL", "CA", "FPRC", "EIC"))
    parser.add_argument("--sample-id", action="append", required=True)
    parser.add_argument("--legacy-root", type=Path, default=None)
    parser.add_argument("--pdb-folder", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--representation-spec", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    representation_spec = RepresentationSpec.read(args.representation_spec) if args.representation_spec else None
    config = load_yaml(args.config)
    tool_config = _tool_config_from_task(config, args=args)
    if representation_spec is not None:
        tool_config = replace(tool_config, representation_mode=representation_spec.mode.value)
    tool = PLBindFeatureTool(tool_config)
    for sample_id in args.sample_id:
        if representation_spec is None:
            result = tool.compute_one(sample_id.lower(), args.invariant, dry_run=args.dry_run)
        else:
            result = tool.compute_with_spec(
                sample_id.lower(),
                args.invariant,
                representation_spec,
                dry_run=args.dry_run,
            )
        print(
            "sample_id={sample_id} invariant={invariant} legacy={legacy} status={status} output={output}".format(
                sample_id=result.sample_id,
                invariant=result.invariant_name,
                legacy=result.legacy_name,
                status=result.status,
                output=result.output_path,
            )
        )
    return 0


def _tool_config_from_task(config: dict, *, args: argparse.Namespace) -> PLBindToolConfig:
    legacy = config.get("legacy", {})
    data_audit = config.get("data_audit", {})
    casf = data_audit.get("casf", {}) if isinstance(data_audit, dict) else {}
    feature_generation = config.get("feature_generation", {})
    if not isinstance(feature_generation, dict):
        raise ValueError("feature_generation config must be a mapping")
    configured_legacy_root = legacy.get("plbind_root") if isinstance(legacy, dict) else None
    legacy_root_value = args.legacy_root or configured_legacy_root
    if not legacy_root_value:
        raise ValueError("legacy.plbind_root or --legacy-root is required")
    legacy_root = Path(legacy_root_value)
    output_root = args.output_root or Path(feature_generation.get("output_root", "cache/plbind_features"))
    if args.pdb_folder is not None:
        pdb_folder = args.pdb_folder
    elif config.get("dataset_manifest") is not None:
        pdb_folder = Path(
            feature_generation.get(
                "structure_staging_root",
                Path(output_root).parent / "plbind_structure_staging",
            )
        )
    else:
        pdb_folder = Path(casf.get("structures_root", ""))
    validate_output_shape = feature_generation.get("validate_output_shape", True)
    if not isinstance(validate_output_shape, bool):
        raise ValueError("feature_generation.validate_output_shape must be true or false")
    if not str(pdb_folder):
        raise ValueError(
            "a PLBind structures root, structure_staging_root, or --pdb-folder is required"
        )
    return PLBindToolConfig(
        legacy_root=legacy_root.expanduser(),
        pdb_folder=pdb_folder.expanduser(),
        output_root=output_root.expanduser(),
        representation_mode=str(config.get("representation_mode", "legacy_casf")),
        dataset_id=str(config.get("task_id", "dataset")),
        validate_output_shape=validate_output_shape,
    )


if __name__ == "__main__":
    raise SystemExit(main())
