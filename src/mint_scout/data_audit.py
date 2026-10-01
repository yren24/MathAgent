from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from mint_scout.compute_feature_batch import _selected_records
from mint_scout.config import load_yaml
from mint_scout.data.casf_index import load_casf_records, missing_structure_paths
from mint_scout.data.element_inventory import (
    build_element_inventory_report,
    iter_elements_from_file,
    render_markdown_report,
)
from mint_scout.data.element_pairs import ElementPairSchema, make_casf_protein_ligand_schema, make_toxicity_schema
from mint_scout.data.structure_staging import file_sha256
from mint_scout.invariants.manifest import stable_hash


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit molecular element frequencies before freezing a schema.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--protein-glob", action="append", default=[])
    parser.add_argument("--ligand-glob", action="append", default=[])
    parser.add_argument("--output-json", type=Path, default=Path("docs/ELEMENT_INVENTORY.json"))
    parser.add_argument("--output-md", type=Path, default=Path("docs/ELEMENT_INVENTORY.md"))
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--manifest-split", choices=("all", "train", "test"), default=None)
    parser.add_argument("--allow-empty", action="store_true")
    args = parser.parse_args(argv)

    config = load_yaml(args.config)
    data_audit_config = config.get("data_audit", {})
    if not isinstance(data_audit_config, dict):
        raise ValueError("data_audit config must be a mapping")
    manifest_split = args.manifest_split or str(
        data_audit_config.get("manifest_split", "train")
    )
    if manifest_split not in {"all", "train", "test"}:
        raise ValueError("data_audit.manifest_split must be all, train, or test")
    sample_ids: tuple[str, ...] = ()
    labels: tuple[float, ...] = ()
    try:
        manifest_inputs = _manifest_modeling_inputs(
            config,
            config_path=args.config,
            max_files=args.max_files,
            split=manifest_split,
        )
        if manifest_inputs is not None:
            protein_paths, ligand_paths, sample_ids, labels = manifest_inputs
        else:
            protein_paths, ligand_paths = _resolve_input_paths(
                cli_protein_globs=args.protein_glob,
                cli_ligand_globs=args.ligand_glob,
                data_audit_config=data_audit_config,
                max_files=args.max_files,
            )
    except FileNotFoundError as exc:
        if not args.allow_empty:
            raise
        print(f"warning: {exc}; continuing with an empty structural audit because --allow-empty was set")
        protein_paths, ligand_paths = (), ()
    if not args.allow_empty and not protein_paths and not ligand_paths:
        raise SystemExit("No files matched. Pass --allow-empty for a structural dry run.")

    schema = _schema_from_config(config)
    report = build_element_inventory_report(
        schema=schema,
        protein_paths=protein_paths,
        ligand_paths=ligand_paths,
        tolerated_elements_by_role=_tolerated_elements_by_role(data_audit_config),
    )

    target_summary = _numeric_summary(labels)
    structure_size_summary = _structure_size_summary(
        protein_paths,
        ligand_paths,
        paired=bool(sample_ids),
    )
    payload = {
        **report.to_dict(),
        "report_schema": "mint-agent.dataset-audit.v1",
        "audit_input_hash": dataset_audit_input_hash(config, config_path=args.config),
        "task_id": config.get("task_id"),
        "modeling_scope": manifest_split if sample_ids else "configured_inputs",
        "sample_count": len(sample_ids) if sample_ids else max(
            (role.file_count for role in report.roles), default=0
        ),
        "sample_ids": list(sample_ids),
        "target_summary": target_summary,
        "structure_size_summary": structure_size_summary,
        "status": "REVIEW" if report.schema_review_required else "PASS",
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    markdown = render_markdown_report(report).rstrip() + "\n\n" + _render_dataset_profile_markdown(
        target_summary=target_summary,
        structure_size_summary=structure_size_summary,
    )
    args.output_md.write_text(markdown, encoding="utf-8")

    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_md}")
    print(f"schema_review_required={str(report.schema_review_required).lower()}")
    return 0


def _manifest_modeling_inputs(
    config: dict,
    *,
    config_path: Path,
    max_files: int | None,
    split: str = "train",
) -> tuple[tuple[Path, ...], tuple[Path, ...], tuple[str, ...], tuple[float, ...]] | None:
    if config.get("dataset_manifest") is None:
        return None
    records = _selected_records(
        config,
        sample_ids=None,
        split=split,
        offset=0,
        limit=max_files,
        config_path=config_path,
    )
    protein_paths: list[Path] = []
    ligand_paths: list[Path] = []
    sample_ids: list[str] = []
    labels: list[float] = []
    for record in records:
        if record.protein_path is None or record.ligand_path is None:
            raise FileNotFoundError(
                f"Sample {record.pdb_id!r} is missing protein/ligand structure paths"
            )
        if record.label is None:
            raise ValueError(
                f"Modeling sample {record.pdb_id!r} has no numeric target"
            )
        protein_paths.append(record.protein_path)
        ligand_paths.append(record.ligand_path)
        sample_ids.append(record.pdb_id)
        labels.append(float(record.label))
    return (
        tuple(protein_paths),
        tuple(ligand_paths),
        tuple(sample_ids),
        tuple(labels),
    )


def dataset_audit_input_hash(config: dict, *, config_path: str | Path) -> str:
    raw_manifest = config.get("dataset_manifest")
    manifest_sha256 = None
    if isinstance(raw_manifest, dict) and raw_manifest.get("path"):
        manifest_path = Path(str(raw_manifest["path"])).expanduser()
        if not manifest_path.is_absolute():
            manifest_path = Path(config_path).expanduser().resolve().parent / manifest_path
        if manifest_path.is_file():
            manifest_sha256 = file_sha256(manifest_path)
    return stable_hash(
        {
            "hash_schema": "mint-agent.dataset-audit-input.v1",
            "system_type": config.get("system_type"),
            "pair_schema": config.get("pair_schema"),
            "dataset_manifest": raw_manifest,
            "dataset_manifest_sha256": manifest_sha256,
            "data_audit": config.get("data_audit"),
        }
    )


def _numeric_summary(values: Iterable[float]) -> dict[str, object] | None:
    values = tuple(values)
    if not values:
        return None
    array = np.asarray(values, dtype=float)
    quantiles = np.quantile(array, (0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0))
    return {
        "count": int(array.size),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "quantiles": {
            name: float(value)
            for name, value in zip(
                ("q00", "q05", "q25", "q50", "q75", "q95", "q100"),
                quantiles,
            )
        },
    }


def _structure_size_summary(
    protein_paths: tuple[Path, ...],
    ligand_paths: tuple[Path, ...],
    *,
    paired: bool,
) -> dict[str, object]:
    protein_counts = tuple(sum(1 for _ in iter_elements_from_file(path)) for path in protein_paths)
    ligand_counts = tuple(sum(1 for _ in iter_elements_from_file(path)) for path in ligand_paths)
    total_counts: tuple[int, ...] = ()
    if paired:
        if len(protein_counts) != len(ligand_counts):
            raise ValueError("Paired manifest inputs must have matching protein and ligand counts")
        total_counts = tuple(
            protein_count + ligand_count
            for protein_count, ligand_count in zip(protein_counts, ligand_counts)
        )
    return {
        "protein_atom_count": _numeric_summary(protein_counts),
        "ligand_atom_count": _numeric_summary(ligand_counts),
        "total_atom_count": _numeric_summary(total_counts),
    }


def _render_dataset_profile_markdown(
    *,
    target_summary: dict[str, object] | None,
    structure_size_summary: dict[str, object],
) -> str:
    lines = [
        "# Dataset Profile",
        "",
        "These are descriptive statistics, not selection thresholds or scientific constants.",
        "",
        "| Quantity | Count | Min | Q05 | Median | Q95 | Max | Mean | Std |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    rows = [("Target", target_summary)]
    rows.extend(
        (
            label,
            structure_size_summary.get(key),
        )
        for label, key in (
            ("Protein atoms", "protein_atom_count"),
            ("Ligand atoms", "ligand_atom_count"),
            ("Total atoms", "total_atom_count"),
        )
    )
    for label, summary in rows:
        if not isinstance(summary, dict):
            lines.append(f"| {label} | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |")
            continue
        quantiles = summary["quantiles"]
        assert isinstance(quantiles, dict)
        values = (
            summary["count"],
            quantiles["q00"],
            quantiles["q05"],
            quantiles["q50"],
            quantiles["q95"],
            quantiles["q100"],
            summary["mean"],
            summary["std"],
        )
        lines.append("| " + label + " | " + " | ".join(_format_profile_value(value) for value in values) + " |")
    lines.append("")
    return "\n".join(lines)


def _format_profile_value(value: object) -> str:
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _merge_globs(cli_globs: list[str], config_globs: object) -> list[str]:
    merged = list(cli_globs)
    if isinstance(config_globs, str):
        merged.append(config_globs)
    elif isinstance(config_globs, list):
        merged.extend(str(item) for item in config_globs)
    return merged


def _expand_globs(patterns: Iterable[str], *, max_files: int | None) -> tuple[Path, ...]:
    paths: list[Path] = []
    for pattern in patterns:
        expanded = glob.glob(str(Path(pattern).expanduser()), recursive=True)
        paths.extend(Path(path) for path in expanded)
    unique_paths = tuple(sorted({path for path in paths if path.is_file()}))
    if max_files is None:
        return unique_paths
    if max_files < 0:
        raise ValueError("--max-files must be non-negative")
    return unique_paths[:max_files]


def _resolve_input_paths(
    *,
    cli_protein_globs: list[str],
    cli_ligand_globs: list[str],
    data_audit_config: dict,
    max_files: int | None,
) -> tuple[tuple[Path, ...], tuple[Path, ...]]:
    protein_globs = _merge_globs(cli_protein_globs, data_audit_config.get("protein_globs", []))
    ligand_globs = _merge_globs(cli_ligand_globs, data_audit_config.get("ligand_globs", []))
    if protein_globs or ligand_globs:
        return (
            _expand_globs(protein_globs, max_files=max_files),
            _expand_globs(ligand_globs, max_files=max_files),
        )

    casf_config = data_audit_config.get("casf")
    if not isinstance(casf_config, dict):
        return (), ()

    records = load_casf_records(
        index_root=Path(casf_config["index_root"]).expanduser(),
        structures_root=Path(casf_config["structures_root"]).expanduser(),
        year=int(casf_config.get("year", 2016)),
        protein_template=str(casf_config.get("protein_template", "{pdb}_pocket.pdb")),
        ligand_template=str(casf_config.get("ligand_template", "{pdb}_ligand.mol2")),
    )
    source = str(casf_config.get("source", "refined_index"))
    if source == "train_test":
        records = tuple(record for record in records if record.split in {"train", "test"})
    elif source != "refined_index":
        raise ValueError(f"Unsupported CASF audit source: {source}")

    missing = missing_structure_paths(records)
    if missing:
        preview = ", ".join(str(path) for path in missing[:5])
        raise FileNotFoundError(f"Missing {len(missing)} expected CASF structure files; first missing: {preview}")

    protein_paths = tuple(record.protein_path for record in records if record.protein_path is not None)
    ligand_paths = tuple(record.ligand_path for record in records if record.ligand_path is not None)
    if max_files is not None:
        protein_paths = protein_paths[:max_files]
        ligand_paths = ligand_paths[:max_files]
    return protein_paths, ligand_paths


def _schema_from_config(config: dict) -> ElementPairSchema:
    system_type = config.get("system_type")
    if system_type == "protein_ligand":
        pair_schema = config.get("pair_schema", {})
        ligand_elements = pair_schema.get("ligand_elements")
        schema_id = pair_schema.get("schema_id", "casf_protein_ligand_40_v1")
        if ligand_elements is None:
            return make_casf_protein_ligand_schema(schema_id=schema_id)
        return make_casf_protein_ligand_schema(schema_id=schema_id, ligand_elements=ligand_elements)
    if system_type == "small_molecule":
        return make_toxicity_schema()
    raise ValueError(f"Unsupported system_type for data audit: {system_type}")


def _tolerated_elements_by_role(data_audit_config: dict) -> dict[str, tuple[str, ...]]:
    raw = data_audit_config.get("tolerated_out_of_schema", {})
    if not isinstance(raw, dict):
        raise ValueError("data_audit.tolerated_out_of_schema must be a mapping")
    tolerated: dict[str, tuple[str, ...]] = {}
    for role, values in raw.items():
        if isinstance(values, str):
            tolerated[str(role)] = (values,)
        elif isinstance(values, list):
            tolerated[str(role)] = tuple(str(value) for value in values)
        else:
            raise ValueError(f"Expected list of tolerated elements for role {role}")
    return tolerated


if __name__ == "__main__":
    raise SystemExit(main())
