from __future__ import annotations

import csv
import json
import math
import tarfile
from pathlib import Path
from typing import Any, Callable, Mapping

from mint_scout.config import load_yaml
from mint_scout.data.atom3d_lba import (
    SOURCE_MD5 as ATOM3D_LBA_SOURCE_MD5,
    SOURCE_URL as ATOM3D_LBA_SOURCE_URL,
    prepare_atom3d_lba,
)
from mint_scout.data.bdb2020plus import prepare_bdb2020plus
from mint_scout.data.hiqbind import (
    HiQBindSubsetConfig,
    METADATA_MD5 as HIQBIND_METADATA_MD5,
    METADATA_URL as HIQBIND_METADATA_URL,
    SOURCE_MD5 as HIQBIND_SOURCE_MD5,
    SOURCE_URL as HIQBIND_SOURCE_URL,
    prepare_hiqbind,
)
from mint_scout.data.structure_staging import file_sha256
from mint_scout.invariants.manifest import stable_hash


PREPARATION_SCHEMA = "mint-agent.dataset-preparation.v1"
SUPPORTED_SPLITS = {"train", "validation", "test", "inference"}
PreparationProvider = Callable[[Mapping[str, Any], Path, str], dict[str, Any]]


def prepare_dataset_from_config(
    task_config_path: str | Path,
    *,
    report_path: str | Path,
) -> dict[str, Any]:
    config_path = Path(task_config_path).expanduser().resolve()
    task = load_yaml(config_path)
    dataset_id = str(task.get("task_id") or "")
    if not dataset_id:
        raise ValueError("task configuration requires task_id")
    raw = task.get("dataset_preparation")
    if not isinstance(raw, Mapping):
        raise ValueError("dataset_preparation must be a mapping")
    provider_name = str(raw.get("provider") or "")
    provider = _PROVIDERS.get(provider_name)
    if provider is None:
        raise ValueError(
            f"Unsupported dataset preparation provider {provider_name!r}; "
            f"expected one of {sorted(_PROVIDERS)}"
        )

    expected_manifest = configured_manifest_path(task, config_path=config_path)
    configured_output = _path_from_config(
        raw.get("output_manifest"), config_path=config_path, name="output_manifest"
    )
    if configured_output != expected_manifest:
        raise ValueError(
            "dataset_preparation.output_manifest must match dataset_manifest.path"
        )

    provider_report = provider(raw, config_path, dataset_id)
    manifest_path = Path(str(provider_report["manifest"])).expanduser().resolve()
    if manifest_path != expected_manifest:
        raise ValueError("preparation provider wrote an unexpected manifest path")
    sample_ids = tuple(str(value) for value in provider_report.get("sample_ids", ()))
    if not sample_ids or len(set(sample_ids)) != len(sample_ids):
        raise ValueError("prepared sample IDs must be nonempty and unique")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"preparation did not create manifest: {manifest_path}")

    payload = {
        "report_schema": PREPARATION_SCHEMA,
        "dataset_id": dataset_id,
        "task_id": dataset_id,
        "evidence_scope": "design",
        "status": "PASS",
        "provider": provider_name,
        "task_config": str(config_path),
        "manifest": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "sample_count": len(sample_ids),
        "sample_ids": list(sample_ids),
        "sample_order_hash": stable_hash(sample_ids),
        "provider_report": provider_report,
    }
    destination = Path(report_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload


def configured_manifest_path(
    task: Mapping[str, Any], *, config_path: str | Path
) -> Path:
    raw_manifest = task.get("dataset_manifest")
    if not isinstance(raw_manifest, Mapping) or not raw_manifest.get("path"):
        raise ValueError("dataset_manifest.path is required for dataset preparation")
    path = Path(str(raw_manifest["path"])).expanduser()
    if not path.is_absolute():
        path = Path(config_path).expanduser().resolve().parent / path
    return path.resolve()


def _prepare_paired_structure_manifest(
    raw: Mapping[str, Any], config_path: Path, dataset_id: str
) -> dict[str, Any]:
    metadata_path = _path_from_config(
        raw.get("metadata_path"), config_path=config_path, name="metadata_path"
    )
    structures_root = _path_from_config(
        raw.get("structures_root"), config_path=config_path, name="structures_root"
    )
    output_manifest = _path_from_config(
        raw.get("output_manifest"), config_path=config_path, name="output_manifest"
    )
    columns = _mapping(raw.get("columns", {}), "dataset_preparation.columns")
    templates = _mapping(raw.get("templates"), "dataset_preparation.templates")
    identifier_columns = _mapping(
        raw.get("identifiers", {}), "dataset_preparation.identifiers"
    )
    sample_column = str(columns.get("sample_id", "sample_id"))
    target_column = str(columns.get("target", "target"))
    split_column = str(columns.get("split", "split"))
    protein_template = _required_text(templates, "protein")
    ligand_template = _required_text(templates, "ligand")

    with metadata_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        headers = set(reader.fieldnames or ())
        required = {sample_column, target_column}
        missing = sorted(required - headers)
        if missing:
            raise ValueError(f"metadata CSV is missing columns: {missing}")
        rows = list(reader)
    if not rows:
        raise ValueError("metadata CSV contains no samples")

    manifest_rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for row_number, row in enumerate(rows, start=2):
        sample_id = str(row.get(sample_column) or "").strip()
        if not sample_id:
            raise ValueError(f"metadata row {row_number} has no sample ID")
        if sample_id in seen:
            raise ValueError(f"duplicate sample ID in metadata: {sample_id}")
        seen.add(sample_id)
        target_text = str(row.get(target_column) or "").strip()
        if target_text:
            try:
                target = float(target_text)
            except ValueError as exc:
                raise ValueError(
                    f"metadata row {row_number} target is not numeric: {target_text!r}"
                ) from exc
            if not math.isfinite(target):
                raise ValueError(f"metadata row {row_number} target must be finite")
        split = str(row.get(split_column) or "").strip().lower()
        if split and split not in SUPPORTED_SPLITS:
            raise ValueError(
                f"metadata row {row_number} has unsupported split {split!r}"
            )
        format_values = {**row, "sample_id": sample_id}
        protein_path = _template_path(
            structures_root, protein_template, format_values, row_number=row_number
        )
        ligand_path = _template_path(
            structures_root, ligand_template, format_values, row_number=row_number
        )
        missing_paths = [path for path in (protein_path, ligand_path) if not path.is_file()]
        if missing_paths:
            raise FileNotFoundError(
                f"metadata row {row_number} is missing structure: {missing_paths[0]}"
            )
        output_row = {
            "sample_id": sample_id,
            "target": target_text,
            "split": split,
            "protein_path": str(protein_path),
            "ligand_path": str(ligand_path),
        }
        for identifier, source_column in identifier_columns.items():
            output_row[str(identifier)] = str(row.get(str(source_column)) or "").strip()
        manifest_rows.append(output_row)

    fieldnames = [
        "sample_id",
        "target",
        "split",
        "protein_path",
        "ligand_path",
        *[str(value) for value in identifier_columns],
    ]
    _write_csv(output_manifest, manifest_rows, fieldnames)
    return {
        "provider": "paired_structure_manifest",
        "dataset_id": dataset_id,
        "metadata_path": str(metadata_path),
        "metadata_sha256": file_sha256(metadata_path),
        "structures_root": str(structures_root),
        "manifest": str(output_manifest),
        "sample_ids": [row["sample_id"] for row in manifest_rows],
        "labeled_count": sum(bool(row["target"]) for row in manifest_rows),
        "unlabeled_count": sum(not bool(row["target"]) for row in manifest_rows),
        "preprocessing": "path normalization and strict validation only",
    }


def _prepare_bdb2020plus(
    raw: Mapping[str, Any], config_path: Path, dataset_id: str
) -> dict[str, Any]:
    source_root = _path_from_config(
        raw.get("source_root"), config_path=config_path, name="source_root"
    )
    output_root = _path_from_config(
        raw.get("output_root"), config_path=config_path, name="output_root"
    )
    output_manifest = _path_from_config(
        raw.get("output_manifest"), config_path=config_path, name="output_manifest"
    )
    source_archive = _optional_path(raw.get("source_archive"), config_path=config_path)
    extracted_archive = False
    if not source_root.is_dir():
        if source_archive is None or not source_archive.is_file():
            raise FileNotFoundError(
                f"BDB2020+ source root is missing and no source archive is available: {source_root}"
            )
        extract_root = _optional_path(raw.get("extract_root"), config_path=config_path)
        extract_root = extract_root or source_root.parent
        _extract_tar_archive(source_archive, extract_root)
        extracted_archive = True
        if not source_root.is_dir():
            raise FileNotFoundError(
                f"Archive did not create configured source_root: {source_root}"
            )
    detail_report = output_manifest.with_suffix(".preparation-detail.json")
    raw_split = raw.get("output_split")
    output_split = None if raw_split in {None, "", "none"} else str(raw_split).lower()
    if output_split is not None and output_split not in SUPPORTED_SPLITS:
        raise ValueError(f"unsupported BDB2020+ output_split: {output_split!r}")
    detail = prepare_bdb2020plus(
        source_root=source_root,
        output_root=output_root,
        manifest_path=output_manifest,
        audit_report_path=detail_report,
        source_archive=source_archive,
        obabel=str(raw.get("obabel") or "obabel"),
        pocket_cutoff_angstrom=float(raw.get("pocket_cutoff_angstrom", 10.0)),
        add_hydrogens=bool(raw.get("add_hydrogens", True)),
        output_split=output_split,
    )
    return {
        "provider": "bdb2020plus",
        "dataset_id": dataset_id,
        "manifest": str(output_manifest),
        "sample_ids": [str(sample["sample_id"]) for sample in detail["samples"]],
        "detail_report": str(detail_report),
        "detail_report_sha256": file_sha256(detail_report),
        "output_split": output_split,
        "source_archive_extracted": extracted_archive,
        "preprocessing": detail["preprocessing"],
    }


def _prepare_atom3d_lba(
    raw: Mapping[str, Any], config_path: Path, dataset_id: str
) -> dict[str, Any]:
    source_root = _path_from_config(
        raw.get("source_root"), config_path=config_path, name="source_root"
    )
    source_archive = _path_from_config(
        raw.get("source_archive"), config_path=config_path, name="source_archive"
    )
    extract_root = _path_from_config(
        raw.get("extract_root"), config_path=config_path, name="extract_root"
    )
    output_root = _path_from_config(
        raw.get("output_root"), config_path=config_path, name="output_root"
    )
    output_manifest = _path_from_config(
        raw.get("output_manifest"), config_path=config_path, name="output_manifest"
    )
    raw_splits = raw.get("source_splits", ["test"])
    if not isinstance(raw_splits, list):
        raise ValueError("dataset_preparation.source_splits must be a list")
    raw_output_split = raw.get("output_split")
    output_split = (
        None
        if raw_output_split in {None, "", "none"}
        else str(raw_output_split).lower()
    )
    raw_split_mapping = raw.get("output_split_by_source")
    if raw_split_mapping is None:
        output_split_by_source = None
    else:
        mapping = _mapping(
            raw_split_mapping, "dataset_preparation.output_split_by_source"
        )
        output_split_by_source = {
            str(source).lower(): str(destination).lower()
            for source, destination in mapping.items()
        }
    raw_limit = raw.get("max_samples")
    max_samples = None if raw_limit in {None, "", "all"} else int(raw_limit)
    raw_expected = raw.get("expected_sample_count")
    expected_sample_count = (
        None if raw_expected in {None, ""} else int(raw_expected)
    )
    detail_report = output_manifest.with_suffix(".preparation-detail.json")
    detail = prepare_atom3d_lba(
        source_root=source_root,
        source_archive=source_archive,
        extract_root=extract_root,
        output_root=output_root,
        manifest_path=output_manifest,
        audit_report_path=detail_report,
        source_splits=tuple(str(value) for value in raw_splits),
        output_split=output_split,
        output_split_by_source=output_split_by_source,
        source_url=str(raw.get("source_url") or ATOM3D_LBA_SOURCE_URL),
        source_archive_md5=str(
            raw.get("source_archive_md5") or ATOM3D_LBA_SOURCE_MD5
        ),
        max_samples=max_samples,
        expected_sample_count=expected_sample_count,
    )
    return {
        "provider": "atom3d_lba",
        "dataset_id": dataset_id,
        "manifest": str(output_manifest),
        "sample_ids": [str(sample["sample_id"]) for sample in detail["samples"]],
        "detail_report": str(detail_report),
        "detail_report_sha256": file_sha256(detail_report),
        "source_doi": detail["source_doi"],
        "source_splits": detail["source_splits"],
        "output_split": output_split,
        "output_split_by_source": output_split_by_source,
        "preprocessing": detail["preprocessing"],
    }


def _prepare_hiqbind(
    raw: Mapping[str, Any], config_path: Path, dataset_id: str
) -> dict[str, Any]:
    source_root = _path_from_config(
        raw.get("source_root"), config_path=config_path, name="source_root"
    )
    metadata_path = _path_from_config(
        raw.get("metadata_path"), config_path=config_path, name="metadata_path"
    )
    output_root = _path_from_config(
        raw.get("output_root"), config_path=config_path, name="output_root"
    )
    output_manifest = _path_from_config(
        raw.get("output_manifest"), config_path=config_path, name="output_manifest"
    )
    source_archive = _optional_path(
        raw.get("source_archive"), config_path=config_path
    )
    extract_root = _optional_path(raw.get("extract_root"), config_path=config_path)
    measurements = raw.get("measurements")
    affinity_signs = raw.get("affinity_signs")
    if not isinstance(measurements, list) or not measurements:
        raise ValueError("dataset_preparation.measurements must be a nonempty list")
    if not isinstance(affinity_signs, list) or not affinity_signs:
        raise ValueError("dataset_preparation.affinity_signs must be a nonempty list")
    raw_output_split = raw.get("output_split")
    output_split = (
        None
        if raw_output_split in {None, "", "none"}
        else str(raw_output_split).lower()
    )
    raw_limit = raw.get("max_samples")
    max_samples = None if raw_limit in {None, "", "all"} else int(raw_limit)
    raw_expected = raw.get("expected_sample_count")
    expected_sample_count = (
        None if raw_expected in {None, ""} else int(raw_expected)
    )
    detail_report = output_manifest.with_suffix(".preparation-detail.json")
    detail = prepare_hiqbind(
        source_root=source_root,
        metadata_path=metadata_path,
        output_root=output_root,
        manifest_path=output_manifest,
        audit_report_path=detail_report,
        measurements=tuple(str(value) for value in measurements),
        affinity_signs=tuple(str(value) for value in affinity_signs),
        output_split=output_split,
        source_archive=source_archive,
        extract_root=extract_root,
        source_url=str(raw.get("source_url") or HIQBIND_SOURCE_URL),
        source_archive_md5=str(
            raw.get("source_archive_md5") or HIQBIND_SOURCE_MD5
        ),
        metadata_url=str(raw.get("metadata_url") or HIQBIND_METADATA_URL),
        metadata_md5=str(raw.get("metadata_md5") or HIQBIND_METADATA_MD5),
        max_samples=max_samples,
        expected_sample_count=expected_sample_count,
        obabel=str(raw.get("obabel") or "obabel"),
        pocket_cutoff_angstrom=float(raw.get("pocket_cutoff_angstrom", 10.0)),
        add_hydrogens=bool(raw.get("add_hydrogens", True)),
        invalid_affinity_policy=str(raw.get("invalid_affinity_policy") or "error"),
        subset_config=HiQBindSubsetConfig.from_mapping(
            _mapping(raw.get("selection", {}), "dataset_preparation.selection")
        ),
    )
    return {
        "provider": "hiqbind",
        "dataset_id": dataset_id,
        "manifest": str(output_manifest),
        "sample_ids": [str(sample["sample_id"]) for sample in detail["samples"]],
        "detail_report": str(detail_report),
        "detail_report_sha256": file_sha256(detail_report),
        "source_doi": detail["source_doi"],
        "source_license": detail["source_license"],
        "output_split": output_split,
        "preprocessing": detail["preprocessing"],
        "subset_selection": detail["subset_selection"],
    }


def _path_from_config(value: object, *, config_path: Path, name: str) -> Path:
    if value is None or not str(value):
        raise ValueError(f"dataset_preparation.{name} is required")
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = config_path.parent / path
    return path.resolve()


def _optional_path(value: object, *, config_path: Path) -> Path | None:
    if value is None or not str(value):
        return None
    return _path_from_config(value, config_path=config_path, name="source_archive")


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _required_text(values: Mapping[str, Any], key: str) -> str:
    value = str(values.get(key) or "")
    if not value:
        raise ValueError(f"dataset_preparation.templates.{key} is required")
    return value


def _template_path(
    root: Path,
    template: str,
    values: Mapping[str, Any],
    *,
    row_number: int,
) -> Path:
    try:
        rendered = template.format_map(values)
    except KeyError as exc:
        raise ValueError(
            f"metadata row {row_number} lacks template field {exc.args[0]!r}"
        ) from exc
    path = Path(rendered).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _write_csv(
    path: Path, rows: list[dict[str, str]], fieldnames: list[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _extract_tar_archive(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with tarfile.open(archive, mode="r:*") as handle:
        members = handle.getmembers()
        for member in members:
            target = (root / member.name).resolve()
            if root != target and root not in target.parents:
                raise ValueError(f"archive member escapes extraction root: {member.name}")
            if member.issym() or member.islnk():
                raise ValueError(f"archive links are not allowed: {member.name}")
        handle.extractall(root, members=members)


_PROVIDERS: dict[str, PreparationProvider] = {
    "paired_structure_manifest": _prepare_paired_structure_manifest,
    "bdb2020plus": _prepare_bdb2020plus,
    "atom3d_lba": _prepare_atom3d_lba,
    "hiqbind": _prepare_hiqbind,
}
