from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import sys
import tarfile
import urllib.request
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from mint_scout.data.bdb2020plus import (
    convert_sdf_to_mol2,
    write_ligand_centered_pocket,
)
from mint_scout.data.structure_staging import file_sha256
from mint_scout.invariants.manifest import stable_hash
from mint_scout.scout.sampling import ProbeSamplingConfig, select_probe_samples


SOURCE_DOI = "10.6084/m9.figshare.27430305.v3"
SOURCE_URL = "https://ndownloader.figshare.com/files/52379345"
SOURCE_MD5 = "b6668cc3a54a5bd686683924037a18d6"
METADATA_URL = "https://ndownloader.figshare.com/files/52379375"
METADATA_MD5 = "b473dcfe797dfb3d7dc9ffbec8ccbbd9"
SOURCE_LICENSE = "CC BY 4.0"
SUPPORTED_MEASUREMENTS = {"kd", "ki", "ic50"}
SUPPORTED_SIGNS = {"=", ">=", "<=", "~"}
SUPPORTED_OUTPUT_SPLITS = {"train", "validation", "test", "inference"}
UNIT_TO_MOLAR = {
    "fm": 1e-15,
    "pm": 1e-12,
    "nm": 1e-9,
    "um": 1e-6,
    "mm": 1e-3,
    "m": 1.0,
}


@dataclass(frozen=True)
class HiQBindRecord:
    source_key: str
    pdb_id: str
    ligand_name: str
    ligand_chain: str
    ligand_residue_number: str
    measurement: str
    sign: str
    affinity_value: float
    affinity_unit: str
    target_p_affinity: float
    year: int
    affinity_source: str
    ligand_heavy_atoms: int | None

    @property
    def sample_id(self) -> str:
        return f"{self.pdb_id.lower()}-{stable_hash(self.source_key)[:12]}"


@dataclass(frozen=True)
class HiQBindSubsetConfig:
    strategy: str = "source_order"
    random_seed: int = 2026
    target_quantile_bins: int = 10
    ligand_size_quantile_bins: int = 5
    max_per_pdb_id: int | None = None

    def __post_init__(self) -> None:
        if self.strategy not in {"source_order", "stratified"}:
            raise ValueError(
                "HiQBind subset strategy must be source_order or stratified"
            )
        if self.target_quantile_bins < 1 or self.ligand_size_quantile_bins < 1:
            raise ValueError("HiQBind subset quantile-bin counts must be positive")
        if self.max_per_pdb_id is not None and self.max_per_pdb_id < 1:
            raise ValueError("HiQBind subset max_per_pdb_id must be positive")

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any] | None
    ) -> "HiQBindSubsetConfig":
        return cls(**dict(value or {}))

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "random_seed": self.random_seed,
            "target_quantile_bins": self.target_quantile_bins,
            "ligand_size_quantile_bins": self.ligand_size_quantile_bins,
            "max_per_pdb_id": self.max_per_pdb_id,
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect or prepare the public HiQBind dataset.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    summary = subparsers.add_parser("summarize")
    summary.add_argument("--metadata", type=Path, required=True)
    summary.add_argument("--output", type=Path, required=True)
    preview = subparsers.add_parser("preview-subset")
    preview.add_argument("--metadata", type=Path, required=True)
    preview.add_argument("--measurement", action="append", required=True)
    preview.add_argument("--affinity-sign", action="append", required=True)
    preview.add_argument("--max-samples", type=int, required=True)
    preview.add_argument(
        "--strategy", choices=("source_order", "stratified"), required=True
    )
    preview.add_argument("--random-seed", type=int, required=True)
    preview.add_argument("--target-quantile-bins", type=int, required=True)
    preview.add_argument("--ligand-size-quantile-bins", type=int, required=True)
    preview.add_argument("--max-per-pdb-id", type=int, default=None)
    preview.add_argument(
        "--invalid-affinity-policy", choices=("error", "skip"), default="error"
    )
    preview.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.command == "summarize":
        payload = summarize_hiqbind_metadata(args.metadata)
    else:
        eligible = load_hiqbind_records(
            args.metadata,
            measurements=args.measurement,
            affinity_signs=args.affinity_sign,
            invalid_affinity_policy=args.invalid_affinity_policy,
        )
        selected, selection = select_hiqbind_subset(
            eligible,
            max_samples=args.max_samples,
            config=HiQBindSubsetConfig(
                strategy=args.strategy,
                random_seed=args.random_seed,
                target_quantile_bins=args.target_quantile_bins,
                ligand_size_quantile_bins=args.ligand_size_quantile_bins,
                max_per_pdb_id=args.max_per_pdb_id,
            ),
        )
        payload = {
            "report_schema": "mint-agent.hiqbind-subset-preview.v1",
            "status": "COMPLETE",
            "metadata": str(args.metadata.expanduser().resolve()),
            "selection": selection,
            "sample_ids": [record.sample_id for record in selected],
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if args.command == "summarize":
        print(
            f"rows={payload['row_count']} unique_structures={payload['unique_structure_count']} "
            f"output={args.output}"
        )
    else:
        print(
            f"eligible={payload['selection']['eligible_count']} "
            f"selected={payload['selection']['selected_count']} "
            f"selection_hash={payload['selection']['selection_hash']} "
            f"output={args.output}"
        )
    return 0


def summarize_hiqbind_metadata(metadata_path: Path) -> dict[str, Any]:
    rows = _read_metadata_rows(metadata_path)
    signs: Counter[str] = Counter()
    measurements: Counter[str] = Counter()
    sources: Counter[str] = Counter()
    years: Counter[int] = Counter()
    source_keys: Counter[str] = Counter()
    targets: list[float] = []
    exact_kd_ki_targets: list[float] = []
    invalid_affinity_rows: list[dict[str, Any]] = []
    for line_number, row in rows:
        sign = str(row["Binding Affinity Sign"]).strip()
        measurement = str(row["Binding Affinity Measurement"]).strip().lower()
        source = str(row["Binding Affinity Source"]).strip()
        year = _integer(row["Year"], "Year", line_number)
        source_key = _source_key(row, line_number=line_number)
        signs[sign] += 1
        measurements[measurement] += 1
        sources[source] += 1
        years[year] += 1
        source_keys[source_key] += 1
        try:
            target = _target_p_affinity(row, line_number=line_number)
        except ValueError as exc:
            invalid_affinity_rows.append(
                {
                    "line_number": line_number,
                    "source_key": source_key,
                    "reason": str(exc),
                }
            )
            continue
        targets.append(target)
        if sign == "=" and measurement in {"kd", "ki"}:
            exact_kd_ki_targets.append(target)
    duplicates = sorted(key for key, count in source_keys.items() if count > 1)
    return {
        "report_schema": "mint-agent.hiqbind-metadata-summary.v1",
        "source_doi": SOURCE_DOI,
        "source_license": SOURCE_LICENSE,
        "metadata": str(metadata_path.expanduser().resolve()),
        "metadata_sha256": file_sha256(metadata_path),
        "row_count": len(rows),
        "unique_structure_count": len(source_keys),
        "duplicate_structure_count": len(duplicates),
        "duplicate_structure_examples": duplicates[:20],
        "invalid_affinity_count": len(invalid_affinity_rows),
        "invalid_affinity_examples": invalid_affinity_rows[:20],
        "sign_counts": dict(sorted(signs.items())),
        "measurement_counts": dict(sorted(measurements.items())),
        "source_counts": dict(sorted(sources.items())),
        "year_counts": {str(key): value for key, value in sorted(years.items())},
        "target_p_affinity": _numeric_summary(targets),
        "exact_kd_ki": {
            "count": len(exact_kd_ki_targets),
            "target_p_affinity": _numeric_summary(exact_kd_ki_targets),
        },
        "status": "REVIEW" if duplicates or invalid_affinity_rows else "PASS",
    }


def load_hiqbind_records(
    metadata_path: Path,
    *,
    measurements: Iterable[str],
    affinity_signs: Iterable[str],
    max_samples: int | None = None,
    invalid_affinity_policy: str = "error",
) -> tuple[HiQBindRecord, ...]:
    normalized_measurements, normalized_signs = _normalize_selection_filters(
        measurements, affinity_signs
    )
    if max_samples is not None and max_samples < 1:
        raise ValueError("HiQBind max_samples must be positive")
    if invalid_affinity_policy not in {"error", "skip"}:
        raise ValueError("HiQBind invalid_affinity_policy must be error or skip")

    records: list[HiQBindRecord] = []
    seen: set[str] = set()
    for line_number, row in _read_metadata_rows(metadata_path):
        measurement = str(row["Binding Affinity Measurement"]).strip().lower()
        sign = str(row["Binding Affinity Sign"]).strip()
        if measurement not in normalized_measurements or sign not in normalized_signs:
            continue
        try:
            target_p_affinity = _target_p_affinity(
                row, line_number=line_number
            )
            affinity_value = _positive_float(
                row["Binding Affinity Value"],
                "Binding Affinity Value",
                line_number,
            )
            affinity_unit = _required(
                row, "Binding Affinity Unit", line_number
            )
        except ValueError:
            if invalid_affinity_policy == "skip":
                continue
            raise
        source_key = _source_key(row, line_number=line_number)
        if source_key in seen:
            raise ValueError(
                f"HiQBind selection gives multiple labels to structure {source_key!r}"
            )
        seen.add(source_key)
        heavy_atoms_text = str(row.get("Ligand NumHeavyAtoms") or "").strip()
        heavy_atoms = (
            _integer(heavy_atoms_text, "Ligand NumHeavyAtoms", line_number)
            if heavy_atoms_text
            else None
        )
        records.append(
            HiQBindRecord(
                source_key=source_key,
                pdb_id=_required(row, "PDBID", line_number).lower(),
                ligand_name=_required(row, "Ligand Name", line_number),
                ligand_chain=_required(row, "Ligand Chain", line_number),
                ligand_residue_number=_required(
                    row, "Ligand Residue Number", line_number
                ),
                measurement=measurement,
                sign=sign,
                affinity_value=affinity_value,
                affinity_unit=affinity_unit,
                target_p_affinity=target_p_affinity,
                year=_integer(row["Year"], "Year", line_number),
                affinity_source=str(row.get("Binding Affinity Source") or "").strip(),
                ligand_heavy_atoms=heavy_atoms,
            )
        )
    records.sort(key=lambda record: record.source_key)
    if max_samples is not None:
        records = records[:max_samples]
    if not records:
        raise ValueError("HiQBind filters selected no records")
    return tuple(records)


def summarize_hiqbind_selection(
    metadata_path: Path,
    *,
    measurements: Iterable[str],
    affinity_signs: Iterable[str],
) -> dict[str, Any]:
    normalized_measurements, normalized_signs = _normalize_selection_filters(
        measurements, affinity_signs
    )
    candidate_count = 0
    valid_targets: list[float] = []
    invalid_rows: list[dict[str, Any]] = []
    for line_number, row in _read_metadata_rows(metadata_path):
        measurement = str(row["Binding Affinity Measurement"]).strip().lower()
        sign = str(row["Binding Affinity Sign"]).strip()
        if measurement not in normalized_measurements or sign not in normalized_signs:
            continue
        candidate_count += 1
        try:
            valid_targets.append(_target_p_affinity(row, line_number=line_number))
        except ValueError as exc:
            invalid_rows.append(
                {
                    "line_number": line_number,
                    "source_key": _source_key(row, line_number=line_number),
                    "reason": str(exc),
                }
            )
    return {
        "report_schema": "mint-agent.hiqbind-selection-summary.v1",
        "measurements": list(normalized_measurements),
        "affinity_signs": list(normalized_signs),
        "candidate_count": candidate_count,
        "valid_count": len(valid_targets),
        "invalid_affinity_count": len(invalid_rows),
        "invalid_affinity_examples": invalid_rows[:20],
        "target_p_affinity": _numeric_summary(valid_targets),
        "status": "REVIEW" if invalid_rows else "PASS",
    }


def select_hiqbind_subset(
    records: Sequence[HiQBindRecord],
    *,
    max_samples: int | None,
    config: HiQBindSubsetConfig,
) -> tuple[tuple[HiQBindRecord, ...], dict[str, Any]]:
    eligible = tuple(sorted(records, key=lambda record: record.source_key))
    if not eligible:
        raise ValueError("HiQBind subset selection requires eligible records")
    if max_samples is not None and max_samples < 1:
        raise ValueError("HiQBind max_samples must be positive")

    candidate_pool = eligible
    if max_samples is not None and config.max_per_pdb_id is not None:
        ranked = sorted(
            eligible,
            key=lambda record: (
                stable_hash(
                    {
                        "random_seed": config.random_seed,
                        "pdb_id": record.pdb_id,
                        "source_key": record.source_key,
                    }
                ),
                record.source_key,
            ),
        )
        counts: Counter[str] = Counter()
        capped = []
        for record in ranked:
            if counts[record.pdb_id] >= config.max_per_pdb_id:
                continue
            capped.append(record)
            counts[record.pdb_id] += 1
        candidate_pool = tuple(sorted(capped, key=lambda record: record.source_key))

    requested = len(candidate_pool) if max_samples is None else max_samples
    if requested > len(candidate_pool):
        raise ValueError(
            "HiQBind subset policy leaves too few records: "
            f"requested {requested}, available {len(candidate_pool)}"
        )
    if requested == len(candidate_pool):
        selected = candidate_pool
    elif config.strategy == "source_order":
        selected = candidate_pool[:requested]
    else:
        size_values, _ = _ligand_size_values(candidate_pool)
        selection = select_probe_samples(
            sample_ids=[record.sample_id for record in candidate_pool],
            targets=[record.target_p_affinity for record in candidate_pool],
            structure_sizes=size_values,
            config=ProbeSamplingConfig(
                fraction=1.0,
                min_samples=requested,
                max_samples=requested,
                target_quantile_bins=config.target_quantile_bins,
                size_quantile_bins=config.ligand_size_quantile_bins,
                min_pair_support=0,
                augmentation_fraction_limit=0.0,
                random_seed=config.random_seed,
            ),
        )
        selected_ids = set(selection.sample_ids)
        selected = tuple(
            record for record in candidate_pool if record.sample_id in selected_ids
        )
    if len(selected) != requested:
        raise AssertionError("HiQBind subset selection returned the wrong sample count")

    eligible_sizes, eligible_imputed = _ligand_size_values(eligible)
    selected_sizes, selected_imputed = _ligand_size_values(selected)
    selection_report: dict[str, Any] = {
        "config": config.to_dict(),
        "eligible_count": len(eligible),
        "candidate_count_after_group_cap": len(candidate_pool),
        "selected_count": len(selected),
        "eligible_pdb_group_count": len({record.pdb_id for record in eligible}),
        "selected_pdb_group_count": len({record.pdb_id for record in selected}),
        "target_p_affinity": _distribution_comparison(
            [record.target_p_affinity for record in eligible],
            [record.target_p_affinity for record in selected],
        ),
        "ligand_heavy_atoms": {
            **_distribution_comparison(eligible_sizes, selected_sizes),
            "eligible_imputed_count": eligible_imputed,
            "selected_imputed_count": selected_imputed,
        },
        "year": _distribution_comparison(
            [float(record.year) for record in eligible],
            [float(record.year) for record in selected],
        ),
        "measurement": _categorical_comparison(
            [record.measurement for record in eligible],
            [record.measurement for record in selected],
        ),
        "affinity_source": _categorical_comparison(
            [record.affinity_source for record in eligible],
            [record.affinity_source for record in selected],
        ),
    }
    selection_report["selection_hash"] = stable_hash(
        {
            "config": selection_report["config"],
            "sample_ids": [record.sample_id for record in selected],
        }
    )
    return selected, selection_report


def prepare_hiqbind(
    *,
    source_root: Path,
    metadata_path: Path,
    output_root: Path,
    manifest_path: Path,
    audit_report_path: Path,
    measurements: Iterable[str],
    affinity_signs: Iterable[str],
    output_split: str | None = None,
    source_archive: Path | None = None,
    extract_root: Path | None = None,
    source_url: str = SOURCE_URL,
    source_archive_md5: str = SOURCE_MD5,
    metadata_url: str = METADATA_URL,
    metadata_md5: str = METADATA_MD5,
    max_samples: int | None = None,
    expected_sample_count: int | None = None,
    obabel: str = "obabel",
    pocket_cutoff_angstrom: float = 10.0,
    add_hydrogens: bool = True,
    invalid_affinity_policy: str = "error",
    subset_config: HiQBindSubsetConfig = HiQBindSubsetConfig(),
) -> dict[str, Any]:
    measurements = tuple(measurements)
    affinity_signs = tuple(affinity_signs)
    if output_split is not None and output_split not in SUPPORTED_OUTPUT_SPLITS:
        raise ValueError(f"unsupported HiQBind output_split: {output_split!r}")
    if not math.isfinite(pocket_cutoff_angstrom) or pocket_cutoff_angstrom <= 0.0:
        raise ValueError("pocket_cutoff_angstrom must be finite and positive")
    if expected_sample_count is not None and expected_sample_count < 1:
        raise ValueError("HiQBind expected_sample_count must be positive")

    source_root = source_root.expanduser().resolve()
    metadata_path = metadata_path.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    manifest_path = manifest_path.expanduser().resolve()
    audit_report_path = audit_report_path.expanduser().resolve()
    source_archive = source_archive.expanduser().resolve() if source_archive else None
    extract_root = (extract_root or source_root.parent).expanduser().resolve()
    metadata_downloaded = _ensure_download(
        metadata_path, url=metadata_url, expected_md5=metadata_md5
    )
    archive_downloaded = False
    archive_extracted = False
    if not source_root.is_dir():
        if source_archive is None:
            raise FileNotFoundError(
                "HiQBind source_root is missing and source_archive is not configured"
            )
        archive_downloaded = _ensure_download(
            source_archive, url=source_url, expected_md5=source_archive_md5
        )
        _extract_tar_archive(source_archive, extract_root)
        archive_extracted = True
    if not source_root.is_dir():
        raise FileNotFoundError(
            f"HiQBind archive did not create configured source_root: {source_root}"
        )
    if source_archive is not None and source_archive.is_file():
        observed_md5 = _file_digest(source_archive, "md5")
        if observed_md5 != source_archive_md5.lower():
            raise ValueError(
                "HiQBind archive checksum mismatch: "
                f"expected {source_archive_md5.lower()}, observed {observed_md5}"
            )

    eligible_records = load_hiqbind_records(
        metadata_path,
        measurements=measurements,
        affinity_signs=affinity_signs,
        invalid_affinity_policy=invalid_affinity_policy,
    )
    records, subset_selection = select_hiqbind_subset(
        eligible_records,
        max_samples=max_samples,
        config=subset_config,
    )
    selection_summary = summarize_hiqbind_selection(
        metadata_path,
        measurements=measurements,
        affinity_signs=affinity_signs,
    )
    missing_sources = _missing_structure_paths(source_root, records)
    if missing_sources and source_archive is not None:
        archive_downloaded = (
            _ensure_download(
                source_archive, url=source_url, expected_md5=source_archive_md5
            )
            or archive_downloaded
        )
        _extract_tar_archive(source_archive, extract_root)
        archive_extracted = True
        missing_sources = _missing_structure_paths(source_root, records)
    if missing_sources:
        raise FileNotFoundError(
            "Missing HiQBind source structures after preparation: "
            + "; ".join(str(path) for path in missing_sources[:5])
        )
    if expected_sample_count is not None and len(records) != expected_sample_count:
        raise ValueError(
            "HiQBind sample-count mismatch: "
            f"expected {expected_sample_count}, observed {len(records)}"
        )

    rows: list[dict[str, str]] = []
    samples: list[dict[str, Any]] = []
    for record in records:
        source_protein, source_ligand = _structure_paths(source_root, record)
        sample_root = output_root / record.sample_id
        protein_path = sample_root / f"{record.sample_id}_pocket.pdb"
        ligand_path = sample_root / f"{record.sample_id}_ligand.mol2"
        conversion = convert_sdf_to_mol2(
            source_ligand,
            ligand_path,
            obabel=obabel,
            add_hydrogens=add_hydrogens,
        )
        pocket = write_ligand_centered_pocket(
            source_protein,
            source_ligand,
            protein_path,
            cutoff_angstrom=pocket_cutoff_angstrom,
        )
        rows.append(
            {
                "sample_id": record.sample_id,
                "target": f"{record.target_p_affinity:.15g}",
                "split": output_split or "",
                "protein_path": str(protein_path),
                "ligand_path": str(ligand_path),
                "pdb_id": record.pdb_id,
                "hiqbind_id": record.source_key,
                "measurement": record.measurement,
                "affinity_source": record.affinity_source,
                "year": str(record.year),
            }
        )
        samples.append(
            {
                "sample_id": record.sample_id,
                "hiqbind_id": record.source_key,
                "pdb_id": record.pdb_id,
                "target_p_affinity": record.target_p_affinity,
                "measurement": record.measurement,
                "sign": record.sign,
                "year": record.year,
                "affinity_source": record.affinity_source,
                "ligand_heavy_atoms": record.ligand_heavy_atoms,
                "source_protein": str(source_protein),
                "source_ligand": str(source_ligand),
                "source_protein_sha256": file_sha256(source_protein),
                "source_ligand_sha256": file_sha256(source_ligand),
                "prepared_protein": str(protein_path),
                "prepared_ligand": str(ligand_path),
                "prepared_protein_sha256": file_sha256(protein_path),
                "prepared_ligand_sha256": file_sha256(ligand_path),
                "pocket": pocket,
                "ligand_conversion": conversion,
            }
        )
    _write_manifest(manifest_path, rows)
    payload = {
        "report_schema": "mint-agent.hiqbind-preparation.v1",
        "dataset_id": "hiqbind",
        "evidence_scope": "design",
        "status": "PASS",
        "run_kind": "engineering_subset" if max_samples is not None else "full_dataset",
        "source_doi": SOURCE_DOI,
        "source_url": source_url,
        "source_license": SOURCE_LICENSE,
        "source_root": str(source_root),
        "source_archive": str(source_archive) if source_archive else None,
        "source_archive_md5": (
            _file_digest(source_archive, "md5")
            if source_archive is not None and source_archive.is_file()
            else None
        ),
        "source_archive_downloaded": archive_downloaded,
        "source_archive_extracted": archive_extracted,
        "metadata": str(metadata_path),
        "metadata_md5": _file_digest(metadata_path, "md5"),
        "metadata_downloaded": metadata_downloaded,
        "metadata_summary": summarize_hiqbind_metadata(metadata_path),
        "selection_summary": selection_summary,
        "measurements": sorted({record.measurement for record in records}),
        "affinity_signs": sorted({record.sign for record in records}),
        "invalid_affinity_policy": invalid_affinity_policy,
        "output_split": output_split,
        "max_samples": max_samples,
        "max_samples_applied": (
            max_samples is not None and selection_summary["valid_count"] > len(samples)
        ),
        "subset_selection": subset_selection,
        "sample_count": len(samples),
        "sample_order_hash": stable_hash([row["sample_id"] for row in rows]),
        "manifest": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "preprocessing": {
            "protein": "complete residues with any ATOM coordinate within the configured ligand cutoff",
            "pocket_cutoff_angstrom": pocket_cutoff_angstrom,
            "ligand": "Open Babel refined-SDF-to-MOL2 conversion",
            "add_hydrogens": add_hydrogens,
            "coordinate_transform": "none",
        },
        "target": {
            "source_field": "Log Binding Affinity",
            "transform": "negative of source log10 molar affinity",
            "interpretation": "pAffinity = -log10(molar affinity)",
            "used_for_representation_design": False,
        },
        "samples": samples,
    }
    audit_report_path.parent.mkdir(parents=True, exist_ok=True)
    audit_report_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload


def _read_metadata_rows(metadata_path: Path) -> list[tuple[int, dict[str, str]]]:
    path = metadata_path.expanduser().resolve(strict=True)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {
            "PDBID",
            "Year",
            "Ligand Name",
            "Ligand Chain",
            "Ligand Residue Number",
            "Binding Affinity Measurement",
            "Binding Affinity Sign",
            "Binding Affinity Value",
            "Binding Affinity Unit",
            "Log Binding Affinity",
        }
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"HiQBind metadata is missing columns: {sorted(missing)}")
        rows = [(line_number, dict(row)) for line_number, row in enumerate(reader, start=2)]
    if not rows:
        raise ValueError("HiQBind metadata contains no rows")
    return rows


def _normalize_selection_filters(
    measurements: Iterable[str], affinity_signs: Iterable[str]
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    normalized_measurements = tuple(
        dict.fromkeys(str(value).strip().lower() for value in measurements)
    )
    normalized_signs = tuple(
        dict.fromkeys(str(value).strip() for value in affinity_signs)
    )
    if not normalized_measurements or set(normalized_measurements) - SUPPORTED_MEASUREMENTS:
        raise ValueError(
            f"HiQBind measurements must be selected from {sorted(SUPPORTED_MEASUREMENTS)}"
        )
    if not normalized_signs or set(normalized_signs) - SUPPORTED_SIGNS:
        raise ValueError(
            f"HiQBind affinity signs must be selected from {sorted(SUPPORTED_SIGNS)}"
        )
    return normalized_measurements, normalized_signs


def _source_key(row: dict[str, str], *, line_number: int) -> str:
    return "_".join(
        _required(row, key, line_number)
        for key in ("PDBID", "Ligand Name", "Ligand Chain", "Ligand Residue Number")
    )


def _structure_paths(
    source_root: Path, record: HiQBindRecord
) -> tuple[Path, Path]:
    source_dir = source_root / record.pdb_id / record.source_key
    return (
        source_dir / f"{record.source_key}_protein_refined.pdb",
        source_dir / f"{record.source_key}_ligand_refined.sdf",
    )


def _missing_structure_paths(
    source_root: Path, records: Iterable[HiQBindRecord]
) -> list[Path]:
    missing: list[Path] = []
    for record in records:
        missing.extend(
            path for path in _structure_paths(source_root, record) if not path.is_file()
        )
    return missing


def _target_p_affinity(row: dict[str, str], *, line_number: int) -> float:
    value = _positive_float(
        row["Binding Affinity Value"], "Binding Affinity Value", line_number
    )
    unit = _required(row, "Binding Affinity Unit", line_number)
    normalized_unit = unit.strip().lower().replace("μ", "u").replace("µ", "u")
    scale = UNIT_TO_MOLAR.get(normalized_unit)
    if scale is None:
        raise ValueError(f"Unsupported HiQBind affinity unit on line {line_number}: {unit!r}")
    expected_log_molar = math.log10(value * scale)
    source_log_molar = _finite_float(
        row["Log Binding Affinity"], "Log Binding Affinity", line_number
    )
    if not math.isclose(source_log_molar, expected_log_molar, abs_tol=1e-6):
        raise ValueError(
            f"HiQBind log affinity disagrees with value/unit on line {line_number}"
        )
    return -source_log_molar


def _required(row: dict[str, str], key: str, line_number: int) -> str:
    value = str(row.get(key) or "").strip()
    if not value:
        raise ValueError(f"Missing HiQBind {key!r} on line {line_number}")
    return value


def _finite_float(value: object, name: str, line_number: int) -> float:
    try:
        parsed = float(str(value))
    except ValueError as exc:
        raise ValueError(f"Invalid HiQBind {name} on line {line_number}: {value!r}") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"Non-finite HiQBind {name} on line {line_number}")
    return parsed


def _positive_float(value: object, name: str, line_number: int) -> float:
    parsed = _finite_float(value, name, line_number)
    if parsed <= 0.0:
        raise ValueError(f"HiQBind {name} must be positive on line {line_number}")
    return parsed


def _integer(value: object, name: str, line_number: int) -> int:
    parsed = _finite_float(value, name, line_number)
    if not parsed.is_integer():
        raise ValueError(f"HiQBind {name} must be an integer on line {line_number}")
    return int(parsed)


def _numeric_summary(values: list[float]) -> dict[str, Any] | None:
    if not values:
        return None
    array = np.asarray(values, dtype=float)
    quantiles = np.quantile(array, (0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0))
    return {
        "count": int(array.size),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "quantiles": {
            name: float(value)
            for name, value in zip(
                ("q00", "q05", "q25", "q50", "q75", "q95", "q100"),
                quantiles,
            )
        },
    }


def _ligand_size_values(
    records: Sequence[HiQBindRecord],
) -> tuple[list[float], int]:
    observed = [
        float(record.ligand_heavy_atoms)
        for record in records
        if record.ligand_heavy_atoms is not None
    ]
    if not observed:
        raise ValueError("HiQBind subset selection has no ligand heavy-atom values")
    median = float(np.median(np.asarray(observed, dtype=float)))
    values = [
        float(record.ligand_heavy_atoms)
        if record.ligand_heavy_atoms is not None
        else median
        for record in records
    ]
    return values, len(records) - len(observed)


def _distribution_comparison(
    population_values: Sequence[float], selected_values: Sequence[float]
) -> dict[str, Any]:
    population = np.asarray(population_values, dtype=float)
    selected = np.asarray(selected_values, dtype=float)
    if population.ndim != 1 or selected.ndim != 1:
        raise ValueError("HiQBind distribution inputs must be one-dimensional")
    if population.size == 0 or selected.size == 0:
        raise ValueError("HiQBind distribution inputs must be nonempty")
    if not np.all(np.isfinite(population)) or not np.all(np.isfinite(selected)):
        raise ValueError("HiQBind distribution inputs must be finite")
    values = np.unique(np.concatenate((population, selected)))
    population_cdf = (
        np.searchsorted(np.sort(population), values, side="right") / population.size
    )
    selected_cdf = (
        np.searchsorted(np.sort(selected), values, side="right") / selected.size
    )
    population_std = float(np.std(population))
    mean_difference = float(np.mean(selected) - np.mean(population))
    return {
        "eligible": _numeric_summary(population.tolist()),
        "selected": _numeric_summary(selected.tolist()),
        "ks_distance": float(np.max(np.abs(population_cdf - selected_cdf))),
        "mean_difference": mean_difference,
        "standardized_mean_difference": (
            mean_difference / population_std if population_std > 0.0 else 0.0
        ),
    }


def _categorical_comparison(
    population_values: Sequence[str], selected_values: Sequence[str]
) -> dict[str, Any]:
    population = Counter(str(value) for value in population_values)
    selected = Counter(str(value) for value in selected_values)
    categories = tuple(sorted(set(population) | set(selected)))
    rows = []
    for category in categories:
        population_share = population[category] / sum(population.values())
        selected_share = selected[category] / sum(selected.values())
        rows.append(
            {
                "category": category,
                "eligible_count": population[category],
                "selected_count": selected[category],
                "eligible_share": population_share,
                "selected_share": selected_share,
                "share_difference": selected_share - population_share,
            }
        )
    return {
        "max_abs_share_difference": max(
            (abs(row["share_difference"]) for row in rows), default=0.0
        ),
        "categories": rows,
    }


def _ensure_download(path: Path, *, url: str, expected_md5: str) -> bool:
    if path.is_file():
        observed = _file_digest(path, "md5")
        if observed != expected_md5.lower():
            raise ValueError(
                f"HiQBind checksum mismatch for {path}: expected {expected_md5}, observed {observed}"
            )
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    temporary.unlink(missing_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "mint-agent/1"})
    try:
        with urllib.request.urlopen(request) as response, temporary.open("wb") as handle:
            shutil.copyfileobj(response, handle, length=1024 * 1024)
        observed = _file_digest(temporary, "md5")
        if observed != expected_md5.lower():
            raise ValueError(
                f"HiQBind download checksum mismatch: expected {expected_md5}, observed {observed}"
            )
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return True


def _extract_tar_archive(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with tarfile.open(archive, mode="r:*") as handle:
        extraction_options = (
            {"filter": "data"} if sys.version_info >= (3, 12) else {}
        )
        for member in handle:
            target = (root / member.name).resolve()
            if root != target and root not in target.parents:
                raise ValueError(f"archive member escapes extraction root: {member.name}")
            if member.issym() or member.islnk():
                raise ValueError(f"archive links are not allowed: {member.name}")
            handle.extract(member, root, **extraction_options)


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    fieldnames = [
        "sample_id",
        "target",
        "split",
        "protein_path",
        "ligand_path",
        "pdb_id",
        "hiqbind_id",
        "measurement",
        "affinity_source",
        "year",
    ]
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _file_digest(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().lower()


if __name__ == "__main__":
    raise SystemExit(main())
