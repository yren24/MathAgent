from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.optimize import linear_sum_assignment

from mint_scout.data.geometry import AtomCloud, load_atom_cloud
from mint_scout.data.structure_staging import file_sha256
from mint_scout.invariants.manifest import stable_hash


SOURCE_URL = "https://github.com/THGLab/LP-PDBBind/tree/master/dataset"


@dataclass(frozen=True)
class BDB2020PlusRecord:
    pdb_id: str
    target_pka: float
    source_protein: Path
    source_ligand: Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Prepare the public BDB2020+ external binding-affinity benchmark."
    )
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audit-report", type=Path, required=True)
    parser.add_argument("--source-archive", type=Path, default=None)
    parser.add_argument("--obabel", default="obabel")
    parser.add_argument("--pocket-cutoff-angstrom", type=float, default=10.0)
    parser.add_argument("--without-hydrogens", action="store_true")
    args = parser.parse_args(argv)

    report = prepare_bdb2020plus(
        source_root=args.source_root,
        output_root=args.output_root,
        manifest_path=args.manifest,
        audit_report_path=args.audit_report,
        source_archive=args.source_archive,
        obabel=args.obabel,
        pocket_cutoff_angstrom=args.pocket_cutoff_angstrom,
        add_hydrogens=not args.without_hydrogens,
    )
    print(
        f"dataset=bdb2020plus samples={report['sample_count']} "
        f"status={report['status']} manifest={args.manifest}"
    )
    return 0


def prepare_bdb2020plus(
    *,
    source_root: Path,
    output_root: Path,
    manifest_path: Path,
    audit_report_path: Path,
    source_archive: Path | None = None,
    obabel: str = "obabel",
    pocket_cutoff_angstrom: float = 10.0,
    add_hydrogens: bool = True,
    output_split: str | None = "test",
) -> dict[str, object]:
    if not math.isfinite(pocket_cutoff_angstrom) or pocket_cutoff_angstrom <= 0.0:
        raise ValueError("pocket_cutoff_angstrom must be finite and positive")
    source_root = source_root.expanduser().resolve(strict=True)
    output_root = output_root.expanduser().resolve()
    metadata_path = source_root / "BDB2020+.csv"
    records = load_bdb2020plus_records(metadata_path, source_root / "dataset")
    obabel_version = _obabel_version(obabel)

    samples: list[dict[str, object]] = []
    manifest_rows: list[dict[str, object]] = []
    for record in records:
        sample_root = output_root / record.pdb_id
        sample_root.mkdir(parents=True, exist_ok=True)
        ligand_path = sample_root / f"{record.pdb_id}_ligand.mol2"
        pocket_path = sample_root / f"{record.pdb_id}_pocket.pdb"
        conversion = convert_sdf_to_mol2(
            record.source_ligand,
            ligand_path,
            obabel=obabel,
            add_hydrogens=add_hydrogens,
        )
        pocket = write_ligand_centered_pocket(
            record.source_protein,
            record.source_ligand,
            pocket_path,
            cutoff_angstrom=pocket_cutoff_angstrom,
        )
        manifest_rows.append(
            {
                "sample_id": record.pdb_id,
                "target": f"{record.target_pka:.15g}",
                "split": output_split or "",
                "protein_path": str(pocket_path),
                "ligand_path": str(ligand_path),
                "pdb_id": record.pdb_id,
            }
        )
        samples.append(
            {
                "sample_id": record.pdb_id,
                "target_pka": record.target_pka,
                "source_protein": str(record.source_protein),
                "source_ligand": str(record.source_ligand),
                "prepared_protein": str(pocket_path),
                "prepared_ligand": str(ligand_path),
                "source_protein_sha256": file_sha256(record.source_protein),
                "source_ligand_sha256": file_sha256(record.source_ligand),
                "prepared_protein_sha256": file_sha256(pocket_path),
                "prepared_ligand_sha256": file_sha256(ligand_path),
                "pocket": pocket,
                "ligand_conversion": conversion,
            }
        )

    _write_manifest(manifest_path, manifest_rows)
    payload: dict[str, object] = {
        "report_schema": "mint-agent.bdb2020plus-preparation.v1",
        "dataset_id": "bdb2020plus",
        "evidence_scope": "external_test",
        "status": "PASS",
        "source_url": SOURCE_URL,
        "source_root": str(source_root),
        "source_metadata": str(metadata_path),
        "source_metadata_sha256": file_sha256(metadata_path),
        "source_archive": str(source_archive.resolve()) if source_archive else None,
        "source_archive_sha256": (
            file_sha256(source_archive) if source_archive is not None else None
        ),
        "sample_count": len(samples),
        "sample_order_hash": stable_hash([row["sample_id"] for row in manifest_rows]),
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": file_sha256(manifest_path),
        "preprocessing": {
            "protein": "complete residues with any ATOM coordinate within cutoff of a ligand atom",
            "pocket_cutoff_angstrom": pocket_cutoff_angstrom,
            "ligand": "Open Babel SDF-to-MOL2 conversion",
            "add_hydrogens": add_hydrogens,
            "obabel_executable": obabel,
            "obabel_version": obabel_version,
        },
        "target": {
            "column": "pKa",
            "interpretation": "source-provided negative log10 molar affinity",
            "used_for_representation_design": False,
        },
        "samples": samples,
    }
    audit_report_path.parent.mkdir(parents=True, exist_ok=True)
    audit_report_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload


def load_bdb2020plus_records(
    metadata_path: Path, structures_root: Path
) -> tuple[BDB2020PlusRecord, ...]:
    if not metadata_path.is_file():
        raise FileNotFoundError(f"BDB2020+ metadata does not exist: {metadata_path}")
    rows: list[BDB2020PlusRecord] = []
    seen: set[str] = set()
    with metadata_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"pdbid", "accurate", "pKa"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"BDB2020+ metadata is missing columns: {sorted(missing)}")
        for line_number, row in enumerate(reader, start=2):
            if not _parse_bool(row.get("accurate")):
                continue
            pdb_id = str(row.get("pdbid") or "").strip().upper()
            if not pdb_id:
                raise ValueError(f"Missing pdbid on metadata line {line_number}")
            if pdb_id.casefold() in seen:
                raise ValueError(f"Duplicate BDB2020+ pdbid: {pdb_id}")
            seen.add(pdb_id.casefold())
            try:
                target_pka = float(str(row.get("pKa") or ""))
            except ValueError as exc:
                raise ValueError(f"Invalid pKa for {pdb_id}: {row.get('pKa')!r}") from exc
            if not math.isfinite(target_pka):
                raise ValueError(f"Non-finite pKa for {pdb_id}")
            sample_root = structures_root / pdb_id
            protein = sample_root / "protein.pdb"
            ligand = sample_root / "ligand.sdf"
            for path in (protein, ligand):
                if not path.is_file():
                    raise FileNotFoundError(f"Missing BDB2020+ structure for {pdb_id}: {path}")
            rows.append(BDB2020PlusRecord(pdb_id, target_pka, protein, ligand))
    if not rows:
        raise ValueError("BDB2020+ metadata contains no accurate labeled records")
    return tuple(rows)


def convert_sdf_to_mol2(
    source: Path,
    output: Path,
    *,
    obabel: str,
    add_hydrogens: bool,
    coordinate_tolerance: float = 1e-3,
) -> dict[str, object]:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}.tmp{output.suffix}")
    command = [obabel, "-isdf", str(source), "-omol2", "-O", str(temporary)]
    if add_hydrogens:
        command.append("-h")
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
        legacy_section_reordered = normalize_mol2_for_legacy(temporary)
        validation = validate_ligand_conversion(
            source, temporary, coordinate_tolerance=coordinate_tolerance
        )
        os.replace(temporary, output)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return {
        **validation,
        "command": command,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
        "add_hydrogens": add_hydrogens,
        "legacy_section_reordered": legacy_section_reordered,
    }


def normalize_mol2_for_legacy(path: Path) -> bool:
    """Place BOND immediately after ATOM for the audited fixed-column reader."""
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    section_starts = [
        index for index, line in enumerate(lines) if line.startswith("@<TRIPOS>")
    ]
    section_names = {lines[index].strip(): index for index in section_starts}
    atom_start = section_names.get("@<TRIPOS>ATOM")
    bond_start = section_names.get("@<TRIPOS>BOND")
    if atom_start is None or bond_start is None:
        raise ValueError(f"MOL2 must contain ATOM and BOND sections: {path}")
    atom_end = next(
        (index for index in section_starts if index > atom_start), len(lines)
    )
    if atom_end == bond_start:
        return False
    if bond_start < atom_end:
        raise ValueError(f"MOL2 BOND section precedes ATOM section: {path}")
    bond_end = next(
        (index for index in section_starts if index > bond_start), len(lines)
    )
    reordered = (
        lines[:atom_end]
        + lines[bond_start:bond_end]
        + lines[atom_end:bond_start]
        + lines[bond_end:]
    )
    path.write_text("".join(reordered), encoding="utf-8")
    return True


def validate_ligand_conversion(
    source: Path,
    converted: Path,
    *,
    coordinate_tolerance: float = 1e-3,
) -> dict[str, object]:
    source_cloud = load_atom_cloud(source)
    converted_cloud = load_atom_cloud(converted)
    source_heavy = _without_hydrogens(source_cloud)
    converted_heavy = _without_hydrogens(converted_cloud)
    if Counter(source_heavy.elements) != Counter(converted_heavy.elements):
        raise ValueError(
            f"Heavy-atom composition changed during ligand conversion: {source} -> {converted}"
        )
    maximum_displacement = _maximum_element_matched_displacement(
        source_heavy, converted_heavy
    )
    if maximum_displacement > coordinate_tolerance:
        raise ValueError(
            f"Heavy-atom coordinates moved by {maximum_displacement:.6g} A during conversion"
        )
    return {
        "source_atom_count": len(source_cloud.elements),
        "converted_atom_count": len(converted_cloud.elements),
        "converted_hydrogen_count": converted_cloud.elements.count("H"),
        "heavy_atom_count": len(source_heavy.elements),
        "maximum_heavy_atom_displacement_angstrom": maximum_displacement,
        "coordinate_tolerance_angstrom": coordinate_tolerance,
    }


def write_ligand_centered_pocket(
    protein_path: Path,
    ligand_path: Path,
    output_path: Path,
    *,
    cutoff_angstrom: float,
    chunk_size: int = 4096,
) -> dict[str, object]:
    if cutoff_angstrom <= 0.0 or not math.isfinite(cutoff_angstrom):
        raise ValueError("cutoff_angstrom must be finite and positive")
    ligand = load_atom_cloud(ligand_path)
    atom_rows = tuple(_iter_pdb_atom_lines(protein_path))
    if not atom_rows:
        raise ValueError(f"No protein ATOM records parsed from {protein_path}")
    coordinates = np.asarray([row[1] for row in atom_rows], dtype=float)
    selected_atom = np.zeros(len(atom_rows), dtype=bool)
    cutoff_squared = cutoff_angstrom**2
    for start in range(0, len(atom_rows), chunk_size):
        block = coordinates[start : start + chunk_size]
        squared = np.sum(
            (block[:, None, :] - ligand.coordinates[None, :, :]) ** 2, axis=2
        )
        selected_atom[start : start + len(block)] = np.any(
            squared <= cutoff_squared, axis=1
        )
    selected_residues = {
        atom_rows[index][2] for index in np.flatnonzero(selected_atom)
    }
    if not selected_residues:
        raise ValueError(
            f"No protein residues are within {cutoff_angstrom:g} A of {ligand_path}"
        )
    selected_lines = [row[0] for row in atom_rows if row[2] in selected_residues]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(selected_lines) + "END\n", encoding="utf-8")
    return {
        "source_atom_count": len(atom_rows),
        "pocket_atom_count": len(selected_lines),
        "pocket_residue_count": len(selected_residues),
        "cutoff_angstrom": cutoff_angstrom,
    }


def _iter_pdb_atom_lines(
    path: Path,
) -> Iterable[tuple[str, tuple[float, float, float], tuple[str, str, str]]]:
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.startswith("ATOM"):
                continue
            try:
                coordinate = (
                    float(line[30:38]),
                    float(line[38:46]),
                    float(line[46:54]),
                )
            except (ValueError, IndexError) as exc:
                raise ValueError(
                    f"Malformed ATOM coordinate in {path} on line {line_number}"
                ) from exc
            residue = (line[21:22], line[22:26], line[26:27])
            yield line if line.endswith("\n") else line + "\n", coordinate, residue


def _without_hydrogens(cloud: AtomCloud) -> AtomCloud:
    indices = np.asarray(
        [index for index, element in enumerate(cloud.elements) if element != "H"],
        dtype=int,
    )
    return AtomCloud(
        elements=tuple(cloud.elements[index] for index in indices),
        coordinates=cloud.coordinates[indices],
    )


def _maximum_element_matched_displacement(left: AtomCloud, right: AtomCloud) -> float:
    maximum = 0.0
    for element in sorted(set(left.elements)):
        left_coordinates = left.coordinates_for(element)
        right_coordinates = right.coordinates_for(element)
        distances = np.sqrt(
            np.sum(
                (left_coordinates[:, None, :] - right_coordinates[None, :, :]) ** 2,
                axis=2,
            )
        )
        row_indices, column_indices = linear_sum_assignment(distances)
        maximum = max(maximum, float(np.max(distances[row_indices, column_indices])))
    return maximum


def _write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "sample_id",
                "target",
                "split",
                "protein_path",
                "ligand_path",
                "pdb_id",
            ),
        )
        writer.writeheader()
        writer.writerows(rows)


def _parse_bool(value: object) -> bool:
    return str(value or "").strip().casefold() in {"1", "true", "yes", "y"}


def _obabel_version(executable: str) -> str:
    completed = subprocess.run(
        [executable, "-V"], check=True, capture_output=True, text=True
    )
    return (completed.stdout or completed.stderr).strip()


if __name__ == "__main__":
    raise SystemExit(main())
