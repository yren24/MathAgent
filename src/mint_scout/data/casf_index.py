from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path


LEGACY_BAD_PDB_IDS = ("3tei", "4as6")


@dataclass(frozen=True)
class CasfRecord:
    pdb_id: str
    label: float | None
    split: str | None
    protein_path: Path | None = None
    ligand_path: Path | None = None
    group_id: str | None = None


def load_casf_records(
    *,
    index_root: Path,
    structures_root: Path | None = None,
    year: int = 2016,
    protein_template: str = "{pdb}_pocket.pdb",
    ligand_template: str = "{pdb}_ligand.mol2",
    exclude_pdb_ids: tuple[str, ...] = LEGACY_BAD_PDB_IDS,
) -> tuple[CasfRecord, ...]:
    labels = load_refined_index(index_root / f"{year}_INDEX_refined.data")
    split_by_pdb = load_legacy_split(index_root=index_root, year=year, exclude_pdb_ids=exclude_pdb_ids)
    excluded = set(exclude_pdb_ids)
    records = []
    for pdb_id in sorted(labels):
        if pdb_id in excluded:
            continue
        protein_path = None
        ligand_path = None
        if structures_root is not None:
            protein_path = structures_root / pdb_id / protein_template.format(pdb=pdb_id)
            ligand_path = structures_root / pdb_id / ligand_template.format(pdb=pdb_id)
        records.append(
            CasfRecord(
                pdb_id=pdb_id,
                label=labels[pdb_id],
                split=split_by_pdb.get(pdb_id),
                protein_path=protein_path,
                ligand_path=ligand_path,
            )
        )
    return tuple(records)


def load_refined_index(path: Path) -> dict[str, float]:
    labels: dict[str, float] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if len(parts) < 4:
            continue
        pdb_id = parts[0].lower()
        labels[pdb_id] = float(parts[3])
    if not labels:
        raise ValueError(f"No CASF records parsed from {path}")
    return labels


def load_legacy_split(
    *,
    index_root: Path,
    year: int,
    exclude_pdb_ids: tuple[str, ...] = LEGACY_BAD_PDB_IDS,
) -> dict[str, str]:
    excluded = set(exclude_pdb_ids)
    split_by_pdb: dict[str, str] = {}
    for split in ("train", "test"):
        path = index_root / f"{split}_data_{year}.txt"
        pdb_ids = _load_pdb_list(path)
        for pdb_id in pdb_ids:
            if pdb_id in excluded:
                continue
            if pdb_id in split_by_pdb:
                raise ValueError(f"{pdb_id} appears in multiple split files")
            split_by_pdb[pdb_id] = split
    return split_by_pdb


def _load_pdb_list(path: Path) -> tuple[str, ...]:
    parsed = ast.literal_eval(path.read_text(encoding="utf-8"))
    if not isinstance(parsed, list):
        raise ValueError(f"Expected a Python list in {path}")
    return tuple(str(item).lower() for item in parsed)


def missing_structure_paths(records: tuple[CasfRecord, ...]) -> tuple[Path, ...]:
    missing = []
    for record in records:
        for path in (record.protein_path, record.ligand_path):
            if path is not None and not path.exists():
                missing.append(path)
    return tuple(missing)
