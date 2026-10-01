from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from mint_scout.data.casf_index import CasfRecord


@dataclass(frozen=True)
class StagedProteinLigandInput:
    sample_id: str
    protein_source: Path
    ligand_source: Path
    protein_sha256: str
    ligand_sha256: str
    combined_sha256: str
    protein_staged_path: Path
    ligand_staged_path: Path

    def to_manifest_dict(self) -> dict[str, object]:
        return {
            "combined_sha256": self.combined_sha256,
            "protein": {
                "source_path": str(self.protein_source),
                "sha256": self.protein_sha256,
                "staged_path": str(self.protein_staged_path),
            },
            "ligand": {
                "source_path": str(self.ligand_source),
                "sha256": self.ligand_sha256,
                "staged_path": str(self.ligand_staged_path),
            },
        }


def stage_protein_ligand_input(
    record: CasfRecord,
    *,
    staging_root: str | Path,
    create_links: bool = True,
) -> StagedProteinLigandInput:
    _assert_safe_sample_id(record.pdb_id)
    if record.protein_path is None or record.ligand_path is None:
        raise ValueError(
            f"Sample {record.pdb_id!r} requires explicit protein and ligand paths"
        )
    protein_source = record.protein_path.expanduser().resolve(strict=True)
    ligand_source = record.ligand_path.expanduser().resolve(strict=True)
    protein_sha256 = file_sha256(protein_source)
    ligand_sha256 = file_sha256(ligand_source)
    combined_sha256 = hashlib.sha256(
        json.dumps(
            {
                "ligand_sha256": ligand_sha256,
                "protein_sha256": protein_sha256,
                "sample_id": record.pdb_id,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    sample_root = Path(staging_root).expanduser().resolve() / record.pdb_id
    protein_staged_path = sample_root / f"{record.pdb_id}_pocket.pdb"
    ligand_staged_path = sample_root / f"{record.pdb_id}_ligand.mol2"
    if create_links:
        sample_root.mkdir(parents=True, exist_ok=True)
        _ensure_symlink(protein_staged_path, protein_source)
        _ensure_symlink(ligand_staged_path, ligand_source)
    return StagedProteinLigandInput(
        sample_id=record.pdb_id,
        protein_source=protein_source,
        ligand_source=ligand_source,
        protein_sha256=protein_sha256,
        ligand_sha256=ligand_sha256,
        combined_sha256=combined_sha256,
        protein_staged_path=protein_staged_path,
        ligand_staged_path=ligand_staged_path,
    )


def file_sha256(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _ensure_symlink(destination: Path, source: Path) -> None:
    try:
        destination.symlink_to(source)
    except FileExistsError:
        if not destination.is_symlink():
            raise ValueError(
                f"Refusing to replace non-symlink staging path: {destination}"
            )
        try:
            existing_source = destination.resolve(strict=True)
        except FileNotFoundError as exc:
            raise ValueError(f"Staging symlink is broken: {destination}") from exc
        if existing_source != source:
            raise ValueError(
                f"Staging path {destination} already points to {existing_source}, expected {source}"
            )


def _assert_safe_sample_id(sample_id: str) -> None:
    if (
        not sample_id
        or sample_id in {".", ".."}
        or Path(sample_id).name != sample_id
        or any(separator in sample_id for separator in ("/", "\\"))
    ):
        raise ValueError(f"sample_id must be a path-safe component: {sample_id!r}")
