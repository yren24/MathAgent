from __future__ import annotations

import hashlib

import pytest

from mint_scout.data.casf_index import CasfRecord
from mint_scout.data.structure_staging import stage_protein_ligand_input


def _record(tmp_path, sample_id="sample-a"):
    protein = tmp_path / "arbitrary_receptor_name.pdb"
    ligand = tmp_path / "arbitrary_compound_name.mol2"
    protein.write_bytes(b"protein-v1")
    ligand.write_bytes(b"ligand-v1")
    return CasfRecord(sample_id, 7.0, "train", protein, ligand)


def test_structure_staging_hashes_and_links_arbitrary_paths(tmp_path):
    record = _record(tmp_path)
    staged = stage_protein_ligand_input(
        record,
        staging_root=tmp_path / "staging",
    )

    assert staged.protein_staged_path.name == "sample-a_pocket.pdb"
    assert staged.ligand_staged_path.name == "sample-a_ligand.mol2"
    assert staged.protein_staged_path.resolve() == record.protein_path.resolve()
    assert staged.ligand_staged_path.resolve() == record.ligand_path.resolve()
    assert staged.protein_sha256 == hashlib.sha256(b"protein-v1").hexdigest()
    assert staged.ligand_sha256 == hashlib.sha256(b"ligand-v1").hexdigest()

    repeated = stage_protein_ligand_input(
        record,
        staging_root=tmp_path / "staging",
    )
    assert repeated == staged


def test_structure_hash_changes_when_source_content_changes(tmp_path):
    record = _record(tmp_path)
    first = stage_protein_ligand_input(
        record,
        staging_root=tmp_path / "staging",
    )
    record.ligand_path.write_bytes(b"ligand-v2")
    second = stage_protein_ligand_input(
        record,
        staging_root=tmp_path / "staging",
    )

    assert first.ligand_sha256 != second.ligand_sha256
    assert first.combined_sha256 != second.combined_sha256


def test_structure_staging_refuses_conflicting_existing_path(tmp_path):
    record = _record(tmp_path)
    destination = tmp_path / "staging" / record.pdb_id
    destination.mkdir(parents=True)
    (destination / f"{record.pdb_id}_pocket.pdb").write_text(
        "do not replace",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Refusing to replace"):
        stage_protein_ligand_input(record, staging_root=tmp_path / "staging")


def test_structure_staging_rejects_unsafe_sample_id(tmp_path):
    record = _record(tmp_path, sample_id="../escape")
    with pytest.raises(ValueError, match="path-safe"):
        stage_protein_ligand_input(record, staging_root=tmp_path / "staging")
