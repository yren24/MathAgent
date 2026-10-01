from __future__ import annotations

from pathlib import Path

import pytest

from mint_scout.data.bdb2020plus import (
    load_bdb2020plus_records,
    normalize_mol2_for_legacy,
    validate_ligand_conversion,
    write_ligand_centered_pocket,
)


def _pdb_atom(
    serial: int,
    atom_name: str,
    residue_name: str,
    chain: str,
    residue_id: int,
    x: float,
    y: float,
    z: float,
    element: str,
) -> str:
    return (
        f"ATOM  {serial:5d} {atom_name:^4s} {residue_name:>3s} {chain}{residue_id:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00          {element:>2s}\n"
    )


def _write_sdf(path: Path) -> None:
    path.write_text(
        "ligand\n"
        "  test\n"
        "\n"
        "  2  1  0  0  0  0  0  0  0  0999 V2000\n"
        "    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n"
        "    1.2000    0.0000    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0\n"
        "  1  2  1  0\n"
        "M  END\n"
        "$$$$\n",
        encoding="utf-8",
    )


def test_write_ligand_centered_pocket_keeps_complete_near_residues(tmp_path):
    ligand = tmp_path / "ligand.sdf"
    _write_sdf(ligand)
    protein = tmp_path / "protein.pdb"
    protein.write_text(
        _pdb_atom(1, "CA", "ALA", "A", 1, 2.0, 0.0, 0.0, "C")
        + _pdb_atom(2, "CB", "ALA", "A", 1, 12.0, 0.0, 0.0, "C")
        + _pdb_atom(3, "CA", "GLY", "A", 2, 20.0, 0.0, 0.0, "C"),
        encoding="utf-8",
    )
    output = tmp_path / "pocket.pdb"

    report = write_ligand_centered_pocket(
        protein, ligand, output, cutoff_angstrom=10.0
    )

    text = output.read_text(encoding="utf-8")
    assert report["pocket_residue_count"] == 1
    assert report["pocket_atom_count"] == 2
    assert "ALA" in text
    assert "GLY" not in text


def test_validate_ligand_conversion_accepts_added_hydrogens_without_heavy_atom_motion(
    tmp_path,
):
    source = tmp_path / "ligand.sdf"
    _write_sdf(source)
    converted = tmp_path / "ligand.mol2"
    converted.write_text(
        "@<TRIPOS>MOLECULE\nligand\n3 2 0 0 0\nSMALL\nNO_CHARGES\n\n"
        "@<TRIPOS>ATOM\n"
        "1 C1 0.0000 0.0000 0.0000 C.3 1 LIG 0.0\n"
        "2 O1 1.2000 0.0000 0.0000 O.3 1 LIG 0.0\n"
        "3 H1 -0.5000 0.0000 0.0000 H 1 LIG 0.0\n"
        "@<TRIPOS>BOND\n1 1 2 1\n2 1 3 1\n",
        encoding="utf-8",
    )

    report = validate_ligand_conversion(source, converted)

    assert report["heavy_atom_count"] == 2
    assert report["converted_hydrogen_count"] == 1
    assert report["maximum_heavy_atom_displacement_angstrom"] == pytest.approx(0.0)


def test_normalize_mol2_places_bonds_immediately_after_atoms(tmp_path):
    ligand = tmp_path / "charged.mol2"
    ligand.write_text(
        "@<TRIPOS>MOLECULE\ncharged\n1 0 0 0 0\nSMALL\nGASTEIGER\n\n"
        "@<TRIPOS>ATOM\n"
        "1 N1 0.0 0.0 0.0 N.4 1 LIG 1.0\n"
        "@<TRIPOS>UNITY_ATOM_ATTR\n1 1\ncharge 1\n"
        "@<TRIPOS>BOND\n",
        encoding="utf-8",
    )

    changed = normalize_mol2_for_legacy(ligand)

    text = ligand.read_text(encoding="utf-8")
    assert changed is True
    assert text.index("@<TRIPOS>ATOM") < text.index("@<TRIPOS>BOND")
    assert text.index("@<TRIPOS>BOND") < text.index("@<TRIPOS>UNITY_ATOM_ATTR")
    assert normalize_mol2_for_legacy(ligand) is False


def test_load_bdb2020plus_records_uses_only_accurate_rows(tmp_path):
    structures = tmp_path / "dataset"
    for pdb_id in ("1ABC", "2DEF"):
        sample_root = structures / pdb_id
        sample_root.mkdir(parents=True)
        (sample_root / "protein.pdb").write_text("ATOM\n", encoding="utf-8")
        (sample_root / "ligand.sdf").write_text("ligand\n", encoding="utf-8")
    metadata = tmp_path / "BDB2020+.csv"
    metadata.write_text(
        "pdbid,accurate,pKa\n1abc,True,7.25\n2def,False,8.0\n",
        encoding="utf-8",
    )

    records = load_bdb2020plus_records(metadata, structures)

    assert len(records) == 1
    assert records[0].pdb_id == "1ABC"
    assert records[0].target_pka == pytest.approx(7.25)
