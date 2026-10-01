from __future__ import annotations

import csv
import hashlib
import tarfile
from pathlib import Path

import pytest

import mint_scout.data.hiqbind as hiqbind


HEADER = (
    "PDBID,Resolution,Year,Ligand Name,Ligand Chain,Ligand Residue Number,"
    "Binding Affinity Measurement,Binding Affinity Sign,Binding Affinity Value,"
    "Binding Affinity Unit,Log Binding Affinity,Binding Affinity Source,"
    "Binding Affinity Annotation,Protein UniProtID,Protein UniProtName,Ligand SMILES,"
    "Ligand MW,Ligand LogP,Ligand TPSA,Ligand NumRotBond,Ligand NumHeavyAtoms,"
    "Ligand NumHDon,Ligand NumHAcc,Ligand QED\n"
)


def _row(
    pdb_id: str,
    ligand: str,
    measurement: str,
    sign: str,
    value: float,
    unit: str,
    log_molar: float,
) -> str:
    return (
        f"{pdb_id},2.0,2022,{ligand},A,101,{measurement},{sign},{value},{unit},"
        f"{log_molar},MOAD,{measurement}={value}{unit},P1,Protein,C,12.0,0.0,"
        "1.0,0,1,0,0,0.5\n"
    )


def _metadata(path: Path) -> None:
    path.write_text(
        HEADER
        + _row("1abc", "LIG", "ki", "=", 10.0, "nM", -8.0)
        + _row("2def", "DRG", "ic50", "=", 1.0, "uM", -6.0)
        + _row("3ghi", "CMP", "kd", ">=", 100.0, "nM", -7.0),
        encoding="utf-8",
    )


def test_hiqbind_loader_filters_measurement_and_sign_and_converts_to_paffinity(
    tmp_path: Path,
):
    metadata = tmp_path / "hiqbind.csv"
    _metadata(metadata)

    records = hiqbind.load_hiqbind_records(
        metadata, measurements=("kd", "ki"), affinity_signs=("=",)
    )

    assert len(records) == 1
    assert records[0].source_key == "1abc_LIG_A_101"
    assert records[0].target_p_affinity == pytest.approx(8.0)
    assert records[0].sample_id.startswith("1abc-")


def _selection_record(index: int, *, pdb_id: str | None = None):
    return hiqbind.HiQBindRecord(
        source_key=f"{pdb_id or f'p{index:04d}'}_L{index:04d}_A_1",
        pdb_id=pdb_id or f"p{index:04d}",
        ligand_name=f"L{index:04d}",
        ligand_chain="A",
        ligand_residue_number="1",
        measurement="kd" if index % 2 else "ki",
        sign="=",
        affinity_value=1.0,
        affinity_unit="nm",
        target_p_affinity=3.0 + 8.0 * index / 199.0,
        year=1990 + index % 31,
        affinity_source=("MOAD", "BindingDB", "BioLiP")[index % 3],
        ligand_heavy_atoms=5 + index % 46,
    )


def test_stratified_hiqbind_subset_is_deterministic_and_distribution_aware():
    records = tuple(_selection_record(index) for index in range(200))
    config = hiqbind.HiQBindSubsetConfig(
        strategy="stratified",
        random_seed=2026,
        target_quantile_bins=10,
        ligand_size_quantile_bins=5,
        max_per_pdb_id=1,
    )

    first, report = hiqbind.select_hiqbind_subset(
        records, max_samples=50, config=config
    )
    second, repeated = hiqbind.select_hiqbind_subset(
        records, max_samples=50, config=config
    )

    assert [record.sample_id for record in first] == [
        record.sample_id for record in second
    ]
    assert report["selection_hash"] == repeated["selection_hash"]
    assert len(first) == 50
    assert report["selected_pdb_group_count"] == 50
    assert report["target_p_affinity"]["ks_distance"] < 0.10
    assert report["ligand_heavy_atoms"]["ks_distance"] < 0.10
    assert [record.source_key for record in first] != [
        record.source_key for record in records[:50]
    ]


def test_stratified_hiqbind_subset_enforces_pdb_group_cap():
    records = tuple(
        _selection_record(index, pdb_id=f"p{index // 3:03d}")
        for index in range(60)
    )

    selected, report = hiqbind.select_hiqbind_subset(
        records,
        max_samples=15,
        config=hiqbind.HiQBindSubsetConfig(
            strategy="stratified", max_per_pdb_id=1
        ),
    )

    assert len({record.pdb_id for record in selected}) == 15
    assert report["candidate_count_after_group_cap"] == 20


def test_hiqbind_summary_profiles_all_rows_and_clean_affinity_subset(tmp_path: Path):
    metadata = tmp_path / "hiqbind.csv"
    _metadata(metadata)

    report = hiqbind.summarize_hiqbind_metadata(metadata)

    assert report["status"] == "PASS"
    assert report["row_count"] == 3
    assert report["unique_structure_count"] == 3
    assert report["measurement_counts"] == {"ic50": 1, "kd": 1, "ki": 1}
    assert report["sign_counts"] == {"=": 2, ">=": 1}
    assert report["exact_kd_ki"]["count"] == 1


def test_hiqbind_selection_summary_reports_invalid_rows_inside_filters(
    tmp_path: Path,
):
    metadata = tmp_path / "hiqbind.csv"
    invalid = _row("4jkl", "BAD", "ki", "=", 10.0, "nM", -8.0).replace(
        ",nM,-8.0,", ",,,"
    )
    metadata.write_text(
        HEADER
        + _row("1abc", "LIG", "ki", "=", 10.0, "nM", -8.0)
        + _row("2def", "DRG", "kd", "=", 1.0, "uM", -6.0)
        + invalid,
        encoding="utf-8",
    )

    report = hiqbind.summarize_hiqbind_selection(
        metadata, measurements=("kd", "ki"), affinity_signs=("=",)
    )

    assert report["candidate_count"] == 3
    assert report["valid_count"] == 2
    assert report["invalid_affinity_count"] == 1
    assert report["status"] == "REVIEW"


def test_hiqbind_preparation_writes_standard_unsplit_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    metadata = tmp_path / "hiqbind.csv"
    metadata.write_text(
        HEADER + _row("1abc", "LIG", "ki", "=", 10.0, "nM", -8.0),
        encoding="utf-8",
    )
    source_root = tmp_path / "raw_data_hiq_sm"
    source_key = "1abc_LIG_A_101"
    source_dir = source_root / "1abc" / source_key
    source_dir.mkdir(parents=True)
    protein = source_dir / f"{source_key}_protein_refined.pdb"
    ligand = source_dir / f"{source_key}_ligand_refined.sdf"
    protein.write_text("ATOM\n", encoding="utf-8")
    ligand.write_text("ligand\n", encoding="utf-8")

    def fake_convert(source, output, **_kwargs):
        assert source == ligand
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("@<TRIPOS>MOLECULE\n", encoding="utf-8")
        return {"maximum_heavy_atom_displacement_angstrom": 0.0}

    def fake_pocket(source, _ligand, output, **_kwargs):
        assert source == protein
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("ATOM\nEND\n", encoding="utf-8")
        return {"pocket_atom_count": 1, "pocket_residue_count": 1}

    monkeypatch.setattr(hiqbind, "convert_sdf_to_mol2", fake_convert)
    monkeypatch.setattr(hiqbind, "write_ligand_centered_pocket", fake_pocket)
    manifest = tmp_path / "prepared" / "manifest.csv"
    report = hiqbind.prepare_hiqbind(
        source_root=source_root,
        metadata_path=metadata,
        output_root=tmp_path / "prepared" / "structures",
        manifest_path=manifest,
        audit_report_path=tmp_path / "prepared" / "detail.json",
        measurements=("kd", "ki"),
        affinity_signs=("=",),
        metadata_md5=hashlib.md5(metadata.read_bytes()).hexdigest(),
    )

    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert report["status"] == "PASS"
    assert report["run_kind"] == "full_dataset"
    assert report["selection_summary"]["candidate_count"] == 1
    assert report["selection_summary"]["valid_count"] == 1
    assert report["max_samples_applied"] is False
    assert report["subset_selection"]["config"]["strategy"] == "source_order"
    assert rows[0]["target"] == "8"
    assert rows[0]["split"] == ""
    assert rows[0]["hiqbind_id"] == source_key
    assert Path(rows[0]["protein_path"]).is_file()
    assert Path(rows[0]["ligand_path"]).is_file()


def test_hiqbind_loader_rejects_multiple_labels_for_one_structure(tmp_path: Path):
    metadata = tmp_path / "hiqbind.csv"
    row = _row("1abc", "LIG", "ki", "=", 10.0, "nM", -8.0)
    metadata.write_text(HEADER + row + row, encoding="utf-8")

    with pytest.raises(ValueError, match="multiple labels"):
        hiqbind.load_hiqbind_records(
            metadata, measurements=("ki",), affinity_signs=("=",)
        )


def test_hiqbind_preparation_recovers_an_incomplete_extraction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    metadata = tmp_path / "hiqbind.csv"
    metadata.write_text(
        HEADER + _row("1abc", "LIG", "ki", "=", 10.0, "nM", -8.0),
        encoding="utf-8",
    )
    source_key = "1abc_LIG_A_101"
    archive_tree = tmp_path / "archive" / "raw_data_hiq_sm" / "1abc" / source_key
    archive_tree.mkdir(parents=True)
    (archive_tree / f"{source_key}_protein_refined.pdb").write_text(
        "ATOM\n", encoding="utf-8"
    )
    (archive_tree / f"{source_key}_ligand_refined.sdf").write_text(
        "ligand\n", encoding="utf-8"
    )
    archive = tmp_path / "hiqbind.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        handle.add(archive_tree.parents[1], arcname="raw_data_hiq_sm")

    extract_root = tmp_path / "extracted"
    source_root = extract_root / "raw_data_hiq_sm"
    source_root.mkdir(parents=True)

    def fake_convert(_source, output, **_kwargs):
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("@<TRIPOS>MOLECULE\n", encoding="utf-8")
        return {"maximum_heavy_atom_displacement_angstrom": 0.0}

    def fake_pocket(_source, _ligand, output, **_kwargs):
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("ATOM\nEND\n", encoding="utf-8")
        return {"pocket_atom_count": 1, "pocket_residue_count": 1}

    monkeypatch.setattr(hiqbind, "convert_sdf_to_mol2", fake_convert)
    monkeypatch.setattr(hiqbind, "write_ligand_centered_pocket", fake_pocket)
    report = hiqbind.prepare_hiqbind(
        source_root=source_root,
        source_archive=archive,
        source_archive_md5=hashlib.md5(archive.read_bytes()).hexdigest(),
        extract_root=extract_root,
        metadata_path=metadata,
        metadata_md5=hashlib.md5(metadata.read_bytes()).hexdigest(),
        output_root=tmp_path / "prepared" / "structures",
        manifest_path=tmp_path / "prepared" / "manifest.csv",
        audit_report_path=tmp_path / "prepared" / "detail.json",
        measurements=("ki",),
        affinity_signs=("=",),
    )

    assert report["status"] == "PASS"
    assert report["source_archive_extracted"] is True


def test_hiqbind_invalid_affinity_requires_an_explicit_skip_policy(tmp_path: Path):
    metadata = tmp_path / "hiqbind.csv"
    invalid = _row("1abc", "LIG", "ki", "=", 10.0, "nM", -8.0).replace(
        ",nM,-8.0,", ",,,"
    )
    metadata.write_text(
        HEADER
        + invalid
        + _row("2def", "DRG", "kd", "=", 1.0, "uM", -6.0),
        encoding="utf-8",
    )

    summary = hiqbind.summarize_hiqbind_metadata(metadata)
    assert summary["status"] == "REVIEW"
    assert summary["invalid_affinity_count"] == 1
    with pytest.raises(ValueError, match="Binding Affinity Unit"):
        hiqbind.load_hiqbind_records(
            metadata, measurements=("kd", "ki"), affinity_signs=("=",)
        )
    records = hiqbind.load_hiqbind_records(
        metadata,
        measurements=("kd", "ki"),
        affinity_signs=("=",),
        invalid_affinity_policy="skip",
    )
    assert [record.pdb_id for record in records] == ["2def"]
