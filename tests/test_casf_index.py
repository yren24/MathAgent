from mint_scout.data.casf_index import load_casf_records, load_legacy_split, load_refined_index
from mint_scout.data_audit import main as data_audit_main


def test_refined_index_parser_skips_header_and_reads_labels(tmp_path):
    index_path = tmp_path / "2016_INDEX_refined.data"
    index_path.write_text(
        "# header\n"
        "# another header\n"
        "10gs  1.90  1998   5.82  Ki=1.5uM      // 10gs.pdf (VWW)\n",
        encoding="utf-8",
    )

    assert load_refined_index(index_path) == {"10gs": 5.82}


def test_legacy_split_parser_filters_bad_ids_and_detects_split(tmp_path):
    (tmp_path / "train_data_2016.txt").write_text("['10gs', '3tei']", encoding="utf-8")
    (tmp_path / "test_data_2016.txt").write_text("['11gs', '4as6']", encoding="utf-8")

    assert load_legacy_split(index_root=tmp_path, year=2016) == {
        "10gs": "train",
        "11gs": "test",
    }


def test_load_casf_records_builds_expected_structure_paths(tmp_path):
    structures = tmp_path / "all-pdbs"
    for pdb_id in ("10gs", "11gs"):
        folder = structures / pdb_id
        folder.mkdir(parents=True)
        (folder / f"{pdb_id}_pocket.pdb").write_text("", encoding="utf-8")
        (folder / f"{pdb_id}_ligand.mol2").write_text("", encoding="utf-8")
    (tmp_path / "2016_INDEX_refined.data").write_text(
        "10gs  1.90  1998   5.82  Ki=1.5uM      // 10gs.pdf (VWW)\n"
        "11gs  2.00  1999   6.00  Ki=1.0uM      // 11gs.pdf (VWW)\n",
        encoding="utf-8",
    )
    (tmp_path / "train_data_2016.txt").write_text("['10gs']", encoding="utf-8")
    (tmp_path / "test_data_2016.txt").write_text("['11gs']", encoding="utf-8")

    records = load_casf_records(index_root=tmp_path, structures_root=structures, year=2016)

    assert [record.pdb_id for record in records] == ["10gs", "11gs"]
    assert records[0].split == "train"
    assert records[1].split == "test"
    assert records[0].protein_path == structures / "10gs" / "10gs_pocket.pdb"
    assert records[0].ligand_path == structures / "10gs" / "10gs_ligand.mol2"


def test_data_audit_cli_uses_casf_config_paths(tmp_path):
    structures = tmp_path / "all-pdbs"
    folder = structures / "10gs"
    folder.mkdir(parents=True)
    (folder / "10gs_pocket.pdb").write_text(
        "ATOM      1  N   ALA A   1      11.104  13.207   9.211  1.00 20.00           N\n",
        encoding="utf-8",
    )
    (folder / "10gs_ligand.mol2").write_text(
        "@<TRIPOS>MOLECULE\n"
        "ligand\n"
        "@<TRIPOS>ATOM\n"
        "1 C1 0.0 0.0 0.0 C.3 1 LIG 0.0\n",
        encoding="utf-8",
    )
    (tmp_path / "2016_INDEX_refined.data").write_text(
        "10gs  1.90  1998   5.82  Ki=1.5uM      // 10gs.pdf (VWW)\n",
        encoding="utf-8",
    )
    (tmp_path / "train_data_2016.txt").write_text("['10gs']", encoding="utf-8")
    (tmp_path / "test_data_2016.txt").write_text("[]", encoding="utf-8")
    config = tmp_path / "task.yaml"
    output_json = tmp_path / "inventory.json"
    output_md = tmp_path / "inventory.md"
    config.write_text(
        "task_id: toy\n"
        "system_type: protein_ligand\n"
        "pair_schema:\n"
        "  schema_id: casf_protein_ligand_40_v1\n"
        "  ligand_elements: [C, N, O, S, P, F, Cl, Br, I, H]\n"
        "data_audit:\n"
        "  casf:\n"
        "    year: 2016\n"
        "    index_root: " + str(tmp_path) + "\n"
        "    structures_root: " + str(structures) + "\n",
        encoding="utf-8",
    )

    data_audit_main(
        [
            "--config",
            str(config),
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
        ]
    )

    assert '"schema_review_required": false' in output_json.read_text(encoding="utf-8")


def test_data_audit_allow_empty_tolerates_missing_casf_index(tmp_path):
    config = tmp_path / "task.yaml"
    output_json = tmp_path / "inventory.json"
    output_md = tmp_path / "inventory.md"
    config.write_text(
        "task_id: missing\n"
        "system_type: protein_ligand\n"
        "pair_schema:\n"
        "  schema_id: casf_protein_ligand_40_v1\n"
        "  ligand_elements: [C, N, O, S, P, F, Cl, Br, I, H]\n"
        "data_audit:\n"
        "  casf:\n"
        "    year: 2016\n"
        "    source: refined_index\n"
        "    index_root: /definitely/missing/index\n"
        "    structures_root: /definitely/missing/all-pdbs\n",
        encoding="utf-8",
    )

    code = data_audit_main(
        [
            "--config",
            str(config),
            "--allow-empty",
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
        ]
    )

    assert code == 0
    assert output_json.exists()
