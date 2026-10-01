import json

from mint_scout.data.element_inventory import (
    build_element_inventory_report,
    iter_elements_from_file,
)
from mint_scout.data.element_pairs import make_casf_protein_ligand_schema
from mint_scout.data_audit import main as data_audit_main


def test_pdb_fallback_does_not_read_alpha_carbon_as_calcium(tmp_path):
    pdb = tmp_path / "protein.pdb"
    pdb.write_text(
        "ATOM      1  CA  ALA A   1      11.104  13.207   9.211  1.00 20.00\n"
        "ATOM      2  SG  CYS A   2      10.104  12.207   8.211  1.00 20.00           S\n",
        encoding="utf-8",
    )

    assert tuple(iter_elements_from_file(pdb)) == ("C", "S")


def test_inventory_reports_role_specific_out_of_schema_elements(tmp_path):
    protein = tmp_path / "protein.pdb"
    ligand = tmp_path / "ligand.mol2"
    protein.write_text(
        "ATOM      1  N   ALA A   1      11.104  13.207   9.211  1.00 20.00           N\n"
        "ATOM      2  CA  ALA A   1      11.104  13.207   9.211  1.00 20.00           C\n",
        encoding="utf-8",
    )
    ligand.write_text(
        "@<TRIPOS>MOLECULE\n"
        "ligand\n"
        "@<TRIPOS>ATOM\n"
        "1 C1 0.0 0.0 0.0 C.3 1 LIG 0.0\n"
        "2 CL1 1.0 0.0 0.0 Cl 1 LIG 0.0\n"
        "3 ZN1 2.0 0.0 0.0 Zn 1 LIG 0.0\n",
        encoding="utf-8",
    )

    report = build_element_inventory_report(
        schema=make_casf_protein_ligand_schema(),
        protein_paths=(protein,),
        ligand_paths=(ligand,),
    )

    protein_role, ligand_role = report.roles
    assert protein_role.atom_counts == {"C": 1, "N": 1}
    assert ligand_role.atom_counts == {"C": 1, "Cl": 1, "Zn": 1}
    assert protein_role.out_of_schema == ()
    assert ligand_role.out_of_schema == ("Zn",)
    assert report.schema_review_required is True


def test_inventory_can_tolerate_legacy_ignored_protein_hydrogen(tmp_path):
    protein = tmp_path / "protein.pdb"
    protein.write_text(
        "ATOM      1  H   ALA A   1      11.104  13.207   9.211  1.00 20.00           H\n",
        encoding="utf-8",
    )

    report = build_element_inventory_report(
        schema=make_casf_protein_ligand_schema(),
        protein_paths=(protein,),
        tolerated_elements_by_role={"protein": ("H",)},
    )

    protein_role = report.roles[0]
    assert protein_role.atom_counts == {"H": 1}
    assert protein_role.out_of_schema == ()
    assert report.schema_review_required is False


def test_data_audit_cli_writes_json_and_markdown(tmp_path):
    config = tmp_path / "task.yaml"
    ligand = tmp_path / "ligand.xyz"
    output_json = tmp_path / "inventory.json"
    output_md = tmp_path / "inventory.md"
    config.write_text(
        "task_id: toy\n"
        "system_type: protein_ligand\n"
        "pair_schema:\n"
        "  schema_id: casf_protein_ligand_40_v1\n"
        "  ligand_elements: [C, N, O, S, P, F, Cl, Br, I, H]\n",
        encoding="utf-8",
    )
    ligand.write_text("2\ncomment\nC 0 0 0\nSe 1 0 0\n", encoding="utf-8")

    exit_code = data_audit_main(
        [
            "--config",
            str(config),
            "--ligand-glob",
            str(ligand),
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
        ]
    )

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert payload["schema_review_required"] is True
    assert payload["roles"][1]["out_of_schema"] == ["Se"]
    assert "Element Inventory Audit" in output_md.read_text(encoding="utf-8")


def test_data_audit_uses_manifest_modeling_pool_and_reports_targets(tmp_path):
    protein = tmp_path / "renamed-pocket.pdb"
    ligand = tmp_path / "renamed-compound.mol2"
    protein.write_text(
        "ATOM      1  C   ALA A   1       0.000   0.000   0.000  1.00 20.00           C\n",
        encoding="utf-8",
    )
    ligand.write_text(
        "@<TRIPOS>MOLECULE\nligand\n@<TRIPOS>ATOM\n"
        "1 N1 1.0 0.0 0.0 N.3 1 LIG 0.0\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "samples.csv"
    manifest.write_text(
        "sample_id,target,protein_path,ligand_path\n"
        f"sample-a,6.5,{protein.name},{ligand.name}\n",
        encoding="utf-8",
    )
    config = tmp_path / "task.yaml"
    config.write_text(
        "task_id: generic-binding\n"
        "system_type: protein_ligand\n"
        "task_type: regression\n"
        "primary_metric: pcc\n"
        "dataset_manifest:\n"
        "  path: samples.csv\n"
        "  label_name: pK\n",
        encoding="utf-8",
    )
    output_json = tmp_path / "audit.json"

    data_audit_main(
        [
            "--config",
            str(config),
            "--output-json",
            str(output_json),
            "--output-md",
            str(tmp_path / "audit.md"),
        ]
    )

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["report_schema"] == "mint-agent.dataset-audit.v1"
    assert len(payload["audit_input_hash"]) == 16
    assert payload["sample_ids"] == ["sample-a"]
    assert payload["target_summary"]["mean"] == 6.5
    assert payload["structure_size_summary"]["protein_atom_count"]["mean"] == 1.0
    assert payload["structure_size_summary"]["ligand_atom_count"]["mean"] == 1.0
    assert payload["structure_size_summary"]["total_atom_count"]["mean"] == 2.0
    assert payload["status"] == "PASS"
    markdown = (tmp_path / "audit.md").read_text(encoding="utf-8")
    assert "# Dataset Profile" in markdown
    assert "| Total atoms | 1 | 2" in markdown


def test_data_audit_can_select_external_test_manifest_rows(tmp_path):
    protein = tmp_path / "pocket.pdb"
    ligand = tmp_path / "ligand.mol2"
    protein.write_text(
        "ATOM      1  C   ALA A   1       0.000   0.000   0.000  1.00 20.00           C\n",
        encoding="utf-8",
    )
    ligand.write_text(
        "@<TRIPOS>MOLECULE\nligand\n@<TRIPOS>ATOM\n"
        "1 N1 1.0 0.0 0.0 N.3 1 LIG 0.0\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "samples.csv"
    manifest.write_text(
        "sample_id,target,split,protein_path,ligand_path\n"
        f"external-a,7.5,test,{protein.name},{ligand.name}\n",
        encoding="utf-8",
    )
    config = tmp_path / "task.yaml"
    config.write_text(
        "task_id: external-binding\n"
        "system_type: protein_ligand\n"
        "task_type: regression\n"
        "dataset_manifest:\n"
        "  path: samples.csv\n"
        "data_audit:\n"
        "  manifest_split: test\n",
        encoding="utf-8",
    )
    output_json = tmp_path / "audit.json"

    data_audit_main(
        [
            "--config",
            str(config),
            "--output-json",
            str(output_json),
            "--output-md",
            str(tmp_path / "audit.md"),
        ]
    )

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["modeling_scope"] == "test"
    assert payload["sample_ids"] == ["external-a"]
    assert payload["target_summary"]["mean"] == 7.5
