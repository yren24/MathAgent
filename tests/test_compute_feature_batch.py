import json
from pathlib import Path

from mint_scout.artifact_memory import inspect_artifact
from mint_scout.compute_feature_batch import main as compute_feature_batch_main
from mint_scout.execution.jobs import inspect_feature_manifest


def _write_batch_fixture(tmp_path: Path) -> Path:
    structures = tmp_path / "all-pdbs"
    for pdb_id in ("10gs", "11gs", "12gs"):
        folder = structures / pdb_id
        folder.mkdir(parents=True)
        (folder / f"{pdb_id}_pocket.pdb").write_text("", encoding="utf-8")
        (folder / f"{pdb_id}_ligand.mol2").write_text("", encoding="utf-8")
    (tmp_path / "2016_INDEX_refined.data").write_text(
        "10gs  1.90  1998   5.82  Ki=1.5uM      // 10gs.pdf (VWW)\n"
        "11gs  2.00  1999   6.00  Ki=1.0uM      // 11gs.pdf (VWW)\n"
        "12gs  2.00  1999   7.00  Ki=1.0uM      // 12gs.pdf (VWW)\n",
        encoding="utf-8",
    )
    (tmp_path / "train_data_2016.txt").write_text("['10gs', '11gs']", encoding="utf-8")
    (tmp_path / "test_data_2016.txt").write_text("['12gs']", encoding="utf-8")
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    (legacy / "feature.py").write_text(
        "from pathlib import Path\n"
        "class ProteinLigand:\n"
        "    def __init__(self, pdb, typ, pdb_folder=None, pdb_feature_folder=None):\n"
        "        Path(pdb_feature_folder).mkdir(parents=True, exist_ok=True)\n"
        "        Path(pdb_feature_folder, f'{pdb}.npy').write_bytes(typ.encode())\n",
        encoding="utf-8",
    )
    config = tmp_path / "task.yaml"
    config.write_text(
        "task_id: toy\n"
        "system_type: protein_ligand\n"
        "legacy:\n"
        "  plbind_root: " + str(legacy) + "\n"
        "data_audit:\n"
        "  casf:\n"
        "    year: 2016\n"
        "    index_root: " + str(tmp_path) + "\n"
        "    structures_root: " + str(structures) + "\n"
        "feature_generation:\n"
        "  output_root: " + str(tmp_path / "features") + "\n"
        "  validate_output_shape: false\n",
        encoding="utf-8",
    )
    return config


def test_compute_feature_batch_dry_run_writes_manifest(tmp_path: Path):
    config = _write_batch_fixture(tmp_path)
    manifest = tmp_path / "runs" / "manifest.jsonl"

    code = compute_feature_batch_main(
        [
            "--config",
            str(config),
            "--invariant",
            "PL",
            "--split",
            "train",
            "--limit",
            "2",
            "--manifest",
            str(manifest),
            "--dry-run",
        ]
    )

    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    assert code == 0
    assert [row["sample_id"] for row in rows] == ["10gs", "11gs"]
    assert {row["status"] for row in rows} == {"dry_run"}
    assert rows[0]["output_path"].endswith("/PL/10gs.npy")


def test_compute_feature_batch_records_missing_structure_failure(tmp_path: Path):
    config = _write_batch_fixture(tmp_path)
    (tmp_path / "all-pdbs" / "11gs" / "11gs_ligand.mol2").unlink()
    manifest = tmp_path / "runs" / "manifest.jsonl"

    code = compute_feature_batch_main(
        [
            "--config",
            str(config),
            "--invariant",
            "PL",
            "--split",
            "train",
            "--limit",
            "2",
            "--manifest",
            str(manifest),
        ]
    )

    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    assert code == 1
    assert rows[0]["status"] == "computed"
    assert rows[1]["status"] == "failed"
    assert rows[1]["error_type"] == "FileNotFoundError"


def test_compute_feature_batch_reads_probe_ids_from_json(tmp_path: Path):
    config = _write_batch_fixture(tmp_path)
    selection = tmp_path / "probe.json"
    selection.write_text(json.dumps({"probe_sample_ids": ["11gs"]}), encoding="utf-8")
    manifest = tmp_path / "runs" / "manifest.jsonl"

    code = compute_feature_batch_main(
        [
            "--config",
            str(config),
            "--invariant",
            "PL",
            "--split",
            "train",
            "--sample-id-file",
            str(selection),
            "--manifest",
            str(manifest),
            "--dry-run",
        ]
    )

    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    assert code == 0
    assert [row["sample_id"] for row in rows] == ["11gs"]


def test_manifest_batch_stages_arbitrary_paths_and_keys_cache_by_content(tmp_path: Path):
    source_root = tmp_path / "source"
    source_root.mkdir()
    protein = source_root / "receptor-any-name.pdb"
    ligand = source_root / "compound-any-name.mol2"
    protein.write_text("protein-v1", encoding="utf-8")
    ligand.write_text("ligand-v1", encoding="utf-8")
    manifest_csv = tmp_path / "samples.csv"
    manifest_csv.write_text(
        "sample_id,target,split,protein_path,ligand_path\n"
        f"custom-1,6.4,train,{protein},{ligand}\n",
        encoding="utf-8",
    )
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    (legacy / "feature.py").write_text(
        "from pathlib import Path\n"
        "class ProteinLigand:\n"
        "    def __init__(self, pdb, typ, pdb_folder=None, pdb_feature_folder=None):\n"
        "        root = Path(pdb_folder, pdb)\n"
        "        assert Path(root, f'{pdb}_pocket.pdb').is_file()\n"
        "        assert Path(root, f'{pdb}_ligand.mol2').is_file()\n"
        "        Path(pdb_feature_folder).mkdir(parents=True, exist_ok=True)\n"
        "        Path(pdb_feature_folder, f'{pdb}.npy').write_bytes(typ.encode())\n",
        encoding="utf-8",
    )
    config = tmp_path / "task.yaml"
    config.write_text(
        "task_id: custom-binding\n"
        "system_type: protein_ligand\n"
        "task_type: regression\n"
        "dataset_manifest:\n"
        "  path: samples.csv\n"
        "legacy:\n"
        f"  plbind_root: {legacy}\n"
        "feature_generation:\n"
        f"  output_root: {tmp_path / 'features'}\n"
        f"  structure_staging_root: {tmp_path / 'staging'}\n"
        "  validate_output_shape: false\n",
        encoding="utf-8",
    )

    first_manifest = tmp_path / "first.jsonl"
    first_code = compute_feature_batch_main(
        [
            "--config",
            str(config),
            "--invariant",
            "PL",
            "--split",
            "train",
            "--manifest",
            str(first_manifest),
        ]
    )
    first = json.loads(first_manifest.read_text(encoding="utf-8"))
    assert first_code == 0
    assert first["status"] == "computed"
    assert first["structure_inputs"]["protein"]["source_path"] == str(protein)
    assert "input-" + first["structure_inputs"]["combined_sha256"] in first["output_path"]
    assert (tmp_path / "staging" / "custom-1" / "custom-1_pocket.pdb").is_symlink()
    inspected = inspect_feature_manifest(first_manifest)
    assert inspected["state"] == "COMPLETE"
    assert inspected["structure_input_set_hash"]
    assert inspect_artifact(first_manifest).metadata["structure_input_set_hash"]

    ligand.write_text("ligand-v2", encoding="utf-8")
    second_manifest = tmp_path / "second.jsonl"
    second_code = compute_feature_batch_main(
        [
            "--config",
            str(config),
            "--invariant",
            "PL",
            "--split",
            "train",
            "--manifest",
            str(second_manifest),
        ]
    )
    second = json.loads(second_manifest.read_text(encoding="utf-8"))
    assert second_code == 0
    assert second["status"] == "computed"
    assert second["output_path"] != first["output_path"]
    assert Path(first["output_path"]).is_file()
    assert Path(second["output_path"]).is_file()
