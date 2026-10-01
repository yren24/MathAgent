from __future__ import annotations

import json

import pytest

from mint_scout.data.manifest_io import (
    load_configured_protein_ligand_records,
    load_dataset_manifest,
    task_card_from_config,
)
from mint_scout.schemas import (
    EvaluationMode,
    ManifestValidationError,
    route_evaluation_mode,
)


def _structure_files(tmp_path, sample_id: str):
    protein = tmp_path / f"{sample_id}_protein.pdb"
    ligand = tmp_path / f"{sample_id}_ligand.mol2"
    protein.write_text("protein", encoding="utf-8")
    ligand.write_text("ligand", encoding="utf-8")
    return protein, ligand


def test_csv_manifest_resolves_roles_and_routes_explicit_labeled_test(tmp_path):
    train_protein, train_ligand = _structure_files(tmp_path, "train-a")
    test_protein, test_ligand = _structure_files(tmp_path, "test-a")
    manifest_path = tmp_path / "samples.csv"
    manifest_path.write_text(
        "id,y,partition,receptor,compound,pdb_id\n"
        f"train-a,6.2,train,{train_protein.name},{train_ligand.name},1abc\n"
        f"test-a,7.1,test,{test_protein.name},{test_ligand.name},2def\n",
        encoding="utf-8",
    )

    manifest = load_dataset_manifest(
        manifest_path,
        dataset_id="binding-set",
        system_type="protein_ligand",
        columns={
            "sample_id": "id",
            "target": "y",
            "split": "partition",
            "roles": {"protein": "receptor", "ligand": "compound"},
            "identifiers": {"pdb_id": "pdb_id"},
        },
        label_name="pK",
    )

    card = task_card_from_config(
        {
            "task_id": "binding-set",
            "system_type": "protein_ligand",
            "task_type": "regression",
            "primary_metric": "pcc",
            "dataset_manifest": {
                "path": manifest_path.name,
                "label_name": "pK",
                "columns": {
                    "sample_id": "id",
                    "target": "y",
                    "split": "partition",
                    "roles": {"protein": "receptor", "ligand": "compound"},
                    "identifiers": {"pdb_id": "pdb_id"},
                },
            },
        },
        config_path=tmp_path / "task.yaml",
    )

    assert manifest.samples[0].role_paths["protein"] == train_protein
    assert manifest.samples[0].identifiers == {"pdb_id": "1abc"}
    assert card is not None
    plan = route_evaluation_mode(card)
    assert plan.mode == EvaluationMode.EXPLICIT_LABELED_TEST
    assert plan.modeling_sample_ids == ("train-a",)
    assert plan.evaluation_sample_ids == ("test-a",)


def test_jsonl_manifest_routes_labeled_and_unlabeled_samples(tmp_path):
    protein_a, ligand_a = _structure_files(tmp_path, "a")
    protein_b, ligand_b = _structure_files(tmp_path, "b")
    manifest_path = tmp_path / "samples.jsonl"
    rows = [
        {
            "sample_id": "a",
            "target": 5.0,
            "split": "train",
            "protein_path": str(protein_a),
            "ligand_path": str(ligand_a),
        },
        {
            "sample_id": "b",
            "target": None,
            "split": "inference",
            "protein_path": str(protein_b),
            "ligand_path": str(ligand_b),
        },
    ]
    manifest_path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    card = task_card_from_config(
        {
            "task_id": "mixed",
            "system_type": "protein_ligand",
            "task_type": "regression",
            "dataset_manifest": {"path": str(manifest_path)},
        },
        config_path=tmp_path / "task.yaml",
    )

    assert card is not None
    plan = route_evaluation_mode(card)
    assert plan.mode == EvaluationMode.LABELED_CV_PLUS_UNLABELED_INFERENCE
    assert plan.modeling_sample_ids == ("a",)
    assert plan.inference_sample_ids == ("b",)


def test_configured_records_select_validation_without_merging_test(tmp_path):
    rows = []
    for sample_id, split in (
        ("train-a", "train"),
        ("val-a", "validation"),
        ("test-a", "test"),
    ):
        protein, ligand = _structure_files(tmp_path, sample_id)
        rows.append(f"{sample_id},1.0,{split},{protein},{ligand}\n")
    manifest_path = tmp_path / "official.csv"
    manifest_path.write_text(
        "sample_id,target,split,protein_path,ligand_path\n" + "".join(rows),
        encoding="utf-8",
    )
    config = {
        "task_id": "official",
        "system_type": "protein_ligand",
        "task_type": "regression",
        "dataset_manifest": {"path": str(manifest_path)},
    }

    records = load_configured_protein_ligand_records(
        config, config_path=tmp_path / "task.yaml", split="validation"
    )

    assert [record.pdb_id for record in records] == ["val-a"]
    assert records[0].split == "validation"


def test_configured_records_carry_the_explicit_cv_group_identifier(tmp_path):
    protein, ligand = _structure_files(tmp_path, "chain-a")
    manifest_path = tmp_path / "grouped.csv"
    manifest_path.write_text(
        "sample_id,target,protein_path,ligand_path,pdb_id\n"
        f"chain-a,7.0,{protein},{ligand},1abc\n",
        encoding="utf-8",
    )
    config = {
        "task_id": "grouped",
        "system_type": "protein_ligand",
        "task_type": "regression",
        "cv": {"group_identifier": "pdb_id"},
        "dataset_manifest": {
            "path": str(manifest_path),
            "columns": {"identifiers": {"pdb_id": "pdb_id"}},
        },
    }

    records = load_configured_protein_ligand_records(
        config, config_path=tmp_path / "task.yaml", split="train"
    )

    assert records[0].group_id == "1abc"


@pytest.mark.parametrize("target", ["not-a-number", "nan", "inf"])
def test_manifest_rejects_invalid_numeric_targets(tmp_path, target):
    protein, ligand = _structure_files(tmp_path, "a")
    manifest_path = tmp_path / "samples.csv"
    manifest_path.write_text(
        "sample_id,target,protein_path,ligand_path\n"
        f"a,{target},{protein},{ligand}\n",
        encoding="utf-8",
    )

    with pytest.raises(ManifestValidationError, match="target"):
        load_dataset_manifest(
            manifest_path,
            dataset_id="bad-target",
            system_type="protein_ligand",
        )


def test_manifest_rejects_duplicate_ids_and_missing_files(tmp_path):
    protein, ligand = _structure_files(tmp_path, "a")
    duplicate_path = tmp_path / "duplicate.csv"
    duplicate_path.write_text(
        "sample_id,target,protein_path,ligand_path\n"
        f"a,1.0,{protein},{ligand}\n"
        f"a,2.0,{protein},{ligand}\n",
        encoding="utf-8",
    )
    with pytest.raises(ManifestValidationError, match="Duplicate sample ids"):
        load_dataset_manifest(
            duplicate_path,
            dataset_id="duplicates",
            system_type="protein_ligand",
        )

    missing_path = tmp_path / "missing.csv"
    missing_path.write_text(
        "sample_id,target,protein_path,ligand_path\n"
        f"b,1.0,{tmp_path / 'missing.pdb'},{ligand}\n",
        encoding="utf-8",
    )
    with pytest.raises(ManifestValidationError, match="missing structure path"):
        load_dataset_manifest(
            missing_path,
            dataset_id="missing",
            system_type="protein_ligand",
        )


def test_missing_role_is_preserved_for_needs_user_input_routing(tmp_path):
    protein, _ = _structure_files(tmp_path, "a")
    manifest_path = tmp_path / "samples.csv"
    manifest_path.write_text(
        "sample_id,target,protein_path,ligand_path\n"
        f"a,1.0,{protein},\n",
        encoding="utf-8",
    )
    card = task_card_from_config(
        {
            "task_id": "ambiguous",
            "system_type": "protein_ligand",
            "task_type": "regression",
            "dataset_manifest": {"path": str(manifest_path)},
        },
        config_path=tmp_path / "task.yaml",
    )

    assert card is not None
    plan = route_evaluation_mode(card)
    assert plan.mode == EvaluationMode.NEEDS_USER_INPUT
    assert "ligand" in plan.warnings[0]
