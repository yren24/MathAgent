from __future__ import annotations

import csv
import json
from pathlib import Path

from mint_scout.data.preparation import PREPARATION_SCHEMA, prepare_dataset_from_config
from mint_scout.execution.jobs import inspect_dataset_preparation_report


def test_paired_structure_provider_builds_a_valid_full_cv_manifest(tmp_path: Path):
    structures = tmp_path / "structures"
    for sample_id in ("a", "b"):
        sample_root = structures / sample_id
        sample_root.mkdir(parents=True)
        (sample_root / f"{sample_id}_protein.pdb").write_text(
            "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 20.00           C\n",
            encoding="utf-8",
        )
        (sample_root / f"{sample_id}_ligand.mol2").write_text(
            "@<TRIPOS>MOLECULE\nligand\n@<TRIPOS>ATOM\n"
            "1 N1 2.0 0.0 0.0 N.3 1 LIG 0.0\n@<TRIPOS>BOND\n",
            encoding="utf-8",
        )
    metadata = tmp_path / "metadata.csv"
    metadata.write_text(
        "id,affinity,fold\na,7.1,\nb,8.2,\n", encoding="utf-8"
    )
    manifest = tmp_path / "prepared" / "manifest.csv"
    task = tmp_path / "task.yaml"
    task.write_text(
        "\n".join(
            [
                "task_id: toy",
                "system_type: protein_ligand",
                "task_type: regression",
                "dataset_preparation:",
                "  provider: paired_structure_manifest",
                f"  metadata_path: {metadata}",
                f"  structures_root: {structures}",
                f"  output_manifest: {manifest}",
                "  columns:",
                "    sample_id: id",
                "    target: affinity",
                "    split: fold",
                "  templates:",
                '    protein: "{sample_id}/{sample_id}_protein.pdb"',
                '    ligand: "{sample_id}/{sample_id}_ligand.mol2"',
                "dataset_manifest:",
                f"  path: {manifest}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    report_path = tmp_path / "preparation.json"

    report = prepare_dataset_from_config(task, report_path=report_path)

    assert report["report_schema"] == PREPARATION_SCHEMA
    assert report["sample_ids"] == ["a", "b"]
    assert report["provider_report"]["labeled_count"] == 2
    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["split"] for row in rows] == ["", ""]
    assert rows[0]["protein_path"].endswith("a/a_protein.pdb")
    assert inspect_dataset_preparation_report(report_path)["state"] == "COMPLETE"


def test_preparation_report_detects_manifest_mutation(tmp_path: Path):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("sample_id\na\n", encoding="utf-8")
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(
            {
                "report_schema": PREPARATION_SCHEMA,
                "dataset_id": "toy",
                "status": "PASS",
                "manifest": str(manifest),
                "manifest_sha256": "wrong",
                "sample_count": 1,
                "sample_ids": ["a"],
            }
        ),
        encoding="utf-8",
    )

    inspection = inspect_dataset_preparation_report(report)

    assert inspection["state"] == "INVALID"
    assert "hash changed" in inspection["error"]
