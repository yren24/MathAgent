from __future__ import annotations

import json

from mint_scout.design_representation import (
    _representation_design_input_hash_without_policy_extensions,
    main,
    representation_design_input_hash,
)


def _write_complex(root, sample_id: str, distance: float):
    sample_root = root / sample_id
    sample_root.mkdir(parents=True)
    (sample_root / f"{sample_id}_pocket.pdb").write_text(
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 20.00           C\n",
        encoding="utf-8",
    )
    (sample_root / f"{sample_id}_ligand.mol2").write_text(
        "@<TRIPOS>MOLECULE\nligand\n@<TRIPOS>ATOM\n"
        f"1 N1 {distance:.3f} 0.000 0.000 N.3 1 LIG 0.0\n",
        encoding="utf-8",
    )


def test_design_representation_cli_writes_frozen_adaptive_spec(tmp_path):
    structures = tmp_path / "structures"
    index = tmp_path / "index"
    index.mkdir()
    for sample_id, distance in (("a001", 3.0), ("a002", 5.0)):
        _write_complex(structures, sample_id, distance)
    (index / "2016_INDEX_refined.data").write_text(
        "a001 2.0 2000 5.0 Kd=1nM\na002 2.0 2000 6.0 Kd=1nM\n",
        encoding="utf-8",
    )
    (index / "train_data_2016.txt").write_text("['a001', 'a002']", encoding="utf-8")
    (index / "test_data_2016.txt").write_text("[]", encoding="utf-8")
    config = tmp_path / "task.yaml"
    config.write_text(
        "task_id: toy\n"
        "system_type: protein_ligand\n"
        "data_audit:\n"
        "  casf:\n"
        "    year: 2016\n"
        f"    index_root: {index}\n"
        f"    structures_root: {structures}\n"
        "representation_design:\n"
        "  support:\n"
        "    min_element_support_samples: 1\n"
        "    min_element_support_fraction: 0.0\n"
        "    min_pair_support_samples: 1\n"
        "    min_pair_support_fraction: 0.0\n"
        "    max_element_pair_channels: 50\n"
        "    small_molecule_self_pairs: false\n"
        "  filtration:\n"
        "    local_distance_quantile: 0.5\n"
        "    dataset_distance_quantile: 0.5\n"
        "    margin_factor: 1.0\n"
        "    default_step_angstrom: 0.1\n"
        "    allowed_steps_angstrom: [0.1, 0.2, 0.5, 1.0]\n"
        "    max_points: 200\n"
        "    include_zero: true\n",
        encoding="utf-8",
    )
    output = tmp_path / "representation.json"

    code = main(["--config", str(config), "--split", "train", "--output", str(output)])

    report = json.loads(output.read_text(encoding="utf-8"))
    spec = report["representation_spec"]
    assert code == 0
    assert report["run_kind"] == "full_modeling_pool_design"
    assert report["sample_count"] == 2
    assert len(report["design_input_hash"]) == 16
    assert spec["mode"] == "dataset_adaptive"
    assert spec["frozen"] is True
    assert spec["pair_order"] == [["C", "N"]]
    assert spec["filtration_profiles"]["PH"] == spec["filtration_profiles"]["PL"]
    assert spec["filtration_profiles"]["PH"]["stop"] == 4.0


def test_representation_design_input_hash_changes_with_filtration_policy():
    baseline = {
        "task_id": "same-data",
        "system_type": "protein_ligand",
        "representation_mode": "dataset_adaptive",
        "representation_design": {
            "filtration": {"dataset_distance_quantile": 0.95}
        },
    }
    extended = {
        **baseline,
        "representation_design": {
            "filtration": {"dataset_distance_quantile": 0.99}
        },
    }

    baseline_hash = representation_design_input_hash(
        baseline,
        data_audit_content_sha256="audit-sha",
        split="train",
        offset=0,
        limit=None,
    )
    extended_hash = representation_design_input_hash(
        extended,
        data_audit_content_sha256="audit-sha",
        split="train",
        offset=0,
        limit=None,
    )

    assert baseline_hash != extended_hash


def test_design_representation_cli_accepts_legacy_submitted_input_hash(tmp_path):
    structures = tmp_path / "structures"
    index = tmp_path / "index"
    index.mkdir()
    for sample_id, distance in (("a001", 3.0), ("a002", 5.0)):
        _write_complex(structures, sample_id, distance)
    (index / "2016_INDEX_refined.data").write_text(
        "a001 2.0 2000 5.0 Kd=1nM\na002 2.0 2000 6.0 Kd=1nM\n",
        encoding="utf-8",
    )
    (index / "train_data_2016.txt").write_text("['a001', 'a002']", encoding="utf-8")
    (index / "test_data_2016.txt").write_text("[]", encoding="utf-8")
    config = tmp_path / "task.yaml"
    config.write_text(
        "task_id: toy\n"
        "system_type: protein_ligand\n"
        "representation_mode: dataset_adaptive\n"
        "hydrogen_request:\n"
        "  mode: exclude\n"
        "data_audit:\n"
        "  tolerated_out_of_schema:\n"
        "    protein: [H]\n"
        "  casf:\n"
        "    year: 2016\n"
        f"    index_root: {index}\n"
        f"    structures_root: {structures}\n"
        "representation_design:\n"
        "  support:\n"
        "    min_element_support_samples: 1\n"
        "    min_element_support_fraction: 0.0\n"
        "    min_pair_support_samples: 1\n"
        "    min_pair_support_fraction: 0.0\n"
        "    max_element_pair_channels: 50\n"
        "    small_molecule_self_pairs: false\n"
        "  filtration:\n"
        "    local_distance_quantile: 0.5\n"
        "    dataset_distance_quantile: 0.5\n"
        "    margin_factor: 1.0\n"
        "    default_step_angstrom: 0.1\n"
        "    allowed_steps_angstrom: [0.1, 0.2, 0.5, 1.0]\n"
        "    max_points: 200\n"
        "    include_zero: true\n",
        encoding="utf-8",
    )
    task = {
        "task_id": "toy",
        "system_type": "protein_ligand",
        "representation_mode": "dataset_adaptive",
        "hydrogen_request": {"mode": "exclude"},
        "data_audit": {
            "tolerated_out_of_schema": {"protein": ["H"]},
            "casf": {
                "year": 2016,
                "index_root": str(index),
                "structures_root": str(structures),
            },
        },
        "representation_design": {
            "support": {
                "min_element_support_samples": 1,
                "min_element_support_fraction": 0.0,
                "min_pair_support_samples": 1,
                "min_pair_support_fraction": 0.0,
                "max_element_pair_channels": 50,
                "small_molecule_self_pairs": False,
            },
            "filtration": {
                "local_distance_quantile": 0.5,
                "dataset_distance_quantile": 0.5,
                "margin_factor": 1.0,
                "default_step_angstrom": 0.1,
                "allowed_steps_angstrom": [0.1, 0.2, 0.5, 1.0],
                "max_points": 200,
                "include_zero": True,
            },
        },
    }
    legacy_hash = _representation_design_input_hash_without_policy_extensions(
        task,
        data_audit_content_sha256=None,
        split="train",
        offset=0,
        limit=None,
    )
    current_hash = representation_design_input_hash(
        task,
        data_audit_content_sha256=None,
        split="train",
        offset=0,
        limit=None,
    )
    output = tmp_path / "representation.json"

    code = main(
        [
            "--config",
            str(config),
            "--split",
            "train",
            "--expected-input-hash",
            legacy_hash,
            "--output",
            str(output),
        ]
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert code == 0
    assert legacy_hash != current_hash
    assert report["design_input_hash"] == current_hash
