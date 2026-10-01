from __future__ import annotations

import json

import numpy as np
import pytest

from mint_scout.data.casf_index import CasfRecord
from mint_scout.evaluate_split_gbt import (
    _assert_disjoint_splits,
    evaluate_split_gbt,
    select_validation_subset,
)
from mint_scout.invariants.plbind_tools import expected_feature_shape
from mint_scout.representation import make_legacy_casf_representation_spec


def test_validation_selection_finds_complementary_pair():
    y_true = np.asarray([0.0, 1.0, 2.0, 3.0, 4.0])
    predictions = {
        "PH": np.asarray([0.0, 0.4, 2.4, 2.6, 4.0]),
        "PL": np.asarray([0.0, 1.6, 1.6, 3.4, 4.0]),
    }

    selected, scores = select_validation_subset(
        y_true, predictions, metric_name="RMSE"
    )

    assert selected == ("PH", "PL")
    assert scores[selected] == pytest.approx(0.0)


def test_validation_selection_prefers_simpler_subset_on_exact_tie():
    y_true = np.asarray([0.0, 1.0, 2.0])
    predictions = {
        "CA": y_true.copy(),
        "PL": y_true.copy(),
    }

    selected, _ = select_validation_subset(y_true, predictions, metric_name="PCC")

    assert selected == ("CA",)


def test_split_protocol_rejects_case_insensitive_id_overlap():
    train = (CasfRecord("1abc", 1.0, "train"),)
    validation = (CasfRecord("1ABC", 2.0, "validation"),)

    with pytest.raises(ValueError, match="must be disjoint"):
        _assert_disjoint_splits(train, validation)


def test_split_evaluator_runs_validation_selection_then_final_test(tmp_path):
    representation = make_legacy_casf_representation_spec()
    representation_path = tmp_path / "representation.json"
    representation.write(representation_path)
    split_rows = {
        "train": [(f"train-{index}", float(index)) for index in range(8)],
        "validation": [(f"val-{index}", float(index + 1)) for index in range(4)],
        "test": [(f"test-{index}", float(index + 2)) for index in range(4)],
    }
    manifest_lines = ["sample_id,target,split,protein_path,ligand_path"]
    for split, rows in split_rows.items():
        for sample_id, target in rows:
            protein = tmp_path / f"{sample_id}.pdb"
            ligand = tmp_path / f"{sample_id}.mol2"
            protein.write_text("ATOM\n", encoding="utf-8")
            ligand.write_text("@<TRIPOS>MOLECULE\n", encoding="utf-8")
            manifest_lines.append(
                f"{sample_id},{target},{split},{protein},{ligand}"
            )
    dataset_manifest = tmp_path / "dataset.csv"
    dataset_manifest.write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
    task_config = tmp_path / "task.yaml"
    task_config.write_text(
        "task_id: split-toy\n"
        "system_type: protein_ligand\n"
        "task_type: regression\n"
        "primary_metric: pcc\n"
        "dataset_manifest:\n"
        f"  path: {dataset_manifest}\n",
        encoding="utf-8",
    )
    gbt_config = tmp_path / "gbt.yaml"
    gbt_config.write_text(
        "n_estimators: 5\n"
        "n_runs: 1\n"
        "max_depth: 2\n"
        "min_samples_split: 2\n"
        "learning_rate: 0.1\n"
        "subsample: 1.0\n"
        "max_features: sqrt\n"
        "random_state: 42\n"
        "normalize: StandardScaler\n",
        encoding="utf-8",
    )

    manifest_values = {split: [] for split in split_rows}
    qc_paths = {}
    for split, rows in split_rows.items():
        feature_manifests = {}
        for invariant in ("CA", "PL"):
            feature_manifest = tmp_path / f"{split}-{invariant}.jsonl"
            entries = []
            shape = expected_feature_shape(invariant, representation)
            for row_index, (sample_id, target) in enumerate(rows):
                output = (
                    tmp_path
                    / f"repr-{representation.spec_hash}"
                    / invariant
                    / f"{sample_id}.npy"
                )
                output.parent.mkdir(parents=True, exist_ok=True)
                values = np.full(shape, target, dtype=np.float32)
                values.reshape(-1)[:3] = (target, target * target, row_index)
                np.save(output, values, allow_pickle=False)
                entries.append(
                    json.dumps(
                        {
                            "sample_id": sample_id,
                            "invariant": invariant,
                            "status": "computed",
                            "output_path": str(output),
                        }
                    )
                )
            feature_manifest.write_text("\n".join(entries) + "\n", encoding="utf-8")
            feature_manifests[invariant] = feature_manifest
            manifest_values[split].append(f"{invariant}={feature_manifest}")
        qc_path = tmp_path / f"{split}-qc.json"
        qc_path.write_text(
            json.dumps(
                {
                    "report_schema": "mint-agent.feature-qc.v1",
                    "status": "PASS",
                    "representation_hash": representation.spec_hash,
                    "sample_ids": [sample_id for sample_id, _ in rows],
                    "feature_manifests": {
                        name: str(path) for name, path in feature_manifests.items()
                    },
                }
            ),
            encoding="utf-8",
        )
        qc_paths[split] = qc_path

    report = evaluate_split_gbt(
        task_config_path=task_config,
        representation_path=representation_path,
        gbt_config_path=gbt_config,
        train_manifest_values=manifest_values["train"],
        validation_manifest_values=manifest_values["validation"],
        test_manifest_values=manifest_values["test"],
        train_qc_report=qc_paths["train"],
        validation_qc_report=qc_paths["validation"],
        test_qc_report=qc_paths["test"],
        n_bootstrap=0,
        bootstrap_seed=2026,
    )

    assert report["status"] == "COMPLETE"
    assert report["sample_counts"] == {"train": 8, "validation": 4, "test": 4}
    assert report["protocol"]["selection_frozen_before_test_scoring"] is True
    assert report["protocol"]["test_used_for_selection"] is False
    assert report["protocol"]["test_query_count"] == 1
    assert report["selection"]["selected_subset"]
    assert report["final_test"]["bootstrap_pcc_95"]["valid_replicates"] == 0
