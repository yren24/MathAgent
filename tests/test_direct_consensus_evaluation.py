from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from mint_scout.evaluate_direct_consensus_gbt import evaluate_direct_test_consensus_gbt
from mint_scout.invariants.plbind_tools import expected_feature_shape
from mint_scout.representation import make_legacy_casf_representation_spec


def test_direct_consensus_reports_every_subset_without_test_selection(tmp_path, monkeypatch):
    representation = make_legacy_casf_representation_spec()
    representation_path = tmp_path / "representation.json"
    representation.write(representation_path)
    rows = {
        "train": [(f"train-{index}", float(index)) for index in range(6)],
        "test": [(f"test-{index}", float(index + 1)) for index in range(4)],
    }
    manifest_rows = ["sample_id,target,split,protein_path,ligand_path"]
    for split, samples in rows.items():
        for sample_id, target in samples:
            protein = tmp_path / f"{sample_id}.pdb"
            ligand = tmp_path / f"{sample_id}.mol2"
            protein.write_text("ATOM\n", encoding="utf-8")
            ligand.write_text("@<TRIPOS>MOLECULE\n", encoding="utf-8")
            manifest_rows.append(f"{sample_id},{target},{split},{protein},{ligand}")
    dataset_manifest = tmp_path / "dataset.csv"
    dataset_manifest.write_text("\n".join(manifest_rows) + "\n", encoding="utf-8")
    task = tmp_path / "task.yaml"
    task.write_text(
        "task_id: direct-consensus-toy\n"
        "system_type: protein_ligand\n"
        "task_type: regression\n"
        "primary_metric: pcc\n"
        "dataset_manifest:\n"
        f"  path: {dataset_manifest}\n",
        encoding="utf-8",
    )
    gbt = tmp_path / "gbt.yaml"
    gbt.write_text(
        "n_estimators: 3\nn_runs: 1\nmax_depth: 2\nmin_samples_split: 2\n"
        "learning_rate: 0.1\nsubsample: 1.0\nmax_features: sqrt\n"
        "random_state: 42\nnormalize: StandardScaler\n",
        encoding="utf-8",
    )

    manifests: dict[str, dict[str, Path]] = {"train": {}, "test": {}}
    for split, samples in rows.items():
        for invariant, offset in (("PL", 0.0), ("PH", 10.0)):
            entries = []
            shape = expected_feature_shape(invariant, representation)
            for sample_id, target in samples:
                output = (
                    tmp_path
                    / f"repr-{representation.spec_hash}"
                    / split
                    / invariant
                    / f"{sample_id}.npy"
                )
                output.parent.mkdir(parents=True, exist_ok=True)
                values = np.zeros(shape, dtype=np.float32)
                values.reshape(-1)[0] = target + offset
                np.save(output, values, allow_pickle=False)
                entries.append(
                    json.dumps(
                        {
                            "sample_id": sample_id,
                            "split": split,
                            "invariant": invariant,
                            "status": "computed",
                            "output_path": str(output),
                        }
                    )
                )
            path = tmp_path / f"{split}-{invariant}.jsonl"
            path.write_text("\n".join(entries) + "\n", encoding="utf-8")
            manifests[split][invariant] = path

    fitted_markers = []

    def fake_fit_predict(train_features, y_train, test_features, *, config):
        assert len(train_features) == len(y_train) == len(rows["train"])
        fitted_markers.append(float(train_features.reshape(len(train_features), -1)[0, 0]))
        return test_features.reshape(len(test_features), -1)[:, 0]

    monkeypatch.setattr(
        "mint_scout.evaluate_direct_consensus_gbt.fit_predict_gbt", fake_fit_predict
    )
    partial = evaluate_direct_test_consensus_gbt(
        task_config_path=task,
        representation_path=representation_path,
        gbt_config_path=gbt,
        train_manifest_values=[f"PL={manifests['train']['PL']}"],
        test_manifest_values=[f"PL={manifests['test']['PL']}"],
    )
    prior_report = tmp_path / "partial.json"
    prior_report.write_text(json.dumps(partial), encoding="utf-8")
    report = evaluate_direct_test_consensus_gbt(
        task_config_path=task,
        representation_path=representation_path,
        gbt_config_path=gbt,
        train_manifest_values=[
            f"PL={manifests['train']['PL']}",
            f"PH={manifests['train']['PH']}",
        ],
        test_manifest_values=[
            f"PL={manifests['test']['PL']}",
            f"PH={manifests['test']['PH']}",
        ],
        prior_report_paths=[prior_report],
    )

    assert report["evaluation_mode"] == "direct_train_test_consensus_posthoc"
    assert report["selection_prohibited"] is True
    assert report["protocol"]["full_train_cross_validation"] is False
    assert report["protocol"]["test_used_for_candidate_selection"] is False
    assert report["reused_prior_prediction_invariants"] == ["PL"]
    assert fitted_markers == [0.0, 10.0]
    assert {tuple(row["subset"]) for row in report["subset_metrics"]} == {
        ("PH",),
        ("PL",),
        ("PH", "PL"),
    }
    assert set(report["test_predictions"]) == {"PH", "PL"}
