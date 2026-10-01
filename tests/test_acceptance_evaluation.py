from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from mint_scout.evaluate_acceptance_gbt import (
    evaluate_acceptance_gbt,
    evaluate_progressive_acceptance_gbt,
)
from mint_scout.execution.artifacts import ScoutExecutionArtifact
from mint_scout.invariants.plbind_tools import expected_feature_shape
from mint_scout.models.gbt import GBTConfig
from mint_scout.representation import make_legacy_casf_representation_spec


def test_acceptance_evaluation_uses_three_run_consensus_without_full_train_cv(tmp_path):
    representation = make_legacy_casf_representation_spec()
    representation_path = tmp_path / "representation.json"
    representation.write(representation_path)
    rows = {
        "train": [(f"train-{index}", float(index)) for index in range(6)],
        "test": [(f"test-{index}", float(index + 1)) for index in range(3)],
    }
    dataset_rows = ["sample_id,target,split,protein_path,ligand_path"]
    for split, samples in rows.items():
        for sample_id, target in samples:
            protein = tmp_path / f"{sample_id}.pdb"
            ligand = tmp_path / f"{sample_id}.mol2"
            protein.write_text("ATOM\n", encoding="utf-8")
            ligand.write_text("@<TRIPOS>MOLECULE\n", encoding="utf-8")
            dataset_rows.append(f"{sample_id},{target},{split},{protein},{ligand}")
    dataset_manifest = tmp_path / "dataset.csv"
    dataset_manifest.write_text("\n".join(dataset_rows) + "\n", encoding="utf-8")
    task = tmp_path / "task.yaml"
    task.write_text(
        "task_id: acceptance-toy\n"
        "system_type: protein_ligand\n"
        "task_type: regression\n"
        "primary_metric: pcc\n"
        "dataset_manifest:\n"
        f"  path: {dataset_manifest}\n",
        encoding="utf-8",
    )
    gbt = GBTConfig(
        n_estimators=3,
        n_runs=3,
        max_depth=2,
        min_samples_split=2,
        learning_rate=0.1,
        subsample=1.0,
        max_features="sqrt",
        random_state=42,
    )
    gbt_path = tmp_path / "gbt.yaml"
    gbt_path.write_text(
        "n_estimators: 3\nn_runs: 3\nmax_depth: 2\nmin_samples_split: 2\n"
        "learning_rate: 0.1\nsubsample: 1.0\nmax_features: sqrt\n"
        "random_state: 42\nnormalize: StandardScaler\n",
        encoding="utf-8",
    )
    scout = tmp_path / "scout.json"
    ScoutExecutionArtifact(
        artifact_version="mint-agent.scout-execution.v1",
        modeling_sample_ids=tuple(sample_id for sample_id, _ in rows["train"]),
        representation_hash=representation.spec_hash,
        frozen_priority_order=(("PL",),),
        full_fold_assignment={
            sample_id: index % 2
            for index, (sample_id, _) in enumerate(rows["train"])
        },
        target_metric="PCC",
        target_direction="higher",
        target_value=-1.0,
        target_source="user",
        gbt_parameter_hash=gbt.parameter_hash,
        probe_hash="probe-hash",
    ).write(scout)

    manifests = {}
    qcs = {}
    shape = expected_feature_shape("PL", representation)
    for split, samples in rows.items():
        manifest = tmp_path / f"{split}-PL.jsonl"
        entries = []
        for index, (sample_id, target) in enumerate(samples):
            output = tmp_path / f"repr-{representation.spec_hash}" / split / f"{sample_id}.npy"
            output.parent.mkdir(parents=True, exist_ok=True)
            values = np.full(shape, target, dtype=np.float32)
            values.reshape(-1)[:3] = (target, target * target, index)
            np.save(output, values, allow_pickle=False)
            entries.append(
                json.dumps(
                    {
                        "sample_id": sample_id,
                        "split": split,
                        "invariant": "PL",
                        "status": "computed",
                        "output_path": str(output),
                    }
                )
            )
        manifest.write_text("\n".join(entries) + "\n", encoding="utf-8")
        manifests[split] = manifest
        qc = tmp_path / f"{split}-qc.json"
        qc.write_text(
            json.dumps(
                {
                    "report_schema": "mint-agent.feature-qc.v1",
                    "status": "PASS",
                    "representation_hash": representation.spec_hash,
                    "sample_ids": [sample_id for sample_id, _ in samples],
                    "feature_manifests": {"PL": str(manifest)},
                }
            ),
            encoding="utf-8",
        )
        qcs[split] = qc

    report = evaluate_acceptance_gbt(
        task_config_path=task,
        representation_path=representation_path,
        gbt_config_path=gbt_path,
        scout_artifact_path=scout,
        candidate_rank=1,
        train_manifest_values=[f"PL={manifests['train']}"],
        evaluation_manifest_values=[f"PL={manifests['test']}"],
        train_qc_report=qcs["train"],
        evaluation_qc_report=qcs["test"],
    )

    assert report["status"] == "TARGET_REACHED"
    assert report["gbt_config"]["n_runs"] == 3
    assert report["run_aggregation"] == "mean_predictions_then_compute_metric"
    assert report["protocol"]["full_train_cross_validation"] is False
    assert report["protocol"]["evaluation_used_for_candidate_selection"] is True
    assert report["protocol"]["independent_test_available"] is False


def test_progressive_acceptance_scores_all_available_subsets_and_reuses_predictions(
    tmp_path, monkeypatch
):
    representation = make_legacy_casf_representation_spec()
    representation_path = tmp_path / "representation.json"
    representation.write(representation_path)
    rows = {
        "train": [(f"train-{index}", float(index)) for index in range(6)],
        "test": [(f"test-{index}", float(index + 1)) for index in range(3)],
    }
    dataset_rows = ["sample_id,target,split,protein_path,ligand_path"]
    for split, samples in rows.items():
        for sample_id, target in samples:
            protein = tmp_path / f"{sample_id}.pdb"
            ligand = tmp_path / f"{sample_id}.mol2"
            protein.write_text("ATOM\n", encoding="utf-8")
            ligand.write_text("@<TRIPOS>MOLECULE\n", encoding="utf-8")
            dataset_rows.append(f"{sample_id},{target},{split},{protein},{ligand}")
    dataset_manifest = tmp_path / "dataset.csv"
    dataset_manifest.write_text("\n".join(dataset_rows) + "\n", encoding="utf-8")
    task = tmp_path / "task.yaml"
    task.write_text(
        "task_id: progressive-acceptance-toy\n"
        "system_type: protein_ligand\n"
        "task_type: regression\n"
        "primary_metric: pcc\n"
        "selection_preferences:\n"
        "  selection_objective: satisfy_target\n"
        "dataset_manifest:\n"
        f"  path: {dataset_manifest}\n",
        encoding="utf-8",
    )
    gbt = GBTConfig(
        n_estimators=3,
        n_runs=3,
        max_depth=2,
        min_samples_split=2,
        learning_rate=0.1,
        subsample=1.0,
        max_features="sqrt",
        random_state=42,
    )
    gbt_path = tmp_path / "gbt.yaml"
    gbt_path.write_text(
        "n_estimators: 3\nn_runs: 3\nmax_depth: 2\nmin_samples_split: 2\n"
        "learning_rate: 0.1\nsubsample: 1.0\nmax_features: sqrt\n"
        "random_state: 42\nnormalize: StandardScaler\n",
        encoding="utf-8",
    )
    scout = tmp_path / "scout.json"
    ScoutExecutionArtifact(
        artifact_version="mint-agent.scout-execution.v1",
        modeling_sample_ids=tuple(sample_id for sample_id, _ in rows["train"]),
        representation_hash=representation.spec_hash,
        frozen_priority_order=(
            ("PL", "PH", "EIC"),
            ("PL", "PH"),
            ("PL", "EIC"),
            ("PH", "EIC"),
            ("PL",),
            ("PH",),
            ("EIC",),
        ),
        full_fold_assignment={
            sample_id: index % 2
            for index, (sample_id, _) in enumerate(rows["train"])
        },
        target_metric="PCC",
        target_direction="higher",
        target_value=2.0,
        target_source="user",
        gbt_parameter_hash=gbt.parameter_hash,
        probe_hash="probe-hash",
    ).write(scout)

    manifests: dict[str, dict[str, object]] = {"train": {}, "test": {}}
    for split, samples in rows.items():
        for invariant in ("PL", "PH", "EIC"):
            manifest = tmp_path / f"{split}-{invariant}.jsonl"
            entries = []
            shape = expected_feature_shape(invariant, representation)
            for index, (sample_id, target) in enumerate(samples):
                output = (
                    tmp_path
                    / f"repr-{representation.spec_hash}"
                    / split
                    / invariant
                    / f"{sample_id}.npy"
                )
                output.parent.mkdir(parents=True, exist_ok=True)
                values = np.full(shape, target + index, dtype=np.float32)
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
            manifest.write_text("\n".join(entries) + "\n", encoding="utf-8")
            manifests[split][invariant] = manifest

    def qc(split: str, invariants: tuple[str, ...]):
        path = tmp_path / f"{split}-{'-'.join(invariants)}-qc.json"
        path.write_text(
            json.dumps(
                {
                    "report_schema": "mint-agent.feature-qc.v1",
                    "status": "PASS",
                    "representation_hash": representation.spec_hash,
                    "sample_ids": [sample_id for sample_id, _ in rows[split]],
                    "feature_manifests": {
                        name: str(manifests[split][name]) for name in invariants
                    },
                }
            ),
            encoding="utf-8",
        )
        return path

    calls: list[int] = []

    def fake_fit_predict(x_train, y_train, x_evaluation, *, config):
        calls.append(x_train.shape[0])
        return np.arange(len(x_evaluation), dtype=float)

    monkeypatch.setattr(
        "mint_scout.evaluate_acceptance_gbt.fit_predict_gbt", fake_fit_predict
    )
    first = evaluate_progressive_acceptance_gbt(
        task_config_path=task,
        representation_path=representation_path,
        gbt_config_path=gbt_path,
        scout_artifact_path=scout,
        max_acquisitions=1,
        train_manifest_values=[f"PL={manifests['train']['PL']}"],
        evaluation_manifest_values=[f"PL={manifests['test']['PL']}"],
        train_qc_report=qc("train", ("PL",)),
        evaluation_qc_report=qc("test", ("PL",)),
    )
    first_path = tmp_path / "first.json"
    first_path.write_text(json.dumps(first), encoding="utf-8")
    second = evaluate_progressive_acceptance_gbt(
        task_config_path=task,
        representation_path=representation_path,
        gbt_config_path=gbt_path,
        scout_artifact_path=scout,
        max_acquisitions=3,
        train_manifest_values=[
            f"PL={manifests['train']['PL']}",
            f"PH={manifests['train']['PH']}",
            f"EIC={manifests['train']['EIC']}",
        ],
        evaluation_manifest_values=[
            f"PL={manifests['test']['PL']}",
            f"PH={manifests['test']['PH']}",
            f"EIC={manifests['test']['EIC']}",
        ],
        train_qc_report=qc("train", ("PL", "PH", "EIC")),
        evaluation_qc_report=qc("test", ("PL", "PH", "EIC")),
        prior_evaluation_report=first_path,
    )
    assert first["status"] == "ACQUISITION_LIMIT_REACHED"
    assert first["objective_complete"] is False
    assert first["prediction_reuse"]["newly_fitted_invariants"] == ["PL"]
    assert second["prediction_reuse"]["reused_invariants"] == ["PL"]
    assert second["prediction_reuse"]["newly_fitted_invariants"] == ["PH", "EIC"]
    assert {tuple(row["subset"]) for row in second["subset_scores"]} == {
        ("PL",),
        ("PH",),
        ("EIC",),
        ("PL", "PH"),
        ("PL", "EIC"),
        ("PH", "EIC"),
        ("PL", "PH", "EIC"),
    }
    assert len(calls) == 3


def test_maximize_rank1_does_not_stop_when_an_early_subset_reaches_target(
    tmp_path, monkeypatch
):
    representation = make_legacy_casf_representation_spec()
    representation_path = tmp_path / "representation.json"
    representation.write(representation_path)
    samples = {
        "train": [(f"train-{index}", float(index)) for index in range(6)],
        "test": [(f"test-{index}", float(index + 1)) for index in range(3)],
    }
    dataset_rows = ["sample_id,target,split,protein_path,ligand_path"]
    for split, rows in samples.items():
        for sample_id, target in rows:
            protein = tmp_path / f"{sample_id}.pdb"
            ligand = tmp_path / f"{sample_id}.mol2"
            protein.write_text("ATOM\n", encoding="utf-8")
            ligand.write_text("@<TRIPOS>MOLECULE\n", encoding="utf-8")
            dataset_rows.append(f"{sample_id},{target},{split},{protein},{ligand}")
    dataset_manifest = tmp_path / "dataset.csv"
    dataset_manifest.write_text("\n".join(dataset_rows) + "\n", encoding="utf-8")
    task = tmp_path / "task.yaml"
    task.write_text(
        "task_id: maximize-rank1-toy\n"
        "system_type: protein_ligand\n"
        "task_type: regression\n"
        "primary_metric: pcc\n"
        "selection_preferences:\n"
        "  selection_objective: maximize_rank1\n"
        "dataset_manifest:\n"
        f"  path: {dataset_manifest}\n",
        encoding="utf-8",
    )
    gbt = GBTConfig(
        n_estimators=3,
        n_runs=3,
        max_depth=2,
        min_samples_split=2,
        learning_rate=0.1,
        subsample=1.0,
        max_features="sqrt",
        random_state=42,
    )
    gbt_path = tmp_path / "gbt.yaml"
    gbt_path.write_text(
        "n_estimators: 3\nn_runs: 3\nmax_depth: 2\nmin_samples_split: 2\n"
        "learning_rate: 0.1\nsubsample: 1.0\nmax_features: sqrt\n"
        "random_state: 42\nnormalize: StandardScaler\n",
        encoding="utf-8",
    )
    scout_path = tmp_path / "scout.json"
    ScoutExecutionArtifact(
        artifact_version="mint-agent.scout-execution.v1",
        modeling_sample_ids=tuple(sample_id for sample_id, _ in samples["train"]),
        representation_hash=representation.spec_hash,
        frozen_priority_order=(("PL", "PH"), ("PL",), ("PH",)),
        full_fold_assignment={
            sample_id: index % 2
            for index, (sample_id, _) in enumerate(samples["train"])
        },
        target_metric="PCC",
        target_direction="higher",
        target_value=0.5,
        target_source="user",
        gbt_parameter_hash=gbt.parameter_hash,
        probe_hash="probe-hash",
    ).write(scout_path)

    manifests: dict[str, dict[str, Path]] = {"train": {}, "test": {}}
    for split, rows in samples.items():
        for invariant in ("PL", "PH"):
            manifest = tmp_path / f"{split}-{invariant}.jsonl"
            entries = []
            shape = expected_feature_shape(invariant, representation)
            for index, (sample_id, target) in enumerate(rows):
                output = (
                    tmp_path
                    / f"repr-{representation.spec_hash}"
                    / split
                    / invariant
                    / f"{sample_id}.npy"
                )
                output.parent.mkdir(parents=True, exist_ok=True)
                np.save(output, np.full(shape, target + index, dtype=np.float32))
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
            manifest.write_text("\n".join(entries) + "\n", encoding="utf-8")
            manifests[split][invariant] = manifest

    def qc(split: str, invariants: tuple[str, ...]) -> Path:
        path = tmp_path / f"{split}-{'-'.join(invariants)}-qc.json"
        path.write_text(
            json.dumps(
                {
                    "report_schema": "mint-agent.feature-qc.v1",
                    "status": "PASS",
                    "representation_hash": representation.spec_hash,
                    "sample_ids": [sample_id for sample_id, _ in samples[split]],
                    "feature_manifests": {
                        name: str(manifests[split][name]) for name in invariants
                    },
                }
            ),
            encoding="utf-8",
        )
        return path

    calls: list[int] = []

    def fake_fit_predict(x_train, y_train, x_evaluation, *, config):
        calls.append(x_train.shape[0])
        return np.arange(len(x_evaluation), dtype=float)

    monkeypatch.setattr(
        "mint_scout.evaluate_acceptance_gbt.fit_predict_gbt", fake_fit_predict
    )
    first = evaluate_progressive_acceptance_gbt(
        task_config_path=task,
        representation_path=representation_path,
        gbt_config_path=gbt_path,
        scout_artifact_path=scout_path,
        max_acquisitions=1,
        train_manifest_values=[f"PL={manifests['train']['PL']}"],
        evaluation_manifest_values=[f"PL={manifests['test']['PL']}"],
        train_qc_report=qc("train", ("PL",)),
        evaluation_qc_report=qc("test", ("PL",)),
    )
    first_path = tmp_path / "first.json"
    first_path.write_text(json.dumps(first), encoding="utf-8")
    second = evaluate_progressive_acceptance_gbt(
        task_config_path=task,
        representation_path=representation_path,
        gbt_config_path=gbt_path,
        scout_artifact_path=scout_path,
        max_acquisitions=2,
        train_manifest_values=[
            f"PL={manifests['train']['PL']}",
            f"PH={manifests['train']['PH']}",
        ],
        evaluation_manifest_values=[
            f"PL={manifests['test']['PL']}",
            f"PH={manifests['test']['PH']}",
        ],
        train_qc_report=qc("train", ("PL", "PH")),
        evaluation_qc_report=qc("test", ("PL", "PH")),
        prior_evaluation_report=first_path,
    )
    direct = evaluate_progressive_acceptance_gbt(
        task_config_path=task,
        representation_path=representation_path,
        gbt_config_path=gbt_path,
        scout_artifact_path=scout_path,
        max_acquisitions=2,
        train_manifest_values=[
            f"PL={manifests['train']['PL']}",
            f"PH={manifests['train']['PH']}",
        ],
        evaluation_manifest_values=[
            f"PL={manifests['test']['PL']}",
            f"PH={manifests['test']['PH']}",
        ],
        train_qc_report=qc("train", ("PL", "PH")),
        evaluation_qc_report=qc("test", ("PL", "PH")),
    )

    assert first["target_reached"] is True
    assert first["objective_complete"] is False
    assert first["status"] == "ACQUISITION_LIMIT_REACHED"
    assert second["objective_complete"] is True
    assert second["status"] == "MAXIMIZATION_COMPLETE"
    assert {tuple(row["subset"]) for row in second["subset_scores"]} == {
        ("PL",),
        ("PH",),
        ("PL", "PH"),
    }
    assert direct["status"] == "MAXIMIZATION_COMPLETE"
    assert direct["prediction_reuse"]["reused_invariants"] == []
    assert direct["prediction_reuse"]["newly_fitted_invariants"] == ["PL", "PH"]
    assert len(calls) == 4


def test_maximize_top_k_scores_only_frozen_top_k_candidates_directly(
    tmp_path, monkeypatch
):
    representation = make_legacy_casf_representation_spec()
    representation_path = tmp_path / "representation.json"
    representation.write(representation_path)
    samples = {
        "train": [(f"train-{index}", float(index)) for index in range(6)],
        "test": [(f"test-{index}", float(index + 1)) for index in range(3)],
    }
    dataset_rows = ["sample_id,target,split,protein_path,ligand_path"]
    for split, rows in samples.items():
        for sample_id, target in rows:
            protein = tmp_path / f"{sample_id}.pdb"
            ligand = tmp_path / f"{sample_id}.mol2"
            protein.write_text("ATOM\n", encoding="utf-8")
            ligand.write_text("@<TRIPOS>MOLECULE\n", encoding="utf-8")
            dataset_rows.append(f"{sample_id},{target},{split},{protein},{ligand}")
    dataset_manifest = tmp_path / "dataset.csv"
    dataset_manifest.write_text("\n".join(dataset_rows) + "\n", encoding="utf-8")
    task = tmp_path / "task.yaml"
    task.write_text(
        "task_id: maximize-top-k-toy\n"
        "system_type: protein_ligand\n"
        "task_type: regression\n"
        "primary_metric: pcc\n"
        "selection_preferences:\n"
        "  selection_objective: maximize_top_k\n"
        "  max_candidate_rank: 2\n"
        "dataset_manifest:\n"
        f"  path: {dataset_manifest}\n",
        encoding="utf-8",
    )
    gbt = GBTConfig(
        n_estimators=3,
        n_runs=3,
        max_depth=2,
        min_samples_split=2,
        learning_rate=0.1,
        subsample=1.0,
        max_features="sqrt",
        random_state=42,
    )
    gbt_path = tmp_path / "gbt.yaml"
    gbt_path.write_text(
        "n_estimators: 3\nn_runs: 3\nmax_depth: 2\nmin_samples_split: 2\n"
        "learning_rate: 0.1\nsubsample: 1.0\nmax_features: sqrt\n"
        "random_state: 42\nnormalize: StandardScaler\n",
        encoding="utf-8",
    )
    priority_order = (
        ("PL", "PH"),
        ("EIC",),
        ("PL",),
        ("PH",),
        ("PL", "EIC"),
        ("PH", "EIC"),
        ("PL", "PH", "EIC"),
    )
    scout_path = tmp_path / "scout.json"
    ScoutExecutionArtifact(
        artifact_version="mint-agent.scout-execution.v1",
        modeling_sample_ids=tuple(sample_id for sample_id, _ in samples["train"]),
        representation_hash=representation.spec_hash,
        frozen_priority_order=priority_order,
        full_fold_assignment={
            sample_id: index % 2
            for index, (sample_id, _) in enumerate(samples["train"])
        },
        target_metric="PCC",
        target_direction="higher",
        target_value=0.5,
        target_source="user",
        gbt_parameter_hash=gbt.parameter_hash,
        probe_hash="probe-hash",
    ).write(scout_path)

    manifests: dict[str, dict[str, Path]] = {"train": {}, "test": {}}
    for split, rows in samples.items():
        for invariant in ("PL", "PH", "EIC"):
            manifest = tmp_path / f"{split}-{invariant}.jsonl"
            entries = []
            shape = expected_feature_shape(invariant, representation)
            for index, (sample_id, target) in enumerate(rows):
                output = (
                    tmp_path
                    / f"repr-{representation.spec_hash}"
                    / split
                    / invariant
                    / f"{sample_id}.npy"
                )
                output.parent.mkdir(parents=True, exist_ok=True)
                np.save(output, np.full(shape, target + index, dtype=np.float32))
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
            manifest.write_text("\n".join(entries) + "\n", encoding="utf-8")
            manifests[split][invariant] = manifest

    def qc(split: str, invariants: tuple[str, ...]) -> Path:
        path = tmp_path / f"{split}-{'-'.join(invariants)}-qc.json"
        path.write_text(
            json.dumps(
                {
                    "report_schema": "mint-agent.feature-qc.v1",
                    "status": "PASS",
                    "representation_hash": representation.spec_hash,
                    "sample_ids": [sample_id for sample_id, _ in samples[split]],
                    "feature_manifests": {
                        name: str(manifests[split][name]) for name in invariants
                    },
                }
            ),
            encoding="utf-8",
        )
        return path

    calls: list[int] = []

    def fake_fit_predict(x_train, y_train, x_evaluation, *, config):
        calls.append(x_train.shape[0])
        return np.arange(len(x_evaluation), dtype=float)

    monkeypatch.setattr(
        "mint_scout.evaluate_acceptance_gbt.fit_predict_gbt", fake_fit_predict
    )
    report = evaluate_progressive_acceptance_gbt(
        task_config_path=task,
        representation_path=representation_path,
        gbt_config_path=gbt_path,
        scout_artifact_path=scout_path,
        max_acquisitions=3,
        train_manifest_values=[
            f"PL={manifests['train']['PL']}",
            f"PH={manifests['train']['PH']}",
            f"EIC={manifests['train']['EIC']}",
        ],
        evaluation_manifest_values=[
            f"PL={manifests['test']['PL']}",
            f"PH={manifests['test']['PH']}",
            f"EIC={manifests['test']['EIC']}",
        ],
        train_qc_report=qc("train", ("PL", "PH", "EIC")),
        evaluation_qc_report=qc("test", ("PL", "PH", "EIC")),
    )

    assert report["status"] == "MAXIMIZATION_COMPLETE"
    assert report["selection_objective"] == "maximize_top_k"
    assert report["candidate_rank_limit"] == 2
    assert report["candidate_pool"] == [["PL", "PH"], ["EIC"]]
    assert report["minimum_acquisitions_for_objective"] == 3
    assert {tuple(row["subset"]) for row in report["subset_scores"]} == {
        ("PL", "PH"),
        ("EIC",),
    }
    assert report["protocol"]["all_available_subsets_scored"] is False
    assert report["prediction_reuse"]["newly_fitted_invariants"] == [
        "PL",
        "PH",
        "EIC",
    ]
    assert len(calls) == 3
