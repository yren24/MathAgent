from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from mint_scout.evaluate_frozen_test_gbt import (
    _load_selection_report,
    evaluate_frozen_test_gbt,
)
from mint_scout.evaluate_validation_gbt import evaluate_validation_gbt
from mint_scout.evaluate_validation_gbt import _load_prior_predictions
from mint_scout.execution.artifacts import ScoutExecutionArtifact
from mint_scout.execution.jobs import inspect_validation_evaluation_report
from mint_scout.invariants.manifest import stable_hash
from mint_scout.invariants.plbind_tools import expected_feature_shape
from mint_scout.models.gbt import GBTConfig
from mint_scout.representation import make_legacy_casf_representation_spec


def test_validation_selection_then_frozen_test_uses_only_selected_features(
    tmp_path: Path,
):
    representation = make_legacy_casf_representation_spec()
    representation_path = tmp_path / "representation.json"
    representation.write(representation_path)
    rows = {
        "train": [(f"train-{index}", float(index)) for index in range(8)],
        "validation": [(f"val-{index}", float(index + 1)) for index in range(4)],
        "test": [(f"test-{index}", float(index + 2)) for index in range(4)],
    }
    manifest_lines = ["sample_id,target,split,protein_path,ligand_path"]
    for split, samples in rows.items():
        for sample_id, target in samples:
            protein = tmp_path / f"{sample_id}.pdb"
            ligand = tmp_path / f"{sample_id}.mol2"
            protein.write_text("ATOM\n", encoding="utf-8")
            ligand.write_text("@<TRIPOS>MOLECULE\n", encoding="utf-8")
            manifest_lines.append(f"{sample_id},{target},{split},{protein},{ligand}")
    dataset_manifest = tmp_path / "dataset.csv"
    dataset_manifest.write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
    task_config = tmp_path / "task.yaml"
    task_config.write_text(
        "task_id: split-toy\n"
        "system_type: protein_ligand\n"
        "task_type: regression\n"
        "primary_metric: pcc\n"
        "selection_preferences:\n"
        "  selection_objective: maximize_rank1\n"
        "dataset_manifest:\n"
        f"  path: {dataset_manifest}\n",
        encoding="utf-8",
    )
    gbt_config_path = tmp_path / "gbt.yaml"
    gbt_config_path.write_text(
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
    gbt_hash = GBTConfig(
        n_estimators=5,
        n_runs=1,
        max_depth=2,
        min_samples_split=2,
        learning_rate=0.1,
        subsample=1.0,
        max_features="sqrt",
        random_state=42,
        normalize="StandardScaler",
    ).parameter_hash
    scout_path = tmp_path / "scout.json"
    ScoutExecutionArtifact(
        artifact_version="mint-agent.scout-execution.v1",
        modeling_sample_ids=tuple(sample_id for sample_id, _ in rows["train"]),
        representation_hash=representation.spec_hash,
        frozen_priority_order=(("PL",),),
        full_fold_assignment={
            sample_id: index % 2 for index, (sample_id, _) in enumerate(rows["train"])
        },
        target_metric="PCC",
        target_direction="higher",
        target_value=-1.0,
        target_source="user",
        gbt_parameter_hash=gbt_hash,
        probe_hash="probe-hash",
    ).write(scout_path)

    manifests: dict[str, Path] = {}
    qcs: dict[str, Path] = {}
    shape = expected_feature_shape("PL", representation)
    for split, samples in rows.items():
        manifest = tmp_path / f"{split}-PL.jsonl"
        entries = []
        for index, (sample_id, target) in enumerate(samples):
            output = (
                tmp_path
                / f"repr-{representation.spec_hash}"
                / "PL"
                / f"{sample_id}.npy"
            )
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
                        "wall_seconds": 0.1,
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

    selection = evaluate_validation_gbt(
        task_config_path=task_config,
        representation_path=representation_path,
        gbt_config_path=gbt_config_path,
        scout_artifact_path=scout_path,
        train_manifest_values=[f"PL={manifests['train']}"],
        validation_manifest_values=[f"PL={manifests['validation']}"],
        train_qc_report=qcs["train"],
        validation_qc_report=qcs["validation"],
        max_acquisitions=1,
    )
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(json.dumps(selection), encoding="utf-8")
    inspected_selection = inspect_validation_evaluation_report(selection_path)
    final = evaluate_frozen_test_gbt(
        task_config_path=task_config,
        representation_path=representation_path,
        gbt_config_path=gbt_config_path,
        selection_report_path=selection_path,
        train_manifest_values=[f"PL={manifests['train']}"],
        validation_manifest_values=[f"PL={manifests['validation']}"],
        test_manifest_values=[f"PL={manifests['test']}"],
        train_qc_report=qcs["train"],
        validation_qc_report=qcs["validation"],
        test_qc_report=qcs["test"],
        n_bootstrap=0,
        bootstrap_seed=2026,
    )

    assert selection["status"] == "MAXIMIZATION_COMPLETE"
    assert selection["selection_objective"] == "maximize_rank1"
    assert selection["objective_complete"] is True
    assert inspected_selection["state"] == "COMPLETE"
    assert inspected_selection["report_status"] == "MAXIMIZATION_COMPLETE"
    assert inspected_selection["selection_objective"] == "maximize_rank1"
    assert selection["selection"]["selected_subset"] == ["PL"]
    assert selection["prediction_reuse"]["reused_invariants"] == []
    assert selection["prediction_reuse"]["newly_fitted_invariants"] == ["PL"]
    assert final["selected_subset"] == ["PL"]
    assert final["status"] == "TARGET_REACHED"
    assert final["target_reached"] is True
    assert final["protocol"]["test_used_for_selection"] is False
    assert final["protocol"]["test_query_count"] == 1
    assert final["protocol"]["refit_splits"] == ["train", "validation"]


def test_frozen_test_rejects_an_intermediate_acquisition_stage(tmp_path: Path):
    selection = tmp_path / "selection.json"
    selection.write_text(
        json.dumps(
            {"status": "ACQUISITION_LIMIT_REACHED", "evidence_scope": "validation",}
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="not a frozen terminal selection"):
        _load_selection_report(selection)


def test_later_validation_stage_loads_only_the_prior_prediction_prefix(tmp_path: Path):
    representation = make_legacy_casf_representation_spec()
    scout_path = tmp_path / "scout.json"
    train_ids = ("train-a", "train-b", "train-c")
    validation_ids = ("val-a", "val-b")
    prior_path = tmp_path / "validation-stage-1.json"
    prior_path.write_text(
        json.dumps(
            {
                "report_schema": "mint-agent.validation-gbt-evaluation.v1",
                "status": "ACQUISITION_LIMIT_REACHED",
                "invariants": ["PL"],
                "representation_hash": representation.spec_hash,
                "gbt_parameter_hash": "gbt-hash",
                "scout_artifact": str(scout_path),
                "target_metric": "PCC",
                "target_value": 0.8,
                "training_sample_order_hash": stable_hash(train_ids),
                "sample_ids": list(validation_ids),
                "validation_predictions": {"PL": {"val-a": 1.25, "val-b": 2.5}},
            }
        ),
        encoding="utf-8",
    )

    predictions, prior_sha256 = _load_prior_predictions(
        prior_path,
        acquired=("PL", "PH"),
        train_ids=train_ids,
        validation_ids=validation_ids,
        representation=representation,
        gbt_parameter_hash="gbt-hash",
        scout_artifact_path=scout_path,
        target_metric="PCC",
        target_value=0.8,
    )

    assert tuple(predictions) == ("PL",)
    assert predictions["PL"].tolist() == [1.25, 2.5]
    assert prior_sha256 is not None

    with pytest.raises(ValueError, match="selection objective does not match"):
        _load_prior_predictions(
            prior_path,
            acquired=("PL", "PH"),
            train_ids=train_ids,
            validation_ids=validation_ids,
            representation=representation,
            gbt_parameter_hash="gbt-hash",
            scout_artifact_path=scout_path,
            target_metric="PCC",
            target_value=0.8,
            selection_objective="maximize_rank1",
        )
