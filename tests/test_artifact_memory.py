from __future__ import annotations

import json
from dataclasses import asdict

from mint_scout.artifact_memory import ArtifactRegistry, inspect_artifact, main, register_batch
from mint_scout.models.gbt import GBTConfig


def test_register_and_query_full_train_model_evaluation(tmp_path):
    artifact = tmp_path / "phase4.json"
    artifact.write_text(
        json.dumps(
            {
                "report_schema": "mint-agent.phase4.casf.v1",
                "task_id": "casf2016",
                "run_kind": "full_execution",
                "split": "train",
                "evaluation_mode": "full_labeled_cv",
                "sample_ids": ["a", "b"],
                "representation_hash": "repr-1",
                "status": "TARGET_REACHED",
                "feature_manifests": {"PL": "/features/pl.jsonl"},
                "result": {
                    "selected_subset": ["PL"],
                    "acquisition_order": ["PL"],
                },
            }
        ),
        encoding="utf-8",
    )
    registry_path = tmp_path / "artifacts.sqlite"

    assert main(
        [
            "register",
            "--registry",
            str(registry_path),
            "--artifact",
            str(artifact),
        ]
    ) == 0
    records = ArtifactRegistry(registry_path).query(
        artifact_kind="model_evaluation",
        dataset_id="casf2016",
        evidence_scope="full_train",
        invariant="PL",
    )

    assert len(records) == 1
    assert records[0].sample_count == 2
    assert records[0].representation_hash == "repr-1"
    assert records[0].status == "TARGET_REACHED"


def test_feature_manifest_requires_explicit_scope_and_representation_override(tmp_path):
    manifest = tmp_path / "features.jsonl"
    manifest.write_text(
        "\n".join(
            json.dumps(
                {
                    "sample_id": sample_id,
                    "invariant": "PL",
                    "split": "train",
                    "status": "cached",
                }
            )
            for sample_id in ("a", "b", "c")
        )
        + "\n",
        encoding="utf-8",
    )
    registry_path = tmp_path / "artifacts.sqlite"

    assert main(
        [
            "register",
            "--registry",
            str(registry_path),
            "--artifact",
            str(manifest),
            "--dataset-id",
            "casf2016",
            "--evidence-scope",
            "full_train",
            "--representation-hash",
            "repr-1",
        ]
    ) == 0
    records = ArtifactRegistry(registry_path).query(
        artifact_kind="feature_manifest",
        evidence_scope="full_train",
        invariant="PL",
    )

    assert len(records) == 1
    assert records[0].sample_count == 3
    assert records[0].representation_hash == "repr-1"
    assert records[0].status == "COMPLETE"
    assert records[0].metadata == {
        "cached_count": 3,
        "computed_count": 0,
        "failed_count": 0,
    }


def test_reregistering_changed_path_replaces_stale_record(tmp_path):
    artifact = tmp_path / "audit.json"
    artifact.write_text(
        json.dumps(
            {
                "report_schema": "mint-agent.probe-representativeness-audit.v1",
                "probe_sample_count": 10,
                "status": "PASS",
            }
        ),
        encoding="utf-8",
    )
    registry = ArtifactRegistry(tmp_path / "artifacts.sqlite")
    first = inspect_artifact(artifact)
    registry.register(first)
    artifact.write_text(
        json.dumps(
            {
                "report_schema": "mint-agent.probe-representativeness-audit.v1",
                "probe_sample_count": 10,
                "status": "WARN",
            }
        ),
        encoding="utf-8",
    )
    second = inspect_artifact(artifact)
    registry.register(second)

    records = registry.query(artifact_kind="probe_audit")

    assert len(records) == 1
    assert records[0].artifact_id != first.artifact_id
    assert records[0].status == "WARN"


def test_query_missing_registry_is_empty(tmp_path):
    assert ArtifactRegistry(tmp_path / "missing.sqlite").query(invariant="PL") == ()


def test_report_with_invariant_list_is_inspected_as_agent_run(tmp_path):
    artifact = tmp_path / "agent-run.json"
    artifact.write_text(
        json.dumps(
            {
                "report_schema": "mint-agent.graph-run.v1",
                "dataset_id": "casf2016",
                "evidence_scope": "full_train",
                "invariants": ["PL"],
                "sample_count": 3772,
                "sample_order_hash": "order-hash",
                "status": "COMPLETE",
            }
        ),
        encoding="utf-8",
    )

    record = inspect_artifact(artifact)

    assert record.artifact_kind == "agent_run"
    assert record.invariants == ("PL",)
    assert record.sample_count == 3772
    assert record.sample_order_hash == "order-hash"


def test_filtration_comparison_is_registered_as_scientific_evidence(tmp_path):
    artifact = tmp_path / "filtration-comparison.json"
    artifact.write_text(
        json.dumps(
            {
                "report_schema": "mint-agent.filtration-comparison.v1",
                "dataset_id": "hiqbind",
                "evidence_scope": "probe",
                "sample_count": 50,
                "invariants": ["PH", "FPRC"],
                "status": "COMPLETE",
                "comparison_hash": "comparison-1",
            }
        ),
        encoding="utf-8",
    )

    record = inspect_artifact(artifact)

    assert record.artifact_kind == "filtration_comparison"
    assert record.invariants == ("FPRC", "PH")
    assert record.sample_count == 50
    assert record.metadata["comparison_hash"] == "comparison-1"


def test_split_gbt_report_registers_as_external_test_model_evaluation(tmp_path):
    artifact = tmp_path / "split-evaluation.json"
    artifact.write_text(
        json.dumps(
            {
                "report_schema": "mint-agent.split-gbt-evaluation.v1",
                "status": "COMPLETE",
                "dataset_id": "official-data",
                "evidence_scope": "external_test",
                "invariants": ["PH", "PL"],
                "feature_manifests": {
                    "train": {"PH": "/train/ph.jsonl", "PL": "/train/pl.jsonl"},
                    "validation": {
                        "PH": "/validation/ph.jsonl",
                        "PL": "/validation/pl.jsonl",
                    },
                    "test": {"PH": "/test/ph.jsonl", "PL": "/test/pl.jsonl"},
                },
                "representation_hash": "repr-1",
                "selection_hash": "selection-1",
                "sample_counts": {"train": 10, "validation": 4, "test": 5},
                "sample_order_hashes": {
                    "train": "train-order",
                    "validation": "validation-order",
                    "test": "test-order",
                },
                "gbt_parameter_hash": "gbt-1",
            }
        ),
        encoding="utf-8",
    )

    record = inspect_artifact(artifact)

    assert record.artifact_kind == "model_evaluation"
    assert record.evidence_scope == "external_test"
    assert record.split == "test"
    assert record.sample_count == 5
    assert record.sample_order_hash == "test-order"
    assert record.selection_hash == "selection-1"
    assert record.invariants == ("PH", "PL")


def test_outlier_diagnostic_preserves_qc_and_population_identity(tmp_path):
    artifact = tmp_path / "diagnostic.json"
    artifact.write_text(
        json.dumps(
            {
                "report_schema": "mint-agent.feature-outlier-diagnostic.v1",
                "status": "COMPLETE",
                "dataset_id": "casf2016",
                "evidence_scope": "probe",
                "invariant": "FPRC",
                "representation_hash": "repr-1",
                "selection_hash": "selection-1",
                "sample_ids": ["a", "b"],
                "diagnostic_hash": "diagnostic-1",
                "source": {
                    "feature_manifest": "/features/fprc.jsonl",
                    "feature_manifest_sha256": "manifest-sha",
                    "feature_qc_hash": "qc-1",
                },
            }
        ),
        encoding="utf-8",
    )

    record = inspect_artifact(artifact)

    assert record.artifact_kind == "feature_outlier_diagnostic"
    assert record.invariants == ("FPRC",)
    assert record.sample_count == 2
    assert record.selection_hash == "selection-1"
    assert record.metadata["feature_qc_hash"] == "qc-1"
    assert record.metadata["diagnostic_hash"] == "diagnostic-1"


def test_batch_registration_applies_explicit_evidence_metadata(tmp_path):
    manifest = tmp_path / "features.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "sample_id": "a",
                "invariant": "PL",
                "split": "train",
                "status": "cached",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    registry_path = tmp_path / "registry.sqlite"
    spec = tmp_path / "batch.yaml"
    spec.write_text(
        f"registry: {registry_path}\n"
        "artifacts:\n"
        f"  - path: {manifest}\n"
        "    dataset_id: casf2016\n"
        "    evidence_scope: full_train\n"
        "    representation_hash: repr-1\n",
        encoding="utf-8",
    )

    records = register_batch(spec)

    assert len(records) == 1
    assert records[0].dataset_id == "casf2016"
    assert records[0].evidence_scope == "full_train"
    assert records[0].representation_hash == "repr-1"


def test_legacy_scout_oof_reads_nested_probe_identity(tmp_path):
    artifact = tmp_path / "legacy-scout.json"
    artifact.write_text(
        json.dumps(
            {
                "report_schema": "mint-agent.casf-scout.v1",
                "task_id": "casf2016",
                "representation_hash": "repr-1",
                "selection_hash": "selection-1",
                "feature_manifests": {"PL": "/features/pl.jsonl"},
                "scout": {
                    "probe_sample_ids": ["a", "b"],
                    "gbt_parameter_hash": "gbt-1",
                    "probe_hash": "probe-1",
                },
            }
        ),
        encoding="utf-8",
    )

    record = inspect_artifact(artifact)

    assert record.artifact_kind == "scout_oof"
    assert record.sample_count == 2
    assert record.sample_order_hash is not None
    assert record.metadata["gbt_parameter_hash"] == "gbt-1"
    assert record.metadata["probe_hash"] == "probe-1"


def test_scout_execution_priority_is_available_in_artifact_memory(tmp_path):
    artifact = tmp_path / "scout-execution.json"
    artifact.write_text(
        json.dumps(
            {
                "artifact_version": "mint-agent.scout-execution.v1",
                "modeling_sample_ids": ["a", "b"],
                "representation_hash": "repr-1",
                "frozen_priority_order": [["PL"], ["PH", "PL"]],
                "full_fold_assignment": {"a": 0, "b": 1},
                "target_metric": "PCC",
                "target_direction": "higher",
                "target_value": 0.75,
                "target_source": "probe",
                "gbt_parameter_hash": "gbt-1",
                "probe_hash": "probe-1",
            }
        ),
        encoding="utf-8",
    )

    record = inspect_artifact(artifact)

    assert record.artifact_kind == "scout_execution_plan"
    assert record.invariants == ("PH", "PL")
    assert record.metadata["frozen_priority_order"] == [["PL"], ["PH", "PL"]]
    assert record.metadata["fold_assignment_hash"] is not None


def test_legacy_evaluation_derives_auditable_hash_from_embedded_gbt_config(tmp_path):
    artifact = tmp_path / "legacy-evaluation.json"
    config = GBTConfig()
    artifact.write_text(
        json.dumps(
            {
                "report_schema": "mint-agent.phase4.casf.v1",
                "task_id": "casf2016",
                "run_kind": "full_execution",
                "evidence_scope": "full_train",
                "sample_ids": ["a", "b"],
                "representation_hash": "repr-1",
                "target_metric": "PCC",
                "target_value": 0.75,
                "status": "TARGET_REACHED",
                "feature_manifests": {"PL": "/features/pl.jsonl"},
                "scout_artifact": "/runs/scout.json",
                "fold_source": "scout_artifact",
                "gbt_config": asdict(config),
                "result": {
                    "selected_subset": ["PL"],
                    "evaluation_fold_assignment": {"a": 0, "b": 1},
                },
            }
        ),
        encoding="utf-8",
    )

    record = inspect_artifact(artifact)

    assert record.metadata["gbt_parameter_hash"] == config.parameter_hash
    assert record.metadata["gbt_parameter_hash_source"] == (
        "derived_from_embedded_gbt_config"
    )
    assert record.metadata["fold_assignment_hash"] is not None
    assert record.metadata["scout_artifact"] == "/runs/scout.json"
    assert record.metadata["fold_source"] == "scout_artifact"


def test_external_gbt_evaluation_is_queryable_as_external_test_memory(tmp_path):
    artifact = tmp_path / "external-evaluation.json"
    artifact.write_text(
        json.dumps(
            {
                "report_schema": "mint-agent.external-gbt-evaluation.v1",
                "external_dataset_id": "bdb2020plus",
                "evidence_scope": "external_test",
                "evaluation_sample_count": 115,
                "evaluation_sample_order_hash": "external-order",
                "representation_hash": "repr-1",
                "invariants": ["PL"],
                "gbt_parameter_hash": "gbt-1",
                "evaluation_hash": "evaluation-1",
                "status": "COMPLETE",
            }
        ),
        encoding="utf-8",
    )

    record = inspect_artifact(artifact)

    assert record.artifact_kind == "model_evaluation"
    assert record.dataset_id == "bdb2020plus"
    assert record.split == "test"
    assert record.evidence_scope == "external_test"
    assert record.sample_count == 115
    assert record.sample_order_hash == "external-order"
    assert record.invariants == ("PL",)
    assert record.metadata["evaluation_hash"] == "evaluation-1"


def test_bdb_preparation_is_queryable_as_external_test_memory(tmp_path):
    artifact = tmp_path / "preparation.json"
    artifact.write_text(
        json.dumps(
            {
                "report_schema": "mint-agent.bdb2020plus-preparation.v1",
                "dataset_id": "bdb2020plus",
                "evidence_scope": "external_test",
                "sample_count": 115,
                "sample_order_hash": "external-order",
                "status": "PASS",
            }
        ),
        encoding="utf-8",
    )

    record = inspect_artifact(artifact)

    assert record.artifact_kind == "dataset_preparation"
    assert record.dataset_id == "bdb2020plus"
    assert record.split == "test"
    assert record.sample_count == 115
    assert record.sample_order_hash == "external-order"
