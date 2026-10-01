import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from mint_scout.execution.jobs import (
    _acceptance_evaluation_record_for_job,
    _assert_validation_evaluation_inputs,
    create_job_ledger,
    inspect_acceptance_evaluation_report,
    inspect_frozen_test_evaluation_report,
)
from mint_scout.execution.artifacts import ScoutExecutionArtifact
from mint_scout.invariants.manifest import stable_hash
from mint_scout.execution.profile import (
    AcceptanceEvaluationJobRequest,
    FrozenTestEvaluationJobRequest,
    ValidationEvaluationJobRequest,
    build_acceptance_evaluation_job_plan,
    build_frozen_test_evaluation_job_plan,
    build_validation_evaluation_job_plan,
    load_execution_profile,
)


def test_acceptance_plan_is_accepted_by_job_ledger():
    profile = load_execution_profile(Path("configs/execution/sapelo2.yaml"))
    plan = build_acceptance_evaluation_job_plan(
        profile,
        AcceptanceEvaluationJobRequest(
            dataset_id="toy",
            invariants=("EIC", "PL"),
            representation_hash="repr-1",
            run_id="acceptance-ledger",
            task_config="configs/tasks/toy.yaml",
            gbt_config="configs/gbt/plbind_adaptive_gbt.yaml",
            representation_spec="/scratch/repr.json",
            scout_artifact="/scratch/scout.json",
            candidate_rank=2,
            train_feature_qc_report="/scratch/train-qc.json",
            evaluation_feature_qc_report="/scratch/test-qc.json",
            train_feature_manifests={
                "EIC": "/scratch/train-eic.jsonl",
                "PL": "/scratch/train-pl.jsonl",
            },
            evaluation_feature_manifests={
                "EIC": "/scratch/test-eic.jsonl",
                "PL": "/scratch/test-pl.jsonl",
            },
        ),
    )

    ledger = create_job_ledger(
        {"run_id": "acceptance-ledger", "evaluation_jobs": [plan.to_dict()]},
        source_plan="plan.json",
    )

    assert ledger["status"] == "PLANNED"
    assert ledger["jobs"][0]["plan"]["job_kind"] == "acceptance_evaluation"
    assert ledger["jobs"][0]["plan"]["input_manifests"] == {
        "EVALUATION_EIC": "/scratch/test-eic.jsonl",
        "EVALUATION_PL": "/scratch/test-pl.jsonl",
        "TRAIN_EIC": "/scratch/train-eic.jsonl",
        "TRAIN_PL": "/scratch/train-pl.jsonl",
    }


@pytest.mark.parametrize(
    "status", ["TARGET_REACHED", "CANDIDATE_REJECTED", "TARGET_NOT_REACHED"]
)
def test_acceptance_inspector_preserves_candidate_status(tmp_path: Path, status: str):
    report = tmp_path / "acceptance.json"
    report.write_text(
        json.dumps(
            {
                "report_schema": "mint-agent.acceptance-gbt-evaluation.v1",
                "status": status,
                "dataset_id": "toy",
                "evidence_scope": "acceptance_test",
                "representation_hash": "repr-1",
                "scout_artifact": "/scratch/scout.json",
                "candidate_rank": 1,
                "candidate_count": 3,
                "selected_subset": ["EIC", "PL"],
                "invariants": ["EIC", "PL"],
                "selected_score": 0.7,
                "target_metric": "PCC",
                "target_value": 0.75,
                "gbt_parameter_hash": "gbt-1",
                "protocol": {
                    "full_train_cross_validation": False,
                    "evaluation_used_for_candidate_selection": True,
                    "independent_test_available": False,
                    "acceptance_query_index": 1,
                },
                "sample_counts": {"train": 6, "acceptance": 3},
                "sample_order_hashes": {
                    "train": "train-hash",
                    "acceptance": "acceptance-hash",
                },
                "feature_manifests": {
                    "train": {
                        "EIC": "/scratch/train-eic.jsonl",
                        "PL": "/scratch/train-pl.jsonl",
                    },
                    "acceptance": {
                        "EIC": "/scratch/test-eic.jsonl",
                        "PL": "/scratch/test-pl.jsonl",
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    inspected = inspect_acceptance_evaluation_report(report)

    assert inspected["state"] == "COMPLETE"
    assert inspected["report_status"] == status
    assert inspected["sample_count"] == 3
    assert inspected["candidate_rank"] == 1


def test_progressive_acceptance_inspector_preserves_maximization_contract(
    tmp_path: Path,
):
    report = tmp_path / "progressive-acceptance.json"
    report.write_text(
        json.dumps(
            {
                "report_schema": "mint-agent.progressive-acceptance-gbt-evaluation.v1",
                "status": "MAXIMIZATION_COMPLETE",
                "dataset_id": "toy",
                "evidence_scope": "acceptance_test",
                "representation_hash": "repr-1",
                "scout_artifact": "/scratch/scout.json",
                "invariants": ["PL", "PH"],
                "max_acquisitions": 2,
                "selection_objective": "maximize_rank1",
                "objective_complete": True,
                "selected_subset": ["PL", "PH"],
                "selected_score": 0.81,
                "target_metric": "PCC",
                "target_value": 0.75,
                "gbt_parameter_hash": "gbt-1",
                "protocol": {
                    "full_train_cross_validation": False,
                    "evaluation_used_for_candidate_selection": True,
                    "independent_test_available": False,
                    "all_available_subsets_scored": True,
                    "single_invariant_predictions_reused": True,
                    "acceptance_query_index": 2,
                },
                "sample_counts": {"train": 6, "acceptance": 3},
                "sample_order_hashes": {
                    "train": "train-hash",
                    "acceptance": "acceptance-hash",
                },
                "feature_manifests": {
                    "train": {
                        "PL": "/scratch/train-pl.jsonl",
                        "PH": "/scratch/train-ph.jsonl",
                    },
                    "acceptance": {
                        "PL": "/scratch/test-pl.jsonl",
                        "PH": "/scratch/test-ph.jsonl",
                    },
                },
                "acceptance_predictions": {
                    "PL": {"a": 1.0, "b": 2.0, "c": 3.0},
                    "PH": {"a": 1.1, "b": 2.1, "c": 3.1},
                },
            }
        ),
        encoding="utf-8",
    )

    inspected = inspect_acceptance_evaluation_report(report)

    assert inspected["state"] == "COMPLETE"
    assert inspected["report_status"] == "MAXIMIZATION_COMPLETE"
    assert inspected["max_acquisitions"] == 2
    assert inspected["selected_subset"] == ("PL", "PH")
    assert inspected["selection_objective"] == "maximize_rank1"


def test_progressive_acceptance_inspector_accepts_top_k_protocol(
    tmp_path: Path,
):
    report = tmp_path / "progressive-top-k-acceptance.json"
    report.write_text(
        json.dumps(
            {
                "report_schema": "mint-agent.progressive-acceptance-gbt-evaluation.v1",
                "status": "MAXIMIZATION_COMPLETE",
                "dataset_id": "toy",
                "evidence_scope": "acceptance_test",
                "representation_hash": "repr-1",
                "scout_artifact": "/scratch/scout.json",
                "invariants": ["EIC", "FPRC", "PH", "PL", "CA"],
                "max_acquisitions": 5,
                "selection_objective": "maximize_top_k",
                "objective_complete": True,
                "candidate_rank_limit": 5,
                "selected_subset": ["EIC", "FPRC", "PH", "PL"],
                "selected_score": 0.83,
                "target_metric": "PCC",
                "target_value": 0.75,
                "gbt_parameter_hash": "gbt-1",
                "protocol": {
                    "full_train_cross_validation": False,
                    "evaluation_used_for_candidate_selection": True,
                    "independent_test_available": False,
                    "all_available_subsets_scored": False,
                    "single_invariant_predictions_reused": True,
                    "acceptance_query_index": 5,
                },
                "sample_counts": {"train": 6, "acceptance": 3},
                "sample_order_hashes": {
                    "train": "train-hash",
                    "acceptance": "acceptance-hash",
                },
                "feature_manifests": {
                    "train": {
                        "CA": "/scratch/train-ca.jsonl",
                        "PH": "/scratch/train-ph.jsonl",
                        "FPRC": "/scratch/train-fprc.jsonl",
                        "EIC": "/scratch/train-eic.jsonl",
                        "PL": "/scratch/train-pl.jsonl",
                    },
                    "acceptance": {
                        "CA": "/scratch/test-ca.jsonl",
                        "PH": "/scratch/test-ph.jsonl",
                        "FPRC": "/scratch/test-fprc.jsonl",
                        "EIC": "/scratch/test-eic.jsonl",
                        "PL": "/scratch/test-pl.jsonl",
                    },
                },
                "acceptance_predictions": {
                    "EIC": {"a": 1.0, "b": 2.0, "c": 3.0},
                    "FPRC": {"a": 1.1, "b": 2.1, "c": 3.1},
                    "PH": {"a": 1.2, "b": 2.2, "c": 3.2},
                    "PL": {"a": 1.3, "b": 2.3, "c": 3.3},
                    "CA": {"a": 1.4, "b": 2.4, "c": 3.4},
                },
            }
        ),
        encoding="utf-8",
    )

    inspected = inspect_acceptance_evaluation_report(report)

    assert inspected["state"] == "COMPLETE"
    assert inspected["report_status"] == "MAXIMIZATION_COMPLETE"
    assert inspected["selection_objective"] == "maximize_top_k"
    assert inspected["selected_subset"] == ("EIC", "FPRC", "PH", "PL")


def test_acceptance_reconciliation_uses_qc_for_representation_certification(
    tmp_path: Path,
):
    train_output = tmp_path / "train-one.npy"
    train_output_two = tmp_path / "train-two.npy"
    evaluation_output = tmp_path / "evaluation.npy"
    train_output.write_bytes(b"train-feature")
    train_output_two.write_bytes(b"train-feature-two")
    evaluation_output.write_bytes(b"evaluation-feature")
    train_manifest = tmp_path / "train-eic.jsonl"
    evaluation_manifest = tmp_path / "evaluation-eic.jsonl"
    train_manifest.write_text(
        "\n".join(
            json.dumps(
                {
                    "sample_id": sample_id,
                    "invariant": "EIC",
                    "status": "computed",
                    "output_path": str(output_path),
                }
            )
            for sample_id, output_path in (
                ("train-one", train_output),
                ("train-two", train_output_two),
            )
        )
        + "\n",
        encoding="utf-8",
    )
    evaluation_manifest.write_text(
        json.dumps(
            {
                "sample_id": "test-one",
                "invariant": "EIC",
                "status": "computed",
                "output_path": str(evaluation_output),
            }
        )
        + "\n",
        encoding="utf-8",
    )

    train_qc = tmp_path / "train-qc.json"
    evaluation_qc = tmp_path / "evaluation-qc.json"
    for path, scope, sample_ids, manifest in (
        (train_qc, "full_train", ["train-one", "train-two"], train_manifest),
        (evaluation_qc, "external_test", ["test-one"], evaluation_manifest),
    ):
        path.write_text(
            json.dumps(
                {
                    "report_schema": "mint-agent.feature-qc.v1",
                    "status": "PASS",
                    "dataset_id": "toy",
                    "evidence_scope": scope,
                    "representation_hash": "repr-1",
                    "sample_ids": sample_ids,
                    "feature_manifests": {"EIC": str(manifest)},
                }
            ),
            encoding="utf-8",
        )

    scout_path = tmp_path / "scout.json"
    ScoutExecutionArtifact(
        artifact_version="mint-agent.scout-execution.v1",
        modeling_sample_ids=("train-one", "train-two"),
        representation_hash="repr-1",
        frozen_priority_order=(("EIC",),),
        full_fold_assignment={"train-one": 0, "train-two": 1},
        target_metric="PCC",
        target_direction="higher",
        target_value=0.75,
        target_source="user",
        gbt_parameter_hash="gbt-1",
        probe_hash="probe-1",
        ranking_policy="hierarchical_empirical_v1",
    ).write(scout_path)

    profile = replace(
        load_execution_profile(Path("configs/execution/sapelo2.yaml")),
        run_root=str(tmp_path),
        log_root=str(tmp_path),
    )
    plan = build_acceptance_evaluation_job_plan(
        profile,
        AcceptanceEvaluationJobRequest(
            dataset_id="toy",
            invariants=("EIC",),
            representation_hash="repr-1",
            run_id="acceptance-reconcile",
            task_config="configs/tasks/toy.yaml",
            gbt_config="configs/gbt/plbind_adaptive_gbt.yaml",
            representation_spec=str(tmp_path / "representation.json"),
            scout_artifact=str(scout_path),
            candidate_rank=1,
            train_feature_qc_report=str(train_qc),
            evaluation_feature_qc_report=str(evaluation_qc),
            train_feature_manifests={"EIC": str(train_manifest)},
            evaluation_feature_manifests={"EIC": str(evaluation_manifest)},
        ),
    )
    report_path = Path(plan.manifest_path)
    report_path.write_text(
        json.dumps(
            {
                "report_schema": "mint-agent.acceptance-gbt-evaluation.v1",
                "status": "TARGET_REACHED",
                "dataset_id": "toy",
                "evidence_scope": "acceptance_test",
                "representation_hash": "repr-1",
                "scout_artifact": str(scout_path),
                "candidate_rank": 1,
                "candidate_count": 1,
                "selected_subset": ["EIC"],
                "invariants": ["EIC"],
                "selected_score": 0.8,
                "target_metric": "PCC",
                "target_value": 0.75,
                "gbt_parameter_hash": "gbt-1",
                "protocol": {
                    "full_train_cross_validation": False,
                    "evaluation_used_for_candidate_selection": True,
                    "independent_test_available": False,
                    "acceptance_query_index": 1,
                },
                "sample_counts": {"train": 2, "acceptance": 1},
                "sample_order_hashes": {
                    "train": stable_hash(("train-one", "train-two")),
                    "acceptance": stable_hash(("test-one",)),
                },
                "feature_manifests": {
                    "train": {"EIC": str(train_manifest)},
                    "acceptance": {"EIC": str(evaluation_manifest)},
                },
            }
        ),
        encoding="utf-8",
    )

    record = _acceptance_evaluation_record_for_job(
        {"plan": plan.to_dict(), "plan_id": plan.plan_id, "attempts": []},
        {"run_id": "acceptance-reconcile"},
    )

    assert record.status == "TARGET_REACHED"
    assert record.representation_hash == "repr-1"


def test_validation_plan_is_accepted_by_job_ledger():
    profile = load_execution_profile(Path("configs/execution/sapelo2.yaml"))
    plan = build_validation_evaluation_job_plan(
        profile,
        ValidationEvaluationJobRequest(
            dataset_id="toy",
            invariants=("PL", "PH"),
            representation_hash="repr-1",
            run_id="validation-ledger",
            task_config="configs/tasks/toy.yaml",
            gbt_config="configs/gbt/plbind_fixed_gbt.yaml",
            representation_spec="/scratch/repr.json",
            scout_artifact="/scratch/scout.json",
            train_feature_qc_report="/scratch/train-qc.json",
            validation_feature_qc_report="/scratch/validation-qc.json",
            train_feature_manifests={
                "PL": "/scratch/train-pl.jsonl",
                "PH": "/scratch/train-ph.jsonl",
            },
            validation_feature_manifests={
                "PL": "/scratch/validation-pl.jsonl",
                "PH": "/scratch/validation-ph.jsonl",
            },
            max_acquisitions=2,
            prior_evaluation_report="/scratch/validation-stage-1.json",
        ),
    )

    ledger = create_job_ledger(
        {"run_id": "validation-ledger", "evaluation_jobs": [plan.to_dict()]},
        source_plan="plan.json",
    )

    assert ledger["status"] == "PLANNED"
    assert ledger["jobs"][0]["plan"]["job_kind"] == "validation_evaluation"
    assert ledger["jobs"][0]["plan"]["input_manifests"]["PRIOR_EVALUATION"] == (
        "/scratch/validation-stage-1.json"
    )


def test_frozen_test_plan_is_accepted_by_job_ledger():
    profile = load_execution_profile(Path("configs/execution/sapelo2.yaml"))
    plan = build_frozen_test_evaluation_job_plan(
        profile,
        FrozenTestEvaluationJobRequest(
            dataset_id="toy",
            invariants=("PL",),
            representation_hash="repr-1",
            run_id="test-ledger",
            task_config="configs/tasks/toy.yaml",
            gbt_config="configs/gbt/plbind_fixed_gbt.yaml",
            representation_spec="/scratch/repr.json",
            selection_report="/scratch/selection.json",
            train_feature_qc_report="/scratch/train-qc.json",
            validation_feature_qc_report="/scratch/validation-qc.json",
            test_feature_qc_report="/scratch/test-qc.json",
            train_feature_manifests={"PL": "/scratch/train-pl.jsonl"},
            validation_feature_manifests={"PL": "/scratch/validation-pl.jsonl"},
            test_feature_manifests={"PL": "/scratch/test-pl.jsonl"},
        ),
    )

    ledger = create_job_ledger(
        {"run_id": "test-ledger", "evaluation_jobs": [plan.to_dict()]},
        source_plan="plan.json",
    )

    assert ledger["status"] == "PLANNED"
    assert ledger["jobs"][0]["plan"]["job_kind"] == "frozen_test_evaluation"


@pytest.mark.parametrize("status", ["TARGET_REACHED", "TARGET_NOT_REACHED"])
def test_frozen_test_inspector_preserves_terminal_target_status(
    tmp_path: Path, status: str
):
    report = tmp_path / "final-test.json"
    report.write_text(
        json.dumps(
            {
                "report_schema": "mint-agent.frozen-test-gbt-evaluation.v1",
                "status": status,
                "dataset_id": "toy",
                "evidence_scope": "external_test",
                "representation_hash": "repr-1",
                "selection_hash": "selection-1",
                "selected_subset": ["PL"],
                "gbt_parameter_hash": "gbt-1",
                "selection_report_sha256": "selection-sha",
                "protocol": {
                    "test_used_for_selection": False,
                    "selection_frozen_before_test_feature_loading": True,
                    "test_query_count": 1,
                },
                "sample_counts": {"train": 2, "validation": 0, "test": 1},
                "sample_order_hashes": {
                    "train": "train-hash",
                    "validation": None,
                    "test": "test-hash",
                },
                "feature_manifests": {
                    "train": {"PL": "/scratch/train-pl.jsonl"},
                    "validation": {},
                    "test": {"PL": "/scratch/test-pl.jsonl"},
                },
            }
        ),
        encoding="utf-8",
    )

    inspected = inspect_frozen_test_evaluation_report(report)

    assert inspected["state"] == "COMPLETE"
    assert inspected["report_status"] == status


def test_later_validation_reconciliation_separates_prior_from_feature_inputs(
    tmp_path: Path,
):
    prior = tmp_path / "validation-stage-1.json"
    prior.write_text("{}\n", encoding="utf-8")
    prior_sha256 = hashlib.sha256(prior.read_bytes()).hexdigest()
    plan = {
        "input_manifests": {
            "TRAIN_CA": "/scratch/train-ca.jsonl",
            "TRAIN_EIC": "/scratch/train-eic.jsonl",
            "VALIDATION_CA": "/scratch/validation-ca.jsonl",
            "VALIDATION_EIC": "/scratch/validation-eic.jsonl",
            "PRIOR_EVALUATION": str(prior),
        }
    }
    payload = {
        "feature_manifests": {
            "train": {
                "CA": "/scratch/train-ca.jsonl",
                "EIC": "/scratch/train-eic.jsonl",
            },
            "validation": {
                "CA": "/scratch/validation-ca.jsonl",
                "EIC": "/scratch/validation-eic.jsonl",
            },
        },
        "prediction_reuse": {
            "prior_evaluation_report": str(prior),
            "prior_evaluation_sha256": prior_sha256,
        },
    }

    _assert_validation_evaluation_inputs(plan, payload)
