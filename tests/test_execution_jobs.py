from __future__ import annotations

import json
from pathlib import Path

from mint_scout.artifact_memory import ArtifactRegistry, inspect_artifact
from mint_scout.execution.jobs import (
    CommandResult,
    create_job_ledger,
    inspect_dataset_audit_report,
    inspect_feature_manifest,
    inspect_feature_qc_report,
    inspect_filtration_audit_report,
    inspect_model_evaluation_report,
    inspect_representation_design_report,
    inspect_scout_combined_report,
    inspect_scout_oof_report,
    load_job_ledger,
    parse_sbatch_job_id,
    reconcile_completed_jobs,
    refresh_job_ledger,
    resume_job_ledger,
    submit_ready_jobs,
    write_job_ledger,
)
from mint_scout.execution.artifacts import ScoutExecutionArtifact
from mint_scout.manage_jobs import main as manage_jobs_main
from mint_scout.representation import make_legacy_casf_representation_spec


def test_dataset_audit_job_is_validated_and_registered(tmp_path: Path):
    report_path = tmp_path / "dataset-audit.json"
    report_path.write_text(
        json.dumps(
            {
                "report_schema": "mint-agent.dataset-audit.v1",
                "task_id": "toy",
                "modeling_scope": "train",
                "sample_count": 2,
                "sample_ids": ["a", "b"],
                "schema_id": "adaptive",
                "schema_review_required": False,
                "roles": [],
                "target_summary": {"count": 2},
                "status": "PASS",
            }
        ),
        encoding="utf-8",
    )
    plan = {
        "scheduler": "slurm",
        "job_kind": "dataset_audit",
        "dataset_id": "toy",
        "invariant": "DATASET",
        "invariants": ["DATASET"],
        "evidence_scope": "design",
        "representation_hash": "not_applicable",
        "working_directory": str(tmp_path),
        "manifest_path": str(report_path),
        "environment": {
            "TASK_CONFIG": str(tmp_path / "task.yaml"),
            "OUTPUT_JSON": str(report_path),
        },
        "input_manifests": {},
        "submit_command": ["sbatch", "run_dataset_audit.sbatch"],
    }
    ledger = create_job_ledger(
        {"run_id": "run", "dataset_id": "toy", "setup_jobs": [plan]},
        source_plan=tmp_path / "graph.json",
    )
    ledger["jobs"][0]["state"] = "COMPLETE"

    assert inspect_dataset_audit_report(report_path)["state"] == "COMPLETE"
    reconcile_completed_jobs(ledger, registry=tmp_path / "artifacts.sqlite")

    records = ArtifactRegistry(tmp_path / "artifacts.sqlite").query(
        artifact_kind="dataset_audit", dataset_id="toy"
    )
    assert ledger["jobs"][0]["state"] == "REGISTERED"
    assert records[0].status == "PASS"
    assert records[0].sample_order_hash is not None


def test_reconcile_failed_job_can_be_retried_after_report_repair(tmp_path: Path):
    report_path = tmp_path / "dataset-audit.json"

    def write_report(sample_ids: list[str]) -> None:
        report_path.write_text(
            json.dumps(
                {
                    "report_schema": "mint-agent.dataset-audit.v1",
                    "task_id": "toy",
                    "modeling_scope": "train",
                    "sample_count": len(sample_ids),
                    "sample_ids": sample_ids,
                    "schema_id": "adaptive",
                    "schema_review_required": False,
                    "roles": [],
                    "target_summary": {"count": len(sample_ids)},
                    "status": "PASS",
                }
            ),
            encoding="utf-8",
        )

    plan = {
        "scheduler": "slurm",
        "job_kind": "dataset_audit",
        "dataset_id": "toy",
        "invariant": "DATASET",
        "invariants": ["DATASET"],
        "evidence_scope": "design",
        "representation_hash": "not_applicable",
        "working_directory": str(tmp_path),
        "manifest_path": str(report_path),
        "environment": {
            "TASK_CONFIG": str(tmp_path / "task.yaml"),
            "OUTPUT_JSON": str(report_path),
        },
        "input_manifests": {},
        "submit_command": ["sbatch", "run_dataset_audit.sbatch"],
    }
    ledger = create_job_ledger(
        {"run_id": "run", "dataset_id": "toy", "setup_jobs": [plan]},
        source_plan=tmp_path / "graph.json",
    )

    write_report(["a", "b"])
    ledger["jobs"][0]["state"] = "COMPLETE"
    reconcile_completed_jobs(ledger, registry=tmp_path / "artifacts.sqlite")
    assert ledger["jobs"][0]["state"] == "REGISTERED"

    write_report(["a", "b", "c"])
    ledger["jobs"][0]["state"] = "COMPLETE"
    reconcile_completed_jobs(ledger, registry=tmp_path / "artifacts.sqlite")
    assert ledger["jobs"][0]["state"] == "RECONCILE_FAILED"
    assert ledger["status"] == "NEEDS_REVIEW"

    write_report(["a", "b"])
    reconcile_completed_jobs(ledger, registry=tmp_path / "artifacts.sqlite")
    assert ledger["jobs"][0]["state"] == "REGISTERED"
    assert ledger["status"] == "COMPLETE"
    assert "reconciliation_error" not in ledger["jobs"][0]


def test_representation_design_reconcile_accepts_compatible_input_hash_mismatch(
    tmp_path: Path,
):
    spec = make_legacy_casf_representation_spec()
    audit = tmp_path / "dataset-audit.json"
    audit.write_text(
        json.dumps(
            {
                "report_schema": "mint-agent.dataset-audit.v1",
                "task_id": "toy",
                "modeling_scope": "train",
                "sample_count": 2,
                "sample_ids": ["a", "b"],
                "schema_id": "adaptive",
                "schema_review_required": False,
                "roles": [],
                "target_summary": {"count": 2},
                "status": "PASS",
            }
        ),
        encoding="utf-8",
    )
    report = tmp_path / "representation.json"
    report.write_text(
        json.dumps(
            {
                "report_schema": "mint-agent.representation-design.v1",
                "run_kind": "full_modeling_pool_design",
                "task_id": "toy",
                "dataset_id": "toy",
                "evidence_scope": "design",
                "modeling_scope": "train",
                "sample_count": 2,
                "sample_ids": ["a", "b"],
                "representation_spec": spec.to_dict(),
                "representation_hash": spec.spec_hash,
                "geometry_profile": {},
                "design_input_hash": "current-hash",
                "status": "COMPLETE",
            }
        ),
        encoding="utf-8",
    )
    plan = {
        "scheduler": "slurm",
        "job_kind": "representation_design",
        "dataset_id": "toy",
        "invariant": "REPRESENTATION",
        "invariants": ["REPRESENTATION"],
        "evidence_scope": "design",
        "representation_hash": "pending",
        "working_directory": str(tmp_path),
        "manifest_path": str(report),
        "environment": {
            "TASK_CONFIG": str(tmp_path / "task.yaml"),
            "DATA_AUDIT_REPORT": str(audit),
            "DESIGN_INPUT_HASH": "submitted-legacy-hash",
            "REPORT": str(report),
        },
        "input_manifests": {"DATA_AUDIT": str(audit)},
        "submit_command": ["sbatch", "run_representation_design.sbatch"],
    }
    ledger = create_job_ledger(
        {"run_id": "run", "dataset_id": "toy", "setup_jobs": [plan]},
        source_plan=tmp_path / "graph.json",
    )
    ledger["jobs"][0]["state"] = "COMPLETE"

    reconcile_completed_jobs(ledger, registry=tmp_path / "artifacts.sqlite")

    records = ArtifactRegistry(tmp_path / "artifacts.sqlite").query(
        artifact_kind="representation_design", dataset_id="toy"
    )
    assert ledger["jobs"][0]["state"] == "REGISTERED"
    assert records[0].metadata["design_input_hash"] == "current-hash"
    assert records[0].metadata["submitted_design_input_hash"] == "submitted-legacy-hash"
    assert records[0].metadata["design_input_hash_mismatch_accepted"] is True


def test_progressive_acceptance_evaluation_plan_validates_with_max_acquisitions(
    tmp_path: Path,
):
    output = tmp_path / "acceptance-stage-01.json"
    plan = {
        "profile_id": "test",
        "scheduler": "slurm",
        "job_kind": "acceptance_evaluation",
        "dataset_id": "toy",
        "invariant": "EIC",
        "invariants": ["EIC"],
        "evidence_scope": "acceptance_test",
        "representation_hash": "repr-1",
        "job_name": "mint-acceptance-evaluation-eic",
        "working_directory": "/project/mint-agent",
        "script_path": "/project/mint-agent/scripts/slurm/run_acceptance_gbt_evaluation.sbatch",
        "resources": {},
        "environment": {
            "ACCEPTANCE_MODE": "progressive",
            "TASK_CONFIG": "/project/mint-agent/task.yaml",
            "GBT_CONFIG": "/project/mint-agent/gbt.yaml",
            "REPRESENTATION_SPEC": "/project/mint-agent/repr.json",
            "SCOUT_ARTIFACT": "/project/mint-agent/scout.json",
            "TRAIN_QC_REPORT": "/project/mint-agent/train-qc.json",
            "EVALUATION_QC_REPORT": "/project/mint-agent/test-qc.json",
            "FEATURE_INVARIANTS": "EIC",
            "MAX_ACQUISITIONS": "1",
            "FEATURE_MANIFEST_TRAIN_EIC": "/project/mint-agent/train-eic.jsonl",
            "FEATURE_MANIFEST_EVALUATION_EIC": "/project/mint-agent/test-eic.jsonl",
            "OUTPUT": str(output),
        },
        "input_manifests": {
            "TRAIN_EIC": "/project/mint-agent/train-eic.jsonl",
            "EVALUATION_EIC": "/project/mint-agent/test-eic.jsonl",
        },
        "manifest_path": str(output),
        "stdout_path": "/scratch/acceptance-%j.out",
        "stderr_path": "/scratch/acceptance-%j.err",
        "submit_command": ["sbatch", "acceptance.sbatch"],
    }

    ledger = create_job_ledger(
        {"run_id": "acceptance-run", "dataset_id": "toy", "evaluation_jobs": [plan]},
        source_plan=tmp_path / "graph.json",
    )

    assert ledger["jobs"][0]["plan"]["environment"]["MAX_ACQUISITIONS"] == "1"


def test_probe_selection_job_accepts_filtration_repair_representation(
    tmp_path: Path,
):
    spec = make_legacy_casf_representation_spec()
    repair = tmp_path / "repair.json"
    repair.write_text(
        json.dumps(
            {
                "report_schema": "mint-agent.filtration-repair.v1",
                "status": "COMPLETE",
                "dataset_id": "old-run",
                "evidence_scope": "design",
                "audit_evidence_scope": "probe",
                "representation_hash": spec.spec_hash,
                "representation_spec": spec.to_dict(),
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "probe.json"
    output.write_text(
        json.dumps(
            {
                "report_schema": "mint-agent.casf-probe-selection.v1",
                "status": "COMPLETE",
                "task_id": "toy",
                "evidence_scope": "probe",
                "representation_hash": spec.spec_hash,
                "selection_hash": "sel-1",
                "modeling_sample_ids": ["a", "b", "c"],
                "probe_sample_ids": ["a", "b"],
            }
        ),
        encoding="utf-8",
    )
    plan = {
        "profile_id": "test",
        "scheduler": "slurm",
        "job_kind": "probe_selection",
        "dataset_id": "toy",
        "invariant": "PROBE",
        "invariants": ["PROBE"],
        "evidence_scope": "probe",
        "representation_hash": spec.spec_hash,
        "working_directory": "/project/mint-agent",
        "manifest_path": str(output),
        "environment": {
            "TASK_CONFIG": str(tmp_path / "task.yaml"),
            "SCOUT_CONFIG": str(tmp_path / "scout.yaml"),
            "REPRESENTATION_SPEC": str(repair),
            "OUTPUT": str(output),
        },
        "input_manifests": {"REPRESENTATION": str(repair)},
        "submit_command": ["sbatch", "probe.sbatch"],
    }
    ledger = create_job_ledger(
        {"run_id": "run", "dataset_id": "toy", "setup_jobs": [plan]},
        source_plan=tmp_path / "graph.json",
    )
    ledger["jobs"][0]["state"] = "COMPLETE"

    inspected = inspect_representation_design_report(repair)
    assert inspected["state"] == "COMPLETE"
    assert inspected["dataset_id"] == "old-run"
    assert inspected["sample_order_hash"] is None
    reconcile_completed_jobs(ledger, registry=tmp_path / "artifacts.sqlite")

    records = ArtifactRegistry(tmp_path / "artifacts.sqlite").query(
        artifact_kind="probe_selection", dataset_id="toy"
    )
    assert ledger["jobs"][0]["state"] == "REGISTERED"
    assert records[0].representation_hash == spec.spec_hash


def _plan(manifest: Path) -> dict:
    return {
        "profile_id": "test",
        "scheduler": "slurm",
        "job_kind": "feature_batch",
        "dataset_id": "toy",
        "invariant": "PL",
        "evidence_scope": "full_train",
        "representation_hash": "repr-1",
        "job_name": "mint-feature-pl-full_train",
        "working_directory": "/project/mint-agent",
        "script_path": "/project/mint-agent/scripts/slurm/run_feature_batch.sbatch",
        "resources": {},
        "environment": {"INVARIANT": "PL", "SPLIT": "train"},
        "manifest_path": str(manifest),
        "stdout_path": "/scratch/log-%j.out",
        "stderr_path": "/scratch/log-%j.err",
        "submit_command": ["sbatch", "--job-name=mint-feature-pl", "feature.sbatch"],
    }


def _report(manifest: Path) -> dict:
    return {
        "report_schema": "mint-agent.graph-run.v1",
        "run_id": "run-1",
        "dataset_id": "toy",
        "feature_jobs": [_plan(manifest)],
    }


def _qc_plan(report: Path, feature_manifest: Path) -> dict:
    return {
        "profile_id": "test",
        "scheduler": "slurm",
        "job_kind": "feature_qc",
        "dataset_id": "toy",
        "invariant": "PL",
        "invariants": ["PL"],
        "evidence_scope": "full_train",
        "representation_hash": "repr-1",
        "job_name": "mint-qc-pl-full_train",
        "working_directory": "/project/mint-agent",
        "script_path": "/project/mint-agent/scripts/slurm/run_feature_qc.sbatch",
        "resources": {},
        "environment": {
            "FEATURE_INVARIANTS": "PL",
            "FEATURE_MANIFEST_PL": str(feature_manifest),
            "OUTPUT": str(report),
        },
        "input_manifests": {"PL": str(feature_manifest)},
        "manifest_path": str(report),
        "stdout_path": "/scratch/qc-log-%j.out",
        "stderr_path": "/scratch/qc-log-%j.err",
        "submit_command": ["sbatch", "--job-name=mint-qc-pl", "qc.sbatch"],
    }


def _filtration_plan(report: Path, feature_manifest: Path) -> dict:
    plan = _qc_plan(report, feature_manifest)
    plan.update(
        {
            "job_kind": "filtration_audit",
            "job_name": "mint-filtration-pl-full_train",
            "script_path": "/project/mint-agent/scripts/slurm/run_filtration_audit.sbatch",
            "submit_command": [
                "sbatch",
                "--job-name=mint-filtration-pl",
                "filtration.sbatch",
            ],
        }
    )
    return plan


def _scout_oof_plan(
    report: Path,
    execution: Path,
    feature_manifest: Path,
    probe: Path,
    qc_report: Path,
    audit_report: Path,
) -> dict:
    return {
        "profile_id": "test",
        "scheduler": "slurm",
        "job_kind": "scout_oof",
        "dataset_id": "toy",
        "invariant": "PL",
        "invariants": ["PL"],
        "evidence_scope": "probe",
        "representation_hash": "repr-1",
        "job_name": "mint-scout-oof-pl",
        "working_directory": "/project/mint-agent",
        "script_path": "/project/mint-agent/scripts/slurm/run_scout_oof.sbatch",
        "resources": {},
        "environment": {
            "FEATURE_INVARIANTS": "PL",
            "FEATURE_MANIFEST_PL": str(feature_manifest),
            "PROBE_SELECTION": str(probe),
            "FEATURE_QC_REPORT": str(qc_report),
            "FILTRATION_AUDIT_REPORT": str(audit_report),
            "OUTPUT": str(report),
            "EXECUTION_ARTIFACT": str(execution),
        },
        "input_manifests": {"PL": str(feature_manifest)},
        "manifest_path": str(report),
        "stdout_path": "/scratch/scout-oof-%j.out",
        "stderr_path": "/scratch/scout-oof-%j.err",
        "submit_command": ["sbatch", "scout-oof.sbatch"],
    }


def _write_probe_scout_inputs(tmp_path: Path) -> dict[str, Path]:
    feature_path_a = tmp_path / "a.npy"
    feature_path_b = tmp_path / "b.npy"
    feature_path_a.write_bytes(b"feature-a")
    feature_path_b.write_bytes(b"feature-b")
    manifest = tmp_path / "PL.jsonl"
    manifest.write_text(
        "\n".join(
            json.dumps(
                {
                    "sample_id": sample_id,
                    "invariant": "PL",
                    "status": "computed",
                    "output_path": str(feature_path),
                }
            )
            for sample_id, feature_path in (
                ("a", feature_path_a),
                ("b", feature_path_b),
            )
        )
        + "\n",
        encoding="utf-8",
    )
    probe = tmp_path / "probe.json"
    probe.write_text(
        json.dumps(
            {
                "report_schema": "mint-agent.casf-probe-selection.v1",
                "representation_hash": "repr-1",
                "selection_hash": "selection-1",
                "probe_sample_ids": ["a", "b"],
                "modeling_sample_ids": ["a", "b"],
            }
        ),
        encoding="utf-8",
    )
    qc_report = tmp_path / "qc.json"
    qc_report.write_text(
        json.dumps(
            {
                "report_schema": "mint-agent.feature-qc.v1",
                "status": "PASS",
                "dataset_id": "toy",
                "evidence_scope": "probe",
                "representation_hash": "repr-1",
                "selection_hash": "selection-1",
                "sample_ids": ["a", "b"],
                "feature_manifests": {"PL": str(manifest)},
            }
        ),
        encoding="utf-8",
    )
    audit_report = tmp_path / "filtration.json"
    audit_report.write_text(
        json.dumps(
            {
                "report_schema": "mint-agent.filtration-axis-audit.v1",
                "status": "PASS",
                "dataset_id": "toy",
                "evidence_scope": "probe",
                "representation_hash": "repr-1",
                "selection_hash": "selection-1",
                "sample_ids": ["a", "b"],
                "feature_manifests": {"PL": str(manifest)},
                "invariants": {"PL": {"recommendation": "KEEP"}},
            }
        ),
        encoding="utf-8",
    )
    return {
        "manifest": manifest,
        "probe": probe,
        "qc": qc_report,
        "audit": audit_report,
    }


def test_submit_and_completed_manifest_are_audited(tmp_path: Path):
    manifest = tmp_path / "PL.jsonl"
    output_a = tmp_path / "a.npy"
    output_b = tmp_path / "b.npy"
    output_a.write_bytes(b"feature-a")
    output_b.write_bytes(b"feature-b")
    ledger = create_job_ledger(_report(manifest), source_plan=tmp_path / "plan.json")

    def submit_runner(command, cwd):
        assert command[0] == "sbatch"
        assert cwd == "/project/mint-agent"
        return CommandResult(0, "Submitted batch job 12345\n", "")

    submit_ready_jobs(ledger, execute=True, runner=submit_runner)
    assert ledger["status"] == "ACTIVE"
    assert ledger["jobs"][0]["attempts"][0]["job_id"] == "12345"

    manifest.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "sample_id": "a",
                        "invariant": "PL",
                        "status": "computed",
                        "output_path": str(output_a),
                    }
                ),
                json.dumps(
                    {
                        "sample_id": "b",
                        "invariant": "PL",
                        "status": "cached",
                        "output_path": str(output_b),
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    def status_runner(command, cwd):
        assert cwd is None
        if command[0] == "sacct":
            return CommandResult(0, "12345|COMPLETED|0:0\n12345.batch|COMPLETED|0:0\n", "")
        raise AssertionError(command)

    refresh_job_ledger(ledger, runner=status_runner)
    assert ledger["status"] == "COMPLETE"
    assert ledger["jobs"][0]["manifest"]["sample_count"] == 2

    registry_path = tmp_path / "artifacts.sqlite"
    ledger_path = tmp_path / "ledger.json"
    write_job_ledger(ledger, ledger_path)
    assert (
        manage_jobs_main(
            [
                "reconcile",
                "--ledger",
                str(ledger_path),
                "--registry",
                str(registry_path),
                "--offline",
            ]
        )
        == 0
    )
    ledger = load_job_ledger(ledger_path)
    assert ledger["status"] == "COMPLETE"
    assert ledger["jobs"][0]["state"] == "REGISTERED"
    records = ArtifactRegistry(registry_path).query(
        artifact_kind="feature_manifest",
        dataset_id="toy",
        evidence_scope="full_train",
        invariant="PL",
        representation_hash="repr-1",
        status="COMPLETE",
    )
    assert len(records) == 1
    assert records[0].sample_count == 2
    assert records[0].metadata["slurm_job_id"] == "12345"


def test_completed_scheduler_without_manifest_needs_resume(tmp_path: Path):
    ledger = create_job_ledger(_report(tmp_path / "missing.jsonl"), source_plan="plan.json")
    submit_ready_jobs(
        ledger,
        execute=True,
        runner=lambda command, cwd: CommandResult(0, "Submitted batch job 7\n", ""),
    )

    def completed_runner(command, cwd):
        return CommandResult(0, "7|COMPLETED|0:0\n", "")

    refresh_job_ledger(ledger, runner=completed_runner)
    assert ledger["status"] == "NEEDS_RESUME"
    assert ledger["jobs"][0]["state"] == "OUTPUT_INCOMPLETE"


def test_failed_job_is_resubmitted_as_a_new_attempt(tmp_path: Path):
    ledger = create_job_ledger(_report(tmp_path / "PL.jsonl"), source_plan="plan.json")
    submitted_ids = iter(("10", "11"))

    def submit_runner(command, cwd):
        return CommandResult(0, f"Submitted batch job {next(submitted_ids)}\n", "")

    submit_ready_jobs(ledger, execute=True, runner=submit_runner)
    refresh_job_ledger(
        ledger,
        runner=lambda command, cwd: CommandResult(0, "10|TIMEOUT|0:0\n", ""),
    )
    assert ledger["jobs"][0]["state"] == "NEEDS_RESUME"

    resume_job_ledger(ledger, execute=True, runner=submit_runner)
    attempts = ledger["jobs"][0]["attempts"]
    assert [attempt["job_id"] for attempt in attempts] == ["10", "11"]
    assert ledger["jobs"][0]["state"] == "ACTIVE"


def test_unknown_job_is_not_resubmitted_without_opt_in(tmp_path: Path):
    ledger = create_job_ledger(_report(tmp_path / "PL.jsonl"), source_plan="plan.json")
    submit_ready_jobs(
        ledger,
        execute=True,
        runner=lambda command, cwd: CommandResult(0, "Submitted batch job 20\n", ""),
    )

    def unknown_runner(command, cwd):
        return CommandResult(1, "", "not found")

    refresh_job_ledger(ledger, runner=unknown_runner)
    resume_job_ledger(
        ledger,
        execute=True,
        runner=lambda command, cwd: (_ for _ in ()).throw(AssertionError("must not submit")),
    )
    assert len(ledger["jobs"][0]["attempts"]) == 1


def test_manifest_failure_and_cli_dry_run(tmp_path: Path):
    manifest = tmp_path / "PL.jsonl"
    manifest.write_text(
        json.dumps({"sample_id": "a", "invariant": "PL", "status": "failed"})
        + "\n",
        encoding="utf-8",
    )
    assert inspect_feature_manifest(manifest)["state"] == "FAILED"
    assert parse_sbatch_job_id("Submitted batch job 999") == "999"

    plan_path = tmp_path / "plan.json"
    ledger_path = tmp_path / "ledger.json"
    plan_path.write_text(json.dumps(_report(manifest)), encoding="utf-8")
    assert (
        manage_jobs_main(
            ["submit", "--plan", str(plan_path), "--ledger", str(ledger_path)]
        )
        == 0
    )
    ledger = load_job_ledger(ledger_path)
    assert ledger["status"] == "PLANNED"
    assert ledger["jobs"][0]["attempts"] == []
    artifact = inspect_artifact(ledger_path)
    assert artifact.artifact_kind == "job_ledger"
    assert artifact.invariants == ("PL",)


def test_offline_status_refreshes_manifest_without_scheduler_query(tmp_path: Path):
    manifest = tmp_path / "PL.jsonl"
    plan_path = tmp_path / "plan.json"
    ledger_path = tmp_path / "ledger.json"
    plan_path.write_text(json.dumps(_report(manifest)), encoding="utf-8")
    manage_jobs_main(["submit", "--plan", str(plan_path), "--ledger", str(ledger_path)])
    output = tmp_path / "a.npy"
    output.write_bytes(b"feature-a")
    manifest.write_text(
        json.dumps(
            {
                "sample_id": "a",
                "invariant": "PL",
                "status": "computed",
                "output_path": str(output),
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert manage_jobs_main(["status", "--ledger", str(ledger_path), "--offline"]) == 0
    ledger = load_job_ledger(ledger_path)
    assert ledger["jobs"][0]["manifest"]["state"] == "COMPLETE"


def test_completed_scheduler_with_missing_feature_file_is_not_reconciled(tmp_path: Path):
    manifest = tmp_path / "PL.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "sample_id": "a",
                "invariant": "PL",
                "status": "computed",
                "output_path": str(tmp_path / "missing.npy"),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    ledger = create_job_ledger(_report(manifest), source_plan="plan.json")
    submit_ready_jobs(
        ledger,
        execute=True,
        runner=lambda command, cwd: CommandResult(0, "Submitted batch job 30\n", ""),
    )
    refresh_job_ledger(
        ledger,
        runner=lambda command, cwd: CommandResult(0, "30|COMPLETED|0:0\n", ""),
    )

    assert ledger["jobs"][0]["manifest"]["state"] == "OUTPUT_MISSING"
    assert ledger["jobs"][0]["state"] == "OUTPUT_INCOMPLETE"
    reconcile_completed_jobs(ledger, registry=tmp_path / "artifacts.sqlite")
    assert ArtifactRegistry(tmp_path / "artifacts.sqlite").query() == ()


def test_reconcile_rejects_manifest_invariant_mismatch(tmp_path: Path):
    manifest = tmp_path / "PL.jsonl"
    output = tmp_path / "a.npy"
    output.write_bytes(b"feature-a")
    manifest.write_text(
        json.dumps(
            {
                "sample_id": "a",
                "invariant": "PH",
                "status": "computed",
                "output_path": str(output),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    ledger = create_job_ledger(_report(manifest), source_plan="plan.json")
    submit_ready_jobs(
        ledger,
        execute=True,
        runner=lambda command, cwd: CommandResult(0, "Submitted batch job 31\n", ""),
    )
    refresh_job_ledger(
        ledger,
        runner=lambda command, cwd: CommandResult(0, "31|COMPLETED|0:0\n", ""),
    )

    reconcile_completed_jobs(ledger, registry=tmp_path / "artifacts.sqlite")
    assert ledger["status"] == "NEEDS_REVIEW"
    assert ledger["jobs"][0]["state"] == "RECONCILE_FAILED"
    assert "invariant" in ledger["jobs"][0]["reconciliation_error"]


def test_probe_feature_reconciliation_freezes_selection_identity(tmp_path: Path):
    feature_path = tmp_path / "feature.npy"
    feature_path.write_bytes(b"feature")
    manifest = tmp_path / "PL-probe.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "sample_id": "a",
                "invariant": "PL",
                "status": "computed",
                "output_path": str(feature_path),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    selection = tmp_path / "probe.json"
    selection.write_text(
        json.dumps(
            {
                "representation_hash": "repr-1",
                "selection_hash": "selection-1",
                "probe_sample_ids": ["a"],
            }
        ),
        encoding="utf-8",
    )
    plan = _plan(manifest)
    plan["evidence_scope"] = "probe"
    plan["environment"].update(
        {
            "SAMPLE_ID_FILE": str(selection),
            "SELECTION_HASH": "selection-1",
        }
    )
    ledger = create_job_ledger(
        {"run_id": "probe-run", "dataset_id": "toy", "feature_jobs": [plan]},
        source_plan=tmp_path / "plan.json",
    )
    submit_ready_jobs(
        ledger,
        execute=True,
        runner=lambda command, cwd: CommandResult(0, "Submitted batch job 32\n", ""),
    )
    refresh_job_ledger(
        ledger,
        runner=lambda command, cwd: CommandResult(0, "32|COMPLETED|0:0\n", ""),
    )

    registry_path = tmp_path / "artifacts.sqlite"
    reconcile_completed_jobs(ledger, registry=registry_path)

    records = ArtifactRegistry(registry_path).query(
        artifact_kind="feature_manifest", evidence_scope="probe"
    )
    assert ledger["jobs"][0]["state"] == "REGISTERED"
    assert records[0].selection_hash == "selection-1"


def test_job_plan_id_detects_changed_scientific_identity(tmp_path: Path):
    report = _report(tmp_path / "PL.jsonl")
    ledger = create_job_ledger(report, source_plan="plan.json")
    tampered = _report(tmp_path / "PL.jsonl")
    tampered["feature_jobs"][0]["plan_id"] = ledger["jobs"][0]["plan_id"]
    tampered["feature_jobs"][0]["representation_hash"] = "changed"

    try:
        create_job_ledger(tampered, source_plan="plan.json")
    except ValueError as exc:
        assert "plan_id" in str(exc)
    else:
        raise AssertionError("tampered scientific job identity was accepted")


def test_completed_qc_job_is_validated_and_registered(tmp_path: Path):
    feature_path = tmp_path / "feature.npy"
    feature_path.write_bytes(b"feature")
    manifest = tmp_path / "PL.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "sample_id": "a",
                "invariant": "PL",
                "status": "computed",
                "output_path": str(feature_path),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    qc_report = tmp_path / "qc.json"
    qc_report.write_text(
        json.dumps(
            {
                "report_schema": "mint-agent.feature-qc.v1",
                "status": "WARN",
                "dataset_id": "toy",
                "evidence_scope": "full_train",
                "representation_hash": "repr-1",
                "sample_ids": ["a"],
                "feature_manifests": {"PL": str(manifest)},
            }
        ),
        encoding="utf-8",
    )
    plan_report = {
        "run_id": "qc-run",
        "dataset_id": "toy",
        "qc_jobs": [_qc_plan(qc_report, manifest)],
    }
    ledger = create_job_ledger(plan_report, source_plan="qc-plan.json")
    submit_ready_jobs(
        ledger,
        execute=True,
        runner=lambda command, cwd: CommandResult(0, "Submitted batch job 55\n", ""),
    )
    refresh_job_ledger(
        ledger,
        runner=lambda command, cwd: CommandResult(0, "55|COMPLETED|0:0\n", ""),
    )
    registry_path = tmp_path / "artifacts.sqlite"
    reconcile_completed_jobs(ledger, registry=registry_path)

    assert inspect_feature_qc_report(qc_report)["report_status"] == "WARN"
    assert ledger["jobs"][0]["state"] == "REGISTERED"
    records = ArtifactRegistry(registry_path).query(
        artifact_kind="feature_qc",
        dataset_id="toy",
        evidence_scope="full_train",
        invariant="PL",
        representation_hash="repr-1",
    )
    assert len(records) == 1
    assert records[0].status == "WARN"


def test_completed_filtration_job_is_validated_and_registered(tmp_path: Path):
    feature_path = tmp_path / "feature.npy"
    feature_path.write_bytes(b"feature")
    manifest = tmp_path / "PL.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "sample_id": "a",
                "invariant": "PL",
                "status": "computed",
                "output_path": str(feature_path),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    audit_report = tmp_path / "filtration.json"
    audit_report.write_text(
        json.dumps(
            {
                "report_schema": "mint-agent.filtration-axis-audit.v1",
                "status": "WARN",
                "dataset_id": "toy",
                "evidence_scope": "full_train",
                "representation_hash": "repr-1",
                "sample_ids": ["a"],
                "feature_manifests": {"PL": str(manifest)},
                "invariants": {"PL": {"recommendation": "EXTEND"}},
            }
        ),
        encoding="utf-8",
    )
    ledger = create_job_ledger(
        {
            "run_id": "filtration-run",
            "dataset_id": "toy",
            "filtration_audit_jobs": [
                _filtration_plan(audit_report, manifest)
            ],
        },
        source_plan="filtration-plan.json",
    )
    submit_ready_jobs(
        ledger,
        execute=True,
        runner=lambda command, cwd: CommandResult(0, "Submitted batch job 56\n", ""),
    )
    refresh_job_ledger(
        ledger,
        runner=lambda command, cwd: CommandResult(0, "56|COMPLETED|0:0\n", ""),
    )
    registry_path = tmp_path / "artifacts.sqlite"
    reconcile_completed_jobs(ledger, registry=registry_path)

    assert inspect_filtration_audit_report(audit_report)["report_status"] == "WARN"
    assert ledger["jobs"][0]["state"] == "REGISTERED"
    records = ArtifactRegistry(registry_path).query(
        artifact_kind="filtration_audit",
        dataset_id="toy",
        evidence_scope="full_train",
        invariant="PL",
        representation_hash="repr-1",
    )
    assert len(records) == 1
    assert records[0].status == "WARN"


def test_completed_scout_oof_job_is_validated_and_registered(tmp_path: Path):
    inputs = _write_probe_scout_inputs(tmp_path)
    scout_report = tmp_path / "scout-PL.json"
    scout_report.write_text(
        json.dumps(
            {
                "report_schema": "mint-agent.casf-scout.v1",
                "dataset_id": "toy",
                "evidence_scope": "probe",
                "status": "COMPLETE",
                "representation_hash": "repr-1",
                "selection_hash": "selection-1",
                "sample_ids": ["a", "b"],
                "feature_manifests": {"PL": str(inputs["manifest"])},
                "feature_qc_report": str(inputs["qc"]),
                "filtration_audit_report": str(inputs["audit"]),
                "scout": {"gbt_parameter_hash": "gbt-1"},
            }
        ),
        encoding="utf-8",
    )
    execution = tmp_path / "scout-PL-execution.json"
    ScoutExecutionArtifact(
        artifact_version="mint-agent.scout-execution.v1",
        modeling_sample_ids=("a", "b"),
        representation_hash="repr-1",
        frozen_priority_order=(("PL",),),
        full_fold_assignment={"a": 0, "b": 1},
        target_metric="PCC",
        target_direction="higher",
        target_value=0.5,
        target_source="config",
        gbt_parameter_hash="gbt-1",
        probe_hash="probe-1",
    ).write(execution)
    plan = _scout_oof_plan(
        scout_report,
        execution,
        inputs["manifest"],
        inputs["probe"],
        inputs["qc"],
        inputs["audit"],
    )
    ledger = create_job_ledger(
        {"run_id": "scout-oof-run", "dataset_id": "toy", "scout_jobs": [plan]},
        source_plan="scout-oof-plan.json",
    )
    submit_ready_jobs(
        ledger,
        execute=True,
        runner=lambda command, cwd: CommandResult(0, "Submitted batch job 57\n", ""),
    )
    refresh_job_ledger(
        ledger,
        runner=lambda command, cwd: CommandResult(0, "57|COMPLETED|0:0\n", ""),
    )
    registry_path = tmp_path / "artifacts.sqlite"
    reconcile_completed_jobs(ledger, registry=registry_path)

    assert inspect_scout_oof_report(scout_report)["state"] == "COMPLETE"
    assert ledger["jobs"][0]["state"] == "REGISTERED"
    records = ArtifactRegistry(registry_path).query(
        artifact_kind="scout_oof",
        dataset_id="toy",
        evidence_scope="probe",
        invariant="PL",
        representation_hash="repr-1",
        status="COMPLETE",
    )
    assert len(records) == 1
    assert records[0].selection_hash == "selection-1"
    assert records[0].metadata["gbt_parameter_hash"] == "gbt-1"


def test_completed_scout_combine_job_registers_execution_artifact(tmp_path: Path):
    inputs = _write_probe_scout_inputs(tmp_path)
    scout_report = tmp_path / "scout-PL.json"
    scout_report.write_text(
        json.dumps(
            {
                "report_schema": "mint-agent.casf-scout.v1",
                "dataset_id": "toy",
                "evidence_scope": "probe",
                "status": "COMPLETE",
                "representation_hash": "repr-1",
                "selection_hash": "selection-1",
                "sample_ids": ["a", "b"],
                "feature_manifests": {"PL": str(inputs["manifest"])},
                "feature_qc_report": str(inputs["qc"]),
                "filtration_audit_report": str(inputs["audit"]),
                "scout": {"gbt_parameter_hash": "gbt-1"},
            }
        ),
        encoding="utf-8",
    )
    combined_report = tmp_path / "combined.json"
    combined_report.write_text(
        json.dumps(
            {
                "report_schema": "mint-agent.combined-scout.v1",
                "dataset_id": "toy",
                "evidence_scope": "probe",
                "status": "COMPLETE",
                "representation_hash": "repr-1",
                "selection_hash": "selection-1",
                "sample_ids": ["a", "b"],
                "invariants": ["PL"],
                "source_oof_reports": {"PL": str(scout_report)},
                "gbt_parameter_hash": "gbt-1",
                "ranking_policy": "hierarchical_empirical_v1",
            }
        ),
        encoding="utf-8",
    )
    execution = tmp_path / "scout-execution.json"
    ScoutExecutionArtifact(
        artifact_version="mint-agent.scout-execution.v1",
        modeling_sample_ids=("a", "b"),
        representation_hash="repr-1",
        frozen_priority_order=(("PL",),),
        full_fold_assignment={"a": 0, "b": 1},
        target_metric="PCC",
        target_direction="higher",
        target_value=0.5,
        target_source="config",
        gbt_parameter_hash="gbt-1",
        probe_hash="probe-1",
    ).write(execution)
    plan = {
        "profile_id": "test",
        "scheduler": "slurm",
        "job_kind": "scout_combine",
        "dataset_id": "toy",
        "invariant": "PL",
        "invariants": ["PL"],
        "evidence_scope": "probe",
        "representation_hash": "repr-1",
        "job_name": "mint-scout-combine-pl",
        "working_directory": "/project/mint-agent",
        "script_path": "/project/mint-agent/scripts/slurm/run_scout_combine.sbatch",
        "resources": {},
        "environment": {
            "SOURCE_INVARIANTS": "PL",
            "SCOUT_REPORT_PL": str(scout_report),
            "PROBE_SELECTION": str(inputs["probe"]),
            "FEATURE_QC_REPORT": str(inputs["qc"]),
            "RANKING_POLICY": "hierarchical_empirical_v1",
            "COMBINED_REPORT": str(combined_report),
            "EXECUTION_ARTIFACT": str(execution),
        },
        "input_manifests": {"PL": str(scout_report)},
        "manifest_path": str(execution),
        "stdout_path": "/scratch/scout-combine-%j.out",
        "stderr_path": "/scratch/scout-combine-%j.err",
        "submit_command": ["sbatch", "scout-combine.sbatch"],
    }
    ledger = create_job_ledger(
        {"run_id": "scout-combine-run", "dataset_id": "toy", "scout_jobs": [plan]},
        source_plan="scout-combine-plan.json",
    )
    submit_ready_jobs(
        ledger,
        execute=True,
        runner=lambda command, cwd: CommandResult(0, "Submitted batch job 58\n", ""),
    )
    refresh_job_ledger(
        ledger,
        runner=lambda command, cwd: CommandResult(0, "58|COMPLETED|0:0\n", ""),
    )
    registry_path = tmp_path / "artifacts.sqlite"
    reconcile_completed_jobs(ledger, registry=registry_path)

    assert inspect_scout_combined_report(combined_report)["state"] == "COMPLETE"
    assert ledger["jobs"][0]["state"] == "REGISTERED"
    records = ArtifactRegistry(registry_path).query(
        artifact_kind="scout_execution_plan",
        dataset_id="toy",
        evidence_scope="probe",
        invariant="PL",
        representation_hash="repr-1",
        status="COMPLETE",
    )
    assert len(records) == 1
    assert records[0].selection_hash == "selection-1"
    assert records[0].sample_count == 2


def test_completed_model_evaluation_is_validated_and_registered(tmp_path: Path):
    feature_a = tmp_path / "a.npy"
    feature_b = tmp_path / "b.npy"
    feature_a.write_bytes(b"feature-a")
    feature_b.write_bytes(b"feature-b")
    manifest = tmp_path / "PL-full.jsonl"
    manifest.write_text(
        "\n".join(
            json.dumps(
                {
                    "sample_id": sample_id,
                    "invariant": "PL",
                    "status": "computed",
                    "output_path": str(feature_path),
                }
            )
            for sample_id, feature_path in (("a", feature_a), ("b", feature_b))
        )
        + "\n",
        encoding="utf-8",
    )
    qc_report = tmp_path / "full-qc.json"
    qc_report.write_text(
        json.dumps(
            {
                "report_schema": "mint-agent.feature-qc.v1",
                "status": "PASS",
                "dataset_id": "toy",
                "evidence_scope": "full_train",
                "representation_hash": "repr-1",
                "sample_ids": ["a", "b"],
                "feature_manifests": {"PL": str(manifest)},
            }
        ),
        encoding="utf-8",
    )
    scout = tmp_path / "scout.json"
    ScoutExecutionArtifact(
        artifact_version="mint-agent.scout-execution.v1",
        modeling_sample_ids=("a", "b"),
        representation_hash="repr-1",
        frozen_priority_order=(("PL",), ("PH",)),
        full_fold_assignment={"a": 0, "b": 1},
        target_metric="PCC",
        target_direction="higher",
        target_value=0.5,
        target_source="config",
        gbt_parameter_hash="gbt-1",
        probe_hash="probe-1",
    ).write(scout)
    output = tmp_path / "evaluation.json"
    output.write_text(
        json.dumps(
            {
                "report_schema": "mint-agent.phase4.casf.v1",
                "dataset_id": "toy",
                "task_id": "toy",
                "evidence_scope": "full_train",
                "run_kind": "full_execution",
                "status": "TARGET_REACHED",
                "split": "train",
                "representation_hash": "repr-1",
                "sample_ids": ["a", "b"],
                "feature_manifests": {"PL": str(manifest)},
                "feature_qc_report": str(qc_report),
                "scout_artifact": str(scout),
                "gbt_parameter_hash": "gbt-1",
                "max_acquisitions": 1,
                "target_metric": "PCC",
                "target_value": 0.5,
                "result": {
                    "selected_subset": ["PL"],
                    "evaluation_fold_assignment": {"a": 0, "b": 1},
                },
            }
        ),
        encoding="utf-8",
    )
    plan = {
        "profile_id": "test",
        "scheduler": "slurm",
        "job_kind": "model_evaluation",
        "dataset_id": "toy",
        "invariant": "PL",
        "invariants": ["PL"],
        "evidence_scope": "full_train",
        "representation_hash": "repr-1",
        "job_name": "mint-evaluate-gbt-pl",
        "working_directory": "/project/mint-agent",
        "script_path": "/project/mint-agent/scripts/slurm/run_model_evaluation.sbatch",
        "resources": {},
        "environment": {
            "TASK_CONFIG": "/project/mint-agent/task.yaml",
            "GBT_CONFIG": "/project/mint-agent/gbt.yaml",
            "REPRESENTATION_SPEC": "/project/mint-agent/repr.json",
            "SCOUT_ARTIFACT": str(scout),
            "FEATURE_QC_REPORT": str(qc_report),
            "FEATURE_INVARIANTS": "PL",
            "MAX_ACQUISITIONS": "1",
            "FEATURE_MANIFEST_PL": str(manifest),
            "OUTPUT": str(output),
        },
        "input_manifests": {"PL": str(manifest)},
        "manifest_path": str(output),
        "stdout_path": "/scratch/evaluation-%j.out",
        "stderr_path": "/scratch/evaluation-%j.err",
        "submit_command": ["sbatch", "evaluation.sbatch"],
    }
    ledger = create_job_ledger(
        {"run_id": "evaluation-run", "dataset_id": "toy", "evaluation_jobs": [plan]},
        source_plan="evaluation-plan.json",
    )
    submit_ready_jobs(
        ledger,
        execute=True,
        runner=lambda command, cwd: CommandResult(0, "Submitted batch job 59\n", ""),
    )
    refresh_job_ledger(
        ledger,
        runner=lambda command, cwd: CommandResult(0, "59|COMPLETED|0:0\n", ""),
    )
    registry_path = tmp_path / "artifacts.sqlite"
    reconcile_completed_jobs(ledger, registry=registry_path)

    assert inspect_model_evaluation_report(output)["state"] == "COMPLETE"
    assert ledger["jobs"][0]["state"] == "REGISTERED"
    records = ArtifactRegistry(registry_path).query(
        artifact_kind="model_evaluation",
        dataset_id="toy",
        evidence_scope="full_train",
        invariant="PL",
        representation_hash="repr-1",
        status="TARGET_REACHED",
    )
    assert len(records) == 1
    assert records[0].metadata["gbt_parameter_hash"] == "gbt-1"
