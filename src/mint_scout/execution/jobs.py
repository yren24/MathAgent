from __future__ import annotations

import hashlib
import json
import re
import subprocess
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple

from mint_scout.artifact_memory import (
    EVIDENCE_SCOPES,
    ArtifactRecord,
    ArtifactRegistry,
    inspect_artifact,
)
from mint_scout.execution.profile import feature_job_plan_id
from mint_scout.execution.artifacts import ScoutExecutionArtifact
from mint_scout.invariants.manifest import stable_hash
from mint_scout.representation import RepresentationSpec


LEDGER_SCHEMA = "mint-agent.slurm-job-ledger.v1"
ACTIVE_SLURM_STATES = {
    "CONFIGURING",
    "PENDING",
    "RUNNING",
    "COMPLETING",
    "REQUEUED",
    "RESIZING",
    "SUSPENDED",
}
FAILED_SLURM_STATES = {
    "BOOT_FAIL",
    "CANCELLED",
    "DEADLINE",
    "FAILED",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PREEMPTED",
    "REVOKED",
    "TIMEOUT",
}
RESUMABLE_JOB_STATES = {"PLANNED", "NEEDS_RESUME", "OUTPUT_INCOMPLETE"}


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: str
    stderr: str


CommandRunner = Callable[[Tuple[str, ...], Optional[str]], CommandResult]


def _plans_from_report(plan_report: Mapping[str, Any]) -> list[Any]:
    plans: list[Any] = []
    for key in (
        "preparation_jobs",
        "setup_jobs",
        "feature_jobs",
        "qc_jobs",
        "filtration_audit_jobs",
        "feature_diagnostic_jobs",
        "scout_jobs",
        "evaluation_jobs",
    ):
        value = plan_report.get(key, [])
        if not isinstance(value, list):
            raise ValueError(f"graph report {key} must be a list")
        plans.extend(value)
    if not plans:
        raise ValueError("graph report contains no executable job plans")
    return plans


def create_job_ledger(
    plan_report: Mapping[str, Any], *, source_plan: str | Path
) -> dict[str, Any]:
    raw_plans = _plans_from_report(plan_report)
    now = _utc_now()
    jobs = [
        _new_job(_validated_plan(plan, index)) for index, plan in enumerate(raw_plans)
    ]
    ledger = {
        "report_schema": LEDGER_SCHEMA,
        "source_plan": str(Path(source_plan).expanduser().resolve()),
        "run_id": plan_report.get("run_id"),
        "dataset_id": plan_report.get("dataset_id"),
        "created_at": now,
        "updated_at": now,
        "status": "PLANNED",
        "jobs": jobs,
    }
    _refresh_summary(ledger)
    return ledger


def merge_job_plans(
    ledger: dict[str, Any], plan_report: Mapping[str, Any]
) -> dict[str, Any]:
    _validate_ledger(ledger)
    raw_plans = _plans_from_report(plan_report)
    jobs = _jobs(ledger)
    by_id = {str(job["plan_id"]): job for job in jobs}
    for index, raw_plan in enumerate(raw_plans):
        plan = _validated_plan(raw_plan, index)
        plan_id = _plan_id(plan)
        existing = by_id.get(plan_id)
        if existing is None:
            job = _new_job(plan)
            jobs.append(job)
            by_id[plan_id] = job
        elif existing.get("plan") != plan:
            raise ValueError(f"job plan {plan_id} changed after ledger creation")
    ledger["updated_at"] = _utc_now()
    _refresh_summary(ledger)
    return ledger


def submit_ready_jobs(
    ledger: dict[str, Any], *, execute: bool, runner: CommandRunner | None = None
) -> dict[str, Any]:
    _validate_ledger(ledger)
    command_runner = runner or run_command
    for job in _jobs(ledger):
        if job.get("state") not in RESUMABLE_JOB_STATES:
            continue
        if not execute:
            continue
        _submit_one(job, command_runner)
    ledger["updated_at"] = _utc_now()
    _refresh_summary(ledger)
    return ledger


def refresh_job_ledger(
    ledger: dict[str, Any],
    *,
    query_scheduler: bool = True,
    runner: CommandRunner | None = None,
) -> dict[str, Any]:
    _validate_ledger(ledger)
    command_runner = runner or run_command
    for job in _jobs(ledger):
        attempts = job.get("attempts", [])
        current = attempts[-1] if attempts else None
        if query_scheduler and current and current.get("job_id"):
            scheduler = query_slurm_job(str(current["job_id"]), runner=command_runner)
            current["scheduler_state"] = scheduler["state"]
            current["exit_code"] = scheduler.get("exit_code")
            current["queried_at"] = _utc_now()
            if scheduler.get("error"):
                current["query_error"] = scheduler["error"]
            else:
                current.pop("query_error", None)
            job["scheduler_state"] = scheduler["state"]
        job["manifest"] = inspect_job_output(job["plan"])
        job["state"] = _derive_job_state(job)
    ledger["updated_at"] = _utc_now()
    _refresh_summary(ledger)
    return ledger


def resume_job_ledger(
    ledger: dict[str, Any],
    *,
    execute: bool,
    resubmit_unknown: bool = False,
    runner: CommandRunner | None = None,
) -> dict[str, Any]:
    _validate_ledger(ledger)
    command_runner = runner or run_command
    eligible = set(RESUMABLE_JOB_STATES)
    if resubmit_unknown:
        eligible.add("UNKNOWN")
    for job in _jobs(ledger):
        if job.get("state") not in eligible:
            continue
        if not execute:
            continue
        _submit_one(job, command_runner)
    ledger["updated_at"] = _utc_now()
    _refresh_summary(ledger)
    return ledger


def reconcile_completed_jobs(
    ledger: dict[str, Any], *, registry: ArtifactRegistry | str | Path
) -> dict[str, Any]:
    _validate_ledger(ledger)
    artifact_registry = (
        registry
        if isinstance(registry, ArtifactRegistry)
        else ArtifactRegistry(registry)
    )
    for job in _jobs(ledger):
        if job.get("state") == "REGISTERED":
            continue
        if job.get("state") not in {"COMPLETE", "RECONCILE_FAILED"}:
            continue
        try:
            record = _artifact_record_for_job(job, ledger)
            _assert_no_registered_content_conflict(artifact_registry, record)
            artifact_registry.register(record)
        except (OSError, ValueError) as exc:
            job["state"] = "RECONCILE_FAILED"
            job["reconciliation_error"] = f"{type(exc).__name__}: {exc}"
            continue
        job.pop("reconciliation_error", None)
        job["registration"] = {
            "artifact_id": record.artifact_id,
            "artifact_kind": record.artifact_kind,
            "content_sha256": record.content_sha256,
            "registry_path": str(artifact_registry.path),
            "registered_at": _utc_now(),
        }
        job["state"] = "REGISTERED"
    ledger["updated_at"] = _utc_now()
    _refresh_summary(ledger)
    return ledger


def query_slurm_job(
    job_id: str, *, runner: CommandRunner | None = None
) -> dict[str, Any]:
    if not re.fullmatch(r"\d+(?:_\d+)?", job_id):
        raise ValueError(f"Invalid Slurm job id: {job_id!r}")
    command_runner = runner or run_command
    sacct = command_runner(
        (
            "sacct",
            "-j",
            job_id,
            "--format=JobIDRaw,State,ExitCode",
            "--parsable2",
            "--noheader",
        ),
        None,
    )
    if sacct.returncode == 0:
        parsed = _parse_sacct(sacct.stdout, job_id)
        if parsed is not None:
            return parsed

    squeue = command_runner(("squeue", "-h", "-j", job_id, "-o", "%i|%T"), None)
    if squeue.returncode == 0:
        parsed = _parse_squeue(squeue.stdout, job_id)
        if parsed is not None:
            return parsed
    errors = [value.strip() for value in (sacct.stderr, squeue.stderr) if value.strip()]
    return {
        "state": "UNKNOWN",
        "exit_code": None,
        "error": "; ".join(errors) or "job was not found in sacct or squeue",
    }


def inspect_dataset_audit_report(path: str | Path) -> dict[str, Any]:
    report_path = Path(path).expanduser()
    if not report_path.is_file():
        return {"state": "MISSING", "path": str(report_path), "sample_count": 0}
    try:
        payload = _read_json_object(report_path)
        if payload.get("report_schema") != "mint-agent.dataset-audit.v1":
            raise ValueError("unsupported dataset-audit report schema")
        report_status = str(payload.get("status") or "")
        if report_status not in {"PASS", "REVIEW"}:
            raise ValueError("dataset audit has no valid status")
        sample_ids = tuple(str(value) for value in payload.get("sample_ids", ()))
        if len(set(sample_ids)) != len(sample_ids):
            raise ValueError("dataset-audit sample IDs are duplicated")
        sample_count = int(payload.get("sample_count") or 0)
        if sample_ids and sample_count != len(sample_ids):
            raise ValueError("dataset-audit sample count does not match its IDs")
        if sample_count < 1:
            raise ValueError("dataset audit contains no structures")
        if bool(payload.get("schema_review_required")) != (report_status == "REVIEW"):
            raise ValueError("dataset-audit status disagrees with schema review flag")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return {
            "state": "INVALID",
            "path": str(report_path),
            "sample_count": 0,
            "error": str(exc),
        }
    return {
        "state": "COMPLETE",
        "path": str(report_path),
        "sample_count": sample_count,
        "sample_order_hash": stable_hash(sample_ids) if sample_ids else None,
        "content_sha256": _file_sha256(report_path),
        "dataset_id": payload.get("task_id") or payload.get("dataset_id"),
        "evidence_scope": "design",
        "report_status": report_status,
    }


def inspect_dataset_preparation_report(path: str | Path) -> dict[str, Any]:
    report_path = Path(path).expanduser()
    if not report_path.is_file():
        return {"state": "MISSING", "path": str(report_path), "sample_count": 0}
    try:
        payload = _read_json_object(report_path)
        if payload.get("report_schema") != "mint-agent.dataset-preparation.v1":
            raise ValueError("unsupported dataset-preparation report schema")
        if payload.get("status") != "PASS":
            raise ValueError("dataset preparation has no passing status")
        sample_ids = tuple(str(value) for value in payload.get("sample_ids", ()))
        if not sample_ids or len(set(sample_ids)) != len(sample_ids):
            raise ValueError("dataset-preparation sample IDs are missing or duplicated")
        if int(payload.get("sample_count") or 0) != len(sample_ids):
            raise ValueError("dataset-preparation sample count does not match its IDs")
        manifest_path = Path(str(payload.get("manifest") or "")).expanduser()
        if not manifest_path.is_file():
            raise ValueError("prepared dataset manifest is missing")
        manifest_sha256 = _file_sha256(manifest_path)
        if manifest_sha256 != payload.get("manifest_sha256"):
            raise ValueError("prepared dataset manifest hash changed after preparation")
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        return {
            "state": "INVALID",
            "path": str(report_path),
            "sample_count": 0,
            "error": str(exc),
        }
    return {
        "state": "COMPLETE",
        "path": str(report_path),
        "report_schema": "mint-agent.representation-design.v1",
        "sample_count": len(sample_ids),
        "sample_order_hash": stable_hash(sample_ids),
        "content_sha256": _file_sha256(report_path),
        "dataset_id": payload.get("dataset_id") or payload.get("task_id"),
        "evidence_scope": "design",
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "provider": payload.get("provider"),
        "report_status": "PASS",
    }


def inspect_representation_design_report(path: str | Path) -> dict[str, Any]:
    report_path = Path(path).expanduser()
    if not report_path.is_file():
        return {"state": "MISSING", "path": str(report_path), "sample_count": 0}
    try:
        payload = _read_json_object(report_path)
        schema = payload.get("report_schema")
        if schema == "mint-agent.filtration-repair.v1":
            return _inspect_filtration_repair_representation(report_path, payload)
        if schema != "mint-agent.representation-design.v1":
            raise ValueError("unsupported representation-design report schema")
        sample_ids = tuple(str(value) for value in payload.get("sample_ids", ()))
        if not sample_ids or len(set(sample_ids)) != len(sample_ids):
            raise ValueError(
                "representation-design sample IDs are missing or duplicated"
            )
        if int(payload.get("sample_count") or 0) != len(sample_ids):
            raise ValueError(
                "representation-design sample count does not match its IDs"
            )
        spec = RepresentationSpec.from_dict(
            dict(payload.get("representation_spec", {}))
        )
        spec.assert_frozen()
        if payload.get("representation_hash") != spec.spec_hash:
            raise ValueError(
                "representation-design hash does not match its frozen spec"
            )
        if payload.get("status") not in {None, "COMPLETE"}:
            raise ValueError("representation-design report is not complete")
        design_input_hash = (
            str(payload["design_input_hash"])
            if payload.get("design_input_hash") is not None
            else None
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        return {
            "state": "INVALID",
            "path": str(report_path),
            "sample_count": 0,
            "error": str(exc),
        }
    return {
        "state": "COMPLETE",
        "path": str(report_path),
        "sample_count": len(sample_ids),
        "sample_order_hash": stable_hash(sample_ids),
        "content_sha256": _file_sha256(report_path),
        "dataset_id": payload.get("task_id") or payload.get("dataset_id"),
        "evidence_scope": "design",
        "representation_hash": spec.spec_hash,
        "design_input_hash": design_input_hash,
        "report_status": "COMPLETE",
    }


def _inspect_filtration_repair_representation(
    report_path: Path, payload: Mapping[str, Any]
) -> dict[str, Any]:
    if payload.get("evidence_scope") != "design":
        raise ValueError("filtration repair representation must have evidence_scope=design")
    if str(payload.get("audit_evidence_scope") or "").lower() in {"test", "external_test"}:
        raise ValueError("filtration repair representation must not use test evidence")
    spec = RepresentationSpec.from_dict(dict(payload.get("representation_spec", {})))
    spec.assert_frozen()
    if payload.get("representation_hash") != spec.spec_hash:
        raise ValueError("filtration repair hash does not match its frozen spec")
    if payload.get("status") not in {None, "COMPLETE"}:
        raise ValueError("filtration repair report is not complete")
    sample_ids = tuple(str(value) for value in payload.get("sample_ids", ()))
    if sample_ids and len(set(sample_ids)) != len(sample_ids):
        raise ValueError("filtration repair sample IDs are duplicated")
    return {
        "state": "COMPLETE",
        "path": str(report_path),
        "report_schema": "mint-agent.filtration-repair.v1",
        "sample_count": len(sample_ids) if sample_ids else None,
        "sample_order_hash": stable_hash(sample_ids) if sample_ids else None,
        "content_sha256": _file_sha256(report_path),
        "dataset_id": payload.get("task_id") or payload.get("dataset_id"),
        "evidence_scope": "design",
        "representation_hash": spec.spec_hash,
        "design_input_hash": None,
        "report_status": "COMPLETE",
    }


def inspect_probe_selection_report(path: str | Path) -> dict[str, Any]:
    report_path = Path(path).expanduser()
    if not report_path.is_file():
        return {"state": "MISSING", "path": str(report_path), "sample_count": 0}
    try:
        payload = _read_json_object(report_path)
        if payload.get("report_schema") != "mint-agent.casf-probe-selection.v1":
            raise ValueError("unsupported probe-selection report schema")
        modeling_ids = tuple(
            str(value) for value in payload.get("modeling_sample_ids", ())
        )
        probe_ids = tuple(str(value) for value in payload.get("probe_sample_ids", ()))
        if not modeling_ids or len(set(modeling_ids)) != len(modeling_ids):
            raise ValueError("probe-selection modeling IDs are missing or duplicated")
        if not probe_ids or len(set(probe_ids)) != len(probe_ids):
            raise ValueError("probe-selection sample IDs are missing or duplicated")
        if not set(probe_ids).issubset(modeling_ids):
            raise ValueError("probe selection is not a subset of the modeling pool")
        if not payload.get("representation_hash") or not payload.get("selection_hash"):
            raise ValueError("probe selection lacks representation or selection hash")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return {
            "state": "INVALID",
            "path": str(report_path),
            "sample_count": 0,
            "error": str(exc),
        }
    return {
        "state": "COMPLETE",
        "path": str(report_path),
        "sample_count": len(probe_ids),
        "sample_order_hash": stable_hash(probe_ids),
        "modeling_sample_order_hash": stable_hash(modeling_ids),
        "content_sha256": _file_sha256(report_path),
        "dataset_id": payload.get("task_id") or payload.get("dataset_id"),
        "evidence_scope": "probe",
        "representation_hash": payload.get("representation_hash"),
        "selection_hash": payload.get("selection_hash"),
        "shared_probe_hash": payload.get("shared_probe_hash"),
        "report_status": "COMPLETE",
    }


def inspect_probe_audit_report(path: str | Path) -> dict[str, Any]:
    report_path = Path(path).expanduser()
    if not report_path.is_file():
        return {"state": "MISSING", "path": str(report_path), "sample_count": 0}
    try:
        payload = _read_json_object(report_path)
        if (
            payload.get("report_schema")
            != "mint-agent.probe-representativeness-audit.v1"
        ):
            raise ValueError("unsupported probe-audit report schema")
        report_status = str(payload.get("status") or "")
        if report_status not in {"PASS", "WARN"}:
            raise ValueError("probe audit has no valid status")
        modeling_ids = tuple(
            str(value) for value in payload.get("modeling_sample_ids", ())
        )
        probe_ids = tuple(str(value) for value in payload.get("probe_sample_ids", ()))
        if not modeling_ids or len(set(modeling_ids)) != len(modeling_ids):
            raise ValueError("probe-audit modeling IDs are missing or duplicated")
        if not probe_ids or len(set(probe_ids)) != len(probe_ids):
            raise ValueError("probe-audit sample IDs are missing or duplicated")
        if not set(probe_ids).issubset(modeling_ids):
            raise ValueError("audited probe is not a subset of the modeling pool")
        if not payload.get("representation_hash") or not payload.get("selection_hash"):
            raise ValueError("probe audit lacks representation or selection hash")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return {
            "state": "INVALID",
            "path": str(report_path),
            "sample_count": 0,
            "error": str(exc),
        }
    return {
        "state": "COMPLETE",
        "path": str(report_path),
        "sample_count": len(probe_ids),
        "sample_order_hash": stable_hash(probe_ids),
        "modeling_sample_order_hash": stable_hash(modeling_ids),
        "content_sha256": _file_sha256(report_path),
        "dataset_id": payload.get("task_id") or payload.get("dataset_id"),
        "evidence_scope": "probe",
        "representation_hash": payload.get("representation_hash"),
        "selection_hash": payload.get("selection_hash"),
        "report_status": report_status,
    }


def inspect_feature_manifest(path: str | Path) -> dict[str, Any]:
    manifest = Path(path).expanduser()
    if not manifest.is_file():
        return {"state": "MISSING", "path": str(manifest), "sample_count": 0}
    sample_ids: list[str] = []
    statuses: list[str] = []
    invariants: list[str] = []
    output_paths: list[str] = []
    structure_hashes: list[str] = []
    missing_outputs: list[str] = []
    try:
        with manifest.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"line {line_number} is not a JSON object")
                sample_id = row.get("sample_id")
                status = row.get("status")
                invariant = row.get("invariant")
                if not sample_id or not status or not invariant:
                    raise ValueError(
                        f"line {line_number} lacks sample_id, status, or invariant"
                    )
                sample_ids.append(str(sample_id))
                statuses.append(str(status))
                invariants.append(str(invariant).upper())
                structure_inputs = row.get("structure_inputs")
                if structure_inputs is not None:
                    if not isinstance(structure_inputs, Mapping):
                        raise ValueError(
                            f"line {line_number} structure_inputs is not an object"
                        )
                    combined_hash = structure_inputs.get("combined_sha256")
                    if not isinstance(combined_hash, str) or not re.fullmatch(
                        r"[0-9a-f]{64}", combined_hash
                    ):
                        raise ValueError(
                            f"line {line_number} has invalid combined structure SHA-256"
                        )
                    structure_hashes.append(combined_hash)
                if status in {"cached", "computed"}:
                    output_path = row.get("output_path")
                    if output_path:
                        normalized_output = str(Path(str(output_path)).expanduser())
                        output_paths.append(normalized_output)
                        if not Path(normalized_output).is_file():
                            missing_outputs.append(normalized_output)
                    else:
                        missing_outputs.append(f"{sample_id}:<missing-output-path>")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return {
            "state": "INVALID",
            "path": str(manifest),
            "sample_count": len(sample_ids),
            "error": str(exc),
        }
    if not sample_ids:
        return {"state": "EMPTY", "path": str(manifest), "sample_count": 0}
    if len(set(sample_ids)) != len(sample_ids):
        return {
            "state": "INVALID",
            "path": str(manifest),
            "sample_count": len(sample_ids),
            "error": "duplicate sample IDs",
        }
    invariant_set = sorted(set(invariants))
    if len(invariant_set) != 1:
        return {
            "state": "INVALID",
            "path": str(manifest),
            "sample_count": len(sample_ids),
            "invariants": invariant_set,
            "error": "feature manifest must contain exactly one invariant",
        }
    if len(set(output_paths)) != len(output_paths):
        return {
            "state": "INVALID",
            "path": str(manifest),
            "sample_count": len(sample_ids),
            "invariants": invariant_set,
            "error": "feature manifest contains duplicate output paths",
        }
    if structure_hashes and len(structure_hashes) != len(sample_ids):
        return {
            "state": "INVALID",
            "path": str(manifest),
            "sample_count": len(sample_ids),
            "invariants": invariant_set,
            "error": "feature manifest mixes hashed and unhashed structure inputs",
        }
    failed_count = sum(status == "failed" for status in statuses)
    complete_count = sum(status in {"cached", "computed"} for status in statuses)
    if failed_count:
        state = "FAILED"
    elif complete_count == len(statuses):
        state = "OUTPUT_MISSING" if missing_outputs else "COMPLETE"
    else:
        state = "INCOMPLETE"
    return {
        "state": state,
        "path": str(manifest),
        "sample_count": len(sample_ids),
        "sample_order_hash": stable_hash(tuple(sample_ids)),
        "content_sha256": _file_sha256(manifest),
        "invariants": invariant_set,
        "output_count": len(output_paths),
        "missing_output_count": len(missing_outputs),
        "missing_output_examples": missing_outputs[:10],
        "computed_count": sum(status == "computed" for status in statuses),
        "cached_count": sum(status == "cached" for status in statuses),
        "failed_count": failed_count,
        "structure_input_set_hash": (
            stable_hash(tuple(structure_hashes)) if structure_hashes else None
        ),
    }


def inspect_feature_qc_report(path: str | Path) -> dict[str, Any]:
    report_path = Path(path).expanduser()
    if not report_path.is_file():
        return {"state": "MISSING", "path": str(report_path), "sample_count": 0}
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("QC report must contain a JSON object")
        if payload.get("report_schema") != "mint-agent.feature-qc.v1":
            raise ValueError("unsupported feature-QC report schema")
        if payload.get("status") not in {"PASS", "WARN", "FAIL"}:
            raise ValueError("feature-QC report has no valid status")
        sample_ids = tuple(str(value) for value in payload.get("sample_ids", ()))
        if not sample_ids or len(set(sample_ids)) != len(sample_ids):
            raise ValueError("feature-QC report sample IDs are missing or duplicated")
        manifests = payload.get("feature_manifests")
        if not isinstance(manifests, Mapping) or not manifests:
            raise ValueError("feature-QC report has no feature manifests")
        invariants = sorted(str(name).upper() for name in manifests)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return {
            "state": "INVALID",
            "path": str(report_path),
            "sample_count": 0,
            "error": str(exc),
        }
    return {
        "state": "COMPLETE",
        "path": str(report_path),
        "sample_count": len(sample_ids),
        "sample_order_hash": stable_hash(sample_ids),
        "content_sha256": _file_sha256(report_path),
        "invariants": invariants,
        "report_status": str(payload["status"]),
        "representation_hash": payload.get("representation_hash"),
        "selection_hash": payload.get("selection_hash"),
        "dataset_id": payload.get("dataset_id"),
        "evidence_scope": payload.get("evidence_scope"),
    }


def inspect_filtration_audit_report(path: str | Path) -> dict[str, Any]:
    report_path = Path(path).expanduser()
    if not report_path.is_file():
        return {"state": "MISSING", "path": str(report_path), "sample_count": 0}
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("filtration audit must contain a JSON object")
        if payload.get("report_schema") != "mint-agent.filtration-axis-audit.v1":
            raise ValueError("unsupported filtration-audit report schema")
        if payload.get("status") not in {"PASS", "WARN", "FAIL"}:
            raise ValueError("filtration-audit report has no valid status")
        sample_ids = tuple(str(value) for value in payload.get("sample_ids", ()))
        if not sample_ids or len(set(sample_ids)) != len(sample_ids):
            raise ValueError("filtration-audit sample IDs are missing or duplicated")
        manifests = payload.get("feature_manifests")
        if not isinstance(manifests, Mapping) or not manifests:
            raise ValueError("filtration-audit report has no feature manifests")
        invariants = sorted(str(name).upper() for name in manifests)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return {
            "state": "INVALID",
            "path": str(report_path),
            "sample_count": 0,
            "error": str(exc),
        }
    return {
        "state": "COMPLETE",
        "path": str(report_path),
        "sample_count": len(sample_ids),
        "sample_order_hash": stable_hash(sample_ids),
        "content_sha256": _file_sha256(report_path),
        "invariants": invariants,
        "report_status": str(payload["status"]),
        "representation_hash": payload.get("representation_hash"),
        "selection_hash": payload.get("selection_hash"),
        "dataset_id": payload.get("dataset_id"),
        "evidence_scope": payload.get("evidence_scope"),
    }


def inspect_feature_outlier_diagnostic_report(path: str | Path) -> dict[str, Any]:
    report_path = Path(path).expanduser()
    if not report_path.is_file():
        return {"state": "MISSING", "path": str(report_path), "sample_count": 0}
    try:
        payload = _read_json_object(report_path)
        if payload.get("report_schema") != "mint-agent.feature-outlier-diagnostic.v1":
            raise ValueError("unsupported feature-outlier diagnostic schema")
        if payload.get("status") != "COMPLETE":
            raise ValueError("feature-outlier diagnostic is not complete")
        sample_ids = tuple(str(value) for value in payload.get("sample_ids", ()))
        if not sample_ids or len(set(sample_ids)) != len(sample_ids):
            raise ValueError(
                "feature-outlier diagnostic sample IDs are missing or duplicated"
            )
        invariant = str(payload.get("invariant") or "").upper()
        if not invariant:
            raise ValueError("feature-outlier diagnostic invariant is missing")
        source = payload.get("source")
        if not isinstance(source, Mapping) or not source.get("feature_manifest"):
            raise ValueError("feature-outlier diagnostic source manifest is missing")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return {
            "state": "INVALID",
            "path": str(report_path),
            "sample_count": 0,
            "error": str(exc),
        }
    return {
        "state": "COMPLETE",
        "path": str(report_path),
        "sample_count": len(sample_ids),
        "sample_order_hash": stable_hash(sample_ids),
        "content_sha256": _file_sha256(report_path),
        "invariants": [invariant],
        "report_status": "COMPLETE",
        "representation_hash": payload.get("representation_hash"),
        "selection_hash": payload.get("selection_hash"),
        "dataset_id": payload.get("dataset_id"),
        "evidence_scope": payload.get("evidence_scope"),
        "feature_manifest": source.get("feature_manifest"),
        "feature_qc_hash": source.get("feature_qc_hash"),
        "diagnostic_hash": payload.get("diagnostic_hash"),
    }


def inspect_scout_oof_report(path: str | Path) -> dict[str, Any]:
    report_path = Path(path).expanduser()
    if not report_path.is_file():
        return {"state": "MISSING", "path": str(report_path), "sample_count": 0}
    try:
        payload = _read_json_object(report_path)
        if payload.get("report_schema") != "mint-agent.casf-scout.v1":
            raise ValueError("unsupported Scout OOF report schema")
        if payload.get("status") not in {None, "COMPLETE"}:
            raise ValueError("Scout OOF report is not complete")
        sample_ids = tuple(str(value) for value in payload.get("sample_ids", ()))
        if not sample_ids:
            scout = payload.get("scout")
            if isinstance(scout, Mapping):
                sample_ids = tuple(
                    str(value) for value in scout.get("probe_sample_ids", ())
                )
        if not sample_ids or len(set(sample_ids)) != len(sample_ids):
            raise ValueError("Scout OOF sample IDs are missing or duplicated")
        manifests = payload.get("feature_manifests")
        if not isinstance(manifests, Mapping) or len(manifests) != 1:
            raise ValueError(
                "Scout OOF report must contain exactly one feature manifest"
            )
        invariants = tuple(sorted(str(name).upper() for name in manifests))
        scout = payload.get("scout")
        if not isinstance(scout, Mapping) or not scout.get("gbt_parameter_hash"):
            raise ValueError("Scout OOF report has no GBT parameter hash")
    except (OSError, ValueError, json.JSONDecodeError, KeyError) as exc:
        return {
            "state": "INVALID",
            "path": str(report_path),
            "sample_count": 0,
            "error": str(exc),
        }
    return {
        "state": "COMPLETE",
        "path": str(report_path),
        "sample_count": len(sample_ids),
        "sample_order_hash": stable_hash(sample_ids),
        "content_sha256": _file_sha256(report_path),
        "invariants": list(invariants),
        "representation_hash": payload.get("representation_hash"),
        "dataset_id": payload.get("dataset_id") or payload.get("task_id"),
        "evidence_scope": payload.get("evidence_scope") or "probe",
        "selection_hash": payload.get("selection_hash"),
        "gbt_parameter_hash": scout.get("gbt_parameter_hash"),
    }


def inspect_scout_execution_artifact(path: str | Path) -> dict[str, Any]:
    artifact_path = Path(path).expanduser()
    if not artifact_path.is_file():
        return {"state": "MISSING", "path": str(artifact_path), "sample_count": 0}
    try:
        artifact = ScoutExecutionArtifact.read(artifact_path)
        if artifact.artifact_version != "mint-agent.scout-execution.v1":
            raise ValueError("unsupported Scout execution artifact schema")
        invariants = sorted(
            {
                str(name).upper()
                for subset in artifact.frozen_priority_order
                for name in subset
            }
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        return {
            "state": "INVALID",
            "path": str(artifact_path),
            "sample_count": 0,
            "error": str(exc),
        }
    return {
        "state": "COMPLETE",
        "path": str(artifact_path),
        "sample_count": len(artifact.modeling_sample_ids),
        "sample_order_hash": stable_hash(artifact.modeling_sample_ids),
        "content_sha256": _file_sha256(artifact_path),
        "invariants": invariants,
        "representation_hash": artifact.representation_hash,
        "gbt_parameter_hash": artifact.gbt_parameter_hash,
        "probe_hash": artifact.probe_hash,
        "ranking_policy": artifact.ranking_policy,
        "target_metric": artifact.target_metric,
        "target_value": artifact.target_value,
    }


def inspect_scout_combined_report(path: str | Path) -> dict[str, Any]:
    report_path = Path(path).expanduser()
    if not report_path.is_file():
        return {"state": "MISSING", "path": str(report_path), "sample_count": 0}
    try:
        payload = _read_json_object(report_path)
        if payload.get("report_schema") != "mint-agent.combined-scout.v1":
            raise ValueError("unsupported combined Scout report schema")
        if payload.get("status") not in {None, "COMPLETE"}:
            raise ValueError("combined Scout report is not complete")
        sample_ids = tuple(str(value) for value in payload.get("sample_ids", ()))
        invariants = sorted(
            str(value).upper() for value in payload.get("invariants", ())
        )
        sources = payload.get("source_oof_reports")
        if not sample_ids or len(set(sample_ids)) != len(sample_ids):
            raise ValueError("combined Scout sample IDs are missing or duplicated")
        if not invariants or not isinstance(sources, Mapping):
            raise ValueError("combined Scout invariants or sources are missing")
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        return {
            "state": "INVALID",
            "path": str(report_path),
            "sample_count": 0,
            "error": str(exc),
        }
    return {
        "state": "COMPLETE",
        "path": str(report_path),
        "sample_count": len(sample_ids),
        "sample_order_hash": stable_hash(sample_ids),
        "content_sha256": _file_sha256(report_path),
        "invariants": invariants,
        "representation_hash": payload.get("representation_hash"),
        "dataset_id": payload.get("dataset_id"),
        "evidence_scope": payload.get("evidence_scope") or "probe",
        "selection_hash": payload.get("selection_hash"),
        "gbt_parameter_hash": payload.get("gbt_parameter_hash"),
        "ranking_policy": payload.get("ranking_policy") or "legacy_unspecified",
    }


def inspect_model_evaluation_report(path: str | Path) -> dict[str, Any]:
    report_path = Path(path).expanduser()
    if not report_path.is_file():
        return {"state": "MISSING", "path": str(report_path), "sample_count": 0}
    try:
        payload = _read_json_object(report_path)
        if payload.get("report_schema") != "mint-agent.phase4.casf.v1":
            raise ValueError("unsupported model-evaluation report schema")
        report_status = str(payload.get("status") or "")
        if report_status not in {
            "TARGET_REACHED",
            "TARGET_NOT_REACHED",
            "ACQUISITION_LIMIT_REACHED",
        }:
            raise ValueError("model-evaluation report has no final status")
        if payload.get("run_kind") != "full_execution":
            raise ValueError("engineering smoke output is not a model evaluation")
        sample_ids = tuple(str(value) for value in payload.get("sample_ids", ()))
        if not sample_ids or len(set(sample_ids)) != len(sample_ids):
            raise ValueError("model-evaluation sample IDs are missing or duplicated")
        manifests = payload.get("feature_manifests")
        if not isinstance(manifests, Mapping) or not manifests:
            raise ValueError("model-evaluation report has no feature manifests")
        invariants = sorted(str(name).upper() for name in manifests)
        if not payload.get("gbt_parameter_hash"):
            raise ValueError("model-evaluation report has no GBT parameter hash")
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        return {
            "state": "INVALID",
            "path": str(report_path),
            "sample_count": 0,
            "error": str(exc),
        }
    return {
        "state": "COMPLETE",
        "path": str(report_path),
        "sample_count": len(sample_ids),
        "sample_order_hash": stable_hash(sample_ids),
        "content_sha256": _file_sha256(report_path),
        "invariants": invariants,
        "representation_hash": payload.get("representation_hash"),
        "dataset_id": payload.get("dataset_id") or payload.get("task_id"),
        "evidence_scope": payload.get("evidence_scope"),
        "gbt_parameter_hash": payload.get("gbt_parameter_hash"),
        "target_metric": payload.get("target_metric"),
        "target_value": payload.get("target_value"),
        "report_status": report_status,
    }


def inspect_acceptance_evaluation_report(path: str | Path) -> dict[str, Any]:
    report_path = Path(path).expanduser()
    if not report_path.is_file():
        return {"state": "MISSING", "path": str(report_path), "sample_count": 0}
    try:
        payload = _read_json_object(report_path)
        if (
            payload.get("report_schema")
            == "mint-agent.progressive-acceptance-gbt-evaluation.v1"
        ):
            return _inspect_progressive_acceptance_report(report_path, payload)
        if payload.get("report_schema") != "mint-agent.acceptance-gbt-evaluation.v1":
            raise ValueError("unsupported acceptance-evaluation report schema")
        report_status = str(payload.get("status") or "")
        if report_status not in {
            "TARGET_REACHED",
            "CANDIDATE_REJECTED",
            "TARGET_NOT_REACHED",
        }:
            raise ValueError("acceptance-evaluation report has no valid status")
        if payload.get("evidence_scope") != "acceptance_test":
            raise ValueError(
                "acceptance-evaluation report has the wrong evidence scope"
            )
        protocol = payload.get("protocol")
        if not isinstance(protocol, Mapping) or (
            protocol.get("full_train_cross_validation") is not False
            or protocol.get("evaluation_used_for_candidate_selection") is not True
            or protocol.get("independent_test_available") is not False
        ):
            raise ValueError("acceptance-evaluation protocol guard failed")
        candidate_rank = int(payload.get("candidate_rank") or 0)
        candidate_count = int(payload.get("candidate_count") or 0)
        if (
            candidate_rank < 1
            or candidate_count < candidate_rank
            or int(protocol.get("acceptance_query_index") or 0) != candidate_rank
        ):
            raise ValueError("acceptance-evaluation candidate identity is invalid")
        counts = payload.get("sample_counts")
        hashes = payload.get("sample_order_hashes")
        if not isinstance(counts, Mapping) or any(
            int(counts.get(split) or 0) < 1 for split in ("train", "acceptance")
        ):
            raise ValueError("acceptance-evaluation sample counts are missing")
        if not isinstance(hashes, Mapping) or any(
            not hashes.get(split) for split in ("train", "acceptance")
        ):
            raise ValueError("acceptance-evaluation sample hashes are missing")
        subset = tuple(
            str(value).upper() for value in payload.get("selected_subset", ())
        )
        invariants = tuple(
            str(value).upper() for value in payload.get("invariants", ())
        )
        if (
            not subset
            or len(set(subset)) != len(subset)
            or set(invariants) != set(subset)
        ):
            raise ValueError("acceptance-evaluation invariant subset is invalid")
        manifests = payload.get("feature_manifests")
        if not isinstance(manifests, Mapping):
            raise ValueError("acceptance-evaluation feature manifests are missing")
        for split in ("train", "acceptance"):
            split_manifests = manifests.get(split)
            if not isinstance(split_manifests, Mapping) or set(split_manifests) != set(
                subset
            ):
                raise ValueError(
                    f"acceptance-evaluation {split} manifests are inconsistent"
                )
        if not payload.get("gbt_parameter_hash"):
            raise ValueError("acceptance-evaluation report has no GBT parameter hash")
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        return {
            "state": "INVALID",
            "path": str(report_path),
            "sample_count": 0,
            "error": str(exc),
        }
    return {
        "state": "COMPLETE",
        "path": str(report_path),
        "sample_count": int(counts["acceptance"]),
        "sample_order_hash": str(hashes["acceptance"]),
        "content_sha256": _file_sha256(report_path),
        "invariants": sorted(subset),
        "representation_hash": payload.get("representation_hash"),
        "dataset_id": payload.get("dataset_id"),
        "evidence_scope": "acceptance_test",
        "gbt_parameter_hash": payload.get("gbt_parameter_hash"),
        "target_metric": payload.get("target_metric"),
        "target_value": payload.get("target_value"),
        "report_status": report_status,
        "candidate_rank": candidate_rank,
        "candidate_count": candidate_count,
        "selected_score": payload.get("selected_score"),
        "scout_artifact": payload.get("scout_artifact"),
    }


def _inspect_progressive_acceptance_report(
    report_path: Path, payload: Mapping[str, Any]
) -> dict[str, Any]:
    try:
        report_status = str(payload.get("status") or "")
        if report_status not in {
            "TARGET_REACHED",
            "ACQUISITION_LIMIT_REACHED",
            "TARGET_NOT_REACHED",
            "MAXIMIZATION_COMPLETE",
        }:
            raise ValueError("progressive acceptance report has no valid status")
        if payload.get("evidence_scope") != "acceptance_test":
            raise ValueError("progressive acceptance report has the wrong scope")
        selection_objective = str(payload.get("selection_objective") or "")
        if selection_objective not in {
            "satisfy_target",
            "maximize_rank1",
            "maximize_all",
            "maximize_top_k",
        }:
            raise ValueError("progressive acceptance objective is invalid")
        protocol = payload.get("protocol")
        if not isinstance(protocol, Mapping):
            raise ValueError("progressive acceptance protocol guard failed")
        all_subsets_required = selection_objective != "maximize_top_k"
        if (
            protocol.get("full_train_cross_validation") is not False
            or protocol.get("evaluation_used_for_candidate_selection") is not True
            or protocol.get("independent_test_available") is not False
            or protocol.get("all_available_subsets_scored") is not (
                True if all_subsets_required else False
            )
            or protocol.get("single_invariant_predictions_reused") is not True
        ):
            raise ValueError("progressive acceptance protocol guard failed")
        invariants = tuple(
            str(value).upper() for value in payload.get("invariants", ())
        )
        max_acquisitions = int(payload.get("max_acquisitions") or 0)
        if (
            not invariants
            or len(set(invariants)) != len(invariants)
            or max_acquisitions != len(invariants)
            or int(protocol.get("acceptance_query_index") or 0)
            != max_acquisitions
        ):
            raise ValueError("progressive acceptance acquisition identity is invalid")
        subset = tuple(
            str(value).upper() for value in payload.get("selected_subset", ())
        )
        if not subset or not set(subset).issubset(invariants):
            raise ValueError("progressive acceptance selected subset is invalid")
        counts = payload.get("sample_counts")
        hashes = payload.get("sample_order_hashes")
        if not isinstance(counts, Mapping) or any(
            int(counts.get(split) or 0) < 1 for split in ("train", "acceptance")
        ):
            raise ValueError("progressive acceptance sample counts are missing")
        if not isinstance(hashes, Mapping) or any(
            not hashes.get(split) for split in ("train", "acceptance")
        ):
            raise ValueError("progressive acceptance sample hashes are missing")
        manifests = payload.get("feature_manifests")
        if not isinstance(manifests, Mapping):
            raise ValueError("progressive acceptance manifests are missing")
        for split in ("train", "acceptance"):
            split_manifests = manifests.get(split)
            if not isinstance(split_manifests, Mapping) or {
                str(name).upper() for name in split_manifests
            } != set(invariants):
                raise ValueError(
                    f"progressive acceptance {split} manifests are inconsistent"
                )
        predictions = payload.get("acceptance_predictions")
        if not isinstance(predictions, Mapping) or set(predictions) != set(invariants):
            raise ValueError("progressive acceptance predictions are incomplete")
        if not isinstance(payload.get("objective_complete"), bool):
            raise ValueError("progressive acceptance objective state is missing")
        if not payload.get("gbt_parameter_hash"):
            raise ValueError("progressive acceptance report has no GBT hash")
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        return {
            "state": "INVALID",
            "path": str(report_path),
            "sample_count": 0,
            "error": str(exc),
        }
    return {
        "state": "COMPLETE",
        "path": str(report_path),
        "sample_count": int(counts["acceptance"]),
        "sample_order_hash": str(hashes["acceptance"]),
        "content_sha256": _file_sha256(report_path),
        "invariants": invariants,
        "representation_hash": payload.get("representation_hash"),
        "dataset_id": payload.get("dataset_id"),
        "evidence_scope": "acceptance_test",
        "gbt_parameter_hash": payload.get("gbt_parameter_hash"),
        "target_metric": payload.get("target_metric"),
        "target_value": payload.get("target_value"),
        "report_status": report_status,
        "max_acquisitions": max_acquisitions,
        "selected_subset": subset,
        "selected_score": payload.get("selected_score"),
        "scout_artifact": payload.get("scout_artifact"),
        "selection_objective": selection_objective,
        "candidate_rank_limit": payload.get("candidate_rank_limit"),
    }


def inspect_validation_evaluation_report(path: str | Path) -> dict[str, Any]:
    report_path = Path(path).expanduser()
    if not report_path.is_file():
        return {"state": "MISSING", "path": str(report_path), "sample_count": 0}
    try:
        payload = _read_json_object(report_path)
        if payload.get("report_schema") != "mint-agent.validation-gbt-evaluation.v1":
            raise ValueError("unsupported validation-evaluation report schema")
        report_status = str(payload.get("status") or "")
        if report_status not in {
            "TARGET_REACHED",
            "TARGET_NOT_REACHED",
            "ACQUISITION_LIMIT_REACHED",
            "MAXIMIZATION_COMPLETE",
        }:
            raise ValueError("validation-evaluation report has no final status")
        if payload.get("evidence_scope") != "validation":
            raise ValueError(
                "validation-evaluation report has the wrong evidence scope"
            )
        sample_ids = tuple(str(value) for value in payload.get("sample_ids", ()))
        if not sample_ids or len(set(sample_ids)) != len(sample_ids):
            raise ValueError("validation sample IDs are missing or duplicated")
        manifests = payload.get("feature_manifests")
        if not isinstance(manifests, Mapping):
            raise ValueError("validation-evaluation feature manifests are missing")
        train = manifests.get("train")
        validation = manifests.get("validation")
        if not isinstance(train, Mapping) or not isinstance(validation, Mapping):
            raise ValueError("validation-evaluation split manifests are missing")
        invariants = tuple(
            str(value).upper() for value in payload.get("invariants", ())
        )
        if (
            not invariants
            or set(train) != set(invariants)
            or set(validation) != set(invariants)
        ):
            raise ValueError(
                "validation-evaluation invariant manifests are inconsistent"
            )
        selection = payload.get("selection")
        if not isinstance(selection, Mapping) or not selection.get("selected_subset"):
            raise ValueError("validation-evaluation selection is missing")
        selected_subset = {str(value).upper() for value in selection["selected_subset"]}
        if not selected_subset.issubset(set(invariants)):
            raise ValueError(
                "validation selected subset is outside acquired invariants"
            )
        if not payload.get("gbt_parameter_hash"):
            raise ValueError("validation-evaluation report has no GBT parameter hash")
        selection_objective = str(
            payload.get("selection_objective") or "satisfy_target"
        )
        if selection_objective not in {
            "satisfy_target",
            "maximize_rank1",
            "maximize_all",
        }:
            raise ValueError("validation-evaluation objective is invalid")
        reuse = payload.get("prediction_reuse")
        if not isinstance(reuse, Mapping):
            raise ValueError("validation-evaluation prediction reuse record is missing")
        reused = tuple(
            str(value).upper() for value in reuse.get("reused_invariants", ())
        )
        newly_fitted = tuple(
            str(value).upper() for value in reuse.get("newly_fitted_invariants", ())
        )
        if reused + newly_fitted != invariants or len(newly_fitted) != 1:
            raise ValueError(
                "validation-evaluation prediction reuse prefix is inconsistent"
            )
        if bool(reused) != bool(reuse.get("prior_evaluation_report")):
            raise ValueError(
                "validation-evaluation prior report provenance is inconsistent"
            )
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        return {
            "state": "INVALID",
            "path": str(report_path),
            "sample_count": 0,
            "error": str(exc),
        }
    return {
        "state": "COMPLETE",
        "path": str(report_path),
        "sample_count": len(sample_ids),
        "sample_order_hash": stable_hash(sample_ids),
        "content_sha256": _file_sha256(report_path),
        "invariants": invariants,
        "representation_hash": payload.get("representation_hash"),
        "dataset_id": payload.get("dataset_id"),
        "evidence_scope": "validation",
        "selection_hash": payload.get("selection_hash"),
        "selected_subset": [
            str(value).upper() for value in selection["selected_subset"]
        ],
        "gbt_parameter_hash": payload.get("gbt_parameter_hash"),
        "target_metric": payload.get("target_metric"),
        "target_value": payload.get("target_value"),
        "report_status": report_status,
        "selection_objective": selection_objective,
    }


def inspect_frozen_test_evaluation_report(path: str | Path) -> dict[str, Any]:
    report_path = Path(path).expanduser()
    if not report_path.is_file():
        return {"state": "MISSING", "path": str(report_path), "sample_count": 0}
    try:
        payload = _read_json_object(report_path)
        if payload.get("report_schema") != "mint-agent.frozen-test-gbt-evaluation.v1":
            raise ValueError("unsupported frozen-test report schema")
        report_status = str(payload.get("status") or "")
        if (
            report_status not in {"TARGET_REACHED", "TARGET_NOT_REACHED"}
            or payload.get("evidence_scope") != "external_test"
        ):
            raise ValueError("frozen-test report is not a complete external evaluation")
        protocol = payload.get("protocol")
        if not isinstance(protocol, Mapping):
            raise ValueError("frozen-test protocol record is missing")
        if (
            protocol.get("test_used_for_selection") is not False
            or protocol.get("selection_frozen_before_test_feature_loading") is not True
            or int(protocol.get("test_query_count") or 0) != 1
        ):
            raise ValueError("frozen-test protocol guard failed")
        counts = payload.get("sample_counts")
        hashes = payload.get("sample_order_hashes")
        if not isinstance(counts, Mapping) or int(counts.get("test") or 0) < 1:
            raise ValueError("frozen-test sample count is missing")
        if not isinstance(hashes, Mapping) or not hashes.get("test"):
            raise ValueError("frozen-test sample order hash is missing")
        subset = tuple(
            str(value).upper() for value in payload.get("selected_subset", ())
        )
        if not subset or len(set(subset)) != len(subset):
            raise ValueError("frozen-test selected subset is missing or duplicated")
        manifests = payload.get("feature_manifests")
        if not isinstance(manifests, Mapping):
            raise ValueError("frozen-test feature manifests are missing")
        for split in ("train", "test"):
            split_manifests = manifests.get(split)
            if not isinstance(split_manifests, Mapping) or set(split_manifests) != set(
                subset
            ):
                raise ValueError(
                    f"frozen-test {split} manifests do not match the frozen subset"
                )
        if not payload.get("gbt_parameter_hash") or not payload.get(
            "selection_report_sha256"
        ):
            raise ValueError("frozen-test model or selection identity is missing")
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        return {
            "state": "INVALID",
            "path": str(report_path),
            "sample_count": 0,
            "error": str(exc),
        }
    return {
        "state": "COMPLETE",
        "path": str(report_path),
        "sample_count": int(counts["test"]),
        "sample_order_hash": str(hashes["test"]),
        "content_sha256": _file_sha256(report_path),
        "invariants": sorted(subset),
        "representation_hash": payload.get("representation_hash"),
        "dataset_id": payload.get("dataset_id"),
        "evidence_scope": "external_test",
        "selection_hash": payload.get("selection_hash"),
        "selected_subset": list(subset),
        "gbt_parameter_hash": payload.get("gbt_parameter_hash"),
        "report_status": report_status,
    }


def inspect_job_output(plan: Mapping[str, Any]) -> dict[str, Any]:
    if plan.get("job_kind") == "dataset_preparation":
        return inspect_dataset_preparation_report(plan["manifest_path"])
    if plan.get("job_kind") == "dataset_audit":
        return inspect_dataset_audit_report(plan["manifest_path"])
    if plan.get("job_kind") == "representation_design":
        return inspect_representation_design_report(plan["manifest_path"])
    if plan.get("job_kind") == "probe_selection":
        return inspect_probe_selection_report(plan["manifest_path"])
    if plan.get("job_kind") == "probe_audit":
        return inspect_probe_audit_report(plan["manifest_path"])
    if plan.get("job_kind") == "feature_qc":
        return inspect_feature_qc_report(plan["manifest_path"])
    if plan.get("job_kind") == "filtration_audit":
        return inspect_filtration_audit_report(plan["manifest_path"])
    if plan.get("job_kind") == "feature_outlier_diagnostic":
        return inspect_feature_outlier_diagnostic_report(plan["manifest_path"])
    if plan.get("job_kind") == "scout_oof":
        report = inspect_scout_oof_report(plan["manifest_path"])
        auxiliary = inspect_scout_execution_artifact(
            str(plan.get("environment", {}).get("EXECUTION_ARTIFACT", ""))
        )
        if report.get("state") == "COMPLETE" and auxiliary.get("state") != "COMPLETE":
            return {
                **report,
                "state": "INCOMPLETE",
                "error": "Scout OOF execution artifact is missing or invalid",
            }
        return {**report, "execution_artifact": auxiliary}
    if plan.get("job_kind") == "scout_combine":
        artifact = inspect_scout_execution_artifact(plan["manifest_path"])
        combined = inspect_scout_combined_report(
            str(plan.get("environment", {}).get("COMBINED_REPORT", ""))
        )
        if artifact.get("state") == "COMPLETE" and combined.get("state") != "COMPLETE":
            return {
                **artifact,
                "state": "INCOMPLETE",
                "error": "combined Scout report is missing or invalid",
            }
        return {**artifact, "combined_report": combined}
    if plan.get("job_kind") == "model_evaluation":
        return inspect_model_evaluation_report(plan["manifest_path"])
    if plan.get("job_kind") == "acceptance_evaluation":
        return inspect_acceptance_evaluation_report(plan["manifest_path"])
    if plan.get("job_kind") == "validation_evaluation":
        return inspect_validation_evaluation_report(plan["manifest_path"])
    if plan.get("job_kind") == "frozen_test_evaluation":
        return inspect_frozen_test_evaluation_report(plan["manifest_path"])
    return inspect_feature_manifest(plan["manifest_path"])


def load_job_ledger(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("job ledger must contain a JSON object")
    _validate_ledger(payload)
    return payload


def write_job_ledger(ledger: Mapping[str, Any], path: str | Path) -> None:
    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(
        json.dumps(ledger, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(destination)


def ledger_summary(ledger: Mapping[str, Any]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for job in ledger.get("jobs", []):
        state = str(job.get("state", "UNKNOWN"))
        counts[state] = counts.get(state, 0) + 1
    return {
        "report_schema": ledger.get("report_schema"),
        "status": ledger.get("status"),
        "job_count": sum(counts.values()),
        "state_counts": dict(sorted(counts.items())),
    }


def run_command(command: tuple[str, ...], cwd: str | None) -> CommandResult:
    try:
        completed = subprocess.run(
            command, cwd=cwd, check=False, capture_output=True, text=True,
        )
    except OSError as exc:
        return CommandResult(returncode=127, stdout="", stderr=str(exc))
    return CommandResult(
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def _submit_one(job: dict[str, Any], runner: CommandRunner) -> None:
    plan = _validated_plan(job.get("plan"), 0)
    if str(job.get("plan_id") or "") != plan["plan_id"]:
        raise ValueError("job ledger plan_id does not match the embedded job plan")
    job["plan"] = plan
    command = tuple(str(value) for value in plan["submit_command"])
    if not _is_slurm_submit_command(command) or plan.get("scheduler") != "slurm":
        raise ValueError(
            f"job plan {job['plan_id']} is not an auditable Slurm submission"
        )
    result = runner(command, str(plan["working_directory"]))
    submitted_at = _utc_now()
    attempt = {
        "attempt": len(job.get("attempts", [])) + 1,
        "submitted_at": submitted_at,
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
        "job_id": None,
        "scheduler_state": "SUBMIT_FAILED",
        "exit_code": None,
    }
    if result.returncode == 0:
        try:
            attempt["job_id"] = parse_sbatch_job_id(result.stdout)
        except ValueError as exc:
            attempt["stderr"] = str(exc)
        else:
            attempt["scheduler_state"] = "SUBMITTED"
    job.setdefault("attempts", []).append(attempt)
    job["scheduler_state"] = attempt["scheduler_state"]
    job["state"] = _derive_job_state(job)


def parse_sbatch_job_id(output: str) -> str:
    match = re.search(r"\bSubmitted\s+batch\s+job\s+(\d+(?:_\d+)?)\b", output)
    if match is None:
        raise ValueError(f"Could not parse Slurm job id from sbatch output: {output!r}")
    return match.group(1)


def _parse_sacct(output: str, job_id: str) -> dict[str, Any] | None:
    fallback = None
    for line in output.splitlines():
        fields = line.strip().split("|")
        if len(fields) < 2 or not fields[0]:
            continue
        row_id = fields[0]
        parsed = {
            "state": _normalize_slurm_state(fields[1]),
            "exit_code": fields[2] if len(fields) > 2 and fields[2] else None,
        }
        if row_id == job_id:
            return parsed
        if row_id.split(".", 1)[0] == job_id and fallback is None:
            fallback = parsed
    return fallback


def _parse_squeue(output: str, job_id: str) -> dict[str, Any] | None:
    for line in output.splitlines():
        fields = line.strip().split("|", 1)
        if len(fields) == 2 and fields[0] == job_id:
            return {"state": _normalize_slurm_state(fields[1]), "exit_code": None}
    return None


def _normalize_slurm_state(value: str) -> str:
    return value.strip().upper().split()[0].rstrip("+") if value.strip() else "UNKNOWN"


def _new_job(plan: dict[str, Any]) -> dict[str, Any]:
    plan_id = _plan_id(plan)
    plan["plan_id"] = plan_id
    return {
        "plan_id": plan_id,
        "plan": plan,
        "attempts": [],
        "scheduler_state": "NOT_SUBMITTED",
        "manifest": inspect_job_output(plan),
        "state": "PLANNED",
    }


def _validated_plan(raw_plan: Any, index: int) -> dict[str, Any]:
    if not isinstance(raw_plan, Mapping):
        raise ValueError(f"job plan {index} must be an object")
    plan = dict(raw_plan)
    required = {
        "scheduler",
        "job_kind",
        "dataset_id",
        "invariant",
        "evidence_scope",
        "representation_hash",
        "working_directory",
        "manifest_path",
        "submit_command",
    }
    missing = sorted(required - set(plan))
    if missing:
        raise ValueError(f"job plan {index} lacks required fields: {missing}")
    command = plan["submit_command"]
    if (
        plan["scheduler"] != "slurm"
        or not isinstance(command, Sequence)
        or isinstance(command, (str, bytes))
        or not command
        or not _is_slurm_submit_command(tuple(str(value) for value in command))
    ):
        raise ValueError(f"job plan {index} is not a Slurm sbatch plan")
    plan["submit_command"] = [str(value) for value in command]
    for key in ("dataset_id", "invariant", "evidence_scope", "representation_hash"):
        if not str(plan.get(key) or ""):
            raise ValueError(f"job plan {index}.{key} must be nonempty")
    plan["invariant"] = str(plan["invariant"]).upper()
    if plan["evidence_scope"] not in EVIDENCE_SCOPES:
        raise ValueError(
            f"job plan {index}.evidence_scope is invalid: {plan['evidence_scope']!r}"
        )
    environment = plan.get("environment")
    if not isinstance(environment, Mapping):
        raise ValueError(f"job plan {index}.environment must be an object")
    raw_invariants = plan.get("invariants", (plan["invariant"],))
    if (
        not isinstance(raw_invariants, Sequence)
        or isinstance(raw_invariants, (str, bytes))
        or not raw_invariants
    ):
        raise ValueError(f"job plan {index}.invariants must be a nonempty list")
    plan["invariants"] = sorted({str(value).upper() for value in raw_invariants})
    job_kind = str(plan.get("job_kind"))
    if job_kind in {
        "dataset_preparation",
        "dataset_audit",
        "representation_design",
        "probe_selection",
        "probe_audit",
    }:
        expected_invariant = {
            "dataset_preparation": "DATASET",
            "dataset_audit": "DATASET",
            "representation_design": "REPRESENTATION",
            "probe_selection": "PROBE",
            "probe_audit": "PROBE",
        }[job_kind]
        if plan["invariant"] != expected_invariant or plan["invariants"] != [
            expected_invariant
        ]:
            raise ValueError(f"job plan {index} has an invalid setup invariant")
        required_environment = {
            "dataset_preparation": ("TASK_CONFIG", "OUTPUT_REPORT"),
            "dataset_audit": ("TASK_CONFIG", "OUTPUT_JSON"),
            "representation_design": ("TASK_CONFIG", "DATA_AUDIT_REPORT", "REPORT",),
            "probe_selection": (
                "TASK_CONFIG",
                "SCOUT_CONFIG",
                "REPRESENTATION_SPEC",
                "OUTPUT",
            ),
            "probe_audit": (
                "TASK_CONFIG",
                "SCOUT_CONFIG",
                "PROBE_SELECTION",
                "OUTPUT",
            ),
        }[job_kind]
        for key in required_environment:
            if not str(environment.get(key) or ""):
                raise ValueError(f"job plan {index} setup job has no {key}")
        output_key = {
            "dataset_preparation": "OUTPUT_REPORT",
            "dataset_audit": "OUTPUT_JSON",
            "representation_design": "REPORT",
            "probe_selection": "OUTPUT",
            "probe_audit": "OUTPUT",
        }[job_kind]
        if str(environment[output_key]) != str(plan["manifest_path"]):
            raise ValueError(f"job plan {index} setup output is inconsistent")
        raw_inputs = plan.get("input_manifests", {})
        if not isinstance(raw_inputs, Mapping):
            raise ValueError(f"job plan {index}.input_manifests must be an object")
        normalized_inputs = {
            str(name).upper(): str(path) for name, path in raw_inputs.items()
        }
        expected_inputs = {
            "dataset_preparation": {},
            "dataset_audit": {},
            "representation_design": {
                "DATA_AUDIT": str(environment.get("DATA_AUDIT_REPORT") or "")
            },
            "probe_selection": {
                "REPRESENTATION": str(environment.get("REPRESENTATION_SPEC") or "")
            },
            "probe_audit": {
                "PROBE_SELECTION": str(environment.get("PROBE_SELECTION") or "")
            },
        }[job_kind]
        if job_kind == "probe_selection" and environment.get("FROZEN_SAMPLE_SOURCE"):
            expected_inputs["FROZEN_SAMPLE_SOURCE"] = str(
                environment["FROZEN_SAMPLE_SOURCE"]
            )
        if normalized_inputs != expected_inputs:
            raise ValueError(f"job plan {index} setup inputs are inconsistent")
        plan["input_manifests"] = normalized_inputs
    elif job_kind == "feature_batch":
        if plan["invariants"] != [plan["invariant"]]:
            raise ValueError(f"job plan {index} feature invariant list is inconsistent")
        if str(environment.get("INVARIANT") or "").upper() != plan["invariant"]:
            raise ValueError(
                f"job plan {index} invariant disagrees with its environment"
            )
        if plan["evidence_scope"] == "probe":
            if not str(environment.get("SAMPLE_ID_FILE") or ""):
                raise ValueError(f"job plan {index} probe feature has no sample source")
            if not str(environment.get("SELECTION_HASH") or ""):
                raise ValueError(
                    f"job plan {index} probe feature has no selection hash"
                )
        plan["input_manifests"] = {}
    elif job_kind in {
        "feature_qc",
        "filtration_audit",
        "feature_outlier_diagnostic",
        "scout_oof",
    }:
        input_manifests = plan.get("input_manifests")
        if not isinstance(input_manifests, Mapping) or not input_manifests:
            raise ValueError(
                f"job plan {index}.input_manifests must be a nonempty object"
            )
        normalized_manifests = {
            str(name).upper(): str(path) for name, path in input_manifests.items()
        }
        if sorted(normalized_manifests) != plan["invariants"]:
            raise ValueError(
                f"job plan {index} analysis invariants do not match its manifests"
            )
        environment_invariants = sorted(
            value.upper()
            for value in str(environment.get("FEATURE_INVARIANTS") or "").split(":")
            if value
        )
        if environment_invariants != plan["invariants"]:
            raise ValueError(
                f"job plan {index} analysis invariants disagree with its environment"
            )
        for name, path in normalized_manifests.items():
            if str(environment.get(f"FEATURE_MANIFEST_{name}") or "") != path:
                raise ValueError(
                    f"job plan {index} analysis manifest disagrees with its environment"
                )
        if str(environment.get("OUTPUT") or "") != str(plan["manifest_path"]):
            raise ValueError(
                f"job plan {index} analysis output disagrees with its environment"
            )
        if (
            job_kind
            in {"feature_qc", "filtration_audit", "feature_outlier_diagnostic",}
            and plan["evidence_scope"] == "probe"
            and not str(environment.get("PROBE_SELECTION") or "")
        ):
            raise ValueError(f"job plan {index} probe analysis has no frozen selection")
        plan["input_manifests"] = normalized_manifests
        if job_kind == "scout_oof" and len(plan["invariants"]) != 1:
            raise ValueError(
                f"job plan {index} Scout OOF job must contain one invariant"
            )
        if job_kind == "feature_outlier_diagnostic":
            if len(plan["invariants"]) != 1:
                raise ValueError(
                    f"job plan {index} feature diagnostic must contain one invariant"
                )
            for key in ("FEATURE_QC_REPORT", "REPRESENTATION_SPEC"):
                if not str(environment.get(key) or ""):
                    raise ValueError(
                        f"job plan {index} feature diagnostic has no {key}"
                    )
    elif job_kind == "scout_combine":
        input_reports = plan.get("input_manifests")
        if not isinstance(input_reports, Mapping) or not input_reports:
            raise ValueError(
                f"job plan {index}.input_manifests must contain Scout reports"
            )
        normalized_reports = {
            str(name).upper(): str(path) for name, path in input_reports.items()
        }
        if sorted(normalized_reports) != plan["invariants"]:
            raise ValueError(
                f"job plan {index} Scout reports do not match its invariants"
            )
        environment_invariants = sorted(
            value.upper()
            for value in str(environment.get("SOURCE_INVARIANTS") or "").split(":")
            if value
        )
        if environment_invariants != plan["invariants"]:
            raise ValueError(
                f"job plan {index} Scout sources disagree with its environment"
            )
        for name, path in normalized_reports.items():
            if str(environment.get(f"SCOUT_REPORT_{name}") or "") != path:
                raise ValueError(f"job plan {index} Scout source path is inconsistent")
        if str(environment.get("EXECUTION_ARTIFACT") or "") != str(
            plan["manifest_path"]
        ):
            raise ValueError(f"job plan {index} Scout execution output is inconsistent")
        if not str(environment.get("COMBINED_REPORT") or ""):
            raise ValueError(f"job plan {index} has no combined Scout report path")
        if str(environment.get("RANKING_POLICY") or "") != (
            "hierarchical_empirical_v1"
        ):
            raise ValueError(
                f"job plan {index} has no supported Scout ranking policy"
            )
        plan["input_manifests"] = normalized_reports
    elif job_kind == "model_evaluation":
        if plan["evidence_scope"] != "full_train":
            raise ValueError(f"job plan {index} model evaluation must be full_train")
        input_manifests = plan.get("input_manifests")
        if not isinstance(input_manifests, Mapping) or not input_manifests:
            raise ValueError(f"job plan {index}.input_manifests must contain features")
        normalized_manifests = {
            str(name).upper(): str(path) for name, path in input_manifests.items()
        }
        if sorted(normalized_manifests) != plan["invariants"]:
            raise ValueError(
                f"job plan {index} evaluation features do not match its invariants"
            )
        environment_invariants = sorted(
            value.upper()
            for value in str(environment.get("FEATURE_INVARIANTS") or "").split(":")
            if value
        )
        if environment_invariants != plan["invariants"]:
            raise ValueError(
                f"job plan {index} evaluation invariants disagree with its environment"
            )
        for name, path in normalized_manifests.items():
            if str(environment.get(f"FEATURE_MANIFEST_{name}") or "") != path:
                raise ValueError(
                    f"job plan {index} evaluation manifest disagrees with its environment"
                )
        for key in (
            "TASK_CONFIG",
            "GBT_CONFIG",
            "REPRESENTATION_SPEC",
            "SCOUT_ARTIFACT",
            "FEATURE_QC_REPORT",
        ):
            if not str(environment.get(key) or ""):
                raise ValueError(f"job plan {index} evaluation has no {key}")
        try:
            max_acquisitions = int(environment.get("MAX_ACQUISITIONS") or 0)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"job plan {index} evaluation acquisition limit is invalid"
            ) from exc
        if max_acquisitions != len(plan["invariants"]):
            raise ValueError(
                f"job plan {index} evaluation acquisition limit is inconsistent"
            )
        if str(environment.get("OUTPUT") or "") != str(plan["manifest_path"]):
            raise ValueError(f"job plan {index} evaluation output is inconsistent")
        plan["input_manifests"] = normalized_manifests
    elif job_kind == "acceptance_evaluation":
        if plan["evidence_scope"] != "acceptance_test":
            raise ValueError(
                f"job plan {index} acceptance evaluation must use acceptance_test evidence"
            )
        input_manifests = plan.get("input_manifests")
        if not isinstance(input_manifests, Mapping) or not input_manifests:
            raise ValueError(
                f"job plan {index}.input_manifests must contain acceptance features"
            )
        normalized_manifests = {
            str(name).upper(): str(path) for name, path in input_manifests.items()
        }
        expected_inputs = {
            f"{split}_{name}"
            for name in plan["invariants"]
            for split in ("TRAIN", "EVALUATION")
        }
        if set(normalized_manifests) != expected_inputs:
            raise ValueError(
                f"job plan {index} acceptance feature inputs are inconsistent"
            )
        for key, path in normalized_manifests.items():
            if str(environment.get(f"FEATURE_MANIFEST_{key}") or "") != path:
                raise ValueError(
                    f"job plan {index} acceptance feature environment is inconsistent"
                )
        environment_invariants = {
            value.upper()
            for value in str(environment.get("FEATURE_INVARIANTS") or "").split(":")
            if value
        }
        if environment_invariants != set(plan["invariants"]):
            raise ValueError(
                f"job plan {index} acceptance invariant list is inconsistent"
            )
        for key in (
            "TASK_CONFIG",
            "GBT_CONFIG",
            "REPRESENTATION_SPEC",
            "SCOUT_ARTIFACT",
            "TRAIN_QC_REPORT",
            "EVALUATION_QC_REPORT",
        ):
            if not str(environment.get(key) or ""):
                raise ValueError(f"job plan {index} acceptance evaluation has no {key}")
        if str(environment.get("ACCEPTANCE_MODE") or "") == "progressive":
            try:
                max_acquisitions = int(environment.get("MAX_ACQUISITIONS") or 0)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"job plan {index} acceptance acquisition limit is invalid"
                ) from exc
            if max_acquisitions != len(plan["invariants"]):
                raise ValueError(
                    f"job plan {index} acceptance acquisition limit is inconsistent"
                )
        else:
            try:
                candidate_rank = int(environment.get("CANDIDATE_RANK") or 0)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"job plan {index} acceptance candidate rank is invalid"
                ) from exc
            if candidate_rank < 1:
                raise ValueError(
                    f"job plan {index} acceptance candidate rank is invalid"
                )
        if str(environment.get("OUTPUT") or "") != str(plan["manifest_path"]):
            raise ValueError(
                f"job plan {index} acceptance evaluation output is inconsistent"
            )
        plan["input_manifests"] = normalized_manifests
    elif job_kind in {"validation_evaluation", "frozen_test_evaluation"}:
        expected_scope = (
            "validation" if job_kind == "validation_evaluation" else "external_test"
        )
        if plan["evidence_scope"] != expected_scope:
            raise ValueError(
                f"job plan {index} {job_kind} must use {expected_scope} evidence"
            )
        input_manifests = plan.get("input_manifests")
        if not isinstance(input_manifests, Mapping) or not input_manifests:
            raise ValueError(
                f"job plan {index}.input_manifests must contain split features"
            )
        normalized_manifests = {
            str(name).upper(): str(path) for name, path in input_manifests.items()
        }
        expected_inputs = {
            f"{split}_{name}"
            for name in plan["invariants"]
            for split in (
                ("TRAIN", "VALIDATION")
                if job_kind == "validation_evaluation"
                else ("TRAIN", "TEST")
            )
        }
        validation_keys = {f"VALIDATION_{name}" for name in plan["invariants"]}
        if job_kind == "validation_evaluation" and environment.get(
            "PRIOR_EVALUATION_REPORT"
        ):
            expected_inputs.add("PRIOR_EVALUATION")
        if job_kind == "frozen_test_evaluation" and environment.get(
            "VALIDATION_QC_REPORT"
        ):
            expected_inputs |= validation_keys
        if set(normalized_manifests) != expected_inputs:
            raise ValueError(f"job plan {index} split feature inputs are inconsistent")
        for key, path in normalized_manifests.items():
            environment_key = (
                "PRIOR_EVALUATION_REPORT"
                if key == "PRIOR_EVALUATION"
                else f"FEATURE_MANIFEST_{key}"
            )
            if str(environment.get(environment_key) or "") != path:
                raise ValueError(
                    f"job plan {index} split feature environment is inconsistent"
                )
        environment_invariants = {
            value.upper()
            for value in str(environment.get("FEATURE_INVARIANTS") or "").split(":")
            if value
        }
        if environment_invariants != set(plan["invariants"]):
            raise ValueError(
                f"job plan {index} split evaluation invariant list is inconsistent"
            )
        required_environment = [
            "TASK_CONFIG",
            "GBT_CONFIG",
            "REPRESENTATION_SPEC",
            "TRAIN_QC_REPORT",
        ]
        if job_kind == "validation_evaluation":
            required_environment.extend(
                ["SCOUT_ARTIFACT", "VALIDATION_QC_REPORT", "MAX_ACQUISITIONS"]
            )
            try:
                max_acquisitions = int(environment.get("MAX_ACQUISITIONS") or 0)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"job plan {index} validation acquisition limit is invalid"
                ) from exc
            if max_acquisitions != len(plan["invariants"]):
                raise ValueError(
                    f"job plan {index} validation acquisition limit is inconsistent"
                )
            prior = str(environment.get("PRIOR_EVALUATION_REPORT") or "")
            if len(plan["invariants"]) == 1 and prior:
                raise ValueError(
                    f"job plan {index} first validation stage cannot reuse predictions"
                )
            if len(plan["invariants"]) > 1 and not prior:
                raise ValueError(
                    f"job plan {index} later validation stage must reuse predictions"
                )
        else:
            required_environment.extend(
                ["SELECTION_REPORT", "TEST_QC_REPORT", "N_BOOTSTRAP", "BOOTSTRAP_SEED"]
            )
        for key in required_environment:
            if not str(environment.get(key) or ""):
                raise ValueError(f"job plan {index} split evaluation has no {key}")
        if str(environment.get("OUTPUT") or "") != str(plan["manifest_path"]):
            raise ValueError(
                f"job plan {index} split evaluation output is inconsistent"
            )
        plan["input_manifests"] = normalized_manifests
    else:
        raise ValueError(f"job plan {index} has unsupported job_kind {job_kind!r}")
    expected_plan_id = feature_job_plan_id(plan)
    if plan.get("plan_id") not in {None, expected_plan_id}:
        raise ValueError(f"job plan {index}.plan_id does not match its contents")
    plan["plan_id"] = expected_plan_id
    return plan


def _plan_id(plan: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in plan.items() if key != "plan_id"}
    return str(plan.get("plan_id") or stable_hash(payload))


def _is_slurm_submit_command(command: Sequence[str]) -> bool:
    if not command:
        return False
    if command[0] == "sbatch":
        return True
    if command[0] != "env":
        return False
    return "sbatch" in command[1:]


def _derive_job_state(job: Mapping[str, Any]) -> str:
    registration = job.get("registration")
    if isinstance(registration, Mapping):
        manifest = job.get("manifest")
        manifest_state = (
            manifest.get("state") if isinstance(manifest, Mapping) else None
        )
        manifest_sha = (
            manifest.get("content_sha256") if isinstance(manifest, Mapping) else None
        )
        if manifest_state == "COMPLETE" and manifest_sha == registration.get(
            "content_sha256"
        ):
            return "REGISTERED"
        return "REGISTRATION_CONFLICT"
    attempts = job.get("attempts", [])
    if not attempts:
        return "PLANNED"
    scheduler_state = str(job.get("scheduler_state") or "UNKNOWN")
    manifest_state = str(job.get("manifest", {}).get("state") or "MISSING")
    if scheduler_state in ACTIVE_SLURM_STATES or scheduler_state == "SUBMITTED":
        return "ACTIVE"
    if scheduler_state == "COMPLETED":
        return "COMPLETE" if manifest_state == "COMPLETE" else "OUTPUT_INCOMPLETE"
    if scheduler_state in FAILED_SLURM_STATES or scheduler_state == "SUBMIT_FAILED":
        return "NEEDS_RESUME"
    return "UNKNOWN"


def _refresh_summary(ledger: dict[str, Any]) -> None:
    states = [str(job.get("state", "UNKNOWN")) for job in _jobs(ledger)]
    if states and all(state in {"COMPLETE", "REGISTERED"} for state in states):
        status = "COMPLETE"
    elif any(state == "ACTIVE" for state in states):
        status = "ACTIVE"
    elif any(state in {"NEEDS_RESUME", "OUTPUT_INCOMPLETE"} for state in states):
        status = "NEEDS_RESUME"
    elif any(
        state in {"RECONCILE_FAILED", "REGISTRATION_CONFLICT"} for state in states
    ):
        status = "NEEDS_REVIEW"
    elif any(state == "UNKNOWN" for state in states):
        status = "UNKNOWN"
    else:
        status = "PLANNED"
    ledger["status"] = status


def _validate_ledger(ledger: Mapping[str, Any]) -> None:
    if ledger.get("report_schema") != LEDGER_SCHEMA:
        raise ValueError(
            f"Unsupported job ledger schema: {ledger.get('report_schema')!r}"
        )
    if not isinstance(ledger.get("jobs"), list):
        raise ValueError("job ledger jobs must be a list")


def _jobs(ledger: Mapping[str, Any]) -> list[dict[str, Any]]:
    jobs = ledger.get("jobs")
    if not isinstance(jobs, list):
        raise ValueError("job ledger jobs must be a list")
    return jobs


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _artifact_record_for_job(
    job: Mapping[str, Any], ledger: Mapping[str, Any]
) -> ArtifactRecord:
    plan = _validated_plan(job.get("plan"), 0)
    if str(job.get("plan_id") or "") != plan["plan_id"]:
        raise ValueError("job ledger plan_id does not match the embedded job plan")
    if plan["job_kind"] == "dataset_audit":
        return _dataset_audit_record_for_job(job, ledger)
    if plan["job_kind"] == "dataset_preparation":
        return _dataset_preparation_record_for_job(job, ledger)
    if plan["job_kind"] == "representation_design":
        return _representation_design_record_for_job(job, ledger)
    if plan["job_kind"] == "probe_selection":
        return _probe_selection_record_for_job(job, ledger)
    if plan["job_kind"] == "probe_audit":
        return _probe_audit_record_for_job(job, ledger)
    if plan["job_kind"] == "feature_qc":
        return _feature_qc_record_for_job(job, ledger)
    if plan["job_kind"] == "filtration_audit":
        return _filtration_audit_record_for_job(job, ledger)
    if plan["job_kind"] == "feature_outlier_diagnostic":
        return _feature_outlier_diagnostic_record_for_job(job, ledger)
    if plan["job_kind"] == "scout_oof":
        return _scout_oof_record_for_job(job, ledger)
    if plan["job_kind"] == "scout_combine":
        return _scout_combine_record_for_job(job, ledger)
    if plan["job_kind"] == "model_evaluation":
        return _model_evaluation_record_for_job(job, ledger)
    if plan["job_kind"] == "acceptance_evaluation":
        return _acceptance_evaluation_record_for_job(job, ledger)
    if plan["job_kind"] == "validation_evaluation":
        return _validation_evaluation_record_for_job(job, ledger)
    if plan["job_kind"] == "frozen_test_evaluation":
        return _frozen_test_evaluation_record_for_job(job, ledger)
    return _feature_record_for_job(job, ledger)


def _dataset_preparation_record_for_job(
    job: Mapping[str, Any], ledger: Mapping[str, Any]
) -> ArtifactRecord:
    plan = _validated_plan(job.get("plan"), 0)
    output = inspect_dataset_preparation_report(plan["manifest_path"])
    if output.get("state") != "COMPLETE":
        raise ValueError("dataset-preparation output is missing or invalid")
    if output.get("dataset_id") != plan["dataset_id"]:
        raise ValueError("dataset-preparation task does not match the submitted plan")
    record = inspect_artifact(plan["manifest_path"])
    if record.artifact_kind != "dataset_preparation":
        raise ValueError("output was not recognized as a dataset-preparation artifact")
    return _replace_setup_record(
        record,
        job=job,
        ledger=ledger,
        dataset_id=str(plan["dataset_id"]),
        evidence_scope="design",
        status="PASS",
    )


def _dataset_audit_record_for_job(
    job: Mapping[str, Any], ledger: Mapping[str, Any]
) -> ArtifactRecord:
    plan = _validated_plan(job.get("plan"), 0)
    output = inspect_dataset_audit_report(plan["manifest_path"])
    if output.get("state") != "COMPLETE":
        raise ValueError("dataset-audit output is missing or invalid")
    if output.get("dataset_id") != plan["dataset_id"]:
        raise ValueError("dataset-audit task does not match the submitted plan")
    record = inspect_artifact(plan["manifest_path"])
    if record.artifact_kind != "dataset_audit":
        raise ValueError("output was not recognized as a dataset-audit artifact")
    return _replace_setup_record(
        record,
        job=job,
        ledger=ledger,
        dataset_id=str(plan["dataset_id"]),
        evidence_scope="design",
        status=str(output["report_status"]),
    )


def _representation_design_record_for_job(
    job: Mapping[str, Any], ledger: Mapping[str, Any]
) -> ArtifactRecord:
    plan = _validated_plan(job.get("plan"), 0)
    output = inspect_representation_design_report(plan["manifest_path"])
    if output.get("state") != "COMPLETE":
        raise ValueError("representation-design output is missing or invalid")
    if output.get("dataset_id") != plan["dataset_id"]:
        raise ValueError("representation-design task does not match the submitted plan")
    submitted_design_input_hash = plan["environment"].get("DESIGN_INPUT_HASH")
    design_input_hash_mismatch = (
        output.get("design_input_hash") is not None
        and submitted_design_input_hash is not None
        and output.get("design_input_hash") != submitted_design_input_hash
    )
    audit = inspect_dataset_audit_report(plan["input_manifests"]["DATA_AUDIT"])
    if audit.get("state") != "COMPLETE" or audit.get("report_status") != "PASS":
        raise ValueError("representation design lacks a passing dataset audit")
    if audit.get("dataset_id") != plan["dataset_id"]:
        raise ValueError("representation design uses a different dataset audit")
    if audit.get("sample_order_hash") is not None and audit.get(
        "sample_order_hash"
    ) != output.get("sample_order_hash"):
        raise ValueError(
            "representation design sample order differs from its data audit"
        )
    record = inspect_artifact(plan["manifest_path"])
    if record.artifact_kind != "representation_design":
        raise ValueError("output was not recognized as a representation artifact")
    if design_input_hash_mismatch:
        record = replace(
            record,
            metadata={
                **dict(record.metadata),
                "submitted_design_input_hash": str(submitted_design_input_hash),
                "design_input_hash_mismatch_accepted": True,
            },
        )
    return _replace_setup_record(
        record,
        job=job,
        ledger=ledger,
        dataset_id=str(plan["dataset_id"]),
        evidence_scope="design",
        representation_hash=str(output["representation_hash"]),
        status=str(output["report_status"]),
    )


def _probe_selection_record_for_job(
    job: Mapping[str, Any], ledger: Mapping[str, Any]
) -> ArtifactRecord:
    plan = _validated_plan(job.get("plan"), 0)
    output = inspect_probe_selection_report(plan["manifest_path"])
    if output.get("state") != "COMPLETE":
        raise ValueError("probe-selection output is missing or invalid")
    if output.get("dataset_id") != plan["dataset_id"]:
        raise ValueError("probe-selection task does not match the submitted plan")
    if output.get("representation_hash") != plan["representation_hash"]:
        raise ValueError("probe selection uses a different representation")
    representation = inspect_representation_design_report(
        plan["input_manifests"]["REPRESENTATION"]
    )
    if representation.get("state") != "COMPLETE":
        raise ValueError("probe selection lacks a valid representation")
    representation_dataset_mismatch = (
        representation.get("dataset_id") != plan["dataset_id"]
    )
    reusable_filtration_repair = (
        representation.get("report_schema") == "mint-agent.filtration-repair.v1"
        and representation.get("representation_hash") == plan["representation_hash"]
    )
    if (
        (representation_dataset_mismatch and not reusable_filtration_repair)
        or representation.get("representation_hash") != plan["representation_hash"]
        or (
            representation.get("sample_order_hash") is not None
            and representation.get("sample_order_hash")
            != output.get("modeling_sample_order_hash")
        )
    ):
        raise ValueError("probe selection is incompatible with its representation")
    record = inspect_artifact(plan["manifest_path"])
    if record.artifact_kind != "probe_selection":
        raise ValueError("output was not recognized as a probe-selection artifact")
    return _replace_setup_record(
        record,
        job=job,
        ledger=ledger,
        dataset_id=str(plan["dataset_id"]),
        evidence_scope="probe",
        representation_hash=str(output["representation_hash"]),
        selection_hash=str(output["selection_hash"]),
        status="COMPLETE",
    )


def _probe_audit_record_for_job(
    job: Mapping[str, Any], ledger: Mapping[str, Any]
) -> ArtifactRecord:
    plan = _validated_plan(job.get("plan"), 0)
    output = inspect_probe_audit_report(plan["manifest_path"])
    if output.get("state") != "COMPLETE":
        raise ValueError("probe-audit output is missing or invalid")
    if output.get("dataset_id") != plan["dataset_id"]:
        raise ValueError("probe-audit task does not match the submitted plan")
    if output.get("representation_hash") != plan["representation_hash"]:
        raise ValueError("probe audit uses a different representation")
    selection = inspect_probe_selection_report(
        plan["input_manifests"]["PROBE_SELECTION"]
    )
    if selection.get("state") != "COMPLETE":
        raise ValueError("probe audit lacks a valid probe selection")
    for key in (
        "dataset_id",
        "representation_hash",
        "selection_hash",
        "sample_order_hash",
        "modeling_sample_order_hash",
    ):
        if output.get(key) != selection.get(key):
            raise ValueError(f"probe audit disagrees with probe selection on {key}")
    record = inspect_artifact(plan["manifest_path"])
    if record.artifact_kind != "probe_audit":
        raise ValueError("output was not recognized as a probe-audit artifact")
    return _replace_setup_record(
        record,
        job=job,
        ledger=ledger,
        dataset_id=str(plan["dataset_id"]),
        evidence_scope="probe",
        representation_hash=str(output["representation_hash"]),
        selection_hash=str(output["selection_hash"]),
        status=str(output["report_status"]),
    )


def _replace_setup_record(
    record: ArtifactRecord,
    *,
    job: Mapping[str, Any],
    ledger: Mapping[str, Any],
    dataset_id: str,
    evidence_scope: str,
    status: str,
    representation_hash: str | None = None,
    selection_hash: str | None = None,
) -> ArtifactRecord:
    attempts = job.get("attempts", [])
    latest_attempt = attempts[-1] if attempts else {}
    return replace(
        record,
        dataset_id=dataset_id,
        split="train",
        evidence_scope=evidence_scope,
        representation_hash=representation_hash or record.representation_hash,
        selection_hash=selection_hash or record.selection_hash,
        status=status,
        metadata={
            **dict(record.metadata),
            "job_plan_id": str(job["plan_id"]),
            "slurm_job_id": latest_attempt.get("job_id"),
            "source_graph_run_id": ledger.get("run_id"),
            "reconciled_from_job_ledger": True,
        },
    )


def _feature_record_for_job(
    job: Mapping[str, Any], ledger: Mapping[str, Any]
) -> ArtifactRecord:
    plan = _validated_plan(job.get("plan"), 0)
    if str(job.get("plan_id") or "") != plan["plan_id"]:
        raise ValueError("job ledger plan_id does not match the embedded job plan")
    manifest = inspect_feature_manifest(plan["manifest_path"])
    if manifest.get("state") != "COMPLETE":
        raise ValueError(
            f"feature manifest is not complete: {manifest.get('state', 'UNKNOWN')}"
        )
    expected_invariant = str(plan["invariant"]).upper()
    if manifest.get("invariants") != [expected_invariant]:
        raise ValueError(
            "feature manifest invariant does not match the submitted job plan"
        )
    record = inspect_artifact(plan["manifest_path"])
    if record.status != "COMPLETE":
        raise ValueError(f"artifact inspection returned status {record.status!r}")
    if record.sample_count != manifest.get(
        "sample_count"
    ) or record.sample_order_hash != manifest.get("sample_order_hash"):
        raise ValueError("manifest identity changed during reconciliation")
    environment = plan["environment"]
    split = str(environment.get("SPLIT") or "") or None
    selection_hash = None
    if plan["evidence_scope"] == "probe":
        selection_path = Path(str(environment["SAMPLE_ID_FILE"]))
        selection = _read_json_object(selection_path)
        selection_hash = str(selection.get("selection_hash") or "")
        if not selection_hash or selection_hash != environment.get("SELECTION_HASH"):
            raise ValueError("probe feature selection hash does not match its job plan")
        probe_ids = tuple(str(value) for value in selection.get("probe_sample_ids", ()))
        if stable_hash(probe_ids) != manifest.get("sample_order_hash"):
            raise ValueError(
                "probe feature sample order does not match its frozen selection"
            )
        if selection.get("representation_hash") != plan["representation_hash"]:
            raise ValueError("probe feature selection uses a different representation")
    attempts = job.get("attempts", [])
    latest_attempt = attempts[-1] if attempts else {}
    return replace(
        record,
        dataset_id=str(plan["dataset_id"]),
        split=split,
        evidence_scope=str(plan["evidence_scope"]),
        invariants=(expected_invariant,),
        representation_hash=str(plan["representation_hash"]),
        selection_hash=selection_hash,
        metadata={
            **dict(record.metadata),
            "job_plan_id": str(job["plan_id"]),
            "slurm_job_id": latest_attempt.get("job_id"),
            "source_graph_run_id": ledger.get("run_id"),
            "reconciled_from_job_ledger": True,
        },
    )


def _feature_qc_record_for_job(
    job: Mapping[str, Any], ledger: Mapping[str, Any]
) -> ArtifactRecord:
    plan = _validated_plan(job.get("plan"), 0)
    if str(job.get("plan_id") or "") != plan["plan_id"]:
        raise ValueError("job ledger plan_id does not match the embedded job plan")
    output = inspect_feature_qc_report(plan["manifest_path"])
    if output.get("state") != "COMPLETE":
        raise ValueError(
            f"feature-QC report is not complete: {output.get('state', 'UNKNOWN')}"
        )
    report_path = Path(plan["manifest_path"])
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    expected_invariants = tuple(plan["invariants"])
    if tuple(output.get("invariants", ())) != expected_invariants:
        raise ValueError("feature-QC invariants do not match the submitted job plan")
    if output.get("dataset_id") != plan["dataset_id"]:
        raise ValueError("feature-QC dataset does not match the submitted job plan")
    if output.get("evidence_scope") != plan["evidence_scope"]:
        raise ValueError(
            "feature-QC evidence scope does not match the submitted job plan"
        )
    if output.get("representation_hash") != plan["representation_hash"]:
        raise ValueError(
            "feature-QC representation hash does not match the submitted job plan"
        )
    selection_hash = _validate_probe_analysis_source(plan, output)
    recorded_manifests = {
        str(name).upper(): str(path)
        for name, path in payload.get("feature_manifests", {}).items()
    }
    if recorded_manifests != plan["input_manifests"]:
        raise ValueError(
            "feature-QC report inputs do not match the submitted manifests"
        )
    for name, path in plan["input_manifests"].items():
        manifest = inspect_feature_manifest(path)
        if manifest.get("state") != "COMPLETE":
            raise ValueError(f"feature-QC input manifest {name} is no longer complete")
        if manifest.get("invariants") != [name]:
            raise ValueError(
                f"feature-QC input manifest {name} has the wrong invariant"
            )
        if manifest.get("sample_count") != output.get("sample_count") or manifest.get(
            "sample_order_hash"
        ) != output.get("sample_order_hash"):
            raise ValueError(
                f"feature-QC sample axis does not match input manifest {name}"
            )
    record = inspect_artifact(report_path)
    if record.artifact_kind != "feature_qc":
        raise ValueError("QC output was not recognized as a feature-QC artifact")
    attempts = job.get("attempts", [])
    latest_attempt = attempts[-1] if attempts else {}
    return replace(
        record,
        dataset_id=str(plan["dataset_id"]),
        split=_scope_to_split_for_record(str(plan["evidence_scope"])),
        evidence_scope=str(plan["evidence_scope"]),
        invariants=expected_invariants,
        representation_hash=str(plan["representation_hash"]),
        selection_hash=selection_hash or record.selection_hash,
        metadata={
            **dict(record.metadata),
            "job_plan_id": str(job["plan_id"]),
            "slurm_job_id": latest_attempt.get("job_id"),
            "source_graph_run_id": ledger.get("run_id"),
            "reconciled_from_job_ledger": True,
        },
    )


def _filtration_audit_record_for_job(
    job: Mapping[str, Any], ledger: Mapping[str, Any]
) -> ArtifactRecord:
    plan = _validated_plan(job.get("plan"), 0)
    if str(job.get("plan_id") or "") != plan["plan_id"]:
        raise ValueError("job ledger plan_id does not match the embedded job plan")
    output = inspect_filtration_audit_report(plan["manifest_path"])
    if output.get("state") != "COMPLETE":
        raise ValueError(
            f"filtration-audit report is not complete: {output.get('state', 'UNKNOWN')}"
        )
    report_path = Path(plan["manifest_path"])
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    expected_invariants = tuple(plan["invariants"])
    if tuple(output.get("invariants", ())) != expected_invariants:
        raise ValueError(
            "filtration-audit invariants do not match the submitted job plan"
        )
    if output.get("dataset_id") != plan["dataset_id"]:
        raise ValueError(
            "filtration-audit dataset does not match the submitted job plan"
        )
    if output.get("evidence_scope") != plan["evidence_scope"]:
        raise ValueError("filtration-audit scope does not match the submitted job plan")
    if output.get("representation_hash") != plan["representation_hash"]:
        raise ValueError(
            "filtration-audit representation does not match the submitted job plan"
        )
    selection_hash = _validate_probe_analysis_source(plan, output)
    recorded_manifests = {
        str(name).upper(): str(path)
        for name, path in payload.get("feature_manifests", {}).items()
    }
    if recorded_manifests != plan["input_manifests"]:
        raise ValueError("filtration-audit inputs do not match the submitted manifests")
    for name, path in plan["input_manifests"].items():
        manifest = inspect_feature_manifest(path)
        if manifest.get("state") != "COMPLETE":
            raise ValueError(
                f"filtration-audit input manifest {name} is no longer complete"
            )
        if manifest.get("invariants") != [name]:
            raise ValueError(
                f"filtration-audit input manifest {name} has the wrong invariant"
            )
        if manifest.get("sample_count") != output.get("sample_count") or manifest.get(
            "sample_order_hash"
        ) != output.get("sample_order_hash"):
            raise ValueError(
                f"filtration-audit sample axis does not match input manifest {name}"
            )
    record = inspect_artifact(report_path)
    if record.artifact_kind != "filtration_audit":
        raise ValueError("output was not recognized as a filtration-audit artifact")
    attempts = job.get("attempts", [])
    latest_attempt = attempts[-1] if attempts else {}
    return replace(
        record,
        dataset_id=str(plan["dataset_id"]),
        split=_scope_to_split_for_record(str(plan["evidence_scope"])),
        evidence_scope=str(plan["evidence_scope"]),
        invariants=expected_invariants,
        representation_hash=str(plan["representation_hash"]),
        selection_hash=selection_hash or record.selection_hash,
        metadata={
            **dict(record.metadata),
            "job_plan_id": str(job["plan_id"]),
            "slurm_job_id": latest_attempt.get("job_id"),
            "source_graph_run_id": ledger.get("run_id"),
            "reconciled_from_job_ledger": True,
        },
    )


def _feature_outlier_diagnostic_record_for_job(
    job: Mapping[str, Any], ledger: Mapping[str, Any]
) -> ArtifactRecord:
    plan = _validated_plan(job.get("plan"), 0)
    if str(job.get("plan_id") or "") != plan["plan_id"]:
        raise ValueError(
            "job ledger plan_id does not match the embedded diagnostic plan"
        )
    output = inspect_feature_outlier_diagnostic_report(plan["manifest_path"])
    if output.get("state") != "COMPLETE":
        raise ValueError(
            f"feature diagnostic is not complete: {output.get('state', 'UNKNOWN')}"
        )
    expected_invariant = str(plan["invariant"])
    if output.get("invariants") != [expected_invariant]:
        raise ValueError("feature diagnostic invariant does not match its job plan")
    for field in ("dataset_id", "evidence_scope", "representation_hash"):
        if output.get(field) != plan.get(field):
            raise ValueError(f"feature diagnostic {field} does not match its job plan")
    selection_hash = _validate_probe_analysis_source(plan, output)
    feature_manifest = plan["input_manifests"][expected_invariant]
    if output.get("feature_manifest") != feature_manifest:
        raise ValueError(
            "feature diagnostic source manifest does not match its job plan"
        )
    manifest = inspect_feature_manifest(feature_manifest)
    if manifest.get("state") != "COMPLETE":
        raise ValueError("feature diagnostic input manifest is no longer complete")
    if manifest.get("sample_count") != output.get("sample_count") or manifest.get(
        "sample_order_hash"
    ) != output.get("sample_order_hash"):
        raise ValueError("feature diagnostic sample axis differs from its manifest")
    qc_path = Path(str(plan["environment"]["FEATURE_QC_REPORT"]))
    qc = inspect_feature_qc_report(qc_path)
    if qc.get("state") != "COMPLETE":
        raise ValueError("feature diagnostic lacks a valid feature-QC report")
    qc_payload = _read_json_object(qc_path)
    if output.get("feature_qc_hash") != qc_payload.get("qc_hash"):
        raise ValueError("feature diagnostic feature-QC hash does not match")
    certified = {
        str(name).upper(): str(path)
        for name, path in qc_payload.get("feature_manifests", {}).items()
    }
    if certified.get(expected_invariant) != feature_manifest:
        raise ValueError(
            "feature diagnostic manifest is not certified by its QC report"
        )
    record = inspect_artifact(plan["manifest_path"])
    if record.artifact_kind != "feature_outlier_diagnostic":
        raise ValueError("output was not recognized as a feature diagnostic artifact")
    attempts = job.get("attempts", [])
    latest_attempt = attempts[-1] if attempts else {}
    return replace(
        record,
        dataset_id=str(plan["dataset_id"]),
        split=_scope_to_split_for_record(str(plan["evidence_scope"])),
        evidence_scope=str(plan["evidence_scope"]),
        invariants=(expected_invariant,),
        representation_hash=str(plan["representation_hash"]),
        selection_hash=selection_hash or record.selection_hash,
        status="COMPLETE",
        metadata={
            **dict(record.metadata),
            "diagnostic_hash": output.get("diagnostic_hash"),
            "feature_qc_hash": output.get("feature_qc_hash"),
            "job_plan_id": str(job["plan_id"]),
            "slurm_job_id": latest_attempt.get("job_id"),
            "source_graph_run_id": ledger.get("run_id"),
            "reconciled_from_job_ledger": True,
        },
    )


def _scout_oof_record_for_job(
    job: Mapping[str, Any], ledger: Mapping[str, Any]
) -> ArtifactRecord:
    plan = _validated_plan(job.get("plan"), 0)
    if str(job.get("plan_id") or "") != plan["plan_id"]:
        raise ValueError(
            "job ledger plan_id does not match the embedded Scout OOF plan"
        )
    output = inspect_job_output(plan)
    if output.get("state") != "COMPLETE":
        raise ValueError(f"Scout OOF output is not complete: {output.get('state')}")
    report_path = Path(plan["manifest_path"])
    payload = _read_json_object(report_path)
    invariant = plan["invariants"][0]
    if output.get("invariants") != [invariant]:
        raise ValueError("Scout OOF invariant does not match the submitted plan")
    if output.get("dataset_id") != plan["dataset_id"]:
        raise ValueError("Scout OOF dataset does not match the submitted plan")
    if output.get("evidence_scope") != "probe":
        raise ValueError("Scout OOF evidence scope must be probe")
    if output.get("representation_hash") != plan["representation_hash"]:
        raise ValueError("Scout OOF representation does not match the submitted plan")
    recorded_manifests = {
        str(name).upper(): str(path)
        for name, path in payload.get("feature_manifests", {}).items()
    }
    if recorded_manifests != plan["input_manifests"]:
        raise ValueError("Scout OOF feature manifest does not match the submitted plan")
    feature_manifest = inspect_feature_manifest(plan["input_manifests"][invariant])
    if feature_manifest.get("state") != "COMPLETE":
        raise ValueError("Scout OOF source feature manifest is no longer complete")
    if feature_manifest.get("sample_count") != output.get(
        "sample_count"
    ) or feature_manifest.get("sample_order_hash") != output.get("sample_order_hash"):
        raise ValueError("Scout OOF sample axis does not match its feature manifest")
    environment = plan["environment"]
    probe = _read_json_object(Path(str(environment["PROBE_SELECTION"])))
    probe_ids = tuple(str(value) for value in probe.get("probe_sample_ids", ()))
    if stable_hash(probe_ids) != output.get("sample_order_hash"):
        raise ValueError("Scout OOF sample order does not match the frozen probe")
    if probe.get("selection_hash") != output.get("selection_hash"):
        raise ValueError("Scout OOF selection hash does not match the frozen probe")
    if payload.get("feature_qc_report") != environment.get("FEATURE_QC_REPORT"):
        raise ValueError("Scout OOF feature-QC source does not match the plan")
    if payload.get("filtration_audit_report") != environment.get(
        "FILTRATION_AUDIT_REPORT"
    ):
        raise ValueError("Scout OOF filtration-audit source does not match the plan")
    for source_name, inspector in (
        ("FEATURE_QC_REPORT", inspect_feature_qc_report),
        ("FILTRATION_AUDIT_REPORT", inspect_filtration_audit_report),
    ):
        source = inspector(str(environment[source_name]))
        if source.get("state") != "COMPLETE":
            raise ValueError(f"Scout OOF {source_name} is missing or invalid")
        if (
            source.get("sample_count") != output.get("sample_count")
            or source.get("sample_order_hash") != output.get("sample_order_hash")
            or source.get("representation_hash") != plan["representation_hash"]
        ):
            raise ValueError(f"Scout OOF {source_name} is scientifically incompatible")
    record = inspect_artifact(report_path)
    if record.artifact_kind != "scout_oof":
        raise ValueError("Scout OOF output was not recognized as a Scout artifact")
    attempts = job.get("attempts", [])
    latest_attempt = attempts[-1] if attempts else {}
    return replace(
        record,
        dataset_id=str(plan["dataset_id"]),
        split="train",
        evidence_scope="probe",
        invariants=(invariant,),
        representation_hash=str(plan["representation_hash"]),
        selection_hash=str(output["selection_hash"]),
        status="COMPLETE",
        metadata={
            **dict(record.metadata),
            "gbt_parameter_hash": output.get("gbt_parameter_hash"),
            "job_plan_id": str(job["plan_id"]),
            "slurm_job_id": latest_attempt.get("job_id"),
            "source_graph_run_id": ledger.get("run_id"),
            "reconciled_from_job_ledger": True,
        },
    )


def _scout_combine_record_for_job(
    job: Mapping[str, Any], ledger: Mapping[str, Any]
) -> ArtifactRecord:
    plan = _validated_plan(job.get("plan"), 0)
    if str(job.get("plan_id") or "") != plan["plan_id"]:
        raise ValueError(
            "job ledger plan_id does not match the embedded Scout combine plan"
        )
    output = inspect_job_output(plan)
    if output.get("state") != "COMPLETE":
        raise ValueError(f"Scout combine output is not complete: {output.get('state')}")
    environment = plan["environment"]
    combined_path = Path(str(environment["COMBINED_REPORT"]))
    combined = inspect_scout_combined_report(combined_path)
    payload = _read_json_object(combined_path)
    expected_invariants = tuple(plan["invariants"])
    if tuple(combined.get("invariants", ())) != expected_invariants:
        raise ValueError("combined Scout invariants do not match the submitted plan")
    if combined.get("dataset_id") != plan["dataset_id"]:
        raise ValueError("combined Scout dataset does not match the submitted plan")
    if combined.get("evidence_scope") != "probe":
        raise ValueError("combined Scout evidence scope must be probe")
    if combined.get("representation_hash") != plan["representation_hash"]:
        raise ValueError(
            "combined Scout representation does not match the submitted plan"
        )
    recorded_sources = {
        str(name).upper(): str(path)
        for name, path in payload.get("source_oof_reports", {}).items()
    }
    if recorded_sources != plan["input_manifests"]:
        raise ValueError(
            "combined Scout source reports do not match the submitted plan"
        )
    for name, path in plan["input_manifests"].items():
        source = inspect_scout_oof_report(path)
        if source.get("state") != "COMPLETE" or source.get("invariants") != [name]:
            raise ValueError(f"combined Scout source {name} is missing or invalid")
        if (
            source.get("sample_count") != combined.get("sample_count")
            or source.get("sample_order_hash") != combined.get("sample_order_hash")
            or source.get("selection_hash") != combined.get("selection_hash")
            or source.get("gbt_parameter_hash") != combined.get("gbt_parameter_hash")
            or source.get("representation_hash") != plan["representation_hash"]
        ):
            raise ValueError(
                f"combined Scout source {name} is scientifically incompatible"
            )
    if output.get("invariants") != list(expected_invariants):
        raise ValueError(
            "Scout execution priority does not match the combined invariants"
        )
    if output.get("representation_hash") != plan["representation_hash"] or output.get(
        "gbt_parameter_hash"
    ) != combined.get("gbt_parameter_hash"):
        raise ValueError("Scout execution artifact does not match the combined report")
    expected_ranking_policy = str(environment.get("RANKING_POLICY") or "")
    if (
        not expected_ranking_policy
        or output.get("ranking_policy") != expected_ranking_policy
        or combined.get("ranking_policy") != expected_ranking_policy
    ):
        raise ValueError("Scout ranking policy does not match the submitted plan")
    probe = _read_json_object(Path(str(environment["PROBE_SELECTION"])))
    modeling_ids = tuple(str(value) for value in probe.get("modeling_sample_ids", ()))
    if stable_hash(modeling_ids) != output.get("sample_order_hash"):
        raise ValueError(
            "Scout execution modeling order does not match the frozen probe"
        )
    record = inspect_artifact(plan["manifest_path"])
    if record.artifact_kind != "scout_execution_plan":
        raise ValueError("Scout output was not recognized as an execution artifact")
    attempts = job.get("attempts", [])
    latest_attempt = attempts[-1] if attempts else {}
    return replace(
        record,
        dataset_id=str(plan["dataset_id"]),
        split="train",
        evidence_scope="probe",
        invariants=expected_invariants,
        representation_hash=str(plan["representation_hash"]),
        selection_hash=str(combined.get("selection_hash")),
        status="COMPLETE",
        metadata={
            **dict(record.metadata),
            "combined_report": str(combined_path),
            "gbt_parameter_hash": combined.get("gbt_parameter_hash"),
            "job_plan_id": str(job["plan_id"]),
            "ranking_policy": output.get("ranking_policy"),
            "slurm_job_id": latest_attempt.get("job_id"),
            "source_graph_run_id": ledger.get("run_id"),
            "reconciled_from_job_ledger": True,
        },
    )


def _model_evaluation_record_for_job(
    job: Mapping[str, Any], ledger: Mapping[str, Any]
) -> ArtifactRecord:
    plan = _validated_plan(job.get("plan"), 0)
    if str(job.get("plan_id") or "") != plan["plan_id"]:
        raise ValueError("job ledger plan_id does not match the evaluation plan")
    output = inspect_model_evaluation_report(plan["manifest_path"])
    if output.get("state") != "COMPLETE":
        raise ValueError(
            f"model-evaluation output is not complete: {output.get('state')}"
        )
    expected_invariants = tuple(plan["invariants"])
    if tuple(output.get("invariants", ())) != expected_invariants:
        raise ValueError("model-evaluation invariants do not match the submitted plan")
    if output.get("dataset_id") != plan["dataset_id"]:
        raise ValueError("model-evaluation dataset does not match the submitted plan")
    if output.get("evidence_scope") != plan["evidence_scope"]:
        raise ValueError("model-evaluation scope does not match the submitted plan")
    if output.get("representation_hash") != plan["representation_hash"]:
        raise ValueError(
            "model-evaluation representation does not match the submitted plan"
        )
    report_path = Path(plan["manifest_path"])
    payload = _read_json_object(report_path)
    recorded_manifests = {
        str(name).upper(): str(path)
        for name, path in payload.get("feature_manifests", {}).items()
    }
    if recorded_manifests != plan["input_manifests"]:
        raise ValueError(
            "model-evaluation feature inputs do not match the submitted plan"
        )
    for name, path in plan["input_manifests"].items():
        manifest = inspect_feature_manifest(path)
        if manifest.get("state") != "COMPLETE":
            raise ValueError(f"model-evaluation feature manifest {name} is incomplete")
        if manifest.get("invariants") != [name]:
            raise ValueError(
                f"model-evaluation feature manifest {name} is incompatible"
            )
        if manifest.get("sample_count") != output.get("sample_count") or manifest.get(
            "sample_order_hash"
        ) != output.get("sample_order_hash"):
            raise ValueError(
                f"model-evaluation sample axis does not match feature manifest {name}"
            )
    environment = plan["environment"]
    if payload.get("scout_artifact") != environment.get("SCOUT_ARTIFACT"):
        raise ValueError(
            "model-evaluation Scout source does not match the submitted plan"
        )
    scout = inspect_scout_execution_artifact(str(environment["SCOUT_ARTIFACT"]))
    if scout.get("state") != "COMPLETE":
        raise ValueError("model-evaluation Scout artifact is missing or invalid")
    scout_invariants = set(scout.get("invariants", ()))
    if (
        not set(expected_invariants).issubset(scout_invariants)
        or scout.get("sample_count") != output.get("sample_count")
        or scout.get("sample_order_hash") != output.get("sample_order_hash")
        or scout.get("representation_hash") != plan["representation_hash"]
        or scout.get("gbt_parameter_hash") != output.get("gbt_parameter_hash")
    ):
        raise ValueError(
            "model-evaluation Scout artifact is scientifically incompatible"
        )
    scout_artifact = ScoutExecutionArtifact.read(str(environment["SCOUT_ARTIFACT"]))
    result_payload = payload.get("result")
    if not isinstance(result_payload, Mapping):
        raise ValueError("model-evaluation result payload is missing")
    try:
        target_value_matches = float(payload.get("target_value")) == float(
            scout_artifact.target_value
        )
    except (TypeError, ValueError):
        target_value_matches = False
    if (
        str(payload.get("target_metric") or "").upper()
        != scout_artifact.target_metric.upper()
        or not target_value_matches
        or result_payload.get("evaluation_fold_assignment")
        != dict(scout_artifact.full_fold_assignment)
    ):
        raise ValueError("model-evaluation target or folds disagree with frozen Scout")
    acquisition_order: list[str] = []
    for subset in scout_artifact.frozen_priority_order:
        for invariant in subset:
            normalized = str(invariant).upper()
            if normalized not in acquisition_order:
                acquisition_order.append(normalized)
    if set(acquisition_order[: len(expected_invariants)]) != set(expected_invariants):
        raise ValueError(
            "model-evaluation invariants are not a frozen acquisition prefix"
        )
    if int(payload.get("max_acquisitions") or 0) != len(expected_invariants):
        raise ValueError("model-evaluation acquisition limit does not match the plan")
    if payload.get("feature_qc_report") != environment.get("FEATURE_QC_REPORT"):
        raise ValueError("model-evaluation QC source does not match the submitted plan")
    qc = inspect_feature_qc_report(str(environment["FEATURE_QC_REPORT"]))
    if qc.get("state") != "COMPLETE" or qc.get("report_status") == "FAIL":
        raise ValueError("model-evaluation feature QC is missing or failed")
    if (
        qc.get("sample_count") != output.get("sample_count")
        or qc.get("sample_order_hash") != output.get("sample_order_hash")
        or qc.get("representation_hash") != plan["representation_hash"]
    ):
        raise ValueError("model-evaluation feature QC is scientifically incompatible")
    qc_payload = _read_json_object(Path(str(environment["FEATURE_QC_REPORT"])))
    qc_manifests = {
        str(name).upper(): str(path)
        for name, path in qc_payload.get("feature_manifests", {}).items()
    }
    if any(
        qc_manifests.get(name) != path for name, path in plan["input_manifests"].items()
    ):
        raise ValueError("model-evaluation feature manifests are not certified by QC")
    record = inspect_artifact(report_path)
    if record.artifact_kind != "model_evaluation":
        raise ValueError("output was not recognized as a model-evaluation artifact")
    attempts = job.get("attempts", [])
    latest_attempt = attempts[-1] if attempts else {}
    return replace(
        record,
        dataset_id=str(plan["dataset_id"]),
        split="train",
        evidence_scope="full_train",
        invariants=expected_invariants,
        representation_hash=str(plan["representation_hash"]),
        status=str(output["report_status"]),
        metadata={
            **dict(record.metadata),
            "gbt_parameter_hash": output.get("gbt_parameter_hash"),
            "job_plan_id": str(job["plan_id"]),
            "slurm_job_id": latest_attempt.get("job_id"),
            "source_graph_run_id": ledger.get("run_id"),
            "reconciled_from_job_ledger": True,
        },
    )


def _validation_evaluation_record_for_job(
    job: Mapping[str, Any], ledger: Mapping[str, Any]
) -> ArtifactRecord:
    plan = _validated_plan(job.get("plan"), 0)
    output = inspect_validation_evaluation_report(plan["manifest_path"])
    if output.get("state") != "COMPLETE":
        raise ValueError("validation-evaluation output is missing or invalid")
    if output.get("dataset_id") != plan["dataset_id"]:
        raise ValueError("validation-evaluation dataset does not match its plan")
    if output.get("representation_hash") != plan["representation_hash"]:
        raise ValueError("validation-evaluation representation does not match its plan")
    if set(output.get("invariants", ())) != set(plan["invariants"]):
        raise ValueError("validation-evaluation invariants do not match its plan")
    payload = _read_json_object(Path(plan["manifest_path"]))
    _assert_validation_evaluation_inputs(plan, payload)
    environment = plan["environment"]
    if payload.get("scout_artifact") != environment.get("SCOUT_ARTIFACT"):
        raise ValueError("validation-evaluation Scout source does not match its plan")
    scout = inspect_scout_execution_artifact(str(environment["SCOUT_ARTIFACT"]))
    if (
        scout.get("state") != "COMPLETE"
        or scout.get("representation_hash") != plan["representation_hash"]
        or scout.get("gbt_parameter_hash") != output.get("gbt_parameter_hash")
    ):
        raise ValueError("validation-evaluation Scout source is incompatible")
    for split, qc_key in (
        ("train", "TRAIN_QC_REPORT"),
        ("validation", "VALIDATION_QC_REPORT"),
    ):
        qc_path = str(environment[qc_key])
        qc = inspect_feature_qc_report(qc_path)
        if qc.get("state") != "COMPLETE" or qc.get("report_status") == "FAIL":
            raise ValueError(f"validation-evaluation {split} QC is missing or failed")
        expected_count = (
            int(payload["training_sample_count"])
            if split == "train"
            else output["sample_count"]
        )
        expected_hash = (
            str(payload["training_sample_order_hash"])
            if split == "train"
            else output["sample_order_hash"]
        )
        if (
            qc.get("sample_count") != expected_count
            or qc.get("sample_order_hash") != expected_hash
            or qc.get("representation_hash") != plan["representation_hash"]
        ):
            raise ValueError(f"validation-evaluation {split} QC is incompatible")
    record = inspect_artifact(plan["manifest_path"])
    if record.artifact_kind != "model_evaluation":
        raise ValueError("validation output was not recognized as a model evaluation")
    attempts = job.get("attempts", [])
    latest_attempt = attempts[-1] if attempts else {}
    return replace(
        record,
        dataset_id=str(plan["dataset_id"]),
        split="validation",
        evidence_scope="validation",
        invariants=tuple(plan["invariants"]),
        representation_hash=str(plan["representation_hash"]),
        selection_hash=str(output["selection_hash"]),
        status=str(output["report_status"]),
        metadata={
            **dict(record.metadata),
            "selected_subset": output.get("selected_subset"),
            "gbt_parameter_hash": output.get("gbt_parameter_hash"),
            "target_metric": output.get("target_metric"),
            "target_value": output.get("target_value"),
            "selection_objective": output.get("selection_objective"),
            "job_plan_id": str(job["plan_id"]),
            "slurm_job_id": latest_attempt.get("job_id"),
            "source_graph_run_id": ledger.get("run_id"),
            "reconciled_from_job_ledger": True,
        },
    )


def _acceptance_evaluation_record_for_job(
    job: Mapping[str, Any], ledger: Mapping[str, Any]
) -> ArtifactRecord:
    plan = _validated_plan(job.get("plan"), 0)
    output = inspect_acceptance_evaluation_report(plan["manifest_path"])
    if output.get("state") != "COMPLETE":
        raise ValueError("acceptance-evaluation output is missing or invalid")
    if output.get("dataset_id") != plan["dataset_id"]:
        raise ValueError("acceptance-evaluation dataset does not match its plan")
    if output.get("representation_hash") != plan["representation_hash"]:
        raise ValueError("acceptance-evaluation representation does not match its plan")
    if {str(value).upper() for value in output.get("invariants", ())} != {
        str(value).upper() for value in plan["invariants"]
    }:
        raise ValueError("acceptance-evaluation invariants do not match its plan")
    if output.get("evidence_scope") != plan["evidence_scope"]:
        raise ValueError("acceptance-evaluation scope does not match its plan")

    payload = _read_json_object(Path(plan["manifest_path"]))
    recorded_inputs: dict[str, str] = {}
    manifests = payload.get("feature_manifests")
    if not isinstance(manifests, Mapping):
        raise ValueError("acceptance-evaluation feature inputs are missing")
    for split, prefix in (("train", "TRAIN"), ("acceptance", "EVALUATION")):
        split_manifests = manifests.get(split)
        if not isinstance(split_manifests, Mapping):
            raise ValueError(f"acceptance-evaluation {split} features are missing")
        for invariant, path in split_manifests.items():
            recorded_inputs[f"{prefix}_{str(invariant).upper()}"] = str(path)
    if recorded_inputs != plan["input_manifests"]:
        raise ValueError("acceptance-evaluation feature inputs do not match its plan")

    environment = plan["environment"]
    if payload.get("scout_artifact") != environment.get("SCOUT_ARTIFACT"):
        raise ValueError("acceptance-evaluation Scout source does not match its plan")
    scout_summary = inspect_scout_execution_artifact(str(environment["SCOUT_ARTIFACT"]))
    if (
        scout_summary.get("state") != "COMPLETE"
        or scout_summary.get("representation_hash") != plan["representation_hash"]
        or scout_summary.get("gbt_parameter_hash") != output.get("gbt_parameter_hash")
    ):
        raise ValueError("acceptance-evaluation Scout source is incompatible")
    scout = ScoutExecutionArtifact.read(str(environment["SCOUT_ARTIFACT"]))
    progressive = environment.get("ACCEPTANCE_MODE") == "progressive"
    candidate_rank: int | None = None
    max_acquisitions: int | None = None
    if progressive:
        max_acquisitions = int(environment["MAX_ACQUISITIONS"])
        if output.get("max_acquisitions") != max_acquisitions:
            raise ValueError(
                "progressive acceptance stage does not match its plan"
            )
        acquisition_order = tuple(
            dict.fromkeys(
                invariant
                for subset in scout.frozen_priority_order
                for invariant in subset
            )
        )
        expected_stage = acquisition_order[:max_acquisitions]
        output_acquisition_order = tuple(
            str(value).upper() for value in payload.get("acquisition_order", ())
        )
        if (
            output_acquisition_order != expected_stage
            or {str(value).upper() for value in plan["invariants"]}
            != set(expected_stage)
        ):
            raise ValueError(
                "progressive acceptance stage is not the frozen acquisition prefix"
            )
        prior_path = environment.get("PRIOR_EVALUATION_REPORT")
        prediction_reuse = payload.get("prediction_reuse")
        if not isinstance(prediction_reuse, Mapping):
            raise ValueError("progressive acceptance reuse metadata is missing")
        payload_prior = prediction_reuse.get("prior_evaluation_report")
        if payload_prior != prior_path:
            raise ValueError("progressive acceptance prior report does not match")
    else:
        candidate_rank = int(environment["CANDIDATE_RANK"])
        if candidate_rank > len(scout.frozen_priority_order):
            raise ValueError("acceptance-evaluation candidate rank is outside Scout")
        if output.get("candidate_rank") != candidate_rank:
            raise ValueError(
                "acceptance-evaluation candidate rank does not match its plan"
            )
        expected_subset = tuple(scout.frozen_priority_order[candidate_rank - 1])
        if set(expected_subset) != set(plan["invariants"]):
            raise ValueError(
                "acceptance-evaluation candidate is not the frozen Scout rank"
            )
    try:
        target_matches = float(output.get("target_value")) == float(scout.target_value)
    except (TypeError, ValueError):
        target_matches = False
    if (
        str(output.get("target_metric") or "").upper() != scout.target_metric.upper()
        or not target_matches
    ):
        raise ValueError("acceptance-evaluation target disagrees with frozen Scout")

    counts = payload["sample_counts"]
    hashes = payload["sample_order_hashes"]
    for key, manifest_path in plan["input_manifests"].items():
        prefix, invariant = key.split("_", 1)
        split = "train" if prefix == "TRAIN" else "acceptance"
        manifest = inspect_feature_manifest(manifest_path)
        if (
            manifest.get("state") != "COMPLETE"
            or manifest.get("invariants") != [invariant]
            or manifest.get("sample_count") != int(counts[split])
            or manifest.get("sample_order_hash") != hashes[split]
        ):
            raise ValueError(f"acceptance-evaluation input {key} is incompatible")

    for split, qc_key, input_prefix in (
        ("train", "TRAIN_QC_REPORT", "TRAIN"),
        ("acceptance", "EVALUATION_QC_REPORT", "EVALUATION"),
    ):
        qc_path = str(environment[qc_key])
        qc = inspect_feature_qc_report(qc_path)
        if (
            qc.get("state") != "COMPLETE"
            or qc.get("report_status") == "FAIL"
            or qc.get("sample_count") != int(counts[split])
            or qc.get("sample_order_hash") != hashes[split]
            or qc.get("representation_hash") != plan["representation_hash"]
        ):
            raise ValueError(f"acceptance-evaluation {split} QC is incompatible")
        qc_payload = _read_json_object(Path(qc_path))
        qc_manifests = {
            f"{input_prefix}_{str(name).upper()}": str(path)
            for name, path in qc_payload.get("feature_manifests", {}).items()
        }
        expected_qc_inputs = {
            key: path
            for key, path in plan["input_manifests"].items()
            if key.startswith(f"{input_prefix}_")
        }
        if qc_manifests != expected_qc_inputs:
            raise ValueError(
                f"acceptance-evaluation {split} features are not certified by QC"
            )

    record = inspect_artifact(plan["manifest_path"])
    if record.artifact_kind != "model_evaluation":
        raise ValueError("acceptance output was not recognized as a model evaluation")
    attempts = job.get("attempts", [])
    latest_attempt = attempts[-1] if attempts else {}
    return replace(
        record,
        dataset_id=str(plan["dataset_id"]),
        split="test",
        evidence_scope="acceptance_test",
        invariants=tuple(plan["invariants"]),
        representation_hash=str(plan["representation_hash"]),
        sample_count=int(output["sample_count"]),
        sample_order_hash=str(output["sample_order_hash"]),
        status=str(output["report_status"]),
        metadata={
            **dict(record.metadata),
            "selected_subset": list(
                output.get("selected_subset") or plan["invariants"]
            ),
            "selected_score": output.get("selected_score"),
            "candidate_rank": candidate_rank,
            "max_acquisitions": max_acquisitions,
            "candidate_count": output.get("candidate_count"),
            "candidate_rank_limit": output.get("candidate_rank_limit"),
            "selection_objective": output.get("selection_objective"),
            "scout_artifact": str(environment["SCOUT_ARTIFACT"]),
            "gbt_parameter_hash": output.get("gbt_parameter_hash"),
            "target_metric": output.get("target_metric"),
            "target_value": output.get("target_value"),
            "job_plan_id": str(job["plan_id"]),
            "slurm_job_id": latest_attempt.get("job_id"),
            "source_graph_run_id": ledger.get("run_id"),
            "reconciled_from_job_ledger": True,
        },
    )


def _frozen_test_evaluation_record_for_job(
    job: Mapping[str, Any], ledger: Mapping[str, Any]
) -> ArtifactRecord:
    plan = _validated_plan(job.get("plan"), 0)
    output = inspect_frozen_test_evaluation_report(plan["manifest_path"])
    if output.get("state") != "COMPLETE":
        raise ValueError("frozen-test output is missing or invalid")
    if output.get("dataset_id") != plan["dataset_id"]:
        raise ValueError("frozen-test dataset does not match its plan")
    if output.get("representation_hash") != plan["representation_hash"]:
        raise ValueError("frozen-test representation does not match its plan")
    if tuple(output.get("invariants", ())) != tuple(plan["invariants"]):
        raise ValueError("frozen-test invariants do not match its plan")
    payload = _read_json_object(Path(plan["manifest_path"]))
    if (
        _flatten_split_manifests(payload.get("feature_manifests"))
        != plan["input_manifests"]
    ):
        raise ValueError("frozen-test feature inputs do not match its plan")
    environment = plan["environment"]
    selection_path = Path(str(environment["SELECTION_REPORT"]))
    if payload.get("selection_report") != str(selection_path) or payload.get(
        "selection_report_sha256"
    ) != _file_sha256(selection_path):
        raise ValueError("frozen-test selection source does not match its plan")
    selection = inspect_artifact(selection_path)
    selected_subset = tuple(
        str(value).upper() for value in selection.metadata.get("selected_subset", ())
    )
    if (
        selection.artifact_kind != "model_evaluation"
        or selection.evidence_scope not in {"probe", "full_train", "validation"}
        or set(selected_subset) != set(plan["invariants"])
    ):
        raise ValueError("frozen-test selection artifact is incompatible")
    counts = payload["sample_counts"]
    hashes = payload["sample_order_hashes"]
    for key, manifest_path in plan["input_manifests"].items():
        split, invariant = key.split("_", 1)
        manifest = inspect_feature_manifest(manifest_path)
        split_name = split.lower()
        if (
            manifest.get("state") != "COMPLETE"
            or manifest.get("invariants") != [invariant]
            or manifest.get("sample_count") != int(counts[split_name])
            or manifest.get("sample_order_hash") != hashes[split_name]
        ):
            raise ValueError(f"frozen-test input {key} is incompatible")
    for split, qc_key in (
        ("train", "TRAIN_QC_REPORT"),
        ("validation", "VALIDATION_QC_REPORT"),
        ("test", "TEST_QC_REPORT"),
    ):
        qc_path = environment.get(qc_key)
        if not qc_path:
            continue
        qc = inspect_feature_qc_report(str(qc_path))
        if (
            qc.get("state") != "COMPLETE"
            or qc.get("report_status") == "FAIL"
            or qc.get("sample_count") != int(counts[split])
            or qc.get("sample_order_hash") != hashes[split]
            or qc.get("representation_hash") != plan["representation_hash"]
        ):
            raise ValueError(f"frozen-test {split} QC is incompatible")
    record = inspect_artifact(plan["manifest_path"])
    if record.artifact_kind != "model_evaluation":
        raise ValueError("frozen-test output was not recognized as a model evaluation")
    attempts = job.get("attempts", [])
    latest_attempt = attempts[-1] if attempts else {}
    return replace(
        record,
        dataset_id=str(plan["dataset_id"]),
        split="test",
        evidence_scope="external_test",
        invariants=tuple(plan["invariants"]),
        representation_hash=str(plan["representation_hash"]),
        status="COMPLETE",
        metadata={
            **dict(record.metadata),
            "selected_subset": list(plan["invariants"]),
            "gbt_parameter_hash": output.get("gbt_parameter_hash"),
            "selection_report": str(selection_path),
            "job_plan_id": str(job["plan_id"]),
            "slurm_job_id": latest_attempt.get("job_id"),
            "source_graph_run_id": ledger.get("run_id"),
            "reconciled_from_job_ledger": True,
        },
    )


def _flatten_split_manifests(raw: Any) -> dict[str, str]:
    if not isinstance(raw, Mapping):
        raise ValueError("split feature manifests must be an object")
    flattened: dict[str, str] = {}
    for split, manifests in raw.items():
        if not isinstance(manifests, Mapping):
            raise ValueError("split feature manifest entry must be an object")
        for invariant, path in manifests.items():
            flattened[f"{str(split).upper()}_{str(invariant).upper()}"] = str(path)
    return flattened


def _assert_validation_evaluation_inputs(
    plan: Mapping[str, Any], payload: Mapping[str, Any]
) -> None:
    planned_inputs = {
        str(name).upper(): str(path)
        for name, path in dict(plan["input_manifests"]).items()
    }
    prior_path = planned_inputs.pop("PRIOR_EVALUATION", None)
    if _flatten_split_manifests(payload.get("feature_manifests")) != planned_inputs:
        raise ValueError("validation-evaluation feature inputs do not match its plan")

    reuse = payload.get("prediction_reuse")
    if not isinstance(reuse, Mapping):
        raise ValueError("validation-evaluation prediction reuse record is missing")
    recorded_prior = reuse.get("prior_evaluation_report")
    if recorded_prior != prior_path:
        raise ValueError("validation-evaluation prior input does not match its plan")
    recorded_sha256 = reuse.get("prior_evaluation_sha256")
    if prior_path is None:
        if recorded_sha256 is not None:
            raise ValueError(
                "first validation stage unexpectedly records prior content"
            )
        return
    if recorded_sha256 != _file_sha256(Path(prior_path)):
        raise ValueError("validation-evaluation prior input hash changed")


def _validate_probe_analysis_source(
    plan: Mapping[str, Any], output: Mapping[str, Any]
) -> str | None:
    if plan.get("evidence_scope") != "probe":
        return None
    environment = plan.get("environment", {})
    selection = _read_json_object(Path(str(environment["PROBE_SELECTION"])))
    selection_hash = str(selection.get("selection_hash") or "")
    if not selection_hash or output.get("selection_hash") != selection_hash:
        raise ValueError(
            "probe analysis selection hash does not match its sample source"
        )
    probe_ids = tuple(str(value) for value in selection.get("probe_sample_ids", ()))
    if stable_hash(probe_ids) != output.get("sample_order_hash"):
        raise ValueError(
            "probe analysis sample order does not match its frozen selection"
        )
    if selection.get("representation_hash") != plan.get("representation_hash"):
        raise ValueError("probe analysis selection uses a different representation")
    return selection_hash


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def _assert_no_registered_content_conflict(
    registry: ArtifactRegistry, record: ArtifactRecord
) -> None:
    existing_records = registry.query(
        artifact_kind=record.artifact_kind, dataset_id=record.dataset_id,
    )
    for existing in existing_records:
        if existing.path != record.path:
            continue
        if existing.content_sha256 != record.content_sha256:
            raise ValueError(
                "artifact path is already registered with different content"
            )
        return


def _scope_to_split_for_record(scope: str) -> str | None:
    if scope in {"probe", "train", "full_train"}:
        return "train"
    if scope == "validation":
        return "validation"
    if scope in {"test", "external_test"}:
        return "test"
    return None
