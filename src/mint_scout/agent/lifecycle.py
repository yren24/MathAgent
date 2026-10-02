from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping
from uuid import uuid4

from mint_scout.execution.jobs import (
    CommandRunner,
    create_job_ledger,
    ledger_summary,
    load_job_ledger,
    parse_sbatch_job_id,
    reconcile_completed_jobs,
    refresh_job_ledger,
    resume_job_ledger,
    run_command,
    submit_ready_jobs,
    write_job_ledger,
)
from mint_scout.execution.profile import load_execution_profile
from mint_scout.invariants.manifest import stable_hash
from mint_scout.run_agent_graph import main as run_graph_main


LIFECYCLE_SCHEMA = "mint-agent.lifecycle.v1"
TERMINAL_STATUSES = {"COMPLETE", "COMPLETE_TARGET_NOT_REACHED"}
JOB_PLAN_KEYS = (
    "preparation_jobs",
    "setup_jobs",
    "feature_jobs",
    "qc_jobs",
    "filtration_audit_jobs",
    "feature_diagnostic_jobs",
    "scout_jobs",
    "evaluation_jobs",
)

GraphRunner = Callable[[Mapping[str, Any], Path], Mapping[str, Any]]


def initialize_lifecycle(
    path: str | Path,
    *,
    task_config: str | Path,
    registry: str | Path,
    dataset_id: str,
    invariants: tuple[str, ...],
    execution_profile: str | Path,
    evidence_scope: str = "full_train",
    representation_spec: str | Path | None = None,
    sample_id_file: str | Path | None = None,
    qc_config: str | Path = "configs/scout/v1.yaml",
    audit_config: str | Path = "configs/scout/v1.yaml",
    feature_diagnostic_top_k: int = 10,
    scout_config: str | Path = "configs/scout/v1.yaml",
    gbt_config: str | Path = "configs/gbt/plbind_adaptive_gbt.yaml",
    user_target: float | None = None,
    llm_scientific_mode: str = "disabled",
    llm_model: str | None = None,
    llm_api_key_env: str = "OPENAI_API_KEY",
    llm_env_file: str | Path | None = None,
    llm_cache_dir: str | Path | None = None,
    llm_max_candidates: int = 10,
    llm_timeout_seconds: float = 180.0,
    lifecycle_id: str | None = None,
) -> dict[str, Any]:
    destination = Path(path).expanduser().resolve()
    normalized_invariants = tuple(
        dict.fromkeys(str(value).upper() for value in invariants)
    )
    if not dataset_id or not normalized_invariants:
        raise ValueError("dataset_id and at least one invariant are required")
    if feature_diagnostic_top_k < 1:
        raise ValueError("feature_diagnostic_top_k must be positive")
    if llm_scientific_mode not in {"disabled", "shadow", "advisory"}:
        raise ValueError("unsupported llm_scientific_mode")
    if llm_max_candidates < 1:
        raise ValueError("llm_max_candidates must be positive")
    if llm_timeout_seconds <= 0:
        raise ValueError("llm_timeout_seconds must be positive")
    resolved_lifecycle_id = lifecycle_id or f"mint-agent-{uuid4()}"
    config = {
        "task_config": str(task_config),
        "registry": str(registry),
        "dataset_id": str(dataset_id),
        "invariants": list(normalized_invariants),
        "execution_profile": str(execution_profile),
        "evidence_scope": str(evidence_scope),
        "representation_spec": _optional_string(representation_spec),
        "sample_id_file": _optional_string(sample_id_file),
        "qc_config": str(qc_config),
        "audit_config": str(audit_config),
        "feature_diagnostic_top_k": feature_diagnostic_top_k,
        "scout_config": str(scout_config),
        "gbt_config": str(gbt_config),
        "user_target": user_target,
        "llm_scientific_mode": llm_scientific_mode,
        "llm_model": _optional_string(llm_model),
        "llm_api_key_env": str(llm_api_key_env),
        "llm_env_file": _optional_string(llm_env_file),
        "llm_cache_dir": _optional_string(llm_cache_dir),
        "llm_max_candidates": llm_max_candidates,
        "llm_timeout_seconds": float(llm_timeout_seconds),
        "thread_id": resolved_lifecycle_id,
    }
    if destination.exists():
        existing = load_lifecycle(destination)
        if existing["config"] != config:
            raise ValueError("existing lifecycle configuration is immutable")
        return existing
    now = _utc_now()
    state = {
        "report_schema": LIFECYCLE_SCHEMA,
        "lifecycle_id": resolved_lifecycle_id,
        "status": "INITIALIZED",
        "config": config,
        "created_at": now,
        "updated_at": now,
        "active_stage_index": None,
        "stages": [],
        "continuations": [],
        "events": [{"at": now, "event": "LIFECYCLE_INITIALIZED",}],
    }
    write_lifecycle(state, destination)
    return state


def advance_lifecycle(
    path: str | Path,
    *,
    execute: bool = False,
    auto_continue: bool = False,
    continuation_script: str | Path | None = None,
    project_root: str | Path | None = None,
    resume_failed: bool = False,
    max_attempts: int = 2,
    stop_after_stage: int | None = None,
    graph_runner: GraphRunner | None = None,
    command_runner: CommandRunner | None = None,
) -> dict[str, Any]:
    destination = Path(path).expanduser().resolve()
    state = load_lifecycle(destination)
    if state["status"] in TERMINAL_STATUSES:
        if not isinstance(state.get("report_bundle"), Mapping):
            _attach_pipeline_report(state, destination)
            write_lifecycle(state, destination)
        return state
    if auto_continue and (continuation_script is None or project_root is None):
        profile = load_execution_profile(state["config"]["execution_profile"])
        continuation_script = continuation_script or profile.lifecycle_script
        project_root = project_root or profile.project_root
    if auto_continue and not execute:
        raise ValueError("auto_continue requires execute=True")
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")
    if stop_after_stage is not None and stop_after_stage < 0:
        raise ValueError("stop_after_stage must be non-negative")
    runner = command_runner or run_command

    active_stage = _active_stage(state)
    if active_stage is not None:
        if active_stage.get("ledger"):
            if not _advance_active_stage(
                state,
                active_stage,
                destination=destination,
                execute=execute,
                auto_continue=auto_continue,
                continuation_script=continuation_script,
                project_root=project_root,
                resume_failed=resume_failed,
                max_attempts=max_attempts,
                stop_after_stage=stop_after_stage,
                command_runner=runner,
            ):
                write_lifecycle(state, destination)
                return state
        elif resume_failed and active_stage.get("status") in {
            "NEEDS_REVIEW",
            "NEEDS_USER_ACTION",
        }:
            active_stage["resume_override_at"] = _utc_now()
            state["active_stage_index"] = None
            state["status"] = "ADVANCING"
            _event(
                state,
                "GRAPH_STAGE_RESUME_OVERRIDE",
                iteration=active_stage.get("iteration"),
                previous_status=active_stage.get("status"),
            )
        else:
            state["status"] = "NEEDS_REVIEW"
            _event(
                state,
                "ACTIVE_STAGE_HAS_NO_LEDGER",
                iteration=active_stage.get("iteration"),
                stage_status=active_stage.get("status"),
            )
            write_lifecycle(state, destination)
            return state

    iteration = len(state["stages"])
    report_path = _stage_path(destination, "graph", iteration)
    ledger_path = _stage_path(destination, "ledger", iteration)
    report = dict((graph_runner or _run_graph)(state["config"], report_path))
    _write_json(report, report_path)
    graph_status = str(report.get("status") or "")
    plan_signature = _plan_signature(report)
    stage = {
        "iteration": iteration,
        "graph_report": str(report_path),
        "graph_status": graph_status,
        "ledger": None,
        "status": "GRAPH_COMPLETE",
        "continuation": None,
        "plan_signature": plan_signature,
    }
    state["stages"].append(stage)
    _event(
        state,
        "GRAPH_EVALUATED",
        iteration=iteration,
        graph_status=graph_status,
        graph_report=str(report_path),
    )

    repeated_stage = next(
        (
            previous
            for previous in state["stages"][:-1]
            if plan_signature
            and previous.get("plan_signature") == plan_signature
            and previous.get("status") == "REGISTERED"
        ),
        None,
    )
    if repeated_stage is not None:
        stage["status"] = "NEEDS_REVIEW"
        stage["reason"] = "graph repeated a previously registered job plan"
        state["status"] = "NEEDS_REVIEW"
    elif graph_status == "COMPLETE":
        stage["status"] = "COMPLETE"
        state["status"] = "COMPLETE"
    elif graph_status == "TARGET_NOT_REACHED":
        stage["status"] = "COMPLETE_TARGET_NOT_REACHED"
        state["status"] = "COMPLETE_TARGET_NOT_REACHED"
    elif graph_status.startswith("NEEDS_") and _job_plan_count(report):
        ledger = create_job_ledger(report, source_plan=report_path)
        write_job_ledger(ledger, ledger_path)
        stage["ledger"] = str(ledger_path)
        stage["status"] = "PLANNED"
        state["active_stage_index"] = iteration
        if execute:
            submit_ready_jobs(ledger, execute=True, runner=runner)
            write_job_ledger(ledger, ledger_path)
            _event(
                state,
                "JOBS_SUBMITTED",
                iteration=iteration,
                ledger=str(ledger_path),
                summary=ledger_summary(ledger),
            )
        _set_waiting_status(state, ledger)
        if auto_continue and ledger.get("status") == "ACTIVE":
            write_lifecycle(state, destination)
            _schedule_continuation(
                state,
                stage,
                ledger,
                destination=destination,
                continuation_script=Path(str(continuation_script)),
                project_root=project_root,
                resume_failed=resume_failed,
                max_attempts=max_attempts,
                stop_after_stage=stop_after_stage,
                command_runner=runner,
            )
    elif graph_status.startswith("NEEDS_"):
        stage["status"] = "NEEDS_USER_ACTION"
        state["status"] = "NEEDS_USER_ACTION"
    else:
        stage["status"] = "NEEDS_REVIEW"
        state["status"] = "NEEDS_REVIEW"

    if state["status"] in TERMINAL_STATUSES:
        _attach_pipeline_report(state, destination)
    write_lifecycle(state, destination)
    return state


def load_lifecycle(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("lifecycle state must contain a JSON object")
    _validate_lifecycle(payload)
    return payload


def write_lifecycle(state: Mapping[str, Any], path: str | Path) -> None:
    _validate_lifecycle(state)
    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    updated_at = _utc_now()
    if isinstance(state, dict):
        state["updated_at"] = updated_at
    payload = dict(state)
    payload["updated_at"] = updated_at
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(destination)


def lifecycle_summary(state: Mapping[str, Any]) -> dict[str, Any]:
    active = _active_stage(state)
    return {
        "report_schema": state.get("report_schema"),
        "lifecycle_id": state.get("lifecycle_id"),
        "status": state.get("status"),
        "stage_count": len(state.get("stages", [])),
        "active_stage_index": state.get("active_stage_index"),
        "active_ledger": active.get("ledger") if active else None,
        "continuation_count": len(state.get("continuations", [])),
    }


def _advance_active_stage(
    state: dict[str, Any],
    stage: dict[str, Any],
    *,
    destination: Path,
    execute: bool,
    auto_continue: bool,
    continuation_script: str | Path | None,
    project_root: str | Path | None,
    resume_failed: bool,
    max_attempts: int,
    stop_after_stage: int | None,
    command_runner: CommandRunner,
) -> bool:
    ledger_path = Path(str(stage["ledger"]))
    ledger = load_job_ledger(ledger_path)
    refresh_job_ledger(ledger, query_scheduler=True, runner=command_runner)
    if ledger.get("status") == "COMPLETE":
        reconcile_completed_jobs(ledger, registry=state["config"]["registry"])
    if ledger.get("status") == "NEEDS_REVIEW" and resume_failed:
        reconcile_completed_jobs(ledger, registry=state["config"]["registry"])
    if ledger.get("status") == "NEEDS_RESUME" and resume_failed:
        exhausted = [
            job["plan_id"]
            for job in ledger.get("jobs", [])
            if job.get("state") in {"NEEDS_RESUME", "OUTPUT_INCOMPLETE"}
            and len(job.get("attempts", [])) >= max_attempts
        ]
        if exhausted:
            stage["status"] = "NEEDS_REVIEW"
            stage["attempts_exhausted"] = exhausted
            state["status"] = "NEEDS_REVIEW"
            write_job_ledger(ledger, ledger_path)
            return False
        if execute:
            resume_job_ledger(ledger, execute=True, runner=command_runner)
            stage.pop("attempts_exhausted", None)
            _event(
                state,
                "JOBS_RESUBMITTED",
                iteration=stage["iteration"],
                ledger=str(ledger_path),
                summary=ledger_summary(ledger),
            )
    write_job_ledger(ledger, ledger_path)

    if ledger.get("status") == "COMPLETE":
        stage.pop("attempts_exhausted", None)
        stage["status"] = "REGISTERED"
        state["active_stage_index"] = None
        state["status"] = "ADVANCING"
        _event(
            state,
            "JOBS_RECONCILED",
            iteration=stage["iteration"],
            ledger=str(ledger_path),
        )
        if stop_after_stage is not None and stage["iteration"] >= stop_after_stage:
            state["status"] = "PAUSED_AFTER_STAGE"
            _event(
                state,
                "LIFECYCLE_PAUSED",
                iteration=stage["iteration"],
                reason="configured stop_after_stage reached",
            )
            return False
        return True

    if ledger.get("status") == "PLANNED" and execute:
        submit_ready_jobs(ledger, execute=True, runner=command_runner)
        write_job_ledger(ledger, ledger_path)
    _set_waiting_status(state, ledger)
    if auto_continue and ledger.get("status") == "ACTIVE":
        write_lifecycle(state, destination)
        _schedule_continuation(
            state,
            stage,
            ledger,
            destination=destination,
            continuation_script=Path(str(continuation_script)),
            project_root=project_root,
            resume_failed=resume_failed,
            max_attempts=max_attempts,
            stop_after_stage=stop_after_stage,
            command_runner=command_runner,
        )
    return False


def _schedule_continuation(
    state: dict[str, Any],
    stage: dict[str, Any],
    ledger: Mapping[str, Any],
    *,
    destination: Path,
    continuation_script: Path,
    project_root: str | Path | None,
    resume_failed: bool,
    max_attempts: int,
    stop_after_stage: int | None,
    command_runner: CommandRunner,
) -> None:
    dependency_ids = tuple(
        str(attempts[-1]["job_id"])
        for job in ledger.get("jobs", [])
        if (attempts := job.get("attempts")) and attempts[-1].get("job_id")
    )
    if not dependency_ids:
        state["status"] = "NEEDS_REVIEW"
        stage["status"] = "NEEDS_REVIEW"
        stage["continuation_error"] = "active ledger has no submitted Slurm job IDs"
        return
    existing = stage.get("continuation")
    if (
        isinstance(existing, Mapping)
        and tuple(existing.get("dependency_job_ids", ())) == dependency_ids
    ):
        return
    export_values = [
        "ALL",
        f"LIFECYCLE={destination}",
        "EXECUTE=1",
        "AUTO_CONTINUE=1",
        f"AUTO_RESUME_FAILED={1 if resume_failed else 0}",
        f"MAX_ATTEMPTS={max_attempts}",
    ]
    if stop_after_stage is not None:
        export_values.append(f"STOP_AFTER_STAGE={stop_after_stage}")
    scheduler_log_options: tuple[str, ...] = ()
    try:
        profile = load_execution_profile(state["config"]["execution_profile"])
    except FileNotFoundError:
        profile = None
    if profile is not None:
        log_root = Path(profile.log_root)
        scheduler_log_options = (
            f"--output={log_root}/mint-agent-lifecycle-%j.out",
            f"--error={log_root}/mint-agent-lifecycle-%j.err",
        )
    command = (
        "sbatch",
        f"--dependency=afterany:{':'.join(dependency_ids)}",
        f"--export={','.join(export_values)}",
        *scheduler_log_options,
        str(continuation_script),
    )
    result = command_runner(command, _optional_string(project_root))
    continuation = {
        "submitted_at": _utc_now(),
        "dependency_job_ids": list(dependency_ids),
        "command": list(command),
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
        "job_id": None,
    }
    if result.returncode == 0:
        try:
            continuation["job_id"] = parse_sbatch_job_id(result.stdout)
        except ValueError as exc:
            continuation["stderr"] = str(exc)
    if continuation["job_id"] is None:
        state["status"] = "NEEDS_REVIEW"
        stage["status"] = "NEEDS_REVIEW"
        stage["continuation_error"] = (
            continuation["stderr"] or "continuation submission failed"
        )
    else:
        stage["continuation"] = continuation
        state["continuations"].append(continuation)
        _event(
            state,
            "CONTINUATION_SUBMITTED",
            iteration=stage["iteration"],
            continuation_job_id=continuation["job_id"],
            dependency_job_ids=list(dependency_ids),
        )


def _run_graph(config: Mapping[str, Any], output: Path) -> Mapping[str, Any]:
    argv = [
        "--task-config",
        str(config["task_config"]),
        "--registry",
        str(config["registry"]),
        "--dataset-id",
        str(config["dataset_id"]),
        "--evidence-scope",
        str(config["evidence_scope"]),
        "--execution-profile",
        str(config["execution_profile"]),
        "--qc-config",
        str(config["qc_config"]),
        "--audit-config",
        str(config["audit_config"]),
        "--feature-diagnostic-top-k",
        str(config.get("feature_diagnostic_top_k", 10)),
        "--scout-config",
        str(config["scout_config"]),
        "--gbt-config",
        str(config["gbt_config"]),
        "--thread-id",
        str(config["thread_id"]),
        "--output",
        str(output),
        "--llm-scientific-mode",
        str(config.get("llm_scientific_mode", "disabled")),
        "--llm-api-key-env",
        str(config.get("llm_api_key_env", "OPENAI_API_KEY")),
        "--llm-max-candidates",
        str(config.get("llm_max_candidates", 10)),
        "--llm-timeout-seconds",
        str(config.get("llm_timeout_seconds", 180.0)),
    ]
    for invariant in config["invariants"]:
        argv.extend(("--invariant", str(invariant)))
    for option, key in (
        ("--representation-spec", "representation_spec"),
        ("--sample-id-file", "sample_id_file"),
    ):
        if config.get(key):
            argv.extend((option, str(config[key])))
    if config.get("user_target") is not None:
        argv.extend(("--user-target", str(config["user_target"])))
    if config.get("llm_model"):
        argv.extend(("--llm-model", str(config["llm_model"])))
    if config.get("llm_env_file"):
        argv.extend(("--llm-env-file", str(config["llm_env_file"])))
    if config.get("llm_cache_dir"):
        argv.extend(("--llm-cache-dir", str(config["llm_cache_dir"])))
    run_graph_main(argv)
    return _read_json(output)


def _job_plan_count(report: Mapping[str, Any]) -> int:
    return sum(
        len(value)
        for key in JOB_PLAN_KEYS
        if isinstance((value := report.get(key)), list)
    )


def _plan_signature(report: Mapping[str, Any]) -> str | None:
    plans = [
        plan
        for key in JOB_PLAN_KEYS
        if isinstance((values := report.get(key)), list)
        for plan in values
    ]
    return stable_hash(plans) if plans else None


def _active_stage(state: Mapping[str, Any]) -> dict[str, Any] | None:
    index = state.get("active_stage_index")
    if index is None:
        return None
    stages = state.get("stages")
    if not isinstance(index, int) or not isinstance(stages, list):
        raise ValueError("lifecycle active stage is invalid")
    return stages[index]


def _set_waiting_status(state: dict[str, Any], ledger: Mapping[str, Any]) -> None:
    status = str(ledger.get("status") or "UNKNOWN")
    active = _active_stage(state)
    if status == "ACTIVE":
        state["status"] = "WAITING_FOR_JOBS"
        if active is not None:
            active["status"] = "WAITING_FOR_JOBS"
    elif status == "PLANNED":
        state["status"] = "PLANNED"
        if active is not None:
            active["status"] = "PLANNED"
    else:
        state["status"] = "NEEDS_REVIEW"
        if active is not None:
            active["status"] = "NEEDS_REVIEW"


def _validate_lifecycle(state: Mapping[str, Any]) -> None:
    if state.get("report_schema") != LIFECYCLE_SCHEMA:
        raise ValueError(
            f"Unsupported lifecycle schema: {state.get('report_schema')!r}"
        )
    if not isinstance(state.get("config"), Mapping):
        raise ValueError("lifecycle config is missing")
    if not isinstance(state.get("stages"), list):
        raise ValueError("lifecycle stages must be a list")
    if not isinstance(state.get("events"), list):
        raise ValueError("lifecycle events must be a list")


def _stage_path(state_path: Path, kind: str, iteration: int) -> Path:
    root = state_path.parent / f"{state_path.stem}.artifacts"
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{kind}_{iteration:03d}.json"


def _attach_pipeline_report(state: dict[str, Any], state_path: Path) -> None:
    from mint_scout.pipeline_reporting import write_pipeline_report

    bundle = write_pipeline_report(state=state, state_path=state_path)
    state["report_bundle"] = bundle.to_dict()
    _event(state, "PIPELINE_REPORT_WRITTEN", **bundle.to_dict())


def _event(state: dict[str, Any], event: str, **details: Any) -> None:
    state["events"].append({"at": _utc_now(), "event": event, **details})


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def _write_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    normalized = str(value)
    return normalized if normalized else None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
