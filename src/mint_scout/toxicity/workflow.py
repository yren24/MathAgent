from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from mint_scout.config import load_yaml
from mint_scout.data.manifest_io import task_card_from_config
from mint_scout.execution.profile import load_execution_profile
from mint_scout.execution.jobs import query_slurm_job
from mint_scout.run_pipeline import PipelineLaunchPlan
from mint_scout.toxicity.selection_contract import ToxicitySelectionContract


TOXICITY_WORKFLOW_VERSION = "mint-agent.toxicity-workflow.v1"


def build_toxicity_launch_plan(path: str | Path) -> PipelineLaunchPlan:
    config_path = Path(path).expanduser().resolve()
    raw = load_yaml(config_path)
    if str(raw.get("version") or "") != TOXICITY_WORKFLOW_VERSION:
        raise ValueError(
            f"toxicity workflow config version must be {TOXICITY_WORKFLOW_VERSION!r}"
        )
    repository_root = Path(str(raw.get("repository_root") or config_path.parent)).expanduser()
    if not repository_root.is_absolute():
        repository_root = (config_path.parent / repository_root).resolve()
    else:
        repository_root = repository_root.resolve()
    profile_path = _resolve_configured_path(
        raw.get("execution_profile"), repository_root, "execution_profile"
    )
    task_path = _resolve_configured_path(
        raw.get("task_config"), repository_root, "task_config"
    )
    task = load_yaml(task_path)
    selection = ToxicitySelectionContract.from_task(task_path)
    if raw.get("primary_metric") is not None and str(raw["primary_metric"]).upper() != selection.primary_metric:
        raise ValueError("toxicity pipeline primary_metric differs from task_config")
    if raw.get("invariants") is not None and tuple(raw["invariants"]) != selection.requested_invariants:
        raise ValueError("toxicity pipeline invariants differ from task_config")
    if raw.get("selection_objective") is not None and raw["selection_objective"] != selection.selection_objective:
        raise ValueError("toxicity pipeline selection_objective differs from task_config")
    if raw.get("user_target") is not None and float(raw["user_target"]) != selection.user_target:
        raise ValueError("toxicity pipeline user_target differs from task_config")
    profile = load_execution_profile(profile_path)
    if not profile.supports_slurm:
        raise ValueError("toxicity workflow requires a Slurm execution profile")
    dataset_id = str(raw.get("dataset_id") or task.get("task_id") or "")
    if not dataset_id or dataset_id != str(task.get("task_id") or ""):
        raise ValueError("toxicity workflow dataset_id must match task_config.task_id")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", dataset_id):
        raise ValueError(
            "toxicity workflow dataset_id must contain only letters, digits, dot, dash, or underscore"
        )
    run_id = str(raw.get("run_id") or "")
    if not run_id or not re.fullmatch(r"[A-Za-z0-9_.-]+", run_id):
        raise ValueError(
            "toxicity workflow run_id must contain only letters, digits, dot, dash, or underscore"
        )
    workflow_root = Path(
        str(
            raw.get("workflow_root")
            or Path(profile.run_root) / "toxicity" / dataset_id / run_id
        )
    ).expanduser()
    registry_path = Path(
        str(
            raw.get("registry_path")
            or Path(profile.scratch_root) / "memory" / "artifacts.sqlite"
        )
    ).expanduser()
    receipt_path = Path(
        str(raw.get("receipt_path") or workflow_root / "toxicity.launch.json")
    ).expanduser()
    state_path = Path(
        str(raw.get("state_path") or workflow_root / "toxicity.workflow.json")
    ).expanduser()
    resources = profile.slurm_defaults
    if resources is None:
        raise ValueError("toxicity workflow execution profile requires slurm_defaults")

    molecule_dirs = _configured_molecule_dirs(
        task,
        task_path=task_path,
        fallback=raw.get("molecule_dirs"),
    )
    environment = {
        "MINT_AGENT_ROOT": profile.project_root,
        "TOXICITY_WORKFLOW_CONFIG": _remote_path(config_path, repository_root, profile.project_root),
        "TASK_CONFIG": _remote_path(task_path, repository_root, profile.project_root),
        "DATASET_ID": dataset_id,
        "PRIMARY_METRIC": selection.primary_metric,
        "RUN_ID": run_id,
        "MANIFEST_PATH": str(raw.get("dataset_manifest") or task["dataset_manifest"]["path"]),
        "DESIGN_ROOT": str(raw.get("design_root") or workflow_root / "design"),
        "ADVISORY_ROOT": str(raw.get("advisory_root") or workflow_root / "design-advisory"),
        "PROBE_ROOT": str(raw.get("probe_root") or workflow_root / "probe"),
        "FEATURE_ROOT": str(raw.get("feature_root") or Path(profile.cache_root) / "toxicity_features" / dataset_id),
        "FINAL_ROOT": str(raw.get("final_root") or workflow_root / "final"),
        "WORKFLOW_JOBS_PATH": str(workflow_root / "toxicity.jobs.json"),
        "GBT_CONFIG": str(raw.get("gbt_config") or "configs/gbt/toxicity_probe_gbt.yaml"),
        "MOLECULE_DIRS": ":".join(molecule_dirs),
        "LEGACY_ROOT": str(
            raw.get("legacy_root")
            or Path(profile.scratch_root) / "external" / "toxicity-phase0" / "legacy"
        ),
        "LLM_ENABLED": "0",
    }
    environment.update(_llm_environment(raw.get("llm_scientific")))
    _validate_environment(environment)
    submit_script = _remote_path(
        repository_root / "scripts" / "sapelo2" / "submit_toxicity_workflow.sh",
        repository_root,
        profile.project_root,
    )
    job_name = f"mint-tox-{_safe_token(dataset_id)}"
    submit_command = (
        "env",
        *(f"{key}={value}" for key, value in environment.items()),
        "sbatch",
        f"--job-name={job_name}",
        f"--partition={resources.partition}",
        "--ntasks=1",
        "--cpus-per-task=1",
        "--mem=2gb",
        "--time=00:20:00",
        f"--output={profile.log_root}/{job_name}-%j.out",
        f"--error={profile.log_root}/{job_name}-%j.err",
        submit_script,
    )
    return PipelineLaunchPlan(
        config_path=str(config_path),
        repository_root=str(repository_root),
        dataset_id=dataset_id,
        run_id=run_id,
        task_config=str(task_path),
        execution_profile=str(profile_path),
        state_path=str(state_path),
        registry_path=str(registry_path),
        receipt_path=str(receipt_path),
        submit_command=submit_command,
    )


def write_toxicity_workflow_state(
    *,
    launch_plan: PipelineLaunchPlan,
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    state = {
        "report_schema": "mint-agent.toxicity-workflow-state.v1",
        "dataset_id": launch_plan.dataset_id,
        "run_id": launch_plan.run_id,
        "status": "SUBMITTED",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "receipt": dict(receipt),
        "task_config": launch_plan.task_config,
        "pipeline_config": launch_plan.config_path,
    }
    destination = Path(launch_plan.state_path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(
        json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(destination)
    return state


def toxicity_workflow_status(plan: PipelineLaunchPlan) -> dict[str, Any]:
    final_path = _final_report_path(plan)
    result: dict[str, Any] = {
        "report_schema": "mint-agent.pipeline-status.v1",
        "dataset_id": plan.dataset_id,
        "run_id": plan.run_id,
        "pipeline_kind": "small_molecule_toxicity_gbt_v1",
        "state_path": plan.state_path,
        "final_report": str(final_path),
    }
    if final_path.is_file():
        report = _load_final_report(final_path)
        result.update(
            status=report["status"],
            selected_subset=report["selected_subset"],
            primary_metric=report["primary_metric"],
            user_target=report.get("user_target"),
            final_target_met=report.get("final_target_met"),
            final_metrics=report["final_test"]["metrics"],
        )
        return result

    state_path = Path(plan.state_path)
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if (
            not isinstance(state, dict)
            or state.get("dataset_id") != plan.dataset_id
            or state.get("run_id") != plan.run_id
        ):
            raise ValueError("toxicity workflow state does not match launch plan")
        job_id = str(state.get("receipt", {}).get("job_id") or "")
        result["launch_job_id"] = job_id
        if job_id:
            scheduler = query_slurm_job(job_id)
            result["launch_scheduler"] = scheduler
            if scheduler.get("state") in {
                "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY",
            }:
                result["status"] = "LAUNCH_FAILED"
                return result
        jobs_path = _jobs_path(plan)
        if jobs_path.is_file():
            jobs = json.loads(jobs_path.read_text(encoding="utf-8"))
            if (
                not isinstance(jobs, dict)
                or jobs.get("dataset_id") != plan.dataset_id
                or jobs.get("run_id") != plan.run_id
            ):
                raise ValueError("toxicity job receipt does not match launch plan")
            final_job = (
                str(jobs.get("final_combine_job") or "")
                if str(jobs.get("launch_job_id") or "") == job_id
                else ""
            )
            if final_job:
                result["final_job_id"] = final_job
                final_scheduler = query_slurm_job(final_job)
                result["final_scheduler"] = final_scheduler
                if final_scheduler.get("state") in {
                    "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY",
                }:
                    result["status"] = "FAILED"
                    return result
                if final_scheduler.get("state") == "COMPLETED":
                    result["status"] = "NEEDS_REVIEW"
                    result["reason"] = "final job completed without a valid final report"
                    return result
        result["status"] = "SUBMITTED"
        return result

    receipt_path = Path(plan.receipt_path)
    if receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        result["status"] = "LAUNCHING"
        result["launch_job_id"] = receipt["job_id"]
        result["launch_scheduler"] = query_slurm_job(str(receipt["job_id"]))
        return result
    result["status"] = "NOT_STARTED"
    return result


def toxicity_workflow_report(plan: PipelineLaunchPlan) -> dict[str, Any]:
    final_path = _final_report_path(plan)
    if not final_path.is_file():
        raise FileNotFoundError(f"toxicity final report is not ready: {final_path}")
    report = _load_final_report(final_path)
    return {
        "report_schema": "mint-agent.workflow-report.v1",
        "dataset_id": plan.dataset_id,
        "run_id": plan.run_id,
        "pipeline_kind": "small_molecule_toxicity_gbt_v1",
        "status": report["status"],
        "final_report": str(final_path),
        "selected_subset": report["selected_subset"],
        "primary_metric": report["primary_metric"],
        "selection_objective": report.get("selection_objective"),
        "user_target": report.get("user_target"),
        "probe_target_met": report.get("probe_target_met"),
        "final_target_met": report.get("final_target_met"),
        "final_metrics": report["final_test"]["metrics"],
        "protocol": report["protocol"],
    }


def _final_report_path(plan: PipelineLaunchPlan) -> Path:
    raw = load_yaml(Path(plan.config_path))
    workflow_root = Path(plan.state_path).parent
    final_root = Path(str(raw.get("final_root") or workflow_root / "final"))
    return final_root / "gbt" / "toxicity_final_test_report.json"


def _jobs_path(plan: PipelineLaunchPlan) -> Path:
    prefix = "WORKFLOW_JOBS_PATH="
    for argument in plan.submit_command:
        if argument.startswith(prefix):
            return Path(argument[len(prefix):])
    return Path(plan.state_path).parent / "toxicity.jobs.json"


def _load_final_report(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(report, dict)
        or report.get("report_schema") != "mint-agent.toxicity-final-evaluation.v1"
        or report.get("status") not in {"COMPLETE", "TARGET_NOT_REACHED"}
        or not isinstance(report.get("selected_subset"), list)
        or not isinstance(report.get("final_test"), dict)
        or not isinstance(report["final_test"].get("metrics"), dict)
    ):
        raise ValueError(f"toxicity final report is incomplete or invalid: {path}")
    return report


def _configured_molecule_dirs(
    task: Mapping[str, Any],
    *,
    task_path: Path,
    fallback: object,
) -> tuple[str, ...]:
    configured = _string_sequence(fallback)
    if configured:
        return configured
    card = task_card_from_config(task, config_path=task_path)
    if card is None:
        raise ValueError("toxicity task requires dataset_manifest")
    parents = sorted(
        {
            str(path.expanduser().resolve().parent)
            for sample in card.dataset.samples
            for role, path in sample.role_paths.items()
            if role == "molecule"
        }
    )
    if not parents:
        raise ValueError("toxicity manifest does not contain molecule paths")
    return tuple(parents)


def _llm_environment(value: object) -> dict[str, str]:
    if not isinstance(value, Mapping):
        return {}
    mode = str(value.get("mode") or "disabled")
    if mode not in {"disabled", "shadow", "advisory"}:
        raise ValueError("toxicity llm_scientific.mode must be disabled, shadow, or advisory")
    if mode == "disabled":
        return {}
    result = {
        "LLM_ENABLED": "1" if mode == "advisory" else "0",
        "LLM_SCIENTIFIC_MODE": mode,
    }
    for source, target in (
        ("model", "LLM_MODEL"),
        ("api_key_env", "LLM_API_KEY_ENV"),
        ("env_file", "LLM_ENV_FILE"),
        ("cache_dir", "LLM_CACHE_DIR"),
    ):
        item = value.get(source)
        if item is not None:
            result[target] = str(item)
    return result


def _resolve_configured_path(value: object, root: Path, name: str) -> Path:
    if value is None:
        raise ValueError(f"toxicity workflow {name} is required")
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _remote_path(path: Path, repository_root: Path, remote_project_root: str) -> str:
    try:
        relative = path.resolve().relative_to(repository_root.resolve())
    except ValueError:
        return str(path)
    return str(Path(remote_project_root) / relative)


def _string_sequence(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError("toxicity workflow molecule_dirs must be a list")
    result = tuple(str(item) for item in value if str(item).strip())
    if len(set(result)) != len(result):
        raise ValueError("toxicity workflow molecule_dirs must not contain duplicates")
    return result


def _validate_environment(environment: Mapping[str, str]) -> None:
    for key, value in environment.items():
        if not re.fullmatch(r"[A-Z_][A-Z0-9_]*", key):
            raise ValueError(f"invalid toxicity workflow environment key: {key!r}")
        if "\x00" in value or "\n" in value:
            raise ValueError(f"invalid toxicity workflow environment value for {key}")


def _safe_token(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value)[:80] or "dataset"
