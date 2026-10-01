from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from mint_scout.agent.lifecycle import lifecycle_summary, load_lifecycle
from mint_scout.config import load_yaml
from mint_scout.execution.jobs import parse_sbatch_job_id, query_slurm_job
from mint_scout.execution.profile import ExecutionProfile, load_execution_profile


PIPELINE_CONFIG_VERSION = "mint-agent.pipeline.v1"


@dataclass(frozen=True)
class PipelineLaunchPlan:
    config_path: str
    repository_root: str
    dataset_id: str
    run_id: str
    task_config: str
    execution_profile: str
    state_path: str
    registry_path: str
    receipt_path: str
    submit_command: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "submit_command": list(self.submit_command)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Launch and inspect a complete config-driven mint-agent pipeline."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("plan", "submit", "status", "resume", "report"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--config", type=Path, required=True)
        if command in {"submit", "resume"}:
            subparser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)

    from mint_scout.agent.workflow import workflow_for_config

    workflow = workflow_for_config(args.config)
    plan = workflow.build_launch_plan(args.config)
    if args.command == "plan":
        print(json.dumps(plan.to_dict(), indent=2, sort_keys=True))
        return 0
    if args.command == "status":
        summary = workflow.status(plan)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0 if summary["status"] != "NOT_STARTED" else 2
    if args.command == "report":
        print(json.dumps(workflow.report(plan), indent=2, sort_keys=True))
        return 0
    if args.command == "resume" and not workflow.resumable:
        raise ValueError(
            "toxicity uses a Slurm dependency chain; inspect status and failed jobs "
            "before submitting a new run"
        )
    if args.command == "resume" and not Path(plan.state_path).is_file():
        raise FileNotFoundError(
            f"Cannot resume before lifecycle state exists: {plan.state_path}"
        )
    if not args.execute:
        print(json.dumps(plan.to_dict(), indent=2, sort_keys=True))
        return 0
    receipt = workflow.submit(plan)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


def build_pipeline_launch_plan(path: str | Path) -> PipelineLaunchPlan:
    config_path = Path(path).expanduser().resolve()
    raw = load_yaml(config_path)
    version = str(raw.get("version") or "")
    if version != PIPELINE_CONFIG_VERSION:
        raise ValueError(f"pipeline config version must be {PIPELINE_CONFIG_VERSION!r}")
    repository_root = _configured_repository_root(config_path, raw)
    task_path = _repository_path(raw.get("task_config"), repository_root, "task_config")
    profile_path = _repository_path(
        raw.get("execution_profile"), repository_root, "execution_profile"
    )
    task = load_yaml(task_path)
    profile = load_execution_profile(profile_path)
    if not profile.supports_slurm:
        raise ValueError("V1 complete pipeline launcher requires a Slurm profile")
    dataset_id = str(raw.get("dataset_id") or task.get("task_id") or "")
    if not dataset_id or dataset_id != str(task.get("task_id") or ""):
        raise ValueError("pipeline dataset_id must match task_config.task_id")
    invariants = _invariants(raw.get("invariants", task.get("invariants")))
    run_id = str(raw.get("run_id") or "")
    if not run_id or not re.fullmatch(r"[A-Za-z0-9_.-]+", run_id):
        raise ValueError(
            "pipeline run_id must contain only letters, digits, dot, dash, or underscore"
        )

    state_path = Path(
        str(
            raw.get("state_path")
            or Path(profile.run_root) / "pipelines" / f"{dataset_id}_{run_id}.json"
        )
    ).expanduser()
    registry_path = Path(
        str(
            raw.get("registry_path")
            or Path(profile.scratch_root) / "memory" / "artifacts.sqlite"
        )
    ).expanduser()
    receipt_path = Path(
        str(raw.get("receipt_path") or state_path.with_suffix(".launch.json"))
    ).expanduser()
    lifecycle_id = str(raw.get("lifecycle_id") or f"{dataset_id}-{run_id}")
    resources = profile.slurm_defaults
    if resources is None:
        raise ValueError("execution profile requires slurm_defaults")

    environment = {
        "MINT_AGENT_ROOT": profile.project_root,
        "RUN_ROOT": profile.run_root,
        "REGISTRY": str(registry_path),
        "TASK_CONFIG": _remote_repository_path(task_path, repository_root, profile),
        "EXECUTION_PROFILE": _remote_repository_path(
            profile_path, repository_root, profile
        ),
        "DATASET_ID": dataset_id,
        "INVARIANTS": ",".join(invariants),
        "LIFECYCLE": str(state_path),
        "LIFECYCLE_ID": lifecycle_id,
        "EXECUTE": "1",
        "AUTO_CONTINUE": "1",
        "AUTO_RESUME_FAILED": "1"
        if bool(raw.get("auto_resume_failed", False))
        else "0",
        "MAX_ATTEMPTS": str(int(raw.get("max_attempts", 2))),
        "EVIDENCE_SCOPE": str(raw.get("evidence_scope", "full_train")),
        "QC_CONFIG": str(raw.get("qc_config", "configs/scout/v1.yaml")),
        "AUDIT_CONFIG": str(raw.get("audit_config", "configs/scout/v1.yaml")),
        "SCOUT_CONFIG": str(raw.get("scout_config", "configs/scout/v1.yaml")),
        "GBT_CONFIG": str(
            raw.get("gbt_config", "configs/gbt/plbind_adaptive_gbt.yaml")
        ),
    }
    environment.update(_llm_scientific_environment(raw.get("llm_scientific")))
    if raw.get("user_target") is not None:
        environment["USER_TARGET"] = str(float(raw["user_target"]))
    if raw.get("feature_diagnostic_top_k") is not None:
        environment["FEATURE_DIAGNOSTIC_TOP_K"] = str(
            int(raw["feature_diagnostic_top_k"])
        )
    if raw.get("stop_after_stage") is not None:
        stop_after_stage = int(raw["stop_after_stage"])
        if stop_after_stage < 0:
            raise ValueError("pipeline stop_after_stage must be non-negative")
        environment["STOP_AFTER_STAGE"] = str(stop_after_stage)
    _validate_environment(environment)
    job_name = f"mint-pipeline-{_safe_token(dataset_id)}"
    lifecycle_script = _remote_path(profile.project_root, profile.lifecycle_script)
    submit_command = (
        "env",
        *(f"{key}={value}" for key, value in environment.items()),
        "sbatch",
        f"--job-name={job_name}",
        f"--partition={resources.partition}",
        "--ntasks=1",
        "--cpus-per-task=1",
        "--mem=2gb",
        "--time=00:15:00",
        f"--output={profile.log_root}/{job_name}-%j.out",
        f"--error={profile.log_root}/{job_name}-%j.err",
        lifecycle_script,
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


def submit_pipeline(plan: PipelineLaunchPlan) -> dict[str, Any]:
    completed = subprocess.run(
        plan.submit_command,
        cwd=plan.repository_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "pipeline submission failed")
    job_id = parse_sbatch_job_id(completed.stdout)
    receipt = {
        "report_schema": "mint-agent.pipeline-launch.v1",
        "dataset_id": plan.dataset_id,
        "run_id": plan.run_id,
        "submitted_at": datetime.now(timezone.utc).isoformat(),
        "job_id": job_id,
        "state_path": plan.state_path,
        "registry_path": plan.registry_path,
        "submit_command": list(plan.submit_command),
    }
    destination = Path(plan.receipt_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(destination)
    return receipt


def pipeline_status(plan: PipelineLaunchPlan) -> dict[str, Any]:
    state_path = Path(plan.state_path)
    if state_path.is_file():
        return {
            "report_schema": "mint-agent.pipeline-status.v1",
            **lifecycle_summary(load_lifecycle(state_path)),
            "state_path": plan.state_path,
        }
    receipt_path = Path(plan.receipt_path)
    if receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        scheduler = query_slurm_job(str(receipt["job_id"]))
        return {
            "report_schema": "mint-agent.pipeline-status.v1",
            "dataset_id": plan.dataset_id,
            "run_id": plan.run_id,
            "status": "LAUNCHING",
            "launch_job_id": receipt["job_id"],
            "launch_scheduler": scheduler,
            "state_path": plan.state_path,
        }
    return {
        "report_schema": "mint-agent.pipeline-status.v1",
        "dataset_id": plan.dataset_id,
        "run_id": plan.run_id,
        "status": "NOT_STARTED",
        "state_path": plan.state_path,
    }


def _invariants(value: object) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("pipeline invariants must be a nonempty list")
    invariants = tuple(dict.fromkeys(str(item).upper() for item in value))
    supported = {"PH", "PL", "CA", "FPRC", "EIC"}
    unknown = sorted(set(invariants) - supported)
    if unknown:
        raise ValueError(f"unsupported pipeline invariants: {unknown}")
    return invariants


def _llm_scientific_environment(value: object) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("pipeline llm_scientific must be a mapping")
    allowed = {
        "mode",
        "model",
        "api_key_env",
        "env_file",
        "cache_dir",
        "max_candidates",
        "timeout_seconds",
    }
    unknown = sorted(str(key) for key in value if key not in allowed)
    if unknown:
        raise ValueError(f"unknown pipeline llm_scientific keys: {unknown}")
    mode = str(value.get("mode", "disabled"))
    if mode not in {"disabled", "shadow", "advisory"}:
        raise ValueError(
            "pipeline llm_scientific.mode must be disabled, shadow, or advisory"
        )
    environment = {
        "LLM_SCIENTIFIC_MODE": mode,
    }
    api_key_env = str(value.get("api_key_env") or "OPENAI_API_KEY")
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", api_key_env):
        raise ValueError(
            "pipeline llm_scientific.api_key_env must be a shell identifier"
        )
    environment["LLM_API_KEY_ENV"] = api_key_env
    max_candidates = int(value.get("max_candidates", 10))
    if max_candidates < 1:
        raise ValueError("pipeline llm_scientific.max_candidates must be positive")
    environment["LLM_MAX_CANDIDATES"] = str(max_candidates)
    timeout_seconds = float(value.get("timeout_seconds", 180.0))
    if timeout_seconds <= 0:
        raise ValueError("pipeline llm_scientific.timeout_seconds must be positive")
    environment["LLM_TIMEOUT_SECONDS"] = str(timeout_seconds)
    for key, env_key in (
        ("model", "LLM_MODEL"),
        ("env_file", "LLM_ENV_FILE"),
        ("cache_dir", "LLM_CACHE_DIR"),
    ):
        raw = value.get(key)
        if raw is not None and str(raw):
            environment[env_key] = str(raw)
    return environment


def _repository_root(path: Path) -> Path:
    start = path if path.is_dir() else path.parent
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise ValueError(f"Could not locate repository root above {path}")


def _configured_repository_root(config_path: Path, config: Mapping[str, Any]) -> Path:
    configured = config.get("repository_root")
    if configured is None:
        return _repository_root(config_path)
    root = Path(str(configured)).expanduser()
    if not root.is_absolute():
        root = config_path.parent / root
    root = root.resolve()
    if not (root / "pyproject.toml").is_file():
        raise ValueError(
            f"pipeline repository_root does not contain pyproject.toml: {root}"
        )
    return root


def _repository_path(value: object, root: Path, name: str) -> Path:
    if value is None or not str(value):
        raise ValueError(f"pipeline {name} is required")
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _remote_repository_path(path: Path, root: Path, profile: ExecutionProfile) -> str:
    try:
        relative = path.relative_to(root)
    except ValueError:
        return str(path)
    return str(Path(profile.project_root) / relative)


def _remote_path(root: str, path: str) -> str:
    return path if path.startswith("/") else str(Path(root) / path)


def _safe_token(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)


def _validate_environment(environment: Mapping[str, str]) -> None:
    for key, value in environment.items():
        if any(character in value for character in (",", "\n", "\r")):
            if key == "INVARIANTS":
                continue
            raise ValueError(f"pipeline export value for {key} is not Slurm-safe")


if __name__ == "__main__":
    raise SystemExit(main())
