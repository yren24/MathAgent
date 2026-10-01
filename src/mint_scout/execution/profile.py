from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from mint_scout.config import load_yaml
from mint_scout.invariants.manifest import stable_hash


@dataclass(frozen=True)
class SlurmResources:
    partition: str
    ntasks: int
    cpus_per_task: int
    mem: str
    time: str


@dataclass(frozen=True)
class ExecutionProfile:
    profile_id: str
    profile_type: str
    login_host: str | None
    username: str | None
    project_root: str
    scratch_root: str
    cache_root: str
    run_root: str
    log_root: str
    python_executable: str
    module_load: tuple[str, ...]
    module_setup: str | None
    plbind_root: str | None
    slurm_defaults: SlurmResources | None
    slurm_setup_defaults: SlurmResources | None
    slurm_feature_defaults: SlurmResources | None
    slurm_qc_defaults: SlurmResources | None
    slurm_filtration_audit_defaults: SlurmResources | None
    slurm_scout_defaults: SlurmResources | None
    slurm_evaluation_defaults: SlurmResources | None
    dataset_preparation_script: str
    dataset_audit_script: str
    representation_design_script: str
    probe_selection_script: str
    probe_audit_script: str
    feature_batch_script: str
    feature_qc_script: str
    filtration_audit_script: str
    feature_outlier_diagnostic_script: str
    scout_oof_script: str
    scout_combine_script: str
    model_evaluation_script: str
    validation_evaluation_script: str
    frozen_test_evaluation_script: str
    acceptance_evaluation_script: str
    lifecycle_script: str

    @property
    def supports_slurm(self) -> bool:
        return self.profile_type in {"slurm", "slurm_hpc"}


@dataclass(frozen=True)
class FeatureJobRequest:
    dataset_id: str
    invariant: str
    evidence_scope: str
    representation_hash: str
    run_id: str
    task_config: str
    representation_spec: str | None = None
    sample_id_file: str | None = None
    selection_hash: str | None = None


@dataclass(frozen=True)
class DatasetAuditJobRequest:
    dataset_id: str
    run_id: str
    task_config: str


@dataclass(frozen=True)
class DatasetPreparationJobRequest:
    dataset_id: str
    run_id: str
    task_config: str
    module_load: tuple[str, ...] = ()
    replace_module_environment: bool = False
    python_executable: str | None = None


@dataclass(frozen=True)
class RepresentationDesignJobRequest:
    dataset_id: str
    run_id: str
    task_config: str
    data_audit_report: str
    design_input_hash: str


@dataclass(frozen=True)
class ProbeSelectionJobRequest:
    dataset_id: str
    representation_hash: str
    run_id: str
    task_config: str
    scout_config: str
    representation_spec: str
    frozen_sample_source: str | None = None


@dataclass(frozen=True)
class ProbeAuditJobRequest:
    dataset_id: str
    representation_hash: str
    run_id: str
    task_config: str
    scout_config: str
    probe_selection: str


@dataclass(frozen=True)
class FeatureQCJobRequest:
    dataset_id: str
    invariants: tuple[str, ...]
    evidence_scope: str
    representation_hash: str
    run_id: str
    representation_spec: str
    feature_manifests: Mapping[str, str]
    qc_config: str
    sample_id_file: str | None = None


@dataclass(frozen=True)
class FiltrationAuditJobRequest:
    dataset_id: str
    invariants: tuple[str, ...]
    evidence_scope: str
    representation_hash: str
    run_id: str
    representation_spec: str
    feature_manifests: Mapping[str, str]
    audit_config: str
    sample_id_file: str | None = None


@dataclass(frozen=True)
class FeatureOutlierDiagnosticJobRequest:
    dataset_id: str
    invariant: str
    evidence_scope: str
    representation_hash: str
    run_id: str
    representation_spec: str
    feature_manifest: str
    feature_qc_report: str
    sample_id_file: str
    top_k: int = 10


@dataclass(frozen=True)
class ScoutOOFJobRequest:
    dataset_id: str
    invariant: str
    representation_hash: str
    run_id: str
    task_config: str
    scout_config: str
    representation_spec: str
    probe_selection: str
    feature_manifest: str
    feature_qc_report: str
    filtration_audit_report: str
    gbt_config: str = "configs/gbt/plbind_adaptive_gbt.yaml"
    user_target: float | None = None


@dataclass(frozen=True)
class ScoutCombineJobRequest:
    dataset_id: str
    invariants: tuple[str, ...]
    representation_hash: str
    run_id: str
    scout_config: str
    representation_spec: str
    probe_selection: str
    scout_reports: Mapping[str, str]
    feature_qc_report: str
    user_target: float | None = None
    llm_prior_order: tuple[tuple[str, ...], ...] = ()
    ranking_policy: str = "hierarchical_empirical_v1"


@dataclass(frozen=True)
class ModelEvaluationJobRequest:
    dataset_id: str
    invariants: tuple[str, ...]
    evidence_scope: str
    representation_hash: str
    run_id: str
    task_config: str
    gbt_config: str
    representation_spec: str
    scout_artifact: str
    feature_qc_report: str
    feature_manifests: Mapping[str, str]
    max_acquisitions: int


@dataclass(frozen=True)
class ValidationEvaluationJobRequest:
    dataset_id: str
    invariants: tuple[str, ...]
    representation_hash: str
    run_id: str
    task_config: str
    gbt_config: str
    representation_spec: str
    scout_artifact: str
    train_feature_qc_report: str
    validation_feature_qc_report: str
    train_feature_manifests: Mapping[str, str]
    validation_feature_manifests: Mapping[str, str]
    max_acquisitions: int
    prior_evaluation_report: str | None = None


@dataclass(frozen=True)
class FrozenTestEvaluationJobRequest:
    dataset_id: str
    invariants: tuple[str, ...]
    representation_hash: str
    run_id: str
    task_config: str
    gbt_config: str
    representation_spec: str
    selection_report: str
    train_feature_qc_report: str
    test_feature_qc_report: str
    train_feature_manifests: Mapping[str, str]
    test_feature_manifests: Mapping[str, str]
    validation_feature_qc_report: str | None = None
    validation_feature_manifests: Mapping[str, str] | None = None
    n_bootstrap: int = 1000
    bootstrap_seed: int = 2026


@dataclass(frozen=True)
class AcceptanceEvaluationJobRequest:
    dataset_id: str
    invariants: tuple[str, ...]
    representation_hash: str
    run_id: str
    task_config: str
    gbt_config: str
    representation_spec: str
    scout_artifact: str
    train_feature_qc_report: str
    evaluation_feature_qc_report: str
    train_feature_manifests: Mapping[str, str]
    evaluation_feature_manifests: Mapping[str, str]
    candidate_rank: int | None = None
    max_acquisitions: int | None = None
    prior_evaluation_report: str | None = None


@dataclass(frozen=True)
class SlurmJobPlan:
    plan_id: str
    profile_id: str
    scheduler: str
    job_kind: str
    dataset_id: str
    invariant: str
    invariants: tuple[str, ...]
    evidence_scope: str
    representation_hash: str
    job_name: str
    working_directory: str
    script_path: str
    resources: SlurmResources
    environment: Mapping[str, str]
    input_manifests: Mapping[str, str]
    manifest_path: str
    stdout_path: str
    stderr_path: str
    submit_command: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "profile_id": self.profile_id,
            "scheduler": self.scheduler,
            "job_kind": self.job_kind,
            "dataset_id": self.dataset_id,
            "invariant": self.invariant,
            "invariants": list(self.invariants),
            "evidence_scope": self.evidence_scope,
            "representation_hash": self.representation_hash,
            "job_name": self.job_name,
            "working_directory": self.working_directory,
            "script_path": self.script_path,
            "resources": {
                "partition": self.resources.partition,
                "ntasks": self.resources.ntasks,
                "cpus_per_task": self.resources.cpus_per_task,
                "mem": self.resources.mem,
                "time": self.resources.time,
            },
            "environment": dict(self.environment),
            "input_manifests": dict(self.input_manifests),
            "manifest_path": self.manifest_path,
            "stdout_path": self.stdout_path,
            "stderr_path": self.stderr_path,
            "submit_command": list(self.submit_command),
        }


def feature_job_plan_id(plan: Mapping[str, Any]) -> str:
    return stable_hash(
        {
            "profile_id": plan.get("profile_id"),
            "job_kind": plan.get("job_kind"),
            "dataset_id": plan.get("dataset_id"),
            "invariant": str(plan.get("invariant") or "").upper(),
            "invariants": sorted(
                str(value).upper() for value in plan.get("invariants", ())
            ),
            "evidence_scope": plan.get("evidence_scope"),
            "representation_hash": plan.get("representation_hash"),
            "input_manifests": plan.get("input_manifests", {}),
            "manifest_path": plan.get("manifest_path"),
            "submit_command": plan.get("submit_command"),
        }
    )


def load_execution_profile(path: str | Path) -> ExecutionProfile:
    data = load_yaml(path)
    return execution_profile_from_mapping(data)


def execution_profile_from_mapping(data: Mapping[str, Any]) -> ExecutionProfile:
    paths = _mapping(data, "paths")
    host = data.get("host", {})
    if not isinstance(host, Mapping):
        raise ValueError("execution profile host must be a mapping")
    python = data.get("python", {})
    if not isinstance(python, Mapping):
        raise ValueError("execution profile python must be a mapping")
    scripts = data.get("scripts", {})
    if scripts is None:
        scripts = {}
    if not isinstance(scripts, Mapping):
        raise ValueError("execution profile scripts must be a mapping")
    tools = data.get("tools", {})
    if tools is None:
        tools = {}
    if not isinstance(tools, Mapping):
        raise ValueError("execution profile tools must be a mapping")

    profile_id = str(data.get("profile_id") or "")
    profile_type = str(data.get("profile_type") or "")
    if not profile_id:
        raise ValueError("execution profile requires profile_id")
    if not profile_type:
        raise ValueError("execution profile requires profile_type")

    return ExecutionProfile(
        profile_id=profile_id,
        profile_type=profile_type,
        login_host=_optional_str(host.get("login")),
        username=_optional_str(host.get("user")),
        project_root=_required_str(paths, "project_root"),
        scratch_root=_required_str(paths, "scratch_root"),
        cache_root=_required_str(paths, "cache_root"),
        run_root=_required_str(paths, "run_root"),
        log_root=_required_str(paths, "log_root"),
        python_executable=str(python.get("executable") or "python"),
        module_load=tuple(str(value) for value in python.get("module_load", ())),
        module_setup=_optional_str(python.get("module_setup")),
        plbind_root=_optional_str(tools.get("plbind_root")),
        slurm_defaults=_resources(data.get("slurm_defaults")),
        slurm_setup_defaults=_resources(data.get("slurm_setup_defaults")),
        slurm_feature_defaults=_resources(data.get("slurm_feature_defaults")),
        slurm_qc_defaults=_resources(data.get("slurm_qc_defaults")),
        slurm_filtration_audit_defaults=_resources(
            data.get("slurm_filtration_audit_defaults")
        ),
        slurm_scout_defaults=_resources(data.get("slurm_scout_defaults")),
        slurm_evaluation_defaults=_resources(data.get("slurm_evaluation_defaults")),
        dataset_preparation_script=str(
            scripts.get("dataset_preparation")
            or "scripts/slurm/run_dataset_preparation.sbatch"
        ),
        dataset_audit_script=str(
            scripts.get("dataset_audit") or "scripts/slurm/run_dataset_audit.sbatch"
        ),
        representation_design_script=str(
            scripts.get("representation_design")
            or "scripts/slurm/run_representation_design.sbatch"
        ),
        probe_selection_script=str(
            scripts.get("probe_selection")
            or "scripts/slurm/run_probe_selection.sbatch"
        ),
        probe_audit_script=str(
            scripts.get("probe_audit") or "scripts/slurm/run_probe_audit.sbatch"
        ),
        feature_batch_script=str(scripts.get("feature_batch") or "scripts/slurm/run_feature_batch.sbatch"),
        feature_qc_script=str(scripts.get("feature_qc") or "scripts/slurm/run_feature_qc.sbatch"),
        filtration_audit_script=str(
            scripts.get("filtration_audit")
            or "scripts/slurm/run_filtration_audit.sbatch"
        ),
        feature_outlier_diagnostic_script=str(
            scripts.get("feature_outlier_diagnostic")
            or "scripts/slurm/run_feature_outlier_diagnostic.sbatch"
        ),
        scout_oof_script=str(
            scripts.get("scout_oof") or "scripts/slurm/run_scout_oof.sbatch"
        ),
        scout_combine_script=str(
            scripts.get("scout_combine")
            or "scripts/slurm/run_scout_combine.sbatch"
        ),
        model_evaluation_script=str(
            scripts.get("model_evaluation")
            or "scripts/slurm/run_model_evaluation.sbatch"
        ),
        validation_evaluation_script=str(
            scripts.get("validation_evaluation")
            or "scripts/slurm/run_validation_gbt_evaluation.sbatch"
        ),
        frozen_test_evaluation_script=str(
            scripts.get("frozen_test_evaluation")
            or "scripts/slurm/run_frozen_test_gbt_evaluation.sbatch"
        ),
        acceptance_evaluation_script=str(
            scripts.get("acceptance_evaluation")
            or "scripts/slurm/run_acceptance_gbt_evaluation.sbatch"
        ),
        lifecycle_script=str(
            scripts.get("lifecycle")
            or "scripts/slurm/run_agent_lifecycle.sbatch"
        ),
    )


def build_dataset_preparation_job_plan(
    profile: ExecutionProfile, request: DatasetPreparationJobRequest
) -> SlurmJobPlan:
    run_token = _safe_token(request.run_id)
    output = (
        Path(profile.run_root)
        / f"dataset_preparation_{request.dataset_id}_{run_token}.json"
    )
    environment = _setup_environment(
        profile,
        {
            "TASK_CONFIG": _remote_path(profile.project_root, request.task_config),
            "OUTPUT_REPORT": str(output),
        },
    )
    if request.module_load:
        environment["DATASET_PREPARATION_MODULES"] = ":".join(request.module_load)
        environment["DATASET_PREPARATION_MODULE_PURGE"] = (
            "1" if request.replace_module_environment else "0"
        )
        _validate_export_values(environment)
    if request.python_executable:
        environment["PYTHON_BIN"] = str(request.python_executable)
        _validate_export_values(environment)
    return _build_setup_job_plan(
        profile,
        job_kind="dataset_preparation",
        dataset_id=request.dataset_id,
        invariant="DATASET",
        evidence_scope="design",
        representation_hash="not_applicable",
        run_id=request.run_id,
        script_path=profile.dataset_preparation_script,
        output_path=output,
        environment=environment,
        input_manifests={},
    )


def build_dataset_audit_job_plan(
    profile: ExecutionProfile, request: DatasetAuditJobRequest
) -> SlurmJobPlan:
    run_token = _safe_token(request.run_id)
    output = Path(profile.run_root) / f"dataset_audit_{request.dataset_id}_{run_token}.json"
    environment = _setup_environment(
        profile,
        {
            "TASK_CONFIG": _remote_path(profile.project_root, request.task_config),
            "OUTPUT_JSON": str(output),
            "OUTPUT_MD": str(output.with_suffix(".md")),
        },
    )
    return _build_setup_job_plan(
        profile,
        job_kind="dataset_audit",
        dataset_id=request.dataset_id,
        invariant="DATASET",
        evidence_scope="design",
        representation_hash="not_applicable",
        run_id=request.run_id,
        script_path=profile.dataset_audit_script,
        output_path=output,
        environment=environment,
        input_manifests={},
    )


def build_representation_design_job_plan(
    profile: ExecutionProfile, request: RepresentationDesignJobRequest
) -> SlurmJobPlan:
    run_token = _safe_token(request.run_id)
    output = Path(profile.run_root) / f"representation_{request.dataset_id}_{run_token}.json"
    audit_path = _remote_path(profile.project_root, request.data_audit_report)
    environment = _setup_environment(
        profile,
        {
            "TASK_CONFIG": _remote_path(profile.project_root, request.task_config),
            "DATA_AUDIT_REPORT": audit_path,
            "DESIGN_INPUT_HASH": request.design_input_hash,
            "SPLIT": "train",
            "OFFSET": "0",
            "LIMIT": "all",
            "REPORT": str(output),
        },
    )
    return _build_setup_job_plan(
        profile,
        job_kind="representation_design",
        dataset_id=request.dataset_id,
        invariant="REPRESENTATION",
        evidence_scope="design",
        representation_hash="pending",
        run_id=request.run_id,
        script_path=profile.representation_design_script,
        output_path=output,
        environment=environment,
        input_manifests={"DATA_AUDIT": audit_path},
    )


def build_probe_selection_job_plan(
    profile: ExecutionProfile, request: ProbeSelectionJobRequest
) -> SlurmJobPlan:
    run_token = _safe_token(request.run_id)
    repr_token = _safe_token(request.representation_hash)[:12]
    output = (
        Path(profile.run_root)
        / f"probe_selection_{request.dataset_id}_{repr_token}_{run_token}.json"
    )
    representation_path = _remote_path(profile.project_root, request.representation_spec)
    frozen_source = (
        _remote_path(profile.project_root, request.frozen_sample_source)
        if request.frozen_sample_source
        else None
    )
    environment = _setup_environment(
        profile,
        {
            "TASK_CONFIG": _remote_path(profile.project_root, request.task_config),
            "SCOUT_CONFIG": _remote_path(profile.project_root, request.scout_config),
            "REPRESENTATION_SPEC": representation_path,
            "OUTPUT": str(output),
            **({"FROZEN_SAMPLE_SOURCE": frozen_source} if frozen_source else {}),
        },
    )
    inputs = {"REPRESENTATION": representation_path}
    if frozen_source:
        inputs["FROZEN_SAMPLE_SOURCE"] = frozen_source
    return _build_setup_job_plan(
        profile,
        job_kind="probe_selection",
        dataset_id=request.dataset_id,
        invariant="PROBE",
        evidence_scope="probe",
        representation_hash=request.representation_hash,
        run_id=request.run_id,
        script_path=profile.probe_selection_script,
        output_path=output,
        environment=environment,
        input_manifests=inputs,
    )


def build_probe_audit_job_plan(
    profile: ExecutionProfile, request: ProbeAuditJobRequest
) -> SlurmJobPlan:
    run_token = _safe_token(request.run_id)
    repr_token = _safe_token(request.representation_hash)[:12]
    output = (
        Path(profile.run_root)
        / f"probe_audit_{request.dataset_id}_{repr_token}_{run_token}.json"
    )
    probe_path = _remote_path(profile.project_root, request.probe_selection)
    environment = _setup_environment(
        profile,
        {
            "TASK_CONFIG": _remote_path(profile.project_root, request.task_config),
            "SCOUT_CONFIG": _remote_path(profile.project_root, request.scout_config),
            "PROBE_SELECTION": probe_path,
            "OUTPUT": str(output),
        },
    )
    return _build_setup_job_plan(
        profile,
        job_kind="probe_audit",
        dataset_id=request.dataset_id,
        invariant="PROBE",
        evidence_scope="probe",
        representation_hash=request.representation_hash,
        run_id=request.run_id,
        script_path=profile.probe_audit_script,
        output_path=output,
        environment=environment,
        input_manifests={"PROBE_SELECTION": probe_path},
    )


def build_feature_job_plan(profile: ExecutionProfile, request: FeatureJobRequest) -> SlurmJobPlan:
    if not profile.supports_slurm:
        raise ValueError(f"Profile {profile.profile_id!r} does not support Slurm feature jobs")
    resources = profile.slurm_feature_defaults or profile.slurm_defaults
    if resources is None:
        raise ValueError(f"Profile {profile.profile_id!r} does not define Slurm resources")

    invariant = request.invariant.upper()
    scope = request.evidence_scope
    split = _scope_to_split(scope)
    run_token = _safe_token(request.run_id)
    repr_token = _safe_token(request.representation_hash)[:12]
    job_name = f"mint-feature-{invariant.lower()}-{scope}"
    manifest = (
        Path(profile.run_root)
        / f"feature_{request.dataset_id}_{invariant}_{scope}_{repr_token}_{run_token}.jsonl"
    )
    script = _remote_path(profile.project_root, profile.feature_batch_script)
    stdout = Path(profile.log_root) / f"{job_name}-%j.out"
    stderr = Path(profile.log_root) / f"{job_name}-%j.err"

    environment = {
        "MINT_AGENT_ROOT": profile.project_root,
        "TASK_CONFIG": _remote_path(profile.project_root, request.task_config),
        "RUN_ROOT": profile.run_root,
        "CACHE_ROOT": profile.cache_root,
        "LOG_ROOT": profile.log_root,
        "PYTHON_BIN": profile.python_executable,
        "INVARIANT": invariant,
        "SPLIT": split,
        "LIMIT": "all",
        "MANIFEST": str(manifest),
    }
    if profile.module_load:
        environment["MINT_MODULES"] = ":".join(profile.module_load)
    if profile.module_setup:
        environment["MINT_MODULE_SETUP"] = _remote_path(
            profile.project_root, profile.module_setup
        )
    if request.representation_spec:
        environment["REPRESENTATION_SPEC"] = _remote_path(
            profile.project_root, request.representation_spec
        )
    if request.sample_id_file:
        environment["SAMPLE_ID_FILE"] = _remote_path(
            profile.project_root, request.sample_id_file
        )
    if request.selection_hash:
        environment["SELECTION_HASH"] = request.selection_hash
    if profile.plbind_root:
        environment["LEGACY_ROOT"] = profile.plbind_root
    _validate_export_values(environment)

    submit_command = (
        "env",
        *(f"{key}={value}" for key, value in environment.items()),
        "sbatch",
        f"--job-name={job_name}",
        f"--partition={resources.partition}",
        f"--ntasks={resources.ntasks}",
        f"--cpus-per-task={resources.cpus_per_task}",
        f"--mem={resources.mem}",
        f"--time={resources.time}",
        f"--output={stdout}",
        f"--error={stderr}",
        script,
    )
    plan_id = feature_job_plan_id(
        {
            "profile_id": profile.profile_id,
            "job_kind": "feature_batch",
            "dataset_id": request.dataset_id,
            "invariant": invariant,
            "invariants": (invariant,),
            "evidence_scope": scope,
            "representation_hash": request.representation_hash,
            "input_manifests": {},
            "manifest_path": str(manifest),
            "submit_command": submit_command,
        }
    )
    return SlurmJobPlan(
        plan_id=plan_id,
        profile_id=profile.profile_id,
        scheduler="slurm",
        job_kind="feature_batch",
        dataset_id=request.dataset_id,
        invariant=invariant,
        invariants=(invariant,),
        evidence_scope=scope,
        representation_hash=request.representation_hash,
        job_name=job_name,
        working_directory=profile.project_root,
        script_path=script,
        resources=resources,
        environment=environment,
        input_manifests={},
        manifest_path=str(manifest),
        stdout_path=str(stdout),
        stderr_path=str(stderr),
        submit_command=submit_command,
    )


def build_feature_qc_job_plan(
    profile: ExecutionProfile, request: FeatureQCJobRequest
) -> SlurmJobPlan:
    if not profile.supports_slurm:
        raise ValueError(f"Profile {profile.profile_id!r} does not support Slurm QC jobs")
    resources = (
        profile.slurm_qc_defaults
        or profile.slurm_defaults
        or profile.slurm_feature_defaults
    )
    if resources is None:
        raise ValueError(f"Profile {profile.profile_id!r} does not define Slurm resources")

    invariants = tuple(sorted({str(value).upper() for value in request.invariants}))
    if not invariants:
        raise ValueError("Feature QC job requires at least one invariant")
    manifests = {str(key).upper(): str(value) for key, value in request.feature_manifests.items()}
    if set(manifests) != set(invariants):
        raise ValueError("Feature QC manifests must match the requested invariants")
    if any(not name.replace("_", "").isalnum() for name in invariants):
        raise ValueError("Feature QC invariant names must be environment-variable safe")

    scope = request.evidence_scope
    run_token = _safe_token(request.run_id)
    repr_token = _safe_token(request.representation_hash)[:12]
    invariant_token = "-".join(name.lower() for name in invariants)
    job_name = f"mint-qc-{invariant_token}-{scope}"
    output = (
        Path(profile.run_root)
        / (
            f"feature_qc_{request.dataset_id}_{invariant_token}_{scope}_"
            f"{repr_token}_{run_token}.json"
        )
    )
    script = _remote_path(profile.project_root, profile.feature_qc_script)
    stdout = Path(profile.log_root) / f"{job_name}-%j.out"
    stderr = Path(profile.log_root) / f"{job_name}-%j.err"
    remote_manifests = {
        name: _remote_path(profile.project_root, path) for name, path in manifests.items()
    }
    sample_id_file = request.sample_id_file or remote_manifests[invariants[0]]
    environment = {
        "MINT_AGENT_ROOT": profile.project_root,
        "RUN_ROOT": profile.run_root,
        "LOG_ROOT": profile.log_root,
        "PYTHON_BIN": profile.python_executable,
        "DATASET_ID": request.dataset_id,
        "EVIDENCE_SCOPE": scope,
        "REPRESENTATION_SPEC": _remote_path(profile.project_root, request.representation_spec),
        "QC_CONFIG": _remote_path(profile.project_root, request.qc_config),
        "FEATURE_INVARIANTS": ":".join(invariants),
        "OUTPUT": str(output),
    }
    sample_source = _remote_path(profile.project_root, sample_id_file)
    if scope == "probe":
        environment["PROBE_SELECTION"] = sample_source
    else:
        environment["SAMPLE_ID_FILE"] = sample_source
    for name, path in remote_manifests.items():
        environment[f"FEATURE_MANIFEST_{name}"] = path
    if profile.module_load:
        environment["MINT_MODULES"] = ":".join(profile.module_load)
    if profile.module_setup:
        environment["MINT_MODULE_SETUP"] = _remote_path(
            profile.project_root, profile.module_setup
        )
    _validate_export_values(environment)

    submit_command = (
        "env",
        *(f"{key}={value}" for key, value in environment.items()),
        "sbatch",
        f"--job-name={job_name}",
        f"--partition={resources.partition}",
        f"--ntasks={resources.ntasks}",
        f"--cpus-per-task={resources.cpus_per_task}",
        f"--mem={resources.mem}",
        f"--time={resources.time}",
        f"--output={stdout}",
        f"--error={stderr}",
        script,
    )
    invariant_label = "+".join(invariants)
    plan_id = feature_job_plan_id(
        {
            "profile_id": profile.profile_id,
            "job_kind": "feature_qc",
            "dataset_id": request.dataset_id,
            "invariant": invariant_label,
            "invariants": invariants,
            "evidence_scope": scope,
            "representation_hash": request.representation_hash,
            "input_manifests": remote_manifests,
            "manifest_path": str(output),
            "submit_command": submit_command,
        }
    )
    return SlurmJobPlan(
        plan_id=plan_id,
        profile_id=profile.profile_id,
        scheduler="slurm",
        job_kind="feature_qc",
        dataset_id=request.dataset_id,
        invariant=invariant_label,
        invariants=invariants,
        evidence_scope=scope,
        representation_hash=request.representation_hash,
        job_name=job_name,
        working_directory=profile.project_root,
        script_path=script,
        resources=resources,
        environment=environment,
        input_manifests=remote_manifests,
        manifest_path=str(output),
        stdout_path=str(stdout),
        stderr_path=str(stderr),
        submit_command=submit_command,
    )


def build_filtration_audit_job_plan(
    profile: ExecutionProfile, request: FiltrationAuditJobRequest
) -> SlurmJobPlan:
    if not profile.supports_slurm:
        raise ValueError(
            f"Profile {profile.profile_id!r} does not support Slurm filtration-audit jobs"
        )
    resources = (
        profile.slurm_filtration_audit_defaults
        or profile.slurm_defaults
        or profile.slurm_qc_defaults
        or profile.slurm_feature_defaults
    )
    if resources is None:
        raise ValueError(f"Profile {profile.profile_id!r} does not define Slurm resources")

    invariants = tuple(sorted({str(value).upper() for value in request.invariants}))
    if not invariants:
        raise ValueError("Filtration audit job requires at least one invariant")
    manifests = {
        str(key).upper(): str(value) for key, value in request.feature_manifests.items()
    }
    if set(manifests) != set(invariants):
        raise ValueError("Filtration audit manifests must match the requested invariants")
    if any(not name.replace("_", "").isalnum() for name in invariants):
        raise ValueError("Filtration audit invariant names must be environment-variable safe")

    scope = request.evidence_scope
    run_token = _safe_token(request.run_id)
    repr_token = _safe_token(request.representation_hash)[:12]
    invariant_token = "-".join(name.lower() for name in invariants)
    job_name = f"mint-filtration-{invariant_token}-{scope}"
    output = (
        Path(profile.run_root)
        / (
            f"filtration_audit_{request.dataset_id}_{invariant_token}_{scope}_"
            f"{repr_token}_{run_token}.json"
        )
    )
    script = _remote_path(profile.project_root, profile.filtration_audit_script)
    stdout = Path(profile.log_root) / f"{job_name}-%j.out"
    stderr = Path(profile.log_root) / f"{job_name}-%j.err"
    remote_manifests = {
        name: _remote_path(profile.project_root, path) for name, path in manifests.items()
    }
    sample_id_file = request.sample_id_file or remote_manifests[invariants[0]]
    environment = {
        "MINT_AGENT_ROOT": profile.project_root,
        "RUN_ROOT": profile.run_root,
        "LOG_ROOT": profile.log_root,
        "PYTHON_BIN": profile.python_executable,
        "DATASET_ID": request.dataset_id,
        "EVIDENCE_SCOPE": scope,
        "REPRESENTATION_SPEC": _remote_path(
            profile.project_root, request.representation_spec
        ),
        "AUDIT_CONFIG": _remote_path(profile.project_root, request.audit_config),
        "FEATURE_INVARIANTS": ":".join(invariants),
        "OUTPUT": str(output),
    }
    sample_source = _remote_path(profile.project_root, sample_id_file)
    if scope == "probe":
        environment["PROBE_SELECTION"] = sample_source
    else:
        environment["SAMPLE_ID_FILE"] = sample_source
    for name, path in remote_manifests.items():
        environment[f"FEATURE_MANIFEST_{name}"] = path
    if profile.module_load:
        environment["MINT_MODULES"] = ":".join(profile.module_load)
    if profile.module_setup:
        environment["MINT_MODULE_SETUP"] = _remote_path(
            profile.project_root, profile.module_setup
        )
    _validate_export_values(environment)

    submit_command = (
        "env",
        *(f"{key}={value}" for key, value in environment.items()),
        "sbatch",
        f"--job-name={job_name}",
        f"--partition={resources.partition}",
        f"--ntasks={resources.ntasks}",
        f"--cpus-per-task={resources.cpus_per_task}",
        f"--mem={resources.mem}",
        f"--time={resources.time}",
        f"--output={stdout}",
        f"--error={stderr}",
        script,
    )
    invariant_label = "+".join(invariants)
    plan_id = feature_job_plan_id(
        {
            "profile_id": profile.profile_id,
            "job_kind": "filtration_audit",
            "dataset_id": request.dataset_id,
            "invariant": invariant_label,
            "invariants": invariants,
            "evidence_scope": scope,
            "representation_hash": request.representation_hash,
            "input_manifests": remote_manifests,
            "manifest_path": str(output),
            "submit_command": submit_command,
        }
    )
    return SlurmJobPlan(
        plan_id=plan_id,
        profile_id=profile.profile_id,
        scheduler="slurm",
        job_kind="filtration_audit",
        dataset_id=request.dataset_id,
        invariant=invariant_label,
        invariants=invariants,
        evidence_scope=scope,
        representation_hash=request.representation_hash,
        job_name=job_name,
        working_directory=profile.project_root,
        script_path=script,
        resources=resources,
        environment=environment,
        input_manifests=remote_manifests,
        manifest_path=str(output),
        stdout_path=str(stdout),
        stderr_path=str(stderr),
        submit_command=submit_command,
    )


def build_feature_outlier_diagnostic_job_plan(
    profile: ExecutionProfile, request: FeatureOutlierDiagnosticJobRequest
) -> SlurmJobPlan:
    if not profile.supports_slurm:
        raise ValueError(
            f"Profile {profile.profile_id!r} does not support Slurm diagnostic jobs"
        )
    resources = (
        profile.slurm_qc_defaults
        or profile.slurm_defaults
        or profile.slurm_feature_defaults
    )
    if resources is None:
        raise ValueError(f"Profile {profile.profile_id!r} does not define Slurm resources")
    if request.top_k < 1:
        raise ValueError("Feature outlier diagnostic top_k must be positive")

    invariant = request.invariant.upper()
    scope = request.evidence_scope
    run_token = _safe_token(request.run_id)
    repr_token = _safe_token(request.representation_hash)[:12]
    job_name = f"mint-diagnostic-{invariant.lower()}-{scope}"
    output = (
        Path(profile.run_root)
        / f"feature_outlier_diagnostic_{request.dataset_id}_{invariant}_{scope}_{repr_token}_{run_token}.json"
    )
    script = _remote_path(profile.project_root, profile.feature_outlier_diagnostic_script)
    stdout = Path(profile.log_root) / f"{job_name}-%j.out"
    stderr = Path(profile.log_root) / f"{job_name}-%j.err"
    feature_manifest = _remote_path(profile.project_root, request.feature_manifest)
    sample_source = _remote_path(profile.project_root, request.sample_id_file)
    environment = {
        "MINT_AGENT_ROOT": profile.project_root,
        "RUN_ROOT": profile.run_root,
        "LOG_ROOT": profile.log_root,
        "PYTHON_BIN": profile.python_executable,
        "DATASET_ID": request.dataset_id,
        "EVIDENCE_SCOPE": scope,
        "REPRESENTATION_SPEC": _remote_path(
            profile.project_root, request.representation_spec
        ),
        "FEATURE_INVARIANTS": invariant,
        f"FEATURE_MANIFEST_{invariant}": feature_manifest,
        "FEATURE_QC_REPORT": _remote_path(
            profile.project_root, request.feature_qc_report
        ),
        "TOP_K": str(request.top_k),
        "OUTPUT": str(output),
    }
    if scope == "probe":
        environment["PROBE_SELECTION"] = sample_source
    else:
        environment["SAMPLE_ID_FILE"] = sample_source
    if profile.module_load:
        environment["MINT_MODULES"] = ":".join(profile.module_load)
    if profile.module_setup:
        environment["MINT_MODULE_SETUP"] = _remote_path(
            profile.project_root, profile.module_setup
        )
    _validate_export_values(environment)

    submit_command = (
        "env",
        *(f"{key}={value}" for key, value in environment.items()),
        "sbatch",
        f"--job-name={job_name}",
        f"--partition={resources.partition}",
        f"--ntasks={resources.ntasks}",
        f"--cpus-per-task={resources.cpus_per_task}",
        f"--mem={resources.mem}",
        f"--time={resources.time}",
        f"--output={stdout}",
        f"--error={stderr}",
        script,
    )
    plan_id = feature_job_plan_id(
        {
            "profile_id": profile.profile_id,
            "job_kind": "feature_outlier_diagnostic",
            "dataset_id": request.dataset_id,
            "invariant": invariant,
            "invariants": (invariant,),
            "evidence_scope": scope,
            "representation_hash": request.representation_hash,
            "input_manifests": {invariant: feature_manifest},
            "manifest_path": str(output),
            "submit_command": submit_command,
        }
    )
    return SlurmJobPlan(
        plan_id=plan_id,
        profile_id=profile.profile_id,
        scheduler="slurm",
        job_kind="feature_outlier_diagnostic",
        dataset_id=request.dataset_id,
        invariant=invariant,
        invariants=(invariant,),
        evidence_scope=scope,
        representation_hash=request.representation_hash,
        job_name=job_name,
        working_directory=profile.project_root,
        script_path=script,
        resources=resources,
        environment=environment,
        input_manifests={invariant: feature_manifest},
        manifest_path=str(output),
        stdout_path=str(stdout),
        stderr_path=str(stderr),
        submit_command=submit_command,
    )


def build_scout_oof_job_plan(
    profile: ExecutionProfile, request: ScoutOOFJobRequest
) -> SlurmJobPlan:
    if not profile.supports_slurm:
        raise ValueError(f"Profile {profile.profile_id!r} does not support Slurm Scout jobs")
    resources = profile.slurm_scout_defaults or profile.slurm_defaults
    if resources is None:
        raise ValueError(f"Profile {profile.profile_id!r} does not define Scout resources")
    invariant = request.invariant.upper()
    dataset_token = _compact_path_token(request.dataset_id, max_length=48)
    run_token = _compact_path_token(request.run_id, max_length=48)
    repr_token = _safe_token(request.representation_hash)[:12]
    job_name = f"mint-scout-oof-{invariant.lower()}"
    report = (
        Path(profile.run_root)
        / f"scout_oof_{dataset_token}_{invariant}_{repr_token}_{run_token}.json"
    )
    execution = (
        Path(profile.run_root)
        / f"scout_oof_execution_{dataset_token}_{invariant}_{repr_token}_{run_token}.json"
    )
    script = _remote_path(profile.project_root, profile.scout_oof_script)
    stdout = Path(profile.log_root) / f"{job_name}-%j.out"
    stderr = Path(profile.log_root) / f"{job_name}-%j.err"
    feature_manifest = _remote_path(profile.project_root, request.feature_manifest)
    environment = {
        "MINT_AGENT_ROOT": profile.project_root,
        "RUN_ROOT": profile.run_root,
        "LOG_ROOT": profile.log_root,
        "PYTHON_BIN": profile.python_executable,
        "DATASET_ID": request.dataset_id,
        "TASK_CONFIG": _remote_path(profile.project_root, request.task_config),
        "SCOUT_CONFIG": _remote_path(profile.project_root, request.scout_config),
        "GBT_CONFIG": _remote_path(profile.project_root, request.gbt_config),
        "REPRESENTATION_SPEC": _remote_path(
            profile.project_root, request.representation_spec
        ),
        "PROBE_SELECTION": _remote_path(profile.project_root, request.probe_selection),
        "FEATURE_QC_REPORT": _remote_path(
            profile.project_root, request.feature_qc_report
        ),
        "FILTRATION_AUDIT_REPORT": _remote_path(
            profile.project_root, request.filtration_audit_report
        ),
        "FEATURE_INVARIANTS": invariant,
        f"FEATURE_MANIFEST_{invariant}": feature_manifest,
        "OUTPUT": str(report),
        "EXECUTION_ARTIFACT": str(execution),
    }
    if request.user_target is not None:
        environment["USER_TARGET"] = str(float(request.user_target))
    if profile.module_load:
        environment["MINT_MODULES"] = ":".join(profile.module_load)
    if profile.module_setup:
        environment["MINT_MODULE_SETUP"] = _remote_path(
            profile.project_root, profile.module_setup
        )
    _validate_export_values(environment)
    submit_command = (
        "env",
        *(f"{key}={value}" for key, value in environment.items()),
        "sbatch",
        f"--job-name={job_name}",
        f"--partition={resources.partition}",
        f"--ntasks={resources.ntasks}",
        f"--cpus-per-task={resources.cpus_per_task}",
        f"--mem={resources.mem}",
        f"--time={resources.time}",
        f"--output={stdout}",
        f"--error={stderr}",
        script,
    )
    input_manifests = {invariant: feature_manifest}
    plan_id = feature_job_plan_id(
        {
            "profile_id": profile.profile_id,
            "job_kind": "scout_oof",
            "dataset_id": request.dataset_id,
            "invariant": invariant,
            "invariants": (invariant,),
            "evidence_scope": "probe",
            "representation_hash": request.representation_hash,
            "input_manifests": input_manifests,
            "manifest_path": str(report),
            "submit_command": submit_command,
        }
    )
    return SlurmJobPlan(
        plan_id=plan_id,
        profile_id=profile.profile_id,
        scheduler="slurm",
        job_kind="scout_oof",
        dataset_id=request.dataset_id,
        invariant=invariant,
        invariants=(invariant,),
        evidence_scope="probe",
        representation_hash=request.representation_hash,
        job_name=job_name,
        working_directory=profile.project_root,
        script_path=script,
        resources=resources,
        environment=environment,
        input_manifests=input_manifests,
        manifest_path=str(report),
        stdout_path=str(stdout),
        stderr_path=str(stderr),
        submit_command=submit_command,
    )


def build_scout_combine_job_plan(
    profile: ExecutionProfile, request: ScoutCombineJobRequest
) -> SlurmJobPlan:
    if not profile.supports_slurm:
        raise ValueError(f"Profile {profile.profile_id!r} does not support Slurm Scout jobs")
    resources = profile.slurm_defaults or profile.slurm_scout_defaults
    if resources is None:
        raise ValueError(f"Profile {profile.profile_id!r} does not define Slurm resources")
    invariants = tuple(sorted({str(value).upper() for value in request.invariants}))
    reports = {str(key).upper(): str(value) for key, value in request.scout_reports.items()}
    if not invariants or set(reports) != set(invariants):
        raise ValueError("Scout source reports must match the requested invariants")
    if request.ranking_policy != "hierarchical_empirical_v1":
        raise ValueError("unsupported Scout ranking policy")
    dataset_token = _compact_path_token(request.dataset_id, max_length=48)
    run_token = _compact_path_token(request.run_id, max_length=48)
    repr_token = _safe_token(request.representation_hash)[:12]
    invariant_token = _compact_path_token(
        "-".join(name.lower() for name in invariants), max_length=40
    )
    policy_token = _compact_path_token(request.ranking_policy, max_length=32)
    job_name = f"mint-scout-combine-{invariant_token}"
    combined_report = (
        Path(profile.run_root)
        / f"scout_combined_{dataset_token}_{invariant_token}_{repr_token}_{run_token}_{policy_token}.json"
    )
    execution = (
        Path(profile.run_root)
        / f"scout_execution_{dataset_token}_{invariant_token}_{repr_token}_{run_token}_{policy_token}.json"
    )
    script = _remote_path(profile.project_root, profile.scout_combine_script)
    stdout = Path(profile.log_root) / f"{job_name}-%j.out"
    stderr = Path(profile.log_root) / f"{job_name}-%j.err"
    remote_reports = {
        name: _remote_path(profile.project_root, path) for name, path in reports.items()
    }
    environment = {
        "MINT_AGENT_ROOT": profile.project_root,
        "RUN_ROOT": profile.run_root,
        "LOG_ROOT": profile.log_root,
        "PYTHON_BIN": profile.python_executable,
        "DATASET_ID": request.dataset_id,
        "SCOUT_CONFIG": _remote_path(profile.project_root, request.scout_config),
        "REPRESENTATION_SPEC": _remote_path(
            profile.project_root, request.representation_spec
        ),
        "PROBE_SELECTION": _remote_path(profile.project_root, request.probe_selection),
        "FEATURE_QC_REPORT": _remote_path(
            profile.project_root, request.feature_qc_report
        ),
        "RANKING_POLICY": request.ranking_policy,
        "SOURCE_INVARIANTS": ":".join(invariants),
        "COMBINED_REPORT": str(combined_report),
        "EXECUTION_ARTIFACT": str(execution),
    }
    for name, path in remote_reports.items():
        environment[f"SCOUT_REPORT_{name}"] = path
    if request.llm_prior_order:
        environment["LLM_PRIOR_CANDIDATES"] = ";".join(
            ",".join(subset) for subset in request.llm_prior_order
        )
    if request.user_target is not None:
        environment["USER_TARGET"] = str(float(request.user_target))
    if profile.module_load:
        environment["MINT_MODULES"] = ":".join(profile.module_load)
    if profile.module_setup:
        environment["MINT_MODULE_SETUP"] = _remote_path(
            profile.project_root, profile.module_setup
        )
    _validate_export_values(environment)
    submit_command = (
        "env",
        *(f"{key}={value}" for key, value in environment.items()),
        "sbatch",
        f"--job-name={job_name}",
        f"--partition={resources.partition}",
        f"--ntasks={resources.ntasks}",
        f"--cpus-per-task={resources.cpus_per_task}",
        f"--mem={resources.mem}",
        f"--time={resources.time}",
        f"--output={stdout}",
        f"--error={stderr}",
        script,
    )
    invariant_label = "+".join(invariants)
    plan_id = feature_job_plan_id(
        {
            "profile_id": profile.profile_id,
            "job_kind": "scout_combine",
            "dataset_id": request.dataset_id,
            "invariant": invariant_label,
            "invariants": invariants,
            "evidence_scope": "probe",
            "representation_hash": request.representation_hash,
            "input_manifests": remote_reports,
            "manifest_path": str(execution),
            "submit_command": submit_command,
        }
    )
    return SlurmJobPlan(
        plan_id=plan_id,
        profile_id=profile.profile_id,
        scheduler="slurm",
        job_kind="scout_combine",
        dataset_id=request.dataset_id,
        invariant=invariant_label,
        invariants=invariants,
        evidence_scope="probe",
        representation_hash=request.representation_hash,
        job_name=job_name,
        working_directory=profile.project_root,
        script_path=script,
        resources=resources,
        environment=environment,
        input_manifests=remote_reports,
        manifest_path=str(execution),
        stdout_path=str(stdout),
        stderr_path=str(stderr),
        submit_command=submit_command,
    )


def build_model_evaluation_job_plan(
    profile: ExecutionProfile, request: ModelEvaluationJobRequest
) -> SlurmJobPlan:
    if not profile.supports_slurm:
        raise ValueError(
            f"Profile {profile.profile_id!r} does not support Slurm evaluation jobs"
        )
    if request.evidence_scope != "full_train":
        raise ValueError("V1 GBT evaluation jobs currently support full_train only")
    resources = (
        profile.slurm_evaluation_defaults
        or profile.slurm_scout_defaults
        or profile.slurm_defaults
    )
    if resources is None:
        raise ValueError(f"Profile {profile.profile_id!r} does not define evaluation resources")
    invariants = tuple(sorted({str(value).upper() for value in request.invariants}))
    manifests = {
        str(key).upper(): str(value) for key, value in request.feature_manifests.items()
    }
    if not invariants or set(manifests) != set(invariants):
        raise ValueError("Evaluation manifests must match the requested invariants")
    if request.max_acquisitions != len(invariants):
        raise ValueError("Evaluation acquisition limit must match the acquired prefix size")
    run_token = _safe_token(request.run_id)
    repr_token = _safe_token(request.representation_hash)[:12]
    invariant_token = "-".join(name.lower() for name in invariants)
    job_name = f"mint-evaluate-gbt-{invariant_token}"
    output = (
        Path(profile.run_root)
        / f"model_evaluation_{request.dataset_id}_{invariant_token}_{repr_token}_{run_token}.json"
    )
    script = _remote_path(profile.project_root, profile.model_evaluation_script)
    stdout = Path(profile.log_root) / f"{job_name}-%j.out"
    stderr = Path(profile.log_root) / f"{job_name}-%j.err"
    remote_manifests = {
        name: _remote_path(profile.project_root, path) for name, path in manifests.items()
    }
    environment = {
        "MINT_AGENT_ROOT": profile.project_root,
        "RUN_ROOT": profile.run_root,
        "LOG_ROOT": profile.log_root,
        "PYTHON_BIN": profile.python_executable,
        "DATASET_ID": request.dataset_id,
        "EVIDENCE_SCOPE": request.evidence_scope,
        "TASK_CONFIG": _remote_path(profile.project_root, request.task_config),
        "GBT_CONFIG": _remote_path(profile.project_root, request.gbt_config),
        "REPRESENTATION_SPEC": _remote_path(
            profile.project_root, request.representation_spec
        ),
        "SCOUT_ARTIFACT": _remote_path(profile.project_root, request.scout_artifact),
        "FEATURE_QC_REPORT": _remote_path(
            profile.project_root, request.feature_qc_report
        ),
        "FEATURE_INVARIANTS": ":".join(invariants),
        "MAX_ACQUISITIONS": str(request.max_acquisitions),
        "OUTPUT": str(output),
    }
    for name, path in remote_manifests.items():
        environment[f"FEATURE_MANIFEST_{name}"] = path
    if profile.module_load:
        environment["MINT_MODULES"] = ":".join(profile.module_load)
    if profile.module_setup:
        environment["MINT_MODULE_SETUP"] = _remote_path(
            profile.project_root, profile.module_setup
        )
    _validate_export_values(environment)
    submit_command = (
        "env",
        *(f"{key}={value}" for key, value in environment.items()),
        "sbatch",
        f"--job-name={job_name}",
        f"--partition={resources.partition}",
        f"--ntasks={resources.ntasks}",
        f"--cpus-per-task={resources.cpus_per_task}",
        f"--mem={resources.mem}",
        f"--time={resources.time}",
        f"--output={stdout}",
        f"--error={stderr}",
        script,
    )
    invariant_label = "+".join(invariants)
    plan_id = feature_job_plan_id(
        {
            "profile_id": profile.profile_id,
            "job_kind": "model_evaluation",
            "dataset_id": request.dataset_id,
            "invariant": invariant_label,
            "invariants": invariants,
            "evidence_scope": request.evidence_scope,
            "representation_hash": request.representation_hash,
            "input_manifests": remote_manifests,
            "manifest_path": str(output),
            "submit_command": submit_command,
        }
    )
    return SlurmJobPlan(
        plan_id=plan_id,
        profile_id=profile.profile_id,
        scheduler="slurm",
        job_kind="model_evaluation",
        dataset_id=request.dataset_id,
        invariant=invariant_label,
        invariants=invariants,
        evidence_scope=request.evidence_scope,
        representation_hash=request.representation_hash,
        job_name=job_name,
        working_directory=profile.project_root,
        script_path=script,
        resources=resources,
        environment=environment,
        input_manifests=remote_manifests,
        manifest_path=str(output),
        stdout_path=str(stdout),
        stderr_path=str(stderr),
        submit_command=submit_command,
    )


def build_validation_evaluation_job_plan(
    profile: ExecutionProfile, request: ValidationEvaluationJobRequest
) -> SlurmJobPlan:
    train_manifests = {
        str(name).upper(): str(path)
        for name, path in request.train_feature_manifests.items()
    }
    validation_manifests = {
        str(name).upper(): str(path)
        for name, path in request.validation_feature_manifests.items()
    }
    invariants = _validated_evaluation_invariants(
        request.invariants,
        train_manifests,
        validation_manifests,
    )
    if request.max_acquisitions != len(invariants):
        raise ValueError("Validation acquisition limit must match the acquired prefix size")
    if len(invariants) == 1 and request.prior_evaluation_report is not None:
        raise ValueError("The first validation stage cannot have a prior evaluation")
    if len(invariants) > 1 and request.prior_evaluation_report is None:
        raise ValueError("Later validation stages require a prior evaluation")
    inputs = {
        f"TRAIN_{name}": _remote_path(profile.project_root, train_manifests[name])
        for name in invariants
    }
    inputs.update(
        {
            f"VALIDATION_{name}": _remote_path(
                profile.project_root, validation_manifests[name]
            )
            for name in invariants
        }
    )
    environment = {
        "MINT_AGENT_ROOT": profile.project_root,
        "RUN_ROOT": profile.run_root,
        "LOG_ROOT": profile.log_root,
        "PYTHON_BIN": profile.python_executable,
        "DATASET_ID": request.dataset_id,
        "EVIDENCE_SCOPE": "validation",
        "TASK_CONFIG": _remote_path(profile.project_root, request.task_config),
        "GBT_CONFIG": _remote_path(profile.project_root, request.gbt_config),
        "REPRESENTATION_SPEC": _remote_path(profile.project_root, request.representation_spec),
        "SCOUT_ARTIFACT": _remote_path(profile.project_root, request.scout_artifact),
        "TRAIN_QC_REPORT": _remote_path(profile.project_root, request.train_feature_qc_report),
        "VALIDATION_QC_REPORT": _remote_path(
            profile.project_root, request.validation_feature_qc_report
        ),
        "FEATURE_INVARIANTS": ":".join(invariants),
        "MAX_ACQUISITIONS": str(request.max_acquisitions),
    }
    for key, path in inputs.items():
        environment[f"FEATURE_MANIFEST_{key}"] = path
    if request.prior_evaluation_report is not None:
        prior = _remote_path(profile.project_root, request.prior_evaluation_report)
        environment["PRIOR_EVALUATION_REPORT"] = prior
        inputs["PRIOR_EVALUATION"] = prior
    return _build_evaluation_plan(
        profile=profile,
        job_kind="validation_evaluation",
        dataset_id=request.dataset_id,
        invariants=invariants,
        evidence_scope="validation",
        representation_hash=request.representation_hash,
        run_id=request.run_id,
        script_path=profile.validation_evaluation_script,
        output_stem="validation_evaluation",
        environment=environment,
        input_manifests=inputs,
    )


def build_frozen_test_evaluation_job_plan(
    profile: ExecutionProfile, request: FrozenTestEvaluationJobRequest
) -> SlurmJobPlan:
    train = {
        str(name).upper(): str(path)
        for name, path in request.train_feature_manifests.items()
    }
    validation = {
        str(name).upper(): str(path)
        for name, path in (request.validation_feature_manifests or {}).items()
    }
    test = {
        str(name).upper(): str(path)
        for name, path in request.test_feature_manifests.items()
    }
    mappings: list[Mapping[str, str]] = [
        train,
        test,
    ]
    if validation:
        mappings.append(validation)
    invariants = _validated_evaluation_invariants(request.invariants, *mappings)
    if bool(validation) != bool(request.validation_feature_qc_report):
        raise ValueError("Validation manifests and QC report must be provided together")
    if request.n_bootstrap < 0:
        raise ValueError("n_bootstrap must be non-negative")
    inputs: dict[str, str] = {}
    for split, manifests in (
        ("TRAIN", train),
        ("VALIDATION", validation),
        ("TEST", test),
    ):
        for name in invariants:
            if name in manifests:
                inputs[f"{split}_{name}"] = _remote_path(
                    profile.project_root, manifests[name]
                )
    environment = {
        "MINT_AGENT_ROOT": profile.project_root,
        "RUN_ROOT": profile.run_root,
        "LOG_ROOT": profile.log_root,
        "PYTHON_BIN": profile.python_executable,
        "DATASET_ID": request.dataset_id,
        "EVIDENCE_SCOPE": "external_test",
        "TASK_CONFIG": _remote_path(profile.project_root, request.task_config),
        "GBT_CONFIG": _remote_path(profile.project_root, request.gbt_config),
        "REPRESENTATION_SPEC": _remote_path(profile.project_root, request.representation_spec),
        "SELECTION_REPORT": _remote_path(profile.project_root, request.selection_report),
        "TRAIN_QC_REPORT": _remote_path(profile.project_root, request.train_feature_qc_report),
        "TEST_QC_REPORT": _remote_path(profile.project_root, request.test_feature_qc_report),
        "FEATURE_INVARIANTS": ":".join(invariants),
        "N_BOOTSTRAP": str(request.n_bootstrap),
        "BOOTSTRAP_SEED": str(request.bootstrap_seed),
    }
    if request.validation_feature_qc_report:
        environment["VALIDATION_QC_REPORT"] = _remote_path(
            profile.project_root, request.validation_feature_qc_report
        )
    for key, path in inputs.items():
        environment[f"FEATURE_MANIFEST_{key}"] = path
    return _build_evaluation_plan(
        profile=profile,
        job_kind="frozen_test_evaluation",
        dataset_id=request.dataset_id,
        invariants=invariants,
        evidence_scope="external_test",
        representation_hash=request.representation_hash,
        run_id=request.run_id,
        script_path=profile.frozen_test_evaluation_script,
        output_stem="frozen_test_evaluation",
        environment=environment,
        input_manifests=inputs,
    )


def build_acceptance_evaluation_job_plan(
    profile: ExecutionProfile, request: AcceptanceEvaluationJobRequest
) -> SlurmJobPlan:
    if (request.candidate_rank is None) == (request.max_acquisitions is None):
        raise ValueError(
            "Provide exactly one of candidate_rank or max_acquisitions"
        )
    if request.candidate_rank is not None and request.candidate_rank < 1:
        raise ValueError("candidate_rank must be positive")
    if request.max_acquisitions is not None and request.max_acquisitions < 1:
        raise ValueError("max_acquisitions must be positive")
    train = {
        str(name).upper(): str(path)
        for name, path in request.train_feature_manifests.items()
    }
    evaluation = {
        str(name).upper(): str(path)
        for name, path in request.evaluation_feature_manifests.items()
    }
    invariants = _validated_evaluation_invariants(
        request.invariants, train, evaluation
    )
    inputs: dict[str, str] = {}
    for split, manifests in (("TRAIN", train), ("EVALUATION", evaluation)):
        for name in invariants:
            inputs[f"{split}_{name}"] = _remote_path(
                profile.project_root, manifests[name]
            )
    environment = {
        "MINT_AGENT_ROOT": profile.project_root,
        "RUN_ROOT": profile.run_root,
        "LOG_ROOT": profile.log_root,
        "PYTHON_BIN": profile.python_executable,
        "DATASET_ID": request.dataset_id,
        "EVIDENCE_SCOPE": "acceptance_test",
        "TASK_CONFIG": _remote_path(profile.project_root, request.task_config),
        "GBT_CONFIG": _remote_path(profile.project_root, request.gbt_config),
        "REPRESENTATION_SPEC": _remote_path(
            profile.project_root, request.representation_spec
        ),
        "SCOUT_ARTIFACT": _remote_path(
            profile.project_root, request.scout_artifact
        ),
        "ACCEPTANCE_MODE": (
            "progressive" if request.max_acquisitions is not None else "candidate"
        ),
        "TRAIN_QC_REPORT": _remote_path(
            profile.project_root, request.train_feature_qc_report
        ),
        "EVALUATION_QC_REPORT": _remote_path(
            profile.project_root, request.evaluation_feature_qc_report
        ),
        "FEATURE_INVARIANTS": ":".join(invariants),
    }
    if request.candidate_rank is not None:
        environment["CANDIDATE_RANK"] = str(request.candidate_rank)
    if request.max_acquisitions is not None:
        if request.max_acquisitions != len(invariants):
            raise ValueError(
                "max_acquisitions must equal the acquired invariant prefix size"
            )
        environment["MAX_ACQUISITIONS"] = str(request.max_acquisitions)
        if request.prior_evaluation_report is not None:
            environment["PRIOR_EVALUATION_REPORT"] = _remote_path(
                profile.project_root, request.prior_evaluation_report
            )
    for key, path in inputs.items():
        environment[f"FEATURE_MANIFEST_{key}"] = path
    return _build_evaluation_plan(
        profile=profile,
        job_kind="acceptance_evaluation",
        dataset_id=request.dataset_id,
        invariants=invariants,
        evidence_scope="acceptance_test",
        representation_hash=request.representation_hash,
        run_id=(
            f"{request.run_id}-stage-{request.max_acquisitions:02d}"
            if request.max_acquisitions is not None
            else f"{request.run_id}-candidate-{request.candidate_rank:02d}"
        ),
        script_path=profile.acceptance_evaluation_script,
        output_stem=(
            f"acceptance_evaluation_stage{request.max_acquisitions:02d}"
            if request.max_acquisitions is not None
            else f"acceptance_evaluation_rank{request.candidate_rank:02d}"
        ),
        environment=environment,
        input_manifests=inputs,
    )


def _validated_evaluation_invariants(
    raw_invariants: tuple[str, ...], *manifest_mappings: Mapping[str, str]
) -> tuple[str, ...]:
    invariants = tuple(dict.fromkeys(str(value).upper() for value in raw_invariants))
    if not invariants:
        raise ValueError("Evaluation requires at least one invariant")
    for manifests in manifest_mappings:
        normalized = {str(name).upper() for name in manifests}
        if normalized != set(invariants):
            raise ValueError("Evaluation manifests must match the requested invariants")
    return invariants


def _build_evaluation_plan(
    *,
    profile: ExecutionProfile,
    job_kind: str,
    dataset_id: str,
    invariants: tuple[str, ...],
    evidence_scope: str,
    representation_hash: str,
    run_id: str,
    script_path: str,
    output_stem: str,
    environment: Mapping[str, str],
    input_manifests: Mapping[str, str],
) -> SlurmJobPlan:
    if not profile.supports_slurm:
        raise ValueError(f"Profile {profile.profile_id!r} does not support Slurm evaluation jobs")
    resources = profile.slurm_evaluation_defaults or profile.slurm_scout_defaults or profile.slurm_defaults
    if resources is None:
        raise ValueError(f"Profile {profile.profile_id!r} does not define evaluation resources")
    run_token = _safe_token(run_id)
    repr_token = _safe_token(representation_hash)[:12]
    invariant_token = "-".join(name.lower() for name in invariants)
    job_name = f"mint-{job_kind.replace('_', '-')}-{invariant_token}"
    output = Path(profile.run_root) / (
        f"{output_stem}_{dataset_id}_{invariant_token}_{repr_token}_{run_token}.json"
    )
    full_environment = dict(environment)
    full_environment["OUTPUT"] = str(output)
    if profile.module_load:
        full_environment["MINT_MODULES"] = ":".join(profile.module_load)
    if profile.module_setup:
        full_environment["MINT_MODULE_SETUP"] = _remote_path(
            profile.project_root, profile.module_setup
        )
    _validate_export_values(full_environment)
    script = _remote_path(profile.project_root, script_path)
    stdout = Path(profile.log_root) / f"{job_name}-%j.out"
    stderr = Path(profile.log_root) / f"{job_name}-%j.err"
    submit_command = (
        "env",
        *(f"{key}={value}" for key, value in full_environment.items()),
        "sbatch",
        f"--job-name={job_name}",
        f"--partition={resources.partition}",
        f"--ntasks={resources.ntasks}",
        f"--cpus-per-task={resources.cpus_per_task}",
        f"--mem={resources.mem}",
        f"--time={resources.time}",
        f"--output={stdout}",
        f"--error={stderr}",
        script,
    )
    invariant_label = "+".join(invariants)
    plan_id = feature_job_plan_id(
        {
            "profile_id": profile.profile_id,
            "job_kind": job_kind,
            "dataset_id": dataset_id,
            "invariant": invariant_label,
            "invariants": invariants,
            "evidence_scope": evidence_scope,
            "representation_hash": representation_hash,
            "input_manifests": input_manifests,
            "manifest_path": str(output),
            "submit_command": submit_command,
        }
    )
    return SlurmJobPlan(
        plan_id=plan_id,
        profile_id=profile.profile_id,
        scheduler="slurm",
        job_kind=job_kind,
        dataset_id=dataset_id,
        invariant=invariant_label,
        invariants=invariants,
        evidence_scope=evidence_scope,
        representation_hash=representation_hash,
        job_name=job_name,
        working_directory=profile.project_root,
        script_path=script,
        resources=resources,
        environment=full_environment,
        input_manifests=dict(input_manifests),
        manifest_path=str(output),
        stdout_path=str(stdout),
        stderr_path=str(stderr),
        submit_command=submit_command,
    )


def _setup_environment(
    profile: ExecutionProfile, values: Mapping[str, str]
) -> dict[str, str]:
    environment = {
        "MINT_AGENT_ROOT": profile.project_root,
        "RUN_ROOT": profile.run_root,
        "LOG_ROOT": profile.log_root,
        "PYTHON_BIN": profile.python_executable,
        **dict(values),
    }
    if profile.module_load:
        environment["MINT_MODULES"] = ":".join(profile.module_load)
    if profile.module_setup:
        environment["MINT_MODULE_SETUP"] = _remote_path(
            profile.project_root, profile.module_setup
        )
    _validate_export_values(environment)
    return environment


def _build_setup_job_plan(
    profile: ExecutionProfile,
    *,
    job_kind: str,
    dataset_id: str,
    invariant: str,
    evidence_scope: str,
    representation_hash: str,
    run_id: str,
    script_path: str,
    output_path: Path,
    environment: Mapping[str, str],
    input_manifests: Mapping[str, str],
) -> SlurmJobPlan:
    if not profile.supports_slurm:
        raise ValueError(
            f"Profile {profile.profile_id!r} does not support Slurm setup jobs"
        )
    resources = profile.slurm_setup_defaults or profile.slurm_defaults
    if resources is None:
        raise ValueError(f"Profile {profile.profile_id!r} does not define setup resources")
    job_name = f"mint-{job_kind.replace('_', '-')}"
    script = _remote_path(profile.project_root, script_path)
    stdout = Path(profile.log_root) / f"{job_name}-%j.out"
    stderr = Path(profile.log_root) / f"{job_name}-%j.err"
    normalized_invariant = invariant.upper()
    normalized_inputs = {str(key).upper(): str(value) for key, value in input_manifests.items()}
    submit_command = (
        "env",
        *(f"{key}={value}" for key, value in environment.items()),
        "sbatch",
        f"--job-name={job_name}",
        f"--partition={resources.partition}",
        f"--ntasks={resources.ntasks}",
        f"--cpus-per-task={resources.cpus_per_task}",
        f"--mem={resources.mem}",
        f"--time={resources.time}",
        f"--output={stdout}",
        f"--error={stderr}",
        script,
    )
    plan_payload = {
        "profile_id": profile.profile_id,
        "job_kind": job_kind,
        "dataset_id": dataset_id,
        "invariant": normalized_invariant,
        "invariants": (normalized_invariant,),
        "evidence_scope": evidence_scope,
        "representation_hash": representation_hash,
        "input_manifests": normalized_inputs,
        "manifest_path": str(output_path),
        "submit_command": submit_command,
    }
    return SlurmJobPlan(
        plan_id=feature_job_plan_id(plan_payload),
        profile_id=profile.profile_id,
        scheduler="slurm",
        job_kind=job_kind,
        dataset_id=dataset_id,
        invariant=normalized_invariant,
        invariants=(normalized_invariant,),
        evidence_scope=evidence_scope,
        representation_hash=representation_hash,
        job_name=job_name,
        working_directory=profile.project_root,
        script_path=script,
        resources=resources,
        environment=dict(environment),
        input_manifests=normalized_inputs,
        manifest_path=str(output_path),
        stdout_path=str(stdout),
        stderr_path=str(stderr),
        submit_command=submit_command,
    )


def _mapping(data: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = data.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"execution profile {key} must be a mapping")
    return value


def _required_str(data: Mapping[str, Any], key: str) -> str:
    value = data.get(key)
    if value is None or str(value) == "":
        raise ValueError(f"execution profile requires {key}")
    return str(value)


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text or None


def _resources(value: Any) -> SlurmResources | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("Slurm resource config must be a mapping")
    return SlurmResources(
        partition=str(value.get("partition") or "batch"),
        ntasks=int(value.get("ntasks") or 1),
        cpus_per_task=int(value.get("cpus_per_task") or 1),
        mem=str(value.get("mem") or "8gb"),
        time=str(value.get("time") or "01:00:00"),
    )


def _scope_to_split(scope: str) -> str:
    if scope in {"full_train", "train", "probe"}:
        return "train"
    if scope == "validation":
        return "validation"
    if scope in {"test", "external_test"}:
        return "test"
    if scope == "all":
        return "all"
    raise ValueError(f"Unsupported evidence scope for feature job planning: {scope!r}")


def _safe_token(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)


def _compact_path_token(value: str, *, max_length: int = 48) -> str:
    token = _safe_token(str(value))
    if len(token) <= max_length:
        return token
    digest = stable_hash(str(value))[:10]
    prefix_length = max(1, max_length - len(digest) - 1)
    return f"{token[:prefix_length]}-{digest}"


def _remote_path(root: str, path: str) -> str:
    if path.startswith("/"):
        return path
    return str(Path(root) / path)


def _validate_export_values(environment: Mapping[str, str]) -> None:
    for key, value in environment.items():
        if any(character in value for character in (",", "\n", "\r")):
            raise ValueError(
                f"Slurm export value for {key} contains an unsupported comma or newline"
            )
