from __future__ import annotations

import json
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

from mint_scout.artifact_memory import (
    ArtifactRecord,
    ArtifactRegistry,
    inspect_artifact,
)
from mint_scout.config import load_yaml
from mint_scout.controlled_experiment import (
    ProbeComparisonPolicy,
    RepresentationComparisonPolicy,
    build_controlled_experiment_matrix,
    choose_probe_candidate,
    choose_representation_candidate,
    materialize_controlled_experiment,
)
from mint_scout.agent.state import MintAgentState
from mint_scout.agent.llm_scientific import (
    LLMScientificContext,
    experiment_planning_advice,
    scientific_critic_advice,
)
from mint_scout.data.manifest_io import task_card_from_config
from mint_scout.data.preparation import configured_manifest_path
from mint_scout.data_audit import dataset_audit_input_hash
from mint_scout.design_representation import representation_design_input_hash
from mint_scout.execution.artifacts import ScoutExecutionArtifact
from mint_scout.filtration_repair import (
    FiltrationRepairConfig,
    repair_filtration_profiles,
)
from mint_scout.invariants.manifest import stable_hash
from mint_scout.models.gbt import GBTConfig
from mint_scout.representation import RepresentationSpec
from mint_scout.execution.profile import (
    AcceptanceEvaluationJobRequest,
    DatasetAuditJobRequest,
    DatasetPreparationJobRequest,
    FeatureJobRequest,
    FrozenTestEvaluationJobRequest,
    FeatureOutlierDiagnosticJobRequest,
    FeatureQCJobRequest,
    FiltrationAuditJobRequest,
    ModelEvaluationJobRequest,
    ProbeAuditJobRequest,
    ProbeSelectionJobRequest,
    RepresentationDesignJobRequest,
    ScoutCombineJobRequest,
    ScoutOOFJobRequest,
    ValidationEvaluationJobRequest,
    build_dataset_audit_job_plan,
    build_acceptance_evaluation_job_plan,
    build_dataset_preparation_job_plan,
    build_feature_job_plan,
    build_feature_outlier_diagnostic_job_plan,
    build_feature_qc_job_plan,
    build_frozen_test_evaluation_job_plan,
    build_filtration_audit_job_plan,
    build_model_evaluation_job_plan,
    build_probe_audit_job_plan,
    build_probe_selection_job_plan,
    build_representation_design_job_plan,
    build_scout_combine_job_plan,
    build_scout_oof_job_plan,
    build_validation_evaluation_job_plan,
    load_execution_profile,
)
from mint_scout.feature_health import summarize_feature_health_files
from mint_scout.sample_size import LabeledSamplePolicy
from mint_scout.scout.pipeline import ScoutConfig
from mint_scout.schemas import (
    EvaluationMode,
    ManifestValidationError,
    route_evaluation_mode,
)


@dataclass(frozen=True)
class AgentGraphContext:
    task_config: Path
    registry: Path
    execution_profile: Path | None = None
    llm_scientific: LLMScientificContext | None = None


def _gbt_config_hash(path: str | Path) -> str | None:
    try:
        payload = load_yaml(Path(path))
        allowed = {field.name for field in fields(GBTConfig)}
        return GBTConfig(
            **{
                key: value
                for key, value in dict(payload).items()
                if key in allowed
            }
        ).parameter_hash
    except (OSError, TypeError, ValueError):
        return None


def _gbt_config_for_frozen_scout(state: Mapping[str, Any]) -> str:
    requested = str(
        state.get("request", {}).get("gbt_config")
        or "configs/gbt/plbind_adaptive_gbt.yaml"
    )
    scout = state.get("scout")
    scout_path = (
        scout.get("path")
        if isinstance(scout, Mapping)
        else None
    )
    if not scout_path:
        return requested
    try:
        frozen = ScoutExecutionArtifact.read(Path(str(scout_path)))
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return requested
    if _gbt_config_hash(requested) == frozen.gbt_parameter_hash:
        return requested
    for candidate in (
        "configs/gbt/plbind_adaptive_gbt.yaml",
        "configs/gbt/plbind_fixed_gbt.yaml",
    ):
        if _gbt_config_hash(candidate) == frozen.gbt_parameter_hash:
            return candidate
    return requested


def build_agent_graph(context: AgentGraphContext, *, checkpointer=None):
    try:
        from langgraph.graph import END, START, StateGraph
    except ImportError as exc:
        raise RuntimeError(
            "LangGraph orchestration requires the optional mint-agent[agent] dependencies"
        ) from exc

    registry = ArtifactRegistry(context.registry)
    execution_profile = (
        load_execution_profile(context.execution_profile)
        if context.execution_profile
        else None
    )
    llm_scientific = context.llm_scientific or LLMScientificContext()

    def dataset_preparation_node(state: MintAgentState) -> dict[str, Any]:
        task = load_yaml(context.task_config)
        raw_manifest = task.get("dataset_manifest")
        if raw_manifest is None:
            return {"route": "continue", "node_trace": ["dataset_preparation_node"]}
        try:
            manifest_path = configured_manifest_path(
                task, config_path=context.task_config
            )
        except ValueError as exc:
            return _failure(str(exc), "dataset_preparation_node")
        if manifest_path.is_file():
            return {"route": "continue", "node_trace": ["dataset_preparation_node"]}

        preparation = task.get("dataset_preparation")
        if not isinstance(preparation, Mapping):
            return {"route": "continue", "node_trace": ["dataset_preparation_node"]}
        preparation_jobs = []
        if execution_profile is not None:
            preparation_jobs.append(
                build_dataset_preparation_job_plan(
                    execution_profile,
                    DatasetPreparationJobRequest(
                        dataset_id=str(task.get("task_id") or ""),
                        run_id=state.get("run_id", "manual"),
                        task_config=str(context.task_config),
                        module_load=_preparation_modules(preparation),
                        replace_module_environment=bool(
                            preparation.get("replace_module_environment", False)
                        ),
                        python_executable=(
                            str(preparation["python_executable"])
                            if preparation.get("python_executable")
                            else None
                        ),
                    ),
                ).to_dict()
            )
        return {
            "route": "blocked",
            "status": "NEEDS_DATASET_PREPARATION",
            "preparation_jobs": preparation_jobs,
            "next_actions": [
                "Review and submit the deterministic dataset-preparation job."
                if preparation_jobs
                else "Run the configured deterministic dataset-preparation provider."
            ],
            "node_trace": ["dataset_preparation_node"],
        }

    def project_context_node(state: MintAgentState) -> dict[str, Any]:
        task = load_yaml(context.task_config)
        request = state.get("request", {})
        intent = task.get("intent")
        intent = intent if isinstance(intent, Mapping) else {}
        llm_intent = intent.get("llm")
        llm_intent = (
            dict(llm_intent)
            if isinstance(llm_intent, Mapping)
            else {"used": False, "used_for_numeric_decisions": False,}
        )
        llm_intent["used_for_numeric_decisions"] = False
        dataset_id = str(request.get("dataset_id") or task.get("task_id") or "")
        if not dataset_id:
            return _failure(
                "Task configuration does not identify a dataset", "project_context_node"
            )
        configured_dataset_id = str(task.get("task_id") or dataset_id)
        if (
            task.get("dataset_manifest") is not None
            and dataset_id != configured_dataset_id
        ):
            return _failure(
                "Requested dataset_id does not match the configured DatasetManifest",
                "project_context_node",
            )
        task_summary = {
            "task_id": configured_dataset_id,
            "dataset_id": dataset_id,
            "system_type": task.get("system_type"),
            "task_type": task.get("task_type"),
            "primary_metric": task.get("primary_metric"),
            "intent_source": intent.get("source") or "structured_config",
            "llm_intent": llm_intent,
            "selection_preferences": (
                dict(task["selection_preferences"])
                if isinstance(task.get("selection_preferences"), Mapping)
                else {}
            ),
        }
        try:
            task_card = task_card_from_config(
                task,
                config_path=context.task_config,
                user_target=request.get("user_target"),
            )
        except (FileNotFoundError, ManifestValidationError) as exc:
            return {
                "task": task_summary,
                "route": "blocked",
                "status": "INVALID_DATASET_MANIFEST",
                "errors": [str(exc)],
                "node_trace": ["project_context_node"],
            }
        if task_card is None:
            return {
                "task": task_summary,
                "route": "continue",
                "node_trace": ["project_context_node"],
            }

        evaluation_plan = route_evaluation_mode(task_card)
        task_summary.update(
            {
                "manifest_source": task_card.dataset.source,
                "label_name": task_card.dataset.label_name,
                "sample_count": len(task_card.dataset.samples),
                "labeled_sample_count": len(task_card.dataset.labeled_samples),
                "unlabeled_sample_count": len(task_card.dataset.unlabeled_samples),
                "evaluation_mode": evaluation_plan.mode.value,
                "modeling_sample_ids": list(evaluation_plan.modeling_sample_ids),
                "validation_sample_ids": list(evaluation_plan.validation_sample_ids),
                "evaluation_sample_ids": list(evaluation_plan.evaluation_sample_ids),
                "inference_sample_ids": list(evaluation_plan.inference_sample_ids),
                "warnings": list(evaluation_plan.warnings),
            }
        )
        blocked_status = {
            EvaluationMode.UNSUPPORTED_TASK: "UNSUPPORTED_TASK_TYPE",
            EvaluationMode.NEEDS_USER_INPUT: "NEEDS_USER_INPUT",
            EvaluationMode.LABEL_RETRIEVAL_REQUIRED: "NEEDS_LABEL_RETRIEVAL",
            EvaluationMode.INSUFFICIENT_LABELS: "INSUFFICIENT_LABELS",
        }.get(evaluation_plan.mode)
        if blocked_status is not None:
            return {
                "task": task_summary,
                "route": "blocked",
                "status": blocked_status,
                "next_actions": list(evaluation_plan.warnings),
                "node_trace": ["project_context_node"],
            }
        raw_label_policy = task.get("labels")
        if isinstance(raw_label_policy, Mapping):
            try:
                label_policy = LabeledSamplePolicy.from_mapping(raw_label_policy)
            except ValueError as exc:
                return _failure(str(exc), "project_context_node")
            small_data_warning = label_policy.warning(
                len(evaluation_plan.modeling_sample_ids)
            )
            task_summary["label_sample_policy"] = label_policy.to_dict()
            if small_data_warning is not None:
                task_summary["warnings"].append(small_data_warning)
                if not label_policy.allow_small_data_override:
                    return {
                        "task": task_summary,
                        "route": "blocked",
                        "status": "NEEDS_SMALL_DATA_CONFIRMATION",
                        "next_actions": [
                            "Set labels.allow_small_data_override=true only for an explicitly approved engineering run."
                        ],
                        "node_trace": ["project_context_node"],
                    }
        return {
            "task": task_summary,
            "route": "continue",
            "node_trace": ["project_context_node"],
        }

    def artifact_memory_node(state: MintAgentState) -> dict[str, Any]:
        dataset_id = state["task"]["dataset_id"]
        records = registry.query(dataset_id=dataset_id)
        return {
            "artifacts": [_record_summary(record) for record in records],
            "dataset_preparation": _latest(
                [_record_summary(record) for record in records],
                kind="dataset_preparation",
                evidence_scope="design",
            ),
            "node_trace": ["artifact_memory_node"],
        }

    def data_audit_node(state: MintAgentState) -> dict[str, Any]:
        artifacts = state.get("artifacts", [])
        task = load_yaml(context.task_config)
        expected_input_hash = dataset_audit_input_hash(
            task, config_path=context.task_config,
        )
        audit = _latest(
            [
                artifact
                for artifact in artifacts
                if artifact.get("metadata", {}).get("audit_input_hash")
                == expected_input_hash
            ],
            kind="dataset_audit",
            evidence_scope="design",
        )
        if audit is None and not any(
            key in task for key in ("dataset_manifest", "pair_schema", "data_audit")
        ):
            audit = _latest(artifacts, kind="dataset_audit", evidence_scope="design",)
        task_supports_data_audit = any(
            key in task for key in ("dataset_manifest", "pair_schema", "data_audit")
        )
        if audit is None:
            legacy_probe_audit = _latest(artifacts, kind="probe_audit")
            if legacy_probe_audit is not None and not task_supports_data_audit:
                return {
                    "route": "continue",
                    "node_trace": ["data_audit_node"],
                }
            setup_jobs = []
            if execution_profile is not None:
                setup_jobs.append(
                    build_dataset_audit_job_plan(
                        execution_profile,
                        DatasetAuditJobRequest(
                            dataset_id=state["task"]["dataset_id"],
                            run_id=state.get("run_id", "manual"),
                            task_config=str(context.task_config),
                        ),
                    ).to_dict()
                )
            return {
                "route": "blocked",
                "status": "NEEDS_DATA_AUDIT",
                "setup_jobs": setup_jobs,
                "next_actions": [
                    "Review and submit the deterministic dataset-audit job."
                    if setup_jobs
                    else "Run and register a deterministic dataset audit."
                ],
                "node_trace": ["data_audit_node"],
            }
        if audit.get("status") == "REVIEW":
            return {
                "dataset_audit": audit,
                "route": "blocked",
                "status": "NEEDS_SCHEMA_REVIEW",
                "next_actions": [
                    "Review out-of-schema elements and update the configurable representation policy."
                ],
                "node_trace": ["data_audit_node"],
            }
        return {
            "dataset_audit": audit,
            "route": "continue",
            "node_trace": ["data_audit_node"],
        }

    def experiment_planning_node(state: MintAgentState) -> dict[str, Any]:
        advice = experiment_planning_advice(state, llm_scientific)
        if (
            llm_scientific.mode == "advisory"
            and advice.get("status") == "VALIDATED"
            and isinstance(advice.get("llm"), Mapping)
            and advice["llm"].get("used", False)
        ):
            advice = dict(advice)
            advice["llm"] = {
                **dict(advice["llm"]),
                "used_for_numeric_decisions": False,
            }
            advice["applied_to_execution"] = True
        result: dict[str, Any] = {
            "experiment_plan": advice,
            "route": "continue",
            "node_trace": ["experiment_planning_node"],
        }
        structured = advice.get("advice")
        has_setting_variants = isinstance(structured, Mapping) and bool(
            structured.get("probe_variants")
            or structured.get("representation_variants")
        )
        if has_setting_variants:
            try:
                result["controlled_experiment"] = build_controlled_experiment_matrix(
                    task_config=load_yaml(context.task_config),
                    scout_config=load_yaml(
                        state.get("request", {}).get("scout_config")
                        or "configs/scout/v1.yaml"
                    ),
                    experiment_plan=advice,
                )
            except (OSError, TypeError, ValueError) as exc:
                return _failure(
                    f"Validated LLM setting proposals could not be materialized: {exc}",
                    "experiment_planning_node",
                )
        return result

    def representation_design_node(state: MintAgentState) -> dict[str, Any]:
        task = load_yaml(context.task_config)
        data_audit = state.get("dataset_audit")
        expected_input_hash = None
        if data_audit is not None:
            expected_input_hash = representation_design_input_hash(
                task,
                data_audit_content_sha256=data_audit.get("content_sha256"),
                split="train",
                offset=0,
                limit=None,
            )
        artifacts = state.get("artifacts", [])
        candidates = artifacts
        # Fully specified tasks use content-addressed representation designs.
        # Minimal legacy task files predate that contract and remain readable.
        if expected_input_hash is not None and "representation_design" in task:
            candidates = [
                artifact
                for artifact in candidates
                if artifact.get("metadata", {}).get("design_input_hash")
                == expected_input_hash
            ]
        representation = _latest(candidates, kind="representation_design")
        if representation is None and expected_input_hash is not None:
            representation = _resume_representation_from_downstream(artifacts)
        if representation is None or not representation.get("representation_hash"):
            setup_jobs = []
            if (
                execution_profile is not None
                and data_audit is not None
                and expected_input_hash is not None
            ):
                setup_jobs.append(
                    build_representation_design_job_plan(
                        execution_profile,
                        RepresentationDesignJobRequest(
                            dataset_id=state["task"]["dataset_id"],
                            run_id=state.get("run_id", "manual"),
                            task_config=str(context.task_config),
                            data_audit_report=data_audit["path"],
                            design_input_hash=expected_input_hash,
                        ),
                    ).to_dict()
                )
            return {
                "route": "blocked",
                "status": "NEEDS_REPRESENTATION",
                "setup_jobs": setup_jobs,
                "next_actions": [
                    "Review and submit the train-only representation-design job."
                    if setup_jobs
                    else "Design and freeze a train-only RepresentationSpec."
                ],
                "node_trace": ["representation_design_node"],
            }
        repaired_representation = _latest_repair_in_lineage(
            artifacts,
            representation,
            config=_filtration_repair_config(task),
        )
        if repaired_representation is not None:
            representation = repaired_representation
        return {
            "representation_hash": representation["representation_hash"],
            "representation_spec_path": representation["path"],
            "route": "continue",
            "node_trace": ["representation_design_node"],
        }

    def probe_setup_node(state: MintAgentState) -> dict[str, Any]:
        artifacts = state.get("artifacts", [])
        representation_hash = state["representation_hash"]
        probe = _latest(
            artifacts,
            kind="probe_selection",
            evidence_scope="probe",
            representation_hash=representation_hash,
        )
        scout_config = str(
            state.get("request", {}).get("scout_config") or "configs/scout/v1.yaml"
        )
        if probe is None or not probe.get("selection_hash"):
            setup_jobs = []
            representation_spec = state.get("representation_spec_path")
            if execution_profile is not None and representation_spec:
                setup_jobs.append(
                    build_probe_selection_job_plan(
                        execution_profile,
                        ProbeSelectionJobRequest(
                            dataset_id=state["task"]["dataset_id"],
                            representation_hash=representation_hash,
                            run_id=state.get("run_id", "manual"),
                            task_config=str(context.task_config),
                            scout_config=scout_config,
                            representation_spec=str(representation_spec),
                        ),
                    ).to_dict()
                )
            return {
                "route": "blocked",
                "status": "NEEDS_PROBE_SELECTION",
                "setup_jobs": setup_jobs,
                "next_actions": [
                    "Review and submit the deterministic probe-selection job."
                    if setup_jobs
                    else "Select and register a frozen representative probe."
                ],
                "node_trace": ["probe_setup_node"],
            }
        audit = _latest(
            artifacts,
            kind="probe_audit",
            evidence_scope="probe",
            representation_hash=representation_hash,
            selection_hash=str(probe["selection_hash"]),
        )
        if audit is None:
            legacy_audit = _latest(
                artifacts, kind="probe_audit", evidence_scope="probe",
            )
            if (
                legacy_audit is not None
                and legacy_audit.get("selection_hash") is None
                and legacy_audit.get("representation_hash")
                in {None, representation_hash}
            ):
                audit = legacy_audit
        if audit is None:
            setup_jobs = []
            if execution_profile is not None:
                setup_jobs.append(
                    build_probe_audit_job_plan(
                        execution_profile,
                        ProbeAuditJobRequest(
                            dataset_id=state["task"]["dataset_id"],
                            representation_hash=representation_hash,
                            run_id=state.get("run_id", "manual"),
                            task_config=str(context.task_config),
                            scout_config=scout_config,
                            probe_selection=probe["path"],
                        ),
                    ).to_dict()
                )
            return {
                "probe_selection": probe,
                "route": "blocked",
                "status": "NEEDS_PROBE_AUDIT",
                "setup_jobs": setup_jobs,
                "next_actions": [
                    "Review and submit the probe-representativeness audit job."
                    if setup_jobs
                    else "Audit and register the frozen probe."
                ],
                "node_trace": ["probe_setup_node"],
            }
        return {
            "probe_selection": probe,
            "probe_audit": audit,
            "route": "continue",
            "node_trace": ["probe_setup_node"],
        }

    def controlled_probe_node(state: MintAgentState) -> dict[str, Any]:
        matrix = state.get("controlled_experiment")
        if not isinstance(matrix, Mapping) or not matrix.get("execution_allowed"):
            return {"route": "continue", "node_trace": ["controlled_probe_node"]}
        probe_candidates = matrix.get("probe_candidates")
        if not isinstance(probe_candidates, list) or len(probe_candidates) <= 1:
            return {"route": "continue", "node_trace": ["controlled_probe_node"]}
        baseline_probe = state.get("probe_selection")
        baseline_audit = state.get("probe_audit")
        if not isinstance(baseline_probe, Mapping) or not isinstance(
            baseline_audit, Mapping
        ):
            return _failure(
                "Controlled Probe comparison requires the baseline probe and audit.",
                "controlled_probe_node",
            )
        representation_spec = state.get("representation_spec_path")
        if not representation_spec:
            return _failure(
                "Controlled Probe comparison requires a representation spec path.",
                "controlled_probe_node",
            )
        if execution_profile is None:
            return {
                "route": "blocked",
                "status": "NEEDS_CONTROLLED_PROBE_EXECUTION_PROFILE",
                "next_actions": [
                    "Provide an execution profile before running controlled Probe alternatives."
                ],
                "node_trace": ["controlled_probe_node"],
            }

        materialized = materialize_controlled_experiment(
            matrix,
            output_dir=_controlled_experiment_output_dir(
                execution_profile.run_root,
                dataset_id=str(state["task"]["dataset_id"]),
                matrix=matrix,
                representation_hash=state.get("representation_hash"),
            ),
        )
        try:
            _register_json_artifact(materialized["manifest_path"], registry=registry)
        except (OSError, ValueError) as exc:
            return _failure(
                f"Could not register controlled experiment matrix: {exc}",
                "controlled_probe_node",
            )

        artifacts = state.get("artifacts", [])
        descriptors: list[dict[str, Any]] = [
            {
                "candidate_id": "baseline-probe",
                "selection": baseline_probe["path"],
                "audit": baseline_audit["path"],
            }
        ]
        missing_selection_jobs: list[dict[str, Any]] = []
        pending_audit_jobs: list[dict[str, Any]] = []
        scout_config_by_id = {
            str(candidate["id"]): str(candidate.get("config_path") or "")
            for candidate in materialized["probe_candidates"]
            if isinstance(candidate, Mapping)
        }
        for candidate in materialized["probe_candidates"]:
            if (
                not isinstance(candidate, Mapping)
                or candidate.get("source") == "baseline"
            ):
                continue
            candidate_id = str(candidate["id"])
            run_id = f"{state.get('run_id', 'manual')}-probe-{candidate_id}"
            scout_config = scout_config_by_id[candidate_id]
            selection_plan = build_probe_selection_job_plan(
                execution_profile,
                ProbeSelectionJobRequest(
                    dataset_id=state["task"]["dataset_id"],
                    representation_hash=state["representation_hash"],
                    run_id=run_id,
                    task_config=str(context.task_config),
                    scout_config=scout_config,
                    representation_spec=str(representation_spec),
                ),
            ).to_dict()
            selection = _artifact_by_path(artifacts, selection_plan["manifest_path"])
            if selection is None or not selection.get("selection_hash"):
                missing_selection_jobs.append(selection_plan)
                continue
            audit_plan = build_probe_audit_job_plan(
                execution_profile,
                ProbeAuditJobRequest(
                    dataset_id=state["task"]["dataset_id"],
                    representation_hash=state["representation_hash"],
                    run_id=run_id,
                    task_config=str(context.task_config),
                    scout_config=scout_config,
                    probe_selection=selection["path"],
                ),
            ).to_dict()
            audit = _artifact_by_path(artifacts, audit_plan["manifest_path"])
            if audit is None:
                pending_audit_jobs.append(audit_plan)
                continue
            descriptors.append(
                {
                    "candidate_id": candidate_id,
                    "selection": selection["path"],
                    "audit": audit["path"],
                }
            )
        if missing_selection_jobs:
            return {
                "route": "blocked",
                "status": "NEEDS_CONTROLLED_PROBE_SELECTION",
                "setup_jobs": missing_selection_jobs,
                "controlled_experiment": materialized,
                "next_actions": [
                    "Review and submit controlled Probe selection jobs proposed by the validated LLM plan."
                ],
                "node_trace": ["controlled_probe_node"],
            }
        if pending_audit_jobs:
            return {
                "route": "blocked",
                "status": "NEEDS_CONTROLLED_PROBE_AUDIT",
                "setup_jobs": pending_audit_jobs,
                "controlled_experiment": materialized,
                "next_actions": [
                    "Review and submit controlled Probe representativeness audit jobs."
                ],
                "node_trace": ["controlled_probe_node"],
            }

        try:
            selected = choose_probe_candidate(
                descriptors,
                policy=_probe_policy_from_matrix(materialized),
            )
            selected_path = (
                Path(str(materialized["materialized_root"]))
                / "controlled_probe_selection.json"
            )
            _write_json_if_same(selected, selected_path)
            _register_json_artifact(selected_path, registry=registry)
        except (OSError, TypeError, ValueError) as exc:
            return _failure(
                f"Controlled Probe comparison failed: {exc}", "controlled_probe_node",
            )
        selected_probe = _artifact_by_path(
            [dict(baseline_probe), *artifacts,],
            str(selected["selected_probe_selection"]),
        )
        selected_audit = _artifact_by_path(
            [dict(baseline_audit), *artifacts,], str(selected["selected_probe_audit"]),
        )
        if selected_probe is None or selected_audit is None:
            return _failure(
                "Controlled Probe comparison selected an unregistered artifact.",
                "controlled_probe_node",
            )
        return {
            "controlled_experiment": materialized,
            "controlled_probe_selection": selected,
            "probe_selection": selected_probe,
            "probe_audit": selected_audit,
            "route": "continue",
            "node_trace": ["controlled_probe_node"],
        }

    def feature_planning_node(state: MintAgentState) -> dict[str, Any]:
        request = state.get("request", {})
        candidate_universe = tuple(
            dict.fromkeys(str(value).upper() for value in request.get("invariants", ()))
        )
        if not candidate_universe:
            return _failure("No invariants were requested", "feature_planning_node")
        scope = str(request.get("evidence_scope", "full_train"))
        if scope != "full_train":
            return {
                "route": "blocked",
                "status": "UNSUPPORTED_PROGRESSIVE_SCOPE",
                "next_actions": [
                    "Use a registered evaluation runner for non-full-train evidence."
                ],
                "node_trace": ["feature_planning_node"],
            }
        representation_hash = state["representation_hash"]
        priority_order = _scout_priority_order(state["scout"])
        if not priority_order:
            return {
                "route": "blocked",
                "status": "INVALID_SCOUT_ARTIFACT",
                "next_actions": [
                    "Regenerate the Scout artifact with a frozen candidate priority order."
                ],
                "node_trace": ["feature_planning_node"],
            }
        universe = set(candidate_universe)
        if any(not set(subset).issubset(universe) for subset in priority_order):
            return {
                "route": "blocked",
                "status": "INCOMPATIBLE_SCOUT_ARTIFACT",
                "next_actions": [
                    "Regenerate Scout for the exact requested invariant universe."
                ],
                "node_trace": ["feature_planning_node"],
            }

        acquisition_order = _flatten_priority_order(priority_order)
        if set(acquisition_order) != universe:
            return {
                "route": "blocked",
                "status": "INCOMPATIBLE_SCOUT_ARTIFACT",
                "next_actions": [
                    "Regenerate Scout with every requested invariant in its frozen queue."
                ],
                "node_trace": ["feature_planning_node"],
            }
        acquisition_stages = tuple(
            acquisition_order[:index] for index in range(1, len(acquisition_order) + 1)
        )
        completed_stages: list[dict[str, Any]] = []
        pending_stage: tuple[str, ...] | None = None
        for stage_index, stage in enumerate(acquisition_stages):
            evaluation = _latest(
                state.get("artifacts", []),
                kind="model_evaluation",
                evidence_scope=scope,
                representation_hash=representation_hash,
                invariants=list(stage),
                exact_invariants=True,
            )
            if evaluation is not None and not _same_evaluation_contract(
                state["scout"], evaluation
            ):
                evaluation = None
            if evaluation is None or evaluation.get("status") not in {
                "COMPLETE",
                "TARGET_REACHED",
                "TARGET_NOT_REACHED",
                "ACQUISITION_LIMIT_REACHED",
            }:
                pending_stage = stage
                break
            completed_stages.append(
                {
                    "stage_index": stage_index,
                    "acquired_invariants": list(stage),
                    "evaluation_path": evaluation["path"],
                    "status": evaluation.get("status"),
                }
            )
            if evaluation.get("status") in {"COMPLETE", "TARGET_REACHED"}:
                return {
                    "feature_plan": {
                        "candidate_universe": list(candidate_universe),
                        "priority_order": [list(value) for value in priority_order],
                        "acquisition_order": list(acquisition_order),
                        "completed_stages": completed_stages,
                        "target_reached_after": list(stage),
                        "evidence_scope": scope,
                        "available": {},
                        "manifests": {},
                        "missing": [],
                        "cache_hit_count": 0,
                    },
                    "evaluation": evaluation,
                    "route": "complete",
                    "status": "COMPLETE",
                    "node_trace": ["feature_planning_node"],
                }
            if evaluation.get("status") == "TARGET_NOT_REACHED":
                if stage != acquisition_stages[-1]:
                    return {
                        "route": "blocked",
                        "status": "INCOMPATIBLE_MODEL_EVALUATION",
                        "next_actions": [
                            "Rerun the intermediate stage with an acquisition limit."
                        ],
                        "node_trace": ["feature_planning_node"],
                    }

        if pending_stage is None:
            final_evaluation = None
            if completed_stages:
                final_evaluation = _latest(
                    state.get("artifacts", []),
                    kind="model_evaluation",
                    evidence_scope=scope,
                    representation_hash=representation_hash,
                    invariants=completed_stages[-1]["acquired_invariants"],
                    exact_invariants=True,
                )
            return {
                "feature_plan": {
                    "candidate_universe": list(candidate_universe),
                    "priority_order": [list(value) for value in priority_order],
                    "acquisition_order": list(acquisition_order),
                    "completed_stages": completed_stages,
                    "target_reached_after": None,
                    "evidence_scope": scope,
                    "available": {},
                    "manifests": {},
                    "missing": [],
                    "cache_hit_count": 0,
                },
                "evaluation": final_evaluation,
                "route": "complete",
                "status": "TARGET_NOT_REACHED",
                "node_trace": ["feature_planning_node"],
            }

        available: dict[str, str] = {}
        manifests: dict[str, dict[str, Any]] = {}
        all_missing: list[str] = []
        for invariant in pending_stage:
            record = _latest(
                state.get("artifacts", []),
                kind="feature_manifest",
                invariant=invariant,
                evidence_scope=scope,
                representation_hash=representation_hash,
                status="COMPLETE",
            )
            if record is None:
                all_missing.append(invariant)
            else:
                available[invariant] = record["path"]
                manifests[invariant] = record
        missing = all_missing[:1]
        route = "reuse" if not all_missing else "compute"
        return {
            "feature_plan": {
                "candidate_universe": list(candidate_universe),
                "priority_order": [list(value) for value in priority_order],
                "acquisition_order": list(acquisition_order),
                "completed_stages": completed_stages,
                "stage_index": len(completed_stages),
                "requested": list(pending_stage),
                "evidence_scope": scope,
                "available": available,
                "manifests": manifests,
                "missing": missing,
                "remaining_missing": all_missing,
                "cache_hit_count": len(available),
            },
            "route": route,
            "node_trace": ["feature_planning_node"],
        }

    def feature_tool_node(state: MintAgentState) -> dict[str, Any]:
        missing = state["feature_plan"]["missing"]
        feature_jobs = []
        if execution_profile is not None:
            for invariant in missing:
                plan = build_feature_job_plan(
                    execution_profile,
                    FeatureJobRequest(
                        dataset_id=state["task"]["dataset_id"],
                        invariant=invariant,
                        evidence_scope=state["feature_plan"]["evidence_scope"],
                        representation_hash=state["representation_hash"],
                        run_id=state.get("run_id", "manual"),
                        task_config=str(context.task_config),
                        representation_spec=(
                            state.get("request", {}).get("representation_spec")
                            or state.get("representation_spec_path")
                        ),
                        sample_id_file=state.get("request", {}).get("sample_id_file"),
                    ),
                )
                feature_jobs.append(plan.to_dict())
        return {
            "route": "blocked",
            "status": "NEEDS_FEATURE_COMPUTE",
            "feature_jobs": feature_jobs,
            "next_actions": [_feature_compute_action(missing, bool(feature_jobs))],
            "node_trace": ["feature_tool_node"],
        }

    def feature_qc_node(state: MintAgentState) -> dict[str, Any]:
        scope = state["feature_plan"]["evidence_scope"]
        requested = state["feature_plan"]["requested"]
        qc = _latest(
            state.get("artifacts", []),
            kind="feature_qc",
            evidence_scope=scope,
            representation_hash=state["representation_hash"],
            invariants=requested,
        )
        if qc is None:
            qc_jobs = []
            representation_spec = state.get("request", {}).get(
                "representation_spec"
            ) or state.get("representation_spec_path")
            if execution_profile is not None and representation_spec:
                plan = build_feature_qc_job_plan(
                    execution_profile,
                    FeatureQCJobRequest(
                        dataset_id=state["task"]["dataset_id"],
                        invariants=tuple(requested),
                        evidence_scope=scope,
                        representation_hash=state["representation_hash"],
                        run_id=state.get("run_id", "manual"),
                        representation_spec=str(representation_spec),
                        feature_manifests=state["feature_plan"]["available"],
                        qc_config=str(
                            state.get("request", {}).get("qc_config")
                            or "configs/scout/v1.yaml"
                        ),
                        sample_id_file=state.get("request", {}).get("sample_id_file"),
                    ),
                )
                qc_jobs.append(plan.to_dict())
            return {
                "route": "blocked",
                "status": "NEEDS_FEATURE_QC",
                "qc_jobs": qc_jobs,
                "next_actions": [
                    "Review and submit the generated Slurm feature-QC job plan."
                    if qc_jobs
                    else "Run feature QC against the frozen manifests."
                ],
                "node_trace": ["feature_qc_node"],
            }
        if qc.get("status") == "FAIL":
            return {
                "qc": qc,
                "route": "blocked",
                "status": "FEATURE_QC_FAILED",
                "next_actions": [
                    "Review the registered feature-QC errors before modeling."
                ],
                "node_trace": ["feature_qc_node"],
            }
        incompatible = [
            invariant
            for invariant, manifest in state["feature_plan"]["manifests"].items()
            if not _same_sample_axis(manifest, qc)
        ]
        if incompatible:
            return {
                "route": "blocked",
                "status": "INCOMPATIBLE_FEATURE_QC",
                "next_actions": [
                    "Regenerate feature QC with the same sample order as: "
                    + ", ".join(incompatible)
                ],
                "node_trace": ["feature_qc_node"],
            }
        return {"qc": qc, "route": "continue", "node_trace": ["feature_qc_node"]}

    def filtration_audit_node(state: MintAgentState) -> dict[str, Any]:
        requested = state["feature_plan"]["requested"]
        scope = state["feature_plan"]["evidence_scope"]
        audit = _latest(
            state.get("artifacts", []),
            kind="filtration_audit",
            evidence_scope=scope,
            representation_hash=state["representation_hash"],
            invariants=requested,
        )
        if audit is None:
            audit_jobs = []
            representation_spec = state.get("request", {}).get(
                "representation_spec"
            ) or state.get("representation_spec_path")
            if execution_profile is not None and representation_spec:
                plan = build_filtration_audit_job_plan(
                    execution_profile,
                    FiltrationAuditJobRequest(
                        dataset_id=state["task"]["dataset_id"],
                        invariants=tuple(requested),
                        evidence_scope=scope,
                        representation_hash=state["representation_hash"],
                        run_id=state.get("run_id", "manual"),
                        representation_spec=str(representation_spec),
                        feature_manifests=state["feature_plan"]["available"],
                        audit_config=str(
                            state.get("request", {}).get("audit_config")
                            or "configs/scout/v1.yaml"
                        ),
                        sample_id_file=state.get("request", {}).get("sample_id_file"),
                    ),
                )
                audit_jobs.append(plan.to_dict())
            return {
                "route": "blocked",
                "status": "NEEDS_FILTRATION_AUDIT",
                "filtration_audit_jobs": audit_jobs,
                "next_actions": [
                    "Review and submit the generated Slurm filtration-audit job plan."
                    if audit_jobs
                    else "Run the filtration-axis audit on the frozen features."
                ],
                "node_trace": ["filtration_audit_node"],
            }
        if audit.get("status") == "FAIL":
            return {
                "filtration_audit": audit,
                "route": "blocked",
                "status": "FILTRATION_AUDIT_FAILED",
                "next_actions": [
                    "Review the registered filtration-audit errors before modeling."
                ],
                "node_trace": ["filtration_audit_node"],
            }
        incompatible = [
            invariant
            for invariant, manifest in state["feature_plan"]["manifests"].items()
            if not _same_sample_axis(manifest, audit)
        ]
        if incompatible:
            return {
                "route": "blocked",
                "status": "INCOMPATIBLE_FILTRATION_AUDIT",
                "next_actions": [
                    "Regenerate the filtration audit with the same sample order as: "
                    + ", ".join(incompatible)
                ],
                "node_trace": ["filtration_audit_node"],
            }
        return {
            "filtration_audit": audit,
            "route": "continue",
            "node_trace": ["filtration_audit_node"],
        }

    def scout_node(state: MintAgentState) -> dict[str, Any]:
        artifacts = state.get("artifacts", [])
        requested = list(
            dict.fromkeys(
                str(value).upper()
                for value in state.get("request", {}).get("invariants", ())
            )
        )
        if not requested:
            return _failure("No invariants were requested", "scout_node")
        representation_hash = state["representation_hash"]
        probe = _latest(
            artifacts,
            kind="probe_selection",
            evidence_scope="probe",
            representation_hash=representation_hash,
        )
        if probe is None or not probe.get("selection_hash"):
            return {
                "route": "blocked",
                "status": "NEEDS_SCOUT_INPUTS",
                "next_actions": [
                    "Select and register a frozen probe for this representation."
                ],
                "node_trace": ["scout_node"],
            }

        selection_hash = str(probe["selection_hash"])
        user_target = state.get("request", {}).get("user_target")
        scout_config = str(
            state.get("request", {}).get("scout_config") or "configs/scout/v1.yaml"
        )
        expected_gbt_parameter_hash = ScoutConfig.from_mapping(
            load_yaml(scout_config)
        ).gbt.parameter_hash
        scout = _latest(
            artifacts,
            kind="scout_execution_plan",
            evidence_scope="probe",
            representation_hash=representation_hash,
            selection_hash=selection_hash,
            invariants=requested,
            exact_invariants=True,
            target_value=(float(user_target) if user_target is not None else None),
            gbt_parameter_hash=expected_gbt_parameter_hash,
            ranking_policy="hierarchical_empirical_v1",
        )
        representation_spec = state.get("request", {}).get(
            "representation_spec"
        ) or state.get("representation_spec_path")
        probe_manifests: dict[str, dict[str, Any]] = {}
        missing_probe_features: list[str] = []
        for invariant in requested:
            manifest = _latest(
                artifacts,
                kind="feature_manifest",
                invariant=invariant,
                evidence_scope="probe",
                representation_hash=representation_hash,
                selection_hash=selection_hash,
                status="COMPLETE",
            )
            if manifest is None:
                missing_probe_features.append(invariant)
            else:
                probe_manifests[invariant] = manifest
        if missing_probe_features:
            feature_jobs = []
            if execution_profile is not None and representation_spec:
                for invariant in missing_probe_features:
                    feature_jobs.append(
                        build_feature_job_plan(
                            execution_profile,
                            FeatureJobRequest(
                                dataset_id=state["task"]["dataset_id"],
                                invariant=invariant,
                                evidence_scope="probe",
                                representation_hash=representation_hash,
                                run_id=state.get("run_id", "manual"),
                                task_config=str(context.task_config),
                                representation_spec=str(representation_spec),
                                sample_id_file=probe["path"],
                                selection_hash=selection_hash,
                            ),
                        ).to_dict()
                    )
            return {
                "route": "blocked",
                "status": "NEEDS_SCOUT_FEATURES",
                "feature_jobs": feature_jobs,
                "next_actions": [
                    "Review and submit independent frozen-probe feature jobs for: "
                    + ", ".join(missing_probe_features)
                    if feature_jobs
                    else "Compute and register frozen-probe features for: "
                    + ", ".join(missing_probe_features)
                ],
                "node_trace": ["scout_node"],
            }

        probe_qc = _latest(
            artifacts,
            kind="feature_qc",
            evidence_scope="probe",
            representation_hash=representation_hash,
            selection_hash=selection_hash,
            invariants=requested,
            exact_invariants=True,
        )
        if probe_qc is None:
            qc_jobs = []
            if execution_profile is not None and representation_spec:
                qc_jobs.append(
                    build_feature_qc_job_plan(
                        execution_profile,
                        FeatureQCJobRequest(
                            dataset_id=state["task"]["dataset_id"],
                            invariants=tuple(requested),
                            evidence_scope="probe",
                            representation_hash=representation_hash,
                            run_id=state.get("run_id", "manual"),
                            representation_spec=str(representation_spec),
                            feature_manifests={
                                name: manifest["path"]
                                for name, manifest in probe_manifests.items()
                            },
                            qc_config=str(
                                state.get("request", {}).get("qc_config")
                                or "configs/scout/v1.yaml"
                            ),
                            sample_id_file=probe["path"],
                        ),
                    ).to_dict()
                )
            return {
                "route": "blocked",
                "status": "NEEDS_SCOUT_QC",
                "qc_jobs": qc_jobs,
                "next_actions": [
                    "Review and submit the probe-scoped feature-QC job."
                    if qc_jobs
                    else "Run and register probe-scoped feature QC."
                ],
                "node_trace": ["scout_node"],
            }
        if probe_qc.get("status") == "FAIL":
            return {
                "route": "blocked",
                "status": "PROBE_FEATURE_QC_FAILED",
                "qc": probe_qc,
                "next_actions": [
                    "Review probe feature anomalies before running filtration or GBT."
                ],
                "node_trace": ["scout_node"],
            }

        probe_audit = _latest(
            artifacts,
            kind="filtration_audit",
            evidence_scope="probe",
            representation_hash=representation_hash,
            selection_hash=selection_hash,
            invariants=requested,
            exact_invariants=True,
        )
        if probe_audit is None:
            audit_jobs = []
            if execution_profile is not None and representation_spec:
                audit_jobs.append(
                    build_filtration_audit_job_plan(
                        execution_profile,
                        FiltrationAuditJobRequest(
                            dataset_id=state["task"]["dataset_id"],
                            invariants=tuple(requested),
                            evidence_scope="probe",
                            representation_hash=representation_hash,
                            run_id=state.get("run_id", "manual"),
                            representation_spec=str(representation_spec),
                            feature_manifests={
                                name: manifest["path"]
                                for name, manifest in probe_manifests.items()
                            },
                            audit_config=str(
                                state.get("request", {}).get("audit_config")
                                or "configs/scout/v1.yaml"
                            ),
                            sample_id_file=probe["path"],
                        ),
                    ).to_dict()
                )
            return {
                "route": "blocked",
                "status": "NEEDS_SCOUT_FILTRATION_AUDIT",
                "filtration_audit_jobs": audit_jobs,
                "next_actions": [
                    "Review and submit the probe-scoped filtration-audit job."
                    if audit_jobs
                    else "Run and register a probe-scoped filtration audit."
                ],
                "node_trace": ["scout_node"],
            }
        if probe_audit.get("status") == "FAIL":
            return {
                "route": "blocked",
                "status": "PROBE_FILTRATION_AUDIT_FAILED",
                "filtration_audit": probe_audit,
                "next_actions": [
                    "Review probe filtration behavior before running GBT Scout."
                ],
                "node_trace": ["scout_node"],
            }

        incompatible_inputs = [
            invariant
            for invariant, manifest in probe_manifests.items()
            if not _same_sample_axis(manifest, probe_qc)
            or not _same_sample_axis(manifest, probe_audit)
        ]
        if incompatible_inputs:
            return {
                "route": "blocked",
                "status": "INCOMPATIBLE_SCOUT_INPUTS",
                "next_actions": [
                    "Regenerate probe evidence with one frozen sample order for: "
                    + ", ".join(incompatible_inputs)
                ],
                "node_trace": ["scout_node"],
            }

        try:
            repair = _materialize_filtration_repair(
                representation_spec_path=representation_spec,
                representation_hash=representation_hash,
                audit=probe_audit,
                task=load_yaml(context.task_config),
                task_id=state["task"]["dataset_id"],
                requested_invariants=requested,
                registry=registry,
            )
        except ValueError as exc:
            return _failure(str(exc), "scout_node")
        if repair is not None:
            return {
                "artifacts": [*artifacts, repair],
                "representation_hash": repair["representation_hash"],
                "representation_spec_path": repair["path"],
                "filtration_repair": repair,
                "filtration_audit": probe_audit,
                "route": "repair",
                "status": "FILTRATION_REPAIR_CREATED",
                "next_actions": [
                    "Recompute the frozen probe features only for the repaired "
                    "method-specific filtration profiles."
                ],
                "node_trace": ["scout_node"],
            }

        diagnostic_invariants, qc_hash = _qc_outlier_invariants(
            probe_qc["path"], requested
        )
        feature_diagnostics: dict[str, dict[str, Any]] = {}
        missing_diagnostics = []
        for invariant in diagnostic_invariants:
            diagnostic = _latest(
                artifacts,
                kind="feature_outlier_diagnostic",
                invariant=invariant,
                evidence_scope="probe",
                representation_hash=representation_hash,
                selection_hash=selection_hash,
                status="COMPLETE",
            )
            metadata = diagnostic.get("metadata", {}) if diagnostic else {}
            if (
                diagnostic is None
                or not _same_sample_axis(diagnostic, probe_qc)
                or not isinstance(metadata, Mapping)
                or metadata.get("feature_qc_hash") != qc_hash
            ):
                missing_diagnostics.append(invariant)
            else:
                feature_diagnostics[invariant] = diagnostic
        if missing_diagnostics:
            diagnostic_jobs = []
            if execution_profile is not None and representation_spec:
                top_k = int(
                    state.get("request", {}).get("feature_diagnostic_top_k", 10)
                )
                for invariant in missing_diagnostics:
                    diagnostic_jobs.append(
                        build_feature_outlier_diagnostic_job_plan(
                            execution_profile,
                            FeatureOutlierDiagnosticJobRequest(
                                dataset_id=state["task"]["dataset_id"],
                                invariant=invariant,
                                evidence_scope="probe",
                                representation_hash=representation_hash,
                                run_id=state.get("run_id", "manual"),
                                representation_spec=str(representation_spec),
                                feature_manifest=probe_manifests[invariant]["path"],
                                feature_qc_report=probe_qc["path"],
                                sample_id_file=probe["path"],
                                top_k=top_k,
                            ),
                        ).to_dict()
                    )
            return {
                "route": "blocked",
                "status": "NEEDS_FEATURE_DIAGNOSTIC",
                "feature_diagnostic_jobs": diagnostic_jobs,
                "qc": probe_qc,
                "filtration_audit": probe_audit,
                "next_actions": [
                    "Review and submit read-only feature diagnostics for: "
                    + ", ".join(missing_diagnostics)
                    if diagnostic_jobs
                    else "Run read-only feature diagnostics for: "
                    + ", ".join(missing_diagnostics)
                ],
                "node_trace": ["scout_node"],
            }

        if scout is not None:
            return {
                "scout": scout,
                "qc": probe_qc,
                "filtration_audit": probe_audit,
                "feature_diagnostics": feature_diagnostics,
                "route": "continue",
                "node_trace": ["scout_node"],
            }

        scout_reports: dict[str, dict[str, Any]] = {}
        missing_oof: list[str] = []
        for invariant in requested:
            report = _latest(
                artifacts,
                kind="scout_oof",
                invariant=invariant,
                evidence_scope="probe",
                representation_hash=representation_hash,
                selection_hash=selection_hash,
                status="COMPLETE",
                gbt_parameter_hash=expected_gbt_parameter_hash,
            )
            if report is None or not _same_sample_axis(report, probe_qc):
                missing_oof.append(invariant)
            else:
                scout_reports[invariant] = report

        if missing_oof:
            scout_jobs = []
            if execution_profile is not None and representation_spec:
                for invariant in missing_oof:
                    plan = build_scout_oof_job_plan(
                        execution_profile,
                        ScoutOOFJobRequest(
                            dataset_id=state["task"]["dataset_id"],
                            invariant=invariant,
                            representation_hash=representation_hash,
                            run_id=state.get("run_id", "manual"),
                            task_config=str(context.task_config),
                            scout_config=scout_config,
                            gbt_config=str(
                                state.get("request", {}).get("gbt_config")
                                or "configs/gbt/plbind_adaptive_gbt.yaml"
                            ),
                            representation_spec=str(representation_spec),
                            probe_selection=probe["path"],
                            feature_manifest=probe_manifests[invariant]["path"],
                            feature_qc_report=probe_qc["path"],
                            filtration_audit_report=probe_audit["path"],
                            user_target=(
                                float(user_target) if user_target is not None else None
                            ),
                        ),
                    )
                    scout_jobs.append(plan.to_dict())
            return {
                "route": "blocked",
                "status": "NEEDS_SCOUT_OOF",
                "scout_jobs": scout_jobs,
                "feature_diagnostics": feature_diagnostics,
                "next_actions": [
                    "Review and submit one independent Scout OOF job per missing invariant: "
                    + ", ".join(missing_oof)
                    if scout_jobs
                    else "Run independent Scout OOF CV for: " + ", ".join(missing_oof)
                ],
                "node_trace": ["scout_node"],
            }

        scout_jobs = []
        if execution_profile is not None and representation_spec:
            plan = build_scout_combine_job_plan(
                execution_profile,
                ScoutCombineJobRequest(
                    dataset_id=state["task"]["dataset_id"],
                    invariants=tuple(requested),
                    representation_hash=representation_hash,
                    run_id=state.get("run_id", "manual"),
                    scout_config=scout_config,
                    representation_spec=str(representation_spec),
                    probe_selection=probe["path"],
                    scout_reports={
                        invariant: scout_reports[invariant]["path"]
                        for invariant in requested
                    },
                    feature_qc_report=probe_qc["path"],
                    user_target=(
                        float(user_target) if user_target is not None else None
                    ),
                    llm_prior_order=(),
                ),
            )
            scout_jobs.append(plan.to_dict())
        return {
            "route": "blocked",
            "status": "NEEDS_SCOUT_COMBINE",
            "scout_jobs": scout_jobs,
            "feature_diagnostics": feature_diagnostics,
            "next_actions": [
                "Review and submit the deterministic Scout OOF combine job."
                if scout_jobs
                else "Combine the compatible Scout OOF reports."
            ],
            "node_trace": ["scout_node"],
        }

    def scientific_critic_node(state: MintAgentState) -> dict[str, Any]:
        advice = scientific_critic_advice(state, llm_scientific)
        return {
            "scientific_critique": advice,
            "route": "continue",
            "node_trace": ["scientific_critic_node"],
        }

    def controlled_representation_node(state: MintAgentState) -> dict[str, Any]:
        matrix = state.get("controlled_experiment")
        if not isinstance(matrix, Mapping) or not matrix.get("execution_allowed"):
            return {"route": "continue", "node_trace": []}
        representation_candidates = matrix.get("representation_candidates")
        if (
            not isinstance(representation_candidates, list)
            or len(representation_candidates) <= 1
        ):
            return {"route": "continue", "node_trace": []}
        if execution_profile is None:
            return {
                "route": "blocked",
                "status": "NEEDS_CONTROLLED_REPRESENTATION_EXECUTION_PROFILE",
                "next_actions": [
                    "Provide an execution profile before comparing representation candidates."
                ],
                "node_trace": ["controlled_representation_node"],
            }
        data_audit = state.get("dataset_audit")
        if not isinstance(data_audit, Mapping) or not data_audit.get("path"):
            return _failure(
                "Controlled representation comparison requires a registered train-only data audit.",
                "controlled_representation_node",
            )

        materialized = (
            dict(matrix)
            if matrix.get("materialized_root") and matrix.get("manifest_path")
            else materialize_controlled_experiment(
                matrix,
                output_dir=_controlled_experiment_output_dir(
                    execution_profile.run_root,
                    dataset_id=str(state["task"]["dataset_id"]),
                    matrix=matrix,
                    representation_hash=state.get("representation_hash"),
                ),
            )
        )
        if not matrix.get("manifest_path"):
            try:
                _register_json_artifact(
                    materialized["manifest_path"], registry=registry
                )
            except (OSError, ValueError) as exc:
                return _failure(
                    f"Could not register controlled experiment matrix: {exc}",
                    "controlled_representation_node",
                )

        baseline_representation_path = state.get("representation_spec_path")
        baseline_probe = state.get("probe_selection")
        baseline_qc = state.get("qc")
        baseline_filtration = state.get("filtration_audit")
        baseline_scout_execution = state.get("scout")
        if (
            not all(
                isinstance(value, Mapping)
                for value in (
                    baseline_probe,
                    baseline_qc,
                    baseline_filtration,
                    baseline_scout_execution,
                )
            )
            or not baseline_representation_path
        ):
            return _failure(
                "Controlled representation comparison requires completed baseline Probe evidence.",
                "controlled_representation_node",
            )
        baseline_combined = baseline_scout_execution.get("metadata", {}).get(
            "combined_report"
        )
        if not baseline_combined:
            return _failure(
                "Baseline Scout execution does not reference its combined Probe report.",
                "controlled_representation_node",
            )

        artifacts = state.get("artifacts", [])
        requested = tuple(
            dict.fromkeys(
                str(value).upper()
                for value in state.get("request", {}).get("invariants", ())
            )
        )
        scout_config = str(
            state.get("request", {}).get("scout_config") or "configs/scout/v1.yaml"
        )
        descriptors: list[dict[str, Any]] = [
            {
                "candidate_id": "baseline-representation",
                "llm_prior_rank": None,
                "representation": str(baseline_representation_path),
                "probe": baseline_probe["path"],
                "scout": str(baseline_combined),
                "scout_execution": baseline_scout_execution["path"],
                "qc": baseline_qc["path"],
                "filtration": baseline_filtration["path"],
            }
        ]
        setup_jobs: list[dict[str, Any]] = []
        feature_jobs: list[dict[str, Any]] = []
        qc_jobs: list[dict[str, Any]] = []
        filtration_jobs: list[dict[str, Any]] = []
        candidate_probe_audits: dict[str, dict[str, Any]] = {}
        candidate_scout_contexts: dict[str, dict[str, Any]] = {}

        for candidate in materialized["representation_candidates"]:
            if (
                not isinstance(candidate, Mapping)
                or candidate.get("source") == "baseline"
            ):
                continue
            candidate_id = str(candidate["id"])
            run_id = f"{state.get('run_id', 'manual')}-representation-{candidate_id}"
            candidate_task_path = str(candidate.get("config_path") or "")
            candidate_task = load_yaml(candidate_task_path)
            design_hash = representation_design_input_hash(
                candidate_task,
                data_audit_content_sha256=data_audit.get("content_sha256"),
                split="train",
                offset=0,
                limit=None,
            )
            design_plan = build_representation_design_job_plan(
                execution_profile,
                RepresentationDesignJobRequest(
                    dataset_id=state["task"]["dataset_id"],
                    run_id=run_id,
                    task_config=candidate_task_path,
                    data_audit_report=data_audit["path"],
                    design_input_hash=design_hash,
                ),
            ).to_dict()
            representation = _artifact_by_path(artifacts, design_plan["manifest_path"])
            if representation is None:
                setup_jobs.append(design_plan)
                continue
            representation_hash = str(representation.get("representation_hash") or "")
            if not representation_hash:
                return _failure(
                    f"Controlled representation {candidate_id!r} has no representation hash.",
                    "controlled_representation_node",
                )

            probe_plan = build_probe_selection_job_plan(
                execution_profile,
                ProbeSelectionJobRequest(
                    dataset_id=state["task"]["dataset_id"],
                    representation_hash=representation_hash,
                    run_id=run_id,
                    task_config=candidate_task_path,
                    scout_config=scout_config,
                    representation_spec=representation["path"],
                    frozen_sample_source=baseline_probe["path"],
                ),
            ).to_dict()
            probe = _artifact_by_path(artifacts, probe_plan["manifest_path"])
            if probe is None:
                setup_jobs.append(probe_plan)
                continue
            probe_audit_plan = build_probe_audit_job_plan(
                execution_profile,
                ProbeAuditJobRequest(
                    dataset_id=state["task"]["dataset_id"],
                    representation_hash=representation_hash,
                    run_id=run_id,
                    task_config=candidate_task_path,
                    scout_config=scout_config,
                    probe_selection=probe["path"],
                ),
            ).to_dict()
            probe_audit = _artifact_by_path(
                artifacts, probe_audit_plan["manifest_path"]
            )
            if probe_audit is None:
                setup_jobs.append(probe_audit_plan)
                continue
            candidate_probe_audits[candidate_id] = probe_audit

            manifests: dict[str, dict[str, Any]] = {}
            for invariant in requested:
                plan = build_feature_job_plan(
                    execution_profile,
                    FeatureJobRequest(
                        dataset_id=state["task"]["dataset_id"],
                        invariant=invariant,
                        evidence_scope="probe",
                        representation_hash=representation_hash,
                        run_id=run_id,
                        task_config=candidate_task_path,
                        representation_spec=representation["path"],
                        sample_id_file=probe["path"],
                        selection_hash=probe["selection_hash"],
                    ),
                ).to_dict()
                manifest = _artifact_by_path(artifacts, plan["manifest_path"])
                if manifest is None:
                    feature_jobs.append(plan)
                else:
                    manifests[invariant] = manifest
            if len(manifests) != len(requested):
                continue

            qc_plan = build_feature_qc_job_plan(
                execution_profile,
                FeatureQCJobRequest(
                    dataset_id=state["task"]["dataset_id"],
                    invariants=requested,
                    evidence_scope="probe",
                    representation_hash=representation_hash,
                    run_id=run_id,
                    representation_spec=representation["path"],
                    feature_manifests={
                        name: manifest["path"] for name, manifest in manifests.items()
                    },
                    qc_config=str(
                        state.get("request", {}).get("qc_config")
                        or "configs/scout/v1.yaml"
                    ),
                    sample_id_file=probe["path"],
                ),
            ).to_dict()
            qc = _artifact_by_path(artifacts, qc_plan["manifest_path"])
            if qc is None:
                qc_jobs.append(qc_plan)
                continue

            filtration_plan = build_filtration_audit_job_plan(
                execution_profile,
                FiltrationAuditJobRequest(
                    dataset_id=state["task"]["dataset_id"],
                    invariants=requested,
                    evidence_scope="probe",
                    representation_hash=representation_hash,
                    run_id=run_id,
                    representation_spec=representation["path"],
                    feature_manifests={
                        name: manifest["path"] for name, manifest in manifests.items()
                    },
                    audit_config=str(
                        state.get("request", {}).get("audit_config")
                        or "configs/scout/v1.yaml"
                    ),
                    sample_id_file=probe["path"],
                ),
            ).to_dict()
            filtration = _artifact_by_path(artifacts, filtration_plan["manifest_path"])
            if filtration is None:
                filtration_jobs.append(filtration_plan)
                continue

            if qc.get("status") == "FAIL" or filtration.get("status") == "FAIL":
                descriptors.append(
                    {
                        "candidate_id": candidate_id,
                        "llm_prior_rank": candidate.get("llm_prior_rank"),
                        "representation": representation["path"],
                        "probe": probe["path"],
                        "scout": None,
                        "scout_execution": None,
                        "qc": qc["path"],
                        "filtration": filtration["path"],
                    }
                )
                continue
            descriptors.append(
                {
                    "candidate_id": candidate_id,
                    "llm_prior_rank": candidate.get("llm_prior_rank"),
                    "representation": representation["path"],
                    "probe": probe["path"],
                    "scout": None,
                    "scout_execution": None,
                    "qc": qc["path"],
                    "filtration": filtration["path"],
                }
            )
            candidate_scout_contexts[candidate_id] = {
                "run_id": run_id,
                "task_config": candidate_task_path,
                "representation": representation,
                "representation_hash": representation_hash,
                "probe": probe,
                "qc": qc,
                "filtration": filtration,
                "manifests": manifests,
            }

        for status, key, jobs, action in (
            (
                "NEEDS_CONTROLLED_REPRESENTATION_SETUP",
                "setup_jobs",
                setup_jobs,
                "Run train-only representation, frozen-probe, and probe-audit setup jobs.",
            ),
            (
                "NEEDS_CONTROLLED_REPRESENTATION_FEATURES",
                "feature_jobs",
                feature_jobs,
                "Compute isolated Probe features for representation candidates.",
            ),
            (
                "NEEDS_CONTROLLED_REPRESENTATION_QC",
                "qc_jobs",
                qc_jobs,
                "Run feature QC for representation candidates.",
            ),
            (
                "NEEDS_CONTROLLED_REPRESENTATION_FILTRATION_AUDIT",
                "filtration_audit_jobs",
                filtration_jobs,
                "Audit filtration behavior for representation candidates.",
            ),
        ):
            if jobs:
                return {
                    "route": "blocked",
                    "status": status,
                    key: jobs,
                    "controlled_experiment": materialized,
                    "next_actions": [action],
                    "node_trace": ["controlled_representation_node"],
                }

        try:
            selection = choose_representation_candidate(
                descriptors,
                primary_metric=str(state.get("task", {}).get("target_metric") or "PCC"),
                policy=_representation_policy_from_matrix(materialized),
            )
            selection_path = (
                Path(str(materialized["materialized_root"]))
                / "controlled_representation_selection.v2.json"
            )
            _write_json_if_same(selection, selection_path)
            _register_json_artifact(selection_path, registry=registry)
        except (OSError, TypeError, ValueError) as exc:
            return _failure(
                f"Controlled representation comparison failed: {exc}",
                "controlled_representation_node",
            )

        selected_id = str(selection["selected_candidate_id"])
        selected_representation = _artifact_by_path(
            artifacts, str(selection["selected_representation"])
        )
        selected_probe = _artifact_by_path(
            [dict(baseline_probe), *artifacts], str(selection["selected_probe"])
        )
        selected_qc = _artifact_by_path(
            [dict(baseline_qc), *artifacts], str(selection["selected_qc"])
        )
        selected_filtration = _artifact_by_path(
            [dict(baseline_filtration), *artifacts],
            str(selection["selected_filtration"]),
        )
        selected_scout: Mapping[str, Any] | None = None
        if selected_id == "baseline-representation":
            selected_scout = baseline_scout_execution
        else:
            scout_context = candidate_scout_contexts.get(selected_id)
            if scout_context is None:
                return _failure(
                    "Selected representation has no feature-audit context.",
                    "controlled_representation_node",
                )
            oof_reports: dict[str, dict[str, Any]] = {}
            scout_jobs: list[dict[str, Any]] = []
            for invariant in requested:
                plan = build_scout_oof_job_plan(
                    execution_profile,
                    ScoutOOFJobRequest(
                        dataset_id=state["task"]["dataset_id"],
                        invariant=invariant,
                        representation_hash=scout_context["representation_hash"],
                        run_id=scout_context["run_id"],
                        task_config=scout_context["task_config"],
                        scout_config=scout_config,
                        gbt_config=str(
                            state.get("request", {}).get("gbt_config")
                            or "configs/gbt/plbind_adaptive_gbt.yaml"
                        ),
                        representation_spec=scout_context["representation"]["path"],
                        probe_selection=scout_context["probe"]["path"],
                        feature_manifest=scout_context["manifests"][invariant]["path"],
                        feature_qc_report=scout_context["qc"]["path"],
                        filtration_audit_report=scout_context["filtration"]["path"],
                        user_target=(
                            float(state.get("request", {})["user_target"])
                            if state.get("request", {}).get("user_target") is not None
                            else None
                        ),
                    ),
                ).to_dict()
                report = _artifact_by_path(artifacts, plan["manifest_path"])
                if report is None:
                    scout_jobs.append(plan)
                else:
                    oof_reports[invariant] = report
            if scout_jobs:
                return {
                    "route": "blocked",
                    "status": "NEEDS_CONTROLLED_REPRESENTATION_SCOUT",
                    "scout_jobs": scout_jobs,
                    "controlled_experiment": materialized,
                    "controlled_representation_selection": selection,
                    "next_actions": [
                        "Run method Scout only for the selected representation."
                    ],
                    "node_trace": ["controlled_representation_node"],
                }
            combine_plan = build_scout_combine_job_plan(
                execution_profile,
                ScoutCombineJobRequest(
                    dataset_id=state["task"]["dataset_id"],
                    invariants=requested,
                    representation_hash=scout_context["representation_hash"],
                    run_id=scout_context["run_id"],
                    scout_config=scout_config,
                    representation_spec=scout_context["representation"]["path"],
                    probe_selection=scout_context["probe"]["path"],
                    scout_reports={
                        name: report["path"] for name, report in oof_reports.items()
                    },
                    feature_qc_report=scout_context["qc"]["path"],
                    user_target=(
                        float(state.get("request", {})["user_target"])
                        if state.get("request", {}).get("user_target") is not None
                        else None
                    ),
                    llm_prior_order=(),
                ),
            ).to_dict()
            selected_scout = _artifact_by_path(
                artifacts, combine_plan["manifest_path"]
            )
            if selected_scout is not None and (
                selected_scout.get("metadata", {}).get("ranking_policy")
                != "hierarchical_empirical_v1"
            ):
                selected_scout = None
            if selected_scout is None:
                return {
                    "route": "blocked",
                    "status": "NEEDS_CONTROLLED_REPRESENTATION_SCOUT",
                    "scout_jobs": [combine_plan],
                    "controlled_experiment": materialized,
                    "controlled_representation_selection": selection,
                    "next_actions": [
                        "Combine the selected representation's method Scout evidence."
                    ],
                    "node_trace": ["controlled_representation_node"],
                }
        if not all(
            isinstance(value, Mapping)
            for value in (
                selected_representation,
                selected_probe,
                selected_qc,
                selected_filtration,
                selected_scout,
            )
        ):
            return _failure(
                "Controlled representation selection references an unregistered artifact.",
                "controlled_representation_node",
            )
        selected_probe_audit = (
            state.get("probe_audit")
            if selected_id == "baseline-representation"
            else candidate_probe_audits.get(selected_id)
        )
        return {
            "controlled_experiment": materialized,
            "controlled_representation_selection": selection,
            "representation_hash": selected_representation["representation_hash"],
            "representation_spec_path": selected_representation["path"],
            "probe_selection": selected_probe,
            "probe_audit": selected_probe_audit,
            "qc": selected_qc,
            "filtration_audit": selected_filtration,
            "scout": selected_scout,
            "route": "continue",
            "node_trace": ["controlled_representation_node"],
        }

    def evaluation_node(state: MintAgentState) -> dict[str, Any]:
        requested = state["feature_plan"]["requested"]
        evaluation = _latest(
            state.get("artifacts", []),
            kind="model_evaluation",
            evidence_scope=state["feature_plan"]["evidence_scope"],
            representation_hash=state["representation_hash"],
            invariants=requested,
            exact_invariants=True,
        )
        if evaluation is None:
            evaluation_jobs = []
            representation_spec = state.get("request", {}).get(
                "representation_spec"
            ) or state.get("representation_spec_path")
            if (
                execution_profile is not None
                and representation_spec
                and state["feature_plan"]["evidence_scope"] == "full_train"
            ):
                plan = build_model_evaluation_job_plan(
                    execution_profile,
                    ModelEvaluationJobRequest(
                        dataset_id=state["task"]["dataset_id"],
                        invariants=tuple(requested),
                        evidence_scope="full_train",
                        representation_hash=state["representation_hash"],
                        run_id=state.get("run_id", "manual"),
                        task_config=str(context.task_config),
                        gbt_config=_gbt_config_for_frozen_scout(state),
                        representation_spec=str(representation_spec),
                        scout_artifact=state["scout"]["path"],
                        feature_qc_report=state["qc"]["path"],
                        feature_manifests=state["feature_plan"]["available"],
                        max_acquisitions=len(requested),
                    ),
                )
                evaluation_jobs.append(plan.to_dict())
            return {
                "route": "blocked",
                "status": "NEEDS_MODEL_EVALUATION",
                "evaluation_jobs": evaluation_jobs,
                "next_actions": [
                    "Review and submit the generated full-train GBT evaluation job plan."
                    if evaluation_jobs
                    else "Run a registered model runner on the requested evidence scope."
                ],
                "node_trace": ["evaluation_node"],
            }
        manifests = state["feature_plan"]["manifests"].values()
        if any(not _same_sample_axis(manifest, evaluation) for manifest in manifests):
            return {
                "route": "blocked",
                "status": "INCOMPATIBLE_MODEL_EVALUATION",
                "next_actions": [
                    "Rerun evaluation with the registered full-train feature sample order."
                ],
                "node_trace": ["evaluation_node"],
            }
        if not _same_evaluation_contract(state["scout"], evaluation):
            return {
                "route": "blocked",
                "status": "INCOMPATIBLE_MODEL_EVALUATION",
                "next_actions": [
                    "Rerun evaluation with the frozen Scout sample order, GBT, metric, and target."
                ],
                "node_trace": ["evaluation_node"],
            }
        return {
            "evaluation": evaluation,
            "route": "continue",
            "status": (
                "COMPLETE"
                if evaluation.get("status") == "TARGET_REACHED"
                else evaluation.get("status") or "COMPLETE"
            ),
            "node_trace": ["evaluation_node"],
        }

    def split_selection_node(state: MintAgentState) -> dict[str, Any]:
        selection_preferences = state.get("task", {}).get(
            "selection_preferences", {}
        )
        selection_preferences = (
            selection_preferences
            if isinstance(selection_preferences, Mapping)
            else {}
        )
        selection_objective = str(
            selection_preferences.get("selection_objective")
            or "satisfy_target"
        )
        requested = tuple(
            dict.fromkeys(
                str(value).upper()
                for value in state.get("request", {}).get("invariants", ())
            )
        )
        priority_order = _scout_priority_order(state["scout"])
        acquisition_order = _flatten_priority_order(priority_order)
        if not requested or set(acquisition_order) != set(requested):
            return _failure(
                "Scout priority does not match the requested invariant universe",
                "split_selection_node",
            )
        try:
            stages = _stages_for_progressive_objective(
                priority_order,
                selection_objective=selection_objective,
                preferences=selection_preferences,
            )
            candidate_rank_limit = _candidate_rank_limit_for_objective(
                selection_objective,
                selection_preferences,
                priority_order,
            )
        except ValueError as exc:
            return _failure(str(exc), "split_selection_node")
        completed: list[dict[str, Any]] = []
        pending: tuple[str, ...] | None = None
        for stage in stages:
            evaluation = _latest(
                state.get("artifacts", []),
                kind="model_evaluation",
                evidence_scope="validation",
                representation_hash=state["representation_hash"],
                invariants=list(stage),
                exact_invariants=True,
            )
            if evaluation is not None and not _same_split_selection_contract(
                state["scout"], evaluation
            ):
                evaluation = None
            metadata = evaluation.get("metadata", {}) if evaluation else {}
            if evaluation is not None and (
                (metadata.get("selection_objective") or "satisfy_target")
                != selection_objective
                or metadata.get("candidate_rank_limit") != candidate_rank_limit
            ):
                evaluation = None
            if evaluation is None:
                pending = stage
                break
            completed.append(
                {
                    "acquired_invariants": list(stage),
                    "evaluation_path": evaluation["path"],
                    "status": evaluation.get("status"),
                    "selection_objective": selection_objective,
                }
            )
            if evaluation.get("status") in {
                "TARGET_REACHED",
                "MAXIMIZATION_COMPLETE",
            } or (
                stage == stages[-1] and evaluation.get("status") == "TARGET_NOT_REACHED"
            ):
                return {
                    "selection_evaluation": evaluation,
                    "split_feature_plan": {
                        "selection_objective": selection_objective,
                        "candidate_rank_limit": candidate_rank_limit,
                        "acquisition_order": list(acquisition_order),
                        "completed_stages": completed,
                    },
                    "route": "continue",
                    "status": evaluation.get("status"),
                    "node_trace": ["split_selection_node"],
                }
            if evaluation.get("status") != "ACQUISITION_LIMIT_REACHED":
                return {
                    "route": "blocked",
                    "status": "INCOMPATIBLE_VALIDATION_EVALUATION",
                    "next_actions": [
                        "Regenerate the validation stage with its frozen acquisition limit."
                    ],
                    "node_trace": ["split_selection_node"],
                }
        if selection_objective == "maximize_rank1" and pending is not None:
            for prefix_size in range(len(pending) - 1, 0, -1):
                prefix = pending[:prefix_size]
                prior = _latest(
                    state.get("artifacts", []),
                    kind="model_evaluation",
                    evidence_scope="validation",
                    representation_hash=state["representation_hash"],
                    invariants=list(prefix),
                    exact_invariants=True,
                )
                if prior is None or not _same_split_selection_contract(
                    state["scout"], prior
                ):
                    continue
                if (
                    prior.get("metadata", {}).get("selection_objective")
                    or "satisfy_target"
                ) != selection_objective or prior.get("metadata", {}).get(
                    "candidate_rank_limit"
                ) != candidate_rank_limit:
                    continue
                if prior.get("status") != "ACQUISITION_LIMIT_REACHED":
                    continue
                completed.append(
                    {
                        "acquired_invariants": list(prefix),
                        "evaluation_path": prior["path"],
                        "status": prior.get("status"),
                        "selection_objective": selection_objective,
                    }
                )
                break
        if pending is None:
            return _failure(
                "Validation selection ended without a frozen subset",
                "split_selection_node",
            )

        features: dict[str, dict[str, dict[str, Any]]] = {}
        missing_features: list[tuple[str, str]] = []
        for scope in ("full_train", "validation"):
            features[scope] = {}
            for invariant in pending:
                manifest = _latest(
                    state.get("artifacts", []),
                    kind="feature_manifest",
                    invariant=invariant,
                    evidence_scope=scope,
                    representation_hash=state["representation_hash"],
                    status="COMPLETE",
                )
                if manifest is None:
                    missing_features.append((scope, invariant))
                else:
                    features[scope][invariant] = manifest
        if missing_features:
            scheduled_missing = (
                missing_features
                if _parallel_acquisition_objective(selection_objective)
                else missing_features[:1]
            )
            feature_jobs = []
            representation_spec = state.get("request", {}).get(
                "representation_spec"
            ) or state.get("representation_spec_path")
            if execution_profile is not None and representation_spec:
                for scope, invariant in scheduled_missing:
                    feature_jobs.append(
                        build_feature_job_plan(
                            execution_profile,
                            FeatureJobRequest(
                                dataset_id=state["task"]["dataset_id"],
                                invariant=invariant,
                                evidence_scope=scope,
                                representation_hash=state["representation_hash"],
                                run_id=state.get("run_id", "manual"),
                                task_config=str(context.task_config),
                                representation_spec=str(representation_spec),
                            ),
                        ).to_dict()
                    )
            return {
                "split_feature_plan": {
                    "selection_objective": selection_objective,
                    "candidate_rank_limit": candidate_rank_limit,
                    "stage": list(pending),
                    "missing": list(dict.fromkeys(item[1] for item in scheduled_missing)),
                    "missing_by_scope": {
                        scope: [
                            invariant
                            for item_scope, invariant in scheduled_missing
                            if item_scope == scope
                        ]
                        for scope in dict.fromkeys(item[0] for item in scheduled_missing)
                    },
                    "parallel_feature_submission": _parallel_acquisition_objective(
                        selection_objective
                    ),
                    "completed_stages": completed,
                },
                "feature_jobs": feature_jobs,
                "route": "blocked",
                "status": "NEEDS_SELECTION_FEATURE_COMPUTE",
                "next_actions": [
                    "Compute all currently missing selected-candidate features in parallel."
                    if _parallel_acquisition_objective(selection_objective)
                    else (
                        f"Compute {scheduled_missing[0][1]} features for the "
                        f"{scheduled_missing[0][0]} selection split."
                    )
                ],
                "node_trace": ["split_selection_node"],
            }

        qcs: dict[str, dict[str, Any]] = {}
        audits: dict[str, dict[str, Any]] = {}
        for scope in ("full_train", "validation"):
            qc = _latest(
                state.get("artifacts", []),
                kind="feature_qc",
                evidence_scope=scope,
                representation_hash=state["representation_hash"],
                invariants=list(pending),
            )
            if qc is None:
                qc_jobs = []
                representation_spec = state.get("request", {}).get(
                    "representation_spec"
                ) or state.get("representation_spec_path")
                if execution_profile is not None and representation_spec:
                    qc_jobs.append(
                        build_feature_qc_job_plan(
                            execution_profile,
                            FeatureQCJobRequest(
                                dataset_id=state["task"]["dataset_id"],
                                invariants=pending,
                                evidence_scope=scope,
                                representation_hash=state["representation_hash"],
                                run_id=state.get("run_id", "manual"),
                                representation_spec=str(representation_spec),
                                feature_manifests={
                                    name: item["path"]
                                    for name, item in features[scope].items()
                                },
                                qc_config=str(
                                    state.get("request", {}).get("qc_config")
                                    or "configs/scout/v1.yaml"
                                ),
                            ),
                        ).to_dict()
                    )
                return {
                "split_feature_plan": {
                    "selection_objective": selection_objective,
                    "candidate_rank_limit": candidate_rank_limit,
                    "stage": list(pending),
                    "scope": scope,
                    "completed_stages": completed,
                    },
                    "qc_jobs": qc_jobs,
                    "route": "blocked",
                    "status": "NEEDS_SELECTION_FEATURE_QC",
                    "next_actions": [
                        f"Run feature QC for the {scope} selection split."
                    ],
                    "node_trace": ["split_selection_node"],
                }
            if qc.get("status") == "FAIL":
                return {
                    "route": "blocked",
                    "status": "SELECTION_FEATURE_QC_FAILED",
                    "next_actions": [f"Review failed {scope} feature QC."],
                    "node_trace": ["split_selection_node"],
                }
            qcs[scope] = qc
            audit = _latest(
                state.get("artifacts", []),
                kind="filtration_audit",
                evidence_scope=scope,
                representation_hash=state["representation_hash"],
                invariants=list(pending),
            )
            if audit is None:
                audit_jobs = []
                representation_spec = state.get("request", {}).get(
                    "representation_spec"
                ) or state.get("representation_spec_path")
                if execution_profile is not None and representation_spec:
                    audit_jobs.append(
                        build_filtration_audit_job_plan(
                            execution_profile,
                            FiltrationAuditJobRequest(
                                dataset_id=state["task"]["dataset_id"],
                                invariants=pending,
                                evidence_scope=scope,
                                representation_hash=state["representation_hash"],
                                run_id=state.get("run_id", "manual"),
                                representation_spec=str(representation_spec),
                                feature_manifests={
                                    name: item["path"]
                                    for name, item in features[scope].items()
                                },
                                audit_config=str(
                                    state.get("request", {}).get("audit_config")
                                    or "configs/scout/v1.yaml"
                                ),
                            ),
                        ).to_dict()
                    )
                return {
                "split_feature_plan": {
                    "selection_objective": selection_objective,
                    "candidate_rank_limit": candidate_rank_limit,
                    "stage": list(pending),
                    "scope": scope,
                    "completed_stages": completed,
                    },
                    "filtration_audit_jobs": audit_jobs,
                    "route": "blocked",
                    "status": "NEEDS_SELECTION_FILTRATION_AUDIT",
                    "next_actions": [
                        f"Audit filtration behavior for the {scope} selection split."
                    ],
                    "node_trace": ["split_selection_node"],
                }
            if audit.get("status") == "FAIL":
                return {
                    "route": "blocked",
                    "status": "SELECTION_FILTRATION_AUDIT_FAILED",
                    "next_actions": [f"Review failed {scope} filtration audit."],
                    "node_trace": ["split_selection_node"],
                }
            audits[scope] = audit

        evaluation_jobs = []
        representation_spec = state.get("request", {}).get(
            "representation_spec"
        ) or state.get("representation_spec_path")
        if execution_profile is not None and representation_spec:
            evaluation_jobs.append(
                build_validation_evaluation_job_plan(
                    execution_profile,
                    ValidationEvaluationJobRequest(
                        dataset_id=state["task"]["dataset_id"],
                        invariants=pending,
                        representation_hash=state["representation_hash"],
                        run_id=state.get("run_id", "manual"),
                        task_config=str(context.task_config),
                        gbt_config=_gbt_config_for_frozen_scout(state),
                        representation_spec=str(representation_spec),
                        scout_artifact=state["scout"]["path"],
                        train_feature_qc_report=qcs["full_train"]["path"],
                        validation_feature_qc_report=qcs["validation"]["path"],
                        train_feature_manifests={
                            name: item["path"]
                            for name, item in features["full_train"].items()
                        },
                        validation_feature_manifests={
                            name: item["path"]
                            for name, item in features["validation"].items()
                        },
                        max_acquisitions=len(pending),
                        prior_evaluation_report=(
                            completed[-1]["evaluation_path"] if completed else None
                        ),
                    ),
                ).to_dict()
            )
        return {
            "split_feature_plan": {
                "selection_objective": selection_objective,
                "candidate_rank_limit": candidate_rank_limit,
                "stage": list(pending),
                "completed_stages": completed,
                "manifests": {
                    scope: {name: item["path"] for name, item in values.items()}
                    for scope, values in features.items()
                },
            },
            "split_qc": qcs,
            "split_filtration_audits": audits,
            "evaluation_jobs": evaluation_jobs,
            "route": "blocked",
            "status": "NEEDS_VALIDATION_EVALUATION",
            "next_actions": ["Run the next frozen progressive stage on validation."],
            "node_trace": ["split_selection_node"],
        }

    def acceptance_selection_node(state: MintAgentState) -> dict[str, Any]:
        priority_order = _scout_priority_order(state["scout"])
        if not priority_order:
            return _failure(
                "No frozen Probe ranking is available", "acceptance_selection_node"
            )
        acquisition_order = _flatten_priority_order(priority_order)
        selection_preferences = state.get("task", {}).get(
            "selection_preferences", {}
        )
        selection_preferences = (
            selection_preferences
            if isinstance(selection_preferences, Mapping)
            else {}
        )
        selection_objective = str(
            selection_preferences.get("selection_objective")
            or "satisfy_target"
        )
        requested = tuple(
            dict.fromkeys(
                str(value).upper()
                for value in state.get("request", {}).get("invariants", ())
            )
        )
        if not requested or set(acquisition_order) != set(requested):
            return _failure(
                "Scout priority does not match the requested invariant universe",
                "acceptance_selection_node",
            )
        try:
            stages = _stages_for_progressive_objective(
                priority_order,
                selection_objective=selection_objective,
                preferences=selection_preferences,
            )
            candidate_rank_limit = _candidate_rank_limit_for_objective(
                selection_objective,
                selection_preferences,
                priority_order,
            )
        except ValueError as exc:
            return _failure(str(exc), "acceptance_selection_node")
        artifacts = state.get("artifacts", [])
        completed: list[dict[str, Any]] = []
        pending: tuple[str, ...] | None = None
        for stage in stages:
            evaluation = _latest(
                artifacts,
                kind="model_evaluation",
                evidence_scope="acceptance_test",
                representation_hash=state["representation_hash"],
                invariants=list(stage),
                exact_invariants=True,
            )
            metadata = evaluation.get("metadata", {}) if evaluation else {}
            if evaluation is not None and (
                metadata.get("scout_artifact") != state["scout"].get("path")
                or int(metadata.get("max_acquisitions") or 0) != len(stage)
                    or metadata.get("gbt_parameter_hash")
                    != state["scout"].get("metadata", {}).get("gbt_parameter_hash")
                    or metadata.get("selection_objective") != selection_objective
                    or metadata.get("candidate_rank_limit") != candidate_rank_limit
                ):
                evaluation = None
            if evaluation is None:
                pending = stage
                break
            completed.append(
                {
                    "max_acquisitions": len(stage),
                    "acquired_invariants": list(stage),
                    "status": evaluation.get("status"),
                    "evaluation_path": evaluation.get("path"),
                    "selected_score": evaluation.get("metadata", {}).get(
                        "selected_score"
                    ),
                    "selected_subset": evaluation.get("metadata", {}).get(
                        "selected_subset"
                    ),
                }
            )
            if evaluation.get("status") in {
                "TARGET_REACHED",
                "MAXIMIZATION_COMPLETE",
            }:
                return {
                    "acceptance_plan": {
                        "mode": "progressive_probe_ranked_acceptance",
                        "full_train_cross_validation": False,
                        "selection_objective": selection_objective,
                        "candidate_rank_limit": candidate_rank_limit,
                        "acquisition_order": list(acquisition_order),
                        "completed_stages": completed,
                        "accepted_candidate": evaluation.get("metadata", {}).get(
                            "selected_subset"
                        ),
                    },
                    "evaluation": evaluation,
                    "route": "complete",
                    "status": "COMPLETE",
                    "node_trace": ["acceptance_selection_node"],
                }
            if evaluation.get("status") == "TARGET_NOT_REACHED" and stage == stages[-1]:
                return {
                    "acceptance_plan": {
                        "mode": "progressive_probe_ranked_acceptance",
                        "full_train_cross_validation": False,
                        "selection_objective": selection_objective,
                        "candidate_rank_limit": candidate_rank_limit,
                        "acquisition_order": list(acquisition_order),
                        "completed_stages": completed,
                        "accepted_candidate": None,
                    },
                    "evaluation": evaluation,
                    "route": "complete",
                    "status": "TARGET_NOT_REACHED",
                    "node_trace": ["acceptance_selection_node"],
                }
            if evaluation.get("status") != "ACQUISITION_LIMIT_REACHED":
                return {
                    "route": "blocked",
                    "status": "INCOMPATIBLE_ACCEPTANCE_EVALUATION",
                    "next_actions": [
                        "Regenerate the progressive acceptance acquisition stage."
                    ],
                    "node_trace": ["acceptance_selection_node"],
                }

        if selection_objective == "maximize_rank1" and pending is not None:
            for prefix_size in range(len(pending) - 1, 0, -1):
                prefix = pending[:prefix_size]
                prior = _latest(
                    artifacts,
                    kind="model_evaluation",
                    evidence_scope="acceptance_test",
                    representation_hash=state["representation_hash"],
                    invariants=list(prefix),
                    exact_invariants=True,
                )
                metadata = prior.get("metadata", {}) if prior else {}
                if prior is None or (
                    metadata.get("scout_artifact") != state["scout"].get("path")
                    or int(metadata.get("max_acquisitions") or 0) != len(prefix)
                    or metadata.get("gbt_parameter_hash")
                    != state["scout"].get("metadata", {}).get("gbt_parameter_hash")
                    or metadata.get("selection_objective") != selection_objective
                    or metadata.get("candidate_rank_limit") != candidate_rank_limit
                    or prior.get("status") != "ACQUISITION_LIMIT_REACHED"
                ):
                    continue
                completed.append(
                    {
                        "max_acquisitions": len(prefix),
                        "acquired_invariants": list(prefix),
                        "status": prior.get("status"),
                        "evaluation_path": prior.get("path"),
                        "selected_score": metadata.get("selected_score"),
                        "selected_subset": metadata.get("selected_subset"),
                    }
                )
                break

        if pending is None:
            return _failure(
                "Acceptance search ended without a terminal result",
                "acceptance_selection_node",
            )

        features: dict[str, dict[str, dict[str, Any]]] = {}
        missing_features: list[tuple[str, str]] = []
        for scope in ("full_train", "external_test"):
            features[scope] = {}
            for invariant in pending:
                manifest = _latest(
                    artifacts,
                    kind="feature_manifest",
                    invariant=invariant,
                    evidence_scope=scope,
                    representation_hash=state["representation_hash"],
                    status="COMPLETE",
                )
                if manifest is None:
                    missing_features.append((scope, invariant))
                else:
                    features[scope][invariant] = manifest
        if missing_features:
            scheduled_missing = (
                missing_features
                if _parallel_acquisition_objective(selection_objective)
                else missing_features[:1]
            )
            feature_jobs = []
            representation_spec = state.get("request", {}).get(
                "representation_spec"
            ) or state.get("representation_spec_path")
            if execution_profile is not None and representation_spec:
                for scope, invariant in scheduled_missing:
                    feature_jobs.append(
                        build_feature_job_plan(
                            execution_profile,
                            FeatureJobRequest(
                                dataset_id=state["task"]["dataset_id"],
                                invariant=invariant,
                                evidence_scope=scope,
                                representation_hash=state["representation_hash"],
                                run_id=state.get("run_id", "manual"),
                                task_config=str(context.task_config),
                                representation_spec=str(representation_spec),
                            ),
                        ).to_dict()
                    )
            return {
                "acceptance_plan": {
                    "mode": "progressive_probe_ranked_acceptance",
                    "selection_objective": selection_objective,
                    "candidate_rank_limit": candidate_rank_limit,
                    "stage": list(pending),
                    "completed_stages": completed,
                    "missing": list(dict.fromkeys(item[1] for item in scheduled_missing)),
                    "missing_by_scope": {
                        scope: [
                            invariant
                            for item_scope, invariant in scheduled_missing
                            if item_scope == scope
                        ]
                        for scope in dict.fromkeys(item[0] for item in scheduled_missing)
                    },
                    "parallel_feature_submission": _parallel_acquisition_objective(
                        selection_objective
                    ),
                },
                "feature_jobs": feature_jobs,
                "route": "blocked",
                "status": "NEEDS_ACCEPTANCE_FEATURE_COMPUTE",
                "next_actions": [
                    "Compute all currently missing selected-candidate features in parallel."
                    if _parallel_acquisition_objective(selection_objective)
                    else (
                        f"Compute {scheduled_missing[0][1]} features for "
                        f"{scheduled_missing[0][0]}."
                    )
                ],
                "node_trace": ["acceptance_selection_node"],
            }

        qcs: dict[str, dict[str, Any]] = {}
        audits: dict[str, dict[str, Any]] = {}
        for scope in ("full_train", "external_test"):
            qc = _latest(
                artifacts,
                kind="feature_qc",
                evidence_scope=scope,
                representation_hash=state["representation_hash"],
                invariants=list(pending),
            )
            if qc is None:
                qc_jobs = []
                representation_spec = state.get("request", {}).get(
                    "representation_spec"
                ) or state.get("representation_spec_path")
                if execution_profile is not None and representation_spec:
                    qc_jobs.append(
                        build_feature_qc_job_plan(
                            execution_profile,
                            FeatureQCJobRequest(
                                dataset_id=state["task"]["dataset_id"],
                                invariants=pending,
                                evidence_scope=scope,
                                representation_hash=state["representation_hash"],
                                run_id=state.get("run_id", "manual"),
                                representation_spec=str(representation_spec),
                                feature_manifests={
                                    name: item["path"]
                                    for name, item in features[scope].items()
                                },
                                qc_config=str(
                                    state.get("request", {}).get("qc_config")
                                    or "configs/scout/v1.yaml"
                                ),
                            ),
                        ).to_dict()
                    )
                return {
                    "acceptance_plan": {
                        "mode": "progressive_probe_ranked_acceptance",
                        "selection_objective": selection_objective,
                        "candidate_rank_limit": candidate_rank_limit,
                        "stage": list(pending),
                        "completed_stages": completed,
                        "scope": scope,
                    },
                    "qc_jobs": qc_jobs,
                    "route": "blocked",
                    "status": "NEEDS_ACCEPTANCE_FEATURE_QC",
                    "next_actions": [f"Run feature QC for {scope}."],
                    "node_trace": ["acceptance_selection_node"],
                }
            if qc.get("status") == "FAIL":
                return {
                    "route": "blocked",
                    "status": "ACCEPTANCE_FEATURE_QC_FAILED",
                    "next_actions": [f"Review failed {scope} feature QC."],
                    "node_trace": ["acceptance_selection_node"],
                }
            qcs[scope] = qc
            audit = _latest(
                artifacts,
                kind="filtration_audit",
                evidence_scope=scope,
                representation_hash=state["representation_hash"],
                invariants=list(pending),
            )
            if audit is None:
                audit_jobs = []
                representation_spec = state.get("request", {}).get(
                    "representation_spec"
                ) or state.get("representation_spec_path")
                if execution_profile is not None and representation_spec:
                    audit_jobs.append(
                        build_filtration_audit_job_plan(
                            execution_profile,
                            FiltrationAuditJobRequest(
                                dataset_id=state["task"]["dataset_id"],
                                invariants=pending,
                                evidence_scope=scope,
                                representation_hash=state["representation_hash"],
                                run_id=state.get("run_id", "manual"),
                                representation_spec=str(representation_spec),
                                feature_manifests={
                                    name: item["path"]
                                    for name, item in features[scope].items()
                                },
                                audit_config=str(
                                    state.get("request", {}).get("audit_config")
                                    or "configs/scout/v1.yaml"
                                ),
                            ),
                        ).to_dict()
                    )
                return {
                    "filtration_audit_jobs": audit_jobs,
                    "route": "blocked",
                    "status": "NEEDS_ACCEPTANCE_FILTRATION_AUDIT",
                    "next_actions": [f"Audit filtration behavior for {scope}."],
                    "node_trace": ["acceptance_selection_node"],
                }
            if audit.get("status") == "FAIL":
                return {
                    "route": "blocked",
                    "status": "ACCEPTANCE_FILTRATION_AUDIT_FAILED",
                    "next_actions": [f"Review failed {scope} filtration audit."],
                    "node_trace": ["acceptance_selection_node"],
                }
            audits[scope] = audit

        evaluation_jobs = []
        representation_spec = state.get("request", {}).get(
            "representation_spec"
        ) or state.get("representation_spec_path")
        if execution_profile is not None and representation_spec:
            evaluation_jobs.append(
                build_acceptance_evaluation_job_plan(
                    execution_profile,
                    AcceptanceEvaluationJobRequest(
                        dataset_id=state["task"]["dataset_id"],
                        invariants=pending,
                        representation_hash=state["representation_hash"],
                        run_id=state.get("run_id", "manual"),
                        task_config=str(context.task_config),
                        gbt_config=_gbt_config_for_frozen_scout(state),
                        representation_spec=str(representation_spec),
                        scout_artifact=state["scout"]["path"],
                        max_acquisitions=len(pending),
                        prior_evaluation_report=(
                            completed[-1]["evaluation_path"] if completed else None
                        ),
                        train_feature_qc_report=qcs["full_train"]["path"],
                        evaluation_feature_qc_report=qcs["external_test"]["path"],
                        train_feature_manifests={
                            name: item["path"]
                            for name, item in features["full_train"].items()
                        },
                        evaluation_feature_manifests={
                            name: item["path"]
                            for name, item in features["external_test"].items()
                        },
                    ),
                ).to_dict()
            )
        return {
            "acceptance_plan": {
                "mode": "progressive_probe_ranked_acceptance",
                "selection_objective": selection_objective,
                "candidate_rank_limit": candidate_rank_limit,
                "stage": list(pending),
                "acquisition_order": list(acquisition_order),
                "completed_stages": completed,
                "full_train_cross_validation": False,
                "run_aggregation": "mean_predictions_then_compute_metric",
            },
            "split_qc": qcs,
            "split_filtration_audits": audits,
            "evaluation_jobs": evaluation_jobs,
            "route": "blocked",
            "status": "NEEDS_ACCEPTANCE_EVALUATION",
            "next_actions": [
                "Fit only the newly acquired invariant and score every available subset."
            ],
            "node_trace": ["acceptance_selection_node"],
        }

    def final_test_node(state: MintAgentState) -> dict[str, Any]:
        selection = state.get("selection_evaluation") or state.get("evaluation")
        if (
            not isinstance(selection, Mapping)
            and state.get("task", {}).get("evaluation_mode")
            == EvaluationMode.EXPLICIT_LABELED_TEST.value
            and isinstance(state.get("scout"), Mapping)
        ):
            try:
                selection = _materialize_probe_rank_selection(
                    state=state,
                    task=load_yaml(context.task_config),
                    registry=registry,
                )
            except (OSError, TypeError, ValueError) as exc:
                return _failure(
                    f"Frozen Probe-rank selection could not be materialized: {exc}",
                    "final_test_node",
                )
        if not isinstance(selection, Mapping) or not selection.get("path"):
            return _failure("No frozen selection is available", "final_test_node")
        selected_subset = _selected_subset_from_artifact(selection)
        if not selected_subset:
            return {
                "route": "blocked",
                "status": "INVALID_FROZEN_SELECTION",
                "next_actions": [
                    "Regenerate the selection report with selected_subset."
                ],
                "node_trace": ["final_test_node"],
            }
        existing = _latest(
            state.get("artifacts", []),
            kind="model_evaluation",
            evidence_scope="external_test",
            representation_hash=state["representation_hash"],
            invariants=list(selected_subset),
            exact_invariants=True,
        )
        if (
            existing is not None
            and existing.get("metadata", {}).get("selection_report")
            == selection["path"]
        ):
            evaluation_status = str(existing.get("status") or "")
            final_status = (
                "TARGET_NOT_REACHED"
                if evaluation_status == "TARGET_NOT_REACHED"
                else "COMPLETE"
            )
            return {
                "evaluation": existing,
                "final_test_plan": {
                    "selected_subset": list(selected_subset),
                    "selection_report": selection["path"],
                    "test_query_count": 1,
                },
                "route": "complete",
                "status": final_status,
                "node_trace": ["final_test_node"],
            }

        mode = str(state.get("task", {}).get("evaluation_mode") or "")
        scopes = ["full_train"]
        if mode == EvaluationMode.EXPLICIT_VALIDATION_AND_TEST.value:
            scopes.append("validation")
        scopes.append("external_test")
        features: dict[str, dict[str, dict[str, Any]]] = {}
        for scope in scopes:
            features[scope] = {}
            for invariant in selected_subset:
                manifest = _latest(
                    state.get("artifacts", []),
                    kind="feature_manifest",
                    invariant=invariant,
                    evidence_scope=scope,
                    representation_hash=state["representation_hash"],
                    status="COMPLETE",
                )
                if manifest is None:
                    feature_jobs = []
                    representation_spec = state.get("request", {}).get(
                        "representation_spec"
                    ) or state.get("representation_spec_path")
                    if execution_profile is not None and representation_spec:
                        feature_jobs.append(
                            build_feature_job_plan(
                                execution_profile,
                                FeatureJobRequest(
                                    dataset_id=state["task"]["dataset_id"],
                                    invariant=invariant,
                                    evidence_scope=scope,
                                    representation_hash=state["representation_hash"],
                                    run_id=state.get("run_id", "manual"),
                                    task_config=str(context.task_config),
                                    representation_spec=str(representation_spec),
                                ),
                            ).to_dict()
                        )
                    return {
                        "final_test_plan": {
                            "selected_subset": list(selected_subset),
                            "selection_report": selection["path"],
                            "scope": scope,
                            "missing": [invariant],
                        },
                        "feature_jobs": feature_jobs,
                        "route": "blocked",
                        "status": "NEEDS_FINAL_TEST_FEATURE_COMPUTE",
                        "next_actions": [
                            f"Compute only selected {invariant} features for {scope}."
                        ],
                        "node_trace": ["final_test_node"],
                    }
                features[scope][invariant] = manifest

        qcs: dict[str, dict[str, Any]] = {}
        audits: dict[str, dict[str, Any]] = {}
        for scope in scopes:
            qc = _latest(
                state.get("artifacts", []),
                kind="feature_qc",
                evidence_scope=scope,
                representation_hash=state["representation_hash"],
                invariants=list(selected_subset),
            )
            if qc is None:
                qc_jobs = []
                representation_spec = state.get("request", {}).get(
                    "representation_spec"
                ) or state.get("representation_spec_path")
                if execution_profile is not None and representation_spec:
                    qc_jobs.append(
                        build_feature_qc_job_plan(
                            execution_profile,
                            FeatureQCJobRequest(
                                dataset_id=state["task"]["dataset_id"],
                                invariants=selected_subset,
                                evidence_scope=scope,
                                representation_hash=state["representation_hash"],
                                run_id=state.get("run_id", "manual"),
                                representation_spec=str(representation_spec),
                                feature_manifests={
                                    name: item["path"]
                                    for name, item in features[scope].items()
                                },
                                qc_config=str(
                                    state.get("request", {}).get("qc_config")
                                    or "configs/scout/v1.yaml"
                                ),
                            ),
                        ).to_dict()
                    )
                return {
                    "final_test_plan": {
                        "selected_subset": list(selected_subset),
                        "selection_report": selection["path"],
                        "scope": scope,
                    },
                    "qc_jobs": qc_jobs,
                    "route": "blocked",
                    "status": "NEEDS_FINAL_TEST_FEATURE_QC",
                    "next_actions": [f"Run selected-feature QC for {scope}."],
                    "node_trace": ["final_test_node"],
                }
            if qc.get("status") == "FAIL":
                return {
                    "route": "blocked",
                    "status": "FINAL_TEST_FEATURE_QC_FAILED",
                    "next_actions": [f"Review failed selected-feature QC for {scope}."],
                    "node_trace": ["final_test_node"],
                }
            qcs[scope] = qc
            audit = _latest(
                state.get("artifacts", []),
                kind="filtration_audit",
                evidence_scope=scope,
                representation_hash=state["representation_hash"],
                invariants=list(selected_subset),
            )
            if audit is None:
                audit_jobs = []
                representation_spec = state.get("request", {}).get(
                    "representation_spec"
                ) or state.get("representation_spec_path")
                if execution_profile is not None and representation_spec:
                    audit_jobs.append(
                        build_filtration_audit_job_plan(
                            execution_profile,
                            FiltrationAuditJobRequest(
                                dataset_id=state["task"]["dataset_id"],
                                invariants=selected_subset,
                                evidence_scope=scope,
                                representation_hash=state["representation_hash"],
                                run_id=state.get("run_id", "manual"),
                                representation_spec=str(representation_spec),
                                feature_manifests={
                                    name: item["path"]
                                    for name, item in features[scope].items()
                                },
                                audit_config=str(
                                    state.get("request", {}).get("audit_config")
                                    or "configs/scout/v1.yaml"
                                ),
                            ),
                        ).to_dict()
                    )
                return {
                    "filtration_audit_jobs": audit_jobs,
                    "route": "blocked",
                    "status": "NEEDS_FINAL_TEST_FILTRATION_AUDIT",
                    "next_actions": [
                        f"Audit selected-feature filtration behavior for {scope}."
                    ],
                    "node_trace": ["final_test_node"],
                }
            if audit.get("status") == "FAIL":
                return {
                    "route": "blocked",
                    "status": "FINAL_TEST_FILTRATION_AUDIT_FAILED",
                    "next_actions": [f"Review failed filtration audit for {scope}."],
                    "node_trace": ["final_test_node"],
                }
            audits[scope] = audit

        evaluation_jobs = []
        representation_spec = state.get("request", {}).get(
            "representation_spec"
        ) or state.get("representation_spec_path")
        if execution_profile is not None and representation_spec:
            evaluation_jobs.append(
                build_frozen_test_evaluation_job_plan(
                    execution_profile,
                    FrozenTestEvaluationJobRequest(
                        dataset_id=state["task"]["dataset_id"],
                        invariants=selected_subset,
                        representation_hash=state["representation_hash"],
                        run_id=state.get("run_id", "manual"),
                        task_config=str(context.task_config),
                        gbt_config=_gbt_config_for_frozen_scout(state),
                        representation_spec=str(representation_spec),
                        selection_report=str(selection["path"]),
                        train_feature_qc_report=qcs["full_train"]["path"],
                        test_feature_qc_report=qcs["external_test"]["path"],
                        train_feature_manifests={
                            name: item["path"]
                            for name, item in features["full_train"].items()
                        },
                        test_feature_manifests={
                            name: item["path"]
                            for name, item in features["external_test"].items()
                        },
                        validation_feature_qc_report=(
                            qcs["validation"]["path"] if "validation" in qcs else None
                        ),
                        validation_feature_manifests=(
                            {
                                name: item["path"]
                                for name, item in features["validation"].items()
                            }
                            if "validation" in features
                            else None
                        ),
                        n_bootstrap=int(
                            state.get("request", {}).get(
                                "test_bootstrap_replicates", 1000
                            )
                        ),
                        bootstrap_seed=int(
                            state.get("request", {}).get("test_bootstrap_seed", 2026)
                        ),
                    ),
                ).to_dict()
            )
        return {
            "final_test_plan": {
                "selected_subset": list(selected_subset),
                "selection_report": selection["path"],
                "feature_manifests": {
                    scope: {name: item["path"] for name, item in values.items()}
                    for scope, values in features.items()
                },
                "test_query_count": 0,
            },
            "split_qc": qcs,
            "split_filtration_audits": audits,
            "evaluation_jobs": evaluation_jobs,
            "route": "blocked",
            "status": "NEEDS_FINAL_TEST_EVALUATION",
            "next_actions": [
                "Refit the frozen subset and query the protected test split exactly once."
            ],
            "node_trace": ["final_test_node"],
        }

    def report_node(state: MintAgentState) -> dict[str, Any]:
        status = state.get(
            "status", "COMPLETE" if state.get("evaluation") else "BLOCKED"
        )
        qc = state.get("qc")
        filtration_audit = state.get("filtration_audit")
        feature_health = None
        if isinstance(qc, Mapping) and isinstance(filtration_audit, Mapping):
            if qc.get("path") and filtration_audit.get("path"):
                feature_health = summarize_feature_health_files(
                    qc["path"],
                    filtration_audit["path"],
                    identity_hint={
                        field: qc.get(field)
                        for field in (
                            "dataset_id",
                            "evidence_scope",
                            "representation_hash",
                            "selection_hash",
                            "sample_count",
                        )
                    },
                )
        report = {
            "report_schema": "mint-agent.graph-run.v1",
            "run_id": state.get("run_id"),
            "status": status,
            "task": state.get("task"),
            "representation_hash": state.get("representation_hash"),
            "representation_spec_path": state.get("representation_spec_path"),
            "preparation_jobs": state.get("preparation_jobs", []),
            "dataset_preparation": state.get("dataset_preparation"),
            "setup_jobs": state.get("setup_jobs", []),
            "dataset_audit": state.get("dataset_audit"),
            "experiment_plan": state.get("experiment_plan"),
            "controlled_experiment": state.get("controlled_experiment"),
            "controlled_probe_selection": state.get("controlled_probe_selection"),
            "controlled_representation_selection": state.get(
                "controlled_representation_selection"
            ),
            "probe_selection": state.get("probe_selection"),
            "probe_audit": state.get("probe_audit"),
            "feature_plan": state.get("feature_plan"),
            "feature_jobs": state.get("feature_jobs", []),
            "qc_jobs": state.get("qc_jobs", []),
            "filtration_audit_jobs": state.get("filtration_audit_jobs", []),
            "feature_diagnostic_jobs": state.get("feature_diagnostic_jobs", []),
            "scout_jobs": state.get("scout_jobs", []),
            "evaluation_jobs": state.get("evaluation_jobs", []),
            "split_feature_plan": state.get("split_feature_plan"),
            "split_qc": state.get("split_qc", {}),
            "split_filtration_audits": state.get("split_filtration_audits", {}),
            "selection_evaluation": state.get("selection_evaluation"),
            "acceptance_plan": state.get("acceptance_plan"),
            "final_test_plan": state.get("final_test_plan"),
            "qc": qc,
            "filtration_audit": filtration_audit,
            "feature_health": feature_health,
            "feature_diagnostics": state.get("feature_diagnostics", {}),
            "scout": state.get("scout"),
            "scientific_critique": state.get("scientific_critique"),
            "evaluation": state.get("evaluation"),
            "next_actions": state.get("next_actions", []),
            "errors": state.get("errors", []),
            "scientific_control": {
                "llm_used_for_intent": bool(
                    state.get("task", {}).get("llm_intent", {}).get("used", False)
                ),
                "llm_used_for_numeric_decisions": False,
                "llm_scientific_mode": llm_scientific.mode,
                "llm_used_for_experiment_planning": _llm_stage_used(
                    state.get("experiment_plan")
                ),
                "llm_used_for_scientific_critique": _llm_stage_used(
                    state.get("scientific_critique")
                ),
                "llm_advice_applied_to_execution": bool(
                    state.get("experiment_plan", {}).get("applied_to_execution", False)
                    if isinstance(state.get("experiment_plan"), Mapping)
                    else False
                ),
                "test_used_for_representation_selection": False,
                "evaluation_set_used_for_candidate_selection": bool(state.get("acceptance_plan")),
                "test_acceptance_used": bool(state.get("acceptance_plan")),
                "full_train_cross_validation_used": _full_train_cv_used(state),
                "independent_test_available": (
                    False
                    if state.get("acceptance_plan")
                    else bool(state.get("final_test_plan"))
                    if state.get("task", {}).get("evaluation_mode")
                    in {
                        EvaluationMode.EXPLICIT_LABELED_TEST.value,
                        EvaluationMode.EXPLICIT_VALIDATION_AND_TEST.value,
                    }
                    else None
                ),
                "artifact_memory_read_only": True,
            },
        }
        return {"final_report": report, "status": status, "node_trace": ["report_node"]}

    builder = StateGraph(MintAgentState)
    builder.add_node("dataset_preparation", dataset_preparation_node)
    builder.add_node("project_context", project_context_node)
    builder.add_node("artifact_memory", artifact_memory_node)
    builder.add_node("data_audit", data_audit_node)
    builder.add_node("experiment_planning", experiment_planning_node)
    builder.add_node("representation_design", representation_design_node)
    builder.add_node("probe_setup", probe_setup_node)
    builder.add_node("controlled_probe", controlled_probe_node)
    builder.add_node("feature_planning", feature_planning_node)
    builder.add_node("feature_tool", feature_tool_node)
    builder.add_node("feature_qc", feature_qc_node)
    builder.add_node("filtration_audit", filtration_audit_node)
    builder.add_node("scout", scout_node)
    builder.add_node("controlled_representation", controlled_representation_node)
    builder.add_node("scientific_critic", scientific_critic_node)
    builder.add_node("evaluation", evaluation_node)
    builder.add_node("split_selection", split_selection_node)
    builder.add_node("acceptance_selection", acceptance_selection_node)
    builder.add_node("final_test", final_test_node)
    builder.add_node("report", report_node)
    builder.add_edge(START, "dataset_preparation")
    builder.add_conditional_edges(
        "dataset_preparation",
        _continue_or_report,
        {"continue": "project_context", "report": "report"},
    )
    builder.add_conditional_edges(
        "project_context",
        _continue_or_report,
        {"continue": "artifact_memory", "report": "report"},
    )
    builder.add_edge("artifact_memory", "data_audit")
    builder.add_conditional_edges(
        "data_audit",
        _continue_or_report,
        {"continue": "experiment_planning", "report": "report"},
    )
    builder.add_conditional_edges(
        "experiment_planning",
        _continue_or_report,
        {"continue": "representation_design", "report": "report"},
    )
    builder.add_conditional_edges(
        "representation_design",
        _continue_or_report,
        {"continue": "probe_setup", "report": "report"},
    )
    builder.add_conditional_edges(
        "probe_setup",
        _continue_or_report,
        {"continue": "controlled_probe", "report": "report"},
    )
    builder.add_conditional_edges(
        "controlled_probe",
        _continue_or_report,
        {"continue": "scout", "report": "report"},
    )
    builder.add_conditional_edges(
        "scout",
        _scout_route,
        {
            "continue": "controlled_representation",
            "repair": "probe_setup",
            "report": "report",
        },
    )
    builder.add_conditional_edges(
        "controlled_representation",
        _continue_or_report,
        {"continue": "scientific_critic", "report": "report"},
    )
    builder.add_conditional_edges(
        "scientific_critic",
        _post_scout_route,
        {
            "feature": "feature_planning",
            "split": "split_selection",
            "acceptance": "acceptance_selection",
            "final_test": "final_test",
            "report": "report",
        },
    )
    builder.add_conditional_edges(
        "split_selection",
        _continue_or_report,
        {"continue": "final_test", "report": "report"},
    )
    builder.add_edge("acceptance_selection", "report")
    builder.add_conditional_edges(
        "feature_planning",
        _feature_route,
        {
            "reuse": "feature_qc",
            "compute": "feature_tool",
            "finalize": "final_test",
            "complete": "report",
            "report": "report",
        },
    )
    builder.add_edge("feature_tool", "report")
    builder.add_conditional_edges(
        "feature_qc",
        _continue_or_report,
        {"continue": "filtration_audit", "report": "report"},
    )
    builder.add_conditional_edges(
        "filtration_audit",
        _continue_or_report,
        {"continue": "evaluation", "report": "report"},
    )
    builder.add_edge("evaluation", "report")
    builder.add_edge("final_test", "report")
    builder.add_edge("report", END)
    return builder.compile(checkpointer=checkpointer)


def _continue_or_report(state: MintAgentState) -> Literal["continue", "report"]:
    return "continue" if state.get("route") == "continue" else "report"


def _scout_route(state: MintAgentState) -> Literal["continue", "repair", "report"]:
    if state.get("route") == "repair":
        return "repair"
    return _continue_or_report(state)


def _feature_route(
    state: MintAgentState,
) -> Literal["reuse", "compute", "finalize", "complete", "report"]:
    route = state.get("route")
    if (
        route == "complete"
        and state.get("task", {}).get("evaluation_mode")
        == EvaluationMode.EXPLICIT_LABELED_TEST.value
    ):
        return "finalize"
    return route if route in {"reuse", "compute", "complete"} else "report"


def _post_scout_route(
    state: MintAgentState,
) -> Literal["feature", "split", "acceptance", "final_test", "report"]:
    if state.get("route") != "continue":
        return "report"
    if (
        state.get("task", {}).get("evaluation_mode")
        == EvaluationMode.EXPLICIT_VALIDATION_AND_TEST.value
    ):
        return "split"
    if (
        state.get("task", {}).get("evaluation_mode")
        == EvaluationMode.EXPLICIT_LABELED_TEST.value
    ):
        if _test_evaluation_policy(state) == "test_acceptance":
            return "acceptance"
        return "final_test"
    return "feature"


def _full_train_cv_used(state: Mapping[str, Any]) -> bool:
    evaluation = state.get("evaluation")
    if isinstance(evaluation, Mapping):
        return evaluation.get("evidence_scope") == "full_train"
    return False


def _test_evaluation_policy(state: Mapping[str, Any]) -> str:
    preferences = state.get("task", {}).get("selection_preferences", {})
    if not isinstance(preferences, Mapping):
        return "frozen_once"
    return str(preferences.get("test_evaluation_policy") or "frozen_once")


def _materialize_probe_rank_selection(
    *,
    state: Mapping[str, Any],
    task: Mapping[str, Any],
    registry: ArtifactRegistry,
) -> dict[str, Any]:
    scout = state.get("scout")
    if not isinstance(scout, Mapping) or not scout.get("path"):
        raise ValueError("a Scout artifact is required")
    priority_order = _scout_priority_order(dict(scout))
    if not priority_order:
        raise ValueError("Scout artifact has no frozen priority order")
    selected_subset = tuple(priority_order[0])
    metadata = scout.get("metadata")
    metadata = metadata if isinstance(metadata, Mapping) else {}
    target_metric = str(metadata.get("target_metric") or task.get("primary_metric") or "PCC").upper()
    target_value = float(metadata.get("target_value", 0.0))
    selection_objective = str(
        task.get("selection_preferences", {}).get("selection_objective")
        if isinstance(task.get("selection_preferences"), Mapping)
        else "satisfy_target"
    )
    if not selection_objective:
        selection_objective = "satisfy_target"
    output_root = Path(str(scout["path"])).expanduser().resolve().parent
    destination = output_root / "probe_rank_selections" / (
        f"rank1-{str(scout.get('content_sha256') or '')[:12] or 'scout'}-"
        f"{'-'.join(selected_subset)}.json"
    )
    payload: dict[str, Any] = {
        "report_schema": "mint-agent.probe-rank-selection.v1",
        "status": "MAXIMIZATION_COMPLETE",
        "dataset_id": str(task.get("task_id") or state.get("task", {}).get("dataset_id") or ""),
        "evidence_scope": "probe",
        "selection_source": "frozen_probe_rank1",
        "selection_objective": selection_objective,
        "selected_subset": list(selected_subset),
        "invariants": list(selected_subset),
        "representation_hash": state["representation_hash"],
        "scout_artifact": str(scout["path"]),
        "scout_content_sha256": scout.get("content_sha256"),
        "ranking_policy": metadata.get("ranking_policy") or "hierarchical_empirical_v1",
        "gbt_parameter_hash": metadata.get("gbt_parameter_hash"),
        "target_metric": target_metric,
        "target_value": target_value,
        "target_source": metadata.get("target_source") or "probe_derived",
        "test_used_for_selection": False,
        "protocol": {
            "selection_evidence_scope": "probe",
            "selection_frozen_before_test_feature_loading": True,
            "test_used_for_selection": False,
            "test_query_budget": 1,
        },
    }
    payload["selection_hash"] = stable_hash(payload)
    _write_json_if_same(payload, destination)
    return _record_summary(_register_json_artifact(destination, registry=registry))


def _preparation_modules(config: Mapping[str, Any]) -> tuple[str, ...]:
    values = config.get("module_load", [])
    if not isinstance(values, list) or not all(
        isinstance(value, str) and value for value in values
    ):
        raise ValueError(
            "dataset_preparation.module_load must be a list of module names"
        )
    return tuple(values)


def _failure(message: str, node: str) -> dict[str, Any]:
    return {
        "route": "blocked",
        "status": "BLOCKED",
        "errors": [message],
        "node_trace": [node],
    }


def _llm_stage_used(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    llm = value.get("llm")
    return bool(isinstance(llm, Mapping) and llm.get("used", False))


def _qc_outlier_invariants(
    path: str | Path, requested: list[str]
) -> tuple[tuple[str, ...], str | None]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return (), None
    if (
        not isinstance(payload, Mapping)
        or payload.get("report_schema") != "mint-agent.feature-qc.v1"
    ):
        raise ValueError(f"Unsupported feature-QC report for diagnostics: {path}")
    universe = {str(value).upper() for value in requested}
    invariants = {
        str(issue.get("invariant", "")).upper()
        for issue in payload.get("issues", ())
        if isinstance(issue, Mapping)
        and issue.get("sample_id") is not None
        and issue.get("issue") == "robust_outlier"
    }
    return (
        tuple(sorted(invariants & universe)),
        (str(payload["qc_hash"]) if payload.get("qc_hash") is not None else None),
    )


def _feature_compute_action(missing: list[str], has_job_plan: bool) -> str:
    suffix = ": " + ", ".join(missing)
    if has_job_plan:
        return "Review and submit the generated Slurm feature job plan" + suffix
    return "Submit isolated registered feature tools for" + suffix


def _artifact_by_path(
    artifacts: list[dict[str, Any]], path: str | Path
) -> dict[str, Any] | None:
    target = _path_key(path)
    for artifact in artifacts:
        artifact_path = artifact.get("path")
        if artifact_path is not None and _path_key(artifact_path) == target:
            return artifact
    return None


def _register_json_artifact(
    path: str | Path, *, registry: ArtifactRegistry, dataset_id: str | None = None
) -> ArtifactRecord:
    record = inspect_artifact(path)
    if dataset_id is not None:
        record = replace(record, dataset_id=dataset_id)
    registry.register(record)
    return record


def _existing_filtration_repair_is_compatible(
    payload: Mapping[str, Any],
    *,
    parent_representation_hash: str,
    repaired_representation_hash: str,
) -> bool:
    if payload.get("report_schema") != "mint-agent.filtration-repair.v1":
        return False
    return (
        str(payload.get("parent_representation_hash") or "")
        == parent_representation_hash
        and str(payload.get("representation_hash") or "")
        == repaired_representation_hash
        and str(payload.get("audit_representation_hash") or "")
        == parent_representation_hash
    )


def _controlled_experiment_output_dir(
    run_root: str | Path,
    *,
    dataset_id: str,
    matrix: Mapping[str, Any],
    representation_hash: object | None,
) -> Path:
    base = (
        Path(run_root)
        / "controlled_experiments"
        / dataset_id
        / str(matrix["experiment_hash"])
    )
    if representation_hash is None:
        return base
    representation_key = str(representation_hash).strip()
    if not representation_key:
        return base
    return base / representation_key[:16]


def _write_json_if_same(payload: Mapping[str, Any], path: str | Path) -> None:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if destination.is_file():
        existing = destination.read_text(encoding="utf-8")
        if existing == content:
            return
        raise ValueError(
            f"refusing to overwrite existing graph artifact: {destination}"
        )
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(destination)


def _probe_policy_from_matrix(matrix: Mapping[str, Any]) -> ProbeComparisonPolicy:
    policy = matrix.get("probe_comparison_policy")
    if not isinstance(policy, Mapping):
        return ProbeComparisonPolicy()
    thresholds = policy.get("thresholds")
    return ProbeComparisonPolicy.from_mapping(
        thresholds if isinstance(thresholds, Mapping) else None
    )


def _representation_policy_from_matrix(
    matrix: Mapping[str, Any],
) -> RepresentationComparisonPolicy:
    policy = matrix.get("representation_comparison_policy")
    return RepresentationComparisonPolicy.from_mapping(
        policy if isinstance(policy, Mapping) else None
    )


def _path_key(path: str | Path) -> str:
    value = Path(str(path)).expanduser()
    try:
        return str(value.resolve())
    except OSError:
        return str(value)


def _record_summary(record: ArtifactRecord) -> dict[str, Any]:
    return {
        "artifact_id": record.artifact_id,
        "kind": record.artifact_kind,
        "path": record.path,
        "content_sha256": record.content_sha256,
        "dataset_id": record.dataset_id,
        "split": record.split,
        "evidence_scope": record.evidence_scope,
        "invariants": list(record.invariants),
        "representation_hash": record.representation_hash,
        "selection_hash": record.selection_hash,
        "sample_count": record.sample_count,
        "sample_order_hash": record.sample_order_hash,
        "status": record.status,
        "report_schema": record.report_schema,
        "metadata": dict(record.metadata),
    }


def _filtration_repair_config(task: Mapping[str, Any]) -> FiltrationRepairConfig:
    design = task.get("representation_design", {})
    if not isinstance(design, Mapping):
        return FiltrationRepairConfig()
    repair = design.get("filtration_repair")
    return FiltrationRepairConfig.from_mapping(
        repair if isinstance(repair, Mapping) else None
    )


def _latest_repair_in_lineage(
    artifacts: list[dict[str, Any]],
    baseline: Mapping[str, Any],
    *,
    config: FiltrationRepairConfig,
) -> dict[str, Any] | None:
    """Return the newest bounded repair descendant of a base design artifact."""

    if not config.enabled or config.max_rounds == 0:
        return None
    current = dict(baseline)
    current_hash = str(current.get("representation_hash") or "")
    for _ in range(config.max_rounds):
        candidates = []
        for artifact in artifacts:
            if artifact.get("kind") != "filtration_repair":
                continue
            if artifact.get("status") != "COMPLETE":
                continue
            metadata = artifact.get("metadata")
            if not isinstance(metadata, Mapping):
                continue
            if str(metadata.get("parent_representation_hash") or "") != current_hash:
                continue
            candidates.append(artifact)
        if not candidates:
            break
        current = max(
            candidates,
            key=lambda item: (
                int(item.get("metadata", {}).get("repair_round", 0)),
                str(item.get("path") or ""),
            ),
        )
        current_hash = str(current.get("representation_hash") or "")
    return current if current is not baseline and current != baseline else None


def _resume_representation_from_downstream(
    artifacts: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Use an already evidenced representation when resuming a reviewed run."""

    downstream_kinds = {
        "probe_selection",
        "probe_audit",
        "feature_manifest",
        "feature_qc",
        "filtration_audit",
        "scout_oof",
        "scout_combine",
    }
    downstream_hashes = {
        str(artifact.get("representation_hash"))
        for artifact in artifacts
        if artifact.get("kind") in downstream_kinds
        and artifact.get("representation_hash")
    }
    if not downstream_hashes:
        return None
    candidates = [
        artifact
        for artifact in artifacts
        if artifact.get("kind") in {"representation_design", "filtration_repair"}
        and str(artifact.get("representation_hash")) in downstream_hashes
    ]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda artifact: (
            artifact.get("kind") == "filtration_repair",
            str(artifact.get("path") or ""),
        ),
    )


def _materialize_filtration_repair(
    *,
    representation_spec_path: str | Path | None,
    representation_hash: str,
    audit: Mapping[str, Any],
    task: Mapping[str, Any],
    task_id: str,
    requested_invariants: Sequence[str],
    registry: ArtifactRegistry,
) -> dict[str, Any] | None:
    config = _filtration_repair_config(task)
    if not config.enabled or representation_spec_path is None:
        return None
    full_audit = _full_filtration_audit_report(audit)
    try:
        representation = RepresentationSpec.read(representation_spec_path)
        if representation.spec_hash != representation_hash:
            raise ValueError("representation path and active representation hash disagree")
        result = repair_filtration_profiles(
            representation,
            full_audit,
            requested_invariants=requested_invariants,
            config=config,
        )
    except (OSError, TypeError, ValueError) as exc:
        raise ValueError(f"filtration repair could not be materialized: {exc}") from exc
    if not result.changed:
        return None
    audit_path = audit.get("path")
    output_parent = (
        Path(str(audit_path)).expanduser().resolve().parent
        if audit_path
        else Path(representation_spec_path).expanduser().resolve().parent
    )
    repair_path = output_parent / "filtration_repairs" / (
        f"repair-{representation.spec_hash[:12]}-"
        f"{result.representation_spec.spec_hash[:12]}.json"
    )
    report = result.report(
        task_id=task_id,
        audit=full_audit,
        requested_invariants=requested_invariants,
    )
    report["filtration_audit_path"] = full_audit.get("path") or audit.get("path")
    report["filtration_audit_content_sha256"] = audit.get("content_sha256")
    if repair_path.is_file():
        try:
            existing_payload = json.loads(repair_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"existing filtration repair artifact is not valid JSON: {repair_path}"
            ) from exc
        if not isinstance(existing_payload, Mapping) or not _existing_filtration_repair_is_compatible(
            existing_payload,
            parent_representation_hash=representation.spec_hash,
            repaired_representation_hash=result.representation_spec.spec_hash,
        ):
            raise ValueError(
                f"refusing to reuse incompatible filtration repair artifact: {repair_path}"
            )
    else:
        _write_json_if_same(report, repair_path)
    return _record_summary(
        _register_json_artifact(repair_path, registry=registry, dataset_id=task_id)
    )


def _full_filtration_audit_report(audit: Mapping[str, Any]) -> Mapping[str, Any]:
    """Load the full audit report when the graph only has an artifact summary."""

    invariants = audit.get("invariants")
    if isinstance(invariants, Mapping):
        return audit
    path = audit.get("path")
    if not path:
        return audit
    with Path(str(path)).expanduser().open("r", encoding="utf-8") as handle:
        full_audit = json.load(handle)
    if not isinstance(full_audit, Mapping):
        raise ValueError("filtration audit report must be a mapping")
    full_audit = dict(full_audit)
    full_audit.setdefault("path", str(path))
    return full_audit


def _latest(
    artifacts: list[dict[str, Any]],
    *,
    kind: str,
    invariant: str | None = None,
    invariants: list[str] | None = None,
    evidence_scope: str | None = None,
    representation_hash: str | None = None,
    status: str | None = None,
    selection_hash: str | None = None,
    exact_invariants: bool = False,
    target_value: float | None = None,
    gbt_parameter_hash: str | None = None,
    ranking_policy: str | None = None,
) -> dict[str, Any] | None:
    required = {value.upper() for value in invariants or []}
    matches = []
    for artifact in artifacts:
        if artifact.get("kind") != kind:
            continue
        names = {str(value).upper() for value in artifact.get("invariants", [])}
        if invariant is not None and invariant.upper() not in names:
            continue
        if required and not required.issubset(names):
            continue
        if exact_invariants and names != required:
            continue
        if (
            evidence_scope is not None
            and artifact.get("evidence_scope") != evidence_scope
        ):
            continue
        if (
            representation_hash is not None
            and artifact.get("representation_hash") != representation_hash
        ):
            continue
        if status is not None and artifact.get("status") != status:
            continue
        if (
            selection_hash is not None
            and artifact.get("selection_hash") != selection_hash
        ):
            continue
        if target_value is not None:
            metadata = artifact.get("metadata")
            try:
                artifact_target = float(metadata["target_value"])
            except (KeyError, TypeError, ValueError):
                continue
            if abs(artifact_target - target_value) > 1.0e-12:
                continue
        if gbt_parameter_hash is not None:
            metadata = artifact.get("metadata")
            if not isinstance(metadata, Mapping) or str(
                metadata.get("gbt_parameter_hash") or ""
            ) != str(gbt_parameter_hash):
                continue
        if ranking_policy is not None:
            metadata = artifact.get("metadata")
            if not isinstance(metadata, Mapping) or str(
                metadata.get("ranking_policy") or ""
            ) != str(ranking_policy):
                continue
        matches.append(artifact)
    return matches[-1] if matches else None


def _same_sample_axis(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_count = left.get("sample_count")
    right_count = right.get("sample_count")
    left_hash = left.get("sample_order_hash")
    right_hash = right.get("sample_order_hash")
    return (
        left_count is not None
        and right_count is not None
        and int(left_count) == int(right_count)
        and bool(left_hash)
        and left_hash == right_hash
    )


def _same_evaluation_contract(
    scout: dict[str, Any], evaluation: dict[str, Any]
) -> bool:
    if not _same_sample_axis(scout, evaluation):
        return False
    if scout.get("representation_hash") != evaluation.get("representation_hash"):
        return False
    scout_metadata = scout.get("metadata")
    evaluation_metadata = evaluation.get("metadata")
    if not isinstance(scout_metadata, dict) or not isinstance(
        evaluation_metadata, dict
    ):
        return False
    for key in ("gbt_parameter_hash", "target_metric", "target_value"):
        if key not in scout_metadata or key not in evaluation_metadata:
            return False
    scout_fold_hash = scout_metadata.get("fold_assignment_hash")
    evaluation_fold_hash = evaluation_metadata.get("fold_assignment_hash")
    if scout_fold_hash and evaluation_fold_hash:
        same_folds = scout_fold_hash == evaluation_fold_hash
    else:
        same_folds = (
            evaluation_metadata.get("scout_artifact") == scout.get("path")
            and evaluation_metadata.get("fold_source") == "scout_artifact"
        )
    try:
        same_target = float(scout_metadata["target_value"]) == float(
            evaluation_metadata["target_value"]
        )
    except (TypeError, ValueError):
        return False
    return (
        scout_metadata["gbt_parameter_hash"]
        == evaluation_metadata["gbt_parameter_hash"]
        and str(scout_metadata["target_metric"]).upper()
        == str(evaluation_metadata["target_metric"]).upper()
        and same_folds
        and same_target
    )


def _same_split_selection_contract(
    scout: dict[str, Any], evaluation: dict[str, Any]
) -> bool:
    if scout.get("representation_hash") != evaluation.get("representation_hash"):
        return False
    scout_metadata = scout.get("metadata")
    evaluation_metadata = evaluation.get("metadata")
    if not isinstance(scout_metadata, Mapping) or not isinstance(
        evaluation_metadata, Mapping
    ):
        return False
    try:
        target_matches = float(scout_metadata["target_value"]) == float(
            evaluation_metadata["target_value"]
        )
    except (KeyError, TypeError, ValueError):
        return False
    return (
        evaluation_metadata.get("scout_artifact") == scout.get("path")
        and evaluation_metadata.get("gbt_parameter_hash")
        == scout_metadata.get("gbt_parameter_hash")
        and str(evaluation_metadata.get("target_metric") or "").upper()
        == str(scout_metadata.get("target_metric") or "").upper()
        and target_matches
    )


def _selected_subset_from_artifact(artifact: Mapping[str, Any]) -> tuple[str, ...]:
    metadata = artifact.get("metadata")
    raw = metadata.get("selected_subset") if isinstance(metadata, Mapping) else None
    if raw is None and artifact.get("path"):
        try:
            payload = json.loads(
                Path(str(artifact["path"])).read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError):
            return ()
        selection = payload.get("selection") if isinstance(payload, Mapping) else None
        result = payload.get("result") if isinstance(payload, Mapping) else None
        raw = payload.get("selected_subset") if isinstance(payload, Mapping) else None
        if raw is None and isinstance(selection, Mapping):
            raw = selection.get("selected_subset")
        if raw is None and isinstance(result, Mapping):
            raw = result.get("selected_subset")
    if not isinstance(raw, (list, tuple)) or not raw:
        return ()
    subset = tuple(str(value).upper() for value in raw)
    if len(set(subset)) != len(subset):
        return ()
    return subset


def _scout_priority_order(scout: dict[str, Any]) -> tuple[tuple[str, ...], ...]:
    metadata = scout.get("metadata")
    raw_priority = (
        metadata.get("frozen_priority_order") if isinstance(metadata, dict) else None
    )
    if raw_priority is None:
        try:
            raw_priority = ScoutExecutionArtifact.read(
                scout["path"]
            ).frozen_priority_order
        except (KeyError, OSError, ValueError, TypeError):
            return ()
    if not isinstance(raw_priority, (list, tuple)):
        return ()
    priority: list[tuple[str, ...]] = []
    for raw_subset in raw_priority:
        if not isinstance(raw_subset, (list, tuple)):
            return ()
        subset = tuple(str(value).upper() for value in raw_subset)
        if not subset or len(set(subset)) != len(subset):
            return ()
        priority.append(subset)
    if len(set(priority)) != len(priority):
        return ()
    return tuple(priority)


def _flatten_priority_order(
    priority_order: tuple[tuple[str, ...], ...]
) -> tuple[str, ...]:
    acquisition_order: list[str] = []
    for subset in priority_order:
        for invariant in subset:
            if invariant not in acquisition_order:
                acquisition_order.append(invariant)
    return tuple(acquisition_order)


def _candidate_rank_limit_for_objective(
    selection_objective: str,
    preferences: Mapping[str, Any],
    priority_order: tuple[tuple[str, ...], ...],
) -> int | None:
    if selection_objective != "maximize_top_k":
        return None
    raw_limit = preferences.get("max_candidate_rank", 5)
    try:
        limit = int(raw_limit)
    except (TypeError, ValueError) as exc:
        raise ValueError("max_candidate_rank must be a positive integer") from exc
    if limit < 1:
        raise ValueError("max_candidate_rank must be a positive integer")
    return min(limit, len(priority_order))


def _stages_for_progressive_objective(
    priority_order: tuple[tuple[str, ...], ...],
    *,
    selection_objective: str,
    preferences: Mapping[str, Any],
) -> tuple[tuple[str, ...], ...]:
    acquisition_order = _flatten_priority_order(priority_order)
    if selection_objective == "maximize_rank1":
        rank1_size = len(tuple(dict.fromkeys(priority_order[0])))
        return (acquisition_order[:rank1_size],)
    if selection_objective == "maximize_top_k":
        candidate_rank_limit = _candidate_rank_limit_for_objective(
            selection_objective, preferences, priority_order
        )
        assert candidate_rank_limit is not None
        return (_flatten_priority_order(priority_order[:candidate_rank_limit]),)
    if selection_objective in {"satisfy_target", "maximize_all"}:
        return tuple(
            acquisition_order[:index] for index in range(1, len(acquisition_order) + 1)
        )
    raise ValueError("Unsupported selection objective")


def _parallel_acquisition_objective(selection_objective: str) -> bool:
    return selection_objective in {"maximize_rank1", "maximize_top_k"}
    build_dataset_audit_job_plan,
