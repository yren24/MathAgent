from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from mint_scout.artifact_memory import ArtifactRegistry, inspect_artifact
from mint_scout.config import load_yaml
from mint_scout.data.element_pairs import SupportThresholds
from mint_scout.data.hiqbind import (
    HiQBindSubsetConfig,
    SUPPORTED_MEASUREMENTS,
    SUPPORTED_OUTPUT_SPLITS,
    SUPPORTED_SIGNS,
)
from mint_scout.data.manifest_io import task_card_from_config
from mint_scout.execution.profile import load_execution_profile
from mint_scout.filtration import FiltrationConfig
from mint_scout.invariants.manifest import stable_hash
from mint_scout.openai_responses import request_structured_output
from mint_scout.run_pipeline import PipelineLaunchPlan
from mint_scout.sample_size import LabeledSamplePolicy
from mint_scout.scout.pipeline import ScoutConfig
from mint_scout.schemas import (
    EvaluationMode,
    EvaluationPlan,
    SampleRecord,
    route_evaluation_mode,
)


REQUEST_VERSION = "mint-agent.user-request.v1"
PREFLIGHT_SCHEMA = "mint-agent.user-preflight.v1"
REPRESENTATION_PROFILE_VERSION = "mint-agent.representation-profile.v1"
DEFAULT_REPRESENTATION_PROFILE = (
    "configs/representation/protein_ligand_adaptive_v1.yaml"
)
SUPPORTED_INVARIANTS = ("PH", "PL", "CA", "FPRC", "EIC")
BLOCKED_EVALUATION_MODES = {
    EvaluationMode.LABEL_RETRIEVAL_REQUIRED,
    EvaluationMode.INSUFFICIENT_LABELS,
    EvaluationMode.NEEDS_USER_INPUT,
    EvaluationMode.UNSUPPORTED_TASK,
}


@dataclass(frozen=True)
class UserRunBundle:
    status: str
    request_id: str
    request_hash: str
    run_id: str
    pipeline_kind: str
    task_config: str
    pipeline_config: str
    preflight_report: str
    launch_plan: PipelineLaunchPlan
    evaluation_mode: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "launch_plan": self.launch_plan.to_dict(),
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate a user request and launch the deterministic mint-agent pipeline."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--request", type=Path)
    source.add_argument("--prompt")
    source.add_argument("--prompt-file", type=Path)
    parser.add_argument("--model", default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--execution-profile", default=None)
    parser.add_argument("--representation-profile", default=None)
    parser.add_argument("--manifest-path", default=None)
    parser.add_argument("--manifest-profile-max-rows", type=int, default=200)
    parser.add_argument(
        "--llm-scientific-mode",
        choices=("disabled", "shadow", "advisory"),
        default="disabled",
    )
    parser.add_argument("--llm-scientific-model", default=None)
    parser.add_argument("--llm-scientific-env-file", default=None)
    parser.add_argument("--llm-scientific-cache-dir", default=None)
    parser.add_argument("--llm-scientific-max-candidates", type=int, default=10)
    parser.add_argument("--llm-scientific-timeout-seconds", type=float, default=180.0)
    args = parser.parse_args(argv)

    if args.request is not None:
        bundle = prepare_user_request(args.request, output_dir=args.output_dir)
    else:
        prompt = args.prompt
        if args.prompt_file is not None:
            prompt = args.prompt_file.read_text(encoding="utf-8")
        model = args.model or os.environ.get("OPENAI_MODEL")
        if not model:
            raise ValueError("Natural-language intake requires --model or OPENAI_MODEL")
        api_key = os.environ.get(args.api_key_env)
        if not api_key:
            raise ValueError(
                f"Natural-language intake requires API credentials in {args.api_key_env}"
            )
        from mint_scout.agent.llm_intake import LLMIntakeContext, run_llm_intake

        llm_scientific = None
        if args.llm_scientific_mode != "disabled":
            llm_scientific = {
                "mode": args.llm_scientific_mode,
                "model": args.llm_scientific_model or model,
                "api_key_env": args.api_key_env,
                "env_file": args.llm_scientific_env_file,
                "cache_dir": args.llm_scientific_cache_dir,
                "max_candidates": args.llm_scientific_max_candidates,
                "timeout_seconds": args.llm_scientific_timeout_seconds,
            }
            llm_scientific = {
                key: value for key, value in llm_scientific.items() if value is not None
            }

        result = run_llm_intake(
            str(prompt),
            context=LLMIntakeContext(
                model=model,
                api_key=api_key,
                base_dir=Path.cwd(),
                output_dir=args.output_dir,
                execute=args.execute,
                execution_profile=args.execution_profile,
                representation_profile=args.representation_profile,
                manifest_path=args.manifest_path,
                llm_scientific=llm_scientific,
                manifest_profile_max_rows=args.manifest_profile_max_rows,
            ),
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return (
            0
            if result.get("status")
            not in {"LLM_ERROR", "INVALID_REQUEST", "SUBMISSION_ERROR",}
            else 2
        )

    result: dict[str, Any] = bundle.to_dict()
    if args.execute and bundle.status in {"READY", "READY_FOR_PREPARATION"}:
        from mint_scout.agent.workflow import submit_prepared_bundle

        result["launch_receipt"] = submit_prepared_bundle(result)
        result["status"] = "SUBMITTED"
    elif args.execute:
        result["submission_blocked"] = True
        result["submission_block_reason"] = bundle.status
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def prepare_user_request(
    path: str | Path, *, output_dir: str | Path | None = None
) -> UserRunBundle:
    request_path = Path(path).expanduser().resolve()
    return prepare_user_mapping(
        load_yaml(request_path),
        base_dir=request_path.parent,
        source=str(request_path),
        output_dir=output_dir,
        llm_provenance={"used": False, "used_for_numeric_decisions": False},
    )


def prepare_user_mapping(
    raw: Mapping[str, Any],
    *,
    base_dir: str | Path,
    source: str,
    output_dir: str | Path | None = None,
    llm_provenance: Mapping[str, Any] | None = None,
    dataset_onboarding: Mapping[str, Any] | None = None,
) -> UserRunBundle:
    request = dict(raw)
    llm = _normalize_llm_provenance(llm_provenance)
    onboarding = _normalize_dataset_onboarding(dataset_onboarding)
    _reject_unknown(
        request,
        {
            "version",
            "request_id",
            "goal",
            "system_type",
            "task_type",
            "primary_metric",
            "execution_profile",
            "representation_profile",
            "dataset",
            "run",
            "labels",
            "label_retrieval_allowed",
            "hydrogen_policy",
            "llm_scientific",
        },
        "user request",
    )
    if request.get("version") != REQUEST_VERSION:
        raise ValueError(f"user request version must be {REQUEST_VERSION!r}")
    request_id = _identifier(request.get("request_id"), "request_id")
    goal = _required_text(request.get("goal"), "goal")
    system_type = _required_text(request.get("system_type"), "system_type")
    task_type = _required_text(request.get("task_type"), "task_type")
    primary_metric = _normalize_primary_metric(
        request.get("primary_metric", "PCC2" if system_type == "small_molecule" else "PCC")
    )
    if system_type != "protein_ligand":
        if system_type == "small_molecule":
            return _prepare_toxicity_user_mapping(
                request,
                base_dir=base_dir,
                source=source,
                output_dir=output_dir,
                llm=llm,
                onboarding=onboarding,
                request_id=request_id,
                goal=goal,
                task_type=task_type,
                primary_metric=primary_metric,
            )
        raise ValueError(
            "V1 execution supports system_type=protein_ligand or "
            "system_type=small_molecule for the toxicity adapter"
        )
    if task_type != "regression":
        raise ValueError("V1 feature selection supports task_type=regression")
    if primary_metric != "PCC":
        raise ValueError("V1 stopping and ranking require primary_metric=PCC")

    root = Path(base_dir).expanduser().resolve()
    profile_path = _resolve_path(
        request.get("execution_profile"), root, "execution_profile"
    )
    if not profile_path.is_file():
        raise FileNotFoundError(f"execution profile does not exist: {profile_path}")
    profile = load_execution_profile(profile_path)
    if not profile.supports_slurm:
        raise ValueError("V1 user start requires a Slurm execution profile")
    if not profile.plbind_root:
        raise ValueError("execution profile tools.plbind_root is required")

    representation_profile_path = _representation_profile_path(
        request.get("representation_profile"),
        base_dir=root,
        project_root=Path(profile.project_root),
    )
    representation_profile = _load_representation_profile(representation_profile_path)
    representation_profile_hash = stable_hash(representation_profile)

    dataset = _mapping(request.get("dataset"), "dataset")
    _reject_unknown(
        dataset,
        {
            "manifest_path",
            "label_name",
            "validate_paths",
            "columns",
            "source",
            "preparation",
            "cv_group_identifier",
        },
        "dataset",
    )
    manifest_path = _resolve_path(
        dataset.get("manifest_path"), root, "dataset.manifest_path"
    )
    validate_paths = dataset.get("validate_paths", True)
    if not isinstance(validate_paths, bool):
        raise ValueError("dataset.validate_paths must be true or false")
    columns = _normalize_columns(dataset.get("columns"), system_type=system_type)
    cv_group_identifier = _optional_text(dataset.get("cv_group_identifier"))
    configured_identifiers = set(columns.get("identifiers", {}))
    if (
        cv_group_identifier is not None
        and cv_group_identifier not in configured_identifiers
    ):
        raise ValueError(
            "dataset.cv_group_identifier must name a configured dataset.columns "
            "identifier"
        )
    preparation = _normalize_preparation(
        dataset.get("preparation"), base_dir=root, manifest_path=manifest_path,
    )
    if not manifest_path.is_file() and preparation is None:
        raise FileNotFoundError(
            "dataset manifest does not exist and dataset.preparation is not configured: "
            f"{manifest_path}"
        )

    run = _mapping(request.get("run", {}), "run")
    _reject_unknown(
        run,
        {
            "run_id",
            "invariants",
            "user_target",
            "selection_objective",
            "max_candidate_rank",
            "test_evaluation_policy",
            "qc_config",
            "audit_config",
            "scout_config",
            "gbt_config",
            "feature_diagnostic_top_k",
            "auto_resume_failed",
            "max_attempts",
            "stop_after_stage",
            "test_evaluation_policy",
        },
        "run",
    )
    invariants_explicit = "invariants" in run and run.get("invariants") is not None
    invariants = _invariants(run.get("invariants", list(SUPPORTED_INVARIANTS)))
    user_target = _optional_target(run.get("user_target"))
    selection_objective = str(run.get("selection_objective") or "satisfy_target")
    if selection_objective not in {
        "satisfy_target",
        "maximize_rank1",
        "maximize_top_k",
        "maximize_all",
    }:
        raise ValueError(
            "run.selection_objective must be satisfy_target, maximize_rank1, "
            "maximize_top_k, or maximize_all"
        )
    max_candidate_rank = (
        _positive_int(run["max_candidate_rank"], "run.max_candidate_rank")
        if run.get("max_candidate_rank") is not None
        else None
    )
    if selection_objective == "maximize_top_k" and max_candidate_rank is None:
        max_candidate_rank = 5
    test_evaluation_policy = str(run.get("test_evaluation_policy") or "frozen_once")
    if test_evaluation_policy not in {"frozen_once", "test_acceptance"}:
        raise ValueError(
            "run.test_evaluation_policy must be frozen_once or test_acceptance"
        )
    target_source = "user" if user_target is not None else "probe_derived"
    method_source = "user" if invariants_explicit else "default_all_supported"
    hydrogen_request = _normalize_hydrogen_request(request.get("hydrogen_policy"))
    request_hash = stable_hash(
        _canonical_request(
            request,
            profile_path,
            manifest_path,
            representation_profile_path,
            representation_profile_hash,
        )
    )
    run_id = _optional_identifier(run.get("run_id")) or _identifier(
        f"{request_id}-{request_hash[:8]}", "run.run_id"
    )
    feature_diagnostic_top_k = _positive_int(
        run.get("feature_diagnostic_top_k", 10), "run.feature_diagnostic_top_k"
    )
    max_attempts = _positive_int(run.get("max_attempts", 2), "run.max_attempts")
    auto_resume_failed = run.get("auto_resume_failed", False)
    if not isinstance(auto_resume_failed, bool):
        raise ValueError("run.auto_resume_failed must be true or false")
    llm_scientific = _normalize_llm_scientific_config(request.get("llm_scientific"))
    stop_after_stage = (
        _nonnegative_int(run["stop_after_stage"], "run.stop_after_stage")
        if run.get("stop_after_stage") is not None
        else None
    )
    scout_config_path = _profile_path(
        profile.project_root, run.get("scout_config", "configs/scout/v1.yaml")
    )
    configured_label_policy = ScoutConfig.from_mapping(
        load_yaml(scout_config_path)
    ).labels
    label_overrides = _mapping(request.get("labels", {}), "labels")
    label_policy = LabeledSamplePolicy.from_mapping(
        {**configured_label_policy.to_dict(), **label_overrides}
    )

    destination = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else Path(profile.run_root) / "intake" / request_id / run_id
    )
    task_path = destination / "task.yaml"
    pipeline_path = destination / "pipeline.yaml"
    preflight_path = destination / "preflight.json"
    resolved_request_path = destination / "request.resolved.yaml"

    task_payload: dict[str, Any] = {
        "task_id": request_id,
        "project_name": "mint-agent",
        "research_goal": goal,
        "intent": {
            "source": (
                "openai_responses_api" if llm.get("used") else "structured_request"
            ),
            "llm": llm,
            **({"dataset_onboarding": onboarding} if onboarding else {}),
        },
        "system_type": system_type,
        "task_type": task_type,
        "primary_metric": primary_metric.lower(),
        "dataset_manifest": {
            "path": str(manifest_path),
            "label_name": _optional_text(dataset.get("label_name")),
            "source": _optional_text(dataset.get("source")) or source,
            "validate_paths": validate_paths,
            "columns": columns,
        },
        "invariants": list(invariants),
        "selection_preferences": {
            "method_source": method_source,
            "target_source": target_source,
            "selection_objective": selection_objective,
            **(
                {"max_candidate_rank": max_candidate_rank}
                if max_candidate_rank is not None
                else {}
            ),
            "test_evaluation_policy": test_evaluation_policy,
        },
        "legacy": {"plbind_root": profile.plbind_root},
        "feature_generation": {
            "output_root": str(
                Path(profile.cache_root) / "plbind_features" / request_id
            ),
            "structure_staging_root": str(
                Path(profile.cache_root) / "plbind_structure_staging" / request_id
            ),
            "validate_output_shape": True,
        },
        "label_retrieval": {
            "allowed": bool(request.get("label_retrieval_allowed", False))
        },
        "labels": label_policy.to_dict(),
        **representation_profile,
    }
    task_payload["hydrogen_request"] = hydrogen_request
    if preparation is not None:
        task_payload["dataset_preparation"] = preparation
    if cv_group_identifier is not None:
        task_payload["cv"] = {
            **_mapping(task_payload.get("cv", {}), "cv"),
            "group_identifier": cv_group_identifier,
        }

    pipeline_payload: dict[str, Any] = {
        "version": "mint-agent.pipeline.v1",
        "repository_root": profile.project_root,
        "run_id": run_id,
        "task_config": str(task_path),
        "execution_profile": str(profile_path),
        "dataset_id": request_id,
        "invariants": list(invariants),
        "evidence_scope": "full_train",
        "qc_config": _profile_path(
            profile.project_root, run.get("qc_config", "configs/scout/v1.yaml")
        ),
        "audit_config": _profile_path(
            profile.project_root, run.get("audit_config", "configs/scout/v1.yaml")
        ),
        "scout_config": scout_config_path,
        "gbt_config": _profile_path(
            profile.project_root,
            run.get("gbt_config", "configs/gbt/plbind_adaptive_gbt.yaml"),
        ),
        "feature_diagnostic_top_k": feature_diagnostic_top_k,
        "auto_resume_failed": auto_resume_failed,
        "max_attempts": max_attempts,
        "selection_objective": selection_objective,
        **(
            {"max_candidate_rank": max_candidate_rank}
            if max_candidate_rank is not None
            else {}
        ),
        "test_evaluation_policy": test_evaluation_policy,
    }
    if user_target is not None:
        pipeline_payload["user_target"] = user_target
    if stop_after_stage is not None:
        pipeline_payload["stop_after_stage"] = stop_after_stage
    if llm_scientific:
        pipeline_payload["llm_scientific"] = llm_scientific

    evaluation_mode: str | None = None
    dataset_summary: dict[str, Any]
    sample_order_hash: str | None = None
    warnings: list[str] = []
    small_data_confirmation_required = False
    if manifest_path.is_file():
        card = task_card_from_config(task_payload, config_path=task_path)
        if card is None:
            raise ValueError("generated task did not produce a dataset manifest")
        evaluation = route_evaluation_mode(card)
        evaluation_mode = evaluation.mode.value
        warnings.extend(evaluation.warnings)
        if evaluation.mode in BLOCKED_EVALUATION_MODES:
            detail = "; ".join(evaluation.warnings) or evaluation.mode.value
            raise ValueError(f"user request cannot be executed: {detail}")
        scout = load_yaml(pipeline_payload["scout_config"])
        probe = _mapping(scout.get("probe", {}), "scout.probe")
        cv_folds = _positive_int(probe.get("cv_folds", 5), "probe.cv_folds")
        _validate_evaluation_sample_counts(evaluation, cv_folds=cv_folds)
        small_data_warning = label_policy.warning(len(evaluation.modeling_sample_ids))
        if small_data_warning is not None:
            warnings.append(small_data_warning)
            small_data_confirmation_required = (
                not label_policy.allow_small_data_override
            )
        leakage = _split_role_path_leakage(card.dataset.samples)
        if leakage:
            raise ValueError(
                "identical structure inputs occur across protected splits: "
                + "; ".join(leakage[:5])
            )
        identifier_leakage = _split_identifier_leakage(card.dataset.samples)
        if identifier_leakage:
            raise ValueError(
                "identifier groups occur across protected splits: "
                + "; ".join(identifier_leakage[:5])
            )
        identifier_groups = _identifier_group_summary(
            card.dataset.samples,
            modeling_sample_ids=set(evaluation.modeling_sample_ids),
        )
        for identifier, summary in identifier_groups.items():
            if summary["repeated_group_count"] and identifier != cv_group_identifier:
                warnings.append(
                    f"Modeling identifier {identifier!r} has "
                    f"{summary['repeated_group_count']} repeated groups; independent "
                    "sample-level CV may place related structures in different folds."
                )
        split_counts: dict[str, int] = {}
        for sample in card.dataset.samples:
            key = sample.split.value if sample.split is not None else "unspecified"
            split_counts[key] = split_counts.get(key, 0) + 1
        sample_order_hash = stable_hash(
            tuple(sample.sample_id for sample in card.dataset.samples)
        )
        dataset_summary = {
            "manifest_exists": True,
            "sample_count": len(card.dataset.samples),
            "labeled_count": len(card.dataset.labeled_samples),
            "unlabeled_count": len(card.dataset.unlabeled_samples),
            "split_counts": split_counts,
            "modeling_sample_count": len(evaluation.modeling_sample_ids),
            "validation_sample_count": len(evaluation.validation_sample_ids),
            "evaluation_sample_count": len(evaluation.evaluation_sample_ids),
            "inference_sample_count": len(evaluation.inference_sample_ids),
            "identifier_groups": identifier_groups,
            "cv_group_identifier": cv_group_identifier,
        }
    else:
        dataset_summary = {
            "manifest_exists": False,
            "preparation_provider": preparation["provider"],
            "sample_count": None,
        }
        warnings.append(
            "Manifest validation is deferred until the configured preparation job completes."
        )

    if small_data_confirmation_required:
        status = "NEEDS_SMALL_DATA_CONFIRMATION"
    else:
        status = "READY" if manifest_path.is_file() else "READY_FOR_PREPARATION"
    preflight = {
        "report_schema": PREFLIGHT_SCHEMA,
        "dataset_id": request_id,
        "evidence_scope": "design",
        "status": status,
        "request_id": request_id,
        "request_hash": request_hash,
        "run_id": run_id,
        "source": source,
        "goal": goal,
        "system_type": system_type,
        "task_type": task_type,
        "primary_metric": primary_metric,
        "user_target": user_target,
        "target_source": target_source,
        "selection_objective": selection_objective,
        **(
            {"max_candidate_rank": max_candidate_rank}
            if max_candidate_rank is not None
            else {}
        ),
        "test_evaluation_policy": test_evaluation_policy,
        "invariants": list(invariants),
        "method_source": method_source,
        "hydrogen_policy": hydrogen_request,
        "sample_count": dataset_summary["sample_count"],
        "sample_order_hash": sample_order_hash,
        "evaluation_mode": evaluation_mode,
        "dataset": {"manifest": str(manifest_path), **dataset_summary},
        "execution": {
            "profile": str(profile_path),
            "profile_id": profile.profile_id,
            "scheduler": "slurm",
            "project_root": profile.project_root,
            "run_root": profile.run_root,
        },
        "representation": {
            "profile": str(representation_profile_path),
            "profile_hash": representation_profile_hash,
            "mode": representation_profile["representation_mode"],
        },
        "generated": {
            "task_config": str(task_path),
            "pipeline_config": str(pipeline_path),
            "resolved_request": str(resolved_request_path),
        },
        "protocol_guards": {
            "llm_used_for_numeric_decisions": False,
            "user_splits_are_preserved": True,
            "test_used_for_selection": False,
            "test_evaluation_policy": test_evaluation_policy,
            "feature_tools_are_isolated": True,
            "grouped_cv_identifier": cv_group_identifier,
            "min_labeled_samples": label_policy.min_labeled_samples,
            "allow_small_data_override": label_policy.allow_small_data_override,
            "small_data_override_used": bool(
                label_policy.allow_small_data_override
                and dataset_summary.get("modeling_sample_count", 0)
                < label_policy.min_labeled_samples
            ),
            "llm_scientific_mode": llm_scientific.get("mode", "disabled"),
        },
        "llm": llm,
        **({"dataset_onboarding": onboarding} if onboarding else {}),
        "warnings": warnings,
    }
    resolved_request = {
        **request,
        "execution_profile": str(profile_path),
        "representation_profile": str(representation_profile_path),
        "dataset": {
            **dataset,
            "manifest_path": str(manifest_path),
            "columns": columns,
            "preparation": preparation,
        },
        "labels": label_policy.to_dict(),
        "hydrogen_policy": hydrogen_request,
        "run": {
            **run,
            "run_id": run_id,
            "invariants": list(invariants),
            "user_target": user_target,
            "selection_objective": selection_objective,
            **(
                {"max_candidate_rank": max_candidate_rank}
                if max_candidate_rank is not None
                else {}
            ),
            "test_evaluation_policy": test_evaluation_policy,
        },
        "llm_scientific": llm_scientific or {"mode": "disabled"},
    }
    _write_yaml_immutable(resolved_request_path, resolved_request)
    _write_yaml_immutable(task_path, task_payload)
    _write_yaml_immutable(pipeline_path, pipeline_payload)
    _write_json_immutable(preflight_path, preflight)
    from mint_scout.agent.workflow import workflow_for_config

    launch_plan = workflow_for_config(pipeline_path).build_launch_plan(pipeline_path)
    ArtifactRegistry(launch_plan.registry_path).register(
        inspect_artifact(preflight_path)
    )
    return UserRunBundle(
        status=status,
        request_id=request_id,
        request_hash=request_hash,
        run_id=run_id,
        pipeline_kind="protein_ligand_agent_lifecycle_v1",
        task_config=str(task_path),
        pipeline_config=str(pipeline_path),
        preflight_report=str(preflight_path),
        launch_plan=launch_plan,
        evaluation_mode=evaluation_mode,
    )


def _prepare_toxicity_user_mapping(
    raw: Mapping[str, Any],
    *,
    base_dir: str | Path,
    source: str,
    output_dir: str | Path | None,
    llm: Mapping[str, Any],
    onboarding: Mapping[str, Any],
    request_id: str,
    goal: str,
    task_type: str,
    primary_metric: str,
) -> UserRunBundle:
    if task_type != "regression":
        raise ValueError("toxicity V1 supports task_type=regression")
    if primary_metric not in {"PCC2", "PCC", "R2"}:
        raise ValueError("toxicity V1 supports primary_metric=PCC2, PCC, or R2")
    root = Path(base_dir).expanduser().resolve()
    profile_path = _resolve_path(
        raw.get("execution_profile"), root, "execution_profile"
    )
    if not profile_path.is_file():
        raise FileNotFoundError(f"execution profile does not exist: {profile_path}")
    profile = load_execution_profile(profile_path)
    if not profile.supports_slurm:
        raise ValueError("toxicity user start requires a Slurm execution profile")

    dataset = _mapping(raw.get("dataset"), "dataset")
    _reject_unknown(
        dataset,
        {
            "manifest_path",
            "label_name",
            "validate_paths",
            "columns",
            "source",
            "preparation",
            "cv_group_identifier",
        },
        "dataset",
    )
    if dataset.get("preparation") is not None:
        raise ValueError("toxicity V1 currently requires an existing manifest")
    manifest_path = _resolve_path(
        dataset.get("manifest_path"), root, "dataset.manifest_path"
    )
    if not manifest_path.is_file():
        raise FileNotFoundError(f"toxicity manifest does not exist: {manifest_path}")
    validate_paths = dataset.get("validate_paths", True)
    if not isinstance(validate_paths, bool):
        raise ValueError("dataset.validate_paths must be true or false")
    columns = _normalize_columns(dataset.get("columns"), system_type="small_molecule")

    run = _mapping(raw.get("run", {}), "run")
    _reject_unknown(
        run,
        {
            "run_id",
            "invariants",
            "user_target",
            "selection_objective",
            "qc_config",
            "audit_config",
            "scout_config",
            "gbt_config",
            "feature_diagnostic_top_k",
            "auto_resume_failed",
            "max_attempts",
            "stop_after_stage",
            "test_evaluation_policy",
        },
        "run",
    )
    invariants_explicit = "invariants" in run and run.get("invariants") is not None
    invariants = _invariants(run.get("invariants", list(SUPPORTED_INVARIANTS)))
    user_target = _optional_target(run.get("user_target"), metric=primary_metric)
    selection_objective = str(run.get("selection_objective") or "maximize_rank1")
    if selection_objective not in {"satisfy_target", "maximize_rank1"}:
        raise ValueError(
            "toxicity run.selection_objective must be satisfy_target or maximize_rank1"
        )
    if selection_objective == "satisfy_target" and user_target is None:
        raise ValueError("toxicity satisfy_target requires run.user_target")
    test_evaluation_policy = str(run.get("test_evaluation_policy") or "frozen_once")
    if test_evaluation_policy not in {"frozen_once", "test_acceptance"}:
        raise ValueError(
            "run.test_evaluation_policy must be frozen_once or test_acceptance"
        )
    llm_scientific = _normalize_llm_scientific_config(raw.get("llm_scientific"))
    label_policy = LabeledSamplePolicy.from_mapping(
        _mapping(raw.get("labels", {}), "labels")
    )
    request_hash = stable_hash(
        _canonical_request(raw, profile_path, manifest_path, None, "toxicity-v1")
    )
    run_id = _optional_identifier(run.get("run_id")) or _identifier(
        f"{request_id}-{request_hash[:8]}", "run.run_id"
    )
    destination = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else Path(profile.run_root) / "intake" / request_id / run_id
    )
    task_path = destination / "task.yaml"
    pipeline_path = destination / "pipeline.yaml"
    preflight_path = destination / "preflight.json"
    resolved_request_path = destination / "request.resolved.yaml"
    workflow_root = Path(profile.run_root) / "toxicity" / request_id / run_id

    task_payload: dict[str, Any] = {
        "task_id": request_id,
        "project_name": "mint-agent",
        "research_goal": goal,
        "intent": {
            "source": (
                "openai_responses_api" if llm.get("used") else "structured_request"
            ),
            "llm": dict(llm),
            **({"dataset_onboarding": onboarding} if onboarding else {}),
        },
        "system_type": "small_molecule",
        "task_type": "regression",
        "primary_metric": primary_metric.lower(),
        "dataset_manifest": {
            "path": str(manifest_path),
            "label_name": _optional_text(dataset.get("label_name")) or "target",
            "source": _optional_text(dataset.get("source")) or source,
            "validate_paths": validate_paths,
            "columns": columns,
        },
        "invariants": list(invariants),
        "selection_preferences": {
            "method_source": (
                "user" if invariants_explicit else "default_all_supported"
            ),
            "target_source": "user" if user_target is not None else "none",
            "selection_objective": selection_objective,
            "user_target": user_target,
            "test_evaluation_policy": test_evaluation_policy,
        },
        "labels": label_policy.to_dict(),
        "label_retrieval": {
            "allowed": bool(raw.get("label_retrieval_allowed", False))
        },
        "hydrogen_request": _normalize_hydrogen_request(raw.get("hydrogen_policy")),
        "toxicity_workflow": {
            "adapter": "legacy_toxicity_small_molecule_v1",
            "primary_metric": primary_metric,
            "llm_can_add_representation_candidates": True,
            "llm_used_for_numeric_ranking": False,
        },
    }
    pipeline_payload: dict[str, Any] = {
        "version": "mint-agent.toxicity-workflow.v1",
        "repository_root": profile.project_root,
        "run_id": run_id,
        "task_config": str(task_path),
        "execution_profile": str(profile_path),
        "dataset_id": request_id,
        "dataset_manifest": str(manifest_path),
        "invariants": list(invariants),
        "selection_objective": selection_objective,
        "primary_metric": primary_metric,
        "workflow_root": str(workflow_root),
        "design_root": str(workflow_root / "design"),
        "advisory_root": str(workflow_root / "design-advisory"),
        "probe_root": str(workflow_root / "probe"),
        "feature_root": str(Path(profile.cache_root) / "toxicity_features" / request_id),
        "final_root": str(workflow_root / "final"),
        "gbt_config": _profile_path(
            profile.project_root,
            run.get("gbt_config", "configs/gbt/toxicity_probe_gbt.yaml"),
        ),
        "llm_scientific": llm_scientific or {"mode": "disabled"},
    }
    if user_target is not None:
        pipeline_payload["user_target"] = user_target

    card = task_card_from_config(task_payload, config_path=task_path)
    if card is None:
        raise ValueError("generated toxicity task did not produce a dataset manifest")
    evaluation = route_evaluation_mode(card)
    warnings = list(evaluation.warnings)
    if evaluation.mode in BLOCKED_EVALUATION_MODES:
        detail = "; ".join(evaluation.warnings) or evaluation.mode.value
        raise ValueError(f"toxicity request cannot be executed: {detail}")
    split_counts = {
        name: sum(sample.split is not None and sample.split.value == name for sample in card.dataset.samples)
        for name in ("train", "validation", "test")
    }
    if not split_counts["train"] or not split_counts["test"] or split_counts["validation"]:
        raise ValueError(
            "toxicity V1 requires explicit train/test splits and does not yet support "
            "a separate validation split"
        )
    leakage = _split_role_path_leakage(card.dataset.samples)
    if leakage:
        raise ValueError(
            "identical molecule inputs occur across protected splits: "
            + "; ".join(leakage[:5])
        )
    small_data_warning = label_policy.warning(len(evaluation.modeling_sample_ids))
    status = "READY"
    if small_data_warning is not None:
        warnings.append(small_data_warning)
        if not label_policy.allow_small_data_override:
            status = "NEEDS_SMALL_DATA_CONFIRMATION"

    preflight = {
        "report_schema": PREFLIGHT_SCHEMA,
        "dataset_id": request_id,
        "evidence_scope": "design",
        "status": status,
        "request_id": request_id,
        "request_hash": request_hash,
        "run_id": run_id,
        "source": source,
        "goal": goal,
        "system_type": "small_molecule",
        "task_type": "regression",
        "primary_metric": primary_metric,
        "user_target": user_target,
        "selection_objective": selection_objective,
        "invariants": list(invariants),
        "sample_count": len(card.dataset.samples),
        "evaluation_mode": evaluation.mode.value,
        "dataset": {
            "manifest": str(manifest_path),
            "sample_count": len(card.dataset.samples),
            "labeled_count": len(card.dataset.labeled_samples),
            "modeling_sample_count": len(evaluation.modeling_sample_ids),
            "evaluation_sample_count": len(evaluation.evaluation_sample_ids),
        },
        "execution": {
            "profile": str(profile_path),
            "profile_id": profile.profile_id,
            "scheduler": "slurm",
            "project_root": profile.project_root,
            "run_root": profile.run_root,
        },
        "toxicity_workflow": {
            "workflow_root": str(workflow_root),
            "design_root": pipeline_payload["design_root"],
            "advisory_root": pipeline_payload["advisory_root"],
            "probe_root": pipeline_payload["probe_root"],
            "final_root": pipeline_payload["final_root"],
            "final_report": str(
                workflow_root / "final" / "gbt" / "toxicity_final_test_report.json"
            ),
        },
        "protocol_guards": {
            "llm_used_for_numeric_decisions": False,
            "user_splits_are_preserved": True,
            "test_used_for_selection": False,
            "full_train_cross_validation_used": False,
            "final_test_query_budget": 1,
            "test_evaluation_policy": test_evaluation_policy,
            "feature_tools_are_isolated": True,
            "llm_scientific_mode": llm_scientific.get("mode", "disabled"),
        },
        "llm": dict(llm),
        **({"dataset_onboarding": onboarding} if onboarding else {}),
        "warnings": warnings,
    }
    resolved_request = {
        **raw,
        "execution_profile": str(profile_path),
        "dataset": {
            **dataset,
            "manifest_path": str(manifest_path),
            "columns": columns,
        },
        "labels": label_policy.to_dict(),
        "run": {
            **run,
            "run_id": run_id,
            "invariants": list(invariants),
            "user_target": user_target,
            "selection_objective": selection_objective,
            "test_evaluation_policy": test_evaluation_policy,
        },
        "llm_scientific": llm_scientific or {"mode": "disabled"},
    }
    _write_yaml_immutable(resolved_request_path, resolved_request)
    _write_yaml_immutable(task_path, task_payload)
    _write_yaml_immutable(pipeline_path, pipeline_payload)
    _write_json_immutable(preflight_path, preflight)
    from mint_scout.agent.workflow import workflow_for_config

    launch_plan = workflow_for_config(pipeline_path).build_launch_plan(pipeline_path)
    ArtifactRegistry(launch_plan.registry_path).register(
        inspect_artifact(preflight_path)
    )
    return UserRunBundle(
        status=status,
        request_id=request_id,
        request_hash=request_hash,
        run_id=run_id,
        pipeline_kind="small_molecule_toxicity_gbt_v1",
        task_config=str(task_path),
        pipeline_config=str(pipeline_path),
        preflight_report=str(preflight_path),
        launch_plan=launch_plan,
        evaluation_mode=evaluation.mode.value,
    )


def request_from_prompt(
    prompt: str,
    *,
    model: str,
    api_key: str,
    endpoint: str = "https://api.openai.com/v1/responses",
    timeout: float = 60.0,
    opener: Callable[..., Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not prompt.strip():
        raise ValueError("prompt must not be empty")
    if not model.strip():
        raise ValueError("model must not be empty")
    if not api_key.strip():
        raise ValueError("api_key must not be empty")
    parsed, provenance, _body = request_structured_output(
        prompt,
        instructions=(
            "Convert the user's molecular modeling request into the supplied schema. "
            "Do not invent filesystem paths, column names, splits, labels, targets, or "
            "scientific thresholds. execution_profile and representation_profile are "
            "local YAML file paths, not model families; GBT is a model family and must "
            "never be placed in either profile field. manifest_path is a local CSV, "
            "JSONL, or NDJSON path. Use all five invariant tools unless the user "
            "explicitly restricts them. For hydrogen_policy, use include or exclude only "
            "when the user explicitly requests it; otherwise use auto and assess whether "
            "explicit hydrogen is likely relevant to the scientific task. Unknown "
            "optional values must be null. Set run.selection_objective to "
            "satisfy_target when the user asks to meet a threshold, maximize_rank1 "
            "when the user asks for only the single best-ranked practical candidate, "
            "maximize_top_k when the user asks for best-available mode or a top-k "
            "candidate pool without early stopping, and maximize_all only when the "
            "user explicitly asks to evaluate every available method. For "
            "maximize_top_k, set run.max_candidate_rank to the requested k or null "
            "to use the pipeline default. Set "
            "run.test_evaluation_policy to frozen_once unless the user explicitly asks "
            "to use the protected test split as an iterative acceptance loop."
        ),
        schema=_prompt_request_schema(),
        schema_name="mint_agent_user_request",
        model=model,
        api_key=api_key,
        endpoint=endpoint,
        timeout=timeout,
        opener=opener,
    )
    discarded_fields = _discard_invalid_prompt_paths(parsed)
    if discarded_fields:
        provenance["discarded_fields"] = discarded_fields
    provenance["prompt_hash"] = stable_hash(prompt)
    return parsed, provenance


def _discard_invalid_prompt_paths(parsed: Mapping[str, Any]) -> list[str]:
    if not isinstance(parsed, dict):
        return []
    discarded: list[str] = []
    for field in ("execution_profile", "representation_profile"):
        value = parsed.get(field)
        if value is not None and not _looks_like_local_file(
            value, suffixes={".yaml", ".yml"}
        ):
            parsed[field] = None
            discarded.append(field)
    dataset = parsed.get("dataset")
    if isinstance(dataset, dict):
        value = dataset.get("manifest_path")
        if value is not None and not _looks_like_local_file(
            value, suffixes={".csv", ".jsonl", ".ndjson"}
        ):
            dataset["manifest_path"] = None
            discarded.append("dataset.manifest_path")
    return discarded


def _looks_like_local_file(value: object, *, suffixes: set[str]) -> bool:
    text = _optional_text(value)
    if text is None or "://" in text or "\n" in text or "\x00" in text:
        return False
    return Path(text).suffix.lower() in suffixes


def _prompt_request_schema() -> dict[str, Any]:
    nullable_string = {"type": ["string", "null"]}
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "version",
            "request_id",
            "goal",
            "system_type",
            "task_type",
            "primary_metric",
            "execution_profile",
            "representation_profile",
            "dataset",
            "run",
            "labels",
            "label_retrieval_allowed",
            "hydrogen_policy",
        ],
        "properties": {
            "version": {"type": "string", "enum": [REQUEST_VERSION]},
            "request_id": {"type": "string"},
            "goal": {"type": "string"},
            "system_type": {
                "type": "string",
                "enum": ["protein_ligand", "small_molecule"],
            },
            "task_type": {"type": "string", "enum": ["regression", "classification"],},
            "primary_metric": {"type": "string"},
            "execution_profile": nullable_string,
            "representation_profile": nullable_string,
            "dataset": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "manifest_path",
                    "label_name",
                    "validate_paths",
                    "columns",
                    "source",
                    "preparation",
                    "cv_group_identifier",
                ],
                "properties": {
                    "manifest_path": nullable_string,
                    "label_name": nullable_string,
                    "validate_paths": {"type": "boolean"},
                    "source": nullable_string,
                    "preparation": {"type": "null"},
                    "cv_group_identifier": nullable_string,
                    "columns": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": [
                            "sample_id",
                            "target",
                            "split",
                            "roles",
                            "identifiers",
                        ],
                        "properties": {
                            "sample_id": nullable_string,
                            "target": nullable_string,
                            "split": nullable_string,
                            "roles": {
                                "type": "object",
                                "additionalProperties": False,
                                "required": ["protein", "ligand", "molecule"],
                                "properties": {
                                    "protein": nullable_string,
                                    "ligand": nullable_string,
                                    "molecule": nullable_string,
                                },
                            },
                            "identifiers": {
                                "type": "object",
                                "additionalProperties": False,
                                "required": ["pdb_id", "smiles", "source_filename"],
                                "properties": {
                                    "pdb_id": nullable_string,
                                    "smiles": nullable_string,
                                    "source_filename": nullable_string,
                                },
                            },
                        },
                    },
                },
            },
            "run": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "run_id",
                    "invariants",
                    "user_target",
                    "selection_objective",
                    "max_candidate_rank",
                    "test_evaluation_policy",
                ],
                "properties": {
                    "run_id": nullable_string,
                    "invariants": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": list(SUPPORTED_INVARIANTS),
                        },
                    },
                    "user_target": {"type": ["number", "null"]},
                    "selection_objective": {
                        "type": "string",
                        "enum": [
                            "satisfy_target",
                            "maximize_rank1",
                            "maximize_top_k",
                            "maximize_all",
                        ],
                    },
                    "max_candidate_rank": {"type": ["integer", "null"], "minimum": 1},
                    "test_evaluation_policy": {
                        "type": ["string", "null"],
                        "enum": ["frozen_once", "test_acceptance", None],
                    },
                },
            },
            "labels": {
                "type": "object",
                "additionalProperties": False,
                "required": ["min_labeled_samples", "allow_small_data_override",],
                "properties": {
                    "min_labeled_samples": {"type": ["integer", "null"]},
                    "allow_small_data_override": {"type": "boolean"},
                },
            },
            "label_retrieval_allowed": {"type": "boolean"},
            "hydrogen_policy": {
                "type": "object",
                "additionalProperties": False,
                "required": ["mode", "task_relevance", "rationale"],
                "properties": {
                    "mode": {
                        "type": "string",
                        "enum": ["auto", "include", "exclude"],
                    },
                    "task_relevance": {
                        "type": "string",
                        "enum": [
                            "likely_relevant",
                            "unlikely_relevant",
                            "uncertain",
                        ],
                    },
                    "rationale": nullable_string,
                },
            },
        },
    }


def _normalize_hydrogen_request(value: object) -> dict[str, Any]:
    if value is None:
        return {
            "mode": "auto",
            "task_relevance": "uncertain",
            "rationale": None,
            "source": "default_auto",
        }
    raw = _mapping(value, "hydrogen_policy")
    _reject_unknown(
        raw, {"mode", "task_relevance", "rationale"}, "hydrogen_policy"
    )
    mode = str(raw.get("mode") or "auto")
    if mode not in {"auto", "include", "exclude"}:
        raise ValueError("hydrogen_policy.mode must be auto, include, or exclude")
    relevance = str(raw.get("task_relevance") or "uncertain")
    if relevance not in {"likely_relevant", "unlikely_relevant", "uncertain"}:
        raise ValueError(
            "hydrogen_policy.task_relevance must be likely_relevant, "
            "unlikely_relevant, or uncertain"
        )
    return {
        "mode": mode,
        "task_relevance": relevance,
        "rationale": _optional_text(raw.get("rationale")),
        "source": "request",
    }


def _normalize_llm_provenance(value: Mapping[str, Any] | None,) -> dict[str, Any]:
    source = value or {}
    result: dict[str, Any] = {
        "used": bool(source.get("used", False)),
        "used_for_numeric_decisions": False,
    }
    for key in (
        "provider",
        "endpoint",
        "model_requested",
        "model_returned",
        "response_id",
        "prompt_hash",
        "input_hash",
        "store",
    ):
        item = source.get(key)
        if isinstance(item, (str, bool, int, float)) or item is None:
            if item is not None:
                result[key] = item
    discarded_fields = source.get("discarded_fields")
    if isinstance(discarded_fields, list):
        result["discarded_fields"] = [
            str(item) for item in discarded_fields if isinstance(item, str)
        ]
    usage = source.get("usage")
    if isinstance(usage, Mapping):
        safe_usage = {
            key: int(usage[key])
            for key in ("input_tokens", "output_tokens", "total_tokens")
            if isinstance(usage.get(key), int) and not isinstance(usage.get(key), bool)
        }
        if safe_usage:
            result["usage"] = safe_usage
    calls = source.get("calls")
    if isinstance(calls, list):
        safe_calls = [
            normalized
            for item in calls
            if isinstance(item, Mapping)
            if (normalized := _normalize_llm_call(item))
        ]
        if safe_calls:
            result["calls"] = safe_calls
    return result


def _normalize_llm_call(value: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {"used_for_numeric_decisions": False}
    for key in (
        "stage",
        "provider",
        "endpoint",
        "model_requested",
        "model_returned",
        "response_id",
        "prompt_hash",
        "input_hash",
        "store",
    ):
        item = value.get(key)
        if isinstance(item, (str, bool, int, float)) or item is None:
            if item is not None:
                result[key] = item
    discarded_fields = value.get("discarded_fields")
    if isinstance(discarded_fields, list):
        result["discarded_fields"] = [
            str(item) for item in discarded_fields if isinstance(item, str)
        ]
    usage = value.get("usage")
    if isinstance(usage, Mapping):
        safe_usage = {
            key: int(usage[key])
            for key in ("input_tokens", "output_tokens", "total_tokens")
            if isinstance(usage.get(key), int) and not isinstance(usage.get(key), bool)
        }
        if safe_usage:
            result["usage"] = safe_usage
    return result if result.get("stage") else {}


def _normalize_dataset_onboarding(value: Mapping[str, Any] | None,) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not value:
        return {}
    result: dict[str, Any] = {
        "numeric_decisions_allowed": False,
    }
    for key in ("report_schema", "status"):
        item = value.get(key)
        if isinstance(item, str) and item.strip():
            result[key] = item.strip()
    profile = _normalize_manifest_profile(value.get("manifest_profile"))
    if profile:
        result["manifest_profile"] = profile
        result["manifest_profile_hash"] = stable_hash(profile)
    mapping = _normalize_column_mapping(value.get("mapping"))
    if mapping:
        result["mapping"] = mapping
    llm = value.get("llm")
    if isinstance(llm, Mapping):
        safe_llm = _normalize_llm_call(llm)
        if safe_llm:
            result["llm"] = safe_llm
    return result


def _normalize_manifest_profile(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    result: dict[str, Any] = {}
    for key in ("report_schema", "format"):
        item = value.get(key)
        if isinstance(item, str) and item.strip():
            result[key] = item.strip()
    for key in ("row_count", "rows_profiled", "profile_limit"):
        item = value.get(key)
        if isinstance(item, int) and not isinstance(item, bool) and item >= 0:
            result[key] = item
    columns = []
    raw_columns = value.get("columns")
    if isinstance(raw_columns, list):
        for raw in raw_columns:
            if not isinstance(raw, Mapping) or not isinstance(raw.get("name"), str):
                continue
            column: dict[str, Any] = {"name": raw["name"]}
            for key in ("profiled_count", "non_empty_count", "missing_count"):
                item = raw.get(key)
                if isinstance(item, int) and not isinstance(item, bool) and item >= 0:
                    column[key] = item
            for key in ("numeric_fraction", "path_like_fraction", "unique_fraction"):
                item = raw.get(key)
                if (
                    isinstance(item, (int, float))
                    and not isinstance(item, bool)
                    and 0.0 <= float(item) <= 1.0
                ):
                    column[key] = float(item)
            for key in ("file_suffixes", "recognized_split_values"):
                item = raw.get(key)
                if isinstance(item, list) and all(
                    isinstance(entry, str) for entry in item
                ):
                    column[key] = list(item)
            if raw.get("recognized_split_values_scope") == "full_manifest":
                column["recognized_split_values_scope"] = "full_manifest"
            columns.append(column)
    if columns:
        result["columns"] = columns
    result["privacy"] = {
        "raw_values_included": False,
        "sample_ids_included": False,
        "filesystem_paths_included": False,
    }
    return result


def _normalize_column_mapping(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    fields = (
        "sample_id",
        "target",
        "split",
        "protein",
        "ligand",
        "molecule",
        "pdb_id",
        "smiles",
        "source_filename",
    )
    result: dict[str, Any] = {}
    raw_mapping = value.get("mapping")
    if isinstance(raw_mapping, Mapping):
        result["mapping"] = {
            field: raw_mapping.get(field)
            for field in fields
            if raw_mapping.get(field) is None or isinstance(raw_mapping.get(field), str)
        }
    raw_confidence = value.get("confidence")
    if isinstance(raw_confidence, Mapping):
        result["confidence"] = {
            field: raw_confidence[field]
            for field in fields
            if raw_confidence.get(field) in {"high", "medium", "low"}
        }
    confirmations = value.get("requires_confirmation")
    if isinstance(confirmations, list):
        result["requires_confirmation"] = [
            field for field in confirmations if field in fields
        ]
    evidence = value.get("evidence")
    if isinstance(evidence, Mapping):
        result["evidence"] = {
            field: evidence[field]
            for field in fields
            if isinstance(evidence.get(field), str)
        }
    summary = value.get("summary")
    if isinstance(summary, str):
        result["summary"] = summary
    return result


def _normalize_columns(value: object, *, system_type: str = "protein_ligand") -> dict[str, Any]:
    columns = _mapping(value or {}, "dataset.columns")
    _reject_unknown(
        columns,
        {"sample_id", "target", "split", "roles", "identifiers"},
        "dataset.columns",
    )
    default_roles = (
        {"molecule": "molecule_path"}
        if system_type == "small_molecule"
        else {"protein": "protein_path", "ligand": "ligand_path"}
    )
    roles = _mapping(columns.get("roles", default_roles), "dataset.columns.roles")
    required_roles = ("molecule",) if system_type == "small_molecule" else ("protein", "ligand")
    unknown_roles = sorted(set(roles) - {"protein", "ligand", "molecule"})
    if unknown_roles:
        raise ValueError(f"dataset.columns.roles has unsupported roles: {unknown_roles}")
    missing_roles = [role for role in required_roles if _optional_text(roles.get(role)) is None]
    if missing_roles:
        raise ValueError(
            "dataset.columns.roles is missing required roles: "
            + ", ".join(missing_roles)
        )
    normalized = {
        "sample_id": _optional_text(columns.get("sample_id")) or "sample_id",
        "target": _optional_text(columns.get("target")) or "target",
        "split": _optional_text(columns.get("split")) or "split",
        "roles": {
            role: _required_text(roles.get(role), f"roles.{role}")
            for role in required_roles
        },
    }
    identifiers = _mapping(
        columns.get("identifiers", {}), "dataset.columns.identifiers"
    )
    normalized_identifiers = {
        str(key): str(value)
        for key, value in identifiers.items()
        if _optional_text(value) is not None
    }
    if normalized_identifiers:
        normalized["identifiers"] = normalized_identifiers
    return normalized


def _normalize_llm_scientific_config(value: object) -> dict[str, Any]:
    if value is None:
        return {}
    config = _mapping(value, "llm_scientific")
    _reject_unknown(
        config,
        {
            "mode",
            "model",
            "api_key_env",
            "env_file",
            "cache_dir",
            "max_candidates",
            "timeout_seconds",
        },
        "llm_scientific",
    )
    mode = _optional_text(config.get("mode")) or "disabled"
    if mode not in {"disabled", "shadow", "advisory"}:
        raise ValueError("llm_scientific.mode must be disabled, shadow, or advisory")
    result: dict[str, Any] = {"mode": mode}
    api_key_env = _optional_text(config.get("api_key_env"))
    if api_key_env is not None:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", api_key_env):
            raise ValueError("llm_scientific.api_key_env must be a shell identifier")
        result["api_key_env"] = api_key_env
    for key in ("model", "env_file", "cache_dir"):
        text = _optional_text(config.get(key))
        if text is not None:
            result[key] = text
    if config.get("max_candidates") is not None:
        result["max_candidates"] = _positive_int(
            config["max_candidates"], "llm_scientific.max_candidates"
        )
    if config.get("timeout_seconds") is not None:
        timeout_seconds = float(config["timeout_seconds"])
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("llm_scientific.timeout_seconds must be positive")
        result["timeout_seconds"] = timeout_seconds
    return result


def _normalize_primary_metric(value: object) -> str:
    text = _required_text(value, "primary_metric")
    token = re.sub(r"[^A-Za-z0-9]+", "", text).upper()
    aliases = {
        "PCC": "PCC",
        "PEARSON": "PCC",
        "PEARSONR": "PCC",
        "PEARSONCORRELATION": "PCC",
        "PCC2": "PCC2",
        "PCCSQUARED": "PCC2",
        "PEARSONSQUARED": "PCC2",
        "PEARSONRSQUARED": "PCC2",
        "PEARSONCORRELATIONSQUARED": "PCC2",
        "R2": "R2",
        "RSQUARED": "R2",
    }
    return aliases.get(token, text.upper())


def _normalize_preparation(
    value: object, *, base_dir: Path, manifest_path: Path
) -> dict[str, Any] | None:
    if value is None:
        return None
    preparation = _mapping(value, "dataset.preparation")
    provider = _required_text(
        preparation.get("provider"), "dataset.preparation.provider"
    )
    supported = {
        "paired_structure_manifest",
        "bdb2020plus",
        "atom3d_lba",
        "hiqbind",
    }
    if provider not in supported:
        raise ValueError(
            f"unsupported dataset preparation provider {provider!r}; expected {sorted(supported)}"
        )
    normalized = dict(preparation)
    configured_output = normalized.get("output_manifest")
    if configured_output is not None:
        output_path = _resolve_path(
            configured_output, base_dir, "dataset.preparation.output_manifest"
        )
        if output_path != manifest_path:
            raise ValueError(
                "dataset.preparation.output_manifest must match dataset.manifest_path"
            )
    normalized["output_manifest"] = str(manifest_path)
    for key in (
        "metadata_path",
        "structures_root",
        "source_root",
        "output_root",
        "source_archive",
        "extract_root",
        "archive_path",
        "lmdb_path",
    ):
        if normalized.get(key) is not None:
            normalized[key] = str(
                _resolve_path(normalized[key], base_dir, f"dataset.preparation.{key}")
            )
    if provider == "hiqbind":
        normalized = _normalize_hiqbind_preparation(normalized)
    return normalized


def _normalize_hiqbind_preparation(value: Mapping[str, Any]) -> dict[str, Any]:
    allowed = {
        "provider",
        "metadata_path",
        "source_root",
        "output_root",
        "output_manifest",
        "source_archive",
        "extract_root",
        "source_url",
        "source_archive_md5",
        "metadata_url",
        "metadata_md5",
        "measurements",
        "affinity_signs",
        "invalid_affinity_policy",
        "output_split",
        "max_samples",
        "expected_sample_count",
        "obabel",
        "pocket_cutoff_angstrom",
        "add_hydrogens",
        "selection",
    }
    _reject_unknown(value, allowed, "dataset.preparation[hiqbind]")
    normalized = dict(value)
    for key in ("metadata_path", "source_root", "output_root", "output_manifest"):
        _required_text(normalized.get(key), f"dataset.preparation.{key}")

    measurements = normalized.get("measurements")
    if not isinstance(measurements, list) or not measurements:
        raise ValueError("dataset.preparation.measurements must be a nonempty list")
    normalized_measurements = list(
        dict.fromkeys(
            _required_text(item, "measurements").lower() for item in measurements
        )
    )
    unknown_measurements = sorted(set(normalized_measurements) - SUPPORTED_MEASUREMENTS)
    if unknown_measurements:
        raise ValueError(f"unsupported HiQBind measurements: {unknown_measurements}")
    normalized["measurements"] = normalized_measurements

    signs = normalized.get("affinity_signs")
    if not isinstance(signs, list) or not signs:
        raise ValueError("dataset.preparation.affinity_signs must be a nonempty list")
    normalized_signs = list(
        dict.fromkeys(_required_text(item, "affinity_signs") for item in signs)
    )
    unknown_signs = sorted(set(normalized_signs) - SUPPORTED_SIGNS)
    if unknown_signs:
        raise ValueError(f"unsupported HiQBind affinity signs: {unknown_signs}")
    normalized["affinity_signs"] = normalized_signs

    invalid_policy = str(normalized.get("invalid_affinity_policy") or "error").lower()
    if invalid_policy not in {"error", "skip"}:
        raise ValueError(
            "dataset.preparation.invalid_affinity_policy must be error or skip"
        )
    normalized["invalid_affinity_policy"] = invalid_policy

    split = _optional_text(normalized.get("output_split"))
    normalized_split = (
        None if split is None or split.lower() == "none" else split.lower()
    )
    if normalized_split is not None and normalized_split not in SUPPORTED_OUTPUT_SPLITS:
        raise ValueError(f"unsupported HiQBind output_split: {normalized_split!r}")
    normalized["output_split"] = normalized_split

    for key in ("max_samples", "expected_sample_count"):
        raw = normalized.get(key)
        normalized[key] = (
            None
            if raw is None or str(raw).strip().lower() in {"", "all", "none"}
            else _positive_int(raw, f"dataset.preparation.{key}")
        )
    cutoff = float(normalized.get("pocket_cutoff_angstrom", 10.0))
    if not math.isfinite(cutoff) or cutoff <= 0.0:
        raise ValueError("dataset.preparation.pocket_cutoff_angstrom must be positive")
    normalized["pocket_cutoff_angstrom"] = cutoff
    add_hydrogens = normalized.get("add_hydrogens", True)
    if not isinstance(add_hydrogens, bool):
        raise ValueError("dataset.preparation.add_hydrogens must be true or false")
    normalized["add_hydrogens"] = add_hydrogens

    raw_selection = _mapping(
        normalized.get("selection", {}), "dataset.preparation.selection"
    )
    _reject_unknown(
        raw_selection,
        {
            "strategy",
            "random_seed",
            "target_quantile_bins",
            "ligand_size_quantile_bins",
            "max_per_pdb_id",
        },
        "dataset.preparation.selection",
    )
    normalized["selection"] = HiQBindSubsetConfig.from_mapping(raw_selection).to_dict()

    for key in ("source_archive_md5", "metadata_md5"):
        digest = _optional_text(normalized.get(key))
        if digest is not None and not re.fullmatch(r"[0-9a-fA-F]{32}", digest):
            raise ValueError(f"dataset.preparation.{key} must be a 32-digit MD5")
        if digest is not None:
            normalized[key] = digest.lower()
    for key in ("source_url", "metadata_url", "obabel"):
        if key in normalized:
            normalized[key] = _required_text(
                normalized[key], f"dataset.preparation.{key}"
            )
    return normalized


def _canonical_request(
    request: Mapping[str, Any],
    profile_path: Path,
    manifest_path: Path,
    representation_profile_path: Path,
    representation_profile_hash: str,
) -> dict[str, Any]:
    return {
        **request,
        "execution_profile": str(profile_path),
        "representation_profile": str(representation_profile_path),
        "representation_profile_hash": representation_profile_hash,
        "dataset": {
            **dict(_mapping(request.get("dataset"), "dataset")),
            "manifest_path": str(manifest_path),
        },
    }


def _representation_profile_path(
    value: object, *, base_dir: Path, project_root: Path
) -> Path:
    configured = _optional_text(value)
    path = (
        _resolve_path(configured, base_dir, "representation_profile")
        if configured is not None
        else (project_root / DEFAULT_REPRESENTATION_PROFILE).resolve()
    )
    if not path.is_file():
        raise FileNotFoundError(f"representation profile does not exist: {path}")
    return path


def _load_representation_profile(path: Path) -> dict[str, Any]:
    profile = load_yaml(path)
    _reject_unknown(
        profile,
        {
            "version",
            "representation_mode",
            "pair_schema",
            "representation_design",
            "cv",
            "data_audit",
        },
        "representation profile",
    )
    if profile.get("version") != REPRESENTATION_PROFILE_VERSION:
        raise ValueError(
            "representation profile version must be "
            f"{REPRESENTATION_PROFILE_VERSION!r}"
        )
    if profile.get("representation_mode") != "dataset_adaptive":
        raise ValueError("V1 representation profile must use dataset_adaptive mode")

    pair_schema = _mapping(profile.get("pair_schema"), "pair_schema")
    _reject_unknown(
        pair_schema,
        {
            "schema_id",
            "protein_elements",
            "ligand_elements",
            "rule",
            "expected_pair_count",
        },
        "pair_schema",
    )
    protein_elements = _element_list(
        pair_schema.get("protein_elements"), "pair_schema.protein_elements"
    )
    ligand_elements = _element_list(
        pair_schema.get("ligand_elements"), "pair_schema.ligand_elements"
    )
    expected_pair_count = _positive_int(
        pair_schema.get("expected_pair_count"), "pair_schema.expected_pair_count"
    )
    if expected_pair_count != len(protein_elements) * len(ligand_elements):
        raise ValueError(
            "pair_schema.expected_pair_count must equal the role-aware Cartesian product"
        )

    design = _mapping(profile.get("representation_design"), "representation_design")
    _reject_unknown(
        design,
        {
            "adapter_supported_elements",
            "hydrogen_policy",
            "metal_awareness",
            "support",
            "filtration",
            "filtration_repair",
            "eic_tau_start",
            "eic_tau_stop",
            "eic_tau_step",
            "saturation_audit_enabled",
            "distance_chunk_size",
        },
        "representation_design",
    )
    adapters = _mapping(
        design.get("adapter_supported_elements"),
        "representation_design.adapter_supported_elements",
    )
    if set(adapters) != {"protein", "ligand"}:
        raise ValueError(
            "representation_design.adapter_supported_elements must define protein and ligand"
        )
    for role in ("protein", "ligand"):
        _element_list(
            adapters[role], f"representation_design.adapter_supported_elements.{role}",
        )
    hydrogen = _mapping(
        design.get("hydrogen_policy", {}),
        "representation_design.hydrogen_policy",
    )
    _reject_unknown(
        hydrogen,
        {"default_mode", "min_explicit_sample_fraction"},
        "representation_design.hydrogen_policy",
    )
    if hydrogen.get("default_mode", "auto") not in {"auto", "include", "exclude"}:
        raise ValueError(
            "representation_design.hydrogen_policy.default_mode must be auto, "
            "include, or exclude"
        )
    hydrogen_fraction = float(hydrogen.get("min_explicit_sample_fraction", 0.95))
    if not 0.0 <= hydrogen_fraction <= 1.0:
        raise ValueError(
            "representation_design.hydrogen_policy.min_explicit_sample_fraction "
            "must be in [0, 1]"
        )
    metal = _mapping(
        design.get("metal_awareness", {}),
        "representation_design.metal_awareness",
    )
    _reject_unknown(
        metal,
        {
            "enabled",
            "protein_elements",
            "min_sample_fraction",
            "min_samples",
            "max_ligand_distance_angstrom",
            "min_proximal_samples",
        },
        "representation_design.metal_awareness",
    )
    if "protein_elements" in metal:
        _element_list(
            metal["protein_elements"],
            "representation_design.metal_awareness.protein_elements",
        )
    metal_fraction = float(metal.get("min_sample_fraction", 0.005))
    if not 0.0 <= metal_fraction <= 1.0:
        raise ValueError(
            "representation_design.metal_awareness.min_sample_fraction must be in [0, 1]"
        )
    _nonnegative_int(
        metal.get("min_samples", 10),
        "representation_design.metal_awareness.min_samples",
    )
    _nonnegative_int(
        metal.get("min_proximal_samples", 10),
        "representation_design.metal_awareness.min_proximal_samples",
    )
    if float(metal.get("max_ligand_distance_angstrom", 6.0)) <= 0.0:
        raise ValueError(
            "representation_design.metal_awareness.max_ligand_distance_angstrom "
            "must be positive"
        )
    support = _mapping(design.get("support", {}), "representation_design.support")
    filtration = _mapping(
        design.get("filtration", {}), "representation_design.filtration"
    )
    try:
        SupportThresholds(**support)
        FiltrationConfig(**filtration)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid representation profile: {exc}") from exc
    repair = _mapping(
        design.get("filtration_repair", {}),
        "representation_design.filtration_repair",
    )
    _reject_unknown(
        repair,
        {
            "enabled",
            "method_specific",
            "max_rounds",
            "extend_factor",
            "shorten_factor",
            "min_stop_angstrom",
            "max_stop_angstrom",
            "trailing_buffer_points",
        },
        "representation_design.filtration_repair",
    )
    try:
        from mint_scout.filtration_repair import FiltrationRepairConfig

        FiltrationRepairConfig.from_mapping(repair)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid filtration repair policy: {exc}") from exc
    tau_start = float(design.get("eic_tau_start", 0.2))
    tau_stop = float(design.get("eic_tau_stop", 5.0))
    tau_step = float(design.get("eic_tau_step", 0.1))
    if tau_start <= 0.0 or tau_stop <= tau_start or tau_step <= 0.0:
        raise ValueError("EIC tau start/stop/step must be positive and ordered")
    _positive_int(
        design.get("distance_chunk_size", 2048),
        "representation_design.distance_chunk_size",
    )

    cv = _mapping(profile.get("cv", {}), "cv")
    _reject_unknown(cv, {"n_splits", "split_seed"}, "cv")
    if _positive_int(cv.get("n_splits", 5), "cv.n_splits") < 2:
        raise ValueError("cv.n_splits must be at least 2")
    int(cv.get("split_seed", 2026))

    data_audit = _mapping(profile.get("data_audit", {}), "data_audit")
    _reject_unknown(
        data_audit, {"manifest_split", "tolerated_out_of_schema"}, "data_audit"
    )
    if data_audit.get("manifest_split", "train") != "train":
        raise ValueError("V1 representation design must audit only the training split")
    tolerated = _mapping(
        data_audit.get("tolerated_out_of_schema", {}),
        "data_audit.tolerated_out_of_schema",
    )
    for role, elements in tolerated.items():
        if role not in {"protein", "ligand"}:
            raise ValueError(f"unsupported tolerated element role: {role}")
        _element_list(elements, f"data_audit.tolerated_out_of_schema.{role}")

    return {key: value for key, value in profile.items() if key != "version"}


def _element_list(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a nonempty list")
    elements = tuple(_required_text(item, name) for item in value)
    if len(set(elements)) != len(elements):
        raise ValueError(f"{name} must not contain duplicates")
    return elements


def _validate_evaluation_sample_counts(
    evaluation: EvaluationPlan, *, cv_folds: int
) -> None:
    if len(evaluation.modeling_sample_ids) < cv_folds:
        raise ValueError(
            "modeling sample count is smaller than the configured CV fold count: "
            f"{len(evaluation.modeling_sample_ids)} < {cv_folds}"
        )
    if evaluation.validation_sample_ids and len(evaluation.validation_sample_ids) < 2:
        raise ValueError("PCC validation requires at least two validation samples")
    if evaluation.evaluation_sample_ids and len(evaluation.evaluation_sample_ids) < 2:
        raise ValueError("PCC evaluation requires at least two evaluation samples")


def _split_role_path_leakage(samples: tuple[SampleRecord, ...]) -> list[str]:
    owners: dict[tuple[tuple[str, str], ...], tuple[str, str]] = {}
    leaks: list[str] = []
    for sample in samples:
        if sample.split is None or not sample.role_paths:
            continue
        identity = tuple(
            sorted(
                (str(role), str(path.resolve()))
                for role, path in sample.role_paths.items()
            )
        )
        prior = owners.get(identity)
        current = (sample.sample_id, sample.split.value)
        if prior is not None and prior[1] != current[1]:
            leaks.append(f"{prior[0]} ({prior[1]}) and {current[0]} ({current[1]})")
        else:
            owners[identity] = current
    return leaks


def _split_identifier_leakage(samples: tuple[SampleRecord, ...]) -> list[str]:
    owners: dict[tuple[str, str], tuple[str, str]] = {}
    leaks: list[str] = []
    for sample in samples:
        if sample.split is None:
            continue
        for identifier, value in sample.identifiers.items():
            key = (identifier, value.casefold())
            current = (sample.sample_id, sample.split.value)
            prior = owners.get(key)
            if prior is not None and prior[1] != current[1]:
                leaks.append(
                    f"{identifier}={value!r}: {prior[0]} ({prior[1]}) and "
                    f"{current[0]} ({current[1]})"
                )
            else:
                owners[key] = current
    return leaks


def _identifier_group_summary(
    samples: tuple[SampleRecord, ...], *, modeling_sample_ids: set[str]
) -> dict[str, dict[str, Any]]:
    counts_by_identifier: dict[str, Counter[str]] = {}
    for sample in samples:
        if sample.sample_id not in modeling_sample_ids:
            continue
        for identifier, value in sample.identifiers.items():
            counts_by_identifier.setdefault(identifier, Counter())[
                value.casefold()
            ] += 1

    result: dict[str, dict[str, Any]] = {}
    for identifier, counts in sorted(counts_by_identifier.items()):
        repeated = sorted(
            ((value, count) for value, count in counts.items() if count > 1),
            key=lambda item: (-item[1], item[0]),
        )
        result[identifier] = {
            "present_sample_count": sum(counts.values()),
            "unique_group_count": len(counts),
            "repeated_group_count": len(repeated),
            "samples_in_repeated_groups": sum(count for _, count in repeated),
            "maximum_group_size": max(counts.values(), default=0),
            "repeated_group_examples": [
                {"value": value, "sample_count": count}
                for value, count in repeated[:10]
            ],
        }
    return result


def _profile_path(project_root: str, value: object) -> str:
    text = _required_text(value, "pipeline config path")
    return text if text.startswith("/") else str(Path(project_root) / text)


def _invariants(value: object) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("run.invariants must be a nonempty list")
    normalized = tuple(dict.fromkeys(str(item).upper() for item in value))
    unknown = sorted(set(normalized) - set(SUPPORTED_INVARIANTS))
    if unknown:
        raise ValueError(f"unsupported invariants: {unknown}")
    return normalized


def _optional_target(value: object, *, metric: str = "PCC") -> float | None:
    if value is None or value == "":
        return None
    target = float(value)
    if not math.isfinite(target) or target > 1.0:
        raise ValueError(f"run.user_target for {metric} must be finite and at most 1")
    if metric == "PCC2" and target < 0.0:
        raise ValueError("run.user_target for PCC2 must be between 0 and 1")
    if metric == "PCC" and target < -1.0:
        raise ValueError("run.user_target for PCC must be between -1 and 1")
    return target


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _nonnegative_int(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a non-negative integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return result


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return dict(value)


def _reject_unknown(value: Mapping[str, Any], allowed: set[str], name: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"unknown {name} fields: {unknown}")


def _resolve_path(value: object, base_dir: Path, name: str) -> Path:
    text = _required_text(value, name)
    path = Path(text).expanduser()
    return path.resolve() if path.is_absolute() else (base_dir / path).resolve()


def _required_text(value: object, name: str) -> str:
    text = _optional_text(value)
    if text is None:
        raise ValueError(f"{name} is required")
    return text


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _identifier(value: object, name: str) -> str:
    text = _required_text(value, name)
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", text):
        raise ValueError(
            f"{name} must contain only letters, digits, dot, dash, or underscore"
        )
    return text


def _optional_identifier(value: object) -> str | None:
    text = _optional_text(value)
    return None if text is None else _identifier(text, "run.run_id")


def _write_yaml_immutable(path: Path, payload: Mapping[str, Any]) -> None:
    import yaml

    text = yaml.safe_dump(dict(payload), sort_keys=False)
    _write_text_immutable(path, text)


def _write_json_immutable(path: Path, payload: Mapping[str, Any]) -> None:
    _write_text_immutable(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_text_immutable(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise ValueError(f"refusing to overwrite immutable intake artifact: {path}")
        return
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())
