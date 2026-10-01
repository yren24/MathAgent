from __future__ import annotations

import copy
import hashlib
import operator
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Callable, Mapping, TypedDict

from mint_scout.dataset_onboarding import (
    apply_column_mapping,
    available_column_names,
    configured_column_errors,
    missing_required_columns,
    profile_manifest,
    request_manifest_column_mapping,
)
from mint_scout.agent.workflow import submit_prepared_bundle
from mint_scout.user_intake import prepare_user_mapping, request_from_prompt


PromptInterpreter = Callable[..., tuple[dict[str, Any], dict[str, Any]]]
RequestPreparer = Callable[..., Any]
BundleSubmitter = Callable[[Mapping[str, Any]], Mapping[str, Any]]
ManifestProfiler = Callable[..., dict[str, Any]]
ColumnMapper = Callable[..., tuple[dict[str, Any], dict[str, Any]]]


class LLMIntakeState(TypedDict, total=False):
    prompt: str
    request: dict[str, Any]
    llm: dict[str, Any]
    explicit_context: dict[str, str]
    manifest_profile: dict[str, Any]
    onboarding: dict[str, Any]
    bundle: dict[str, Any]
    launch_receipt: dict[str, Any]
    status: str
    missing_fields: list[str]
    errors: list[str]
    next_actions: list[str]
    node_trace: Annotated[list[str], operator.add]


@dataclass(frozen=True)
class LLMIntakeContext:
    model: str
    api_key: str
    base_dir: Path
    output_dir: Path | None = None
    execute: bool = False
    execution_profile: str | None = None
    representation_profile: str | None = None
    manifest_path: str | None = None
    llm_scientific: Mapping[str, Any] | None = None
    manifest_profile_max_rows: int = 200
    interpreter: PromptInterpreter = request_from_prompt
    manifest_profiler: ManifestProfiler = profile_manifest
    column_mapper: ColumnMapper = request_manifest_column_mapping
    preparer: RequestPreparer = prepare_user_mapping
    submitter: BundleSubmitter | None = None


def build_llm_intake_graph(context: LLMIntakeContext, *, checkpointer=None):
    try:
        from langgraph.graph import END, START, StateGraph
    except ImportError as exc:
        raise RuntimeError(
            "LLM intake orchestration requires the optional mint-agent[agent] dependencies"
        ) from exc

    def intent_parser_node(state: LLMIntakeState) -> dict[str, Any]:
        prompt = state.get("prompt", "")
        try:
            request, provenance = context.interpreter(
                prompt,
                model=context.model,
                api_key=context.api_key,
            )
        except (RuntimeError, ValueError) as exc:
            return {
                "status": "LLM_ERROR",
                "errors": [str(exc)],
                "next_actions": [
                    "Check the OpenAI model, API credentials, and request text."
                ],
                "node_trace": ["intent_parser_node"],
            }
        request = _ensure_request_id(request, prompt=prompt)
        provenance = _append_llm_call({}, "intent_parser", provenance)
        return {
            "request": dict(request),
            "llm": provenance,
            "status": "INTENT_PARSED",
            "node_trace": ["intent_parser_node"],
        }

    def explicit_context_node(state: LLMIntakeState) -> dict[str, Any]:
        request = copy.deepcopy(state.get("request", {}))
        applied: dict[str, str] = {}
        conflicts: list[str] = []
        _apply_path_override(
            request,
            "execution_profile",
            context.execution_profile,
            base_dir=context.base_dir,
            applied=applied,
            conflicts=conflicts,
        )
        _apply_path_override(
            request,
            "representation_profile",
            context.representation_profile,
            base_dir=context.base_dir,
            applied=applied,
            conflicts=conflicts,
        )
        dataset = request.get("dataset")
        if isinstance(dataset, dict):
            _apply_path_override(
                dataset,
                "manifest_path",
                context.manifest_path,
                base_dir=context.base_dir,
                applied=applied,
                conflicts=conflicts,
                field_name="dataset.manifest_path",
            )
        if context.llm_scientific:
            request["llm_scientific"] = dict(context.llm_scientific)
        if conflicts:
            return {
                "request": request,
                "explicit_context": applied,
                "status": "INVALID_REQUEST",
                "errors": conflicts,
                "next_actions": [
                    "Resolve conflicts between the prompt and explicit CLI paths."
                ],
                "node_trace": ["explicit_context_node"],
            }
        missing = _missing_execution_fields(request)
        if "dataset.manifest_path" in missing:
            return {
                "request": request,
                "explicit_context": applied,
                "status": "NEEDS_USER_INPUT",
                "missing_fields": missing,
                "next_actions": [_missing_field_action(missing)],
                "node_trace": ["explicit_context_node"],
            }
        dataset = request.get("dataset")
        manifest_path = (
            dataset.get("manifest_path") if isinstance(dataset, Mapping) else None
        )
        if manifest_path and request.get("system_type") in {
            "protein_ligand",
            "small_molecule",
        }:
            return {
                "request": request,
                "explicit_context": applied,
                "status": "READY_FOR_ONBOARDING",
                "node_trace": ["explicit_context_node"],
            }
        if missing:
            return {
                "request": request,
                "explicit_context": applied,
                "status": "NEEDS_USER_INPUT",
                "missing_fields": missing,
                "next_actions": [_missing_field_action(missing)],
                "node_trace": ["explicit_context_node"],
            }
        return {
            "request": request,
            "explicit_context": applied,
            "status": "READY_FOR_PREFLIGHT",
            "node_trace": ["explicit_context_node"],
        }

    def manifest_profiler_node(state: LLMIntakeState) -> dict[str, Any]:
        request = state["request"]
        dataset = request.get("dataset", {})
        manifest_path = dataset.get("manifest_path")
        try:
            profile = context.manifest_profiler(
                manifest_path,
                max_rows=context.manifest_profile_max_rows,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            return {
                "status": "NEEDS_USER_INPUT",
                "errors": [str(exc)],
                "next_actions": [
                    "Provide a readable CSV, JSONL, or NDJSON dataset manifest."
                ],
                "node_trace": ["manifest_profiler_node"],
            }
        available = available_column_names(profile)
        errors = configured_column_errors(request, available_columns=available)
        if errors:
            return {
                "manifest_profile": profile,
                "status": "INVALID_REQUEST",
                "errors": errors,
                "next_actions": [
                    "Correct the configured manifest column names or omit them for onboarding."
                ],
                "node_trace": ["manifest_profiler_node"],
            }
        missing_columns = missing_required_columns(request)
        if missing_columns:
            return {
                "manifest_profile": profile,
                "status": "READY_FOR_COLUMN_MAPPING",
                "node_trace": ["manifest_profiler_node"],
            }
        missing_execution = _missing_execution_fields(request)
        if missing_execution:
            return {
                "manifest_profile": profile,
                "onboarding": {
                    "status": "VALIDATED_EXPLICIT_COLUMNS",
                    "llm_used": False,
                },
                "status": "NEEDS_USER_INPUT",
                "missing_fields": missing_execution,
                "next_actions": [_missing_field_action(missing_execution)],
                "node_trace": ["manifest_profiler_node"],
            }
        return {
            "manifest_profile": profile,
            "onboarding": {
                "status": "VALIDATED_EXPLICIT_COLUMNS",
                "llm_used": False,
            },
            "status": "READY_FOR_PREFLIGHT",
            "node_trace": ["manifest_profiler_node"],
        }

    def column_mapper_node(state: LLMIntakeState) -> dict[str, Any]:
        profile = state["manifest_profile"]
        try:
            mapping, provenance = context.column_mapper(
                profile,
                request=state["request"],
                model=context.model,
                api_key=context.api_key,
            )
        except (RuntimeError, ValueError) as exc:
            return {
                "status": "LLM_ERROR",
                "errors": [str(exc)],
                "next_actions": [
                    "Check the model response or provide explicit manifest column mappings."
                ],
                "node_trace": ["column_mapper_node"],
            }
        request, errors = apply_column_mapping(
            state["request"],
            mapping,
            available_columns=available_column_names(profile),
        )
        combined_llm = _append_llm_call(
            state.get("llm", {}), "dataset_column_mapper", provenance
        )
        onboarding = {
            "report_schema": "mint-agent.dataset-column-mapping.v1",
            "status": "NEEDS_USER_CONFIRMATION" if errors else "MAPPED",
            "mapping": mapping,
            "llm": _safe_llm_call("dataset_column_mapper", provenance),
            "numeric_decisions_allowed": False,
        }
        if errors:
            return {
                "request": request,
                "llm": combined_llm,
                "onboarding": onboarding,
                "status": "NEEDS_USER_INPUT",
                "errors": errors,
                "next_actions": [
                    "Confirm the ambiguous manifest fields or provide explicit column names."
                ],
                "node_trace": ["column_mapper_node"],
            }
        missing_execution = _missing_execution_fields(request)
        if missing_execution:
            return {
                "request": request,
                "llm": combined_llm,
                "onboarding": onboarding,
                "status": "NEEDS_USER_INPUT",
                "missing_fields": missing_execution,
                "next_actions": [_missing_field_action(missing_execution)],
                "node_trace": ["column_mapper_node"],
            }
        return {
            "request": request,
            "llm": combined_llm,
            "onboarding": onboarding,
            "status": "READY_FOR_PREFLIGHT",
            "node_trace": ["column_mapper_node"],
        }

    def deterministic_preflight_node(state: LLMIntakeState) -> dict[str, Any]:
        try:
            bundle = context.preparer(
                state["request"],
                base_dir=context.base_dir,
                source="openai_responses_api",
                output_dir=context.output_dir,
                llm_provenance=state.get("llm"),
                dataset_onboarding={
                    **state.get("onboarding", {}),
                    "manifest_profile": state.get("manifest_profile", {}),
                },
            )
        except FileNotFoundError as exc:
            return {
                "status": "NEEDS_USER_INPUT",
                "errors": [str(exc)],
                "next_actions": [
                    "Provide an existing execution profile and dataset manifest path."
                ],
                "node_trace": ["deterministic_preflight_node"],
            }
        except ValueError as exc:
            message = str(exc)
            unsupported = (
                "V1 execution supports" in message
                or "V1 feature selection supports" in message
            )
            return {
                "status": "UNSUPPORTED_TASK_TYPE" if unsupported else "INVALID_REQUEST",
                "errors": [message],
                "next_actions": [
                    "Use a supported protein-ligand regression request or revise "
                    "the supplied metadata."
                ],
                "node_trace": ["deterministic_preflight_node"],
            }
        return {
            "bundle": bundle.to_dict(),
            "status": bundle.status,
            "node_trace": ["deterministic_preflight_node"],
        }

    def optional_submit_node(state: LLMIntakeState) -> dict[str, Any]:
        bundle = state["bundle"]
        submitter = context.submitter or _submit_prepared_bundle
        try:
            receipt = dict(submitter(bundle))
        except (OSError, RuntimeError, ValueError) as exc:
            return {
                "status": "SUBMISSION_ERROR",
                "errors": [str(exc)],
                "next_actions": [
                    "Inspect the generated preflight and retry submission after "
                    "fixing the scheduler issue."
                ],
                "node_trace": ["optional_submit_node"],
            }
        return {
            "launch_receipt": receipt,
            "status": "SUBMITTED",
            "node_trace": ["optional_submit_node"],
        }

    def after_intent(state: LLMIntakeState) -> str:
        return "context" if state.get("status") == "INTENT_PARSED" else "end"

    def after_context(state: LLMIntakeState) -> str:
        if state.get("status") == "READY_FOR_ONBOARDING":
            return "onboarding"
        if state.get("status") == "READY_FOR_PREFLIGHT":
            return "preflight"
        return "end"

    def after_manifest_profile(state: LLMIntakeState) -> str:
        if state.get("status") == "READY_FOR_COLUMN_MAPPING":
            return "mapping"
        if state.get("status") == "READY_FOR_PREFLIGHT":
            return "preflight"
        return "end"

    def after_mapping(state: LLMIntakeState) -> str:
        return "preflight" if state.get("status") == "READY_FOR_PREFLIGHT" else "end"

    def after_preflight(state: LLMIntakeState) -> str:
        if context.execute and state.get("status") in {
            "READY",
            "READY_FOR_PREPARATION",
        }:
            return "submit"
        return "end"

    builder = StateGraph(LLMIntakeState)
    builder.add_node("intent_parser", intent_parser_node)
    builder.add_node("explicit_context", explicit_context_node)
    builder.add_node("manifest_profiler", manifest_profiler_node)
    builder.add_node("column_mapper", column_mapper_node)
    builder.add_node("deterministic_preflight", deterministic_preflight_node)
    builder.add_node("optional_submit", optional_submit_node)
    builder.add_edge(START, "intent_parser")
    builder.add_conditional_edges(
        "intent_parser", after_intent, {"context": "explicit_context", "end": END}
    )
    builder.add_conditional_edges(
        "explicit_context",
        after_context,
        {
            "onboarding": "manifest_profiler",
            "preflight": "deterministic_preflight",
            "end": END,
        },
    )
    builder.add_conditional_edges(
        "manifest_profiler",
        after_manifest_profile,
        {
            "mapping": "column_mapper",
            "preflight": "deterministic_preflight",
            "end": END,
        },
    )
    builder.add_conditional_edges(
        "column_mapper",
        after_mapping,
        {"preflight": "deterministic_preflight", "end": END},
    )
    builder.add_conditional_edges(
        "deterministic_preflight",
        after_preflight,
        {"submit": "optional_submit", "end": END},
    )
    builder.add_edge("optional_submit", END)
    return builder.compile(checkpointer=checkpointer)


def run_llm_intake(
    prompt: str,
    *,
    context: LLMIntakeContext,
) -> dict[str, Any]:
    result = build_llm_intake_graph(context).invoke({"prompt": prompt})
    return public_intake_result(result)


def public_intake_result(state: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: state[key]
        for key in (
            "status",
            "request",
            "llm",
            "explicit_context",
            "manifest_profile",
            "onboarding",
            "bundle",
            "launch_receipt",
            "missing_fields",
            "errors",
            "next_actions",
            "node_trace",
        )
        if key in state
    }


def _ensure_request_id(request: Mapping[str, Any], *, prompt: str) -> dict[str, Any]:
    normalized = dict(request)
    request_id = normalized.get("request_id")
    if not isinstance(request_id, str) or not request_id.strip():
        digest = hashlib.sha256(prompt.strip().encode("utf-8")).hexdigest()[:12]
        normalized["request_id"] = f"natural-{digest}"
    return normalized


def _apply_path_override(
    owner: dict[str, Any],
    key: str,
    override: str | None,
    *,
    base_dir: Path,
    applied: dict[str, str],
    conflicts: list[str],
    field_name: str | None = None,
) -> None:
    if override is None:
        return
    name = field_name or key
    existing = owner.get(key)
    if existing is not None and not _same_path(existing, override, base_dir=base_dir):
        conflicts.append(
            f"{name} conflicts with the explicit command-line value"
        )
        return
    owner[key] = override
    applied[name] = override


def _same_path(left: object, right: object, *, base_dir: Path) -> bool:
    def resolve(value: object) -> Path:
        path = Path(str(value)).expanduser()
        return (base_dir / path).resolve() if not path.is_absolute() else path.resolve()

    return resolve(left) == resolve(right)


def _missing_execution_fields(request: Mapping[str, Any]) -> list[str]:
    missing: list[str] = []
    if not request.get("execution_profile"):
        missing.append("execution_profile")
    dataset = request.get("dataset")
    if not isinstance(dataset, Mapping):
        missing.append("dataset")
    elif not dataset.get("manifest_path") and not dataset.get("preparation"):
        missing.append("dataset.manifest_path")
    return missing


def _missing_field_action(missing: list[str]) -> str:
    return "Provide the missing execution fields: " + ", ".join(missing)


def _submit_prepared_bundle(bundle: Mapping[str, Any]) -> Mapping[str, Any]:
    return submit_prepared_bundle(bundle)


def _append_llm_call(
    existing: Mapping[str, Any], stage: str, provenance: Mapping[str, Any]
) -> dict[str, Any]:
    safe_call = _safe_llm_call(stage, provenance)
    if existing:
        result = dict(existing)
        calls = [
            dict(item)
            for item in existing.get("calls", ())
            if isinstance(item, Mapping)
        ]
    else:
        result = {key: value for key, value in safe_call.items() if key != "stage"}
        calls = []
    calls.append(safe_call)
    result["calls"] = calls
    result["used"] = True
    result["used_for_numeric_decisions"] = False
    totals = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    found_usage = False
    for call in calls:
        usage = call.get("usage")
        if not isinstance(usage, Mapping):
            continue
        found_usage = True
        for key in totals:
            value = usage.get(key)
            if isinstance(value, int) and not isinstance(value, bool):
                totals[key] += value
    if found_usage:
        result["usage"] = totals
    return result


def _safe_llm_call(stage: str, provenance: Mapping[str, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = {"stage": stage, "used_for_numeric_decisions": False}
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
        value = provenance.get(key)
        if isinstance(value, (str, bool, int, float)) or value is None:
            if value is not None:
                safe[key] = value
    discarded_fields = provenance.get("discarded_fields")
    if isinstance(discarded_fields, list):
        safe["discarded_fields"] = [
            str(value) for value in discarded_fields if isinstance(value, str)
        ]
    usage = provenance.get("usage")
    if isinstance(usage, Mapping):
        safe_usage = {
            key: int(usage[key])
            for key in ("input_tokens", "output_tokens", "total_tokens")
            if isinstance(usage.get(key), int)
            and not isinstance(usage.get(key), bool)
        }
        if safe_usage:
            safe["usage"] = safe_usage
    return safe
