from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, Literal, Mapping, Tuple

from mint_scout.invariants.manifest import stable_hash
from mint_scout.openai_responses import request_structured_output


SCIENTIFIC_ADVICE_SCHEMA = "mint-agent.llm-scientific-advice.v2"
ADVICE_CACHE_SCHEMA_VERSION = "mint-agent.llm-advice-cache.v4"
LLMMode = Literal["disabled", "shadow", "advisory"]
AdviceGenerator = Callable[..., Tuple[Dict[str, Any], Dict[str, Any]]]

_KNOWN_INVARIANTS = frozenset({"PH", "PL", "CA", "FPRC", "EIC"})
_STRATIFICATION_FIELDS = frozenset({"target", "molecular_size", "element_coverage"})
_RISK_CHECKS = frozenset(
    {
        "no_test_evidence",
        "split_leakage",
        "feature_quality",
        "filtration_boundary",
        "ranking_stability",
        "computational_cost",
        "shared_probe_folds",
    }
)
_PROBE_VARIANT_LIMIT = 2
_REPRESENTATION_VARIANT_LIMIT = 2
_ALLOWED_PROBE_FRACTIONS = (0.10, 0.20, 0.30)
_ALLOWED_PROBE_MAX_SAMPLES = (300, 500, 750, 1000)
_ALLOWED_TARGET_BINS = (5, 10, 15)
_ALLOWED_SIZE_BINS = (3, 5, 8)
_ALLOWED_PAIR_SUPPORT = (1, 3, 5)
_ALLOWED_AUGMENTATION_FRACTIONS = (0.05, 0.10, 0.20)
_ALLOWED_DISTANCE_QUANTILES = (0.90, 0.95, 0.975, 0.99)
_ALLOWED_MARGIN_FACTORS = (1.0, 1.10, 1.20)
_ALLOWED_PAIR_SUPPORT_FRACTIONS = (0.0025, 0.005, 0.01)
_ALLOWED_PAIR_SUPPORT_SAMPLES = (5, 10, 20)
_ALLOWED_PAIR_CHANNELS = (20, 30, 40, 50)
_FORBIDDEN_TEST_EVIDENCE = re.compile(
    r"\b(?:test|held[- ]out)\s+(?:label|score|metric|performance|result)s?\b",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class LLMScientificContext:
    mode: LLMMode = "disabled"
    model: str | None = None
    api_key: str | None = None
    cache_dir: Path | None = None
    planner: AdviceGenerator | None = None
    critic: AdviceGenerator | None = None
    max_candidates: int = 10
    timeout_seconds: float = 180.0

    def __post_init__(self) -> None:
        if self.mode not in {"disabled", "shadow", "advisory"}:
            raise ValueError(f"Unsupported LLM scientific mode: {self.mode!r}")
        if self.max_candidates < 1:
            raise ValueError("max_candidates must be positive")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.mode != "disabled" and (self.planner is None or self.critic is None):
            if not self.model or not self.api_key:
                raise ValueError(
                    "Enabled LLM scientific modes require a model and API key"
                )


def experiment_planning_advice(
    state: Mapping[str, Any], context: LLMScientificContext
) -> dict[str, Any]:
    facts = experiment_planning_facts(state)
    fallback = deterministic_experiment_plan(facts)
    return _run_advisor(
        stage="experiment_planning",
        facts=facts,
        context=context,
        generator=context.planner
        or partial(generate_experiment_plan, timeout=context.timeout_seconds),
        validator=lambda value: validate_experiment_plan(
            value,
            allowed_invariants=tuple(facts["allowed_invariants"]),
            max_candidates=context.max_candidates,
        ),
        fallback=fallback,
    )


def scientific_critic_advice(
    state: Mapping[str, Any], context: LLMScientificContext
) -> dict[str, Any]:
    facts = scientific_critic_facts(state)
    fallback = deterministic_scientific_critique(facts)
    return _run_advisor(
        stage="scientific_critic",
        facts=facts,
        context=context,
        generator=context.critic
        or partial(generate_scientific_critique, timeout=context.timeout_seconds),
        validator=validate_scientific_critique,
        fallback=fallback,
    )


def experiment_planning_facts(state: Mapping[str, Any]) -> dict[str, Any]:
    task = state.get("task")
    task = task if isinstance(task, Mapping) else {}
    request = state.get("request")
    request = request if isinstance(request, Mapping) else {}
    audit = state.get("dataset_audit")
    audit = audit if isinstance(audit, Mapping) else {}
    audit_payload = _read_json_mapping(audit.get("path"))
    allowed = tuple(
        dict.fromkeys(
            str(value).upper()
            for value in request.get("invariants", ())
            if str(value).upper() in _KNOWN_INVARIANTS
        )
    )
    return {
        "facts_schema": "mint-agent.experiment-planning-facts.v1",
        "task": {
            "system_type": task.get("system_type"),
            "task_type": task.get("task_type"),
            "primary_metric": task.get("primary_metric"),
            "evaluation_mode": task.get("evaluation_mode"),
            "sample_count": task.get("sample_count"),
            "labeled_sample_count": task.get("labeled_sample_count"),
            "label_name": task.get("label_name"),
        },
        "allowed_invariants": list(allowed),
        "data_summary": {
            "modeling_scope": audit_payload.get("modeling_scope"),
            "sample_count": audit_payload.get("sample_count")
            or audit.get("sample_count"),
            "target_summary": _numeric_summary(audit_payload.get("target_summary")),
            "structure_size_summary": _structure_size_summary(
                audit_payload.get("structure_size_summary")
            ),
            "element_summary": _element_summary(audit_payload),
            "audit_status": audit_payload.get("status") or audit.get("status"),
        },
        "information_boundary": {
            "sample_ids_included": False,
            "paths_included": False,
            "test_labels_included": False,
            "test_metrics_included": False,
            "recommendations_require_validation": True,
        },
    }


def scientific_critic_facts(state: Mapping[str, Any]) -> dict[str, Any]:
    task = state.get("task")
    task = task if isinstance(task, Mapping) else {}
    request = state.get("request")
    request = request if isinstance(request, Mapping) else {}
    scout = state.get("scout")
    scout = scout if isinstance(scout, Mapping) else {}
    scout_metadata = scout.get("metadata")
    scout_metadata = scout_metadata if isinstance(scout_metadata, Mapping) else {}
    qc = state.get("qc")
    qc = qc if isinstance(qc, Mapping) else {}
    filtration = state.get("filtration_audit")
    filtration = filtration if isinstance(filtration, Mapping) else {}
    scout_payload = _read_json_mapping(scout.get("path"))
    qc_payload = _read_json_mapping(qc.get("path"))
    filtration_payload = _read_json_mapping(filtration.get("path"))
    requested = tuple(
        dict.fromkeys(
            str(value).upper()
            for value in request.get("invariants", ())
            if str(value).upper() in _KNOWN_INVARIANTS
        )
    )
    return {
        "facts_schema": "mint-agent.scientific-critic-facts.v1",
        "task": {
            "system_type": task.get("system_type"),
            "task_type": task.get("task_type"),
            "primary_metric": task.get("primary_metric"),
            "evaluation_mode": task.get("evaluation_mode"),
        },
        "allowed_invariants": list(requested),
        "probe_evidence": {
            "sample_count": scout_payload.get("sample_count")
            or scout.get("sample_count"),
            "consensus_metrics": _candidate_summaries(
                scout_payload.get("consensus_metrics"), limit=10
            ),
            "priority_order": _safe_priority_order(
                scout_payload.get("frozen_priority_order")
                or scout_metadata.get("frozen_priority_order"),
                requested,
            ),
            "target": _target_summary(scout_payload.get("target")),
        },
        "quality_evidence": {
            "feature_qc": _quality_summary(qc_payload, qc),
            "filtration_audit": _quality_summary(filtration_payload, filtration),
        },
        "information_boundary": {
            "evidence_scope": "probe",
            "sample_ids_included": False,
            "paths_included": False,
            "test_labels_included": False,
            "test_metrics_included": False,
            "critic_can_change_selection": False,
        },
    }


def deterministic_experiment_plan(facts: Mapping[str, Any]) -> dict[str, Any]:
    invariants = [str(value) for value in facts.get("allowed_invariants", ())]
    evaluation_mode = facts.get("task", {}).get("evaluation_mode")
    sampling_strategy = (
        "provided_split"
        if evaluation_mode in {"explicit_labeled_test", "explicit_validation_and_test"}
        else "group_aware_stratified"
    )
    return {
        "objective": "Prioritize validated probe experiments without using held-out evidence.",
        "probe_strategy": {
            "sampling_strategy": sampling_strategy,
            "stratify_by": ["target", "molecular_size", "element_coverage"],
            "group_by": "provided_group_identifier",
            "preserve_extremes": True,
            "rationale": "Use the configured deterministic representative-probe policy.",
        },
        "representation_hypotheses": [
            {
                "id": "baseline-adaptive-representation",
                "category": "quality_control",
                "methods": invariants,
                "proposal": "Audit the configured adaptive representation before modeling.",
                "evidence_needed": [
                    "train-only element inventory",
                    "feature quality control",
                    "filtration audit",
                ],
            }
        ],
        "probe_variants": [],
        "representation_variants": [],
        "candidate_priorities": [
            {
                "rank": index,
                "invariants": [invariant],
                "rationale": "Acquire independent probe evidence before combination ranking.",
                "expected_cost": "unknown",
            }
            for index, invariant in enumerate(invariants, start=1)
        ],
        "risk_checks": [
            "no_test_evidence",
            "split_leakage",
            "feature_quality",
            "filtration_boundary",
            "ranking_stability",
            "computational_cost",
        ],
    }


def deterministic_scientific_critique(facts: Mapping[str, Any],) -> dict[str, Any]:
    probe = facts.get("probe_evidence", {})
    has_ranking = bool(probe.get("priority_order"))
    return {
        "evidence_assessment": "preliminary" if has_ranking else "insufficient",
        "ranking_interpretation": (
            "Use the deterministic probe ranking as preliminary evidence only."
            if has_ranking
            else "No validated probe ranking is available yet."
        ),
        "stability_findings": [
            "Ranking stability must be checked before full-data acquisition."
        ],
        "cost_findings": [
            "Use measured feature acquisition cost when candidates are statistically similar."
        ],
        "quality_findings": [
            "Feature QC and filtration audits remain mandatory execution gates."
        ],
        "recommended_next_action": (
            "continue_deterministic_pipeline"
            if has_ranking
            else "gather_more_probe_evidence"
        ),
        "rationale": "The deterministic pipeline retains control of selection and execution.",
    }


def generate_experiment_plan(
    facts: Mapping[str, Any], *, model: str, api_key: str, timeout: float = 180.0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    plan, provenance, _body = request_structured_output(
        json.dumps(facts, indent=2, sort_keys=True),
        instructions=(
            "Act as a constrained molecular-science experiment planner. Propose a "
            "probe strategy, at most two bounded probe variants, at most two bounded "
            "adaptive-representation variants, representation hypotheses, and a "
            "budget-aware order of candidate mathematical invariant subsets. The "
            "deterministic baseline is injected separately, so variants must be genuine "
            "alternatives. Use only the supplied summary. "
            "Never request or infer sample identities, file paths, held-out labels, test "
            "metrics, or test performance. Use only allowed invariants. Treat all numeric "
            "defaults as configurable. Your plan is advisory and will be validated and "
            "executed by deterministic Python code."
        ),
        schema=_experiment_plan_schema(),
        schema_name="mint_agent_experiment_plan",
        model=model,
        api_key=api_key,
        timeout=timeout,
    )
    provenance["input_hash"] = stable_hash(facts)
    return plan, provenance


def generate_scientific_critique(
    facts: Mapping[str, Any], *, model: str, api_key: str, timeout: float = 180.0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    critique, provenance, _body = request_structured_output(
        json.dumps(facts, indent=2, sort_keys=True),
        instructions=(
            "Act as a constrained scientific critic. Interpret only the supplied probe, "
            "quality, stability, and cost evidence. Do not recompute metrics, change the "
            "ranking, claim full-data or held-out performance, or recommend access to test "
            "labels. Identify uncertainty and choose one allowed next action."
        ),
        schema=_scientific_critique_schema(),
        schema_name="mint_agent_scientific_critique",
        model=model,
        api_key=api_key,
        timeout=timeout,
    )
    provenance["input_hash"] = stable_hash(facts)
    return critique, provenance


def validate_experiment_plan(
    plan: Mapping[str, Any], *, allowed_invariants: tuple[str, ...], max_candidates: int
) -> dict[str, Any]:
    if not isinstance(plan, Mapping):
        raise ValueError("experiment plan must be an object")
    allowed = set(allowed_invariants)
    probe = _required_mapping(plan, "probe_strategy")
    fields = probe.get("stratify_by")
    if not isinstance(fields, list) or not fields:
        raise ValueError("probe_strategy.stratify_by must be a non-empty list")
    normalized_fields = tuple(str(value) for value in fields)
    if len(set(normalized_fields)) != len(normalized_fields):
        raise ValueError("probe_strategy.stratify_by must not contain duplicates")
    if any(value not in _STRATIFICATION_FIELDS for value in normalized_fields):
        raise ValueError("probe strategy contains an unsupported stratification field")

    _validate_probe_variants(plan.get("probe_variants"))
    _validate_representation_variants(plan.get("representation_variants"))

    candidates = plan.get("candidate_priorities")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("candidate_priorities must be a non-empty list")
    if len(candidates) > max_candidates:
        raise ValueError("candidate_priorities exceeds the configured budget")
    ranks: list[int] = []
    seen: set[tuple[str, ...]] = set()
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            raise ValueError("each candidate priority must be an object")
        rank = candidate.get("rank")
        if not isinstance(rank, int):
            raise ValueError("candidate rank must be an integer")
        ranks.append(rank)
        names = _validated_invariant_subset(candidate.get("invariants"), allowed)
        if names in seen:
            raise ValueError("candidate invariant subsets must be unique")
        seen.add(names)
    if ranks != list(range(1, len(candidates) + 1)):
        raise ValueError("candidate ranks must be consecutive and start at one")

    hypotheses = plan.get("representation_hypotheses")
    if not isinstance(hypotheses, list):
        raise ValueError("representation_hypotheses must be a list")
    for hypothesis in hypotheses:
        if not isinstance(hypothesis, Mapping):
            raise ValueError("each representation hypothesis must be an object")
        _validated_invariant_subset(hypothesis.get("methods"), allowed)
    risk_checks = plan.get("risk_checks")
    if not isinstance(risk_checks, list):
        raise ValueError("risk_checks must be a list")
    normalized_risks = tuple(str(value) for value in risk_checks)
    if len(set(normalized_risks)) != len(normalized_risks):
        raise ValueError("risk_checks must not contain duplicates")
    if not set(normalized_risks).issubset(_RISK_CHECKS):
        raise ValueError("risk_checks contains an unsupported value")
    _reject_forbidden_test_evidence(plan)
    return json.loads(json.dumps(plan))


def validate_scientific_critique(critique: Mapping[str, Any],) -> dict[str, Any]:
    if not isinstance(critique, Mapping):
        raise ValueError("scientific critique must be an object")
    action = critique.get("recommended_next_action")
    allowed_actions = {
        "continue_deterministic_pipeline",
        "gather_more_probe_evidence",
        "revise_representation_with_new_run",
        "stop_for_review",
    }
    if action not in allowed_actions:
        raise ValueError("scientific critique contains an unsupported next action")
    _reject_forbidden_test_evidence(critique)
    return json.loads(json.dumps(critique))


def _validate_probe_variants(value: Any) -> None:
    if not isinstance(value, list):
        raise ValueError("probe_variants must be a list")
    if len(value) > _PROBE_VARIANT_LIMIT:
        raise ValueError("probe_variants exceeds the configured experiment budget")
    seen_ids: set[str] = set()
    seen_settings: set[tuple[Any, ...]] = set()
    for variant in value:
        if not isinstance(variant, Mapping):
            raise ValueError("each probe variant must be an object")
        variant_id = _validated_variant_id(variant.get("id"), seen_ids)
        settings = (
            _allowed_value(variant, "fraction", _ALLOWED_PROBE_FRACTIONS),
            _allowed_value(variant, "max_samples", _ALLOWED_PROBE_MAX_SAMPLES),
            _allowed_value(variant, "target_quantile_bins", _ALLOWED_TARGET_BINS),
            _allowed_value(variant, "size_quantile_bins", _ALLOWED_SIZE_BINS),
            _allowed_value(variant, "min_pair_support", _ALLOWED_PAIR_SUPPORT),
            _allowed_value(
                variant, "augmentation_fraction_limit", _ALLOWED_AUGMENTATION_FRACTIONS,
            ),
        )
        if settings in seen_settings:
            raise ValueError("probe variants must contain unique settings")
        seen_ids.add(variant_id)
        seen_settings.add(settings)


def _validate_representation_variants(value: Any) -> None:
    if not isinstance(value, list):
        raise ValueError("representation_variants must be a list")
    if len(value) > _REPRESENTATION_VARIANT_LIMIT:
        raise ValueError(
            "representation_variants exceeds the configured experiment budget"
        )
    seen_ids: set[str] = set()
    seen_settings: set[tuple[Any, ...]] = set()
    for variant in value:
        if not isinstance(variant, Mapping):
            raise ValueError("each representation variant must be an object")
        variant_id = _validated_variant_id(variant.get("id"), seen_ids)
        settings = (
            _allowed_value(
                variant, "local_distance_quantile", _ALLOWED_DISTANCE_QUANTILES
            ),
            _allowed_value(
                variant, "dataset_distance_quantile", _ALLOWED_DISTANCE_QUANTILES
            ),
            _allowed_value(variant, "margin_factor", _ALLOWED_MARGIN_FACTORS),
            _allowed_value(
                variant, "min_pair_support_fraction", _ALLOWED_PAIR_SUPPORT_FRACTIONS,
            ),
            _allowed_value(
                variant, "min_pair_support_samples", _ALLOWED_PAIR_SUPPORT_SAMPLES,
            ),
            _allowed_value(
                variant, "max_element_pair_channels", _ALLOWED_PAIR_CHANNELS
            ),
            _allowed_value(variant, "fixed_point_count", (50,)),
        )
        if settings in seen_settings:
            raise ValueError("representation variants must contain unique settings")
        seen_ids.add(variant_id)
        seen_settings.add(settings)


def _validated_variant_id(value: Any, seen: set[str]) -> str:
    variant_id = str(value or "")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", variant_id):
        raise ValueError("experiment variant id must be a safe non-empty identifier")
    if variant_id in seen or variant_id in {
        "baseline",
        "baseline-probe",
        "baseline-representation",
    }:
        raise ValueError(
            "experiment variant ids must be unique and cannot use reserved baseline ids"
        )
    return variant_id


def _allowed_value(value: Mapping[str, Any], key: str, allowed: tuple[Any, ...]) -> Any:
    candidate = value.get(key)
    if candidate not in allowed:
        raise ValueError(f"{key} must be one of {list(allowed)}")
    return candidate


def _run_advisor(
    *,
    stage: str,
    facts: dict[str, Any],
    context: LLMScientificContext,
    generator: AdviceGenerator,
    validator: Callable[[Mapping[str, Any]], dict[str, Any]],
    fallback: dict[str, Any],
) -> dict[str, Any]:
    input_hash = stable_hash(
        {
            "advice_cache_schema": ADVICE_CACHE_SCHEMA_VERSION,
            "stage": stage,
            "facts": facts,
        }
    )
    if context.mode == "disabled":
        return _advice_artifact(
            stage=stage,
            mode=context.mode,
            input_hash=input_hash,
            status="DISABLED",
            advice=fallback,
            provenance={"used": False},
            validation_errors=[],
        )

    cached = _load_cached_advice(context.cache_dir, stage, input_hash)
    if cached is not None:
        reused = dict(cached)
        # The scientific proposal is cacheable across observation modes, but
        # execution authority belongs to the current request, never the cache.
        reused["mode"] = context.mode
        reused["applied_to_execution"] = False
        reused["cache_hit"] = True
        return reused

    validation_errors: list[str] = []
    try:
        advice, provenance = generator(
            facts, model=str(context.model), api_key=str(context.api_key),
        )
        validated = validator(advice)
        status = "VALIDATED"
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        validated = fallback
        provenance = {
            "used": context.mode != "disabled",
            "model_requested": context.model,
        }
        validation_errors.append(str(exc))
        status = "FALLBACK"
    artifact = _advice_artifact(
        stage=stage,
        mode=context.mode,
        input_hash=input_hash,
        status=status,
        advice=validated,
        provenance=provenance,
        validation_errors=validation_errors,
    )
    if status == "VALIDATED":
        _write_cached_advice(context.cache_dir, stage, input_hash, artifact)
    return artifact


def _advice_artifact(
    *,
    stage: str,
    mode: LLMMode,
    input_hash: str,
    status: str,
    advice: Mapping[str, Any],
    provenance: Mapping[str, Any],
    validation_errors: list[str],
) -> dict[str, Any]:
    return {
        "report_schema": SCIENTIFIC_ADVICE_SCHEMA,
        "advice_cache_schema": ADVICE_CACHE_SCHEMA_VERSION,
        "stage": stage,
        "mode": mode,
        "status": status,
        "input_hash": input_hash,
        "advice": dict(advice),
        "validation": {
            "passed": not validation_errors,
            "errors": list(validation_errors),
        },
        "llm": {
            **dict(provenance),
            "used_for_numeric_decisions": False,
            "input_policy": "summary_only_no_paths_ids_or_test_evidence",
        },
        "applied_to_execution": False,
        "cache_hit": False,
    }


def _load_cached_advice(
    cache_dir: Path | None, stage: str, input_hash: str
) -> dict[str, Any] | None:
    path = _cache_path(cache_dir, stage, input_hash)
    if path is None or not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    validation = payload.get("validation") if isinstance(payload, Mapping) else None
    if (
        not isinstance(payload, dict)
        or payload.get("report_schema") != SCIENTIFIC_ADVICE_SCHEMA
        or payload.get("stage") != stage
        or payload.get("input_hash") != input_hash
        or payload.get("status") != "VALIDATED"
        or not isinstance(validation, Mapping)
        or validation.get("passed") is not True
    ):
        return None
    return payload


def _write_cached_advice(
    cache_dir: Path | None, stage: str, input_hash: str, artifact: Mapping[str, Any],
) -> None:
    path = _cache_path(cache_dir, stage, input_hash)
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    try:
        temporary.replace(path)
    except OSError:
        temporary.unlink(missing_ok=True)


def _cache_path(cache_dir: Path | None, stage: str, input_hash: str) -> Path | None:
    if cache_dir is None:
        return None
    return Path(cache_dir).expanduser().resolve() / stage / f"{input_hash}.json"


def _experiment_plan_schema() -> dict[str, Any]:
    invariant_array = {
        "type": "array",
        "items": {"type": "string", "enum": sorted(_KNOWN_INVARIANTS)},
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "objective",
            "probe_strategy",
            "probe_variants",
            "representation_variants",
            "representation_hypotheses",
            "candidate_priorities",
            "risk_checks",
        ],
        "properties": {
            "objective": {"type": "string"},
            "probe_strategy": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "sampling_strategy",
                    "stratify_by",
                    "group_by",
                    "preserve_extremes",
                    "rationale",
                ],
                "properties": {
                    "sampling_strategy": {
                        "type": "string",
                        "enum": [
                            "stratified",
                            "group_aware_stratified",
                            "provided_split",
                            "deterministic_existing",
                        ],
                    },
                    "stratify_by": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": sorted(_STRATIFICATION_FIELDS),
                        },
                    },
                    "group_by": {
                        "type": "string",
                        "enum": ["none", "provided_group_identifier"],
                    },
                    "preserve_extremes": {"type": "boolean"},
                    "rationale": {"type": "string"},
                },
            },
            "probe_variants": _probe_variant_schema(),
            "representation_variants": _representation_variant_schema(),
            "representation_hypotheses": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "id",
                        "category",
                        "methods",
                        "proposal",
                        "evidence_needed",
                    ],
                    "properties": {
                        "id": {"type": "string"},
                        "category": {
                            "type": "string",
                            "enum": [
                                "element_groups",
                                "filtration_range",
                                "method_subset",
                                "quality_control",
                            ],
                        },
                        "methods": invariant_array,
                        "proposal": {"type": "string"},
                        "evidence_needed": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                },
            },
            "candidate_priorities": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["rank", "invariants", "rationale", "expected_cost",],
                    "properties": {
                        "rank": {"type": "integer"},
                        "invariants": invariant_array,
                        "rationale": {"type": "string"},
                        "expected_cost": {
                            "type": "string",
                            "enum": ["low", "medium", "high", "unknown"],
                        },
                    },
                },
            },
            "risk_checks": {
                "type": "array",
                "items": {"type": "string", "enum": sorted(_RISK_CHECKS),},
            },
        },
    }


def _probe_variant_schema() -> dict[str, Any]:
    return {
        "type": "array",
        "items": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "id",
                "fraction",
                "max_samples",
                "target_quantile_bins",
                "size_quantile_bins",
                "min_pair_support",
                "augmentation_fraction_limit",
                "rationale",
            ],
            "properties": {
                "id": {"type": "string"},
                "fraction": {"type": "number", "enum": list(_ALLOWED_PROBE_FRACTIONS),},
                "max_samples": {
                    "type": "integer",
                    "enum": list(_ALLOWED_PROBE_MAX_SAMPLES),
                },
                "target_quantile_bins": {
                    "type": "integer",
                    "enum": list(_ALLOWED_TARGET_BINS),
                },
                "size_quantile_bins": {
                    "type": "integer",
                    "enum": list(_ALLOWED_SIZE_BINS),
                },
                "min_pair_support": {
                    "type": "integer",
                    "enum": list(_ALLOWED_PAIR_SUPPORT),
                },
                "augmentation_fraction_limit": {
                    "type": "number",
                    "enum": list(_ALLOWED_AUGMENTATION_FRACTIONS),
                },
                "rationale": {"type": "string"},
            },
        },
    }


def _representation_variant_schema() -> dict[str, Any]:
    return {
        "type": "array",
        "items": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "id",
                "local_distance_quantile",
                "dataset_distance_quantile",
                "margin_factor",
                "min_pair_support_fraction",
                "min_pair_support_samples",
                "max_element_pair_channels",
                "fixed_point_count",
                "rationale",
            ],
            "properties": {
                "id": {"type": "string"},
                "local_distance_quantile": {
                    "type": "number",
                    "enum": list(_ALLOWED_DISTANCE_QUANTILES),
                },
                "dataset_distance_quantile": {
                    "type": "number",
                    "enum": list(_ALLOWED_DISTANCE_QUANTILES),
                },
                "margin_factor": {
                    "type": "number",
                    "enum": list(_ALLOWED_MARGIN_FACTORS),
                },
                "min_pair_support_fraction": {
                    "type": "number",
                    "enum": list(_ALLOWED_PAIR_SUPPORT_FRACTIONS),
                },
                "min_pair_support_samples": {
                    "type": "integer",
                    "enum": list(_ALLOWED_PAIR_SUPPORT_SAMPLES),
                },
                "max_element_pair_channels": {
                    "type": "integer",
                    "enum": list(_ALLOWED_PAIR_CHANNELS),
                },
                "fixed_point_count": {"type": "integer", "enum": [50]},
                "rationale": {"type": "string"},
            },
        },
    }


def _scientific_critique_schema() -> dict[str, Any]:
    string_array = {"type": "array", "items": {"type": "string"}}
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "evidence_assessment",
            "ranking_interpretation",
            "stability_findings",
            "cost_findings",
            "quality_findings",
            "recommended_next_action",
            "rationale",
        ],
        "properties": {
            "evidence_assessment": {
                "type": "string",
                "enum": ["insufficient", "preliminary", "adequate"],
            },
            "ranking_interpretation": {"type": "string"},
            "stability_findings": string_array,
            "cost_findings": string_array,
            "quality_findings": string_array,
            "recommended_next_action": {
                "type": "string",
                "enum": [
                    "continue_deterministic_pipeline",
                    "gather_more_probe_evidence",
                    "revise_representation_with_new_run",
                    "stop_for_review",
                ],
            },
            "rationale": {"type": "string"},
        },
    }


def _validated_invariant_subset(values: Any, allowed: set[str]) -> tuple[str, ...]:
    if not isinstance(values, list) or not values:
        raise ValueError("invariant subset must be a non-empty list")
    normalized = tuple(dict.fromkeys(str(value).upper() for value in values))
    if len(normalized) != len(values):
        raise ValueError("invariant subset contains duplicates")
    if not set(normalized).issubset(allowed):
        raise ValueError("plan contains an invariant outside the allowed universe")
    return normalized


def _reject_forbidden_test_evidence(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if _FORBIDDEN_TEST_EVIDENCE.search(str(key)):
                raise ValueError("advice refers to forbidden held-out evidence")
            _reject_forbidden_test_evidence(nested)
    elif isinstance(value, list):
        for nested in value:
            _reject_forbidden_test_evidence(nested)
    elif isinstance(value, str) and _FORBIDDEN_TEST_EVIDENCE.search(value):
        raise ValueError("advice refers to forbidden held-out evidence")


def _required_mapping(value: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    nested = value.get(key)
    if not isinstance(nested, Mapping):
        raise ValueError(f"{key} must be an object")
    return nested


def _read_json_mapping(path: Any) -> dict[str, Any]:
    if not isinstance(path, (str, Path)) or not str(path):
        return {}
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _numeric_summary(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    quantiles = value.get("quantiles")
    quantiles = quantiles if isinstance(quantiles, Mapping) else {}
    return {
        key: value.get(key) for key in ("count", "minimum", "maximum", "mean", "std")
    } | {
        "quantiles": {
            key: quantiles.get(key)
            for key in ("q00", "q05", "q25", "q50", "q75", "q95", "q100")
        }
    }


def _structure_size_summary(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    return {
        key: _numeric_summary(value.get(key))
        for key in ("protein_atom_count", "ligand_atom_count", "total_atom_count",)
    }


def _element_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    roles = payload.get("roles")
    if not isinstance(roles, list):
        return {}
    summary: dict[str, Any] = {}
    for role in roles:
        if not isinstance(role, Mapping):
            continue
        name = str(role.get("role") or "")
        if not name:
            continue
        counts = role.get("element_counts")
        counts = counts if isinstance(counts, Mapping) else {}
        summary[name] = {
            "file_count": role.get("file_count"),
            "atom_count": role.get("atom_count"),
            "element_counts": {
                str(key): counts[key] for key in sorted(counts, key=str)
            },
            "out_of_schema_elements": sorted(
                str(value) for value in role.get("out_of_schema_elements", ())
            ),
        }
    return summary


def _candidate_summaries(value: Any, *, limit: int) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    summaries = []
    allowed = {
        "subset_id",
        "invariants",
        "primary_score",
        "bootstrap_ci_low",
        "bootstrap_ci_high",
        "bootstrap_std",
        "top_tier",
        "top1_frequency",
        "median_rank",
        "estimated_full_acquisition_wall_seconds",
    }
    for candidate in value[:limit]:
        if isinstance(candidate, Mapping):
            summaries.append(
                {key: candidate.get(key) for key in allowed if key in candidate}
            )
    return summaries


def _safe_priority_order(value: Any, allowed: tuple[str, ...]) -> list[list[str]]:
    if not isinstance(value, list):
        return []
    universe = set(allowed)
    result = []
    for subset in value:
        if not isinstance(subset, list):
            continue
        normalized = [str(item).upper() for item in subset]
        if normalized and set(normalized).issubset(universe):
            result.append(normalized)
    return result[:10]


def _target_summary(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    return {
        key: value.get(key)
        for key in ("metric", "direction", "value", "source")
        if key in value
    }


def _quality_summary(
    payload: Mapping[str, Any], record: Mapping[str, Any]
) -> dict[str, Any]:
    issues = payload.get("issues")
    issue_counts: dict[str, int] = {}
    if isinstance(issues, list):
        for issue in issues:
            if isinstance(issue, Mapping):
                name = str(issue.get("issue") or issue.get("code") or "unknown")
                issue_counts[name] = issue_counts.get(name, 0) + 1
    return {
        "status": payload.get("status") or record.get("status"),
        "sample_count": payload.get("sample_count") or record.get("sample_count"),
        "issue_counts": issue_counts,
    }
