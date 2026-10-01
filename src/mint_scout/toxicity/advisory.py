from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Mapping

from mint_scout.invariants.manifest import stable_hash
from mint_scout.openai_responses import request_structured_output
from mint_scout.toxicity.design import (
    ALLOWED_DISTANCE_QUANTILES,
    ALLOWED_MARGIN_FACTORS,
    ALLOWED_PAIR_CAPS,
    ALLOWED_SUPPORT_FRACTIONS,
    ALLOWED_SUPPORT_SAMPLES,
    DESIGN_SCHEMA,
    ToxicityAdaptiveProposal,
)
from mint_scout.toxicity.probe import ToxicityProbePolicy


ADVISORY_SCHEMA = "mint-agent.toxicity-representation-advisory.v1"
AdviceGenerator = Callable[..., Any]
ALLOWED_PROBE_SIZES = (300, 500, 750, 1000)
ALLOWED_TARGET_BINS = (5, 10, 15)
ALLOWED_SIZE_BINS = (3, 5, 8)
ALLOWED_PROBE_PAIR_SUPPORT = (1, 3, 5)


def toxicity_advisory_facts(design_report: Mapping[str, Any]) -> dict[str, Any]:
    if design_report.get("report_schema") != DESIGN_SCHEMA:
        raise ValueError("unrecognized toxicity representation design report")
    if design_report.get("evidence_scope") != "train_only":
        raise ValueError("toxicity advisory requires a train-only design report")
    if design_report.get("test_structures_used") is not False:
        raise ValueError("toxicity advisory cannot use test structures")
    if design_report.get("test_labels_used") is not False:
        raise ValueError("toxicity advisory cannot use test labels")

    audits = design_report.get("proposal_audits")
    audits = audits if isinstance(audits, list) else []
    deterministic_audit = next(
        (
            audit
            for audit in audits
            if isinstance(audit, Mapping)
            and isinstance(audit.get("proposal"), Mapping)
            and audit["proposal"].get("source") == "deterministic"
        ),
        {},
    )
    element_presence = design_report.get("observed_element_presence")
    element_presence = (
        dict(element_presence) if isinstance(element_presence, Mapping) else {}
    )
    out_of_schema = design_report.get("out_of_schema_element_presence")
    out_of_schema = dict(out_of_schema) if isinstance(out_of_schema, Mapping) else {}
    sample_count = int(design_report.get("training_sample_count") or 0)
    presence_fraction = {
        str(element): float(count) / sample_count if sample_count else 0.0
        for element, count in sorted(element_presence.items(), key=lambda item: str(item[0]))
    }
    return {
        "facts_schema": "mint-agent.toxicity-representation-advisory-facts.v1",
        "task": {
            "system_type": "small_molecule",
            "task_type": "toxicity_regression",
            "primary_metric": str(
                design_report.get("scientific_controls", {}).get("primary_metric", "PCC2")
            ),
        },
        "training_summary": {
            "sample_count": sample_count,
            "element_sample_presence": element_presence,
            "element_presence_fraction": presence_fraction,
            "out_of_schema_element_presence": out_of_schema,
        },
        "deterministic_reference": {
            "proposal": deterministic_audit.get("proposal"),
            "selected_pair_count": deterministic_audit.get("selected_pair_count"),
            "selected_pairs": deterministic_audit.get("selected_pairs"),
            "pair_cap_applied": deterministic_audit.get("pair_cap_applied"),
            "legacy_pair_geometry": deterministic_audit.get("legacy_pair_geometry"),
            "adaptive_pair_geometry": deterministic_audit.get(
                "adaptive_pair_geometry"
            ),
        },
        "allowed_values": {
            "include_hydrogen": [True, False],
            "support_fractions": list(ALLOWED_SUPPORT_FRACTIONS),
            "support_samples": list(ALLOWED_SUPPORT_SAMPLES),
            "pair_caps": list(ALLOWED_PAIR_CAPS),
            "distance_quantiles": list(ALLOWED_DISTANCE_QUANTILES),
            "margin_factors": list(ALLOWED_MARGIN_FACTORS),
            "fixed_point_count": [50],
            "bond_delta": [0.45],
            "probe_candidate_sizes": list(ALLOWED_PROBE_SIZES),
            "probe_target_quantile_bins": list(ALLOWED_TARGET_BINS),
            "probe_size_quantile_bins": list(ALLOWED_SIZE_BINS),
            "probe_min_pair_support": list(ALLOWED_PROBE_PAIR_SUPPORT),
        },
        "scientific_controls": {
            "proposal_limit": 2,
            "probe_strategy_limit": 1,
            "legacy_baseline_always_retained": True,
            "deterministic_candidate_always_retained": True,
            "eic_tau_profile_fixed_to_audited_legacy": True,
            "llm_proposals_are_hypotheses_not_evidence": True,
            "llm_probe_strategy_is_validated_by_python": True,
            "llm_used_for_numeric_ranking": False,
        },
        "information_boundary": {
            "sample_ids_included": False,
            "filesystem_paths_included": False,
            "test_structures_included": False,
            "test_labels_included": False,
            "test_metrics_included": False,
        },
    }


def generate_toxicity_advice(
    facts: Mapping[str, Any],
    *,
    model: str,
    api_key: str,
    timeout: float = 180.0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    advice, provenance, _body = request_structured_output(
        json.dumps(facts, indent=2, sort_keys=True),
        instructions=(
            "Act as a constrained molecular-representation advisor for a small-molecule "
            "toxicity regression task. Propose one or two scientifically distinct "
            "adaptive element-pair and distance-filtration settings using only the "
            "allowed values and anonymous train-only summary. Also propose one bounded "
            "train-only Probe strategy using only allowed candidate sizes and bin counts. "
            "Hydrogen may be retained "
            "when it plausibly carries toxicity-relevant structure, or omitted as a "
            "controlled ablation. Do not claim that a proposal will improve prediction. "
            "Do not request sample identities, paths, held-out data, test labels, or test "
            "metrics. The legacy and deterministic candidates are retained separately. "
            "Your proposals are hypotheses that deterministic Python validation, feature "
            "quality control, and probe experiments must evaluate."
        ),
        schema=_toxicity_advice_schema(),
        schema_name="mint_agent_toxicity_representation_advice",
        model=model,
        api_key=api_key,
        timeout=timeout,
    )
    provenance["input_hash"] = stable_hash(facts)
    return advice, provenance


def validate_toxicity_advice(
    advice: Mapping[str, Any],
) -> tuple[tuple[ToxicityAdaptiveProposal, ...], dict[str, Any], ToxicityProbePolicy]:
    if not isinstance(advice, Mapping):
        raise ValueError("toxicity representation advice must be an object")
    unknown = sorted(set(advice) - {"proposals", "probe_strategy", "overall_rationale"})
    if unknown:
        raise ValueError(f"unknown toxicity advisory fields: {unknown}")
    proposals = advice.get("proposals")
    if not isinstance(proposals, list) or not 1 <= len(proposals) <= 2:
        raise ValueError("toxicity advisory must contain one or two proposals")
    overall_rationale = advice.get("overall_rationale")
    if not isinstance(overall_rationale, str) or not overall_rationale.strip():
        raise ValueError("toxicity advisory requires an overall rationale")
    probe_strategy = advice.get("probe_strategy")
    if not isinstance(probe_strategy, Mapping):
        raise ValueError("toxicity advisory requires probe_strategy")
    probe_policy, normalized_probe = validate_probe_strategy(probe_strategy)

    normalized: list[ToxicityAdaptiveProposal] = []
    normalized_rows: list[dict[str, Any]] = []
    for item in proposals:
        if not isinstance(item, Mapping):
            raise ValueError("each toxicity advisory proposal must be an object")
        rationale = item.get("rationale")
        if not isinstance(rationale, str) or not rationale.strip():
            raise ValueError("each toxicity advisory proposal requires a rationale")
        reported_source = item.get("source")
        if reported_source is not None and reported_source != "llm":
            raise ValueError("normalized toxicity advisory source must be llm")
        proposal_fields = {
            key: value
            for key, value in item.items()
            if key not in {"rationale", "source"}
        }
        proposal = ToxicityAdaptiveProposal.from_mapping(
            proposal_fields, source="llm"
        )
        normalized.append(proposal)
        normalized_rows.append({**asdict(proposal), "rationale": rationale.strip()})
    ids = [proposal.proposal_id for proposal in normalized]
    if len(set(ids)) != len(ids):
        raise ValueError("toxicity advisory proposal ids must be unique")
    return tuple(normalized), {
        "proposals": normalized_rows,
        "probe_strategy": normalized_probe,
        "overall_rationale": overall_rationale.strip(),
    }, probe_policy


def validate_probe_strategy(
    raw: Mapping[str, Any],
) -> tuple[ToxicityProbePolicy, dict[str, Any]]:
    allowed = {
        "candidate_sizes",
        "target_quantile_bins",
        "size_quantile_bins",
        "min_pair_support",
        "rationale",
    }
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"unknown toxicity probe_strategy fields: {unknown}")
    candidate_sizes = raw.get("candidate_sizes")
    if not isinstance(candidate_sizes, list) or not candidate_sizes:
        raise ValueError("toxicity probe_strategy requires candidate_sizes")
    sizes = tuple(int(size) for size in candidate_sizes)
    if any(size not in ALLOWED_PROBE_SIZES for size in sizes):
        raise ValueError("unsupported toxicity probe candidate size")
    if tuple(sorted(sizes)) != sizes or len(set(sizes)) != len(sizes):
        raise ValueError("toxicity probe candidate sizes must be unique and increasing")
    target_bins = int(raw.get("target_quantile_bins"))
    size_bins = int(raw.get("size_quantile_bins"))
    min_pair_support = int(raw.get("min_pair_support"))
    if target_bins not in ALLOWED_TARGET_BINS:
        raise ValueError("unsupported toxicity probe target_quantile_bins")
    if size_bins not in ALLOWED_SIZE_BINS:
        raise ValueError("unsupported toxicity probe size_quantile_bins")
    if min_pair_support not in ALLOWED_PROBE_PAIR_SUPPORT:
        raise ValueError("unsupported toxicity probe min_pair_support")
    rationale = raw.get("rationale")
    if not isinstance(rationale, str) or not rationale.strip():
        raise ValueError("toxicity probe_strategy requires a rationale")
    policy = ToxicityProbePolicy(
        candidate_sizes=sizes,
        target_quantile_bins=target_bins,
        size_quantile_bins=size_bins,
        min_pair_support=min_pair_support,
    )
    normalized = {
        "candidate_sizes": list(policy.candidate_sizes),
        "target_quantile_bins": policy.target_quantile_bins,
        "size_quantile_bins": policy.size_quantile_bins,
        "min_pair_support": policy.min_pair_support,
        "rationale": rationale.strip(),
    }
    return policy, normalized


def run_toxicity_advisory(
    design_report: Mapping[str, Any],
    *,
    model: str,
    api_key: str,
    planner: AdviceGenerator | None = None,
    timeout: float = 180.0,
    cache_dir: str | Path | None = None,
) -> tuple[dict[str, Any], tuple[ToxicityAdaptiveProposal, ...]]:
    facts = toxicity_advisory_facts(design_report)
    input_hash = stable_hash(facts)
    cached = _load_cached_advisory(cache_dir, input_hash)
    if cached is not None:
        return cached
    generator = planner or generate_toxicity_advice
    try:
        advice, provenance = generator(
            facts, model=model, api_key=api_key, timeout=timeout
        )
        proposals, validated, probe_policy = validate_toxicity_advice(advice)
        status = "VALIDATED"
        errors: list[str] = []
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        proposals = ()
        probe_policy = ToxicityProbePolicy()
        validated = {
            "proposals": [],
            "probe_strategy": _probe_policy_to_advice(probe_policy, ""),
            "overall_rationale": "",
        }
        provenance = {"used": True, "model_requested": model}
        status = "FALLBACK"
        errors = [str(exc)]
    artifact = {
        "report_schema": ADVISORY_SCHEMA,
        "status": status,
        "input_hash": input_hash,
        "facts": facts,
        "advice": validated,
        "validation": {"passed": not errors, "errors": errors},
        "llm": {
            **dict(provenance),
            "used_for_numeric_ranking": False,
            "input_policy": "summary_only_no_paths_ids_or_test_evidence",
        },
        "applied_to_execution": status == "VALIDATED",
        "cache_hit": False,
    }
    if status == "VALIDATED":
        _write_cached_advisory(cache_dir, input_hash, artifact)
    return artifact, proposals


def probe_policy_from_advisory_artifact(path: str | Path) -> ToxicityProbePolicy:
    artifact = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if (
        not isinstance(artifact, Mapping)
        or artifact.get("report_schema") != ADVISORY_SCHEMA
        or artifact.get("status") != "VALIDATED"
    ):
        raise ValueError("toxicity advisory artifact is not a validated advisory")
    advice = artifact.get("advice")
    if not isinstance(advice, Mapping):
        raise ValueError("toxicity advisory artifact has no advice")
    probe_strategy = advice.get("probe_strategy")
    if not isinstance(probe_strategy, Mapping):
        raise ValueError("toxicity advisory artifact has no probe_strategy")
    policy, _normalized = validate_probe_strategy(probe_strategy)
    return policy


def _probe_policy_to_advice(
    policy: ToxicityProbePolicy, rationale: str
) -> dict[str, Any]:
    return {
        "candidate_sizes": list(policy.candidate_sizes),
        "target_quantile_bins": policy.target_quantile_bins,
        "size_quantile_bins": policy.size_quantile_bins,
        "min_pair_support": policy.min_pair_support,
        "rationale": rationale,
    }


def write_toxicity_advisory(
    *,
    artifact: Mapping[str, Any],
    proposals: tuple[ToxicityAdaptiveProposal, ...],
    artifact_path: str | Path,
    proposals_path: str | Path,
) -> None:
    artifact_destination = Path(artifact_path).expanduser().resolve()
    proposals_destination = Path(proposals_path).expanduser().resolve()
    artifact_destination.parent.mkdir(parents=True, exist_ok=True)
    proposals_destination.parent.mkdir(parents=True, exist_ok=True)
    artifact_destination.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    rows = []
    for proposal in proposals:
        row = asdict(proposal)
        row.pop("source", None)
        rows.append(row)
    proposals_destination.write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _load_cached_advisory(
    cache_dir: str | Path | None, input_hash: str
) -> tuple[dict[str, Any], tuple[ToxicityAdaptiveProposal, ...]] | None:
    path = _advisory_cache_path(cache_dir, input_hash)
    if path is None or not path.is_file():
        return None
    try:
        artifact = json.loads(path.read_text(encoding="utf-8"))
        if (
            not isinstance(artifact, dict)
            or artifact.get("report_schema") != ADVISORY_SCHEMA
            or artifact.get("status") != "VALIDATED"
            or artifact.get("input_hash") != input_hash
        ):
            return None
        validation = artifact.get("validation")
        if not isinstance(validation, Mapping) or validation.get("passed") is not True:
            return None
        advice = artifact.get("advice")
        if not isinstance(advice, Mapping):
            return None
        proposals, normalized, _probe_policy = validate_toxicity_advice(advice)
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None
    reused = dict(artifact)
    reused["advice"] = normalized
    reused["cache_hit"] = True
    reused["applied_to_execution"] = True
    return reused, proposals


def _write_cached_advisory(
    cache_dir: str | Path | None,
    input_hash: str,
    artifact: Mapping[str, Any],
) -> None:
    path = _advisory_cache_path(cache_dir, input_hash)
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


def _advisory_cache_path(
    cache_dir: str | Path | None, input_hash: str
) -> Path | None:
    if cache_dir is None:
        return None
    return (
        Path(cache_dir).expanduser().resolve()
        / "toxicity_representation"
        / f"{input_hash}.json"
    )


def _toxicity_advice_schema() -> dict[str, Any]:
    proposal = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "proposal_id",
            "include_hydrogen",
            "min_element_support_fraction",
            "min_pair_support_fraction",
            "min_support_samples",
            "max_pair_channels",
            "local_distance_quantile",
            "dataset_distance_quantile",
            "margin_factor",
            "fixed_point_count",
            "bond_delta",
            "rationale",
        ],
        "properties": {
            "proposal_id": {"type": "string"},
            "include_hydrogen": {"type": "boolean"},
            "min_element_support_fraction": {
                "type": "number",
                "enum": list(ALLOWED_SUPPORT_FRACTIONS),
            },
            "min_pair_support_fraction": {
                "type": "number",
                "enum": list(ALLOWED_SUPPORT_FRACTIONS),
            },
            "min_support_samples": {
                "type": "integer",
                "enum": list(ALLOWED_SUPPORT_SAMPLES),
            },
            "max_pair_channels": {
                "type": "integer",
                "enum": list(ALLOWED_PAIR_CAPS),
            },
            "local_distance_quantile": {
                "type": "number",
                "enum": list(ALLOWED_DISTANCE_QUANTILES),
            },
            "dataset_distance_quantile": {
                "type": "number",
                "enum": list(ALLOWED_DISTANCE_QUANTILES),
            },
            "margin_factor": {
                "type": "number",
                "enum": list(ALLOWED_MARGIN_FACTORS),
            },
            "fixed_point_count": {"type": "integer", "enum": [50]},
            "bond_delta": {"type": "number", "enum": [0.45]},
            "rationale": {"type": "string"},
        },
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["proposals", "probe_strategy", "overall_rationale"],
        "properties": {
            "proposals": {"type": "array", "items": proposal},
            "probe_strategy": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "candidate_sizes",
                    "target_quantile_bins",
                    "size_quantile_bins",
                    "min_pair_support",
                    "rationale",
                ],
                "properties": {
                    "candidate_sizes": {
                        "type": "array",
                        "items": {
                            "type": "integer",
                            "enum": list(ALLOWED_PROBE_SIZES),
                        },
                    },
                    "target_quantile_bins": {
                        "type": "integer",
                        "enum": list(ALLOWED_TARGET_BINS),
                    },
                    "size_quantile_bins": {
                        "type": "integer",
                        "enum": list(ALLOWED_SIZE_BINS),
                    },
                    "min_pair_support": {
                        "type": "integer",
                        "enum": list(ALLOWED_PROBE_PAIR_SUPPORT),
                    },
                    "rationale": {"type": "string"},
                },
            },
            "overall_rationale": {"type": "string"},
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate bounded LLM toxicity representation proposals."
    )
    parser.add_argument("--design-report", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--proposals", type=Path, required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--cache-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    model = args.model or os.environ.get("OPENAI_MODEL")
    if not model:
        raise ValueError("toxicity advisory requires --model or OPENAI_MODEL")
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise ValueError(f"toxicity advisory requires {args.api_key_env}")
    report = json.loads(args.design_report.read_text(encoding="utf-8"))
    if not isinstance(report, Mapping):
        raise ValueError("toxicity design report must contain a JSON object")
    artifact, proposals = run_toxicity_advisory(
        report,
        model=model,
        api_key=api_key,
        timeout=args.timeout,
        cache_dir=args.cache_dir,
    )
    write_toxicity_advisory(
        artifact=artifact,
        proposals=proposals,
        artifact_path=args.artifact,
        proposals_path=args.proposals,
    )
    print(
        f"status={artifact['status']} proposals={len(proposals)} "
        f"artifact={args.artifact}"
    )
    return 0 if artifact["status"] == "VALIDATED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
