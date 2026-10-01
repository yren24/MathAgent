from __future__ import annotations

import argparse
import json
import operator
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Callable, Mapping, TypedDict

from mint_scout.invariants.manifest import stable_hash
from mint_scout.openai_responses import request_structured_output


EXPLANATION_SCHEMA = "mint-agent.llm-explanation.v1"
ExplanationGenerator = Callable[..., tuple[dict[str, Any], dict[str, Any]]]


class LLMExplanationState(TypedDict, total=False):
    summary_path: str
    summary: dict[str, Any]
    explanation: dict[str, Any]
    artifact: dict[str, Any]
    output_path: str
    status: str
    errors: list[str]
    node_trace: Annotated[list[str], operator.add]


@dataclass(frozen=True)
class LLMExplanationContext:
    model: str
    api_key: str
    language: str = "English"
    output_path: Path | None = None
    generator: ExplanationGenerator | None = None


def build_llm_explanation_graph(
    context: LLMExplanationContext, *, checkpointer=None
):
    try:
        from langgraph.graph import END, START, StateGraph
    except ImportError as exc:
        raise RuntimeError(
            "LLM explanation orchestration requires the optional mint-agent[agent] dependencies"
        ) from exc

    def load_summary_node(state: LLMExplanationState) -> dict[str, Any]:
        path = Path(state.get("summary_path", "")).expanduser().resolve()
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return {
                "status": "INVALID_REPORT",
                "errors": [str(exc)],
                "node_trace": ["load_summary_node"],
            }
        if (
            not isinstance(payload, dict)
            or payload.get("report_schema") != "mint-agent.pipeline-report.v1"
        ):
            return {
                "status": "INVALID_REPORT",
                "errors": ["summary must use mint-agent.pipeline-report.v1"],
                "node_trace": ["load_summary_node"],
            }
        return {
            "summary": payload,
            "status": "REPORT_LOADED",
            "node_trace": ["load_summary_node"],
        }

    def explanation_node(state: LLMExplanationState) -> dict[str, Any]:
        generator = context.generator or explain_pipeline_summary
        try:
            explanation, provenance = generator(
                state["summary"],
                model=context.model,
                api_key=context.api_key,
                language=context.language,
            )
        except (RuntimeError, ValueError) as exc:
            return {
                "status": "LLM_ERROR",
                "errors": [str(exc)],
                "node_trace": ["explanation_node"],
            }
        artifact = {
            "report_schema": EXPLANATION_SCHEMA,
            "status": "COMPLETE",
            "source_summary_hash": stable_hash(state["summary"]),
            "source_lifecycle_id": state["summary"].get("lifecycle_id"),
            "language": context.language,
            "explanation": explanation,
            "llm": {
                **provenance,
                "used_for_numeric_decisions": False,
                "input_policy": "paths_and_sample_ids_omitted",
            },
            "scientific_control": {
                "source_artifacts_immutable": True,
                "metrics_recomputed_by_llm": False,
                "selection_changed_by_llm": False,
            },
        }
        return {
            "explanation": explanation,
            "artifact": artifact,
            "status": "EXPLANATION_READY",
            "node_trace": ["explanation_node"],
        }

    def write_explanation_node(state: LLMExplanationState) -> dict[str, Any]:
        summary_path = Path(state["summary_path"]).expanduser().resolve()
        destination = context.output_path or (
            summary_path.parent / f"llm_explanation_{_safe_name(context.model)}.json"
        )
        destination = destination.expanduser().resolve()
        if destination.exists():
            return {
                "status": "OUTPUT_EXISTS",
                "errors": [f"refusing to overwrite existing explanation: {destination}"],
                "node_trace": ["write_explanation_node"],
            }
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(state["artifact"], indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return {
            "output_path": str(destination),
            "status": "COMPLETE",
            "node_trace": ["write_explanation_node"],
        }

    def after_load(state: LLMExplanationState) -> str:
        return "explain" if state.get("status") == "REPORT_LOADED" else "end"

    def after_explanation(state: LLMExplanationState) -> str:
        return "write" if state.get("status") == "EXPLANATION_READY" else "end"

    builder = StateGraph(LLMExplanationState)
    builder.add_node("load_summary", load_summary_node)
    builder.add_node("explanation", explanation_node)
    builder.add_node("write_explanation", write_explanation_node)
    builder.add_edge(START, "load_summary")
    builder.add_conditional_edges(
        "load_summary", after_load, {"explain": "explanation", "end": END}
    )
    builder.add_conditional_edges(
        "explanation",
        after_explanation,
        {"write": "write_explanation", "end": END},
    )
    builder.add_edge("write_explanation", END)
    return builder.compile(checkpointer=checkpointer)


def explain_pipeline_summary(
    summary: Mapping[str, Any],
    *,
    model: str,
    api_key: str,
    language: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    facts = _explanation_facts(summary)
    explanation, provenance, _body = request_structured_output(
        json.dumps(facts, indent=2, sort_keys=True),
        instructions=(
            f"Explain this completed mint-agent run in {language}. Use only facts in "
            "the supplied JSON. Copy reported metrics without recomputing them. Do not "
            "change the selected invariant subset, scientific target, representation, "
            "or warnings. Clearly distinguish probe evidence from full evaluation. "
            "Do not invent missing facts or numerical recommendations."
        ),
        schema=_explanation_output_schema(),
        schema_name="mint_agent_pipeline_explanation",
        model=model,
        api_key=api_key,
    )
    provenance["input_hash"] = stable_hash(facts)
    return explanation, provenance


def public_explanation_result(state: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: state[key]
        for key in (
            "status",
            "output_path",
            "explanation",
            "errors",
            "node_trace",
        )
        if key in state
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate a non-numeric LLM explanation from a pipeline summary."
    )
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--language", default="English")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    model = args.model or os.environ.get("OPENAI_MODEL")
    if not model:
        raise ValueError("LLM explanation requires --model or OPENAI_MODEL")
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise ValueError(
            f"LLM explanation requires API credentials in {args.api_key_env}"
        )
    context = LLMExplanationContext(
        model=model,
        api_key=api_key,
        language=args.language,
        output_path=args.output,
    )
    state = build_llm_explanation_graph(context).invoke(
        {"summary_path": str(args.summary)}
    )
    result = public_explanation_result(state)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("status") == "COMPLETE" else 2


def _explanation_facts(summary: Mapping[str, Any]) -> dict[str, Any]:
    scientific = summary.get("scientific_summary", {})
    if not isinstance(scientific, Mapping):
        scientific = {}
    scout = scientific.get("scout", {})
    if not isinstance(scout, Mapping):
        scout = {}
    candidates = scout.get("candidates", [])
    top_candidates = []
    if isinstance(candidates, list):
        for candidate in candidates[:5]:
            if not isinstance(candidate, Mapping):
                continue
            top_candidates.append(
                {
                    key: candidate.get(key)
                    for key in (
                        "subset_id",
                        "invariants",
                        "primary_score",
                        "bootstrap_ci_low",
                        "bootstrap_ci_high",
                        "top_tier",
                        "top1_frequency",
                        "median_rank",
                        "estimated_full_acquisition_wall_seconds",
                    )
                }
            )
    preparation = scientific.get("preparation", {})
    if not isinstance(preparation, Mapping):
        preparation = {}
    return {
        "lifecycle_id": summary.get("lifecycle_id"),
        "lifecycle_status": summary.get("lifecycle_status"),
        "dataset_id": summary.get("dataset_id"),
        "requested_invariants": summary.get("requested_invariants"),
        "scientific_control": summary.get("scientific_control"),
        "preparation": {
            "provider": preparation.get("provider"),
            "sample_count": preparation.get("sample_count"),
        },
        "representation": scientific.get("representation"),
        "probe_feature_qc": scientific.get("probe_feature_qc"),
        "probe_filtration_audit": scientific.get("probe_filtration_audit"),
        "full_feature_qc": scientific.get("feature_qc"),
        "full_filtration_audit": scientific.get("filtration_audit"),
        "scout": {
            "status": scout.get("status"),
            "sample_count": scout.get("sample_count"),
            "primary_metric": scout.get("primary_metric"),
            "target_value": scout.get("target_value"),
            "target_source": scout.get("target_source"),
            "top_candidates": top_candidates,
        },
        "evaluation": scientific.get("evaluation"),
        "execution": {
            "stage_count": summary.get("stage_count"),
            "job_count": len(summary.get("jobs", [])),
        },
    }


def _explanation_output_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "overview",
            "selection_reason",
            "quality_findings",
            "limitations",
            "next_steps",
        ],
        "properties": {
            "overview": {"type": "string"},
            "selection_reason": {"type": "string"},
            "quality_findings": {
                "type": "array",
                "items": {"type": "string"},
            },
            "limitations": {
                "type": "array",
                "items": {"type": "string"},
            },
            "next_steps": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
    }


def _safe_name(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-.")
    return normalized or "model"


if __name__ == "__main__":
    raise SystemExit(main())
