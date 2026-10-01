from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from uuid import uuid4

from mint_scout.agent.graph import AgentGraphContext, build_agent_graph
from mint_scout.agent.llm_scientific import LLMScientificContext


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the cache-first mint-agent LangGraph workflow."
    )
    parser.add_argument("--task-config", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--invariant", action="append", required=True)
    parser.add_argument("--evidence-scope", default="full_train")
    parser.add_argument("--execution-profile", type=Path, default=None)
    parser.add_argument("--representation-spec", default=None)
    parser.add_argument("--sample-id-file", default=None)
    parser.add_argument("--qc-config", default="configs/scout/v1.yaml")
    parser.add_argument("--audit-config", default="configs/scout/v1.yaml")
    parser.add_argument("--feature-diagnostic-top-k", type=int, default=10)
    parser.add_argument("--scout-config", default="configs/scout/v1.yaml")
    parser.add_argument("--user-target", type=float, default=None)
    parser.add_argument("--gbt-config", default="configs/gbt/plbind_adaptive_gbt.yaml")
    parser.add_argument("--thread-id", default=None)
    parser.add_argument(
        "--llm-scientific-mode",
        choices=("disabled", "shadow", "advisory"),
        default="disabled",
    )
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--llm-api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--llm-env-file", type=Path, default=None)
    parser.add_argument("--llm-cache-dir", type=Path, default=None)
    parser.add_argument("--llm-max-candidates", type=int, default=10)
    parser.add_argument("--llm-timeout-seconds", type=float, default=180.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-mermaid", type=Path, default=None)
    args = parser.parse_args(argv)

    try:
        from langgraph.checkpoint.memory import InMemorySaver
    except ImportError as exc:
        raise RuntimeError(
            "Install the optional mint-agent[agent] dependencies"
        ) from exc

    thread_id = args.thread_id or f"mint-agent-{uuid4()}"
    llm_context = _llm_scientific_context(args)
    graph = build_agent_graph(
        AgentGraphContext(
            task_config=args.task_config,
            registry=args.registry,
            execution_profile=args.execution_profile,
            llm_scientific=llm_context,
        ),
        checkpointer=InMemorySaver(),
    )
    state = graph.invoke(
        {
            "run_id": thread_id,
            "request": {
                "dataset_id": args.dataset_id,
                "invariants": [value.upper() for value in args.invariant],
                "evidence_scope": args.evidence_scope,
                "representation_spec": args.representation_spec,
                "sample_id_file": args.sample_id_file,
                "qc_config": args.qc_config,
                "audit_config": args.audit_config,
                "feature_diagnostic_top_k": args.feature_diagnostic_top_k,
                "scout_config": args.scout_config,
                "user_target": args.user_target,
                "gbt_config": args.gbt_config,
            },
            "node_trace": [],
        },
        {"configurable": {"thread_id": thread_id}},
    )
    report = dict(state["final_report"])
    task_summary = state.get("task", {})
    report.update(
        {
            "dataset_id": task_summary.get("dataset_id", args.dataset_id),
            "evidence_scope": state["request"]["evidence_scope"],
            "invariants": list(state["request"]["invariants"]),
            "node_trace": list(state.get("node_trace", [])),
            "artifact_snapshot_count": len(state.get("artifacts", [])),
            "sample_count": state.get("evaluation", {}).get("sample_count"),
            "sample_order_hash": state.get("evaluation", {}).get("sample_order_hash"),
            "preparation_jobs": state.get("preparation_jobs", []),
            "feature_jobs": state.get("feature_jobs", []),
            "qc_jobs": state.get("qc_jobs", []),
            "filtration_audit_jobs": state.get("filtration_audit_jobs", []),
            "feature_diagnostic_jobs": state.get("feature_diagnostic_jobs", []),
            "scout_jobs": state.get("scout_jobs", []),
            "evaluation_jobs": state.get("evaluation_jobs", []),
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if args.output_mermaid is not None:
        args.output_mermaid.parent.mkdir(parents=True, exist_ok=True)
        args.output_mermaid.write_text(
            graph.get_graph().draw_mermaid() + "\n", encoding="utf-8"
        )
    print(
        f"thread_id={thread_id} status={report.get('status')} "
        f"trace={','.join(state.get('node_trace', []))} output={args.output}"
    )
    return _exit_code(str(report.get("status") or ""))


def _llm_scientific_context(args: argparse.Namespace) -> LLMScientificContext:
    if args.llm_scientific_mode == "disabled":
        return LLMScientificContext(mode="disabled")
    file_environment = _read_llm_env_file(args.llm_env_file)
    model = (
        args.llm_model
        or file_environment.get("OPENAI_MODEL")
        or os.environ.get("OPENAI_MODEL")
    )
    api_key = file_environment.get(args.llm_api_key_env) or os.environ.get(
        args.llm_api_key_env
    )
    if not model:
        raise ValueError(
            "enabled LLM scientific mode requires --llm-model or OPENAI_MODEL"
        )
    if not api_key:
        raise ValueError(f"enabled LLM scientific mode requires {args.llm_api_key_env}")
    cache_dir = args.llm_cache_dir or (
        args.registry.expanduser().resolve().parent / "llm_scientific"
    )
    return LLMScientificContext(
        mode=args.llm_scientific_mode,
        model=model,
        api_key=api_key,
        cache_dir=cache_dir,
        max_candidates=args.llm_max_candidates,
        timeout_seconds=args.llm_timeout_seconds,
    )


def _read_llm_env_file(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    resolved = path.expanduser().resolve()
    values: dict[str, str] = {}
    for line_number, raw_line in enumerate(
        resolved.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            raise ValueError(
                f"invalid LLM environment entry at {resolved}:{line_number}"
            )
        name, value = line.split("=", 1)
        name = name.strip()
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            raise ValueError(
                f"invalid LLM environment name at {resolved}:{line_number}"
            )
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        values[name] = value
    return values


def _exit_code(status: str) -> int:
    if status == "COMPLETE" or status.startswith("NEEDS_"):
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
