from __future__ import annotations

import argparse
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


GRAPH_SCHEMA = "mint-agent.toxicity-agent-graph.v1"


@dataclass(frozen=True)
class ToxicityGraphContext:
    environment: Mapping[str, str]


def build_toxicity_agent_graph(context: ToxicityGraphContext, *, checkpointer=None):
    try:
        from langgraph.graph import END, START, StateGraph
    except ImportError as exc:
        raise RuntimeError(
            "Toxicity agent workflow requires the optional LangGraph dependency"
        ) from exc

    def context_node(state: dict[str, Any]) -> dict[str, Any]:
        env = _toxicity_environment(context.environment)
        return {
            "execute": bool(state.get("execute", False)),
            "environment": env,
            "dataset_id": env["DATASET_ID"],
            "run_id": env.get("RUN_ID", ""),
            "llm_scientific_mode": env.get("LLM_SCIENTIFIC_MODE", "disabled"),
            "llm_advisory_enabled": env.get("LLM_ENABLED") == "1",
            "node_trace": ["context_node"],
        }

    def submit_chain_node(state: dict[str, Any]) -> dict[str, Any]:
        passthrough = {
            "environment": state.get("environment", {}),
            "dataset_id": state.get("dataset_id"),
            "run_id": state.get("run_id"),
            "llm_scientific_mode": state.get("llm_scientific_mode"),
            "llm_advisory_enabled": state.get("llm_advisory_enabled", False),
        }
        if not state.get("execute", False):
            return {
                **passthrough,
                "status": "PLANNED",
                "next_actions": ["Submit the toxicity dependency chain."],
                "node_trace": ["submit_chain_node"],
            }
        receipt = submit_toxicity_dependency_chain(state["environment"])
        return {
            **passthrough,
            "status": "SUBMITTED",
            "jobs": receipt,
            "node_trace": ["submit_chain_node"],
        }

    def report_node(state: dict[str, Any]) -> dict[str, Any]:
        return {
            "final_report": {
                "report_schema": GRAPH_SCHEMA,
                "status": state.get("status", "UNKNOWN"),
                "dataset_id": state.get("dataset_id"),
                "run_id": state.get("run_id"),
                "llm_scientific_mode": state.get("llm_scientific_mode"),
                "llm_advisory_enabled": state.get("llm_advisory_enabled", False),
                "llm_probe_strategy_can_affect_probe_policy": bool(
                    state.get("llm_advisory_enabled", False)
                ),
                "jobs": state.get("jobs"),
                "next_actions": state.get("next_actions", []),
                "node_trace": state.get("node_trace", []),
            },
            "node_trace": ["report_node"],
        }

    builder = StateGraph(dict)
    builder.add_node("context", context_node)
    builder.add_node("submit_chain", submit_chain_node)
    builder.add_node("report", report_node)
    builder.add_edge(START, "context")
    builder.add_edge("context", "submit_chain")
    builder.add_edge("submit_chain", "report")
    builder.add_edge("report", END)
    return builder.compile(checkpointer=checkpointer)


def submit_toxicity_dependency_chain(environment: Mapping[str, str]) -> dict[str, Any]:
    env = _toxicity_environment(environment)
    _prepare_directories(env)
    common = {
        "MINT_AGENT_ROOT": env["MINT_AGENT_ROOT"],
        "MANIFEST_PATH": env["MANIFEST_PATH"],
        "MOLECULE_DIRS": env["MOLECULE_DIRS"],
        "DATASET_ID": env["DATASET_ID"],
        "FEATURE_ROOT": env["FEATURE_ROOT"],
        "LEGACY_ROOT": env["LEGACY_ROOT"],
        "GBT_CONFIG": env["GBT_CONFIG"],
        "PRIMARY_METRIC": env["PRIMARY_METRIC"],
        "TASK_CONFIG": env["TASK_CONFIG"],
    }

    design_job = _submit(
        env,
        common,
        "scripts/slurm/run_toxicity_design.sbatch",
        extra={"OUTPUT_ROOT": env["DESIGN_ROOT"]},
    )

    effective_design_root = env["DESIGN_ROOT"]
    if env["LLM_ENABLED"] == "1":
        advisory_job = _submit(
            env,
            common,
            "scripts/slurm/run_toxicity_advisory.sbatch",
            dependency=f"afterok:{design_job}",
            extra={
                "BASE_DESIGN_ROOT": env["DESIGN_ROOT"],
                "OUTPUT_ROOT": env["ADVISORY_ROOT"],
                "LLM_MODEL": env["LLM_MODEL"],
                "LLM_ENV_FILE": env["LLM_ENV_FILE"],
                "LLM_CACHE_DIR": env["LLM_CACHE_DIR"],
                "PRIMARY_METRIC": env["PRIMARY_METRIC"],
            },
        )
        effective_design_root = env["ADVISORY_ROOT"]
    else:
        advisory_job = design_job

    feature_plan_job = _submit(
        env,
        common,
        "scripts/slurm/run_toxicity_feature_plan.sbatch",
        dependency=f"afterok:{advisory_job}",
        extra={
            "DESIGN_ROOT": effective_design_root,
            "OUTPUT_ROOT": env["FEATURE_PLAN_ROOT"],
        },
    )
    probe_extra = {
        "MANIFEST_ROOT": env["DESIGN_ROOT"],
        "DESIGN_ROOT": effective_design_root,
        "OUTPUT_ROOT": env["PROBE_ROOT"],
    }
    if env["LLM_ENABLED"] == "1":
        probe_extra["ADVISORY_ARTIFACT"] = (
            f"{env['ADVISORY_ROOT']}/toxicity_llm_advisory.json"
        )
    probe_job = _submit(
        env,
        common,
        "scripts/slurm/run_toxicity_probe_selection.sbatch",
        dependency=f"afterok:{advisory_job}",
        extra=probe_extra,
    )
    probe_feature_job = _submit(
        env,
        common,
        "scripts/slurm/run_toxicity_probe_feature_array.sbatch",
        dependency=f"afterok:{feature_plan_job}:{probe_job}",
        array="0-79%20",
        extra={
            "BASE_DESIGN_ROOT": env["DESIGN_ROOT"],
            "PROBE_ROOT": env["PROBE_ROOT"],
            "PLAN_ROOT": env["FEATURE_PLAN_ROOT"],
            "FEATURE_ROOT": env["FEATURE_ROOT"],
        },
    )
    probe_qc_job = _submit(
        env,
        common,
        "scripts/slurm/run_toxicity_feature_qc.sbatch",
        dependency=f"afterok:{probe_feature_job}",
        extra={
            "FEATURE_ROOT": env["FEATURE_PLAN_ROOT"],
            "PLAN_PATH": f"{env['FEATURE_PLAN_ROOT']}/toxicity_feature_plan.json",
            "QC_OUTPUT": f"{env['FEATURE_PLAN_ROOT']}/toxicity_feature_qc.json",
            "EXPECTED_SAMPLE_COUNT": "",
        },
    )
    representation_job = _submit(
        env,
        common,
        "scripts/slurm/run_toxicity_representation_selection.sbatch",
        dependency=f"afterok:{probe_qc_job}",
        extra={
            "DESIGN_ROOT": effective_design_root,
            "PROBE_ROOT": env["PROBE_ROOT"],
            "FEATURE_ROOT": env["FEATURE_PLAN_ROOT"],
        },
    )
    repair_job = _submit(
        env,
        common,
        "scripts/slurm/run_toxicity_filtration_repair.sbatch",
        dependency=f"afterok:{representation_job}",
        extra={
            "PROBE_ROOT": env["PROBE_ROOT"],
            "FEATURE_ROOT": env["FEATURE_PLAN_ROOT"],
            "REPAIR_ROOT": env["REPAIR_ROOT"],
        },
    )
    repair_feature_job = _submit(
        env,
        common,
        "scripts/slurm/run_toxicity_repair_feature_array.sbatch",
        dependency=f"afterok:{repair_job}",
        array="0-9%10",
        extra={
            "BASE_DESIGN_ROOT": env["DESIGN_ROOT"],
            "PROBE_ROOT": env["PROBE_ROOT"],
            "REPAIR_ROOT": env["REPAIR_ROOT"],
            "FEATURE_ROOT": env["FEATURE_ROOT"],
        },
    )
    repair_qc_job = _submit(
        env,
        common,
        "scripts/slurm/run_toxicity_feature_qc.sbatch",
        dependency=f"afterok:{repair_feature_job}",
        extra={
            "FEATURE_ROOT": env["REPAIR_ROOT"],
            "PLAN_PATH": f"{env['REPAIR_ROOT']}/toxicity_repair_feature_plan.json",
            "QC_OUTPUT": f"{env['REPAIR_ROOT']}/toxicity_repair_feature_qc.json",
            "EXPECTED_SAMPLE_COUNT": "",
        },
    )
    probe_gbt_job = _submit(
        env,
        common,
        "scripts/slurm/run_toxicity_probe_gbt_array.sbatch",
        dependency=f"afterok:{repair_qc_job}",
        array="0-4%5",
        extra={
            "PROBE_ROOT": env["PROBE_ROOT"],
            "REPAIR_ROOT": env["REPAIR_ROOT"],
            "GBT_ROOT": f"{env['REPAIR_ROOT']}/gbt",
        },
    )
    probe_combine_job = _submit(
        env,
        common,
        "scripts/slurm/run_toxicity_probe_gbt_combine.sbatch",
        dependency=f"afterok:{probe_gbt_job}",
        extra={
            "PROBE_ROOT": env["PROBE_ROOT"],
            "REPAIR_ROOT": env["REPAIR_ROOT"],
            "GBT_ROOT": f"{env['REPAIR_ROOT']}/gbt",
        },
    )
    final_plan_job = _submit(
        env,
        common,
        "scripts/slurm/run_toxicity_final_plan.sbatch",
        dependency=f"afterok:{probe_combine_job}",
        extra={
            "PROBE_ROOT": env["PROBE_ROOT"],
            "REPAIR_ROOT": env["REPAIR_ROOT"],
            "FINAL_ROOT": env["FINAL_ROOT"],
        },
    )
    final_feature_job = _submit(
        env,
        common,
        "scripts/slurm/run_toxicity_final_feature_array.sbatch",
        dependency=f"afterok:{final_plan_job}",
        array="0-9%10",
        extra={
            "DESIGN_ROOT": env["DESIGN_ROOT"],
            "FINAL_ROOT": env["FINAL_ROOT"],
            "FEATURE_ROOT": env["FEATURE_ROOT"],
        },
    )
    final_gbt_job = _submit(
        env,
        common,
        "scripts/slurm/run_toxicity_final_gbt_array.sbatch",
        dependency=f"afterok:{final_feature_job}",
        array="0-4%5",
        extra={"FINAL_ROOT": env["FINAL_ROOT"]},
    )
    final_combine_job = _submit(
        env,
        common,
        "scripts/slurm/run_toxicity_final_gbt_combine.sbatch",
        dependency=f"afterok:{final_gbt_job}",
        extra={"FINAL_ROOT": env["FINAL_ROOT"]},
    )

    receipt = {
        "report_schema": "mint-agent.toxicity-jobs.v1",
        "dataset_id": env["DATASET_ID"],
        "run_id": env.get("RUN_ID", ""),
        "launch_job_id": env.get("SLURM_JOB_ID", ""),
        "llm_scientific_mode": env.get("LLM_SCIENTIFIC_MODE", "disabled"),
        "llm_probe_strategy_enabled": env["LLM_ENABLED"] == "1",
        "design_job": design_job,
        "advisory_job": advisory_job,
        "feature_plan_job": feature_plan_job,
        "probe_job": probe_job,
        "probe_feature_job": probe_feature_job,
        "probe_qc_job": probe_qc_job,
        "representation_job": representation_job,
        "repair_job": repair_job,
        "repair_feature_job": repair_feature_job,
        "repair_qc_job": repair_qc_job,
        "probe_gbt_job": probe_gbt_job,
        "probe_combine_job": probe_combine_job,
        "final_plan_job": final_plan_job,
        "final_feature_job": final_feature_job,
        "final_gbt_job": final_gbt_job,
        "final_combine_job": final_combine_job,
        "final_report": f"{env['FINAL_ROOT']}/gbt/toxicity_final_test_report.json",
    }
    _write_json(receipt, Path(env["WORKFLOW_JOBS_PATH"]))
    return receipt


def _submit(
    env: Mapping[str, str],
    common: Mapping[str, str],
    script: str,
    *,
    dependency: str | None = None,
    array: str | None = None,
    extra: Mapping[str, str] | None = None,
) -> str:
    job_env = os.environ.copy()
    job_env.update(common)
    job_env.update({key: str(value) for key, value in (extra or {}).items()})
    command = ["sbatch", "--parsable"]
    if dependency:
        command.append(f"--dependency={dependency}")
    if array:
        command.append(f"--array={array}")
    command.append(script)
    completed = subprocess.run(
        command,
        cwd=env["MINT_AGENT_ROOT"],
        env=job_env,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip())
    return completed.stdout.strip().splitlines()[-1]


def _toxicity_environment(raw: Mapping[str, str]) -> dict[str, str]:
    root = raw.get("MINT_AGENT_ROOT") or str(Path.cwd())
    work_root = raw.get("MATHAGENT_WORK_ROOT") or str(Path.cwd() / "runs")
    design_root = raw.get("DESIGN_ROOT") or f"{work_root}/toxicity/design-v1"
    probe_root = raw.get("PROBE_ROOT") or f"{work_root}/toxicity/probe-v1"
    final_root = raw.get("FINAL_ROOT") or f"{work_root}/toxicity/final-v1"
    return {
        "MINT_AGENT_ROOT": root,
        "DESIGN_ROOT": design_root,
        "ADVISORY_ROOT": raw.get("ADVISORY_ROOT")
        or f"{work_root}/toxicity/design-v1-advisory",
        "PROBE_ROOT": probe_root,
        "FEATURE_PLAN_ROOT": raw.get("FEATURE_PLAN_ROOT") or f"{probe_root}/features",
        "FEATURE_ROOT": raw.get("FEATURE_ROOT")
        or f"{work_root}/toxicity/feature-cache",
        "REPAIR_ROOT": raw.get("REPAIR_ROOT") or f"{probe_root}/repair-v1",
        "FINAL_ROOT": final_root,
        "LEGACY_ROOT": raw.get("LEGACY_ROOT")
        or str(Path(root) / "external" / "toxicity-phase0" / "legacy"),
        "MANIFEST_PATH": raw.get("MANIFEST_PATH") or "",
        "MOLECULE_DIRS": raw.get("MOLECULE_DIRS") or "",
        "DATASET_ID": raw.get("DATASET_ID") or "LD50",
        "GBT_CONFIG": raw.get("GBT_CONFIG") or f"{root}/configs/gbt/toxicity_probe_gbt.yaml",
        "PRIMARY_METRIC": raw.get("PRIMARY_METRIC") or "PCC2",
        "TASK_CONFIG": raw.get("TASK_CONFIG") or "",
        "LLM_ENABLED": raw.get("LLM_ENABLED") or "0",
        "LLM_SCIENTIFIC_MODE": raw.get("LLM_SCIENTIFIC_MODE") or "disabled",
        "LLM_MODEL": raw.get("LLM_MODEL") or "",
        "LLM_ENV_FILE": raw.get("LLM_ENV_FILE")
        or "~/.config/mathagent/openai.env",
        "LLM_CACHE_DIR": raw.get("LLM_CACHE_DIR")
        or f"{work_root}/memory/llm",
        "WORKFLOW_JOBS_PATH": raw.get("WORKFLOW_JOBS_PATH")
        or f"{Path(final_root).parent}/toxicity.jobs.json",
        "RUN_ID": raw.get("RUN_ID") or "",
        "SLURM_JOB_ID": raw.get("SLURM_JOB_ID") or "",
    }


def _prepare_directories(env: Mapping[str, str]) -> None:
    for path in (
        str(Path(env["WORKFLOW_JOBS_PATH"]).parent / "logs"),
        env["DESIGN_ROOT"],
        env["PROBE_ROOT"],
        env["FINAL_ROOT"],
        str(Path(env["WORKFLOW_JOBS_PATH"]).parent),
    ):
        Path(path).mkdir(parents=True, exist_ok=True)


def _write_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the LangGraph-supervised toxicity workflow launcher."
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    graph = build_toxicity_agent_graph(
        ToxicityGraphContext(environment=dict(os.environ))
    )
    result = graph.invoke({"execute": args.execute, "node_trace": []})
    report = result["final_report"]
    if args.output is not None:
        _write_json(report, args.output)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
