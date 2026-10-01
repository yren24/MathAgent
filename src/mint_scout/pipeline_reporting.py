from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from mint_scout.config import load_yaml


PIPELINE_REPORT_SCHEMA = "mint-agent.pipeline-report.v1"


@dataclass(frozen=True)
class PipelineReportBundle:
    directory: Path
    summary: Path
    final_report: Path
    run_manifest: Path

    def to_dict(self) -> dict[str, str]:
        return {
            "directory": str(self.directory),
            "summary": str(self.summary),
            "final_report": str(self.final_report),
            "run_manifest": str(self.run_manifest),
        }


def write_pipeline_report(
    *, state: Mapping[str, Any], state_path: str | Path,
) -> PipelineReportBundle:
    source = Path(state_path).expanduser().resolve()
    root = source.parent / f"{source.stem}.artifacts"
    root.mkdir(parents=True, exist_ok=True)

    stages, jobs, artifacts = _collect_execution_records(state)
    payloads = _latest_artifact_payloads(artifacts)
    final_graph = _latest_graph_payload(stages)
    payloads = _supplement_payloads_from_graph(payloads, final_graph)
    task_config = _task_config_payload(state)
    scientific = _scientific_summary(final_graph, payloads, task_config)
    summary_payload = {
        "report_schema": PIPELINE_REPORT_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "lifecycle_id": state.get("lifecycle_id"),
        "lifecycle_status": state.get("status"),
        "lifecycle_state": str(source),
        "dataset_id": state.get("config", {}).get("dataset_id"),
        "requested_invariants": list(state.get("config", {}).get("invariants", [])),
        "evidence_scope": state.get("config", {}).get("evidence_scope"),
        "stage_count": len(stages),
        "continuation_count": len(state.get("continuations", [])),
        "scientific_control": final_graph.get("scientific_control", {}),
        "scientific_summary": scientific,
        "stages": stages,
        "jobs": jobs,
        "artifacts": artifacts,
    }

    summary_path = root / "pipeline_summary.json"
    report_path = root / "final_report.md"
    manifest_path = root / "run_manifest.json"
    _write_json(summary_path, summary_payload)
    report_path.write_text(_render_markdown(summary_payload), encoding="utf-8")
    _write_json(
        manifest_path,
        {
            "report_schema": "mint-agent.pipeline-run-manifest.v1",
            "created_at": summary_payload["created_at"],
            "lifecycle_id": state.get("lifecycle_id"),
            "lifecycle_status": state.get("status"),
            "lifecycle_state": str(source),
            "generated_files": [_file_record(summary_path), _file_record(report_path),],
            "registered_artifacts": artifacts,
        },
    )
    return PipelineReportBundle(root, summary_path, report_path, manifest_path)


def _collect_execution_records(
    state: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    stages: list[dict[str, Any]] = []
    jobs: list[dict[str, Any]] = []
    artifacts: list[dict[str, Any]] = []
    for stage in state.get("stages", []):
        if not isinstance(stage, Mapping):
            continue
        stage_index = int(stage.get("iteration", len(stages)))
        stage_row = {
            "iteration": stage_index,
            "graph_status": stage.get("graph_status"),
            "status": stage.get("status"),
            "graph_report": stage.get("graph_report"),
            "ledger": stage.get("ledger"),
        }
        stages.append(stage_row)
        ledger_path = stage.get("ledger")
        ledger = _read_json_object(ledger_path) if ledger_path else None
        if ledger is None:
            continue
        for job in ledger.get("jobs", []):
            if not isinstance(job, Mapping):
                continue
            plan = job.get("plan") if isinstance(job.get("plan"), Mapping) else {}
            registration = (
                job.get("registration")
                if isinstance(job.get("registration"), Mapping)
                else {}
            )
            attempts = [
                attempt
                for attempt in job.get("attempts", [])
                if isinstance(attempt, Mapping)
            ]
            job_ids = [
                str(attempt["job_id"]) for attempt in attempts if attempt.get("job_id")
            ]
            output_path = plan.get("manifest_path")
            row = {
                "stage": stage_index,
                "plan_id": job.get("plan_id"),
                "job_kind": plan.get("job_kind"),
                "invariants": list(plan.get("invariants", [])),
                "state": job.get("state"),
                "scheduler_state": job.get("scheduler_state"),
                "job_ids": job_ids,
                "attempt_count": len(attempts),
                "output_path": output_path,
                "artifact_id": registration.get("artifact_id"),
                "artifact_kind": registration.get("artifact_kind"),
            }
            jobs.append(row)
            if registration.get("artifact_id") and output_path:
                artifacts.append(
                    {
                        "stage": stage_index,
                        "artifact_id": registration.get("artifact_id"),
                        "artifact_kind": registration.get("artifact_kind"),
                        "path": output_path,
                        "content_sha256": registration.get("content_sha256"),
                        "status": job.get("state"),
                    }
                )
    return stages, jobs, artifacts


def _latest_artifact_payloads(
    artifacts: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    payloads: dict[str, dict[str, Any]] = {}
    for artifact in artifacts:
        kind = str(artifact.get("artifact_kind") or "")
        payload = _read_json_object(artifact.get("path"))
        if kind and payload is not None:
            payloads[kind] = payload
    return payloads


def _latest_graph_payload(stages: list[dict[str, Any]]) -> dict[str, Any]:
    for stage in reversed(stages):
        payload = _read_json_object(stage.get("graph_report"))
        if payload is not None:
            return payload
    return {}


def _supplement_payloads_from_graph(
    payloads: Mapping[str, Mapping[str, Any]], final_graph: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    supplemented = dict(payloads)
    references = {
        "dataset_preparation": _artifact_path(final_graph.get("dataset_preparation")),
        "representation_design": final_graph.get("representation_spec_path"),
        "model_evaluation": _artifact_path(final_graph.get("evaluation")),
        "probe_feature_qc": _artifact_path(final_graph.get("qc")),
        "probe_filtration_audit": _artifact_path(final_graph.get("filtration_audit")),
    }
    scout = final_graph.get("scout")
    if isinstance(scout, Mapping):
        metadata = scout.get("metadata")
        if isinstance(metadata, Mapping):
            references["scout_combined"] = metadata.get("combined_report")
    for kind, path in references.items():
        if kind in supplemented:
            continue
        payload = _read_json_object(path)
        if payload is not None:
            supplemented[kind] = payload
    return supplemented


def _scientific_summary(
    final_graph: Mapping[str, Any],
    payloads: Mapping[str, Mapping[str, Any]],
    task_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    preparation = payloads.get("dataset_preparation", {})
    representation = payloads.get("representation_design", {})
    representation_spec = representation.get("representation_spec", {})
    if not isinstance(representation_spec, Mapping):
        representation_spec = {}
    qc = payloads.get("feature_qc", {})
    filtration = payloads.get("filtration_audit", {})
    probe_qc = payloads.get("probe_feature_qc", {})
    probe_filtration = payloads.get("probe_filtration_audit", {})
    scout = payloads.get("scout_combined", {})
    evaluation = payloads.get("model_evaluation", {})
    result = evaluation.get("result", {})
    if not isinstance(result, Mapping):
        result = {}
    evaluation_metrics = result.get("evaluation_metrics", {})
    if not isinstance(evaluation_metrics, Mapping):
        evaluation_metrics = {}
    selected_metrics = evaluation_metrics.get("selected_subset", {})
    if not isinstance(selected_metrics, Mapping):
        selected_metrics = {}

    filtration_summary = _summarize_filtration_audit(filtration)
    probe_filtration_summary = _summarize_filtration_audit(probe_filtration)

    qc_invariants = qc.get("invariants", {})
    qc_shapes = {}
    if isinstance(qc_invariants, Mapping):
        for name, item in qc_invariants.items():
            if isinstance(item, Mapping):
                qc_shapes[str(name)] = item.get("expected_shape")

    pair_names = representation_spec.get("pair_names", [])
    pair_count = (
        len(pair_names) if isinstance(pair_names, list) and pair_names else None
    )
    if pair_count is None:
        pair_count = _pair_count_from_qc(qc_shapes)
    filtration_profiles = representation_spec.get("filtration_profiles", {})
    if not isinstance(filtration_profiles, Mapping) or not filtration_profiles:
        filtration_profiles = _profiles_from_audit(filtration_summary, qc_shapes)

    scout_target = scout.get("target", {})
    if not isinstance(scout_target, Mapping):
        scout_target = {}
    scout_candidates = []
    for candidate in scout.get("consensus_metrics", []):
        if not isinstance(candidate, Mapping):
            continue
        secondary = candidate.get("secondary_metrics", {})
        if not isinstance(secondary, Mapping):
            secondary = {}
        scout_candidates.append(
            {
                "subset_id": candidate.get("subset_id"),
                "invariants": list(candidate.get("invariants", [])),
                "primary_score": candidate.get("nominal_primary_score"),
                "bootstrap_ci_low": candidate.get("bootstrap_ci_low"),
                "bootstrap_ci_high": candidate.get("bootstrap_ci_high"),
                "secondary_metrics": dict(secondary),
                "top_tier": candidate.get("top_tier"),
                "top1_frequency": candidate.get("top1_frequency"),
                "median_rank": candidate.get("median_rank"),
                "estimated_full_acquisition_wall_seconds": candidate.get(
                    "estimated_full_acquisition_wall_seconds"
                ),
                "probe_performance_component": candidate.get(
                    "probe_performance_component"
                ),
                "ranking_stability_component": candidate.get(
                    "ranking_stability_component"
                ),
                "feature_quality_component": candidate.get("feature_quality_component"),
                "computational_efficiency_component": candidate.get(
                    "computational_efficiency_component"
                ),
                "llm_prior_component": candidate.get("llm_prior_component"),
                "final_score": candidate.get("final_score"),
            }
        )
    if evaluation.get("report_schema") in {
        "mint-agent.acceptance-gbt-evaluation.v1",
        "mint-agent.progressive-acceptance-gbt-evaluation.v1",
    }:
        acceptance_metrics = evaluation.get("metrics", {})
        selected_metrics = (
            acceptance_metrics if isinstance(acceptance_metrics, Mapping) else {}
        )
        selected_subset = evaluation.get("selected_subset")
        split_counts = evaluation.get("sample_counts", {})
        evaluation_sample_count = (
            split_counts.get("acceptance")
            if isinstance(split_counts, Mapping)
            else None
        )
    elif evaluation.get("report_schema") == "mint-agent.frozen-test-gbt-evaluation.v1":
        final_test = evaluation.get("final_test", {})
        if not isinstance(final_test, Mapping):
            final_test = {}
        frozen_metrics = final_test.get("metrics", {})
        selected_metrics = frozen_metrics if isinstance(frozen_metrics, Mapping) else {}
        selected_subset = evaluation.get("selected_subset")
        split_counts = evaluation.get("sample_counts", {})
        evaluation_sample_count = (
            split_counts.get("test") if isinstance(split_counts, Mapping) else None
        )
    else:
        selected_subset = result.get("selected_subset")
        evaluation_sample_count = (
            len(evaluation.get("sample_ids", []))
            if isinstance(evaluation.get("sample_ids"), list)
            else None
        )
    task = final_graph.get("task", {})
    if not isinstance(task, Mapping):
        task = {}
    sample_count = preparation.get("sample_count")
    if sample_count is None:
        sample_count = task.get("sample_count")
    manifest = preparation.get("manifest")
    if manifest is None:
        manifest = _dataset_manifest_path(task_config)
    target_reached = evaluation.get("target_reached")
    if target_reached is None and evaluation.get("status") in {
        "TARGET_REACHED",
        "TARGET_NOT_REACHED",
    }:
        target_reached = evaluation.get("status") == "TARGET_REACHED"
    representation_selection = final_graph.get("controlled_representation_selection")
    if not isinstance(representation_selection, Mapping):
        representation_selection = {}

    return {
        "task": task,
        "preparation": {
            "provider": preparation.get("provider"),
            "sample_count": sample_count,
            "manifest": manifest,
        },
        "representation": {
            "representation_hash": (
                representation.get("representation_hash")
                or final_graph.get("representation_hash")
            ),
            "mode": representation_spec.get("mode"),
            "pair_count": pair_count,
            "filtration_profiles": filtration_profiles,
        },
        "representation_selection": {
            "selection_basis": representation_selection.get("selection_basis"),
            "selected_candidate_id": representation_selection.get(
                "selected_candidate_id"
            ),
            "decision_policy": representation_selection.get("decision_policy"),
            "decision_trace": representation_selection.get("decision_trace", []),
            "test_evidence_used": representation_selection.get("test_evidence_used"),
            "llm_prior_used": representation_selection.get("llm_prior_used"),
            "model_performance_used": representation_selection.get(
                "model_performance_used"
            ),
            "ranking_stability_used": representation_selection.get(
                "ranking_stability_used"
            ),
            "runtime_cost_used": representation_selection.get("runtime_cost_used"),
        },
        "feature_qc": {
            "status": qc.get("status"),
            "blocking_issue_count": qc.get("blocking_issue_count"),
            "warning_issue_count": qc.get("warning_issue_count"),
            "expected_shapes": qc_shapes,
        },
        "filtration_audit": {
            "status": filtration.get("status"),
            "invariants": filtration_summary,
        },
        "probe_feature_qc": {
            "status": probe_qc.get("status"),
            "blocking_issue_count": probe_qc.get("blocking_issue_count"),
            "warning_issue_count": probe_qc.get("warning_issue_count"),
        },
        "probe_filtration_audit": {
            "status": probe_filtration.get("status"),
            "invariants": probe_filtration_summary,
        },
        "scout": {
            "status": scout.get("status"),
            "sample_count": scout.get("sample_count"),
            "primary_metric": scout_target.get("metric"),
            "target_value": scout_target.get("value"),
            "target_source": scout_target.get("source"),
            "priority_order": scout.get("frozen_priority_order", []),
            "candidates": scout_candidates,
        },
        "evaluation": {
            "status": evaluation.get("status"),
            "evaluation_mode": evaluation.get("evaluation_mode"),
            "sample_count": evaluation_sample_count,
            "selected_subset": selected_subset,
            "metrics": dict(selected_metrics),
            "target_metric": evaluation.get("target_metric"),
            "target_value": evaluation.get("target_value"),
            "target_source": evaluation.get("target_source"),
            "target_score": evaluation.get("target_score"),
            "target_reached": target_reached,
            "candidate_rank": evaluation.get("candidate_rank"),
            "candidate_count": evaluation.get("candidate_count"),
            "run_aggregation": evaluation.get("run_aggregation"),
            "protocol": dict(evaluation.get("protocol", {}))
            if isinstance(evaluation.get("protocol"), Mapping)
            else {},
        },
        "llm_scientific": {
            "experiment_plan": _advice_summary(final_graph.get("experiment_plan")),
            "scientific_critique": _advice_summary(
                final_graph.get("scientific_critique")
            ),
        },
    }


def _render_markdown(payload: Mapping[str, Any]) -> str:
    scientific = payload["scientific_summary"]
    task = scientific["task"]
    if not isinstance(task, Mapping):
        task = {}
    evaluation = scientific["evaluation"]
    scout = scientific["scout"]
    representation = scientific["representation"]
    representation_selection = scientific.get("representation_selection", {})
    representation_selection = (
        representation_selection
        if isinstance(representation_selection, Mapping)
        else {}
    )
    qc = scientific["feature_qc"]
    filtration = scientific["filtration_audit"]
    probe_qc = scientific["probe_feature_qc"]
    probe_filtration = scientific["probe_filtration_audit"]
    preparation = scientific["preparation"]
    llm_scientific = scientific.get("llm_scientific", {})
    llm_scientific = llm_scientific if isinstance(llm_scientific, Mapping) else {}
    controls = payload.get("scientific_control", {})
    controls = controls if isinstance(controls, Mapping) else {}
    selected = evaluation.get("selected_subset") or []
    selected_text = "+".join(str(value) for value in selected) if selected else "n/a"
    llm_intent = task.get("llm_intent", {})
    selection_preferences = task.get("selection_preferences", {})
    selection_preferences = (
        selection_preferences if isinstance(selection_preferences, Mapping) else {}
    )
    intent_lines = []
    if isinstance(llm_intent, Mapping) and llm_intent.get("used"):
        intent_lines = [
            "- Intent model: `"
            f"{_text(llm_intent.get('model_returned') or llm_intent.get('model_requested'))}`",
            f"- Intent response: `{_text(llm_intent.get('response_id'))}`",
        ]
    lines = [
        "# MathAgent Pipeline Report",
        "",
        "## Run Identity",
        "",
        f"- Lifecycle: `{_text(payload.get('lifecycle_id'))}`",
        f"- Dataset: `{_text(payload.get('dataset_id'))}`",
        f"- Status: `{_text(payload.get('lifecycle_status'))}`",
        f"- Evidence scope: `{_text(payload.get('evidence_scope'))}`",
        f"- Requested invariants: `{'+'.join(payload.get('requested_invariants', []))}`",
        f"- Method source: `{_text(selection_preferences.get('method_source'))}`",
        f"- Target source: `{_text(selection_preferences.get('target_source'))}`",
        f"- Stages: `{payload.get('stage_count')}`",
        f"- Intent source: `{_text(task.get('intent_source'))}`",
        *intent_lines,
        "",
        "## Data And Representation",
        "",
        f"- Preparation provider: `{_text(preparation.get('provider'))}`",
        f"- Samples: `{_text(preparation.get('sample_count'))}`",
        f"- Standard manifest: `{_text(preparation.get('manifest'))}`",
        f"- Representation: `{_text(representation.get('mode'))}`",
        f"- Representation hash: `{_text(representation.get('representation_hash'))}`",
        f"- Retained element pairs: `{_text(representation.get('pair_count'))}`",
        "",
        "### Filtration Profiles",
        "",
        "| Invariant | Points | Start | Stop | Step |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    profiles = representation.get("filtration_profiles", {})
    if isinstance(profiles, Mapping):
        for name, profile in sorted(profiles.items()):
            if isinstance(profile, Mapping):
                lines.append(
                    f"| {_cell(name)} | {_cell(profile.get('num_points'))} | "
                    f"{_cell(profile.get('start'))} | {_cell(profile.get('stop'))} | "
                    f"{_cell(profile.get('step'))} |"
                )

    if representation_selection.get("selected_candidate_id") is not None:
        lines.extend(
            [
                "",
                "### Representation Selection",
                "",
                f"- Decision rule: `{_text(representation_selection.get('selection_basis'))}`",
                f"- Selected candidate: `{_text(representation_selection.get('selected_candidate_id'))}`",
                f"- Test evidence used: `{_text(representation_selection.get('test_evidence_used'))}`",
                f"- Model performance used: `{_text(representation_selection.get('model_performance_used'))}`",
                f"- Ranking stability used: `{_text(representation_selection.get('ranking_stability_used'))}`",
                f"- Runtime cost used: `{_text(representation_selection.get('runtime_cost_used'))}`",
                f"- LLM numerical prior used: `{_text(representation_selection.get('llm_prior_used'))}`",
            ]
        )

    lines.extend(
        ["", "## Quality Gates", "",]
    )
    if probe_qc.get("status") is not None:
        lines.append(
            f"- Probe feature QC: `{_text(probe_qc.get('status'))}` "
            f"(blocking `{_text(probe_qc.get('blocking_issue_count'))}`, "
            f"warnings `{_text(probe_qc.get('warning_issue_count'))}`)"
        )
    if probe_filtration.get("status") is not None:
        lines.append(
            f"- Probe filtration audit: `{_text(probe_filtration.get('status'))}`"
        )
        _append_filtration_rows(
            lines, probe_filtration.get("invariants", {}), label_prefix="Probe ",
        )
    lines.extend(
        [
            f"- Full-acquired feature QC: `{_text(qc.get('status'))}` "
            f"(blocking `{_text(qc.get('blocking_issue_count'))}`, warnings `{_text(qc.get('warning_issue_count'))}`)",
            f"- Full-acquired filtration audit: `{_text(filtration.get('status'))}`",
        ]
    )
    _append_filtration_rows(lines, filtration.get("invariants", {}))
    lines.append("")

    scout_candidates = scout.get("candidates", [])
    if scout_candidates:
        lines.extend(
            [
                "",
                "## Probe Scout",
                "",
                f"- Samples: `{_text(scout.get('sample_count'))}`",
                f"- Primary metric: `{_text(scout.get('primary_metric'))}`",
                f"- Target: `{_number(scout.get('target_value'))}` "
                f"(`{_text(scout.get('target_source'))}`)",
                "",
                "| Subset | Probe score | Bootstrap CI | Stability | Quality | Efficiency | Diagnostic composite |",
                "| --- | ---: | --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for candidate in scout_candidates:
            if not isinstance(candidate, Mapping):
                continue
            interval = (
                f"[{_number(candidate.get('bootstrap_ci_low'))}, "
                f"{_number(candidate.get('bootstrap_ci_high'))}]"
            )
            lines.append(
                f"| {_cell(candidate.get('subset_id'))} | "
                f"{_number(candidate.get('primary_score'))} | {interval} | "
                f"{_number(candidate.get('ranking_stability_component'))} | "
                f"{_number(candidate.get('feature_quality_component'))} | "
                f"{_number(candidate.get('computational_efficiency_component'))} | "
                f"{_number(candidate.get('final_score'))} |"
            )

    lines.extend(
        [
            "",
            "## Model Evaluation",
            "",
            f"- Protocol: `{_text(evaluation.get('evaluation_mode'))}`",
            f"- Status: `{_text(evaluation.get('status'))}`",
            f"- Selected subset: `{selected_text}`",
            f"- Candidate rank: `{_text(evaluation.get('candidate_rank'))}` / "
            f"`{_text(evaluation.get('candidate_count'))}`",
            f"- Run aggregation: `{_text(evaluation.get('run_aggregation'))}`",
            f"- Target: `{_text(evaluation.get('target_metric'))} >= {_number(evaluation.get('target_value'))}` "
            f"(`{_text(evaluation.get('target_source'))}`)",
            f"- Target reached: `{_text(evaluation.get('target_reached'))}`",
            "",
            "| Metric | Score |",
            "| --- | ---: |",
        ]
    )
    for metric, value in sorted(evaluation.get("metrics", {}).items()):
        lines.append(f"| {_cell(metric)} | {_number(value)} |")

    if evaluation.get("target_reached") is False:
        lines.extend(
            [
                "",
                "The frozen candidate did not reach the requested target. Under the "
                "configured method set, representation candidates, and compute budget, "
                "this run cannot satisfy the requested performance requirement.",
            ]
        )

    lines.extend(
        [
            "",
            "## Execution Audit",
            "",
            "| Stage | Graph status | Job kind | Slurm job | Scheduler | Artifact |",
            "| ---: | --- | --- | --- | --- | --- |",
        ]
    )
    jobs_by_stage: dict[int, list[Mapping[str, Any]]] = {}
    for job in payload.get("jobs", []):
        if isinstance(job, Mapping):
            jobs_by_stage.setdefault(int(job.get("stage", -1)), []).append(job)
    for stage in payload.get("stages", []):
        if not isinstance(stage, Mapping):
            continue
        stage_index = int(stage.get("iteration", -1))
        stage_jobs = jobs_by_stage.get(stage_index, [])
        if not stage_jobs:
            lines.append(
                f"| {stage_index} | {_cell(stage.get('graph_status'))} | - | - | - | - |"
            )
            continue
        for job in stage_jobs:
            lines.append(
                f"| {stage_index} | {_cell(stage.get('graph_status'))} | "
                f"{_cell(job.get('job_kind'))} | {_cell(','.join(job.get('job_ids', [])))} | "
                f"{_cell(job.get('scheduler_state'))} | {_cell(job.get('artifact_id'))} |"
            )

    lines.extend(
        [
            "",
            "## Scientific Controls",
            "",
            f"- LLM used for intent: `{_text(controls.get('llm_used_for_intent'))}`",
            f"- LLM scientific mode: `{_text(controls.get('llm_scientific_mode'))}`",
            "- LLM used for experiment planning: `"
            f"{_text(controls.get('llm_used_for_experiment_planning'))}`",
            "- LLM used for scientific critique: `"
            f"{_text(controls.get('llm_used_for_scientific_critique'))}`",
            "- LLM advice applied to execution: `"
            f"{_text(controls.get('llm_advice_applied_to_execution'))}`",
            "- LLM used for numeric decisions: `"
            f"{_text(controls.get('llm_used_for_numeric_decisions'))}`",
            "- Evaluation set used for candidate selection: `"
            f"{_text(controls.get('evaluation_set_used_for_candidate_selection'))}`",
            "- Full-train CV used: `"
            f"{_text(controls.get('full_train_cross_validation_used'))}`",
            "- Independent test available: `"
            f"{_text(controls.get('independent_test_available'))}`",
            "",
            "Numeric decisions in this report are copied from registered deterministic artifacts. "
            "The report does not rerun models or alter the frozen representation.",
            "",
        ]
    )
    _append_advice_rows(
        lines, "Experiment planning", llm_scientific.get("experiment_plan"),
    )
    _append_advice_rows(
        lines, "Scientific critique", llm_scientific.get("scientific_critique"),
    )
    return "\n".join(lines)


def _advice_summary(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    advice = value.get("advice")
    advice = advice if isinstance(advice, Mapping) else {}
    llm = value.get("llm")
    llm = llm if isinstance(llm, Mapping) else {}
    validation = value.get("validation")
    validation = validation if isinstance(validation, Mapping) else {}
    summary: dict[str, Any] = {
        "stage": value.get("stage"),
        "mode": value.get("mode"),
        "status": value.get("status"),
        "cache_hit": bool(value.get("cache_hit", False)),
        "applied_to_execution": bool(value.get("applied_to_execution", False)),
        "llm_used": bool(llm.get("used", False)),
        "used_for_numeric_decisions": bool(
            llm.get("used_for_numeric_decisions", False)
        ),
        "validation_passed": validation.get("passed"),
        "validation_errors": list(validation.get("errors", []))
        if isinstance(validation.get("errors"), list)
        else [],
    }
    if "objective" in advice:
        summary["objective"] = advice.get("objective")
    candidates = advice.get("candidate_priorities")
    if isinstance(candidates, list):
        rows = []
        for item in candidates:
            if not isinstance(item, Mapping):
                continue
            invariants = item.get("invariants")
            rows.append(
                {
                    "rank": item.get("rank"),
                    "invariants": list(invariants)
                    if isinstance(invariants, list)
                    else [],
                    "expected_cost": item.get("expected_cost"),
                }
            )
        summary["candidate_priorities"] = rows
    if "recommended_next_action" in advice:
        summary["recommended_next_action"] = advice.get("recommended_next_action")
    if "evidence_assessment" in advice:
        summary["evidence_assessment"] = advice.get("evidence_assessment")
    return summary


def _append_advice_rows(lines: list[str], label: str, value: object) -> None:
    if not isinstance(value, Mapping) or not value:
        return
    lines.extend(
        [
            "",
            f"### {label}",
            "",
            f"- Status: `{_text(value.get('status'))}`",
            f"- Mode: `{_text(value.get('mode'))}`",
            f"- LLM used: `{_text(value.get('llm_used'))}`",
            f"- Cache hit: `{_text(value.get('cache_hit'))}`",
            f"- Applied to execution: `{_text(value.get('applied_to_execution'))}`",
        ]
    )
    if value.get("objective"):
        lines.append(f"- Objective: {_text(value.get('objective'))}")
    if value.get("recommended_next_action"):
        lines.append(
            "- Recommended next action: `"
            f"{_text(value.get('recommended_next_action'))}`"
        )
    candidates = value.get("candidate_priorities")
    if isinstance(candidates, list) and candidates:
        lines.extend(
            ["", "| Rank | Invariants | Expected cost |", "| ---: | --- | --- |",]
        )
        for item in candidates:
            if not isinstance(item, Mapping):
                continue
            invariants = item.get("invariants", [])
            invariant_text = (
                "+".join(str(name) for name in invariants)
                if isinstance(invariants, list)
                else _text(invariants)
            )
            lines.append(
                f"| {_cell(item.get('rank'))} | {_cell(invariant_text)} | "
                f"{_cell(item.get('expected_cost'))} |"
            )


def _summarize_filtration_audit(audit: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    invariants = audit.get("invariants", {})
    summary: dict[str, dict[str, Any]] = {}
    if not isinstance(invariants, Mapping):
        return summary
    for name, item in invariants.items():
        if not isinstance(item, Mapping):
            continue
        summary[str(name)] = {
            key: item.get(key)
            for key in (
                "status",
                "recommendation",
                "axis_start",
                "axis_stop",
                "axis_step",
                "tail_zero_fraction_mean",
                "tail_relative_l2_change_max",
            )
        }
    return summary


def _append_filtration_rows(
    lines: list[str], invariant_audits: object, *, label_prefix: str = "",
) -> None:
    if not isinstance(invariant_audits, Mapping):
        return
    for name, item in sorted(invariant_audits.items()):
        if isinstance(item, Mapping):
            lines.append(
                f"- {label_prefix}`{name}` recommendation: "
                f"`{_text(item.get('recommendation'))}`; "
                f"tail zero fraction `{_number(item.get('tail_zero_fraction_mean'))}`; "
                f"tail relative L2 max `{_number(item.get('tail_relative_l2_change_max'))}`"
            )


def _read_json_object(path: object) -> dict[str, Any] | None:
    if path is None:
        return None
    source = Path(str(path)).expanduser()
    if source.suffix.lower() != ".json" or not source.is_file():
        return None
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _task_config_payload(state: Mapping[str, Any]) -> dict[str, Any] | None:
    config = state.get("config")
    if not isinstance(config, Mapping):
        return None
    task_config = config.get("task_config")
    if not task_config:
        return None
    path = Path(str(task_config)).expanduser()
    if not path.is_file():
        return None
    try:
        payload = load_yaml(path)
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _dataset_manifest_path(task_config: Mapping[str, Any] | None) -> str | None:
    if not isinstance(task_config, Mapping):
        return None
    manifest = task_config.get("dataset_manifest")
    if not isinstance(manifest, Mapping):
        return None
    path = manifest.get("path")
    return str(path) if path else None


def _artifact_path(value: object) -> object:
    if isinstance(value, Mapping):
        return value.get("path")
    return None


def _pair_count_from_qc(qc_shapes: Mapping[str, object]) -> int | None:
    counts = {
        int(shape[1])
        for shape in qc_shapes.values()
        if isinstance(shape, list) and len(shape) >= 2 and isinstance(shape[1], int)
    }
    return counts.pop() if len(counts) == 1 else None


def _profiles_from_audit(
    filtration_summary: Mapping[str, Mapping[str, Any]],
    qc_shapes: Mapping[str, object],
) -> dict[str, dict[str, Any]]:
    profiles = {}
    for name, audit in filtration_summary.items():
        shape = qc_shapes.get(name)
        points = shape[0] if isinstance(shape, list) and shape else None
        profiles[name] = {
            "num_points": points,
            "start": audit.get("axis_start"),
            "stop": audit.get("axis_stop"),
            "step": audit.get("axis_step"),
        }
    return profiles


def _file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _number(value: object) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.6g}"
    return _text(value)


def _cell(value: object) -> str:
    return _text(value).replace("|", "\\|")


def _text(value: object) -> str:
    return "n/a" if value is None or value == "" else str(value)
