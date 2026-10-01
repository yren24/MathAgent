from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from mint_scout.execution.engine import ExecutionResult
from mint_scout.labels import LabelRetrievalResult
from mint_scout.representation import RepresentationSpec
from mint_scout.schemas import EvaluationMode, EvaluationPlan, TaskCard
from mint_scout.scout.pipeline import ProbeResult


@dataclass(frozen=True)
class RunArtifactBundle:
    run_directory: Path
    artifacts: Mapping[str, Path]


def write_run_artifacts(
    *,
    run_directory: str | Path,
    task_card: TaskCard,
    evaluation_plan: EvaluationPlan,
    representation_spec: RepresentationSpec,
    probe_result: ProbeResult,
    execution_result: ExecutionResult,
    timing_seconds: Mapping[str, float],
    anomalies: Sequence[str] = (),
    label_retrieval: LabelRetrievalResult | None = None,
) -> RunArtifactBundle:
    representation_spec.assert_frozen()
    if representation_spec.spec_hash != execution_result.representation_hash:
        raise ValueError("Execution and report RepresentationSpec hashes do not match")
    if probe_result.gbt_parameter_hash != execution_result.gbt_parameter_hash:
        raise ValueError("Probe and execution GBT parameter hashes do not match")
    if execution_result.evaluation_mode != evaluation_plan.mode:
        raise ValueError("Execution result and evaluation plan modes do not match")
    if not timing_seconds or any(float(value) < 0 for value in timing_seconds.values()):
        raise ValueError("timing_seconds must contain non-negative stage timings")

    root = Path(run_directory)
    root.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, Path] = {}
    all_anomalies = tuple(
        dict.fromkeys(
            tuple(str(item) for item in representation_spec.parameters.get("anomalies", ()))
            + tuple(probe_result.selection.warnings)
            + tuple(probe_result.target.warnings)
            + tuple(str(item) for item in anomalies)
        )
    )

    def json_artifact(name: str, payload: Any) -> None:
        path = root / name
        path.write_text(json.dumps(_plain(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        artifacts[name] = path

    json_artifact("task_card.json", _task_payload(task_card, evaluation_plan))
    representation_path = root / "representation_spec.json"
    representation_spec.write(representation_path)
    artifacts[representation_path.name] = representation_path
    json_artifact("target_spec.json", asdict(probe_result.target))
    json_artifact("anomalies.json", {"anomalies": list(all_anomalies)})

    _write_element_support(root / "element_support.csv", representation_spec)
    artifacts["element_support.csv"] = root / "element_support.csv"
    _write_pair_support(root / "element_pair_support.csv", representation_spec)
    artifacts["element_pair_support.csv"] = root / "element_pair_support.csv"
    _write_probe_ids(root / "probe_sample_ids.csv", probe_result)
    artifacts["probe_sample_ids.csv"] = root / "probe_sample_ids.csv"
    _write_probe_folds(root / "probe_fold_assignments.csv", probe_result)
    artifacts["probe_fold_assignments.csv"] = root / "probe_fold_assignments.csv"
    _write_probe_ranking(root / "probe_consensus_ranking.csv", probe_result)
    artifacts["probe_consensus_ranking.csv"] = root / "probe_consensus_ranking.csv"
    _write_probe_single_metrics(root / "probe_single_invariant_metrics.csv", probe_result)
    artifacts["probe_single_invariant_metrics.csv"] = root / "probe_single_invariant_metrics.csv"
    _write_probe_ranking(root / "probe_bootstrap_summary.csv", probe_result)
    artifacts["probe_bootstrap_summary.csv"] = root / "probe_bootstrap_summary.csv"
    _write_acquisitions(root / "acquisition_log.jsonl", execution_result)
    artifacts["acquisition_log.jsonl"] = root / "acquisition_log.jsonl"
    _write_full_metrics(root / "full_evaluation_metrics.csv", execution_result)
    artifacts["full_evaluation_metrics.csv"] = root / "full_evaluation_metrics.csv"
    _write_timing(root / "timing.csv", timing_seconds)
    artifacts["timing.csv"] = root / "timing.csv"

    if execution_result.inference is not None:
        _write_predictions(root / "predictions.csv", execution_result)
        artifacts["predictions.csv"] = root / "predictions.csv"
        _write_disagreement(root / "prediction_disagreement.csv", execution_result)
        artifacts["prediction_disagreement.csv"] = root / "prediction_disagreement.csv"
    if label_retrieval is not None:
        _write_label_provenance(root / "label_provenance.csv", label_retrieval)
        artifacts["label_provenance.csv"] = root / "label_provenance.csv"

    report_path = root / "final_report.md"
    report_path.write_text(
        _render_final_report(
            task_card=task_card,
            plan=evaluation_plan,
            representation=representation_spec,
            probe=probe_result,
            execution=execution_result,
            timing=timing_seconds,
            anomalies=all_anomalies,
        ),
        encoding="utf-8",
    )
    artifacts[report_path.name] = report_path
    json_artifact(
        "run_manifest.json",
        {
            "schema": "mint-agent.run-manifest.v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "task_id": task_card.task_id,
            "dataset_id": task_card.dataset.dataset_id,
            "representation_hash": representation_spec.spec_hash,
            "probe_hash": probe_result.probe_hash,
            "status": execution_result.status.value,
            "artifact_files": sorted((*artifacts, "run_manifest.json")),
        },
    )
    return RunArtifactBundle(root, dict(artifacts))


def _task_payload(task: TaskCard, plan: EvaluationPlan) -> dict[str, Any]:
    dataset = task.dataset
    return {
        "task_id": task.task_id,
        "dataset_id": dataset.dataset_id,
        "system_type": dataset.system_type,
        "task_type": task.task_type.value,
        "target_name": dataset.label_name,
        "primary_metric": task.target_metric.upper(),
        "user_target": task.user_target,
        "label_retrieval_allowed": task.label_retrieval_allowed,
        "evaluation_mode": plan.mode.value,
        "modeling_sample_count": len(plan.modeling_sample_ids),
        "validation_sample_count": len(plan.validation_sample_ids),
        "evaluation_sample_count": len(plan.evaluation_sample_ids),
        "inference_sample_count": len(plan.inference_sample_ids),
        "warnings": list(plan.warnings),
    }


def _write_element_support(path: Path, representation: RepresentationSpec) -> None:
    rows = representation.parameters.get("element_support", ())
    _write_csv(
        path,
        rows,
        ("role", "element", "atom_count", "sample_presence", "support_fraction", "retained"),
    )


def _write_pair_support(path: Path, representation: RepresentationSpec) -> None:
    rows = representation.parameters.get("pair_support", ())
    normalized = []
    for row in rows:
        item = dict(row)
        pair = item.pop("pair", ())
        item["pair"] = "+".join(pair)
        normalized.append(item)
    _write_csv(
        path,
        normalized,
        ("pair", "sample_presence", "support_fraction", "retained", "selected"),
    )


def _write_probe_ids(path: Path, probe: ProbeResult) -> None:
    _write_csv(path, ({"sample_id": value} for value in probe.selection.sample_ids), ("sample_id",))


def _write_probe_folds(path: Path, probe: ProbeResult) -> None:
    _write_csv(
        path,
        (
            {"sample_id": sample_id, "fold_id": probe.fold_assignment[sample_id]}
            for sample_id in probe.selection.sample_ids
        ),
        ("sample_id", "fold_id"),
    )


def _write_probe_ranking(path: Path, probe: ProbeResult) -> None:
    rows = []
    for rank, summary in enumerate(probe.ranking.summaries, start=1):
        row = asdict(summary)
        row["rank"] = rank
        row["invariants"] = "+".join(summary.invariants)
        row.update({f"secondary_{key}": value for key, value in summary.secondary_metrics.items()})
        row.pop("secondary_metrics", None)
        rows.append(row)
    fields = tuple(dict.fromkeys(key for row in rows for key in row))
    _write_csv(path, rows, fields)


def _write_probe_single_metrics(path: Path, probe: ProbeResult) -> None:
    rows = []
    for summary in probe.ranking.summaries:
        if summary.subset_size != 1:
            continue
        row = {
            "invariant": summary.invariants[0],
            "primary_metric": probe.ranking.primary_metric,
            "primary_score": summary.nominal_primary_score,
        }
        row.update({f"secondary_{key}": value for key, value in summary.secondary_metrics.items()})
        rows.append(row)
    fields = tuple(dict.fromkeys(key for row in rows for key in row))
    _write_csv(path, rows, fields or ("invariant", "primary_metric", "primary_score"))


def _write_acquisitions(path: Path, execution: ExecutionResult) -> None:
    lines = []
    cumulative = []
    for index, record in enumerate(execution.acquisition_records, start=1):
        cumulative.append(record.invariant)
        lines.append(
            json.dumps(
                {
                    "round": index,
                    "invariant": record.invariant,
                    "cached_count": record.cached_count,
                    "computed_count": record.computed_count,
                    "cost": record.cost,
                    "acquired_invariants": list(cumulative),
                },
                sort_keys=True,
            )
        )
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _write_full_metrics(path: Path, execution: ExecutionResult) -> None:
    rows = []
    for round_index, round_result in enumerate(execution.rounds, start=1):
        for subset, score in round_result.subset_scores.items():
            rows.append(
                {
                    "round": round_index,
                    "evaluation_protocol": execution.evaluation_mode.value,
                    "acquired_invariants": "+".join(round_result.acquired_invariants),
                    "subset": "+".join(subset),
                    "metric": execution.target_metric,
                    "score": score,
                    "target_value": execution.target_value,
                    "target_reached": round_result.target_reached and subset == round_result.selected_subset,
                }
            )
    _write_csv(
        path,
        rows,
        (
            "round",
            "evaluation_protocol",
            "acquired_invariants",
            "subset",
            "metric",
            "score",
            "target_value",
            "target_reached",
        ),
    )


def _write_predictions(path: Path, execution: ExecutionResult) -> None:
    assert execution.inference is not None
    _write_csv(
        path,
        (
            {"sample_id": sample_id, "prediction": float(execution.inference.predictions[index])}
            for index, sample_id in enumerate(execution.inference.sample_ids)
        ),
        ("sample_id", "prediction"),
    )


def _write_disagreement(path: Path, execution: ExecutionResult) -> None:
    assert execution.inference is not None
    inference = execution.inference
    rows = []
    for index, sample_id in enumerate(inference.sample_ids):
        rows.append(
            {
                "sample_id": sample_id,
                "available": inference.disagreement_std is not None,
                "std_across_invariants": (
                    "" if inference.disagreement_std is None else float(inference.disagreement_std[index])
                ),
                "range_across_invariants": (
                    "" if inference.disagreement_range is None else float(inference.disagreement_range[index])
                ),
            }
        )
    _write_csv(
        path,
        rows,
        ("sample_id", "available", "std_across_invariants", "range_across_invariants"),
    )


def _write_label_provenance(path: Path, result: LabelRetrievalResult) -> None:
    rows = [_plain(asdict(record)) for record in result.provenance_records]
    fields = tuple(dict.fromkeys(key for row in rows for key in row))
    _write_csv(path, rows, fields or ("sample_id",))


def _write_timing(path: Path, timing: Mapping[str, float]) -> None:
    _write_csv(
        path,
        ({"stage": stage, "wall_seconds": float(seconds)} for stage, seconds in sorted(timing.items())),
        ("stage", "wall_seconds"),
    )


def _render_final_report(
    *,
    task_card: TaskCard,
    plan: EvaluationPlan,
    representation: RepresentationSpec,
    probe: ProbeResult,
    execution: ExecutionResult,
    timing: Mapping[str, float],
    anomalies: Sequence[str],
) -> str:
    library = tuple(representation.parameters.get("invariant_library", ()))
    acquired = tuple(execution.acquisition_order)
    skipped = tuple(name for name in library if name not in acquired)
    lead = probe.ranking.summaries[0]
    lines = [
        "# MathAgent Final Report",
        "",
        f"- Task: `{task_card.task_id}`",
        f"- System: `{task_card.dataset.system_type}` regression",
        f"- Evaluation protocol: `{plan.mode.value}`",
        f"- Representation: `{representation.mode.value}` (`{representation.spec_hash}`)",
        f"- Retained element-pair channels: `{len(representation.pair_order)}`",
        f"- Probe samples: `{probe.selection.final_size}`",
        f"- Probe OOF lead: `{'+'.join(lead.invariants)}`; {probe.ranking.primary_metric} `{lead.nominal_primary_score:.6g}`",
        f"- Acceptance target: `{probe.target.metric} {probe.target.direction} {probe.target.value:.6g}` (`{probe.target.source}`)",
        f"- Execution status: `{execution.status.value}`",
        f"- Selected subset: `{'+'.join(execution.selected_subset)}`",
        f"- Acceptance score: `{execution.selected_score:.6g}`",
        f"- Acquired invariants: `{', '.join(acquired)}`",
        f"- Skipped invariants: `{', '.join(skipped) if skipped else 'none'}`",
        f"- Acceptance queries: `{execution.n_acceptance_queries}`",
        f"- Evaluation set used for acceptance: `{str(execution.evaluation_set_used_for_acceptance).lower()}`",
        f"- Untouched final test: `{execution.untouched_final_test}`",
        "",
        "Probe OOF performance is prioritization evidence. The acceptance score above uses the stated evaluation protocol.",
        "",
        "## Timing",
        "",
    ]
    lines.extend(f"- {stage}: `{seconds:.3f}` seconds" for stage, seconds in sorted(timing.items()))
    lines.extend(("", "## Anomalies", ""))
    lines.extend(f"- `{item}`" for item in anomalies)
    if not anomalies:
        lines.append("- None")
    return "\n".join(lines) + "\n"


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]] | Any, fields: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(_plain(dict(row)))


def _plain(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value
