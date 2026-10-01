from __future__ import annotations

import json
from pathlib import Path

from mint_scout.pipeline_reporting import write_pipeline_report


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_pipeline_report_summarizes_registered_scientific_artifacts(tmp_path: Path):
    preparation = tmp_path / "preparation.json"
    representation = tmp_path / "representation.json"
    qc = tmp_path / "qc.json"
    filtration = tmp_path / "filtration.json"
    evaluation = tmp_path / "evaluation.json"
    _write_json(
        preparation,
        {
            "provider": "external-provider",
            "sample_count": 2,
            "manifest": str(tmp_path / "samples.csv"),
        },
    )
    _write_json(
        representation,
        {
            "representation_hash": "repr-1",
            "representation_spec": {
                "mode": "dataset_adaptive",
                "pair_names": ["protein:C|ligand:C"],
                "filtration_profiles": {
                    "PL": {
                        "num_points": 50,
                        "start": 0.0,
                        "stop": 10.0,
                        "step": 10.0 / 49.0,
                    }
                },
            },
        },
    )
    _write_json(
        qc,
        {
            "status": "PASS",
            "blocking_issue_count": 0,
            "warning_issue_count": 0,
            "invariants": {"PL": {"expected_shape": [50, 1, 8]}},
        },
    )
    _write_json(
        filtration,
        {
            "status": "PASS",
            "invariants": {
                "PL": {
                    "status": "PASS",
                    "recommendation": "KEEP",
                    "axis_start": 0.0,
                    "axis_stop": 10.0,
                    "axis_step": 10.0 / 49.0,
                    "tail_zero_fraction_mean": 0.25,
                    "tail_relative_l2_change_max": 0.04,
                }
            },
        },
    )
    _write_json(
        evaluation,
        {
            "status": "TARGET_REACHED",
            "evaluation_mode": "full_labeled_cv",
            "sample_ids": ["a", "b"],
            "target_metric": "PCC",
            "target_value": 0.5,
            "target_source": "user",
            "result": {
                "selected_subset": ["PL"],
                "evaluation_metrics": {
                    "selected_subset": {
                        "PCC": 0.7,
                        "RMSE": 1.1,
                    }
                },
            },
        },
    )
    graph = tmp_path / "graph.json"
    _write_json(
        graph,
        {
            "status": "COMPLETE",
            "task": {
                "evaluation_mode": "full_labeled_cv",
                "sample_count": 2,
                "intent_source": "openai_responses_api",
                "llm_intent": {
                    "used": True,
                    "model_requested": "test-model",
                    "response_id": "resp_1",
                },
            },
            "dataset_preparation": {"path": str(preparation)},
            "scientific_control": {
                "llm_used_for_intent": True,
                "llm_scientific_mode": "shadow",
                "llm_used_for_experiment_planning": True,
                "llm_used_for_scientific_critique": True,
                "llm_advice_applied_to_execution": False,
                "llm_used_for_numeric_decisions": False,
            },
            "experiment_plan": {
                "stage": "experiment_planning",
                "mode": "shadow",
                "status": "VALIDATED",
                "cache_hit": False,
                "applied_to_execution": False,
                "validation": {"passed": True, "errors": []},
                "llm": {"used": True, "used_for_numeric_decisions": False},
                "advice": {
                    "objective": "Plan train-only probe experiments.",
                    "candidate_priorities": [
                        {
                            "rank": 1,
                            "invariants": ["PL"],
                            "expected_cost": "unknown",
                        }
                    ],
                },
            },
            "scientific_critique": {
                "stage": "scientific_critic",
                "mode": "shadow",
                "status": "VALIDATED",
                "cache_hit": False,
                "applied_to_execution": False,
                "validation": {"passed": True, "errors": []},
                "llm": {"used": True, "used_for_numeric_decisions": False},
                "advice": {
                    "recommended_next_action": "continue_deterministic_pipeline",
                    "evidence_assessment": "preliminary",
                },
            },
        },
    )
    ledger = tmp_path / "ledger.json"
    artifacts = [
        ("representation_design", representation),
        ("feature_qc", qc),
        ("filtration_audit", filtration),
        ("model_evaluation", evaluation),
    ]
    _write_json(
        ledger,
        {
            "jobs": [
                {
                    "plan_id": f"plan-{index}",
                    "state": "REGISTERED",
                    "scheduler_state": "COMPLETED",
                    "attempts": [{"job_id": str(100 + index)}],
                    "plan": {
                        "job_kind": kind,
                        "invariants": ["PL"],
                        "manifest_path": str(path),
                    },
                    "registration": {
                        "artifact_id": f"artifact-{index}",
                        "artifact_kind": kind,
                        "content_sha256": f"sha-{index}",
                    },
                }
                for index, (kind, path) in enumerate(artifacts)
            ]
        },
    )
    state_path = tmp_path / "lifecycle.json"
    state = {
        "lifecycle_id": "toy-run",
        "status": "COMPLETE",
        "config": {
            "dataset_id": "toy",
            "invariants": ["PL"],
            "evidence_scope": "full_train",
        },
        "continuations": [],
        "stages": [
            {
                "iteration": 0,
                "graph_status": "COMPLETE",
                "status": "COMPLETE",
                "graph_report": str(graph),
                "ledger": str(ledger),
            }
        ],
    }
    _write_json(state_path, state)

    bundle = write_pipeline_report(state=state, state_path=state_path)

    summary = json.loads(bundle.summary.read_text(encoding="utf-8"))
    assert summary["scientific_summary"]["evaluation"]["metrics"]["PCC"] == 0.7
    assert summary["scientific_summary"]["evaluation"]["target_reached"] is True
    assert summary["scientific_summary"]["preparation"]["provider"] == "external-provider"
    assert summary["scientific_summary"]["representation"]["pair_count"] == 1
    assert summary["scientific_summary"]["feature_qc"]["status"] == "PASS"
    assert summary["scientific_summary"]["llm_scientific"]["experiment_plan"][
        "status"
    ] == "VALIDATED"
    assert summary["scientific_summary"]["llm_scientific"]["scientific_critique"][
        "recommended_next_action"
    ] == "continue_deterministic_pipeline"
    assert len(summary["artifacts"]) == 4
    report = bundle.final_report.read_text(encoding="utf-8")
    assert "| PCC | 0.7 |" in report
    assert "Intent source: `openai_responses_api`" in report
    assert "Intent model: `test-model`" in report
    assert "Intent response: `resp_1`" in report
    assert "Target reached: `True`" in report
    assert "LLM used for intent: `True`" in report
    assert "LLM scientific mode: `shadow`" in report
    assert "LLM used for experiment planning: `True`" in report
    assert "LLM used for scientific critique: `True`" in report
    assert "Applied to execution: `False`" in report
    assert "Plan train-only probe experiments." in report
    assert "continue_deterministic_pipeline" in report
    assert "LLM used for numeric decisions: `False`" in report
    assert "`PL` recommendation: `KEEP`" in report
    assert "artifact-3" in report
    manifest = json.loads(bundle.run_manifest.read_text(encoding="utf-8"))
    assert len(manifest["generated_files"]) == 2
    assert len(manifest["registered_artifacts"]) == 4


def test_pipeline_report_loads_reused_artifacts_from_final_graph(tmp_path: Path):
    representation = tmp_path / "representation.json"
    evaluation = tmp_path / "evaluation.json"
    scout = tmp_path / "scout.json"
    _write_json(
        representation,
        {
            "representation_hash": "repr-reused",
            "representation_spec": {
                "mode": "dataset_adaptive",
                "pair_names": ["protein:C|ligand:N"],
                "filtration_profiles": {},
            },
        },
    )
    _write_json(
        evaluation,
        {
            "status": "TARGET_REACHED",
            "evaluation_mode": "full_labeled_cv",
            "sample_ids": ["a", "b"],
            "target_metric": "PCC",
            "target_value": 0.6,
            "target_source": "probe_derived",
            "result": {
                "selected_subset": ["PL"],
                "evaluation_metrics": {"selected_subset": {"PCC": 0.7}},
            },
        },
    )
    _write_json(
        scout,
        {
            "status": "COMPLETE",
            "sample_count": 2,
            "target": {"metric": "PCC", "value": 0.6, "source": "probe_derived"},
            "frozen_priority_order": [["PL"], ["PH", "PL"]],
            "consensus_metrics": [
                {
                    "subset_id": "PH+PL",
                    "invariants": ["PH", "PL"],
                    "nominal_primary_score": 0.68,
                    "bootstrap_ci_low": 0.61,
                    "bootstrap_ci_high": 0.74,
                    "secondary_metrics": {"RMSE": 1.2, "MAE": 0.9, "R2": 0.46},
                    "top_tier": True,
                }
            ],
        },
    )
    graph = tmp_path / "graph.json"
    _write_json(
        graph,
        {
            "status": "COMPLETE",
            "representation_hash": "repr-reused",
            "representation_spec_path": str(representation),
            "evaluation": {"path": str(evaluation)},
            "scout": {"metadata": {"combined_report": str(scout)}},
        },
    )
    state_path = tmp_path / "lifecycle.json"
    state = {
        "lifecycle_id": "reused-run",
        "status": "COMPLETE",
        "config": {
            "dataset_id": "toy",
            "invariants": ["PL", "PH"],
            "evidence_scope": "full_train",
        },
        "continuations": [],
        "stages": [
            {
                "iteration": 0,
                "graph_status": "COMPLETE",
                "status": "COMPLETE",
                "graph_report": str(graph),
            }
        ],
    }
    _write_json(state_path, state)

    bundle = write_pipeline_report(state=state, state_path=state_path)

    summary = json.loads(bundle.summary.read_text(encoding="utf-8"))
    scientific = summary["scientific_summary"]
    assert scientific["representation"]["representation_hash"] == "repr-reused"
    assert scientific["representation"]["pair_count"] == 1
    assert scientific["evaluation"]["metrics"]["PCC"] == 0.7
    assert scientific["scout"]["candidates"][0]["subset_id"] == "PH+PL"
    report = bundle.final_report.read_text(encoding="utf-8")
    assert "## Probe Scout" in report
    assert "| PH+PL | 0.68 | [0.61, 0.74] |" in report


def test_pipeline_report_preserves_probe_warnings_after_full_acquisition(tmp_path: Path):
    probe_qc = tmp_path / "probe-qc.json"
    probe_filtration = tmp_path / "probe-filtration.json"
    full_qc = tmp_path / "full-qc.json"
    full_filtration = tmp_path / "full-filtration.json"
    _write_json(
        probe_qc,
        {
            "status": "PASS",
            "blocking_issue_count": 0,
            "warning_issue_count": 0,
        },
    )
    _write_json(
        probe_filtration,
        {
            "status": "WARN",
            "invariants": {
                "PH": {
                    "status": "WARN",
                    "recommendation": "EXTEND",
                    "tail_zero_fraction_mean": 0.0,
                    "tail_relative_l2_change_max": 0.2,
                }
            },
        },
    )
    _write_json(
        full_qc,
        {
            "status": "PASS",
            "blocking_issue_count": 0,
            "warning_issue_count": 0,
            "invariants": {"CA": {"expected_shape": [50, 1, 8]}},
        },
    )
    _write_json(
        full_filtration,
        {
            "status": "PASS",
            "invariants": {
                "CA": {
                    "status": "PASS",
                    "recommendation": "KEEP",
                    "tail_zero_fraction_mean": 0.5,
                    "tail_relative_l2_change_max": 0.03,
                }
            },
        },
    )
    graph = tmp_path / "graph.json"
    _write_json(
        graph,
        {
            "status": "COMPLETE",
            "qc": {"path": str(probe_qc)},
            "filtration_audit": {"path": str(probe_filtration)},
        },
    )
    ledger = tmp_path / "ledger.json"
    _write_json(
        ledger,
        {
            "jobs": [
                {
                    "plan_id": f"plan-{index}",
                    "state": "REGISTERED",
                    "scheduler_state": "COMPLETED",
                    "attempts": [{"job_id": str(200 + index)}],
                    "plan": {
                        "job_kind": kind,
                        "invariants": ["CA"],
                        "manifest_path": str(path),
                    },
                    "registration": {
                        "artifact_id": f"artifact-{index}",
                        "artifact_kind": kind,
                        "content_sha256": f"sha-{index}",
                    },
                }
                for index, (kind, path) in enumerate(
                    (("feature_qc", full_qc), ("filtration_audit", full_filtration))
                )
            ]
        },
    )
    state_path = tmp_path / "lifecycle.json"
    state = {
        "lifecycle_id": "probe-warning-run",
        "status": "COMPLETE",
        "config": {
            "dataset_id": "toy",
            "invariants": ["PH", "CA"],
            "evidence_scope": "full_train",
        },
        "continuations": [],
        "stages": [
            {
                "iteration": 0,
                "graph_status": "COMPLETE",
                "status": "COMPLETE",
                "graph_report": str(graph),
                "ledger": str(ledger),
            }
        ],
    }
    _write_json(state_path, state)

    bundle = write_pipeline_report(state=state, state_path=state_path)

    summary = json.loads(bundle.summary.read_text(encoding="utf-8"))
    scientific = summary["scientific_summary"]
    assert scientific["probe_filtration_audit"]["status"] == "WARN"
    assert scientific["probe_filtration_audit"]["invariants"]["PH"]["recommendation"] == "EXTEND"
    assert scientific["filtration_audit"]["invariants"]["CA"]["recommendation"] == "KEEP"
    report = bundle.final_report.read_text(encoding="utf-8")
    assert "Probe filtration audit: `WARN`" in report
    assert "Probe `PH` recommendation: `EXTEND`" in report
    assert "Full-acquired filtration audit: `PASS`" in report
    assert "`CA` recommendation: `KEEP`" in report


def test_pipeline_report_reads_frozen_external_test_metrics(tmp_path: Path):
    evaluation = tmp_path / "frozen-test.json"
    manifest = tmp_path / "manifest.csv"
    task_config = tmp_path / "task.yaml"
    task_config.write_text(
        "task_id: toy\n"
        "system_type: protein_ligand\n"
        "task_type: regression\n"
        "dataset_manifest:\n"
        f"  path: {manifest}\n",
        encoding="utf-8",
    )
    _write_json(
        evaluation,
        {
            "report_schema": "mint-agent.frozen-test-gbt-evaluation.v1",
            "status": "COMPLETE",
            "evaluation_mode": "explicit_validation_and_test",
            "evidence_scope": "external_test",
            "selected_subset": ["PL", "PH"],
            "sample_counts": {"train": 8, "validation": 3, "test": 4},
            "target_metric": "PCC",
            "target_value": 0.7,
            "target_source": "user",
            "final_test": {"metrics": {"PCC": 0.81, "RMSE": 1.2}},
        },
    )
    graph = tmp_path / "graph.json"
    _write_json(
        graph,
        {
            "status": "COMPLETE",
            "task": {
                "sample_count": 15,
                "manifest_source": "openai_responses_api",
            },
            "evaluation": {"path": str(evaluation)},
            "scientific_control": {"llm_used_for_numeric_decisions": False},
        },
    )
    state_path = tmp_path / "lifecycle.json"
    state = {
        "lifecycle_id": "frozen-test-run",
        "status": "COMPLETE",
        "config": {
            "dataset_id": "toy",
            "task_config": str(task_config),
            "invariants": ["PL", "PH"],
            "evidence_scope": "external_test",
        },
        "continuations": [],
        "stages": [
            {
                "iteration": 0,
                "graph_status": "COMPLETE",
                "status": "COMPLETE",
                "graph_report": str(graph),
                "ledger": None,
            }
        ],
    }
    _write_json(state_path, state)

    bundle = write_pipeline_report(state=state, state_path=state_path)
    summary = json.loads(bundle.summary.read_text(encoding="utf-8"))

    assert summary["scientific_summary"]["evaluation"]["sample_count"] == 4
    assert summary["scientific_summary"]["preparation"]["sample_count"] == 15
    assert summary["scientific_summary"]["preparation"]["manifest"] == str(manifest)
    assert summary["scientific_summary"]["evaluation"]["selected_subset"] == ["PL", "PH"]
    assert summary["scientific_summary"]["evaluation"]["metrics"]["PCC"] == 0.81
    report = bundle.final_report.read_text(encoding="utf-8")
    assert "Samples: `15`" in report
    assert "| PCC | 0.81 |" in report
