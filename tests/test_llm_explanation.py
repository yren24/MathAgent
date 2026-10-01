from __future__ import annotations

import json
from pathlib import Path

from mint_scout.agent.llm_explanation import (
    LLMExplanationContext,
    _explanation_facts,
    build_llm_explanation_graph,
    public_explanation_result,
)


def _summary() -> dict:
    return {
        "report_schema": "mint-agent.pipeline-report.v1",
        "lifecycle_id": "toy-run",
        "lifecycle_status": "COMPLETE",
        "dataset_id": "toy",
        "requested_invariants": ["CA", "PH"],
        "stage_count": 4,
        "scientific_control": {"llm_used_for_numeric_decisions": False},
        "scientific_summary": {
            "preparation": {"sample_count": 500, "manifest": "/private/data.csv"},
            "representation": {"representation_hash": "repr", "pair_count": 32},
            "probe_feature_qc": {"status": "PASS"},
            "probe_filtration_audit": {
                "status": "WARN",
                "invariants": {"PH": {"recommendation": "EXTEND"}},
            },
            "feature_qc": {"status": "PASS"},
            "filtration_audit": {"status": "PASS"},
            "scout": {
                "status": "COMPLETE",
                "sample_count": 300,
                "primary_metric": "PCC",
                "target_value": 0.53,
                "target_source": "probe_derived",
                "candidates": [
                    {
                        "subset_id": "CA+PH",
                        "primary_score": 0.59,
                        "sample_ids": ["must-not-be-sent"],
                    }
                ],
            },
            "evaluation": {
                "status": "TARGET_REACHED",
                "selected_subset": ["CA"],
                "metrics": {"PCC": 0.62},
            },
        },
        "jobs": [{"job_id": "1", "private_path": "/private/job"}],
        "artifacts": [{"path": "/private/artifact"}],
    }


def test_explanation_facts_omit_artifact_paths_and_sample_ids():
    facts = _explanation_facts(_summary())
    serialized = json.dumps(facts)

    assert "/private/artifact" not in serialized
    assert "/private/job" not in serialized
    assert "/private/data.csv" not in serialized
    assert "must-not-be-sent" not in serialized
    assert facts["evaluation"]["metrics"]["PCC"] == 0.62


def test_llm_explanation_graph_writes_audited_non_numeric_artifact(tmp_path: Path):
    summary_path = tmp_path / "pipeline_summary.json"
    output_path = tmp_path / "explanation.json"
    summary_path.write_text(json.dumps(_summary()), encoding="utf-8")

    def generator(summary, *, model, api_key, language):
        assert summary["dataset_id"] == "toy"
        assert model == "test-model"
        assert api_key == "top-secret"
        assert language == "Chinese"
        return (
            {
                "overview": "Run completed.",
                "selection_reason": "CA reached the target.",
                "quality_findings": ["Probe PH requested filtration review."],
                "limitations": ["This is a subset study."],
                "next_steps": ["Review the warning."],
            },
            {"used": True, "model_requested": model, "response_id": "resp_1"},
        )

    context = LLMExplanationContext(
        model="test-model",
        api_key="top-secret",
        language="Chinese",
        output_path=output_path,
        generator=generator,
    )
    state = build_llm_explanation_graph(context).invoke(
        {"summary_path": str(summary_path)}
    )
    result = public_explanation_result(state)

    assert result["status"] == "COMPLETE"
    artifact = json.loads(output_path.read_text(encoding="utf-8"))
    assert artifact["scientific_control"] == {
        "source_artifacts_immutable": True,
        "metrics_recomputed_by_llm": False,
        "selection_changed_by_llm": False,
    }
    assert artifact["llm"]["used_for_numeric_decisions"] is False
    assert "top-secret" not in json.dumps(artifact)
    assert state["node_trace"] == [
        "load_summary_node",
        "explanation_node",
        "write_explanation_node",
    ]


def test_llm_explanation_graph_rejects_non_pipeline_report(tmp_path: Path):
    summary_path = tmp_path / "wrong.json"
    summary_path.write_text(json.dumps({"report_schema": "wrong"}), encoding="utf-8")
    context = LLMExplanationContext(
        model="test-model",
        api_key="top-secret",
        generator=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("generator must not run")
        ),
    )

    state = build_llm_explanation_graph(context).invoke(
        {"summary_path": str(summary_path)}
    )

    assert state["status"] == "INVALID_REPORT"
    assert state["node_trace"] == ["load_summary_node"]
