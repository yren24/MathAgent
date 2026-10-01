from __future__ import annotations

from types import SimpleNamespace

import pytest

from mint_scout.toxicity.agent_graph import (
    ToxicityGraphContext,
    build_toxicity_agent_graph,
    submit_toxicity_dependency_chain,
)


def test_toxicity_dependency_chain_wires_llm_probe_strategy(
    tmp_path, monkeypatch
):
    submissions = []

    def fake_run(command, *, cwd, env, check, capture_output, text):
        submissions.append({"command": list(command), "cwd": cwd, "env": dict(env)})
        return SimpleNamespace(
            returncode=0,
            stdout=f"{9000 + len(submissions)}\n",
            stderr="",
        )

    monkeypatch.setattr(
        "mint_scout.toxicity.agent_graph._prepare_directories",
        lambda _env: None,
    )
    monkeypatch.setattr("subprocess.run", fake_run)

    root = tmp_path / "repo"
    jobs_path = tmp_path / "toxicity.jobs.json"
    receipt = submit_toxicity_dependency_chain(
        {
            "MINT_AGENT_ROOT": str(root),
            "DESIGN_ROOT": str(tmp_path / "design"),
            "ADVISORY_ROOT": str(tmp_path / "advisory"),
            "PROBE_ROOT": str(tmp_path / "probe"),
            "FEATURE_PLAN_ROOT": str(tmp_path / "probe" / "features"),
            "FEATURE_ROOT": str(tmp_path / "features"),
            "REPAIR_ROOT": str(tmp_path / "probe" / "repair"),
            "FINAL_ROOT": str(tmp_path / "final"),
            "WORKFLOW_JOBS_PATH": str(jobs_path),
            "MANIFEST_PATH": str(tmp_path / "manifest.csv"),
            "MOLECULE_DIRS": str(tmp_path / "molecules"),
            "LLM_ENABLED": "1",
            "LLM_SCIENTIFIC_MODE": "advisory",
            "LLM_MODEL": "gpt-test",
        }
    )

    advisory = next(
        item
        for item in submissions
        if item["command"][-1].endswith("run_toxicity_advisory.sbatch")
    )
    probe = next(
        item
        for item in submissions
        if item["command"][-1].endswith("run_toxicity_probe_selection.sbatch")
    )

    assert receipt["llm_probe_strategy_enabled"] is True
    assert receipt["advisory_job"] != receipt["design_job"]
    assert advisory["env"]["LLM_MODEL"] == "gpt-test"
    assert probe["env"]["ADVISORY_ARTIFACT"].endswith(
        "advisory/toxicity_llm_advisory.json"
    )
    assert jobs_path.exists()


def test_toxicity_langgraph_launcher_reports_llm_probe_strategy(monkeypatch):
    pytest.importorskip("langgraph.graph")

    monkeypatch.setattr(
        "mint_scout.toxicity.agent_graph.submit_toxicity_dependency_chain",
        lambda _env: {
            "report_schema": "mint-agent.toxicity-jobs.v1",
            "llm_probe_strategy_enabled": True,
            "final_report": "/tmp/final.json",
        },
    )
    graph = build_toxicity_agent_graph(
        ToxicityGraphContext(
            environment={
                "DATASET_ID": "LD50",
                "RUN_ID": "smoke",
                "LLM_ENABLED": "1",
                "LLM_SCIENTIFIC_MODE": "advisory",
            }
        )
    )
    result = graph.invoke({"execute": True})

    report = result["final_report"]
    assert report["status"] == "SUBMITTED"
    assert report["dataset_id"] == "LD50"
    assert report["llm_advisory_enabled"] is True
    assert report["llm_probe_strategy_can_affect_probe_policy"] is True
