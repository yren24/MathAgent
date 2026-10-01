from __future__ import annotations

import json
from pathlib import Path

import pytest

import mint_scout.agent.lifecycle as lifecycle
from mint_scout.agent.lifecycle import (
    advance_lifecycle,
    initialize_lifecycle,
    load_lifecycle,
    write_lifecycle,
)
from mint_scout.artifact_memory import ArtifactRegistry
from mint_scout.execution.jobs import CommandResult, load_job_ledger


def _initialize(tmp_path: Path) -> Path:
    state_path = tmp_path / "lifecycle.json"
    initialize_lifecycle(
        state_path,
        task_config=tmp_path / "task.yaml",
        registry=tmp_path / "artifacts.sqlite",
        dataset_id="toy",
        invariants=("PL", "PH"),
        execution_profile=tmp_path / "profile.yaml",
        representation_spec=tmp_path / "representation.json",
        lifecycle_id="toy-lifecycle",
    )
    return state_path


def _feature_plan(manifest: Path, working_directory: Path) -> dict:
    return {
        "profile_id": "test",
        "scheduler": "slurm",
        "job_kind": "feature_batch",
        "dataset_id": "toy",
        "invariant": "PL",
        "invariants": ["PL"],
        "evidence_scope": "full_train",
        "representation_hash": "repr-1",
        "job_name": "mint-feature-pl",
        "working_directory": str(working_directory),
        "script_path": str(working_directory / "feature.sbatch"),
        "resources": {},
        "environment": {"INVARIANT": "PL", "SPLIT": "train"},
        "manifest_path": str(manifest),
        "stdout_path": str(working_directory / "feature-%j.out"),
        "stderr_path": str(working_directory / "feature-%j.err"),
        "submit_command": ["sbatch", "feature.sbatch"],
    }


def _needs_feature_report(manifest: Path, working_directory: Path) -> dict:
    return {
        "report_schema": "mint-agent.graph-run.v1",
        "run_id": "toy-lifecycle",
        "dataset_id": "toy",
        "status": "NEEDS_FEATURE_COMPUTE",
        "feature_jobs": [_feature_plan(manifest, working_directory)],
        "qc_jobs": [],
        "filtration_audit_jobs": [],
        "scout_jobs": [],
        "evaluation_jobs": [],
    }


def test_lifecycle_dry_run_persists_graph_plan_and_ledger(tmp_path: Path):
    state_path = _initialize(tmp_path)
    manifest = tmp_path / "PL.jsonl"

    state = advance_lifecycle(
        state_path,
        graph_runner=lambda config, output: _needs_feature_report(manifest, tmp_path),
    )

    assert state["status"] == "PLANNED"
    assert state["active_stage_index"] == 0
    stage = state["stages"][0]
    assert Path(stage["graph_report"]).is_file()
    ledger = load_job_ledger(stage["ledger"])
    assert ledger["status"] == "PLANNED"
    assert ledger["jobs"][0]["attempts"] == []


def test_lifecycle_passes_llm_scientific_args_to_graph_runner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    state_path = tmp_path / "lifecycle.json"
    state = initialize_lifecycle(
        state_path,
        task_config=tmp_path / "task.yaml",
        registry=tmp_path / "artifacts.sqlite",
        dataset_id="toy",
        invariants=("PL",),
        execution_profile=tmp_path / "profile.yaml",
        llm_scientific_mode="shadow",
        llm_model="gpt-5",
        llm_api_key_env="OPENAI_API_KEY",
        llm_env_file="~/.config/mathagent/openai.env",
        llm_cache_dir="/path/to/workdir/mathagent/llm-cache",
        llm_max_candidates=4,
        llm_timeout_seconds=240,
        lifecycle_id="toy-lifecycle",
    )
    captured: dict[str, list[str]] = {}

    def fake_run_graph_main(argv):
        captured["argv"] = list(argv)
        output = Path(argv[argv.index("--output") + 1])
        output.write_text(
            json.dumps(
                {"report_schema": "mint-agent.graph-run.v1", "status": "COMPLETE",}
            ),
            encoding="utf-8",
        )
        return 0

    monkeypatch.setattr(lifecycle, "run_graph_main", fake_run_graph_main)

    lifecycle._run_graph(state["config"], tmp_path / "graph.json")

    argv = captured["argv"]
    assert "--llm-scientific-mode" in argv
    assert argv[argv.index("--llm-scientific-mode") + 1] == "shadow"
    assert argv[argv.index("--llm-model") + 1] == "gpt-5"
    assert argv[argv.index("--llm-env-file") + 1].endswith("openai.env")
    assert argv[argv.index("--llm-cache-dir") + 1].endswith("llm-cache")
    assert argv[argv.index("--llm-max-candidates") + 1] == "4"
    assert argv[argv.index("--llm-timeout-seconds") + 1] == "240.0"


def test_lifecycle_submits_dependency_continuation(tmp_path: Path):
    state_path = _initialize(tmp_path)
    manifest = tmp_path / "PL.jsonl"
    commands = []

    def runner(command, cwd):
        commands.append(command)
        job_id = "102" if any("--dependency=" in value for value in command) else "101"
        return CommandResult(0, f"Submitted batch job {job_id}\n", "")

    state = advance_lifecycle(
        state_path,
        execute=True,
        auto_continue=True,
        continuation_script="scripts/run-agent.sbatch",
        project_root=tmp_path,
        graph_runner=lambda config, output: _needs_feature_report(manifest, tmp_path),
        command_runner=runner,
    )

    assert state["status"] == "WAITING_FOR_JOBS"
    assert len(commands) == 2
    continuation = state["stages"][0]["continuation"]
    assert continuation["job_id"] == "102"
    assert continuation["dependency_job_ids"] == ["101"]
    assert "--dependency=afterany:101" in continuation["command"]


def test_lifecycle_pauses_after_reconciling_configured_stage(tmp_path: Path):
    state_path = _initialize(tmp_path)
    manifest = tmp_path / "PL.jsonl"

    def submit_runner(command, cwd):
        job_id = "302" if any("--dependency=" in value for value in command) else "301"
        return CommandResult(0, f"Submitted batch job {job_id}\n", "")

    first = advance_lifecycle(
        state_path,
        execute=True,
        auto_continue=True,
        continuation_script="scripts/run-agent.sbatch",
        project_root=tmp_path,
        stop_after_stage=0,
        graph_runner=lambda config, output: _needs_feature_report(manifest, tmp_path),
        command_runner=submit_runner,
    )
    continuation_command = first["stages"][0]["continuation"]["command"]
    assert any("STOP_AFTER_STAGE=0" in value for value in continuation_command)

    output_a = tmp_path / "a.npy"
    output_a.write_bytes(b"a")
    manifest.write_text(
        json.dumps(
            {
                "sample_id": "a",
                "invariant": "PL",
                "split": "train",
                "status": "computed",
                "output_path": str(output_a),
            }
        )
        + "\n",
        encoding="utf-8",
    )

    def completed_runner(command, cwd):
        assert command[0] == "sacct"
        return CommandResult(0, "301|COMPLETED|0:0\n", "")

    def unexpected_graph(config, output):
        raise AssertionError("graph must not run after the configured stop stage")

    paused = advance_lifecycle(
        state_path,
        execute=True,
        auto_continue=True,
        continuation_script="scripts/run-agent.sbatch",
        project_root=tmp_path,
        stop_after_stage=0,
        graph_runner=unexpected_graph,
        command_runner=completed_runner,
    )

    assert paused["status"] == "PAUSED_AFTER_STAGE"
    assert paused["active_stage_index"] is None
    assert paused["stages"][0]["status"] == "REGISTERED"
    assert paused["events"][-1]["event"] == "LIFECYCLE_PAUSED"

    resumed = advance_lifecycle(
        state_path,
        graph_runner=lambda config, output: {
            "report_schema": "mint-agent.graph-run.v1",
            "run_id": "toy-lifecycle",
            "dataset_id": "toy",
            "status": "COMPLETE",
        },
    )
    assert resumed["status"] == "COMPLETE"
    assert Path(resumed["report_bundle"]["final_report"]).is_file()
    assert Path(resumed["report_bundle"]["summary"]).is_file()
    assert resumed["events"][-1]["event"] == "PIPELINE_REPORT_WRITTEN"


def test_lifecycle_pauses_when_requested_stage_was_already_passed(tmp_path: Path):
    state_path = _initialize(tmp_path)
    manifest = tmp_path / "PL.jsonl"

    def submit_runner(command, cwd):
        job_id = "402" if any("--dependency=" in value for value in command) else "401"
        return CommandResult(0, f"Submitted batch job {job_id}\n", "")

    advance_lifecycle(
        state_path,
        execute=True,
        auto_continue=True,
        continuation_script="scripts/run-agent.sbatch",
        project_root=tmp_path,
        graph_runner=lambda config, output: _needs_feature_report(manifest, tmp_path),
        command_runner=submit_runner,
    )
    state = load_lifecycle(state_path)
    state["stages"][0]["iteration"] = 2
    write_lifecycle(state, state_path)
    output = tmp_path / "a.npy"
    output.write_bytes(b"a")
    manifest.write_text(
        json.dumps(
            {
                "sample_id": "a",
                "invariant": "PL",
                "split": "train",
                "status": "computed",
                "output_path": str(output),
            }
        )
        + "\n",
        encoding="utf-8",
    )

    paused = advance_lifecycle(
        state_path,
        execute=True,
        auto_continue=True,
        continuation_script="scripts/run-agent.sbatch",
        project_root=tmp_path,
        stop_after_stage=0,
        graph_runner=lambda config, output: pytest.fail("graph should not advance"),
        command_runner=lambda command, cwd: CommandResult(
            0, "401|COMPLETED|0:0\n", "",
        ),
    )

    assert paused["status"] == "PAUSED_AFTER_STAGE"
    assert paused["stages"][0]["iteration"] == 2


def test_lifecycle_rejects_negative_stop_stage(tmp_path: Path):
    state_path = _initialize(tmp_path)

    try:
        advance_lifecycle(state_path, stop_after_stage=-1)
    except ValueError as exc:
        assert str(exc) == "stop_after_stage must be non-negative"
    else:
        raise AssertionError("negative stop stage must be rejected")


def test_lifecycle_does_not_retry_failed_job_without_opt_in(tmp_path: Path):
    state_path = _initialize(tmp_path)
    manifest = tmp_path / "PL.jsonl"

    first = advance_lifecycle(
        state_path,
        execute=True,
        graph_runner=lambda config, output: _needs_feature_report(manifest, tmp_path),
        command_runner=lambda command, cwd: CommandResult(
            0, "Submitted batch job 401\n", ""
        ),
    )
    assert first["status"] == "WAITING_FOR_JOBS"

    def failed_runner(command, cwd):
        assert command[0] == "sacct"
        return CommandResult(0, "401|FAILED|1:0\n", "")

    stopped = advance_lifecycle(
        state_path,
        execute=True,
        resume_failed=False,
        graph_runner=lambda config, output: (_ for _ in ()).throw(
            AssertionError("graph must not advance after a failed job")
        ),
        command_runner=failed_runner,
    )

    ledger = load_job_ledger(stopped["stages"][0]["ledger"])
    assert stopped["status"] == "NEEDS_REVIEW"
    assert ledger["status"] == "NEEDS_RESUME"
    assert len(ledger["jobs"][0]["attempts"]) == 1


def test_lifecycle_enforces_failed_job_attempt_limit(tmp_path: Path):
    state_path = _initialize(tmp_path)
    manifest = tmp_path / "PL.jsonl"

    first = advance_lifecycle(
        state_path,
        execute=True,
        graph_runner=lambda config, output: _needs_feature_report(manifest, tmp_path),
        command_runner=lambda command, cwd: CommandResult(
            0, "Submitted batch job 501\n", ""
        ),
    )
    assert first["status"] == "WAITING_FOR_JOBS"

    def retry_runner(command, cwd):
        if command[0] == "sacct":
            return CommandResult(0, "501|FAILED|1:0\n", "")
        assert command[0] == "sbatch"
        return CommandResult(0, "Submitted batch job 502\n", "")

    retried = advance_lifecycle(
        state_path,
        execute=True,
        resume_failed=True,
        max_attempts=2,
        command_runner=retry_runner,
    )
    assert retried["status"] == "WAITING_FOR_JOBS"

    def exhausted_runner(command, cwd):
        assert command[0] == "sacct"
        return CommandResult(0, "502|FAILED|1:0\n", "")

    stopped = advance_lifecycle(
        state_path,
        execute=True,
        resume_failed=True,
        max_attempts=2,
        graph_runner=lambda config, output: (_ for _ in ()).throw(
            AssertionError("graph must not advance after retry exhaustion")
        ),
        command_runner=exhausted_runner,
    )

    ledger = load_job_ledger(stopped["stages"][0]["ledger"])
    assert stopped["status"] == "NEEDS_REVIEW"
    assert stopped["stages"][0]["attempts_exhausted"] == [ledger["jobs"][0]["plan_id"]]
    assert len(ledger["jobs"][0]["attempts"]) == 2


def test_lifecycle_reconciles_then_advances_graph_to_completion(tmp_path: Path):
    state_path = _initialize(tmp_path)
    manifest = tmp_path / "PL.jsonl"
    graph_reports = [
        _needs_feature_report(manifest, tmp_path),
        {
            "report_schema": "mint-agent.graph-run.v1",
            "run_id": "toy-lifecycle",
            "dataset_id": "toy",
            "status": "COMPLETE",
            "feature_jobs": [],
            "qc_jobs": [],
            "filtration_audit_jobs": [],
            "scout_jobs": [],
            "evaluation_jobs": [],
        },
    ]

    def graph_runner(config, output):
        return graph_reports.pop(0)

    def submit_runner(command, cwd):
        return CommandResult(0, "Submitted batch job 201\n", "")

    first = advance_lifecycle(
        state_path,
        execute=True,
        graph_runner=graph_runner,
        command_runner=submit_runner,
    )
    assert first["status"] == "WAITING_FOR_JOBS"

    output_a = tmp_path / "a.npy"
    output_b = tmp_path / "b.npy"
    output_a.write_bytes(b"a")
    output_b.write_bytes(b"b")
    manifest.write_text(
        "\n".join(
            json.dumps(
                {
                    "sample_id": sample_id,
                    "invariant": "PL",
                    "split": "train",
                    "status": "computed",
                    "output_path": str(output),
                }
            )
            for sample_id, output in (("a", output_a), ("b", output_b))
        )
        + "\n",
        encoding="utf-8",
    )

    def completed_runner(command, cwd):
        assert command[0] == "sacct"
        return CommandResult(0, "201|COMPLETED|0:0\n", "")

    final = advance_lifecycle(
        state_path,
        execute=True,
        graph_runner=graph_runner,
        command_runner=completed_runner,
    )

    assert final["status"] == "COMPLETE"
    assert final["active_stage_index"] is None
    assert [stage["status"] for stage in final["stages"]] == [
        "REGISTERED",
        "COMPLETE",
    ]
    records = ArtifactRegistry(tmp_path / "artifacts.sqlite").query(
        artifact_kind="feature_manifest", dataset_id="toy", invariant="PL",
    )
    assert len(records) == 1
    assert load_lifecycle(state_path)["status"] == "COMPLETE"


def test_lifecycle_resume_failed_graph_only_stage(tmp_path: Path):
    state_path = _initialize(tmp_path)
    state = load_lifecycle(state_path)
    graph_report = tmp_path / "graph_000.json"
    graph_report.write_text(
        json.dumps(
            {
                "report_schema": "mint-agent.graph-run.v1",
                "status": "BLOCKED",
                "errors": ["schema mismatch fixed by a later code change"],
            }
        ),
        encoding="utf-8",
    )
    state["active_stage_index"] = 0
    state["status"] = "NEEDS_REVIEW"
    state["stages"].append(
        {
            "iteration": 0,
            "graph_report": str(graph_report),
            "graph_status": "BLOCKED",
            "ledger": None,
            "status": "NEEDS_REVIEW",
            "continuation": None,
            "plan_signature": None,
        }
    )
    write_lifecycle(state, state_path)

    resumed = advance_lifecycle(
        state_path,
        resume_failed=True,
        graph_runner=lambda config, output: {
            "report_schema": "mint-agent.graph-run.v1",
            "run_id": "toy-lifecycle",
            "dataset_id": "toy",
            "status": "COMPLETE",
            "feature_jobs": [],
            "qc_jobs": [],
            "filtration_audit_jobs": [],
            "scout_jobs": [],
            "evaluation_jobs": [],
        },
    )

    assert resumed["status"] == "COMPLETE"
    assert resumed["active_stage_index"] is None
    assert resumed["stages"][0]["status"] == "NEEDS_REVIEW"
    assert resumed["stages"][0]["resume_override_at"]
    assert resumed["stages"][1]["status"] == "COMPLETE"
