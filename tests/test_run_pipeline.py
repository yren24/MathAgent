from pathlib import Path

from mint_scout.config import load_yaml
from mint_scout.run_pipeline import build_pipeline_launch_plan, main, pipeline_status


def test_bdb_pipeline_builds_one_command_for_complete_lifecycle():
    plan = build_pipeline_launch_plan(
        Path("configs/pipelines/bdb2020plus_fullcv_sapelo2.yaml")
    )

    command = list(plan.submit_command)
    assert plan.dataset_id == "bdb2020plus-fullcv"
    assert "INVARIANTS=PL" in command
    assert "EXECUTE=1" in command
    assert "AUTO_CONTINUE=1" in command
    assert command[-1].endswith("scripts/sapelo2/run_agent_lifecycle.sbatch")


def test_atom3d_pipeline_is_configured_for_complete_lifecycle():
    plan = build_pipeline_launch_plan(
        Path("configs/pipelines/atom3d_lba_test_fullcv_sapelo2.yaml")
    )

    command = list(plan.submit_command)
    assert not any(value.startswith("STOP_AFTER_STAGE=") for value in command)
    assert plan.dataset_id == "atom3d-lba-test-fullcv"


def test_official_atom3d_split_pipeline_uses_manifest_driven_graph_routing():
    plan = build_pipeline_launch_plan(
        Path("configs/pipelines/atom3d_lba_identity30_official_sapelo2.yaml")
    )

    assert plan.dataset_id == "atom3d-lba-identity30-official"
    assert "INVARIANTS=PH,PL,CA,FPRC,EIC" in plan.submit_command
    assert "EVIDENCE_SCOPE=full_train" in plan.submit_command


def test_pipeline_status_is_not_started_without_state_or_receipt(tmp_path: Path):
    source = Path("configs/pipelines/bdb2020plus_fullcv_sapelo2.yaml")
    plan = build_pipeline_launch_plan(source)
    isolated = plan.__class__(
        **{
            **plan.__dict__,
            "state_path": str(tmp_path / "state.json"),
            "receipt_path": str(tmp_path / "receipt.json"),
        }
    )

    assert pipeline_status(isolated)["status"] == "NOT_STARTED"


def test_pipeline_plan_cli_prints_the_launch_plan(capsys):
    code = main(
        ["plan", "--config", "configs/pipelines/bdb2020plus_fullcv_sapelo2.yaml",]
    )

    assert code == 0
    assert '"AUTO_CONTINUE=1"' in capsys.readouterr().out


def test_pipeline_config_can_live_outside_repository_with_explicit_root(tmp_path: Path):
    source = Path("configs/pipelines/bdb2020plus_fullcv_sapelo2.yaml").resolve()
    payload = load_yaml(source)
    payload["repository_root"] = str(Path.cwd().resolve())
    payload["task_config"] = str(
        Path("configs/tasks/bdb2020plus_fullcv_sapelo2.yaml").resolve()
    )
    payload["execution_profile"] = str(Path("configs/execution/sapelo2.yaml").resolve())
    destination = tmp_path / "generated" / "pipeline.yaml"
    destination.parent.mkdir()
    import yaml

    destination.write_text(yaml.safe_dump(payload), encoding="utf-8")

    plan = build_pipeline_launch_plan(destination)

    assert plan.repository_root == str(Path.cwd().resolve())
    assert plan.config_path == str(destination)


def test_pipeline_passes_llm_scientific_config_without_secret_value(tmp_path: Path):
    source = Path("configs/pipelines/bdb2020plus_fullcv_sapelo2.yaml").resolve()
    payload = load_yaml(source)
    payload["repository_root"] = str(Path.cwd().resolve())
    payload["task_config"] = str(
        Path("configs/tasks/bdb2020plus_fullcv_sapelo2.yaml").resolve()
    )
    payload["execution_profile"] = str(Path("configs/execution/sapelo2.yaml").resolve())
    payload["llm_scientific"] = {
        "mode": "shadow",
        "model": "gpt-5",
        "api_key_env": "OPENAI_API_KEY",
        "env_file": "~/.config/mathagent/openai.env",
        "cache_dir": "/path/to/workdir/mathagent/llm-scientific-cache",
        "max_candidates": 7,
        "timeout_seconds": 240,
    }
    destination = tmp_path / "pipeline.yaml"
    import yaml

    destination.write_text(yaml.safe_dump(payload), encoding="utf-8")

    plan = build_pipeline_launch_plan(destination)
    command = list(plan.submit_command)

    assert "LLM_SCIENTIFIC_MODE=shadow" in command
    assert "LLM_MODEL=gpt-5" in command
    assert "LLM_API_KEY_ENV=OPENAI_API_KEY" in command
    assert "LLM_MAX_CANDIDATES=7" in command
    assert "LLM_TIMEOUT_SECONDS=240.0" in command
    assert not any("sk-" in value for value in command)
