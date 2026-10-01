from __future__ import annotations

import json
from pathlib import Path

import pytest

from mint_scout.artifact_memory import ArtifactRegistry
from mint_scout.config import load_yaml
from mint_scout.user_intake import (
    REQUEST_VERSION,
    prepare_user_mapping,
    prepare_user_request,
    request_from_prompt,
)
from mint_scout.agent.workflow import workflow_for_config
import mint_scout.agent.workflow as domain_workflow
from mint_scout.run_pipeline import main as pipeline_main
from mint_scout.user_intake import main as intake_main


def _write_profile(tmp_path: Path) -> Path:
    repository = Path.cwd().resolve()
    profile = tmp_path / "profile.yaml"
    profile.write_text(
        "profile_id: test-slurm\n"
        "profile_type: slurm_hpc\n"
        "host:\n"
        "  login: cluster.example.edu\n"
        "  user: scientist\n"
        "paths:\n"
        f"  project_root: {repository}\n"
        f"  scratch_root: {tmp_path / 'scratch'}\n"
        f"  cache_root: {tmp_path / 'scratch' / 'cache'}\n"
        f"  run_root: {tmp_path / 'scratch' / 'runs'}\n"
        f"  log_root: {tmp_path / 'scratch' / 'logs'}\n"
        "python:\n"
        "  executable: python\n"
        "  module_load: []\n"
        "tools:\n"
        f"  plbind_root: {tmp_path / 'legacy' / 'plbind'}\n"
        "slurm_defaults:\n"
        "  partition: batch\n"
        "  ntasks: 1\n"
        "  cpus_per_task: 1\n"
        "  mem: 2gb\n"
        "  time: '00:15:00'\n",
        encoding="utf-8",
    )
    return profile


def _write_manifest(tmp_path: Path, *, include_splits: bool = True) -> Path:
    rows = ["sample_id,target,split,protein_path,ligand_path,pdb_id"]
    split_values = [
        "train",
        "train",
        "train",
        "train",
        "train",
        "validation",
        "validation",
        "test",
        "test",
    ]
    for index, split in enumerate(split_values):
        protein = tmp_path / f"p{index}.pdb"
        ligand = tmp_path / f"l{index}.mol2"
        protein.write_text("ATOM\n", encoding="utf-8")
        ligand.write_text("@<TRIPOS>MOLECULE\n", encoding="utf-8")
        rows.append(
            f"s{index},{index + 1},{split if include_splits else ''},"
            f"{protein},{ligand},s{index}"
        )
    manifest = tmp_path / "samples.csv"
    manifest.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return manifest


def _write_toxicity_manifest(tmp_path: Path) -> Path:
    molecule_dir = tmp_path / "molecules"
    molecule_dir.mkdir()
    rows = ["sample_id,target,split,molecule_path,smiles,source_filename"]
    splits = ["train", "train", "train", "train", "train", "test", "test"]
    for index, split in enumerate(splits):
        molecule = molecule_dir / f"tox{index}.mol2"
        molecule.write_text("@<TRIPOS>MOLECULE\nsmall\n", encoding="utf-8")
        rows.append(
            f"tox{index},{1.0 + index / 10:.3f},{split},{molecule},C,tox{index}.mol2"
        )
    manifest = tmp_path / "toxicity.csv"
    manifest.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return manifest


def _toxicity_request(*, manifest: Path, profile: Path) -> dict:
    return {
        "version": REQUEST_VERSION,
        "request_id": "toxicity-toy",
        "goal": "Predict LD50 toxicity from small-molecule structures.",
        "system_type": "small_molecule",
        "task_type": "regression",
        "primary_metric": "PCC2",
        "execution_profile": str(profile),
        "dataset": {
            "manifest_path": str(manifest),
            "label_name": "LD50",
            "validate_paths": True,
            "columns": {
                "sample_id": "sample_id",
                "target": "target",
                "split": "split",
                "roles": {"molecule": "molecule_path"},
                "identifiers": {
                    "smiles": "smiles",
                    "source_filename": "source_filename",
                },
            },
        },
        "run": {
            "run_id": "tox-v1",
            "invariants": ["PH", "PL", "CA", "FPRC", "EIC"],
            "selection_objective": "maximize_rank1",
        },
        "labels": {
            "min_labeled_samples": 5,
            "allow_small_data_override": False,
        },
        "label_retrieval_allowed": False,
        "hydrogen_policy": {
            "mode": "auto",
            "task_relevance": "uncertain",
            "rationale": "Let the toxicity representation advisory decide whether H is useful.",
        },
        "llm_scientific": {
            "mode": "advisory",
            "model": "test-model",
            "env_file": "/secure/openai.env",
            "cache_dir": "/scratch/test/llm-cache",
        },
    }


def _write_request(
    tmp_path: Path,
    *,
    manifest: Path,
    profile: Path,
    run_id: str | None = "intake-v1",
    min_labeled_samples: int = 5,
    allow_small_data_override: bool = False,
) -> Path:
    run_id_line = f"  run_id: {run_id}\n" if run_id is not None else ""
    request = tmp_path / "request.yaml"
    request.write_text(
        f"version: {REQUEST_VERSION}\n"
        "request_id: intake-toy\n"
        "goal: Predict binding affinity.\n"
        "system_type: protein_ligand\n"
        "task_type: regression\n"
        "primary_metric: PCC\n"
        f"execution_profile: {profile}\n"
        "labels:\n"
        f"  min_labeled_samples: {min_labeled_samples}\n"
        f"  allow_small_data_override: {str(allow_small_data_override).lower()}\n"
        "dataset:\n"
        f"  manifest_path: {manifest}\n"
        "  label_name: affinity\n"
        "  validate_paths: true\n"
        "  columns:\n"
        "    sample_id: sample_id\n"
        "    target: target\n"
        "    split: split\n"
        "    roles:\n"
        "      protein: protein_path\n"
        "      ligand: ligand_path\n"
        "    identifiers:\n"
        "      pdb_id: pdb_id\n"
        "run:\n"
        f"{run_id_line}"
        "  invariants: [PH, PL, CA, FPRC, EIC]\n",
        encoding="utf-8",
    )
    return request


def test_user_request_generates_valid_split_pipeline_outside_repository(tmp_path: Path):
    profile = _write_profile(tmp_path)
    manifest = _write_manifest(tmp_path)
    request = _write_request(tmp_path, manifest=manifest, profile=profile)
    output = tmp_path / "generated"

    bundle = prepare_user_request(request, output_dir=output)

    assert bundle.status == "READY"
    assert bundle.evaluation_mode == "explicit_validation_and_test"
    assert bundle.launch_plan.repository_root == str(Path.cwd().resolve())
    task = load_yaml(bundle.task_config)
    pipeline = load_yaml(bundle.pipeline_config)
    preflight = json.loads(Path(bundle.preflight_report).read_text(encoding="utf-8"))
    assert task["legacy"]["plbind_root"].endswith("/legacy/plbind")
    assert task["intent"] == {
        "source": "structured_request",
        "llm": {"used": False, "used_for_numeric_decisions": False},
    }
    assert task["dataset_manifest"]["path"] == str(manifest)
    assert task["selection_preferences"] == {
        "method_source": "user",
        "target_source": "probe_derived",
        "selection_objective": "satisfy_target",
        "test_evaluation_policy": "frozen_once",
    }
    assert task["representation_mode"] == "dataset_adaptive"
    assert task["representation_design"]["filtration"]["fixed_point_count"] == 50
    assert task["representation_design"]["support"]["max_element_pair_channels"] == 50
    assert set(task["data_audit"]["tolerated_out_of_schema"]["protein"]) == {
        "H",
        "Ca",
        "Cd",
        "Co",
        "Cs",
        "Cu",
        "F",
        "Fe",
        "Hg",
        "K",
        "Mg",
        "Mn",
        "Na",
        "Ni",
        "P",
        "Se",
        "Sr",
        "Zn",
    }
    assert pipeline["repository_root"] == str(Path.cwd().resolve())
    assert pipeline["task_config"] == str(output / "task.yaml")
    assert preflight["dataset"]["split_counts"] == {
        "test": 2,
        "train": 5,
        "validation": 2,
    }
    assert preflight["protocol_guards"]["test_used_for_selection"] is False
    assert preflight["llm"]["used_for_numeric_decisions"] is False
    assert preflight["representation"]["mode"] == "dataset_adaptive"
    assert preflight["representation"]["profile_hash"]
    assert preflight["method_source"] == "user"
    assert preflight["target_source"] == "probe_derived"
    records = ArtifactRegistry(bundle.launch_plan.registry_path).query(
        artifact_kind="user_preflight", dataset_id="intake-toy",
    )
    assert len(records) == 1
    assert records[0].content_sha256


def test_small_molecule_toxicity_request_generates_toxicity_workflow_bundle(
    tmp_path: Path,
):
    profile = _write_profile(tmp_path)
    manifest = _write_toxicity_manifest(tmp_path)
    request = _toxicity_request(manifest=manifest, profile=profile)

    bundle = prepare_user_mapping(
        request,
        base_dir=tmp_path,
        source="openai_responses_api",
        output_dir=tmp_path / "toxicity-generated",
        llm_provenance={"used": True, "model_requested": "test-model"},
    )

    task = load_yaml(bundle.task_config)
    pipeline = load_yaml(bundle.pipeline_config)
    preflight = json.loads(Path(bundle.preflight_report).read_text(encoding="utf-8"))

    assert bundle.status == "READY"
    assert bundle.pipeline_kind == "small_molecule_toxicity_gbt_v1"
    assert bundle.evaluation_mode == "explicit_labeled_test"
    assert pipeline["version"] == "mint-agent.toxicity-workflow.v1"
    assert pipeline["llm_scientific"]["mode"] == "advisory"
    assert task["system_type"] == "small_molecule"
    assert task["dataset_manifest"]["columns"]["roles"] == {
        "molecule": "molecule_path"
    }
    assert preflight["protocol_guards"]["test_used_for_selection"] is False
    assert preflight["protocol_guards"]["final_test_query_budget"] == 1
    assert preflight["toxicity_workflow"]["final_report"].endswith(
        "/final/gbt/toxicity_final_test_report.json"
    )
    assert any(
        "submit_toxicity_workflow.sh" in part
        for part in bundle.launch_plan.submit_command
    )
    assert "MOLECULE_DIRS=" in " ".join(bundle.launch_plan.submit_command)
    assert workflow_for_config(bundle.pipeline_config).pipeline_kind == bundle.pipeline_kind
    assert "WORKFLOW_JOBS_PATH=" in " ".join(bundle.launch_plan.submit_command)


def test_toxicity_user_methods_metric_and_target_reach_slurm_plan(tmp_path: Path):
    profile = _write_profile(tmp_path)
    manifest = _write_toxicity_manifest(tmp_path)
    request = _toxicity_request(manifest=manifest, profile=profile)
    request["primary_metric"] = "R2"
    request["run"]["invariants"] = ["PL", "EIC"]
    request["run"]["selection_objective"] = "satisfy_target"
    request["run"]["user_target"] = 0.7

    bundle = prepare_user_mapping(
        request, base_dir=tmp_path, source="structured_request",
        output_dir=tmp_path / "toxicity-r2",
    )
    task = load_yaml(bundle.task_config)
    pipeline = load_yaml(bundle.pipeline_config)
    command = bundle.launch_plan.submit_command
    assert task["invariants"] == ["PL", "EIC"]
    assert task["selection_preferences"]["user_target"] == 0.7
    assert pipeline["primary_metric"] == "R2"
    assert pipeline["selection_objective"] == "satisfy_target"
    assert "PRIMARY_METRIC=R2" in command
    assert f"TASK_CONFIG={bundle.task_config}" in command


def test_toxicity_rejects_target_objective_without_target(tmp_path: Path):
    profile = _write_profile(tmp_path)
    manifest = _write_toxicity_manifest(tmp_path)
    request = _toxicity_request(manifest=manifest, profile=profile)
    request["run"]["selection_objective"] = "satisfy_target"
    with pytest.raises(ValueError, match="requires run.user_target"):
        prepare_user_mapping(
            request, base_dir=tmp_path, source="structured_request",
            output_dir=tmp_path / "invalid-target",
        )


def test_both_domain_workflows_preserve_their_existing_launch_plans(tmp_path: Path):
    profile = _write_profile(tmp_path)
    binding_manifest = _write_manifest(tmp_path)
    binding_request = _write_request(
        tmp_path, manifest=binding_manifest, profile=profile,
    )
    binding = prepare_user_request(
        binding_request, output_dir=tmp_path / "binding-run",
    )
    toxicity_manifest = _write_toxicity_manifest(tmp_path)
    toxicity = prepare_user_mapping(
        _toxicity_request(manifest=toxicity_manifest, profile=profile),
        base_dir=tmp_path,
        source="structured_request",
        output_dir=tmp_path / "toxicity-run",
    )

    assert binding.launch_plan.submit_command[-1].endswith(
        "scripts/slurm/run_agent_lifecycle.sbatch"
    )
    assert toxicity.launch_plan.submit_command[-1].endswith(
        "scripts/sapelo2/submit_toxicity_workflow.sh"
    )
    assert workflow_for_config(binding.pipeline_config).pipeline_kind == binding.pipeline_kind
    assert workflow_for_config(toxicity.pipeline_config).pipeline_kind == toxicity.pipeline_kind
    assert load_yaml(binding.task_config)["primary_metric"] == "pcc"
    assert load_yaml(toxicity.task_config)["primary_metric"] == "pcc2"


def test_structured_toxicity_submission_records_state(tmp_path: Path, monkeypatch, capsys):
    profile = _write_profile(tmp_path)
    manifest = _write_toxicity_manifest(tmp_path)
    request_path = tmp_path / "toxicity-request.yaml"
    import yaml

    request_path.write_text(
        yaml.safe_dump(_toxicity_request(manifest=manifest, profile=profile)),
        encoding="utf-8",
    )
    submitted = []

    def fake_submit(plan):
        submitted.append(plan)
        return {"job_id": "12345"}

    monkeypatch.setattr(domain_workflow, "submit_pipeline", fake_submit)
    assert intake_main([
        "--request", str(request_path),
        "--output-dir", str(tmp_path / "submitted"),
        "--execute",
    ]) == 0
    output = json.loads(capsys.readouterr().out)
    state = json.loads(Path(submitted[0].state_path).read_text(encoding="utf-8"))
    assert output["status"] == "SUBMITTED"
    assert state["status"] == "SUBMITTED"
    assert state["receipt"]["job_id"] == "12345"


def test_toxicity_uses_common_pipeline_plan_status_and_report_cli(
    tmp_path: Path, capsys,
):
    profile = _write_profile(tmp_path)
    manifest = _write_toxicity_manifest(tmp_path)
    bundle = prepare_user_mapping(
        _toxicity_request(manifest=manifest, profile=profile),
        base_dir=tmp_path,
        source="structured_request",
        output_dir=tmp_path / "toxicity-cli",
    )
    args = ["--config", bundle.pipeline_config]
    assert pipeline_main(["plan", *args]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["submit_command"][-1].endswith("submit_toxicity_workflow.sh")
    assert pipeline_main(["status", *args]) == 2
    assert json.loads(capsys.readouterr().out)["status"] == "NOT_STARTED"

    final_path = (
        Path(bundle.launch_plan.state_path).parent
        / "final" / "gbt" / "toxicity_final_test_report.json"
    )
    final_path.parent.mkdir(parents=True)
    final_path.write_text(json.dumps({
        "report_schema": "mint-agent.toxicity-final-evaluation.v1",
        "status": "COMPLETE",
        "selected_subset": ["EIC", "FPRC"],
        "primary_metric": "PCC2",
        "protocol": {"test_used_for_selection": False},
        "final_test": {"metrics": {"PCC2": 0.62, "R2": 0.61}},
    }), encoding="utf-8")
    assert pipeline_main(["status", *args]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "COMPLETE"
    assert pipeline_main(["report", *args]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["selected_subset"] == ["EIC", "FPRC"]
    assert report["final_metrics"]["PCC2"] == 0.62


def test_toxicity_status_reports_failed_final_job(tmp_path: Path, monkeypatch):
    from mint_scout.toxicity import workflow as toxicity_workflow

    profile = _write_profile(tmp_path)
    manifest = _write_toxicity_manifest(tmp_path)
    bundle = prepare_user_mapping(
        _toxicity_request(manifest=manifest, profile=profile),
        base_dir=tmp_path,
        source="structured_request",
        output_dir=tmp_path / "toxicity-status",
    )
    plan = bundle.launch_plan
    toxicity_workflow.write_toxicity_workflow_state(
        launch_plan=plan, receipt={"job_id": "111"},
    )
    jobs_path = Path(next(
        argument.removeprefix("WORKFLOW_JOBS_PATH=")
        for argument in plan.submit_command
        if argument.startswith("WORKFLOW_JOBS_PATH=")
    ))
    jobs_path.parent.mkdir(parents=True, exist_ok=True)
    jobs_path.write_text(json.dumps({
        "dataset_id": plan.dataset_id,
        "run_id": plan.run_id,
        "launch_job_id": "111",
        "final_combine_job": "222",
    }), encoding="utf-8")
    monkeypatch.setattr(
        toxicity_workflow, "query_slurm_job",
        lambda job_id: {"state": "FAILED" if job_id == "222" else "COMPLETED"},
    )
    status = toxicity_workflow.toxicity_workflow_status(plan)
    assert status["status"] == "FAILED"
    assert status["final_job_id"] == "222"


def test_small_molecule_toxicity_normalizes_pcc_squared_metric(tmp_path: Path):
    profile = _write_profile(tmp_path)
    manifest = _write_toxicity_manifest(tmp_path)
    request = _toxicity_request(manifest=manifest, profile=profile)
    request["primary_metric"] = "PCC squared"

    bundle = prepare_user_mapping(
        request,
        base_dir=tmp_path,
        source="openai_responses_api",
        output_dir=tmp_path / "toxicity-pcc-squared",
    )

    task = load_yaml(bundle.task_config)
    pipeline = load_yaml(bundle.pipeline_config)
    preflight = json.loads(Path(bundle.preflight_report).read_text(encoding="utf-8"))
    assert task["primary_metric"] == "pcc2"
    assert pipeline["primary_metric"] == "PCC2"
    assert preflight["primary_metric"] == "PCC2"


def test_omitted_methods_and_target_use_audited_defaults(tmp_path: Path):
    profile = _write_profile(tmp_path)
    manifest = _write_manifest(tmp_path)
    request = load_yaml(_write_request(tmp_path, manifest=manifest, profile=profile))
    request["run"].pop("invariants")

    bundle = prepare_user_mapping(
        request, base_dir=tmp_path, source="test", output_dir=tmp_path / "defaulted",
    )

    task = load_yaml(bundle.task_config)
    pipeline = load_yaml(bundle.pipeline_config)
    preflight = json.loads(Path(bundle.preflight_report).read_text(encoding="utf-8"))
    assert task["invariants"] == ["PH", "PL", "CA", "FPRC", "EIC"]
    assert task["selection_preferences"] == {
        "method_source": "default_all_supported",
        "target_source": "probe_derived",
        "selection_objective": "satisfy_target",
        "test_evaluation_policy": "frozen_once",
    }
    assert "user_target" not in pipeline
    assert preflight["method_source"] == "default_all_supported"
    assert preflight["target_source"] == "probe_derived"


def test_maximize_rank1_objective_is_propagated_to_generated_configs(
    tmp_path: Path,
):
    profile = _write_profile(tmp_path)
    manifest = _write_manifest(tmp_path)
    request = load_yaml(_write_request(tmp_path, manifest=manifest, profile=profile))
    request["run"]["selection_objective"] = "maximize_rank1"

    bundle = prepare_user_mapping(
        request,
        base_dir=tmp_path,
        source="test",
        output_dir=tmp_path / "maximize-rank1",
    )
    task = load_yaml(bundle.task_config)
    pipeline = load_yaml(bundle.pipeline_config)
    preflight = json.loads(Path(bundle.preflight_report).read_text(encoding="utf-8"))
    resolved = load_yaml(Path(bundle.pipeline_config).with_name("request.resolved.yaml"))

    assert task["selection_preferences"]["selection_objective"] == "maximize_rank1"
    assert pipeline["selection_objective"] == "maximize_rank1"
    assert preflight["selection_objective"] == "maximize_rank1"
    assert resolved["run"]["selection_objective"] == "maximize_rank1"


def test_maximize_top_k_objective_is_propagated_to_generated_configs(
    tmp_path: Path,
):
    profile = _write_profile(tmp_path)
    manifest = _write_manifest(tmp_path)
    request = load_yaml(_write_request(tmp_path, manifest=manifest, profile=profile))
    request["run"]["selection_objective"] = "maximize_top_k"
    request["run"]["max_candidate_rank"] = 4

    bundle = prepare_user_mapping(
        request,
        base_dir=tmp_path,
        source="test",
        output_dir=tmp_path / "maximize-top-k",
    )
    task = load_yaml(bundle.task_config)
    pipeline = load_yaml(bundle.pipeline_config)
    preflight = json.loads(Path(bundle.preflight_report).read_text(encoding="utf-8"))
    resolved = load_yaml(Path(bundle.pipeline_config).with_name("request.resolved.yaml"))

    assert task["selection_preferences"]["selection_objective"] == "maximize_top_k"
    assert task["selection_preferences"]["max_candidate_rank"] == 4
    assert pipeline["selection_objective"] == "maximize_top_k"
    assert pipeline["max_candidate_rank"] == 4
    assert preflight["selection_objective"] == "maximize_top_k"
    assert preflight["max_candidate_rank"] == 4
    assert resolved["run"]["selection_objective"] == "maximize_top_k"
    assert resolved["run"]["max_candidate_rank"] == 4


def test_llm_provenance_is_allowlisted_before_configs_are_written(tmp_path: Path):
    profile = _write_profile(tmp_path)
    manifest = _write_manifest(tmp_path)
    request = load_yaml(_write_request(tmp_path, manifest=manifest, profile=profile))

    bundle = prepare_user_mapping(
        request,
        base_dir=tmp_path,
        source="openai_responses_api",
        output_dir=tmp_path / "llm-generated",
        llm_provenance={
            "used": True,
            "provider": "openai",
            "model_requested": "test-model",
            "response_id": "resp_1",
            "prompt_hash": "prompt-hash",
            "api_key": "must-not-be-written",
            "arbitrary_secret": "must-not-be-written",
            "used_for_numeric_decisions": True,
            "calls": [
                {
                    "stage": "intent_parser",
                    "model_requested": "test-model",
                    "response_id": "resp_1",
                    "usage": {
                        "input_tokens": 12,
                        "output_tokens": 4,
                        "total_tokens": 16,
                    },
                    "api_key": "nested-secret",
                }
            ],
        },
        dataset_onboarding={
            "report_schema": "mint-agent.dataset-column-mapping.v1",
            "status": "MAPPED",
            "numeric_decisions_allowed": True,
            "manifest_profile": {
                "report_schema": "mint-agent.dataset-onboarding.v1",
                "format": "csv",
                "row_count": 9,
                "rows_profiled": 9,
                "profile_limit": 200,
                "columns": [
                    {
                        "name": "target",
                        "profiled_count": 9,
                        "non_empty_count": 9,
                        "missing_count": 0,
                        "numeric_fraction": 1.0,
                        "path_like_fraction": 0.0,
                        "unique_fraction": 1.0,
                        "file_suffixes": [],
                        "recognized_split_values": [],
                        "raw_value": "must-not-be-written",
                    }
                ],
                "raw_rows": ["nested-secret"],
            },
            "mapping": {
                "mapping": {
                    "sample_id": "sample_id",
                    "target": "target",
                    "split": "split",
                    "protein": "protein_path",
                    "ligand": "ligand_path",
                    "pdb_id": "pdb_id",
                },
                "confidence": {
                    field: "high"
                    for field in (
                        "sample_id",
                        "target",
                        "split",
                        "protein",
                        "ligand",
                        "pdb_id",
                    )
                },
                "requires_confirmation": [],
                "evidence": {
                    field: "Profile evidence."
                    for field in (
                        "sample_id",
                        "target",
                        "split",
                        "protein",
                        "ligand",
                        "pdb_id",
                    )
                },
                "summary": "Mapped from a redacted profile.",
                "secret": "nested-secret",
            },
            "llm": {
                "stage": "dataset_column_mapper",
                "model_requested": "test-model",
                "response_id": "resp_2",
                "api_key": "nested-secret",
            },
            "secret": "nested-secret",
        },
    )

    task = load_yaml(bundle.task_config)
    preflight = json.loads(Path(bundle.preflight_report).read_text(encoding="utf-8"))
    serialized = json.dumps({"task": task, "preflight": preflight})
    assert task["intent"]["source"] == "openai_responses_api"
    assert task["intent"]["llm"]["model_requested"] == "test-model"
    assert task["intent"]["llm"]["used_for_numeric_decisions"] is False
    assert task["intent"]["llm"]["calls"][0]["stage"] == "intent_parser"
    assert task["intent"]["llm"]["calls"][0]["usage"]["total_tokens"] == 16
    onboarding = task["intent"]["dataset_onboarding"]
    assert onboarding["numeric_decisions_allowed"] is False
    assert onboarding["manifest_profile_hash"]
    assert onboarding["mapping"]["mapping"]["target"] == "target"
    assert onboarding["llm"]["response_id"] == "resp_2"
    assert preflight["dataset_onboarding"] == onboarding
    assert "must-not-be-written" not in serialized
    assert "nested-secret" not in serialized


def test_user_request_passes_bounded_execution_control_to_pipeline(tmp_path: Path):
    profile = _write_profile(tmp_path)
    manifest = _write_manifest(tmp_path)
    request = _write_request(tmp_path, manifest=manifest, profile=profile)
    request.write_text(
        request.read_text(encoding="utf-8") + "  stop_after_stage: 0\n",
        encoding="utf-8",
    )

    bundle = prepare_user_request(request)

    pipeline = load_yaml(bundle.pipeline_config)
    assert pipeline["stop_after_stage"] == 0


def test_user_request_passes_llm_scientific_config_to_pipeline(tmp_path: Path):
    profile = _write_profile(tmp_path)
    manifest = _write_manifest(tmp_path)
    request = _write_request(tmp_path, manifest=manifest, profile=profile)
    request.write_text(
        request.read_text(encoding="utf-8")
        + "llm_scientific:\n"
        + "  mode: shadow\n"
        + "  model: gpt-5\n"
        + "  api_key_env: OPENAI_API_KEY\n"
        + "  env_file: ~/.config/mathagent/openai.env\n"
        + "  max_candidates: 6\n"
        + "  timeout_seconds: 240\n",
        encoding="utf-8",
    )

    bundle = prepare_user_request(request, output_dir=tmp_path / "generated-llm")

    pipeline = load_yaml(bundle.pipeline_config)
    preflight = json.loads(Path(bundle.preflight_report).read_text(encoding="utf-8"))
    assert pipeline["llm_scientific"] == {
        "mode": "shadow",
        "model": "gpt-5",
        "api_key_env": "OPENAI_API_KEY",
        "env_file": "~/.config/mathagent/openai.env",
        "max_candidates": 6,
        "timeout_seconds": 240.0,
    }
    assert preflight["protocol_guards"]["llm_scientific_mode"] == "shadow"
    assert not any("sk-" in value for value in bundle.launch_plan.submit_command)


def test_user_request_accepts_a_configurable_representation_profile(tmp_path: Path):
    profile = _write_profile(tmp_path)
    manifest = _write_manifest(tmp_path)
    request = _write_request(tmp_path, manifest=manifest, profile=profile)
    source = Path("configs/representation/protein_ligand_adaptive_v1.yaml")
    custom = tmp_path / "representation.yaml"
    custom.write_text(
        source.read_text(encoding="utf-8").replace(
            "fixed_point_count: 50", "fixed_point_count: 25"
        ),
        encoding="utf-8",
    )
    request.write_text(
        request.read_text(encoding="utf-8").replace(
            "dataset:\n", f"representation_profile: {custom}\ndataset:\n"
        ),
        encoding="utf-8",
    )

    bundle = prepare_user_request(request)

    task = load_yaml(bundle.task_config)
    preflight = json.loads(Path(bundle.preflight_report).read_text(encoding="utf-8"))
    assert task["representation_design"]["filtration"]["fixed_point_count"] == 25
    assert preflight["representation"]["profile"] == str(custom)


def test_request_without_split_routes_to_full_labeled_cv_and_stable_run_id(
    tmp_path: Path,
):
    profile = _write_profile(tmp_path)
    manifest = _write_manifest(tmp_path, include_splits=False)
    request = _write_request(tmp_path, manifest=manifest, profile=profile, run_id=None)

    first = prepare_user_request(request)
    second = prepare_user_request(request)

    assert first.evaluation_mode == "full_labeled_cv"
    assert first.run_id == second.run_id
    assert first.run_id.startswith("intake-toy-")


def test_small_labeled_modeling_pool_requires_explicit_confirmation(tmp_path: Path):
    profile = _write_profile(tmp_path)
    manifest = _write_manifest(tmp_path, include_splits=False)
    request = _write_request(
        tmp_path, manifest=manifest, profile=profile, min_labeled_samples=300,
    )

    bundle = prepare_user_request(request)

    preflight = json.loads(Path(bundle.preflight_report).read_text(encoding="utf-8"))
    assert bundle.status == "NEEDS_SMALL_DATA_CONFIRMATION"
    assert preflight["status"] == "NEEDS_SMALL_DATA_CONFIRMATION"
    assert preflight["protocol_guards"]["small_data_override_used"] is False
    assert any(
        "SMALL_LABELED_MODELING_POOL" in warning for warning in preflight["warnings"]
    )


def test_explicit_small_data_override_is_warned_and_propagated(tmp_path: Path):
    profile = _write_profile(tmp_path)
    manifest = _write_manifest(tmp_path, include_splits=False)
    request = _write_request(
        tmp_path,
        manifest=manifest,
        profile=profile,
        min_labeled_samples=300,
        allow_small_data_override=True,
    )

    bundle = prepare_user_request(request)

    task = load_yaml(bundle.task_config)
    preflight = json.loads(Path(bundle.preflight_report).read_text(encoding="utf-8"))
    assert bundle.status == "READY"
    assert task["labels"] == {
        "min_labeled_samples": 300,
        "allow_small_data_override": True,
    }
    assert preflight["protocol_guards"]["small_data_override_used"] is True
    assert any("engineering run" in warning for warning in preflight["warnings"])


def test_request_propagates_an_explicit_cv_group_identifier(tmp_path: Path):
    profile = _write_profile(tmp_path)
    manifest = _write_manifest(tmp_path, include_splits=False)
    request = _write_request(tmp_path, manifest=manifest, profile=profile)
    request.write_text(
        request.read_text(encoding="utf-8").replace(
            "  label_name: affinity\n",
            "  label_name: affinity\n  cv_group_identifier: pdb_id\n",
        ),
        encoding="utf-8",
    )

    bundle = prepare_user_request(request)

    task = load_yaml(bundle.task_config)
    preflight = json.loads(Path(bundle.preflight_report).read_text(encoding="utf-8"))
    assert task["cv"]["group_identifier"] == "pdb_id"
    assert preflight["protocol_guards"]["grouped_cv_identifier"] == "pdb_id"


def test_request_refuses_missing_manifest_without_preparation(tmp_path: Path):
    profile = _write_profile(tmp_path)
    request = _write_request(
        tmp_path, manifest=tmp_path / "missing.csv", profile=profile,
    )

    with pytest.raises(FileNotFoundError, match="dataset.preparation"):
        prepare_user_request(request)


def test_request_rejects_exact_structure_leakage_across_splits(tmp_path: Path):
    profile = _write_profile(tmp_path)
    manifest = _write_manifest(tmp_path)
    rows = manifest.read_text(encoding="utf-8").splitlines()
    train = rows[1].split(",")
    test = rows[-1].split(",")
    test[3:5] = train[3:5]
    rows[-1] = ",".join(test)
    manifest.write_text("\n".join(rows) + "\n", encoding="utf-8")
    request = _write_request(tmp_path, manifest=manifest, profile=profile)

    with pytest.raises(ValueError, match="protected splits"):
        prepare_user_request(request)


def test_request_rejects_identifier_group_leakage_across_splits(tmp_path: Path):
    profile = _write_profile(tmp_path)
    manifest = _write_manifest(tmp_path)
    rows = manifest.read_text(encoding="utf-8").splitlines()
    first = rows[1].split(",")
    last = rows[-1].split(",")
    last[-1] = first[-1]
    rows[-1] = ",".join(last)
    manifest.write_text("\n".join(rows) + "\n", encoding="utf-8")
    request = _write_request(tmp_path, manifest=manifest, profile=profile)

    with pytest.raises(ValueError, match="identifier groups"):
        prepare_user_request(request)


def test_unsplit_repeated_identifier_groups_are_reported(tmp_path: Path):
    profile = _write_profile(tmp_path)
    manifest = _write_manifest(tmp_path, include_splits=False)
    rows = manifest.read_text(encoding="utf-8").splitlines()
    first = rows[1].split(",")
    second = rows[2].split(",")
    second[-1] = first[-1]
    rows[2] = ",".join(second)
    manifest.write_text("\n".join(rows) + "\n", encoding="utf-8")
    request = _write_request(tmp_path, manifest=manifest, profile=profile)

    bundle = prepare_user_request(request)

    preflight = json.loads(Path(bundle.preflight_report).read_text(encoding="utf-8"))
    groups = preflight["dataset"]["identifier_groups"]["pdb_id"]
    assert groups["repeated_group_count"] == 1
    assert groups["maximum_group_size"] == 2
    assert any("sample-level CV" in warning for warning in preflight["warnings"])


def test_request_with_preparation_defers_manifest_validation(tmp_path: Path):
    profile = _write_profile(tmp_path)
    manifest = tmp_path / "prepared.csv"
    metadata = tmp_path / "metadata.csv"
    structures = tmp_path / "structures"
    metadata.write_text("sample_id,target\na,1.0\n", encoding="utf-8")
    structures.mkdir()
    request = _write_request(tmp_path, manifest=manifest, profile=profile)
    text = request.read_text(encoding="utf-8").replace(
        "  columns:\n",
        "  preparation:\n"
        "    provider: paired_structure_manifest\n"
        f"    metadata_path: {metadata}\n"
        f"    structures_root: {structures}\n"
        "    columns:\n"
        "      sample_id: sample_id\n"
        "      target: target\n"
        "    templates:\n"
        "      protein: '{sample_id}/{sample_id}_protein.pdb'\n"
        "      ligand: '{sample_id}/{sample_id}_ligand.mol2'\n"
        "  columns:\n",
    )
    request.write_text(text, encoding="utf-8")

    bundle = prepare_user_request(request)

    assert bundle.status == "READY_FOR_PREPARATION"
    assert bundle.evaluation_mode is None
    task = load_yaml(bundle.task_config)
    assert task["dataset_preparation"]["output_manifest"] == str(manifest)


def test_hiqbind_preparation_is_validated_during_user_preflight(tmp_path: Path):
    profile = _write_profile(tmp_path)
    manifest = tmp_path / "hiqbind" / "manifest.csv"
    request = _write_request(tmp_path, manifest=manifest, profile=profile)
    text = request.read_text(encoding="utf-8").replace(
        "  columns:\n",
        "  preparation:\n"
        "    provider: hiqbind\n"
        f"    metadata_path: {tmp_path / 'hiqbind.csv'}\n"
        f"    source_root: {tmp_path / 'raw_data_hiq_sm'}\n"
        f"    output_root: {tmp_path / 'prepared'}\n"
        "    measurements: [ki, potency]\n"
        "    affinity_signs: ['=']\n"
        "  columns:\n",
    )
    request.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported HiQBind measurements"):
        prepare_user_request(request)


def test_hiqbind_stratified_selection_is_normalized_during_preflight(tmp_path: Path):
    profile = _write_profile(tmp_path)
    manifest = tmp_path / "hiqbind" / "manifest.csv"
    request = _write_request(tmp_path, manifest=manifest, profile=profile)
    text = request.read_text(encoding="utf-8").replace(
        "  columns:\n",
        "  preparation:\n"
        "    provider: hiqbind\n"
        f"    metadata_path: {tmp_path / 'hiqbind.csv'}\n"
        f"    source_root: {tmp_path / 'raw_data_hiq_sm'}\n"
        f"    output_root: {tmp_path / 'prepared'}\n"
        "    measurements: [kd, ki]\n"
        "    affinity_signs: ['=']\n"
        "    max_samples: 500\n"
        "    expected_sample_count: 500\n"
        "    selection:\n"
        "      strategy: stratified\n"
        "      random_seed: 2026\n"
        "      target_quantile_bins: 10\n"
        "      ligand_size_quantile_bins: 5\n"
        "      max_per_pdb_id: 1\n"
        "  columns:\n",
    )
    request.write_text(text, encoding="utf-8")

    bundle = prepare_user_request(request)
    task = load_yaml(bundle.task_config)

    assert bundle.status == "READY_FOR_PREPARATION"
    assert task["dataset_preparation"]["selection"] == {
        "strategy": "stratified",
        "random_seed": 2026,
        "target_quantile_bins": 10,
        "ligand_size_quantile_bins": 5,
        "max_per_pdb_id": 1,
    }


class _FakeResponse:
    def __init__(self, payload: dict):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


def test_natural_language_intake_uses_strict_non_numeric_llm_boundary():
    generated = {
        "version": REQUEST_VERSION,
        "request_id": "prompt-toy",
        "goal": "Predict binding affinity.",
        "system_type": "protein_ligand",
        "task_type": "regression",
        "primary_metric": "PCC",
        "execution_profile": "/cluster/profile.yaml",
        "representation_profile": None,
        "dataset": {
            "manifest_path": "/data/samples.csv",
            "label_name": "affinity",
            "cv_group_identifier": None,
            "validate_paths": True,
            "source": None,
            "preparation": None,
            "columns": {
                "sample_id": "sample_id",
                "target": "target",
                "split": "split",
                "roles": {"protein": "protein_path", "ligand": "ligand_path"},
                "identifiers": {"pdb_id": "pdb_id"},
            },
        },
        "run": {
            "run_id": None,
            "invariants": ["PH", "PL", "CA", "FPRC", "EIC"],
            "user_target": None,
            "selection_objective": "maximize_rank1",
            "max_candidate_rank": None,
            "test_evaluation_policy": None,
        },
        "labels": {"min_labeled_samples": None, "allow_small_data_override": False,},
        "label_retrieval_allowed": False,
        "hydrogen_policy": {
            "mode": "auto",
            "task_relevance": "unlikely_relevant",
            "rationale": "Explicit H is not required for this binding-affinity request.",
        },
    }
    captured: dict = {}

    def opener(request, *, timeout):
        captured["body"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return _FakeResponse(
            {
                "id": "resp_123",
                "model": "new-model",
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {"type": "output_text", "text": json.dumps(generated)}
                        ],
                    }
                ],
            }
        )

    request, provenance = request_from_prompt(
        "Use my labeled manifest and return the best practical result.",
        model="new-model",
        api_key="secret",
        opener=opener,
    )

    assert request == generated
    assert captured["body"]["store"] is False
    assert captured["body"]["text"]["format"]["strict"] is True
    assert "best-available mode" in captured["body"]["instructions"]
    assert request["run"]["selection_objective"] == "maximize_rank1"
    assert provenance["response_id"] == "resp_123"
    assert provenance["used_for_numeric_decisions"] is False

    generated["execution_profile"] = "GBT"
    generated["representation_profile"] = "adaptive"
    generated["dataset"]["manifest_path"] = "ATOM3D"
    sanitized, corrected_provenance = request_from_prompt(
        "Use GBT on my supplied dataset.",
        model="new-model",
        api_key="secret",
        opener=opener,
    )

    assert sanitized["execution_profile"] is None
    assert sanitized["representation_profile"] is None
    assert sanitized["dataset"]["manifest_path"] is None
    assert corrected_provenance["discarded_fields"] == [
        "execution_profile",
        "representation_profile",
        "dataset.manifest_path",
    ]
    assert "GBT is a model family" in captured["body"]["instructions"]
