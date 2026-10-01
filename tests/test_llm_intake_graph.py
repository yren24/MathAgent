from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import mint_scout.agent.workflow as domain_workflow
from mint_scout.agent.llm_intake import (
    LLMIntakeContext,
    _submit_prepared_bundle,
    run_llm_intake,
)
from mint_scout.user_intake import REQUEST_VERSION


def _generated_request() -> dict:
    return {
        "version": REQUEST_VERSION,
        "request_id": "natural-binding-run",
        "goal": "Predict protein-ligand binding affinity.",
        "system_type": "protein_ligand",
        "task_type": "regression",
        "primary_metric": "PCC",
        "execution_profile": None,
        "representation_profile": None,
        "dataset": {
            "manifest_path": None,
            "label_name": "affinity",
            "validate_paths": True,
            "source": None,
            "preparation": None,
            "cv_group_identifier": "pdb_id",
            "columns": {
                "sample_id": "sample_id",
                "target": "target",
                "split": "split",
                "roles": {
                    "protein": "protein_path",
                    "ligand": "ligand_path",
                },
                "identifiers": {"pdb_id": "pdb_id"},
            },
        },
        "run": {
            "run_id": None,
            "invariants": ["PH", "PL", "CA", "FPRC", "EIC"],
            "user_target": None,
            "selection_objective": "satisfy_target",
        },
        "labels": {
            "min_labeled_samples": None,
            "allow_small_data_override": False,
        },
        "label_retrieval_allowed": False,
        "hydrogen_policy": {
            "mode": "auto",
            "task_relevance": "unlikely_relevant",
            "rationale": "Generic binding-affinity prediction does not require explicit H.",
        },
    }


def _generated_toxicity_request() -> dict:
    request = _generated_request()
    request.update(
        {
            "request_id": "natural-toxicity-run",
            "goal": "Predict LD50 toxicity.",
            "system_type": "small_molecule",
            "primary_metric": "PCC2",
        }
    )
    request["dataset"] = {
        "manifest_path": None,
        "label_name": "LD50",
        "validate_paths": True,
        "source": None,
        "preparation": None,
        "cv_group_identifier": None,
        "columns": {
            "sample_id": "sample_id",
            "target": "target",
            "split": "split",
            "roles": {"molecule": "molecule_path"},
            "identifiers": {"smiles": "smiles", "source_filename": "source_filename"},
        },
    }
    request["run"]["selection_objective"] = "maximize_rank1"
    request["hydrogen_policy"] = {
        "mode": "auto",
        "task_relevance": "uncertain",
        "rationale": "Toxicity may or may not require explicit hydrogen.",
    }
    return request


def _interpreter(prompt: str, *, model: str, api_key: str):
    assert prompt == "Run binding affinity prediction."
    assert model == "test-model"
    assert api_key == "top-secret"
    return _generated_request(), {
        "used": True,
        "model_requested": model,
        "used_for_numeric_decisions": False,
    }


def _write_manifest(path: Path) -> Path:
    path.write_text(
        "sample_id,target,split,protein_path,ligand_path,pdb_id\n"
        "one,7.0,train,protein.pdb,ligand.mol2,1abc\n",
        encoding="utf-8",
    )
    return path


def _write_toxicity_manifest(path: Path) -> Path:
    path.write_text(
        "sample_id,target,split,molecule_path,smiles,source_filename\n"
        "one,2.0,train,one.mol2,C,one.mol2\n",
        encoding="utf-8",
    )
    return path


def test_llm_intake_graph_requests_missing_execution_context(tmp_path: Path):
    def unexpected_preparer(*_args, **_kwargs):
        raise AssertionError("preflight must not run with missing paths")

    result = run_llm_intake(
        "Run binding affinity prediction.",
        context=LLMIntakeContext(
            model="test-model",
            api_key="top-secret",
            base_dir=tmp_path,
            interpreter=_interpreter,
            preparer=unexpected_preparer,
        ),
    )

    assert result["status"] == "NEEDS_USER_INPUT"
    assert result["missing_fields"] == [
        "execution_profile",
        "dataset.manifest_path",
    ]
    assert result["node_trace"] == [
        "intent_parser_node",
        "explicit_context_node",
    ]
    assert "top-secret" not in json.dumps(result)


@dataclass
class _FakeBundle:
    status: str = "READY"

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "pipeline_config": "/runs/pipeline.yaml",
        }


def test_llm_intake_graph_applies_explicit_paths_then_runs_preflight(
    tmp_path: Path,
):
    captured: dict = {}

    def preparer(request, **kwargs):
        captured["request"] = request
        captured["kwargs"] = kwargs
        return _FakeBundle()

    profile = tmp_path / "sapelo2.yaml"
    manifest = _write_manifest(tmp_path / "samples.csv")
    result = run_llm_intake(
        "Run binding affinity prediction.",
        context=LLMIntakeContext(
            model="test-model",
            api_key="top-secret",
            base_dir=tmp_path,
            execution_profile=str(profile),
            manifest_path=str(manifest),
            interpreter=_interpreter,
            preparer=preparer,
        ),
    )

    assert result["status"] == "READY"
    assert captured["request"]["execution_profile"] == str(profile)
    assert captured["request"]["dataset"]["manifest_path"] == str(manifest)
    assert captured["kwargs"]["source"] == "openai_responses_api"
    assert result["explicit_context"] == {
        "execution_profile": str(profile),
        "dataset.manifest_path": str(manifest),
    }
    assert result["node_trace"] == [
        "intent_parser_node",
        "explicit_context_node",
        "manifest_profiler_node",
        "deterministic_preflight_node",
    ]


def test_llm_intake_graph_onboards_small_molecule_manifests(tmp_path: Path):
    captured: dict = {}

    def interpreter(prompt: str, *, model: str, api_key: str):
        assert prompt == "Run toxicity prediction."
        return _generated_toxicity_request(), {
            "used": True,
            "model_requested": model,
            "used_for_numeric_decisions": False,
        }

    def preparer(request, **kwargs):
        captured["request"] = request
        captured["kwargs"] = kwargs
        return _FakeBundle()

    manifest = _write_toxicity_manifest(tmp_path / "toxicity.csv")
    result = run_llm_intake(
        "Run toxicity prediction.",
        context=LLMIntakeContext(
            model="test-model",
            api_key="top-secret",
            base_dir=tmp_path,
            execution_profile=str(tmp_path / "sapelo2.yaml"),
            manifest_path=str(manifest),
            interpreter=interpreter,
            preparer=preparer,
        ),
    )

    assert result["status"] == "READY"
    assert captured["request"]["system_type"] == "small_molecule"
    assert captured["request"]["dataset"]["columns"]["roles"] == {
        "molecule": "molecule_path"
    }
    assert result["node_trace"] == [
        "intent_parser_node",
        "explicit_context_node",
        "manifest_profiler_node",
        "deterministic_preflight_node",
    ]


def test_llm_intake_graph_generates_stable_request_id_when_llm_leaves_it_empty(
    tmp_path: Path,
):
    captured: dict = {}

    def interpreter(prompt: str, *, model: str, api_key: str):
        request = _generated_request()
        request["request_id"] = ""
        return request, {"used": True, "used_for_numeric_decisions": False}

    def preparer(request, **_kwargs):
        captured["request"] = request
        return _FakeBundle()

    manifest = _write_manifest(tmp_path / "samples.csv")
    result = run_llm_intake(
        "Run binding affinity prediction.",
        context=LLMIntakeContext(
            model="test-model",
            api_key="top-secret",
            base_dir=tmp_path,
            execution_profile=str(tmp_path / "sapelo2.yaml"),
            manifest_path=str(manifest),
            interpreter=interpreter,
            preparer=preparer,
        ),
    )

    assert result["status"] == "READY"
    assert captured["request"]["request_id"] == "natural-c80b3823d102"


def test_llm_intake_graph_applies_explicit_scientific_advisory_config(
    tmp_path: Path,
):
    captured: dict = {}

    def preparer(request, **_kwargs):
        captured["request"] = request
        return _FakeBundle()

    manifest = _write_manifest(tmp_path / "samples.csv")
    result = run_llm_intake(
        "Run binding affinity prediction.",
        context=LLMIntakeContext(
            model="test-model",
            api_key="top-secret",
            base_dir=tmp_path,
            execution_profile=str(tmp_path / "sapelo2.yaml"),
            manifest_path=str(manifest),
            llm_scientific={
                "mode": "advisory",
                "model": "test-model",
                "env_file": "/secure/openai.env",
            },
            interpreter=_interpreter,
            preparer=preparer,
        ),
    )

    assert result["status"] == "READY"
    assert captured["request"]["llm_scientific"] == {
        "mode": "advisory",
        "model": "test-model",
        "env_file": "/secure/openai.env",
    }


def test_llm_intake_graph_submits_only_after_ready_preflight(tmp_path: Path):
    receipts: list[dict] = []

    def submitter(bundle):
        receipts.append(dict(bundle))
        return {"job_id": "12345"}

    manifest = _write_manifest(tmp_path / "samples.csv")
    result = run_llm_intake(
        "Run binding affinity prediction.",
        context=LLMIntakeContext(
            model="test-model",
            api_key="top-secret",
            base_dir=tmp_path,
            execute=True,
            execution_profile=str(tmp_path / "sapelo2.yaml"),
            manifest_path=str(manifest),
            interpreter=_interpreter,
            preparer=lambda *_args, **_kwargs: _FakeBundle(),
            submitter=submitter,
        ),
    )

    assert result["status"] == "SUBMITTED"
    assert result["launch_receipt"] == {"job_id": "12345"}
    assert receipts == [
        {"status": "READY", "pipeline_config": "/runs/pipeline.yaml"}
    ]
    assert result["node_trace"][-1] == "optional_submit_node"


def test_default_llm_intake_submitter_uses_deterministic_pipeline_plan(monkeypatch):
    captured: dict = {}

    def fake_submit_pipeline(plan):
        captured["plan"] = plan
        return {"job_id": "submitted-by-pipeline"}

    monkeypatch.setattr(domain_workflow, "submit_pipeline", fake_submit_pipeline)

    receipt = _submit_prepared_bundle(
        {
            "status": "READY",
            "pipeline_config": "configs/pipelines/atom3d_lba_identity30_official_sapelo2.yaml",
        }
    )

    assert receipt == {"job_id": "submitted-by-pipeline"}
    assert captured["plan"].dataset_id == "atom3d-lba-identity30-official"
    assert captured["plan"].submit_command[-1].endswith(
        "scripts/sapelo2/run_agent_lifecycle.sbatch"
    )
    assert "EXECUTE=1" in captured["plan"].submit_command
    assert "AUTO_CONTINUE=1" in captured["plan"].submit_command


def test_llm_intake_graph_profiles_and_maps_missing_manifest_columns(tmp_path: Path):
    manifest = _write_manifest(tmp_path / "samples.csv")
    generated = _generated_request()
    generated["dataset"]["columns"] = {
        "sample_id": None,
        "target": None,
        "split": None,
        "roles": {"protein": None, "ligand": None},
        "identifiers": {"pdb_id": None},
    }
    captured: dict = {}

    def interpreter(*_args, **_kwargs):
        return generated, {
            "used": True,
            "model_requested": "test-model",
            "response_id": "resp_intent",
            "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        }

    def mapper(profile, *, request, model, api_key):
        assert profile["privacy"]["raw_values_included"] is False
        assert request["dataset"]["manifest_path"] == str(manifest)
        assert model == "test-model"
        assert api_key == "top-secret"
        return {
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
                    "sample_id", "target", "split", "protein", "ligand", "pdb_id"
                )
            },
            "requires_confirmation": [],
            "evidence": {
                field: "Profile evidence."
                for field in (
                    "sample_id", "target", "split", "protein", "ligand", "pdb_id"
                )
            },
            "summary": "Mapped from the redacted manifest profile.",
        }, {
            "used": True,
            "model_requested": model,
            "response_id": "resp_mapping",
            "usage": {"input_tokens": 20, "output_tokens": 10, "total_tokens": 30},
            "api_key": "must-not-leak",
        }

    def preparer(request, **_kwargs):
        captured["request"] = request
        return _FakeBundle()

    result = run_llm_intake(
        "Run binding affinity prediction.",
        context=LLMIntakeContext(
            model="test-model",
            api_key="top-secret",
            base_dir=tmp_path,
            execution_profile=str(tmp_path / "sapelo2.yaml"),
            manifest_path=str(manifest),
            interpreter=interpreter,
            column_mapper=mapper,
            preparer=preparer,
        ),
    )

    assert result["status"] == "READY"
    assert captured["request"]["dataset"]["columns"]["roles"]["protein"] == "protein_path"
    assert result["onboarding"]["status"] == "MAPPED"
    assert result["llm"]["usage"] == {
        "input_tokens": 30,
        "output_tokens": 15,
        "total_tokens": 45,
    }
    assert [call["stage"] for call in result["llm"]["calls"]] == [
        "intent_parser",
        "dataset_column_mapper",
    ]
    assert "must-not-leak" not in json.dumps(result)
    assert result["node_trace"] == [
        "intent_parser_node",
        "explicit_context_node",
        "manifest_profiler_node",
        "column_mapper_node",
        "deterministic_preflight_node",
    ]


def test_llm_intake_graph_stops_for_ambiguous_required_mapping(tmp_path: Path):
    manifest = _write_manifest(tmp_path / "samples.csv")
    generated = _generated_request()
    generated["dataset"]["columns"] = {
        "sample_id": None,
        "target": None,
        "split": None,
        "roles": {"protein": None, "ligand": None},
        "identifiers": {"pdb_id": None},
    }

    def interpreter(*_args, **_kwargs):
        return generated, {"used": True, "model_requested": "test-model"}

    def mapper(*_args, **_kwargs):
        mapping = {
            "sample_id": "sample_id",
            "target": "target",
            "split": "split",
            "protein": "protein_path",
            "ligand": "ligand_path",
            "pdb_id": "pdb_id",
        }
        return {
            "mapping": mapping,
            "confidence": {**{field: "high" for field in mapping}, "target": "medium"},
            "requires_confirmation": ["target"],
            "evidence": {field: "Profile evidence." for field in mapping},
            "summary": "The target requires confirmation.",
        }, {"used": True, "model_requested": "test-model"}

    result = run_llm_intake(
        "Run binding affinity prediction.",
        context=LLMIntakeContext(
            model="test-model",
            api_key="top-secret",
            base_dir=tmp_path,
            execution_profile=str(tmp_path / "sapelo2.yaml"),
            manifest_path=str(manifest),
            interpreter=interpreter,
            column_mapper=mapper,
            preparer=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("preflight must not run")
            ),
        ),
    )

    assert result["status"] == "NEEDS_USER_INPUT"
    assert result["onboarding"]["status"] == "NEEDS_USER_CONFIRMATION"
    assert result["errors"] == [
        "Manifest column mapping for target requires user confirmation."
    ]
    assert result["node_trace"][-1] == "column_mapper_node"
