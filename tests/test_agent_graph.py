from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("langgraph")

from langgraph.checkpoint.memory import InMemorySaver

from mint_scout.agent.graph import (
    AgentGraphContext,
    _controlled_experiment_output_dir,
    build_agent_graph,
)
from mint_scout.agent.llm_scientific import LLMScientificContext
from mint_scout.artifact_memory import ArtifactRecord, ArtifactRegistry
from mint_scout.config import load_yaml
from mint_scout.data_audit import dataset_audit_input_hash
from mint_scout.design_representation import representation_design_input_hash
from mint_scout.execution.jobs import (
    CommandResult,
    create_job_ledger,
    reconcile_completed_jobs,
    refresh_job_ledger,
    submit_ready_jobs,
)
from mint_scout.run_agent_graph import _exit_code, main as run_graph_main
from mint_scout.scout.pipeline import ScoutConfig

_TEST_GBT_HASH = ScoutConfig.from_mapping(
    load_yaml("configs/scout/v1.yaml")
).gbt.parameter_hash


def test_controlled_experiment_output_dir_is_representation_scoped(
    tmp_path: Path,
):
    matrix = {"experiment_hash": "experiment-abc"}

    legacy = _controlled_experiment_output_dir(
        tmp_path,
        dataset_id="toy",
        matrix=matrix,
        representation_hash=None,
    )
    first = _controlled_experiment_output_dir(
        tmp_path,
        dataset_id="toy",
        matrix=matrix,
        representation_hash="1a1b76bf85eb9999",
    )
    repaired = _controlled_experiment_output_dir(
        tmp_path,
        dataset_id="toy",
        matrix=matrix,
        representation_hash="43fd1b5d62536957",
    )

    assert legacy == tmp_path / "controlled_experiments" / "toy" / "experiment-abc"
    assert first == legacy / "1a1b76bf85eb9999"
    assert repaired == legacy / "43fd1b5d62536957"
    assert first != repaired


def _record(
    kind,
    path,
    *,
    scope="probe",
    invariants=(),
    representation_hash=None,
    selection_hash=None,
    status="COMPLETE",
    sample_order_hash="order",
    metadata=None,
):
    resolved_metadata = dict(metadata or {})
    if kind == "scout_execution_plan":
        resolved_metadata.setdefault("frozen_priority_order", [list(invariants)])
        resolved_metadata.setdefault("gbt_parameter_hash", _TEST_GBT_HASH)
        resolved_metadata.setdefault("ranking_policy", "hierarchical_empirical_v1")
        resolved_metadata.setdefault("target_metric", "PCC")
        resolved_metadata.setdefault("target_value", 0.7)
        resolved_metadata.setdefault("fold_assignment_hash", "folds")
    if kind == "model_evaluation":
        resolved_metadata.setdefault("gbt_parameter_hash", _TEST_GBT_HASH)
        resolved_metadata.setdefault("target_metric", "PCC")
        resolved_metadata.setdefault("target_value", 0.7)
        resolved_metadata.setdefault("fold_assignment_hash", "folds")
    return ArtifactRecord(
        artifact_id=f"id-{kind}-{path.name}",
        artifact_kind=kind,
        path=str(path),
        content_sha256="sha",
        dataset_id="toy",
        split="train",
        evidence_scope=scope,
        invariants=tuple(invariants),
        representation_hash=representation_hash,
        selection_hash=selection_hash,
        sample_count=2,
        sample_order_hash=sample_order_hash,
        status=status,
        report_schema=None,
        metadata=resolved_metadata,
    )


def _register_probe_prerequisites(
    registry: ArtifactRegistry,
    tmp_path: Path,
    invariants: tuple[str, ...],
    *,
    representation_hash: str = "repr",
    selection_hash: str = "selection",
) -> None:
    records = [
        _record(
            "probe_selection",
            tmp_path / "probe-selection.json",
            representation_hash=representation_hash,
            selection_hash=selection_hash,
        ),
        _record(
            "probe_audit",
            tmp_path / "probe-audit.json",
            representation_hash=representation_hash,
            selection_hash=selection_hash,
        ),
        *[
            _record(
                "feature_manifest",
                tmp_path / f"probe-{invariant}.jsonl",
                invariants=(invariant,),
                representation_hash=representation_hash,
                selection_hash=selection_hash,
            )
            for invariant in invariants
        ],
        _record(
            "feature_qc",
            tmp_path / "probe-qc.json",
            invariants=invariants,
            representation_hash=representation_hash,
            selection_hash=selection_hash,
        ),
        _record(
            "filtration_audit",
            tmp_path / "probe-filtration.json",
            invariants=invariants,
            representation_hash=representation_hash,
            selection_hash=selection_hash,
        ),
    ]
    for record in records:
        registry.register(record)


def _register_dataset_audit_pass(
    registry: ArtifactRegistry,
    tmp_path: Path,
    task_path: Path,
) -> None:
    audit_path = tmp_path / "dataset-audit-pass.json"
    audit_path.write_text(
        json.dumps(
            {
                "report_schema": "mint-agent.dataset-audit.v1",
                "status": "PASS",
                "task_id": load_yaml(task_path).get("task_id", "toy"),
                "sample_count": 0,
                "sample_ids": [],
            }
        ),
        encoding="utf-8",
    )
    registry.register(
        _record(
            "dataset_audit",
            audit_path,
            scope="design",
            status="PASS",
            metadata={
                "audit_input_hash": dataset_audit_input_hash(
                    load_yaml(task_path), config_path=task_path,
                )
            },
        )
    )


def test_cached_graph_reaches_evaluation_without_compute(tmp_path: Path):
    task = tmp_path / "task.yaml"
    task.write_text(
        "task_id: toy\nsystem_type: protein_ligand\ntask_type: regression\nprimary_metric: pcc\n"
    )
    registry_path = tmp_path / "artifacts.sqlite"
    registry = ArtifactRegistry(registry_path)
    _register_dataset_audit_pass(registry, tmp_path, task)
    _register_probe_prerequisites(registry, tmp_path, ("PL",))
    records = [
        _record("probe_audit", tmp_path / "audit.json"),
        _record(
            "representation_design",
            tmp_path / "representation.json",
            representation_hash="repr",
        ),
        _record(
            "feature_manifest",
            tmp_path / "PL.jsonl",
            scope="full_train",
            invariants=("PL",),
            representation_hash="repr",
        ),
        _record(
            "feature_qc",
            tmp_path / "qc.json",
            scope="full_train",
            invariants=("PL",),
            representation_hash="repr",
        ),
        _record(
            "filtration_audit",
            tmp_path / "filtration.json",
            scope="full_train",
            invariants=("PL",),
            representation_hash="repr",
        ),
        _record(
            "scout_execution_plan",
            tmp_path / "scout.json",
            invariants=("PL",),
            representation_hash="repr",
            selection_hash="selection",
        ),
        _record(
            "model_evaluation",
            tmp_path / "evaluation.json",
            scope="full_train",
            invariants=("PL",),
            representation_hash="repr",
        ),
    ]
    for record in records:
        registry.register(record)
    graph = build_agent_graph(
        AgentGraphContext(task_config=task, registry=registry_path),
        checkpointer=InMemorySaver(),
    )
    result = graph.invoke(
        {
            "run_id": "thread-1",
            "request": {
                "dataset_id": "toy",
                "invariants": ["PL"],
                "evidence_scope": "full_train",
            },
            "node_trace": [],
        },
        {"configurable": {"thread_id": "thread-1"}},
    )
    assert result["status"] == "COMPLETE"
    assert "feature_tool_node" not in result["node_trace"]
    assert (
        result["final_report"]["scientific_control"]["llm_used_for_numeric_decisions"]
        is False
    )


def test_legacy_weighted_scout_cache_is_recombined_for_hierarchical_ranking(
    tmp_path: Path,
):
    task = tmp_path / "task.yaml"
    task.write_text(
        "task_id: toy\nsystem_type: protein_ligand\ntask_type: regression\nprimary_metric: pcc\n"
    )
    registry_path = tmp_path / "artifacts.sqlite"
    registry = ArtifactRegistry(registry_path)
    _register_dataset_audit_pass(registry, tmp_path, task)
    _register_probe_prerequisites(registry, tmp_path, ("PL",))
    for record in (
        _record(
            "representation_design",
            tmp_path / "representation.json",
            representation_hash="repr",
        ),
        _record(
            "scout_oof",
            tmp_path / "scout-oof.json",
            invariants=("PL",),
            representation_hash="repr",
            selection_hash="selection",
            metadata={"gbt_parameter_hash": _TEST_GBT_HASH},
        ),
        _record(
            "scout_execution_plan",
            tmp_path / "legacy-scout.json",
            invariants=("PL",),
            representation_hash="repr",
            selection_hash="selection",
            metadata={
                "gbt_parameter_hash": _TEST_GBT_HASH,
                "ranking_policy": "legacy_weighted_composite",
                "target_metric": "PCC",
                "target_value": 0.7,
            },
        ),
    ):
        registry.register(record)
    graph = build_agent_graph(AgentGraphContext(task_config=task, registry=registry_path))

    result = graph.invoke(
        {
            "run_id": "hierarchical-cache-upgrade",
            "request": {
                "dataset_id": "toy",
                "invariants": ["PL"],
                "evidence_scope": "full_train",
            },
            "node_trace": [],
        }
    )

    assert result["status"] == "NEEDS_SCOUT_COMBINE"


def test_train_test_workflow_defaults_probe_rank_to_one_frozen_test_query(
    tmp_path: Path,
):
    dataset = tmp_path / "dataset.csv"
    rows = ["sample_id,target,split,protein_path,ligand_path"]
    for split, count in (("train", 4), ("test", 2)):
        for index in range(count):
            sample_id = f"{split}-{index}"
            protein = tmp_path / f"{sample_id}.pdb"
            ligand = tmp_path / f"{sample_id}.mol2"
            protein.write_text("ATOM\n", encoding="utf-8")
            ligand.write_text("@<TRIPOS>MOLECULE\n", encoding="utf-8")
            rows.append(f"{sample_id},{index},{split},{protein},{ligand}")
    dataset.write_text("\n".join(rows) + "\n", encoding="utf-8")
    task = tmp_path / "task.yaml"
    task.write_text(
        "task_id: toy\n"
        "system_type: protein_ligand\n"
        "task_type: regression\n"
        "primary_metric: pcc\n"
        "dataset_manifest:\n"
        f"  path: {dataset}\n",
        encoding="utf-8",
    )
    registry_path = tmp_path / "artifacts.sqlite"
    registry = ArtifactRegistry(registry_path)
    _register_dataset_audit_pass(registry, tmp_path, task)
    _register_probe_prerequisites(registry, tmp_path, ("PL",))
    for record in (
        _record(
            "representation_design",
            tmp_path / "representation.json",
            representation_hash="repr",
        ),
        _record(
            "scout_execution_plan",
            tmp_path / "scout.json",
            invariants=("PL",),
            representation_hash="repr",
            selection_hash="selection",
            metadata={
                "frozen_priority_order": [["PL"]],
                "gbt_parameter_hash": _TEST_GBT_HASH,
                "target_metric": "PCC",
                "target_value": 0.7,
            },
        ),
        _record(
            "feature_manifest",
            tmp_path / "full-PL.jsonl",
            scope="full_train",
            invariants=("PL",),
            representation_hash="repr",
        ),
    ):
        registry.register(record)
    graph = build_agent_graph(
        AgentGraphContext(
            task_config=task,
            registry=registry_path,
            execution_profile=Path("configs/execution/sapelo2.yaml"),
        )
    )

    result = graph.invoke(
        {
            "run_id": "probe-ranked-acceptance",
            "request": {
                "dataset_id": "toy",
                "invariants": ["PL"],
                "representation_spec": str(tmp_path / "representation.json"),
            },
            "node_trace": [],
        }
    )

    assert result["status"] == "NEEDS_FINAL_TEST_FEATURE_COMPUTE"
    assert result["final_test_plan"]["selected_subset"] == ["PL"]
    assert result["final_test_plan"]["selection_report"].endswith(
        "/probe_rank_selections/rank1-sha-PL.json"
    )
    assert result["feature_jobs"][0]["evidence_scope"] == "external_test"
    assert result["final_report"]["scientific_control"]["test_acceptance_used"] is False
    assert "evaluation_node" not in result["node_trace"]


def test_maximize_rank1_graph_continues_after_an_early_target_reaching_stage(
    tmp_path: Path,
):
    dataset = tmp_path / "dataset.csv"
    rows = ["sample_id,target,split,protein_path,ligand_path"]
    for split, count in (("train", 4), ("test", 2)):
        for index in range(count):
            sample_id = f"{split}-{index}"
            protein = tmp_path / f"{sample_id}.pdb"
            ligand = tmp_path / f"{sample_id}.mol2"
            protein.write_text("ATOM\n", encoding="utf-8")
            ligand.write_text("@<TRIPOS>MOLECULE\n", encoding="utf-8")
            rows.append(f"{sample_id},{index},{split},{protein},{ligand}")
    dataset.write_text("\n".join(rows) + "\n", encoding="utf-8")
    task = tmp_path / "task.yaml"
    task.write_text(
        "task_id: toy\n"
        "system_type: protein_ligand\n"
        "task_type: regression\n"
        "primary_metric: pcc\n"
        "selection_preferences:\n"
        "  selection_objective: maximize_rank1\n"
        "  test_evaluation_policy: test_acceptance\n"
        "dataset_manifest:\n"
        f"  path: {dataset}\n",
        encoding="utf-8",
    )
    registry_path = tmp_path / "artifacts.sqlite"
    registry = ArtifactRegistry(registry_path)
    universe = ("PL", "PH")
    _register_dataset_audit_pass(registry, tmp_path, task)
    _register_probe_prerequisites(registry, tmp_path, universe)
    scout_path = tmp_path / "scout.json"
    for record in (
        _record(
            "representation_design",
            tmp_path / "representation.json",
            representation_hash="repr",
        ),
        _record(
            "scout_execution_plan",
            scout_path,
            invariants=universe,
            representation_hash="repr",
            selection_hash="selection",
            metadata={
                "frozen_priority_order": [["PL", "PH"], ["PL"], ["PH"]],
                "gbt_parameter_hash": _TEST_GBT_HASH,
                "target_metric": "PCC",
                "target_value": 0.7,
            },
        ),
        _record(
            "model_evaluation",
            tmp_path / "stage-01.json",
            scope="acceptance_test",
            invariants=("PL",),
            representation_hash="repr",
            status="ACQUISITION_LIMIT_REACHED",
            metadata={
                "scout_artifact": str(scout_path),
                "max_acquisitions": 1,
                "gbt_parameter_hash": _TEST_GBT_HASH,
                "selection_objective": "maximize_rank1",
                "selected_subset": ["PL"],
                "selected_score": 0.8,
            },
        ),
        _record(
            "feature_manifest",
            tmp_path / "full-PL.jsonl",
            scope="full_train",
            invariants=("PL",),
            representation_hash="repr",
        ),
        _record(
            "feature_manifest",
            tmp_path / "test-PL.jsonl",
            scope="external_test",
            invariants=("PL",),
            representation_hash="repr",
        ),
    ):
        registry.register(record)
    graph = build_agent_graph(
        AgentGraphContext(
            task_config=task,
            registry=registry_path,
            execution_profile=Path("configs/execution/sapelo2.yaml"),
        )
    )

    result = graph.invoke(
        {
            "run_id": "maximize-rank1",
            "request": {
                "dataset_id": "toy",
                "invariants": list(universe),
                "representation_spec": str(tmp_path / "representation.json"),
            },
            "node_trace": [],
        }
    )

    assert result["status"] == "NEEDS_ACCEPTANCE_FEATURE_COMPUTE"
    assert result["acceptance_plan"]["selection_objective"] == "maximize_rank1"
    assert result["acceptance_plan"]["stage"] == ["PL", "PH"]
    assert result["acceptance_plan"]["completed_stages"][0]["selected_score"] == 0.8
    assert result["acceptance_plan"]["missing"] == ["PH"]
    assert result["acceptance_plan"]["parallel_feature_submission"] is True
    assert {
        (job["invariant"], job["evidence_scope"])
        for job in result["feature_jobs"]
    } == {("PH", "full_train"), ("PH", "external_test")}


def test_maximize_top_k_acceptance_computes_selected_candidate_union_in_parallel(
    tmp_path: Path,
):
    dataset = tmp_path / "dataset.csv"
    rows = ["sample_id,target,split,protein_path,ligand_path"]
    for split, count in (("train", 4), ("test", 2)):
        for index in range(count):
            sample_id = f"{split}-{index}"
            protein = tmp_path / f"{sample_id}.pdb"
            ligand = tmp_path / f"{sample_id}.mol2"
            protein.write_text("ATOM\n", encoding="utf-8")
            ligand.write_text("@<TRIPOS>MOLECULE\n", encoding="utf-8")
            rows.append(f"{sample_id},{index},{split},{protein},{ligand}")
    dataset.write_text("\n".join(rows) + "\n", encoding="utf-8")
    task = tmp_path / "task.yaml"
    task.write_text(
        "task_id: toy\n"
        "system_type: protein_ligand\n"
        "task_type: regression\n"
        "primary_metric: pcc\n"
        "selection_preferences:\n"
        "  selection_objective: maximize_top_k\n"
        "  max_candidate_rank: 2\n"
        "  test_evaluation_policy: test_acceptance\n"
        "dataset_manifest:\n"
        f"  path: {dataset}\n",
        encoding="utf-8",
    )
    registry_path = tmp_path / "artifacts.sqlite"
    registry = ArtifactRegistry(registry_path)
    universe = ("PL", "PH", "EIC")
    _register_dataset_audit_pass(registry, tmp_path, task)
    _register_probe_prerequisites(registry, tmp_path, universe)
    scout_path = tmp_path / "scout.json"
    for record in (
        _record(
            "representation_design",
            tmp_path / "representation.json",
            representation_hash="repr",
        ),
        _record(
            "scout_execution_plan",
            scout_path,
            invariants=universe,
            representation_hash="repr",
            selection_hash="selection",
            metadata={
                "frozen_priority_order": [
                    ["PL", "PH"],
                    ["EIC"],
                    ["PL"],
                    ["PH"],
                    ["PL", "EIC"],
                    ["PH", "EIC"],
                    ["PL", "PH", "EIC"],
                ],
                "gbt_parameter_hash": _TEST_GBT_HASH,
                "target_metric": "PCC",
                "target_value": 0.7,
            },
        ),
    ):
        registry.register(record)
    graph = build_agent_graph(
        AgentGraphContext(
            task_config=task,
            registry=registry_path,
            execution_profile=Path("configs/execution/sapelo2.yaml"),
        )
    )

    result = graph.invoke(
        {
            "run_id": "maximize-top-k",
            "request": {
                "dataset_id": "toy",
                "invariants": list(universe),
                "representation_spec": str(tmp_path / "representation.json"),
            },
            "node_trace": [],
        }
    )

    assert result["status"] == "NEEDS_ACCEPTANCE_FEATURE_COMPUTE"
    assert result["acceptance_plan"]["selection_objective"] == "maximize_top_k"
    assert result["acceptance_plan"]["candidate_rank_limit"] == 2
    assert result["acceptance_plan"]["stage"] == ["PL", "PH", "EIC"]
    assert result["acceptance_plan"]["parallel_feature_submission"] is True
    assert {
        (job["invariant"], job["evidence_scope"])
        for job in result["feature_jobs"]
    } == {
        ("PL", "full_train"),
        ("PH", "full_train"),
        ("EIC", "full_train"),
        ("PL", "external_test"),
        ("PH", "external_test"),
        ("EIC", "external_test"),
    }


def test_shadow_llm_scientific_nodes_are_audited_without_numeric_control(
    tmp_path: Path,
):
    task = tmp_path / "task.yaml"
    task.write_text(
        "task_id: toy\nsystem_type: protein_ligand\ntask_type: regression\n"
    )
    registry_path = tmp_path / "artifacts.sqlite"
    registry = ArtifactRegistry(registry_path)
    _register_dataset_audit_pass(registry, tmp_path, task)
    _register_probe_prerequisites(registry, tmp_path, ("PL",))
    for record in (
        _record("probe_audit", tmp_path / "audit.json"),
        _record(
            "representation_design",
            tmp_path / "representation.json",
            representation_hash="repr",
        ),
        _record(
            "feature_manifest",
            tmp_path / "PL.jsonl",
            scope="full_train",
            invariants=("PL",),
            representation_hash="repr",
        ),
        _record(
            "feature_qc",
            tmp_path / "qc.json",
            scope="full_train",
            invariants=("PL",),
            representation_hash="repr",
        ),
        _record(
            "filtration_audit",
            tmp_path / "filtration.json",
            scope="full_train",
            invariants=("PL",),
            representation_hash="repr",
        ),
        _record(
            "scout_execution_plan",
            tmp_path / "scout.json",
            invariants=("PL",),
            representation_hash="repr",
            selection_hash="selection",
        ),
        _record(
            "model_evaluation",
            tmp_path / "evaluation.json",
            scope="full_train",
            invariants=("PL",),
            representation_hash="repr",
        ),
    ):
        registry.register(record)

    def planner(_facts, *, model, api_key):
        assert model == "test-model"
        assert api_key == "top-secret"
        return (
            {
                "objective": "Prioritize train-only probe evidence.",
                "probe_strategy": {
                    "sampling_strategy": "group_aware_stratified",
                    "stratify_by": ["target", "molecular_size"],
                    "group_by": "provided_group_identifier",
                    "preserve_extremes": True,
                    "rationale": "Use only train/probe summaries.",
                },
                "probe_variants": [],
                "representation_variants": [],
                "representation_hypotheses": [
                    {
                        "id": "pl-first",
                        "category": "method_subset",
                        "methods": ["PL"],
                        "proposal": "Start with PL on the probe.",
                        "evidence_needed": ["probe CV"],
                    }
                ],
                "candidate_priorities": [
                    {
                        "rank": 1,
                        "invariants": ["PL"],
                        "rationale": "Probe evidence first.",
                        "expected_cost": "unknown",
                    }
                ],
                "risk_checks": ["no_test_evidence", "feature_quality"],
            },
            {"used": True, "model_requested": model, "response_id": "resp_plan"},
        )

    def critic(_facts, *, model, api_key):
        assert model == "test-model"
        assert api_key == "top-secret"
        return (
            {
                "evidence_assessment": "preliminary",
                "ranking_interpretation": "Treat probe ranking as preliminary.",
                "stability_findings": ["Check bootstrap stability."],
                "cost_findings": ["Use measured cost for ties."],
                "quality_findings": ["Keep QC gates."],
                "recommended_next_action": "continue_deterministic_pipeline",
                "rationale": "Deterministic code executes the run.",
            },
            {"used": True, "model_requested": model, "response_id": "resp_critic"},
        )

    graph = build_agent_graph(
        AgentGraphContext(
            task_config=task,
            registry=registry_path,
            llm_scientific=LLMScientificContext(
                mode="shadow",
                model="test-model",
                api_key="top-secret",
                planner=planner,
                critic=critic,
            ),
        )
    )
    result = graph.invoke(
        {
            "run_id": "shadow-llm",
            "request": {
                "dataset_id": "toy",
                "invariants": ["PL"],
                "evidence_scope": "full_train",
            },
            "node_trace": [],
        }
    )

    controls = result["final_report"]["scientific_control"]
    assert result["status"] == "COMPLETE"
    assert "experiment_planning_node" in result["node_trace"]
    assert "scientific_critic_node" in result["node_trace"]
    assert controls["llm_scientific_mode"] == "shadow"
    assert controls["llm_used_for_experiment_planning"] is True
    assert controls["llm_used_for_scientific_critique"] is True
    assert controls["llm_advice_applied_to_execution"] is False
    assert controls["llm_used_for_numeric_decisions"] is False
    assert "top-secret" not in json.dumps(result)


def test_advisory_llm_probe_variant_materializes_controlled_probe_job(tmp_path: Path,):
    task = tmp_path / "task.yaml"
    task.write_text(
        "task_id: toy\n"
        "system_type: protein_ligand\n"
        "task_type: regression\n"
        "representation_design:\n"
        "  support:\n"
        "    min_element_support_fraction: 0.005\n"
        "    min_element_support_samples: 10\n"
        "    min_pair_support_fraction: 0.005\n"
        "    min_pair_support_samples: 10\n"
        "    max_element_pair_channels: 50\n"
        "    small_molecule_self_pairs: false\n"
        "  filtration:\n"
        "    local_distance_quantile: 0.95\n"
        "    dataset_distance_quantile: 0.95\n"
        "    margin_factor: 1.10\n"
        "    default_step_angstrom: 0.1\n"
        "    allowed_steps_angstrom: [0.1, 0.2]\n"
        "    max_points: 200\n"
        "    include_zero: true\n"
        "    fixed_point_count: 50\n",
        encoding="utf-8",
    )
    scout = tmp_path / "scout.yaml"
    scout.write_text(
        "probe:\n"
        "  fraction: 0.20\n"
        "  min_samples: 300\n"
        "  max_samples: 1000\n"
        "  target_quantile_bins: 10\n"
        "  size_quantile_bins: 5\n"
        "  min_pair_support: 3\n"
        "  augmentation_fraction_limit: 0.10\n"
        "ranking:\n"
        "  weights:\n"
        "    probe_performance: 0.50\n"
        "    ranking_stability: 0.20\n"
        "    feature_quality: 0.15\n"
        "    computational_efficiency: 0.10\n"
        "    llm_prior: 0.05\n"
        "controlled_experiment:\n"
        "  max_probe_alternatives: 2\n"
        "  max_representation_alternatives: 2\n"
        "  tie_tolerance: 1.0e-12\n",
        encoding="utf-8",
    )
    profile = tmp_path / "execution.yaml"
    profile.write_text(
        "profile_id: local-test\n"
        "profile_type: slurm_hpc\n"
        "paths:\n"
        f"  project_root: {Path.cwd()}\n"
        f"  scratch_root: {tmp_path / 'scratch'}\n"
        f"  cache_root: {tmp_path / 'cache'}\n"
        f"  run_root: {tmp_path / 'runs'}\n"
        f"  log_root: {tmp_path / 'logs'}\n"
        "python:\n"
        "  executable: python\n"
        "scripts:\n"
        "  dataset_audit: scripts/slurm/run_dataset_audit.sbatch\n"
        "  representation_design: scripts/slurm/run_representation_design.sbatch\n"
        "  probe_selection: scripts/slurm/run_probe_selection.sbatch\n"
        "  probe_audit: scripts/slurm/run_probe_audit.sbatch\n"
        "slurm_defaults:\n"
        "  partition: batch\n"
        "  ntasks: 1\n"
        "  cpus_per_task: 1\n"
        "  mem: 1gb\n"
        "  time: '00:30:00'\n"
        "slurm_setup_defaults:\n"
        "  partition: batch\n"
        "  ntasks: 1\n"
        "  cpus_per_task: 1\n"
        "  mem: 1gb\n"
        "  time: '00:30:00'\n",
        encoding="utf-8",
    )
    audit_path = tmp_path / "dataset-audit.json"
    audit_path.write_text(
        json.dumps(
            {
                "report_schema": "mint-agent.dataset-audit.v1",
                "status": "PASS",
                "task_id": "toy",
                "sample_count": 4,
                "sample_ids": ["a", "b", "c", "d"],
            }
        ),
        encoding="utf-8",
    )
    registry_path = tmp_path / "artifacts.sqlite"
    registry = ArtifactRegistry(registry_path)
    registry.register(
        _record(
            "dataset_audit",
            audit_path,
            scope="design",
            status="PASS",
            metadata={
                "audit_input_hash": dataset_audit_input_hash(
                    load_yaml(task), config_path=task
                )
            },
        )
    )
    registry.register(
        _record(
            "representation_design",
            tmp_path / "representation.json",
            scope="design",
            representation_hash="repr",
            metadata={
                "design_input_hash": representation_design_input_hash(
                    load_yaml(task),
                    data_audit_content_sha256="sha",
                    split="train",
                    offset=0,
                    limit=None,
                )
            },
        )
    )
    registry.register(
        _record(
            "probe_selection",
            tmp_path / "probe.json",
            representation_hash="repr",
            selection_hash="selection",
        )
    )
    registry.register(
        _record(
            "probe_audit",
            tmp_path / "probe-audit.json",
            representation_hash="repr",
            selection_hash="selection",
        )
    )

    def planner(_facts, *, model, api_key):
        return (
            {
                "objective": "Compare one bounded probe alternative.",
                "probe_strategy": {
                    "sampling_strategy": "group_aware_stratified",
                    "stratify_by": ["target", "molecular_size"],
                },
                "probe_variants": [
                    {
                        "id": "larger-probe",
                        "fraction": 0.30,
                        "max_samples": 750,
                        "target_quantile_bins": 15,
                        "size_quantile_bins": 8,
                        "min_pair_support": 5,
                        "augmentation_fraction_limit": 0.20,
                        "rationale": "Improve representativeness.",
                    }
                ],
                "representation_variants": [],
                "representation_hypotheses": [],
                "candidate_priorities": [
                    {
                        "rank": 1,
                        "invariants": ["PL"],
                        "rationale": "Start with PL.",
                        "expected_cost": "low",
                    }
                ],
                "risk_checks": ["no_test_evidence"],
            },
            {"used": True, "model_requested": model, "response_id": "resp_plan"},
        )

    graph = build_agent_graph(
        AgentGraphContext(
            task_config=task,
            registry=registry_path,
            execution_profile=profile,
            llm_scientific=LLMScientificContext(
                mode="advisory",
                model="test-model",
                api_key="top-secret",
                planner=planner,
                critic=planner,
            ),
        )
    )

    result = graph.invoke(
        {
            "run_id": "controlled-probe",
            "request": {
                "dataset_id": "toy",
                "invariants": ["PL"],
                "scout_config": str(scout),
            },
            "node_trace": [],
        }
    )

    assert result["status"] == "NEEDS_CONTROLLED_PROBE_SELECTION"
    assert result["setup_jobs"][0]["job_kind"] == "probe_selection"
    assert (
        "larger-probe.scout.json"
        in result["setup_jobs"][0]["environment"]["SCOUT_CONFIG"]
    )
    assert result["controlled_experiment"]["execution_allowed"] is True
    assert result["node_trace"][-2:] == ["controlled_probe_node", "report_node"]


def test_advisory_representation_variant_routes_through_controlled_comparison(
    tmp_path: Path,
):
    task = tmp_path / "task.yaml"
    task.write_text(
        "task_id: toy\n"
        "system_type: protein_ligand\n"
        "task_type: regression\n"
        "primary_metric: pcc\n"
        "representation_design:\n"
        "  support:\n"
        "    min_element_support_fraction: 0.005\n"
        "    min_element_support_samples: 10\n"
        "    min_pair_support_fraction: 0.005\n"
        "    min_pair_support_samples: 10\n"
        "    max_element_pair_channels: 50\n"
        "    small_molecule_self_pairs: false\n"
        "  filtration:\n"
        "    local_distance_quantile: 0.95\n"
        "    dataset_distance_quantile: 0.95\n"
        "    margin_factor: 1.10\n"
        "    default_step_angstrom: 0.1\n"
        "    allowed_steps_angstrom: [0.1, 0.2]\n"
        "    max_points: 200\n"
        "    include_zero: true\n"
        "    fixed_point_count: 50\n",
        encoding="utf-8",
    )
    scout = tmp_path / "scout.yaml"
    scout.write_text(
        "probe:\n"
        "  fraction: 0.20\n"
        "  min_samples: 300\n"
        "  max_samples: 1000\n"
        "  target_quantile_bins: 10\n"
        "  size_quantile_bins: 5\n"
        "  min_pair_support: 3\n"
        "  augmentation_fraction_limit: 0.10\n"
        "controlled_experiment:\n"
        "  max_probe_alternatives: 2\n"
        "  max_representation_alternatives: 2\n"
        "  representation_comparison:\n"
        "    require_train_only: true\n"
        "    minimum_pair_coverage: 0.0\n"
        "    pair_coverage_tolerance: 0.0\n"
        "    filtration_issue_tolerance: 0\n"
        "    feature_warning_tolerance: 0\n"
        "    max_total_feature_dimensions: null\n"
        "    dimension_tolerance: 0\n",
        encoding="utf-8",
    )
    custom_scout_gbt_hash = ScoutConfig.from_mapping(
        load_yaml(scout)
    ).gbt.parameter_hash
    profile = tmp_path / "execution.yaml"
    profile.write_text(
        "profile_id: local-test\n"
        "profile_type: slurm_hpc\n"
        "paths:\n"
        f"  project_root: {Path.cwd()}\n"
        f"  scratch_root: {tmp_path / 'scratch'}\n"
        f"  cache_root: {tmp_path / 'cache'}\n"
        f"  run_root: {tmp_path / 'runs'}\n"
        f"  log_root: {tmp_path / 'logs'}\n"
        "python:\n"
        "  executable: python\n"
        "scripts:\n"
        "  representation_design: scripts/slurm/run_representation_design.sbatch\n"
        "slurm_defaults:\n"
        "  partition: batch\n"
        "  ntasks: 1\n"
        "  cpus_per_task: 1\n"
        "  mem: 1gb\n"
        "  time: '00:30:00'\n"
        "slurm_setup_defaults:\n"
        "  partition: batch\n"
        "  ntasks: 1\n"
        "  cpus_per_task: 1\n"
        "  mem: 1gb\n"
        "  time: '00:30:00'\n",
        encoding="utf-8",
    )
    audit_path = tmp_path / "dataset-audit.json"
    audit_path.write_text(
        json.dumps(
            {
                "report_schema": "mint-agent.dataset-audit.v1",
                "status": "PASS",
                "task_id": "toy",
                "sample_count": 4,
                "sample_ids": ["a", "b", "c", "d"],
            }
        ),
        encoding="utf-8",
    )
    registry_path = tmp_path / "artifacts.sqlite"
    registry = ArtifactRegistry(registry_path)
    records = [
        _record(
            "dataset_audit",
            audit_path,
            scope="design",
            status="PASS",
            metadata={
                "audit_input_hash": dataset_audit_input_hash(
                    load_yaml(task), config_path=task
                )
            },
        ),
        _record(
            "representation_design",
            tmp_path / "representation.json",
            scope="design",
            representation_hash="repr",
            metadata={
                "design_input_hash": representation_design_input_hash(
                    load_yaml(task),
                    data_audit_content_sha256="sha",
                    split="train",
                    offset=0,
                    limit=None,
                )
            },
        ),
        _record(
            "probe_selection",
            tmp_path / "probe.json",
            representation_hash="repr",
            selection_hash="selection",
        ),
        _record(
            "probe_audit",
            tmp_path / "probe-audit.json",
            representation_hash="repr",
            selection_hash="selection",
        ),
        _record(
            "feature_manifest",
            tmp_path / "probe-PL.jsonl",
            invariants=("PL",),
            representation_hash="repr",
            selection_hash="selection",
        ),
        _record(
            "feature_qc",
            tmp_path / "probe-qc.json",
            invariants=("PL",),
            representation_hash="repr",
            selection_hash="selection",
        ),
        _record(
            "filtration_audit",
            tmp_path / "probe-filtration.json",
            invariants=("PL",),
            representation_hash="repr",
            selection_hash="selection",
        ),
        _record(
            "scout_execution_plan",
            tmp_path / "scout-execution.json",
            invariants=("PL",),
            representation_hash="repr",
            selection_hash="selection",
            metadata={
                "combined_report": str(tmp_path / "combined-scout.json"),
                "gbt_parameter_hash": custom_scout_gbt_hash,
            },
        ),
    ]
    for record in records:
        registry.register(record)

    def planner(_facts, *, model, api_key):
        return (
            {
                "objective": "Compare one bounded adaptive representation.",
                "probe_strategy": {
                    "sampling_strategy": "group_aware_stratified",
                    "stratify_by": ["target", "molecular_size"],
                },
                "probe_variants": [],
                "representation_variants": [
                    {
                        "id": "q99-range",
                        "local_distance_quantile": 0.95,
                        "dataset_distance_quantile": 0.99,
                        "margin_factor": 1.10,
                        "min_pair_support_fraction": 0.005,
                        "min_pair_support_samples": 10,
                        "max_element_pair_channels": 40,
                        "fixed_point_count": 50,
                        "rationale": "Test a longer train-only distance range.",
                    }
                ],
                "representation_hypotheses": [],
                "candidate_priorities": [
                    {
                        "rank": 1,
                        "invariants": ["PL"],
                        "rationale": "Use PL as the first empirical check.",
                        "expected_cost": "low",
                    }
                ],
                "risk_checks": ["no_test_evidence", "shared_probe_folds"],
            },
            {"used": True, "model_requested": model, "response_id": "resp_plan"},
        )

    graph = build_agent_graph(
        AgentGraphContext(
            task_config=task,
            registry=registry_path,
            execution_profile=profile,
            llm_scientific=LLMScientificContext(
                mode="advisory",
                model="test-model",
                api_key="top-secret",
                planner=planner,
                critic=planner,
            ),
        )
    )
    result = graph.invoke(
        {
            "run_id": "controlled-representation",
            "request": {
                "dataset_id": "toy",
                "invariants": ["PL"],
                "scout_config": str(scout),
            },
            "node_trace": [],
        }
    )

    assert result["status"] == "NEEDS_CONTROLLED_REPRESENTATION_SETUP"
    assert [job["job_kind"] for job in result["setup_jobs"]] == [
        "representation_design"
    ]
    assert result["controlled_experiment"]["execution_allowed"] is True
    policy = result["controlled_experiment"]["representation_comparison_policy"]
    assert policy["decision_rule"] == "hierarchical_representation_suitability"
    assert policy["llm_prior_used"] is False
    assert result["node_trace"][-2:] == [
        "controlled_representation_node",
        "report_node",
    ]


def test_three_way_split_routes_to_validation_feature_acquisition(tmp_path: Path):
    dataset = tmp_path / "dataset.csv"
    rows = ["sample_id,target,split,protein_path,ligand_path"]
    for split, count in (("train", 4), ("validation", 2), ("test", 2)):
        for index in range(count):
            sample_id = f"{split}-{index}"
            protein = tmp_path / f"{sample_id}.pdb"
            ligand = tmp_path / f"{sample_id}.mol2"
            protein.write_text("ATOM\n", encoding="utf-8")
            ligand.write_text("@<TRIPOS>MOLECULE\n", encoding="utf-8")
            rows.append(f"{sample_id},{index},{split},{protein},{ligand}")
    dataset.write_text("\n".join(rows) + "\n", encoding="utf-8")
    task = tmp_path / "task.yaml"
    task.write_text(
        "task_id: toy\n"
        "system_type: protein_ligand\n"
        "task_type: regression\n"
        "primary_metric: pcc\n"
        "dataset_manifest:\n"
        f"  path: {dataset}\n",
        encoding="utf-8",
    )
    registry_path = tmp_path / "artifacts.sqlite"
    registry = ArtifactRegistry(registry_path)
    _register_dataset_audit_pass(registry, tmp_path, task)
    _register_probe_prerequisites(registry, tmp_path, ("PL",))
    registry.register(
        _record(
            "representation_design",
            tmp_path / "representation.json",
            representation_hash="repr",
        )
    )
    registry.register(
        _record(
            "scout_execution_plan",
            tmp_path / "scout.json",
            invariants=("PL",),
            representation_hash="repr",
            selection_hash="selection",
        )
    )
    graph = build_agent_graph(
        AgentGraphContext(
            task_config=task,
            registry=registry_path,
            execution_profile=Path("configs/execution/sapelo2.yaml"),
        )
    )

    result = graph.invoke(
        {
            "run_id": "strict-split",
            "request": {"dataset_id": "toy", "invariants": ["PL"]},
            "node_trace": [],
        }
    )

    assert result["task"]["evaluation_mode"] == "explicit_validation_and_test"
    assert result["status"] == "NEEDS_SELECTION_FEATURE_COMPUTE"
    assert result["feature_jobs"][0]["evidence_scope"] == "full_train"
    assert result["split_feature_plan"]["stage"] == ["PL"]


def test_frozen_validation_selection_only_requests_selected_test_features(
    tmp_path: Path,
):
    dataset = tmp_path / "dataset.csv"
    rows = ["sample_id,target,split,protein_path,ligand_path"]
    for split, count in (("train", 4), ("validation", 2), ("test", 2)):
        for index in range(count):
            sample_id = f"{split}-{index}"
            protein = tmp_path / f"{sample_id}.pdb"
            ligand = tmp_path / f"{sample_id}.mol2"
            protein.write_text("ATOM\n", encoding="utf-8")
            ligand.write_text("@<TRIPOS>MOLECULE\n", encoding="utf-8")
            rows.append(f"{sample_id},{index},{split},{protein},{ligand}")
    dataset.write_text("\n".join(rows) + "\n", encoding="utf-8")
    task = tmp_path / "task.yaml"
    task.write_text(
        "task_id: toy\n"
        "system_type: protein_ligand\n"
        "task_type: regression\n"
        "primary_metric: pcc\n"
        "dataset_manifest:\n"
        f"  path: {dataset}\n",
        encoding="utf-8",
    )
    registry_path = tmp_path / "artifacts.sqlite"
    registry = ArtifactRegistry(registry_path)
    _register_dataset_audit_pass(registry, tmp_path, task)
    _register_probe_prerequisites(registry, tmp_path, ("PH", "PL"))
    representation = _record(
        "representation_design",
        tmp_path / "representation.json",
        representation_hash="repr",
    )
    scout = _record(
        "scout_execution_plan",
        tmp_path / "scout.json",
        invariants=("PH", "PL"),
        representation_hash="repr",
        selection_hash="selection",
        metadata={
            "frozen_priority_order": [["PL"], ["PH"]],
            "gbt_parameter_hash": _TEST_GBT_HASH,
            "target_metric": "PCC",
            "target_value": 0.7,
            "fold_assignment_hash": "folds",
        },
    )
    records = [representation, scout]
    for scope in ("full_train", "validation"):
        records.extend(
            [
                _record(
                    "feature_manifest",
                    tmp_path / f"{scope}-PL.jsonl",
                    scope=scope,
                    invariants=("PL",),
                    representation_hash="repr",
                ),
                _record(
                    "feature_qc",
                    tmp_path / f"{scope}-qc.json",
                    scope=scope,
                    invariants=("PL",),
                    representation_hash="repr",
                ),
                _record(
                    "filtration_audit",
                    tmp_path / f"{scope}-audit.json",
                    scope=scope,
                    invariants=("PL",),
                    representation_hash="repr",
                ),
            ]
        )
    records.append(
        _record(
            "model_evaluation",
            tmp_path / "validation-evaluation.json",
            scope="validation",
            invariants=("PL",),
            representation_hash="repr",
            status="TARGET_REACHED",
            metadata={
                "selected_subset": ["PL"],
                "scout_artifact": str(tmp_path / "scout.json"),
                "gbt_parameter_hash": _TEST_GBT_HASH,
                "target_metric": "PCC",
                "target_value": 0.7,
            },
        )
    )
    for record in records:
        registry.register(record)
    graph = build_agent_graph(
        AgentGraphContext(
            task_config=task,
            registry=registry_path,
            execution_profile=Path("configs/execution/sapelo2.yaml"),
        )
    )

    result = graph.invoke(
        {
            "run_id": "strict-final",
            "request": {"dataset_id": "toy", "invariants": ["PH", "PL"]},
            "node_trace": [],
        }
    )

    assert result["status"] == "NEEDS_FINAL_TEST_FEATURE_COMPUTE"
    assert result["final_test_plan"]["selected_subset"] == ["PL"]
    assert result["feature_jobs"][0]["invariant"] == "PL"
    assert result["feature_jobs"][0]["evidence_scope"] == "external_test"


def test_maximize_rank1_validation_continues_after_an_early_target(
    tmp_path: Path,
):
    dataset = tmp_path / "dataset.csv"
    rows = ["sample_id,target,split,protein_path,ligand_path"]
    for split, count in (("train", 4), ("validation", 2), ("test", 2)):
        for index in range(count):
            sample_id = f"{split}-{index}"
            protein = tmp_path / f"{sample_id}.pdb"
            ligand = tmp_path / f"{sample_id}.mol2"
            protein.write_text("ATOM\n", encoding="utf-8")
            ligand.write_text("@<TRIPOS>MOLECULE\n", encoding="utf-8")
            rows.append(f"{sample_id},{index},{split},{protein},{ligand}")
    dataset.write_text("\n".join(rows) + "\n", encoding="utf-8")
    task = tmp_path / "task.yaml"
    task.write_text(
        "task_id: toy\n"
        "system_type: protein_ligand\n"
        "task_type: regression\n"
        "primary_metric: pcc\n"
        "selection_preferences:\n"
        "  selection_objective: maximize_rank1\n"
        "dataset_manifest:\n"
        f"  path: {dataset}\n",
        encoding="utf-8",
    )
    registry_path = tmp_path / "artifacts.sqlite"
    registry = ArtifactRegistry(registry_path)
    universe = ("PL", "PH")
    _register_dataset_audit_pass(registry, tmp_path, task)
    _register_probe_prerequisites(registry, tmp_path, universe)
    scout_path = tmp_path / "scout.json"
    for record in (
        _record(
            "representation_design",
            tmp_path / "representation.json",
            representation_hash="repr",
        ),
        _record(
            "scout_execution_plan",
            scout_path,
            invariants=universe,
            representation_hash="repr",
            selection_hash="selection",
            metadata={
                "frozen_priority_order": [["PL", "PH"], ["PL"], ["PH"]],
                "gbt_parameter_hash": _TEST_GBT_HASH,
                "target_metric": "PCC",
                "target_value": 0.7,
            },
        ),
        _record(
            "model_evaluation",
            tmp_path / "validation-stage-01.json",
            scope="validation",
            invariants=("PL",),
            representation_hash="repr",
            status="ACQUISITION_LIMIT_REACHED",
            metadata={
                "scout_artifact": str(scout_path),
                "gbt_parameter_hash": _TEST_GBT_HASH,
                "target_metric": "PCC",
                "target_value": 0.7,
                "selection_objective": "maximize_rank1",
                "selected_subset": ["PL"],
            },
        ),
        _record(
            "feature_manifest",
            tmp_path / "full-PL.jsonl",
            scope="full_train",
            invariants=("PL",),
            representation_hash="repr",
        ),
        _record(
            "feature_manifest",
            tmp_path / "validation-PL.jsonl",
            scope="validation",
            invariants=("PL",),
            representation_hash="repr",
        ),
    ):
        registry.register(record)
    graph = build_agent_graph(
        AgentGraphContext(
            task_config=task,
            registry=registry_path,
            execution_profile=Path("configs/execution/sapelo2.yaml"),
        )
    )

    result = graph.invoke(
        {
            "run_id": "maximize-rank1-validation",
            "request": {
                "dataset_id": "toy",
                "invariants": list(universe),
                "representation_spec": str(tmp_path / "representation.json"),
            },
            "node_trace": [],
        }
    )

    assert result["status"] == "NEEDS_SELECTION_FEATURE_COMPUTE"
    assert result["split_feature_plan"]["selection_objective"] == "maximize_rank1"
    assert result["split_feature_plan"]["stage"] == ["PL", "PH"]
    assert result["split_feature_plan"]["completed_stages"][0][
        "acquired_invariants"
    ] == ["PL"]
    assert result["split_feature_plan"]["missing"] == ["PH"]
    assert result["split_feature_plan"]["parallel_feature_submission"] is True
    assert {
        (job["invariant"], job["evidence_scope"])
        for job in result["feature_jobs"]
    } == {("PH", "full_train"), ("PH", "validation")}


def test_cached_scout_cannot_bypass_missing_probe_qc(tmp_path: Path):
    task = tmp_path / "task.yaml"
    task.write_text(
        "task_id: toy\nsystem_type: protein_ligand\ntask_type: regression\n"
    )
    registry_path = tmp_path / "artifacts.sqlite"
    registry = ArtifactRegistry(registry_path)
    for record in (
        _record(
            "representation_design",
            tmp_path / "representation.json",
            representation_hash="repr",
        ),
        _record(
            "probe_selection",
            tmp_path / "probe-selection.json",
            representation_hash="repr",
            selection_hash="selection",
        ),
        _record(
            "probe_audit",
            tmp_path / "probe-audit.json",
            representation_hash="repr",
            selection_hash="selection",
        ),
        _record(
            "feature_manifest",
            tmp_path / "probe-PL.jsonl",
            invariants=("PL",),
            representation_hash="repr",
            selection_hash="selection",
        ),
        _record(
            "scout_execution_plan",
            tmp_path / "scout.json",
            invariants=("PL",),
            representation_hash="repr",
            selection_hash="selection",
        ),
    ):
        registry.register(record)

    graph = build_agent_graph(
        AgentGraphContext(task_config=task, registry=registry_path)
    )
    result = graph.invoke(
        {
            "run_id": "cached-scout-missing-qc",
            "request": {
                "dataset_id": "toy",
                "invariants": ["PL"],
                "evidence_scope": "full_train",
            },
            "node_trace": [],
        }
    )

    assert result["status"] == "NEEDS_SCOUT_QC"
    assert result.get("scout") is None


def test_probe_outlier_warning_routes_through_read_only_diagnostic(tmp_path: Path):
    task = tmp_path / "task.yaml"
    task.write_text(
        "task_id: toy\nsystem_type: protein_ligand\ntask_type: regression\n"
    )
    registry_path = tmp_path / "artifacts.sqlite"
    registry = ArtifactRegistry(registry_path)
    _register_probe_prerequisites(registry, tmp_path, ("FPRC",))
    registry.register(
        _record(
            "representation_design",
            tmp_path / "representation.json",
            representation_hash="repr",
        )
    )
    qc_path = tmp_path / "probe-qc.json"
    qc_path.write_text(
        json.dumps(
            {
                "report_schema": "mint-agent.feature-qc.v1",
                "status": "WARN",
                "qc_hash": "qc-hash",
                "sample_ids": ["a", "b"],
                "feature_manifests": {"FPRC": str(tmp_path / "probe-FPRC.jsonl")},
                "issues": [
                    {
                        "invariant": "FPRC",
                        "sample_id": "b",
                        "issue": "robust_outlier",
                        "severity": "warning",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    graph = build_agent_graph(
        AgentGraphContext(
            task_config=task,
            registry=registry_path,
            execution_profile=Path("configs/execution/sapelo2.yaml"),
        )
    )
    request = {
        "dataset_id": "toy",
        "invariants": ["FPRC"],
        "evidence_scope": "full_train",
        "representation_spec": str(tmp_path / "representation.json"),
        "feature_diagnostic_top_k": 7,
    }

    missing = graph.invoke(
        {"run_id": "diagnostic-plan", "request": request, "node_trace": []}
    )

    assert missing["status"] == "NEEDS_FEATURE_DIAGNOSTIC"
    assert len(missing["feature_diagnostic_jobs"]) == 1
    plan = missing["feature_diagnostic_jobs"][0]
    assert plan["job_kind"] == "feature_outlier_diagnostic"
    assert plan["invariant"] == "FPRC"
    assert plan["environment"]["TOP_K"] == "7"
    assert plan["environment"]["FEATURE_QC_REPORT"] == str(qc_path)

    registry.register(
        _record(
            "feature_outlier_diagnostic",
            tmp_path / "diagnostic.json",
            invariants=("FPRC",),
            representation_hash="repr",
            selection_hash="selection",
            metadata={"feature_qc_hash": "qc-hash"},
        )
    )
    resumed = build_agent_graph(
        AgentGraphContext(task_config=task, registry=registry_path)
    ).invoke({"run_id": "diagnostic-reuse", "request": request, "node_trace": []})

    assert resumed["status"] == "NEEDS_SCOUT_OOF"
    assert resumed["feature_diagnostics"]["FPRC"]["path"].endswith("diagnostic.json")


def test_graph_reads_generic_manifest_and_records_full_cv_scope(tmp_path: Path):
    protein = tmp_path / "sample_protein.pdb"
    ligand = tmp_path / "sample_ligand.mol2"
    protein.write_text("protein", encoding="utf-8")
    ligand.write_text("ligand", encoding="utf-8")
    manifest = tmp_path / "samples.csv"
    manifest.write_text(
        "sample_id,target,protein_path,ligand_path\n"
        f"sample,6.5,{protein.name},{ligand.name}\n",
        encoding="utf-8",
    )
    task = tmp_path / "task.yaml"
    task.write_text(
        "task_id: generic-binding\n"
        "intent:\n"
        "  source: openai_responses_api\n"
        "  llm:\n"
        "    used: true\n"
        "    model_requested: test-model\n"
        "    response_id: resp_1\n"
        "system_type: protein_ligand\n"
        "task_type: regression\n"
        "primary_metric: pcc\n"
        "dataset_manifest:\n"
        "  path: samples.csv\n",
        encoding="utf-8",
    )
    graph = build_agent_graph(
        AgentGraphContext(task_config=task, registry=tmp_path / "artifacts.sqlite")
    )

    result = graph.invoke(
        {
            "run_id": "generic-manifest",
            "request": {
                "dataset_id": "generic-binding",
                "invariants": ["PL"],
                "evidence_scope": "full_train",
            },
            "node_trace": [],
        }
    )

    assert result["status"] == "NEEDS_DATA_AUDIT"
    assert result["task"]["evaluation_mode"] == "full_labeled_cv"
    assert result["task"]["intent_source"] == "openai_responses_api"
    assert result["task"]["llm_intent"]["model_requested"] == "test-model"
    assert result["final_report"]["scientific_control"]["llm_used_for_intent"] is True
    assert (
        result["final_report"]["scientific_control"]["llm_used_for_numeric_decisions"]
        is False
    )
    assert result["task"]["modeling_sample_ids"] == ["sample"]
    assert result["node_trace"] == [
        "dataset_preparation_node",
        "project_context_node",
        "artifact_memory_node",
        "data_audit_node",
        "report_node",
    ]


def test_graph_stops_when_manifest_is_missing_a_required_role(tmp_path: Path):
    protein = tmp_path / "sample_protein.pdb"
    protein.write_text("protein", encoding="utf-8")
    manifest = tmp_path / "samples.csv"
    manifest.write_text(
        "sample_id,target,protein_path,ligand_path\n" f"sample,6.5,{protein.name},\n",
        encoding="utf-8",
    )
    task = tmp_path / "task.yaml"
    task.write_text(
        "task_id: ambiguous-binding\n"
        "system_type: protein_ligand\n"
        "task_type: regression\n"
        "dataset_manifest:\n"
        "  path: samples.csv\n",
        encoding="utf-8",
    )
    graph = build_agent_graph(
        AgentGraphContext(task_config=task, registry=tmp_path / "artifacts.sqlite")
    )

    result = graph.invoke(
        {
            "run_id": "missing-role",
            "request": {
                "dataset_id": "ambiguous-binding",
                "invariants": ["PL"],
                "evidence_scope": "full_train",
            },
            "node_trace": [],
        }
    )

    assert result["status"] == "NEEDS_USER_INPUT"
    assert "ligand" in result["task"]["warnings"][0]
    assert result["node_trace"] == [
        "dataset_preparation_node",
        "project_context_node",
        "report_node",
    ]


def test_missing_feature_routes_to_tool_plan_and_stops(tmp_path: Path):
    task = tmp_path / "task.yaml"
    task.write_text(
        "task_id: toy\nsystem_type: protein_ligand\ntask_type: regression\n"
    )
    registry_path = tmp_path / "artifacts.sqlite"
    registry = ArtifactRegistry(registry_path)
    _register_probe_prerequisites(registry, tmp_path, ("PH",))
    registry.register(_record("probe_audit", tmp_path / "audit.json"))
    registry.register(
        _record(
            "representation_design",
            tmp_path / "representation.json",
            representation_hash="repr",
        )
    )
    registry.register(
        _record(
            "scout_execution_plan",
            tmp_path / "scout.json",
            invariants=("PH",),
            representation_hash="repr",
            selection_hash="selection",
        )
    )
    graph = build_agent_graph(
        AgentGraphContext(task_config=task, registry=registry_path)
    )
    result = graph.invoke(
        {
            "run_id": "thread-2",
            "request": {
                "dataset_id": "toy",
                "invariants": ["PH"],
                "evidence_scope": "full_train",
            },
            "node_trace": [],
        }
    )
    assert result["status"] == "NEEDS_FEATURE_COMPUTE"
    assert result["feature_plan"]["missing"] == ["PH"]
    assert "feature_tool_node" in result["node_trace"]


def test_new_dataset_graph_emits_first_setup_job(tmp_path: Path):
    task = tmp_path / "task.yaml"
    task.write_text(
        "task_id: new-data\nsystem_type: protein_ligand\ntask_type: regression\n",
        encoding="utf-8",
    )
    graph = build_agent_graph(
        AgentGraphContext(
            task_config=task,
            registry=tmp_path / "artifacts.sqlite",
            execution_profile=Path("configs/execution/sapelo2.yaml"),
        )
    )

    result = graph.invoke(
        {
            "run_id": "onboard-one",
            "request": {
                "dataset_id": "new-data",
                "invariants": ["PL"],
                "evidence_scope": "full_train",
            },
            "node_trace": [],
        }
    )

    assert result["status"] == "NEEDS_DATA_AUDIT"
    assert len(result["setup_jobs"]) == 1
    assert result["setup_jobs"][0]["job_kind"] == "dataset_audit"
    assert result["node_trace"] == [
        "dataset_preparation_node",
        "project_context_node",
        "artifact_memory_node",
        "data_audit_node",
        "report_node",
    ]


def test_graph_blocks_unconfirmed_small_labeled_modeling_pool(tmp_path: Path):
    dataset = tmp_path / "dataset.csv"
    rows = ["sample_id,target,protein_path,ligand_path"]
    for index in range(5):
        protein = tmp_path / f"p{index}.pdb"
        ligand = tmp_path / f"l{index}.mol2"
        protein.write_text("ATOM\n", encoding="utf-8")
        ligand.write_text("@<TRIPOS>MOLECULE\n", encoding="utf-8")
        rows.append(f"s{index},{index},{protein},{ligand}")
    dataset.write_text("\n".join(rows) + "\n", encoding="utf-8")
    task = tmp_path / "task.yaml"
    task.write_text(
        "task_id: toy\n"
        "system_type: protein_ligand\n"
        "task_type: regression\n"
        "primary_metric: pcc\n"
        "labels:\n"
        "  min_labeled_samples: 300\n"
        "  allow_small_data_override: false\n"
        "dataset_manifest:\n"
        f"  path: {dataset}\n"
        "  columns:\n"
        "    roles:\n"
        "      protein: protein_path\n"
        "      ligand: ligand_path\n",
        encoding="utf-8",
    )
    graph = build_agent_graph(
        AgentGraphContext(task_config=task, registry=tmp_path / "artifacts.sqlite",)
    )

    result = graph.invoke(
        {
            "run_id": "small-data",
            "request": {
                "dataset_id": "toy",
                "invariants": ["PL"],
                "evidence_scope": "full_train",
            },
            "node_trace": [],
        }
    )

    assert result["status"] == "NEEDS_SMALL_DATA_CONFIRMATION"
    assert result["task"]["label_sample_policy"]["min_labeled_samples"] == 300
    assert "artifact_memory_node" not in result["node_trace"]


def test_changed_explicit_audit_policy_invalidates_legacy_cached_audit(tmp_path: Path):
    task = tmp_path / "task.yaml"
    task.write_text(
        "task_id: toy\n"
        "system_type: protein_ligand\n"
        "task_type: regression\n"
        "data_audit:\n"
        "  tolerated_out_of_schema:\n"
        "    protein: [H, Zn]\n",
        encoding="utf-8",
    )
    registry_path = tmp_path / "artifacts.sqlite"
    ArtifactRegistry(registry_path).register(
        _record(
            "dataset_audit",
            tmp_path / "legacy-audit.json",
            scope="design",
            status="PASS",
        )
    )
    graph = build_agent_graph(
        AgentGraphContext(
            task_config=task,
            registry=registry_path,
            execution_profile=Path("configs/execution/sapelo2.yaml"),
        )
    )

    result = graph.invoke(
        {
            "run_id": "changed-audit-policy",
            "request": {
                "dataset_id": "toy",
                "invariants": ["PL"],
                "evidence_scope": "full_train",
            },
            "node_trace": [],
        }
    )

    assert result["status"] == "NEEDS_DATA_AUDIT"
    assert result["setup_jobs"][0]["job_kind"] == "dataset_audit"


def test_changed_representation_policy_invalidates_cached_design(tmp_path: Path):
    dataset = tmp_path / "dataset.csv"
    rows = ["sample_id,target,protein_path,ligand_path"]
    for index in range(2):
        protein = tmp_path / f"p{index}.pdb"
        ligand = tmp_path / f"l{index}.mol2"
        protein.write_text("ATOM\n", encoding="utf-8")
        ligand.write_text("@<TRIPOS>MOLECULE\n", encoding="utf-8")
        rows.append(f"s{index},{index},{protein},{ligand}")
    dataset.write_text("\n".join(rows) + "\n", encoding="utf-8")
    task = tmp_path / "task.yaml"
    task.write_text(
        "task_id: toy\n"
        "system_type: protein_ligand\n"
        "task_type: regression\n"
        "primary_metric: pcc\n"
        "representation_mode: dataset_adaptive\n"
        "representation_design:\n"
        "  filtration:\n"
        "    dataset_distance_quantile: 0.99\n"
        "dataset_manifest:\n"
        f"  path: {dataset}\n"
        "  columns:\n"
        "    roles:\n"
        "      protein: protein_path\n"
        "      ligand: ligand_path\n",
        encoding="utf-8",
    )
    registry_path = tmp_path / "artifacts.sqlite"
    registry = ArtifactRegistry(registry_path)
    registry.register(
        _record(
            "dataset_audit",
            tmp_path / "audit.json",
            scope="design",
            status="PASS",
            metadata={
                "audit_input_hash": dataset_audit_input_hash(
                    load_yaml(task), config_path=task
                )
            },
        )
    )
    registry.register(
        _record(
            "representation_design",
            tmp_path / "old-representation.json",
            scope="design",
            representation_hash="old-repr",
            metadata={"design_input_hash": "old-q95-input"},
        )
    )
    graph = build_agent_graph(
        AgentGraphContext(task_config=task, registry=registry_path)
    )

    result = graph.invoke(
        {
            "run_id": "changed-representation",
            "request": {
                "dataset_id": "toy",
                "invariants": ["PL"],
                "evidence_scope": "full_train",
            },
            "node_trace": [],
        }
    )

    assert result["status"] == "NEEDS_REPRESENTATION"
    assert result.get("representation_hash") is None


def test_missing_manifest_routes_to_dataset_preparation_before_audit(tmp_path: Path):
    task = tmp_path / "task.yaml"
    task.write_text(
        "task_id: new-data\n"
        "system_type: protein_ligand\n"
        "task_type: regression\n"
        "dataset_preparation:\n"
        "  provider: paired_structure_manifest\n"
        "  module_load: []\n"
        f"  output_manifest: {tmp_path / 'manifest.csv'}\n"
        "dataset_manifest:\n"
        f"  path: {tmp_path / 'manifest.csv'}\n",
        encoding="utf-8",
    )
    graph = build_agent_graph(
        AgentGraphContext(
            task_config=task,
            registry=tmp_path / "artifacts.sqlite",
            execution_profile=Path("configs/execution/sapelo2.yaml"),
        )
    )

    result = graph.invoke(
        {
            "run_id": "prepare-first",
            "request": {
                "dataset_id": "new-data",
                "invariants": ["PL"],
                "evidence_scope": "full_train",
            },
            "node_trace": [],
        }
    )

    assert result["status"] == "NEEDS_DATASET_PREPARATION"
    assert [job["job_kind"] for job in result["preparation_jobs"]] == [
        "dataset_preparation"
    ]
    assert result["node_trace"] == [
        "dataset_preparation_node",
        "report_node",
    ]


@pytest.mark.parametrize(
    ("completed_steps", "expected_status", "expected_job_kind"),
    [
        (("audit",), "NEEDS_REPRESENTATION", "representation_design"),
        (("audit", "representation"), "NEEDS_PROBE_SELECTION", "probe_selection"),
        (("audit", "representation", "probe"), "NEEDS_PROBE_AUDIT", "probe_audit",),
    ],
)
def test_graph_plans_exactly_next_onboarding_step(
    tmp_path: Path,
    completed_steps: tuple[str, ...],
    expected_status: str,
    expected_job_kind: str,
):
    task = tmp_path / "task.yaml"
    task.write_text(
        "task_id: toy\nsystem_type: protein_ligand\ntask_type: regression\n",
        encoding="utf-8",
    )
    registry_path = tmp_path / "artifacts.sqlite"
    registry = ArtifactRegistry(registry_path)
    registry.register(
        _record(
            "dataset_audit",
            tmp_path / "dataset-audit.json",
            scope="design",
            status="PASS",
        )
    )
    if "representation" in completed_steps:
        registry.register(
            _record(
                "representation_design",
                tmp_path / "representation.json",
                scope="design",
                representation_hash="repr",
            )
        )
    if "probe" in completed_steps:
        registry.register(
            _record(
                "probe_selection",
                tmp_path / "probe.json",
                representation_hash="repr",
                selection_hash="selection",
            )
        )
    graph = build_agent_graph(
        AgentGraphContext(
            task_config=task,
            registry=registry_path,
            execution_profile=Path("configs/execution/sapelo2.yaml"),
        )
    )

    result = graph.invoke(
        {
            "run_id": "onboard-next",
            "request": {
                "dataset_id": "toy",
                "invariants": ["PL"],
                "evidence_scope": "full_train",
            },
            "node_trace": [],
        }
    )

    assert result["status"] == expected_status
    assert [job["job_kind"] for job in result["setup_jobs"]] == [expected_job_kind]


def test_scout_input_pipeline_plans_probe_features_then_qc_then_filtration(
    tmp_path: Path,
):
    task = tmp_path / "task.yaml"
    task.write_text(
        "task_id: toy\nsystem_type: protein_ligand\ntask_type: regression\n",
        encoding="utf-8",
    )
    registry_path = tmp_path / "artifacts.sqlite"
    registry = ArtifactRegistry(registry_path)
    selection_hash = "frozen-probe"
    for record in (
        _record(
            "dataset_audit",
            tmp_path / "dataset-audit.json",
            scope="design",
            status="PASS",
        ),
        _record(
            "representation_design",
            tmp_path / "representation.json",
            scope="design",
            representation_hash="repr",
        ),
        _record(
            "probe_selection",
            tmp_path / "probe.json",
            representation_hash="repr",
            selection_hash=selection_hash,
        ),
        _record(
            "probe_audit",
            tmp_path / "probe-audit.json",
            representation_hash="repr",
            selection_hash=selection_hash,
            status="PASS",
        ),
    ):
        registry.register(record)

    def invoke(run_id: str):
        return build_agent_graph(
            AgentGraphContext(
                task_config=task,
                registry=registry_path,
                execution_profile=Path("configs/execution/sapelo2.yaml"),
            )
        ).invoke(
            {
                "run_id": run_id,
                "request": {
                    "dataset_id": "toy",
                    "invariants": ["PL", "PH"],
                    "evidence_scope": "full_train",
                },
                "node_trace": [],
            }
        )

    missing_features = invoke("probe-features")
    assert missing_features["status"] == "NEEDS_SCOUT_FEATURES"
    assert [job["invariant"] for job in missing_features["feature_jobs"]] == [
        "PL",
        "PH",
    ]
    assert all(
        job["environment"]["SELECTION_HASH"] == selection_hash
        for job in missing_features["feature_jobs"]
    )

    for invariant in ("PL", "PH"):
        registry.register(
            _record(
                "feature_manifest",
                tmp_path / f"probe-{invariant}.jsonl",
                invariants=(invariant,),
                representation_hash="repr",
                selection_hash=selection_hash,
            )
        )
    missing_qc = invoke("probe-qc")
    assert missing_qc["status"] == "NEEDS_SCOUT_QC"
    assert len(missing_qc["qc_jobs"]) == 1
    assert missing_qc["qc_jobs"][0]["evidence_scope"] == "probe"
    assert missing_qc["qc_jobs"][0]["environment"]["PROBE_SELECTION"] == str(
        tmp_path / "probe.json"
    )

    registry.register(
        _record(
            "feature_qc",
            tmp_path / "probe-qc.json",
            invariants=("PL", "PH"),
            representation_hash="repr",
            selection_hash=selection_hash,
            status="PASS",
        )
    )
    missing_filtration = invoke("probe-filtration")
    assert missing_filtration["status"] == "NEEDS_SCOUT_FILTRATION_AUDIT"
    assert len(missing_filtration["filtration_audit_jobs"]) == 1
    assert missing_filtration["filtration_audit_jobs"][0]["evidence_scope"] == "probe"
    assert missing_filtration["filtration_audit_jobs"][0]["environment"][
        "PROBE_SELECTION"
    ] == str(tmp_path / "probe.json")


def test_missing_feature_includes_slurm_plan_when_profile_is_available(tmp_path: Path):
    task = tmp_path / "task.yaml"
    task.write_text(
        "task_id: toy\nsystem_type: protein_ligand\ntask_type: regression\n"
    )
    profile = tmp_path / "profile.yaml"
    profile.write_text(
        "\n".join(
            [
                "profile_id: test_slurm",
                "profile_type: slurm_hpc",
                "host:",
                "  login: cluster.example.edu",
                "  user: user1",
                "paths:",
                "  project_root: /home/user1/mint-agent",
                "  scratch_root: /scratch/user1/mint-agent",
                "  cache_root: /scratch/user1/mathagent/cache",
                "  run_root: /scratch/user1/mathagent/runs",
                "  log_root: /scratch/user1/mathagent/logs",
                "python:",
                "  executable: python",
                "  module_load: []",
                "scripts:",
                "  feature_batch: scripts/slurm/run_feature_batch.sbatch",
                "slurm_feature_defaults:",
                "  partition: batch",
                "  ntasks: 1",
                "  cpus_per_task: 2",
                "  mem: 12gb",
                "  time: '03:00:00'",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    registry_path = tmp_path / "artifacts.sqlite"
    registry = ArtifactRegistry(registry_path)
    _register_probe_prerequisites(registry, tmp_path, ("PH",))
    registry.register(_record("probe_audit", tmp_path / "audit.json"))
    registry.register(
        _record(
            "representation_design",
            tmp_path / "representation.json",
            representation_hash="repr",
        )
    )
    registry.register(
        _record(
            "scout_execution_plan",
            tmp_path / "scout.json",
            invariants=("PH",),
            representation_hash="repr",
            selection_hash="selection",
        )
    )
    graph = build_agent_graph(
        AgentGraphContext(
            task_config=task, registry=registry_path, execution_profile=profile
        )
    )

    result = graph.invoke(
        {
            "run_id": "thread-3",
            "request": {
                "dataset_id": "toy",
                "invariants": ["PH"],
                "evidence_scope": "full_train",
                "representation_spec": "/scratch/user1/mathagent/runs/repr.json",
            },
            "node_trace": [],
        }
    )

    assert result["status"] == "NEEDS_FEATURE_COMPUTE"
    assert result["feature_jobs"][0]["profile_id"] == "test_slurm"
    assert result["feature_jobs"][0]["environment"]["INVARIANT"] == "PH"
    assert (
        result["feature_jobs"][0]["environment"]["REPRESENTATION_SPEC"]
        == "/scratch/user1/mathagent/runs/repr.json"
    )
    assert result["final_report"]["feature_jobs"] == result["feature_jobs"]


def test_reconciled_feature_is_reused_and_graph_advances_to_qc(tmp_path: Path):
    task = tmp_path / "task.yaml"
    task.write_text(
        "task_id: toy\nsystem_type: protein_ligand\ntask_type: regression\n"
    )
    profile = tmp_path / "profile.yaml"
    profile.write_text(
        "\n".join(
            [
                "profile_id: test_slurm",
                "profile_type: slurm_hpc",
                "paths:",
                f"  project_root: {tmp_path}",
                f"  scratch_root: {tmp_path}",
                f"  cache_root: {tmp_path / 'cache'}",
                f"  run_root: {tmp_path / 'runs'}",
                f"  log_root: {tmp_path / 'logs'}",
                "python:",
                "  executable: python",
                "scripts:",
                "  feature_batch: scripts/slurm/run_feature_batch.sbatch",
                "slurm_feature_defaults:",
                "  partition: batch",
                "  ntasks: 1",
                "  cpus_per_task: 1",
                "  mem: 1gb",
                "  time: '00:10:00'",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    registry_path = tmp_path / "artifacts.sqlite"
    registry = ArtifactRegistry(registry_path)
    _register_probe_prerequisites(registry, tmp_path, ("PH",))
    registry.register(_record("probe_audit", tmp_path / "audit.json"))
    registry.register(
        _record(
            "representation_design",
            tmp_path / "representation.json",
            representation_hash="repr",
        )
    )
    registry.register(
        _record(
            "scout_execution_plan",
            tmp_path / "scout.json",
            invariants=("PH",),
            representation_hash="repr",
            selection_hash="selection",
        )
    )
    request = {
        "dataset_id": "toy",
        "invariants": ["PH"],
        "evidence_scope": "full_train",
    }
    graph = build_agent_graph(
        AgentGraphContext(
            task_config=task, registry=registry_path, execution_profile=profile
        )
    )
    first = graph.invoke(
        {"run_id": "reconcile-test", "request": request, "node_trace": []}
    )
    plan = first["feature_jobs"][0]
    output_a = tmp_path / "feature-a.npy"
    output_b = tmp_path / "feature-b.npy"
    output_a.write_bytes(b"feature-a")
    output_b.write_bytes(b"feature-b")
    manifest = Path(plan["manifest_path"])
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "sample_id": "a",
                        "split": "train",
                        "invariant": "PH",
                        "status": "computed",
                        "output_path": str(output_a),
                    }
                ),
                json.dumps(
                    {
                        "sample_id": "b",
                        "split": "train",
                        "invariant": "PH",
                        "status": "computed",
                        "output_path": str(output_b),
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    ledger = create_job_ledger(
        first["final_report"], source_plan=tmp_path / "graph.json"
    )
    submit_ready_jobs(
        ledger,
        execute=True,
        runner=lambda command, cwd: CommandResult(0, "Submitted batch job 42\n", ""),
    )
    refresh_job_ledger(
        ledger, runner=lambda command, cwd: CommandResult(0, "42|COMPLETED|0:0\n", ""),
    )
    reconcile_completed_jobs(ledger, registry=registry_path)

    second = graph.invoke(
        {"run_id": "reconcile-test-2", "request": request, "node_trace": []}
    )
    assert ledger["jobs"][0]["state"] == "REGISTERED"
    assert second["feature_plan"]["cache_hit_count"] == 1
    assert second["status"] == "NEEDS_FEATURE_QC"
    assert second["qc_jobs"][0]["job_kind"] == "feature_qc"
    assert second["qc_jobs"][0]["input_manifests"] == {"PH": str(manifest)}
    assert "feature_tool_node" not in second["node_trace"]
    assert "feature_qc_node" in second["node_trace"]


def test_missing_scope_matched_filtration_audit_emits_slurm_plan(tmp_path: Path):
    task = tmp_path / "task.yaml"
    task.write_text(
        "task_id: toy\nsystem_type: protein_ligand\ntask_type: regression\n"
    )
    profile = tmp_path / "profile.yaml"
    profile.write_text(
        "\n".join(
            [
                "profile_id: test_slurm",
                "profile_type: slurm_hpc",
                "paths:",
                f"  project_root: {tmp_path}",
                f"  scratch_root: {tmp_path}",
                f"  cache_root: {tmp_path / 'cache'}",
                f"  run_root: {tmp_path / 'runs'}",
                f"  log_root: {tmp_path / 'logs'}",
                "python:",
                "  executable: python",
                "scripts:",
                "  filtration_audit: scripts/slurm/run_filtration_audit.sbatch",
                "slurm_defaults:",
                "  partition: batch",
                "  ntasks: 1",
                "  cpus_per_task: 1",
                "  mem: 1gb",
                "  time: '00:10:00'",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    registry_path = tmp_path / "artifacts.sqlite"
    registry = ArtifactRegistry(registry_path)
    _register_probe_prerequisites(registry, tmp_path, ("PL",))
    for record in (
        _record("probe_audit", tmp_path / "probe-audit.json"),
        _record(
            "representation_design",
            tmp_path / "representation.json",
            representation_hash="repr",
        ),
        _record(
            "feature_manifest",
            tmp_path / "PL.jsonl",
            scope="full_train",
            invariants=("PL",),
            representation_hash="repr",
        ),
        _record(
            "feature_qc",
            tmp_path / "qc.json",
            scope="full_train",
            invariants=("PL",),
            representation_hash="repr",
        ),
        _record(
            "filtration_audit",
            tmp_path / "wrong-scope-filtration.json",
            scope="probe",
            invariants=("PL",),
            representation_hash="repr",
        ),
        _record(
            "scout_execution_plan",
            tmp_path / "scout.json",
            invariants=("PL",),
            representation_hash="repr",
            selection_hash="selection",
        ),
    ):
        registry.register(record)
    graph = build_agent_graph(
        AgentGraphContext(
            task_config=task, registry=registry_path, execution_profile=profile,
        )
    )

    result = graph.invoke(
        {
            "run_id": "filtration-plan",
            "request": {
                "dataset_id": "toy",
                "invariants": ["PL"],
                "evidence_scope": "full_train",
            },
            "node_trace": [],
        }
    )

    assert result["status"] == "NEEDS_FILTRATION_AUDIT"
    plan = result["filtration_audit_jobs"][0]
    assert plan["job_kind"] == "filtration_audit"
    assert plan["evidence_scope"] == "full_train"
    assert plan["input_manifests"] == {"PL": str(tmp_path / "PL.jsonl")}


def test_graph_rejects_evaluation_with_different_sample_order(tmp_path: Path):
    task = tmp_path / "task.yaml"
    task.write_text(
        "task_id: toy\nsystem_type: protein_ligand\ntask_type: regression\n"
    )
    registry_path = tmp_path / "artifacts.sqlite"
    registry = ArtifactRegistry(registry_path)
    _register_probe_prerequisites(registry, tmp_path, ("PL",))
    records = [
        _record("probe_audit", tmp_path / "audit.json"),
        _record(
            "representation_design",
            tmp_path / "representation.json",
            representation_hash="repr",
        ),
        _record(
            "feature_manifest",
            tmp_path / "PL.jsonl",
            scope="full_train",
            invariants=("PL",),
            representation_hash="repr",
        ),
        _record(
            "feature_qc",
            tmp_path / "qc.json",
            scope="full_train",
            invariants=("PL",),
            representation_hash="repr",
        ),
        _record(
            "filtration_audit",
            tmp_path / "filtration.json",
            scope="full_train",
            invariants=("PL",),
            representation_hash="repr",
        ),
        _record(
            "scout_execution_plan",
            tmp_path / "scout.json",
            invariants=("PL",),
            representation_hash="repr",
            selection_hash="selection",
        ),
        _record(
            "model_evaluation",
            tmp_path / "evaluation.json",
            scope="full_train",
            invariants=("PL",),
            representation_hash="repr",
            sample_order_hash="different-order",
        ),
    ]
    for record in records:
        registry.register(record)
    graph = build_agent_graph(
        AgentGraphContext(task_config=task, registry=registry_path)
    )

    result = graph.invoke(
        {
            "run_id": "thread-mismatch",
            "request": {
                "dataset_id": "toy",
                "invariants": ["PL"],
                "evidence_scope": "full_train",
            },
            "node_trace": [],
        }
    )

    assert result["status"] == "INCOMPATIBLE_MODEL_EVALUATION"
    assert result["final_report"]["evaluation"] is None


def test_graph_rejects_evaluation_from_a_different_scout_target(tmp_path: Path):
    task = tmp_path / "task.yaml"
    task.write_text(
        "task_id: toy\nsystem_type: protein_ligand\ntask_type: regression\n"
    )
    registry_path = tmp_path / "artifacts.sqlite"
    registry = ArtifactRegistry(registry_path)
    _register_probe_prerequisites(registry, tmp_path, ("PL",))
    for record in (
        _record("probe_audit", tmp_path / "audit.json"),
        _record(
            "representation_design",
            tmp_path / "representation.json",
            representation_hash="repr",
        ),
        _record(
            "feature_manifest",
            tmp_path / "PL.jsonl",
            scope="full_train",
            invariants=("PL",),
            representation_hash="repr",
        ),
        _record(
            "feature_qc",
            tmp_path / "qc.json",
            scope="full_train",
            invariants=("PL",),
            representation_hash="repr",
        ),
        _record(
            "filtration_audit",
            tmp_path / "filtration.json",
            scope="full_train",
            invariants=("PL",),
            representation_hash="repr",
        ),
        _record(
            "scout_execution_plan",
            tmp_path / "scout.json",
            invariants=("PL",),
            representation_hash="repr",
            selection_hash="selection",
            metadata={"target_value": 0.8},
        ),
        _record(
            "model_evaluation",
            tmp_path / "evaluation.json",
            scope="full_train",
            invariants=("PL",),
            representation_hash="repr",
            status="TARGET_REACHED",
            metadata={"target_value": 0.7},
        ),
    ):
        registry.register(record)
    graph = build_agent_graph(
        AgentGraphContext(task_config=task, registry=registry_path)
    )

    result = graph.invoke(
        {
            "run_id": "target-mismatch",
            "request": {
                "dataset_id": "toy",
                "invariants": ["PL"],
                "evidence_scope": "full_train",
            },
            "node_trace": [],
        }
    )

    assert result["status"] == "INCOMPATIBLE_MODEL_EVALUATION"
    assert result["final_report"]["evaluation"] is None


def test_graph_accepts_legacy_evaluation_with_exact_frozen_scout_source(tmp_path: Path):
    task = tmp_path / "task.yaml"
    task.write_text(
        "task_id: toy\nsystem_type: protein_ligand\ntask_type: regression\n"
    )
    registry_path = tmp_path / "artifacts.sqlite"
    registry = ArtifactRegistry(registry_path)
    _register_probe_prerequisites(registry, tmp_path, ("PL",))
    scout_path = tmp_path / "scout.json"
    scout = _record(
        "scout_execution_plan",
        scout_path,
        invariants=("PL",),
        representation_hash="repr",
        selection_hash="selection",
    )
    evaluation = _record(
        "model_evaluation",
        tmp_path / "evaluation.json",
        scope="full_train",
        invariants=("PL",),
        representation_hash="repr",
        status="TARGET_REACHED",
        metadata={
            "fold_assignment_hash": None,
            "scout_artifact": str(scout_path),
            "fold_source": "scout_artifact",
        },
    )
    for record in (
        _record("probe_audit", tmp_path / "audit.json"),
        _record(
            "representation_design",
            tmp_path / "representation.json",
            representation_hash="repr",
        ),
        scout,
        evaluation,
    ):
        registry.register(record)
    graph = build_agent_graph(
        AgentGraphContext(task_config=task, registry=registry_path)
    )

    result = graph.invoke(
        {
            "run_id": "legacy-source",
            "request": {
                "dataset_id": "toy",
                "invariants": ["PL"],
                "evidence_scope": "full_train",
            },
            "node_trace": [],
        }
    )

    assert result["status"] == "COMPLETE"
    assert result["feature_plan"]["target_reached_after"] == ["PL"]


def test_graph_stops_on_registered_failed_feature_qc(tmp_path: Path):
    task = tmp_path / "task.yaml"
    task.write_text(
        "task_id: toy\nsystem_type: protein_ligand\ntask_type: regression\n"
    )
    registry_path = tmp_path / "artifacts.sqlite"
    registry = ArtifactRegistry(registry_path)
    _register_probe_prerequisites(registry, tmp_path, ("PL",))
    for record in (
        _record("probe_audit", tmp_path / "audit.json"),
        _record(
            "representation_design",
            tmp_path / "representation.json",
            representation_hash="repr",
        ),
        _record(
            "feature_manifest",
            tmp_path / "PL.jsonl",
            scope="full_train",
            invariants=("PL",),
            representation_hash="repr",
            selection_hash="selection",
        ),
        _record(
            "feature_qc",
            tmp_path / "qc.json",
            scope="full_train",
            invariants=("PL",),
            representation_hash="repr",
            status="FAIL",
        ),
        _record(
            "scout_execution_plan",
            tmp_path / "scout.json",
            invariants=("PL",),
            representation_hash="repr",
            selection_hash="selection",
        ),
    ):
        registry.register(record)
    graph = build_agent_graph(
        AgentGraphContext(task_config=task, registry=registry_path)
    )

    result = graph.invoke(
        {
            "run_id": "failed-qc",
            "request": {
                "dataset_id": "toy",
                "invariants": ["PL"],
                "evidence_scope": "full_train",
            },
            "node_trace": [],
        }
    )

    assert result["status"] == "FEATURE_QC_FAILED"
    assert "filtration_audit_node" not in result["node_trace"]


def test_progressive_graph_stops_after_first_gbt_stage_reaches_target(tmp_path: Path):
    task = tmp_path / "task.yaml"
    task.write_text(
        "task_id: toy\nsystem_type: protein_ligand\ntask_type: regression\n"
    )
    registry_path = tmp_path / "artifacts.sqlite"
    registry = ArtifactRegistry(registry_path)
    universe = ("PH", "PL", "CA", "FPRC", "EIC")
    priority = [["PL"], ["PH", "PL"], ["FPRC", "PL"], ["EIC"], ["CA"]]
    _register_probe_prerequisites(registry, tmp_path, universe)
    for record in (
        _record("probe_audit", tmp_path / "audit.json"),
        _record(
            "representation_design",
            tmp_path / "representation.json",
            representation_hash="repr",
        ),
        _record(
            "scout_execution_plan",
            tmp_path / "scout.json",
            invariants=universe,
            representation_hash="repr",
            selection_hash="selection",
            metadata={"frozen_priority_order": priority},
        ),
        _record(
            "model_evaluation",
            tmp_path / "PL-evaluation.json",
            scope="full_train",
            invariants=("PL",),
            representation_hash="repr",
            status="TARGET_REACHED",
        ),
    ):
        registry.register(record)
    graph = build_agent_graph(
        AgentGraphContext(task_config=task, registry=registry_path)
    )

    result = graph.invoke(
        {
            "run_id": "progressive-stop",
            "request": {
                "dataset_id": "toy",
                "invariants": list(universe),
                "evidence_scope": "full_train",
            },
            "node_trace": [],
        }
    )

    assert result["status"] == "COMPLETE"
    assert result["feature_plan"]["acquisition_order"] == [
        "PL",
        "PH",
        "FPRC",
        "EIC",
        "CA",
    ]
    assert result["feature_plan"]["target_reached_after"] == ["PL"]
    assert result["evaluation"]["path"].endswith("PL-evaluation.json")
    assert "feature_tool_node" not in result["node_trace"]
    assert "feature_qc_node" not in result["node_trace"]


def test_progressive_graph_plans_only_next_missing_invariant(tmp_path: Path):
    task = tmp_path / "task.yaml"
    task.write_text(
        "task_id: toy\nsystem_type: protein_ligand\ntask_type: regression\n"
    )
    registry_path = tmp_path / "artifacts.sqlite"
    registry = ArtifactRegistry(registry_path)
    universe = ("PH", "PL", "CA", "FPRC", "EIC")
    priority = [["PL"], ["PH", "PL"], ["FPRC", "PL"], ["EIC"], ["CA"]]
    _register_probe_prerequisites(registry, tmp_path, universe)
    for record in (
        _record("probe_audit", tmp_path / "audit.json"),
        _record(
            "representation_design",
            tmp_path / "representation.json",
            representation_hash="repr",
        ),
        _record(
            "scout_execution_plan",
            tmp_path / "scout.json",
            invariants=universe,
            representation_hash="repr",
            selection_hash="selection",
            metadata={"frozen_priority_order": priority},
        ),
        _record(
            "model_evaluation",
            tmp_path / "PL-evaluation.json",
            scope="full_train",
            invariants=("PL",),
            representation_hash="repr",
            status="ACQUISITION_LIMIT_REACHED",
        ),
        _record(
            "feature_manifest",
            tmp_path / "PL.jsonl",
            scope="full_train",
            invariants=("PL",),
            representation_hash="repr",
        ),
    ):
        registry.register(record)
    graph = build_agent_graph(
        AgentGraphContext(task_config=task, registry=registry_path)
    )

    result = graph.invoke(
        {
            "run_id": "progressive-next",
            "request": {
                "dataset_id": "toy",
                "invariants": list(universe),
                "evidence_scope": "full_train",
            },
            "node_trace": [],
        }
    )

    assert result["status"] == "NEEDS_FEATURE_COMPUTE"
    assert result["feature_plan"]["completed_stages"][0]["status"] == (
        "ACQUISITION_LIMIT_REACHED"
    )
    assert result["feature_plan"]["requested"] == ["PL", "PH"]
    assert result["feature_plan"]["available"] == {"PL": str(tmp_path / "PL.jsonl")}
    assert result["feature_plan"]["missing"] == ["PH"]
    assert result["feature_plan"]["remaining_missing"] == ["PH"]


def test_incompatible_scout_oof_emits_plan_only_for_stale_invariant(tmp_path: Path):
    task = tmp_path / "task.yaml"
    task.write_text(
        "task_id: toy\nsystem_type: protein_ligand\ntask_type: regression\n"
    )
    registry_path = tmp_path / "artifacts.sqlite"
    registry = ArtifactRegistry(registry_path)
    selection_hash = "frozen-probe"
    expected_gbt_hash = ScoutConfig.from_mapping(
        load_yaml("configs/scout/v1.yaml")
    ).gbt.parameter_hash
    for record in (
        _record("probe_audit", tmp_path / "audit.json"),
        _record(
            "representation_design",
            tmp_path / "representation.json",
            representation_hash="repr",
        ),
        _record(
            "feature_manifest",
            tmp_path / "full-PH.jsonl",
            scope="full_train",
            invariants=("PH",),
            representation_hash="repr",
        ),
        _record(
            "feature_manifest",
            tmp_path / "full-PL.jsonl",
            scope="full_train",
            invariants=("PL",),
            representation_hash="repr",
        ),
        _record(
            "feature_qc",
            tmp_path / "full-qc.json",
            scope="full_train",
            invariants=("PH", "PL"),
            representation_hash="repr",
        ),
        _record(
            "filtration_audit",
            tmp_path / "full-filtration.json",
            scope="full_train",
            invariants=("PH", "PL"),
            representation_hash="repr",
        ),
        _record(
            "probe_selection",
            tmp_path / "probe.json",
            representation_hash="repr",
            selection_hash=selection_hash,
        ),
        _record(
            "feature_manifest",
            tmp_path / "probe-PH.jsonl",
            invariants=("PH",),
            representation_hash="repr",
            selection_hash=selection_hash,
        ),
        _record(
            "feature_manifest",
            tmp_path / "probe-PL.jsonl",
            invariants=("PL",),
            representation_hash="repr",
            selection_hash=selection_hash,
        ),
        _record(
            "feature_qc",
            tmp_path / "probe-qc.json",
            invariants=("PH", "PL"),
            representation_hash="repr",
            selection_hash=selection_hash,
        ),
        _record(
            "filtration_audit",
            tmp_path / "probe-filtration.json",
            invariants=("PH", "PL"),
            representation_hash="repr",
            selection_hash=selection_hash,
        ),
        _record(
            "scout_execution_plan",
            tmp_path / "scout-superset.json",
            invariants=("CA", "PH", "PL"),
            representation_hash="repr",
            selection_hash=selection_hash,
        ),
        _record(
            "scout_oof",
            tmp_path / "stale-scout-PH.json",
            invariants=("PH",),
            representation_hash="repr",
            selection_hash=selection_hash,
            metadata={"gbt_parameter_hash": "stale-gbt"},
        ),
        _record(
            "scout_oof",
            tmp_path / "current-scout-PL.json",
            invariants=("PL",),
            representation_hash="repr",
            selection_hash=selection_hash,
            metadata={"gbt_parameter_hash": expected_gbt_hash},
        ),
    ):
        registry.register(record)
    graph = build_agent_graph(
        AgentGraphContext(
            task_config=task,
            registry=registry_path,
            execution_profile=Path("configs/execution/sapelo2.yaml"),
        )
    )

    result = graph.invoke(
        {
            "run_id": "scout-oof-plan",
            "request": {
                "dataset_id": "toy",
                "invariants": ["PH", "PL"],
                "evidence_scope": "full_train",
                "representation_spec": str(tmp_path / "representation.json"),
            },
            "node_trace": [],
        }
    )

    assert result["status"] == "NEEDS_SCOUT_OOF"
    assert [job["invariant"] for job in result["scout_jobs"]] == ["PH"]
    assert all(job["job_kind"] == "scout_oof" for job in result["scout_jobs"])
    assert result["scout_jobs"][0]["input_manifests"] == {
        "PH": str(tmp_path / "probe-PH.jsonl")
    }


def test_complete_scout_oof_set_recombines_when_cached_target_differs(tmp_path: Path):
    task = tmp_path / "task.yaml"
    task.write_text(
        "task_id: toy\nsystem_type: protein_ligand\ntask_type: regression\n"
    )
    registry_path = tmp_path / "artifacts.sqlite"
    registry = ArtifactRegistry(registry_path)
    selection_hash = "frozen-probe"
    expected_gbt_hash = ScoutConfig.from_mapping(
        load_yaml("configs/scout/v1.yaml")
    ).gbt.parameter_hash
    for record in (
        _record("probe_audit", tmp_path / "audit.json"),
        _record(
            "representation_design",
            tmp_path / "representation.json",
            representation_hash="repr",
        ),
        _record(
            "feature_manifest",
            tmp_path / "full-PL.jsonl",
            scope="full_train",
            invariants=("PL",),
            representation_hash="repr",
        ),
        _record(
            "feature_qc",
            tmp_path / "full-qc.json",
            scope="full_train",
            invariants=("PL",),
            representation_hash="repr",
        ),
        _record(
            "filtration_audit",
            tmp_path / "full-filtration.json",
            scope="full_train",
            invariants=("PL",),
            representation_hash="repr",
        ),
        _record(
            "probe_selection",
            tmp_path / "probe.json",
            representation_hash="repr",
            selection_hash=selection_hash,
        ),
        _record(
            "feature_manifest",
            tmp_path / "probe-PL.jsonl",
            invariants=("PL",),
            representation_hash="repr",
            selection_hash=selection_hash,
        ),
        _record(
            "feature_qc",
            tmp_path / "probe-qc.json",
            invariants=("PL",),
            representation_hash="repr",
            selection_hash=selection_hash,
        ),
        _record(
            "filtration_audit",
            tmp_path / "probe-filtration.json",
            invariants=("PL",),
            representation_hash="repr",
            selection_hash=selection_hash,
        ),
        _record(
            "scout_oof",
            tmp_path / "scout-PL.json",
            invariants=("PL",),
            representation_hash="repr",
            selection_hash=selection_hash,
            metadata={"gbt_parameter_hash": expected_gbt_hash},
        ),
        _record(
            "scout_execution_plan",
            tmp_path / "stale-scout.json",
            invariants=("PL",),
            representation_hash="repr",
            selection_hash=selection_hash,
            metadata={"target_value": 0.65, "gbt_parameter_hash": expected_gbt_hash,},
        ),
    ):
        registry.register(record)
    graph = build_agent_graph(
        AgentGraphContext(
            task_config=task,
            registry=registry_path,
            execution_profile=Path("configs/execution/sapelo2.yaml"),
        )
    )

    result = graph.invoke(
        {
            "run_id": "scout-combine-plan",
            "request": {
                "dataset_id": "toy",
                "invariants": ["PL"],
                "evidence_scope": "full_train",
                "representation_spec": str(tmp_path / "representation.json"),
                "user_target": 0.75,
            },
            "node_trace": [],
        }
    )

    assert result["status"] == "NEEDS_SCOUT_COMBINE"
    assert len(result["scout_jobs"]) == 1
    assert result["scout_jobs"][0]["job_kind"] == "scout_combine"
    assert result["scout_jobs"][0]["input_manifests"] == {
        "PL": str(tmp_path / "scout-PL.json")
    }
    assert result["scout_jobs"][0]["environment"]["USER_TARGET"] == "0.75"


def test_missing_model_evaluation_emits_registered_gbt_runner_plan(tmp_path: Path):
    task = tmp_path / "task.yaml"
    task.write_text(
        "task_id: toy\nsystem_type: protein_ligand\ntask_type: regression\n"
    )
    registry_path = tmp_path / "artifacts.sqlite"
    registry = ArtifactRegistry(registry_path)
    _register_probe_prerequisites(registry, tmp_path, ("PL",))
    for record in (
        _record("probe_audit", tmp_path / "audit.json"),
        _record(
            "representation_design",
            tmp_path / "representation.json",
            representation_hash="repr",
        ),
        _record(
            "feature_manifest",
            tmp_path / "full-PL.jsonl",
            scope="full_train",
            invariants=("PL",),
            representation_hash="repr",
        ),
        _record(
            "feature_qc",
            tmp_path / "full-qc.json",
            scope="full_train",
            invariants=("PL",),
            representation_hash="repr",
        ),
        _record(
            "filtration_audit",
            tmp_path / "full-filtration.json",
            scope="full_train",
            invariants=("PL",),
            representation_hash="repr",
        ),
        _record(
            "scout_execution_plan",
            tmp_path / "scout.json",
            invariants=("PL",),
            representation_hash="repr",
            selection_hash="selection",
        ),
    ):
        registry.register(record)
    graph = build_agent_graph(
        AgentGraphContext(
            task_config=task,
            registry=registry_path,
            execution_profile=Path("configs/execution/sapelo2.yaml"),
        )
    )

    result = graph.invoke(
        {
            "run_id": "evaluation-plan",
            "request": {
                "dataset_id": "toy",
                "invariants": ["PL"],
                "evidence_scope": "full_train",
                "representation_spec": str(tmp_path / "representation.json"),
            },
            "node_trace": [],
        }
    )

    assert result["status"] == "NEEDS_MODEL_EVALUATION"
    assert len(result["evaluation_jobs"]) == 1
    assert result["evaluation_jobs"][0]["job_kind"] == "model_evaluation"
    assert result["evaluation_jobs"][0]["environment"]["SCOUT_ARTIFACT"] == str(
        tmp_path / "scout.json"
    )


def test_cli_writes_top_level_auditable_report(tmp_path: Path):
    task = tmp_path / "task.yaml"
    task.write_text(
        "task_id: toy\nsystem_type: protein_ligand\ntask_type: regression\n"
    )
    registry_path = tmp_path / "artifacts.sqlite"
    registry = ArtifactRegistry(registry_path)
    _register_probe_prerequisites(registry, tmp_path, ("PL",))
    for record in (
        _record("probe_audit", tmp_path / "audit.json"),
        _record(
            "representation_design",
            tmp_path / "representation.json",
            representation_hash="repr",
        ),
        _record(
            "feature_manifest",
            tmp_path / "PL.jsonl",
            scope="full_train",
            invariants=("PL",),
            representation_hash="repr",
        ),
        _record(
            "feature_qc",
            tmp_path / "qc.json",
            scope="full_train",
            invariants=("PL",),
            representation_hash="repr",
        ),
        _record(
            "filtration_audit",
            tmp_path / "filtration.json",
            scope="full_train",
            invariants=("PL",),
            representation_hash="repr",
        ),
        _record(
            "scout_execution_plan",
            tmp_path / "scout.json",
            invariants=("PL",),
            representation_hash="repr",
            selection_hash="selection",
        ),
        _record(
            "model_evaluation",
            tmp_path / "evaluation.json",
            scope="full_train",
            invariants=("PL",),
            representation_hash="repr",
        ),
    ):
        registry.register(record)
    output = tmp_path / "run.json"

    assert (
        run_graph_main(
            [
                "--task-config",
                str(task),
                "--registry",
                str(registry_path),
                "--dataset-id",
                "toy",
                "--invariant",
                "PL",
                "--thread-id",
                "cli-test",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["report_schema"] == "mint-agent.graph-run.v1"
    assert report["dataset_id"] == "toy"
    assert report["invariants"] == ["PL"]
    assert report["artifact_snapshot_count"] == 12
    assert report["sample_count"] == 2
    assert report["sample_order_hash"] == "order"
    assert report["node_trace"][-1] == "report_node"
    assert "artifacts" not in report


def test_cli_preserves_early_manifest_validation_error(tmp_path: Path):
    task = tmp_path / "task.yaml"
    task.write_text(
        "task_id: invalid-manifest\n"
        "system_type: protein_ligand\n"
        "task_type: regression\n"
        "dataset_manifest:\n"
        "  path: missing.csv\n",
        encoding="utf-8",
    )
    output = tmp_path / "run.json"

    code = run_graph_main(
        [
            "--task-config",
            str(task),
            "--registry",
            str(tmp_path / "artifacts.sqlite"),
            "--dataset-id",
            "invalid-manifest",
            "--invariant",
            "PL",
            "--thread-id",
            "invalid-manifest-test",
            "--output",
            str(output),
        ]
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert code == 2
    assert report["status"] == "INVALID_DATASET_MANIFEST"
    assert report["dataset_id"] == "invalid-manifest"
    assert "does not exist" in report["errors"][0]
    assert report["node_trace"] == [
        "dataset_preparation_node",
        "project_context_node",
        "report_node",
    ]


def test_graph_cli_treats_actionable_needs_state_as_successful_run():
    assert _exit_code("COMPLETE") == 0
    assert _exit_code("NEEDS_FEATURE_COMPUTE") == 0
    assert _exit_code("INCOMPATIBLE_FEATURE_QC") == 2
