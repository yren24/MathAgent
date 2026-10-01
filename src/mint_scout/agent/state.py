from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict


class MintAgentState(TypedDict, total=False):
    run_id: str
    request: dict[str, Any]
    task: dict[str, Any]
    representation_hash: str
    representation_spec_path: str
    artifacts: list[dict[str, Any]]
    preparation_jobs: list[dict[str, Any]]
    dataset_preparation: dict[str, Any]
    setup_jobs: list[dict[str, Any]]
    dataset_audit: dict[str, Any]
    experiment_plan: dict[str, Any]
    controlled_experiment: dict[str, Any]
    controlled_probe_selection: dict[str, Any]
    controlled_representation_selection: dict[str, Any]
    probe_selection: dict[str, Any]
    probe_audit: dict[str, Any]
    feature_plan: dict[str, Any]
    feature_jobs: list[dict[str, Any]]
    qc_jobs: list[dict[str, Any]]
    filtration_audit_jobs: list[dict[str, Any]]
    feature_diagnostic_jobs: list[dict[str, Any]]
    scout_jobs: list[dict[str, Any]]
    evaluation_jobs: list[dict[str, Any]]
    split_feature_plan: dict[str, Any]
    split_qc: dict[str, dict[str, Any]]
    split_filtration_audits: dict[str, dict[str, Any]]
    selection_evaluation: dict[str, Any]
    acceptance_plan: dict[str, Any]
    final_test_plan: dict[str, Any]
    qc: dict[str, Any]
    filtration_audit: dict[str, Any]
    filtration_repair: dict[str, Any]
    feature_health: dict[str, Any]
    feature_diagnostics: dict[str, dict[str, Any]]
    scout: dict[str, Any]
    scientific_critique: dict[str, Any]
    evaluation: dict[str, Any]
    route: str
    status: str
    next_actions: list[str]
    final_report: dict[str, Any]
    report_bundle: dict[str, str]
    errors: list[str]
    node_trace: Annotated[list[str], operator.add]
