from __future__ import annotations

import json
from pathlib import Path

import pytest

from mint_scout.controlled_experiment import (
    ProbeComparisonPolicy,
    RepresentationComparisonPolicy,
    _pair_coverage,
    build_controlled_experiment_matrix,
    choose_probe_candidate,
    choose_representation_candidate,
    materialize_controlled_experiment,
)


def _write(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _configs() -> tuple[dict, dict]:
    task = {
        "task_id": "toy",
        "system_type": "protein_ligand",
        "representation_design": {
            "support": {
                "min_element_support_fraction": 0.005,
                "min_element_support_samples": 10,
                "min_pair_support_fraction": 0.005,
                "min_pair_support_samples": 10,
                "max_element_pair_channels": 50,
                "small_molecule_self_pairs": False,
            },
            "filtration": {
                "local_distance_quantile": 0.95,
                "dataset_distance_quantile": 0.95,
                "margin_factor": 1.10,
                "default_step_angstrom": 0.1,
                "allowed_steps_angstrom": [0.1, 0.2],
                "max_points": 200,
                "include_zero": True,
                "fixed_point_count": 50,
            },
        },
    }
    scout = {
        "probe": {
            "fraction": 0.20,
            "min_samples": 300,
            "max_samples": 1000,
            "target_quantile_bins": 10,
            "size_quantile_bins": 5,
            "min_pair_support": 3,
            "augmentation_fraction_limit": 0.10,
            "random_seed": 2026,
        }
    }
    return task, scout


def _experiment_plan() -> dict:
    return {
        "mode": "advisory",
        "status": "VALIDATED",
        "input_hash": "input",
        "applied_to_execution": True,
        "advice": {
            "probe_strategy": {"stratify_by": ["target", "molecular_size"]},
            "probe_variants": [
                {
                    "id": "larger-probe",
                    "fraction": 0.30,
                    "max_samples": 750,
                    "target_quantile_bins": 15,
                    "size_quantile_bins": 8,
                    "min_pair_support": 5,
                    "augmentation_fraction_limit": 0.20,
                    "rationale": "Improve rare-pair coverage.",
                }
            ],
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
                    "rationale": "Test a longer global distance range.",
                }
            ],
            "representation_hypotheses": [],
        },
    }


def test_matrix_injects_baseline_and_bounded_llm_candidates(tmp_path: Path):
    task, scout = _configs()
    scout["ranking"] = {
        "weights": {
            "probe_performance": 0.40,
            "ranking_stability": 0.25,
            "feature_quality": 0.20,
            "computational_efficiency": 0.10,
            "llm_prior": 0.05,
        }
    }
    matrix = build_controlled_experiment_matrix(
        task_config=task, scout_config=scout, experiment_plan=_experiment_plan(),
    )

    assert matrix["execution_allowed"] is True
    assert [row["id"] for row in matrix["probe_candidates"]] == [
        "baseline-probe",
        "larger-probe",
    ]
    assert [row["id"] for row in matrix["representation_candidates"]] == [
        "baseline-representation",
        "q99-range",
    ]
    alternative = matrix["representation_candidates"][1]["config"]
    assert alternative["representation_design"]["filtration"]["fixed_point_count"] == 50
    assert (
        alternative["representation_design"]["filtration"]["dataset_distance_quantile"]
        == 0.99
    )
    assert (
        matrix["representation_comparison_policy"]["decision_rule"]
        == "hierarchical_representation_suitability"
    )
    assert matrix["representation_comparison_policy"]["llm_prior_used"] is False
    assert matrix["representation_comparison_policy"]["uses_model_performance"] is False
    assert matrix["representation_comparison_policy"]["uses_ranking_stability"] is False
    assert matrix["representation_comparison_policy"]["uses_runtime_cost"] is False
    assert matrix["probe_comparison_policy"]["uses_weighted_composite"] is False
    assert (
        matrix["probe_comparison_policy"]["decision_rule"]
        == "representativeness_gates_then_minimum_sample_cost"
    )

    materialized = materialize_controlled_experiment(
        matrix, output_dir=tmp_path / "experiment"
    )
    assert Path(materialized["manifest_path"]).is_file()
    assert all(
        Path(row["config_path"]).is_file()
        for row in materialized["representation_candidates"]
    )


def _probe_selection(candidate: str, sample_ids: list[str]) -> dict:
    return {
        "report_schema": "mint-agent.casf-probe-selection.v1",
        "task_id": "toy",
        "representation_hash": "repr",
        "selection_hash": f"selection-{candidate}",
        "modeling_sample_ids": ["a", "b", "c", "d"],
        "probe_sample_ids": sample_ids,
        "modeling_fold_assignment": {"a": 0, "b": 1, "c": 0, "d": 1},
        "probe_fold_assignment": {
            sample_id: {"a": 0, "b": 1, "c": 0, "d": 1}[sample_id]
            for sample_id in sample_ids
        },
    }


def _probe_audit(candidate: str, count: int, error: float) -> dict:
    comparison = {
        "ks_distance": error,
        "standardized_mean_difference": error,
        "extreme_coverage": {"minimum_retained": True, "maximum_retained": True},
    }
    return {
        "report_schema": "mint-agent.probe-representativeness-audit.v1",
        "selection_hash": f"selection-{candidate}",
        "probe_sample_count": count,
        "distribution_comparisons": {
            "target": comparison,
            "total_atom_count": comparison,
        },
        "joint_target_total_size_strata": {
            "total_variation_distance": error,
            "max_abs_cell_share_error": error,
            "all_target_bins_covered": True,
            "all_size_bins_covered": True,
        },
        "element_pair_coverage": {
            "pairs": [{"minimum_met": True}, {"minimum_met": True}],
            "all_support_consistent": True,
            "all_minimum_support_met": True,
        },
        "warnings": [],
        "status": "PASS",
    }


def test_probe_comparison_uses_representativeness_not_model_performance(tmp_path: Path):
    baseline_selection = _write(
        tmp_path / "baseline-selection.json", _probe_selection("baseline", ["a", "b"]),
    )
    baseline_audit = _write(
        tmp_path / "baseline-audit.json", _probe_audit("baseline", 2, 0.30)
    )
    better_selection = _write(
        tmp_path / "better-selection.json", _probe_selection("better", ["a", "b", "c"]),
    )
    better_audit = _write(
        tmp_path / "better-audit.json", _probe_audit("better", 3, 0.02)
    )

    result = choose_probe_candidate(
        [
            {
                "candidate_id": "baseline-probe",
                "selection": baseline_selection,
                "audit": baseline_audit,
            },
            {
                "candidate_id": "better-probe",
                "selection": better_selection,
                "audit": better_audit,
            },
        ]
    )

    assert result["selected_candidate_id"] == "better-probe"
    assert result["test_evidence_used"] is False
    assert (
        result["selection_basis"]
        == "representativeness_gates_then_minimum_sample_cost"
    )
    assert "final_score" not in result["candidates"][0]


def test_probe_comparison_selects_smallest_candidate_after_all_gates_pass(
    tmp_path: Path,
):
    smaller_selection = _write(
        tmp_path / "smaller-selection.json",
        _probe_selection("smaller", ["a", "b"]),
    )
    smaller_audit = _write(
        tmp_path / "smaller-audit.json", _probe_audit("smaller", 750, 0.03)
    )
    larger_selection = _write(
        tmp_path / "larger-selection.json",
        _probe_selection("larger", ["a", "b", "c"]),
    )
    larger_audit = _write(
        tmp_path / "larger-audit.json", _probe_audit("larger", 1000, 0.01)
    )

    result = choose_probe_candidate(
        [
            {
                "candidate_id": "smaller-probe",
                "selection": smaller_selection,
                "audit": smaller_audit,
            },
            {
                "candidate_id": "larger-probe",
                "selection": larger_selection,
                "audit": larger_audit,
            },
        ],
        policy=ProbeComparisonPolicy(),
    )

    assert result["selected_candidate_id"] == "smaller-probe"
    assert result["decision_trace"][1]["minimum_probe_sample_count"] == 750
    assert all(candidate["eligible"] for candidate in result["candidates"])


def test_probe_comparison_rejects_cheaper_candidate_that_fails_a_gate(
    tmp_path: Path,
):
    failing_selection = _write(
        tmp_path / "failing-selection.json",
        _probe_selection("failing", ["a", "b"]),
    )
    failing_audit = _write(
        tmp_path / "failing-audit.json", _probe_audit("failing", 500, 0.11)
    )
    passing_selection = _write(
        tmp_path / "passing-selection.json",
        _probe_selection("passing", ["a", "b", "c"]),
    )
    passing_audit = _write(
        tmp_path / "passing-audit.json", _probe_audit("passing", 1000, 0.02)
    )

    result = choose_probe_candidate(
        [
            {
                "candidate_id": "failing-probe",
                "selection": failing_selection,
                "audit": failing_audit,
            },
            {
                "candidate_id": "passing-probe",
                "selection": passing_selection,
                "audit": passing_audit,
            },
        ]
    )

    assert result["selected_candidate_id"] == "passing-probe"
    failing = next(
        row for row in result["candidates"] if row["candidate_id"] == "failing-probe"
    )
    assert failing["eligible"] is False
    assert failing["gate_failures"] == ["ks_distance", "joint_strata_share"]


def _representation(
    candidate: str,
    *,
    selected_pair_presence: tuple[int, ...] = (3, 3),
    omitted_pair_presence: tuple[int, ...] = (),
    unsupported_observed: dict | None = None,
) -> dict:
    pair_support = [
        {
            "pair": ["C", f"X{index}"],
            "sample_presence": presence,
            "support_fraction": presence / 3.0,
            "retained": True,
            "selected": True,
        }
        for index, presence in enumerate(selected_pair_presence)
    ]
    pair_support.extend(
        {
            "pair": ["C", f"O{index}"],
            "sample_presence": presence,
            "support_fraction": presence / 3.0,
            "retained": True,
            "selected": False,
        }
        for index, presence in enumerate(omitted_pair_presence)
    )
    return {
        "report_schema": "mint-agent.representation-design.v1",
        "representation_hash": f"repr-{candidate}",
        "modeling_scope": "train",
        "sample_count": 3,
        "representation_spec": {
            "frozen": True,
            "pair_order": [row["pair"] for row in pair_support if row["selected"]],
            "parameters": {
                "pair_support": pair_support,
                "adapter_compatibility": {
                    "unsupported_observed": unsupported_observed or {}
                },
            },
        },
    }


def _qc(candidate: str, *, warning_count: int = 0, dimensions: int = 100) -> dict:
    return {
        "status": "WARN" if warning_count else "PASS",
        "evidence_scope": "probe",
        "representation_hash": f"repr-{candidate}",
        "warning_issue_count": warning_count,
        "issues": [
            {"severity": "warning", "issue": "robust_outlier"}
            for _ in range(warning_count)
        ],
        "invariants": {"PL": {"expected_shape": [dimensions]}},
    }


def _filtration(candidate: str, *recommendations: str) -> dict:
    values = recommendations or ("KEEP",)
    return {
        "status": "PASS" if set(values) == {"KEEP"} else "WARN",
        "evidence_scope": "probe",
        "representation_hash": f"repr-{candidate}",
        "invariants": {
            f"I{index}": {"recommendation": recommendation}
            for index, recommendation in enumerate(values)
        },
    }


def _scout(
    candidate: str,
    score: float,
    cost: float,
    *,
    top1_frequency: float = 0.8,
    bootstrap_std: float = 0.03,
    ci_low: float | None = None,
    ci_high: float | None = None,
) -> dict:
    return {
        "report_schema": "mint-agent.combined-scout.v1",
        "representation_hash": f"repr-{candidate}",
        "selection_hash": f"selection-{candidate}",
        "frozen_priority_order": [["PL"]],
        "consensus_metrics": [
            {
                "invariants": ["PL"],
                "nominal_primary_score": score,
                "top1_frequency": top1_frequency,
                "median_rank": 1.0,
                "bootstrap_std": bootstrap_std,
                "bootstrap_ci_low": ci_low,
                "bootstrap_ci_high": ci_high,
                "feature_quality_component": 1.0,
            }
        ],
        "cost_profile": {"PL": {"estimated_full_wall_seconds": cost},},
    }


def test_representation_comparison_uses_label_free_suitability_not_probe_oof(
    tmp_path: Path,
):
    descriptors = []
    for candidate_id, token, score, cost in (
        ("baseline-representation", "baseline", 0.95, 1.0),
        ("adaptive-q99", "adaptive", 0.10, 999.0),
    ):
        probe = _probe_selection(token, ["a", "b", "c"])
        descriptor = {
            "candidate_id": candidate_id,
            "llm_prior_rank": None if candidate_id == "baseline-representation" else 1,
            "representation": _write(
                tmp_path / f"{token}-representation.json", _representation(token)
            ),
            "probe": _write(tmp_path / f"{token}-probe.json", probe),
            # Representation selection must not read model-performance artifacts.
            "scout": tmp_path / f"{token}-intentionally-absent-scout.json",
            "qc": _write(tmp_path / f"{token}-qc.json", _qc(token)),
            "filtration": _write(
                tmp_path / f"{token}-filtration.json",
                _filtration(token, "EXTEND" if token == "baseline" else "KEEP"),
            ),
        }
        descriptors.append(descriptor)

    result = choose_representation_candidate(descriptors, primary_metric="PCC")

    assert result["selected_candidate_id"] == "adaptive-q99"
    assert result["test_evidence_used"] is False
    assert (
        result["selection_basis"]
        == "hierarchical_train_only_representation_suitability"
    )
    assert result["llm_prior_used"] is False
    assert result["model_performance_used"] is False
    assert result["ranking_stability_used"] is False
    assert result["runtime_cost_used"] is False


def test_representation_hierarchy_prefers_coverage_before_feature_health(
    tmp_path: Path,
):
    descriptors = []
    for candidate_id, token, selected, omitted, warnings, llm_rank in (
        ("baseline-representation", "baseline", (3, 3), (), 1, None),
        ("healthy-but-low-coverage", "healthy", (3,), (3,), 0, 2),
        ("llm-first-but-low-coverage", "worse", (3,), (3, 3), 0, 1),
    ):
        descriptors.append(
            {
                "candidate_id": candidate_id,
                "llm_prior_rank": llm_rank,
                "representation": _write(
                    tmp_path / f"{token}-representation.json",
                    _representation(
                        token,
                        selected_pair_presence=selected,
                        omitted_pair_presence=omitted,
                    ),
                ),
                "probe": _write(
                    tmp_path / f"{token}-probe.json",
                    _probe_selection(token, ["a", "b", "c"]),
                ),
                "scout": _write(
                    tmp_path / f"{token}-scout.json",
                    _scout(token, 0.99 if token == "worse" else 0.10, 1.0),
                ),
                "qc": _write(
                    tmp_path / f"{token}-qc.json", _qc(token, warning_count=warnings)
                ),
                "filtration": _write(
                    tmp_path / f"{token}-filtration.json", _filtration(token, "KEEP")
                ),
            }
        )

    result = choose_representation_candidate(
        descriptors,
        primary_metric="PCC",
        policy=RepresentationComparisonPolicy(
            pair_coverage_tolerance=0.0,
            filtration_issue_tolerance=0,
            feature_warning_tolerance=0,
            dimension_tolerance=0,
        ),
    )

    assert result["selected_candidate_id"] == "baseline-representation"
    coverage_stage = next(
        stage
        for stage in result["decision_trace"]
        if stage["stage"] == "element_pair_coverage"
    )
    assert coverage_stage["survivors"] == ["baseline-representation"]


def test_pair_coverage_uses_support_mass_not_raw_channel_fraction():
    representation = _representation(
        "pruned",
        selected_pair_presence=(100, 100),
        omitted_pair_presence=(1,),
    )

    coverage, details = _pair_coverage(
        representation["representation_spec"]["parameters"], representation
    )

    assert coverage == 200 / 201
    assert details["selected_channel_fraction"] == 2 / 3
    assert details["coverage_basis"] == (
        "support_mass_and_non_tolerated_adapter_coverage"
    )


def test_pair_coverage_ignores_tolerated_unsupported_elements():
    representation = _representation(
        "tolerated",
        unsupported_observed={
            "protein": [
                {"element": "H", "sample_presence": 3, "tolerated": True}
            ]
        },
    )

    coverage, details = _pair_coverage(
        representation["representation_spec"]["parameters"], representation
    )

    assert coverage == 1.0
    assert details["adapter_supported_sample_fraction"] == 1.0


def test_pair_coverage_penalizes_non_tolerated_unsupported_elements():
    representation = _representation(
        "unsupported",
        unsupported_observed={
            "protein": [
                {"element": "Xx", "sample_presence": 2, "tolerated": False}
            ]
        },
    )

    coverage, details = _pair_coverage(
        representation["representation_spec"]["parameters"], representation
    )

    assert coverage == pytest.approx(1 / 3)
    assert details["adapter_supported_sample_fraction"] == pytest.approx(1 / 3)
