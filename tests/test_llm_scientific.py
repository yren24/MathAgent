from __future__ import annotations

import json
from pathlib import Path

import pytest

from mint_scout.agent.llm_scientific import (
    LLMScientificContext,
    _experiment_plan_schema,
    deterministic_experiment_plan,
    experiment_planning_advice,
    experiment_planning_facts,
    scientific_critic_advice,
    scientific_critic_facts,
)
from mint_scout.run_agent_graph import _read_llm_env_file


def _audit(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "status": "PASS",
                "sample_count": 12,
                "sample_ids": ["must-not-be-sent"],
                "private_path": "/private/raw.csv",
                "target_summary": {
                    "count": 12,
                    "minimum": 1.0,
                    "maximum": 9.0,
                    "mean": 5.0,
                    "std": 2.0,
                    "quantiles": {"q05": 1.5, "q50": 5.0, "q95": 8.5},
                },
                "roles": [
                    {
                        "role": "protein",
                        "file_count": 12,
                        "atom_count": 1200,
                        "element_counts": {"C": 600, "N": 120, "Zn": 2},
                        "out_of_schema_elements": ["Zn"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def _planning_state(tmp_path: Path) -> dict:
    return {
        "request": {"dataset_id": "toy", "invariants": ["PL", "PH"]},
        "task": {
            "system_type": "protein_ligand",
            "task_type": "regression",
            "primary_metric": "pcc",
            "evaluation_mode": "full_labeled_cv",
            "sample_count": 12,
            "labeled_sample_count": 12,
        },
        "dataset_audit": {"path": str(_audit(tmp_path / "audit.json"))},
    }


def _valid_plan(*, invariants: list[str] | None = None) -> dict:
    chosen = invariants or ["PL"]
    return {
        "objective": "Prioritize probe experiments.",
        "probe_strategy": {
            "sampling_strategy": "group_aware_stratified",
            "stratify_by": ["target", "molecular_size"],
            "group_by": "provided_group_identifier",
            "preserve_extremes": True,
            "rationale": "Use train-only summaries.",
        },
        "probe_variants": [],
        "representation_variants": [],
        "representation_hypotheses": [
            {
                "id": "adaptive-filtration",
                "category": "filtration_range",
                "methods": chosen,
                "proposal": "Check train-only filtration coverage.",
                "evidence_needed": ["filtration audit"],
            }
        ],
        "candidate_priorities": [
            {
                "rank": 1,
                "invariants": chosen,
                "rationale": "Probe first.",
                "expected_cost": "unknown",
            }
        ],
        "risk_checks": ["no_test_evidence", "feature_quality"],
    }


def _valid_critique() -> dict:
    return {
        "evidence_assessment": "preliminary",
        "ranking_interpretation": "Use the probe ranking as preliminary evidence.",
        "stability_findings": ["Check bootstrap stability."],
        "cost_findings": ["Use measured cost for similar candidates."],
        "quality_findings": ["Keep QC gates mandatory."],
        "recommended_next_action": "continue_deterministic_pipeline",
        "rationale": "The deterministic pipeline remains in control.",
    }


def test_openai_experiment_schema_avoids_unsupported_constraint_keywords():
    serialized = json.dumps(_experiment_plan_schema())

    for keyword in ("uniqueItems", "minItems", "maxItems", "minimum"):
        assert keyword not in serialized


def test_duplicate_stratification_is_rejected_by_python_validator(tmp_path: Path):
    def planner(_facts, **_kwargs):
        plan = _valid_plan()
        plan["probe_strategy"]["stratify_by"] = ["target", "target"]
        return plan, {"used": True}

    context = LLMScientificContext(
        mode="shadow",
        model="test-model",
        api_key="top-secret",
        planner=planner,
        critic=lambda *_args, **_kwargs: (_valid_critique(), {"used": True}),
    )

    result = experiment_planning_advice(_planning_state(tmp_path), context)

    assert result["status"] == "FALLBACK"
    assert "must not contain duplicates" in result["validation"]["errors"][0]


def test_empty_hypothesis_methods_are_rejected_by_python_validator(tmp_path: Path):
    def planner(_facts, **_kwargs):
        plan = _valid_plan()
        plan["representation_hypotheses"][0]["methods"] = []
        return plan, {"used": True}

    context = LLMScientificContext(
        mode="shadow",
        model="test-model",
        api_key="top-secret",
        planner=planner,
        critic=lambda *_args, **_kwargs: (_valid_critique(), {"used": True}),
    )

    result = experiment_planning_advice(_planning_state(tmp_path), context)

    assert result["status"] == "FALLBACK"
    assert "non-empty list" in result["validation"]["errors"][0]


def test_experiment_planning_facts_omit_paths_and_sample_ids(tmp_path: Path):
    facts = experiment_planning_facts(_planning_state(tmp_path))
    serialized = json.dumps(facts)

    assert "/private/raw.csv" not in serialized
    assert "must-not-be-sent" not in serialized
    assert str(tmp_path) not in serialized
    assert facts["allowed_invariants"] == ["PL", "PH"]
    assert (
        facts["data_summary"]["element_summary"]["protein"]["element_counts"]["Zn"] == 2
    )
    assert facts["information_boundary"]["test_metrics_included"] is False


def test_disabled_mode_returns_deterministic_plan_without_llm(tmp_path: Path):
    result = experiment_planning_advice(
        _planning_state(tmp_path), LLMScientificContext(mode="disabled")
    )

    assert result["status"] == "DISABLED"
    assert result["llm"]["used"] is False
    assert result["advice"] == deterministic_experiment_plan(
        experiment_planning_facts(_planning_state(tmp_path))
    )


def test_valid_mock_planner_is_validated_and_cached(tmp_path: Path):
    calls = {"count": 0}

    def planner(_facts, *, model, api_key):
        calls["count"] += 1
        assert model == "test-model"
        assert api_key == "top-secret"
        return _valid_plan(), {"used": True, "model_requested": model}

    context = LLMScientificContext(
        mode="shadow",
        model="test-model",
        api_key="top-secret",
        cache_dir=tmp_path / "cache",
        planner=planner,
        critic=lambda *_args, **_kwargs: (_valid_critique(), {"used": True}),
    )

    first = experiment_planning_advice(_planning_state(tmp_path), context)
    second = experiment_planning_advice(_planning_state(tmp_path), context)

    assert first["status"] == "VALIDATED"
    assert first["cache_hit"] is False
    assert second["cache_hit"] is True
    assert calls["count"] == 1
    assert "top-secret" not in json.dumps(first)


def test_cached_shadow_advice_uses_current_advisory_mode(tmp_path: Path):
    calls = {"count": 0}

    def planner(_facts, **_kwargs):
        calls["count"] += 1
        return _valid_plan(), {"used": True}

    cache_dir = tmp_path / "cache"
    shadow = LLMScientificContext(
        mode="shadow",
        model="test-model",
        api_key="top-secret",
        cache_dir=cache_dir,
        planner=planner,
        critic=lambda *_args, **_kwargs: (_valid_critique(), {"used": True}),
    )
    advisory = LLMScientificContext(
        mode="advisory",
        model="test-model",
        api_key="top-secret",
        cache_dir=cache_dir,
        planner=planner,
        critic=lambda *_args, **_kwargs: (_valid_critique(), {"used": True}),
    )

    first = experiment_planning_advice(_planning_state(tmp_path), shadow)
    second = experiment_planning_advice(_planning_state(tmp_path), advisory)

    assert first["mode"] == "shadow"
    assert second["mode"] == "advisory"
    assert second["cache_hit"] is True
    assert second["applied_to_execution"] is False
    assert calls["count"] == 1


def test_fallback_is_not_cached_so_transient_failures_can_retry(tmp_path: Path):
    calls = {"count": 0}

    def planner(_facts, **_kwargs):
        calls["count"] += 1
        raise TimeoutError("temporary read timeout")

    context = LLMScientificContext(
        mode="shadow",
        model="test-model",
        api_key="top-secret",
        cache_dir=tmp_path / "cache",
        planner=planner,
        critic=lambda *_args, **_kwargs: (_valid_critique(), {"used": True}),
    )

    first = experiment_planning_advice(_planning_state(tmp_path), context)
    second = experiment_planning_advice(_planning_state(tmp_path), context)

    assert first["status"] == "FALLBACK"
    assert second["status"] == "FALLBACK"
    assert first["cache_hit"] is False
    assert second["cache_hit"] is False
    assert calls["count"] == 2
    assert not list((tmp_path / "cache").rglob("*.json"))


def test_llm_timeout_must_be_positive():
    with pytest.raises(ValueError, match="timeout_seconds must be positive"):
        LLMScientificContext(timeout_seconds=0)


def test_invalid_planner_output_falls_back_without_blocking(tmp_path: Path):
    def planner(_facts, **_kwargs):
        return _valid_plan(invariants=["EIC"]), {"used": True}

    context = LLMScientificContext(
        mode="advisory",
        model="test-model",
        api_key="top-secret",
        planner=planner,
        critic=lambda *_args, **_kwargs: (_valid_critique(), {"used": True}),
    )

    result = experiment_planning_advice(_planning_state(tmp_path), context)

    assert result["status"] == "FALLBACK"
    assert result["validation"]["passed"] is False
    assert result["applied_to_execution"] is False
    assert result["llm"]["used_for_numeric_decisions"] is False


def test_out_of_policy_adaptive_setting_falls_back(tmp_path: Path):
    def planner(_facts, **_kwargs):
        plan = _valid_plan()
        plan["representation_variants"] = [
            {
                "id": "unsupported-q93",
                "local_distance_quantile": 0.93,
                "dataset_distance_quantile": 0.95,
                "margin_factor": 1.10,
                "min_pair_support_fraction": 0.005,
                "min_pair_support_samples": 10,
                "max_element_pair_channels": 40,
                "fixed_point_count": 50,
                "rationale": "Try an unsupported numerical value.",
            }
        ]
        return plan, {"used": True}

    context = LLMScientificContext(
        mode="advisory",
        model="test-model",
        api_key="top-secret",
        planner=planner,
        critic=lambda *_args, **_kwargs: (_valid_critique(), {"used": True}),
    )

    result = experiment_planning_advice(_planning_state(tmp_path), context)

    assert result["status"] == "FALLBACK"
    assert result["validation"]["passed"] is False
    assert "local_distance_quantile" in result["validation"]["errors"][0]


def test_forbidden_held_out_claim_is_rejected(tmp_path: Path):
    def planner(_facts, **_kwargs):
        plan = _valid_plan()
        plan["objective"] = "Use test performance to choose the best invariant."
        return plan, {"used": True}

    context = LLMScientificContext(
        mode="shadow",
        model="test-model",
        api_key="top-secret",
        planner=planner,
        critic=lambda *_args, **_kwargs: (_valid_critique(), {"used": True}),
    )

    result = experiment_planning_advice(_planning_state(tmp_path), context)

    assert result["status"] == "FALLBACK"
    assert "forbidden" in result["validation"]["errors"][0]


def test_scientific_critic_uses_probe_summaries_only(tmp_path: Path):
    scout = tmp_path / "scout.json"
    scout.write_text(
        json.dumps(
            {
                "sample_count": 4,
                "frozen_priority_order": [["PL"], ["PH", "PL"]],
                "sample_ids": ["must-not-be-sent"],
                "consensus_metrics": [
                    {
                        "subset_id": "PL",
                        "invariants": ["PL"],
                        "primary_score": 0.61,
                        "bootstrap_ci_low": 0.52,
                        "bootstrap_ci_high": 0.69,
                        "private_path": "/private/scout.json",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    state = {
        "request": {"invariants": ["PL", "PH"]},
        "task": {"evaluation_mode": "full_labeled_cv"},
        "scout": {"path": str(scout), "sample_count": 4},
    }
    facts = scientific_critic_facts(state)
    serialized = json.dumps(facts)

    assert "/private/scout.json" not in serialized
    assert "must-not-be-sent" not in serialized
    assert facts["probe_evidence"]["priority_order"] == [["PL"], ["PH", "PL"]]

    context = LLMScientificContext(
        mode="shadow",
        model="test-model",
        api_key="top-secret",
        planner=lambda *_args, **_kwargs: (_valid_plan(), {"used": True}),
        critic=lambda *_args, **_kwargs: (_valid_critique(), {"used": True}),
    )
    critique = scientific_critic_advice(state, context)
    assert critique["status"] == "VALIDATED"
    assert (
        critique["advice"]["recommended_next_action"]
        == "continue_deterministic_pipeline"
    )


def test_llm_env_file_parser_supports_export_and_quotes(tmp_path: Path):
    env_file = tmp_path / "openai.env"
    env_file.write_text(
        "export OPENAI_MODEL='test-model'\nOPENAI_API_KEY=\"top-secret\"\n",
        encoding="utf-8",
    )

    values = _read_llm_env_file(env_file)

    assert values == {
        "OPENAI_MODEL": "test-model",
        "OPENAI_API_KEY": "top-secret",
    }
