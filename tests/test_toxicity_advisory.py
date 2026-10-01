from __future__ import annotations

import json
from pathlib import Path

from mint_scout.toxicity.advisory import (
    _toxicity_advice_schema,
    run_toxicity_advisory,
    toxicity_advisory_facts,
    write_toxicity_advisory,
)
from mint_scout.toxicity.design import DESIGN_SCHEMA


def _design_report() -> dict:
    return {
        "report_schema": DESIGN_SCHEMA,
        "evidence_scope": "train_only",
        "test_structures_used": False,
        "test_labels_used": False,
        "training_sample_count": 100,
        "observed_element_presence": {"H": 90, "C": 100, "N": 70, "O": 80},
        "out_of_schema_element_presence": {"Si": 2},
        "private_path": "/private/train",
        "sample_ids": ["must-not-be-sent"],
        "proposal_audits": [
            {
                "proposal": {
                    "proposal_id": "deterministic-q95",
                    "source": "deterministic",
                },
                "selected_pair_count": 6,
                "selected_pairs": [["C", "O"], ["C", "N"]],
                "pair_cap_applied": False,
                "legacy_pair_geometry": {"max_filtration_angstrom": 9.5},
                "adaptive_pair_geometry": {"max_filtration_angstrom": 8.9},
            }
        ],
    }


def _valid_advice() -> dict:
    return {
        "proposals": [
            {
                "proposal_id": "llm-q90-no-h",
                "include_hydrogen": False,
                "min_element_support_fraction": 0.001,
                "min_pair_support_fraction": 0.001,
                "min_support_samples": 5,
                "max_pair_channels": 45,
                "local_distance_quantile": 0.9,
                "dataset_distance_quantile": 0.9,
                "margin_factor": 1.0,
                "fixed_point_count": 50,
                "bond_delta": 0.45,
                "rationale": "Test a cheaper hydrogen-free representation.",
            }
        ],
        "probe_strategy": {
            "candidate_sizes": [300, 500, 750],
            "target_quantile_bins": 10,
            "size_quantile_bins": 5,
            "min_pair_support": 3,
            "rationale": "Use the standard hierarchical probe sizes.",
        },
        "overall_rationale": "Compare a bounded structural ablation.",
    }


def test_toxicity_advisory_facts_exclude_paths_ids_and_test_evidence():
    facts = toxicity_advisory_facts(_design_report())
    serialized = json.dumps(facts)

    assert "/private/train" not in serialized
    assert "must-not-be-sent" not in serialized
    assert facts["information_boundary"]["test_labels_included"] is False
    assert facts["training_summary"]["element_presence_fraction"]["H"] == 0.9


def test_toxicity_advisory_validates_bounded_proposal_and_writes_design_input(
    tmp_path: Path,
):
    captured = {}

    def planner(facts, **_kwargs):
        captured.update(facts)
        return _valid_advice(), {"used": True, "model_requested": "test-model"}

    artifact, proposals = run_toxicity_advisory(
        _design_report(),
        model="test-model",
        api_key="top-secret",
        planner=planner,
    )
    artifact_path = tmp_path / "advisory.json"
    proposals_path = tmp_path / "proposals.json"
    write_toxicity_advisory(
        artifact=artifact,
        proposals=proposals,
        artifact_path=artifact_path,
        proposals_path=proposals_path,
    )

    proposal_rows = json.loads(proposals_path.read_text(encoding="utf-8"))
    assert captured["information_boundary"]["sample_ids_included"] is False
    assert artifact["status"] == "VALIDATED"
    assert artifact["applied_to_execution"] is True
    assert artifact["advice"]["probe_strategy"]["candidate_sizes"] == [300, 500, 750]
    assert proposals[0].source == "llm"
    assert proposal_rows[0]["proposal_id"] == "llm-q90-no-h"
    assert "source" not in proposal_rows[0]
    assert "rationale" not in proposal_rows[0]


def test_toxicity_advisory_rejects_out_of_policy_values():
    invalid = _valid_advice()
    invalid["proposals"][0]["dataset_distance_quantile"] = 0.99

    artifact, proposals = run_toxicity_advisory(
        _design_report(),
        model="test-model",
        api_key="top-secret",
        planner=lambda *_args, **_kwargs: (invalid, {"used": True}),
    )

    assert artifact["status"] == "FALLBACK"
    assert proposals == ()
    assert "unsupported dataset_distance_quantile" in artifact["validation"]["errors"][0]


def test_toxicity_advisory_rejects_out_of_policy_probe_strategy():
    invalid = _valid_advice()
    invalid["probe_strategy"]["candidate_sizes"] = [250]

    artifact, proposals = run_toxicity_advisory(
        _design_report(),
        model="test-model",
        api_key="top-secret",
        planner=lambda *_args, **_kwargs: (invalid, {"used": True}),
    )

    assert artifact["status"] == "FALLBACK"
    assert proposals == ()
    assert "unsupported toxicity probe candidate size" in artifact["validation"]["errors"][0]


def test_toxicity_advisory_reuses_content_addressed_cache(tmp_path: Path):
    calls = []

    def planner(_facts, **_kwargs):
        calls.append("called")
        return _valid_advice(), {"used": True, "model_requested": "test-model"}

    first, first_proposals = run_toxicity_advisory(
        _design_report(),
        model="test-model",
        api_key="top-secret",
        planner=planner,
        cache_dir=tmp_path / "cache",
    )
    second, second_proposals = run_toxicity_advisory(
        _design_report(),
        model="different-model-is-not-called",
        api_key="different-key-is-not-called",
        planner=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("cached advisory should avoid another API call")
        ),
        cache_dir=tmp_path / "cache",
    )

    assert calls == ["called"]
    assert first["cache_hit"] is False
    assert second["cache_hit"] is True
    assert first_proposals == second_proposals


def test_openai_toxicity_schema_avoids_unsupported_constraint_keywords():
    serialized = json.dumps(_toxicity_advice_schema())

    for keyword in ("uniqueItems", "minItems", "maxItems", "minimum"):
        assert keyword not in serialized
