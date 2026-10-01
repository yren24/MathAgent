from __future__ import annotations

import json

import pytest

from mint_scout.invariants.manifest import stable_hash
from mint_scout.representation import make_legacy_toxicity_representation_spec
from mint_scout.filtration import FiltrationProfile
from mint_scout.toxicity.feature_plan import build_toxicity_feature_plan
from mint_scout.toxicity.filtration_repair import (
    ToxicityFiltrationRepairPolicy,
    build_toxicity_filtration_repair,
    repair_filtration_profile,
)
from mint_scout.toxicity.legacy import TOXICITY_INVARIANTS
from mint_scout.toxicity.selection_contract import ToxicitySelectionContract


def _profile():
    return FiltrationProfile("distance", 0.0, 9.8, 0.2, 50, "test")


def test_extend_keeps_point_count_and_expands_range():
    repaired = repair_filtration_profile(
        _profile(),
        recommendation="EXTEND",
        last_effective_point_index=49,
        policy=ToxicityFiltrationRepairPolicy(extension_factor=1.25),
        method="PH",
    )

    assert repaired.num_points == 50
    assert repaired.stop == pytest.approx(12.25)
    assert repaired.step == pytest.approx(0.25)


def test_shorten_uses_last_effective_point_with_buffer():
    repaired = repair_filtration_profile(
        _profile(),
        recommendation="SHORTEN",
        last_effective_point_index=30,
        policy=ToxicityFiltrationRepairPolicy(shorten_buffer_points=3),
        method="CA",
    )

    assert repaired.num_points == 50
    assert repaired.stop == pytest.approx(6.6)
    assert repaired.step == pytest.approx(6.6 / 49)


def test_keep_returns_original_profile():
    profile = _profile()
    assert (
        repair_filtration_profile(
            profile,
            recommendation="KEEP",
            last_effective_point_index=None,
            policy=ToxicityFiltrationRepairPolicy(),
            method="PL",
        )
        is profile
    )


def test_tau_profile_cannot_be_automatically_extended():
    profile = FiltrationProfile("tau", 0.2, 5.0, 0.2, 25, "legacy")
    with pytest.raises(ValueError, match="distance filtration only"):
        repair_filtration_profile(
            profile,
            recommendation="EXTEND",
            last_effective_point_index=24,
            policy=ToxicityFiltrationRepairPolicy(),
            method="EIC",
        )


@pytest.mark.parametrize("methods", [TOXICITY_INVARIANTS, ("PL", "EIC")])
def test_build_repair_passthrough_when_all_methods_keep(tmp_path, methods):
    spec = make_legacy_toxicity_representation_spec()
    representation_dir = tmp_path / "representations"
    representation_dir.mkdir()
    spec_path = representation_dir / "legacy.json"
    spec.write(spec_path)
    design_path = tmp_path / "design.json"
    design_path.write_text(
        json.dumps(
            {
                "candidates": [
                    {
                        "candidate_id": "legacy",
                        "representation_hash": spec.spec_hash,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    plan = build_toxicity_feature_plan(
        design_report=design_path,
        representation_dir=representation_dir,
        manifest_dir=tmp_path / "probe_manifests",
        selection=ToxicitySelectionContract(requested_invariants=methods),
    )
    plan_path = tmp_path / "feature_plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    qc = {
        "plan_hash": plan["plan_hash"],
        "test_labels_used": False,
        "candidates": [{"candidate_id": "legacy", "hard_gate_passed": True}],
        "tasks": [
            {
                "feature_signature": signature,
                "hard_gate_passed": True,
                "filtration": {
                    "recommendation": "KEEP",
                    "last_effective_point_index": None,
                },
            }
            for signature in plan["candidate_feature_signatures"]["legacy"].values()
        ],
    }
    qc["qc_hash"] = stable_hash(qc)
    qc_path = tmp_path / "feature_qc.json"
    qc_path.write_text(json.dumps(qc), encoding="utf-8")
    selection = {
        "feature_plan_hash": plan["plan_hash"],
        "feature_qc_hash": qc["qc_hash"],
        "test_labels_used": False,
        "selected_candidate_id": "legacy",
        "selected_representation": str(spec_path),
        "selected_method_feature_signatures": plan["candidate_feature_signatures"][
            "legacy"
        ],
    }
    selection["selection_hash"] = stable_hash(selection)
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(json.dumps(selection), encoding="utf-8")

    result = build_toxicity_filtration_repair(
        representation_selection_path=selection_path,
        feature_plan_path=plan_path,
        feature_qc_path=qc_path,
        output_dir=tmp_path / "repair",
    )

    assert result["status"] == "NOT_REQUIRED"
    assert result["changed_methods"] == []
    assert result["reused_methods"] == list(methods)
    repair_plan = json.loads(
        (tmp_path / "repair" / "toxicity_repair_feature_plan.json").read_text(
            encoding="utf-8"
        )
    )
    assert repair_plan["candidate_count"] == 1
    assert repair_plan["deduplicated_task_count"] == len(methods)
    assert repair_plan["selection_contract"]["requested_invariants"] == list(methods)
    assert all(task["execute"] is False for task in repair_plan["tasks"])
