from __future__ import annotations

import json

import pytest

from mint_scout.data.element_pairs import make_toxicity_schema
from mint_scout.filtration import FiltrationProfile
from mint_scout.representation import (
    RepresentationMode,
    RepresentationSpec,
    make_legacy_toxicity_representation_spec,
)
from mint_scout.toxicity.feature_plan import build_toxicity_feature_plan
from mint_scout.toxicity.selection_contract import ToxicitySelectionContract


def test_feature_plan_deduplicates_identical_method_inputs(tmp_path):
    legacy = make_legacy_toxicity_representation_spec()
    profiles = dict(legacy.filtration_profiles)
    profiles["PH"] = FiltrationProfile(
        "distance", 0.0, 11.8, 0.2, 60, "test_ph_only"
    )
    ph_only = RepresentationSpec.from_schema(
        mode=RepresentationMode.DATASET_ADAPTIVE,
        schema=make_toxicity_schema(),
        filtration_profiles=profiles,
        parameters={"system_type": "small_molecule"},
    ).freeze()
    representation_dir = tmp_path / "representations"
    representation_dir.mkdir()
    legacy.write(representation_dir / "legacy.json")
    ph_only.write(representation_dir / "ph-only.json")
    design = {
        "candidates": [
            {
                "candidate_id": "legacy",
                "representation_hash": legacy.spec_hash,
            },
            {
                "candidate_id": "ph-only",
                "representation_hash": ph_only.spec_hash,
            },
        ]
    }
    design_path = tmp_path / "design.json"
    design_path.write_text(json.dumps(design), encoding="utf-8")

    plan = build_toxicity_feature_plan(
        design_report=design_path,
        representation_dir=representation_dir,
        manifest_dir=tmp_path / "manifests",
    )

    assert plan["naive_task_count"] == 10
    assert plan["unique_task_count"] == 6
    assert plan["deduplicated_task_count"] == 4
    assert (
        plan["candidate_feature_signatures"]["legacy"]["PL"]
        == plan["candidate_feature_signatures"]["ph-only"]["PL"]
    )
    assert (
        plan["candidate_feature_signatures"]["legacy"]["PH"]
        != plan["candidate_feature_signatures"]["ph-only"]["PH"]
    )

    restricted = build_toxicity_feature_plan(
        design_report=design_path,
        representation_dir=representation_dir,
        manifest_dir=tmp_path / "restricted-manifests",
        selection=ToxicitySelectionContract(
            requested_invariants=("PL", "EIC"), primary_metric="R2"
        ),
    )
    assert restricted["naive_task_count"] == 4
    assert restricted["unique_task_count"] == 2
    assert restricted["selection_contract"]["primary_metric"] == "R2"
    assert all(
        set(signatures) == {"PL", "EIC"}
        for signatures in restricted["candidate_feature_signatures"].values()
    )
    assert {task["invariant"] for task in restricted["tasks"]} == {"PL", "EIC"}

    design["scientific_controls"] = {"primary_metric": "PCC2"}
    design_path.write_text(json.dumps(design), encoding="utf-8")
    with pytest.raises(ValueError, match="differs from requested metric"):
        build_toxicity_feature_plan(
            design_report=design_path,
            representation_dir=representation_dir,
            manifest_dir=tmp_path / "invalid-manifests",
            selection=ToxicitySelectionContract(primary_metric="R2"),
        )
