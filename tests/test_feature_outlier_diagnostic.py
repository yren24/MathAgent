from __future__ import annotations

import json

import numpy as np
import pytest

from mint_scout.data.element_pairs import ElementPairSchema
from mint_scout.feature_outlier_diagnostic import (
    FPRC_COMPONENT_NAMES,
    flagged_samples_from_qc,
    run_feature_outlier_diagnostic,
)
from mint_scout.filtration import FiltrationProfile
from mint_scout.representation import RepresentationMode, RepresentationSpec


def _representation() -> RepresentationSpec:
    schema = ElementPairSchema(
        schema_id="toy-pairs",
        system_type="protein_ligand",
        left_role="protein",
        right_role="ligand",
        left_elements=("C", "O"),
        right_elements=("C", "N"),
        global_element_order=None,
        pair_generation_rule="test",
        role_aware=True,
        exclude_self_pairs=False,
        deduplicate_unordered_pairs=False,
        pair_order=(("C", "C"), ("O", "N")),
        expected_pair_count=2,
    )
    return RepresentationSpec.from_schema(
        mode=RepresentationMode.DATASET_ADAPTIVE,
        schema=schema,
        filtration_profiles={
            "FPRC": FiltrationProfile("distance", 0.0, 1.0, 1.0, 2, "test")
        },
    ).freeze()


def _manifest(tmp_path, representation, arrays):
    feature_root = (
        tmp_path
        / "features"
        / f"repr-{representation.spec_hash}"
        / "FPRC"
    )
    feature_root.mkdir(parents=True)
    rows = []
    for sample_id, array in arrays.items():
        path = feature_root / f"{sample_id}.npy"
        np.save(path, array)
        rows.append(
            {
                "sample_id": sample_id,
                "invariant": "FPRC",
                "status": "computed",
                "output_path": str(path),
            }
        )
    path = tmp_path / "manifest.jsonl"
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    return path


def test_diagnostic_locates_pair_filtration_and_fprc_component(tmp_path):
    representation = _representation()
    arrays = {}
    for index, sample_id in enumerate(("a", "b", "flagged"), start=1):
        array = np.full((2, 2, 20), float(index), dtype=np.float32)
        if sample_id == "flagged":
            array[1, 1, 19] = 1000.0
        arrays[sample_id] = array
    manifest = _manifest(tmp_path, representation, arrays)

    result = run_feature_outlier_diagnostic(
        representation=representation,
        sample_ids=("a", "b", "flagged"),
        invariant="FPRC",
        manifest_path=manifest,
        flagged_sample_ids=("flagged",),
        top_k=2,
        selection_hash="selection",
        sample_metadata={
            "a": {"structure_size": 10, "size_bin": 0},
            "b": {"structure_size": 20, "size_bin": 1},
            "flagged": {"structure_size": 100, "size_bin": 4},
        },
    )

    assert result["automatic_sample_removal"] is False
    diagnostic = result["diagnostics"][0]
    maximum = diagnostic["maximum_absolute_coordinate"]
    assert maximum["filtration_value"] == 1.0
    assert maximum["element_pair"] == "protein:O|ligand:N"
    assert maximum["component"] == "edge_curvature_third_absolute_deviation_sum"
    assert (
        diagnostic["axes"]["element_pair"]["top_contributors"][0]["label"]
        == "protein:O|ligand:N"
    )
    assert diagnostic["axes"]["component"]["top_contributors"][0]["label"] == FPRC_COMPONENT_NAMES[19]
    assert diagnostic["selection_structure_context"]["descending_size_rank"] == 1
    assert result["population_structure_context"]["maximum"] == 100.0


def test_diagnostic_rejects_sample_outside_frozen_population(tmp_path):
    representation = _representation()
    manifest = _manifest(
        tmp_path,
        representation,
        {"a": np.ones((2, 2, 20), dtype=np.float32)},
    )

    with pytest.raises(ValueError, match="outside the selected population"):
        run_feature_outlier_diagnostic(
            representation=representation,
            sample_ids=("a",),
            invariant="FPRC",
            manifest_path=manifest,
            flagged_sample_ids=("different",),
        )


def test_flagged_samples_are_selected_from_matching_qc_reason():
    report = {
        "issues": [
            {"invariant": "FPRC", "sample_id": "a", "issue": "robust_outlier"},
            {"invariant": "PH", "sample_id": "b", "issue": "robust_outlier"},
            {"invariant": "FPRC", "sample_id": "c", "issue": "constant"},
        ]
    }

    assert flagged_samples_from_qc(report, "FPRC") == ("a",)
