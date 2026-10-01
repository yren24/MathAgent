import json

import pytest

from mint_scout.data.element_pairs import make_casf_protein_ligand_schema
from mint_scout.filtration import FiltrationConfig, choose_distance_step, make_distance_filtration_profile
from mint_scout.representation import (
    RepresentationMode,
    RepresentationSpec,
    make_legacy_casf_representation_spec,
)


def test_distance_filtration_includes_zero_endpoint_count():
    profile = make_distance_filtration_profile(rmax=12.0)

    assert profile.start == 0.0
    assert profile.stop == 12.0
    assert profile.step == 0.1
    assert profile.num_points == 121


def test_distance_step_chooser_respects_point_cap():
    config = FiltrationConfig(max_points=200)

    assert choose_distance_step(30.0, config) == 0.2


def test_distance_step_chooser_raises_when_no_allowed_step_satisfies_cap():
    config = FiltrationConfig(allowed_steps_angstrom=(0.1, 0.2), max_points=10)

    with pytest.raises(ValueError, match="No allowed filtration step"):
        choose_distance_step(30.0, config)


def test_fixed_point_filtration_spans_rmax_with_exact_requested_size():
    config = FiltrationConfig(fixed_point_count=50)

    profile = make_distance_filtration_profile(rmax=12.25, config=config)

    assert profile.start == 0.0
    assert profile.stop == 12.25
    assert profile.num_points == 50
    assert profile.step == pytest.approx(0.25)
    assert profile.source.endswith("_fixed_point_count")


def test_distance_filtration_rounds_stop_up_to_cover_non_aligned_rmax():
    profile = make_distance_filtration_profile(rmax=12.25)

    assert profile.stop == pytest.approx(12.3)
    assert profile.num_points == 124


def test_representation_spec_hash_changes_when_filtration_changes():
    schema = make_casf_protein_ligand_schema()
    spec_a = RepresentationSpec.from_schema(
        mode=RepresentationMode.LEGACY_CASF,
        schema=schema,
        filtration_profiles={"PH": make_distance_filtration_profile(rmax=12.0)},
    )
    spec_b = RepresentationSpec.from_schema(
        mode=RepresentationMode.LEGACY_CASF,
        schema=schema,
        filtration_profiles={"PH": make_distance_filtration_profile(rmax=15.0)},
    )

    assert spec_a.spec_hash != spec_b.spec_hash


def test_freezing_spec_does_not_change_semantic_hash():
    schema = make_casf_protein_ligand_schema()
    spec = RepresentationSpec.from_schema(
        mode=RepresentationMode.LEGACY_CASF,
        schema=schema,
        filtration_profiles={"PH": make_distance_filtration_profile(rmax=15.0)},
    )

    frozen = spec.freeze()
    assert frozen.frozen is True
    assert frozen.spec_hash == spec.spec_hash


def test_unfrozen_spec_rejected_before_feature_generation():
    schema = make_casf_protein_ligand_schema()
    spec = RepresentationSpec.from_schema(
        mode=RepresentationMode.LEGACY_CASF,
        schema=schema,
        filtration_profiles={"PH": make_distance_filtration_profile(rmax=15.0)},
    )

    with pytest.raises(AssertionError):
        spec.assert_frozen()
    spec.freeze().assert_frozen()


def test_representation_spec_nested_parameters_are_immutable():
    schema = make_casf_protein_ligand_schema()
    spec = RepresentationSpec.from_schema(
        mode=RepresentationMode.LEGACY_CASF,
        schema=schema,
        filtration_profiles={"PH": make_distance_filtration_profile(rmax=15.0)},
        parameters={"adapter": {"presets": ["legacy"]}},
    ).freeze()

    with pytest.raises(TypeError):
        spec.parameters["new"] = "value"
    with pytest.raises(TypeError):
        spec.parameters["adapter"]["presets"] = ("changed",)
    assert spec.to_dict()["parameters"] == {"adapter": {"presets": ["legacy"]}}


def test_legacy_casf_representation_records_exact_audited_shapes_and_grids():
    spec = make_legacy_casf_representation_spec()

    assert spec.frozen is True
    assert spec.mode == RepresentationMode.LEGACY_CASF
    assert len(spec.pair_order) == 40
    assert spec.filtration_profiles["PH"].num_points == 150
    assert spec.filtration_profiles["PL"].num_points == 30
    assert spec.filtration_profiles["CA"].num_points == 141
    assert spec.filtration_profiles["FPRC"].num_points == 30
    assert spec.filtration_profiles["EIC"].num_points == 49


def test_representation_spec_reads_nested_design_report_and_verifies_hash(tmp_path):
    spec = make_legacy_casf_representation_spec()
    path = tmp_path / "design.json"
    path.write_text(json.dumps({"representation_spec": spec.to_dict()}), encoding="utf-8")

    loaded = RepresentationSpec.read(path)

    assert loaded.to_dict() == spec.to_dict()


def test_representation_spec_rejects_tampered_payload(tmp_path):
    payload = make_legacy_casf_representation_spec().to_dict()
    payload["parameters"]["tampered"] = True
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="hash mismatch"):
        RepresentationSpec.read(path)
