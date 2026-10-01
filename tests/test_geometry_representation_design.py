from __future__ import annotations

import numpy as np
import pytest

from mint_scout.data.element_pairs import SupportThresholds
from mint_scout.data.geometry import (
    AtomCloud,
    load_atom_cloud,
    profile_protein_ligand_geometry,
    profile_small_molecule_geometry,
)
from mint_scout.filtration import FiltrationConfig
from mint_scout.representation import RepresentationMode
from mint_scout.representation_design import (
    HydrogenPolicyConfig,
    MetalAwarenessConfig,
    RepresentationDesignConfig,
    design_adaptive_representation,
)


def _write_complex(tmp_path, sample_id: str, distance: float):
    protein = tmp_path / f"{sample_id}_protein.pdb"
    ligand = tmp_path / f"{sample_id}_ligand.mol2"
    protein.write_text(
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 20.00           C\n",
        encoding="utf-8",
    )
    ligand.write_text(
        "@<TRIPOS>MOLECULE\nligand\n@<TRIPOS>ATOM\n"
        f"1 N1 {distance:.3f} 0.000 0.000 N.3 1 LIG 0.0\n",
        encoding="utf-8",
    )
    return {"protein": protein, "ligand": ligand}


def test_load_atom_cloud_parses_pdb_and_mol2_coordinates(tmp_path):
    paths = _write_complex(tmp_path, "s1", 3.0)

    protein = load_atom_cloud(paths["protein"])
    ligand = load_atom_cloud(paths["ligand"])

    assert protein.elements == ("C",)
    assert ligand.elements == ("N",)
    assert np.allclose(protein.coordinates[0], [0.0, 0.0, 0.0])
    assert np.allclose(ligand.coordinates[0], [3.0, 0.0, 0.0])


def test_protein_ligand_geometry_uses_local_then_dataset_quantiles():
    clouds = {
        "s1": {
            "protein": AtomCloud(("C",), np.asarray([[0.0, 0.0, 0.0]])),
            "ligand": AtomCloud(("N",), np.asarray([[3.0, 0.0, 0.0]])),
        },
        "s2": {
            "protein": AtomCloud(("C",), np.asarray([[0.0, 0.0, 0.0]])),
            "ligand": AtomCloud(("N",), np.asarray([[5.0, 0.0, 0.0]])),
        },
    }
    profile = profile_protein_ligand_geometry(
        clouds_by_sample=clouds,
        retained_pairs=(("C", "N"),),
        config=FiltrationConfig(
            local_distance_quantile=0.5,
            dataset_distance_quantile=0.5,
            margin_factor=1.0,
        ),
    )

    assert profile.max_filtration_angstrom == 4.0
    assert profile.local_summary_count == 2
    assert profile.pair_observation_counts == {"protein:C|ligand:N": 2}


def test_small_molecule_geometry_gives_each_molecule_one_summary():
    clouds = {
        "m1": {"molecule": AtomCloud(("C", "N"), np.asarray([[0, 0, 0], [2, 0, 0]]))},
        "m2": {"molecule": AtomCloud(("C", "N"), np.asarray([[0, 0, 0], [4, 0, 0]]))},
    }
    profile = profile_small_molecule_geometry(
        clouds_by_sample=clouds,
        retained_pairs=(("C", "N"),),
        config=FiltrationConfig(
            local_distance_quantile=0.5,
            dataset_distance_quantile=0.5,
            margin_factor=1.0,
        ),
    )

    assert profile.max_filtration_angstrom == 3.0
    assert profile.local_summary_count == 2


def test_adaptive_design_freezes_support_geometry_and_common_distance_grid(tmp_path):
    paths = {
        "s1": _write_complex(tmp_path, "s1", 3.0),
        "s2": _write_complex(tmp_path, "s2", 5.0),
    }
    config = RepresentationDesignConfig(
        support=SupportThresholds(
            min_element_support_samples=1,
            min_element_support_fraction=0.0,
            min_pair_support_samples=1,
            min_pair_support_fraction=0.0,
        ),
        filtration=FiltrationConfig(
            local_distance_quantile=0.5,
            dataset_distance_quantile=0.5,
            margin_factor=1.0,
        ),
    )

    result = design_adaptive_representation(
        system_type="protein_ligand",
        role_paths_by_sample=paths,
        config=config,
    )
    spec = result.representation_spec

    assert spec.mode == RepresentationMode.DATASET_ADAPTIVE
    assert spec.frozen is True
    assert spec.pair_order == (("C", "N"),)
    assert {spec.filtration_profiles[name] for name in ("PH", "PL", "CA", "FPRC")} == {
        spec.filtration_profiles["PH"]
    }
    assert spec.filtration_profiles["PH"].stop == 4.0
    assert spec.filtration_profiles["EIC"].scale_kind == "tau"
    assert spec.parameters["modeling_dataset_fingerprint"]
    assert spec.parameters["pair_cap_triggered"] is False


def test_adaptive_design_hash_changes_when_modeling_coordinates_change(tmp_path):
    config = RepresentationDesignConfig(
        support=SupportThresholds(
            min_element_support_samples=1,
            min_element_support_fraction=0.0,
            min_pair_support_samples=1,
            min_pair_support_fraction=0.0,
        )
    )
    first = design_adaptive_representation(
        system_type="protein_ligand",
        role_paths_by_sample={"s1": _write_complex(tmp_path, "first", 3.0)},
        config=config,
    )
    second = design_adaptive_representation(
        system_type="protein_ligand",
        role_paths_by_sample={"s1": _write_complex(tmp_path, "second", 5.0)},
        config=config,
    )

    assert first.representation_spec.spec_hash != second.representation_spec.spec_hash


def test_adaptive_design_records_but_excludes_elements_unsupported_by_adapter(tmp_path):
    paths = _write_complex(tmp_path, "s1", 3.0)
    paths["protein"].write_text(
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 20.00           C\n"
        "HETATM    2 P    PO4 A   2       1.000   0.000   0.000  1.00 20.00           P\n",
        encoding="utf-8",
    )
    config = RepresentationDesignConfig(
        support=SupportThresholds(
            min_element_support_samples=1,
            min_element_support_fraction=0.0,
            min_pair_support_samples=1,
            min_pair_support_fraction=0.0,
        ),
        adapter_supported_elements={"protein": ("C",), "ligand": ("N",)},
    )

    result = design_adaptive_representation(
        system_type="protein_ligand",
        role_paths_by_sample={"s1": paths},
        config=config,
    )
    compatibility = result.representation_spec.parameters["adapter_compatibility"]

    assert result.representation_spec.pair_order == (("C", "N"),)
    assert compatibility["unsupported_observed"]["protein"] == (
        {"element": "P", "sample_presence": 1, "tolerated": False},
    )
    assert "ADAPTER_UNSUPPORTED_ELEMENTS_OBSERVED_PROTEIN" in result.representation_spec.parameters[
        "anomalies"
    ]


def test_hydrogen_auto_excludes_explicit_h_when_task_is_not_relevant(tmp_path):
    paths = _write_complex(tmp_path, "h-excluded", 3.0)
    paths["protein"].write_text(
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 20.00           C\n"
        "ATOM      2  H   ALA A   1       1.000   0.000   0.000  1.00 20.00           H\n",
        encoding="utf-8",
    )
    config = RepresentationDesignConfig(
        support=SupportThresholds(
            min_element_support_samples=1,
            min_element_support_fraction=0.0,
            min_pair_support_samples=1,
            min_pair_support_fraction=0.0,
        ),
        adapter_supported_elements={"protein": ("C",), "ligand": ("N",)},
        hydrogen_policy=HydrogenPolicyConfig(
            mode="auto", task_relevance="unlikely_relevant"
        ),
    )

    result = design_adaptive_representation(
        system_type="protein_ligand",
        role_paths_by_sample={"s1": paths},
        config=config,
    )

    audit = result.representation_spec.parameters["hydrogen_policy"]
    assert result.representation_spec.pair_order == (("C", "N"),)
    assert audit["resolved_mode"] == "exclude"
    assert audit["roles"]["protein"]["sample_presence"] == 1
    assert result.representation_spec.parameters["adapter_compatibility"][
        "unsupported_observed"
    ]["protein"] == ()


def test_hydrogen_auto_blocks_relevant_task_when_adapter_cannot_use_h(tmp_path):
    paths = _write_complex(tmp_path, "h-blocked", 3.0)
    paths["protein"].write_text(
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 20.00           C\n"
        "ATOM      2  H   ALA A   1       1.000   0.000   0.000  1.00 20.00           H\n",
        encoding="utf-8",
    )
    paths["ligand"].write_text(
        "@<TRIPOS>MOLECULE\nligand\n@<TRIPOS>ATOM\n"
        "1 N1 3.000 0.000 0.000 N.3 1 LIG 0.0\n"
        "2 H1 4.000 0.000 0.000 H 1 LIG 0.0\n",
        encoding="utf-8",
    )
    config = RepresentationDesignConfig(
        support=SupportThresholds(
            min_element_support_samples=1,
            min_element_support_fraction=0.0,
            min_pair_support_samples=1,
            min_pair_support_fraction=0.0,
        ),
        adapter_supported_elements={"protein": ("C",), "ligand": ("N", "H")},
        hydrogen_policy=HydrogenPolicyConfig(
            mode="auto",
            task_relevance="likely_relevant",
            min_explicit_sample_fraction=1.0,
        ),
    )

    with pytest.raises(ValueError, match="does not support explicit H for role=protein"):
        design_adaptive_representation(
            system_type="protein_ligand",
            role_paths_by_sample={"s1": paths},
            config=config,
        )


def test_metal_audit_marks_proximal_supported_candidate_as_adapter_unavailable(tmp_path):
    paths = _write_complex(tmp_path, "metal", 3.0)
    paths["protein"].write_text(
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 20.00           C\n"
        "HETATM    2 ZN   ZN  A   2       2.000   0.000   0.000  1.00 20.00          Zn\n",
        encoding="utf-8",
    )
    config = RepresentationDesignConfig(
        support=SupportThresholds(
            min_element_support_samples=1,
            min_element_support_fraction=0.0,
            min_pair_support_samples=1,
            min_pair_support_fraction=0.0,
        ),
        adapter_supported_elements={"protein": ("C",), "ligand": ("N",)},
        tolerated_unsupported_elements={"protein": ("Zn",)},
        metal_awareness=MetalAwarenessConfig(
            protein_elements=("Zn",),
            min_sample_fraction=1.0,
            min_samples=1,
            max_ligand_distance_angstrom=2.0,
            min_proximal_samples=1,
        ),
    )

    result = design_adaptive_representation(
        system_type="protein_ligand",
        role_paths_by_sample={"s1": paths},
        config=config,
    )

    metal = result.representation_spec.parameters["metal_awareness"]["elements"][0]
    assert metal["element"] == "Zn"
    assert metal["eligible"] is True
    assert metal["status"] == "ELIGIBLE_ADAPTER_UNSUPPORTED"
    assert result.representation_spec.parameters["adapter_compatibility"]["anomalies"] == ()


def test_supported_metal_enters_pairs_only_after_presence_and_proximity_gates(tmp_path):
    paths = _write_complex(tmp_path, "metal-supported", 3.0)
    paths["protein"].write_text(
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 20.00           C\n"
        "HETATM    2 ZN   ZN  A   2       2.000   0.000   0.000  1.00 20.00          Zn\n",
        encoding="utf-8",
    )
    config = RepresentationDesignConfig(
        support=SupportThresholds(
            min_element_support_samples=1,
            min_element_support_fraction=0.0,
            min_pair_support_samples=1,
            min_pair_support_fraction=0.0,
        ),
        adapter_supported_elements={"protein": ("C", "Zn"), "ligand": ("N",)},
        metal_awareness=MetalAwarenessConfig(
            protein_elements=("Zn",),
            min_sample_fraction=1.0,
            min_samples=1,
            max_ligand_distance_angstrom=2.0,
            min_proximal_samples=1,
        ),
    )

    result = design_adaptive_representation(
        system_type="protein_ligand",
        role_paths_by_sample={"s1": paths},
        config=config,
    )

    assert result.representation_spec.pair_order == (("C", "N"), ("Zn", "N"))
    metal = result.representation_spec.parameters["metal_awareness"]["elements"][0]
    assert metal["status"] == "ELIGIBLE"
