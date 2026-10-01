import pytest

from mint_scout.data.element_pairs import (
    SupportThresholds,
    build_dataset_adaptive_schema,
    make_casf_protein_ligand_schema,
    make_toxicity_schema,
)


def test_toxicity_schema_exact_30_pairs():
    schema = make_toxicity_schema()
    assert schema.expected_pair_count == 30
    assert schema.pair_names == (
        "H-C",
        "H-N",
        "H-O",
        "H-F",
        "H-P",
        "H-S",
        "H-Cl",
        "H-Br",
        "H-I",
        "C-N",
        "C-O",
        "C-F",
        "C-P",
        "C-S",
        "C-Cl",
        "C-Br",
        "C-I",
        "N-O",
        "N-F",
        "N-P",
        "N-S",
        "N-Cl",
        "N-Br",
        "N-I",
        "O-F",
        "O-P",
        "O-S",
        "O-Cl",
        "O-Br",
        "O-I",
    )


def test_casf_schema_uses_40_role_aware_pairs():
    schema = make_casf_protein_ligand_schema()
    assert schema.expected_pair_count == 40
    assert schema.role_aware is True
    assert schema.pair_names[0] == "protein:C|ligand:C"
    assert schema.pair_names[-1] == "protein:S|ligand:H"


def test_adaptive_protein_ligand_pairs_are_role_aware_and_support_filtered():
    build = build_dataset_adaptive_schema(
        system_type="protein_ligand",
        elements_by_role={
            "protein": {
                "s1": ["C", "N"],
                "s2": ["C"],
                "s3": ["N"],
            },
            "ligand": {
                "s1": ["N"],
                "s2": ["C"],
                "s3": ["C"],
            },
        },
        thresholds=SupportThresholds(
            min_element_support_fraction=0.0,
            min_element_support_samples=1,
            min_pair_support_fraction=0.0,
            min_pair_support_samples=1,
            max_element_pair_channels=50,
        ),
    )

    assert build.schema.role_aware is True
    assert ("C", "N") in build.schema.pair_order
    assert ("N", "C") in build.schema.pair_order
    assert "protein:C|ligand:N" in build.schema.pair_names
    assert "protein:N|ligand:C" in build.schema.pair_names


def test_adaptive_small_molecule_pairs_are_unordered_without_self_pairs():
    build = build_dataset_adaptive_schema(
        system_type="small_molecule",
        elements_by_role={
            "molecule": {
                "s1": ["C", "N", "O"],
                "s2": ["C", "N"],
            }
        },
        thresholds=SupportThresholds(
            min_element_support_fraction=0.0,
            min_element_support_samples=1,
            min_pair_support_fraction=0.0,
            min_pair_support_samples=1,
            max_element_pair_channels=50,
            small_molecule_self_pairs=False,
        ),
    )

    assert build.schema.role_aware is False
    assert ("C", "C") not in build.schema.pair_order
    assert ("C", "N") in build.schema.pair_order
    assert ("N", "C") not in build.schema.pair_order


def test_adaptive_pair_cap_retains_exactly_top_50_with_deterministic_ties():
    protein_elements = [f"P{i:02d}" for i in range(4)]
    ligand_elements = [f"L{i:02d}" for i in range(19)]
    build = build_dataset_adaptive_schema(
        system_type="protein_ligand",
        elements_by_role={
            "protein": {"s1": protein_elements},
            "ligand": {"s1": ligand_elements},
        },
        thresholds=SupportThresholds(
            min_element_support_fraction=0.0,
            min_element_support_samples=1,
            min_pair_support_fraction=0.0,
            min_pair_support_samples=1,
            max_element_pair_channels=50,
        ),
    )

    assert len(build.pair_support) == 76
    assert build.schema.expected_pair_count == 50
    assert build.cap_applied is True
    assert build.schema.pair_order == tuple(sorted(build.schema.pair_order))
    assert sum(record.selected for record in build.pair_support) == 50
    assert any("PAIR_CAP_APPLIED" in warning for warning in build.warnings)


def test_adaptive_pair_filtering_uses_support_not_targets():
    build = build_dataset_adaptive_schema(
        system_type="small_molecule",
        elements_by_role={
            "molecule": {
                "low_label_sample": ["C", "N"],
                "high_label_sample": ["C"],
            }
        },
        thresholds=SupportThresholds(
            min_element_support_fraction=0.0,
            min_element_support_samples=1,
            min_pair_support_fraction=0.0,
            min_pair_support_samples=2,
            max_element_pair_channels=50,
        ),
    )

    assert ("C", "N") not in build.schema.pair_order
    assert build.warnings == ("No element pairs survived support filtering.",)


def test_adaptive_protein_ligand_requires_aligned_role_sample_ids():
    with pytest.raises(ValueError, match="aligned modeling samples"):
        build_dataset_adaptive_schema(
            system_type="protein_ligand",
            elements_by_role={
                "protein": {"s1": ["C"], "s2": ["N"]},
                "ligand": {"s1": ["C"]},
            },
            thresholds=SupportThresholds(
                min_element_support_samples=1,
                min_pair_support_samples=1,
            ),
        )
