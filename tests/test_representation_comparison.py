from mint_scout.data.element_pairs import make_casf_protein_ligand_schema
from mint_scout.filtration import FiltrationProfile
from mint_scout.representation import RepresentationMode, RepresentationSpec
from mint_scout.representation_comparison import build_representation_comparison


def test_same_fold_comparison_attributes_no_material_pair_difference():
    representation = RepresentationSpec.from_schema(
        mode=RepresentationMode.DATASET_ADAPTIVE,
        schema=make_casf_protein_ligand_schema(),
        filtration_profiles={"PL": FiltrationProfile("distance", 0.0, 20.0, 0.5, 41, "toy")},
    ).freeze()
    folds = {"a": 0, "b": 1}
    report = build_representation_comparison(
        representation=representation,
        adaptive_cv={
            "task_id": "toy",
            "sample_ids": ["a", "b"],
            "result": {"selected_score": 0.79},
        },
        scout_execution={"modeling_sample_ids": ["a", "b"], "full_fold_assignment": folds},
        legacy_cv={
            "evaluation_sample_ids": ["a", "b"],
            "fold_assignment": folds,
            "metrics": {"PCC": 0.792},
            "representation": {"name": "legacy"},
        },
        legacy_fixed_test={"metrics": {"PCC": 0.836}},
        filtration_audit={
            "audit_hash": "audit",
            "invariants": {"PL": {"recommendation": "KEEP"}},
        },
        min_delta=0.005,
    )
    comparison = report["same_protocol_comparison"]
    assert comparison["material_difference"] is False
    assert report["representation_difference"]["element_pair_set_changed"] is False
    assert comparison["pcc_delta_legacy_minus_adaptive"] == 0.0020000000000000018
