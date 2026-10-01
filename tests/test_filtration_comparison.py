import pytest

from mint_scout.data.element_pairs import make_casf_protein_ligand_schema
from mint_scout.filtration import FiltrationProfile
from mint_scout.filtration_comparison import build_filtration_comparison
from mint_scout.representation import RepresentationMode, RepresentationSpec


def _representation(stop: float) -> RepresentationSpec:
    profile = FiltrationProfile(
        "distance", 0.0, stop, stop / 49.0, 50, "fixed_point_count"
    )
    return RepresentationSpec.from_schema(
        mode=RepresentationMode.DATASET_ADAPTIVE,
        schema=make_casf_protein_ligand_schema(),
        filtration_profiles={"PH": profile, "FPRC": profile},
    ).freeze()


def _oof(invariant: str, score: float, *, fold_b: int = 1) -> dict:
    return {
        "dataset_id": "toy",
        "evidence_scope": "probe",
        "sample_ids": ["a", "b"],
        "status": "COMPLETE",
        "scout": {
            "fold_assignment": {"a": 0, "b": fold_b},
            "gbt_parameter_hash": "gbt",
            "invariant_oof_predictions": {invariant: {"a": 1.0, "b": 2.0}},
            "consensus_metrics": [
                {
                    "invariants": [invariant],
                    "nominal_primary_score": score,
                    "bootstrap_ci_low": score - 0.1,
                    "bootstrap_ci_high": score + 0.1,
                    "secondary_metrics": {"RMSE": 1.0, "MAE": 0.8, "R2": 0.4},
                }
            ],
        },
    }


def test_filtration_comparison_is_same_protocol_and_small_sample_guarded():
    report = build_filtration_comparison(
        baseline_representation=_representation(19.0),
        candidate_representation=_representation(20.0),
        baseline_oof=[_oof("PH", 0.64), _oof("FPRC", 0.54)],
        candidate_oof=[_oof("PH", 0.66), _oof("FPRC", 0.53)],
        min_pcc_delta=0.01,
        min_labeled_samples=300,
    )

    assert report["protocol"]["same_folds"] is True
    assert report["representation"]["element_pair_schema_changed"] is False
    assert report["comparisons"]["PH"]["outcome"] == "CANDIDATE_MATERIALLY_BETTER"
    assert report["comparisons"]["FPRC"]["outcome"] == "CANDIDATE_MATERIALLY_WORSE"
    assert report["decision"] == {
        "engineering_evidence_only": True,
        "automatic_adoption_permitted": False,
        "recommendation": "DO_NOT_ADOPT_SMALL_SAMPLE",
        "material_improvement_count": 1,
        "material_regression_count": 1,
    }


def test_filtration_comparison_rejects_changed_fold_assignment():
    with pytest.raises(ValueError, match="fold assignments differ"):
        build_filtration_comparison(
            baseline_representation=_representation(19.0),
            candidate_representation=_representation(20.0),
            baseline_oof=[_oof("PH", 0.64)],
            candidate_oof=[_oof("PH", 0.65, fold_b=0)],
            min_pcc_delta=0.01,
            min_labeled_samples=300,
        )
