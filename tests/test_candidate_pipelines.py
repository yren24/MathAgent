from mint_scout.config import load_yaml
from mint_scout.search.candidates import (
    assert_model_conditioned_candidates,
    enabled_candidate_pipelines,
    load_candidate_pipelines,
)


def test_casf2016_search_uses_model_conditioned_candidate_pipelines():
    config = load_yaml("configs/tasks/casf2016.yaml")
    candidates = enabled_candidate_pipelines(config)

    assert config["search"]["selection_unit"] == "candidate_pipeline"
    assert config["search"]["model_conditioned"] is True
    assert {candidate.model for candidate in candidates} == {"legacy_gbt"}
    assert {candidate.model_family for candidate in candidates} == {"gradient_boosting"}
    assert {candidate.input_contract for candidate in candidates} == {"fixed_vector"}
    assert ("PL", "PH") in {candidate.features for candidate in candidates}
    assert_model_conditioned_candidates(candidates)


def test_candidate_pipeline_names_must_be_unique():
    config = {
        "fidelity": {"fractions": [0.1]},
        "search": {
            "candidates": [
                {
                    "name": "dup",
                    "features": ["PL"],
                    "feature_schema": "casf_protein_ligand_40_v1",
                    "input_contract": "fixed_vector",
                    "normalize": "legacy_standard_scaler",
                    "training_protocol": "train_set_cv_5fold",
                    "model": {"name": "legacy_gbt", "family": "gradient_boosting"},
                },
                {
                    "name": "dup",
                    "features": ["PH"],
                    "feature_schema": "casf_protein_ligand_40_v1",
                    "input_contract": "fixed_vector",
                    "normalize": "legacy_standard_scaler",
                    "training_protocol": "train_set_cv_5fold",
                    "model": {"name": "legacy_gbt", "family": "gradient_boosting"},
                },
            ]
        },
    }

    try:
        load_candidate_pipelines(config)
    except ValueError as exc:
        assert "unique" in str(exc)
    else:
        raise AssertionError("Expected duplicate candidate names to be rejected")
