from __future__ import annotations

from pathlib import Path

import pytest

from mint_scout.toxicity.design import (
    ToxicityAdaptiveProposal,
    design_toxicity_candidates,
)


def _write_mol2(path: Path, elements: tuple[str, ...]) -> None:
    atom_lines = []
    for index, element in enumerate(elements, start=1):
        atom_lines.append(
            f"{index:7d} {element}{index:<4d} {(index - 1) * 3.0:9.4f} "
            f"{(index % 2) * 2.5:9.4f} 0.0000 {element}.3 1 MOL 0.0000"
        )
    path.write_text(
        "@<TRIPOS>MOLECULE\n"
        f"{path.stem}\n"
        f"{len(elements)} 0 0 0 0\n"
        "SMALL\nNO_CHARGES\n\n"
        "@<TRIPOS>ATOM\n"
        + "\n".join(atom_lines)
        + "\n@<TRIPOS>BOND\n",
        encoding="utf-8",
    )


def _training_paths(tmp_path: Path) -> dict[str, Path]:
    paths = {}
    for index in range(10):
        path = tmp_path / f"train-{index}.mol2"
        elements = ("H", "C", "N", "O", "Si") if index == 0 else ("H", "C", "N", "O")
        _write_mol2(path, elements)
        paths[path.stem] = path
    return paths


def test_toxicity_design_keeps_legacy_and_builds_three_explainable_adaptive_ablations(
    tmp_path: Path,
):
    result = design_toxicity_candidates(training_paths=_training_paths(tmp_path))

    assert result.training_sample_count == 10
    assert result.out_of_schema_element_presence == {"Si": 1}
    assert [candidate.ablation for candidate in result.candidates] == [
        "legacy_baseline",
        "pairs-only",
        "grid-only",
        "pairs-and-grid",
    ]
    assert len(result.candidates[0].representation_spec.pair_order) == 30
    adaptive_pairs = result.candidates[1].representation_spec.pair_order
    assert adaptive_pairs
    assert all(left != right for left, right in adaptive_pairs)
    assert not any((right, left) in adaptive_pairs for left, right in adaptive_pairs)
    assert all("Si" not in pair for pair in adaptive_pairs)
    assert result.to_dict()["scientific_controls"]["primary_metric"] == "PCC2"


def test_toxicity_adaptive_grid_has_50_train_only_points(tmp_path: Path):
    result = design_toxicity_candidates(training_paths=_training_paths(tmp_path))
    adaptive = next(
        candidate
        for candidate in result.candidates
        if candidate.ablation == "pairs-and-grid"
    )
    profile = adaptive.representation_spec.filtration_profiles["PL"]

    assert profile.num_points == 50
    assert profile.start == 0.0
    assert profile.stop > 0.0
    assert profile.step == pytest.approx(profile.stop / 49)
    assert adaptive.representation_spec.parameters["training_fingerprint"]


def test_llm_can_add_bounded_candidates_but_cannot_remove_baseline(tmp_path: Path):
    proposal = ToxicityAdaptiveProposal(
        proposal_id="llm-q90-no-h",
        source="llm",
        include_hydrogen=False,
        min_element_support_fraction=0.001,
        min_pair_support_fraction=0.001,
        min_support_samples=5,
        max_pair_channels=45,
        local_distance_quantile=0.90,
        dataset_distance_quantile=0.90,
        margin_factor=1.0,
    )

    result = design_toxicity_candidates(
        training_paths=_training_paths(tmp_path),
        llm_proposals=(proposal,),
    )

    assert result.candidates[0].candidate_id == "legacy-30-pairs-legacy-grid"
    assert result.candidates[0].source == "deterministic_baseline"
    assert any(candidate.source == "llm" for candidate in result.candidates)
    llm_pairs = next(
        candidate.representation_spec.pair_order
        for candidate in result.candidates
        if candidate.source == "llm" and candidate.ablation == "pairs-only"
    )
    assert all("H" not in pair for pair in llm_pairs)


def test_llm_proposal_rejects_unbounded_scientific_values():
    with pytest.raises(ValueError, match="unsupported dataset_distance_quantile"):
        ToxicityAdaptiveProposal(
            proposal_id="bad-q99",
            source="llm",
            dataset_distance_quantile=0.99,
        )


def test_toxicity_design_allows_at_most_two_llm_proposals(tmp_path: Path):
    proposals = tuple(
        ToxicityAdaptiveProposal(
            proposal_id=f"llm-{index}",
            source="llm",
        )
        for index in range(3)
    )

    with pytest.raises(ValueError, match="at most two LLM"):
        design_toxicity_candidates(
            training_paths=_training_paths(tmp_path),
            llm_proposals=proposals,
        )
