from __future__ import annotations

import numpy as np

from mint_scout.data.element_pairs import SupportThresholds
from mint_scout.execution import ExecutionConfig, ProgressiveExecutionEngine
from mint_scout.execution.provider import InMemoryFeatureProvider
from mint_scout.models.gbt import GBTConfig
from mint_scout.representation_design import RepresentationDesignConfig, design_adaptive_representation
from mint_scout.schemas import EvaluationMode, EvaluationPlan
from mint_scout.scout.pipeline import ScoutConfig, run_scout
from mint_scout.scout.sampling import ProbeSamplingConfig
from mint_scout.scout.stability import BootstrapConfig


def test_adaptive_design_scout_artifact_and_progressive_execution_end_to_end(tmp_path):
    sample_ids = tuple(f"s{index:02d}" for index in range(20))
    role_paths = {}
    for index, sample_id in enumerate(sample_ids):
        protein = tmp_path / f"{sample_id}.pdb"
        ligand = tmp_path / f"{sample_id}.mol2"
        protein.write_text(
            "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 20.00           C\n",
            encoding="utf-8",
        )
        ligand.write_text(
            "@<TRIPOS>MOLECULE\nligand\n@<TRIPOS>ATOM\n"
            f"1 N1 {2.0 + index / 20.0:.3f} 0.000 0.000 N.3 1 LIG 0.0\n",
            encoding="utf-8",
        )
        role_paths[sample_id] = {"protein": protein, "ligand": ligand}

    design = design_adaptive_representation(
        system_type="protein_ligand",
        role_paths_by_sample=role_paths,
        config=RepresentationDesignConfig(
            support=SupportThresholds(
                min_element_support_samples=1,
                min_element_support_fraction=0.0,
                min_pair_support_samples=1,
                min_pair_support_fraction=0.0,
            )
        ),
    )
    x = np.linspace(-2.0, 2.0, len(sample_ids))
    targets = 2.5 * x + 0.1 * np.sin(x)
    features = {
        "PH": np.column_stack((x, x**2)),
        "PL": x.reshape(-1, 1),
        "CA": np.cos(x).reshape(-1, 1),
        "FPRC": np.column_stack((x, np.sin(x))),
        "EIC": np.sin(2.0 * x).reshape(-1, 1),
    }
    gbt = GBTConfig(
        config_id="adaptive_e2e_test",
        n_estimators=6,
        max_depth=2,
        min_samples_split=2,
        learning_rate=0.1,
        subsample=1.0,
        random_state=4,
        n_runs=1,
    )
    scout = run_scout(
        sample_ids=sample_ids,
        targets=targets,
        structure_sizes=np.arange(10, 30, dtype=float),
        features_by_invariant=features,
        retained_pairs=design.representation_spec.pair_order,
        pair_presence_by_sample={
            sample_id: design.representation_spec.pair_order for sample_id in sample_ids
        },
        user_target=0.4,
        config=ScoutConfig(
            probe=ProbeSamplingConfig(
                fraction=1.0,
                min_samples=20,
                max_samples=20,
                min_pair_support=1,
            ),
            gbt=gbt,
            bootstrap=BootstrapConfig(replicates=5, random_seed=2),
        ),
    )
    artifact = scout.to_execution_artifact(
        representation_hash=design.representation_spec.spec_hash
    )
    provider = InMemoryFeatureProvider(features, sample_ids)
    execution = ProgressiveExecutionEngine(
        provider,
        config=ExecutionConfig(
            target_metric=artifact.target_metric,
            target_value=artifact.target_value,
            representation_hash=artifact.representation_hash,
            require_shared_fold_assignment=True,
            gbt=gbt,
        ),
    ).run_from_scout(
        artifact=artifact,
        plan=EvaluationPlan(EvaluationMode.FULL_LABELED_CV, sample_ids),
        targets_by_sample=dict(zip(sample_ids, targets)),
    )

    assert design.representation_spec.frozen is True
    assert artifact.representation_hash == design.representation_spec.spec_hash
    assert execution.acquisition_order[0] in features
    assert provider.calls[0][0] == execution.acquisition_order[0]
    assert execution.rounds
