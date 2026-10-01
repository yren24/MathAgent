import csv
import json

import numpy as np

from mint_scout.execution import ExecutionConfig, ProgressiveExecutionEngine
from mint_scout.execution.provider import InMemoryFeatureProvider
from mint_scout.models.gbt import GBTConfig
from mint_scout.reporting import write_run_artifacts
from mint_scout.representation import make_legacy_casf_representation_spec
from mint_scout.schemas import (
    DatasetManifest,
    EvaluationMode,
    EvaluationPlan,
    SampleRecord,
    TaskCard,
    TaskType,
)
from mint_scout.scout.pipeline import ScoutConfig, run_scout
from mint_scout.scout.sampling import ProbeSamplingConfig
from mint_scout.scout.stability import BootstrapConfig


def test_report_artifacts_separate_probe_and_acceptance_evidence(tmp_path):
    sample_ids = tuple(f"s{index:02d}" for index in range(20))
    x = np.linspace(-2.0, 2.0, len(sample_ids))
    targets = 2.0 * x + 0.1 * np.sin(x)
    features = {"PL": x.reshape(-1, 1), "CA": np.column_stack((x, x**2))}
    gbt = GBTConfig(
        config_id="report_test",
        n_estimators=5,
        max_depth=2,
        min_samples_split=2,
        learning_rate=0.1,
        subsample=1.0,
        random_state=8,
        n_runs=1,
    )
    scout = run_scout(
        sample_ids=sample_ids,
        targets=targets,
        structure_sizes=np.arange(len(sample_ids), dtype=float),
        features_by_invariant=features,
        user_target=-1.0,
        config=ScoutConfig(
            cv_folds=5,
            probe=ProbeSamplingConfig(fraction=1.0, min_samples=20, max_samples=20),
            gbt=gbt,
            bootstrap=BootstrapConfig(replicates=3, random_seed=4),
        ),
    )
    representation = make_legacy_casf_representation_spec()
    artifact = scout.to_execution_artifact(representation_hash=representation.spec_hash)
    plan = EvaluationPlan(EvaluationMode.FULL_LABELED_CV, sample_ids)
    execution = ProgressiveExecutionEngine(
        InMemoryFeatureProvider(features, sample_ids),
        config=ExecutionConfig(
            target_metric=artifact.target_metric,
            target_value=artifact.target_value,
            representation_hash=artifact.representation_hash,
            require_shared_fold_assignment=True,
            gbt=gbt,
        ),
    ).run_from_scout(
        artifact=artifact,
        plan=plan,
        targets_by_sample=dict(zip(sample_ids, targets)),
    )
    task = TaskCard(
        task_id="report-test",
        dataset=DatasetManifest(
            dataset_id="toy",
            system_type="protein_ligand",
            samples=tuple(SampleRecord(sample_id, float(target)) for sample_id, target in zip(sample_ids, targets)),
            label_name="binding affinity",
        ),
        task_type=TaskType.REGRESSION,
        user_target=-1.0,
    )

    bundle = write_run_artifacts(
        run_directory=tmp_path / "run",
        task_card=task,
        evaluation_plan=plan,
        representation_spec=representation,
        probe_result=scout,
        execution_result=execution,
        timing_seconds={"probe": 1.25, "execution": 2.5},
        anomalies=("TEST_ANOMALY",),
    )

    assert "probe_consensus_ranking.csv" in bundle.artifacts
    assert "probe_single_invariant_metrics.csv" in bundle.artifacts
    assert "full_evaluation_metrics.csv" in bundle.artifacts
    with bundle.artifacts["full_evaluation_metrics.csv"].open() as handle:
        rows = list(csv.DictReader(handle))
    assert rows and all(row["evaluation_protocol"] == "full_labeled_cv" for row in rows)
    assert "nominal_primary_score" not in rows[0]

    manifest = json.loads(bundle.artifacts["run_manifest.json"].read_text())
    assert manifest["representation_hash"] == representation.spec_hash
    assert manifest["probe_hash"] == scout.probe_hash
    report = bundle.artifacts["final_report.md"].read_text()
    assert "Probe OOF lead" in report
    assert "Acceptance score" in report
    assert "TEST_ANOMALY" in report
    assert "probe: `1.250` seconds" in report
    assert "Acquired invariants" in report
    assert "Skipped invariants" in report
