from __future__ import annotations

from pathlib import Path

import pytest

from mint_scout.execution.profile import (
    AcceptanceEvaluationJobRequest,
    DatasetAuditJobRequest,
    DatasetPreparationJobRequest,
    FeatureJobRequest,
    FeatureOutlierDiagnosticJobRequest,
    FeatureQCJobRequest,
    FiltrationAuditJobRequest,
    FrozenTestEvaluationJobRequest,
    ModelEvaluationJobRequest,
    ProbeAuditJobRequest,
    ProbeSelectionJobRequest,
    RepresentationDesignJobRequest,
    ScoutCombineJobRequest,
    ScoutOOFJobRequest,
    ValidationEvaluationJobRequest,
    build_dataset_audit_job_plan,
    build_acceptance_evaluation_job_plan,
    build_dataset_preparation_job_plan,
    build_feature_job_plan,
    build_feature_outlier_diagnostic_job_plan,
    build_feature_qc_job_plan,
    build_filtration_audit_job_plan,
    build_frozen_test_evaluation_job_plan,
    build_model_evaluation_job_plan,
    build_probe_audit_job_plan,
    build_probe_selection_job_plan,
    build_representation_design_job_plan,
    build_scout_combine_job_plan,
    build_scout_oof_job_plan,
    build_validation_evaluation_job_plan,
    load_execution_profile,
)


def test_build_acceptance_evaluation_plan_uses_one_frozen_candidate():
    profile = load_execution_profile(Path("configs/execution/sapelo2.yaml"))
    plan = build_acceptance_evaluation_job_plan(
        profile,
        AcceptanceEvaluationJobRequest(
            dataset_id="toy",
            invariants=("EIC", "PL"),
            representation_hash="repr-hash",
            run_id="acceptance",
            task_config="configs/tasks/toy.yaml",
            gbt_config="configs/gbt/plbind_adaptive_gbt.yaml",
            representation_spec="/scratch/project/repr.json",
            scout_artifact="/scratch/project/scout.json",
            candidate_rank=2,
            train_feature_qc_report="/scratch/project/train-qc.json",
            evaluation_feature_qc_report="/scratch/project/test-qc.json",
            train_feature_manifests={
                "EIC": "/scratch/project/train-eic.jsonl",
                "PL": "/scratch/project/train-pl.jsonl",
            },
            evaluation_feature_manifests={
                "EIC": "/scratch/project/test-eic.jsonl",
                "PL": "/scratch/project/test-pl.jsonl",
            },
        ),
    )

    assert plan.job_kind == "acceptance_evaluation"
    assert plan.evidence_scope == "acceptance_test"
    assert plan.environment["CANDIDATE_RANK"] == "2"
    assert plan.environment["FEATURE_INVARIANTS"] == "EIC:PL"
    assert plan.environment["GBT_CONFIG"].endswith("plbind_adaptive_gbt.yaml")
    assert plan.submit_command[-1].endswith("run_acceptance_gbt_evaluation.sbatch")


def test_build_progressive_acceptance_plan_reuses_the_previous_stage():
    profile = load_execution_profile(Path("configs/execution/sapelo2.yaml"))
    plan = build_acceptance_evaluation_job_plan(
        profile,
        AcceptanceEvaluationJobRequest(
            dataset_id="toy",
            invariants=("PL", "PH"),
            representation_hash="repr-hash",
            run_id="acceptance",
            task_config="configs/tasks/toy.yaml",
            gbt_config="configs/gbt/plbind_adaptive_gbt.yaml",
            representation_spec="/scratch/project/repr.json",
            scout_artifact="/scratch/project/scout.json",
            max_acquisitions=2,
            prior_evaluation_report="/scratch/project/stage-01.json",
            train_feature_qc_report="/scratch/project/train-qc.json",
            evaluation_feature_qc_report="/scratch/project/test-qc.json",
            train_feature_manifests={
                "PL": "/scratch/project/train-pl.jsonl",
                "PH": "/scratch/project/train-ph.jsonl",
            },
            evaluation_feature_manifests={
                "PL": "/scratch/project/test-pl.jsonl",
                "PH": "/scratch/project/test-ph.jsonl",
            },
        ),
    )

    assert plan.environment["ACCEPTANCE_MODE"] == "progressive"
    assert plan.environment["MAX_ACQUISITIONS"] == "2"
    assert plan.environment["PRIOR_EVALUATION_REPORT"].endswith("stage-01.json")
    assert plan.environment["FEATURE_INVARIANTS"] == "PL:PH"
    assert "acceptance_evaluation_stage02" in plan.manifest_path
    assert plan.manifest_path.endswith(".json")


def test_build_setup_plans_form_a_frozen_dependency_chain():
    profile = load_execution_profile(Path("configs/execution/sapelo2.yaml"))
    audit = build_dataset_audit_job_plan(
        profile,
        DatasetAuditJobRequest(
            dataset_id="new-data",
            run_id="setup/one",
            task_config="configs/tasks/new.yaml",
        ),
    )
    representation = build_representation_design_job_plan(
        profile,
        RepresentationDesignJobRequest(
            dataset_id="new-data",
            run_id="setup/one",
            task_config="configs/tasks/new.yaml",
            data_audit_report=audit.manifest_path,
            design_input_hash="design-input-hash",
        ),
    )
    probe = build_probe_selection_job_plan(
        profile,
        ProbeSelectionJobRequest(
            dataset_id="new-data",
            representation_hash="repr-hash",
            run_id="setup/one",
            task_config="configs/tasks/new.yaml",
            scout_config="configs/scout/v1.yaml",
            representation_spec=representation.manifest_path,
        ),
    )
    probe_audit = build_probe_audit_job_plan(
        profile,
        ProbeAuditJobRequest(
            dataset_id="new-data",
            representation_hash="repr-hash",
            run_id="setup/one",
            task_config="configs/tasks/new.yaml",
            scout_config="configs/scout/v1.yaml",
            probe_selection=probe.manifest_path,
        ),
    )

    assert audit.job_kind == "dataset_audit"
    assert audit.environment["OUTPUT_JSON"] == audit.manifest_path
    assert representation.input_manifests == {"DATA_AUDIT": audit.manifest_path}
    assert representation.environment["DESIGN_INPUT_HASH"] == "design-input-hash"
    assert probe.input_manifests == {"REPRESENTATION": representation.manifest_path}
    assert probe_audit.input_manifests == {"PROBE_SELECTION": probe.manifest_path}
    assert probe_audit.resources.time == "02:00:00"
    assert audit.submit_command[-1].endswith("scripts/slurm/run_dataset_audit.sbatch")


def test_probe_plan_can_reuse_one_frozen_sample_axis():
    profile = load_execution_profile(Path("configs/execution/sapelo2.yaml"))
    plan = build_probe_selection_job_plan(
        profile,
        ProbeSelectionJobRequest(
            dataset_id="new-data",
            representation_hash="alternative-repr",
            run_id="shared-probe",
            task_config="configs/tasks/new.yaml",
            scout_config="configs/scout/v1.yaml",
            representation_spec="/scratch/repr-alt.json",
            frozen_sample_source="/scratch/selected-probe.json",
        ),
    )

    assert plan.environment["FROZEN_SAMPLE_SOURCE"] == "/scratch/selected-probe.json"
    assert plan.input_manifests["FROZEN_SAMPLE_SOURCE"] == "/scratch/selected-probe.json"


def test_build_dataset_preparation_plan_uses_provider_modules():
    profile = load_execution_profile(Path("configs/execution/sapelo2.yaml"))
    plan = build_dataset_preparation_job_plan(
        profile,
        DatasetPreparationJobRequest(
            dataset_id="new-data",
            run_id="from-scratch",
            task_config="configs/tasks/new.yaml",
            module_load=("OpenBabel/3.1.1",),
            replace_module_environment=True,
            python_executable="/project/.venv/bin/python",
        ),
    )

    assert plan.job_kind == "dataset_preparation"
    assert plan.invariant == "DATASET"
    assert plan.environment["OUTPUT_REPORT"] == plan.manifest_path
    assert plan.environment["DATASET_PREPARATION_MODULES"] == "OpenBabel/3.1.1"
    assert plan.environment["DATASET_PREPARATION_MODULE_PURGE"] == "1"
    assert plan.environment["PYTHON_BIN"] == "/project/.venv/bin/python"
    assert plan.submit_command[-1].endswith(
        "scripts/slurm/run_dataset_preparation.sbatch"
    )


def test_load_sapelo2_profile_and_build_feature_plan():
    profile = load_execution_profile(Path("configs/execution/sapelo2.yaml"))
    request = FeatureJobRequest(
        dataset_id="casf2016",
        invariant="pl",
        evidence_scope="full_train",
        representation_hash="abcdef1234567890",
        run_id="thread/one",
        task_config="configs/tasks/casf2016_sapelo2.yaml",
        representation_spec="/path/to/workdir/mathagent/runs/repr.json",
    )

    plan = build_feature_job_plan(profile, request)

    assert plan.profile_id == "sapelo2"
    assert plan.scheduler == "slurm"
    assert plan.job_kind == "feature_batch"
    assert plan.dataset_id == "casf2016"
    assert plan.invariant == "PL"
    assert plan.evidence_scope == "full_train"
    assert plan.representation_hash == "abcdef1234567890"
    assert len(plan.plan_id) == 16
    assert plan.resources.mem == "16gb"
    assert plan.environment["TASK_CONFIG"] == "/path/to/MathAgent/configs/tasks/casf2016_sapelo2.yaml"
    assert plan.environment["RUN_ROOT"] == "/path/to/workdir/mathagent/runs"
    assert plan.environment["INVARIANT"] == "PL"
    assert plan.environment["SPLIT"] == "train"
    assert plan.environment["LIMIT"] == "all"
    assert plan.environment["REPRESENTATION_SPEC"] == "/path/to/workdir/mathagent/runs/repr.json"
    assert plan.environment["LEGACY_ROOT"] == "/path/to/legacy/embed_nn/plbind"
    assert plan.environment["MINT_MODULE_SETUP"].endswith("scripts/sapelo2/load_modules.sh")
    assert profile.lifecycle_script == "scripts/sapelo2/run_agent_lifecycle.sbatch"
    assert "feature_casf2016_PL_full_train_abcdef123456_thread_one.jsonl" in plan.manifest_path
    assert plan.submit_command[0] == "env"
    assert "sbatch" in plan.submit_command
    assert plan.submit_command[-1].endswith("scripts/slurm/run_feature_batch.sbatch")


@pytest.mark.parametrize(
    ("scope", "expected_split"),
    (("validation", "validation"), ("external_test", "test")),
)
def test_feature_plan_supports_held_out_split_scopes(scope, expected_split):
    profile = load_execution_profile(Path("configs/execution/sapelo2.yaml"))
    plan = build_feature_job_plan(
        profile,
        FeatureJobRequest(
            dataset_id="official",
            invariant="CA",
            evidence_scope=scope,
            representation_hash="repr-hash",
            run_id="held-out",
            task_config="configs/tasks/official.yaml",
            representation_spec="/scratch/project/repr.json",
        ),
    )

    assert plan.evidence_scope == scope
    assert plan.environment["SPLIT"] == expected_split


def test_feature_plan_rejects_non_slurm_profile(tmp_path: Path):
    profile_path = tmp_path / "local.yaml"
    profile_path.write_text(
        "\n".join(
            [
                "profile_id: local",
                "profile_type: local",
                "paths:",
                "  project_root: /tmp/project",
                "  scratch_root: /tmp/project",
                "  cache_root: /tmp/project/cache",
                "  run_root: /tmp/project/runs",
                "  log_root: /tmp/project/logs",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    profile = load_execution_profile(profile_path)
    request = FeatureJobRequest(
        dataset_id="toy",
        invariant="PH",
        evidence_scope="full_train",
        representation_hash="repr",
        run_id="run",
        task_config="task.yaml",
    )

    with pytest.raises(ValueError, match="does not support Slurm"):
        build_feature_job_plan(profile, request)


def test_build_feature_qc_plan_uses_frozen_manifests():
    profile = load_execution_profile(Path("configs/execution/sapelo2.yaml"))
    plan = build_feature_qc_job_plan(
        profile,
        FeatureQCJobRequest(
            dataset_id="casf2016",
            invariants=("PL", "PH"),
            evidence_scope="full_train",
            representation_hash="abcdef1234567890",
            run_id="qc/one",
            representation_spec="/path/to/workdir/mathagent/runs/repr.json",
            feature_manifests={
                "PH": "/path/to/workdir/mathagent/runs/PH.jsonl",
                "PL": "/path/to/workdir/mathagent/runs/PL.jsonl",
            },
            qc_config="configs/scout/v1.yaml",
        ),
    )

    assert plan.job_kind == "feature_qc"
    assert plan.invariants == ("PH", "PL")
    assert plan.resources.time == "02:00:00"
    assert plan.environment["FEATURE_INVARIANTS"] == "PH:PL"
    assert plan.environment["SAMPLE_ID_FILE"].endswith("/PH.jsonl")
    assert plan.environment["QC_CONFIG"] == "/path/to/MathAgent/configs/scout/v1.yaml"
    assert "_ph-pl_full_train_" in plan.manifest_path
    assert plan.environment["MINT_MODULE_SETUP"].endswith("scripts/sapelo2/load_modules.sh")
    assert plan.submit_command[0] == "env"
    assert "sbatch" in plan.submit_command
    assert plan.submit_command[-1].endswith("scripts/slurm/run_feature_qc.sbatch")


def test_build_filtration_audit_plan_uses_same_sample_source():
    profile = load_execution_profile(Path("configs/execution/sapelo2.yaml"))
    plan = build_filtration_audit_job_plan(
        profile,
        FiltrationAuditJobRequest(
            dataset_id="casf2016",
            invariants=("PL",),
            evidence_scope="full_train",
            representation_hash="abcdef1234567890",
            run_id="audit/one",
            representation_spec="/path/to/workdir/mathagent/runs/repr.json",
            feature_manifests={
                "PL": "/path/to/workdir/mathagent/runs/PL.jsonl"
            },
            audit_config="configs/scout/v1.yaml",
        ),
    )

    assert plan.job_kind == "filtration_audit"
    assert plan.invariants == ("PL",)
    assert plan.environment["SAMPLE_ID_FILE"].endswith("/PL.jsonl")
    assert plan.environment["AUDIT_CONFIG"].endswith("configs/scout/v1.yaml")
    assert "_pl_full_train_" in plan.manifest_path
    assert plan.submit_command[0] == "env"
    assert plan.submit_command[-1].endswith("scripts/slurm/run_filtration_audit.sbatch")


def test_qc_and_filtration_paths_are_unique_per_invariant_set():
    profile = load_execution_profile(Path("configs/execution/sapelo2.yaml"))
    common = {
        "dataset_id": "casf2016",
        "evidence_scope": "full_train",
        "representation_hash": "abcdef1234567890",
        "run_id": "progressive/one",
        "representation_spec": "/path/to/workdir/mathagent/runs/repr.json",
    }
    pl_manifest = "/path/to/workdir/mathagent/runs/PL.jsonl"
    ph_manifest = "/path/to/workdir/mathagent/runs/PH.jsonl"

    qc_pl = build_feature_qc_job_plan(
        profile,
        FeatureQCJobRequest(
            **common,
            invariants=("PL",),
            feature_manifests={"PL": pl_manifest},
            qc_config="configs/scout/v1.yaml",
        ),
    )
    qc_ph_pl = build_feature_qc_job_plan(
        profile,
        FeatureQCJobRequest(
            **common,
            invariants=("PL", "PH"),
            feature_manifests={"PL": pl_manifest, "PH": ph_manifest},
            qc_config="configs/scout/v1.yaml",
        ),
    )
    audit_pl = build_filtration_audit_job_plan(
        profile,
        FiltrationAuditJobRequest(
            **common,
            invariants=("PL",),
            feature_manifests={"PL": pl_manifest},
            audit_config="configs/scout/v1.yaml",
        ),
    )
    audit_ph_pl = build_filtration_audit_job_plan(
        profile,
        FiltrationAuditJobRequest(
            **common,
            invariants=("PL", "PH"),
            feature_manifests={"PL": pl_manifest, "PH": ph_manifest},
            audit_config="configs/scout/v1.yaml",
        ),
    )

    assert qc_pl.manifest_path != qc_ph_pl.manifest_path
    assert audit_pl.manifest_path != audit_ph_pl.manifest_path


def test_build_outlier_diagnostic_plan_is_read_only_and_single_invariant():
    profile = load_execution_profile(Path("configs/execution/sapelo2.yaml"))
    plan = build_feature_outlier_diagnostic_job_plan(
        profile,
        FeatureOutlierDiagnosticJobRequest(
            dataset_id="casf2016",
            invariant="fprc",
            evidence_scope="probe",
            representation_hash="abcdef1234567890",
            run_id="diagnostic/one",
            representation_spec="/path/to/workdir/mathagent/runs/repr.json",
            feature_manifest="/path/to/workdir/mathagent/runs/FPRC.jsonl",
            feature_qc_report="/path/to/workdir/mathagent/runs/qc.json",
            sample_id_file="/path/to/workdir/mathagent/runs/probe.json",
            top_k=7,
        ),
    )

    assert plan.job_kind == "feature_outlier_diagnostic"
    assert plan.invariants == ("FPRC",)
    assert plan.input_manifests == {
        "FPRC": "/path/to/workdir/mathagent/runs/FPRC.jsonl"
    }
    assert plan.environment["PROBE_SELECTION"].endswith("/probe.json")
    assert plan.environment["FEATURE_QC_REPORT"].endswith("/qc.json")
    assert plan.environment["TOP_K"] == "7"
    assert plan.submit_command[-1].endswith(
        "scripts/slurm/run_feature_outlier_diagnostic.sbatch"
    )


def test_build_scout_oof_plan_isolates_one_invariant():
    profile = load_execution_profile(Path("configs/execution/sapelo2.yaml"))
    plan = build_scout_oof_job_plan(
        profile,
        ScoutOOFJobRequest(
            dataset_id="casf2016",
            invariant="pl",
            representation_hash="abcdef1234567890",
            run_id="scout/one",
            task_config="configs/tasks/casf2016_sapelo2.yaml",
            scout_config="configs/scout/v1.yaml",
            representation_spec="/path/to/workdir/mathagent/runs/repr.json",
            probe_selection="/path/to/workdir/mathagent/runs/probe.json",
            feature_manifest="/path/to/workdir/mathagent/runs/PL.jsonl",
            feature_qc_report="/path/to/workdir/mathagent/runs/qc.json",
            filtration_audit_report="/path/to/workdir/mathagent/runs/audit.json",
        ),
    )

    assert plan.job_kind == "scout_oof"
    assert plan.invariants == ("PL",)
    assert plan.resources.mem == "64gb"
    assert plan.environment["FEATURE_INVARIANTS"] == "PL"
    assert plan.environment["GBT_CONFIG"].endswith("plbind_adaptive_gbt.yaml")
    assert plan.input_manifests == {
        "PL": "/path/to/workdir/mathagent/runs/PL.jsonl"
    }
    assert plan.submit_command[-1].endswith("scripts/slurm/run_scout_oof.sbatch")


def test_build_scout_combine_plan_uses_frozen_oof_reports():
    profile = load_execution_profile(Path("configs/execution/sapelo2.yaml"))
    plan = build_scout_combine_job_plan(
        profile,
        ScoutCombineJobRequest(
            dataset_id="casf2016",
            invariants=("PL", "PH"),
            representation_hash="abcdef1234567890",
            run_id="combine/one",
            scout_config="configs/scout/v1.yaml",
            representation_spec="/path/to/workdir/mathagent/runs/repr.json",
            probe_selection="/path/to/workdir/mathagent/runs/probe.json",
            scout_reports={
                "PH": "/path/to/workdir/mathagent/runs/scout-PH.json",
                "PL": "/path/to/workdir/mathagent/runs/scout-PL.json",
            },
            feature_qc_report="/path/to/workdir/mathagent/runs/qc.json",
        ),
    )

    assert plan.job_kind == "scout_combine"
    assert plan.invariants == ("PH", "PL")
    assert plan.environment["SOURCE_INVARIANTS"] == "PH:PL"
    assert plan.environment["RANKING_POLICY"] == "hierarchical_empirical_v1"
    assert plan.manifest_path.endswith("_hierarchical_empirical_v1.json")
    assert plan.input_manifests == {
        "PH": "/path/to/workdir/mathagent/runs/scout-PH.json",
        "PL": "/path/to/workdir/mathagent/runs/scout-PL.json",
    }
    assert plan.submit_command[-1].endswith("scripts/slurm/run_scout_combine.sbatch")


def test_build_scout_combine_plan_compacts_long_output_filename():
    profile = load_execution_profile(Path("configs/execution/sapelo2.yaml"))
    long_dataset = "pdbbind2016-casf2016-best-available-topk-20260925-" * 3
    long_run = (
        "pdbbind2016-casf2016-best-available-topk-20260925-"
        "representation-support_pruned_local95_hierarchical_empirical_v1-"
    ) * 3

    plan = build_scout_combine_job_plan(
        profile,
        ScoutCombineJobRequest(
            dataset_id=long_dataset,
            invariants=("PH", "PL", "CA", "FPRC", "EIC"),
            representation_hash="abcdef1234567890",
            run_id=long_run,
            scout_config="configs/scout/v1.yaml",
            representation_spec="/path/to/workdir/mathagent/runs/repr.json",
            probe_selection="/path/to/workdir/mathagent/runs/probe.json",
            scout_reports={
                "PH": "/path/to/workdir/mathagent/runs/scout-PH.json",
                "PL": "/path/to/workdir/mathagent/runs/scout-PL.json",
                "CA": "/path/to/workdir/mathagent/runs/scout-CA.json",
                "FPRC": "/path/to/workdir/mathagent/runs/scout-FPRC.json",
                "EIC": "/path/to/workdir/mathagent/runs/scout-EIC.json",
            },
            feature_qc_report="/path/to/workdir/mathagent/runs/qc.json",
        ),
    )

    assert len(Path(plan.manifest_path).name) < 255
    assert len(Path(plan.environment["COMBINED_REPORT"]).name) < 255


def test_build_model_evaluation_plan_uses_frozen_scout_and_features():
    profile = load_execution_profile(Path("configs/execution/sapelo2.yaml"))
    plan = build_model_evaluation_job_plan(
        profile,
        ModelEvaluationJobRequest(
            dataset_id="casf2016",
            invariants=("PL",),
            evidence_scope="full_train",
            representation_hash="abcdef1234567890",
            run_id="evaluate/one",
            task_config="configs/tasks/casf2016_sapelo2.yaml",
            gbt_config="configs/gbt/plbind_fixed_gbt.yaml",
            representation_spec="/path/to/workdir/mathagent/runs/repr.json",
            scout_artifact="/path/to/workdir/mathagent/runs/scout.json",
            feature_qc_report="/path/to/workdir/mathagent/runs/qc.json",
            feature_manifests={
                "PL": "/path/to/workdir/mathagent/runs/PL.jsonl"
            },
            max_acquisitions=1,
        ),
    )

    assert plan.job_kind == "model_evaluation"
    assert plan.evidence_scope == "full_train"
    assert plan.resources.mem == "64gb"
    assert plan.environment["SCOUT_ARTIFACT"].endswith("/scout.json")
    assert plan.environment["GBT_CONFIG"].endswith("plbind_fixed_gbt.yaml")
    assert plan.environment["MAX_ACQUISITIONS"] == "1"
    assert plan.input_manifests == {
        "PL": "/path/to/workdir/mathagent/runs/PL.jsonl"
    }
    assert plan.submit_command[-1].endswith(
        "scripts/slurm/run_model_evaluation.sbatch"
    )


def test_build_validation_evaluation_plan_keeps_split_inputs_separate():
    profile = load_execution_profile(Path("configs/execution/sapelo2.yaml"))
    plan = build_validation_evaluation_job_plan(
        profile,
        ValidationEvaluationJobRequest(
            dataset_id="toy",
            invariants=("PL", "PH"),
            representation_hash="repr-1",
            run_id="validation/one",
            task_config="configs/tasks/toy.yaml",
            gbt_config="configs/gbt/plbind_fixed_gbt.yaml",
            representation_spec="/scratch/repr.json",
            scout_artifact="/scratch/scout.json",
            train_feature_qc_report="/scratch/train-qc.json",
            validation_feature_qc_report="/scratch/validation-qc.json",
            train_feature_manifests={"PL": "/scratch/train-pl.jsonl", "PH": "/scratch/train-ph.jsonl"},
            validation_feature_manifests={
                "PL": "/scratch/validation-pl.jsonl",
                "PH": "/scratch/validation-ph.jsonl",
            },
            max_acquisitions=2,
            prior_evaluation_report="/scratch/validation-stage-1.json",
        ),
    )

    assert plan.job_kind == "validation_evaluation"
    assert plan.evidence_scope == "validation"
    assert plan.environment["FEATURE_INVARIANTS"] == "PL:PH"
    assert set(plan.input_manifests) == {
        "TRAIN_PL",
        "TRAIN_PH",
        "VALIDATION_PL",
        "VALIDATION_PH",
        "PRIOR_EVALUATION",
    }
    assert plan.environment["PRIOR_EVALUATION_REPORT"] == "/scratch/validation-stage-1.json"
    assert plan.submit_command[-1].endswith("run_validation_gbt_evaluation.sbatch")


def test_build_frozen_test_plan_only_contains_selected_invariants():
    profile = load_execution_profile(Path("configs/execution/sapelo2.yaml"))
    plan = build_frozen_test_evaluation_job_plan(
        profile,
        FrozenTestEvaluationJobRequest(
            dataset_id="toy",
            invariants=("PL",),
            representation_hash="repr-1",
            run_id="test/one",
            task_config="configs/tasks/toy.yaml",
            gbt_config="configs/gbt/plbind_fixed_gbt.yaml",
            representation_spec="/scratch/repr.json",
            selection_report="/scratch/selection.json",
            train_feature_qc_report="/scratch/train-qc.json",
            test_feature_qc_report="/scratch/test-qc.json",
            train_feature_manifests={"PL": "/scratch/train-pl.jsonl"},
            test_feature_manifests={"PL": "/scratch/test-pl.jsonl"},
        ),
    )

    assert plan.job_kind == "frozen_test_evaluation"
    assert plan.evidence_scope == "external_test"
    assert plan.invariants == ("PL",)
    assert set(plan.input_manifests) == {"TRAIN_PL", "TEST_PL"}
    assert plan.environment["N_BOOTSTRAP"] == "1000"
    assert plan.submit_command[-1].endswith("run_frozen_test_gbt_evaluation.sbatch")
    build_dataset_audit_job_plan,
