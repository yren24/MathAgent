from pathlib import Path

import pytest

from mint_scout.schemas import (
    DatasetManifest,
    EvaluationMode,
    ManifestValidationError,
    SampleRecord,
    SampleSplit,
    TaskCard,
    TaskType,
    route_evaluation_mode,
)


def _task(samples, *, task_type=TaskType.REGRESSION, retrieval=False, roles=()):
    return TaskCard(
        task_id="toy",
        dataset=DatasetManifest(dataset_id="toy-data", system_type="protein_ligand", samples=tuple(samples)),
        task_type=task_type,
        label_retrieval_allowed=retrieval,
        required_roles=tuple(roles),
    )


def test_explicit_train_and_labeled_test_routes_to_external_evaluation():
    plan = route_evaluation_mode(
        _task(
            [
                SampleRecord("a", 1.0, SampleSplit.TRAIN),
                SampleRecord("b", 2.0, SampleSplit.TRAIN),
                SampleRecord("c", 3.0, SampleSplit.TEST),
            ]
        )
    )

    assert plan.mode == EvaluationMode.EXPLICIT_LABELED_TEST
    assert plan.modeling_sample_ids == ("a", "b")
    assert plan.evaluation_sample_ids == ("c",)
    assert plan.inference_sample_ids == ()


def test_explicit_train_validation_test_routes_to_strict_split_protocol():
    plan = route_evaluation_mode(
        _task(
            [
                SampleRecord("train-a", 1.0, SampleSplit.TRAIN),
                SampleRecord("val-a", 2.0, SampleSplit.VALIDATION),
                SampleRecord("test-a", 3.0, SampleSplit.TEST),
            ]
        )
    )

    assert plan.mode == EvaluationMode.EXPLICIT_VALIDATION_AND_TEST
    assert plan.modeling_sample_ids == ("train-a",)
    assert plan.validation_sample_ids == ("val-a",)
    assert plan.evaluation_sample_ids == ("test-a",)


def test_all_labeled_without_split_routes_to_full_labeled_cv():
    plan = route_evaluation_mode(_task([SampleRecord("a", 1.0), SampleRecord("b", 2.0)]))

    assert plan.mode == EvaluationMode.FULL_LABELED_CV
    assert plan.modeling_sample_ids == ("a", "b")


def test_labeled_train_and_unlabeled_test_routes_to_cv_plus_inference():
    plan = route_evaluation_mode(
        _task(
            [
                SampleRecord("a", 1.0, SampleSplit.TRAIN),
                SampleRecord("b", 2.0, SampleSplit.TRAIN),
                SampleRecord("x", None, SampleSplit.TEST),
            ]
        )
    )

    assert plan.mode == EvaluationMode.LABELED_CV_PLUS_UNLABELED_INFERENCE
    assert plan.modeling_sample_ids == ("a", "b")
    assert plan.inference_sample_ids == ("x",)
    assert "predictions only" in plan.warnings[0]


def test_mixed_labels_without_split_routes_to_labeled_pool_plus_inference_pool():
    plan = route_evaluation_mode(_task([SampleRecord("a", 1.0), SampleRecord("x")]))

    assert plan.mode == EvaluationMode.LABELED_CV_PLUS_UNLABELED_INFERENCE
    assert plan.modeling_sample_ids == ("a",)
    assert plan.inference_sample_ids == ("x",)


def test_no_labels_routes_to_retrieval_when_allowed():
    plan = route_evaluation_mode(_task([SampleRecord("x")], retrieval=True))

    assert plan.mode == EvaluationMode.LABEL_RETRIEVAL_REQUIRED


def test_no_labels_without_retrieval_is_insufficient():
    plan = route_evaluation_mode(_task([SampleRecord("x")], retrieval=False))

    assert plan.mode == EvaluationMode.INSUFFICIENT_LABELS
    assert "No labeled samples" in plan.warnings[0]


def test_classification_is_marked_unsupported():
    plan = route_evaluation_mode(_task([SampleRecord("a", 1.0)], task_type=TaskType.CLASSIFICATION))

    assert plan.mode == EvaluationMode.UNSUPPORTED_TASK
    assert plan.modeling_sample_ids == ()


def test_missing_required_role_routes_to_needs_user_input():
    plan = route_evaluation_mode(
        _task(
            [
                SampleRecord("a", 1.0, role_paths={"protein": Path("a.pdb")}),
            ],
            roles=("protein", "ligand"),
        )
    )

    assert plan.mode == EvaluationMode.NEEDS_USER_INPUT
    assert "ligand" in plan.warnings[0]


def test_duplicate_sample_ids_raise_validation_error():
    with pytest.raises(ManifestValidationError):
        DatasetManifest(
            dataset_id="dupe",
            system_type="small_molecule",
            samples=(SampleRecord("a", 1.0), SampleRecord("a", 2.0)),
        )
