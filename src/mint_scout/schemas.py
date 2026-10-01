from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Mapping

from mint_scout.data.element_pairs import SystemType


class TaskType(str, Enum):
    REGRESSION = "regression"
    CLASSIFICATION = "classification"


class SampleSplit(str, Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"
    INFERENCE = "inference"


class EvaluationMode(str, Enum):
    EXPLICIT_VALIDATION_AND_TEST = "explicit_validation_and_test"
    EXPLICIT_LABELED_TEST = "explicit_labeled_test"
    FULL_LABELED_CV = "full_labeled_cv"
    LABELED_CV_PLUS_UNLABELED_INFERENCE = "labeled_cv_plus_unlabeled_inference"
    LABEL_RETRIEVAL_REQUIRED = "label_retrieval_required"
    INSUFFICIENT_LABELS = "insufficient_labels"
    NEEDS_USER_INPUT = "needs_user_input"
    UNSUPPORTED_TASK = "unsupported_task"


class ManifestValidationError(ValueError):
    pass


@dataclass(frozen=True)
class SampleRecord:
    sample_id: str
    target: float | None = None
    split: SampleSplit | None = None
    role_paths: Mapping[str, Path] = field(default_factory=dict)
    identifiers: Mapping[str, str] = field(default_factory=dict)

    @property
    def is_labeled(self) -> bool:
        return self.target is not None


@dataclass(frozen=True)
class DatasetManifest:
    dataset_id: str
    system_type: SystemType
    samples: tuple[SampleRecord, ...]
    label_name: str | None = None
    source: str | None = None

    def __post_init__(self) -> None:
        if not self.dataset_id:
            raise ManifestValidationError("dataset_id is required")
        ids = [sample.sample_id for sample in self.samples]
        duplicates = sorted({sample_id for sample_id in ids if ids.count(sample_id) > 1})
        if duplicates:
            raise ManifestValidationError(f"Duplicate sample ids: {duplicates[:10]}")

    @property
    def labeled_samples(self) -> tuple[SampleRecord, ...]:
        return tuple(sample for sample in self.samples if sample.is_labeled)

    @property
    def unlabeled_samples(self) -> tuple[SampleRecord, ...]:
        return tuple(sample for sample in self.samples if not sample.is_labeled)

    def samples_in_split(self, split: SampleSplit) -> tuple[SampleRecord, ...]:
        return tuple(sample for sample in self.samples if sample.split == split)


@dataclass(frozen=True)
class TaskCard:
    task_id: str
    dataset: DatasetManifest
    task_type: TaskType
    target_metric: str = "PCC"
    user_target: float | None = None
    label_retrieval_allowed: bool = False
    required_roles: tuple[str, ...] = ()


@dataclass(frozen=True)
class EvaluationPlan:
    mode: EvaluationMode
    modeling_sample_ids: tuple[str, ...]
    validation_sample_ids: tuple[str, ...] = ()
    evaluation_sample_ids: tuple[str, ...] = ()
    inference_sample_ids: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


def route_evaluation_mode(task: TaskCard) -> EvaluationPlan:
    if task.task_type != TaskType.REGRESSION:
        return EvaluationPlan(
            mode=EvaluationMode.UNSUPPORTED_TASK,
            modeling_sample_ids=(),
            warnings=("V1 supports regression only.",),
        )

    role_warning = _role_warning(task)
    if role_warning:
        return EvaluationPlan(
            mode=EvaluationMode.NEEDS_USER_INPUT,
            modeling_sample_ids=(),
            warnings=(role_warning,),
        )

    train = task.dataset.samples_in_split(SampleSplit.TRAIN)
    validation = task.dataset.samples_in_split(SampleSplit.VALIDATION)
    test = task.dataset.samples_in_split(SampleSplit.TEST)
    inference = task.dataset.samples_in_split(SampleSplit.INFERENCE)

    labeled_train = tuple(sample for sample in train if sample.is_labeled)
    labeled_validation = tuple(sample for sample in validation if sample.is_labeled)
    labeled_test = tuple(sample for sample in test if sample.is_labeled)
    unlabeled_test = tuple(sample for sample in test if not sample.is_labeled)
    unlabeled_inference = tuple(sample for sample in inference if not sample.is_labeled)

    if train and validation and test:
        if (
            len(labeled_train) == len(train)
            and len(labeled_validation) == len(validation)
            and len(labeled_test) == len(test)
        ):
            return EvaluationPlan(
                mode=EvaluationMode.EXPLICIT_VALIDATION_AND_TEST,
                modeling_sample_ids=_ids(labeled_train),
                validation_sample_ids=_ids(labeled_validation),
                evaluation_sample_ids=_ids(labeled_test),
            )

    if train and test:
        if labeled_train and len(labeled_test) == len(test):
            return EvaluationPlan(
                mode=EvaluationMode.EXPLICIT_LABELED_TEST,
                modeling_sample_ids=_ids(labeled_train),
                evaluation_sample_ids=_ids(labeled_test),
            )
        if labeled_train and (unlabeled_test or unlabeled_inference):
            return EvaluationPlan(
                mode=EvaluationMode.LABELED_CV_PLUS_UNLABELED_INFERENCE,
                modeling_sample_ids=_ids(labeled_train),
                inference_sample_ids=_ids(unlabeled_test + unlabeled_inference),
                warnings=("Unlabeled test/inference samples receive predictions only, not performance metrics.",),
            )

    labeled = task.dataset.labeled_samples
    unlabeled = task.dataset.unlabeled_samples
    if labeled and not unlabeled:
        return EvaluationPlan(
            mode=EvaluationMode.FULL_LABELED_CV,
            modeling_sample_ids=_ids(labeled),
        )
    if labeled and unlabeled:
        return EvaluationPlan(
            mode=EvaluationMode.LABELED_CV_PLUS_UNLABELED_INFERENCE,
            modeling_sample_ids=_ids(labeled),
            inference_sample_ids=_ids(unlabeled),
            warnings=("Mixed labeled/unlabeled samples without a complete labeled test use CV plus inference.",),
        )
    if task.label_retrieval_allowed:
        return EvaluationPlan(
            mode=EvaluationMode.LABEL_RETRIEVAL_REQUIRED,
            modeling_sample_ids=(),
        )
    return EvaluationPlan(
        mode=EvaluationMode.INSUFFICIENT_LABELS,
        modeling_sample_ids=(),
        warnings=("No labeled samples are available for supervised V1 ranking.",),
    )


def _ids(samples: tuple[SampleRecord, ...]) -> tuple[str, ...]:
    return tuple(sample.sample_id for sample in samples)


def _role_warning(task: TaskCard) -> str | None:
    if not task.required_roles:
        return None
    for sample in task.dataset.samples:
        missing = [role for role in task.required_roles if role not in sample.role_paths]
        if missing:
            return f"Sample {sample.sample_id!r} is missing required role paths: {missing}"
    return None
