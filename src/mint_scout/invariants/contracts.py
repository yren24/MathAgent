from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from mint_scout.search.cost import CostRecord


@dataclass(frozen=True)
class FeatureToolResult:
    sample_id: str
    invariant_name: str
    legacy_name: str
    output_path: Path
    cost: CostRecord | None
    status: str


@dataclass(frozen=True)
class InvariantCapability:
    name: str
    legacy_name: str
    description: str
    supported_systems: tuple[str, ...]
    supported_representation_modes: tuple[str, ...]
    expected_shapes: dict[str, tuple[int, ...]]
    domain: str

    def expected_shape(self, representation_mode: str) -> tuple[int, ...]:
        try:
            return self.expected_shapes[representation_mode]
        except KeyError as exc:
            raise KeyError(
                f"{self.name} has no expected shape for representation mode {representation_mode!r}"
            ) from exc


@dataclass(frozen=True)
class FeatureValidationReport:
    shape: tuple[int, ...]
    dtype: str
    all_finite: bool
    nonzero_count: int
    minimum: float
    maximum: float


class InvariantAdapter(Protocol):
    @property
    def capability(self) -> InvariantCapability:
        ...

    def compute_one(self, sample_id: str, *, dry_run: bool = False) -> FeatureToolResult:
        ...


def load_feature_shape(path: str | Path) -> tuple[int, ...]:
    import numpy as np

    return tuple(int(dim) for dim in np.load(path).shape)


def assert_feature_shape(path: str | Path, expected_shape: tuple[int, ...]) -> tuple[int, ...]:
    observed_shape = load_feature_shape(path)
    matches = len(observed_shape) == len(expected_shape) and all(
        expected == -1 or observed == expected
        for observed, expected in zip(observed_shape, expected_shape)
    )
    if not matches:
        raise AssertionError(f"Feature shape mismatch for {path}: expected {expected_shape}, got {observed_shape}")
    return observed_shape


def validate_feature_file(
    path: str | Path,
    expected_shape: tuple[int, ...],
) -> FeatureValidationReport:
    import numpy as np

    array = np.load(path, allow_pickle=False)
    observed_shape = tuple(int(dim) for dim in array.shape)
    matches = len(observed_shape) == len(expected_shape) and all(
        expected == -1 or observed == expected
        for observed, expected in zip(observed_shape, expected_shape)
    )
    if not matches:
        raise AssertionError(
            f"Feature shape mismatch for {path}: expected {expected_shape}, got {observed_shape}"
        )
    if array.size == 0:
        raise ValueError(f"Feature array is empty: {path}")
    all_finite = bool(np.isfinite(array).all())
    if not all_finite:
        raise ValueError(f"Feature array contains NaN or Inf: {path}")
    return FeatureValidationReport(
        shape=observed_shape,
        dtype=str(array.dtype),
        all_finite=True,
        nonzero_count=int(np.count_nonzero(array)),
        minimum=float(np.min(array)),
        maximum=float(np.max(array)),
    )
