from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

from mint_scout.invariants.contracts import FeatureToolResult
from mint_scout.representation import RepresentationSpec


@dataclass(frozen=True)
class FeatureArtifact:
    invariant: str
    sample_ids: tuple[str, ...]
    features: np.ndarray
    cached_sample_ids: tuple[str, ...] = ()
    computed_sample_ids: tuple[str, ...] = ()
    cost: float = 0.0


class FeatureProvider(Protocol):
    def acquire(
        self,
        invariant: str,
        sample_ids: Sequence[str],
        *,
        representation_hash: str,
    ) -> FeatureArtifact:
        """Return features aligned to sample_ids, computing only missing cache entries."""


@dataclass
class InMemoryFeatureProvider:
    features_by_invariant: Mapping[str, np.ndarray]
    all_sample_ids: Sequence[str]
    precomputed: Mapping[str, Sequence[str]] | None = None
    cost_by_invariant: Mapping[str, float] | None = None

    def __post_init__(self) -> None:
        self._sample_ids = tuple(self.all_sample_ids)
        if len(set(self._sample_ids)) != len(self._sample_ids):
            raise ValueError("all_sample_ids contains duplicates")
        self._index = {sample_id: index for index, sample_id in enumerate(self._sample_ids)}
        self._available: dict[tuple[str, str], set[str]] = {}
        for invariant, sample_ids in (self.precomputed or {}).items():
            self._available[(invariant.upper(), "*")] = {str(sample_id) for sample_id in sample_ids}
        self.calls: list[tuple[str, tuple[str, ...], tuple[str, ...]]] = []

    def acquire(
        self,
        invariant: str,
        sample_ids: Sequence[str],
        *,
        representation_hash: str,
    ) -> FeatureArtifact:
        name = invariant.upper()
        requested = tuple(sample_ids)
        if name not in self.features_by_invariant:
            raise KeyError(f"No synthetic features registered for invariant {name!r}")
        missing_ids = [sample_id for sample_id in requested if sample_id not in self._index]
        if missing_ids:
            raise KeyError(f"Unknown sample ids for {name}: {missing_ids[:10]}")
        key = (name, representation_hash)
        available = self._available.setdefault(key, set(self._available.get((name, "*"), set())))
        cached = tuple(sample_id for sample_id in requested if sample_id in available)
        computed = tuple(sample_id for sample_id in requested if sample_id not in available)
        available.update(computed)
        self.calls.append((name, cached, computed))

        matrix = np.asarray(self.features_by_invariant[name], dtype=float)
        row_indices = [self._index[sample_id] for sample_id in requested]
        return FeatureArtifact(
            invariant=name,
            sample_ids=requested,
            features=matrix[row_indices],
            cached_sample_ids=cached,
            computed_sample_ids=computed,
            cost=float((self.cost_by_invariant or {}).get(name, 1.0)) * len(computed),
        )


@dataclass
class PLBindCachedFeatureProvider:
    tool: object
    representation_spec: RepresentationSpec
    manifest_paths: Mapping[str, str | Path] | None = None

    def acquire(
        self,
        invariant: str,
        sample_ids: Sequence[str],
        *,
        representation_hash: str,
    ) -> FeatureArtifact:
        name = invariant.upper()
        requested = tuple(sample_ids)
        self.representation_spec.assert_frozen()
        if representation_hash != self.representation_spec.spec_hash:
            raise ValueError("Requested representation hash does not match provider's frozen spec")
        rows: list[np.ndarray] = []
        cached: list[str] = []
        computed: list[str] = []
        total_cost = 0.0
        expected_shape: tuple[int, ...] | None = None
        manifest_rows = self._manifest_rows(name)
        for sample_id in requested:
            if manifest_rows is not None:
                row = manifest_rows.get(sample_id)
                if row is None:
                    raise KeyError(
                        f"Precomputed {name} manifest has no row for sample {sample_id!r}"
                    )
                output_path = Path(str(row["output_path"])).expanduser()
                if not output_path.is_file():
                    raise FileNotFoundError(
                        f"Precomputed {name} feature does not exist: {output_path}"
                    )
                cached.append(sample_id)
            else:
                result = self.tool.compute_with_spec(
                    sample_id,
                    name,
                    self.representation_spec,
                    dry_run=False,
                )
                _assert_tool_result(result, sample_id=sample_id, invariant=name)
                output_path = Path(result.output_path)
                if result.status == "cached":
                    cached.append(sample_id)
                elif result.status == "computed":
                    computed.append(sample_id)
                    if result.cost is not None:
                        total_cost += result.cost.cpu_core_hours
                else:
                    raise ValueError(
                        f"Unsupported feature result status for execution: {result.status!r}"
                    )
            array = np.load(output_path)
            if expected_shape is None:
                expected_shape = tuple(array.shape)
            elif tuple(array.shape) != expected_shape:
                raise ValueError(
                    f"{name} feature shape mismatch: expected {expected_shape}, got {tuple(array.shape)} "
                    f"for {result.output_path}"
                )
            rows.append(np.asarray(array, dtype=float).reshape(-1))
        if not rows:
            raise ValueError("sample_ids must contain at least one sample")
        return FeatureArtifact(
            invariant=name,
            sample_ids=requested,
            features=np.vstack(rows),
            cached_sample_ids=tuple(cached),
            computed_sample_ids=tuple(computed),
            cost=total_cost,
        )

    def _manifest_rows(self, invariant: str) -> dict[str, Mapping[str, Any]] | None:
        normalized_paths = {
            str(name).upper(): Path(path).expanduser()
            for name, path in (self.manifest_paths or {}).items()
        }
        manifest_path = normalized_paths.get(invariant)
        if manifest_path is None:
            return None
        rows: dict[str, Mapping[str, Any]] = {}
        with manifest_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, Mapping):
                    raise ValueError(
                        f"Feature manifest {manifest_path}:{line_number} is not an object"
                    )
                sample_id = str(row.get("sample_id") or "")
                if not sample_id:
                    raise ValueError(
                        f"Feature manifest {manifest_path}:{line_number} has no sample_id"
                    )
                if sample_id in rows:
                    raise ValueError(
                        f"Feature manifest {manifest_path} has duplicate sample {sample_id!r}"
                    )
                if str(row.get("invariant") or "").upper() != invariant:
                    raise ValueError(
                        f"Feature manifest {manifest_path}:{line_number} invariant mismatch"
                    )
                if row.get("status") not in {"cached", "computed"}:
                    raise ValueError(
                        f"Feature manifest {manifest_path}:{line_number} is not complete"
                    )
                if not row.get("output_path"):
                    raise ValueError(
                        f"Feature manifest {manifest_path}:{line_number} has no output_path"
                    )
                rows[sample_id] = row
        if not rows:
            raise ValueError(f"Feature manifest is empty: {manifest_path}")
        return rows


def _assert_tool_result(result: FeatureToolResult, *, sample_id: str, invariant: str) -> None:
    if result.sample_id != sample_id:
        raise ValueError(f"Feature tool returned sample_id={result.sample_id!r}; expected {sample_id!r}")
    if result.invariant_name.upper() != invariant:
        raise ValueError(f"Feature tool returned invariant={result.invariant_name!r}; expected {invariant!r}")
    if not Path(result.output_path).exists():
        raise FileNotFoundError(f"Feature tool output does not exist: {result.output_path}")
