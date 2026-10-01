from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping


@dataclass(frozen=True)
class ScoutExecutionArtifact:
    artifact_version: str
    modeling_sample_ids: tuple[str, ...]
    representation_hash: str
    frozen_priority_order: tuple[tuple[str, ...], ...]
    full_fold_assignment: Mapping[str, int]
    target_metric: str
    target_direction: str
    target_value: float
    target_source: str
    gbt_parameter_hash: str
    probe_hash: str
    ranking_policy: str = "hierarchical_empirical_v1"

    def __post_init__(self) -> None:
        if not self.modeling_sample_ids or len(set(self.modeling_sample_ids)) != len(self.modeling_sample_ids):
            raise ValueError("modeling_sample_ids must be nonempty and unique")
        if set(self.full_fold_assignment) != set(self.modeling_sample_ids):
            raise ValueError("full_fold_assignment must contain exactly the modeling sample ids")
        if len(set(self.full_fold_assignment.values())) < 2:
            raise ValueError("full_fold_assignment must contain at least two folds")
        if not self.frozen_priority_order:
            raise ValueError("frozen_priority_order cannot be empty")
        if not self.representation_hash or not self.gbt_parameter_hash or not self.probe_hash:
            raise ValueError("representation, GBT, and probe hashes are required")
        if not self.ranking_policy:
            raise ValueError("ranking_policy is required")
        if self.target_direction not in {"higher", "lower"}:
            raise ValueError("target_direction must be 'higher' or 'lower'")
        if not math.isfinite(self.target_value):
            raise ValueError("target_value must be finite")
        object.__setattr__(self, "full_fold_assignment", MappingProxyType(dict(self.full_fold_assignment)))

    def to_dict(self) -> dict[str, object]:
        return {
            "artifact_version": self.artifact_version,
            "modeling_sample_ids": list(self.modeling_sample_ids),
            "representation_hash": self.representation_hash,
            "frozen_priority_order": [list(subset) for subset in self.frozen_priority_order],
            "full_fold_assignment": dict(self.full_fold_assignment),
            "target_metric": self.target_metric,
            "target_direction": self.target_direction,
            "target_value": self.target_value,
            "target_source": self.target_source,
            "gbt_parameter_hash": self.gbt_parameter_hash,
            "probe_hash": self.probe_hash,
            "ranking_policy": self.ranking_policy,
        }

    def write(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    @classmethod
    def read(cls, path: str | Path) -> "ScoutExecutionArtifact":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            artifact_version=str(payload["artifact_version"]),
            modeling_sample_ids=tuple(payload["modeling_sample_ids"]),
            representation_hash=str(payload["representation_hash"]),
            frozen_priority_order=tuple(tuple(subset) for subset in payload["frozen_priority_order"]),
            full_fold_assignment={str(key): int(value) for key, value in payload["full_fold_assignment"].items()},
            target_metric=str(payload["target_metric"]),
            target_direction=str(payload["target_direction"]),
            target_value=float(payload["target_value"]),
            target_source=str(payload["target_source"]),
            gbt_parameter_hash=str(payload["gbt_parameter_hash"]),
            probe_hash=str(payload["probe_hash"]),
            ranking_policy=str(payload.get("ranking_policy") or "legacy_unspecified"),
        )
