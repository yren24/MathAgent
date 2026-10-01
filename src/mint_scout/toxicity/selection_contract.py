from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from mint_scout.config import load_yaml
from mint_scout.toxicity.legacy import TOXICITY_INVARIANTS


SUPPORTED_METRICS = ("PCC2", "PCC", "R2")


@dataclass(frozen=True)
class ToxicitySelectionContract:
    requested_invariants: tuple[str, ...] = TOXICITY_INVARIANTS
    primary_metric: str = "PCC2"
    selection_objective: str = "maximize_rank1"
    user_target: float | None = None

    def __post_init__(self) -> None:
        methods = tuple(str(name).upper() for name in self.requested_invariants)
        if not methods or len(set(methods)) != len(methods):
            raise ValueError("toxicity requested_invariants must be nonempty and unique")
        if set(methods) - set(TOXICITY_INVARIANTS):
            raise ValueError("toxicity requested_invariants contain unsupported methods")
        metric = str(self.primary_metric).upper()
        if metric not in SUPPORTED_METRICS:
            raise ValueError(f"toxicity primary_metric must be one of {SUPPORTED_METRICS}")
        if self.selection_objective not in {"maximize_rank1", "satisfy_target"}:
            raise ValueError("toxicity selection_objective must be maximize_rank1 or satisfy_target")
        if self.selection_objective == "satisfy_target" and self.user_target is None:
            raise ValueError("toxicity satisfy_target requires an explicit user_target")
        if self.user_target is not None:
            target = float(self.user_target)
            if not math.isfinite(target) or target > 1.0:
                raise ValueError("toxicity user_target must be finite and no greater than 1")
            if metric == "PCC2" and target < 0.0:
                raise ValueError("toxicity PCC2 user_target must be between 0 and 1")
            if metric == "PCC" and target < -1.0:
                raise ValueError("toxicity PCC user_target must be between -1 and 1")
        object.__setattr__(self, "requested_invariants", methods)
        object.__setattr__(self, "primary_metric", metric)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["requested_invariants"] = list(self.requested_invariants)
        return payload

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ToxicitySelectionContract":
        return cls(
            requested_invariants=tuple(value.get("requested_invariants", TOXICITY_INVARIANTS)),
            primary_metric=str(value.get("primary_metric", "PCC2")),
            selection_objective=str(value.get("selection_objective", "maximize_rank1")),
            user_target=value.get("user_target"),
        )

    @classmethod
    def from_task(cls, path: str | Path) -> "ToxicitySelectionContract":
        task = load_yaml(Path(path))
        if task.get("system_type") != "small_molecule":
            raise ValueError("toxicity selection requires a small_molecule task")
        preferences = task.get("selection_preferences")
        if not isinstance(preferences, Mapping):
            raise ValueError("toxicity task requires selection_preferences")
        return cls(
            requested_invariants=tuple(task.get("invariants", ())),
            primary_metric=str(task.get("primary_metric", "PCC2")),
            selection_objective=str(preferences.get("selection_objective", "maximize_rank1")),
            user_target=preferences.get("user_target"),
        )


def methods_from_plan(plan: Mapping[str, Any]) -> tuple[str, ...]:
    contract = plan.get("selection_contract")
    if isinstance(contract, Mapping):
        return ToxicitySelectionContract.from_mapping(contract).requested_invariants
    signatures = plan.get("candidate_feature_signatures")
    if not isinstance(signatures, Mapping) or not signatures:
        raise ValueError("toxicity feature plan has no candidate signatures")
    first = next(iter(signatures.values()))
    if not isinstance(first, Mapping):
        raise ValueError("toxicity candidate signatures must be a mapping")
    methods = tuple(name for name in TOXICITY_INVARIANTS if name in first)
    if set(first) != set(methods):
        raise ValueError("toxicity feature plan includes unsupported methods")
    return methods
