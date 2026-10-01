from __future__ import annotations

import argparse
import copy
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from mint_scout.config import load_yaml
from mint_scout.filtration import FiltrationProfile
from mint_scout.invariants.manifest import stable_hash
from mint_scout.representation import RepresentationSpec
from mint_scout.representation_design import DISTANCE_INVARIANTS


REPAIR_SCHEMA = "mint-agent.filtration-repair.v1"


@dataclass(frozen=True)
class FiltrationRepairConfig:
    """Conservative, audit-driven range repair for distance invariants."""

    enabled: bool = False
    method_specific: bool = True
    max_rounds: int = 1
    extend_factor: float = 1.20
    shorten_factor: float = 0.85
    min_stop_angstrom: float = 4.0
    max_stop_angstrom: float = 30.0
    trailing_buffer_points: int = 4

    def __post_init__(self) -> None:
        if self.max_rounds < 0:
            raise ValueError("filtration_repair.max_rounds must be non-negative")
        if self.extend_factor <= 1.0:
            raise ValueError("filtration_repair.extend_factor must exceed 1")
        if not 0.0 < self.shorten_factor < 1.0:
            raise ValueError("filtration_repair.shorten_factor must be in (0, 1)")
        if self.min_stop_angstrom <= 0.0:
            raise ValueError("filtration_repair.min_stop_angstrom must be positive")
        if self.max_stop_angstrom <= self.min_stop_angstrom:
            raise ValueError(
                "filtration_repair.max_stop_angstrom must exceed min_stop_angstrom"
            )
        if self.trailing_buffer_points < 0:
            raise ValueError(
                "filtration_repair.trailing_buffer_points must be non-negative"
            )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "FiltrationRepairConfig":
        if value is None:
            return cls()
        allowed = {
            "enabled",
            "method_specific",
            "max_rounds",
            "extend_factor",
            "shorten_factor",
            "min_stop_angstrom",
            "max_stop_angstrom",
            "trailing_buffer_points",
        }
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise ValueError(f"Unknown filtration_repair fields: {unknown}")
        return cls(**dict(value))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FiltrationRepairResult:
    representation_spec: RepresentationSpec
    decisions: tuple[Mapping[str, Any], ...]
    changed: bool
    repair_round: int
    parent_representation_hash: str

    def report(
        self,
        *,
        task_id: str | None,
        audit: Mapping[str, Any],
        requested_invariants: Sequence[str],
    ) -> dict[str, Any]:
        return {
            "report_schema": REPAIR_SCHEMA,
            "status": "COMPLETE" if self.changed else "NO_REPAIR_NEEDED",
            "task_id": task_id,
            "dataset_id": task_id,
            "evidence_scope": "design",
            "audit_evidence_scope": audit.get("evidence_scope"),
            "audit_representation_hash": audit.get("representation_hash"),
            "parent_representation_hash": self.parent_representation_hash,
            "repair_round": self.repair_round,
            "requested_invariants": list(requested_invariants),
            "policy": _plain_mapping(
                self.representation_spec.parameters.get("filtration_repair_policy", {})
            ),
            "decisions": [_plain_mapping(value) for value in self.decisions],
            "representation_spec": self.representation_spec.to_dict(),
            "representation_hash": self.representation_spec.spec_hash,
        }


def repair_filtration_profiles(
    representation: RepresentationSpec,
    audit: Mapping[str, Any],
    *,
    requested_invariants: Sequence[str],
    config: FiltrationRepairConfig,
) -> FiltrationRepairResult:
    """Build a new frozen representation from train/probe filtration evidence.

    The audit must belong to ``representation``. EIC is deliberately excluded:
    its tau axis is not a Euclidean-distance filtration range.
    """

    _validate_audit_compatibility(representation, audit)
    current_round = _repair_round(representation)
    requested = tuple(dict.fromkeys(str(name).upper() for name in requested_invariants))
    profiles = dict(representation.filtration_profiles)
    audit_invariants = audit.get("invariants", {})
    if not isinstance(audit_invariants, Mapping):
        raise ValueError("filtration audit invariants must be a mapping")

    decisions: list[dict[str, Any]] = []
    allow_changes = config.enabled and current_round < config.max_rounds
    for invariant in requested:
        profile = profiles.get(invariant)
        if invariant not in DISTANCE_INVARIANTS or profile is None:
            continue
        evidence = audit_invariants.get(invariant, {})
        if not isinstance(evidence, Mapping):
            continue
        decision = _repair_decision(
            invariant=invariant,
            profile=profile,
            audit=evidence,
            config=config,
            allow_changes=allow_changes,
        )
        decisions.append(decision)
        if decision["changed"]:
            profiles[invariant] = _with_stop(
                profile, float(decision["new_stop"]), source="audit_method_specific_repair"
            )

    changed = any(bool(decision["changed"]) for decision in decisions)
    if not changed:
        return FiltrationRepairResult(
            representation_spec=representation,
            decisions=tuple(decisions),
            changed=False,
            repair_round=current_round,
            parent_representation_hash=representation.spec_hash,
        )

    parameters = _plain_mapping(representation.parameters)
    next_round = current_round + 1
    parameters["filtration_repair"] = {
        "parent_representation_hash": representation.spec_hash,
        "repair_round": next_round,
        "method_specific": config.method_specific,
        "decisions": copy.deepcopy(decisions),
    }
    parameters["filtration_repair_policy"] = config.to_dict()
    repair_identity = stable_hash(
        {
            "parent_representation_hash": representation.spec_hash,
            "profiles": {name: profile.to_dict() for name, profile in sorted(profiles.items())},
            "repair_round": next_round,
        }
    )
    repaired = RepresentationSpec(
        representation_id=f"{representation.mode.value}_{repair_identity}",
        mode=representation.mode,
        system_type=representation.system_type,
        schema_id=representation.schema_id,
        pair_order=representation.pair_order,
        pair_names=representation.pair_names,
        filtration_profiles=profiles,
        parameters=parameters,
        frozen=True,
    )
    return FiltrationRepairResult(
        representation_spec=repaired,
        decisions=tuple(decisions),
        changed=True,
        repair_round=next_round,
        parent_representation_hash=representation.spec_hash,
    )


def _repair_decision(
    *,
    invariant: str,
    profile: FiltrationProfile,
    audit: Mapping[str, Any],
    config: FiltrationRepairConfig,
    allow_changes: bool,
) -> dict[str, Any]:
    recommendation = str(audit.get("recommendation") or "KEEP").upper()
    reasons = tuple(str(value).upper() for value in audit.get("reason_codes", ()))
    old_stop = float(profile.stop)
    result: dict[str, Any] = {
        "invariant": invariant,
        "recommendation": recommendation,
        "reason_codes": list(reasons),
        "old_stop": old_stop,
        "old_step": float(profile.step),
        "new_stop": old_stop,
        "new_step": float(profile.step),
        "point_count": profile.num_points,
        "changed": False,
        "action": "KEEP",
        "reason": "audit_does_not_request_a_distance_range_change",
    }
    if not allow_changes:
        result["reason"] = "repair_disabled_or_round_limit_reached"
        return result
    if not config.method_specific:
        result["reason"] = "method_specific_repair_disabled"
        return result
    if recommendation == "EXTEND" and "ACTIVE_AT_BOUNDARY" in reasons:
        new_stop = min(config.max_stop_angstrom, old_stop * config.extend_factor)
        result.update(action="EXTEND", reason="active_at_boundary")
    elif recommendation == "SHORTEN" and "ACTIVE_AT_BOUNDARY" not in reasons:
        last_effective = audit.get("last_effective_axis_value")
        try:
            retained_tail_stop = float(last_effective) + (
                config.trailing_buffer_points * float(profile.step)
            )
        except (TypeError, ValueError):
            retained_tail_stop = old_stop * config.shorten_factor
        # Do not remove more than the configured fraction in one repair round.
        new_stop = max(old_stop * config.shorten_factor, retained_tail_stop)
        new_stop = max(config.min_stop_angstrom, min(old_stop, new_stop))
        result.update(action="SHORTEN", reason="saturated_or_sparse_tail")
    elif recommendation == "SHORTEN":
        result["reason"] = "conflicting_active_boundary_and_short_tail_signals"
        return result
    else:
        return result

    new_stop = max(profile.start + (profile.num_points - 1) * 1.0e-9, new_stop)
    if abs(new_stop - old_stop) <= max(1.0e-9, abs(old_stop) * 1.0e-6):
        result["reason"] = "requested_change_was_limited_by_configured_bounds"
        return result
    new_profile = _with_stop(profile, new_stop, source="audit_method_specific_repair")
    result.update(
        new_stop=float(new_profile.stop),
        new_step=float(new_profile.step),
        changed=True,
    )
    return result


def _with_stop(profile: FiltrationProfile, stop: float, *, source: str) -> FiltrationProfile:
    step = (stop - profile.start) / (profile.num_points - 1)
    return FiltrationProfile(
        scale_kind=profile.scale_kind,
        start=float(profile.start),
        stop=float(stop),
        step=float(step),
        num_points=profile.num_points,
        source=source,
    )


def _repair_round(representation: RepresentationSpec) -> int:
    repair = representation.parameters.get("filtration_repair", {})
    if not isinstance(repair, Mapping):
        return 0
    try:
        return max(0, int(repair.get("repair_round", 0)))
    except (TypeError, ValueError):
        return 0


def _validate_audit_compatibility(
    representation: RepresentationSpec, audit: Mapping[str, Any]
) -> None:
    audit_hash = audit.get("representation_hash")
    if audit_hash is not None and str(audit_hash) != representation.spec_hash:
        raise ValueError("filtration audit does not match the representation spec")
    if str(audit.get("status") or "").upper() == "FAIL":
        raise ValueError("cannot repair from a failed filtration audit")


def _plain_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return {str(key): _plain_value(item) for key, item in value.items()}
    return {}


def _plain_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain_value(item) for item in value]
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Create a method-specific filtration repair from an audit report."
    )
    parser.add_argument("--representation-spec", type=Path, required=True)
    parser.add_argument("--filtration-audit", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--task-id", default=None)
    parser.add_argument("--invariant", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    raw_config = load_yaml(args.config)
    design = raw_config.get("representation_design", raw_config)
    if not isinstance(design, Mapping):
        raise ValueError("repair config must contain representation_design")
    repair_config = FiltrationRepairConfig.from_mapping(
        design.get("filtration_repair")
        if isinstance(design.get("filtration_repair"), Mapping)
        else None
    )
    audit = json.loads(args.filtration_audit.read_text(encoding="utf-8"))
    if not isinstance(audit, Mapping):
        raise ValueError("filtration audit must contain an object")
    result = repair_filtration_profiles(
        RepresentationSpec.read(args.representation_spec),
        audit,
        requested_invariants=args.invariant,
        config=repair_config,
    )
    report = result.report(
        task_id=args.task_id,
        audit=audit,
        requested_invariants=args.invariant,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"status={report['status']} representation_hash={report['representation_hash']} output={args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
