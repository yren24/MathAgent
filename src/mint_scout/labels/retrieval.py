from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping, Protocol, Sequence

from mint_scout.schemas import SampleRecord


class MatchStatus(str, Enum):
    EXACT = "exact"
    PROBABLE = "probable"
    AMBIGUOUS = "ambiguous"
    CONFLICT = "conflict"
    NOT_FOUND = "not_found"


@dataclass(frozen=True)
class TargetLabelSpec:
    target_name: str
    unit: str | None = None
    endpoint_definition: str | None = None

    def __post_init__(self) -> None:
        if not self.target_name.strip():
            raise ValueError("target_name is required")


@dataclass(frozen=True)
class RetrievedLabel:
    sample_id: str
    identifier_type: str
    identifier_value: str
    target_name: str
    value: float
    unit: str | None
    endpoint_definition: str | None
    experimental_conditions: str | None
    source_name: str
    source_url_or_record_id: str
    match_status: MatchStatus
    retrieval_timestamp: str
    notes: str | None = None
    raw_value: float | None = None
    raw_unit: str | None = None

    def __post_init__(self) -> None:
        required = {
            "sample_id": self.sample_id,
            "identifier_type": self.identifier_type,
            "identifier_value": self.identifier_value,
            "target_name": self.target_name,
            "source_name": self.source_name,
            "source_url_or_record_id": self.source_url_or_record_id,
            "retrieval_timestamp": self.retrieval_timestamp,
        }
        missing = [name for name, value in required.items() if not str(value).strip()]
        if missing:
            raise ValueError(f"Retrieved labels require grounded provenance fields: {missing}")
        if not math.isfinite(self.value):
            raise ValueError("Retrieved label value must be finite")
        if self.raw_value is not None and not math.isfinite(self.raw_value):
            raise ValueError("Retrieved raw label value must be finite")

    @classmethod
    def grounded(
        cls,
        *,
        sample_id: str,
        identifier_type: str,
        identifier_value: str,
        target_name: str,
        value: float,
        unit: str | None,
        endpoint_definition: str | None,
        source_name: str,
        source_url_or_record_id: str,
        match_status: MatchStatus = MatchStatus.EXACT,
        experimental_conditions: str | None = None,
        notes: str | None = None,
        retrieval_timestamp: str | None = None,
    ) -> "RetrievedLabel":
        return cls(
            sample_id=sample_id,
            identifier_type=identifier_type,
            identifier_value=identifier_value,
            target_name=target_name,
            value=float(value),
            unit=unit,
            endpoint_definition=endpoint_definition,
            experimental_conditions=experimental_conditions,
            source_name=source_name,
            source_url_or_record_id=source_url_or_record_id,
            match_status=match_status,
            retrieval_timestamp=retrieval_timestamp
            or datetime.now(timezone.utc).isoformat(),
            notes=notes,
            raw_value=float(value),
            raw_unit=unit,
        )


class LabelRetrievalProvider(Protocol):
    def retrieve(
        self, sample: SampleRecord, target_spec: TargetLabelSpec
    ) -> Sequence[RetrievedLabel]: ...


@dataclass(frozen=True)
class LabelRetrievalPolicy:
    min_labeled_samples: int = 300
    allow_small_data_override: bool = False
    accepted_match_statuses: tuple[MatchStatus, ...] = (MatchStatus.EXACT,)
    unit_conversion_factors: Mapping[tuple[str, str], float] = field(default_factory=dict)
    conflict_absolute_tolerance: float = 0.0

    def __post_init__(self) -> None:
        if self.min_labeled_samples < 1:
            raise ValueError("min_labeled_samples must be positive")
        if self.conflict_absolute_tolerance < 0:
            raise ValueError("conflict_absolute_tolerance must be non-negative")
        factors = dict(self.unit_conversion_factors)
        if any(not math.isfinite(value) or value <= 0 for value in factors.values()):
            raise ValueError("unit conversion factors must be finite and positive")
        object.__setattr__(self, "unit_conversion_factors", MappingProxyType(factors))

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> "LabelRetrievalPolicy":
        statuses = tuple(
            MatchStatus(str(value))
            for value in config.get("accepted_match_statuses", (MatchStatus.EXACT.value,))
        )
        factors: dict[tuple[str, str], float] = {}
        for conversion in config.get("unit_conversions", ()):
            factors[(str(conversion["from"]), str(conversion["to"]))] = float(
                conversion["factor"]
            )
        return cls(
            min_labeled_samples=int(config.get("min_labeled_samples", 300)),
            allow_small_data_override=bool(config.get("allow_small_data_override", False)),
            accepted_match_statuses=statuses,
            unit_conversion_factors=factors,
            conflict_absolute_tolerance=float(
                config.get("conflict_absolute_tolerance", 0.0)
            ),
        )


@dataclass(frozen=True)
class LabelDecision:
    sample_id: str
    status: str
    reason: str
    normalized_value: float | None = None
    normalized_unit: str | None = None
    records: tuple[RetrievedLabel, ...] = ()


@dataclass(frozen=True)
class LabelRetrievalResult:
    decisions: tuple[LabelDecision, ...]
    accepted_values: Mapping[str, float]
    provenance_records: tuple[RetrievedLabel, ...]
    warnings: tuple[str, ...]
    sufficient_for_modeling: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "accepted_values", MappingProxyType(dict(self.accepted_values)))


def retrieve_and_validate_labels(
    *,
    samples: Sequence[SampleRecord],
    target_spec: TargetLabelSpec,
    provider: LabelRetrievalProvider,
    policy: LabelRetrievalPolicy = LabelRetrievalPolicy(),
) -> LabelRetrievalResult:
    decisions: list[LabelDecision] = []
    provenance: list[RetrievedLabel] = []
    accepted: dict[str, float] = {}

    for sample in samples:
        records = tuple(provider.retrieve(sample, target_spec))
        provenance.extend(records)
        decision = _validate_sample_records(sample, target_spec, records, policy)
        decisions.append(decision)
        if decision.status == "accepted":
            assert decision.normalized_value is not None
            accepted[sample.sample_id] = decision.normalized_value

    warnings: list[str] = []
    sufficient = len(accepted) >= policy.min_labeled_samples
    if not sufficient:
        warnings.append(
            f"INSUFFICIENT_LABELS: recovered {len(accepted)} reliable labels; "
            f"configured minimum is {policy.min_labeled_samples}."
        )
        sufficient = policy.allow_small_data_override and bool(accepted)
        if sufficient:
            warnings.append("SMALL_DATA_OVERRIDE_ENABLED")

    return LabelRetrievalResult(
        decisions=tuple(decisions),
        accepted_values=accepted,
        provenance_records=tuple(provenance),
        warnings=tuple(warnings),
        sufficient_for_modeling=sufficient,
    )


def _validate_sample_records(
    sample: SampleRecord,
    target_spec: TargetLabelSpec,
    records: tuple[RetrievedLabel, ...],
    policy: LabelRetrievalPolicy,
) -> LabelDecision:
    if not records:
        return LabelDecision(sample.sample_id, "not_found", "Provider returned no records")

    eligible: list[RetrievedLabel] = []
    rejection_reasons: list[str] = []
    for record in records:
        reason = _record_rejection_reason(sample, target_spec, record, policy)
        if reason is not None:
            rejection_reasons.append(reason)
            continue
        try:
            eligible.append(_normalize_unit(record, target_spec, policy))
        except ValueError as exc:
            rejection_reasons.append(str(exc))

    if not eligible:
        return LabelDecision(
            sample.sample_id,
            "rejected",
            "; ".join(sorted(set(rejection_reasons))),
            records=records,
        )

    values = [record.value for record in eligible]
    if max(values) - min(values) > policy.conflict_absolute_tolerance:
        conflicts = tuple(replace(record, match_status=MatchStatus.CONFLICT) for record in eligible)
        return LabelDecision(
            sample.sample_id,
            "conflict",
            "Credible records disagree after deterministic unit normalization",
            records=conflicts,
        )

    value = float(sum(values) / len(values))
    return LabelDecision(
        sample.sample_id,
        "accepted",
        "Grounded record passed identifier, target, endpoint, and unit checks",
        normalized_value=value,
        normalized_unit=target_spec.unit,
        records=tuple(eligible),
    )


def _record_rejection_reason(
    sample: SampleRecord,
    target_spec: TargetLabelSpec,
    record: RetrievedLabel,
    policy: LabelRetrievalPolicy,
) -> str | None:
    if record.sample_id != sample.sample_id:
        return "sample_id mismatch"
    expected_identifier = sample.identifiers.get(record.identifier_type)
    if expected_identifier is None or expected_identifier != record.identifier_value:
        return "identifier mismatch or identifier absent from manifest"
    if record.target_name.casefold().strip() != target_spec.target_name.casefold().strip():
        return "target name mismatch"
    if target_spec.endpoint_definition is not None:
        if record.endpoint_definition is None or (
            record.endpoint_definition.casefold().strip()
            != target_spec.endpoint_definition.casefold().strip()
        ):
            return "endpoint mismatch"
    if record.match_status not in policy.accepted_match_statuses:
        return f"match status {record.match_status.value!r} is not accepted"
    return None


def _normalize_unit(
    record: RetrievedLabel,
    target_spec: TargetLabelSpec,
    policy: LabelRetrievalPolicy,
) -> RetrievedLabel:
    if target_spec.unit is None:
        return record
    if record.unit == target_spec.unit:
        return record
    if record.unit is None:
        raise ValueError("retrieved unit is missing")
    factor = policy.unit_conversion_factors.get((record.unit, target_spec.unit))
    if factor is None:
        raise ValueError(f"no approved unit conversion from {record.unit!r} to {target_spec.unit!r}")
    return replace(
        record,
        value=float(record.value * factor),
        unit=target_spec.unit,
        raw_value=record.raw_value if record.raw_value is not None else record.value,
        raw_unit=record.raw_unit if record.raw_unit is not None else record.unit,
    )
